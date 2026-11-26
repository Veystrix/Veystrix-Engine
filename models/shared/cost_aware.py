"""Cost-aware residency scoring for a slot pool of fixed-size expert records.

Decode visits layers in order, so a pool smaller than one token's working set
evicts every record before its layer comes round again under recency-based
policies. This module scores each resident record by how likely it is to be
routed again before it would be evicted: a decayed per-(layer, expert) route
frequency, an estimate of how many visits to that layer fit in the pool's
survival horizon, and small terms for recent use and refill cost. The
frequency table persists across evictions, which is what lets such a pool keep
the records most likely to be routed again instead of the most recent ones.

Eviction choice can never change model output, only which bytes are re-read.
Pure Python and NumPy, main-thread use only.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields

import numpy as np

WEIGHTS_VERSION = 1
_NON_NEGATIVE = ("recent", "frequency", "reuse", "refill", "size")


def _real(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"cost weight {name!r} must be a real number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"cost weight {name!r} must be finite, got {value!r}")
    return value


@dataclass(frozen=True)
class CostWeights:
    """Score weights. ``size`` is constant here (all records are equal size); it documents the ``-e*B_resident`` term."""

    recent: float = 0.1
    frequency: float = 0.2
    reuse: float = 1.0
    refill: float = 0.0
    size: float = 0.0
    alpha: float = 0.3

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, _real(field.name, getattr(self, field.name)))
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"cost weight 'alpha' must be in (0, 1], got {self.alpha!r}")
        for name in _NON_NEGATIVE:
            if getattr(self, name) < 0.0:
                raise ValueError(f"cost weight {name!r} must be >= 0, got {getattr(self, name)!r}")

    @classmethod
    def from_mapping(cls, data) -> "CostWeights":
        if not isinstance(data, dict):
            raise ValueError("cost weights must be a JSON object")
        known = {field.name for field in fields(cls)}
        unknown = sorted(str(key) for key in data if key not in known and key != "version")
        if unknown:
            raise ValueError(f"unknown cost weight keys: {', '.join(unknown)}")
        version = data.get("version", WEIGHTS_VERSION)
        if isinstance(version, bool) or version != WEIGHTS_VERSION:
            raise ValueError(f"unsupported cost weights version {version!r}; expected {WEIGHTS_VERSION}")
        return cls(**{key: value for key, value in data.items() if key in known})

    @classmethod
    def load(cls, path) -> "CostWeights":
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot read cost weights from {path}: {error}") from error
        if not isinstance(data, dict):
            raise ValueError(f"cost weights file {path} is not a JSON object")
        try:
            return cls.from_mapping(data)
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error

    def to_mapping(self) -> dict:
        return {"version": WEIGHTS_VERSION, **asdict(self)}

    @classmethod
    def from_env(cls, environ=None, flag: str = "FLASHNEXT_EXPERT_POOL_COST_WEIGHTS") -> "CostWeights":
        value = (os.environ if environ is None else environ).get(flag, "")
        return cls.load(os.path.expanduser(value)) if value else cls()


class RouteFrequency:
    """Decayed per-(layer, expert) route probability, shared by eviction and prefetch."""

    def __init__(self, layers: int = 48, experts: int = 512, alpha: float = 0.3):
        self.layers, self.experts, self.alpha = layers, experts, alpha
        self.table = np.zeros((layers, experts), dtype=np.float64)

    def observe(self, layer, experts) -> None:
        """One visit to ``layer`` routed ``experts`` (duplicates count once)."""
        if not 0 <= layer < self.layers:
            return
        ids = np.unique(np.fromiter(experts, dtype=np.int64))
        ids = ids[(ids >= 0) & (ids < self.experts)]
        row = self.table[layer]
        row *= 1.0 - self.alpha
        row[ids] += self.alpha

    def probability(self, layer, expert) -> float:
        if 0 <= layer < self.layers and 0 <= expert < self.experts:
            return float(self.table[layer, expert])
        return 0.0

    def predict(self, layer, k, exclude=()) -> list[tuple[int, float]]:
        """Up to ``k`` (expert, probability) pairs, most likely first, ties to the lower id."""
        if not 0 <= layer < self.layers or k <= 0:
            return []
        row = self.table[layer]
        ids = np.flatnonzero(row > 0.0)
        skip = [int(i) for i in exclude]
        if len(ids) and skip:
            ids = ids[~np.isin(ids, skip)]
        if not len(ids):
            return []
        order = np.lexsort((ids, -row[ids]))[:k]
        return [(int(ids[i]), float(row[ids[i]])) for i in order]


class CostAwarePolicy:
    """Slot-indexed residency scoring; the pool owns the key-to-slot map and the free list."""

    _STALE = 1  # seeded entries start this many steps beyond the recency window

    def __init__(self, slots: int, *, layers: int = 48, experts: int = 512,
                 weights: CostWeights | None = None, frequency: RouteFrequency | None = None):
        self.slots, self.layers = slots, layers
        self.weights = weights or CostWeights()
        self.frequency = frequency or RouteFrequency(layers, experts, self.weights.alpha)
        self._layer = np.full(slots, -1, dtype=np.int64)
        self._expert = np.zeros(slots, dtype=np.int64)
        self._occupied = np.zeros(slots, dtype=bool)
        self._last = np.zeros(slots, dtype=np.int64)
        self._step = 0
        self._current = 0
        self._miss_rate = 8.0
        self._refill = np.zeros(layers, dtype=np.float64)
        self._refill_seen = np.zeros(layers, dtype=bool)

    def advance(self, layer) -> None:
        self._step += 1
        self._current = layer

    def touch(self, key, slot) -> None:
        self._last[slot] = self._step

    def insert(self, key, slot, *, seeded: bool = False) -> None:
        if self._occupied[slot]:
            raise ValueError(f"slot {slot} already holds {self.key_of(slot)}")
        self._layer[slot], self._expert[slot] = key
        self._occupied[slot] = True
        self._last[slot] = self._step - self.layers - self._STALE if seeded else self._step

    def discard(self, key, slot) -> None:
        if self.key_of(slot) != tuple(key):
            raise ValueError(f"slot {slot} does not hold {tuple(key)}")
        self._occupied[slot] = False

    def key_of(self, slot) -> tuple[int, int] | None:
        if not self._occupied[slot]:
            return None
        return int(self._layer[slot]), int(self._expert[slot])

    def _scores(self, index) -> np.ndarray:
        """Scores of the occupied slots in ``index`` (higher keeps)."""
        w, layers = self.weights, self.layers
        layer, expert = self._layer[index], self._expert[index]
        table = self.frequency.table
        inside = (layer >= 0) & (layer < layers)
        valid = inside & (layer < table.shape[0]) & (expert >= 0) & (expert < table.shape[1])
        p = np.where(valid, table[np.where(valid, layer, 0), np.where(valid, expert, 0)], 0.0)
        phase = np.where(inside, (layer - self._current) % layers, 0)
        phase = np.where(phase == 0, layers, phase)
        horizon = self.slots / max(self._miss_rate, 0.5)
        visits = np.where(phase <= horizon, np.floor((horizon - phase) / layers) + 1.0, 0.0)
        reuse = 1.0 - (1.0 - p) ** visits
        recent = (self._step - self._last[index]) <= layers
        score = w.recent * recent + w.frequency * p + w.reuse * reuse - w.size
        if w.refill and self._refill_seen.any():
            mean = self._refill[self._refill_seen].mean()
            if mean > 0.0:
                cost = np.where(self._refill_seen, self._refill / mean, 0.0)
                score = score + w.refill * np.where(inside, cost[np.where(inside, layer, 0)], 0.0)
        return score

    def victims(self, count, protected=()) -> list[int]:
        """The ``count`` lowest-scoring unprotected occupied slots, lowest first. Does not modify state."""
        if count <= 0:
            return []
        eligible = self._occupied.copy()
        for slot in protected:
            eligible[slot] = False
        index = np.flatnonzero(eligible)
        if len(index) < count:
            raise RuntimeError(f"need {count} victims but only {len(index)} unprotected slots are occupied")
        order = np.lexsort((index, self._last[index], self._scores(index)))[:count]
        return [int(slot) for slot in index[order]]

    def observe(self, layer, experts, misses: int = 0) -> None:
        self.frequency.observe(layer, experts)
        self._miss_rate = 0.95 * self._miss_rate + 0.05 * misses

    def observe_refill(self, layer, seconds) -> None:
        if not 0 <= layer < self.layers or not math.isfinite(seconds) or seconds < 0:
            return
        if self._refill_seen[layer]:
            self._refill[layer] += 0.2 * (seconds - self._refill[layer])
        else:
            self._refill[layer], self._refill_seen[layer] = seconds, True

    def stats(self) -> dict:
        return {"policy": "cost-aware", "weights": self.weights.to_mapping(), "step": self._step,
                "miss_rate": self._miss_rate, "resident": int(self._occupied.sum())}
