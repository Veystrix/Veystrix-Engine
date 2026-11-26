"""Cache-aware routing for the streamed Nemotron experts. Opt-in; `swap` changes what the model computes.

A decoded token reads its routed experts from the SSD unless the macOS file cache still holds them. When a chosen expert
is not resident and an unchosen one that was used a moment ago scores almost as well, taking the resident one turns a
drive read into a cache read. The idea is the training-free cache prior of "Mixture of Cache-Conditional Experts"
(arXiv 2412.00099) and the same swap `models/flashnext/routing.py` already measured on Qwen (+8%, 16.8% fewer reads).

``VEYSTRIX_NEMOTRON_CACHE_ROUTING``
    ``off`` (default)  exact routing, nothing changes.
    ``observe``        exact routing; counts how many swaps each epsilon would allow. Output is identical.
    ``swap``           swap within ``VEYSTRIX_NEMOTRON_CACHE_EPSILON`` (default 0.02, in router-score units).
``VEYSTRIX_NEMOTRON_CACHE_WINDOW``   tokens an expert counts as resident after use (default 3).
``VEYSTRIX_NEMOTRON_CACHE_STATS``    path; counters are written there as JSON every 64 routed layers.

Only single-row calls (decode) are touched. Prefill and speculative verification route exactly. The mixing weights of
the chosen set are the router's own sigmoid scores, normalised and scaled exactly as `group_expert_select` does.
"""
from __future__ import annotations

import json
import os
from collections import deque

import mlx.core as mx
import numpy as np
from mlx_lm.models import nemotron_h

GRID = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)


def mode() -> str:
    value = os.environ.get("VEYSTRIX_NEMOTRON_CACHE_ROUTING", "off").strip().lower()
    return value if value in ("observe", "swap") else "off"


class _Shared:
    """One per model: counters and settings shared by every MoE layer's gate."""

    def __init__(self, kind: str, epsilon: float, window: int, stats_path: str | None):
        self.kind, self.epsilon, self.window, self.stats_path = kind, epsilon, max(1, window), stats_path
        self.calls = 0
        self.chosen = 0
        self.resident = 0           # chosen experts that were already in the window (exact routing)
        self.swapped = 0            # swaps made (swap mode)
        self.possible = {eps: 0 for eps in GRID}   # observe: misses with a resident candidate within eps
        self.score_gap_sum = 0.0

    def summary(self) -> dict:
        calls = max(self.calls, 1)
        return {"mode": self.kind, "epsilon": self.epsilon, "window": self.window, "layer_calls": self.calls,
                "chosen": self.chosen, "chosen_resident": self.resident,
                "resident_fraction": round(self.resident / max(self.chosen, 1), 4), "swapped": self.swapped,
                "swapped_per_call": round(self.swapped / calls, 4),
                "possible_by_epsilon": {str(eps): n for eps, n in self.possible.items()},
                "possible_fraction_of_misses": {
                    str(eps): round(n / max(self.chosen - self.resident, 1), 4) for eps, n in self.possible.items()}}

    def flush(self) -> None:
        if self.stats_path:
            try:
                with open(self.stats_path, "w") as handle:
                    json.dump(self.summary(), handle)
            except OSError:
                pass


class CacheAwareGate(nemotron_h.MoEGate):
    """`MoEGate` with a cache prior on decode rows. Installed by changing a built gate's class (see `install`)."""

    def __call__(self, x):
        shared = self._cache_shared
        rows = int(np.prod(x.shape[:-1]))
        if rows != 1 or self.n_group != 1:
            return super().__call__(x)
        orig = mx.sigmoid((x @ self.weight.T).astype(mx.float32))
        selection = orig + self.e_score_correction_bias
        select_host = np.asarray(selection).reshape(-1)
        orig_host = np.asarray(orig).reshape(-1)
        k = self.top_k
        ids = np.argpartition(-select_host, k - 1)[:k]
        resident = {e for step in self._cache_history for e in step}
        chosen = set(int(e) for e in ids)
        misses = [e for e in chosen if e not in resident]
        shared.calls += 1
        shared.chosen += k
        shared.resident += k - len(misses)
        if misses:
            candidates = sorted((e for e in resident if e not in chosen), key=lambda e: -select_host[e])
            if candidates:
                best = float(select_host[candidates[0]])
                for eps in GRID:
                    shared.possible[eps] += sum(1 for e in misses if float(select_host[e]) - best <= eps)
            if shared.kind == "swap" and shared.epsilon > 0 and candidates:
                for e in sorted(misses, key=lambda e: select_host[e]):
                    if not candidates:
                        break
                    r = candidates[0]
                    if float(select_host[e]) - float(select_host[r]) <= shared.epsilon:
                        chosen.discard(e)
                        chosen.add(r)
                        candidates.pop(0)
                        shared.swapped += 1
        final = np.array(sorted(chosen), dtype=np.uint32)
        weights = orig_host[final]
        if k > 1 and self.norm_topk_prob:
            weights = weights / (weights.sum() + 1e-20)
        weights = weights * self.routed_scaling_factor
        self._cache_history.append(tuple(int(e) for e in final) if shared.kind == "swap"
                                   else tuple(int(e) for e in ids))
        if shared.calls % 64 == 0:
            shared.flush()
        lead = tuple(x.shape[:-1])
        return mx.array(final).reshape(*lead, k), mx.array(weights.astype(np.float32)).reshape(*lead, k)


def install(model, kind: str | None = None, *, epsilon: float | None = None, window: int | None = None):
    """Switch every MoE gate of a built model to `CacheAwareGate`. Returns the shared counters, or None when off."""
    kind = kind or mode()
    if kind == "off":
        return None
    epsilon = float(os.environ.get("VEYSTRIX_NEMOTRON_CACHE_EPSILON", "0.02")) if epsilon is None else epsilon
    window = int(os.environ.get("VEYSTRIX_NEMOTRON_CACHE_WINDOW", "3")) if window is None else window
    shared = _Shared(kind, epsilon, window, os.environ.get("VEYSTRIX_NEMOTRON_CACHE_STATS"))
    for layer in model.backbone.layers:
        gate = getattr(layer.mixer, "gate", None)
        if isinstance(gate, nemotron_h.MoEGate):
            gate.__class__ = CacheAwareGate
            object.__setattr__(gate, "_cache_shared", shared)
            object.__setattr__(gate, "_cache_history", deque(maxlen=shared.window))
    return shared
