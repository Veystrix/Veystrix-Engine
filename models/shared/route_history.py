"""Expert-route history for cross-layer lookahead prefetch (blueprint MOE-04/05, EXP-K).

A one-layer-ahead router projection can only see ``l + 1``. The layer after that
is two hidden-state transforms away, so a direct projection is not available
before ``l`` finishes. Routed expert sets on coding traffic repeat across turns
and across adjacent decode steps, so the previous pass's top-k is a cheap
advisory predictor for ``l + 2``. The real router still decides what is
computed, so a wrong prediction only wastes prefetch bandwidth and can never
change the answer.
"""

from __future__ import annotations

import numpy as np


class RouteHistory:
    """Exponentially weighted expert frequency per layer, updated from observed routes."""

    def __init__(self, layers, experts: int, *, decay: float = 0.5):
        if not 0.0 <= decay < 1.0:
            raise ValueError("route-history decay must be in [0, 1)")
        self.experts = int(experts)
        self.decay = float(decay)
        self.counts = {int(layer): np.zeros(self.experts, dtype=np.float64) for layer in layers}
        self.passes = 0

    def observe(self, layer: int, ids) -> None:
        counts = self.counts.get(int(layer))
        if counts is None:
            return
        counts *= self.decay
        flat = np.asarray(ids, dtype=np.int64).reshape(-1)
        valid = flat[(flat >= 0) & (flat < self.experts)]
        if valid.size:
            np.add.at(counts, valid, 1.0)

    def finish_pass(self) -> None:
        self.passes += 1

    def predict(self, layer: int, k: int):
        """The ``k`` most frequent experts for ``layer``, highest first. Empty before any observation."""
        counts = self.counts.get(int(layer))
        if counts is None or k <= 0:
            return []
        order = np.argsort(-counts)
        return [int(expert) for expert in order[:k] if counts[expert] > 0]

    def recall(self, layer: int, ids, k: int) -> float:
        """Fraction of the observed ``ids`` that a ``k``-deep history prediction covers."""
        observed = set(int(x) for x in np.asarray(ids).reshape(-1))
        if not observed:
            return 1.0
        return len(observed & set(self.predict(layer, k))) / len(observed)

    def state(self) -> dict:
        return {"experts": self.experts, "decay": self.decay, "passes": self.passes,
                "counts": {str(layer): counts.tolist() for layer, counts in self.counts.items()}}
