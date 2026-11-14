#!/usr/bin/env python3
"""Offline recall check for the route-history lookahead (MOE-04/05).

Reads a Nemotron decode-profile JSON (which records each forward pass's routed
expert ids per MoE layer), then asks how much of the next token's route set a
prediction built from earlier tokens would have covered. This is trace
analysis, not a live model run: it establishes that the advisory lookahead has
signal on the recorded coding turn, not a throughput result.

    python -m models.nemotron35.tests.bench.bench_lookahead_recall PROFILE.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from models.shared.route_history import RouteHistory  # noqa: E402
from veystrix.measurement import MeasurementRun  # noqa: E402
from veystrix.results import output_path  # noqa: E402


def evaluate(profile: dict, *, depth: int, decay: float) -> dict:
    tokens = profile.get("tokens") or []
    layers = sorted({int(layer) for token in tokens for layer in (token.get("routes") or {})})
    if not tokens or not layers:
        raise ValueError("profile has no routed tokens")
    history = RouteHistory(layers, max(max(routes) for token in tokens
                                       for routes in (token.get("routes") or {}).values()) + 1, decay=decay)
    previous: dict[int, set] = {}
    recalls, static_recalls, random_recalls = [], [], []
    observations = 0
    for token in tokens:
        routes = {int(layer): [int(e) for e in experts] for layer, experts in (token.get("routes") or {}).items()}
        for layer, experts in routes.items():
            observed = set(experts)
            predicted = set(history.predict(layer, depth))
            recalls.append(len(observed & predicted) / len(observed))
            static_recalls.append(len(observed & previous.get(layer, set())) / len(observed))
            random_recalls.append(min(depth, len(observed)) / history.experts)
            observations += 1
        for layer, experts in routes.items():
            history.observe(layer, experts)
            previous[layer] = set(experts)
    mean = lambda values: sum(values) / len(values) if values else None  # noqa: E731
    return {"tokens": len(tokens), "layers": len(layers), "observations": observations,
            "depth": depth, "decay": decay,
            "history_recall": round(mean(recalls), 4),
            "last_token_recall": round(mean(static_recalls), 4),
            "random_expectation": round(mean(random_recalls), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--decay", type=float, default=0.5)
    args = parser.parse_args()
    payload = json.loads(args.profile.read_text())
    result = output_path("nemotron35", "lookahead-recall", "lookahead-recall.jsonl")
    run = MeasurementRun(result, runtime="nemotron35", experiment="lookahead-recall", metadata={
        "diagnostic": True, "source": str(args.profile),
        "source_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        "kind": "offline trace analysis; no live model execution"})
    run.start()
    metrics = evaluate(payload, depth=args.depth, decay=args.decay)
    run.arm(arm_id="route-history", condition="history", round_index=1, command=sys.argv, metrics=metrics)
    run.finish("diagnostic", **metrics)
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
