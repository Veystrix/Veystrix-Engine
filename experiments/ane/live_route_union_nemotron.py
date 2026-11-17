"""Live route-union measurement for Nemotron on the current checkpoint.

Authorized guarded run (2026-11-17). Hooks the existing pool.observe_routes
read-only (wraps it; no production code changed) to capture per-token, per-layer
selected expert IDs (top-6 of 128) and compute the consecutive-token union ratio.

Run under real_run_guard.py with the established low-memory envelope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from veystrix.results import output_path  # noqa: E402

PROMPT = "Explain mixture-of-experts routing and top-k expert selection briefly."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=128)
    args = ap.parse_args()

    from veystrix.bench.qualification import load_backend
    ckpt = str(Path.home() / "models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit-mlx-community")
    backend = load_backend("nemotron35", ckpt, args.tokens + 64, None)

    core = backend.core
    # OverlappedSharedMoE is an nn.Module dict container: mixer["switch_mlp"] is
    # a StreamedSwitchMLP whose .pool owns observe_routes.
    pool = None
    for layer in core.model.backbone.layers:
        mixer = getattr(layer, "mixer", None)
        sw = mixer["switch_mlp"] if (mixer is not None and hasattr(mixer, "keys")
                                     and "switch_mlp" in mixer.keys()) else None
        if sw is not None and hasattr(getattr(sw, "pool", None), "observe_routes"):
            pool = sw.pool
            break
    if pool is None:
        raise RuntimeError("could not locate Nemotron ExpertPool with observe_routes")
    # Every MoE layer shares this same pool instance; wrapping it once observes all layers.

    original = pool.observe_routes
    state = {"token": -1}
    per_token = defaultdict(set)

    def observe(layer, ids):
        import numpy as np
        arr = np.asarray(ids).reshape(-1)
        if len(arr) == 6:  # a single decode token routes top-6
            per_token[int(state["token"])].update((int(layer), int(e)) for e in arr)
        return original(layer, ids)

    pool.observe_routes = observe
    per_token_bytes = []
    try:
        ids = backend.tokenizer.encode(PROMPT) if hasattr(backend, "tokenizer") else None
    except Exception:
        ids = None
    from veystrix.bench.qualification import prompt_ids
    ids = prompt_ids(backend, 32)
    backend.reset()
    backend.append_tokens(ids)

    def on_prefilled():
        state["token"] = 0

    def on_token(_v, _p):
        state["token"] += 1

    out, stats = backend.generate(args.tokens, on_prefilled=on_prefilled, on_decode_token=on_token)
    pool.observe_routes = original

    # union ratio: for each consecutive decode token pair, per-layer union/sum
    layers = sorted({l for l, _ in set().union(*per_token.values())})
    tokens = sorted(per_token)
    by_tl = {(l, t): per_token[t] for t in tokens for l in layers}
    ratios = []
    for t1, t2 in zip(tokens, tokens[1:]):
        su = sa = 0
        for l in layers:
            a = {e for (ll, e) in per_token[t1] if ll == l}
            b = {e for (ll, e) in per_token[t2] if ll == l}
            su += len(a | b); sa += len(a) + len(b)
        if sa:
            ratios.append(su / sa)

    msg = {
        "checkpoint": ckpt, "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tokens": stats.tokens, "layers_with_routes": len(layers),
        "routed_per_call_median": statistics.median([len([e for l, e in s]) for s in per_token.values()]) if per_token else None,
        "union_ratio_k2_mean": statistics.mean(ratios) if ratios else None,
        "union_ratio_k2_median": statistics.median(ratios) if ratios else None,
        "union_ratio_k2_p10": sorted(ratios)[len(ratios)//10] if ratios else None,
        "union_ratio_k2_p90": sorted(ratios)[9*len(ratios)//10] if ratios else None,
        "output_sha256_prefix": hashlib.sha256(str(out).encode()).hexdigest()[:16],
    }
    dest = output_path("nemotron35", "live-route-union", "union.json")
    dest.write_text(json.dumps(msg, indent=1))
    print(json.dumps(msg, indent=1), flush=True)
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
