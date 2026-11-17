#!/usr/bin/env python3
"""Phase 0 (offline diagnostic) — exact-zero sparsity of Nemotron's relu2 experts.

Opus's second-miracle "new lever": Nemotron experts compute
    y = W_down . relu2(W_up . x)
so wherever (W_up . x)_j <= 0 the activation is exactly +0 and the matching
column of W_down contributes nothing. If enough activations are exactly zero,
the down-projection chunks containing only zero activations need never be read
from SSD — bit-exact, because the unread weights are multiplied by exactly 0.

This harness measures whether that lever is viable. It hooks ``mlx.nn.relu2``
(which every expert forward calls), loads the real full-size streamed model,
runs a real greedy decode on a fixed prompt, and reports:
  - exact-zero fraction of relu2 activations (Opus kill: <60% zeros -> DROP)
  - fully-zero 64-neuron quantization groups
  - fraction of 4/16/64 KB down-projection chunks that contain any nonzero neuron
    (Opus kill: needed 16 KB chunks > 0.6 -> DROP)
  - per-neuron hotness (for offline co-activation reordering)

This is a DIAGNOSTIC: it does not change model output. No digest gate needed
(no behavior change), but the tool records the greedy token IDs it produced so
the run is reproducible. Reads real weights from SSD, so run under the guard.

Usage:
  scripts/flashnext/real_run_guard.py --log <dir>/guard.jsonl -- \
    .venv/bin/python -m models.nemotron35.tests.bench.bench_relu2_sparsity \
      --checkpoint <path> --tokens 96 --pool-gb 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from veystrix.results import output_path

# ---------------------------------------------------------------------------
# Global relu2 hook — stats accumulate here across the whole run.
# ---------------------------------------------------------------------------
_STATS = {
    "calls": 0, "elements": 0, "nonzero": 0,
    "groups64": 0, "groups64_any_nonzero": 0,
    "chunks": {k: {"total": 0, "needed": 0} for k in (4096, 16384, 65536)},
    "hotness_sum": None,  # np array sized on first call
    "sample_max": 4000,  # cap tensors recorded for per-call detail
    "per_call_nonzero_frac": [],
}
_ORIGINAL_RELU2 = nn.relu2


def _hooked_relu2(x):
    out = _ORIGINAL_RELU2(x)
    # Record stats lazily on a sampled subset (avoid eval cost on every call).
    if _STATS["calls"] < _STATS["sample_max"]:
        try:
            arr = np.asarray(out.astype(mx.float32)).reshape(-1)
            nz = arr != 0.0
            n = arr.size
            _STATS["calls"] += 1
            _STATS["elements"] += n
            _STATS["nonzero"] += int(nz.sum())
            _STATS["per_call_nonzero_frac"].append(float(nz.mean()) if n else 0.0)
            if _STATS["hotness_sum"] is None or _STATS["hotness_sum"].size != n:
                _STATS["hotness_sum"] = np.zeros(n, dtype=np.int64)
            _STATS["hotness_sum"] += nz.astype(np.int64)
            # 64-neuron groups
            pad = (-n) % 64
            g = np.pad(nz, (0, pad)).reshape(-1, 64)
            _STATS["groups64"] += g.shape[0]
            _STATS["groups64_any_nonzero"] += int(g.any(axis=1).sum())
            # chunk-level: fraction of chunks containing any nonzero. Column j
            # of W_down corresponds to activation j; chunk by contiguous columns.
            for chunk_bytes, rec in _STATS["chunks"].items():
                cols = chunk_bytes // 4  # 4 bytes/weight (U32 packed ~ but conservative)
                cols = max(1, cols)
                c = np.pad(nz, (0, (-n) % cols)).reshape(-1, cols)
                rec["total"] += c.shape[0]
                rec["needed"] += int(c.any(axis=1).sum())
        except Exception:
            _STATS["calls"] += 1  # still count, but skip detailed stats
    else:
        out  # noqa
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--pool-gb", type=float, default=2.0)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    nn.relu2 = _hooked_relu2  # install hook BEFORE model load / forward

    from models.nemotron35.streamed_model import load_streamed
    from models.nemotron35.backend_core import EngineCore

    ckpt = Path(args.checkpoint).expanduser().resolve(strict=True)
    config = json.loads((ckpt / "config.json").read_text())
    from models.nemotron35.streamed_model import validate_config
    streamed = load_streamed(ckpt, pool_bytes=int(args.pool_gb * 2 ** 30),
                             verify=False, config_check=validate_config)
    core = EngineCore(streamed, chunk_tokens=256, int8_kv=True)
    cache = core._make_cache()

    # Build a deterministic token prompt from the vocab (no tokenizer needed).
    rng = np.random.default_rng(0)
    vocab = config.get("vocab_size", 256000)
    prompt = rng.integers(0, vocab, size=args.prompt_tokens).tolist()
    ids = mx.array([prompt], dtype=mx.int32)

    began = time.perf_counter()
    produced = []
    try:
        logits = streamed.model(ids, cache=cache)
        mx.eval(logits)
        token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        produced.append(token)
        for _ in range(args.tokens - 1):
            logits = streamed.model(mx.array([[token]], dtype=mx.int32), cache=cache)
            token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            mx.eval(token)
            produced.append(token)
    finally:
        streamed.close()
    elapsed = time.perf_counter() - began

    nonzero_frac = (_STATS["nonzero"] / _STATS["elements"]) if _STATS["elements"] else 0.0
    zero_frac = 1.0 - nonzero_frac
    g64_frac = (_STATS["groups64_any_nonzero"] / _STATS["groups64"]) if _STATS["groups64"] else 0.0
    chunks = {}
    for size, rec in _STATS["chunks"].items():
        chunks[str(size)] = round(rec["needed"] / rec["total"], 4) if rec["total"] else 0.0

    verdict = "KEEP" if (zero_frac >= 0.60 and chunks.get("16384", 1.0) <= 0.6) else "DROP"
    result = {
        "experiment": "relu2-sparsity-phase0", "verdict_hint": verdict,
        "gate": {"note": "diagnostic only; output unchanged by hook", "tokens": len(produced)},
        "zero_fraction": round(zero_frac, 4),
        "nonzero_fraction": round(nonzero_frac, 4),
        "groups64_any_nonzero_fraction": round(g64_frac, 4),
        "needed_chunk_fraction": chunks,
        "relu2_calls_sampled": _STATS["calls"],
        "elements_sampled": _STATS["elements"],
        "per_call_nonzero_frac_mean": round(float(np.mean(_STATS["per_call_nonzero_frac"])), 4)
        if _STATS["per_call_nonzero_frac"] else 0.0,
        "tok_s": len(produced) / elapsed if elapsed else 0.0,
        "tokens": args.tokens, "prompt_tokens": args.prompt_tokens, "pool_gb": args.pool_gb,
        "checkpoint": str(ckpt),
    }
    target = output_path("nemotron35", "relu2-sparsity", "phase0.json", args.json)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
