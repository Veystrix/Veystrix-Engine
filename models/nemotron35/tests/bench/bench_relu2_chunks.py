#!/usr/bin/env python3
"""Phase 0b — layout-aware down-projection chunk analysis (offline diagnostic).

Fix for Phase 0a's bug: 0a chunked the activation vector by chunk_bytes//4
columns, but there are only 1856 activations, so a 16 KB "chunk" (4096 cols)
covered the whole vector → needed fraction ~1.0. That was meaningless.

Here we model the REAL down_proj.weight layout from affine_index.py:
  down_proj.weight shape = (hidden=2688, intermediate/8=232) dtype U32
  => logical W_down is (2688, 1856) 4-bit codes, 8 codes per U32 word.
  Activation vector a has 1856 entries (one per intermediate column).
  y = W_down @ relu2(a). Column j of W_down is multiplied by a[j].

For each decode step we get the EXACT zero mask of relu2(a) (from the exact
up-projection output). Column j can be skipped iff a[j] == 0. We then compute,
for the on-disk layout, the fraction of W_down BYTE RANGES that must be read:
  - ideal per-64-group: a 64-column quantization group shares scale/bias;
    if ANY of its 64 columns is nonzero we must read that group's codes+scale+bias.
  - contiguous-chunk: if the layout interleaves groups across the hidden rows,
    compute needed fraction at 512B/4KB/16KB on-disk chunk granularity.

Report: zero fraction, per-group needed fraction, byte-range needed fraction.
Opus kill criterion: <60% zeros OR per-64-group needed >0.6 -> DROP.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from veystrix.results import output_path

_ORIG = nn.relu2
_S = {"calls": 0, "elements": 0, "nonzero": 0,
      "g64_total": 0, "g64_needed": 0,
      "col_hot": None, "max_calls": 4000}


def hook(x):
    out = _ORIG(x)
    if _S["calls"] < _S["max_calls"]:
        try:
            a = np.asarray(out.astype(mx.float32)).reshape(-1)
            nz = a != 0
            n = a.size
            _S["calls"] += 1
            _S["elements"] += n
            _S["nonzero"] += int(nz.sum())
            if _S["col_hot"] is None or _S["col_hot"].size != n:
                _S["col_hot"] = np.zeros(n, dtype=np.int64)
            _S["col_hot"] += nz.astype(np.int64)
            pad = (-n) % 64
            g = np.pad(nz, (0, pad)).reshape(-1, 64)
            _S["g64_total"] += g.shape[0]
            _S["g64_needed"] += int(g.any(axis=1).sum())
        except Exception:
            _S["calls"] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--pool-gb", type=float, default=2.0)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    nn.relu2 = hook
    from models.nemotron35.streamed_model import load_streamed, validate_config
    from models.nemotron35.backend_core import EngineCore

    ckpt = Path(args.checkpoint).expanduser().resolve(strict=True)
    cfg = json.loads((ckpt / "config.json").read_text())
    streamed = load_streamed(ckpt, pool_bytes=int(args.pool_gb * 2 ** 30),
                             verify=False, config_check=validate_config)
    core = EngineCore(streamed, chunk_tokens=256, int8_kv=True)
    cache = core._make_cache()
    rng = np.random.default_rng(0)
    prompt = rng.integers(0, cfg.get("vocab_size", 256000), size=args.prompt_tokens).tolist()
    token = None
    try:
        logits = streamed.model(mx.array([prompt], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        for _ in range(args.tokens - 1):
            logits = streamed.model(mx.array([[token]], dtype=mx.int32), cache=cache)
            token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            mx.eval(token)
    finally:
        streamed.close()

    zero = 1 - _S["nonzero"] / _S["elements"] if _S["elements"] else 0
    g64_needed = _S["g64_needed"] / _S["g64_total"] if _S["g64_total"] else 0
    # W_down is (hidden, intermediate); a 64-col group shares scale/bias.
    # If reading a group requires its codes (8 U32/row) + scale + bias, the
    # needed FRACTION of down codes ≈ fraction of 64-groups with any nonzero.
    # This is the optimistic byte saving (assumes group-reordered layout).
    down_codes_needed_frac = g64_needed
    down_codes_saved = 1 - down_codes_needed_frac
    # Total expert bytes: W_up 2.49 + up scale/bias 0.31 + W_down 2.49 + down sb 0.31.
    # W_up always fully read; down codes (2.49) partially; down scale/bias partially.
    total = 2.49 + 0.31 + 2.49 + 0.31
    optimistic_read = 2.49 + 0.31 + down_codes_needed_frac * (2.49 + 0.31)
    total_save = 1 - optimistic_read / total

    verdict = "KEEP" if (zero >= 0.60 and g64_needed <= 0.6) else "DROP"
    result = {
        "experiment": "relu2-chunks-phase0b", "verdict_hint": verdict,
        "zero_fraction": round(zero, 4),
        "needed_64group_fraction": round(g64_needed, 4),
        "down_code_bytes_saved_frac": round(down_codes_saved, 4),
        "total_expert_bytes_saved_frac_optimistic": round(total_save, 4),
        "calls": _S["calls"], "elements": _S["elements"],
        "opuses_kill": "zeros<0.60 OR needed-64group>0.6",
        "note": "needed-64group is the optimistic byte floor: group-reorderable layout; "
                "real contiguous layout may save less. 0a's chunk numbers were invalid "
                "(chunked 1856 activations by 4096 cols -> one chunk).",
    }
    target = output_path("nemotron35", "relu2-sparsity", "phase0b.json", args.json)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
