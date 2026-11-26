#!/usr/bin/env python3
"""DIAGNOSTIC ONLY -- OWNER RUNS THIS AGAINST THE REAL CHECKPOINT.

Hooks the production Qwen4ExpPLELayer at decoder layer 1 (the only PLE layer)
and dumps the EXACT tensor that production adds to hidden_states, for:
  * the 16-token prompt prefill (one batched call, S=16), and
  * the first greedy decode steps (S=1).

This is the ground truth needed to compare a C++ ple_forward call against the
real Python path. It intentionally uses the same deterministic prompt as
cpp/engine/tools/capture_real_reference.py so the two references line up.

Usage (do NOT let subagents run this on the real checkpoint):
  /Users/vincent/Documents/Veystrix/.venv/bin/python \
      cpp/engine/tools/dump_ple_real.py \
      --checkpoint /Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP \
      --out cpp/engine/tests/vectors_ple_real \
      --prompt-tokens 16 --tokens 3

Each call directory contains:
  hidden_states.npy   [B,S,10240]  -- tensor passed AS hidden_states to PLE
  input_ids.npy       [B,S] int64  -- raw token ids passed to PLE
  ple_output.npy      [B,S,10240]  -- the value added to hidden_states
  mask.npy            [B,S] bool   -- only when a 2-D mask was passed
  cache_context_before/after.npy   -- cache[3] (last 2 tokens) if present
  cache_conv_before/after.npy      -- cache[2] (short-conv state) if present
meta.json records shapes, dtypes, and the effective RMSNorm convention.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = "/Users/vincent/.local/share/opencode/worktree/dd9a0e/cpp-engine-qdense"
VENV_SITE = "/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages"
for p in (REPO, VENV_SITE):
    if p not in sys.path:
        sys.path.insert(0, p)

import mlx.core as mx  # noqa: E402


def _np(t, dtype=np.float32):
    if t is None:
        return None
    return np.array(mx.astype(t, mx.float32) if t.dtype in (mx.bfloat16, mx.float16) else t,
                    dtype=dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=3)
    args = ap.parse_args()

    from veystrix.backends.flashnext import FlashNextBackend
    from mlx_vlm.models.qwen4_exp import language as q4lang

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the backend normally so loader.py installs all production patches
    # (patch_rmsnorm, compile glue, ngram streaming swap, quantized KV, ...).
    os.environ.setdefault("VEYSTRIX_SAMPLING_GREEDY", "1")
    backend = FlashNextBackend(model_path=args.checkpoint)

    calls: list[dict] = []
    original = q4lang.Qwen4ExpPLELayer.__call__

    def capturing(self, hidden_states, input_ids, cache, mask):
        layer_idx = getattr(getattr(self, "ple_embedding", None), "layer_idx", None)
        # Capture only layer 1 (the sole PLE layer in this checkpoint).
        if layer_idx not in (None, 1):
            return original(self, hidden_states, input_ids, cache, mask)
        idx = len(calls)
        tag = "prefill" if input_ids.shape[1] > 1 else "decode"
        cd = out_dir / f"{tag}_{idx:03d}"
        cd.mkdir(parents=True, exist_ok=True)

        def cache_slot(i):
            try:
                return None if cache is None or cache[i] is None else cache[i]
            except (IndexError, TypeError):
                return None

        ctx_before = cache_slot(3)
        conv_before = cache_slot(2)
        mx.eval(hidden_states, input_ids)
        out = original(self, hidden_states, input_ids, cache, mask)
        mx.eval(out)
        ctx_after = cache_slot(3)
        conv_after = cache_slot(2)

        np.save(cd / "hidden_states.npy", _np(hidden_states))
        np.save(cd / "input_ids.npy", np.array(input_ids, dtype=np.int64))
        np.save(cd / "ple_output.npy", _np(out))
        if mask is not None and isinstance(mask, mx.array) and mask.ndim == 2:
            np.save(cd / "mask.npy", np.array(mask, dtype=np.bool_))
        for name, t in (("cache_context_before", ctx_before),
                        ("cache_context_after", ctx_after)):
            if t is not None:
                np.save(cd / f"{name}.npy", np.array(t, dtype=np.int64))
        for name, t in (("cache_conv_before", conv_before),
                        ("cache_conv_after", conv_after)):
            if t is not None:
                np.save(cd / f"{name}.npy", _np(t))

        calls.append({
            "idx": idx,
            "phase": tag,
            "dir": cd.name,
            "hidden_shape": list(hidden_states.shape),
            "hidden_dtype": str(hidden_states.dtype),
            "input_ids_shape": list(input_ids.shape),
            "input_ids_dtype": str(input_ids.dtype),
            "input_ids_first8": np.array(input_ids).reshape(-1)[:8].tolist(),
            "output_shape": list(out.shape),
            "output_dtype": str(out.dtype),
            "mask_shape": list(mask.shape) if (mask is not None and isinstance(mask, mx.array)) else None,
            "ple_layer_idx": layer_idx,
            "norm_one_centered": getattr(self.norm_key, "_flashnext_one_centered", None),
        })
        return out

    q4lang.Qwen4ExpPLELayer.__call__ = capturing
    try:
        rng = np.random.default_rng(20261117)  # same prompt as capture_real_reference.py
        prompt = rng.integers(0, 200000, size=args.prompt_tokens).tolist()
        backend.append_tokens(prompt)
        tokens = []
        _text, _stats = backend.generate(args.tokens,
                                         on_decode_token=lambda t, _p: tokens.append(int(t)))
    finally:
        q4lang.Qwen4ExpPLELayer.__call__ = original

    meta = {
        "checkpoint": args.checkpoint,
        "prompt_tokens": prompt,
        "generated_tokens": tokens,
        "ple_calls": calls,
        "note": "Production patches active; norm_one_centered=True means gain=weight (no +1).",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"captured {len(calls)} PLE calls to {out_dir}")
    for c in calls:
        print(f"  {c['dir']}: hidden={c['hidden_shape']} ids={c['input_ids_shape']} "
              f"out={c['output_shape']} one_centered={c['norm_one_centered']}")


if __name__ == "__main__":
    main()
