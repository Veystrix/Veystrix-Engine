#!/usr/bin/env python3
"""Generate bit-exact parity vectors for PRODUCTION's fused-Metal MRoPE path.

This drives the REAL production rotary embedding from mlx_vlm with
``fused_apply = True`` (the default on Apple Silicon), so the expected outputs
come from the hand-written Metal kernel in rope_utils._mrope_apply_kernel:

  .venv/lib/python3.12/site-packages/mlx_vlm/models/rope_utils.py
    MRoPERotaryEmbedding.apply_rotary            lines 1244-1287
    _compiled_mrope_apply / _fast_mrope_apply    lines 1007-1025, 693-742
    _mrope_apply_kernel                          lines 568-652
  .venv/lib/python3.12/site-packages/mlx_vlm/models/qwen3_5/language.py
    Qwen3_5RotaryEmbedding (style="interleaved")

For each case we save the real kernel inputs (q, k, position_ids, inv_freq,
position_selector) and the real fused outputs (q_out, k_out), so the C++ port
can be checked byte-for-byte (uint32 view) against production.

bfloat16 note: numpy has no bfloat16, so mx.save cannot write it directly.
For bfloat16 cases the dtype-varying arrays are bit-cast to uint16 via
``mx.view(a, mx.uint16)`` before saving; meta.json sets ``stored_as_uint16``
so the C++ test can reinterpret the bits with ``mx::view(..., mx::bfloat16)``.

Usage:
  .venv/bin/python cpp/engine/tools/gen_rope_fused_vectors.py \
      --out cpp/engine/tests/vectors_rope_fused
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, ".venv/lib/python3.12/site-packages")
from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding  # noqa: E402


def save_bits(path: Path, arr: mx.array, stored_as_uint16: bool):
    """Save an array, bit-casting bfloat16 to uint16 when needed."""
    if stored_as_uint16:
        arr = mx.view(arr, mx.uint16)
    mx.save(str(path), arr)


def emit_case(out: Path, name: str, *, dtype, position_ndim, seed,
              rotary_dim=32, base=10_000_000, mrope_section=(11, 11, 10),
              attention_scaling=1.0, B=2, H=3, T=4, D=128,
              max_pos=2048) -> dict:
    rng = np.random.default_rng(seed)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)

    # --- REAL production rotary embedding; DO NOT force fused_apply=False. ---
    rot = Qwen3_5RotaryEmbedding(
        rotary_dim,
        max_position_embeddings=max_pos,
        base=base,
        mrope_section=list(mrope_section),
    )
    rot.attention_scaling = attention_scaling
    assert rot.fused_apply, "expected the fused-Metal MRoPE path (Metal missing?)"

    # Real-shaped q/k (the indexer path uses [B, H, T, D] with D=head_dim).
    q = mx.array(
        (rng.normal(size=(B, H, T, D)).astype(np.float32) * 0.4)
    ).astype(dtype)
    k = mx.array(
        (rng.normal(size=(B, H, T, D)).astype(np.float32) * 0.4)
    ).astype(dtype)

    # Real position ids: text 2-D [B,T] or MRoPE 3-D [axis,B,T].
    if position_ndim == 2:
        positions = mx.array(
            np.arange(T, dtype=np.int32)[None, :]
            + np.arange(B, dtype=np.int32)[:, None]
        )
    else:
        positions = mx.array(
            np.stack([
                np.arange(T, dtype=np.int32)[None, :]
                + np.arange(B, dtype=np.int32)[:, None]
                + axis * 7
                for axis in range(3)
            ], axis=0)
        )
    assert positions.ndim == position_ndim

    # The exact kernel inputs. inv_freq is float32, selector is int32.
    inv_freq = rot.inv_freq
    position_selector = rot.position_selector

    # --- REAL fused apply_rotary (dispatches to the Metal kernel). ---
    q_out, k_out = rot.apply_rotary(q, k, positions, unsqueeze_dim=1)
    mx.eval(q, k, positions, inv_freq, position_selector, q_out, k_out)
    # Confirm the compiled fused branch was actually exercised.
    assert rot._compiled_apply and (position_ndim in rot._compiled_apply), \
        "fused apply_rotary did not use the compiled Metal kernel"

    stored_as_uint16 = dtype == mx.bfloat16
    for arr_name, arr in {
        "q": q, "k": k, "q_out": q_out, "k_out": k_out,
    }.items():
        save_bits(d / f"{arr_name}.npy", arr, stored_as_uint16)
    mx.save(str(d / "position_ids.npy"), positions)
    mx.save(str(d / "inv_freq.npy"), inv_freq)
    mx.save(str(d / "position_selector.npy"), position_selector)

    meta = {
        "dtype": "bfloat16" if dtype == mx.bfloat16 else
                 ("float16" if dtype == mx.float16 else "float32"),
        "stored_as_uint16": stored_as_uint16,
        "position_ndim": position_ndim,
        "rotary_dim": rotary_dim,
        "base": float(base),
        "mrope_section": list(mrope_section),
        "attention_scaling": attention_scaling,
        "B": B, "H": H, "T": T, "D": D,
        "max_position_embeddings": max_pos,
        "fused_apply": bool(rot.fused_apply),
        "mlx_version": mx.__version__,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"dir": name, **meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors_rope_fused")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = []
    for dtype, tag in ((mx.float32, "f32"),
                       (mx.float16, "f16"),
                       (mx.bfloat16, "bf16")):
        for pdim in (2, 3):
            cases.append(emit_case(
                out, f"case_{tag}_pos{pdim}", dtype=dtype,
                position_ndim=pdim, seed=1000 + len(cases) * 17,
            ))

    (out / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print("wrote", len(cases), "fused-RoPE cases:",
          ", ".join(c["dir"] for c in cases))


if __name__ == "__main__":
    main()
