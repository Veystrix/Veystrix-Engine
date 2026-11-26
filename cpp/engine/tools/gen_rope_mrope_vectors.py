#!/usr/bin/env python3
"""Synthetic Python-oracle vectors for the **full-attention** RoPE config.

This generator is deliberately scoped to the config used by
`full_attention_dense` (real_checkpoint_generate.cpp ~line 521-522):

    rope_dim        = 64
    rope_base       = 10_000_000
    rope_style      = "interleaved"
    rope_mrope      = [11, 11, 10]
    rope_scaling    = 1.0
    max_pos         = 262144

The expected q/k are produced by the REAL production rotary object,
``mlx_vlm.models.rope_utils.MRoPERotaryEmbedding`` (the base class of
``Qwen3_5RotaryEmbedding`` used by qwen4_exp/language.py).  With
``style="interleaved"`` and Metal present, ``rot.fused_apply`` is True and
``apply_rotary`` dispatches to the hand-written Metal MRoPE kernel in
``rope_utils._mrope_apply_kernel`` -- i.e. the production ground truth that the
C++ full-attention path must match.

Position ids are text-like 2-D ``[B, S]`` with the same position on every MRoPE
axis (so the interleaved selector does not change the numeric result; it is
still dumped so the C++ fused API can be exercised verbatim).

Generated files, per case ``<out>/<name>/``:

    q.npy, k.npy                 float32 [B, H, S, D]
    position_ids.npy             int32   [B, S]
    inv_freq.npy                 float32 [D/2]
    position_selector.npy        int32   [D/2]
    q_out.npy, k_out.npy         float32 [B, H, S, D]  (Python fused oracle)
    meta.json

``<out>/cases.json`` lists the case directory names.

Usage:
    .venv/bin/python cpp/engine/tools/gen_rope_mrope_vectors.py \
        --out cpp/engine/tests/vectors_rope_mrope
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, ".venv/lib/python3.12/site-packages")
from mlx_vlm.models.rope_utils import MRoPERotaryEmbedding  # noqa: E402

# ---- The real full-attention config (single source of truth for this tool). --
CFG = dict(
    dim=64,
    base=10_000_000.0,
    mrope_section=(11, 11, 10),
    style="interleaved",
    attention_scaling=1.0,
    max_position_embeddings=262144,
)
B = 1
H = 4  # indexer query heads; RoPE is per-head and head-count agnostic
D = 64  # rotary_dim == head_dim: no pass-through channels in this config


def _make_rot():
    rot = MRoPERotaryEmbedding(
        CFG["dim"],
        max_position_embeddings=CFG["max_position_embeddings"],
        base=CFG["base"],
        mrope_section=list(CFG["mrope_section"]),
        attention_scaling=CFG["attention_scaling"],
        style=CFG["style"],
    )
    if not rot.fused_apply:
        raise RuntimeError(
            "fused-Metal MRoPE unavailable: expected rot.fused_apply=True "
            "(Metal missing or style not interleaved)"
        )
    return rot


def _make_inputs(seed: int, S: int, start_pos: int):
    """Deterministic random q/k and text-like position ids."""
    rng = np.random.default_rng(seed)
    q = mx.array(rng.normal(size=(B, H, S, D)).astype(np.float32))
    k = mx.array(rng.normal(size=(B, H, S, D)).astype(np.float32))
    positions = mx.array(
        (np.arange(S, dtype=np.int32) + np.int32(start_pos))[None, :]
    )
    return q, k, positions


def _bits(a: mx.array) -> np.ndarray:
    """Raw float32 bits as uint32, for reproducibility checks."""
    return np.asarray(mx.view(mx.reshape(a, (-1,)), mx.uint32))


def emit_case(out: Path, name: str, *, S: int, seed: int,
              start_pos: int) -> dict:
    d = out / name
    d.mkdir(parents=True, exist_ok=True)

    rot = _make_rot()
    q, k, positions = _make_inputs(seed, S, start_pos)
    inv_freq = rot.inv_freq
    selector = rot.position_selector

    # ---- REAL production fused apply (the Python oracle). ----
    q_out, k_out = rot.apply_rotary(q, k, positions, unsqueeze_dim=1)
    mx.eval(q, k, positions, inv_freq, selector, q_out, k_out)
    if not (rot._compiled_apply and positions.ndim in rot._compiled_apply):
        raise RuntimeError(
            "apply_rotary did not take the compiled fused-Metal branch"
        )

    # ---- Reproducibility: fresh embedding + fresh RNG must give same bits. ----
    rot2 = _make_rot()
    q2, k2, pos2 = _make_inputs(seed, S, start_pos)
    q2_out, k2_out = rot2.apply_rotary(q2, k2, pos2, unsqueeze_dim=1)
    mx.eval(q2_out, k2_out)
    for tag, a, b in (("q", q, q2), ("k", k, k2),
                      ("q_out", q_out, q2_out), ("k_out", k_out, k2_out),
                      ("position_ids", positions, pos2)):
        if not np.array_equal(_bits(a), _bits(b)):
            raise RuntimeError(f"non-reproducible array: {tag}")
    if not np.array_equal(_bits(inv_freq), _bits(rot2.inv_freq)):
        raise RuntimeError("non-reproducible inv_freq")
    if not np.array_equal(_bits(selector), _bits(rot2.position_selector)):
        raise RuntimeError("non-reproducible position_selector")

    # ---- Dump inputs + oracle outputs. ----
    for arr_name, arr in {
        "q": q, "k": k,
        "position_ids": positions,
        "inv_freq": inv_freq,
        "position_selector": selector,
        "q_out": q_out, "k_out": k_out,
    }.items():
        mx.save(str(d / f"{arr_name}.npy"), arr)

    meta = {
        "dtype": "float32",
        "stored_as_uint16": False,
        "position_ndim": int(positions.ndim),
        "rotary_dim": CFG["dim"],
        "base": CFG["base"],
        "mrope_section": list(CFG["mrope_section"]),
        "attention_scaling": CFG["attention_scaling"],
        "style": CFG["style"],
        "B": B, "H": H, "S": S, "D": D,
        "max_position_embeddings": CFG["max_position_embeddings"],
        "seed": seed,
        "start_pos": start_pos,
        "fused_apply": bool(rot.fused_apply),
        "mlx_version": mx.__version__,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"dir": name, **meta}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors_rope_mrope")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = [
        emit_case(out, "case_s1", S=1, seed=20261118, start_pos=5),
        emit_case(out, "case_s8", S=8, seed=20261119, start_pos=13),
    ]
    (out / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print("wrote", len(cases), "mrope RoPE oracle cases:",
          ", ".join(c["dir"] for c in cases))
    print("config:", json.dumps(CFG, sort_keys=True))
    print("oracle: MRoPERotaryEmbedding.apply_rotary (fused_apply=True, "
          "compiled Metal kernel); reproducibility verified bit-for-bit")


if __name__ == "__main__":
    main()
