#!/usr/bin/env python3
"""Generate bit-exact parity vectors for the FULL QSA indexer path.

This drives the *real* production modules from mlx_vlm:

  .venv/lib/python3.12/site-packages/mlx_vlm/models/qwen4_exp/language.py
    class Qwen4ExpQSAIndexer.__call__   (lines 306-380)
    class QSAKVCache.update_indexer     (lines 54-63)
    _append_indexer_positions           (lines 21-38)
  .venv/lib/python3.12/site-packages/mlx_vlm/models/qwen3_5/language.py
    class Qwen3_5RotaryEmbedding
  .venv/lib/python3.12/site-packages/mlx_vlm/models/rope_utils.py
    class MRoPERotaryEmbedding.apply_rotary / compute_inv_freq / ...

For every case we save ALL inputs, the reference ``qsa_mask`` (selected_tokens)
and the *updated* indexer cache state, so the C++ port can be checked
byte-for-byte (uint32 view) against production.

NOTE ON THE FUSED RoPE PATH
---------------------------
On Apple Silicon MLX builds ``MRoPERotaryEmbedding.fused_apply`` is True and
``apply_rotary`` takes a hand-written Metal kernel that calls ``metal::cos`` /
``metal::sin``.  That path differs from the eager ``mx.cos`` / ``mx.sin`` path
by ~1 float32 ULP.  To obtain *stable, reproducible* bit-exact vectors we force
``rot.fused_apply = False`` here, i.e. the real eager branch of the real
``apply_rotary``.  The C++ port implements that eager math exactly (verified
bit-identical for float32 and float16).  Production's fused path changes only
the low bit of the rotated values; the downstream boolean QSA mask is unchanged
for the well-separated scores used in these tests.  See qsa_full_port.md.

Usage:
  .venv/bin/python cpp/engine/tools/gen_qsa_full_vectors.py \
      --out cpp/engine/tests/vectors_qsa_full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

sys.path.insert(0, ".venv/lib/python3.12/site-packages")
from mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    Qwen4ExpQSAIndexer,
    QSAKVCache,
)
from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding  # noqa: E402


def save(path, arr):
    mx.save(str(path), arr)


def emit_case(out: Path, name: str, *, dtype, B, S, past_len, Hq, Hk, D,
              ratio, budget, eps, rotary_dim, base, mrope_section,
              attention_scaling, max_pos, seed):
    rng = np.random.default_rng(seed)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)

    H = 3 * D + 5  # arbitrary hidden width
    block_topk = budget // ratio

    # --- real rotary embedding (Qwen3_5RotaryEmbedding / MRoPERotaryEmbedding).
    rot = Qwen3_5RotaryEmbedding(
        rotary_dim,
        max_position_embeddings=max_pos,
        base=base,
        mrope_section=list(mrope_section),
    )
    rot.attention_scaling = attention_scaling
    # Force the deterministic eager branch (see module docstring).
    rot.fused_apply = False

    # --- real QSA indexer, weights set to random but well-conditioned values.
    cfg = SimpleNamespace(
        indexer_n_heads=Hq,
        indexer_kv_heads=Hk,
        indexer_head_dim=D,
        indexer_budget=budget,
        indexer_compress_ratio=ratio,
        hidden_size=H,
        rms_norm_eps=eps,
    )
    indexer = Qwen4ExpQSAIndexer(cfg, rot)

    proj_weight = mx.array(
        (rng.normal(size=((Hq + Hk) * D, H)).astype(np.float32) * 0.2)
    ).astype(dtype)
    q_weight = mx.array(rng.normal(size=(D,)).astype(np.float32) * 0.1)
    k_weight = mx.array(rng.normal(size=(D,)).astype(np.float32) * 0.1)
    indexer.index_qk_proj.weight = proj_weight
    indexer.q_layernorm.weight = q_weight
    indexer.k_layernorm.weight = k_weight

    # --- inputs for one decode step.
    x = mx.array(rng.normal(size=(B, S, H)).astype(np.float32) * 0.4).astype(dtype)

    # Past cache state: random keys + monotonically increasing text positions.
    past_keys = mx.array(
        rng.normal(size=(B, past_len, D)).astype(np.float32) * 0.3
    ).astype(dtype)
    past_pos = mx.array(
        np.tile(np.arange(past_len, dtype=np.int32)[None, :], (B, 1))
    )

    # New positions for the current step (2-D text positions).
    position_ids = mx.array(
        np.arange(past_len, past_len + S, dtype=np.int32)[None, :]
        + np.zeros((B, 1), dtype=np.int32)
    )

    cache = QSAKVCache()
    cache.index_keys = past_keys
    cache.index_position_ids = past_pos
    cache.offset = past_len  # __call__ reads past_len from cache.offset

    # --- real decode step.
    mask = indexer(x, cache, position_ids)
    mx.eval(x, proj_weight, q_weight, k_weight, past_keys, past_pos,
            position_ids)

    returned_none = mask is None
    arrays = {
        "x": x,
        "proj_weight": proj_weight,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "past_index_keys": past_keys,
        "past_index_positions": past_pos,
        "position_ids": position_ids,
        "expected_index_keys": cache.index_keys,
        "expected_index_positions": cache.index_position_ids,
    }
    if not returned_none:
        mx.eval(mask)
        arrays["expected_mask"] = mask
    for k, v in arrays.items():
        save(d / f"{k}.npy", v)

    key_len = past_len + S
    meta = {
        "n_heads": Hq,
        "kv_heads": Hk,
        "head_dim": D,
        "compress_ratio": ratio,
        "token_budget": budget,
        "block_topk": block_topk,
        "eps": eps,
        "B": B,
        "S": S,
        "H": H,
        "past_len": past_len,
        "key_len": key_len,
        "max_complete_blocks": key_len // ratio,
        "rotary_dim": rotary_dim,
        "rope_base": float(base),
        "mrope_section": list(mrope_section),
        "rope_style": rot.style,
        "attention_scaling": attention_scaling,
        "max_position_embeddings": max_pos,
        "returned_none": returned_none,
        "dtype": str(dtype),
        "mlx_version": mx.__version__,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"dir": name, **meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors_qsa_full")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    common = dict(
        Hq=2, Hk=1, D=128, ratio=4, budget=16, eps=1e-6,
        rotary_dim=32, base=10_000_000, mrope_section=[11, 11, 10],
        attention_scaling=1.0, max_pos=2048,
    )

    cases = [
        # None path: max_complete_blocks (=2) <= block_topk (=4).
        emit_case(out, "case0_none", dtype=mx.float32,
                  B=1, S=1, past_len=8, seed=101, **common),
        # Selection path: 25 keys -> 6 complete blocks > block_topk 4.
        emit_case(out, "case1_select", dtype=mx.float32,
                  B=1, S=1, past_len=24, seed=202, **common),
        # Batch/seq > 1 selection path: 22 keys -> 5 complete blocks.
        emit_case(out, "case2_select_batch2", dtype=mx.float32,
                  B=2, S=2, past_len=20, seed=303, **common),
        # float16 selection path (dtype casting in RoPE + scoring).
        emit_case(out, "case3_select_float16", dtype=mx.float16,
                  B=1, S=1, past_len=24, seed=404, **common),
        # P5b: long-context selection regime (past_len past the 2048 dense
        # threshold). 2061 keys -> 515 complete blocks > block_topk 4, so the
        # real indexer selects blocks exactly as it does on a real long prompt.
        emit_case(out, "case4_long_select", dtype=mx.float32,
                  B=1, S=1, past_len=2060, seed=505,
                  Hq=2, Hk=1, D=128, ratio=4, budget=16, eps=1e-6,
                  rotary_dim=32, base=10_000_000, mrope_section=[11, 11, 10],
                  attention_scaling=1.0, max_pos=4096),
    ]
    (out / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print("wrote", len(cases), "full QSA cases:", ", ".join(c["dir"] for c in cases))


if __name__ == "__main__":
    main()
