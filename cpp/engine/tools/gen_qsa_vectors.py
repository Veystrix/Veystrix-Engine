#!/usr/bin/env python3
"""Generate bit-exact parity vectors for the QSA indexer primitive.

The reference path below mirrors, op for op, the pre-RoPE portion of
`Qwen4ExpQSAIndexer.__call__` from
  .venv/lib/python3.12/site-packages/mlx_vlm/models/qwen4_exp/language.py
(the file backing `Qwen3.8-Flash-Next`).  The q/k layernorms use the REAL
`Qwen4ExpRMSNorm` class from that module (float32 compute, one-centered gain),
so the C++ port (cpp/engine/src/qsa.cpp) must match byte-for-byte.

RoPE / block selection are intentionally out of scope for this primitive; see
research_notes/Cerebras inference and IO splitting/qsa_port.md.

Usage:
  .venv/bin/python cpp/engine/tools/gen_qsa_vectors.py \
      --out cpp/engine/tests/vectors_qsa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, ".venv/lib/python3.12/site-packages")
from mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    Qwen4ExpRMSNorm,
)


def save(path, arr):
    mx.save(str(path), arr)


def reference(x, proj_weight, q_weight, k_weight, cfg):
    """Exact pre-RoPE op sequence from Qwen4ExpQSAIndexer.__call__."""
    B, S, _ = x.shape
    Hq, Hk = cfg["n_heads"], cfg["kv_heads"]
    D, ratio, eps = cfg["head_dim"], cfg["compress_ratio"], cfg["eps"]

    q_ln = Qwen4ExpRMSNorm(D, eps=eps)
    k_ln = Qwen4ExpRMSNorm(D, eps=eps)
    q_ln.weight = q_weight
    k_ln.weight = k_weight

    # qk = self.index_qk_proj(x).reshape(B, S, Hq + Hk, D)
    qk = (x @ proj_weight.T).reshape(B, S, Hq + Hk, D)
    query = qk[:, :, :Hq]
    raw_keys = qk[:, :, Hq:].squeeze(2)
    query = q_ln(query).transpose(0, 2, 1, 3)

    key_len = raw_keys.shape[1]
    max_complete_blocks = key_len // ratio
    complete_key_len = max_complete_blocks * ratio
    pooled = raw_keys[:, :complete_key_len].reshape(
        B, max_complete_blocks, ratio, D
    )
    pooled = mx.expand_dims(
        k_ln(mx.mean(pooled.astype(mx.float32), axis=2).astype(raw_keys.dtype)),
        axis=1,
    )
    mx.eval(query, pooled)
    return query, pooled


def emit_case(out: Path, name: str, dtype, B, S, Hq, Hk, D, ratio, eps, seed):
    rng = np.random.default_rng(seed)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)

    H = 3 * D + 7  # arbitrary input feature width
    x = mx.array(rng.normal(size=(B, S, H)).astype(np.float32) * 0.4).astype(dtype)
    proj_weight = mx.array(
        rng.normal(size=((Hq + Hk) * D, H)).astype(np.float32) * 0.2
    ).astype(dtype)
    # Qwen4 RMSNorm weights are "one-centered": checkpoint values are small,
    # the effective gain is (1.0 + weight). Negative small values included.
    q_weight = mx.array(rng.normal(size=(D,)).astype(np.float32) * 0.1)
    k_weight = mx.array(rng.normal(size=(D,)).astype(np.float32) * 0.1)

    cfg = {
        "n_heads": Hq,
        "kv_heads": Hk,
        "head_dim": D,
        "compress_ratio": ratio,
        "eps": eps,
    }
    query, pooled_keys = reference(x, proj_weight, q_weight, k_weight, cfg)

    arrays = {
        "x": x,
        "proj_weight": proj_weight,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "query": query,
        "pooled_keys": pooled_keys,
    }
    for k, v in arrays.items():
        save(d / f"{k}.npy", v)
    token_budget = 4 * ratio  # arbitrary but consistent block budget
    meta = {
        **cfg,
        "B": B,
        "S": S,
        "H": H,
        "blocks": S // ratio,
        "token_budget": token_budget,
        "block_topk": token_budget // ratio,
        "dtype": str(dtype),
        "mlx_version": mx.__version__,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"dir": name, **meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors_qsa")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = [
        emit_case(out, "case0_float32_small", mx.float32, 1, 8, 2, 1, 16, 2, 1e-6, 1234),
        emit_case(out, "case1_bfloat16_decode", mx.bfloat16, 1, 16, 4, 1, 32, 4, 1e-6, 77),
        emit_case(out, "case2_float16_batch2", mx.float16, 2, 12, 3, 1, 24, 3, 1e-6, 99),
    ]
    (out / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print("wrote", len(cases), "QSA indexer cases:", ", ".join(c["dir"] for c in cases))


if __name__ == "__main__":
    main()
