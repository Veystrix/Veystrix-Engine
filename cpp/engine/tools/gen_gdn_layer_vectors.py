#!/usr/bin/env python3
"""Generate bit-exact vectors for a FULL single GDN linear-attention layer
(the Qwen4ExpGatedDeltaNet.__call__ DECODE step) by running the REAL mlx_vlm
module, so the C++ composition in gdn_layer.cpp can be checked byte-for-byte.

The real layer is built from the actual ``Qwen4ExpGatedDeltaNet`` class with a
permissive (tiny) config. Dk/Dv are kept as multiples of 32 so the production
fused Metal kernel (Dk/32 threads per tile) is used. Every weight and cache
tensor is seeded deterministically, then one decode step is executed:

    out = model(x, cache=[conv_state, rec_state])

DTYPE: x and conv_state are float32. With bf16 inputs the Python
``_normalize_qk`` operates in bf16 and the fused kernel sees bf16 q/k, while
the already-verified C++ primitive ``gdn_normalize_qk`` is f32-only; using f32
inputs keeps the entire post-projection path f32 so EVERY reused primitive is
bit-exact. Linear / conv / out_proj weights stay bf16 (production dtype) and
MLX promotes the matmul to f32 exactly as nn.Linear does.

Outputs (mx.save) in ``--out``:
    x.npy, w_qkv.npy, w_z.npy, w_b.npy, w_a.npy, conv_w.npy,
    A_log.npy, dt_bias.npy, norm_w.npy, w_out.npy,
    conv_state.npy, rec_state.npy,              (inputs / cache)
    expected_output.npy, expected_conv_state.npy, expected_rec_state.npy,
    meta.json                                   (shapes / dtypes / config)

Usage:
    .venv/bin/python cpp/engine/tools/gen_gdn_layer_vectors.py \
        --out cpp/engine/tests/vectors_gdn_layer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, ".venv/lib/python3.12/site-packages")
import mlx.core as mx  # noqa: E402
from mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    Qwen4ExpGatedDeltaNet,
)


def make_case(cfg: SimpleNamespace, seed: int):
    H = cfg.hidden_size
    Hk, Hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    Dk, Dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    K = cfg.linear_conv_kernel_dim
    key_dim, value_dim = Hk * Dk, Hv * Dv
    conv_dim = 2 * key_dim + value_dim
    B = 1

    rng = np.random.default_rng(seed)
    mxrand = lambda shape, scale=0.1, dtype=mx.float32: mx.array(  # noqa: E731
        (rng.standard_normal(shape) * scale).astype(np.float32)
    ).astype(dtype)

    # Real model, then overwrite every parameter deterministically.
    model = Qwen4ExpGatedDeltaNet(cfg)
    model.in_proj_qkv.weight = mxrand((conv_dim, H), dtype=mx.bfloat16)
    model.in_proj_z.weight = mxrand((value_dim, H), dtype=mx.bfloat16)
    model.in_proj_b.weight = mxrand((Hv, H), dtype=mx.bfloat16)
    model.in_proj_a.weight = mxrand((Hv, H), dtype=mx.bfloat16)
    model.conv1d.weight = mxrand((conv_dim, K, 1), dtype=mx.bfloat16)
    model.A_log = mxrand((Hv,), scale=1.0)                 # f32
    model.dt_bias = mxrand((Hv,), scale=0.5)               # f32
    model.norm.weight = mx.array(np.ones(Dv, dtype=np.float32))  # f32 ones
    model.out_proj.weight = mxrand((H, value_dim), dtype=mx.bfloat16)

    # Decode cache: [B, K-1, conv_dim] (f32 to match x) and [B,Hv,Dv,Dk] f32.
    conv_state = mxrand((B, K - 1, conv_dim), scale=0.1)
    rec_state = mxrand((B, Hv, Dv, Dk), scale=0.05)
    x = mxrand((B, 1, H), scale=0.1)

    cache = [conv_state, rec_state]
    out = model(x, cache=cache)
    mx.eval(out, cache[0], cache[1])

    arrays = {
        "x": x,
        "w_qkv": model.in_proj_qkv.weight,
        "w_z": model.in_proj_z.weight,
        "w_b": model.in_proj_b.weight,
        "w_a": model.in_proj_a.weight,
        "conv_w": model.conv1d.weight,
        "A_log": model.A_log,
        "dt_bias": model.dt_bias,
        "norm_w": model.norm.weight,
        "w_out": model.out_proj.weight,
        "conv_state": conv_state,
        "rec_state": rec_state,
        "expected_output": out,
        "expected_conv_state": cache[0],
        "expected_rec_state": cache[1],
    }
    meta = {
        "config": {
            "hidden_size": H,
            "key_dim": key_dim,
            "value_dim": value_dim,
            "num_k_heads": Hk,
            "num_v_heads": Hv,
            "head_k_dim": Dk,
            "head_v_dim": Dv,
            "conv_kernel": K,
            "eps": cfg.rms_norm_eps,
            "output_activation": cfg.output_gate_type,
            "batch": B,
        },
        "seed": seed,
    }
    return arrays, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors_gdn_layer")
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--hk", type=int, default=2, help="num key heads")
    ap.add_argument("--hv", type=int, default=4, help="num value heads")
    ap.add_argument("--dk", type=int, default=32, help="key head dim (mult of 32)")
    ap.add_argument("--dv", type=int, default=32, help="value head dim (mult of 32)")
    ap.add_argument("--kernel", type=int, default=4, help="conv kernel size K")
    args = ap.parse_args()

    cfg = SimpleNamespace(
        hidden_size=args.hidden,
        linear_num_value_heads=args.hv,
        linear_num_key_heads=args.hk,
        linear_key_head_dim=args.dk,
        linear_value_head_dim=args.dv,
        linear_conv_kernel_dim=args.kernel,
        rms_norm_eps=1e-6,
        output_gate_type="sigmoid",
        hidden_act="silu",
    )
    assert args.hv % args.hk == 0, "Hv must be a multiple of Hk"
    assert args.dk % 32 == 0, "Dk must be a multiple of 32 for the kernel"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arrays, meta = make_case(cfg, args.seed)

    for name, arr in arrays.items():
        mx.save(str(out / f"{name}.npy"), arr)
        meta[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype).split(".")[-1]}
        print(f"wrote {out / (name + '.npy')} shape={arr.shape} dtype={arr.dtype}")

    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out / 'meta.json'}")


if __name__ == "__main__":
    main()
