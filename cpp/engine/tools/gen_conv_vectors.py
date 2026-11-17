#!/usr/bin/env python3
"""Generate bit-exact vectors for the GDN causal-conv1d DECODE chain.

This imports the REAL Python helpers used on the production decode path:

  * ``mlx_vlm.models.qwen3_5.language._qwen3_5_decode_depthwise_conv``
    (the depthwise-conv + cast that ``_causal_conv1d_decode`` dispatches to
    when ``S == 1`` and the conv weight is bf16/fp16), and
  * ``mlx.nn.silu`` (applied by ``Qwen3_5GatedDeltaNet.__call__``).

and mirrors the full decode chain exactly:

    weight = conv1d.weight[:, :, 0].T.astype(mx.float32)   # [K, C] f32
    conv_out = _qwen3_5_decode_depthwise_conv(conv_input, weight)
    silu_out = nn.silu(conv_out)

Outputs (via ``mx.save``), in ``--out``:

    conv_input.npy     [B, K, C] in the half dtype (conv_state ++ mixed_qkv)
    conv_weight.npy    [C, K, 1] in the half dtype (raw nn.Conv1d.weight)
    expected.npy       [B, 1, C] pre-silu depthwise-conv output (half dtype)
    expected_silu.npy  [B, 1, C] post-silu output (half dtype)
    meta.json          shapes / dtypes

Usage:
    .venv/bin/python cpp/engine/tools/gen_conv_vectors.py \
        --out cpp/engine/tests/vectors/conv --dtype bfloat16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Import the actual Python engine so the vectors come from the real helpers.
sys.path.insert(0, ".venv/lib/python3.12/site-packages")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_vlm.models.qwen3_5.language import (  # noqa: E402
    _qwen3_5_decode_depthwise_conv,
)


def make_case(B: int, K: int, C: int, dtype, seed: int):
    """Build one deterministic decode case and its expected outputs."""
    rng = np.random.default_rng(seed)
    ci = (rng.standard_normal((B, K, C)) * 0.5).astype(np.float32)
    cw = (rng.standard_normal((C, K, 1)) * 0.1).astype(np.float32)

    conv_input = mx.array(ci).astype(dtype)   # [B, K, C]
    conv_weight = mx.array(cw).astype(dtype)  # [C, K, 1] raw conv1d weight

    # Exactly the prep in Qwen3_5GatedDeltaNet._causal_conv1d_decode.
    weight = conv_weight[:, :, 0].T.astype(mx.float32)  # [K, C] f32
    mx.eval(weight)

    # The real (mx.compile'd) depthwise-conv helper.
    conv_out = _qwen3_5_decode_depthwise_conv(conv_input, weight)  # [B,1,C]
    silu_out = nn.silu(conv_out)  # nn.silu is itself mx.compile'd
    mx.eval(conv_input, conv_weight, conv_out, silu_out)

    return conv_input, conv_weight, conv_out, silu_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors/conv")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--kernel", type=int, default=4, help="linear_conv_kernel_dim")
    ap.add_argument("--conv-dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261117)
    args = ap.parse_args()

    dtype = getattr(mx, args.dtype)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    conv_input, conv_weight, conv_out, silu_out = make_case(
        args.batch, args.kernel, args.conv_dim, dtype, args.seed
    )

    arrays = {
        "conv_input": conv_input,
        "conv_weight": conv_weight,
        "expected": conv_out,
        "expected_silu": silu_out,
    }
    meta = {"dtype": args.dtype}
    for name, arr in arrays.items():
        mx.save(str(out / f"{name}.npy"), arr)
        meta[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype).split(".")[-1]}
        print(f"wrote {out / (name + '.npy')} shape={arr.shape} dtype={arr.dtype}")

    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out / 'meta.json'}")


if __name__ == "__main__":
    main()
