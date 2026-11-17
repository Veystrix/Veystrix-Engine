#!/usr/bin/env python3
"""Generate bit-exact PREFILL (S>1) vectors for the GDN linear-attention layer.

Runs the REAL ``Qwen4ExpGatedDeltaNet`` (eval mode, seeded, small config) on the
M4 GPU so the production S>1 path is exercised exactly:

  * full-sequence projections over [B,S,H],
  * full depthwise ``self.conv1d`` over concat([conv_state, mixed_qkv]),
  * ``gated_delta_update`` -> ``gated_delta_kernel`` (the fused Metal kernel
    whose internal t-loop IS production prefill; verified: S in {2,5,16} call
    'kernel', never 'chunked'/'ops'),
  * gated RMSNorm + out_proj over all S tokens.

Cases: S in {2, 5, 16} (no mask — the normal single-prompt prefill) plus
S=5 with a bool SSM/padding mask (a couple of False positions) to cover the
masked Metal kernel and the pre-conv mixed_qkv masking.

DTYPE: x and conv/cache states are float32; weights stay bf16/f32 exactly as in
production. MLX promotes matmul/conv to f32 the same way nn.Linear/nn.Conv1d do.
This mirrors the S==1 generator (gen_gdn_layer_vectors.py) so every reused
primitive stays bit-exact.

Layout (--out):
  <out>/w_*.npy, A_log.npy, dt_bias.npy, norm_w.npy            (shared weights)
  <out>/meta.json
  <out>/<case>/x.npy, mask.npy, expected_output.npy,
                expected_conv_state.npy, expected_rec_state.npy

For unmasked cases mask.npy is an empty array (the C++ test treats size 0 as
"no mask" and passes nullptr, matching production's mask=None).

Usage:
    .venv/bin/python cpp/engine/tools/gen_prefill_vectors.py \
        --out cpp/engine/tests/vectors_prefill
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
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedDeltaNet  # noqa: E402


CASES = [
    ("s2", 2, None),
    ("s5", 5, None),
    ("s16", 16, None),
    ("s5_masked", 5, [True, True, False, True, True]),
]


def make_model_and_weights(cfg, seed):
    conv_dim = 2 * cfg.key_dim_total + cfg.value_dim_total
    rng = np.random.default_rng(seed)
    mxrand = lambda shape, scale=0.1, dtype=mx.float32: mx.array(  # noqa: E731
        (rng.standard_normal(shape) * scale).astype(np.float32)
    ).astype(dtype)

    model = Qwen4ExpGatedDeltaNet(cfg)
    model.in_proj_qkv.weight = mxrand((conv_dim, cfg.hidden_size), dtype=mx.bfloat16)
    model.in_proj_z.weight = mxrand((cfg.value_dim_total, cfg.hidden_size), dtype=mx.bfloat16)
    model.in_proj_b.weight = mxrand((cfg.num_v_heads, cfg.hidden_size), dtype=mx.bfloat16)
    model.in_proj_a.weight = mxrand((cfg.num_v_heads, cfg.hidden_size), dtype=mx.bfloat16)
    model.conv1d.weight = mxrand((conv_dim, cfg.linear_conv_kernel_dim, 1), dtype=mx.bfloat16)
    model.A_log = mxrand((cfg.num_v_heads,), scale=1.0)      # f32
    model.dt_bias = mxrand((cfg.num_v_heads,), scale=0.5)    # f32
    model.norm.weight = mx.array(np.ones(cfg.linear_value_head_dim, dtype=np.float32))
    model.out_proj.weight = mxrand((cfg.hidden_size, cfg.value_dim_total), dtype=mx.bfloat16)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors_prefill")
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--hk", type=int, default=2)
    ap.add_argument("--hv", type=int, default=4)
    ap.add_argument("--dk", type=int, default=32, help="mult of 32 for the Metal kernel")
    ap.add_argument("--dv", type=int, default=32)
    ap.add_argument("--kernel", type=int, default=4, help="conv kernel size K")
    args = ap.parse_args()

    mx.set_default_device(mx.gpu)
    assert mx.metal.is_available(), "these vectors require the M4 GPU Metal kernel path"
    assert args.hv % args.hk == 0 and args.dk % 32 == 0

    Hk, Hv = args.hk, args.hv
    Dk, Dv = args.dk, args.dv
    cfg = SimpleNamespace(
        hidden_size=args.hidden,
        linear_num_value_heads=Hv,
        linear_num_key_heads=Hk,
        linear_key_head_dim=Dk,
        linear_value_head_dim=Dv,
        linear_conv_kernel_dim=args.kernel,
        rms_norm_eps=1e-6,
        output_gate_type="sigmoid",
        hidden_act="silu",
        key_dim_total=Hk * Dk,
        value_dim_total=Hv * Dv,
        num_k_heads=Hk,
        num_v_heads=Hv,
    )

    model = make_model_and_weights(cfg, args.seed)
    conv_dim = 2 * cfg.key_dim_total + cfg.value_dim_total
    rng = np.random.default_rng(args.seed + 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shared = {
        "w_qkv": model.in_proj_qkv.weight,
        "w_z": model.in_proj_z.weight,
        "w_b": model.in_proj_b.weight,
        "w_a": model.in_proj_a.weight,
        "conv_w": model.conv1d.weight,
        "A_log": model.A_log,
        "dt_bias": model.dt_bias,
        "norm_w": model.norm.weight,
        "w_out": model.out_proj.weight,
    }
    for name, arr in shared.items():
        mx.save(str(out / f"{name}.npy"), arr)

    config = {
        "hidden_size": args.hidden,
        "key_dim": cfg.key_dim_total,
        "value_dim": cfg.value_dim_total,
        "num_k_heads": Hk,
        "num_v_heads": Hv,
        "head_k_dim": Dk,
        "head_v_dim": Dv,
        "conv_kernel": args.kernel,
        "eps": 1e-6,
        "output_activation": "sigmoid",
        "batch": 1,
        "cases": [c[0] for c in CASES],
        "seed": args.seed,
    }
    (out / "meta.json").write_text(json.dumps(config, indent=2) + "\n")

    for case, S, mask_vals in CASES:
        cdir = out / case
        cdir.mkdir(parents=True, exist_ok=True)
        conv_state = mx.array(
            (rng.standard_normal((1, args.kernel - 1, conv_dim)) * 0.1).astype(np.float32)
        )
        rec_state = mx.array(
            (rng.standard_normal((1, Hv, Dv, Dk)) * 0.05).astype(np.float32)
        )
        x = mx.array((rng.standard_normal((1, S, args.hidden)) * 0.1).astype(np.float32))

        mask = None
        if mask_vals is not None:
            mask = mx.array(np.array([mask_vals], dtype=np.bool_))
        # exactly as production: model(x, mask=mask, cache=[conv, rec]).
        # The layer reassigns cache[0]/cache[1], so the NEW state lives in the
        # list slots; the local conv_state/rec_state stay the INITIAL state.
        cache = [conv_state, rec_state]
        expected = model(x, mask=mask, cache=cache)
        new_conv, new_rec = cache[0], cache[1]
        mx.eval(expected, new_conv, new_rec)

        mx.save(str(cdir / "x.npy"), x)
        mx.save(str(cdir / "mask.npy"),
                mask if mask is not None else mx.array([], dtype=mx.bool_))
        mx.save(str(cdir / "i_conv_state.npy"), conv_state)
        mx.save(str(cdir / "i_rec_state.npy"), rec_state)
        mx.save(str(cdir / "expected_output.npy"), expected)
        mx.save(str(cdir / "expected_conv_state.npy"), new_conv)
        mx.save(str(cdir / "expected_rec_state.npy"), new_rec)
        print(f"wrote case {case}: S={S} mask={'yes' if mask is not None else 'no'} "
              f"out={expected.shape} new_conv={conv_state.shape} new_rec={rec_state.shape}")

    print(f"wrote shared weights and meta to {out}")


if __name__ == "__main__":
    main()
