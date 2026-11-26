#!/usr/bin/env python3
"""Bit-exact vectors for the S>1 Mamba-2 prefill path (plan P5b).

Drives the REAL mlx_lm primitives that NemotronHMamba2Mixer.__call__ uses:
  * nn.Conv1d (depthwise, groups=conv_dim) + silu  -- mixer._conv
  * mlx_lm.models.ssm.ssm_update (seq_len>1 -> ssm_attn chunked scan)
  * MambaRMSNormGated (swiglu + grouped rms_norm)
No mlx_lm source is modified; we reproduce the module's op order with its own
primitives and random but well-conditioned weights.

Usage:
  .venv/bin/python cpp/engine/tools/gen_mamba_prefill_vectors.py \
      --out cpp/engine/tests/vectors_mamba_prefill
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import sys
sys.path.insert(0, ".venv/lib/python3.12/site-packages")
from mlx_lm.models.ssm import ssm_update  # noqa: E402


def save(d: Path, name: str, arr):
    mx.save(str(d / (name + ".npy")), arr)


def emit(d: Path, *, S: int, seed: int, chunks: int):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    B = 1
    H, Dh, G, Ds, K = 4, 8, 2, 16, 4
    I = H * Dh
    CD = I + 2 * G * Ds
    hidden = 32

    dtype = mx.float32
    in_w = mx.array(rng.normal(0, 0.05, (I + CD + H, hidden)), dtype=dtype)
    conv_w = mx.array(rng.normal(0, 0.1, (CD, K, 1)), dtype=dtype)
    conv_b = mx.array(rng.normal(0, 0.01, (CD,)), dtype=dtype)
    A_log = mx.array(np.log(np.arange(1, H + 1, dtype=np.float32)), dtype=mx.float32)
    dt_bias = mx.array(np.ones(H), dtype=mx.float32)
    D = mx.array(np.ones(H), dtype=mx.float32)
    norm_w = mx.array(rng.normal(1.0, 0.05, (I,)), dtype=dtype)
    out_w = mx.array(rng.normal(0, 0.05, (hidden, I)), dtype=dtype)

    conv_state = mx.zeros((B, K - 1, CD), dtype=dtype)
    ssm_state = None
    expected = []

    for c in range(chunks):
        x = mx.array(rng.normal(0, 0.4, (B, S, hidden)), dtype=dtype)
        projected = mx.matmul(x, mx.swapaxes(in_w, 0, 1))
        gate, conv_input, dt = mx.split(projected, [I, I + CD], axis=-1)

        # _conv with cache: pad with conv_state, full depthwise conv, silu.
        padded = mx.concatenate([conv_state, conv_input], axis=1)
        conv_raw = mx.conv1d(padded, conv_w, 1, 0, 1, CD) + conv_b
        conv_out = nn.silu(conv_raw)
        new_conv = padded[:, -(K - 1):, :]

        hs, Bv, Cv = mx.split(conv_out, [I, I + G * Ds], axis=-1)
        # mixer._ssm reshapes to [B,S,H,Dh] / [B,S,G,Ds] before ssm_update.
        hs = mx.reshape(hs, (B, S, H, Dh))
        Bv = mx.reshape(Bv, (B, S, G, Ds))
        Cv = mx.reshape(Cv, (B, S, G, Ds))
        y, new_ssm = ssm_update(hs, A_log, Bv, Cv, D, dt, dt_bias,
                                ssm_state, (0.001, 100.0))
        y = mx.reshape(y, (B, S, I))  # mixer._ssm reshapes back to [B,S,I]

        gated = nn.silu(gate) * y
        unf = mx.reshape(gated, (B, S, G, I // G))
        normed = mx.fast.rms_norm(unf, weight=None, eps=1e-5)
        normed = norm_w * mx.reshape(normed, (B, S, I))
        out = mx.matmul(normed, mx.swapaxes(out_w, 0, 1))

        conv_state, ssm_state = new_conv, new_ssm
        save(d, f"x{c}", x)
        save(d, f"expected_out{c}", out)
        save(d, f"expected_conv_state{c}", conv_state)
        save(d, f"expected_ssm_state{c}", ssm_state)
        expected.append(c)

    for name, arr in [("in_w", in_w), ("conv_w", conv_w), ("conv_b", conv_b),
                      ("A_log", A_log), ("dt_bias", dt_bias), ("D", D),
                      ("norm_w", norm_w), ("out_w", out_w)]:
        save(d, name, arr)
    meta = dict(S=S, chunks=chunks, B=B, H=H, Dh=Dh, G=G, Ds=Ds, K=K,
                I=I, CD=CD, hidden=hidden, eps=1e-5,
                time_step_min=0.001, time_step_max=100.0,
                mlx_version="0.32.2")
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    metas = []
    for name, S in [("s2", 2), ("s7", 7), ("s64", 64), ("s128", 128)]:
        metas.append(dict(name=name, **emit(out / name, S=S, seed=S + 1000,
                                             chunks=2)))
    (out / "cases.json").write_text(json.dumps(metas, indent=2) + "\n")
    print("wrote", len(metas), "mamba prefill cases")


if __name__ == "__main__":
    main()
