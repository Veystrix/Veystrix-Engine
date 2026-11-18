#!/usr/bin/env python3
"""Golden vectors for the Mamba/SSM decode path (kernel-level and layer-level).

KERNEL LEVEL: calls the REAL stock decode kernel ``ssm_update_kernel``
(mlx_lm.models.ssm, the one selected for S=1) on synthetic tiny-but-legal shapes
with a fixed seed. Outputs (y, state_out) are saved byte-for-byte so the C++
test (test_mamba_kernel.cpp) can compare them with a uint32 view — zero
tolerance.

LAYER LEVEL: builds a REAL ``NemotronHMamba2Mixer`` with a tiny config, overwrites
every weight deterministically (all bf16 except A_log/dt_bias/D f32 per plan risk
5), forces eval mode, runs one S=1 decode step and saves the layer output plus
the updated conv/ssm caches and all key intermediates. The C++ test
(test_mamba_layer.cpp) compares byte-for-byte and isolates the first divergence.

NO real checkpoint is used; everything is synthetic. The two non-bit-exact
prefill research paths are explicitly disabled (S=1 also auto-selects stock).

Usage (from the repo root):
  .venv/bin/python cpp/engine/tools/gen_mamba_vectors.py \
      --out cpp/engine/tests/vectors_mamba_kernel \
      --layer-out cpp/engine/tests/vectors_mamba_layer
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Force the stock fused-kernel path (belt-and-braces; we call it directly and
# S=1 already selects stock). The fast_ssm / mamba_scan research paths are NOT
# bit-exact and must never produce goldens (plan risk 3).
os.environ["VEYSTRIX_NEMOTRON_FAST_SSM"] = "0"
os.environ["VEYSTRIX_NEMOTRON_PARALLEL_MAMBA"] = "0"

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm.models.ssm import ssm_update_kernel, compute_dt  # noqa: E402
from mlx_lm.models.nemotron_h import (  # noqa: E402
    ModelArgs,
    NemotronHMamba2Mixer,
)
from mlx_lm.models.cache import ArraysCache  # noqa: E402


# ---------------------------------------------------------------------------
# Kernel-level vectors (unchanged from stage 1)
# ---------------------------------------------------------------------------
def emit_case(out: Path, name: str, *, batch: int, dtype, state_dtype,
              tmin: float, tmax: float, seed: int) -> dict:
    """Create one case directory with all inputs + golden outputs + meta.json."""
    assert mx.metal.is_available(), "fused SSM kernel requires Metal"

    H, Dh, groups, Ds = 8, 32, 2, 32   # tiny but legal (Ds % 32 == 0)
    rng = np.random.default_rng(seed)

    def rand(shape):
        return rng.normal(size=shape).astype(np.float32)

    def as_dtype(shape, dtype_):
        return mx.array(rand(shape)).astype(dtype_)

    # hidden_states X [batch,1,H,Dh] in the requested input dtype (T).
    X = as_dtype((batch, 1, H, Dh), dtype)
    A_log = mx.array(rand((H,)))                                   # f32
    Bv = as_dtype((batch, 1, groups, Ds), dtype)
    Cv = as_dtype((batch, 1, groups, Ds), dtype)
    Dv = as_dtype((H,), dtype)       # mixer casts D to hidden dtype
    dt_raw = as_dtype((batch, 1, H), dtype)
    dt_bias = mx.array(np.ones((H,), dtype=np.float32))              # f32
    state = mx.array(rand((batch, H, Dh, Ds)) * 0.01).astype(state_dtype)

    y, state_out = ssm_update_kernel(
        X, A_log, Bv, Cv, Dv, dt_raw, dt_bias, state, (tmin, tmax))
    mx.eval(y, state_out)

    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    arrays = {
        "X": X, "A_log": A_log, "B": Bv, "C": Cv, "D": Dv,
        "dt_raw": dt_raw, "dt_bias": dt_bias, "state": state,
        "y": y, "state_out": state_out,
    }
    for k, v in arrays.items():
        mx.save(str(d / f"{k}.npy"), v)

    def tstr(x: float) -> str:
        if x == float("inf"):
            return "inf"
        if x == float("-inf"):
            return "-inf"
        return repr(x)

    meta = {
        "dir": name,
        "batch": batch, "H": H, "Dh": Dh, "groups": groups, "Ds": Ds,
        "T": str(dtype), "U": str(state_dtype),
        "tmin": tstr(tmin), "tmax": tstr(tmax),
        "seed": seed, "mlx_version": mx.__version__,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    # Sanity: the recurrence must actually write the state (not the Ds<32 trap).
    if not np.any(np.asarray(state_out.astype(mx.float32)) != np.asarray(state.astype(mx.float32))):
        raise RuntimeError(f"{name}: state_out identical to state_in — kernel did no work")
    print(f"  {name}: y={tuple(y.shape)} {y.dtype}, state_out={tuple(state_out.shape)} {state_out.dtype}")
    return meta


# ---------------------------------------------------------------------------
# Layer-level vectors (NEW — STAGE 2)
# ---------------------------------------------------------------------------
def gen_layer_case(outdir: Path, seed: int) -> dict:
    """Build a real NemotronHMamba2Mixer (tiny config) and run one S=1 step."""
    assert mx.metal.is_available(), "fused SSM kernel requires Metal"

    # Tiny but kernel-legal geometry. Dh % 8 == 0, Ds % 32 == 0, H % G == 0.
    B = 1
    hidden = 256
    H, Dh, G, Ds, K = 16, 64, 2, 128, 4
    inter = H * Dh                              # 1024
    conv_dim = inter + 2 * G * Ds               # 1536
    proj = inter + conv_dim + H                 # 2576
    group_size = inter // G                     # 512
    tmin, tmax = 0.001, 100.0
    eps = 1e-5

    args = ModelArgs(
        model_type="nemotron_h",
        vocab_size=1024,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=1,
        max_position_embeddings=128,
        num_attention_heads=8,
        num_key_value_heads=8,
        attention_bias=False,
        mamba_num_heads=H,
        mamba_head_dim=Dh,
        mamba_proj_bias=False,
        ssm_state_size=Ds,
        conv_kernel=K,
        n_groups=G,
        mlp_bias=False,
        layer_norm_epsilon=eps,
        use_bias=False,
        use_conv_bias=True,
        time_step_limit=(tmin, tmax),
        time_step_min=tmin,
        time_step_max=tmax,
    )
    model = NemotronHMamba2Mixer(args)

    rng = np.random.default_rng(seed)

    def mr(shape, scale=0.1, dtype=mx.float32):
        return mx.array((rng.standard_normal(shape) * scale).astype(np.float32)).astype(dtype)

    # Overwrite every parameter deterministically. Weights bf16 (production),
    # A_log/dt_bias/D f32 (A_log protected from casting; D is cast to hidden
    # dtype by the mixer at nemotron_h.py:195).
    model.in_proj.weight = mr((proj, hidden), 0.05, mx.bfloat16)
    model.conv1d.weight = mr((conv_dim, K, 1), 0.3, mx.bfloat16)
    model.conv1d.bias = mr((conv_dim,), 0.1, mx.bfloat16)   # use_conv_bias=True
    model.A_log = mx.array(np.log(np.arange(1, H + 1, dtype=np.float32)))  # f32
    model.dt_bias = mr((H,), 0.5, mx.float32)               # f32
    model.D = mr((H,), 0.5, mx.float32)                     # f32 (cast by mixer)
    # Checkpoint-cast norm weight is bf16 (cast_predicate excludes only A_log).
    model.norm.weight = mx.array(np.ones((inter,), dtype=np.float32)).astype(mx.bfloat16)
    model.out_proj.weight = mr((hidden, inter), 0.05, mx.bfloat16)
    model.eval()

    # Decode caches: conv_state bf16 [B,K-1,conv_dim]; ssm_state f32 [B,H,Dh,Ds].
    conv_state = mr((B, K - 1, conv_dim), 0.1, mx.bfloat16)
    ssm_state = mr((B, H, Dh, Ds), 0.05, mx.float32)
    x = mr((B, 1, hidden), 0.1, mx.bfloat16)

    cache = ArraysCache(size=2)
    cache[0] = conv_state
    cache[1] = ssm_state
    out = model(x, mask=None, cache=cache)  # mutates cache[0], cache[1]
    mx.eval(out, cache[0], cache[1])

    # Recompute the real intermediates (same ops as __call__) to isolate a
    # failing C++ layer to the exact diverging primitive.
    projected = model.in_proj(x)
    gate, conv_input, dt_raw = mx.split(projected, [inter, inter + conv_dim], axis=-1)

    padded = mx.concatenate([conv_state, conv_input], axis=1)
    conv_out = nn.silu(model.conv1d(padded))                # bf16, conv+bias+silu
    hidden_ssm, Bv, Cv = mx.split(
        conv_out, [inter, inter + G * Ds], axis=-1)
    x4 = hidden_ssm.reshape(B, 1, H, Dh)
    b4 = Bv.reshape(B, 1, G, Ds)
    c4 = Cv.reshape(B, 1, G, Ds)
    dt_post = compute_dt(dt_raw, model.dt_bias, model.time_step_limit)  # f32
    y, state_out = ssm_update_kernel(
        x4, model.A_log, b4, c4, model.D.astype(x4.dtype), dt_raw,
        model.dt_bias, ssm_state, model.time_step_limit)
    normed = model.norm(y.reshape(B, 1, inter), gate)
    mx.eval(projected, gate, conv_input, dt_raw, padded, conv_out, hidden_ssm,
            b4, c4, dt_post, y, state_out, normed)

    arrays = {
        "x": x,
        "w_in": model.in_proj.weight,
        "conv_w": model.conv1d.weight,
        "conv_b": model.conv1d.bias,
        "A_log": model.A_log,
        "dt_bias": model.dt_bias,
        "D": model.D,
        "norm_w": model.norm.weight,
        "w_out": model.out_proj.weight,
        "conv_state": conv_state,
        "ssm_state": ssm_state,
        "i_projected": projected,
        "i_gate": gate,
        "i_conv_input": conv_input,
        "i_dt_raw": dt_raw,
        "i_padded": padded,
        "i_conv_out": conv_out,
        "i_hidden_ssm": x4,
        "i_B": b4,
        "i_C": c4,
        "i_dt_post": dt_post,
        "i_y": y,
        "i_state": state_out,
        "i_normed": normed,
        "expected_output": out,
        "expected_conv_state": cache[0],
        "expected_ssm_state": cache[1],
    }

    outdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": {
            "batch": B, "hidden": hidden, "H": H, "Dh": Dh, "G": G,
            "Ds": Ds, "K": K, "inter": inter, "conv_dim": conv_dim,
            "group_size": group_size, "eps": eps,
            "tmin": tmin, "tmax": tmax, "use_conv_bias": True,
        },
        "seed": seed,
        "mlx_version": mx.__version__,
    }
    for name, arr in arrays.items():
        mx.save(str(outdir / f"{name}.npy"), arr)
        meta[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype).split(".")[-1]}
        print(f"wrote {outdir / (name + '.npy')} shape={arr.shape} dtype={arr.dtype}")
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {outdir / 'meta.json'}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors_mamba_kernel")
    ap.add_argument("--layer-out", default="cpp/engine/tests/vectors_mamba_layer")
    ap.add_argument("--layer-seed", type=int, default=20261118)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [
        emit_case(out, "case_b1_bf16", batch=1, dtype=mx.bfloat16,
                  state_dtype=mx.float32, tmin=0.001, tmax=100.0, seed=11),
        emit_case(out, "case_b2_bf16", batch=2, dtype=mx.bfloat16,
                  state_dtype=mx.float32, tmin=0.001, tmax=100.0, seed=22),
        emit_case(out, "case_b1_f16", batch=1, dtype=mx.float16,
                  state_dtype=mx.float32, tmin=0.001, tmax=100.0, seed=33),
        emit_case(out, "case_b1_f32_inf", batch=1, dtype=mx.float32,
                  state_dtype=mx.float32, tmin=0.0, tmax=float("inf"), seed=44),
    ]
    (out / "cases.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    print(f"wrote {len(cases)} kernel cases to {out}")

    print("generating layer-level case ...")
    gen_layer_case(Path(args.layer_out), args.layer_seed)


if __name__ == "__main__":
    main()
