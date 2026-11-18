#!/usr/bin/env python3
"""Generate kernel-level golden vectors for the fused Mamba/SSM decode kernel.

We call the REAL stock decode kernel ``mlx_lm.models.ssm.ssm_update_kernel``
(the one selected for S=1 decode) on synthetic, tiny-but-legal shapes with a
fixed seed. Outputs (y, state_out) are saved byte-for-byte as .npy so the C++
test (test_mamba_kernel.cpp) can compare them with a uint32 view — zero
tolerance. NO real checkpoint is used; everything is CPU/synthetic.

Kernel constraints (from ssm.py:17-55):
  * Ds % 32 == 0 and Ds >= 32   (n_per_t = Ds / 32 must be >= 1; with Ds=16 the
    loop body never runs and the kernel writes nothing -> meaningless test)
  * H % groups == 0            (g_idx = n / G, G = H / groups)
  * grid.y = Dh divisible by threadgroup.y = 8

The two non-bit-exact prefill research paths are explicitly disabled; calling
ssm_update_kernel directly (S=1, state present) already forces the fused kernel.

Usage (from the repo root):
  .venv/bin/python cpp/engine/tools/gen_mamba_vectors.py \
      --out cpp/engine/tests/vectors_mamba_kernel
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Force the stock fused-kernel path (belt-and-braces; we call it directly).
os.environ["VEYSTRIX_NEMOTRON_FAST_SSM"] = "0"
os.environ["VEYSTRIX_NEMOTRON_PARALLEL_MAMBA"] = "0"

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm.models.ssm import ssm_update_kernel  # noqa: E402


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors_mamba_kernel")
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
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
