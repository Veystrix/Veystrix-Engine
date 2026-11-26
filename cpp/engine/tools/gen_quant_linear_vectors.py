#!/usr/bin/env python3
"""Generate bit-exact fixtures for QuantizedLinear (Q4 affine, group_size 32).

Each fixture is produced from ``nn.Linear(...).to_quantized(gs, bits)`` -- the
exact construction ``models/flashnext/loader.py:209-236`` uses for the dense
projections. A future C++ test (written by the primary agent) can load these
and reproduce the production ``mx.quantized_matmul`` + optional bias path with
``mx::quantized_matmul``.

For every case we save, as one ``.npz`` per case:
  * ``x``        -- input activations        [M, in]   float32
  * ``weight``   -- packed uint32 weight     [out, in/8]
  * ``scales``   -- affine scales            [out, in/gs] float32
  * ``biases``   -- affine zero-points       [out, in/gs] float32
  * ``bias``     -- layer bias (if any)      [out] float32
  * ``expected`` -- QuantizedLinear output   [M, out] float32

The packed weight/scales/biases come straight from the QuantizedLinear's
parameters (``ql.parameters()``). ``expected`` is ``ql(x)``. We call ``ql(x)``
twice per case and assert the two float32 bit patterns are identical, proving
the reference output is reproducible.

Deterministic: numpy and MLX RNGs are seeded with the same constant, and the
linear weights/bias are set explicitly from the seeded numpy RNG rather than
relying on module init. ``meta.json`` records every case and a sha256 of each
array's little-endian bytes.

Run:
  /Users/vincent/Documents/Veystrix/.venv/bin/python \
      cpp/engine/tools/gen_quant_linear_vectors.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

SEED = 20261117
GROUP_SIZE = 32
BITS = 4
MODE = "affine"
DT = np.float32

# (in_features, out_features) -- the first is the model's 2560-wide projection,
# the second is a small config that keeps the fixture fast to load in ctest.
SHAPES = [(2560, 2560), (256, 128)]
MS = [1, 16]
BIASES = [False, True]


def _digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).astype("<f4").tobytes()
                          if arr.dtype == np.float32
                          else np.ascontiguousarray(arr).tobytes()).hexdigest()


def _describe(name: str, arr: np.ndarray) -> dict:
    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": _digest(arr),
        "first": np.ascontiguousarray(arr).reshape(-1)[:4].tolist(),
    }


def build_case(in_features: int, out_features: int, bias: bool, m: int,
               weight_np: np.ndarray, bias_np: np.ndarray | None) -> tuple[dict, dict]:
    """Build a quantized linear from fixed weights, return arrays + description."""
    lin = nn.Linear(in_features, out_features, bias=bias)
    lin.weight = mx.array(weight_np)
    if bias:
        lin.bias = mx.array(bias_np)
    mx.eval(lin.parameters())
    ql = lin.to_quantized(group_size=GROUP_SIZE, bits=BITS, mode=MODE)
    mx.eval(ql.parameters())

    # Input: deterministic from the numpy RNG (advanced after the weight draw).
    x_np = (np.random.randn(m, in_features) * 0.5).astype(DT)
    x = mx.array(x_np)

    y1 = ql(x)
    y2 = ql(x)
    mx.eval(y1, y2)
    expected = np.array(y1, dtype=DT)
    repeat = np.array(y2, dtype=DT)
    if not np.array_equal(expected.view(np.uint32), repeat.view(np.uint32)):
        raise SystemExit(f"NON-REPRODUCIBLE output for case M={m} bias={bias}")

    params = ql.parameters()
    arrays = {
        "x": x_np,
        "weight": np.array(params["weight"]),
        "scales": np.array(params["scales"], dtype=DT),
        "biases": np.array(params["biases"], dtype=DT),
        "expected": expected,
    }
    if bias:
        arrays["bias"] = np.array(params["bias"], dtype=DT)

    description = {
        "in": in_features,
        "out": out_features,
        "M": m,
        "bias": bias,
        "group_size": GROUP_SIZE,
        "bits": BITS,
        "mode": MODE,
        "reproducible": True,
        "arrays": [_describe(name, a) for name, a in arrays.items()],
    }
    return arrays, description


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        Path(__file__).resolve().parent.parent / "tests" / "vectors_quant_linear"))
    args = ap.parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    mx.random.seed(SEED)

    cases = []
    for in_features, out_features in SHAPES:
        for bias in BIASES:
            # Fixed base weight + optional bias for this (in, out, bias) tuple,
            # shared by the M=1 and M=16 cases.
            weight_np = (np.random.randn(out_features, in_features) * 0.05).astype(DT)
            bias_np = (np.random.randn(out_features) * 0.1).astype(DT) if bias else None
            for m in MS:
                arrays, description = build_case(
                    in_features, out_features, bias, m, weight_np, bias_np)
                case_id = f"qlinear_in{in_features}_out{out_features}_bias{int(bias)}_m{m}"
                path = out_dir / f"{case_id}.npz"
                np.savez(path, **arrays)
                description["file"] = path.name
                # Also emit individual .npy files so the C++ ctest can mx::load
                # them (C++ has no npz reader). Deterministic, same bytes.
                case_dir = out_dir / case_id
                case_dir.mkdir(exist_ok=True)
                for a_name, a_arr in arrays.items():
                    np.save(case_dir / f"{a_name}.npy", a_arr)
                description["npy_dir"] = case_id
                cases.append(description)

                print(f"\n=== {case_id} -> {path} ===")
                print(f"  in={in_features} out={out_features} M={m} bias={bias} "
                      f"gs={GROUP_SIZE} bits={BITS} mode={MODE}")
                for name, a in arrays.items():
                    d = _describe(name, a)
                    print(f"  {name:<8} shape={d['shape']} dtype={d['dtype']} "
                          f"sha256={d['sha256'][:16]} first={d['first']}")
                print(f"  reproducible (ql(x) twice, bit-identical): True")

    meta = {"seed": SEED, "group_size": GROUP_SIZE, "bits": BITS,
            "mode": MODE, "cases": cases}
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwrote {len(cases)} fixtures + {meta_path}")


if __name__ == "__main__":
    main()
