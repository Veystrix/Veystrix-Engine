#!/usr/bin/env python3
"""Generate bit-exact golden vectors for the C++ KV tier-quant primitives.

The arrays are produced by the REAL Python path -- the `mx.quantize` /
`mx.dequantize` ops called by models/flashnext/quantized_kv.py
(AffineQuantizedQSAKVCache) and by the REAL
models/flashnext/mixed_kv.py::MixedPrecisionQSAKVCache(mode="tiered4").

Everything is synthetic (tiny random tensors, fixed seed), CPU only, and no
checkpoint or benchmark is read. Every array is saved with mx.save so its
bfloat16/uint32 bits survive byte-for-byte; tests/test_kv_quant.cpp loads them
with mx::load and compares raw bits.

Generated layout (default --out cpp/engine/tests/vectors_kv_quant):
  affine4/  x.npy packed.npy scales.npy biases.npy dequant.npy meta.json
  affine8/  ...(same)
  affine4_wide/ ...(same, D=256 -> 4 groups of 64 per row)
  tiered4/  chunk_k_<i>.npy chunk_v_<i>.npy                 (inputs to replay)
            hist_keys_data/scales/biases.npy               (INT4, [0,history_tokens))
            hist_values_data/scales/biases.npy
            hist_keys_dequant.npy hist_values_dequant.npy
            recent_keys_data/scales/biases.npy             (INT8, [history_tokens,offset))
            recent_values_data/scales/biases.npy
            recent_keys_dequant.npy recent_values_dequant.npy
            meta.json  (chunks, offset, history_tokens, recent_tokens, group_size, ...)

Run (venv with mlx 0.32.2, e.g. /Users/vincent/Documents/Veystrix/.venv/bin/python):
  python cpp/engine/tools/gen_kv_quant_vectors.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# Import the reference modules from THIS worktree (not a globally installed copy).
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.flashnext.mixed_kv import MixedPrecisionQSAKVCache  # noqa: E402
from models.flashnext.quantized_kv import AffineQuantizedQSAKVCache  # noqa: E402

SEED = 20261118
GROUP_SIZE = 64
DT = np.float32


def _raw(a: mx.array) -> bytes:
    """Little-endian raw bytes of an MLX array (bf16 as uint16 pairs)."""
    if a.dtype == mx.bfloat16:
        return np.asarray(mx.view(mx.reshape(a, (-1,)), mx.uint16)).astype("<u2").tobytes()
    return np.asarray(mx.reshape(a, (-1,))).tobytes()


def _describe(name: str, a: mx.array) -> dict:
    return {
        "name": name,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sha256": hashlib.sha256(_raw(a)).hexdigest(),
    }


def _save(cd: Path, name: str, a: mx.array) -> dict:
    mx.eval(a)
    mx.save(str(cd / f"{name}.npy"), a)
    return _describe(name, a)


def _rand_kv(rng: np.random.Generator, shape) -> tuple[mx.array, mx.array]:
    k = mx.array((rng.normal(size=shape) * 0.5).astype(DT)).astype(mx.bfloat16)
    v = mx.array((rng.normal(size=shape) * 0.5).astype(DT)).astype(mx.bfloat16)
    return k, v


def affine_case(out: Path, name: str, shape, bits: int, group_size: int,
                seed: int) -> dict:
    """mx.quantize / mx.dequantize -- the exact ops used by AffineQuantizedQSAKVCache.

    A sanity assertion drives the REAL cache with the same tensor and checks its
    dequantize() output is bit-identical to the standalone dequant we save.
    """
    cd = out / name
    cd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    x, _ = _rand_kv(rng, shape)

    packed, scales, biases = mx.quantize(x, group_size=group_size, bits=bits)
    dequant = mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits)
    mx.eval(x, packed, scales, biases, dequant)

    # Cross-check against the real cache (capacity-padded; only [:offset] is live).
    cache = AffineQuantizedQSAKVCache(group_size=group_size, bits=bits)
    k, v = x, mx.array((rng.normal(size=shape) * 0.5).astype(DT)).astype(mx.bfloat16)
    cache.update_and_fetch(k, v)
    dk, dv = cache.dequantize()
    mx.eval(dk, dv)
    assert dk.dtype == mx.bfloat16, dk.dtype
    if not np.array_equal(_raw(dequant), _raw(dk)):
        raise SystemExit(f"{name}: standalone dequant differs from real cache dequantize()")

    arrays = [_save(cd, n, a) for n, a in (
        ("x", x), ("packed", packed), ("scales", scales),
        ("biases", biases), ("dequant", dequant))]
    meta = {
        "kind": "affine", "name": name, "seed": seed,
        "shape": list(shape), "group_size": group_size, "bits": bits,
        "input_dtype": str(x.dtype), "packed_dtype": str(packed.dtype),
        "scales_dtype": str(scales.dtype), "biases_dtype": str(biases.dtype),
        "dequant_dtype": str(dequant.dtype),
        "cache_cross_check": "AffineQuantizedQSAKVCache.dequantize() bit-identical",
        "arrays": arrays,
    }
    (cd / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[{name}] {shape} gs={group_size} bits={bits}: "
          f"packed={tuple(packed.shape)} scales={tuple(scales.shape)} "
          f"dequant_dtype={dequant.dtype}")
    return meta


def tiered_case(out: Path, chunks, group_size: int, recent_tokens: int,
                seed: int) -> dict:
    """Drive the REAL MixedPrecisionQSAKVCache(mode='tiered4') and save both tiers."""
    cd = out / "tiered4"
    cd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dim = 128

    cache = MixedPrecisionQSAKVCache(
        mode="tiered4", group_size=group_size, recent_tokens=recent_tokens)
    inputs = []
    for i, tokens in enumerate(chunks):
        k, v = _rand_kv(rng, (1, 2, tokens, dim))
        cache.update_and_fetch(k, v)
        inputs.append((k, v))
    mx.eval(cache.keys[0], cache.recent_keys[0])

    hist = cache.history_tokens
    off = cache.offset
    recent = off - hist

    def sl(arrays, start, stop):
        # [..., T, D] packed / scales / biases -> slice the token axis
        return tuple(a[..., start:stop, :] for a in arrays)

    hk_data, hk_scales, hk_biases = sl(cache.keys, 0, hist)
    hv_data, hv_scales, hv_biases = sl(cache.values, 0, hist)
    rk_data, rk_scales, rk_biases = sl(cache.recent_keys, 0, recent)
    rv_data, rv_scales, rv_biases = sl(cache.recent_values, 0, recent)

    hk_deq = mx.dequantize(hk_data, hk_scales, hk_biases, group_size=group_size, bits=4)
    hv_deq = mx.dequantize(hv_data, hv_scales, hv_biases, group_size=group_size, bits=4)
    rk_deq = mx.dequantize(rk_data, rk_scales, rk_biases, group_size=group_size, bits=8)
    rv_deq = mx.dequantize(rv_data, rv_scales, rv_biases, group_size=group_size, bits=8)
    mx.eval(hk_deq, hv_deq, rk_deq, rv_deq)

    arrays = []
    for i, (k, v) in enumerate(inputs):
        arrays += [_save(cd, f"chunk_k_{i}", k), _save(cd, f"chunk_v_{i}", v)]
    for name, a in (
        ("hist_keys_data", hk_data), ("hist_keys_scales", hk_scales),
        ("hist_keys_biases", hk_biases),
        ("hist_values_data", hv_data), ("hist_values_scales", hv_scales),
        ("hist_values_biases", hv_biases),
        ("hist_keys_dequant", hk_deq), ("hist_values_dequant", hv_deq),
        ("recent_keys_data", rk_data), ("recent_keys_scales", rk_scales),
        ("recent_keys_biases", rk_biases),
        ("recent_values_data", rv_data), ("recent_values_scales", rv_scales),
        ("recent_values_biases", rv_biases),
        ("recent_keys_dequant", rk_deq), ("recent_values_dequant", rv_deq),
    ):
        arrays.append(_save(cd, name, a))

    meta = {
        "kind": "tiered4", "seed": seed,
        "source": "models/flashnext/mixed_kv.py::MixedPrecisionQSAKVCache(mode='tiered4')",
        "chunks": list(chunks), "offset": off, "history_tokens": hist,
        "recent_tokens": recent_tokens, "group_size": group_size,
        "key_bits": 4, "value_bits": 4, "recent_bits": 8,
        "head_dim": dim, "heads": 2,
        "arrays": arrays,
    }
    (cd / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[tiered4] chunks={list(chunks)} offset={off} history_tokens={hist} "
          f"recent={recent} hist_int4={tuple(hk_data.shape)} recent_int8={tuple(rk_data.shape)}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        Path(__file__).resolve().parent.parent / "tests" / "vectors_kv_quant"))
    args = ap.parse_args()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    mx.set_default_device(mx.cpu)
    np.random.seed(SEED)
    mx.random.seed(SEED)

    metas = [
        affine_case(out, "affine4", (1, 2, 4, 128), 4, GROUP_SIZE, SEED),
        affine_case(out, "affine8", (1, 2, 4, 128), 8, GROUP_SIZE, SEED + 1),
        affine_case(out, "affine4_wide", (1, 1, 3, 256), 4, GROUP_SIZE, SEED + 2),
        tiered_case(out, [128, 256, 128], GROUP_SIZE, 256, SEED + 3),
    ]
    root_meta = {
        "seed": SEED, "group_size": GROUP_SIZE, "device": "cpu",
        "mlx_version": str(mx.__version__),
        "sources": ["models/flashnext/quantized_kv.py",
                    "models/flashnext/mixed_kv.py"],
        "cases": [m["name"] if "name" in m else "tiered4" for m in metas],
    }
    (out / "meta.json").write_text(json.dumps(root_meta, indent=2) + "\n")
    print(f"\nwrote golden vectors to {out}")


if __name__ == "__main__":
    main()
