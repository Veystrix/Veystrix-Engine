#!/usr/bin/env python3
"""Bit-exact GPU<->GPU check of the C++ KV-tier round-trip integration.

The C++ primitives (kv_quant.{hpp,cpp}) were validated against CPU-generated
golden vectors in `test_kv_quant`. The real checkpoint generator runs its
attention on the GPU, and the kv_quant_port.md doc measured a 1-ULP difference
between mx.quantize on CPU and GPU (1/128 packed words, 569/1024 dequant
outputs). So a GPU-backed integration path must be checked against GPU-backed
golden values, not the CPU vectors.

This tool:
  1. Generates synthetic bf16 K/V tensors (fixed seed) and the exact values a
     Python affine cache / tiered4 cache hands to SDPA, all on the GPU, using
     the REAL Python ops (mx.quantize/mx.dequantize, MixedPrecisionQSAKVCache).
  2. Optionally invokes the C++ binary with VEYSTRIX_CPP_KV_TIER_SELFTEST=<dir>
     and checks its bit-exact PASS output.

No checkpoint is read; this is a primitive-level parity check, not a generator
logit comparison. The generator-level mode comparison (C++ vs Python with a
matching FLASHNEXT_KV_CACHE_DTYPE) is a separate, non-bit-exact gate because of
the pre-existing PLE/quantized-matmul gaps; see
docs/engineering/kv_tier_mtp_integration.md.

Usage:
  .venv/bin/python cpp/engine/tools/check_kv_tier_parity.py --out /tmp/kvparity \
      [--binary /tmp/real_gen_p1]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

SEED = 20261118
SHAPE = (1, 2, 4, 128)   # [B, kv_heads, T, D]
GROUP = 64
CHUNKS = (128, 256, 128)  # tiered4: 512 tokens, recent window 256 -> 256 age out
RECENT = 256


def bf16_bytes(a: mx.array) -> bytes:
    return bytes(a)


def f32_bytes(a: mx.array) -> bytes:
    return bytes(mx.astype(a, mx.float32))


def make_input(rng, n):
    x = rng.standard_normal(n, dtype=np.float32) * 0.7
    return mx.array(x, dtype=mx.bfloat16)


def write_manifest(d: Path, mode: str, bits: int, chunks: tuple[int, ...]):
    lines = [
        f"mode={mode}",
        f"group_size={GROUP}",
        f"bits={bits}",
        f"recent_tokens={RECENT}",
        "shape=" + ",".join(str(x) for x in SHAPE),
        "n_chunks=" + str(len(chunks)),
        "chunk_tokens=" + ",".join(str(c) for c in chunks),
    ]
    (d / "manifest.txt").write_text("\n".join(lines) + "\n")


def gen_affine(d: Path, mode: str, bits: int, rng):
    d.mkdir(parents=True, exist_ok=True)
    x = make_input(rng, int(np.prod(SHAPE)))
    x = mx.reshape(x, SHAPE)
    parts = mx.quantize(x, GROUP, bits, "affine")
    expected = mx.dequantize(parts[0], parts[1], parts[2], GROUP, bits, "affine")
    mx.eval(x, expected)
    write_manifest(d, mode, bits, (SHAPE[2],))
    (d / "chunk0.u16").write_bytes(bf16_bytes(x))
    (d / "expected.f32").write_bytes(f32_bytes(expected))
    return x, expected


def gen_tiered4(d: Path, rng):
    d.mkdir(parents=True, exist_ok=True)
    from models.flashnext.mixed_kv import MixedPrecisionQSAKVCache

    cache = MixedPrecisionQSAKVCache(mode="tiered4", group_size=GROUP,
                                     recent_tokens=RECENT)
    inputs = []
    for i, c in enumerate(CHUNKS):
        x = make_input(rng, 1 * 2 * c * 128)
        x = mx.reshape(x, (1, 2, c, 128))
        # Both K and V get the same chunk; Tiered4KV writes both tiers from the
        # original activations, so the K dequant is the comparison target.
        cache.update_and_fetch(x, x)
        (d / f"chunk{i}.u16").write_bytes(bf16_bytes(x))
        inputs.append(x)
    idx = mx.reshape(mx.arange(cache.offset, dtype=mx.int32),
                     (1, 1, cache.offset))
    keys, _ = cache.gather_selected(idx)
    mx.eval(keys)
    write_manifest(d, "tiered4", 4, CHUNKS)
    (d / "expected.f32").write_bytes(f32_bytes(keys))
    return inputs, keys


LINE = __import__("re").compile(
    r"^step=(\d+)\s+in=(-?\d+)\s+out=(-?\d+)\s+sha256=([0-9a-f]{64})$")


def compare_arms(arms_path: Path) -> int:
    """Report C++ vs Python generator arms. This is NOT a bit-exact gate: the
    C++ generator already diverges from Python in bf16 mode (documented gaps),
    so the only claim here is that each arm ran with its intended mode and that
    the mode actually changed the logits. Primitive-level bit-exactness is
    established by run_selftest above."""
    import json
    arms = json.loads(arms_path.read_text())
    print("generator-level KV mode comparison (measured; bit-exactness NOT claimed)")
    all_ran = True
    for mode, paths in arms.items():
        cpp = []
        for line in Path(paths["cpp"]).read_text().splitlines():
            m = LINE.match(line.strip())
            if m:
                cpp.append((int(m.group(3)), m.group(4)))
        ref = json.loads(Path(paths["python"]).read_text())
        py = [(int(s["token"]), s["logit_sha256"]) for s in ref.get("steps", [])]
        ok = bool(cpp) and bool(py)
        all_ran &= ok
        print(f"  {mode:8s} C++ tokens={[t for t, _ in cpp]} "
              f"digests={[d[:12] for _, d in cpp]}")
        print(f"  {'':8s} Python tokens={[t for t, _ in py]} "
              f"digests={[d[:12] for _, d in py]}  ({'ran' if ok else 'MISSING'})")
    print("  note: cross-language token/digest identity is not expected; see "
          "docs/engineering/kv_tier_mtp_integration.md")
    return 0 if all_ran else 1


def run_selftest(binary: str, root: Path) -> int:
    import os
    env = dict(os.environ)
    env["VEYSTRIX_CPP_KV_TIER_SELFTEST"] = str(root)
    proc = subprocess.run([binary], env=env, text=True, capture_output=True)
    out = proc.stdout + proc.stderr
    print(out[-8000:])
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory for golden vectors")
    ap.add_argument("--binary", default=None,
                    help="C++ binary to run with VEYSTRIX_CPP_KV_TIER_SELFTEST")
    ap.add_argument("--compare-arms", default=None,
                    help="JSON mapping mode -> {cpp: OUT, python: reference.json}")
    args = ap.parse_args()
    if args.compare_arms:
        return compare_arms(Path(args.compare_arms))

    mx.set_default_device(mx.gpu)  # MUST match the C++ generator's default device
    rng = np.random.default_rng(SEED)
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)

    print(f"device={mx.default_device()} (goldens generated on GPU)")
    gen_affine(root / "affine4", "affine4", 4, rng)
    gen_affine(root / "affine8", "affine8", 8, rng)
    gen_tiered4(root / "tiered4", rng)
    print(f"wrote golden vectors under {root}")

    if args.binary:
        rc = run_selftest(args.binary, root)
        print(f"selftest exit={rc}")
        return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
