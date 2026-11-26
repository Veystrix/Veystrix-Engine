#!/usr/bin/env python3
"""OWNER-RUN-safe (no checkpoint): test whether the C++ GDN primitive semantics
diverge from Python's at REAL bf16 shapes.

Two suspected dtype divergences in cpp/engine/src/engine.cpp vs Python/MLX:

  1. gdn_normalize_qk: C++ casts q/k to f32, normalizes + scales in f32 and
     RETURNS f32; Python's Qwen4ExpGatedDeltaNet._normalize_qk operates on bf16
     q/k and returns bf16 (sum/rsqrt/scale all round to bf16).

  2. compute_g_beta: C++ computes softplus via logaddexp(a+dt_bias, f32(0)),
     which promotes the bf16 (a+dt_bias) to f32 BEFORE the logaddexp, so the
     softplus result is f32. Python uses nn.softplus(a+dt_bias) with bf16 input
     -> bf16 result, then promotes only when multiplying by exp(A_log:f32).

Both feed the STATEFUL fused delta kernel, so a 1-ULP difference compounds over
16 tokens x 48 layers. This script quantifies the per-primitive gap in isolation
using the exact MLX primitives, at the real GDN geometry. No checkpoint needed.
"""
from __future__ import annotations

import sys
import numpy as np

VENV_SITE = "/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages"
if VENV_SITE not in sys.path:
    sys.path.insert(0, VENV_SITE)

import mlx.core as mx
import mlx.nn as nn


def stat(name: str, a, b):
    a = np.asarray(a.astype(mx.float32)) if getattr(a, "dtype", None) in (mx.bfloat16, mx.float16) else np.asarray(a, dtype=np.float32)
    b = np.asarray(b.astype(mx.float32)) if getattr(b, "dtype", None) in (mx.bfloat16, mx.float16) else np.asarray(b, dtype=np.float32)
    d = np.abs(a - b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    corr = float(np.dot(a.ravel(), b.ravel()) / denom) if denom else float("nan")
    print(f"  {name:28s} shape={a.shape} maxdiff={d.max():.6g} "
          f"meanabs={d.mean():.3g} corr={corr:.9f} "
          f"exact={(a == b).mean() * 100:.2f}%")


def main():
    mx.random.seed(0)
    B, T, Hk, Dk = 1, 16, 16, 128
    Hv = 48

    print("== 1. _normalize_qk (q,k bf16, real shape) ==")
    q = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    k = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)

    # Python (Qwen4ExpGatedDeltaNet._normalize_qk, language.py:266-272)
    def py_normalize(q, k):
        scale = q.shape[-1] ** -0.5
        qn = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-6)
        kn = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6)
        return qn * scale, kn

    # C++ (engine.cpp:223-229): f32 end-to-end, returns f32
    def cpp_normalize(q, k):
        import math
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        qn = mx.astype(q, mx.float32)
        qn = qn * mx.rsqrt(mx.sum(mx.square(qn), -1, keepdims=True) + 1e-6)
        kn = mx.astype(k, mx.float32)
        kn = kn * mx.rsqrt(mx.sum(mx.square(kn), -1, keepdims=True) + 1e-6)
        return qn * scale, kn

    pq, pk = py_normalize(q, k)
    cq, ck = cpp_normalize(q, k)
    print(f"  python dtype={pq.dtype}  cpp dtype={cq.dtype}")
    stat("q: python(bf16) vs cpp(f32)", pq, cq)
    stat("q: python(bf16) vs cpp->bf16", pq, cq.astype(mx.bfloat16))
    stat("k: python(bf16) vs cpp(f32)", pk, ck)
    stat("k: python(bf16) vs cpp->bf16", pk, ck.astype(mx.bfloat16))

    print("== 2. compute_g_beta (a,b,A_log,dt_bias bf16, real shape) ==")
    a = mx.random.normal((B, T, Hv)).astype(mx.bfloat16)
    b = mx.random.normal((B, T, Hv)).astype(mx.bfloat16)
    A_log = mx.random.uniform(-3, 0, (Hv,)).astype(mx.bfloat16)
    dt_bias = mx.zeros((Hv,)).astype(mx.bfloat16)

    # Python (gated_delta.py:262-268): softplus in bf16, then promote to f32
    def py_gb(A_log, a, b, dt_bias):
        g = mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))
        beta = mx.sigmoid(b)
        return g, beta

    # C++ (engine.cpp:139-148): logaddexp with f32 zero -> f32 softplus
    def cpp_gb(A_log, a, b, dt_bias):
        A = mx.exp(mx.astype(A_log, mx.float32))
        sp = mx.logaddexp(a + dt_bias, mx.array(0.0, dtype=mx.float32))
        g = mx.exp(-A * sp)
        return g, mx.sigmoid(b)

    pg, pb = py_gb(A_log, a, b, dt_bias)
    cg, cb = cpp_gb(A_log, a, b, dt_bias)
    print(f"  softplus dtype: python={(nn.softplus(a + dt_bias)).dtype} "
          f"cpp={(mx.logaddexp(a + dt_bias, mx.array(0.0, dtype=mx.float32))).dtype}")
    print(f"  g dtype: python={pg.dtype} cpp={cg.dtype}")
    stat("g: python vs cpp", pg, cg)
    stat("beta: python vs cpp", pb, cb)

    print("== 3. Is C++ softplus-result -> bf16 enough to match Python? ==")
    sp_py = nn.softplus(a + dt_bias)
    sp_cpp = mx.logaddexp(a + dt_bias, mx.array(0.0, dtype=mx.float32))
    stat("softplus: python(bf16) vs cpp(f32)", sp_py, sp_cpp)
    stat("softplus: python vs cpp->bf16", sp_py, sp_cpp.astype(mx.bfloat16))
    g_fix = mx.exp(-mx.exp(A_log.astype(mx.float32)) * sp_cpp.astype(mx.bfloat16))
    stat("g with cpp softplus cast to bf16", pg, g_fix)


if __name__ == "__main__":
    main()
