#!/usr/bin/env python3
"""Bit-exact vectors for the SORTED Nemotron relu2 MoE path (plan P5b).

Drives the real mlx_lm SwitchMLP sorted branch (indices.size >= 64):
  _gather_sort -> gather_qmm(sorted_indices=True) -> relu2 -> ... -> _scatter_unsort
This is the reduction grouping Python uses for batched prefill (T>=11 at
top-6/128); the unsorted path only matches T=1 decode.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv


def make_projection(E, out, inp, group, rng):
    codes = rng.integers(0, 16, size=(E, out, inp), dtype=np.uint8)
    packed = np.zeros((E, out, inp // 8), dtype=np.uint32)
    for n in range(8):
        packed |= codes[..., n::8].astype(np.uint32) << (n * 4)
    scales = mx.array((rng.random((E, out, inp // group)).astype(np.float32) * 0.5 + 0.5)).astype(mx.bfloat16)
    biases = mx.array((rng.random((E, out, inp // group)).astype(np.float32) * 0.2 - 0.1)).astype(mx.bfloat16)
    return mx.array(packed), scales, biases


def emit(d: Path, T: int, K: int, E: int, H: int, I: int, G: int, seed: int):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    route = mx.array(np.sort(rng.choice(E, (T, K), replace=True), axis=-1).astype(np.uint32))
    scores = mx.array(rng.random((T, K)).astype(np.float32))
    scores = scores / scores.sum(axis=-1, keepdims=True)
    x = mx.array(rng.normal(size=(T, H)).astype(np.float32) * 0.3)

    up_w, up_s, up_b = make_projection(E, I, H, G, rng)
    dn_w, dn_s, dn_b = make_projection(E, H, I, G, rng)

    # Sorted path (exactly SwitchMLP with do_sort=True).
    x4 = mx.expand_dims(x, (-2, -3))
    xs, idx, inv = _gather_sort(x4, route)
    u = mx.gather_qmm(xs, up_w, up_s, up_b, rhs_indices=idx, transpose=True,
                      group_size=G, bits=4, mode="affine", sorted_indices=True)
    a = mx.square(mx.maximum(u, mx.array(0.0)))
    dn = mx.gather_qmm(a, dn_w, dn_s, dn_b, rhs_indices=idx, transpose=True,
                       group_size=G, bits=4, mode="affine", sorted_indices=True)
    dn = dn[inv]
    dn = mx.unflatten(dn, 0, route.shape)
    expected = (mx.squeeze(dn, axis=-2) * scores[..., None]).sum(axis=-2)

    arrays = dict(x=x, route=route, scores=scores,
                  up_w=up_w, up_s=up_s, up_b=up_b,
                  dn_w=dn_w, dn_s=dn_s, dn_b=dn_b, expected=expected)
    for k, v in arrays.items():
        mx.save(str(d / f"{k}.npy"), v)
    (d / "meta.json").write_text(json.dumps(
        dict(T=T, K=K, E=E, H=H, I=I, G=G, sorted=True), indent=2) + "\n")


def main():
    out = Path("cpp/engine/tests/vectors_relu2_sorted")
    for name, T in [("t16", 16), ("t64", 64), ("t128", 128)]:
        emit(out / name, T=T, K=6, E=128, H=64, I=128, G=64, seed=T + 7000)
    print("wrote sorted relu2 cases: t16, t64, t128")


if __name__ == "__main__":
    main()
