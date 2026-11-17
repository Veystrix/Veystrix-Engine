#!/usr/bin/env python3
"""Generate bit-exact parity vectors for the C++ engine from the real Python ops.

Every vector is produced with the SAME MLX 0.32.2 ops the Python engine uses,
so the C++ test (which loads them with mx::load) must match byte-for-byte.

For each case we save:
  router_logits.npy    [tokens, experts] float32
  router_indices.npy   [tokens, k] int32   (expected)
  router_scores.npy    [tokens, k] float32 (expected)
  x.npy                [tokens, hidden] float32
  route.npy            [tokens, k] uint32
  scores.npy           [tokens, k] float32
  gw/gs/gb.npy         gate weight/scales/biases
  uw/us/ub.npy         up
  dw/ds/db.npy         down
  expected.npy         [tokens, hidden] float32 (moe_block output)
  meta.json            shapes/params

Usage:
  .venv/bin/python cpp/engine/tools/gen_vectors.py --out cpp/engine/tests/vectors
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

# Mirror the production router (adaptive_topk._moe_call, threshold 1.0).
def ref_router(logits: mx.array, k: int):
    gates = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    topk_mass = scores.sum(axis=-1, keepdims=True)
    scores = scores / topk_mass
    return inds, scores

# Mirror StreamingSwitchLinear.__call__ = mx.gather_qmm(...).
def ref_projection(x, w, s, b, route, group_size=32, bits=4):
    return mx.gather_qmm(x, w, s, b, rhs_indices=route, transpose=True,
                         group_size=group_size, bits=bits, mode="affine")

def ref_swiglu(gate, up):
    return up * (gate * mx.sigmoid(gate))

def ref_combine(expert_out, scores):
    return (expert_out * scores[..., None]).sum(axis=-2)

def ref_moe_block(x, route, scores, weights, group_size=32, bits=4):
    # Match StreamingSwitchGLU.__call__ decode path exactly.
    x4 = mx.expand_dims(x, (-2, -3))                 # [T,1,1,H]
    g = ref_projection(x4, weights["gw"], weights["gs"], weights["gb"], route, group_size, bits)
    u = ref_projection(x4, weights["uw"], weights["us"], weights["ub"], route, group_size, bits)
    a = ref_swiglu(g, u)                             # [T,k,1,inter]
    d = ref_projection(a, weights["dw"], weights["ds"], weights["db"], route, group_size, bits)
    d = mx.squeeze(d, axis=-2)                       # [T,k,H]
    return ref_combine(d, scores)                    # [T,H]

def make_projection(E, out, hidden, group, rng):
    """Q4 affine pack: uint32 weight (E,out,hidden/8), bf16 scales/biases (E,out,hidden/group)."""
    codes = rng.integers(0, 16, size=(E, out, hidden), dtype=np.uint8)
    packed = np.zeros((E, out, hidden // 8), dtype=np.uint32)
    for n in range(8):
        packed |= codes[..., n::8].astype(np.uint32) << (n * 4)
    scales = mx.array(rng.random((E, out, hidden // group)).astype(np.float32) * 0.5 + 0.5).astype(mx.bfloat16)
    biases = mx.array(rng.random((E, out, hidden // group)).astype(np.float32) * 0.2 - 0.1).astype(mx.bfloat16)
    return mx.array(packed), mx.array(scales), mx.array(biases)

def save_npy(path: Path, arr: mx.array):
    # mx.save writes the native .npy format including bfloat16, which C++
    # mx::load reads back bit-exactly. numpy.save cannot store bfloat16.
    mx.save(str(path), arr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)

    # Production-like shape, small enough for a fast test: E=16 experts,
    # hidden=64, inter=64, group=32, top-4, 2 tokens.
    E, H, I, G, K, T = 16, 64, 64, 32, 4, 2

    # Router vectors.
    logits = mx.array(rng.normal(size=(T, E)).astype(np.float32))
    inds, scores = ref_router(logits, K)
    save_npy(out / "router_logits.npy", logits)
    save_npy(out / "router_indices.npy", inds)
    save_npy(out / "router_scores.npy", scores)

    # MoE block vectors. Use the recorded route as uint32 (as production does).
    route_u32 = mx.array(np.asarray(inds).astype(np.uint32))
    x = mx.array(rng.normal(size=(T, H)).astype(np.float32) * 0.3)
    gw, gs, gb = make_projection(E, H, H, G, rng)   # gate: H -> H
    uw, us, ub = make_projection(E, I, H, G, rng)   # up:   H -> I
    dw, ds, db = make_projection(E, H, I, G, rng)   # down: I -> H
    weights = {"gw": gw, "gs": gs, "gb": gb, "uw": uw, "us": us, "ub": ub,
               "dw": dw, "ds": ds, "db": db}
    expected = ref_moe_block(x, route_u32, scores, weights, G, 4)

    arrays = {
        "x": x, "route": route_u32, "scores": scores,
        "gw": gw, "gs": gs, "gb": gb, "uw": uw, "us": us, "ub": ub,
        "dw": dw, "ds": ds, "db": db, "expected": expected,
    }
    for name, arr in arrays.items():
        save_npy(out / f"{name}.npy", arr)
    meta = {"E": E, "H": H, "I": I, "G": G, "K": K, "T": T,
            "bits": 4, "mlx_version": mx.__version__}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(arrays) + 3} vectors to {out}")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
