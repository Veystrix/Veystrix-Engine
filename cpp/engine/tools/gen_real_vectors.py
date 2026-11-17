#!/usr/bin/env python3
"""Generate a REAL-checkpoint parity case for the C++ engine.

Loads actual Qwen3.8-Flash-Next expert weights (Q4 affine G32) from the
checkpoint via the engine's SafeTensorStore, picks a handful of real experts,
and computes the reference MoE output with the same mlx gather_qmm ops. Saves
the exact tensors + expected output as .npy so the C++ test can reproduce it
bit-for-bit. This proves the C++ engine works on real weights, not just
synthetic ones.

Usage:
  .venv/bin/python cpp/engine/tools/gen_real_vectors.py \
    --checkpoint /Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP \
    --out cpp/engine/tests/vectors/real
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from models.flashnext.store import SafeTensorStore


def save(path, arr):
    mx.save(str(path), arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="cpp/engine/tests/vectors/real")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", type=int, default=8, help="how many real experts to include")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    store = SafeTensorStore(args.checkpoint)
    base = f"language_model.model.layers.{args.layer}.mlp.switch_mlp"
    # Number of experts in this layer from the weight tensor shape.
    wname = f"{base}.gate_proj.weight"
    n_experts = store.shape(wname)[0]
    E = min(args.experts, n_experts)
    rng = np.random.default_rng(2026)

    def part(name, experts, meta=False):
        raw = store.rows_np(f"{base}.{name}", experts)
        arr = mx.array(raw)
        if meta:  # scales/biases are uint16 holding bfloat16 bit patterns
            arr = arr.view(mx.bfloat16)
        return arr

    experts = list(range(E))
    gw = part("gate_proj.weight", experts)
    gs = part("gate_proj.scales", experts, meta=True)
    gb = part("gate_proj.biases", experts, meta=True)
    uw = part("up_proj.weight", experts)
    us = part("up_proj.scales", experts, meta=True)
    ub = part("up_proj.biases", experts, meta=True)
    dw = part("down_proj.weight", experts)
    ds = part("down_proj.scales", experts, meta=True)
    db = part("down_proj.biases", experts, meta=True)

    H = store.shape(wname)[2] * 8          # packed uint32 -> 8 codes/word = hidden
    out_dim_gate = store.shape(wname)[1]    # intermediate width (out of gate)
    # SwiGLU: gate/up output the intermediate dim; down outputs hidden.
    # Verify shapes from the down weight.
    down_name = f"{base}.down_proj.weight"
    down_out = store.shape(down_name)[1]    # hidden
    down_in = store.shape(down_name)[2] * 8 # intermediate
    I = down_in
    G = 32
    K = min(4, E)
    T = 1
    hidden = down_out

    # Router logits over these E real experts, top-K.
    logits = mx.array(rng.normal(size=(T, E)).astype(np.float32))
    gates = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-K, axis=-1)[..., -K:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    route = mx.array(np.asarray(inds).astype(np.uint32))
    x = mx.array(rng.normal(size=(T, hidden)).astype(np.float32) * 0.3)

    x4 = mx.expand_dims(x, (-2, -3))
    def proj(w, s, b, inp):
        return mx.gather_qmm(inp, w, s, b, rhs_indices=route, transpose=True,
                             group_size=G, bits=4, mode="affine")
    g = proj(gw, gs, gb, x4)
    u = proj(uw, us, ub, x4)
    a = u * (g * mx.sigmoid(g))
    d = proj(dw, ds, db, a)
    d = mx.squeeze(d, axis=-2)
    expected = (d * scores[..., None]).sum(axis=-2)

    arrays = {"x": x, "route": route, "scores": scores,
              "gw": gw, "gs": gs, "gb": gb, "uw": uw, "us": us, "ub": ub,
              "dw": dw, "ds": ds, "db": db, "expected": expected}
    for k, v in arrays.items():
        save(out / f"{k}.npy", v)
    meta = {"E": E, "H": hidden, "I": I, "G": G, "K": K, "T": T, "bits": 4,
            "layer": args.layer, "checkpoint": args.checkpoint,
            "n_experts_in_layer": n_experts}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"wrote real-checkpoint vectors to {out}")


if __name__ == "__main__":
    main()
