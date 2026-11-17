#!/usr/bin/env python3
"""Generate bit-exact vectors for the native streaming MoE path.

Uses the REAL SafeTensorStore + Qwen checkpoint (layer 0, a handful of real
experts), a fixed route/x/scores, and the exact mx.gather_qmm SwiGLU path.
Only x/route/scores/expected (+ meta.json) are saved: the C++ test is
deliberately forced to load the expert weights from the real shard itself
through the pager, not from these vectors.

Usage:
  .venv/bin/python cpp/engine/tools/gen_moe_stream_vectors.py \
    --checkpoint /Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP \
    --out cpp/engine/tests/vectors_moe_stream
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
    ap.add_argument("--out", default="cpp/engine/tests/vectors_moe_stream")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", type=int, default=8, help="real experts to include")
    ap.add_argument("--topk", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    store = SafeTensorStore(args.checkpoint)
    prefix = f"language_model.model.layers.{args.layer}.mlp.switch_mlp"
    wname = f"{prefix}.gate_proj.weight"
    n_experts = store.shape(wname)[0]
    E = min(args.experts, n_experts)
    K = min(args.topk, E)
    G = 32
    bits = 4
    T = 1

    def part(name, experts, meta=False):
        raw = store.rows_np(f"{prefix}.{name}", experts)
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

    H = store.shape(wname)[2] * 8          # packed uint32 -> 8 codes/word
    I = store.shape(f"{prefix}.down_proj.weight")[2] * 8

    # Fixed route / x / scores. Using a fixed seed keeps the case reproducible
    # across generator runs, so the checked-in vectors stay valid.
    rng = np.random.default_rng(2026)
    logits = mx.array(rng.normal(size=(T, E)).astype(np.float32))
    gates = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-K, axis=-1)[..., -K:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    route = mx.array(np.asarray(inds).astype(np.uint32))
    x = mx.array(rng.normal(size=(T, H)).astype(np.float32) * 0.3)

    x4 = mx.expand_dims(x, (-2, -3))

    def proj(w, s, b, inp):
        return mx.gather_qmm(inp, w, s, b, rhs_indices=route, transpose=True,
                             group_size=G, bits=bits, mode="affine")

    g = proj(gw, gs, gb, x4)
    u = proj(uw, us, ub, x4)
    a = u * (g * mx.sigmoid(g))
    d = proj(dw, ds, db, a)
    d = mx.squeeze(d, axis=-2)
    expected = (d * scores[..., None]).sum(axis=-2)
    mx.eval(x, route, scores, expected)

    # Only these four vectors are saved; weights MUST come from the shard.
    for name, arr in {"x": x, "route": route, "scores": scores,
                      "expected": expected}.items():
        save(out / f"{name}.npy", arr)

    meta = {"E": E, "H": H, "I": I, "G": G, "K": K, "T": T, "bits": bits,
            "layer": args.layer, "prefix": prefix,
            "checkpoint": args.checkpoint, "n_experts_in_layer": n_experts}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    store.close()
    print(json.dumps(meta, indent=2))
    print(f"wrote streaming-MoE vectors to {out}")


if __name__ == "__main__":
    main()
