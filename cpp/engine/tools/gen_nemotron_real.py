#!/usr/bin/env python3
"""Generate a REAL Nemotron G64-checkpoint parity case for the C++ engine.

Loads actual Nemotron 3.5 Lightning expert weights (Q4 affine G64) via the
engine's streamed loader, computes the reference relu2 MoE block output with
mlx gather_qmm, and saves vectors for the C++ test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def save(path, arr):
    mx.save(str(path), arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="cpp/engine/tests/vectors/real_nemo")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from models.nemotron35.streamed_model import load_streamed, validate_config
    ckpt = Path(args.checkpoint).expanduser().resolve()
    streamed = load_streamed(ckpt, pool_bytes=512 * 1024 * 1024,
                             verify=False, config_check=validate_config)
    index = streamed.reader.index
    rng = np.random.default_rng(31415)
    E, K, T = 16, 6, 1
    H = streamed.config["hidden_size"]         # 2688
    I = streamed.config["moe_intermediate_size"]  # 1856
    G = 64

    # Find a real MoE layer and expert keys from the loaded stack.
    # The streamed model exposes reader tensors by name; read the first MoE layer.
    layer = next(i for i, b in enumerate(streamed.config["layers_block_type"]) if b == "moe")
    # Names in this checkpoint use 'experts.<e>.fc1/fc2' (affine_index convention).
    def tensor(name):
        arr = np.asarray(streamed.reader.get_array(name))
        return mx.array(arr)

    base = f"backbone.layers.{layer}.mixer.experts"
    experts = list(range(E))
    def part(proj, e):
        return tensor(f"{base}.{e}.{proj}")
    # Pack order: gate=fc1 (hidden->inter), up is fused with fc1 in Nemotron?
    # Nemotron relu2 MLP has ONE up projection (fc1) then relu2 then fc2 (down).
    # There is no separate gate. Mirror streamed_experts: fc1 -> relu2 -> fc2.
    up_w = mx.stack([part("fc1.weight", e) for e in experts])
    down_w = mx.stack([part("fc2.weight", e) for e in experts])
    up_s = mx.stack([mx.array(np.asarray(part("fc1.scales", e)).view(np.uint16)).view(mx.bfloat16) for e in experts])
    up_b = mx.stack([mx.array(np.asarray(part("fc1.biases", e)).view(np.uint16)).view(mx.bfloat16) for e in experts])
    down_s = mx.stack([mx.array(np.asarray(part("fc2.scales", e)).view(np.uint16)).view(mx.bfloat16) for e in experts])
    down_b = mx.stack([mx.array(np.asarray(part("fc2.biases", e)).view(np.uint16)).view(mx.bfloat16) for e in experts])

    logits = mx.array(rng.normal(size=(T, E)).astype(np.float32))
    gates = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-K, axis=-1)[..., -K:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    route = mx.array(np.asarray(inds).astype(np.uint32))
    x = mx.array(rng.normal(size=(T, H)).astype(np.float32) * 0.3)

    x4 = mx.expand_dims(x, (-2, -3))
    up = mx.gather_qmm(x4, up_w, up_s, up_b, rhs_indices=route, transpose=True,
                       group_size=G, bits=4, mode="affine")
    a = mx.relu2(up)   # squared ReLU (Nemotron), NOT SwiGLU
    d = mx.gather_qmm(a, down_w, down_s, down_b, rhs_indices=route, transpose=True,
                      group_size=G, bits=4, mode="affine")
    d = mx.squeeze(d, axis=-2)
    expected = (d * scores[..., None]).sum(axis=-2)

    # Nemotron relu2 MoE has no separate gate: set gate = up so swiglu != used.
    # Instead expose a dedicated relu2 path flag to C++ via meta 'activation':'relu2'.
    arrays = {"x": x, "route": route, "scores": scores,
              "gw": up_w, "gs": up_s, "gb": up_b,   # reinterpret up as 'gate'
              "uw": up_w, "us": up_s, "ub": up_b,   # up==gate (ignored for relu2)
              "dw": down_w, "ds": down_s, "db": down_b,
              "expected": expected}
    for k, v in arrays.items():
        save(out / f"{k}.npy", v)
    meta = {"E": E, "H": H, "I": I, "G": G, "K": K, "T": T, "bits": 4,
            "activation": "relu2", "layer": layer}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    streamed.close()
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
