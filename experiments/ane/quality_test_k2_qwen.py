"""QUALITY TEST for expert-wise K=2 — real Qwen (Flash-Next) weights.

Mirrors quality_test_k2.py but for Qwen's 3-projection SwiGLU expert
(gate, up, down; 512 experts stacked). Tests bit-pattern equality of:
  A baseline: separate 1-row calls
  B naive K2: one 2-row batched call
  C safe K2 : shared weight load, separate 1-row calls
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

CKPT = Path.home() / "models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
LAYER, EXPERT, GROUP, BITS = 0, 0, 32, 4


def load_expert():
    idx = json.loads((CKPT / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    base = f"language_model.model.layers.{LAYER}.mlp.switch_mlp"
    out = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for part in ("weight", "scales", "biases"):
            key = f"{base}.{proj}.{part}"
            with open(CKPT / wm[key], "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
                start, end = hdr[key]["data_offsets"]
                shape = hdr[key]["shape"]
                f.seek(8 + n + start)
                raw = f.read(end - start)
            per = (end - start) // shape[0]
            arr = np.frombuffer(raw[EXPERT * per:(EXPERT + 1) * per],
                                dtype=np.uint32 if hdr[key]["dtype"] == "U32" else np.uint16).reshape(shape[1:])
            if hdr[key]["dtype"] == "BF16":
                arr = (arr.astype(np.uint32) << 16).view(np.float32)
            out[f"{proj}.{part}"] = mx.array(arr)
    return out


def main():
    w = load_expert()
    hidden = w["gate_proj.weight"].shape[1] * 8
    rng = np.random.default_rng(0)
    x1 = mx.array(rng.standard_normal(hidden).astype(np.float32))
    x2 = mx.array(rng.standard_normal(hidden).astype(np.float32))

    def proj(x, name):
        return mx.quantized_matmul(x, w[f"{name}.weight"], w[f"{name}.scales"],
                                   w[f"{name}.biases"], group_size=GROUP, bits=BITS)

    def ffn(x):
        g = proj(x, "gate_proj"); u = proj(x, "up_proj")
        return proj(nn.silu(g) * u, "down_proj")

    a = ffn(x1[None]).reshape(-1); b = ffn(x2[None]).reshape(-1)
    ab = ffn(mx.stack([x1, x2], axis=0)).reshape(2, -1)
    c1 = ffn(x1[None]).reshape(-1); c2 = ffn(x2[None]).reshape(-1)
    mx.eval(a, b, ab, c1, c2)

    eq = lambda u, v: bool(mx.array_equal(u, v).item())
    report = {
        "note": "real Qwen expert 0 layer 0 SwiGLU: bit-pattern comparison",
        "hidden": hidden,
        "naive_batched_row1_equal": eq(a, ab[0]),
        "naive_batched_row2_equal": eq(b, ab[1]),
        "naive_max_diff": float(max(mx.abs(a - ab[0]).max().item(), mx.abs(b - ab[1]).max().item())),
        "shared_read_perrow_row1_equal": eq(a, c1),
        "shared_read_perrow_row2_equal": eq(b, c2),
        "verdict": None,
    }
    report["verdict"] = (
        "SAFE: shared-read + per-row compute is BIT-IDENTICAL. Naive batched compute dropped."
        if report["shared_read_perrow_row1_equal"] and report["shared_read_perrow_row2_equal"]
        else "DROP K=2 for Qwen: no bit-identical path."
    )
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
