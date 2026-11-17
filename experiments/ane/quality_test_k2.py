"""QUALITY TEST for expert-wise K=2 — real Nemotron weights, 2026-11-17.

Question: does grouping two tokens by expert preserve bit-identical output?
Two separable design choices are tested:
  (1) batched compute — one 2-row matmul per expert (naive K=2)
  (2) shared read, per-row compute — load expert ONCE, run two 1-row matmuls

Owner rule: ANY quality drop => drop the method entirely.
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

CKPT = Path.home() / "models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit-mlx-community"
LAYER, EXPERT, GROUP, BITS = 1, 0, 64, 4


def load_expert():
    idx = json.loads((CKPT / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    base = f"backbone.layers.{LAYER}.mixer.switch_mlp"
    out = {}
    for proj in ("fc1", "fc2"):
        for part in ("weight", "scales", "biases"):
            key = f"{base}.{proj}.{part}"
            shard = wm[key]
            with open(CKPT / shard, "rb") as f:
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
    hidden = w["fc1.weight"].shape[1] * 8
    rng = np.random.default_rng(0)
    x1 = mx.array(rng.standard_normal(hidden).astype(np.float32))
    x2 = mx.array(rng.standard_normal(hidden).astype(np.float32))

    def ffn(x):  # x: (rows, hidden), uses batched or single-row kernel by row count
        h = mx.quantized_matmul(x, w["fc1.weight"], w["fc1.scales"], w["fc1.biases"],
                                group_size=GROUP, bits=BITS)
        return mx.quantized_matmul(nn.relu2(h), w["fc2.weight"], w["fc2.scales"],
                                   w["fc2.biases"], group_size=GROUP, bits=BITS)

    # A: baseline — separate 1-row calls (two independent weight loads in reality)
    a = ffn(x1[None]).reshape(-1)
    b = ffn(x2[None]).reshape(-1)
    # B: naive K2 — one 2-row call (one load, batched GEMM)
    ab = ffn(mx.stack([x1, x2], axis=0)).reshape(2, -1)
    # C: safe K2 — one load, TWO separate 1-row calls from the same weights
    c1 = ffn(x1[None]).reshape(-1)
    c2 = ffn(x2[None]).reshape(-1)
    mx.eval(a, b, ab, c1, c2)

    eq = lambda u, v: bool(mx.array_equal(u, v).item())
    report = {
        "note": "real Nemotron expert 0, layer 1: bit-pattern comparison",
        "hidden": hidden,
        "naive_batched_row1_equal": eq(a, ab[0]),
        "naive_batched_row2_equal": eq(b, ab[1]),
        "naive_max_diff": float(max(mx.abs(a - ab[0]).max().item(), mx.abs(b - ab[1]).max().item())),
        "shared_read_perrow_row1_equal": eq(a, c1),
        "shared_read_perrow_row2_equal": eq(b, c2),
        "verdict": None,
    }
    if report["shared_read_perrow_row1_equal"] and report["shared_read_perrow_row2_equal"]:
        report["verdict"] = (
            "SAFE: shared-read + per-row compute is BIT-IDENTICAL. "
            "Naive batched compute FAILS and is dropped; K=2 must load once but compute rows separately."
        )
    else:
        report["verdict"] = "DROP K=2 entirely: no bit-identical path."
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
