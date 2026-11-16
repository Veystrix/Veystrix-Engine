"""Export one real Nemotron q_proj to fp16 for the ANE experiment.

Runs in the repo .venv (read-only on the checkpoint). Writes only to the
git-ignored experiments/ane/out/. No model code is modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

ROOT = Path.home() / "models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit-mlx-community"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
LAYER = 5  # first attention layer in layers_block_type
PREFIX = f"backbone.layers.{LAYER}.mixer.q_proj"


def main():
    idx = json.loads((ROOT / "model.safetensors.index.json").read_text())
    shard = idx["weight_map"][f"{PREFIX}.weight"]
    # safetensors' python frameworks decode neither MLX's packed uint32 weight
    # nor bfloat16 scales; read just the three byte ranges from the header and
    # widen bf16 ourselves. Avoids loading the whole 4.8 GB shard.
    import struct
    tensors = {}
    with open(ROOT / shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        for suffix, dtype in (("weight", np.uint32), ("scales", np.uint16), ("biases", np.uint16)):
            key = f"{PREFIX}.{suffix}"
            if key not in header:
                continue
            info = header[key]
            start, end = info["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            arr = np.frombuffer(raw, dtype=dtype)
            if dtype == np.uint16:
                arr = (arr.astype(np.uint32) << 16).view(np.float32)
            tensors[suffix] = mx.array(arr.reshape(info["shape"]))
    missing = {"weight", "scales", "biases"} - tensors.keys()
    if missing:
        raise RuntimeError(f"missing tensors: {sorted(missing)}")

    deq = mx.dequantize(tensors["weight"], tensors["scales"], tensors["biases"],
                        group_size=64, bits=4)
    mx.eval(deq)
    w = np.asarray(deq, dtype=np.float16)
    np.save(OUT / "nemotron_q5_qproj_fp16.npy", w)

    # Fixed random inputs for latency and numerics (a key makes them reproducible).
    for rows in (512, 2048, 4096):
        x = np.asarray(mx.random.normal((rows, w.shape[1]), key=mx.random.key(rows)),
                       dtype=np.float16)
        np.save(OUT / f"nemotron_qproj_x{rows}.npy", x)

    meta = {"layer": LAYER, "prefix": PREFIX, "shard": shard,
            "weight_shape": list(w.shape), "dequant_dtype": str(deq.dtype),
            "group_size": 64, "bits": 4, "fp16_MB": round(w.nbytes / 2**20, 1)}
    (OUT / "nemotron_qproj_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
