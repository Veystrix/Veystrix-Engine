"""Real-weight Nemotron q_proj: ANE vs MLX GPU on prefill shapes.

Reads the fp16 export from export_nemotron_qproj.py. No production code is
touched. Runs under the guard, plugged in.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

OUT = Path(__file__).resolve().parent / "out"
RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)


def build_model(w, rows):
    import coremltools as ct
    from coremltools.models import neural_network as nn, datatypes as dt
    b = nn.NeuralNetworkBuilder(
        input_features=[("x", dt.Array(rows, w.shape[1], 1))],
        output_features=[("y", None)],
        disable_rank5_shape_mapping=True,
    )
    b.add_flatten_to_2d(name="flat", input_name="x", output_name="flat_o", axis=1)
    b.add_inner_product(name="fc", W=w, b=np.zeros(w.shape[0], dtype=np.float32),
                        input_channels=w.shape[1], output_channels=w.shape[0],
                        has_bias=False, input_name="flat_o", output_name="y")
    return ct.models.MLModel(b.spec, compute_units=ct.ComputeUnit.CPU_AND_NE)


def p50(fn, n=20):
    fn()
    for _ in range(2):
        fn()
    s = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        s.append((time.perf_counter() - t0) * 1000)
    s.sort()
    return s[len(s)//2]


def main():
    w = np.load(OUT / "nemotron_q5_qproj_fp16.npy")
    report = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "source": "real Nemotron layer 5 q_proj, dequantized Q4 G64 to fp16",
              "weight_shape": list(w.shape), "cases": []}
    wx = mx.array(w.astype(np.float32))
    for rows in (512, 2048, 4096):
        x = np.load(OUT / f"nemotron_qproj_x{rows}.npy")
        x3 = x[:, :, None]
        model = build_model(w.astype(np.float32), rows)
        ane_ms = p50(lambda: model.predict({"x": x3}))
        xx = mx.array(x.astype(np.float32))
        gpu_ms = p50(lambda: mx.eval(xx @ wx.T))
        y_ane = model.predict({"x": x3})["y"].astype(np.float32)
        y_gpu = np.asarray(xx @ wx.T)
        diff = np.abs(y_ane - y_gpu)
        report["cases"].append({
            "rows": rows,
            "ane_ms": round(ane_ms, 3), "gpu_ms": round(gpu_ms, 3),
            "ane_vs_gpu": round(gpu_ms / ane_ms, 3),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "gpu_abs_max": float(np.abs(y_gpu).max()),
        })
        print(report["cases"][-1])
    dest = RES / "ane_nemotron_qproj.json"
    dest.write_text(json.dumps(report, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
