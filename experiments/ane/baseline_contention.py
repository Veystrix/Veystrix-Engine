"""Reproduce the prefill-relevant contention measurement under the guard.

The 2026-11-07 report found ANE side-load added +0.3% / -0.2% to compute-bound
GPU work but +3.5% to +17.6% to bandwidth-bound (decode-like) work. Prefill is
compute-bound, so this is the favorable case. Two processes, reversed rounds,
as in the original; random weights only.

Run under the guard, plugged in.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
DURATION = 3.0


def _make_spec(rows=256, inner=2560, out=512, seed=0):
    import coremltools as ct
    from coremltools.models import neural_network as nn, datatypes as dt
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((out, inner)) * 0.02).astype(np.float32)
    b = np.zeros(out, dtype=np.float32)
    builder = nn.NeuralNetworkBuilder(
        input_features=[("x", dt.Array(rows, inner, 1))],
        output_features=[("y", None)],
        disable_rank5_shape_mapping=True,
    )
    builder.add_flatten_to_2d(name="flat", input_name="x", output_name="flat_o", axis=1)
    builder.add_inner_product(name="fc", W=w, b=b, input_channels=inner,
                              output_channels=out, has_bias=True,
                              input_name="flat_o", output_name="y")
    return builder.spec


def ane_worker(rate_hz, stop, queue):
    """ANE matmul side-load. rate_hz=0 means flat out."""
    import coremltools as ct
    m = ct.models.MLModel(_make_spec(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    x = np.random.default_rng(7).standard_normal((256, 2560, 1)).astype(np.float32)
    m.predict({"x": x})
    n = 0
    while stop.is_set() is False:
        t0 = time.perf_counter()
        m.predict({"x": x})
        n += 1
        if rate_hz:
            target = 1.0 / rate_hz
            left = target - (time.perf_counter() - t0)
            if left > 0:
                time.sleep(left)
    queue.put(n)


def gpu_worker(stop, queue):
    """Compute-bound GPU matmul, the prefill shape. Returns p50/mean ms."""
    import mlx.core as mx
    a = mx.array(np.random.default_rng(1).standard_normal((1024, 2560)).astype(np.float32))
    b = mx.array(np.random.default_rng(2).standard_normal((2560, 2560)).astype(np.float32))
    def step():
        y = a @ b
        mx.eval(y)
    step()
    samples = []
    while stop.is_set() is False:
        t0 = time.perf_counter()
        step()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    queue.put({"p50_ms": samples[len(samples)//2], "n": len(samples)})


def run_round(mpctx, rate_hz, label, order):
    stop = mpctx.Event()
    q = mpctx.Queue()
    gpu = mpctx.Process(target=gpu_worker, args=(stop, q))
    ane = mpctx.Process(target=ane_worker, args=(rate_hz, stop, q)) if rate_hz is not None else None
    # Start in the requested order, reversed on the second round.
    first, second = (ane, gpu) if order == "ane_first" else (gpu, ane)
    for p in (first, second):
        if p is not None:
            p.start()
    time.sleep(DURATION)
    stop.set()
    results = []
    for p in (gpu, ane):
        if p is not None:
            p.join(timeout=10)
    # Drain: the GPU worker puts a dict, the ANE worker puts an int.
    got = []
    while not q.empty():
        got.append(q.get())
    gpu_res = next((r for r in got if isinstance(r, dict)), {})
    return {"label": label, "rate_hz": rate_hz, "order": order, "gpu": gpu_res}


def main():
    mpctx = mp.get_context("spawn")
    report = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "note": "random weights; GPU compute-bound (prefill shape); 2 reversed rounds",
              "rounds": []}
    for order in ("gpu_first", "ane_first"):
        report["rounds"].append(run_round(mpctx, None, "none", order))
        for rate in (200, 0):
            report["rounds"].append(run_round(mpctx, rate, f"ANE {rate}Hz" if rate else "ANE flat", order))
    dest = OUT / "ane_baseline_contention.json"
    dest.write_text(json.dumps(report, indent=1))
    print(f"wrote {dest}")
    for r in report["rounds"]:
        g = r["gpu"]
        print(f"{r['label']:<12} {r['order']:<10} p50={g.get('p50_ms', 0):.3f} ms n={g.get('n', 0)}")


if __name__ == "__main__":
    main()
