"""Prefill-focused ANE baseline reproduction (owner-approved 2026-11-17).

Recreates the prefill-relevant parts of experiments/ane/REPORT.md, whose scripts
were untracked and are lost. Uses random weights: this measures placement,
latency, memory and contention, NOT predictor accuracy or model quality.

Run under the real-run guard, plugged in, e.g.:
  scripts/flashnext/real_run_guard.py --log results/ane-baseline.jsonl -- \
    experiments/ane/V/bin/python experiments/ane/baseline_prefill.py

No model checkpoint is read. No production file is changed.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

IN_DIM = 2560          # Flash-Next hidden size (router input in REPORT.md)
OUT_DIM = 512          # expert count / top-10-of-512
BATCHES = (1, 16, 256, 1024)
N_WARM = 3
# Short loops: the original report used 1-4 s per case on a fanless Air.
N_MEAS = {"1": 400, "16": 200, "256": 100, "1024": 40}


def _shape(*dims):
    return _dt.Array(*dims)


def build_linear(batch, in_dim=IN_DIM, out_dim=OUT_DIM, *, seed=0):
    """A bare linear predictor, built without PyTorch."""
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((out_dim, in_dim)) * 0.02).astype(np.float32)
    b = np.zeros(out_dim, dtype=np.float32)
    # Core ML neural-network multiarrays must be rank 1 or 3; use rank 3 and
    # flatten to 2D for the inner product, then reshape back to rank 3.
    builder = _nn.NeuralNetworkBuilder(
        input_features=[("x", _shape(batch, in_dim, 1))],
        output_features=[("y", None)],
        disable_rank5_shape_mapping=True,
    )
    builder.add_flatten_to_2d(name="flat_in", input_name="x",
                              output_name="flat_in_out", axis=1)
    builder.add_inner_product(name="fc", W=w, b=b, input_channels=in_dim,
                              output_channels=out_dim, has_bias=True,
                              input_name="flat_in_out", output_name="y")
    return builder.spec, w, b


def measure(fn, n):
    """p50/p95 of wall milliseconds over n calls, after warmup."""
    fn()  # compile / first call
    for _ in range(N_WARM):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {"p50_ms": samples[len(samples) // 2],
            "p95_ms": samples[int(len(samples) * 0.95)],
            "n": n}


def placement(model, label):
    """Static MLComputePlan: which device each layer prefers."""
    try:
        plan = _cp.load_from_path(model.get_compiled_model_path())
        layers = {}
        nn_struct = plan.model_structure.neuralnetwork
        for layer in (nn_struct.layers if nn_struct else []):
            usage = plan.get_compute_device_usage_for_neuralnetwork_layer(layer)
            layers[layer.name] = {
                "type": layer.type,
                "preferred": type(usage.preferred_compute_device).__name__,
                "supported": sorted({type(d).__name__ for d in usage.supported_compute_devices}),
            }
        return {"label": label, "layers": layers}
    except Exception as exc:
        return {"label": label, "error": f"{type(exc).__name__}: {exc}"}


def neural_footprint_mb():
    """ri_neural_footprint for this process (0 if unavailable)."""
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        class RUsage(ctypes.Structure):
            _fields_ = [("ru_utime", ctypes.c_uint64), ("ru_stime", ctypes.c_uint64),
                        ("ru_maxrss", ctypes.c_uint64), ("ru_ixrss", ctypes.c_uint64),
                        ("ru_idrss", ctypes.c_uint64), ("ru_isrss", ctypes.c_uint64),
                        ("ru_minflt", ctypes.c_uint64), ("ru_majflt", ctypes.c_uint64),
                        ("ru_nvcsw", ctypes.c_uint64), ("ru_nivcsw", ctypes.c_uint64),
                        ("ru_nswap", ctypes.c_uint64), ("ru_inblock", ctypes.c_uint64),
                        ("ru_oublock", ctypes.c_uint64), ("ru_msgsnd", ctypes.c_uint64),
                        ("ru_msgrcv", ctypes.c_uint64), ("ru_nsignals", ctypes.c_uint64)]
        #-macos-only rusage v6 with ri_neural_footprint as last field
        class RUsageV6(ctypes.Structure):
            _fields_ = RUsage._fields_ + [("ri_wired_size", ctypes.c_uint64),
                                          ("ri_resident_size", ctypes.c_uint64),
                                          ("ri_phys_footprint", ctypes.c_uint64),
                                          ("ri_pageins", ctypes.c_uint64),
                                          ("ri_neural_footprint", ctypes.c_uint64)]
        u = RUsageV6()
        if libc.getrusage(0, ctypes.byref(u)) == 0:
            return u.ri_neural_footprint / 2**20
    except Exception:
        pass
    return 0.0


def main():
    global _ct, _nn, _dt, _cp
    import coremltools as _ct
    from coremltools.models import neural_network as _nn
    from coremltools.models import datatypes as _dt
    from coremltools.models.compute_plan import MLComputePlan as _cp
    import mlx.core as mx

    report = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "on_ac": _power_state(), "coremltools": _ct.__version__,
              "note": "random weights; placement/latency/memory/contention only",
              "placement": [], "latency": [], "throughput": {}, "contention": []}

    for batch in BATCHES:
        spec, w, b = build_linear(batch)
        x = np.random.default_rng(batch).standard_normal((batch, IN_DIM)).astype(np.float32)
        x3 = x[:, :, None]

        for units, cu in (("coreml-cpu", _ct.ComputeUnit.CPU_ONLY),
                          ("coreml-ne", _ct.ComputeUnit.CPU_AND_NE)):
            m = _ct.models.MLModel(spec, compute_units=cu)
            if units == "coreml-ne":
                report["placement"].append(placement(m, f"router_b{batch}"))
            result = measure(lambda: m.predict({"x": x3}), N_MEAS[str(batch)])
            result.update(model=f"router_b{batch}", units=units, batch=batch,
                          neural_MB=neural_footprint_mb())
            report["latency"].append(result)

        # MLX GPU, same math and same random weights.
        wx = mx.array(w)
        bx = mx.array(b)
        xx = mx.array(x)
        def gpu_fn():
            y = xx @ wx.T + bx
            mx.eval(y)
        result = measure(gpu_fn, N_MEAS[str(batch)])
        result.update(model=f"router_b{batch}", units="mlx-gpu", batch=batch)
        report["latency"].append(result)

    # Large matmul throughput, the prefill-dense shape from REPORT.md section 2.
    for rows, inner, cols in ((1024, 2560, 2560), (512, 2048, 2048)):
        w = (np.random.default_rng(rows).standard_normal((cols, inner)) * 0.02).astype(np.float32)
        x = np.random.default_rng(cols).standard_normal((rows, inner)).astype(np.float32)
        spec, _, _ = build_linear(rows, in_dim=inner, out_dim=cols)
        m = _ct.models.MLModel(spec, compute_units=_ct.ComputeUnit.CPU_AND_NE)
        x3 = x[:, :, None]
        m.predict({"x": x3})
        t0 = time.perf_counter()
        for _ in range(5):
            m.predict({"x": x3})
        t_ne = (time.perf_counter() - t0) / 5
        wx = mx.array(w); xx = mx.array(x)
        def f():
            y = xx @ wx.T; mx.eval(y)
        f()
        t0 = time.perf_counter()
        for _ in range(5):
            f()
        t_gpu = (time.perf_counter() - t0) / 5
        flops = 2 * rows * inner * cols
        report["throughput"][f"{rows}x{inner}x{cols}"] = {
            "ane_tflops": flops / t_ne / 1e12, "gpu_tflops": flops / t_gpu / 1e12}

    dest = OUT / "ane_baseline_prefill.json"
    dest.write_text(json.dumps(report, indent=1))
    print(f"wrote {dest}")


def _power_state():
    try:
        import subprocess
        return subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                              text=True, timeout=5).stdout.strip().splitlines()[:2]
    except Exception:
        return []


if __name__ == "__main__":
    main()
