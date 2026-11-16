"""4-bit palettized Core ML q_proj: does it fit and run on ANE?

Addresses the memory blocker from REPORT_NEMOTRON.md: fp16 duplicates were
~21 MB per projection (546 MB for 26). Palettization should be ~4x smaller
and is ANE's native weight format. Verifies actual ANE placement, latency,
and accuracy against the real dequantized fp16 weights.
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
ROWS = 2048


def build(mode):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.optimize.coreml import (OptimizationConfig, OpPalettizerConfig,
                                             OpLinearQuantizerConfig, palettize_weights,
                                             linear_quantize_weights)
    w = np.load(OUT / "nemotron_q5_qproj_fp16.npy").astype(np.float32)

    @mb.program(input_specs=[mb.TensorSpec(shape=(ROWS, w.shape[1]))])
    def prog(x):
        return mb.matmul(x=x, y=w.T)

    m = ct.convert(prog, minimum_deployment_target=ct.target.macOS15,
                   compute_units=ct.ComputeUnit.CPU_AND_NE)
    if mode == "int8":
        cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel"))
        m = linear_quantize_weights(m, cfg)
    elif mode != "fp16":
        cfg = OptimizationConfig(global_config=OpPalettizerConfig(
            mode=mode.upper(), nbits=4, granularity="per_grouped_channel",
            group_size=64, enable_per_channel_scale=True))
        m = palettize_weights(m, cfg)
    return m


def p50(fn, n=15):
    fn()
    s = []
    for _ in range(n):
        t = time.perf_counter(); fn(); s.append((time.perf_counter()-t)*1000)
    s.sort()
    return s[len(s)//2]


def main():
    x = np.load(OUT / "nemotron_qproj_x2048.npy")
    report = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": ROWS, "modes": {}}
    for mode in ("fp16", "int8"):
        try:
            m = build(mode)
            # Verify placement via compute plan.
            from coremltools.models.compute_plan import MLComputePlan
            plan = MLComputePlan.load_from_path(m.get_compiled_model_path())
            layers = {}
            ns = plan.model_structure.neuralnetwork
            if ns:
                for layer in ns.layers:
                    u = plan.get_compute_device_usage_for_neuralnetwork_layer(layer)
                    layers[layer.name] = type(u.preferred_compute_device).__name__
            else:
                prog = plan.model_structure.program
                for fname, fn in (prog.functions if prog else {}).items():
                    for i, op in enumerate(fn.block.operations):
                        try:
                            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
                            layers[f"{fname}:{op.operator_name}{i}"] = type(u.preferred_compute_device).__name__
                        except Exception:
                            pass
            t = p50(lambda: m.predict({"x": x}))
            y = list(m.predict({"x": x}).values())[0]
            ref = np.load(OUT / "nemotron_q5_qproj_fp16.npy").astype(np.float32)
            ref_y = x @ ref.T
            diff = np.abs(y.astype(np.float32) - ref_y)
            report["modes"][mode] = {"placement": layers, "p50_ms": round(t, 3),
                                     "max_abs_diff": float(diff.max()),
                                     "mean_abs_diff": float(diff.mean())}
            print(mode, report["modes"][mode], flush=True)
        except Exception as exc:
            report["modes"][mode] = {"error": f"{type(exc).__name__}: {exc}"}
            print(mode, "ERROR", str(exc)[:200], flush=True)
    dest = RES / "palettized_qproj.json"
    dest.write_text(json.dumps(report, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
