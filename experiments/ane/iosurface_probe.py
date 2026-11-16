"""Probe Apple's documented zero-copy path: IOSurface-backed MLMultiArray.

If this works, Core ML reads the caller's FP16 buffer with NO internal
allocation — the alternative to problem (3) in the earlier analysis.
"""
from __future__ import annotations

import ctypes
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
ROWS, INNER, OUTDIM = 2048, 2688, 512


def make_pixel_buffer(rows, cols):
    import Quartz
    from Foundation import NSDictionary
    attrs = NSDictionary.dictionaryWithObject_forKey_(
        NSDictionary.dictionary(), Quartz.kCVPixelBufferIOSurfacePropertiesKey)
    status, pb = Quartz.CVPixelBufferCreate(
        None, cols, rows, Quartz.kCVPixelFormatType_OneComponent16Half, attrs, None)
    if status != 0 or pb is None:
        raise RuntimeError(f"CVPixelBufferCreate status {status}")
    return pb


def main():
    import Quartz
    import CoreML
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.models.compute_plan import MLComputePlan

    pb = make_pixel_buffer(ROWS, INNER)
    print("pixel buffer:", pb is not None,
          "width", Quartz.CVPixelBufferGetWidth(pb), "height", Quartz.CVPixelBufferGetHeight(pb),
          "bpr", Quartz.CVPixelBufferGetBytesPerRow(pb), flush=True)

    ma = CoreML.MLMultiArray.alloc().initWithPixelBuffer_shape_(pb, [ROWS, INNER])
    print("MLMultiArray:", ma is not None,
          "dtype", ma.dataType() if ma else None, flush=True)

    # FP16 input model.
    w = np.load(OUT / "nemotron_q5_qproj_fp16.npy").astype(np.float16)

    from coremltools.converters.mil.mil import types as mil_types
    @mb.program(input_specs=[mb.TensorSpec(shape=(ROWS, INNER), dtype=mil_types.fp16)])
    def prog(x):
        return mb.matmul(x=x, y=w.T)

    m = ct.convert(prog, minimum_deployment_target=ct.target.macOS15,
                   compute_units=ct.ComputeUnit.CPU_AND_NE)
    from coremltools.optimize.coreml import OptimizationConfig, OpLinearQuantizerConfig, linear_quantize_weights
    cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel"))
    m = linear_quantize_weights(m, cfg)
    print("model converted", flush=True)

    # Fill the IOSurface backing with a real input, using the documented
    # base-address access (lock while writing, unlock before prediction).
    Quartz.CVPixelBufferLockBaseAddress(pb, 0)
    base = Quartz.CVPixelBufferGetBaseAddress(pb)
    x = np.load(OUT / "nemotron_qproj_x2048.npy").astype(np.float16)
    bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
    src = x.ctypes.data
    dst = int(base)
    row_bytes = INNER * 2
    for r in range(ROWS):
        ctypes.memmove(dst + r * bpr, src + r * row_bytes, row_bytes)
    Quartz.CVPixelBufferUnlockBaseAddress(pb, 0)

    fv = CoreML.MLFeatureValue.featureValueWithMultiArray_(ma)
    provider, perr = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
        {"x": fv}, None)
    result, rerr = m.predict(provider) if False else (None, None)
    # Use coremltools with the IOSurface-backed MultiArray directly.
    for _ in range(2):
        out = m.predict({"x": ma})
    y = list(out.values())[0]
    print("predict OK, out shape", np.asarray(y).shape, flush=True)

    t = []
    for _ in range(15):
        s = time.perf_counter(); m.predict({"x": ma}); t.append((time.perf_counter()-s)*1000)
    t.sort(); print("IOSurface-backed p50 ms", round(t[len(t)//2], 3), flush=True)


if __name__ == "__main__":
    main()
