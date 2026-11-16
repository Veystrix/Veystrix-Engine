"""Zero-copy MLX -> Core ML (ANE) probe using DLPack + native PyObjC.

coremltools' Python predict() copies and marshals. This path builds an
MLMultiArray over the MLX buffer directly (no copy) and calls the native
MLModel API. Goal: find out whether native invocation removes the per-call
overhead that made the end-to-end ANE prototype 2.5x slower.
"""
from __future__ import annotations

import ctypes
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
MODEL_URL = OUT / "native_probe.mlmodelc"


# ---- minimal DLPack C layout (v0) ----
class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("ndim", ctypes.c_int32),
                ("dtype", DLDataType), ("device", DLDevice),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64)]


class DLManagedTensor(ctypes.Structure):
    _fields_ = [("dl_tensor", DLTensor), ("manager_ctx", ctypes.c_void_p),
                ("deleter", ctypes.c_void_p)]


_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_GetPointer.restype = ctypes.c_void_p


def mlx_to_multiarray(a):
    """Create an MLMultiArray sharing MLX's buffer. Caller keeps the returned
    keepalive (the MLX array and its memoryview) alive through prediction."""
    import CoreML
    from Foundation import NSArray
    mx.eval(a)
    mv = memoryview(a)
    data_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mv))
    shape = [int(v) for v in mv.shape]
    # MLX strides are in elements; MLMultiArray wants them in elements too.
    strides = [int(v) // a.itemsize for v in mv.strides]
    arr, aerr = CoreML.MLMultiArray.alloc().initWithDataPointer_shape_dataType_strides_deallocator_error_(
        ctypes.c_void_p(data_ptr), NSArray.arrayWithArray_(shape),
        CoreML.MLMultiArrayDataTypeFloat32, NSArray.arrayWithArray_(strides), None, None)
    assert arr is not None, aerr
    return arr, (a, mv)


def build_model():
    import coremltools as ct
    from coremltools.models import neural_network as nn, datatypes as dt
    rows, inner, out = 512, 2560, 512
    w = (np.random.default_rng(0).standard_normal((out, inner)) * 0.02).astype(np.float32)
    b = nn.NeuralNetworkBuilder(
        input_features=[("x", dt.Array(rows, inner, 1))],
        output_features=[("y", None)], disable_rank5_shape_mapping=True)
    b.add_flatten_to_2d(name="flat", input_name="x", output_name="fo", axis=1)
    b.add_inner_product(name="fc", W=w, b=np.zeros(out, np.float32),
                        input_channels=inner, output_channels=out, has_bias=False,
                        input_name="fo", output_name="y")
    m = ct.models.MLModel(b.spec, compute_units=ct.ComputeUnit.CPU_AND_NE)
    src = str(OUT / "native_probe.mlmodel")
    m.save(src)
    if MODEL_URL.exists():
        import shutil
        shutil.rmtree(MODEL_URL)
    return ct.models.utils.compile_model(src, str(MODEL_URL))


def main():
    dest = build_model()
    import CoreML
    from Foundation import NSURL
    cfg = CoreML.MLModelConfiguration.alloc()
    cfg.setComputeUnits_(CoreML.MLComputeUnitsCPUAndNeuralEngine)
    url = NSURL.fileURLWithPath_(dest)
    model, err = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(url, cfg, None)
    print("native model loaded:", model is not None, "err:", err)
    assert model is not None, err

    x_np = np.random.default_rng(1).standard_normal((512, 2560, 1)).astype(np.float32)

    def numpy_to_multiarray(arr):
        from Foundation import NSArray
        p = ctypes.addressof(ctypes.c_char.from_buffer(memoryview(arr)))
        shp = [int(v) for v in arr.shape]
        st = [int(v) // arr.itemsize for v in arr.strides]
        ma, err = CoreML.MLMultiArray.alloc().initWithDataPointer_shape_dataType_strides_deallocator_error_(
            ctypes.c_void_p(p), NSArray.arrayWithArray_(shp),
            CoreML.MLMultiArrayDataTypeFloat32, NSArray.arrayWithArray_(st), None, None)
        assert ma is not None, err
        return ma

    def predict(ma):
        fv = CoreML.MLFeatureValue.featureValueWithMultiArray_(ma)
        provider, perr = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_({"x": fv}, None)
        assert provider is not None, perr
        result, rerr = model.predictionFromFeatures_error_(provider, None)
        assert result is not None, rerr
        return result

    # Warm up and verify output.
    out = predict(numpy_to_multiarray(x_np))
    y0 = np.asarray(out.featureValueForName_("y").multiArrayValue())
    print("native output shape:", y0.shape)

    def native():
        predict(numpy_to_multiarray(x_np))

    samples = []
    for _ in range(30):
        t0 = time.perf_counter(); native(); samples.append((time.perf_counter()-t0)*1000)
    samples.sort()
    print(f"native numpy-pointer p50 = {samples[len(samples)//2]:.3f} ms")

    import coremltools as ct
    cm = ct.models.MLModel(ct.models.MLModel(dest).get_spec(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    def py():
        cm.predict({"x": x_np})
    s2 = []
    for _ in range(20):
        t0 = time.perf_counter(); py(); s2.append((time.perf_counter()-t0)*1000)
    s2.sort()
    print(f"coremltools predict p50 = {s2[len(s2)//2]:.3f} ms")


if __name__ == "__main__":
    main()
