"""Prototype: INT8 mlprogram Core ML projections on ANE for Nemotron prefill.

Variant of ane_nemotron_hook.py using INT8 quantized mlprograms, which
measured 5.24 ms on ANE vs 17.6 ms MLX GPU for the real q_proj, with half the
weight bytes of fp16. Off unless VEYSTRIX_NEMOTRON_ANE_PREFILL=1.
"""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ENABLED = os.environ.get("VEYSTRIX_NEMOTRON_ANE_PREFILL", "0") == "1"
MIN_ROWS = 384
ACTIVE: set[int] = set()
_MODELS: dict[tuple, object] = {}
_ORIG_LINEAR = nn.QuantizedLinear.__call__

try:
    from models.nemotron35 import attention as _attn
    _ORIG_ATTN = _attn.StreamedAttention.__call__
except Exception:
    _attn = None
    _ORIG_ATTN = None

try:
    from mlx_lm.models.nemotron_h import NemotronHMamba2Mixer as _Mamba
    _ORIG_MAMBA = _Mamba.__call__
except Exception:
    _Mamba = None
    _ORIG_MAMBA = None

TARGET = os.environ.get("ANE_TARGET", "attn")


def _model(linear, rows):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.optimize.coreml import OptimizationConfig, OpLinearQuantizerConfig, linear_quantize_weights
    mx.eval(linear.weight, linear.scales, linear.biases)
    w = np.asarray(mx.dequantize(linear.weight, linear.scales, linear.biases,
                                 group_size=linear.group_size, bits=linear.bits).astype(mx.float32))

    @mb.program(input_specs=[mb.TensorSpec(shape=(rows, w.shape[1]))])
    def prog(x):
        return mb.matmul(x=x, y=w.T)

    m = ct.convert(prog, minimum_deployment_target=ct.target.macOS15,
                   compute_units=ct.ComputeUnit.CPU_AND_NE)
    cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel"))
    return linear_quantize_weights(m, cfg)


def _ane_call(self, x):
    key = id(self)
    if not ENABLED or key not in ACTIVE or x.ndim < 2 or x.shape[-2] < MIN_ROWS:
        return _ORIG_LINEAR(self, x)
    rows = x.shape[-2]
    ck = (key, rows)
    model = _MODELS.get(ck)
    if model is None:
        model = _MODELS[ck] = _model(self, rows)
    inp = np.asarray(x.astype(mx.float16)).reshape(rows, -1).astype(np.float32)
    y = list(model.predict({"x": inp}).values())[0]
    return mx.array(y).astype(x.dtype).reshape(x.shape[:-1] + (y.shape[-1],))


def _attn_call(self, x, mask=None, cache=None):
    if ENABLED:
        limit = int(os.environ.get("ANE_MAX_LAYERS", "13"))
        if len(ACTIVE) // 2 < limit:
            ACTIVE.add(id(self.q_proj))
            ACTIVE.add(id(self.o_proj))
    return _ORIG_ATTN(self, x, mask, cache)


def _mamba_call(self, *args, **kwargs):
    if ENABLED and TARGET == "mamba":
        limit = int(os.environ.get("ANE_MAX_LAYERS", "39"))
        if len(ACTIVE) < limit:
            ACTIVE.add(id(self.in_proj))
    return _ORIG_MAMBA(self, *args, **kwargs)


if ENABLED:
    nn.QuantizedLinear.__call__ = _ane_call
    if _attn is not None and TARGET == "attn":
        _attn.StreamedAttention.__call__ = _attn_call
    if _Mamba is not None and TARGET == "mamba":
        _Mamba.__call__ = _mamba_call
