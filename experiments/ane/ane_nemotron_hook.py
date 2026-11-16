"""Prototype: route Nemotron attention projections to the ANE during prefill.

Import this module before the backend is constructed. Off when
VEYSTRIX_NEMOTRON_ANE_PREFILL is not "1". Lives in experiments/ only; no
production file is modified. The weights come from the live model, dequantized
once and baked into a Core ML model per (layer, row count).

Numerics are fp16, NOT bit-identical to the bf16 reference path.
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
except Exception:  # pragma: no cover - imported before repo on path in some probes
    _attn = None
    _ORIG_ATTN = None


def _coreml_model(linear, rows):
    import coremltools as ct
    from coremltools.models import neural_network as cnn, datatypes as dt
    mx.eval(linear.weight, linear.scales, linear.biases)
    w = np.asarray(mx.dequantize(linear.weight, linear.scales, linear.biases,
                                 group_size=linear.group_size, bits=linear.bits).astype(mx.float32))
    out, inner = w.shape
    b = cnn.NeuralNetworkBuilder(
        input_features=[("x", dt.Array(rows, inner, 1))],
        output_features=[("y", None)],
        disable_rank5_shape_mapping=True,
    )
    b.add_flatten_to_2d(name="flat", input_name="x", output_name="flat_o", axis=1)
    b.add_inner_product(name="fc", W=w, b=np.zeros(out, dtype=np.float32),
                        input_channels=inner, output_channels=out, has_bias=False,
                        input_name="flat_o", output_name="y")
    return ct.models.MLModel(b.spec, compute_units=ct.ComputeUnit.CPU_AND_NE)


def _ane_call(self, x):
    key = id(self)
    if not ENABLED or key not in ACTIVE or x.ndim < 2 or x.shape[-2] < MIN_ROWS:
        return _ORIG_LINEAR(self, x)
    rows = x.shape[-2]
    ck = (key, rows)
    model = _MODELS.get(ck)
    if model is None:
        model = _MODELS[ck] = _coreml_model(self, rows)
    inp = np.asarray(x.astype(mx.float16)).reshape(rows, -1, 1)
    y = model.predict({"x": inp})["y"]
    return mx.array(y).astype(x.dtype).reshape(x.shape[:-1] + (y.shape[-1],))


def _attn_call(self, x, mask=None, cache=None):
    if ENABLED:
        ACTIVE.add(id(self.q_proj))
        ACTIVE.add(id(self.o_proj))
    return _ORIG_ATTN(self, x, mask, cache)


if ENABLED:
    nn.QuantizedLinear.__call__ = _ane_call
    if _attn is not None:
        _attn.StreamedAttention.__call__ = _attn_call
