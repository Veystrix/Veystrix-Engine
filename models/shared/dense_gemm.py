"""Dense GEMM for large quantized projections in prefill (opt-in; shared by Nemotron and Flash-Next).

For a call with many rows, ``mx.dequantize`` followed by a dense bf16 GEMM is slightly faster than ``mx.quantized_matmul`` (the
dense GEMM reaches a higher rate than the fused 4-bit kernel). On the real Mamba projections at 2,048 rows it is 8% faster
(in_proj 38.8 -> 35.8 ms, out_proj 15.5 -> 14.3 ms) and the output is bit-equal to the quantized matmul (``mx.array_equal``).
The weights are still the checkpoint's 4-bit values: they are dequantized to a transient array per call (about 55 MB for the
largest projection) and nothing is kept. Only used for affine-quantized ``nn.QuantizedLinear`` layers without a bias and at
least ``MIN_ROWS`` rows; decode and speculative rows keep the stock path. The caller passes the environment variable that turns it on (``VEYSTRIX_NEMOTRON_DENSE_GEMM`` or ``FLASHNEXT_DENSE_GEMM``).

Bit equality is a property of the shape, not a rule: MLX's quantized kernel and the dense GEMM accumulate in the same order for
some (in, out) shapes and not for others (for example 256 -> 384 and 2688 -> 128 differ at every row count tried, 2688 -> 1856
only below 256 rows). So the first time a shape is seen it is checked on random data at three row counts at and above
``MIN_ROWS``, and only shapes that are bit-equal in all of them take the dense path; the verdict is cached per shape.
"""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

MIN_ROWS = 384
FLASHNEXT_FLAG = "FLASHNEXT_DENSE_GEMM"

_installed = False
_original = None
_VERIFY_ROWS = (MIN_ROWS, MIN_ROWS + 1, 640)
_verdicts = {}


_flags = []


def flag_enabled(flag: str) -> bool:
    value = os.environ.get(flag, "0") or "0"
    if value not in ("0", "1"):
        raise ValueError(f"{flag} must be 0 or 1")
    return value == "1"


def enabled() -> bool:
    return any(flag_enabled(flag) for flag in _flags)


def _bit_equal(self):
    """True when dense and quantized outputs are identical for this layer's shape and quantization at the verified row counts."""
    key = (self.weight.shape, self.scales.shape, self.group_size, self.bits)
    verdict = _verdicts.get(key)
    if verdict is None:
        inputs = self.scales.shape[-1] * self.group_size
        verdict = True
        for rows in _VERIFY_ROWS:
            # An explicit key: the check must not advance the global random state that seeded sampling depends on.
            x = (mx.random.normal((1, rows, inputs), key=mx.random.key(rows)) * 0.5).astype(mx.bfloat16)
            quantized = mx.quantized_matmul(x, self.weight, self.scales, self.biases, transpose=True,
                                            group_size=self.group_size, bits=self.bits)
            dense = x @ mx.dequantize(self.weight, self.scales, self.biases, group_size=self.group_size, bits=self.bits).T
            if not mx.array_equal(quantized, dense).item():
                verdict = False
                break
        _verdicts[key] = verdict
    return verdict


def _call(self, x):
    if (x.ndim >= 2 and x.shape[-2] >= MIN_ROWS and "bias" not in self and getattr(self, "mode", "affine") == "affine"
            and x.dtype == mx.bfloat16 and "biases" in self and _bit_equal(self)):
        weight = mx.dequantize(self.weight, self.scales, self.biases, group_size=self.group_size, bits=self.bits)
        return x @ weight.T
    return _original(self, x)


def install(flag: str) -> None:
    """Route large quantized linears through the dense path when ``flag`` is on (idempotent; flag read per call)."""
    global _installed, _original
    if flag not in _flags:
        _flags.append(flag)
    if _installed:
        return
    _original = nn.QuantizedLinear.__call__
    _installed = True

    def call(self, x):
        if enabled():
            return _call(self, x)
        return _original(self, x)

    nn.QuantizedLinear.__call__ = call

