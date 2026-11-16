import os
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn

from models.nemotron35 import dense_gemm


def _layer(inputs, outputs):
    layer = nn.QuantizedLinear(inputs, outputs, bias=False, group_size=64, bits=4)
    layer.weight = mx.random.randint(0, 2**31 - 1, layer.weight.shape).astype(mx.uint32)
    layer.scales = (mx.random.uniform(shape=layer.scales.shape) * 0.1 + 0.01).astype(mx.bfloat16)
    layer.biases = (mx.random.normal(layer.biases.shape) * 0.01).astype(mx.bfloat16)
    return layer


class DenseGemmTests(unittest.TestCase):
    def setUp(self):
        dense_gemm.install()

    def test_flag_is_validated(self):
        with patch.dict(os.environ, {dense_gemm.FLAG: "2"}):
            with self.assertRaises(ValueError):
                dense_gemm.enabled()

    def test_off_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(dense_gemm.FLAG, None)
            self.assertFalse(dense_gemm.enabled())

    def test_bit_equal_to_the_quantized_matmul(self):
        layer = _layer(2688, 2688)
        for rows in (1, 8, dense_gemm.MIN_ROWS - 1, dense_gemm.MIN_ROWS, 513):
            x = (mx.random.normal((1, rows, 2688)) * 0.5).astype(mx.bfloat16)
            with patch.dict(os.environ, {dense_gemm.FLAG: "0"}):
                expected = layer(x)
            with patch.dict(os.environ, {dense_gemm.FLAG: "1"}):
                actual = layer(x)
            mx.eval(expected, actual)
            self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(mx.array_equal(actual, expected).item(), rows)

    def test_shapes_that_are_not_bit_equal_keep_the_stock_path(self):
        layer = _layer(256, 384)          # the dense GEMM differs from the quantized kernel at every row count for this shape
        x = (mx.random.normal((1, 512, 256)) * 0.5).astype(mx.bfloat16)
        with patch.dict(os.environ, {dense_gemm.FLAG: "0"}):
            expected = layer(x)
        with patch.dict(os.environ, {dense_gemm.FLAG: "1"}):
            actual = layer(x)
        mx.eval(expected, actual)
        self.assertTrue(mx.array_equal(actual, expected).item())
        self.assertFalse(dense_gemm._bit_equal(layer))

    def test_small_row_counts_use_the_stock_path(self):
        layer = _layer(2688, 2688)
        x = mx.ones((1, 4, 2688), dtype=mx.bfloat16)
        with patch.dict(os.environ, {dense_gemm.FLAG: "1"}), patch("mlx.core.dequantize") as dequantize:
            layer(x)
        dequantize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
