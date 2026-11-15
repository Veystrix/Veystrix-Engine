import os
import unittest
from unittest import mock

import mlx.core as mx
import numpy as np
from mlx_lm.models import nemotron_h
from mlx_lm.models.ssm import ssm_attn

from models.nemotron35 import fast_ssm


def make(length, heads=8, groups=2, head_dim=16, width=32, seed=0, with_state=True):
    mx.random.seed(seed)
    x = (mx.random.normal((1, length, heads, head_dim)) * 0.7).astype(mx.bfloat16)
    b = (mx.random.normal((1, length, groups, width)) * 0.7).astype(mx.bfloat16)
    c = (mx.random.normal((1, length, groups, width)) * 0.7).astype(mx.bfloat16)
    dt = mx.random.normal((1, length, heads)).astype(mx.bfloat16)
    a_log = mx.log(mx.arange(1, heads + 1, dtype=mx.float32)) * 0.5
    bias = mx.random.uniform(-4, -1, (heads,))
    skip = mx.random.normal((heads,)).astype(mx.bfloat16)
    state = (mx.random.normal((1, heads, head_dim, width)) * 0.1).astype(mx.float32) if with_state else None
    return x, a_log, b, c, skip, dt, bias, state


def truth(x, a_log, b, c, skip, dt, bias, state):
    """Sequential float64 recurrence on the same (bf16) inputs."""
    f = lambda a: np.array(a.astype(mx.float32), dtype=np.float64)
    X, B, C, DT = f(x)[0], f(b)[0], f(c)[0], f(dt)[0]
    heads, groups = X.shape[1], B.shape[1]
    r = heads // groups
    S = np.zeros((heads, X.shape[2], B.shape[2])) if state is None else f(state)[0]
    A = -np.exp(np.array(a_log, dtype=np.float64))
    Y = np.zeros(X.shape)
    for t in range(X.shape[0]):
        v = DT[t] + np.array(bias, dtype=np.float64)
        v = np.clip(np.log1p(np.exp(-np.abs(v))) + np.maximum(v, 0), 0.001, 100.0)
        S = np.exp(v * A)[:, None, None] * S + (v[:, None] * X[t])[:, :, None] * np.repeat(B[t], r, axis=0)[:, None, :]
        Y[t] = np.einsum("hpd,hd->hp", S, np.repeat(C[t], r, axis=0)) + f(skip)[:, None] * X[t]
    return Y, S


class FastSsmTests(unittest.TestCase):
    def compare(self, length, with_state):
        args = make(length, with_state=with_state)
        stock_y, stock_s = ssm_attn(*args[:7], args[7])
        fast_y, fast_s = fast_ssm.fast_ssm_attn(*args[:7], args[7])
        mx.eval(stock_y, stock_s, fast_y, fast_s)
        self.assertEqual(fast_y.dtype, stock_y.dtype)
        self.assertEqual(fast_y.shape, stock_y.shape)
        scale = float(mx.max(mx.abs(stock_y.astype(mx.float32))))
        diff = float(mx.max(mx.abs(fast_y.astype(mx.float32) - stock_y.astype(mx.float32))))
        self.assertLess(diff, 0.02 * scale)                                  # at most a couple of bf16 steps
        state_scale = float(mx.max(mx.abs(stock_s)))
        self.assertLess(float(mx.max(mx.abs(fast_s - stock_s))), 1e-4 * max(state_scale, 1.0))
        return args, stock_y, stock_s, fast_y, fast_s

    def test_matches_stock_whole_steps_and_ragged_tails(self):
        for length in (64, 256, 512, 700, 1000):
            for with_state in (False, True):
                with self.subTest(length=length, with_state=with_state):
                    self.compare(length, with_state)

    def test_error_against_float64_truth_is_no_worse_than_stock_output(self):
        args = make(300, with_state=True)
        want_y, want_s = truth(*args)
        stock_y, _ = ssm_attn(*args[:7], args[7])
        fast_y, fast_s = fast_ssm.fast_ssm_attn(*args[:7], args[7])
        err = lambda a: float(np.sqrt(np.mean((np.array(a.astype(mx.float32), dtype=np.float64)[0] - want_y) ** 2)))
        self.assertLess(err(fast_y), 1.05 * err(stock_y))
        self.assertLess(float(np.sqrt(np.mean((np.array(fast_s, dtype=np.float64)[0] - want_s) ** 2))), 1e-5)

    def test_wrapper_uses_the_fast_path_only_for_long_unmasked_calls(self):
        args = make(128)
        with mock.patch.object(fast_ssm, "fast_ssm_attn", return_value=("fast", None)) as fast, \
                mock.patch.object(fast_ssm, "_stock_ssm_update", return_value=("stock", None)) as stock:
            self.assertEqual(fast_ssm.ssm_update(*args[:7], args[7])[0], "fast")
            short = make(fast_ssm.MIN_ROWS - 1)
            self.assertEqual(fast_ssm.ssm_update(*short[:7], short[7])[0], "stock")
            self.assertEqual(fast_ssm.ssm_update(*args[:7], args[7], mask=mx.ones((1, 128), dtype=mx.bool_))[0], "stock")
            self.assertEqual(fast_ssm.ssm_update(*args[:7], args[7], lengths=mx.array([128]))[0], "stock")
            with mock.patch.dict(os.environ, {fast_ssm.FLAG: "0"}):
                self.assertEqual(fast_ssm.ssm_update(*args[:7], args[7])[0], "stock")
                self.assertEqual(fast_ssm.fingerprint(), {})
            self.assertEqual(fast_ssm.fingerprint(), {"ssm_prefill": "fused-v1"})
            self.assertEqual(fast.call_count, 1)
            self.assertEqual(stock.call_count, 4)
        with mock.patch.dict(os.environ, {fast_ssm.FLAG: "2"}):
            with self.assertRaises(ValueError):
                fast_ssm.enabled()

    def test_install_routes_the_mixer_and_is_idempotent(self):
        original = nemotron_h.ssm_update
        try:
            fast_ssm.install()
            fast_ssm.install()
            self.assertIs(nemotron_h.ssm_update, fast_ssm.ssm_update)
        finally:
            nemotron_h.ssm_update = original
            fast_ssm._installed = False


if __name__ == "__main__":
    unittest.main()
