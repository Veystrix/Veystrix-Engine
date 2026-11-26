"""Cache-aware routing for streamed Nemotron experts: exact when off or observing, a bounded swap when on."""
import os
import shutil
import tempfile
import unittest
import unittest.mock
from collections import deque
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models import nemotron_h

from models.nemotron35 import cache_routing
from models.nemotron35.streamed_model import load_streamed
from models.nemotron35.tiny_checkpoint import TINY_CONFIG, permissive_config_check, write_checkpoint

_dir = None


def setUpModule():
    global _dir
    _dir = tempfile.mkdtemp(prefix="nemotron-cache-routing-")
    write_checkpoint(Path(_dir) / "ck")


def tearDownModule():
    shutil.rmtree(_dir, ignore_errors=True)


def gate(seed=1):
    mx.random.seed(seed)
    args = nemotron_h.ModelArgs.from_dict(TINY_CONFIG)
    g = nemotron_h.MoEGate(args)
    g.weight = mx.random.normal(g.weight.shape) * 0.5
    g.e_score_correction_bias = mx.random.normal(g.e_score_correction_bias.shape) * 0.05
    return g


def exact(g, x):
    inds, scores = nemotron_h.MoEGate.__call__(g, x)
    order = np.argsort(np.asarray(inds).reshape(-1))
    return np.asarray(inds).reshape(-1)[order], np.asarray(scores).reshape(-1)[order]


class GateTests(unittest.TestCase):
    def install(self, g, kind, epsilon=0.02, window=3):
        shared = cache_routing._Shared(kind, epsilon, window, None)
        g.__class__ = cache_routing.CacheAwareGate
        object.__setattr__(g, "_cache_shared", shared)
        object.__setattr__(g, "_cache_history", deque(maxlen=window))
        return shared

    def test_observe_and_zero_epsilon_route_exactly(self):
        g = gate()
        xs = [mx.random.normal((1, 1, TINY_CONFIG["hidden_size"])) for _ in range(6)]
        want = [exact(g, x) for x in xs]
        for kind, eps in (("observe", 0.02), ("swap", 0.0)):
            g2 = gate()
            self.install(g2, kind, eps)
            for x, (ids, weights) in zip(xs, want):
                inds, scores = g2(x)
                order = np.argsort(np.asarray(inds).reshape(-1))
                np.testing.assert_array_equal(np.asarray(inds).reshape(-1)[order], ids)
                np.testing.assert_allclose(np.asarray(scores).reshape(-1)[order], weights, rtol=1e-5, atol=1e-6)

    def test_prefill_rows_and_off_route_exactly(self):
        g = gate()
        x = mx.random.normal((1, 5, TINY_CONFIG["hidden_size"]))
        want_inds, want_scores = nemotron_h.MoEGate.__call__(g, x)
        self.install(g, "swap", 1.0)
        inds, scores = g(x)
        np.testing.assert_array_equal(np.asarray(inds), np.asarray(want_inds))
        np.testing.assert_allclose(np.asarray(scores), np.asarray(want_scores))

    def test_swap_prefers_recent_experts_and_keeps_weights_normalised(self):
        g = gate()
        shared = self.install(g, "swap", 10.0)          # a tolerance that allows any swap
        first = mx.random.normal((1, 1, TINY_CONFIG["hidden_size"]))
        inds, _ = g(first)
        resident = set(int(e) for e in np.asarray(inds).reshape(-1))
        second = mx.random.normal((1, 1, TINY_CONFIG["hidden_size"])) * 3
        inds2, scores2 = g(second)
        chosen = set(int(e) for e in np.asarray(inds2).reshape(-1))
        self.assertEqual(len(chosen), TINY_CONFIG["num_experts_per_tok"])
        self.assertEqual(chosen, resident)               # every miss was swapped for a resident expert
        self.assertAlmostEqual(float(np.asarray(scores2).sum()), TINY_CONFIG["routed_scaling_factor"], places=4)
        exact_ids, _ = exact(gate(), second)
        self.assertEqual(shared.swapped, len(set(int(e) for e in exact_ids) - resident))

    def test_observe_counts_possible_swaps_without_swapping(self):
        g = gate()
        shared = self.install(g, "observe")
        for _ in range(8):
            g(mx.random.normal((1, 1, TINY_CONFIG["hidden_size"])))
        summary = shared.summary()
        self.assertEqual(summary["swapped"], 0)
        self.assertEqual(summary["layer_calls"], 8)
        counts = list(summary["possible_by_epsilon"].values())
        self.assertEqual(counts, sorted(counts))        # a larger tolerance never allows fewer swaps


class ModelTests(unittest.TestCase):
    def logits(self, kind):
        env = {"VEYSTRIX_NEMOTRON_CACHE_ROUTING": kind}
        with unittest.mock.patch.dict(os.environ, env):
            streamed = load_streamed(Path(_dir) / "ck", pool_bytes=0, config_check=permissive_config_check,
                                     prefetch=False)
        self.addCleanup(streamed.close)
        tokens = mx.array([[3, 5, 7, 9, 11, 2, 4]])
        out = streamed.model(tokens)
        mx.eval(out)
        steps = []
        for t in (6, 8, 1, 4):
            step = streamed.model(mx.array([[t]]))
            mx.eval(step)
            steps.append(np.asarray(step.astype(mx.float32)))
        return np.asarray(out.astype(mx.float32)), steps

    def test_observe_leaves_logits_identical(self):
        base_prefill, base_steps = self.logits("off")
        seen_prefill, seen_steps = self.logits("observe")
        np.testing.assert_allclose(seen_prefill, base_prefill, rtol=1e-4, atol=1e-4)
        for a, b in zip(seen_steps, base_steps):
            np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3)

    def test_off_installs_nothing(self):
        with unittest.mock.patch.dict(os.environ, {"VEYSTRIX_NEMOTRON_CACHE_ROUTING": "off"}):
            streamed = load_streamed(Path(_dir) / "ck", pool_bytes=0, config_check=permissive_config_check,
                                     prefetch=False)
        self.addCleanup(streamed.close)
        for layer in streamed.model.backbone.layers:
            gate_ = getattr(layer.mixer, "gate", None)
            if gate_ is not None:
                self.assertIs(type(gate_), nemotron_h.MoEGate)


if __name__ == "__main__":
    unittest.main()
