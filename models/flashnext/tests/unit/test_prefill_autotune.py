from unittest import mock
import unittest

from models.flashnext import prefill, prefill_autotune as tune
from models.flashnext.tests.unit.test_prefix_session import Base, cache_bits


class SelectionTests(unittest.TestCase):
    def test_measured_shape_cost_and_headroom_choose_different_caps(self):
        sizes = (2048, 1024, 512, 256)
        cost = {2048: 9, 1024: 3, 512: 2, 256: 1.5}
        self.assertEqual(tune.choose(8192, 0, 8 * 2**30, 2048, sizes, seconds=cost).cap, 1024)
        self.assertEqual(tune.choose(8192, 0, 0, 2048, sizes, seconds=cost).cap, 256)
        self.assertFalse(tune.choose(8192, 0, 8 * 2**30, 2048, sizes, seconds={512: 1}).measured)

    def test_cached_context_counts_towards_chunk_cost(self):
        short = tune.choose(1024, 0, 8 * 2**30, 1024, (512, 256))
        long = tune.choose(1024, 8192, 8 * 2**30, 1024, (512, 256))
        self.assertGreater(long.estimated_bytes, short.estimated_bytes)

    def test_existing_prefix_keeps_its_shape_when_only_suffix_fits(self):
        with mock.patch.dict("os.environ", {tune.FLAG: "1", prefill.FLAG_CHUNK_TOKENS: "2048",
                                           tune.SHAPES_FLAG: ""}):
            def peek():
                self.assertEqual(prefill.chunk_tokens(), 2048)
                return 7936

            token = tune.start(8192, 0, available_bytes=2 * 2**30, prior_prefix=(2048, peek))
            try:
                self.assertEqual(tune.CURRENT.get().reusable_tokens, 7936)
                self.assertEqual(prefill.chunk_tokens(), 2048)
            finally:
                tune.CURRENT.reset(token)
            token = tune.start(8192, 0, available_bytes=2 * 2**30, prior_prefix=(2048, lambda: 0))
            try:
                self.assertLess(prefill.chunk_tokens(), 2048)
            finally:
                tune.CURRENT.reset(token)

    def test_context_reset_restores_configured_plan(self):
        with mock.patch.dict("os.environ", {prefill.FLAG_CHUNK_TOKENS: "1024", prefill.FLAG_CHUNK_POLICY: "grid"}):
            token = tune.CURRENT.set(tune.Choice(512, 0, 0, False))
            try:
                self.assertEqual((prefill.chunk_tokens(), prefill.chunk_policy()), (512, "balanced"))
            finally:
                tune.CURRENT.reset(token)
            self.assertEqual((prefill.chunk_tokens(), prefill.chunk_policy()), (1024, "grid"))


class PrefixTests(Base):
    def test_restore_is_exact_at_same_cap_and_isolated_across_caps(self):
        model, session = self.engine()
        prompt = self.system + self.suffix_a
        token = tune.CURRENT.set(tune.Choice(64, 0, 0, False))
        try:
            cold, cache_cold, report = self.go(model, session, prompt)
            self.assertEqual(report.hit_tokens, 0)
            warm, cache_warm, report = self.go(model, session, prompt)
            self.assertEqual(report.hit_tokens, len(self.system))
            self.assertEqual(report.prefilled_tokens, len(self.suffix_a))
            self.assertEqual(cold, warm)
            self.assertEqual(cache_bits(cache_cold), cache_bits(cache_warm))
        finally:
            tune.CURRENT.reset(token)
        token = tune.CURRENT.set(tune.Choice(128, 0, 0, False))
        try:
            _, _, report = self.go(model, session, prompt)
            self.assertEqual(report.hit_tokens, 0)
        finally:
            tune.CURRENT.reset(token)
        token = tune.CURRENT.set(tune.Choice(64, 0, 0, False))
        try:
            _, _, report = self.go(model, session, prompt)
            self.assertEqual(report.hit_tokens, len(self.system))
        finally:
            tune.CURRENT.reset(token)
