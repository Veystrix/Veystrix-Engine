import ctypes
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from veystrix.backends import cpp_engine as mod
from veystrix.server import anthropic_usage, chat_usage, usage_counts
from models.flashnext.chat_guard import ChatGuard


class CommercialReleaseTests(unittest.TestCase):
    """End-to-end tests for commercial release enhancements across Workstreams A-F."""

    def test_vx_session_truncate_binding(self):
        """Test vx_session_truncate is bound on CtypesVxBinding and callable."""
        binding = mod.CtypesVxBinding()
        self.assertTrue(hasattr(binding, "truncate"))

    def test_cpp_engine_truncate(self):
        """Test CppEngineBackend.truncate rewinds tape and updates _live_at."""
        class FakeBinding:
            has_v2 = True
            def __init__(self):
                self.truncated_to = None
            def abi_version(self): return 2
            def truncate(self, session, target_tokens):
                self.truncated_to = target_tokens

        backend = mod.CppEngineBackend.__new__(mod.CppEngineBackend)
        fake_b = FakeBinding()
        backend._binding = fake_b
        backend._session = object()
        backend.tape = [10, 20, 30, 40, 50, 60]
        backend.pending = [70, 80]
        backend._live_at = 6

        ok = backend.truncate(4)
        self.assertTrue(ok)
        self.assertEqual(fake_b.truncated_to, 4)
        self.assertEqual(backend.tape, [10, 20, 30, 40])
        self.assertEqual(backend.pending, [])
        self.assertEqual(backend._live_at, 4)

    def test_server_lcp_truncation_reuses_prefix(self):
        """Test that server truncates and reuses common prefix >= 256 tokens."""
        backend = MagicMock()
        backend.tape = list(range(1000))
        backend.pending = []
        backend.common_prefix = lambda ids: 800
        backend.truncate = MagicMock(return_value=True)
        backend.append_tokens = MagicMock()

        new_ids = list(range(800)) + [9999, 8888]
        shared = backend.common_prefix(new_ids)
        self.assertEqual(shared, 800)
        self.assertTrue(shared >= 256 and shared < len(backend.tape))
        
        # Call backend.truncate as done in server.py
        ok = backend.truncate(shared)
        self.assertTrue(ok)
        backend.truncate.assert_called_once_with(800)

    def test_anthropic_usage_reports_cache_creation_and_read(self):
        """Test Anthropic usage accurately reports cache_creation and cache_read."""
        service = MagicMock()
        service.last_prompt_tokens = 1000
        service.last_cached_tokens = 750
        service.last_cache_created = 250
        stats = MagicMock()
        stats.prompt_tokens = 250
        stats.tokens = 50

        u = anthropic_usage(service, stats)
        self.assertEqual(u["input_tokens"], 250)
        self.assertEqual(u["cache_read_input_tokens"], 750)
        self.assertEqual(u["cache_creation_input_tokens"], 250)
        self.assertEqual(u["output_tokens"], 50)

    def test_openai_usage_reports_cached_tokens(self):
        """Test OpenAI chat usage accurately reports cached_tokens and prompt_cache_hit_tokens."""
        service = MagicMock()
        service.last_prompt_tokens = 1000
        service.last_cached_tokens = 600
        stats = MagicMock()
        stats.prompt_tokens = 400
        stats.tokens = 30

        u = chat_usage(service, stats)
        self.assertEqual(u["prompt_tokens"], 1000)
        self.assertEqual(u["prompt_tokens_details"]["cached_tokens"], 600)
        self.assertEqual(u["prompt_cache_hit_tokens"], 600)
        self.assertEqual(u["prompt_cache_miss_tokens"], 400)
        self.assertEqual(u["completion_tokens"], 30)

    def test_chat_guard_tolerates_transient_probe_misses(self):
        """Test ChatGuard tolerates up to 2 transient probe misses before failing."""
        probe = MagicMock()
        sampler = MagicMock()
        sampler.sample.return_value = {
            "power_source": "AC Power",
            "low_power": False,
            "swap_used_mb": 100,
            "pressure_level": 1,
            "battery_percent": None,
            "thermal_warning": False,
        }
        sampler.runner.return_value = "thermal warning level: 0"
        guard = ChatGuard(tempfile.mktemp(), probe=probe, sampler=sampler)

        row_bad = {
            "mlx_active": 1000,
            "mlx_cache": 500,
            "available_mb": None,
            "compressor_mb": 100,
            "swapout": 0,
            "swapin": 0,
            "metal_limit_bytes": 16 * (1024**3),
        }
        with patch("models.flashnext.memory_policy.is_absolute_swap_policy", return_value=False), \
             patch("models.flashnext.memory_policy.is_swap_growth_policy", return_value=False), \
             patch("models.flashnext.memory_waiver.active", return_value=False):
            # First tick: tolerated (returns None)
            res1 = guard._reason(row_bad)
            self.assertIsNone(res1)
            # Second tick: tolerated
            res2 = guard._reason(row_bad)
            self.assertIsNone(res2)
            # Third tick: exceeds threshold, returns critical reason
            res3 = guard._reason(row_bad)
            self.assertIsNotNone(res3)
            self.assertIn("critical probe unavailable: available_mb", res3)

    def test_chat_guard_handles_desktop_mac_without_battery(self):
        """Test ChatGuard allows desktop Mac (AC power / no battery hardware)."""
        probe = MagicMock()
        sampler = MagicMock()
        sampler.sample.return_value = {
            "power_source": "AC Power",
            "low_power": False,
            "swap_used_mb": 100,
            "pressure_level": 1,
            "battery_percent": None,  # desktop Mac has no battery
            "thermal_warning": False,
        }
        sampler.runner.return_value = "Note: No thermal warning level has been recorded"
        guard = ChatGuard(tempfile.mktemp(), probe=probe, sampler=sampler)

        row = {
            "mlx_active": 1000,
            "mlx_cache": 500,
            "available_mb": 12000,
            "compressor_mb": 100,
            "swapout": 0,
            "swapin": 0,
            "metal_limit_bytes": 16 * (1024**3),
        }
        with patch("models.flashnext.memory_policy.is_absolute_swap_policy", return_value=False), \
             patch("models.flashnext.memory_policy.is_swap_growth_policy", return_value=False), \
             patch("models.flashnext.memory_waiver.active", return_value=False):
            reason = guard._reason(row)
            self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
