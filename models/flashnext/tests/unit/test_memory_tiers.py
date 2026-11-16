"""The elastic planner and pool caps must follow physical memory size."""
from __future__ import annotations

import unittest

from models.flashnext import memory_budget
from models.flashnext.settings import modes
from veystrix.memory_tiers import GIB


class HigherMemoryPoolTests(unittest.TestCase):
    def test_planner_is_available_on_every_supported_size(self):
        for gib in (8, 16, 24, 32, 64, 96, 128):
            with self.subTest(gib=gib):
                self.assertTrue(memory_budget.enabled({}, gib * GIB))

    def test_policy_overrides_still_win(self):
        self.assertFalse(memory_budget.enabled({"VEYSTRIX_MEMORY_POLICY": "legacy"}, 128 * GIB))
        self.assertTrue(memory_budget.enabled({"VEYSTRIX_MEMORY_POLICY": "elastic"}, 128 * GIB))
        self.assertFalse(memory_budget.enabled({"FLASHNEXT_MEMORY_MANAGER": "1"}, 128 * GIB))

    def test_fixed_pool_cap_follows_the_tier(self):
        expected = {16: 2.0, 24: 3.0, 32: 4.0, 64: 6.0, 96: 8.0, 128: 12.0}
        for gib, cap in expected.items():
            with self.subTest(gib=gib):
                gb, _ = modes.choose_pool_gb("auto", physical_bytes=gib * GIB)
                self.assertLessEqual(gb, cap)
        # The 16 GB baseline still refuses a 6 GB pool.
        with self.assertRaises(modes.ModeError):
            modes.choose_pool_gb("6", physical_bytes=16 * GIB)

    def test_larger_machines_accept_larger_pools(self):
        gb, _ = modes.choose_pool_gb("8", physical_bytes=128 * GIB)
        self.assertEqual(gb, 8.0)
        gb, _ = modes.choose_pool_gb("12", physical_bytes=128 * GIB)
        self.assertEqual(gb, 12.0)
        with self.assertRaises(modes.ModeError):
            modes.choose_pool_gb("13", physical_bytes=128 * GIB)

    def test_start_gate_is_unchanged(self):
        gb, reason = modes.choose_pool_gb("auto", available_gb=8.0, physical_bytes=128 * GIB)
        self.assertEqual(gb, 0.0)
        self.assertIn("start gate", reason)


if __name__ == "__main__":
    unittest.main()
