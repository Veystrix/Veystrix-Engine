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
        # 8 GiB is the kernel-addressable ceiling; beyond that is refused.
        from veystrix.memory_tiers import FLASHNEXT_KERNEL_POOL_CAP_BYTES
        ceiling = FLASHNEXT_KERNEL_POOL_CAP_BYTES / 1_000_000_000
        gb, _ = modes.choose_pool_gb(str(ceiling), physical_bytes=128 * GIB)
        self.assertLessEqual(gb * 1_000_000_000, FLASHNEXT_KERNEL_POOL_CAP_BYTES)
        with self.assertRaises(modes.ModeError):
            modes.choose_pool_gb(str(ceiling + 1), physical_bytes=128 * GIB)

    def test_planner_steps_are_unchanged_on_the_baseline(self):
        self.assertEqual(memory_budget.pool_steps(6 * GIB, 16 * GIB), memory_budget.POOL_STEPS)

    def test_planner_can_reach_a_larger_pool_on_a_larger_machine(self):
        from veystrix.memory_tiers import FLASHNEXT_KERNEL_POOL_CAP_BYTES
        cap = FLASHNEXT_KERNEL_POOL_CAP_BYTES
        steps = memory_budget.pool_steps(cap, 128 * GIB)
        self.assertEqual(steps[0], cap)
        self.assertEqual(steps[1:], memory_budget.POOL_STEPS)
        row = memory_budget.plan(
            available=20 * GIB, pressure=1, resident=3 * GIB, state=0, prefix=0,
            pending_tokens=0, prefill_scratch=0, verification_width=0, existing_pool=0,
            cap=cap, previous=None, last_change=0, now=200, physical_bytes=128 * GIB)
        self.assertEqual(row.expert_pool_bytes, cap)

    def test_nemotron_auto_cap_follows_the_tier(self):
        from models.nemotron35 import admission
        from veystrix.memory_tiers import GIB as TIER_GIB

        self.assertEqual(admission.auto_pool_max(16 * TIER_GIB), admission.AUTO_POOL_MAX)
        self.assertGreater(admission.auto_pool_max(128 * TIER_GIB), admission.AUTO_POOL_MAX)

    def test_start_gate_is_unchanged(self):
        gb, reason = modes.choose_pool_gb("auto", available_gb=8.0, physical_bytes=128 * GIB)
        self.assertEqual(gb, 0.0)
        self.assertIn("start gate", reason)




class DeviceMetalLimitTests(unittest.TestCase):
    def test_fallback_when_no_device_is_present(self):
        from models.flashnext import prefill_budget as pb
        from unittest import mock
        with mock.patch.dict("sys.modules", {"mlx.core": None}):
            self.assertEqual(pb.device_metal_limit(fallback=123), 123)
        self.assertGreater(pb.device_metal_limit(), 0)

    def test_fit_accepts_a_device_limit(self):
        from models.flashnext import prefill_budget as pb
        tiny = pb.Budget(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.assertTrue(tiny.fits(False, metal_limit=pb.METAL_LIMIT))
        self.assertFalse(tiny.fits(False, metal_limit=1))

class DeviceMetalLimitTests(unittest.TestCase):
    def test_fallback_when_no_device_is_present(self):
        from models.flashnext import prefill_budget as pb
        from unittest import mock
        with mock.patch.dict("sys.modules", {"mlx.core": None}):
            self.assertEqual(pb.device_metal_limit(fallback=123), 123)
        self.assertGreater(pb.device_metal_limit(), 0)

    def test_fit_accepts_a_device_limit(self):
        from models.flashnext import prefill_budget as pb
        tiny = pb.Budget(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.assertTrue(tiny.fits(False, metal_limit=pb.METAL_LIMIT))
        self.assertFalse(tiny.fits(False, metal_limit=1))


if __name__ == "__main__":
    unittest.main()
