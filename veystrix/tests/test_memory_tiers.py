import unittest

from veystrix.memory_tiers import (
    FLASHNEXT_KERNEL_POOL_CAP_BYTES, GIB, TIERS, MemoryTier, current_tier,
    flashnext_pool_cap_bytes, full_resident_bytes, physical_bytes, tier_for,
)


class MemoryTierTests(unittest.TestCase):
    def test_supported_sizes_map_to_their_tier(self):
        expected = {
            16: "16 GB", 24: "24 GB", 32: "32 GB",
            64: "64 GB", 96: "96 GB", 128: "128 GB",
        }
        for gib, name in expected.items():
            with self.subTest(gib=gib):
                self.assertEqual(tier_for(gib * GIB).name, name)

    def test_baseline_is_preserved_and_caps_are_monotonic(self):
        baseline = tier_for(16 * GIB)
        self.assertEqual(baseline.pool_cap_gb, 2.0)
        self.assertEqual(baseline.nemotron_pool_cap_gb, 4.0)
        caps = [tier.pool_cap_gb for _, tier in reversed(TIERS)]
        self.assertEqual(caps, sorted(caps))
        self.assertTrue(all(tier.pool_cap_gb > 0 for _, tier in TIERS))

    def test_sizes_below_a_threshold_fall_back_to_the_smaller_tier(self):
        self.assertEqual(tier_for(19 * GIB).name, "16 GB")
        self.assertEqual(tier_for(27 * GIB).name, "24 GB")
        self.assertEqual(tier_for(47 * GIB).name, "32 GB")
        self.assertEqual(tier_for(79 * GIB).name, "64 GB")
        self.assertEqual(tier_for(111 * GIB).name, "96 GB")
        self.assertEqual(tier_for(8 * GIB).name, "16 GB")

    def test_environment_override(self):
        self.assertEqual(physical_bytes({"VEYSTRIX_PHYSICAL_MEMORY_GIB": "64"}), 64 * GIB)
        self.assertEqual(current_tier({"VEYSTRIX_PHYSICAL_MEMORY_GIB": "128"}).name, "128 GB")
        self.assertEqual(current_tier({"VEYSTRIX_PHYSICAL_MEMORY_GIB": "bad"}).name,
                         tier_for(physical_bytes({})).name)

    def test_flashnext_pool_is_capped_by_the_kernel_addressing(self):
        # 64/96 GB tier caps (6/8 GB) are below the ceiling; 128 GB (12 GB) is clamped to it.
        for gib in (64, 96):
            with self.subTest(gib=gib):
                tier = tier_for(gib * GIB)
                self.assertEqual(flashnext_pool_cap_bytes(tier), round(tier.pool_cap_gb * 1_000_000_000))
        self.assertEqual(flashnext_pool_cap_bytes(tier_for(128 * GIB)), FLASHNEXT_KERNEL_POOL_CAP_BYTES)

    def test_full_residency_is_fail_closed(self):
        dense, routed = 2 * GIB, 16 * GIB
        # 32 GB passes the OS-reserve rule; 16 GB does not.
        self.assertEqual(full_resident_bytes(routed_expert_bytes=routed, dense_bytes=dense,
                                             total_bytes=32 * GIB), routed)
        self.assertEqual(full_resident_bytes(routed_expert_bytes=routed, dense_bytes=dense,
                                             total_bytes=16 * GIB), 0)

    def test_tier_is_a_frozen_value(self):
        tier = MemoryTier("test", 1.0, 1.0)
        with self.assertRaises(Exception):
            tier.pool_cap_gb = 2.0


if __name__ == "__main__":
    unittest.main()
