import unittest

from models.flashnext import memory_budget as budget


class MemoryBudgetTests(unittest.TestCase):
    def plan(self, **overrides):
        inputs = dict(available=6 * budget.GIB, pressure=1, resident=3 * budget.GIB,
                      state=100, prefix=200, pending_tokens=100, prefill_scratch=0,
                      verification_width=0, existing_pool=0, cap=6 * budget.GIB,
                      previous=None, last_change=0, now=200)
        return budget.plan(**(inputs | overrides))

    def test_six_gb_setting_is_a_cap_and_zero_is_valid(self):
        self.assertEqual(self.plan().expert_pool_bytes, 2 * budget.GIB)
        self.assertEqual(self.plan(cap=0).expert_pool_bytes, 0)
        self.assertEqual(self.plan(cap=600_000_000).expert_pool_bytes, budget.GIB // 2)

    def test_pressure_and_capacity_shrink_without_refusal(self):
        self.assertEqual(self.plan(pressure=2, previous=2 * budget.GIB).expert_pool_bytes, 0)
        tight = self.plan(available=1, previous=2 * budget.GIB)
        self.assertEqual(tight.expert_pool_bytes, 0)
        self.assertLess(tight.projected_available_bytes, 0)

    def test_growth_has_cooldown_and_margin(self):
        self.assertEqual(self.plan(previous=0, now=50).expert_pool_bytes, 0)
        self.assertEqual(self.plan(previous=0, now=200).expert_pool_bytes, 2 * budget.GIB)
        self.assertEqual(self.plan(available=2 * budget.GIB, previous=0).expert_pool_bytes, 0)

    def test_speculative_reservation_and_legacy_switch(self):
        row = self.plan(verification_width=8)
        self.assertEqual(row.speculative_workspace_bytes, 640 * 2**20)
        self.assertEqual(row.pending_state_bytes, 100 * 64 * 1024)
        self.assertEqual(row.resident_model_bytes, 3 * budget.GIB)
        self.assertTrue(budget.enabled({}, 16 * budget.GIB))
        self.assertFalse(budget.enabled({budget.POLICY: "legacy"}, 16 * budget.GIB))
