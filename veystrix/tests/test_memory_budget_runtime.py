"""Dynamic runtime budget across all supported RAM tiers (16..128 GB).

No hardware needed: totals are passed explicitly. The 16 GB case asserts the
validated behavior (no pool, stream), and swap above the hard ceiling always
fails closed to streaming regardless of installed RAM.
"""
from __future__ import annotations

import unittest

from veystrix.memory_budget_runtime import plan
from veystrix.memory_tiers import GIB, swap_ceiling_bytes


class RuntimeBudgetTests(unittest.TestCase):
    BASE = dict(available=10 * GIB, swap_used=0, pressure=1,
                dense_bytes=2 * GIB, routed_expert_bytes=0,
                kv_bytes_per_token=0, session_count=1, context_tokens=0)

    def test_all_tiers_get_a_nonnegative_pool(self):
        for gib in (16, 24, 32, 48, 64, 96, 128):
            with self.subTest(gib=gib):
                b = plan(total=gib * GIB, **self.BASE)
                self.assertGreaterEqual(b.pool_bytes, 0)
                self.assertGreaterEqual(b.max_sessions, 1)
                self.assertIsInstance(b.resident, bool)

    def test_16gb_defaults_to_streaming(self):
        # Measured 16 GB behavior: the OS reserve is 4 GiB, and when available
        # memory is at/below it there is no headroom for an optional pool.
        b = plan(total=16 * GIB, **dict(self.BASE, available=4 * GIB))
        self.assertEqual(b.pool_bytes, 0)

    def test_hard_swap_ceiling_fails_closed(self):
        over = dict(self.BASE, swap_used=swap_ceiling_bytes() + GIB)
        b = plan(total=64 * GIB, **over)
        self.assertEqual(b.pool_bytes, 0)
        self.assertFalse(b.resident)
        self.assertEqual(b.max_sessions, 1)

    def test_warning_pressure_allows_higher_ram_residency_while_critical_fails_closed(self):
        # 64 GB Mac at warning pressure=2 with 40 GiB available still allocates dynamically.
        warn = plan(total=64 * GIB, available=40 * GIB, swap_used=0, pressure=2,
                    dense_bytes=2 * GIB, routed_expert_bytes=16 * GIB, record_bytes=5_612_544)
        self.assertTrue(warn.resident)
        self.assertGreater(warn.pool_bytes, 0)
        # Critical pressure=4 always fails closed across every RAM tier.
        crit = plan(total=64 * GIB, available=40 * GIB, swap_used=0, pressure=4,
                    dense_bytes=2 * GIB, routed_expert_bytes=16 * GIB, record_bytes=5_612_544)
        self.assertEqual((crit.pool_bytes, crit.nemotron_slots, crit.resident, crit.max_sessions),
                         (0, 0, False, 1))

    def test_resident_when_model_fits(self):
        base = dict(self.BASE, available=40 * GIB, routed_expert_bytes=16 * GIB)
        b = plan(total=64 * GIB, **base)
        self.assertTrue(b.resident)
        self.assertGreater(b.nemotron_slots, 0)
        small = plan(total=16 * GIB, **dict(base, available=10 * GIB))
        self.assertFalse(small.resident)

    def test_profile_updates_are_plain_types(self):
        b = plan(total=32 * GIB, **self.BASE)
        u = b.profile_updates()
        self.assertIsInstance(u["expert_pool_bytes"], int)
        self.assertIsInstance(u["resident"], bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)


QWEN = dict(dense_bytes=int(3.85 * GIB), routed_expert_bytes=int(71.78 * GIB), record_bytes=3_072_000)
NEMO = dict(dense_bytes=int(1.17 * GIB), routed_expert_bytes=int(15.39 * GIB), record_bytes=5_612_544)


class DynamicPoolTests(unittest.TestCase):
    """Pool follows free memory and the real model, across 16..128 GB (no hardware)."""

    def run_plan(self, total_gib, model, free_fraction=0.85, **extra):
        total = total_gib * GIB
        return plan(total=total, available=int(total * free_fraction), swap_used=0, pressure=1,
                    **model, **extra)

    def test_qwen_pool_grows_with_machine_size(self):
        pools = [self.run_plan(g, QWEN).pool_bytes for g in (16, 24, 32, 48, 64, 96, 128)]
        self.assertEqual(pools, sorted(pools))
        self.assertGreater(pools[-1], pools[0])
        self.assertLess(pools[0], 12 * GIB)            # 16 GB never gets a big pool

    def test_qwen_only_resident_when_everything_fits(self):
        self.assertFalse(self.run_plan(64, QWEN).resident)
        self.assertFalse(self.run_plan(96, QWEN).resident)
        big = self.run_plan(128, QWEN)
        self.assertTrue(big.resident)
        self.assertGreaterEqual(big.pool_bytes, QWEN["routed_expert_bytes"] - QWEN["record_bytes"])

    def test_nemotron_resident_from_32gb_not_16(self):
        self.assertFalse(self.run_plan(16, NEMO).resident)
        for gib in (32, 48, 64, 96, 128):
            with self.subTest(gib=gib):
                b = self.run_plan(gib, NEMO)
                self.assertTrue(b.resident)
                self.assertEqual(b.nemotron_slots, NEMO["routed_expert_bytes"] // NEMO["record_bytes"])

    def test_16gb_stays_at_the_validated_pool_cap(self):
        for model in (QWEN, NEMO):
            b = self.run_plan(16, model, free_fraction=0.9)
            self.assertLessEqual(b.pool_bytes, 2_000_000_000)
            self.assertEqual(b.nemotron_slots, 0)

    def test_pool_is_whole_records_and_never_exceeds_model(self):
        for gib in (16, 24, 32, 48, 64, 96, 128):
            for model in (QWEN, NEMO):
                b = self.run_plan(gib, model)
                self.assertEqual(b.pool_bytes % model["record_bytes"], 0)
                self.assertLessEqual(b.pool_bytes, model["routed_expert_bytes"])

    def test_less_free_memory_means_smaller_pool(self):
        roomy = self.run_plan(64, QWEN, free_fraction=0.9).pool_bytes
        tight = self.run_plan(64, QWEN, free_fraction=0.5).pool_bytes
        self.assertGreater(roomy, tight)

    def test_sessions_and_context_shrink_the_pool(self):
        one = self.run_plan(32, QWEN, kv_bytes_per_token=100_000, context_tokens=8192, session_count=1)
        four = self.run_plan(32, QWEN, kv_bytes_per_token=100_000, context_tokens=65536, session_count=4)
        self.assertGreaterEqual(one.pool_bytes, four.pool_bytes)

    def test_explicit_cap_and_unknown_model_fallback(self):
        capped = plan(total=128 * GIB, available=100 * GIB, swap_used=0, pressure=1, **QWEN,
                      environ={"VEYSTRIX_POOL_CAP_BYTES": str(4 * GIB)})
        self.assertLessEqual(capped.pool_bytes, 4 * GIB)
        unknown = plan(total=128 * GIB, available=100 * GIB, swap_used=0, pressure=1, environ={})
        self.assertLessEqual(unknown.pool_bytes, 12 * GIB)   # old tier cap, no model knowledge

    def test_16gb_with_real_headroom_state_streams_safely(self):
        for model in (QWEN, NEMO):
            b = plan(total=16 * GIB, available=int(3.5 * GIB), swap_used=int(0.8 * GIB), pressure=1, **model)
            self.assertEqual((b.pool_bytes, b.nemotron_slots, b.resident), (0, 0, False))


class ProbeTests(unittest.TestCase):
    def test_probe_classifies_tensors(self):
        import json, struct, tempfile
        from pathlib import Path
        from veystrix.model_memory import probe_checkpoint
        header = {"__metadata__": {}}
        off = 0
        def add(name, shape, size):
            nonlocal off
            header[name] = {"dtype": "U32", "shape": shape, "data_offsets": [off, off + size]}
            off += size
        for layer in (0, 1):
            add(f"m.layers.{layer}.mlp.switch_mlp.up.weight", [4, 8], 4000)
            add(f"m.layers.{layer}.mlp.shared_expert.up.weight", [8, 8], 100)
            add(f"m.layers.{layer}.attn.q.weight", [8, 8], 50)
        add("m.layers.0.ple.ngram_embedding.shard_1.weight", [100], 700)
        raw = json.dumps(header).encode()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * off)
            m = probe_checkpoint(tmp)
        self.assertEqual((m.routed_expert_bytes, m.moe_layers, m.experts_per_layer, m.record_bytes),
                         (8000, 2, 4, 1000))
        self.assertEqual(m.streamed_table_bytes, 700)
        self.assertEqual(m.resident_dense_bytes, m.total_bytes - 8000 - 700)


class NemotronStreamingPoolDefault(unittest.TestCase):
    def test_streaming_nemotron_skips_the_slot_pool_and_qwen_is_untouched(self):
        import json, os, tempfile
        from pathlib import Path
        from unittest.mock import patch
        from veystrix.backends import cpp_engine
        from veystrix.memory_budget_runtime import RuntimeBudget
        budget = RuntimeBudget(16 * GIB, 8 * GIB, 0, 1, 0, 0, False, 2)
        for model_type, expected in (("nemotron_h", "0"), ("qwen4_exp", None)):
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "config.json").write_text(json.dumps({"model_type": model_type}))
                with patch.dict(os.environ, {}, clear=False), \
                     patch("veystrix.memory_budget_runtime.plan", return_value=budget), \
                     patch.object(cpp_engine, "_runtime_budget", None):
                    for key in ("VEYSTRIX_CPP_NEMO_POOL", "VEYSTRIX_CPP_NEMO_POOL_SLOTS", "VEYSTRIX_CPP_RESIDENT",
                                "VEYSTRIX_CPP_MAX_SESSIONS", "VEYSTRIX_CPP_POOL_BYTES"):
                        os.environ.pop(key, None)
                    cpp_engine.apply_runtime_memory_budget(tmp)
                    self.assertEqual(os.environ.get("VEYSTRIX_CPP_NEMO_POOL"), expected, model_type)
                    self.assertIsNone(os.environ.get("VEYSTRIX_CPP_NEMO_POOL_SLOTS"), model_type)
                    for key in ("VEYSTRIX_CPP_NEMO_POOL", "VEYSTRIX_CPP_NEMO_POOL_SLOTS", "VEYSTRIX_CPP_RESIDENT",
                                "VEYSTRIX_CPP_MAX_SESSIONS"):
                        os.environ.pop(key, None)
