import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx
import numpy as np

from models.flashnext import expert_cache, expert_io_stats, expert_pool
from models.flashnext.store import SafeTensorStore
from models.flashnext.tests.unit.test_prefill_ring import (
    EXPERTS,
    HIDDEN,
    LAYERS,
    PROJECTIONS,
    SHAPES,
    build_checkpoint,
    prefix_of,
    silu_glu,
)


class ShortPrefillPoolReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        build_checkpoint(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        expert_io_stats.reset_stats()
        self.store = SafeTensorStore(self.directory.name)
        self.addCleanup(self.store.close)
        self.glu1 = expert_cache.StreamingSwitchGLU(
            self.store, prefix_of(1), 32, 4, "affine", silu_glu, layer_id=1
        )
        self.glu2 = expert_cache.StreamingSwitchGLU(
            self.store, prefix_of(2), 32, 4, "affine", silu_glu, layer_id=2
        )
        # Create a resident decode pool of 64 MB
        self.pool = expert_pool.ExpertPool(self.store, 64 * 1024 * 1024, 32)
        self.store._flashnext_expert_pool = self.pool
        self.addCleanup(self.pool.close)

    def test_full_pool_hits_produce_bit_identical_output_with_zero_ssd_reads(self):
        # Assign experts 2, 4, 6 of layer 1 to pool
        slots, misses = self.pool.assign(1, [2, 4, 6])
        futures = self.pool.fill(prefix_of(1), misses, expert_cache._submit_read, 2)
        for f in futures:
            f.result()

        # Inputs with 16 tokens routing to experts 2, 4, 6
        mx.random.seed(42)
        x = mx.random.normal((1, 16, HIDDEN)).astype(mx.bfloat16)
        # Routes using only the populated experts [2, 4, 6]
        indices = mx.array([[[2, 4], [4, 6]] * 8], dtype=mx.uint32)

        # Baseline run (reuse OFF)
        with mock.patch.dict(os.environ, {expert_io_stats.FLAG_POOL_REUSE: "0"}):
            y_baseline = self.glu1(x, indices)
            mx.eval(y_baseline)

        # Opt-in run (reuse ON)
        with mock.patch.dict(os.environ, {expert_io_stats.FLAG_POOL_REUSE: "1",
                                          expert_io_stats.FLAG_DIAGNOSTICS: "1"}):
            y_optimized = self.glu1(x, indices)
            mx.eval(y_optimized)

        # 1. Assert bit-identical numerical output
        self.assertTrue(mx.array_equal(y_baseline.view(mx.uint16), y_optimized.view(mx.uint16)))

        # 2. Assert stats show 100% pool hits and 0 SSD reads
        stats = expert_io_stats.get_stats()
        self.assertEqual(stats.pool_hits, 3)    # 3 unique experts: 2, 4, 6
        self.assertEqual(stats.pool_misses, 0)
        self.assertEqual(stats.ssd_reads, 0)
        self.assertGreater(stats.pool_bytes_saved, 0)

    def test_aligned_pack_fills_the_metal_pool_with_exact_bytes(self):
        from models.nemotron35.safetensors_index import index_checkpoint
        from models.shared.expert_pack import ALIGNMENT, build

        root = Path(self.directory.name)
        config = root / "config.json"
        config.write_text("{}")
        self.addCleanup(config.unlink)
        with tempfile.TemporaryDirectory() as folder:
            pack = Path(folder) / "experts.pack"
            size = build(root, index_checkpoint(root), "flashnext", pack, layers=[1])
            self.assertEqual(size % ALIGNMENT, 0)
            with mock.patch.dict(os.environ, {"FLASHNEXT_EXPERT_PACK": str(pack),
                                              expert_pool.NATIVE_PACK_FLAG: "0"}):
                candidate = expert_pool.ExpertPool(self.store, 64 * 1024 * 1024, 32)
            self.addCleanup(candidate.close)
            for layer in (1, 2):
                baseline_slots, misses = self.pool.assign(layer, [1, 3])
                for future in self.pool.fill(prefix_of(layer), misses, expert_cache._submit_read):
                    future.result()
                candidate_slots, misses = candidate.assign(layer, [1, 3])
                futures = candidate.fill(prefix_of(layer), misses, expert_cache._submit_read)
                if layer == 1:
                    self.assertEqual(len(futures), 2)
                for future in futures:
                    future.result()
                for expert in (1, 3):
                    a = expert_pool.HEADER + baseline_slots[expert] * self.pool.stride
                    b = expert_pool.HEADER + candidate_slots[expert] * candidate.stride
                    np.testing.assert_array_equal(self.pool.array[a:a + self.pool.stride],
                                                  candidate.array[b:b + candidate.stride])
            candidate.native_pack = True
            slots, misses = candidate.assign(1, [5, 7])
            baseline_slots, baseline_misses = self.pool.assign(1, [5, 7])
            for future in self.pool.fill(prefix_of(1), baseline_misses, expert_cache._submit_read):
                future.result()
            pending = candidate.fill(prefix_of(1), misses, expert_cache._submit_read)
            self.assertEqual(len(pending), 1)
            pending[0].result()
            for expert in (5, 7):
                a = expert_pool.HEADER + baseline_slots[expert] * self.pool.stride
                b = expert_pool.HEADER + slots[expert] * candidate.stride
                np.testing.assert_array_equal(self.pool.array[a:a + self.pool.stride],
                                              candidate.array[b:b + candidate.stride])

    def test_content_addressed_coalesced_and_mapped_records_match_shards(self):
        import hashlib
        from models.nemotron35.safetensors_index import index_checkpoint
        from models.shared.expert_pack import build

        root = Path(self.directory.name)
        (root / "config.json").write_text("{}")
        with tempfile.TemporaryDirectory() as folder:
            pack, size = build(root, index_checkpoint(root), "flashnext",
                               Path(folder), layers=[1], content_addressed=True)
            self.assertEqual(pack.stem, hashlib.sha256(pack.read_bytes()).hexdigest())
            self.assertEqual(size, pack.stat().st_size)
            baseline_slots, misses = self.pool.assign(1, [2, 3, 5])
            for future in self.pool.fill(prefix_of(1), misses, expert_cache._submit_read):
                future.result()
            for mode in ("coalesced", "mmap"):
                with self.subTest(mode=mode), mock.patch.dict(os.environ, {
                    "FLASHNEXT_EXPERT_PACK": str(pack), "FLASHNEXT_EXPERT_PACK_READ": mode,
                    expert_pool.NATIVE_PACK_FLAG: "0",
                }):
                    candidate = expert_pool.ExpertPool(self.store, 64 * 2**20, 32)
                    try:
                        slots, misses = candidate.assign(1, [5, 3, 2])
                        for future in candidate.fill(prefix_of(1), misses, expert_cache._submit_read):
                            future.result()
                        self.assertEqual(candidate.pack_extents, 2)
                        self.assertEqual(candidate.packed_reads, 3)
                        for expert in (2, 3, 5):
                            a = expert_pool.HEADER + baseline_slots[expert] * self.pool.stride
                            b = expert_pool.HEADER + slots[expert] * candidate.stride
                            np.testing.assert_array_equal(self.pool.array[a:a + self.pool.stride],
                                                          candidate.array[b:b + candidate.stride])
                    finally:
                        candidate.close()

    def test_partial_pool_residency_reads_only_misses_from_ssd(self):
        # Assign only expert 2 of layer 1 to pool (4 and 6 remain misses)
        slots, misses = self.pool.assign(1, [2])
        futures = self.pool.fill(prefix_of(1), misses, expert_cache._submit_read, 2)
        for f in futures:
            f.result()

        mx.random.seed(42)
        x = mx.random.normal((1, 16, HIDDEN)).astype(mx.bfloat16)
        indices = mx.array([[[2, 4], [4, 6]] * 8], dtype=mx.uint32)

        # Baseline run
        with mock.patch.dict(os.environ, {expert_io_stats.FLAG_POOL_REUSE: "0"}):
            y_baseline = self.glu1(x, indices)
            mx.eval(y_baseline)

        # Opt-in run
        with mock.patch.dict(os.environ, {expert_io_stats.FLAG_POOL_REUSE: "1",
                                          expert_io_stats.FLAG_DIAGNOSTICS: "1"}):
            y_optimized = self.glu1(x, indices)
            mx.eval(y_optimized)

        # Output bit-identical
        self.assertTrue(mx.array_equal(y_baseline.view(mx.uint16), y_optimized.view(mx.uint16)))

        # Stats show 1 hit and 2 misses
        stats = expert_io_stats.get_stats()
        self.assertEqual(stats.pool_hits, 1)    # expert 2
        self.assertEqual(stats.pool_misses, 2)  # experts 4 and 6
        self.assertEqual(stats.ssd_reads, 1)

    def test_different_layer_with_same_expert_id_is_not_falsely_hit(self):
        # Assign expert 2 of layer 1
        slots, misses = self.pool.assign(1, [2])
        futures = self.pool.fill(prefix_of(1), misses, expert_cache._submit_read, 2)
        for f in futures:
            f.result()

        # Query layer 2 with expert 2
        mx.random.seed(42)
        x = mx.random.normal((1, 16, HIDDEN)).astype(mx.bfloat16)
        indices = mx.array([[[2, 2]] * 16], dtype=mx.uint32)

        with mock.patch.dict(os.environ, {expert_io_stats.FLAG_POOL_REUSE: "1",
                                          expert_io_stats.FLAG_DIAGNOSTICS: "1"}):
            y = self.glu2(x, indices)
            mx.eval(y)

        # Must be a miss on layer 2 despite being in pool for layer 1!
        stats = expert_io_stats.get_stats()
        self.assertEqual(stats.pool_hits, 0)
        self.assertEqual(stats.pool_misses, 1)

    def test_pool_eviction_removes_from_pool_and_reads_from_ssd(self):
        # Pool with minimum 16 slots
        tiny_pool = expert_pool.ExpertPool(self.store, 4096 + 16 * self.pool.stride, 32)
        self.store._flashnext_expert_pool = tiny_pool
        self.addCleanup(tiny_pool.close)

        # Fill all 16 slots: 8 experts on layer 1 (0..7), 8 on layer 2 (0..7)
        slots1, m1 = tiny_pool.assign(1, list(range(8)))
        futures1 = tiny_pool.fill(prefix_of(1), m1, expert_cache._submit_read, 2)
        for f in futures1:
            f.result()
        slots2, m2 = tiny_pool.assign(2, list(range(8)))
        futures2 = tiny_pool.fill(prefix_of(2), m2, expert_cache._submit_read, 2)
        for f in futures2:
            f.result()

        # Now allocate expert 8 on layer 1, which must evict (1, 0) (oldest)
        slots3, m3 = tiny_pool.assign(1, [8])
        futures3 = tiny_pool.fill(prefix_of(1), m3, expert_cache._submit_read, 2)
        for f in futures3:
            f.result()

        # Expert 0 of layer 1 is evicted from pool
        self.assertNotIn((1, 0), tiny_pool._slot_of)
        self.assertIn((1, 1), tiny_pool._slot_of)
        self.assertIn((1, 8), tiny_pool._slot_of)

        # Now prefill with expert 0
        mx.random.seed(42)
        x = mx.random.normal((1, 16, HIDDEN)).astype(mx.bfloat16)
        indices = mx.array([[[0, 0]] * 16], dtype=mx.uint32)

        with mock.patch.dict(os.environ, {expert_io_stats.FLAG_POOL_REUSE: "1",
                                          expert_io_stats.FLAG_DIAGNOSTICS: "1"}):
            y = self.glu1(x, indices)
            mx.eval(y)

        # Expert 0 must be counted as a miss and fetched from SSD
        stats = expert_io_stats.get_stats()
        self.assertEqual(stats.pool_hits, 0)
        self.assertEqual(stats.pool_misses, 1)
        self.assertEqual(stats.ssd_reads, 1)

    def test_multiple_sequential_requests_preserve_state_and_lru(self):
        # Assign experts 1, 2, 3 to pool
        slots, misses = self.pool.assign(1, [1, 2, 3])
        futures = self.pool.fill(prefix_of(1), misses, expert_cache._submit_read, 2)
        for f in futures:
            f.result()

        mx.random.seed(99)
        x = mx.random.normal((1, 16, HIDDEN)).astype(mx.bfloat16)

        # Turn 1: query expert 1
        with mock.patch.dict(os.environ, {expert_io_stats.FLAG_POOL_REUSE: "1",
                                          expert_io_stats.FLAG_DIAGNOSTICS: "1"}):
            y1 = self.glu1(x, mx.array([[[1, 1]] * 16], dtype=mx.uint32))
            mx.eval(y1)
            # Turn 2: query expert 2 and 3
            y2 = self.glu1(x, mx.array([[[2, 3]] * 16], dtype=mx.uint32))
            mx.eval(y2)

        stats = expert_io_stats.get_stats()
        self.assertEqual(stats.pool_hits, 3)    # 1 on turn 1, 2 on turn 2
        self.assertEqual(stats.pool_misses, 0)
        self.assertEqual(stats.ssd_reads, 0)

    def test_pool_release_safely_falls_back_to_pure_ssd_reads(self):
        # Pool is released / set to None
        self.store._flashnext_expert_pool = None

        mx.random.seed(123)
        x = mx.random.normal((1, 16, HIDDEN)).astype(mx.bfloat16)
        indices = mx.array([[[1, 2]] * 16], dtype=mx.uint32)

        # Baseline run
        with mock.patch.dict(os.environ, {expert_io_stats.FLAG_POOL_REUSE: "0"}):
            y_baseline = self.glu1(x, indices)
            mx.eval(y_baseline)

        # Opt-in run with pool=None
        with mock.patch.dict(os.environ, {expert_io_stats.FLAG_POOL_REUSE: "1",
                                          expert_io_stats.FLAG_DIAGNOSTICS: "1"}):
            y_optimized = self.glu1(x, indices)
            mx.eval(y_optimized)

        self.assertTrue(mx.array_equal(y_baseline.view(mx.uint16), y_optimized.view(mx.uint16)))
        stats = expert_io_stats.get_stats()
        self.assertEqual(stats.pool_hits, 0)
        self.assertEqual(stats.pool_misses, 2)
        self.assertEqual(stats.ssd_reads, 1)

    def test_pool_resizing_during_workload_handles_reallocated_pool(self):
        # Initial pool: populate expert 1 on layer 1
        self.pool.assign(1, [1])
        futures = self.pool.fill(prefix_of(1), [(1, self.pool._slot_of[(1, 1)])], expert_cache._submit_read, 2)
        for f in futures:
            f.result()

        # Simulate pool resize: replace with new pool with different slots
        new_pool = expert_pool.ExpertPool(self.store, 4096 + 20 * self.pool.stride, 32)
        self.store._flashnext_expert_pool = new_pool
        self.addCleanup(new_pool.close)

        # Populate expert 2 in new pool
        new_pool.assign(1, [2])
        futures = new_pool.fill(prefix_of(1), [(2, new_pool._slot_of[(1, 2)])], expert_cache._submit_read, 2)
        for f in futures:
            f.result()

        # Expert 1 is NOT in new_pool, expert 2 IS in new_pool
        mx.random.seed(77)
        x = mx.random.normal((1, 16, HIDDEN)).astype(mx.bfloat16)
        indices = mx.array([[[1, 2]] * 16], dtype=mx.uint32)

        with mock.patch.dict(os.environ, {expert_io_stats.FLAG_POOL_REUSE: "1",
                                          expert_io_stats.FLAG_DIAGNOSTICS: "1"}):
            y = self.glu1(x, indices)
            mx.eval(y)

        stats = expert_io_stats.get_stats()
        self.assertEqual(stats.pool_hits, 1)   # expert 2
        self.assertEqual(stats.pool_misses, 1) # expert 1

    def test_quantized_weight_bytes_exact_equality(self):
        # Assign expert 3 of layer 1 to pool
        slots, misses = self.pool.assign(1, [3])
        futures = self.pool.fill(prefix_of(1), misses, expert_cache._submit_read, 2)
        for f in futures:
            f.result()

        slot = self.pool._slot_of[(1, 3)]
        # For each projection and part, verify that pool bytes match store.rows_np bit-for-bit
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for part in ("weight", "scales", "biases"):
                name = f"{prefix_of(1)}.{proj}.{part}"
                ref = self.store.refs[name]
                expected_np = self.store.rows_np(name, [3])[0]
                expected_bytes = expected_np.view(np.uint8).tobytes()

                offset_in_slot = self.pool.offsets[(proj, part)]
                base = 4096 + slot * self.pool.stride + offset_in_slot
                actual_bytes = self.pool.array[base : base + ref.row_bytes].tobytes()

                self.assertEqual(actual_bytes, expected_bytes, f"Byte mismatch in {name} expert 3!")


if __name__ == "__main__":
    unittest.main()
