"""Expert pool: LRU slot assignment and byte-exact record fills."""
from __future__ import annotations

import json
import os
import random
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
import numpy as np

from models.flashnext import expert_cache, expert_pool
from models.shared.cost_aware import CostAwarePolicy
from models.flashnext.slab_pack import get_slab_layout
from models.flashnext.store import SafeTensorStore

PREFIX = "language_model.model.layers.4.mlp.switch_mlp"
ROWS = {"weight": 819_200, "scales": 102_400, "biases": 102_400}


class _Store:
    dir = "/nonexistent"
    refs = {}


class AssignTests(unittest.TestCase):
    def test_flat_slot_table_survives_eviction_ring_lease_and_invalidation(self):
        from models.flashnext import pool_ring

        with mock.patch.dict(os.environ, {expert_pool.FLAT_SLOTS_FLAG: "1"}):
            pool = self.make(40)
        self.addCleanup(pool.close)
        old, _ = pool.assign(1, range(40))
        first = pool_ring.lease_range(pool, 20)
        self.assertTrue(all(pool._flat_slots[1, expert] == -1 for expert, slot in old.items() if slot >= first))
        active, misses = pool.assign(1, [0, 40, 41])
        self.assertEqual(active[0], old[0])
        self.assertEqual(len(misses), 2)
        pool.invalidate(1, misses)
        self.assertEqual(len(pool.assign(1, [40, 41])[1]), 2)
        pool_ring.return_range(pool, first, 20)
        self.assertEqual({key: int(pool._flat_slots[key]) for key in pool._slot_of}, pool._slot_of)

    def test_least_stale_preserves_next_layer_and_recovers_failed_fills(self):
        with mock.patch.dict(os.environ, {expert_pool.POLICY_FLAG: "least-stale"}):
            pool = self.make(16)
        self.addCleanup(pool.close)
        first, _ = pool.assign(1, range(8))
        second, _ = pool.assign(2, range(8))
        slots, misses = pool.assign(1, [0, 8, 9])
        self.assertEqual(slots[0], first[0])
        self.assertEqual({slot for _, slot in misses}, {first[1], first[2]})
        pool.invalidate(1, misses)
        self.assertEqual(pool.assign(2, range(8)), (second, []))
        self.assertEqual(len(pool.assign(1, [8, 9])[1]), 2)

    def test_s3fifo_does_not_reuse_protected_or_leased_slots(self):
        from models.flashnext import pool_ring

        with mock.patch.dict(os.environ, {expert_pool.POLICY_FLAG: "s3fifo"}):
            pool = self.make(40)
        self.addCleanup(pool.close)
        old, _ = pool.assign(1, list(range(40)))
        first = pool_ring.lease_range(pool, 20)
        retained = [expert for expert, slot in old.items() if slot < first][:5]
        slots, misses = pool.assign(1, retained + list(range(40, 55)))
        self.assertEqual([slots[expert] for expert in retained], [old[expert] for expert in retained])
        self.assertEqual(set(slots.values()), set(range(first)))
        pool.invalidate(1, misses)
        pool_ring.return_range(pool, first, 20)
        _, new_misses = pool.assign(1, list(range(40, 55)))
        self.assertEqual(len(new_misses), 15)

    def make(self, slots):
        stride = get_slab_layout(32).record_stride
        return expert_pool.ExpertPool(_Store(), expert_pool.HEADER + slots * stride)

    def test_hits_misses_and_lru_eviction(self):
        pool = self.make(16)
        slots, misses = pool.assign(1, list(range(10)))
        self.assertEqual(len(misses), 10)
        self.assertEqual(len(set(slots.values())), 10)
        slots2, misses2 = pool.assign(1, [3, 4, 20])
        self.assertEqual(slots2[3], slots[3])
        self.assertEqual([e for e, _ in misses2], [20])
        # Fill the pool, then force evictions: the oldest keys go first and
        # keys routed in the same call survive.
        pool.assign(2, list(range(5)))           # 16 slots now used
        slots3, misses3 = pool.assign(1, [3, 4, 30, 31])
        self.assertEqual(slots3[3], slots[3])
        self.assertEqual(slots3[4], slots[4])
        evicted = {slots[0], slots[1]}
        self.assertEqual({slot for _, slot in misses3}, evicted)
        self.assertEqual(pool.hits, 2 + 2)
        self.assertEqual(pool.misses, 10 + 1 + 5 + 2)

    def test_too_small_pool_is_refused(self):
        with self.assertRaises(ValueError):
            self.make(4)

    def test_oversized_assignment_fails_before_evicting_or_claiming_slots(self):
        pool = self.make(16)
        pool.assign(1, [1, 2, 3])
        before = (list(pool._slot_of.items()), list(pool._free), pool.hits, pool.misses)
        with self.assertRaisesRegex(ValueError, "exceed"):
            pool.assign(1, list(range(17)))
        self.assertEqual(before, (list(pool._slot_of.items()), list(pool._free), pool.hits, pool.misses))

    def test_leased_ring_capacity_prevents_slot_aliasing(self):
        from models.flashnext import pool_ring

        pool = self.make(40)
        self.assertEqual(pool.usable_slots, 40)
        first = pool_ring.lease_range(pool, 20)
        self.assertEqual(first, 20)
        self.assertEqual(pool.usable_slots, 20)

        # 16 distinct experts fits within the 20 unleased slots
        slots, misses = pool.assign(1, list(range(16)))
        self.assertEqual(len(slots), 16)
        self.assertEqual(len(set(slots.values())), 16)
        self.assertTrue(all(s < 20 for s in slots.values()), "all assigned slots must be unleased (< 20)")

        # 25 distinct experts exceeds the 20 unleased slots (even though total slots is 40)
        before = (list(pool._slot_of.items()), list(pool._free), pool.hits, pool.misses)
        with self.assertRaisesRegex(ValueError, "exceed"):
            pool.assign(1, list(range(25)))
        # Pool state must be completely unmutated: no aliasing, no partial eviction
        self.assertEqual(before, (list(pool._slot_of.items()), list(pool._free), pool.hits, pool.misses))
        self.assertEqual(len(set(pool._slot_of.values())), 16, "no duplicated live slot assignments")


class FillTests(unittest.TestCase):
    def test_oversized_batch_uses_streaming_path_without_changing_pool(self):
        pool = AssignTests().make(16)
        pool.assign(4, [30])
        prior = list(pool._slot_of.items())
        gate = SimpleNamespace(cache=SimpleNamespace(store=_Store(), prefix=PREFIX + ".gate_proj"), group_size=32)
        fallback = mock.Mock(return_value=mx.array([1.0], dtype=mx.float32))
        layer = SimpleNamespace(layer_id=4, gate_proj=gate, hits=0, misses=0, _one_pass=fallback)
        routes = list(range(20))
        with mock.patch.object(expert_pool, "for_store", return_value=pool):
            result = expert_cache.StreamingSwitchGLU._pool_pass(layer, "indices", routes, "input", "scores", None, None, None)
        self.assertEqual(result.dtype, mx.bfloat16)
        self.assertEqual(result.tolist(), [1.0])
        fallback.assert_called_once_with(None, "indices", routes, True, "input", "scores", None, None, None)
        self.assertEqual(list(pool._slot_of.items()), prior)

    def test_pool_pass_with_leased_ring_falls_back_to_streamed_path_when_unleased_capacity_exceeded(self):
        from models.flashnext import pool_ring

        pool = AssignTests().make(40)
        pool_ring.lease_range(pool, 20)
        self.assertEqual(pool.usable_slots, 20)

        # Pre-assign 5 experts
        pool.assign(4, [100, 101, 102, 103, 104])
        prior = list(pool._slot_of.items())

        gate = SimpleNamespace(cache=SimpleNamespace(store=_Store(), prefix=PREFIX + ".gate_proj"), group_size=32)
        fallback = mock.Mock(return_value=mx.array([1.0], dtype=mx.float32))
        layer = SimpleNamespace(layer_id=4, gate_proj=gate, hits=0, misses=0, _one_pass=fallback)

        # 8 tokens with top-k routing 25 distinct experts: exceeds the 20 unleased slots (though 25 < total slots 40)
        routes = list(range(25))
        with mock.patch.object(expert_pool, "for_store", return_value=pool):
            result = expert_cache.StreamingSwitchGLU._pool_pass(layer, "indices", routes, "input", "scores", None, None, None)

        self.assertEqual(result.dtype, mx.bfloat16)
        fallback.assert_called_once_with(None, "indices", routes, True, "input", "scores", None, None, None)
        # Verify pool was not corrupted or aliased
        self.assertEqual(list(pool._slot_of.items()), prior)
        self.assertEqual(len(set(pool._slot_of.values())), len(pool._slot_of))

    def test_parallel_descriptor_initialization_opens_once(self):
        pool = AssignTests().make(16)
        def open_slowly(*args):
            time.sleep(0.02)
            return 123
        with mock.patch.object(expert_pool.os, "open", side_effect=open_slowly) as opened, \
             mock.patch.object(expert_pool.fcntl, "fcntl"), \
             mock.patch.dict(os.environ, {expert_pool.BUFFERED_FILL_FLAG: "1"}), \
             ThreadPoolExecutor(8) as workers:
            self.assertEqual(list(workers.map(pool._fill_fd, ["shard"] * 8)), [123] * 8)
        self.assertEqual(opened.call_count, 1)

    def test_partial_submission_is_drained(self):
        pool = AssignTests().make(16)
        _, misses = pool.assign(4, [3])
        first = mock.Mock()
        submit = mock.Mock(side_effect=[first, RuntimeError("submission failed")])
        with self.assertRaisesRegex(RuntimeError, "submission failed"):
            pool.fill(PREFIX, misses, submit)
        first.result.assert_called_once_with()

    def test_failed_pool_pass_drains_workers_and_invalidates_misses(self):
        pool = AssignTests().make(16)
        pool.assign(4, [1])  # existing resident expert survives
        failed = mock.Mock()
        failed.result.side_effect = OSError("read failed")
        later = mock.Mock()
        gate = SimpleNamespace(cache=SimpleNamespace(store=_Store(), prefix=PREFIX + ".gate_proj"), group_size=32)
        layer = SimpleNamespace(layer_id=4, gate_proj=gate, hits=0, misses=0)
        with mock.patch.object(expert_pool, "for_store", return_value=pool), \
             mock.patch.object(pool, "fill", return_value=[failed, later]):
            with self.assertRaisesRegex(OSError, "read failed"):
                expert_cache.StreamingSwitchGLU._pool_pass(layer, None, [1, 2], None, None, None, None, None)
        later.result.assert_called_once_with()
        self.assertIn((4, 1), pool._slot_of)
        self.assertNotIn((4, 2), pool._slot_of)
        _, retry = pool.assign(4, [2])
        self.assertEqual([e for e, _ in retry], [2])

    def test_failed_submission_invalidates_assigned_slot(self):
        pool = AssignTests().make(16)
        first = mock.Mock()
        submit = mock.Mock(side_effect=[first, RuntimeError("submit failed")])
        gate = SimpleNamespace(cache=SimpleNamespace(store=_Store(), prefix=PREFIX + ".gate_proj"), group_size=32)
        layer = SimpleNamespace(layer_id=4, gate_proj=gate, hits=0, misses=0)
        with mock.patch.object(expert_pool, "for_store", return_value=pool), \
             mock.patch.object(expert_cache, "_submit_read", submit):
            with self.assertRaisesRegex(RuntimeError, "submit failed"):
                expert_cache.StreamingSwitchGLU._pool_pass(layer, None, [2], None, None, None, None, None)
        first.result.assert_called_once_with()
        self.assertNotIn((4, 2), pool._slot_of)
        self.assertEqual([e for e, _ in pool.assign(4, [2])[1]], [2])

    def test_fill_places_checkpoint_rows_at_record_offsets(self):
        mx.random.seed(8)
        experts = 5
        tensors = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for part, size in ROWS.items():
                tensors[f"{PREFIX}.{projection}.{part}"] = mx.random.randint(
                    0, 255, (experts, size)
                ).astype(mx.uint8)
        with tempfile.TemporaryDirectory() as directory:
            shard = "model-00001-of-00001.safetensors"
            mx.save_safetensors(os.path.join(directory, shard), tensors)
            with open(os.path.join(directory, "model.safetensors.index.json"), "w") as out:
                json.dump({"weight_map": {key: shard for key in tensors}}, out)
            store = SafeTensorStore(directory)
            layout = get_slab_layout(32)
            pool = expert_pool.ExpertPool(
                store, expert_pool.HEADER + 16 * layout.record_stride)
            workers = ThreadPoolExecutor(4)
            try:
                slots, misses = pool.assign(4, [3, 0, 4])
                futures = pool.fill(PREFIX, misses, workers.submit, chunk=2)
                for future in futures:
                    future.result()
                for expert, slot in slots.items():
                    base = expert_pool.HEADER + slot * layout.record_stride
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        for part, size in ROWS.items():
                            start = base + layout.offset(projection, part)
                            expected = store.rows_np(
                                f"{PREFIX}.{projection}.{part}", [expert]
                            ).tobytes()
                            self.assertEqual(
                                bytes(pool.array[start:start + size]), expected,
                                f"expert {expert} {projection}.{part}",
                            )
            finally:
                workers.shutdown()
                pool.close()
                store.close()


class ConfigTests(unittest.TestCase):
    def test_off_by_default_and_parsed(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(expert_pool.SIZE_FLAG, None)
            self.assertFalse(expert_pool.enabled())
        with mock.patch.dict(os.environ, {expert_pool.SIZE_FLAG: "6"}):
            self.assertEqual(expert_pool.configured_gb(), 6.0)
            self.assertTrue(expert_pool.enabled())

    def test_for_store_allocation_fallback_to_1gb_on_error(self):
        store = _Store()
        with mock.patch.dict(os.environ, {expert_pool.SIZE_FLAG: "2"}):
            self.addCleanup(expert_pool.set_budget_gb, None)
            calls = []

            def fake_init(st, size_bytes, *args, **kwargs):
                calls.append(size_bytes)
                if size_bytes == 2_000_000_000:
                    raise MemoryError("simulated out of memory at 2 GB")
                return None

            with mock.patch.object(expert_pool.ExpertPool, "__init__", side_effect=fake_init):
                pool = expert_pool.for_store(store)
                self.assertIsNotNone(pool)
                self.assertEqual(calls, [2_000_000_000, 1_000_000_000])
                self.assertEqual(expert_pool.configured_gb(), 1.0)

    def test_for_store_allocation_fallback_to_ssd_when_1gb_fails(self):
        store = _Store()
        with mock.patch.dict(os.environ, {expert_pool.SIZE_FLAG: "2"}):
            self.addCleanup(expert_pool.set_budget_gb, None)

            with mock.patch.object(expert_pool.ExpertPool, "__init__", side_effect=MemoryError("OOM")):
                pool = expert_pool.for_store(store)
                # Falls back to pure SSD (pool is None) without raising
                self.assertIsNone(pool)
                self.assertIsNone(getattr(store, "_flashnext_expert_pool", None))


class BufferedFillTests(unittest.TestCase):
    """The opt-in buffered fill reads the same bytes through separate descriptors; the uncached ones stay for the ring."""

    def _run(self, buffered):
        mx.random.seed(11)
        tensors = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for part, size in ROWS.items():
                tensors[f"{PREFIX}.{projection}.{part}"] = mx.random.randint(0, 255, (5, size)).astype(mx.uint8)
        with tempfile.TemporaryDirectory() as directory:
            shard = "model-00001-of-00001.safetensors"
            mx.save_safetensors(os.path.join(directory, shard), tensors)
            with open(os.path.join(directory, "model.safetensors.index.json"), "w") as out:
                json.dump({"weight_map": {key: shard for key in tensors}}, out)
            store = SafeTensorStore(directory)
            layout = get_slab_layout(32)
            pool = expert_pool.ExpertPool(store, expert_pool.HEADER + 16 * layout.record_stride)
            workers = ThreadPoolExecutor(4)
            env = {expert_pool.BUFFERED_FILL_FLAG: "1"} if buffered else {expert_pool.BUFFERED_FILL_FLAG: "0"}
            try:
                with mock.patch.dict(os.environ, env):
                    slots, misses = pool.assign(4, [4, 0, 1, 3])
                    with mock.patch.object(os, "preadv", wraps=os.preadv) as read, \
                            mock.patch.object(workers, "submit", wraps=workers.submit) as submit:
                        for future in pool.fill(PREFIX, misses, submit, chunk=2):
                            future.result()
                    self.assertEqual(read.call_count, 2 * len(tensors))
                    self.assertEqual(submit.call_count, 12)  # 6 weight tasks, 6 small-part tasks
                    data = {}
                    for expert, slot in slots.items():
                        base = expert_pool.HEADER + slot * layout.record_stride
                        for projection in ("gate_proj", "up_proj", "down_proj"):
                            for part, size in ROWS.items():
                                start = base + layout.offset(projection, part)
                                row = bytes(pool.array[start:start + size])
                                self.assertEqual(row, np.array(tensors[f"{PREFIX}.{projection}.{part}"][expert]).tobytes())
                                data[(expert, projection, part)] = row
                    return data, dict(pool._fds), dict(pool._fill_fds)
            finally:
                workers.shutdown()
                pool.close()
                store.close()

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(expert_pool.BUFFERED_FILL_FLAG, None)
            self.assertFalse(expert_pool.buffered_fill_enabled())

    def test_buffered_fill_is_byte_identical_and_uses_separate_descriptors(self):
        reference, ref_uncached, ref_buffered = self._run(False)
        candidate, cand_uncached, cand_buffered = self._run(True)
        self.assertEqual(reference, candidate)
        self.assertTrue(ref_uncached and not ref_buffered)       # reference: only the shared F_NOCACHE descriptors
        self.assertTrue(cand_buffered and not cand_uncached)     # opt-in: own descriptors, shared ones untouched

    def test_close_releases_both_descriptor_sets(self):
        stride = get_slab_layout(32).record_stride
        pool = expert_pool.ExpertPool(_Store(), expert_pool.HEADER + 16 * stride)
        read, write = os.pipe()
        os.close(write)
        pool._fill_fds["x"] = read
        pool.close()
        self.assertEqual(pool._fill_fds, {})
        with self.assertRaises(OSError):
            os.fstat(read)


# ---------------------------------------------------------------------------------------------------------------
# Cost-aware policy, decode prefetch, byte exactness
# ---------------------------------------------------------------------------------------------------------------

STRIDE = get_slab_layout(32).record_stride
POLICIES = ("lru", "s3fifo", "least-stale", "cost-aware")
PREFETCH_ENV = {expert_pool.PREFETCH_FLAG: "1"}
FLAGS = (expert_pool.POLICY_FLAG, expert_pool.FLAT_SLOTS_FLAG, expert_pool.PREFETCH_FLAG,
         expert_pool.PREFETCH_SLOTS_FLAG, expert_pool.PREFETCH_DEPTH_FLAG, expert_pool.PREFETCH_WIDTH_FLAG,
         expert_pool.PREFETCH_MIN_P_FLAG, expert_pool.PREFETCH_WORKERS_FLAG, expert_pool.PREFETCH_GATE_FLAG,
         "FLASHNEXT_EXPERT_POOL_COST_WEIGHTS", "FLASHNEXT_EXPERT_PACK")


def make_pool(total, store=None, **env):
    """A pool of ``total`` physical slots with exactly the given flags set."""
    with mock.patch.dict(os.environ, env):
        for flag in FLAGS:
            if flag not in env:
                os.environ.pop(flag, None)
        return expert_pool.ExpertPool(store or _Store(), expert_pool.HEADER + total * STRIDE)


def cost_pool(total=32, **env):
    return make_pool(total, **{expert_pool.POLICY_FLAG: "cost-aware", **env})


class CostAwareTests(unittest.TestCase):
    def pool(self, total=16, **env):
        pool = cost_pool(total, **env)
        self.addCleanup(pool.close)
        return pool

    def rows(self, pool, fill=1):
        return {part: bytes([fill]) * size for part, size in pool._row_sizes.items()}

    def check(self, pool, leased=0):
        """Flat table, policy and free list all agree with ``_slot_of``; every decode slot is accounted for once."""
        slots = list(pool._slot_of.values())
        self.assertEqual(len(set(slots)), len(slots))
        self.assertFalse(set(slots) & set(pool._free))
        self.assertEqual(len(pool._free) + len(slots) + leased, pool.slots)
        for key, slot in pool._slot_of.items():
            self.assertEqual(pool._cost.key_of(slot), key)
        self.assertEqual(int(pool._cost._occupied.sum()), len(slots))
        if pool._flat_slots is not None:
            flat = {(layer, expert): int(pool._flat_slots[layer, expert])
                    for layer, expert in zip(*np.nonzero(pool._flat_slots >= 0))}
            self.assertEqual(flat, pool._slot_of)

    def test_hits_misses_and_eviction_counter(self):
        pool = self.pool(16)
        slots, misses = pool.assign(1, range(10))
        self.assertEqual((len(misses), pool.evictions), (10, 0))
        again, misses = pool.assign(1, [3, 4, 20])
        self.assertEqual((again[3], again[4]), (slots[3], slots[4]))
        self.assertEqual([expert for expert, _ in misses], [20])
        self.assertEqual((pool.hits, pool.misses), (2, 11))
        pool.assign(2, range(5))                              # 16 slots full
        _, misses = pool.assign(1, [3, 4, 30, 31])
        self.assertEqual((len(misses), pool.evictions), (2, 2))
        self.check(pool)

    def test_every_policy_counts_evictions(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                pool = make_pool(16, **{expert_pool.POLICY_FLAG: policy})
                self.addCleanup(pool.close)
                pool.assign(1, range(16))
                self.assertEqual(pool.evictions, 0)
                pool.assign(2, range(4))
                self.assertEqual(pool.evictions, 4)
                self.assertEqual(pool.stats()["evictions"], 4)

    def test_frequently_routed_key_outlives_newer_cold_keys(self):
        pool = self.pool(16)
        for _ in range(10):
            pool.assign(1, [0, 1])
        for start in range(0, 180, 3):                        # a flood of one-off keys from another layer
            pool.assign(2, [start, start + 1, start + 2])
        self.assertIn((1, 0), pool._slot_of)
        self.assertIn((1, 1), pool._slot_of)
        self.check(pool)

    def test_this_calls_keys_are_never_evicted(self):
        pool = self.pool(16)
        old, _ = pool.assign(1, range(16))
        wanted = [0, 5, 9, 12] + list(range(100, 112))
        slots, misses = pool.assign(1, wanted)
        for expert in (0, 5, 9, 12):
            self.assertEqual(slots[expert], old[expert])
        self.assertEqual(len(set(slots.values())), 16)
        self.assertEqual(len(misses), 12)
        self.check(pool)

    def test_invalidate_returns_slots_and_leaves_no_stale_hit(self):
        pool = self.pool(16)
        pool.assign(1, [1])
        _, misses = pool.assign(1, [2, 3])
        pool.invalidate(1, misses)
        self.assertNotIn((1, 2), pool._slot_of)
        self.assertTrue({slot for _, slot in misses} <= set(pool._free))
        self.check(pool)
        self.assertEqual([expert for expert, _ in pool.assign(1, [2, 3])[1]], [2, 3])

    def test_seed_from_ring_enters_the_policy_as_the_stalest_entry(self):
        pool = self.pool(16)
        self.assertEqual(pool.seed_from_ring(7, 3, self.rows(pool)), sum(pool._row_sizes.values()))
        self.assertEqual(pool.seed_from_ring(7, 3, self.rows(pool)), 0)       # already resident
        seeded = pool._slot_of[(7, 3)]
        self.assertEqual(pool._cost.key_of(seeded), (7, 3))
        pool.assign(1, range(15))                                             # fills the pool
        _, misses = pool.assign(2, [0])
        self.assertEqual(misses[0][1], seeded)                                # the seeded record goes first
        self.assertNotIn((7, 3), pool._slot_of)
        self.check(pool)

    def test_ring_lease_and_return_keep_policy_free_list_and_flat_table_consistent(self):
        from models.flashnext import pool_ring

        pool = self.pool(40, **{expert_pool.FLAT_SLOTS_FLAG: "1"})
        old, _ = pool.assign(1, range(40))
        first = pool_ring.lease_range(pool, 20)
        self.assertTrue(all(pool._flat_slots[1, expert] == -1 for expert, slot in old.items() if slot >= first))
        self.check(pool, leased=20)
        active, misses = pool.assign(1, [0, 40, 41])
        self.assertEqual(active[0], old[0])
        self.assertTrue(all(slot < first for _, slot in misses))
        pool.invalidate(1, misses)
        self.check(pool, leased=20)
        self.assertEqual(len(pool.assign(1, [40, 41])[1]), 2)
        pool_ring.return_range(pool, first, 20)
        self.check(pool)

    def test_flat_slot_table_matches_slot_map_under_random_operations(self):
        from models.flashnext import pool_ring

        pool = self.pool(48, **{expert_pool.FLAT_SLOTS_FLAG: "1"})
        rng, leased, first = random.Random(5), 0, None
        for step in range(400):
            action = rng.random()
            if action < 0.8:
                experts = rng.sample(range(30), rng.randint(1, 6))
                if len(set(experts)) <= pool.usable_slots:
                    _, misses = pool.assign(rng.randrange(5), experts)
                    if rng.random() < 0.15:
                        pool.invalidate(rng.randrange(5), misses)    # wrong layer: must be a no-op for foreign keys
            elif action < 0.9 and first is None:
                first, leased = pool_ring.lease_range(pool, 16), 16
            elif action < 0.95 and first is not None:
                pool_ring.return_range(pool, first, leased)
                first, leased = None, 0
            else:
                pool.seed_from_ring(rng.randrange(5), rng.randrange(30, 60), self.rows(pool))
            self.check(pool, leased)

    def test_slot_conservation_after_random_operations(self):
        pool = self.pool(32)
        rng = random.Random(11)
        for _ in range(300):
            layer, experts = rng.randrange(6), rng.sample(range(40), rng.randint(1, 8))
            slots, misses = pool.assign(layer, experts)
            self.assertEqual(set(slots), set(experts))
            if rng.random() < 0.1:
                pool.invalidate(layer, misses)
            self.check(pool)

    def test_assignment_matches_a_bare_policy_driven_in_the_documented_order(self):
        slots = 32
        pool = self.pool(slots)
        policy, free, mapping = CostAwarePolicy(slots, layers=48, experts=512), list(range(slots - 1, -1, -1)), {}
        rng, hits, misses_total = random.Random(3), 0, 0
        for _ in range(500):
            layer = rng.choice((0, 1, 2, 3, 4, rng.randrange(48)))
            experts = list(dict.fromkeys(min(int(rng.expovariate(0.25)), 60) for _ in range(rng.randint(1, 7))))
            policy.advance(layer)
            hit = {e: mapping[(layer, e)] for e in experts if (layer, e) in mapping}
            for expert, slot in hit.items():
                policy.touch((layer, expert), slot)
            missing = [e for e in experts if e not in hit]
            need = len(missing) - len(free)
            if need > 0:
                for slot in policy.victims(need, protected=hit.values()):
                    key = policy.key_of(slot)
                    policy.discard(key, slot)
                    del mapping[key]
                    free.append(slot)
            expected_misses = []
            for expert in missing:
                slot = free.pop()
                mapping[(layer, expert)] = slot
                policy.insert((layer, expert), slot)
                hit[expert] = slot
                expected_misses.append((expert, slot))
            policy.observe(layer, experts, misses=len(missing))
            got, got_misses = pool.assign(layer, experts)
            hits, misses_total = hits + len(experts) - len(missing), misses_total + len(missing)
            self.assertEqual((pool.hits, pool.misses), (hits, misses_total))
            self.assertEqual(got, hit)
            self.assertEqual(got_misses, expected_misses)
        self.assertGreater(pool.evictions, 0)

    def test_weights_file_refill_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "weights.json")
            with open(path, "w") as out:
                json.dump({"version": 1, "refill": 0.5}, out)
            pool = self.pool(16, **{"FLASHNEXT_EXPERT_POOL_COST_WEIGHTS": path})
            self.assertTrue(pool.wants_refill_cost)
            pool.observe_refill(3, 0.002)
            self.assertTrue(pool._cost._refill_seen[3])
            self.assertEqual(pool.stats()["cost"]["weights"]["refill"], 0.5)
            with open(path, "w") as out:
                out.write("{not json")
            with self.assertRaisesRegex(ValueError, "cannot read cost weights"):
                cost_pool(16, **{"FLASHNEXT_EXPERT_POOL_COST_WEIGHTS": path})
        default = self.pool(16)
        self.assertFalse(default.wants_refill_cost)
        default.observe_refill(3, 0.002)
        self.assertFalse(default._cost._refill_seen.any())
        lru = make_pool(16)
        self.addCleanup(lru.close)
        self.assertFalse(lru.wants_refill_cost)
        lru.observe_refill(3, 0.002)                          # no-op

    def test_stats_are_json_serializable_for_every_policy(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                pool = make_pool(16, **{expert_pool.POLICY_FLAG: policy})
                self.addCleanup(pool.close)
                self.assertIsNone(pool.stats()["hit_rate"])
                pool.assign(1, range(4))
                pool.assign(1, range(6))
                stats = json.loads(json.dumps(pool.stats()))
                self.assertEqual((stats["policy"], stats["slots"], stats["resident"], stats["hits"], stats["misses"]),
                                 (policy, 16, 6, 4, 6))
                self.assertAlmostEqual(stats["hit_rate"], 0.4)
                self.assertEqual("cost" in stats, policy == "cost-aware")
                self.assertIsNone(stats["prefetch"])

    def test_invalid_policy_and_prefetch_flags_name_the_flag(self):
        with self.assertRaisesRegex(ValueError, expert_pool.POLICY_FLAG):
            make_pool(16, **{expert_pool.POLICY_FLAG: "mru"})
        bad = {expert_pool.PREFETCH_FLAG: "yes", expert_pool.PREFETCH_SLOTS_FLAG: "0", expert_pool.PREFETCH_DEPTH_FLAG: "x",
               expert_pool.PREFETCH_WIDTH_FLAG: "-1", expert_pool.PREFETCH_MIN_P_FLAG: "1.5",
               expert_pool.PREFETCH_WORKERS_FLAG: "0", expert_pool.PREFETCH_GATE_FLAG: "2"}
        for flag, value in bad.items():
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, flag):
                    make_pool(32, **{expert_pool.PREFETCH_FLAG: "1", flag: value})


class SizingTests(unittest.TestCase):
    def sized(self, total, **env):
        pool = make_pool(total, **{**PREFETCH_ENV, **env})
        self.addCleanup(pool.close)
        return pool

    def test_staging_is_carved_from_the_same_mapping(self):
        plain, pool = make_pool(64), self.sized(64)
        self.addCleanup(plain.close)
        self.assertEqual((plain.slots, plain.staging_slots, plain.prefetch), (64, 0, None))
        self.assertEqual((pool.slots, pool.staging_slots), (56, 8))     # 16 requested, capped at total // 8
        self.assertEqual(len(pool._mm), len(plain._mm))
        self.assertEqual(sorted(pool._free), list(range(56)))
        self.assertEqual((pool.prefetch.first_slot, pool.prefetch.slots), (56, 8))
        self.assertEqual(pool.stats()["prefetch"]["staging_bytes"], 8 * STRIDE)

    def test_configured_count_below_the_cap_is_honoured(self):
        pool = self.sized(100, **{expert_pool.PREFETCH_SLOTS_FLAG: "4"})
        self.assertEqual((pool.slots, pool.staging_slots), (96, 4))

    def test_sixteen_decode_slot_floor(self):
        pool = self.sized(20)
        self.assertEqual((pool.slots, pool.staging_slots), (18, 2))
        pool = self.sized(17)
        self.assertEqual((pool.slots, pool.staging_slots), (16, 1))

    def test_no_staging_slots_disables_prefetch_and_says_why(self):
        pool = self.sized(16)
        self.assertEqual((pool.slots, pool.staging_slots, pool.prefetch), (16, 0, None))
        stats = json.loads(json.dumps(pool.stats()))
        self.assertIsNone(stats["prefetch"])
        self.assertIn("no staging slots", stats["prefetch_disabled"])
        pool.prefetch_ahead(3)                                 # still a no-op

    def test_staging_slots_stay_out_of_decode_bookkeeping(self):
        pool = self.sized(32, **{expert_pool.FLAT_SLOTS_FLAG: "1", expert_pool.POLICY_FLAG: "cost-aware"})
        self.assertEqual((pool.slots, pool.staging_slots), (28, 4))
        self.assertEqual(pool._cost.slots, 28)
        slots, _ = pool.assign(1, range(20))
        self.assertTrue(all(slot < 28 for slot in slots.values()))
        self.assertEqual(pool.usable_slots, 28)

    def test_defaults_have_no_prefetcher_threads_or_hot_path_work(self):
        before = threading.active_count()
        pool = make_pool(32)
        self.addCleanup(pool.close)
        self.assertEqual(threading.active_count(), before)
        self.assertIsNone(pool.prefetch)
        self.assertIsNone(pool._frequency)
        self.assertEqual((pool.slots, pool.staging_slots), (32, 0))
        pool.prefetch_ahead(3)
        pool.assign(3, [1, 2])
        self.assertIsNone(pool.stats()["prefetch"])
        pool.close()
        self.assertEqual(threading.active_count(), before)


# A real store whose records have deterministic distinct bytes: every (layer, expert, part) row is a ramp offset by a
# per-row constant, so a record placed in the wrong slot, row or part cannot compare equal.
FIXTURE_LAYERS, FIXTURE_EXPERTS = (4, 5, 6, 7), 12
_FIXTURE = {}


def fixture_prefix(layer):
    return f"language_model.model.layers.{layer}.mlp.switch_mlp"


def row_pattern(layer, expert, projection, part):
    size = ROWS[part]
    base = (layer * 53 + expert * 7 + ("gate_proj", "up_proj", "down_proj").index(projection) * 29
            + tuple(ROWS).index(part) * 3) % 251
    return ((np.arange(size, dtype=np.int64) + base) % 251).astype(np.uint8)


def fixture_store():
    if "store" not in _FIXTURE:
        directory = tempfile.mkdtemp(prefix="flashnext-pool-fixture-")
        tensors = {}
        for layer in FIXTURE_LAYERS:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                for part in ROWS:
                    tensors[f"{fixture_prefix(layer)}.{projection}.{part}"] = mx.array(
                        np.stack([row_pattern(layer, e, projection, part) for e in range(FIXTURE_EXPERTS)]))
        shard = "model-00001-of-00001.safetensors"
        mx.save_safetensors(os.path.join(directory, shard), tensors)
        with open(os.path.join(directory, "model.safetensors.index.json"), "w") as out:
            json.dump({"weight_map": {key: shard for key in tensors}}, out)
        _FIXTURE["store"], _FIXTURE["directory"] = SafeTensorStore(directory), directory
        _FIXTURE["expected"] = {}
        unittest.addModuleCleanup(_drop_fixture)
    return _FIXTURE["store"]


def _drop_fixture():
    import shutil

    store = _FIXTURE.pop("store", None)
    if store is not None:
        store.close()
    shutil.rmtree(_FIXTURE.pop("directory", ""), ignore_errors=True)
    _FIXTURE.clear()


def expected_record(layer, expert):
    """[(offset within the record, bytes)] for the nine parts, cached."""
    cache = _FIXTURE["expected"]
    if (layer, expert) not in cache:
        layout = get_slab_layout(32)
        cache[layer, expert] = [(layout.offset(projection, part), row_pattern(layer, expert, projection, part).tobytes())
                                for projection in ("gate_proj", "up_proj", "down_proj") for part in ROWS]
    return cache[layer, expert]


def settle(pool, timeout=20.0):
    """Wait until no prefetch read is queued or running."""
    prefetch = pool.prefetch
    with prefetch._cond:
        assert prefetch._cond.wait_for(lambda: all(entry.state not in (0, 1) for entry in prefetch._live), timeout)


class ExactBytesTests(unittest.TestCase):
    """The checkpoint-free equivalent of the token-digest gate: whatever the policy and prefetch do, every routed
    expert's slot holds exactly the checkpoint record."""

    def setUp(self):
        self.store = fixture_store()
        self.workers = ThreadPoolExecutor(4)
        self.addCleanup(self.workers.shutdown)

    def pool(self, total=32, **env):
        pool = make_pool(total, self.store, **env)
        self.addCleanup(pool.close)
        return pool

    def assert_record(self, pool, layer, expert, slot):
        base = expert_pool.HEADER + slot * pool.stride
        for offset, data in expected_record(layer, expert):
            start = base + offset
            self.assertEqual(pool.array[start:start + len(data)].tobytes(), data, f"layer {layer} expert {expert}")

    def drive(self, pool, accesses=200, seed=0, settle_odds=0.7, per_access=5):
        rng = random.Random(seed)
        weights = [1.0 / (rank + 1) ** 0.7 for rank in range(FIXTURE_EXPERTS)]
        staging = range(pool.slots, pool.slots + pool.staging_slots)
        for step in range(accesses):
            layer = FIXTURE_LAYERS[step % len(FIXTURE_LAYERS)]
            experts = list(dict.fromkeys(rng.choices(range(FIXTURE_EXPERTS), weights, k=per_access)))
            slots, misses = pool.assign(layer, experts)
            futures = pool.fill(fixture_prefix(layer), misses, self.workers.submit, chunk=2)
            pool.prefetch_ahead(layer)
            for future in futures:
                future.result(timeout=30)
            for expert, slot in slots.items():
                self.assertLess(slot, pool.slots)
                self.assert_record(pool, layer, expert, slot)
            self.assertFalse(set(staging) & (set(pool._slot_of.values()) | set(pool._free)))
            if pool.prefetch is not None:
                stats = pool.prefetch.stats()
                self.assertEqual(stats["staged"] + stats["free_slots"] + stats["claimed"], pool.staging_slots)
                if rng.random() < settle_odds:
                    settle(pool)

    def test_every_policy_is_byte_exact_with_prefetch_off_and_on(self):
        for policy in POLICIES:
            for prefetch in (False, True):
                with self.subTest(policy=policy, prefetch=prefetch):
                    env = {expert_pool.POLICY_FLAG: policy}
                    if prefetch:
                        env |= {**PREFETCH_ENV, expert_pool.PREFETCH_MIN_P_FLAG: "0.3", expert_pool.PREFETCH_SLOTS_FLAG: "4"}
                    pool = self.pool(32, **env)
                    self.assertEqual((pool.slots, pool.staging_slots), (28, 4) if prefetch else (32, 0))
                    self.drive(pool)
                    if prefetch:
                        stats = pool.prefetch.stats()
                        self.assertGreater(stats["used"] + stats["late"], 0, stats)
                        self.assertGreater(stats["issued"], 0)
                        settle(pool)
                        stats = pool.prefetch.stats()
                        self.assertEqual((stats["claimed"], stats["staged"] + stats["free_slots"]), (0, 4))
                        json.dumps(pool.stats())

    def test_gate_off_is_byte_exact_too(self):
        pool = self.pool(32, **PREFETCH_ENV, **{expert_pool.PREFETCH_MIN_P_FLAG: "0.3",
                                                 expert_pool.PREFETCH_GATE_FLAG: "0", expert_pool.PREFETCH_SLOTS_FLAG: "4"})
        self.drive(pool, accesses=120, seed=4, settle_odds=0.3)

    def test_prefetch_read_failure_falls_back_to_a_correct_direct_read(self):
        pool = self.pool(32, **PREFETCH_ENV, **{expert_pool.PREFETCH_MIN_P_FLAG: "0.3", expert_pool.PREFETCH_SLOTS_FLAG: "4"})
        original = pool.prefetch._read_record

        def flaky(layer, expert, slot):
            if (layer + expert) % 2:
                raise OSError("injected prefetch failure")
            original(layer, expert, slot)

        pool.prefetch._read_record = flaky
        self.drive(pool, accesses=120, seed=2)
        stats = pool.prefetch.stats()
        self.assertGreater(stats["failed"], 0)
        self.assertGreater(stats["used"] + stats["late"], 0)

    def staged_pool(self, **env):
        """A pool whose layer-5 records 3 and 4 are staged and complete."""
        pool = self.pool(32, **PREFETCH_ENV, **env)
        for layer in (4, 5, 6, 7, 4):
            pool.prefetch.observe(layer)
        pool._frequency.table[5, 3] = pool._frequency.table[5, 4] = 0.9
        self.assertEqual(pool.prefetch.issue(4, pool._slot_of.__contains__), 2)
        settle(pool)
        self.assertEqual(pool.prefetch.stats()["staged"], 2)
        return pool

    def test_demand_failure_invalidates_and_leaves_no_stale_hit(self):
        pool = self.pool(32, **PREFETCH_ENV)
        pool.prefetch.observe(4)
        _, misses = pool.assign(4, [2])
        failed = Future()
        failed.set_exception(OSError("demand read failed"))
        pool.invalidate(4, misses)
        self.assertNotIn((4, 2), pool._slot_of)
        slots, again = pool.assign(4, [2])
        self.assertEqual([expert for expert, _ in again], [2])          # a miss, never a stale hit
        for future in pool.fill(fixture_prefix(4), again, self.workers.submit):
            future.result(timeout=30)
        self.assert_record(pool, 4, 2, slots[2])

    def test_promotion_failure_releases_the_entry_and_invalidates(self):
        pool = self.staged_pool()
        pool.prefetch._copy_record = mock.Mock(side_effect=OSError("copy failed"))
        _, misses = pool.assign(5, [3, 4])
        futures = pool.fill(fixture_prefix(5), misses, self.workers.submit)
        for future in futures:
            with self.assertRaises(OSError):
                future.result(timeout=30)
        pool.invalidate(5, misses)
        stats = pool.prefetch.stats()
        self.assertEqual((stats["claimed"], stats["staged"], stats["free_slots"]), (0, 0, 4))
        self.assertEqual(stats["used"], 0)
        self.assertFalse({(5, 3), (5, 4)} & set(pool._slot_of))
        self.assertEqual(len(pool.assign(5, [3, 4])[1]), 2)

    def test_submit_failure_midway_releases_claimed_entries_and_reopens_the_gate(self):
        pool = self.staged_pool()
        _, misses = pool.assign(5, [3, 4])
        calls = []

        def submit(function, *args):
            calls.append(function)
            if len(calls) == 2:
                raise RuntimeError("submit failed")
            return self.workers.submit(function, *args)

        with self.assertRaisesRegex(RuntimeError, "submit failed"):
            pool.fill(fixture_prefix(5), misses, submit)
        settle(pool)
        stats = pool.prefetch.stats()
        self.assertEqual(stats["claimed"], 0)
        self.assertEqual(stats["staged"] + stats["free_slots"], 4)
        self.assertEqual(pool.prefetch._demand, 0)
        self.assertFalse(pool.prefetch._executor.stats()["held"])
        pool.invalidate(5, misses)

    def test_failure_before_any_future_reopens_the_gate(self):
        pool = self.pool(32, **PREFETCH_ENV)
        _, misses = pool.assign(4, [9])
        with self.assertRaisesRegex(RuntimeError, "no workers"):
            pool.fill(fixture_prefix(4), misses, mock.Mock(side_effect=RuntimeError("no workers")))
        self.assertEqual(pool.prefetch._demand, 0)
        self.assertFalse(pool.prefetch._executor.stats()["held"])

    def test_gate_is_shut_for_real_reads_and_open_when_the_fill_completes(self):
        pool = self.pool(32, **PREFETCH_ENV)
        _, misses = pool.assign(4, [9])
        release, started = threading.Event(), threading.Event()

        def submit(function, *args):
            def slow():
                started.set()
                self.assertTrue(release.wait(20))
                return function(*args)
            return self.workers.submit(slow)

        futures = pool.fill(fixture_prefix(4), misses, submit)
        self.assertTrue(started.wait(20))
        self.assertEqual(pool.prefetch._demand, 1)
        self.assertTrue(pool.prefetch._executor.stats()["held"])
        release.set()
        for future in futures:
            future.result(timeout=30)
        self.assertTrue(self._until(lambda: pool.prefetch._demand == 0))
        self.assertFalse(pool.prefetch._executor.stats()["held"])

    @staticmethod
    def _until(condition, timeout=20.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if condition():
                return True
            threading.Event().wait(0.005)
        return condition()

    def test_close_waits_for_a_blocked_prefetch_read_and_nothing_writes_afterwards(self):
        pool = self.pool(32, **PREFETCH_ENV, **{expert_pool.PREFETCH_MIN_P_FLAG: "0.3", expert_pool.PREFETCH_WORKERS_FLAG: "1"})
        for layer in (4, 5, 6, 7, 4):
            pool.prefetch.observe(layer)
        pool._frequency.table[5, 3] = pool._frequency.table[5, 4] = 0.9
        original, started, release, writes = pool.prefetch._read_record, threading.Event(), threading.Event(), []

        def blocked(layer, expert, slot):
            started.set()
            release.wait(20)
            original(layer, expert, slot)
            writes.append((layer, expert))

        pool.prefetch._read_record = blocked
        pool.prefetch.issue(4, pool._slot_of.__contains__)
        self.assertTrue(started.wait(20))
        closed = threading.Event()
        closer = threading.Thread(target=lambda: (pool.close(), closed.set()))
        closer.start()
        self.assertFalse(closed.wait(0.3))
        release.set()
        self.assertTrue(closed.wait(20))
        closer.join(20)
        done = list(writes)
        self.assertEqual(len(done), 1)                         # the running read finished, the queued one never started
        threading.Event().wait(0.1)
        self.assertEqual(writes, done)
        self.assertFalse(any(thread.is_alive() for thread in pool.prefetch._executor._threads))


class StatsWiringTests(unittest.TestCase):
    def test_kv_diagnostics_reports_expert_pool_stats(self):
        import inspect

        from veystrix.backends import flashnext as backend

        source = inspect.getsource(backend.FlashNextBackend.kv_diagnostics) if hasattr(backend, "FlashNextBackend") else ""
        self.assertIn('"expert_pool_stats": pool.stats() if pool is not None else None', source)

    def test_memory_budget_counts_staging_slots(self):
        from models.flashnext import memory_budget

        pool = make_pool(64, **PREFETCH_ENV)
        self.addCleanup(pool.close)
        pool.assign(1, range(10))
        seen = {}

        def plan(**kwargs):
            seen.update(kwargs)
            return SimpleNamespace(expert_pool_bytes=7)

        store = SimpleNamespace(_flashnext_expert_pool=pool, _flashnext_elastic_pool_bytes=7)
        with mock.patch.object(memory_budget, "plan", plan), \
                mock.patch("models.flashnext.memory_manager.read_system", return_value={"available": 1, "pressure": 1}):
            memory_budget.apply(store, phase="decode", tokens=1, chunk=1, ring=False, verification_width=0,
                                resident=0, cache=[])
        self.assertEqual(seen["existing_pool"], (10 + 8) * STRIDE)


if __name__ == "__main__":
    unittest.main()
