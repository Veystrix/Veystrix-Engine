"""Prefill read statistics (``FLASHNEXT_PREFILL_READ_STATS=1``): what each prefill read, re-read, staged and fell back on.

The recorder is exercised alone (lifecycle, summary shape, nesting) and on the tiny streamed model through all three read
points: chunk-major (``StreamingSwitchGLU._one_pass``), the all-experts layer-major window and the route-union groups.
"""
from __future__ import annotations

import json
import os
import unittest
from unittest import mock

import mlx.core as mx

from models.flashnext import expert_cache, layer_major, prefill_reads
from models.flashnext.tests.unit.test_route_union import UnionBase, spans_of
from models.flashnext.tests.bench.streaming_tiny import EXPERTS, LAYERS, route_sets

SUMMARY_KEYS = {"order", "prompt_tokens", "layers", "calls", "experts_read", "experts_reread", "reread_share", "bytes_read",
                "ring_staged_calls", "fallback_calls", "fallback_reasons", "max_set", "per_layer"}


class FakeGlu:
    layer_id = 3


class RecorderTests(unittest.TestCase):
    def setUp(self):
        prefill_reads.reset()
        self.addCleanup(prefill_reads.reset)
        patcher = mock.patch.dict(os.environ, {})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(prefill_reads.FLAG, None)

    def test_off_by_default_every_hook_is_one_check(self):
        self.assertFalse(prefill_reads.enabled())
        self.assertIsNone(prefill_reads.begin(100))
        self.assertIsNone(prefill_reads.ACTIVE[0])
        prefill_reads.end(None)
        self.assertIsNone(prefill_reads.summary())

    def test_begin_end_summary_for_the_last_completed_prefill(self):
        os.environ[prefill_reads.FLAG] = "1"
        token = prefill_reads.begin(512)
        self.assertIs(prefill_reads.ACTIVE[0], token)
        self.assertIsNone(prefill_reads.begin(9), "a nested prefill does not start a second recording")
        glu = FakeGlu()
        token.note("chunk", glu, [1, 2, 3], True)
        token.note("chunk", glu, [3, 4], False, ring=None)
        prefill_reads.end(token)
        self.assertIsNone(prefill_reads.ACTIVE[0])
        summary = prefill_reads.summary()
        self.assertEqual(set(summary), SUMMARY_KEYS)
        self.assertEqual((summary["order"], summary["prompt_tokens"], summary["layers"], summary["calls"]),
                         ("chunk", 512, 1, 2))
        self.assertEqual((summary["experts_read"], summary["experts_reread"], summary["max_set"]), (5, 1, 3))
        self.assertAlmostEqual(summary["reread_share"], 0.2)
        self.assertEqual((summary["ring_staged_calls"], summary["fallback_calls"]), (1, 1))
        self.assertEqual(summary["fallback_reasons"], {"no_ring": 1})
        self.assertEqual(summary["per_layer"], [{"layer": 3, "calls": 2, "read": 5, "reread": 1, "union": 4}])
        json.dumps(summary)
        summary["per_layer"].clear()
        self.assertEqual(len(prefill_reads.summary()["per_layer"]), 1, "summary() hands out a copy")

    def test_a_failed_prefill_leaves_the_previous_summary_and_clears_the_recording(self):
        os.environ[prefill_reads.FLAG] = "1"
        first = prefill_reads.begin(10)
        first.note("layer", FakeGlu(), [1], True)
        prefill_reads.end(first)
        second = prefill_reads.begin(20)
        second.note("layer", FakeGlu(), [1, 2], True)
        prefill_reads.end(second, ok=False)
        self.assertIsNone(prefill_reads.ACTIVE[0])
        self.assertEqual(prefill_reads.summary()["prompt_tokens"], 10)

    def test_a_prefill_that_read_nothing_leaves_none(self):
        os.environ[prefill_reads.FLAG] = "1"
        done = prefill_reads.begin(10)
        done.note("chunk", FakeGlu(), [1], True)
        prefill_reads.end(done)
        empty = prefill_reads.begin(10)
        prefill_reads.end(empty)
        self.assertIsNone(prefill_reads.summary())

    def test_order_is_the_most_specific_read_point_seen(self):
        os.environ[prefill_reads.FLAG] = "1"
        for notes, expected in ((("chunk",), "chunk"), (("chunk", "layer"), "layer"), (("layer", "layer-union"), "layer-union")):
            token = prefill_reads.begin(1)
            for order in notes:
                token.note(order, FakeGlu(), [1], True)
            prefill_reads.end(token)
            self.assertEqual(prefill_reads.summary()["order"], expected)

    def test_the_reason_comes_from_the_rings_own_counters(self):
        class Ring:
            def __init__(self):
                self.counters = {"fallback_capacity": 0, "fallback_busy": 0, "disabled": None}

            def stats(self):
                return dict(self.counters)

        ring = Ring()
        before = prefill_reads.mark(ring)
        ring.counters["fallback_capacity"] += 1
        os.environ[prefill_reads.FLAG] = "1"
        token = prefill_reads.begin(1)
        token.note("chunk", FakeGlu(), [1], False, ring=ring, before=before)
        prefill_reads.end(token)
        self.assertEqual(prefill_reads.summary()["fallback_reasons"], {"capacity": 1})


class StreamedReadsBase(UnionBase):
    ring_experts = 96

    def reads(self, order, union=False, window=256, prompt=None, **kwargs):
        prefill_reads.reset()
        self.addCleanup(prefill_reads.reset)
        result = self.prefill(order=order, union=union, window=window, prompt=prompt, reads=True, **kwargs)
        self.assertIsNone(prefill_reads.ACTIVE[0])
        return prefill_reads.summary(), result

    def invariants(self, summary):
        self.assertEqual(set(summary), SUMMARY_KEYS)
        json.dumps(summary)
        self.assertEqual(summary["layers"], LAYERS)
        self.assertEqual(sum(row["calls"] for row in summary["per_layer"]), summary["calls"])
        self.assertEqual(sum(row["read"] for row in summary["per_layer"]), summary["experts_read"])
        self.assertEqual(sum(row["reread"] for row in summary["per_layer"]), summary["experts_reread"])
        for row in summary["per_layer"]:
            self.assertEqual(row["read"] - row["reread"], row["union"], "every read is new or a re-read")
            self.assertLessEqual(row["union"], EXPERTS)
        self.assertEqual(summary["ring_staged_calls"] + summary["fallback_calls"], summary["calls"])
        self.assertEqual(sum(summary["fallback_reasons"].values()), summary["fallback_calls"])


class ChunkMajorReadTests(StreamedReadsBase):
    def test_a_multi_chunk_prompt_rereads_experts_and_uses_the_ring(self):
        summary, _ = self.reads("chunk", prompt=self.ids[:256])
        self.invariants(summary)
        self.assertEqual(summary["order"], "chunk")
        self.assertEqual(summary["prompt_tokens"], 256)
        self.assertEqual(summary["calls"], LAYERS * 4, "one read per (layer, 64-token chunk)")
        self.assertGreater(summary["experts_reread"], 0)
        self.assertGreater(summary["reread_share"], 0.5, "dense routing: most of every later chunk is a re-read")
        self.assertEqual(summary["ring_staged_calls"], summary["calls"])
        self.assertEqual(summary["fallback_reasons"], {})
        row_bytes = 7680
        self.assertEqual(summary["bytes_read"], summary["experts_read"] * row_bytes)

    def test_the_counts_equal_the_real_routes(self):
        sets = route_sets(self.model, lambda: self.prefill(order="chunk", prompt=self.ids[:256]))
        summary, _ = self.reads("chunk", prompt=self.ids[:256])
        for row in summary["per_layer"]:
            layer_sets = sets[row["layer"]]
            self.assertEqual(row["read"], sum(len(s) for s in layer_sets))
            self.assertEqual(row["union"], len(set().union(*layer_sets)))

    def test_without_a_ring_every_read_is_a_fallback_for_that_reason(self):
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_EXPERTS": "0"}):
            summary, _ = self.reads("chunk", prompt=self.ids[:256])
        self.assertEqual(summary["ring_staged_calls"], 0)
        self.assertEqual(summary["fallback_reasons"], {"no_ring": summary["calls"]})

    def test_a_ring_that_is_too_small_reports_capacity(self):
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_EXPERTS": "30"}):
            summary, _ = self.reads("chunk", prompt=self.ids[:256])
        self.assertEqual(summary["fallback_reasons"], {"capacity": summary["calls"]})
        self.assertGreater(summary["max_set"], 30)

    def test_the_flag_off_records_nothing_and_changes_no_result(self):
        want = self.prefill(order="chunk", prompt=self.ids[:256])
        prefill_reads.reset()
        again = self.prefill(order="chunk", prompt=self.ids[:256], reads=False)
        self.assertIsNone(prefill_reads.summary())
        self.assertEqual(again["token"], want["token"])


class LayerMajorReadTests(StreamedReadsBase):
    def test_the_all_experts_window_reads_every_expert_once_per_window(self):
        summary, _ = self.reads("layer", window=256, prompt=self.ids[:256])
        self.invariants(summary)
        self.assertEqual(summary["order"], "layer")
        self.assertEqual(summary["calls"], LAYERS)
        self.assertEqual(summary["experts_read"], LAYERS * EXPERTS)
        self.assertEqual(summary["experts_reread"], 0)
        two_windows, _ = self.reads("layer", window=128, prompt=self.ids[:256])
        self.assertEqual(two_windows["calls"], LAYERS * 2)
        self.assertEqual(two_windows["experts_reread"], LAYERS * EXPERTS, "the second window re-reads everything")

    def test_union_reads_each_expert_once_per_group_and_fewer_than_chunk_major(self):
        chunk, _ = self.reads("chunk", prompt=self.ids[:256])
        union, result = self.reads("layer", union=True, window=256, prompt=self.ids[:256])
        self.invariants(union)
        self.assertEqual(union["order"], "layer-union")
        self.assertEqual(union["calls"], LAYERS, "one group per layer: the whole layer fits the 96-expert half")
        self.assertEqual(union["experts_reread"], 0)
        for row in union["per_layer"]:
            self.assertEqual((row["calls"], row["read"], row["reread"]), (1, row["union"], 0))
        self.assertEqual(union["ring_staged_calls"], LAYERS)
        self.assertLess(union["experts_read"], chunk["experts_read"])
        self.assertLess(union["bytes_read"], chunk["bytes_read"])
        item = result["trace"].layers[0]["route_union"]
        self.assertEqual(item["union_experts"], union["experts_read"])
        self.assertEqual(item["chunk_set_experts"], chunk["experts_read"])
        self.assertEqual(item["expert_gb"], round(union["bytes_read"] / 1e9, 3))

    def test_union_with_a_small_ring_reports_one_call_per_group_and_the_capacity_fallback(self):
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_EXPERTS": "80"}):
            sets = route_sets(self.model, lambda: self.prefill(order="chunk", prompt=self.ids[:256]))
            summary, result = self.reads("layer", union=True, window=256, prompt=self.ids[:256])
        self.invariants(summary)
        for row in summary["per_layer"]:
            steps = layer_major.partition_groups(sets[row["layer"]], 80)
            self.assertEqual(row["calls"], len(steps), f"layer {row['layer']}")
            self.assertEqual(row["read"], sum(len(s.experts) for s in steps))
        oversize = sum(1 for layer in sets for step in layer_major.partition_groups(sets[layer], 80) if step.oversize)
        self.assertGreater(oversize, 0)
        self.assertEqual(summary["fallback_calls"], oversize)
        self.assertEqual(summary["fallback_reasons"], {"capacity": oversize})
        self.assertGreater(summary["max_set"], 80)
        self.assertGreater(summary["experts_reread"], 0, "groups of one layer overlap, so later groups re-read some experts")

    def test_union_without_a_ring_falls_back_for_that_reason(self):
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_EXPERTS": "0"}):
            summary, _ = self.reads("layer", union=True, window=256, prompt=self.ids[:256])
        self.assertEqual(summary["fallback_reasons"], {"no_ring": LAYERS})

    def test_a_cancelled_prefill_leaves_no_recording_behind(self):
        calls = []

        def observer(layer):
            calls.append(layer)
            if len(calls) == 5:
                raise RuntimeError("cancel")

        prefill_reads.reset()
        expert_cache.set_prefill_progress(observer)
        try:
            with self.assertRaisesRegex(RuntimeError, "cancel"):
                self.prefill(order="layer", union=True, window=256, prompt=self.ids[:256], reads=True)
        finally:
            expert_cache.set_prefill_progress(None)
        self.assertIsNone(prefill_reads.ACTIVE[0])
        self.assertIsNone(prefill_reads.summary())
        self.assert_restored()


if __name__ == "__main__":
    unittest.main()
