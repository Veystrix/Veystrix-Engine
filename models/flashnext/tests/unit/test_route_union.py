"""Route-union layer-major prefill (``FLASHNEXT_PREFILL_ROUTE_UNION=1``): exactness, grouping, safety, flags and launch wiring.

Checkpoint-free. A tiny Qwen4-Exp language model (all layer kinds, one PLE layer) has its MoE layers swapped for real
``StreamingSwitchGLU`` layers over a synthetic safetensors store (``streaming_tiny``), so the real read path, the real prefill
ring and the real expert pool run. The claim under test is the one the real checkpoint must still confirm with a digest gate:
the route-union order computes every (token, expert) product with the same ``gather_qmm`` on the same rows as chunk-major
order, so caches, hidden states, logits and the sampled token match byte for byte.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

from models.flashnext import expert_cache, layer_major, prefill, prefill_reads, prefill_ring
from models.flashnext.tests.bench.streaming_tiny import StreamingCase, build_stock_model, route_sets
from models.flashnext.tests.unit.test_prefix_cache import cache_bits

UNION_ENV = {layer_major.FLAG_ORDER: "layer", layer_major.FLAG_ROUTE_UNION: "1"}


def spans_of(sizes):
    out, position = [], 0
    for size in sizes:
        out.append((position, position + size))
        position += size
    return out


class FlagTests(unittest.TestCase):
    def test_default_is_off_and_identity_has_no_suffix(self):
        with mock.patch.dict(os.environ, {layer_major.FLAG_ORDER: "layer", layer_major.FLAG_WINDOW: "4096"}):
            os.environ.pop(layer_major.FLAG_ROUTE_UNION, None)
            self.assertFalse(layer_major.route_union_enabled())
            self.assertEqual(layer_major.identity(), f"{layer_major.PLAN_VERSION}/w4096")

    def test_union_adds_a_versioned_suffix_to_the_identity(self):
        with mock.patch.dict(os.environ, {**UNION_ENV, layer_major.FLAG_WINDOW: "1024", layer_major.FLAG_MIN_TOKENS: "256"}):
            self.assertTrue(layer_major.route_union_enabled())
            self.assertEqual(layer_major.identity(), f"{layer_major.PLAN_VERSION}/w1024/min256/union-v1")
        with mock.patch.dict(os.environ, {layer_major.FLAG_ORDER: "chunk", layer_major.FLAG_ROUTE_UNION: "1"}):
            self.assertEqual(layer_major.identity(), "", "chunk-major keeps its empty identity whatever the union flag says")

    def test_bad_value_raises(self):
        with mock.patch.dict(os.environ, {layer_major.FLAG_ROUTE_UNION: "yes"}):
            with self.assertRaises(ValueError):
                layer_major.route_union_enabled()

    def test_layer_prefetch_is_ignored_in_union_mode(self):
        window = layer_major.ExpertWindow(ring=None)
        glu = mock.Mock()
        with mock.patch.dict(os.environ, {**UNION_ENV, layer_major.FLAG_PREFETCH: "1"}):
            self.assertFalse(window.prefetch(glu))
            self.assertEqual(prefill_ring.ring_halves(), 1, "no second half is leased for a prefetch that never happens")
        with mock.patch.dict(os.environ, {layer_major.FLAG_ORDER: "layer", layer_major.FLAG_PREFETCH: "1",
                                         layer_major.FLAG_ROUTE_UNION: "0"}):
            self.assertEqual(prefill_ring.ring_halves(), 2, "without the union flag the existing behaviour is unchanged")


class SplitTests(unittest.TestCase):
    """``layer_first_half`` + ``mlp`` + ``layer_second_half`` is the layer's own ``__call__``, bit for bit."""

    @classmethod
    def setUpClass(cls):
        cls.model = build_stock_model()
        mx.random.seed(5)
        cls.ids = mx.array([[int(t) for t in mx.random.randint(2, 512, (96,)).tolist()]])

    def test_every_layer_kind_composes_bit_identically(self):
        from mlx_vlm.models.qwen4_exp import language as q4

        model, body = self.model, self.model.model
        reference_cache, split_cache = model.make_cache(), model.make_cache()
        for offset, chunk in ((0, self.ids[:, :64]), (64, self.ids[:, 64:96])):       # a first chunk and a continuation
            state = mx.tile(body.embed_tokens(chunk), (1, 1, body.args.hc_count))
            positions = layer_major._position_ids(model, chunk, offset, reference_cache)
            fa = q4._create_qwen3_5_attention_mask(state, reference_cache[body.fa_idx])
            ssm = q4._create_qwen3_5_ssm_mask(state, reference_cache[body.ssm_idx])
            kinds = set()
            ref_state, split_state = state, state
            for index, layer in enumerate(body.layers):
                mask = ssm if layer.is_linear else fa
                kinds.add(("linear" if layer.is_linear else "full", "ple" in layer))
                layer_major.check_layer(layer)
                ref_state = layer(ref_state, chunk, mask=mask, cache=reference_cache[index], position_ids=positions)
                mixed, hyper, weights = layer_major.layer_first_half(
                    layer, split_state, chunk, mask, split_cache[index], positions)
                split_state = layer_major.layer_second_half(layer.mlp(mixed), hyper, weights)
                mx.eval(ref_state, split_state, reference_cache[index].state, split_cache[index].state)
                self.assertTrue(bool(mx.array_equal(ref_state, split_state)), f"layer {index} differs")
            self.assertEqual(kinds, {("linear", False), ("linear", True), ("full", False)})
            self.assertTrue(cache_bits(split_cache) == cache_bits(reference_cache))

    def test_a_changed_or_patched_layer_is_refused(self):
        layer = self.model.model.layers[0]
        layer_major.check_layer(layer)
        with mock.patch("mlx_vlm.models.qwen4_exp.language.Qwen4ExpDecoderLayer.__call__", lambda self, *a, **k: a[0]):
            with self.assertRaises(layer_major.UnionRefused) as caught:
                layer_major.check_layer(layer)
        self.assertEqual(caught.exception.reason, "layer-code")
        with self.assertRaises(layer_major.UnionRefused) as caught:
            layer_major.check_layer(object())
        self.assertEqual(caught.exception.reason, "layer-type")
        broken = build_stock_model().model.layers[0]
        broken.attn_hyper_connection = None
        with self.assertRaises(layer_major.UnionRefused) as caught:
            layer_major.check_layer(broken)
        self.assertEqual(caught.exception.reason, "layer-attributes")


class PartitionTests(unittest.TestCase):
    def test_unbounded_capacity_is_one_group(self):
        steps = layer_major.partition_groups([{1, 2}, {2, 3}, {4}], None)
        self.assertEqual([(s.chunks, s.experts) for s in steps], [([0, 1, 2], [1, 2, 3, 4])])

    def test_greedy_consecutive_groups_whose_union_fits(self):
        steps = layer_major.partition_groups([{1, 2}, {2, 3}, {4, 5}, {5, 6}, {7}], 3)
        self.assertEqual([(s.chunks, s.experts) for s in steps], [([0, 1], [1, 2, 3]), ([2, 3], [4, 5, 6]), ([4], [7])])

    def test_a_chunk_larger_than_the_ring_is_its_own_oversize_step(self):
        steps = layer_major.partition_groups([{1}, {2, 3, 4, 5}, {6}], 3)
        self.assertEqual([(s.chunks, s.oversize) for s in steps], [([0], False), ([1], True), ([2], False)])

    def test_an_unknown_chunk_is_a_legacy_step_that_ends_the_group(self):
        steps = layer_major.partition_groups([{1}, {2}, None, {3}], None)
        self.assertEqual([(s.chunks, s.legacy) for s in steps], [([0, 1], False), ([2], True), ([3], False)])


class UnionBase(StreamingCase):
    """Helpers shared by the model-level tests: chunk-major reference, route-union and today's layer-major runs."""

    def chunk_major(self, ids, spans, offset=0, caches=None, ring=False):
        caches = caches if caches is not None else self.model.make_cache()
        outs = []

        def run():
            for a, b in spans:
                result = self.model(mx.array([ids[a:b]]), cache=caches, return_hidden=True, skip_logits=True)
                mx.eval(result.hidden_states[-1], [c.state for c in caches])
                outs.append(result.hidden_states[-1])

        if ring:
            with prefill_ring.session(self.model, 4096) as granted:
                self.assertIsNotNone(granted, prefill_ring.LAST_REASON[0])
                run()
        else:
            run()
        return caches, outs

    def window_run(self, ids, spans, union, caches=None, offset=0, freeze=(), ring=False, window=None):
        """``run_window`` over ``spans`` in union or today's layer-major order. Returns (caches, hidden outs, report, window)."""
        caches = caches if caches is not None else self.model.make_cache()
        env = {layer_major.FLAG_ORDER: "layer", layer_major.FLAG_ROUTE_UNION: "1" if union else "0"}
        with mock.patch.dict(os.environ, env):
            if ring:
                with prefill_ring.session(self.model, 4096) as granted:
                    self.assertIsNotNone(granted, prefill_ring.LAST_REASON[0])
                    window = layer_major.ExpertWindow(ring=granted)
                    states, report = layer_major.run_window(
                        self.model, mx.array([ids[spans[0][0]:spans[-1][1]]]), caches, spans, offset, window=window,
                        freeze_edges=freeze)
                    mx.eval(states)
                    self.assertEqual(granted.ring.stats()["states"], ["free"] * len(granted.ring.stats()["states"]))
            else:
                window = layer_major.ExpertWindow(ring=None)
                states, report = layer_major.run_window(
                    self.model, mx.array([ids[spans[0][0]:spans[-1][1]]]), caches, spans, offset, window=window,
                    freeze_edges=freeze)
        outs = [self.model.model.hyper_connection_mixer(s) for s in states]
        mx.eval(outs)
        return caches, outs, report, window

    def assert_same(self, got, want, label=""):
        got_caches, got_outs = got[0], got[1]
        want_caches, want_outs = want[0], want[1]
        self.assertTrue(cache_bits(got_caches) == cache_bits(want_caches), f"{label}: cache state differs")
        for g, w in zip(got_outs, want_outs):
            self.assertEqual(np.array(g.astype(mx.float32)).tobytes(), np.array(w.astype(mx.float32)).tobytes(),
                             f"{label}: hidden states differ")

    def prefill(self, order, union=False, window=256, prompt=None, offset=0, cache=None, boundaries=(), reads=False):
        env = {"FLASHNEXT_PREFILL_CHUNK_TOKENS": "64", "FLASHNEXT_PREFILL_CHUNK_POLICY": "grid",
               layer_major.FLAG_ORDER: order, layer_major.FLAG_ROUTE_UNION: "1" if union else "0",
               layer_major.FLAG_INNER_SNAPSHOTS: "0", layer_major.FLAG_MIN_TOKENS: "0",
               prefill_reads.FLAG: "1" if reads else "0"}
        prompt = prompt or self.ids[:700]
        cache = cache if cache is not None else self.model.make_cache()
        logits, trace = [], prefill.PlanTrace()
        prefill.set_logits_hook(lambda row: (mx.eval(row), logits.append(np.array(row.astype(mx.float32)).tobytes())))
        self.addCleanup(prefill.set_logits_hook, None)
        with mock.patch.dict(os.environ, env), mock.patch.object(layer_major, "window_tokens", return_value=window):
            _, _, token = prefill.prefill_target(
                self.model, mx.array([prompt[offset:]]), cache, sampler=lambda row: mx.argmax(row, axis=-1),
                boundaries=tuple(boundaries), offset=offset, trace=trace)
        prefill.set_logits_hook(None)
        return {"token": int(token.item()), "logits": logits, "cache": cache, "trace": trace}


class UnionEquivalenceNoRing(UnionBase):
    ring_experts = 0
    use_ring = False

    def test_equal_chunks(self):
        spans = spans_of([64, 64, 64, 64])
        want = self.chunk_major(self.ids, spans, ring=self.use_ring)
        got = self.window_run(self.ids, spans, union=True, ring=self.use_ring)
        self.assert_same(got, want)
        report = got[2]
        self.assertEqual(report.union_layers, 8)
        self.assertEqual(report.union_refused, {})
        self.assertGreater(sum(row[2] for row in report.union), sum(row[1] for row in report.union),
                           "chunk-major would have read more experts than the union")
        self.assert_restored()

    def test_uneven_chunks_with_a_tiny_chunk_inside_the_window(self):
        spans = spans_of([40, 5, 90, 64, 8, 30])      # 5 and 8 rows take the decode-style MoE path: legacy steps
        want = self.chunk_major(self.ids, spans, ring=self.use_ring)
        got = self.window_run(self.ids, spans, union=True, ring=self.use_ring)
        self.assert_same(got, want)
        self.assertEqual(got[2].union_layers, 8)
        self.assert_restored()

    def test_single_chunk_window(self):
        spans = spans_of([192])
        self.assert_same(self.window_run(self.ids, spans, union=True, ring=self.use_ring),
                         self.chunk_major(self.ids, spans, ring=self.use_ring))

    def test_resuming_from_a_restored_prefix(self):
        absolute = [(100 + a, 100 + b) for a, b in spans_of([64, 64, 64])]
        want_caches = self.model.make_cache()
        self.chunk_major(self.ids, [(0, 100)], caches=want_caches)
        want = self.chunk_major(self.ids, absolute, caches=want_caches, ring=self.use_ring)
        got_caches = self.model.make_cache()
        self.chunk_major(self.ids, [(0, 100)], caches=got_caches)
        got = self.window_run(self.ids[100:], spans_of([64, 64, 64]), union=True, caches=got_caches, offset=100,
                              ring=self.use_ring)
        self.assert_same(got, want)

    def test_prefill_target_matches_chunk_major_token_cache_and_logits(self):
        want = self.prefill(order="chunk")
        got = self.prefill(order="layer", union=True, window=256)
        self.assertEqual(got["token"], want["token"])
        self.assertEqual(got["logits"], want["logits"])
        self.assertTrue(cache_bits(got["cache"]) == cache_bits(want["cache"]))
        self.assertTrue(got["trace"].order.endswith("/union-v1"))
        self.assertGreater(len(got["trace"].windows), 1)
        item = got["trace"].layers[0]
        self.assertEqual(item["execution"], "layer")
        self.assertEqual(item["route_union"]["layers"], 8)
        self.assertEqual(item["route_union"]["columns"][:3], ["layer", "union", "chunk_sets"])
        self.assertNotIn("route_union", want["trace"].layers[0] if want["trace"].layers else {})
        self.assert_restored()

    def test_tail_chunk_after_the_window_stays_exact(self):
        prompt = self.ids[:456]                                  # seven 64-token chunks and a tail of 8 tokens
        edges = (192, 384, 448)                                  # the 448 boundary keeps the 8-token tail a chunk of its own
        want = self.prefill(order="chunk", prompt=prompt, boundaries=edges)
        got = self.prefill(order="layer", union=True, window=512, prompt=prompt, boundaries=edges)
        self.assertEqual(got["token"], want["token"])
        self.assertEqual(got["logits"], want["logits"])
        self.assertTrue(cache_bits(got["cache"]) == cache_bits(want["cache"]))
        self.assertEqual([(row["start"], row["end"], row["execution"]) for row in got["trace"].layers],
                         [(0, 448, "layer"), (448, 456, "chunk")])

    def test_prefill_target_from_a_nonzero_offset(self):
        want_cache = self.prefill(order="chunk", prompt=self.ids[:100])["cache"]
        got_cache = self.prefill(order="chunk", prompt=self.ids[:100])["cache"]
        want = self.prefill(order="chunk", prompt=self.ids[:420], offset=100, cache=want_cache)
        got = self.prefill(order="layer", union=True, window=256, prompt=self.ids[:420], offset=100, cache=got_cache)
        self.assertEqual(got["token"], want["token"])
        self.assertEqual(got["logits"], want["logits"])
        self.assertTrue(cache_bits(got["cache"]) == cache_bits(want["cache"]))


class UnionEquivalenceWithRing(UnionEquivalenceNoRing):
    ring_experts = 96                   # the whole 96-expert layer fits the half: one group per layer
    use_ring = True

    def test_union_stages_through_the_ring_in_one_group_per_layer(self):
        spans = spans_of([64, 64, 64, 64])
        got = self.window_run(self.ids, spans, union=True, ring=True)
        report = got[2]
        self.assertEqual([row[3] for row in report.union], [1] * 8)
        self.assertEqual([row[4] for row in report.union], [1] * 8)
        self.assertEqual(report.ring_layers, 8)
        self.assertEqual(report.fallback_layers, 0)
        self.assertLess(report.expert_bytes, 8 * 96 * 7680, "the union reads fewer bytes than all experts of every layer")
        self.assert_restored()


class UnionGroupsTests(UnionBase):
    ring_experts = 80

    def test_the_fixture_routes_densely_whatever_global_patches_other_tests_left(self):
        """Guards the capacity tests below: a process-wide RMSNorm patch once collapsed routing to 2 experts per chunk."""
        sets = route_sets(self.model, lambda: self.chunk_major(self.ids, spans_of([64, 64]), ring=True))
        self.assertGreater(min(len(s) for layer in sets.values() for s in layer), 40)

    def test_small_ring_needs_several_groups_and_an_oversize_chunk_falls_back(self):
        spans = spans_of([64, 64, 64, 64])
        reference = []
        sets = route_sets(self.model, lambda: reference.append(self.chunk_major(self.ids, spans, ring=True)))
        want = reference[0]
        caches = self.model.make_cache()
        with mock.patch.dict(os.environ, UNION_ENV), prefill_ring.session(self.model, 4096) as ring:
            self.assertEqual(ring.ring.half_experts, 80)
            window = layer_major.ExpertWindow(ring=ring)
            states, report = layer_major.run_window(self.model, mx.array([self.ids[:256]]), caches, spans, 0, window=window)
            outs = [self.model.model.hyper_connection_mixer(s) for s in states]
            mx.eval(outs)
            ring_stats = ring.stats()
            self.assertEqual(ring.ring.stats()["states"], ["free"])
        self.assert_same((caches, outs), want)
        expected_groups = expected_oversize = expected_ring = 0
        for row in report.union:
            layer = row[0]
            steps = layer_major.partition_groups(sets[layer], 80)
            self.assertEqual(row[3], len(steps), f"layer {layer} groups")
            oversize = sum(1 for step in steps if step.oversize)
            self.assertEqual(row[5], oversize, f"layer {layer} fallback groups")
            self.assertEqual(row[4], len(steps) - oversize, f"layer {layer} ring groups")
            self.assertEqual(row[1], len(set().union(*sets[layer])))
            self.assertEqual(row[2], sum(len(s) for s in sets[layer]))
            expected_groups += len(steps)
            expected_oversize += oversize
            expected_ring += len(steps) - oversize
        self.assertGreater(expected_groups, 8, "the ring was meant to force several groups for some layer")
        self.assertGreater(expected_oversize, 0, "the ring was meant to leave a chunk that alone exceeds it")
        self.assertEqual(ring_stats["staged_layers"], expected_ring)
        self.assertEqual(ring_stats["fallback_capacity"], expected_oversize)
        self.assertEqual(report.ring_layers, expected_ring)
        self.assertEqual(report.fallback_layers, expected_oversize)
        self.assertEqual(window.stats["layers"], expected_groups)
        self.assert_restored()

    def test_tiny_ring_sends_every_chunk_to_the_read_path_and_stays_exact(self):
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_EXPERTS": "20"}):
            spans = spans_of([64, 64, 64])
            want = self.chunk_major(self.ids, spans, ring=True)
            got = self.window_run(self.ids, spans, union=True, ring=True)
        self.assert_same(got, want)
        self.assertEqual(got[2].ring_layers, 0)
        self.assertEqual(got[2].fallback_layers, 8 * 3)
        self.assert_restored()


class UnionFreezeTests(UnionBase):
    ring_experts = 96

    def test_inner_snapshot_freeze_edges_equal_todays_layer_major_and_chunk_major(self):
        spans = spans_of([64, 64, 64, 64])
        edges = (128, 192)
        for ring in (False, True):
            union = self.window_run(self.ids, spans, union=True, freeze=edges, ring=ring)
            today = self.window_run(self.ids, spans, union=False, freeze=edges, ring=ring)
            self.assertEqual(sorted(union[2].frozen), sorted(edges))
            self.assertEqual(sorted(union[2].frozen), sorted(today[2].frozen))
            for edge in edges:
                self.assertTrue(cache_bits(union[2].frozen[edge]) == cache_bits(today[2].frozen[edge]), f"edge {edge}")
            # and what chunk-major holds at those edges
            caches = self.model.make_cache()
            for a, b in spans:
                self.model(mx.array([self.ids[a:b]]), cache=caches, return_hidden=True, skip_logits=True)
                mx.eval([c.state for c in caches])
                if b in edges:
                    self.assertTrue(cache_bits(union[2].frozen[b]) == cache_bits(caches), f"edge {b} vs chunk-major")
            self.assert_same(union, today)

    def test_a_frozen_edge_survives_later_chunks_updating_the_live_cache(self):
        spans = spans_of([64, 64, 64, 64])
        union = self.window_run(self.ids, spans, union=True, freeze=(64,))
        self.assertFalse(cache_bits(union[2].frozen[64]) == cache_bits(union[0]))


class UnionFailureTests(UnionBase):
    ring_experts = 96

    def run_failing(self, fail_on, error, spans=None):
        """Run a union window whose progress observer raises on its ``fail_on``-th call; returns the observer's log and a rerun."""
        spans = spans or spans_of([64, 64, 64])
        log = []

        def observer(layer):
            log.append((layer, ring_states()))
            if len(log) == fail_on:
                raise error

        with prefill_ring.session(self.model, 4096) as ring:
            window = layer_major.ExpertWindow(ring=ring)

            def ring_states():
                return list(ring.ring.stats()["states"])

            caches = self.model.make_cache()
            expert_cache.set_prefill_progress(observer)
            try:
                with mock.patch.dict(os.environ, UNION_ENV):
                    with self.assertRaises(type(error)):
                        layer_major.run_window(self.model, mx.array([self.ids[:192]]), caches, spans, 0, window=window)
            finally:
                expert_cache.set_prefill_progress(None)
            self.assertEqual(ring_states(), ["free"], "a failed window must leave no half in use")
            self.assertIsNone(window.route_union)
            self.assertIsNone(window._weights)
            self.assertIsNone(expert_cache._LAYER_WINDOW[0])
            self.assertIsNone(ring.pending)
        # a new ring session on the same pool: the same prefill now succeeds
        again = self.window_run(self.ids, spans, union=True, ring=True)
        return log, again

    def test_failure_during_phase_a_leaves_the_ring_idle_and_the_next_prefill_works(self):
        spans = spans_of([64, 64, 64])
        log, again = self.run_failing(2, RuntimeError("cancelled in phase A"))
        self.assertEqual(len(log), 2)
        self.assertTrue(all(states == ["free"] for _, states in log), "phase A never touches the ring")
        self.assert_same(again, self.chunk_major(self.ids, spans))
        self.assert_restored()

    def test_failure_during_phase_b_after_staging_fences_the_half(self):
        spans = spans_of([64, 64, 64])
        log, again = self.run_failing(4, RuntimeError("cancelled in phase B"))        # 3 probes, then the first run call
        self.assertEqual([states for _, states in log][:3], [["free"]] * 3)
        self.assertEqual(log[3][1], ["computing"], "the half was staged and in use when the cancel arrived")
        self.assert_same(again, self.chunk_major(self.ids, spans))
        self.assert_restored()

    def test_keyboard_interrupt_in_the_middle_of_a_group_is_fenced_too(self):
        log, _ = self.run_failing(5, KeyboardInterrupt())                              # second chunk of the first group
        self.assertEqual(log[4][1], ["computing"])
        self.assert_restored()

    def test_prefill_target_cancelled_in_each_phase_restores_the_ring_and_recovers(self):
        want = self.prefill(order="chunk")
        for fail_on in (3, 40):                 # call 3: a probe of the first layer; call 40: a run call of a later layer
            calls = []

            def observer(layer):
                calls.append(layer)
                if len(calls) == fail_on:
                    raise RuntimeError("cancel")

            expert_cache.set_prefill_progress(observer)
            try:
                with self.assertRaisesRegex(RuntimeError, "cancel"):
                    self.prefill(order="layer", union=True, window=256)
            finally:
                expert_cache.set_prefill_progress(None)
            self.assert_restored()
            got = self.prefill(order="layer", union=True, window=256)
            self.assertEqual(got["token"], want["token"])
            self.assertTrue(cache_bits(got["cache"]) == cache_bits(want["cache"]))
            self.assert_restored()


class FlagOffTests(UnionBase):
    ring_experts = 0

    def test_without_the_flag_the_window_stages_every_expert_and_no_union_object_exists(self):
        spans = spans_of([64, 64, 64])
        with mock.patch.object(layer_major, "RouteUnion", side_effect=AssertionError("union must not run")), \
                mock.patch.object(layer_major.ExpertWindow, "begin_union", side_effect=AssertionError("union must not run")):
            got = self.window_run(self.ids, spans, union=False)
        window, report = got[3], got[2]
        self.assertEqual(window.stats["layers"], 8, "one read per layer for the whole window")
        row_bytes = prefill_reads.expert_bytes(self.model.model.layers[0].mlp.switch_mlp)
        self.assertEqual(window.stats["bytes"], 8 * 96 * row_bytes, "all 96 experts of every layer")
        self.assertEqual((report.union_layers, report.union, report.union_refused), (0, [], {}))
        self.assertIsNone(window.route_union)
        self.assert_same(got, self.chunk_major(self.ids, spans))
        with mock.patch.dict(os.environ, {layer_major.FLAG_ORDER: "layer", layer_major.FLAG_ROUTE_UNION: "0"}):
            self.assertNotIn("union", layer_major.identity())

    def test_a_non_streamed_model_refuses_every_layer_and_runs_the_existing_path(self):
        stock = build_stock_model()
        ids = mx.array([self.ids[:128]])
        spans = spans_of([64, 64])
        with mock.patch.dict(os.environ, UNION_ENV):
            window = layer_major.ExpertWindow(ring=None)
            states, report = layer_major.run_window(stock, ids, stock.make_cache(), spans, 0, window=window)
        self.assertEqual(report.union_layers, 0)
        self.assertEqual(report.union_refused, {"not-streamed": 8})
        reference = stock.make_cache()
        with mock.patch.dict(os.environ, {layer_major.FLAG_ROUTE_UNION: "0"}):
            expected, _ = layer_major.run_window(stock, ids, reference, spans, 0, window=layer_major.ExpertWindow(ring=None))
        for g, w in zip(states, expected):
            self.assertTrue(bool(mx.array_equal(g, w)))

    def test_a_patched_layer_class_makes_the_window_fall_back_and_count_it(self):
        spans = spans_of([64, 64])
        original = type(self.model.model.layers[0]).__call__
        calls = []

        def passthrough(layer, *args, **kwargs):
            calls.append(1)
            return original(layer, *args, **kwargs)

        with mock.patch.dict(os.environ, UNION_ENV), \
                mock.patch("mlx_vlm.models.qwen4_exp.language.Qwen4ExpDecoderLayer.__call__", passthrough):
            window = layer_major.ExpertWindow(ring=None)
            states, report = layer_major.run_window(self.model, mx.array([self.ids[:128]]), self.model.make_cache(), spans, 0,
                                                    window=window)
        self.assertEqual(report.union_refused, {"layer-code": 8})
        self.assertEqual(len(calls), 8 * 2, "the existing layer-major loop ran every (layer, chunk)")


class UnionRobustnessTests(UnionBase):
    ring_experts = 96

    def test_compile_glue_wrapper_is_accepted_and_one_token_chunks_stay_exact(self):
        """Every real load installs compile_glue's decoder wrapper; the union must still run, and a 1-token chunk takes it."""
        from models.flashnext import compile_glue

        original = type(self.model.model.layers[0]).__call__
        spans = spans_of([64, 1, 64, 6, 40])
        with mock.patch("mlx_vlm.models.qwen4_exp.language.Qwen4ExpDecoderLayer.__call__", compile_glue._decoder_call), \
                mock.patch.dict(compile_glue._ORIGINAL, {"decoder": original}), \
                mock.patch.object(compile_glue, "ENABLED", [True]):
            layer_major.check_layer(self.model.model.layers[0])
            want = self.chunk_major(self.ids, spans, ring=True)
            got = self.window_run(self.ids, spans, union=True, ring=True)
        self.assert_same(got, want)
        self.assertEqual(got[2].union_layers, 8)
        self.assertEqual(got[2].union_refused, {})

    def test_a_block_that_does_not_call_the_experts_once_falls_back_per_chunk_and_is_counted(self):
        from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

        original = Qwen3_5MoeSparseMoeBlock.__call__
        spans = spans_of([64, 64, 64])

        def twice(block, x):
            return original(block, x) + original(block, x)

        with mock.patch.object(Qwen3_5MoeSparseMoeBlock, "__call__", twice):
            want = self.chunk_major(self.ids, spans, ring=True)
            got = self.window_run(self.ids, spans, union=True, ring=True)
        self.assert_same(got, want)
        self.assertEqual(got[2].union_refused, {"probe-count": 8 * 3})
        self.assertEqual(got[2].union, [[layer, 0, 0, 0, 0, 0, 0] for layer in range(8)])
        self.assert_restored()

    def test_decode_seeding_hook_gets_the_staged_positions_and_the_groups_routes(self):
        """FLASHNEXT_PREFILL_SEED_DECODE: the ring is told each staged expert's position and the group's routed ids."""
        spans = spans_of([64, 64, 64])
        want = self.chunk_major(self.ids, spans, ring=True)
        calls = []
        real = prefill_ring.PrefillRing.seed_decode

        def spy(ring, layer, positions, routed):
            calls.append((layer, dict(positions), list(routed)))
            return real(ring, layer, positions, routed)

        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_SEED_DECODE": "1"}), \
                mock.patch.object(prefill_ring.PrefillRing, "seed_decode", spy):
            got = self.window_run(self.ids, spans, union=True, ring=True)
        self.assert_same(got, want)
        self.assertEqual([layer for layer, _, _ in calls], list(range(8)), "one seeding per staged group")
        for layer, positions, routed in calls:
            self.assertEqual(sorted(positions.values()), list(range(len(positions))))
            self.assertEqual(set(routed), set(positions), "every staged expert was routed by some chunk of the group")
            self.assertEqual(len(routed), 3 * 64 * 2)

    def test_a_staged_operand_that_misses_a_routed_expert_is_never_used_silently(self):
        spans = spans_of([64, 64])
        real_stage = layer_major.RouteUnion.stage

        def short_stage(union, wanted, hosts):
            real_stage(union, wanted[:-1], hosts)             # drop one routed expert from the operand

        with mock.patch.object(layer_major.RouteUnion, "stage", short_stage):
            with self.assertRaisesRegex(RuntimeError, "outside its staged experts"):
                self.window_run(self.ids, spans, union=True, ring=True)
        self.assert_restored()


class ProbeNeutralityTests(UnionBase):
    """The probe runs the real MoE block once more per chunk; nothing the block records may be counted twice."""

    ring_experts = 96

    def test_adaptive_topk_counters_and_observer_see_each_chunk_once(self):
        from models.flashnext import adaptive_topk
        from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

        spans = spans_of([64, 64, 64, 64])

        def observe(case):
            seen = []
            adaptive_topk.reset_keep_stats()
            adaptive_topk.set_route_observer(lambda layer, rows, scores, keeps: seen.append((layer, len(rows))))
            try:
                with mock.patch.object(Qwen3_5MoeSparseMoeBlock, "__call__", adaptive_topk._moe_call):
                    result = case()
            finally:
                adaptive_topk.set_route_observer(None)
            return seen, adaptive_topk._KEEP_COUNT[0], adaptive_topk._KEEP_SUM[0], result

        chunk_seen, chunk_count, chunk_sum, want = observe(lambda: self.chunk_major(self.ids, spans, ring=True))
        union_seen, union_count, union_sum, got = observe(lambda: self.window_run(self.ids, spans, union=True, ring=True))
        self.assertEqual(sorted(union_seen), sorted(chunk_seen), "the observer saw every (layer, chunk) exactly once")
        self.assertEqual((union_count, union_sum), (chunk_count, chunk_sum))
        self.assertGreater(chunk_count, 0)
        self.assert_same(got, want)
        self.assertIsNone(adaptive_topk._ROUTE_OBSERVER[0])

    def test_probe_restores_overlap_and_observer_even_when_it_raises(self):
        from models.flashnext import adaptive_topk

        marker = object()
        previous = (adaptive_topk._OVERLAP[0], adaptive_topk._ROUTE_OBSERVER[0])
        adaptive_topk._ROUTE_OBSERVER[0] = marker
        try:
            with self.assertRaises(RuntimeError):
                with layer_major._probe_neutral():
                    self.assertFalse(adaptive_topk._OVERLAP[0])
                    self.assertIsNone(adaptive_topk._ROUTE_OBSERVER[0])
                    raise RuntimeError("boom")
            self.assertIs(adaptive_topk._ROUTE_OBSERVER[0], marker)
            self.assertEqual(adaptive_topk._OVERLAP[0], previous[0])
        finally:
            adaptive_topk._ROUTE_OBSERVER[0], adaptive_topk._OVERLAP[0] = previous[1], previous[0]

    def test_route_trace_sees_each_chunk_once_with_the_chunk_major_routes(self):
        spans = spans_of([64, 64, 64, 64])
        want = route_sets(self.model, lambda: self.chunk_major(self.ids, spans, ring=True))
        got = route_sets(self.model, lambda: self.window_run(self.ids, spans, union=True, ring=True))
        self.assertEqual({k: [sorted(s) for s in v] for k, v in got.items()},
                         {k: [sorted(s) for s in v] for k, v in want.items()})
        self.assertEqual({len(v) for v in got.values()}, {4})

    def test_the_probe_reads_nothing(self):
        """Per layer exactly one staging (the whole window fits the 96-expert half): the probe added no read."""
        spans = spans_of([64, 64, 64, 64])
        got = self.window_run(self.ids, spans, union=True, ring=True)
        self.assertEqual(got[3].stats["layers"], 8)
        self.assertEqual(got[2].ring_layers + got[2].fallback_layers, 8)


class ModeTests(unittest.TestCase):
    def test_the_union_profile_extends_the_16gb_profile(self):
        from models.flashnext.prefill_16gb import PREFILL_16GB, PREFILL_16GB_UNION

        self.assertEqual(PREFILL_16GB["FLASHNEXT_PREFILL_ORDER"], "chunk", "the existing profile is unchanged")
        self.assertNotIn("FLASHNEXT_PREFILL_ROUTE_UNION", PREFILL_16GB)
        for key, value in {"FLASHNEXT_PREFILL_ORDER": "layer", "FLASHNEXT_PREFILL_ROUTE_UNION": "1",
                           "FLASHNEXT_LAYER_MAJOR_MIN_TOKENS": "256", "FLASHNEXT_PREFILL_WINDOW_TOKENS": "1024",
                           "FLASHNEXT_PREFILL_RING_EXPERTS": "384"}.items():
            self.assertEqual(PREFILL_16GB_UNION[key], value, key)
        for key in PREFILL_16GB:
            if key not in ("FLASHNEXT_PREFILL_ORDER", "FLASHNEXT_PREFILL_RING_EXPERTS"):
                self.assertEqual(PREFILL_16GB_UNION[key], PREFILL_16GB[key], key)

    def test_the_mode_is_registered_labelled_and_owned(self):
        from models.flashnext.settings import modes

        flags, label = modes.EXPERIMENTAL["prefill-16gb-union"]
        self.assertIn("unmeasured", label)
        self.assertIn("experimental", label)
        self.assertLessEqual(set(flags), modes.OWNED)
        self.assertIn(layer_major.FLAG_ROUTE_UNION, modes.OWNED)
        plan = modes.plan_mode(modes.AGENTIC, enable=("prefill-16gb-union",), free_disk_gb=200)
        for key, value in flags.items():
            self.assertEqual(plan.env[key], value, key)
        self.assertIn("prefill-16gb-union", modes.banner(plan))
        applied = modes.apply_mode({"FLASHNEXT_PREFILL_ROUTE_UNION": "1", "FLASHNEXT_PREFILL_ORDER": "layer"},
                                   modes.plan_mode(modes.AGENTIC, free_disk_gb=200))
        self.assertNotIn("FLASHNEXT_PREFILL_ROUTE_UNION", applied, "an inherited union flag must not leak into a plain mode")
        self.assertEqual(applied["FLASHNEXT_PREFILL_ORDER"], "chunk")

    def test_default_modes_do_not_enable_the_union(self):
        from models.flashnext.settings import modes

        for mode in (modes.CHAT, modes.AGENTIC):
            self.assertNotIn(layer_major.FLAG_ROUTE_UNION, modes.plan_mode(mode, free_disk_gb=200).env)

    def test_union_mode_is_a_layer_major_mode_with_its_own_conflicts(self):
        from models.flashnext.settings import modes

        self.assertIn("prefill-16gb-union", modes.LAYER_MAJOR_MODES)
        with self.assertRaises(modes.ModeError):
            modes.plan_mode(modes.AGENTIC, enable=("prefill-16gb-union", "layer-major-4k"), free_disk_gb=200)
        with self.assertRaises(modes.ModeError):
            modes.plan_mode(modes.AGENTIC, enable=("prefill-16gb-union", "layer-prefetch"), free_disk_gb=200)
        with self.assertRaises(modes.ModeError):
            modes.plan_mode(modes.AGENTIC, enable=("prefill-16gb-union", "prefill-16gb"), free_disk_gb=200)
        with self.assertRaises(modes.ModeError):
            modes.plan_mode(modes.CHAT, enable=("prefill-16gb-union",))
        modes.plan_mode(modes.CHAT, enable=("prefill-16gb",))           # the existing profile is still allowed in chat


if __name__ == "__main__":
    unittest.main()
