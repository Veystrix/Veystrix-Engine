#!/usr/bin/env python3
"""Small exactness test for Qwen4 block verification and cache rollback."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
from mlx_vlm.models.qwen3_5 import language as base_language
from mlx_vlm.models.qwen4_exp.config import ModelConfig, VisionConfig
from mlx_vlm.models.qwen4_exp.language import LanguageModel, TextConfig

from models.flashnext.adaptive_topk import (
    apply as patch_topk,
    set_renorm_blend,
    set_threshold,
)
from models.flashnext.patch_rmsnorm import apply as patch_norm
from models.flashnext.prefill import prefill_target
from models.flashnext.qwen4_verifier import Qwen4ExactSpeculativeVerifier
from models.flashnext.speculative import (
    FastDraftGreedy,
    snapshot_cache,
)


def tiny_language() -> LanguageModel:
    text = TextConfig(
        model_type="qwen4_exp",
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=32,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        hc_count=2,
        hc_lowrank=8,
        head_dim=16,
        layer_types=["linear_attention", "full_attention"],
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ple_conv_kernel_size=2,
        ngram_size=2,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=1,
        split_ngram_parts=1,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
        eos_token_id=2,
    )
    config = ModelConfig(
        text_config=text,
        vision_config=VisionConfig(),
        model_type="qwen4_exp",
        vocab_size=64,
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    model = LanguageModel(text, config)
    model.eval()
    return model


class SpeculativeTinyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        patch_norm()
        patch_topk()
        cls.language = tiny_language()
        base_language._EXACT_SPECULATIVE_VERIFIER = Qwen4ExactSpeculativeVerifier()
        set_threshold(1.0)
        set_renorm_blend(1.0)

    def prefill(self):
        cache = self.language.make_cache()
        prompt = mx.array([[3, 7, 11]], dtype=mx.uint32)
        mx.eval(self.language(prompt, cache=cache).logits)
        return cache

    def test_prompt_lookup_keeps_greedy_tokens_and_continuation(self):
        from unittest import mock
        from models.flashnext.speculative import prompt_lookup_tokens
        from models.shared.prompt_lookup import ThroughputGate

        first = mx.array([5], dtype=mx.uint32)
        expected = [5]
        reference = self.prefill()
        for _ in range(8):
            logits = self.language(mx.array([[expected[-1]]], dtype=mx.uint32), cache=reference).logits
            expected.append(int(mx.argmax(logits[0, -1]).item()))
        for draft in (expected[1:5], [(expected[1] + 1) % 64, *expected[2:5]]):
            candidate = self.prefill()
            history = [59, 5, *draft, 60, 59]
            with mock.patch.object(ThroughputGate, "allow", return_value=True):
                actual = list(prompt_lookup_tokens(
                    self.language, candidate, history, first, 8, set(), lambda: None,
                ))
            self.assertEqual(actual, expected[:8])
            follow = mx.array([[expected[8]]], dtype=mx.uint32)
            a = self.language(follow, cache=candidate).logits
            b = self.language(follow, cache=reference).logits
            self.assertTrue(mx.array_equal(a, b).item())
            # The reference must be at the same offset for the second branch.
            reference = self.prefill()
            for token in expected[:8]:
                mx.eval(self.language(mx.array([[token]], dtype=mx.uint32), cache=reference).logits)

    def test_prompt_lookup_depth_and_acceptance_cooldown(self):
        from models.shared.prompt_lookup import ThroughputGate

        gate = ThroughputGate()
        gate.observe(False, 1, 1)
        for depth in (2, 3):
            self.assertEqual(gate.depth, depth)
            gate.observe(True, depth + 1, 1, proposed=depth)
        self.assertEqual(gate.depth, 4)

        gate = ThroughputGate()
        gate.observe(False, 1, 1)
        gate.observe(True, 1, 0.25, proposed=2)
        self.assertEqual(gate.depth, 1)
        gate.observe(True, 1, 0.25, proposed=1)
        self.assertFalse(gate.allow())
        self.assertEqual(gate.cooldown, 15)

    def test_block_and_rollback_match_single_tokens(self):
        values = [5, 9, 13, 17, 21]
        reference_cache = self.prefill()
        reference_logits = []
        for value in values:
            token = mx.array([[value]], dtype=mx.uint32)
            output = self.language(token, cache=reference_cache)
            mx.eval(output.logits)
            reference_logits.append(output.logits)
        reference_logits = mx.concatenate(reference_logits, axis=1)

        verify_cache = self.prefill()
        snapshot = snapshot_cache(verify_cache)
        output = self.language(
            mx.array([values], dtype=mx.uint32),
            cache=verify_cache,
            speculative_verify=True,
            return_hidden=True,
            skip_logits=True,
        )
        block_logits = self.language.speculative_logits_from_hidden(
            output.hidden_states[-1]
        )
        mx.eval(block_logits)
        self.assertEqual(
            float(mx.max(mx.abs(reference_logits - block_logits)).item()),
            0.0,
        )

        for keep_count in (1, 2):
            verify_cache = self.prefill()
            snapshot = snapshot_cache(verify_cache)
            output = self.language(
                mx.array([values], dtype=mx.uint32),
                cache=verify_cache,
                speculative_verify=True,
                return_hidden=True,
                skip_logits=True,
            )
            decoder = FastDraftGreedy(self.language, object(), depth=2)
            decoder.target_cache = verify_cache
            decoder._rollback_verified(
                snapshot, output, len(values), keep_count
            )
            probe = mx.array([[29]], dtype=mx.uint32)
            actual = self.language(probe, cache=verify_cache).logits

            expected_cache = self.prefill()
            for value in values[:keep_count]:
                token = mx.array([[value]], dtype=mx.uint32)
                mx.eval(self.language(token, cache=expected_cache).logits)
            expected = self.language(probe, cache=expected_cache).logits
            mx.eval(actual, expected)
            self.assertEqual(
                float(mx.max(mx.abs(expected - actual)).item()), 0.0
            )

    def assert_external_draft_exact(
        self,
        fallback_on_reject,
        release_draft_before_verify=False,
        draft_min_margin=None,
        draft_min_block=1,
        draft_margin_tokens=None,
    ):
        target = tiny_language()
        draft = tiny_language()
        prompt = mx.array([[3, 7, 11]], dtype=mx.uint32)
        cache = target.make_cache()
        output = target(prompt, cache=cache)
        token = mx.argmax(output.logits[:, -1, :], axis=-1).astype(mx.uint32)
        expected = []
        for _ in range(16):
            expected.append(int(token.item()))
            output = target(token.reshape(1, 1), cache=cache)
            token = mx.argmax(
                output.logits[:, -1, :], axis=-1
            ).astype(mx.uint32)
            mx.eval(token)

        target._position_ids = None
        target._rope_deltas = None
        decoder = FastDraftGreedy(
            target,
            SimpleNamespace(_read_mode="pread"),
            depth=2,
            draft_language=draft,
            fallback_on_reject=fallback_on_reject,
            release_draft_before_verify=release_draft_before_verify,
            draft_min_margin=draft_min_margin,
            draft_min_block=draft_min_block,
            draft_margin_tokens=draft_margin_tokens,
        )
        decoder.append(prompt)
        actual = list(decoder.generate(16, set()))
        self.assertEqual(actual, expected)
        return decoder

    def test_external_draft_anchor_preserves_greedy_tokens(self):
        self.assert_external_draft_exact(False)

    def test_external_draft_fallback_preserves_greedy_tokens(self):
        self.assert_external_draft_exact(True)

    def test_transient_external_draft_preserves_greedy_tokens(self):
        decoder = self.assert_external_draft_exact(False, True)
        self.assertTrue(decoder.draft_disabled)
        self.assertIsNone(decoder.draft_language)
        self.assertIsNone(decoder.draft_cache)
        self.assertIsNone(decoder.draft_next)

    def test_transient_draft_releases_on_initial_stop(self):
        target = tiny_language()
        draft = tiny_language()
        decoder = FastDraftGreedy(
            target,
            SimpleNamespace(_read_mode="pread"),
            depth=2,
            draft_language=draft,
            release_draft_before_verify=True,
        )
        decoder.append(mx.array([[3, 7, 11]], dtype=mx.uint32))
        stop = int(decoder.next_main.item())

        self.assertEqual(list(decoder.generate(16, {stop})), [])
        self.assertTrue(decoder.draft_disabled)
        self.assertIsNone(decoder.draft_language)
        self.assertIsNone(decoder.draft_cache)
        self.assertIsNone(decoder.draft_next)

    def test_speculative_stop_keeps_cache_for_tool_append(self):
        target = tiny_language()
        prompt = mx.array([[3, 7, 11]], dtype=mx.uint32)

        decoder = FastDraftGreedy(target, SimpleNamespace(_read_mode="pread"), depth=2)
        decoder.append(prompt)
        stop = int(decoder.next_main.item())
        self.assertEqual(list(decoder.generate(16, {stop})), [])

        decoder.append(mx.array([[4]], dtype=mx.uint32))
        actual = int(decoder.next_main.item())

        expected_language = tiny_language()
        expected_cache = expected_language.make_cache()
        expected = prefill_target(
            expected_language,
            mx.concatenate([prompt, mx.array([[4]], dtype=mx.uint32)], axis=1),
            expected_cache,
        )
        expected_next = int(expected[2].item())
        self.assertEqual(actual, expected_next)

    def test_transient_release_is_idempotent(self):
        target = tiny_language()
        decoder = FastDraftGreedy(
            target,
            SimpleNamespace(_read_mode="pread"),
            draft_language=tiny_language(),
            release_draft_before_verify=True,
        )

        decoder._release_transient_draft()
        elapsed = decoder.stats.release_seconds
        decoder._release_transient_draft()

        self.assertEqual(decoder.stats.release_seconds, elapsed)

    def test_transient_draft_releases_after_generation_error(self):
        decoder = FastDraftGreedy(
            tiny_language(),
            SimpleNamespace(_read_mode="pread"),
            draft_language=tiny_language(),
            release_draft_before_verify=True,
        )

        with patch.object(
            decoder,
            "_generate_external",
            side_effect=RuntimeError("draft failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "draft failed"):
                list(decoder.generate(16, set()))

        self.assertTrue(decoder.draft_disabled)
        self.assertIsNone(decoder.draft_language)
        self.assertIsNone(decoder.draft_cache)
        self.assertIsNone(decoder.draft_next)

    def test_confidence_abort_preserves_greedy_tokens(self):
        self.assert_external_draft_exact(False, True, 1e9, 2)

    def test_long_prefill_disables_external_draft_and_stays_exact(self):
        target = tiny_language()
        draft = tiny_language()
        prompt = mx.array([[3, 7, 11]], dtype=mx.uint32)
        reference_cache = target.make_cache()
        reference = target(prompt, cache=reference_cache)
        expected = int(mx.argmax(reference.logits[:, -1, :], axis=-1).item())

        target._position_ids = None
        target._rope_deltas = None
        decoder = FastDraftGreedy(
            target,
            SimpleNamespace(_read_mode="pread"),
            depth=2,
            draft_language=draft,
            release_draft_before_verify=True,
        )
        with patch("models.flashnext.speculative.QSA_CHUNK_THRESHOLD", 2):
            decoder.append(prompt)

        self.assertTrue(decoder.draft_disabled)
        self.assertEqual(int(decoder.next_main.item()), expected)

    def test_exact_verifier_exposes_pre_mixer_residual(self):
        values = [5, 9, 13]
        plain = self.language(
            mx.array([values], dtype=mx.uint32),
            cache=self.prefill(),
            speculative_verify=True,
            return_hidden=True,
            skip_logits=True,
        )
        captured = self.language(
            mx.array([values], dtype=mx.uint32),
            cache=self.prefill(),
            speculative_verify=True,
            return_hidden=True,
            skip_logits=True,
            capture_layer_ids=[],
        )
        # Without the request the sink holds only the final mixed hidden.
        self.assertEqual(len(plain.hidden_states), 1)
        # With [] the sink holds pre-final HC residual first, final last.
        self.assertEqual(len(captured.hidden_states), 2)
        hc_hidden, final_hidden = captured.hidden_states
        self.assertEqual(hc_hidden.shape[-1], 64 * 2)
        self.assertEqual(final_hidden.shape[-1], 64)
        mx.eval(hc_hidden, final_hidden)
        # The final mixed hidden is unchanged by the extra capture.
        self.assertEqual(
            float(
                mx.max(mx.abs(plain.hidden_states[-1] - final_hidden)).item()
            ),
            0.0,
        )

    def test_mtp_multi_cycle_matches_sequential_target(self):
        from models.flashnext.mtp import attach
        from models.flashnext.speculative import MTPGreedy

        language = tiny_language()
        attach(language)
        prompt = mx.array([[3, 7, 11]], dtype=mx.uint32)
        cache = language.make_cache()
        output = language(prompt, cache=cache)
        token = mx.argmax(output.logits[:, -1, :], axis=-1).astype(mx.uint32)
        expected = []
        for _ in range(24):
            expected.append(int(token.item()))
            output = language(token.reshape(1, 1), cache=cache)
            token = mx.argmax(
                output.logits[:, -1, :], axis=-1
            ).astype(mx.uint32)
            mx.eval(token)

        language._position_ids = None
        language._rope_deltas = None
        decoder = MTPGreedy(language, depth=3)
        decoder.append(prompt)
        actual = list(decoder.generate(24, set()))
        self.assertEqual(actual, expected)
        # The draft chain must actually execute, not sit idle.
        self.assertGreater(decoder.stats.drafted, 0)
        self.assertGreater(decoder.stats.cycles, 1)

    def test_mtp_chunked_warmup_matches_target_and_tracks_every_prompt_token(self):
        from unittest.mock import patch
        from models.flashnext.mtp import attach
        from models.flashnext.speculative import MTPGreedy

        language = tiny_language()
        attach(language)
        prompt = mx.array([[3, 7, 11, 13, 19, 23, 29]], dtype=mx.uint32)
        reference = language.make_cache()
        for start in range(0, 7, 3):
            output = language(prompt[:, start:start + 3], cache=reference)
            mx.eval(output.logits)
        expected = []
        for _ in range(12):
            value = int(mx.argmax(output.logits[0, -1]).item())
            expected.append(value)
            output = language(mx.array([[value]], dtype=mx.uint32), cache=reference)
            mx.eval(output.logits)
        language._position_ids = None
        language._rope_deltas = None
        decoder = MTPGreedy(language, depth=3)
        with patch("models.flashnext.prefill.chunk_tokens", return_value=3):
            decoder.append(prompt)
        self.assertEqual(decoder.hist_offset, 7)
        self.assertEqual(list(decoder.generate(12, set())), expected)

    def test_target_replay_replays_single_tokens(self):
        # Recurrent replay must be singleton-equivalent: a batched
        # forward predicts the same tokens but can leave different
        # recurrent states, which diverges later Vontra runs. Lock the
        # one-call-per-token shape here, not just predictions.
        from models.flashnext.speculative import (
            MTPGreedy,
            restore_cache,
            snapshot_cache,
        )

        calls = []

        class FakeLanguage:
            def make_cache(self):
                return [SimpleNamespace(
                    state=mx.zeros((2, 2)), meta_state=None)]

            def __call__(self, ids, cache=None):
                calls.append(tuple(int(v) for v in ids.shape))
                return SimpleNamespace(logits=mx.zeros((1, 1, 4)))

        language = FakeLanguage()
        decoder = MTPGreedy.__new__(MTPGreedy)
        decoder.language = language
        decoder.target_cache = language.make_cache()
        decoder.stats = SimpleNamespace(replayed=0)
        snapshot = snapshot_cache(decoder.target_cache)
        decoder._target_replay(
            snapshot, mx.array([[9, 10, 11]], dtype=mx.uint32))
        self.assertEqual(calls, [(1, 1), (1, 1), (1, 1)])
        self.assertEqual(decoder.stats.replayed, 3)

    def test_target_replay_states_match_singleton(self):
        from models.flashnext.mtp import attach
        from models.flashnext.speculative import (
            MTPGreedy,
            restore_cache,
            snapshot_cache,
        )

        def states(cache):
            return snapshot_cache(cache)

        def state_max_abs(left, right):
            peak = 0.0

            def walk(a, b):
                nonlocal peak
                if isinstance(a, (list, tuple)):
                    for x, y in zip(a, b):
                        walk(x, y)
                    return
                if a is None or b is None:
                    return
                peak = max(peak, float(
                    mx.max(mx.abs(a.astype(mx.float32)
                                  - b.astype(mx.float32))).item()))

            for (a_state, _), (b_state, _) in zip(left, right):
                walk(a_state, b_state)
            return peak

        language = tiny_language()
        attach(language)
        prompt = mx.array([[3, 7, 11]], dtype=mx.uint32)
        cache = language.make_cache()
        mx.eval(language(prompt, cache=cache).logits)
        block = mx.array([[5, 9, 13]], dtype=mx.uint32)
        snapshot = snapshot_cache(cache)

        replayed = language.make_cache()
        restore_cache(replayed, snapshot)
        decoder = MTPGreedy.__new__(MTPGreedy)
        decoder.language = language
        decoder.target_cache = replayed
        decoder.stats = SimpleNamespace(replayed=0)
        decoder._target_replay(snapshot_cache(replayed), block)

        sequential = language.make_cache()
        restore_cache(sequential, snapshot)
        for position in range(block.shape[1]):
            mx.eval(language(block[:, position:position + 1],
                             cache=sequential).logits)

        self.assertEqual(
            state_max_abs(states(replayed), states(sequential)), 0.0)

    @staticmethod
    def _mtp_language():
        """Tiny MTP model with a non-degenerate token stream.

        The synthetic config leaves every norm scale at zero, which collapses
        the output to one repeated token. Unit scales give varied tokens, so
        a wrong cache state shows up as a different token.
        """
        from mlx.utils import tree_map

        from models.flashnext.mtp import attach

        mx.random.seed(0)
        language = tiny_language()
        attach(language)

        def unit_scale(value):
            if value.dtype == mx.float32 and float(mx.max(mx.abs(value)).item()) == 0.0:
                return mx.ones(value.shape, dtype=value.dtype)
            return value

        language.update(tree_map(unit_scale, language.parameters()))
        return language

    @staticmethod
    def _oracle_chain(truth, emitted, vocab, reject=lambda cycle, depth: None):
        """A `_draft_chain` replacement that proposes the true continuation.

        ``emitted`` is the list of tokens yielded so far this turn, so the
        block's first token is ``truth[len(emitted)]`` and the drafts follow
        it. ``reject(cycle, depth)`` names a position to corrupt, or None.
        """
        cycle = [0]

        def chain(decoder, depth=None):
            depth = decoder.depth if depth is None else depth
            start = len(emitted)
            drafts = list(truth[start + 1:start + 1 + depth])
            drafts += [0] * (depth - len(drafts))
            position = reject(cycle[0], depth) if depth else None
            cycle[0] += 1
            if position is not None:
                drafts[position] = (drafts[position] + 1) % vocab
            return drafts

        return chain

    @staticmethod
    def _drain(decoder, count, emitted):
        for token in decoder.generate(count, set()):
            emitted.append(token)
        return emitted

    @staticmethod
    def _sequential(language, prompt, count, cache=None):
        """Greedy tokens from single-token forwards, plus the cache they leave."""
        cache = cache if cache is not None else language.make_cache()
        output = language(prompt, cache=cache)
        mx.eval(output.logits)
        tokens = []
        for _ in range(count):
            token = mx.argmax(output.logits[:, -1, :], axis=-1).astype(mx.uint32)
            tokens.append(int(token.item()))
            output = language(token.reshape(1, 1), cache=cache)
            mx.eval(output.logits)
        return tokens, cache, output

    @staticmethod
    def _cache_max_abs(left, right):
        peak = 0.0

        def walk(a, b):
            nonlocal peak
            if isinstance(a, (list, tuple)):
                for x, y in zip(a, b):
                    walk(x, y)
                return
            if a is None or b is None:
                return
            if a.shape != b.shape:
                raise AssertionError(f"state shapes differ: {a.shape} vs {b.shape}")
            peak = max(peak, float(mx.max(mx.abs(
                a.astype(mx.float32) - b.astype(mx.float32))).item()))

        for (a_state, _), (b_state, _) in zip(
                snapshot_cache(left), snapshot_cache(right)):
            walk(a_state, b_state)
        return peak

    def _two_turn_reference(self, language, first, second, count_first, count_second):
        tokens_first, cache, output = self._sequential(language, first, count_first)
        token = mx.argmax(output.logits[:, -1, :], axis=-1).astype(mx.uint32)
        # Same shape as the decoder's append: one batched forward of the turn.
        output = language(second, cache=cache)
        mx.eval(output.logits)
        tokens_second, _, _ = self._sequential_from(language, cache, output, count_second)
        return tokens_first, tokens_second

    @staticmethod
    def _sequential_from(language, cache, output, count):
        tokens = []
        for _ in range(count):
            token = mx.argmax(output.logits[:, -1, :], axis=-1).astype(mx.uint32)
            tokens.append(int(token.item()))
            output = language(token.reshape(1, 1), cache=cache)
            mx.eval(output.logits)
        return tokens, cache, output

    def _run_two_turns(self, decoder, first, second, count_first, count_second):
        decoder.append(first)
        tokens_first = list(decoder.generate(count_first, set()))
        decoder.append(second)
        tokens_second = list(decoder.generate(count_second, set()))
        return tokens_first, tokens_second

    def test_mtp_adaptive_matches_sequential_target_over_two_turns(self):
        import os
        from models.flashnext.speculative import MTPGreedy

        language = self._mtp_language()
        vocab = int(language.args.vocab_size)
        first = mx.array([[3, 7, 11]], dtype=mx.uint32)
        second = mx.array([[5, 9]], dtype=mx.uint32)
        expected = self._two_turn_reference(language, first, second, 48, 48)

        def run(chain_for):
            language._position_ids = None
            language._rope_deltas = None
            with patch.dict(os.environ, {"FLASHNEXT_MTP_ADAPTIVE": "1"}):
                decoder = MTPGreedy(language, depth=4)
            turns = []
            for ids, truth in ((first, expected[0]), (second, expected[1])):
                emitted = []
                decoder.append(ids)
                with patch.object(MTPGreedy, "_draft_chain", chain_for(truth, emitted)):
                    turns.append(self._drain(decoder, 48, emitted))
            return decoder, turns

        # Real MTP head: random weights, so nearly every draft is rejected.
        decoder, turns = run(lambda truth, emitted: MTPGreedy._draft_chain)
        self.assertEqual(turns, list(expected))
        # Oracle drafts with an occasional rejection at a varying position.
        decoder, turns = run(lambda truth, emitted: self._oracle_chain(
            truth, emitted, vocab,
            reject=lambda cycle, depth: (cycle % 4) % depth if cycle % 4 != 3 else None))
        self.assertEqual(turns, list(expected))
        counts = decoder.stats.depth_counts
        self.assertIn(0, counts)
        self.assertGreater(len(counts), 1)
        self.assertEqual(sum(counts.values()), decoder.stats.cycles)
        self.assertTrue(all(0 <= depth <= 4 for depth in counts))
        self.assertGreater(decoder.stats.accepted, 0)
        # The controller persists across both turns of the conversation.
        self.assertEqual(decoder.stats.adaptive["cycles"], decoder.stats.cycles)
        self.assertGreater(decoder.stats.verify_seconds, 0.0)
        self.assertGreater(decoder.stats.draft_seconds, 0.0)
        self.assertEqual(decoder.stats.drafted, sum(
            depth * count for depth, count in counts.items()))

    def test_mtp_adaptive_follows_depth_changes_and_reset(self):
        import os
        from models.flashnext.speculative import MTPGreedy

        language = self._mtp_language()
        first = mx.array([[3, 7, 11]], dtype=mx.uint32)
        second = mx.array([[5, 9]], dtype=mx.uint32)
        expected = self._two_turn_reference(language, first, second, 15, 15)
        language._position_ids = None
        language._rope_deltas = None
        with patch.dict(os.environ, {"FLASHNEXT_MTP_ADAPTIVE": "1"}):
            decoder = MTPGreedy(language, depth=4)
        self.assertIsNone(decoder._depth_controller)
        decoder.append(first)
        self.assertEqual(list(decoder.generate(15, set())), expected[0])
        controller = decoder._depth_controller
        self.assertEqual(controller.max_depth, 4)
        # The backend lowers the configured depth for a turn: the ceiling follows.
        decoder.depth = 2
        decoder.append(second)
        self.assertEqual(list(decoder.generate(15, set())), expected[1])
        self.assertIsNot(decoder._depth_controller, controller)
        self.assertEqual(decoder._depth_controller.max_depth, 2)
        self.assertLessEqual(max(decoder.stats.depth_counts), 4)
        # A new conversation re-reads the flag and drops the controller.
        with patch.dict(os.environ, {"FLASHNEXT_MTP_ADAPTIVE": "0"}):
            decoder.reset()
        self.assertFalse(decoder._adaptive)
        self.assertIsNone(decoder._depth_controller)
        self.assertEqual(decoder.stats.depth_counts, {})
        with patch.dict(os.environ, {"FLASHNEXT_MTP_ADAPTIVE": "1"}):
            decoder.reset()
        self.assertTrue(decoder._adaptive)

    def test_mtp_forced_depth_schedule_matches_sequential_target(self):
        import itertools
        import os
        from models.flashnext.speculative import MTPGreedy

        language = self._mtp_language()
        vocab = int(language.args.vocab_size)
        prompt = mx.array([[3, 7, 11]], dtype=mx.uint32)
        expected, _, _ = self._sequential(language, prompt, 60)
        for reject in (
            lambda cycle, depth: None,
            lambda cycle, depth: cycle % depth,
        ):
            language._position_ids = None
            language._rope_deltas = None
            with patch.dict(os.environ, {"FLASHNEXT_MTP_ADAPTIVE": "1"}):
                decoder = MTPGreedy(language, depth=4)
            controller = decoder._controller()
            decoder.append(prompt)
            emitted = []
            chain = self._oracle_chain(expected, emitted, vocab, reject)
            with patch.object(
                controller, "choose", side_effect=itertools.cycle(range(5))
            ), patch.object(MTPGreedy, "_draft_chain", chain):
                self._drain(decoder, 48, emitted)
            self.assertEqual(emitted, expected[:48])
            self.assertEqual(set(decoder.stats.depth_counts), {0, 1, 2, 3, 4})

    def test_mtp_forced_rejection_at_every_position_matches_target_and_state(self):
        from models.flashnext.speculative import MTPGreedy

        language = self._mtp_language()
        prompt = mx.array([[3, 7, 11]], dtype=mx.uint32)
        count = 20
        # Extra tokens let the forced drafts run past the end of generation.
        truth, _, _ = self._sequential(language, prompt, count + 8)
        _, reference_cache, _ = self._sequential(language, prompt, count)
        vocab = int(language.args.vocab_size)
        for depth in range(1, 5):
            for position in range(depth):
                with self.subTest(depth=depth, position=position):
                    emitted = []

                    def forced(decoder, chain_depth=None, *, _position=position):
                        self.assertEqual(chain_depth, decoder.depth)
                        start = len(emitted)
                        drafts = truth[start + 1:start + 1 + chain_depth]
                        drafts[_position] = (drafts[_position] + 1) % vocab
                        return drafts

                    language._position_ids = None
                    language._rope_deltas = None
                    decoder = MTPGreedy(language, depth=depth)
                    decoder.append(prompt)
                    with patch.object(MTPGreedy, "_draft_chain", forced):
                        for token in decoder.generate(count, set()):
                            emitted.append(token)
                    self.assertEqual(emitted, truth[:count])
                    # Every cycle rejects at `position`, except the last, which
                    # may be cut short by the token budget.
                    self.assertLessEqual(decoder.stats.accepted,
                                         decoder.stats.cycles * position)
                    self.assertGreater(decoder.stats.replayed, 0)
                    self.assertEqual(
                        self._cache_max_abs(decoder.target_cache, reference_cache),
                        0.0,
                    )

    def test_mtp_flag_off_drafts_configured_depth_without_controller(self):
        import os
        from models.flashnext.speculative import MTPGreedy

        language = self._mtp_language()
        prompt = mx.array([[3, 7, 11]], dtype=mx.uint32)
        expected, _, _ = self._sequential(language, prompt, 24)
        language._position_ids = None
        language._rope_deltas = None
        calls = []
        original = MTPGreedy._draft_chain

        def spy(decoder, depth=None):
            calls.append(depth)
            return original(decoder, depth)

        env = {k: v for k, v in os.environ.items() if k != "FLASHNEXT_MTP_ADAPTIVE"}
        with patch.dict(os.environ, env, clear=True):
            decoder = MTPGreedy(language, depth=3)
        decoder.append(prompt)
        with patch.object(MTPGreedy, "_draft_chain", spy):
            actual = list(decoder.generate(24, set()))
        self.assertEqual(actual, expected)
        self.assertEqual(calls, [3] * decoder.stats.cycles)
        self.assertGreater(decoder.stats.cycles, 1)
        self.assertIsNone(decoder._depth_controller)
        self.assertIsNone(decoder.stats.adaptive)
        self.assertEqual(decoder.stats.depth_counts, {3: decoder.stats.cycles})
        self.assertEqual(decoder.stats.drafted, 3 * decoder.stats.cycles)
        self.assertGreater(decoder.stats.verify_seconds, 0.0)

if __name__ == "__main__":
    unittest.main()
