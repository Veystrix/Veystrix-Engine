"""Disconnects and failures between cooperative Flash-Next singletons on the tiny Qwen4 language model.

Flash-Next has no grouped decode: the scheduler steps one conversation at a time, each inside ``batch_scope``, which
loads that conversation's position ids and rope deltas into the shared language object and puts the outside values
back afterwards. Two conversations are interleaved one token at a time. One of them leaves before each output
boundary 0..N, and separately its forward raises (at the first layer, between the two layers, from the sampler).
The conversation that stays must emit the tokens, and hold the cache tensors and position state, of a run in which
nobody left; the language object's ``_position_ids`` and ``_rope_deltas`` are what they were outside the scope even
when the scope exits by an exception; a conversation created afterwards reproduces its control.
"""
from __future__ import annotations

from contextlib import contextmanager
import unittest
from unittest import mock

import mlx.core as mx
from mlx_vlm.models.qwen3_5 import language as base_language

from models.flashnext.adaptive_topk import apply as patch_topk, set_renorm_blend, set_threshold
from models.flashnext.patch_rmsnorm import apply as patch_norm, restore as restore_norm
from models.flashnext.qwen4_verifier import Qwen4ExactSpeculativeVerifier
from models.flashnext.tests.unit.test_speculative_tiny import tiny_language
from veystrix.backends.flashnext import FlashNextBackend
from veystrix.sampling import Sampler, Sampling

ROUNDS = 5
PROMPTS = [[3, 7, 11], [4, 12, 10, 13], [5, 9, 14, 6, 21]]
THIRD = [8, 15, 22, 17]
OUTSIDE = (mx.array([[41, 42, 43, 44]], dtype=mx.int32), mx.array([[7]], dtype=mx.uint32))


class Injected(Exception):
    pass


def flatten(value):
    if value is None:
        return []
    if isinstance(value, mx.array):
        return [value]
    return [leaf for item in value for leaf in flatten(item)]


def same(left, right):
    a, b = flatten(left), flatten(right)
    return len(a) == len(b) and all(x.shape == y.shape and mx.array_equal(x, y).item() for x, y in zip(a, b))


def cache_state(backend):
    return [entry.state for entry in backend.cache]


class Conversation:
    """One backend driven like ``cooperative.generate``: scoped prefill, then one scoped decode per token."""

    def __init__(self, language, prompt):
        self.language = language
        self.backend = FlashNextBackend.__new__(FlashNextBackend)
        self.backend.language = language
        self.backend.cache = language.make_cache()
        self.backend._batch_positions = (None, None)
        self.sampler = Sampler(Sampling.greedy_settings())
        with self.backend.batch_scope():
            self.token = self.backend.batch_prefill(prompt, 0, True, self.sampler)
        self.out = []

    def step(self, want, sampler=None):
        value = self.token
        with self.backend.batch_scope():
            result = self.backend.batch_decode(value, sampler or self.sampler, want)
        self.out.append(value)
        self.token = result


class FlashNextDisconnectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if patch_norm():               # process-wide: undo it when this class installed it
            cls.addClassCleanup(restore_norm)
        patch_topk()
        cls.language = tiny_language()
        base_language._EXACT_SPECULATIVE_VERIFIER = Qwen4ExactSpeculativeVerifier()
        set_threshold(1.0)
        set_renorm_blend(1.0)

    def setUp(self):
        language = self.language
        saved = (language._position_ids, language._rope_deltas)
        language._position_ids, language._rope_deltas = OUTSIDE

        def restore():
            language._position_ids, language._rope_deltas = saved

        self.addCleanup(restore)
        self.controls = {}

    # -- helpers ---------------------------------------------------------------------------------------
    def control(self, prompt):
        key = tuple(prompt)
        if key not in self.controls:
            conversation = Conversation(self.language, prompt)
            for index in range(ROUNDS):
                conversation.step(index + 1 < ROUNDS)
            self.controls[key] = conversation
        return self.controls[key]

    def assert_outside_positions_untouched(self, label=""):
        self.assertIs(self.language._position_ids, OUTSIDE[0], label)
        self.assertIs(self.language._rope_deltas, OUTSIDE[1], label)

    def assert_matches_control(self, conversation, prompt, label=""):
        control = self.control(prompt)
        self.assertEqual(conversation.out, control.out, label)
        self.assertTrue(same(cache_state(conversation.backend), cache_state(control.backend)), label)
        self.assertTrue(same(conversation.backend._batch_positions, control.backend._batch_positions), label)

    def assert_fresh_conversation_reproduces_control(self, prompt):
        fresh = Conversation(self.language, prompt)
        for index in range(ROUNDS):
            fresh.step(index + 1 < ROUNDS)
        self.assert_matches_control(fresh, prompt, "conversation created after the incident")

    # -- tests -----------------------------------------------------------------------------------------
    def test_scope_restores_the_outside_positions_on_every_exit(self):
        backend = Conversation(self.language, PROMPTS[0]).backend
        self.assert_outside_positions_untouched("after a normal prefill")
        for error in (Injected, KeyboardInterrupt):
            with self.subTest(error=error.__name__):
                with self.assertRaises(error):
                    with backend.batch_scope():
                        self.assertIsNot(self.language._position_ids, OUTSIDE[0])   # the conversation's own
                        raise error
                self.assert_outside_positions_untouched()

    def test_one_conversation_leaves_before_every_output_boundary(self):
        for leaver in range(len(PROMPTS)):
            for boundary in range(ROUNDS + 1):
                with self.subTest(leaver=leaver, boundary=boundary):
                    conversations = [Conversation(self.language, prompt) for prompt in PROMPTS]
                    for index in range(ROUNDS):
                        for number, conversation in enumerate(conversations):
                            if number == leaver and index >= boundary:
                                continue
                            conversation.step(index + 1 < ROUNDS)
                            self.assert_outside_positions_untouched()
                    self.assertEqual(conversations[leaver].out, self.control(PROMPTS[leaver]).out[:boundary])
                    for number, conversation in enumerate(conversations):
                        if number != leaver:
                            self.assert_matches_control(conversation, PROMPTS[number], f"stayer {number}")
                    self.assert_fresh_conversation_reproduces_control(THIRD)
                    self.assert_outside_positions_untouched()

    @contextmanager
    def failing_layer(self, nth):
        """Raise from the nth decoder layer entered inside the block (mid-forward: layer 1 already updated state)."""
        layer_type = type(self.language.model.layers[0])
        original = layer_type.__call__
        entered = [0]

        def wrapper(layer, *args, **kwargs):
            entered[0] += 1
            if entered[0] == nth:
                raise Injected(f"decoder layer {nth}")
            return original(layer, *args, **kwargs)

        with mock.patch.object(layer_type, "__call__", wrapper):
            yield

    def test_a_forward_that_raises_leaves_the_others_and_the_scope_clean(self):
        class RaisingSampler:
            def observe(self, token):
                pass

            def __call__(self, logits):
                raise Injected("sampler")

        for failure in ("layer 1", "layer 2", "sampler"):
            for failed_round in (0, 2):
                with self.subTest(failure=failure, failed_round=failed_round):
                    victim = Conversation(self.language, PROMPTS[0])
                    stayers = [Conversation(self.language, prompt) for prompt in PROMPTS[1:]]
                    for index in range(ROUNDS):
                        if index == failed_round:
                            if failure == "sampler":
                                with self.assertRaises(Injected):
                                    victim.step(True, RaisingSampler())
                            else:
                                with self.failing_layer(int(failure.split()[1])):
                                    with self.assertRaises(Injected):
                                        victim.step(True)
                            self.assert_outside_positions_untouched("after the failed forward")
                        elif index < failed_round:
                            victim.step(True)
                        for conversation in stayers:
                            conversation.step(index + 1 < ROUNDS)
                            self.assert_outside_positions_untouched()
                    for number, conversation in enumerate(stayers):
                        self.assert_matches_control(conversation, PROMPTS[number + 1], f"stayer {number}")
                    self.assert_fresh_conversation_reproduces_control(THIRD)
                    self.assert_outside_positions_untouched()


if __name__ == "__main__":
    unittest.main()
