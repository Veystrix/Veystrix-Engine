"""Disconnects and failures around real Nemotron forks on the tiny synthetic checkpoint.

Three layers, each with injection at every boundary rather than at one:

* ``GroupedDecodeLeaveTests``: ready conversations decode as one group through ``batch_decode_group`` (the
  production call of ``BatchScheduler``); one of them leaves before each output boundary 0..N. The conversations
  that stay must emit the tokens, and hold the cache tensors, of a run in which nobody left.
* ``ForwardFailureTests``: the grouped forward raises mid-layer inside ``EngineCore.decode_many`` (before or after
  each expert fetch, at each MoE layer, at several rounds). The expert pool, its flags and byte accounting come back
  clean, a conversation that was not part of the failed group continues exactly, and brand-new forks (single and
  grouped) reproduce their controls.
* ``ServedDisconnectTests``: real forks through ``cooperative.generate`` and a ``BatchScheduler`` whose loop runs on
  the main thread (MLX work never leaves it); ``should_cancel`` turns true at every check of the request, including
  while a decode quantum is queued in a group of three. Survivors match their solo controls in tokens and cache, the
  reservation is released, and a system prefix stored before the disconnect restores to the cold-prefill state.

An engine error inside a group is not survivable by the members of that group: the scheduler hands the exception to
every row, and the server drops each failed conversation's fork. The tests therefore pin the invariants that do hold
(shared state, bystanders, later requests) and do not claim the half-advanced group members can continue.
"""
from __future__ import annotations

from contextlib import contextmanager
from itertools import permutations
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest import mock

import mlx.core as mx

from models.nemotron35.tests.unit.test_backend import StubTokenizer, machine
from models.nemotron35.tiny_checkpoint import permissive_config_check, write_checkpoint
from veystrix.backends.base import GenerationCancelled
from veystrix.backends.cooperative import PREFILL_TOKENS
from veystrix.backends.nemotron35 import NemotronBackend
from veystrix.batch_scheduler import BatchScheduler
from veystrix.sampling import Sampler, Sampling
from veystrix.tests.test_cooperative_disconnect import CancelAt, Gate, Injected, Member, hold, wait_until

SYSTEM_HEADER = [60, 61]
MESSAGE_END = [62]
ROUNDS = 5
_dir = None


def setUpModule():
    global _dir
    _dir = tempfile.mkdtemp(prefix="nemotron-disconnect-")
    write_checkpoint(Path(_dir) / "ck")


def tearDownModule():
    shutil.rmtree(_dir, ignore_errors=True)


def make(**options):
    prefix = tempfile.mkdtemp(prefix="prefix-", dir=_dir)
    args = dict(authorized=True, config_check=permissive_config_check, machine=machine(), tokenizer=StubTokenizer(),
                pool_bytes=64 * 1024, chunk_tokens=4, session_dir=str(Path(_dir) / "sessions"),
                prefix_dir=prefix, snapshot_tokens=8)
    args.update(options)
    backend = NemotronBackend(Path(_dir) / "ck", **args)
    backend._current_machine = lambda: machine()
    # The stub tokenizer encodes the chat markers as many tokens; the system-prefix plan needs single ids.
    backend._system_header, backend._message_end = list(SYSTEM_HEADER), list(MESSAGE_END)
    return backend


def body(seed, count):
    return [4 + (seed * 17 + index * 5) % 56 for index in range(count)]


def with_system(seed, tail):
    return SYSTEM_HEADER + body(1, 300) + MESSAGE_END + body(seed, tail)


def same_state(left, right):
    if left.core.cached_tokens != right.core.cached_tokens:
        return False
    a, b = left.core.state_arrays(), right.core.state_arrays()
    return len(a) == len(b) and all(x.shape == y.shape and mx.array_equal(x, y).item() for x, y in zip(a, b))


class Conversation:
    """One fork driven the way ``cooperative.generate`` drives it: prefill, then one fed token per decode."""

    def __init__(self, parent, prompt):
        self.fork = parent.fork_for_batch()
        assert self.fork is not None
        self.sampler = Sampler(Sampling.greedy_settings())
        self.token = self.fork.batch_prefill(prompt, 0, True, self.sampler)
        self.out = []

    @property
    def step(self):
        return self.fork, self.token, self.sampler


def single_round(conversation, want):
    fork, token, sampler = conversation.step
    result = fork.batch_decode(token, sampler, want)
    conversation.out.append(token)
    conversation.token = result


def group_round(parent, conversations, want):
    """What ``BatchScheduler._run_decode_group`` does with the rows that are still live."""
    if len(conversations) == 1:
        return single_round(conversations[0], want)
    results = parent.batch_decode_group([(*c.step, want) for c in conversations])
    for conversation, result in zip(conversations, results, strict=True):
        conversation.out.append(conversation.token)
        conversation.token = result


class NemotronCase(unittest.TestCase):
    def setUp(self):
        self.parent = make()
        self.addCleanup(self.parent.close)
        self.pool = self.parent.streamed.pool
        self.controls = {}

    def control(self, prompt, horizon=ROUNDS):
        """The conversation alone: every round through the singleton forward."""
        key = (tuple(prompt), horizon)
        if key not in self.controls:
            conversation = Conversation(self.parent, prompt)
            for round_index in range(horizon):
                single_round(conversation, round_index + 1 < horizon)
            self.controls[key] = conversation
        return self.controls[key]

    def assert_matches_control(self, conversation, prompt, label="", horizon=ROUNDS):
        control = self.control(prompt, horizon)
        self.assertEqual(conversation.out, control.out, label)
        self.assertTrue(same_state(conversation.fork, control.fork), label)

    def assert_pool_idle(self):
        pool = self.pool
        self.assertTrue(pool.speculate)
        self.assertIsNone(pool._held)
        self.assertIsNone(pool._held_layer)
        self.assertIsNone(pool._held_routes)
        self.assertEqual(pool.held_bytes(), 0)
        self.assertFalse(pool.cancel_check())
        self.assertEqual(sum(size for _, size in pool._items.values()), pool.resident_bytes)
        self.assertLessEqual(pool.resident_bytes, pool.budget)

    def assert_new_forks_reproduce_controls(self, single, pair):
        """A fork created after the incident decodes alone, and two of them decode as a group, exactly."""
        alone = Conversation(self.parent, single)
        for round_index in range(ROUNDS):
            single_round(alone, round_index + 1 < ROUNDS)
        self.assert_matches_control(alone, single, "fresh single fork")
        group = [Conversation(self.parent, prompt) for prompt in pair]
        for round_index in range(ROUNDS):
            group_round(self.parent, group, round_index + 1 < ROUNDS)
        for conversation, prompt in zip(group, pair, strict=True):
            self.assert_matches_control(conversation, prompt, "fresh grouped fork")


PROMPTS = [body(2, 3), body(3, 5), body(4, 9)]
THIRD = body(5, 6)
FOURTH = body(6, 4)


class GroupedDecodeLeaveTests(NemotronCase):
    def test_one_row_leaves_the_group_before_every_output_boundary(self):
        for leaver in range(3):
            for boundary in range(ROUNDS + 1):
                with self.subTest(leaver=leaver, boundary=boundary):
                    conversations = [Conversation(self.parent, prompt) for prompt in PROMPTS]
                    for round_index in range(ROUNDS):
                        live = [c for index, c in enumerate(conversations)
                                if not (index == leaver and round_index >= boundary)]
                        group_round(self.parent, live, round_index + 1 < ROUNDS)
                    self.assertEqual(conversations[leaver].out, self.control(PROMPTS[leaver]).out[:boundary])
                    for index, conversation in enumerate(conversations):
                        if index != leaver:
                            self.assert_matches_control(conversation, PROMPTS[index], f"stayer {index}")
                    self.assert_pool_idle()
                    self.assert_new_forks_reproduce_controls(THIRD, [FOURTH, PROMPTS[0]])

    def test_rows_that_finish_at_different_horizons_share_rounds_with_and_without_logits(self):
        """Each request has its own max_tokens: in one round a row may skip the head while another samples."""
        horizons = [3, 5, ROUNDS]
        conversations = [Conversation(self.parent, prompt) for prompt in PROMPTS]
        for round_index in range(ROUNDS):
            live = [(c, h) for c, h in zip(conversations, horizons, strict=True) if round_index < h]
            steps = [(*c.step, round_index + 1 < h) for c, h in live]
            results = (self.parent.batch_decode_group(steps) if len(steps) > 1
                       else [live[0][0].fork.batch_decode(*steps[0])])
            for (conversation, horizon), result in zip(live, results, strict=True):
                self.assertEqual(result is None, round_index + 1 == horizon)
                conversation.out.append(conversation.token)
                conversation.token = result
        for conversation, prompt, horizon in zip(conversations, PROMPTS, horizons, strict=True):
            self.assert_matches_control(conversation, prompt, f"horizon {horizon}", horizon)
        self.assert_pool_idle()

    def test_two_rows_leave_at_different_boundaries_and_the_last_runs_alone(self):
        for first, second in ((0, 1), (1, 3), (2, 4)):
            with self.subTest(first=first, second=second):
                conversations = [Conversation(self.parent, prompt) for prompt in PROMPTS]
                leaves = {0: first, 1: second}
                for round_index in range(ROUNDS):
                    live = [c for index, c in enumerate(conversations) if round_index < leaves.get(index, ROUNDS)]
                    group_round(self.parent, live, round_index + 1 < ROUNDS)
                self.assert_matches_control(conversations[2], PROMPTS[2])
                self.assertEqual(conversations[0].out, self.control(PROMPTS[0]).out[:first])
                self.assertEqual(conversations[1].out, self.control(PROMPTS[1]).out[:second])
                self.assert_pool_idle()


class ForwardFailureTests(NemotronCase):
    """The grouped forward raises mid-layer; nothing outside the failed group may notice."""

    @contextmanager
    def failing(self, point, nth):
        pool = self.pool
        calls = [0]
        if point == "begin_layer":
            original = pool.begin_layer

            def begin(layer, **options):
                if layer == nth:                     # the nth *layer index* (1 and 3 are the MoE layers)
                    raise Injected(f"begin_layer({layer})")
                return original(layer, **options)

            with mock.patch.object(pool, "begin_layer", begin):
                yield
            return
        original = pool.fetch

        def fetch(*args, **options):
            calls[0] += 1
            if point == "fetch_before" and calls[0] == nth:
                raise Injected(f"fetch #{nth} before reading")
            result = original(*args, **options)
            if point == "fetch_after" and calls[0] == nth:
                raise Injected(f"fetch #{nth} after reading")
            return result

        with mock.patch.object(pool, "fetch", fetch):
            yield

    def test_failure_at_each_moe_layer_and_expert_fetch_leaves_the_pool_and_bystanders_intact(self):
        points = [("begin_layer", 1), ("begin_layer", 3)]
        points += [(kind, nth) for kind in ("fetch_before", "fetch_after") for nth in (1, 2, 3, 4)]
        for point, nth in points:
            for failed_round in (0, 2):
                with self.subTest(point=point, nth=nth, failed_round=failed_round):
                    pair = [Conversation(self.parent, prompt) for prompt in PROMPTS[:2]]
                    bystander = Conversation(self.parent, PROMPTS[2])
                    for round_index in range(failed_round):
                        group_round(self.parent, pair, True)
                        single_round(bystander, True)
                    with self.failing(point, nth):
                        with self.assertRaises(Injected):
                            group_round(self.parent, pair, True)
                    self.assert_pool_idle()
                    # The bystander was never in the failed forward: it finishes exactly as it would alone.
                    for round_index in range(failed_round, ROUNDS):
                        single_round(bystander, round_index + 1 < ROUNDS)
                    self.assert_matches_control(bystander, PROMPTS[2], "bystander")
                    self.assert_new_forks_reproduce_controls(THIRD, [FOURTH, PROMPTS[0]])
                    self.assert_pool_idle()


def serve(scheduler, controller, *, seconds=180):
    """Run ``controller`` on a thread and the model loop on this (the main) thread, as the server does."""
    stop, box = threading.Event(), {}

    def run():
        try:
            if not wait_until(lambda: scheduler.loop_running):
                raise TimeoutError("the model loop never started")
            controller()
        except BaseException as error:
            box["error"] = error
        finally:
            stop.set()
            scheduler._wake.set()

    thread = threading.Thread(target=run, daemon=True)
    watchdog = threading.Timer(seconds, lambda: (box.setdefault("error", TimeoutError("served test hung")),
                                                 stop.set()))
    watchdog.daemon = True
    thread.start()
    watchdog.start()
    try:
        scheduler.serve_jobs(stop)
    finally:
        watchdog.cancel()
    thread.join(10)
    if "error" in box:
        raise box["error"]


class ServedDisconnectTests(NemotronCase):
    def scheduler(self):
        return BatchScheduler(headroom=lambda: 1 << 40)

    def forked(self, scheduler, name, prompt, **options):
        return Member(scheduler, self.parent.fork_for_batch(), name, prompt, ROUNDS, **options)

    def assert_scheduler_clean(self, scheduler):
        self.assertEqual(scheduler.admission.used, 0)
        self.assertEqual(scheduler.snapshot()["batch"]["requests"], [])
        self.assertEqual(scheduler.ready.pending(), 0)
        self.assertEqual(scheduler._quanta, {})

    def solo(self, prompt):
        """The conversation alone through the same pipeline on an untouched scheduler (and prefix cache)."""
        key = ("served", tuple(prompt))
        if key not in self.controls:
            parent = make()
            self.addCleanup(parent.close)
            scheduler = self.scheduler()
            member = Member(scheduler, parent.fork_for_batch(), "control", prompt, ROUNDS)

            def run():
                member.start()
                member.join(60)

            serve(scheduler, run)
            self.assertIsNone(member.error)
            self.assertEqual(len(member.tokens), ROUNDS, "the control must not stop early on an EOS token")
            self.controls[key] = member
        return self.controls[key]

    def survivors_match(self, members):
        for member in members:
            self.assertIsNone(member.error, member.label)
            control = self.solo(member.prompt)
            self.assertEqual(member.tokens, control.tokens, member.label)
            self.assertTrue(same_state(member.backend, control.backend), member.label)

    def test_cancel_at_every_check_with_two_conversations_alongside(self):
        prefill_quanta = 1                                  # every prompt here fits one 128-token quantum
        scheduler = self.scheduler()
        probe = CancelAt(None)
        member = self.forked(scheduler, "probe", body(7, 11), cancel=probe)
        serve(scheduler, lambda: (member.start(), member.join(60)))
        self.assertIsNone(member.error)
        self.assertEqual(probe.calls, 1 + prefill_quanta + ROUNDS)
        victim_prompt, others = PROMPTS[0], PROMPTS[1:]
        for k in range(prefill_quanta + 1 + ROUNDS):
            with self.subTest(k=k):
                scheduler = self.scheduler()
                victim = self.forked(scheduler, "victim", victim_prompt, cancel=CancelAt(k))
                members = [self.forked(scheduler, f"other{i}", prompt) for i, prompt in enumerate(others)]
                after = []

                def controller():
                    gate = None
                    if k > prefill_quanta:                          # decode phase: force one group of three
                        gate = Gate(scheduler, k - prefill_quanta - 1, ["victim", "other0", "other1"])
                        for member in (victim, *members):
                            member.gate = gate
                    for member in (victim, *members):
                        member.start()
                    if gate is not None:
                        gate.drive()
                    for member in (victim, *members):
                        member.join(60)
                    follow = self.forked(scheduler, "after", victim_prompt)
                    follow.start()
                    follow.join(60)
                    after.append(follow)
                    self.assert_scheduler_clean(scheduler)

                serve(scheduler, controller)
                self.assertIsInstance(victim.error, GenerationCancelled)
                self.assertEqual(victim.tokens, self.solo(victim_prompt).tokens[:max(0, k - prefill_quanta - 1)])
                self.survivors_match(members)
                self.survivors_match(after)
                if k > prefill_quanta:                      # the two that stayed still ran as one group
                    self.assertGreaterEqual(scheduler.snapshot()["batch"]["max_decode_group"], 2)
                self.assert_pool_idle()

    def test_every_position_of_the_victim_in_a_group_of_three(self):
        prefill_quanta = 1                                  # every prompt here fits one 128-token quantum
        for step in (0, 2, ROUNDS - 1):
            for order in permutations(("victim", "left", "right")):
                with self.subTest(step=step, order=order):
                    scheduler = self.scheduler()
                    prompts = {"victim": PROMPTS[0], "left": PROMPTS[1], "right": PROMPTS[2]}
                    members = {name: self.forked(
                        scheduler, name, prompts[name],
                        cancel=CancelAt(prefill_quanta + 1 + step if name == "victim" else None)) for name in order}

                    def controller():
                        gate = Gate(scheduler, step, order)
                        for member in members.values():
                            member.gate = gate
                            member.start()
                        gate.drive()
                        for member in members.values():
                            member.join(60)
                        self.assert_scheduler_clean(scheduler)

                    serve(scheduler, controller)
                    victim = members["victim"]
                    self.assertIsInstance(victim.error, GenerationCancelled)
                    self.assertEqual(victim.tokens, self.solo(victim.prompt).tokens[:step])
                    self.survivors_match([members["left"], members["right"]])
                    self.assertIn(scheduler.snapshot()["batch"]["max_decode_group"], (2, 3))
                    self.assert_pool_idle()

    def test_prefix_saved_before_the_disconnect_restores_to_the_cold_prefill_state(self):
        victim_prompt, next_prompt = with_system(2, 3), with_system(3, 3)
        anchor = len(SYSTEM_HEADER) + 300 + len(MESSAGE_END)
        anchor_quanta = -(-anchor // PREFILL_TOKENS)       # the quantum that ends on the system boundary
        cold = make(prefix_cache=False)
        self.addCleanup(cold.close)
        cold_fork = cold.fork_for_batch()
        cold_fork.core.prefill(victim_prompt[:anchor])
        total = 1 + anchor_quanta + 1 + ROUNDS             # plan, quanta up to and past the anchor, decode tokens
        for k in range(total):
            with self.subTest(k=k):
                parent = make()
                self.addCleanup(parent.close)
                scheduler = self.scheduler()
                victim = Member(scheduler, parent.fork_for_batch(), "victim", victim_prompt, ROUNDS,
                                cancel=CancelAt(k))

                def controller():
                    victim.start()
                    victim.join(60)
                    self.assert_scheduler_clean(scheduler)

                serve(scheduler, controller)
                self.assertIsInstance(victim.error, GenerationCancelled)
                saved = k > anchor_quanta                   # the save happens inside quantum ``anchor_quanta``
                # Ask the store before anyone else can add to it.
                probe = parent.fork_for_batch()
                probe.pending = list(next_prompt)
                offset, planned = probe.batch_prefix_plan(next_prompt, 0, PREFILL_TOKENS)
                self.assertEqual(planned, anchor)
                if saved:
                    self.assertEqual(offset, anchor)
                    self.assertTrue(same_state(probe, cold_fork), "stored anchor differs from a cold prefill")
                else:
                    self.assertEqual(offset, 0)
                scheduler = self.scheduler()
                restored = Member(scheduler, parent.fork_for_batch(), "restored", next_prompt, ROUNDS)
                serve(scheduler, lambda: (restored.start(), restored.join(60), self.assert_scheduler_clean(scheduler)))
                self.assertIsNone(restored.error)
                hit = getattr(restored.backend.last_prefix, "hit_tokens", 0)
                self.assertEqual(hit, anchor if saved else 0, f"restored {hit} tokens")
                # Whatever was restored or recomputed, the request emits what a cold request emits.
                reference = make()
                self.addCleanup(reference.close)
                control_scheduler = self.scheduler()
                control = Member(control_scheduler, reference.fork_for_batch(), "control", next_prompt, ROUNDS)
                serve(control_scheduler, lambda: (control.start(), control.join(60)))
                self.assertEqual(restored.tokens, control.tokens)
                self.assertTrue(same_state(restored.backend, control.backend))


if __name__ == "__main__":
    unittest.main()
