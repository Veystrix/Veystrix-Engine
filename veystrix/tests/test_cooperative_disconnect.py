"""A client that disconnects at any point of a cooperative request must not disturb anyone else.

Checkpoint-free: a deterministic fake backend (every conversation owns a small running state, the model
object shares a prefix store and records how its scope is used) is driven through the real
``veystrix.backends.cooperative.generate`` and the real ``BatchScheduler.serve_jobs`` loop. The model loop runs
on a worker thread, request threads call ``reserve / submit_quantum / submit_decode / release`` exactly as
``server.py`` does.

The disconnect is injected at every boundary instead of at one: ``should_cancel`` turns true at the k-th check
(before the prefix plan, before each prefill quantum, before each decode token) and the request-thread callbacks
(``on_prefill_progress``, ``on_prefilled``, ``out``, ``on_decode_token``) raise at every step. After each run
the reservation is released, nothing is left queued, the other conversations in the same decode group produced
the tokens they produce alone, the shared prefix store holds only what a cold prefill produces, and the next
request on the same scheduler completes with its control tokens.
"""
from __future__ import annotations

from contextlib import contextmanager
from itertools import count, permutations
from types import SimpleNamespace
import threading
import time
import unittest

from veystrix.backends.base import GenerationCancelled
from veystrix.backends.cooperative import PREFILL_TOKENS, generate
from veystrix.batch_scheduler import BatchScheduler
from veystrix.sampling import Sampling
from veystrix.scheduler import ClientGone

ANCHOR = PREFILL_TOKENS          # the fake's system boundary, one chunk
PROMPT_TOKENS = 300              # chunks of 128 / 128 / 44 with a cold anchor
OUTPUT_TOKENS = 6
MASK = (1 << 61) - 1
BLOCKERS = count()


def fold(state, tokens):
    for token in tokens:
        state = (state * 1000003 + int(token) + 1) & MASK
    return state


def following(state):
    return 5 + state % 50


def prompt_for(member, *, shared_system=False):
    system = [(5 + (j * 7) % 50) if shared_system else (5 + (member * 11 + j * 7) % 50) for j in range(ANCHOR)]
    tail = [5 + (member * 13 + j * 3 + 1) % 50 for j in range(PROMPT_TOKENS - ANCHOR)]
    return system + tail


def wait_until(check, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.002)
    return False


class FakeModel:
    """Everything the forks share: the prefix store and an audit of scope use."""

    def __init__(self):
        self.prefix = {}
        self.depth = 0
        self.outside_scope = []
        self.group_sizes = []
        self.fail_decode = None          # (state seen by the forward, error) to raise once from a decode forward
        self.lock = threading.Lock()


class FakeBackend:
    def __init__(self, model):
        self.model = model
        self.state = 0
        self.pending, self.tape = [], []
        self.last_prefix = None
        self.turn_closed = True
        self.sampling = Sampling.greedy_settings()
        self.stops = set()
        self.tokenizer = SimpleNamespace(decode=lambda ids: "".join(chr(0x4e00 + int(i)) for i in ids))
        self.prefill_calls = 0
        self.decode_calls = 0

    # -- the cooperative surface -----------------------------------------------------------------------
    def batch_context(self):
        return 4096

    def batch_state_bytes(self, tokens):
        return tokens

    def batch_prepare(self, tokens):
        self._in_scope("prepare")

    def batch_prefix_plan(self, prompt, cached, width):
        self._in_scope("plan")
        if cached or len(prompt) <= ANCHOR:
            return cached, None
        key = tuple(prompt[:ANCHOR])
        with self.model.lock:
            saved = self.model.prefix.get(key)
        if saved is None:
            return cached, ANCHOR
        self.state = saved
        self.tape.extend(self.pending[:ANCHOR])
        del self.pending[:ANCHOR]
        return ANCHOR, ANCHOR

    @contextmanager
    def batch_scope(self):
        with self.model.lock:
            self.model.depth += 1
        try:
            yield
        finally:
            with self.model.lock:
                self.model.depth -= 1

    def batch_prefill(self, chunk, offset, final, sampler):
        self._in_scope("prefill")
        assert offset == len(self.tape), (offset, len(self.tape))
        self.prefill_calls += 1
        self.state = fold(self.state, chunk)
        return following(self.state) if final else None

    def batch_save_prefix(self):
        self._in_scope("save")
        with self.model.lock:
            self.model.prefix[tuple(self.tape)] = self.state

    def batch_decode(self, value, sampler, want_logits):
        self._in_scope("decode")
        self.decode_calls += 1
        with self.model.lock:
            failure = self.model.fail_decode
            if failure is not None and failure[0] is self:
                self.model.fail_decode = None
                raise failure[1]
        self.state = fold(self.state, [value])
        return following(self.state) if want_logits else None

    @staticmethod
    def batch_stats(**fields):
        return SimpleNamespace(**fields)

    def _in_scope(self, what):
        with self.model.lock:
            if self.model.depth != 1:
                self.model.outside_scope.append((what, self.model.depth))


class GroupedBackend(FakeBackend):
    """What Nemotron offers: ready decode rows of one model run as one group."""

    def batch_decode_group(self, steps):
        with self.model.lock:
            self.model.group_sizes.append(len(steps))
        return [backend.batch_decode(token, sampler, want) for backend, token, sampler, want in steps]

    def batch_group_key(self):
        return self.model


class CancelAt:
    """``should_cancel`` that turns (and stays) true from its k-th call on."""

    def __init__(self, k):
        self.k, self.calls, self.lock = k, 0, threading.Lock()

    def __call__(self):
        with self.lock:
            index, self.calls = self.calls, self.calls + 1
        return self.k is not None and index >= self.k


class Injected(Exception):
    """A callback-side disconnect (the server raises ClientGone / a safety stop from ``out``)."""


@contextmanager
def hold(scheduler):
    """Occupy the model thread with a quantum that only ends when the block exits."""
    blocker = scheduler.reserve(f"blocker-{next(BLOCKERS)}", 1)
    entered, release = threading.Event(), threading.Event()
    holder = threading.Thread(
        target=scheduler.submit_quantum, args=(blocker, lambda: (entered.set(), release.wait(10)), False))
    holder.start()
    try:
        if not entered.wait(10):
            raise AssertionError("the model thread never entered the blocker")
        yield
    finally:
        release.set()
        holder.join(10)
        scheduler.release(blocker)


class Gate:
    """Hold every member at decode step ``step`` until the scheduler can form one group of them."""

    def __init__(self, scheduler, step, order):
        self.scheduler, self.step, self.order = scheduler, step, list(order)
        self.arrived = {name: threading.Event() for name in order}
        self.go = {name: threading.Event() for name in order}

    def before(self, name, index):
        if index == self.step and name in self.go:
            self.arrived[name].set()
            if not self.go[name].wait(10):
                raise AssertionError(f"{name} was never released at step {index}")

    def drive(self, queued=lambda: None):
        """Queue the members' decode quanta behind a held model thread, in ``order``; then release them."""
        scheduler = self.scheduler
        for name in self.order:
            if not self.arrived[name].wait(10):
                raise AssertionError(f"{name} never reached decode step {self.step}")
        with hold(scheduler):
            for count, name in enumerate(self.order, 1):
                self.go[name].set()
                if not wait_until(lambda: scheduler.ready.pending() == count):
                    raise AssertionError(f"{name} did not queue its decode quantum")
            queued()


class Member(threading.Thread):
    """One request thread, written the way ``ModelService`` drives a cooperative request."""

    def __init__(self, scheduler, backend, name, prompt, tokens, *, cancel=None, gate=None, hooks=None):
        super().__init__(daemon=True)
        self.scheduler, self.backend, self.label = scheduler, backend, name
        self.prompt, self.limit = prompt, tokens
        self.cancel = cancel or CancelAt(None)
        self.gate = gate
        self.hooks = hooks or {}
        self.tokens, self.error, self.text, self.reserved = [], None, None, None

    def run(self):
        scheduler, backend = self.scheduler, self.backend
        backend.pending.extend(self.prompt)
        decode_index = [0]
        try:
            state = self.reserved = scheduler.reserve(
                f"{self.label}-{id(self)}", backend.batch_state_bytes(len(self.prompt) + self.limit))
        except BaseException as error:
            self.error = error
            return

        def decode_token(token, piece):
            self.tokens.append(token)
            self._hook("on_decode_token", len(self.tokens))

        def out(piece):
            self._hook("out", len(self.tokens))

        def arrive():
            index, decode_index[0] = decode_index[0], decode_index[0] + 1
            if self.gate is not None:
                self.gate.before(self.label, index)

        def dispatch_decode(args, cancel):
            arrive()
            return scheduler.submit_decode(state, args, cancel)

        def dispatch(work, decode):
            if decode:                       # singleton backends queue their decode steps as plain quanta
                arrive()
            return scheduler.submit_quantum(state, work, decode)

        progress = [0]

        def prefill_progress(done, total):
            progress[0] += 1
            self._hook("on_prefill_progress", progress[0])

        try:
            self.text, self.stats = generate(
                backend, self.limit,
                dispatch=dispatch, dispatch_decode=dispatch_decode, out=out, on_decode_token=decode_token,
                on_prefilled=lambda: self._hook("on_prefilled", 0),
                on_prefill_progress=prefill_progress, should_cancel=self.cancel)
        except BaseException as error:
            self.error = error
        finally:
            scheduler.release(state)

    def _hook(self, name, count):
        trigger = self.hooks.get(name)
        if trigger is not None and trigger == count:
            raise Injected(f"{name} at {count}")


class CooperativeCase(unittest.TestCase):
    backend_class = GroupedBackend
    controls = {}

    def setUp(self):
        self.loop = None
        self.fresh()
        self.addCleanup(self.stop_loop)

    def fresh(self):
        """A new model, prefix store and scheduler (each sub-case starts from nothing)."""
        self.stop_loop()
        self.model = FakeModel()
        self.scheduler = BatchScheduler(headroom=lambda: 1 << 40)
        self.stop = threading.Event()
        self.loop = threading.Thread(target=self.scheduler.serve_jobs, args=(self.stop,), daemon=True)
        self.loop.start()
        self.assertTrue(wait_until(lambda: self.scheduler.loop_running))

    def stop_loop(self):
        if self.loop is not None:
            self.stop.set()
            self.scheduler._wake.set()
            self.loop.join(5)
            alive, self.loop = self.loop.is_alive(), None
            self.assertFalse(alive, "the model loop did not stop")

    # -- helpers ---------------------------------------------------------------------------------------
    def member(self, name, prompt, tokens=OUTPUT_TOKENS, **options):
        return Member(self.scheduler, self.backend_class(self.model), name, prompt, tokens, **options)

    def control(self, prompt, tokens=OUTPUT_TOKENS):
        """What this prompt emits alone on an untouched scheduler and an empty prefix store."""
        key = (self.backend_class, tuple(prompt), tokens)
        if key not in CooperativeCase.controls:
            model, scheduler, stop = FakeModel(), BatchScheduler(headroom=lambda: 1 << 40), threading.Event()
            loop = threading.Thread(target=scheduler.serve_jobs, args=(stop,), daemon=True)
            loop.start()
            self.assertTrue(wait_until(lambda: scheduler.loop_running))
            try:
                alone = Member(scheduler, self.backend_class(model), "control", prompt, tokens)
                alone.start()
                alone.join(20)
                self.assertIsNone(alone.error)
            finally:
                stop.set()
                scheduler._wake.set()
                loop.join(5)
            expected, state = [], fold(0, prompt)
            for _ in range(tokens):                      # the fake's own arithmetic, computed directly
                expected.append(following(state))
                state = fold(state, [expected[-1]])
            self.assertEqual(alone.tokens, expected)
            CooperativeCase.controls[key] = expected
        return CooperativeCase.controls[key]

    def run_all(self, members, gate=None):
        for member in members:
            member.start()
        if gate is not None:
            gate.drive()
        for member in members:
            member.join(20)
            self.assertFalse(member.is_alive(), f"{member.label} never finished")

    def assert_clean(self, *, followup=None):
        scheduler, model = self.scheduler, self.model
        self.assertEqual(scheduler.admission.used, 0)
        self.assertEqual(scheduler.snapshot()["batch"]["requests"], [])
        self.assertEqual(scheduler.ready.pending(), 0)
        self.assertEqual(scheduler._quanta, {})
        self.assertFalse(scheduler.busy)
        self.assertEqual(model.depth, 0)
        self.assertEqual(model.outside_scope, [])
        for key, state in model.prefix.items():          # only what a cold prefill produces
            self.assertEqual(state, fold(0, key))
        for index, prompt in enumerate(followup or ()):
            after = self.member(f"after{index}", prompt)
            self.run_all([after])
            self.assertIsNone(after.error)
            self.assertEqual(after.tokens, self.control(prompt))
        self.assertEqual(scheduler.admission.used, 0)
        self.assertEqual(scheduler._quanta, {})
        self.assertEqual(model.outside_scope, [])
        for key, state in model.prefix.items():
            self.assertEqual(state, fold(0, key))

    def checks(self):
        """(``should_cancel`` evaluations, prefill quanta) of one uninterrupted cold request."""
        probe = CancelAt(None)
        alone = self.member("probe", prompt_for(99), cancel=probe)
        self.run_all([alone])
        self.assertIsNone(alone.error)
        self.assertEqual(probe.calls, 1 + alone.backend.prefill_calls + OUTPUT_TOKENS)
        self.fresh()
        return probe.calls, alone.backend.prefill_calls

    def survivors_match(self, members):
        for member in members:
            self.assertIsNone(member.error, member.label)
            self.assertEqual(member.tokens, self.control(member.prompt), member.label)

    @staticmethod
    def phase(k, quanta):
        if k == 0:
            return "before the prefix plan"
        if k <= quanta:
            return f"before prefill quantum {k}"
        if k == quanta + 1:
            return "after the last prefill quantum, before decode token 0"
        return f"before decode token {k - quanta - 1}"


class SchedulerBoundaryTests(CooperativeCase):
    """Cancellation as the scheduler sees it, at each of its boundaries."""

    def test_every_cancel_check_of_a_lone_request(self):
        total, quanta = self.checks()
        self.assertEqual(quanta, 3)
        for k in range(total + 1):
            with self.subTest(k=k, phase=self.phase(k, quanta)):
                self.fresh()
                victim = self.member("victim", prompt_for(0, shared_system=True), cancel=CancelAt(k))
                self.run_all([victim])
                reference = self.control(victim.prompt)
                if k < total:
                    self.assertIsInstance(victim.error, GenerationCancelled)
                    self.assertEqual(victim.tokens, reference[:max(0, k - quanta - 1)])
                else:
                    self.assertIsNone(victim.error)
                    self.assertEqual(victim.tokens, reference)
                self.assert_clean(followup=[victim.prompt, prompt_for(1, shared_system=True)])

    def test_cancel_during_prefill_with_other_conversations_interleaved(self):
        total, quanta = self.checks()
        for k in range(quanta + 1):
            with self.subTest(k=k, phase=self.phase(k, quanta)):
                self.fresh()
                victim = self.member("victim", prompt_for(0), cancel=CancelAt(k))
                others = [self.member(f"other{i}", prompt_for(i)) for i in (1, 2)]
                self.run_all([victim, *others])
                self.assertIsInstance(victim.error, GenerationCancelled)
                self.assertEqual(victim.tokens, [])
                self.survivors_match(others)
                self.assert_clean(followup=[victim.prompt, prompt_for(3)])

    def test_cancel_at_every_decode_step_inside_a_group_of_three_at_every_position(self):
        total, quanta = self.checks()
        for step in range(OUTPUT_TOKENS):
            for order in permutations(("victim", "left", "right")):
                with self.subTest(step=step, order=order):
                    self.fresh()
                    prompts = {"victim": prompt_for(0), "left": prompt_for(1), "right": prompt_for(2)}
                    members = {name: self.member(
                        name, prompts[name],
                        cancel=CancelAt(quanta + 1 + step if name == "victim" else None)) for name in order}
                    gate = Gate(self.scheduler, step, order)
                    for member in members.values():
                        member.gate = gate
                    self.run_all(list(members.values()), gate)
                    victim = members["victim"]
                    self.assertIsInstance(victim.error, GenerationCancelled)
                    self.assertEqual(victim.tokens, self.control(victim.prompt)[:step])
                    self.survivors_match([members["left"], members["right"]])
                    self.assert_clean(followup=[victim.prompt, prompt_for(3)])

    def test_group_forced_without_a_disconnect_matches_controls(self):
        """The forcing protocol itself: a three-conversation group forms and is exact."""
        if self.backend_class is not GroupedBackend:
            self.skipTest("singleton backends never form a group")
        names = ("a", "b", "c")
        members = {name: self.member(name, prompt_for(index)) for index, name in enumerate(names)}
        gate = Gate(self.scheduler, 2, names)
        for member in members.values():
            member.gate = gate
        self.run_all(list(members.values()), gate)
        self.survivors_match(members.values())
        self.assertIn(3, self.model.group_sizes)
        self.assertEqual(self.scheduler.snapshot()["batch"]["max_decode_group"], 3)
        self.assert_clean()

    def test_cancel_while_the_quantum_waits_in_the_queue(self):
        """The flag flips after the quantum was queued and before the model thread reached it."""
        flag = threading.Event()
        victim = self.member("victim", prompt_for(0), cancel=flag.is_set)
        survivor = self.member("survivor", prompt_for(1))
        with hold(self.scheduler):
            victim.start()
            survivor.start()
            self.assertTrue(wait_until(lambda: self.scheduler.ready.pending() == 2))
            flag.set()
        for member in (victim, survivor):
            member.join(20)
            self.assertFalse(member.is_alive())
        self.assertIsInstance(victim.error, GenerationCancelled)
        self.assertEqual(victim.backend.tape, [])           # its queued quantum never ran
        self.assertEqual(victim.backend.prefill_calls, 0)
        self.survivors_match([survivor])
        self.assert_clean(followup=[victim.prompt])

    def test_cancel_while_decode_is_queued_in_a_group(self):
        """Both decode quanta are queued as one group; the flag flips before the group runs."""
        if self.backend_class is not GroupedBackend:
            self.skipTest("singleton backends never form a group")
        flag = threading.Event()
        victim = self.member("victim", prompt_for(0), cancel=flag.is_set)
        other = self.member("other", prompt_for(1))
        gate = Gate(self.scheduler, 3, ("victim", "other"))
        victim.gate = other.gate = gate
        victim.start()
        other.start()
        gate.drive(queued=flag.set)
        for member in (victim, other):
            member.join(20)
            self.assertFalse(member.is_alive())
        self.assertIsInstance(victim.error, GenerationCancelled)
        self.assertEqual(victim.tokens, self.control(victim.prompt)[:3])
        self.survivors_match([other])
        self.assert_clean(followup=[victim.prompt])

    def test_callbacks_that_raise_at_every_step(self):
        total, quanta = self.checks()
        points = [("on_prefill_progress", n) for n in range(1, quanta + 1)]
        points += [("on_prefilled", 0)]
        points += [(hook, n) for hook in ("out", "on_decode_token") for n in range(1, OUTPUT_TOKENS + 1)]
        for hook, number in points:
            with self.subTest(hook=hook, number=number):
                self.fresh()
                victim = self.member("victim", prompt_for(0), hooks={hook: number})
                others = [self.member(f"other{i}", prompt_for(i)) for i in (1, 2)]
                self.run_all([victim, *others])
                self.assertIsInstance(victim.error, Injected)
                reference = self.control(victim.prompt)
                self.assertEqual(victim.tokens, reference[:len(victim.tokens)])
                self.survivors_match(others)
                self.assert_clean(followup=[victim.prompt, prompt_for(3)])

    def test_prefix_stored_before_the_disconnect_equals_a_cold_prefill(self):
        """The victim saves the system anchor in its first quantum; later requests restore it exactly."""
        total, quanta = self.checks()
        for k in (2, 3, quanta + 1, total - 1):
            with self.subTest(k=k):
                self.fresh()
                victim = self.member("victim", prompt_for(0, shared_system=True), cancel=CancelAt(k))
                self.run_all([victim])
                self.assertIsInstance(victim.error, GenerationCancelled)
                self.assertEqual(list(self.model.prefix), [tuple(victim.prompt[:ANCHOR])])
                self.assert_clean()
                restored = self.member("restored", prompt_for(1, shared_system=True))
                self.run_all([restored])
                self.assertEqual(restored.backend.prefill_calls, 2)     # 300 tokens less the 128 restored
                self.assertIsNone(restored.error)
                self.assertEqual(restored.tokens, self.control(restored.prompt))
                self.assert_clean()

    def test_prefix_is_not_stored_when_the_disconnect_precedes_the_anchor(self):
        for k in (0, 1):
            with self.subTest(k=k):
                self.fresh()
                victim = self.member("victim", prompt_for(0, shared_system=True), cancel=CancelAt(k))
                self.run_all([victim])
                self.assertIsInstance(victim.error, GenerationCancelled)
                self.assertEqual(self.model.prefix, {})
                self.assert_clean(followup=[victim.prompt])

    def test_a_failing_forward_reaches_exactly_the_conversations_that_were_in_it(self):
        """An engine error is not a disconnect: every row of the failed group gets it, nobody else does."""
        victim = self.member("victim", prompt_for(0))
        others = [self.member(f"other{i}", prompt_for(i)) for i in (1, 2)]
        gate = Gate(self.scheduler, 2, ("victim", "other1", "other2"))
        for member in (victim, *others):
            member.gate = gate
            member.start()
        gate.drive(queued=lambda: setattr(self.model, "fail_decode", (victim.backend, RuntimeError("forward failed"))))
        for member in (victim, *others):
            member.join(20)
            self.assertFalse(member.is_alive())
        grouped = self.backend_class is GroupedBackend
        self.assertIsInstance(victim.error, RuntimeError)
        self.assertEqual(victim.tokens, self.control(victim.prompt)[:2])
        for member in others:
            if grouped:
                self.assertIsInstance(member.error, RuntimeError)
                self.assertEqual(member.tokens, self.control(member.prompt)[:2])
            else:
                self.assertIsNone(member.error)
                self.assertEqual(member.tokens, self.control(member.prompt))
        self.assertIsNone(self.model.fail_decode)
        self.assert_clean(followup=[victim.prompt, prompt_for(1)])

    def test_a_cancelled_group_member_does_not_shift_results_of_the_others(self):
        """Direct check of the result-to-row mapping with the first and a middle row cancelled."""
        if self.backend_class is not GroupedBackend:
            self.skipTest("singleton backends never form a group")
        scheduler = self.scheduler
        states = [scheduler.reserve(name, 1) for name in "wxyz"]
        results, errors = {}, {}

        def ask(index, state):
            fork = self.backend_class(self.model)
            fork.state = index
            try:
                results[index] = scheduler.submit_decode(
                    state, (fork, index + 100, SimpleNamespace(observe=lambda token: None), True),
                    lambda: index in (0, 2))
            except BaseException as error:
                errors[index] = error

        threads = []
        with hold(scheduler):
            for index, state in enumerate(states):
                thread = threading.Thread(target=ask, args=(index, state))
                thread.start()
                threads.append(thread)
                self.assertTrue(wait_until(lambda: scheduler.ready.pending() == index + 1))
        for thread in threads:
            thread.join(10)
            self.assertFalse(thread.is_alive())
        for state in states:
            scheduler.release(state)
        self.assertEqual(set(errors), {0, 2})
        for error in errors.values():
            self.assertIsInstance(error, GenerationCancelled)
        for index in (1, 3):
            self.assertEqual(results[index][0], following(fold(index, [index + 100])))
        self.assertEqual(self.model.group_sizes, [2])
        self.assert_clean()


class SingletonBoundaryTests(SchedulerBoundaryTests):
    """The same sweeps for a backend without ``batch_decode_group`` (the Flash-Next shape)."""

    backend_class = FakeBackend


if __name__ == "__main__":
    unittest.main()
