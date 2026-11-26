"""Exercise model-thread dispatch, priority, lifecycle and isolated HTTP state."""
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest

from veystrix import server
from veystrix.batch_scheduler import BatchScheduler
from veystrix.scheduler import Busy
from veystrix.sampling import Sampling
from veystrix.tests.test_server_runtime import CHAT, FakeBackend, FakeSession, request, wait_until


class CooperativeBackend(FakeBackend):
    def __init__(self, executed, entered, release):
        super().__init__(["ok"])
        self.executed = executed
        self.entered = entered
        self.release = release
        self.sampling = Sampling.greedy_settings()
        self.stops = set()
        self.tokenizer.decode = lambda ids: "".join(map(chr, ids))

    def fork_for_batch(self):
        return CooperativeBackend(self.executed, self.entered, self.release)

    def batch_context(self):
        return 4096

    def batch_state_bytes(self, tokens):
        return tokens

    def batch_prepare(self, tokens):
        pass

    def batch_prefix_plan(self, prompt, cached, width):
        return cached, None

    def batch_scope(self):
        return nullcontext()

    def batch_prefill(self, chunk, offset, final, sampler):
        self.executed.append(("prefill", threading.get_ident(), tuple(self.tape)))
        if not self.entered.is_set():
            self.entered.set()
            self.release.wait(5)
        return 65 if final else None

    def batch_decode(self, value, sampler, want_logits):
        self.executed.append(("decode", threading.get_ident(), tuple(self.tape)))
        return 65

    @staticmethod
    def batch_stats(**fields):
        from veystrix.backends.cpp_engine import Stats

        return Stats(**fields)


class BatchSchedulerTests(unittest.TestCase):
    def start(self, scheduler):
        stop = threading.Event()
        worker = threading.Thread(target=scheduler.serve_jobs, args=(stop,), daemon=True)
        worker.start()
        self.addCleanup(worker.join, 5)
        self.addCleanup(stop.set)
        self.assertTrue(wait_until(lambda: scheduler.loop_running))
        return worker

    def test_decode_priority_and_headroom_refusal(self):
        scheduler = BatchScheduler(headroom=lambda: 10)
        self.start(scheduler)
        first = scheduler.reserve("first", 4)
        second = scheduler.reserve("second", 4)
        self.addCleanup(scheduler.release, first)
        self.addCleanup(scheduler.release, second)
        with self.assertRaises(Busy):
            scheduler.reserve("refused", 4)
        active, release = threading.Event(), threading.Event()
        order = []
        with ThreadPoolExecutor() as callers:
            running = callers.submit(scheduler.submit_quantum, first, lambda: (active.set(), release.wait(3)), False)
            self.assertTrue(active.wait(3))
            decode = scheduler.reserve("decode", 1)
            self.addCleanup(scheduler.release, decode)
            prefill_call = callers.submit(scheduler.submit_quantum, second, lambda: order.append("prefill"), False)
            decode_call = callers.submit(scheduler.submit_quantum, decode, lambda: order.append("decode"), True)
            self.assertTrue(wait_until(lambda: scheduler.ready.pending() == 2))
            release.set()
            for call in (running, prefill_call, decode_call):
                call.result(timeout=3)
        self.assertEqual(order, ["decode", "prefill"])

    def test_ready_decode_rows_form_one_group_and_skip_cancelled_conversations(self):
        scheduler = BatchScheduler(headroom=lambda: 10)
        self.start(scheduler)
        states = [scheduler.reserve(str(index), 1) for index in range(4)]
        for state in states:
            self.addCleanup(scheduler.release, state)
        entered, release = threading.Event(), threading.Event()
        groups = []

        class GroupBackend:
            def batch_scope(self):
                return nullcontext()

            def batch_group_key(self):
                return self

            def batch_decode_group(self, steps):
                groups.append([step[1] for step in steps])
                return [step[1] + 10 for step in steps]

        backend = GroupBackend()
        with ThreadPoolExecutor() as callers:
            first = callers.submit(scheduler.submit_quantum, states[0],
                                   lambda: (entered.set(), release.wait(3)), False)
            self.assertTrue(entered.wait(3))
            waiting = [callers.submit(scheduler.submit_decode, states[index], (backend, index, None, True),
                                      lambda index=index: index == 2)
                       for index in range(1, 4)]
            try:
                self.assertTrue(wait_until(lambda: scheduler.ready.pending() == 3))
            finally:
                release.set()
            first.result(timeout=3)
            self.assertEqual(waiting[0].result(timeout=3)[0], 11)
            from veystrix.backends.base import GenerationCancelled
            with self.assertRaises(GenerationCancelled):
                waiting[1].result(timeout=3)
            self.assertEqual(waiting[2].result(timeout=3)[0], 13)
        self.assertEqual(groups, [[1, 3]])
        self.assertEqual(scheduler.snapshot()["batch"]["max_decode_group"], 2)

    def test_two_http_requests_keep_separate_tapes_on_one_executor(self):
        scheduler = BatchScheduler(headroom=lambda: 2**30)
        worker = self.start(scheduler)
        entered, release = threading.Event(), threading.Event()
        executed = []
        session = FakeSession(["ok"])
        session.backend = CooperativeBackend(executed, entered, release)
        model = server.ModelService(session)
        http = server.VeystrixHTTPServer(("127.0.0.1", 0), model, scheduler=scheduler)
        listener = threading.Thread(target=http.serve_forever, daemon=True)
        listener.start()
        self.addCleanup(http.server_close)
        self.addCleanup(http.shutdown)
        def ask(text):
            body = {**CHAT, "max_tokens": 2, "messages": [{"role": "user", "content": text}]}
            return request(http.server_port, "POST", "/v1/chat/completions", body)
        with ThreadPoolExecutor() as callers:
            first = callers.submit(ask, "first")
            if not entered.wait(3):
                self.fail(f"first request did not enter prefill: {first.result(timeout=3)}")
            second = callers.submit(ask, "second")
            try:
                self.assertTrue(wait_until(lambda: len(scheduler.admission.snapshot()["requests"]) == 2))
            finally:
                release.set()
            for call in (first, second):
                status, body, _ = call.result(timeout=5)
                self.assertEqual(status, 200, body)
        self.assertTrue(executed)
        self.assertEqual({tid for _, tid, _ in executed}, {worker.ident})
        self.assertEqual(session.backend.tape, [])
        self.assertEqual(scheduler.admission.used, 0)
