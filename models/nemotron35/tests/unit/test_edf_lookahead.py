"""Deadline-ordered read scheduling (EXP-I) and route-history lookahead (MOE-04/05)."""

import concurrent.futures as cf
import os
import threading
import time
import unittest
import unittest.mock

import mlx.core as mx

from models.nemotron35.streamed_experts import ExpertPool
from models.shared.edf_io import DeadlineExecutor
from models.shared.route_history import RouteHistory


class DeadlineExecutorTest(unittest.TestCase):
    def test_deadline_then_demand_order(self):
        pool = DeadlineExecutor(1, "test-edf")
        gate, started = threading.Event(), threading.Event()

        def blocker():
            started.set()
            gate.wait(5)
            return "blocker"

        first = pool.submit(blocker, deadline=0, demand=True)
        self.assertTrue(started.wait(2))
        order = []
        futures = [
            pool.submit(lambda: order.append("late-spec"), deadline=2, demand=False),
            pool.submit(lambda: order.append("same-demand"), deadline=1, demand=True),
            pool.submit(lambda: order.append("same-spec"), deadline=1, demand=False),
            pool.submit(lambda: order.append("early-spec"), deadline=0, demand=False),
        ]
        gate.set()
        self.assertEqual(first.result(2), "blocker")
        for future in futures:
            future.result(2)
        pool.shutdown()
        # The blocker already held the only worker, so the queued order is exactly the heap order.
        self.assertEqual(order, ["early-spec", "same-demand", "same-spec", "late-spec"])

    def test_cancel_stale_skips_queued_prefetch(self):
        pool = DeadlineExecutor(1, "test-edf-cancel")
        gate, started = threading.Event(), threading.Event()

        def blocker():
            started.set()
            gate.wait(5)
            return "blocker"

        first = pool.submit(blocker, deadline=0)
        self.assertTrue(started.wait(2))
        stale = pool.submit(lambda: "stale", deadline=1, demand=False)
        fresh = pool.submit(lambda: "fresh", deadline=2, demand=False)
        demand = pool.submit(lambda: "demand", deadline=2, demand=True)
        self.assertEqual(pool.cancel_stale(2), 1)
        gate.set()
        self.assertEqual(first.result(2), "blocker")
        with self.assertRaises(cf.CancelledError):
            stale.result(2)
        self.assertEqual(fresh.result(2), "fresh")
        self.assertEqual(demand.result(2), "demand")
        pool.shutdown()
        stats = pool.stats()
        self.assertGreaterEqual(stats["skipped"], 1)
        self.assertEqual(stats["failed"], 0)

    def test_shutdown_fails_pending(self):
        pool = DeadlineExecutor(1, "test-edf-stop")
        gate, started = threading.Event(), threading.Event()

        def blocker():
            started.set()
            gate.wait(5)
            return 1

        first = pool.submit(blocker)
        self.assertTrue(started.wait(2))
        pending = pool.submit(lambda: 2)
        pool.shutdown(wait=False, cancel_futures=True)
        gate.set()
        with self.assertRaises(cf.CancelledError):
            pending.result(2)
        self.assertEqual(first.result(2), 1)


    def blocked_pool(self, name):
        pool = DeadlineExecutor(1, name)
        gate, started = threading.Event(), threading.Event()

        def blocker():
            started.set()
            gate.wait(5)

        first = pool.submit(blocker, deadline=-1)
        self.assertTrue(started.wait(2))
        return pool, gate, first

    def test_not_held_by_default(self):
        pool = DeadlineExecutor(1, "test-edf-default")
        try:
            self.assertFalse(pool.stats()["held"])
            self.assertEqual(pool.submit(lambda: 1, demand=False).result(2), 1)
        finally:
            pool.shutdown()

    def test_hold_runs_demand_not_speculative(self):
        pool = DeadlineExecutor(1, "test-edf-hold")
        try:
            pool.hold_speculative(True)
            self.assertTrue(pool.stats()["held"])
            spec = pool.submit(lambda: "spec", deadline=0, demand=False)
            demand = pool.submit(lambda: "demand", deadline=5, demand=True)
            self.assertEqual(demand.result(2), "demand")
            # The single worker has since taken another demand item past the earlier speculative one.
            self.assertEqual(pool.submit(lambda: "again", deadline=9).result(2), "again")
            self.assertFalse(spec.done())
            self.assertEqual(pool.pending(), 1)
            pool.hold_speculative(False)
            self.assertEqual(spec.result(2), "spec")
            self.assertFalse(pool.stats()["held"])
        finally:
            pool.shutdown()

    def test_held_demand_runs_in_deadline_order(self):
        pool, gate, first = self.blocked_pool("test-edf-hold-demand")
        try:
            pool.hold_speculative(True)
            order = []
            futures = [pool.submit(lambda: order.append("spec0"), deadline=0, demand=False),
                       pool.submit(lambda: order.append("demand3"), deadline=3, demand=True),
                       pool.submit(lambda: order.append("demand1"), deadline=1, demand=True),
                       pool.submit(lambda: order.append("demand2"), deadline=2, demand=True)]
            gate.set()
            first.result(2)
            for future in futures[1:]:
                future.result(2)
            self.assertEqual(order, ["demand1", "demand2", "demand3"])
            self.assertFalse(futures[0].done())
            pool.hold_speculative(False)
            futures[0].result(2)
            self.assertEqual(order[-1], "spec0")
        finally:
            gate.set()
            pool.shutdown()

    def test_release_runs_speculative_in_deadline_order(self):
        pool, gate, first = self.blocked_pool("test-edf-release")
        try:
            pool.hold_speculative(True)
            order = []
            futures = [pool.submit(lambda d=d: order.append(d), deadline=d, demand=False) for d in (2, 0, 1)]
            pool.hold_speculative(False)
            gate.set()
            first.result(2)
            for future in futures:
                future.result(2)
            self.assertEqual(order, [0, 1, 2])
        finally:
            gate.set()
            pool.shutdown()

    def test_cancel_stale_works_while_held(self):
        pool = DeadlineExecutor(1, "test-edf-hold-cancel")
        try:
            pool.hold_speculative(True)
            stale = pool.submit(lambda: "stale", deadline=1, demand=False)
            fresh = pool.submit(lambda: "fresh", deadline=3, demand=False)
            layer = pool.submit(lambda: "layer", deadline=4, demand=False)
            self.assertEqual(pool.cancel_stale(2), 1)
            self.assertEqual(pool.cancel_layer(lambda deadline: deadline == 4), 1)
            self.assertEqual(pool.pending(), 1)
            with self.assertRaises(cf.CancelledError):
                stale.result(2)
            self.assertTrue(layer.cancelled())
            pool.hold_speculative(False)
            self.assertEqual(fresh.result(2), "fresh")
        finally:
            pool.shutdown()

    def test_shutdown_while_held_resolves_futures(self):
        pool = DeadlineExecutor(2, "test-edf-hold-stop")
        pool.hold_speculative(True)
        queued = [pool.submit(lambda: "never", deadline=d, demand=False) for d in (0, 1)]
        started = time.monotonic()
        pool.shutdown(wait=True)
        self.assertLess(time.monotonic() - started, 3)
        for future in queued:
            with self.assertRaises(RuntimeError):
                future.result(2)
        self.assertTrue(pool.stats()["held"])
        self.assertFalse(any(thread.is_alive() for thread in pool._threads))
        pool = DeadlineExecutor(1, "test-edf-hold-stop-cancel")
        pool.hold_speculative(True)
        cancelled = pool.submit(lambda: "never", deadline=1, demand=False)
        pool.shutdown(cancel_futures=True)
        self.assertTrue(cancelled.cancelled())
        self.assertFalse(any(thread.is_alive() for thread in pool._threads))


class RouteHistoryTest(unittest.TestCase):
    def test_observation_decay_and_prediction(self):
        history = RouteHistory([3], 8, decay=0.5)
        self.assertEqual(history.predict(3, 4), [])
        history.observe(3, [5, 5, 2, 1])
        history.observe(3, [5, 1, 1])
        # Expert 1 decayed to 0.5 then gained 2 (2.5); expert 5 decayed to 1 then gained 1 (2.0).
        self.assertEqual(history.predict(3, 3), [1, 5, 2])
        self.assertAlmostEqual(history.recall(3, [5, 2, 7], 2), 1 / 3)

    def test_unknown_layer_is_inert(self):
        history = RouteHistory([1], 4)
        history.observe(9, [0, 1])
        self.assertEqual(history.predict(9, 2), [])
        self.assertEqual(history.recall(9, [], 2), 1.0)


class FakeReader:
    expert_count, expert_bytes = 16, 512
    layers = (3, 4, 6)
    bytes_read = 0
    read_seconds = 0.0
    mapped, mapped_experts = False, 0     # the pool asks every reader whether experts are mapped

    def __init__(self, delay=0.0):
        self.delay = delay
        self.requests = []
        self._lock = threading.Lock()

    def submit_expert(self, executor, layer, expert, *, deadline=None, demand=True):
        def work():
            if self.delay:
                time.sleep(self.delay)
            with self._lock:
                self.requests.append((layer, expert, deadline, demand))
            return {"fc1.weight": mx.zeros((4,), dtype=mx.uint16),
                    "fc2.weight": mx.zeros((4,), dtype=mx.uint16)}

        if deadline is None:
            return executor.submit(work)
        return executor.submit(work, deadline=deadline, demand=demand)


class PoolIntegrationTest(unittest.TestCase):
    def test_route_history_prefetch_is_used(self):
        reader = FakeReader()
        with unittest.mock.patch.dict(os.environ, {"VEYSTRIX_NEMOTRON_LOOKAHEAD_LAYERS": "1",
                                                   "VEYSTRIX_NEMOTRON_EDF_IO": "0"}):
            pool = ExpertPool(reader, 1 << 20, workers=2, prefetch_workers=2)
        try:
            self.assertIsNotNone(pool.history)
            pool.observe_routes(4, [7, 7, 7, 1])
            self.assertEqual(pool.prefetch_history(4), 2)
            deadline = time.monotonic() + 2
            while not pool._staged and time.monotonic() < deadline:
                time.sleep(0.01)
            arrays = pool.fetch(4, [7, 1])
            self.assertEqual(set(arrays), {7, 1})
            stats = pool.stats()
            self.assertEqual(stats["lookahead"]["issued"], 2)
            self.assertEqual(stats["lookahead"]["used"], 2)
            self.assertTrue(all(request[2] is None for request in reader.requests))
        finally:
            pool.close()

    def test_deadline_executor_receives_layer_deadlines(self):
        reader = FakeReader()
        with unittest.mock.patch.dict(os.environ, {"VEYSTRIX_NEMOTRON_EDF_IO": "1",
                                                   "VEYSTRIX_NEMOTRON_LOOKAHEAD_LAYERS": "0"}):
            pool = ExpertPool(reader, 1 << 20, workers=1, prefetch_workers=1)
        try:
            self.assertIsNotNone(pool.edf)
            pool.prefetch(6, [2], history=False)
            pool.fetch(3, [4])
            self.assertIn("edf", pool.stats())
            seen = {(layer, expert, demand) for layer, expert, _deadline, demand in reader.requests}
            self.assertIn((3, 4, True), seen)
            self.assertIn((6, 2, False), seen)
            self.assertTrue(all(deadline is not None for _layer, _expert, deadline, _demand in reader.requests))
        finally:
            pool.close()


if __name__ == "__main__":
    unittest.main()
