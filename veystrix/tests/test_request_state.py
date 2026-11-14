"""Independent request state, admission cap and decode-first ordering."""

import unittest

from veystrix.request_state import (
    DECODE, PREFILL, AdmissionController, DecodeFirstQueue, RequestRegistry, RequestState, StateRefused)


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class AdmissionTest(unittest.TestCase):
    def test_cap_is_enforced_before_allocation(self):
        controller = AdmissionController(1000)
        first = RequestState("r1", "s1", kv_bytes_per_token=100)
        first.reserve(4)
        controller.admit(first)
        second = RequestState("r2", "s2", kv_bytes_per_token=100)
        second.reserve(7)
        with self.assertRaises(StateRefused):
            controller.admit(second)
        self.assertEqual(controller.refused, 1)
        with self.assertRaises(StateRefused):
            controller.grow("r1", 7)
        self.assertEqual(controller.used, 400)
        controller.grow("r1", 1)
        self.assertEqual(controller.used, 500)
        controller.release("r1")
        controller.admit(second)
        self.assertEqual(controller.used, 700)

    def test_duplicate_and_unknown_requests_are_refused(self):
        controller = AdmissionController(100)
        state = RequestState("r1", "s1")
        controller.admit(state)
        with self.assertRaises(StateRefused):
            controller.admit(state)
        with self.assertRaises(StateRefused):
            controller.grow("missing", 1)
        self.assertFalse(controller.cancel("missing"))
        self.assertTrue(controller.cancel("r1"))
        self.assertTrue(state.cancelled)


class DecodeFirstTest(unittest.TestCase):
    def test_decode_runs_before_a_fresh_prefill(self):
        clock = Clock()
        queue = DecodeFirstQueue(aging_seconds=1.0, clock=clock)
        prefill = RequestState("p", "sp")
        decode = RequestState("d", "sd")
        queue.enqueue(prefill, PREFILL)
        queue.enqueue(decode, DECODE)
        self.assertEqual(queue.next_ready()[0].request_id, "d")
        self.assertEqual(queue.next_ready()[0].request_id, "p")

    def test_long_waiting_prefill_is_lifted_once(self):
        clock = Clock()
        queue = DecodeFirstQueue(aging_seconds=1.0, clock=clock)
        prefill = RequestState("p", "sp")
        prefill.last_active = 0.0
        queue.enqueue(prefill, PREFILL)
        clock.value = 5.0
        decode = RequestState("d", "sd")
        decode.last_active = 5.0
        queue.enqueue(decode, DECODE)
        self.assertEqual(queue.next_ready()[0].request_id, "p")

    def test_cancelled_and_dropped_entries_are_skipped(self):
        queue = DecodeFirstQueue()
        state = RequestState("r", "s")
        queue.enqueue(state, DECODE)
        state.cancel()
        self.assertEqual(queue.next_ready(), (None, None))
        queue.enqueue(state, DECODE)
        self.assertTrue(queue.drop("r"))
        self.assertFalse(queue.drop("r"))
        self.assertEqual(queue.pending(), 0)


class RegistryTest(unittest.TestCase):
    def test_sessions_own_distinct_states_and_idle_eviction_is_bounded(self):
        registry = RequestRegistry(max_states=2)
        first = registry.create("a", "r1")
        second = registry.create("b", "r2")
        first.last_active, second.last_active = 0.0, 100.0
        self.assertEqual(registry.create("a", "r3"), first)   # same live session keeps its state
        third = registry.create("c", "r3")
        self.assertEqual(third.session_id, "c")
        self.assertEqual(registry.evicted, 1)
        self.assertIsNone(registry.get("a"))
        self.assertIsNotNone(registry.get("b"))
        self.assertIsNotNone(registry.get("c"))
        self.assertEqual(registry.snapshot()["max_states"], 2)


if __name__ == "__main__":
    unittest.main()
