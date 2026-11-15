"""Decode prefetch: staging slots, deadline expiry, demand gate, hazards and teardown (fake reads, no model)."""
from __future__ import annotations

import json
import threading
import unittest

from models.flashnext.decode_prefetch import DecodePrefetch

WAIT = 10.0
FIRST = 100


class Frequency:
    def __init__(self, table):
        self.table = table

    def predict(self, layer, k, exclude=()):
        return list(self.table.get(layer, []))[:k]


class IO:
    """Fake record reads and copies; reads can be blocked or failed per key."""

    def __init__(self):
        self.cv = threading.Condition()
        self.started, self.finished, self.copies = [], [], []
        self.block, self.fail = {}, set()

    def read(self, layer, expert, slot):
        with self.cv:
            self.started.append((layer, expert, slot))
            self.cv.notify_all()
        gate = self.block.get((layer, expert))
        if gate is not None:
            assert gate.wait(WAIT), "test never released the read"
        if (layer, expert) in self.fail:
            raise OSError("read failed")
        with self.cv:
            self.finished.append((layer, expert, slot))
            self.cv.notify_all()

    def copy(self, source, destination):
        self.copies.append((source, destination))

    def wait(self, attribute, count):
        with self.cv:
            assert self.cv.wait_for(lambda: len(getattr(self, attribute)) >= count, WAIT), \
                f"only {len(getattr(self, attribute))} {attribute}, wanted {count}"


def make(io, table, slots=4, **kwargs):
    kwargs.setdefault("min_probability", 0.5)
    return DecodePrefetch(first_slot=FIRST, slots=slots, read_record=io.read, copy_record=io.copy,
                          frequency=Frequency(table), **kwargs)


def warm(prefetch):
    """Learn the layer chain 0 -> 1 -> 2 -> 3 -> 0 and stand at layer 1, step 6."""
    for layer in (0, 1, 2, 3, 0, 1):
        prefetch.observe(layer)


def not_resident(_key):
    return False


def invariant(test, prefetch):
    stats = prefetch.stats()
    test.assertEqual(stats["staged"] + stats["free_slots"] + stats["claimed"], stats["staging_slots"])
    return stats


class Base(unittest.TestCase):
    def setUp(self):
        self.io = IO()

    def start(self, table, **kwargs):
        prefetch = make(self.io, table, **kwargs)
        self.addCleanup(prefetch.close)
        return prefetch


class IssueTests(Base):
    def test_issues_nearest_first_filtered_and_within_staging_range(self):
        table = {2: [(7, 0.9), (8, 0.8), (9, 0.7), (1, 0.4)], 3: [(5, 0.95), (6, 0.6)]}
        prefetch = self.start(table, slots=8, width=2, depth=2)
        warm(prefetch)
        self.assertEqual(prefetch.issue(1, lambda key: key == (2, 7)), 4)   # (2,7) resident; width 2 per target
        self.io.wait("finished", 4)
        self.assertEqual({(layer, expert) for layer, expert, _ in self.io.finished},
                         {(2, 8), (2, 9), (3, 5), (3, 6)})
        self.assertTrue(all(FIRST <= slot < FIRST + 8 for *_, slot in self.io.finished))
        self.assertEqual(len({slot for *_, slot in self.io.finished}), 4)
        stats = invariant(self, prefetch)
        self.assertEqual((stats["issued"], stats["staged"], stats["free_slots"]), (4, 4, 4))

    def test_already_staged_keys_are_not_issued_twice(self):
        prefetch = self.start({2: [(7, 0.9)]}, depth=1)
        warm(prefetch)
        self.assertEqual(prefetch.issue(1, not_resident), 1)
        self.assertEqual(prefetch.issue(1, not_resident), 0)

    def test_nothing_is_issued_for_an_unknown_layer_order(self):
        prefetch = self.start({1: [(1, 0.9)], 2: [(1, 0.9)]})
        prefetch.observe(0)
        self.assertEqual(prefetch.issue(0, not_resident), 0)
        prefetch.observe(0)                      # repeats never teach a self-loop
        self.assertEqual(prefetch.issue(0, not_resident), 0)

    def test_probability_threshold(self):
        prefetch = self.start({2: [(1, 0.49)]}, depth=1)
        warm(prefetch)
        self.assertEqual(prefetch.issue(1, not_resident), 0)

    def test_stats_are_json_serializable(self):
        prefetch = self.start({2: [(1, 0.9)]}, depth=1, record_bytes=1000)
        warm(prefetch)
        prefetch.issue(1, not_resident)
        stats = json.loads(json.dumps(prefetch.stats()))
        self.assertEqual(stats["staging_bytes"], 4000)
        self.assertEqual(stats["settings"], {"depth": 1, "width": 2, "min_probability": 0.5, "workers": 2, "gate": True})
        self.assertEqual(stats["precision"], 0.0)


class PromotionTests(Base):
    def test_completed_read_is_copied_not_reread(self):
        prefetch = self.start({2: [(7, 0.9)]}, depth=1)
        warm(prefetch)
        prefetch.issue(1, not_resident)
        self.io.wait("finished", 1)
        entry = prefetch.claim((2, 7))
        self.assertIsNone(prefetch.claim((2, 7)))
        self.assertTrue(entry.ready)
        stats = invariant(self, prefetch)
        self.assertEqual((stats["staged"], stats["claimed"]), (0, 1))
        self.assertTrue(prefetch.finish(entry, 3))
        self.assertEqual(self.io.copies, [(entry.slot, 3)])
        prefetch.release(entry)
        prefetch.release(entry)                  # idempotent
        stats = invariant(self, prefetch)
        self.assertEqual((stats["used"], stats["late"], stats["free_slots"], stats["precision"]), (1, 0, 4, 1.0))
        self.assertEqual(len(self.io.started), 1)

    def test_running_read_is_awaited_then_copied(self):
        self.io.block[(2, 7)] = threading.Event()
        prefetch = self.start({2: [(7, 0.9)]}, depth=1)
        warm(prefetch)
        prefetch.issue(1, not_resident)
        self.io.wait("started", 1)
        entry = prefetch.claim((2, 7))
        waiting, original, result = threading.Event(), prefetch._cond.wait, []

        def wait(*args):
            waiting.set()
            return original(*args)

        prefetch._cond.wait = wait
        worker = threading.Thread(target=lambda: result.append(prefetch.finish(entry, 5)))
        worker.start()
        self.assertTrue(waiting.wait(WAIT))
        self.assertEqual(self.io.copies, [])     # the copy never starts before the read completes
        self.io.block[(2, 7)].set()
        worker.join(WAIT)
        self.assertEqual(result, [True])
        self.assertEqual(self.io.copies, [(entry.slot, 5)])
        prefetch.release(entry)
        stats = invariant(self, prefetch)
        self.assertEqual((stats["late"], stats["used"]), (1, 0))

    def test_queued_read_is_cancelled_not_awaited(self):
        prefetch = self.start({2: [(7, 0.9)]}, depth=1)
        prefetch.demand_begin()                  # the hold is what keeps the read queued
        warm(prefetch)
        prefetch.issue(1, not_resident)
        entry = prefetch.claim((2, 7))
        self.assertFalse(entry.ready)
        self.assertFalse(prefetch.finish(entry, 3))   # returns at once: never waits on a queued read
        prefetch.release(entry)
        stats = invariant(self, prefetch)
        self.assertEqual((stats["cancelled"], stats["free_slots"], stats["used"]), (1, 4, 0))
        prefetch.demand_end()
        self._sentinel(prefetch)
        self.assertEqual(self.io.started, [])    # the cancelled read never runs, so it never writes the slot

    def test_failed_read_frees_the_slot_and_is_not_claimable(self):
        self.io.fail.add((2, 7))
        prefetch = self.start({2: [(7, 0.9)]}, depth=1)
        warm(prefetch)
        prefetch.issue(1, not_resident)
        self.io.wait("started", 1)
        self._idle(prefetch)
        stats = invariant(self, prefetch)
        self.assertEqual((stats["failed"], stats["staged"], stats["free_slots"]), (1, 0, 4))
        self.assertIsNone(prefetch.claim((2, 7)))

    def test_failed_read_of_a_claimed_entry_sends_the_caller_to_a_direct_read(self):
        self.io.block[(2, 7)] = threading.Event()
        self.io.fail.add((2, 7))
        prefetch = self.start({2: [(7, 0.9)]}, depth=1)
        warm(prefetch)
        prefetch.issue(1, not_resident)
        self.io.wait("started", 1)
        entry = prefetch.claim((2, 7))
        self.io.block[(2, 7)].set()
        self.assertFalse(prefetch.finish(entry, 3))
        prefetch.release(entry)
        stats = invariant(self, prefetch)
        self.assertEqual((stats["failed"], stats["free_slots"], self.io.copies), (1, 4, []))

    def test_release_of_a_running_claim_waits_for_the_read(self):
        self.io.block[(2, 7)] = threading.Event()
        prefetch = self.start({2: [(7, 0.9)]}, depth=1)
        warm(prefetch)
        prefetch.issue(1, not_resident)
        self.io.wait("started", 1)
        entry = prefetch.claim((2, 7))
        prefetch.release(entry)                  # e.g. fill raised before submitting the copy task
        self.assertEqual(prefetch.stats()["free_slots"], 3)   # the read may still write this slot
        self.io.block[(2, 7)].set()
        self.io.wait("finished", 1)
        self._idle(prefetch)
        self.assertEqual(invariant(self, prefetch)["free_slots"], 4)

    @staticmethod
    def _sentinel(prefetch):
        """Run a demand item through the workers: proves they are awake, so anything runnable has run."""
        done = threading.Event()
        prefetch._executor.submit(done.set, deadline=0, demand=True)
        assert done.wait(WAIT)

    def _idle(self, prefetch):
        with prefetch._cond:
            self.assertTrue(prefetch._cond.wait_for(
                lambda: all(entry.state not in (0, 1) for entry in prefetch._live), WAIT))


class ExpiryTests(Base):
    def test_unclaimed_queued_predictions_are_cancelled_when_their_layer_passes(self):
        prefetch = self.start({2: [(7, 0.9)], 3: [(5, 0.9)]})
        prefetch.demand_begin()
        warm(prefetch)
        self.assertEqual(prefetch.issue(1, not_resident), 2)    # deadlines 7 (layer 2) and 8 (layer 3)
        prefetch.observe(2)
        self.assertEqual(invariant(self, prefetch)["expired"], 0)
        prefetch.observe(3)                                     # step 8: layer 2's deadline has passed
        stats = invariant(self, prefetch)
        self.assertEqual((stats["expired"], stats["staged"], stats["pending"]), (1, 1, 1))
        prefetch.observe(0)
        stats = invariant(self, prefetch)
        self.assertEqual((stats["expired"], stats["staged"], stats["pending"], stats["free_slots"]), (2, 0, 0, 4))
        prefetch.demand_end()
        self.assertEqual(self.io.started, [])

    def test_completed_prediction_rests_and_can_be_claimed_on_the_next_visit(self):
        prefetch = self.start({2: [(7, 0.9)]}, depth=1)
        warm(prefetch)
        prefetch.issue(1, not_resident)
        self.io.wait("finished", 1)
        for layer in (2, 3, 0, 1, 2):            # the deadline passed at the first of these
            prefetch.observe(layer)
        stats = invariant(self, prefetch)
        self.assertEqual((stats["staged"], stats["expired"]), (1, 0))
        entry = prefetch.claim((2, 7))
        self.assertTrue(prefetch.finish(entry, 1))
        prefetch.release(entry)
        self.assertEqual(prefetch.stats()["used"], 1)

    def test_running_read_survives_its_deadline_and_then_rests(self):
        self.io.block[(2, 7)] = threading.Event()
        prefetch = self.start({2: [(7, 0.9)]}, depth=1)
        warm(prefetch)
        prefetch.issue(1, not_resident)
        self.io.wait("started", 1)
        for layer in (2, 3):
            prefetch.observe(layer)
        stats = invariant(self, prefetch)
        self.assertEqual((stats["staged"], stats["expired"]), (1, 0))
        self.io.block[(2, 7)].set()
        self.io.wait("finished", 1)
        self.assertEqual(invariant(self, prefetch)["staged"], 1)

    def test_displacement_takes_oldest_completed_expired_entry_only(self):
        table = {2: [(1, 0.9), (2, 0.9)], 0: [(5, 0.9), (6, 0.9)]}
        prefetch = self.start(table, slots=2, depth=1, width=2)
        warm(prefetch)                                          # step 6
        self.assertEqual(prefetch.issue(1, not_resident), 2)    # both staging slots, deadline 7
        self.io.wait("finished", 2)
        self.assertEqual(prefetch.issue(1, not_resident), 0)    # nothing free, nothing expired
        prefetch.observe(2)                                     # step 7: deadline == step, still not passed
        prefetch.observe(3)                                     # step 8
        stats = invariant(self, prefetch)
        self.assertEqual((stats["staged"], stats["free_slots"]), (2, 0))
        self.assertEqual(prefetch.issue(3, not_resident), 2)
        stats = invariant(self, prefetch)
        self.assertEqual((stats["wasted"], stats["staged"], stats["free_slots"]), (2, 2, 0))
        self.io.wait("finished", 4)
        self.assertIsNone(prefetch.claim((2, 1)))
        self.assertIsNotNone(prefetch.claim((0, 5)))

    def test_running_and_future_deadline_entries_are_never_displaced(self):
        self.io.block[(2, 1)] = threading.Event()
        table = {2: [(1, 0.9)], 3: [(4, 0.9)], 0: [(5, 0.9), (6, 0.9)]}
        prefetch = self.start(table, slots=2, depth=2, width=1)
        warm(prefetch)
        self.assertEqual(prefetch.issue(1, not_resident), 2)    # (2,1) deadline 7 blocks; (3,4) deadline 8
        self.io.wait("started", 1)
        self.io.wait("finished", 1)                             # (3,4) completed
        for layer in (2, 3):
            prefetch.observe(layer)                             # step 8: (2,1) passed but running, (3,4) not passed
        self.assertEqual(prefetch.issue(3, not_resident), 0)
        self.assertEqual(prefetch.stats()["wasted"], 0)
        prefetch.observe(0)                                     # step 9: (3,4) passed and completed
        self.assertEqual(prefetch.issue(3, not_resident), 1)    # displaces (3,4), never the running (2,1)
        self.assertEqual(prefetch.stats()["wasted"], 1)
        self.assertIsNone(prefetch.claim((3, 4)))
        entry = prefetch.claim((2, 1))
        self.assertIsNotNone(entry)
        self.io.block[(2, 1)].set()
        prefetch.finish(entry, 0)
        prefetch.release(entry)


class GateTests(Base):
    @staticmethod
    def _sentinel(prefetch):
        done = threading.Event()
        prefetch._executor.submit(done.set, deadline=0, demand=True)
        assert done.wait(WAIT)

    def test_prefetch_reads_wait_for_outstanding_demand_reads(self):
        prefetch = self.start({2: [(7, 0.9)]}, depth=1, gate=True)
        warm(prefetch)
        prefetch.demand_begin()
        prefetch.issue(1, not_resident)
        self._sentinel(prefetch)                 # the workers are awake and idle for speculative work
        self.assertEqual(self.io.started, [])
        self.assertEqual(prefetch.stats()["pending"], 1)
        prefetch.demand_end()
        self.io.wait("started", 1)

    def test_gate_nests_by_count(self):
        prefetch = self.start({2: [(7, 0.9)]}, depth=1)
        warm(prefetch)
        prefetch.demand_begin()
        prefetch.demand_begin()
        prefetch.issue(1, not_resident)
        prefetch.demand_end()
        self._sentinel(prefetch)
        self.assertEqual(self.io.started, [])
        prefetch.demand_end()
        self.io.wait("started", 1)

    def test_ungated_prefetch_starts_immediately(self):
        prefetch = self.start({2: [(7, 0.9)]}, depth=1, gate=False)
        warm(prefetch)
        prefetch.demand_begin()
        prefetch.issue(1, not_resident)
        self.io.wait("started", 1)
        self.assertFalse(prefetch._executor.stats()["held"])
        prefetch.demand_end()


class CloseTests(Base):
    def test_close_waits_for_a_running_read_and_cancels_queued_ones(self):
        self.io.block[(2, 1)] = threading.Event()
        prefetch = self.start({2: [(1, 0.9), (2, 0.9), (3, 0.9)]}, depth=1, width=3, workers=1)
        warm(prefetch)
        self.assertEqual(prefetch.issue(1, not_resident), 3)
        self.io.wait("started", 1)
        closed = threading.Event()
        worker = threading.Thread(target=lambda: (prefetch.close(), closed.set()))
        worker.start()
        self.assertFalse(closed.wait(0.3))       # the blocked read has not finished, so close has not returned
        self.io.block[(2, 1)].set()
        self.assertTrue(closed.wait(WAIT))
        worker.join(WAIT)
        seen = list(self.io.started)
        self.assertEqual(seen, [(2, 1, seen[0][2])])             # queued reads never started
        self.assertEqual(len(self.io.finished), 1)
        self.assertEqual(prefetch.issue(1, not_resident), 0)
        self.assertIsNone(prefetch.claim((2, 2)))
        self.assertFalse(any(thread.is_alive() for thread in prefetch._executor._threads))
        prefetch.close()                         # idempotent


if __name__ == "__main__":
    unittest.main()
