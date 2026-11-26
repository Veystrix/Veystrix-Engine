"""Parked C sessions: a client alternating two conversations (qwen-code's side request) resumes the main one."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from veystrix.backends import cpp_engine as mod


class _Binding:
    has_v2 = True

    def __init__(self):
        self.opened, self.closed = 0, []

    def open_session(self, engine):
        self.opened += 1
        return f"s{self.opened}"

    def close_session(self, session):
        self.closed.append(session)


def _backend():
    b = mod.CppEngineBackend.__new__(mod.CppEngineBackend)
    b._binding, b._engine, b._closed, b._batch_mode = _Binding(), "e", False, False
    b._session, b.tape, b.pending, b._live_at, b._seed_applied = "s0", [], [], None, False
    return b


def _live(b, tokens):
    b.tape, b.pending, b._live_at = list(tokens), [], len(tokens)


_MEMORY_GATE = mod.CppEngineBackend._parking_memory_ok


class ParkedSessionTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(mod.CppEngineBackend, "_parking_memory_ok", staticmethod(lambda: True))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_side_request_does_not_evict_the_main_conversation(self):
        b = _backend()
        main = list(range(1000))
        _live(b, main)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(mod.PARKED_ENV, None)
            os.environ.pop(mod.PARKED_TOKENS_ENV, None)
            b.reset()                                    # a side request arrives: main is parked, not closed
            self.assertEqual(b._binding.closed, [])
            _live(b, [7] * 300)                          # the side request ran
            turn2 = main + [5, 6, 7]
            self.assertEqual(b.parked_prefix(turn2), 1000)
            self.assertEqual(b.activate_parked(turn2), 1000)   # main comes back; the side session is parked in turn
            self.assertEqual(b.parked_prefix([7] * 300 + [1]), 300)
        self.assertEqual(b._session, "s0")
        self.assertEqual((b.tape, b._live_at), (main, 1000))
        self.assertEqual(b._binding.closed, [])          # nothing closed: one slot, swapped

    def test_limits_and_off_switch(self):
        b = _backend()
        with patch.dict(os.environ, {mod.PARKED_ENV: "0"}):
            _live(b, range(1000))
            b.reset()
            self.assertEqual(b._binding.closed, ["s0"])
        b = _backend()
        with patch.dict(os.environ, {mod.PARKED_ENV: "1", mod.PARKED_TOKENS_ENV: "5000"}):
            _live(b, range(1000))
            b.reset()
            _live(b, range(2000, 3000))
            b.reset()                                    # count 1: the older parked session is closed
            self.assertEqual(b._binding.closed, ["s0"])
            _live(b, range(10))                          # too short to be worth parking
            b.reset()
            self.assertIn("s2", b._binding.closed)
            self.assertEqual(b.parked_prefix(list(range(1000)) + [1]), 0)
            self.assertEqual(b.parked_prefix(list(range(2000, 3000)) + [1]), 1000)

    def test_low_memory_parks_nothing_and_releases_what_is_parked(self):
        b = _backend()
        _live(b, range(1000))
        b.reset()
        self.assertEqual(b._binding.closed, [])
        with patch.object(mod.CppEngineBackend, "_parking_memory_ok", staticmethod(lambda: False)):
            _live(b, range(2000, 2500))
            b.reset()
        self.assertEqual(b._binding.closed, ["s0", "s1"])
        self.assertEqual(b.parked_prefix(list(range(1001))), 0)

    def test_memory_gate_reads_available_memory(self):
        with patch("veystrix.bench.memory.sample", return_value={"available_bytes": 3 * 2**30}):
            with patch.dict(os.environ, {mod.PARKED_MIN_AVAILABLE_ENV: "4"}):
                self.assertFalse(_MEMORY_GATE())
            with patch.dict(os.environ, {mod.PARKED_MIN_AVAILABLE_ENV: "2"}):
                self.assertTrue(_MEMORY_GATE())

    def test_a_diverged_session_is_never_parked(self):
        b = _backend()
        _live(b, range(1000))
        b._live_at = None                                 # cancelled mid-decode: C state != tape
        b.reset()
        self.assertEqual(b._binding.closed, ["s0"])
        self.assertEqual(b.parked_prefix(list(range(1001))), 0)


if __name__ == "__main__":
    unittest.main()
