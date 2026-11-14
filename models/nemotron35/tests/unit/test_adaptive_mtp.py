"""Adaptive native-MTP draft depth on the tiny synthetic checkpoint."""
import itertools
import os
import shutil
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

from models.nemotron35.backend_core import EngineCore
from models.nemotron35.runtime import Cancelled
from models.nemotron35.streamed_model import load_streamed
from models.nemotron35.tiny_checkpoint import permissive_config_check, write_checkpoint

FLAG = "VEYSTRIX_NEMOTRON_MTP_ADAPTIVE"
VOCAB = 64
PROMPT = [3, 9, 14, 20, 7, 31, 2, 18, 40, 11, 5, 27]
_dir = None


def setUpModule():
    global _dir
    _dir = tempfile.mkdtemp(prefix="nemotron-adaptive-")
    write_checkpoint(Path(_dir) / "ck")


def tearDownModule():
    shutil.rmtree(_dir, ignore_errors=True)


class OracleHead:
    """Stands in for the MTP head: proposes the true continuation.

    ``reject(round, depth_index)`` returns True to corrupt that draft. Round
    boundaries are the ``rewind`` calls the draft loop makes in its ``finally``.
    """

    def __init__(self, truth, emitted, reject=lambda round_, index: False):
        self.truth, self.emitted, self.reject = truth, emitted, reject
        self.kv = SimpleNamespace(offset=0)
        self.round = 0
        self.index = 0
        self.draft_counts = []

    def reset(self):
        self.kv.offset = 0

    def rewind(self, offset):
        self.kv.offset = min(self.kv.offset, offset)
        self.draft_counts.append(self.index)
        self.index = 0
        self.round += 1

    def forward(self, rows, tokens, *, predict=False):
        self.kv.offset += int(tokens.shape[0])
        if not predict:
            return None
        position = len(self.emitted) + self.index + 1
        token = self.truth[position] if position < len(self.truth) else 0
        if self.reject(self.round, self.index):
            token = (token + 1) % VOCAB
        self.index += 1
        return token, rows[-1:]


class AdaptiveMTPTests(unittest.TestCase):
    def core(self):
        model = load_streamed(Path(_dir) / "ck", pool_bytes=0, config_check=permissive_config_check)
        self.addCleanup(model.close)
        return EngineCore(model, chunk_tokens=4)

    @classmethod
    def greedy(cls, count):
        """Plain greedy decode of PROMPT, one token at a time, no MTP."""
        model = load_streamed(Path(_dir) / "ck", pool_bytes=0, config_check=permissive_config_check)
        try:
            core = EngineCore(model, chunk_tokens=4)
            token = int(mx.argmax(core.prefill(PROMPT), axis=-1).item())
            tokens = []
            for _ in range(count):
                tokens.append(token)
                token = int(mx.argmax(core.decode(token), axis=-1).item())
            return tokens
        finally:
            model.close()

    def start(self, truth, emitted, *, reject=lambda round_, index: False, adaptive=False, depth=4):
        core = self.core()
        core.mtp = OracleHead(truth, emitted, reject)
        flags = {FLAG: "1" if adaptive else "0"}
        with unittest.mock.patch.dict(os.environ, flags):
            core.begin_mtp_turn(True)
        first = mx.argmax(core.prefill(PROMPT), axis=-1)
        return core, first

    def drive(self, core, first, horizon, emitted, stops=(), depth=4, cancelled=lambda: False):
        for token in core.native_mtp_tokens(first, horizon, set(stops), cancelled, depth):
            emitted.append(token)
        return emitted

    def spy(self, core, depth=4):
        """Record every observation the controller receives."""
        controller = core._mtp_controller(depth)
        seen = []
        original = controller.observe

        def observe(*args, **kwargs):
            seen.append((args, kwargs))
            return original(*args, **kwargs)

        controller.observe = observe
        return controller, seen

    def test_flag_off_asks_for_the_configured_depth_every_round(self):
        truth, emitted = self.greedy(40), []
        core, first = self.start(truth, emitted, adaptive=False)
        self.drive(core, first, 30, emitted, depth=3)
        self.assertEqual(emitted, truth[:30])
        self.assertIsNone(core._mtp_controller_state)
        self.assertFalse(core.mtp_adaptive)
        self.assertIsNone(core.mtp_adaptive_stats())
        counts = core.mtp.draft_counts
        # Every full round drafts exactly the depth; only the last rounds are cut by the horizon.
        self.assertEqual(counts[:5], [3] * 5)
        self.assertTrue(all(count <= 3 for count in counts))

    def test_flag_on_matches_plain_greedy_and_uses_several_depths(self):
        truth, emitted = self.greedy(80), []
        # Occasional corrupted drafts at varying positions give the controller mixed evidence.
        reject = lambda round_, index: round_ % 5 == 4 and index == round_ % 3
        core, first = self.start(truth, emitted, reject=reject, adaptive=True)
        self.drive(core, first, 70, emitted)
        self.assertEqual(emitted, truth[:70])
        stats = core.mtp_adaptive_stats()
        counts = stats["depth_counts"]
        self.assertIn(0, counts)
        self.assertGreater(len(counts), 1)
        self.assertTrue(all(0 <= depth <= 4 for depth in counts))
        self.assertTrue(core.mtp_adaptive)

    def test_forced_schedule_matches_plain_greedy(self):
        truth = self.greedy(80)
        for reject in (lambda round_, index: False, lambda round_, index: index == round_ % 4):
            emitted = []
            core, first = self.start(truth, emitted, reject=reject, adaptive=True)
            controller, seen = self.spy(core)
            with unittest.mock.patch.object(controller, "choose", side_effect=itertools.cycle(range(5))):
                self.drive(core, first, 60, emitted)
            self.assertEqual(emitted, truth[:60])
            self.assertEqual({args[0] for args, _ in seen}, {0, 1, 2, 3, 4})

    def test_forced_mismatch_at_each_position_records_accepted_and_repair(self):
        truth = self.greedy(60)
        for depth in range(1, 5):
            for position in range(depth):
                with self.subTest(depth=depth, position=position):
                    emitted = []
                    reject = lambda round_, index, p=position: index == p
                    core, first = self.start(truth, emitted, reject=reject, adaptive=True, depth=depth)
                    controller, seen = self.spy(core, depth)
                    with unittest.mock.patch.object(controller, "choose", return_value=depth):
                        self.drive(core, first, 24, emitted, depth=depth)
                    self.assertEqual(emitted, truth[:24])
                    full = [(args, kwargs) for args, kwargs in seen if args[0] == depth]
                    self.assertTrue(full)
                    for args, kwargs in full:
                        self.assertEqual(args[1], position)
                        self.assertGreater(kwargs["repair_seconds"], 0.0)
                        self.assertGreater(kwargs["verify_seconds"], 0.0)
                        self.assertGreater(kwargs["draft_seconds"], 0.0)

    def test_full_acceptance_reports_no_repair(self):
        truth, emitted = self.greedy(30), []
        core, first = self.start(truth, emitted, adaptive=True)
        controller, seen = self.spy(core)
        with unittest.mock.patch.object(controller, "choose", return_value=2):
            self.drive(core, first, 12, emitted)
        self.assertEqual(emitted, truth[:12])
        for args, kwargs in seen:
            if args[0] == 2:
                self.assertEqual(args[1], 2)
                self.assertEqual(kwargs["repair_seconds"], 0.0)

    def test_shortened_drafts_observe_the_drafted_depth(self):
        truth = self.greedy(40)
        # A stop token that is a draft ends the draft loop early; the stop is also the final token.
        index = next(i for i in range(12, len(truth)) if truth[i] not in truth[:i])
        stop = truth[index]
        emitted = []
        core, first = self.start(truth, emitted, adaptive=True)
        controller, seen = self.spy(core)
        verified = []
        original = core.verify_decode

        def verify(token, drafts, **kwargs):
            verified.append(len(drafts))
            return original(token, drafts, **kwargs)

        core.verify_decode = verify
        with unittest.mock.patch.object(controller, "choose", return_value=4):
            self.drive(core, first, 30, emitted, stops={stop})
        self.assertEqual(emitted, truth[:index])
        depths = [args[0] for args, _ in seen]
        self.assertEqual([d for d in depths if d], verified)
        self.assertTrue(all(d <= 4 for d in depths))
        self.assertTrue(any(d < 4 for d in verified))
        # The horizon also limits the draft: never more than horizon - produced - 1.
        emitted = []
        core, first = self.start(truth, emitted, adaptive=True)
        controller, seen = self.spy(core)
        with unittest.mock.patch.object(controller, "choose", return_value=4):
            self.drive(core, first, 7, emitted)
        self.assertEqual(emitted, truth[:7])
        produced = 0
        for args, _ in seen:
            self.assertLessEqual(args[0], 7 - produced - 1 if args[0] else 4)
            produced += args[1] + 1

    def test_plain_step_at_the_horizon_is_not_observed(self):
        truth, emitted = self.greedy(10), []
        core, first = self.start(truth, emitted, adaptive=True)
        controller, seen = self.spy(core)
        with unittest.mock.patch.object(controller, "choose", return_value=0):
            self.drive(core, first, 5, emitted)
        self.assertEqual(emitted, truth[:5])
        # Five rounds, but the last plain step skips logits and measures nothing.
        self.assertEqual(len(seen), 4)
        for args, kwargs in seen:
            self.assertEqual(args[:2], (0, 0))
            self.assertGreater(kwargs["verify_seconds"], 0.0)

    def test_exception_in_verification_is_not_observed_and_cache_is_restored(self):
        truth, emitted = self.greedy(20), []
        core, first = self.start(truth, emitted, adaptive=True)
        controller, seen = self.spy(core)
        before_tokens = core.cached_tokens
        before_state = [array.tolist() for array in core.state_arrays()]
        with unittest.mock.patch.object(controller, "choose", return_value=2), \
                unittest.mock.patch.object(core, "_segment_layer_major", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self.drive(core, first, 10, emitted)
        self.assertEqual(seen, [])
        self.assertEqual(core.cached_tokens, before_tokens)
        self.assertEqual([array.tolist() for array in core.state_arrays()], before_state)

    def test_cancellation_during_verification_is_not_observed(self):
        truth, emitted = self.greedy(20), []
        core, first = self.start(truth, emitted, adaptive=True)
        controller, seen = self.spy(core)
        calls = []

        def cancelled():
            calls.append(1)
            return len(calls) > 1

        with unittest.mock.patch.object(controller, "choose", return_value=2):
            with self.assertRaises(Cancelled):
                self.drive(core, first, 10, emitted, cancelled=cancelled)
        self.assertEqual(seen, [])

    def test_controller_survives_turns_and_is_dropped_by_reset(self):
        truth, emitted = self.greedy(30), []
        core, first = self.start(truth, emitted, adaptive=True)
        self.drive(core, first, 10, emitted)
        controller = core._mtp_controller_state[1]
        with unittest.mock.patch.dict(os.environ, {FLAG: "1"}):
            core.begin_mtp_turn(True)
        self.assertIs(core._mtp_controller_state[1], controller)
        # A changed ceiling builds a fresh controller.
        self.assertIsNot(core._mtp_controller(3), controller)
        self.assertEqual(core._mtp_controller(3).max_depth, 3)
        # The flag is read per turn: turning it off drops the controller.
        with unittest.mock.patch.dict(os.environ, {FLAG: "0"}):
            core.begin_mtp_turn(True)
        self.assertIsNone(core._mtp_controller_state)
        with unittest.mock.patch.dict(os.environ, {FLAG: "1"}):
            core.begin_mtp_turn(True)
        core._mtp_controller(4)
        core.reset()
        self.assertIsNone(core._mtp_controller_state)

    def test_verify_decode_reports_the_round_split(self):
        truth = self.greedy(10)
        core = self.core()
        first = int(mx.argmax(core.prefill(PROMPT), axis=-1).item())
        committed, following = core.verify_decode(first, truth[1:4], cancelled=lambda: False)
        self.assertEqual(committed, truth[:4])
        accepted, verify_seconds, repair_seconds = core.last_round
        self.assertEqual((accepted, repair_seconds), (3, 0.0))
        self.assertGreater(verify_seconds, 0.0)
        core.reset()
        first = int(mx.argmax(core.prefill(PROMPT), axis=-1).item())
        committed, following = core.verify_decode(first, [(truth[1] + 1) % VOCAB, 0], cancelled=lambda: False)
        self.assertEqual(core.last_round[0], len(committed) - 1)
        self.assertEqual(core.last_round[0], 0)
        self.assertGreater(core.last_round[2], 0.0)


if __name__ == "__main__":
    unittest.main()
