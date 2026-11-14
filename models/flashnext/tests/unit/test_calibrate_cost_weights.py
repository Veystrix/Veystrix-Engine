import contextlib
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from models.flashnext.tests.bench import calibrate_cost_weights as C
from models.shared.cost_aware import CostWeights

TINY_GRID = {"recent": (0.0, 0.1), "frequency": (0.2, 1.0), "reuse": (1.0,), "alpha": (0.3, 0.5)}


def synthetic_events():
    """Three turns of 12 tokens over 3 layers; a hot core of experts plus a drifting tail."""
    events = []
    for turn in range(3):
        for token in range(12):
            for layer in range(3):
                events.append([turn, "decode", token, layer, [0, 1, 2 + (token + layer + turn) % 9]])
    return events


def write_trace(folder):
    path = Path(folder) / "trace.json.gz"
    with gzip.open(path, "wt") as handle:
        json.dump({"record_bytes": 1_000_000, "events": synthetic_events()}, handle)
    return path


class SearchTests(unittest.TestCase):
    def test_calibrate_returns_valid_weights_and_never_scores_below_the_start(self):
        events = [event for event in synthetic_events() if event[0] != 2]
        weights, trajectory = C.calibrate(events, [6, 9], grid=TINY_GRID, passes=2)
        self.assertEqual(CostWeights.from_mapping(weights.to_mapping()), weights)
        self.assertEqual((weights.refill, weights.size), (0.0, 0.0))
        start = trajectory[0]
        self.assertEqual((start["pass"], start["weights"]), (0, CostWeights().to_mapping()))
        best = max(row["mean"] for row in trajectory)
        self.assertGreaterEqual(best, start["mean"])
        chosen = C.evaluate(events, [6, 9], weights)
        self.assertAlmostEqual(sum(chosen) / 2, best, 12)

    def test_search_skips_the_current_value_and_covers_every_coordinate(self):
        events = [event for event in synthetic_events() if event[0] == 0]
        _, trajectory = C.calibrate(events, [6], grid=TINY_GRID, passes=1)
        tried = {(row["coordinate"], row["value"]) for row in trajectory[1:]}
        self.assertEqual(tried, {("recent", 0.0), ("frequency", 1.0), ("alpha", 0.5)})  # defaults 0.1, 0.2, 1.0, 0.3 skipped

    def test_flat_objective_keeps_the_defaults(self):
        events = [[0, "decode", t, 0, [t % 2]] for t in range(10)]      # a pool this large always hits after warm-up
        weights, _ = C.calibrate(events, [50], grid=TINY_GRID, passes=1)
        self.assertEqual(weights, CostWeights())


class DecisionTests(unittest.TestCase):
    @staticmethod
    def holdout(default, calibrated):
        return {"default": {"mean": default}, "calibrated": {"mean": calibrated}}

    def test_threshold_is_half_a_percentage_point(self):
        self.assertFalse(C.decide(self.holdout(0.40, 0.4049))["write_weights"])
        self.assertTrue(C.decide(self.holdout(0.40, 0.4051))["write_weights"])
        self.assertFalse(C.decide(self.holdout(0.40, 0.39))["write_weights"])

    def test_holdout_report_has_every_reference(self):
        report = C.holdout_report(synthetic_events(), [6], CostWeights())
        self.assertEqual(set(report), {"default", "calibrated", "lru", "s3fifo", "least-stale", "belady"})
        self.assertEqual(report["default"]["per_size"], report["calibrated"]["per_size"])
        self.assertLessEqual(report["lru"]["mean"], report["belady"]["mean"] + 1e-12)


class MainTests(unittest.TestCase):
    def run_main(self, folder, minimum):
        out, report = Path(folder) / "weights.json", Path(folder) / "calibration.json"
        with mock.patch.object(C, "GRID", TINY_GRID), mock.patch.object(C, "PASSES", 1), \
                mock.patch.object(C, "MIN_GAIN", minimum), \
                mock.patch.object(C, "output_path", return_value=report) as target, \
                contextlib.redirect_stdout(io.StringIO()) as text:
            C.main([str(write_trace(folder)), "--gb", "0.006", "0.009", "--out", str(out)])
        target.assert_called_once_with("flashnext", "cost-weights-calibration", "calibration.json")
        return out, json.loads(report.read_text()), text.getvalue()

    def test_nothing_is_written_when_the_gain_is_under_the_threshold(self):
        with tempfile.TemporaryDirectory() as folder:
            out, report, text = self.run_main(folder, 2.0)
            self.assertFalse(out.exists())
            self.assertFalse(report["decision"]["write_weights"])
            self.assertIsNone(report["out"])
            self.assertIn("the defaults stand", text)

    def test_weights_are_written_when_the_gain_is_over_the_threshold(self):
        with tempfile.TemporaryDirectory() as folder:
            out, report, text = self.run_main(folder, -1.0)
            self.assertTrue(report["decision"]["write_weights"])
            self.assertEqual(CostWeights.load(out), CostWeights.from_mapping(report["weights"]))
            self.assertEqual(report["out"], str(out))

    def test_report_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            _, report, _ = self.run_main(folder, 2.0)
        for key in ("trajectory", "train", "holdout", "decision", "weights", "grid", "trace_sha256", "record_bytes",
                    "train_turns", "holdout_turns", "slots", "gb"):
            self.assertIn(key, report)
        self.assertEqual((report["train_turns"], report["holdout_turns"], report["slots"]), ([0, 1], [2], [6, 9]))

    def test_overlapping_turns_are_refused(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                C.main([str(write_trace(folder)), "--train-turns", "0", "1", "--holdout-turns", "1"])


if __name__ == "__main__":
    unittest.main()
