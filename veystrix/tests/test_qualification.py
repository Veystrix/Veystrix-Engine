from pathlib import Path
import json
import tempfile
import unittest

from veystrix.bench.qualification import FLOOR, assess, guard_passed


class QualificationTests(unittest.TestCase):
    def test_gate_requires_five_guarded_exact_runs_and_stream_percentiles(self):
        good = {
            "output_tokens": 512, "cache_consistent": True, "available_floor_bytes": FLOOR,
            "swap_growth_bytes": 0, "swapout_growth_bytes": 0, "guard_passed": True,
            "token_sha256": "reference", "stream_decode_tok_s": 21,
        }
        def check(rows, count=512, digest="reference"):
            return assess(rows, output_tokens=count, expected_digest=digest)["passed"]

        self.assertTrue(check([good] * 5))
        self.assertFalse(check([good] * 4))
        self.assertFalse(check([good] * 5, count=128))
        self.assertFalse(check([good] * 5, digest=None))
        for bad in (
            {"token_sha256": "changed"}, {"guard_passed": False},
            {"stream_decode_tok_s": 11}, {"stream_decode_tok_s": float("nan")},
            {"swap_growth_bytes": 1}, {"available_floor_bytes": FLOOR - 1},
        ):
            with self.subTest(bad=bad):
                self.assertFalse(check([good] * 5 + [good | bad]))

    def test_terminal_guard_result_is_required(self):
        start = {"event": "start", "refusal": None, "power": {"ac": True, "low_power": False}}
        end = {"event": "end", "stopped": None, "returncode": 0, "available_floor_gib": 3.5,
               "swapout_mib": 0, "swap_peak_mib": 1, "swap_baseline_mib": 1, "thermal_seconds": {"0": 1}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guard.jsonl"
            for terminal, expected in ((None, False), (end, True), (end | {"stopped": "floor"}, False)):
                path.write_text("\n".join(json.dumps(row) for row in (start, terminal) if row))
                self.assertEqual(guard_passed(path), expected)
