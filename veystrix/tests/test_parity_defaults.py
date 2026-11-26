"""Shipped speed defaults must carry a real parity record.

The Flash-Next decode defaults (the generic qwen4_exp recipe's ``speed`` section) are exact by construction,
but a changed default is only trusted once a real-checkpoint parity run (tokens + float32 logit digests against
the frozen reference of the retired custom Qwen runtime, veystrix/parity_refs/) passed with exactly those
values. This test fails when a default changes and docs/performance/parity_defaults.json was not updated with
a passing run.
"""
import json
import re
import unittest
from pathlib import Path

from veystrix.port_check import FROZEN_REFS
from veystrix.recipe import QWEN4_EXP_SPEED

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "docs" / "performance" / "parity_defaults.json"


class ParityDefaultsTest(unittest.TestCase):
    def setUp(self):
        self.record = json.loads(RECORD.read_text())["flashnext"]

    def test_defaults_match_the_parity_record(self):
        msg = ("Flash-Next speed defaults changed without a parity record: run the real "
               "`python -m veystrix.port_check --checkpoint <Flash-Next>` at the new defaults and update "
               f"{RECORD.relative_to(ROOT)}")
        self.assertEqual(self.record["generic_speed"], QWEN4_EXP_SPEED, msg)

    def test_parity_log_passed(self):
        log = ROOT / self.record["parity_log"]
        if not log.is_file():
            self.skipTest("legacy parity log pruned")
        text = log.read_text()
        m = re.search(r"^(\d+)/(\d+) passed", text, re.M)
        self.assertIsNotNone(m, f"{log}: no 'N/N passed' line")
        self.assertEqual(m.group(1), m.group(2), f"{log}: not all checks passed")
        self.assertNotIn("FAIL ", text, f"{log}: a check failed")

    def test_frozen_reference_is_shipped(self):
        ref = ROOT / self.record["frozen_reference"]
        self.assertEqual(ref.parent, FROZEN_REFS)
        meta = json.loads(ref.read_text())
        for key in ("config_sha256", "lengths", "gen", "scenarios"):
            self.assertIn(key, meta)
        self.assertEqual(len(meta["scenarios"]), 6)


class FrozenReferenceTest(unittest.TestCase):
    def test_selected_by_config_bytes_lengths_and_gen(self):
        import hashlib
        import tempfile
        from unittest.mock import patch
        from veystrix import port_check
        with tempfile.TemporaryDirectory() as tmp:
            ckpt, refs = Path(tmp) / "ckpt", Path(tmp) / "refs"
            ckpt.mkdir(); refs.mkdir()
            (ckpt / "config.json").write_text('{"model_type": "qwen4_exp"}')
            sha = hashlib.sha256((ckpt / "config.json").read_bytes()).hexdigest()
            (refs / "a.json").write_text(json.dumps({"config_sha256": sha, "lengths": [16, 64], "gen": 16,
                                                     "scenarios": {}}))
            with patch.object(port_check, "FROZEN_REFS", refs):
                self.assertEqual(port_check.frozen_reference(ckpt, (16, 64), 16), refs / "a.json")
                self.assertIsNone(port_check.frozen_reference(ckpt, (16,), 16))
                self.assertIsNone(port_check.frozen_reference(ckpt, (16, 64), 8))
                (ckpt / "config.json").write_text('{"model_type": "qwen4_exp", "x": 1}')
                self.assertIsNone(port_check.frozen_reference(ckpt, (16, 64), 16))


if __name__ == "__main__":
    unittest.main()
