"""The one-command verification entry points must stay in sync with CI."""
from __future__ import annotations

import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class VerificationEntrypointTests(unittest.TestCase):
    def test_verify_script_covers_every_suite(self):
        script = ROOT / "scripts" / "verify.sh"
        self.assertTrue(script.is_file())
        self.assertTrue(os.access(script, os.X_OK))
        text = script.read_text()
        for fragment in ("veystrix/tests", "secret_scan.py",
                         "swift build", "swift test", "linux_unit_tests.py"):
            self.assertIn(fragment, text)

    def test_make_targets_exist(self):
        makefile = (ROOT / "Makefile").read_text()
        for target in ("verify:", "fast:", "python-only:", "scan:"):
            self.assertIn(target, makefile)

    def test_pull_request_template_keeps_the_safety_gate(self):
        template = (ROOT / ".github" / "pull_request_template.md").read_text()
        self.assertIn("./scripts/verify.sh", template)
        self.assertIn("preflight profile", template)
        self.assertIn("result folders", template)

    def test_artifact_guardrails_are_a_first_class_check(self):
        script = ROOT / "scripts" / "verify.sh"
        self.assertIn("--artifacts", script.read_text())
        guard = (ROOT / "scripts" / "verify_artifacts.py").read_text()
        for fragment in ("no plaintext bytecode", "no product source in bundle",
                         "shield dylib present", "integrity manifest pins files",
                         "naive decompile fails", "key still bound to shield digest"):
            self.assertIn(fragment, guard)
        self.assertIn("artifacts:", (ROOT / "Makefile").read_text())

    def test_contributing_points_at_the_one_command(self):
        self.assertIn("./scripts/verify.sh", (ROOT / "CONTRIBUTING.md").read_text())


if __name__ == "__main__":
    unittest.main()
