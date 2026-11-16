"""Guards for the active documentation index and decision log."""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


class DocsHygieneTests(unittest.TestCase):
    def test_index_status_is_current(self):
        index = (DOCS / "README.md").read_text()
        self.assertIn("2026-11-16", index)
        self.assertIn("two models are supported", index)
        for stale in ("updated 2026-11-10", "branch `qwen3.8-flash-next`",
                      "branch `unlimited-generation`", "branch `overnight-speed"):
            self.assertNotIn(stale, index)

    def test_index_links_resolve(self):
        index = (DOCS / "README.md").read_text()
        for target in re.findall(r"\]\(([^)#]+\.md)\)", index):
            if "://" in target:
                continue
            self.assertTrue((DOCS / target).is_file(), f"broken doc link: {target}")

    def test_decision_log_records_the_model_decisions(self):
        log = (DOCS / "decisions" / "README.md").read_text()
        for number in ("ADR-001", "ADR-002", "ADR-003", "ADR-004", "ADR-005"):
            self.assertIn(number, log)
        self.assertIn("until Qwen 4 Flash is validated", log)

    def test_validation_checklist_has_no_removed_models_or_stale_excuse(self):
        text = (DOCS / "release" / "VALIDATION.md").read_text()
        for removed in ("qwen27b", "K2", "Bonsai", "OWNER TRAVELLING"):
            self.assertNotIn(removed, text)
        self.assertIn("988", text)
        self.assertIn("2,289", text)

    def test_release_process_explains_the_update_button_limitation(self):
        text = (DOCS / "release" / "RELEASE_PROCESS.md").read_text()
        self.assertIn("In-app \"Check for Updates\"", text)
        self.assertIn("appcast.xml", text)
        self.assertIn("does not exist yet", text)

    def test_historical_performance_handoffs_are_marked(self):
        for name in ("handoff-2026-11-14.md", "inference-blueprint-handoff.md",
                     "overnight-2026-11-16.md"):
            self.assertIn("Historical record", (DOCS / "performance" / name).read_text())


if __name__ == "__main__":
    unittest.main()
