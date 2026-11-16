"""The generated results index must stay reproducible and non-destructive."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.results.index_citations import ROOT, citation_counts, render


class ResultsIndexTests(unittest.TestCase):
    def test_counts_only_existing_run_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "results" / "model" / "run-1").mkdir(parents=True)
            (root / "docs" / "a.md").write_text(
                "see results/model/run-1 and results/model/run-1 again, "
                "plus results/model and results/model/missing")
            counts = citation_counts(root)
        self.assertEqual(counts, {"results/model/run-1": 2})

    def test_render_is_a_markdown_table(self):
        from collections import Counter
        text = render(Counter({"results/model/run-1": 3}))
        self.assertIn("## Cited runs", text)
        self.assertIn("| 3 |", text)
        self.assertIn("results/model/run-1", text)
        self.assertIn("does", text)  # the non-destructive promise

    def test_committed_index_is_present_and_current(self):
        index = ROOT / "results" / "INDEX.md"
        self.assertTrue(index.is_file())
        self.assertEqual(index.read_text(), render(citation_counts()))
