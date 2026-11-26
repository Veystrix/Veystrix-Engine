"""Offline tests for the in-TUI /update command (curl and Homebrew install paths)."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from veystrix.tool.harness_adapters import ReplayAdapter
from veystrix.tool.tui import slash
from veystrix.tests.test_tui_fixtures import THEMES, make_loop

RELEASE = {"tag_name": "v9.9.9", "published_at": "2026-11-22T00:00:00Z", "prerelease": True,
           "assets": [{"name": "SHA256SUMS.txt", "browser_download_url": "https://x/SUMS"},
                      {"name": "SHA256SUMS.txt.minisig", "browser_download_url": "https://x/SIG"},
                      {"name": "veystrix-engine-9.9.9-arm64.tar.gz",
                       "browser_download_url": "https://x/a.tar.gz"}]}


class SlashUpdateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        (self.home / ".veystrix").mkdir()
        (self.home / ".veystrix" / "engine").mkdir()
        (self.home / ".veystrix" / "engine" / "BUNDLE-MANIFEST.json").write_text(
            json.dumps({"version": "1.1.0"}))

    def test_check_only_reports_installed_and_latest(self):
        with patch.dict(os.environ, {"VEYSTRIX_TEST_RELEASES": json.dumps(RELEASE)}):
            lines = slash("/update --check", home=self.home, port=1)
        text = "\n".join(lines)
        self.assertIn("installed 1.1.0", text)
        self.assertIn("latest 9.9.9", text)
        self.assertIn("veystrix update", text)

    def test_brew_install_runs_brew_update_and_upgrade(self):
        calls: list[list[str]] = []

        class Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def fake_run(argv, **kwargs):
            calls.append(list(argv))
            return Result()

        with patch.dict(os.environ, {"VEYSTRIX_INSTALLER": "brew"}), \
                patch("shutil.which", return_value="/opt/homebrew/bin/brew"), \
                patch("veystrix.tool.tui.subprocess.run", fake_run):
            lines = slash("/update", home=self.home, port=1)
        self.assertEqual(calls, [["brew", "update"], ["brew", "upgrade", "veystrix"]])
        self.assertIn("Homebrew upgrade finished", "\n".join(lines))

    def test_ui_runs_update_without_blocking_the_screen(self):
        seen = []

        def fake_slash(command, *, home, port):
            seen.append(command)
            return ["  ◆ Veystrix · latest 9.9.9"]

        loop, _, _, _ = make_loop(ReplayAdapter(), slash_fn=fake_slash)
        loop.step(0.0, list("/update --check") + ["enter"])
        self.assertFalse(loop.state.busy)
        self.assertEqual(seen, ["/update --check"])
        # Background result is delivered on the UI thread on the next step.
        self.assertTrue(any("checking for updates" in e.text for e in loop.state.entries))
        for _ in range(20):
            loop.step(0.1)
        self.assertTrue(any("latest 9.9.9" in e.text for e in loop.state.entries))


if __name__ == "__main__":
    unittest.main()
