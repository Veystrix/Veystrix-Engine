"""Offline tests for the Veystrix-branded white-label TUI (M3b).

No real harness binary or real harness config is used. A fake harness process emits
harness banners, telemetry lines, tool calls and streamed tokens; the TUI must show only
Veystrix branding. The engine endpoint is the in-process mock server.
"""
from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from integrations.hermes.mock_veystrix_server import MockVeystrixServer
from veystrix import cli
from veystrix.tool.tui import (CHECKED, DEFAULT_HARNESS, WHITELABEL, banner,
                               branding_clean, format_tool_call, launch_command,
                               run_once, slash, tui, whitelabel_environment)

ROOT = Path(__file__).resolve().parents[2]
FAKE_HARNESS = '''\
import os, sys, time
print("Welcome to OpenCode v2.2.2 - the open source agent")
print("Claude Code 2.1.304 | telemetry enabled | update available: v3")
print("Sign in to your anthropic account to continue")
print("read_file('README.md')")
print("edit_file('README.md', text='hello world')")
print("run_shell('python -m unittest')")
if "--resume" in sys.argv:
    print("resume=" + sys.argv[sys.argv.index("--resume") + 1])
print("DO_NOT_TRACK=" + os.environ.get("DO_NOT_TRACK", ""))
print("DISABLE_TELEMETRY=" + os.environ.get("DISABLE_TELEMETRY", ""))
print("streaming", flush=True)
if os.environ.get("VEYSTRIX_FAKE_HANG"):
    time.sleep(30)
'''


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TerminalTuiTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.script = self.home / "fake_harness.py"
        self.script.write_text(FAKE_HARNESS)
        self.env = dict(os.environ)
        self.env.update({"VEYSTRIX_HOME": str(self.home),
                         "VEYSTRIX_MODEL_ROOT": str(self.home / "models"),
                         "VEYSTRIX_NO_KEYCHAIN": "1"})
        self._patcher = patch.dict(os.environ, self.env, clear=False)
        self._patcher.start()
        self.port = _free_port()
        self.mock = MockVeystrixServer(port=self.port).start()
        self.command = [sys.executable, str(self.script)]

    def tearDown(self):
        self.mock.stop()
        self._patcher.stop()
        self._tmp.cleanup()

    def test_whitelabel_flags_are_doc_verified(self):
        self.assertEqual(CHECKED, "2026-11-18")
        self.assertEqual(DEFAULT_HARNESS, "opencode")
        for name, spec in WHITELABEL.items():
            self.assertTrue(spec["doc"].startswith("https://"), name)
            self.assertTrue(spec["note"], name)
        self.assertEqual(WHITELABEL["opencode"]["settings"]["update"], "disable")
        self.assertEqual(WHITELABEL["claude-code"]["env"]["DISABLE_TELEMETRY"], "1")
        self.assertIn("--no-telemetry", WHITELABEL["qwen-code"]["flags"])
        self.assertTrue(WHITELABEL["pi"]["settings"]["quietStartup"])
        self.assertFalse(WHITELABEL["pi"]["settings"]["enableInstallTelemetry"])
        env = whitelabel_environment("claude-code", self.port, "veystrix-flashnext", "local")
        # Claude Code appends /v1/messages to its base URL itself.
        self.assertEqual(env["ANTHROPIC_BASE_URL"], f"http://127.0.0.1:{self.port}")
        self.assertEqual(env["DISABLE_UPDATES"], "1")

    def test_branding_filter_drops_harness_surface(self):
        self.assertIsNone(branding_clean("OpenCode v2.2.2"))
        self.assertIsNone(branding_clean("New version available"))
        self.assertIsNone(branding_clean("telemetry: anonymous usage enabled"))
        self.assertIn("Veystrix", branding_clean("model from opencode is ready"))

    def test_tui_shows_only_veystrix_branding_and_tool_calls(self):
        result = tui("opencode", home=self.home, port=self.port, once=True, command=self.command)
        text = "\n".join(result["lines"])
        self.assertIn("V E Y S T R I X", text)
        self.assertIn("◆ Veystrix · read_file", text)
        self.assertIn("◆ Veystrix · edit_file", text)
        self.assertIn("◆ Veystrix · run_shell", text)
        self.assertIn("streaming", text)
        for forbidden in ("OpenCode", "Claude Code", "telemetry", "update available", "anthropic account", "v2.2.2"):
            self.assertNotIn(forbidden, text)

    def test_cancel_and_resume(self):
        hanging = dict(self.env, VEYSTRIX_FAKE_HANG="1")
        with patch.dict(os.environ, hanging):
            result = run_once(self.command, {}, harness="opencode", cancel_after=0.3, timeout=5)
        self.assertTrue(result["cancelled"])
        self.assertTrue(any("cancelled" in line for line in result["lines"]))
        launched = [*self.command, "--resume", "session-abc"]
        resumed = run_once(launched, {}, harness="opencode", timeout=5)
        self.assertTrue(any("session-abc" in line for line in resumed["lines"]))
        self.assertIn("--resume", launch_command("claude-code", "veystrix-flashnext", "session-abc"))

    def test_slash_commands_are_branded_native_commands(self):
        self.assertIn("8080", "\n".join(slash("/config get port", home=self.home, port=self.port)))
        self.assertIn("MEASURED", "\n".join(slash("/doctor", home=self.home, port=self.port)))
        self.assertEqual(slash("/quit", home=self.home, port=self.port), ["bye"])

    def test_cli_tui_and_bare_invocation(self):
        out = subprocess.run([sys.executable, "-m", "veystrix", "tui", "--harness", "opencode",
                              "--port", str(self.port), "--once", "--command",
                              " ".join(shlex.quote(arg) for arg in self.command)], cwd=ROOT, env=self.env,
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("V E Y S T R I X", out.stdout)
        self.assertNotIn("Claude Code", out.stdout)
        with patch.object(cli, "_dispatch_tool") as dispatch:
            with patch.object(cli.sys, "argv", ["veystrix"]):
                cli.main()
        dispatch.assert_called_once_with(["tui"])


if __name__ == "__main__":
    unittest.main()
