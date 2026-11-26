"""Offline harness-adapter tests (M3).

No harness binary is launched and no real harness config file is touched. Each adapter's keys
are checked against the doc URL/date recorded in code. The 3-tool-call task runs against the
mock Veystrix server with a temporary HOME: read file, edit file, run shell.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from integrations.hermes.mock_veystrix_server import MockVeystrixServer
from veystrix.tool.harnesses import (
    DOC, NAMES, anthropic_base_url, configure, resolve_model_id, run_environment,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "veystrix-flashnext"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _chat(base: str, key: str, tool: dict) -> dict:
    body = json.dumps({"model": MODEL_ID, "stream": False,
                       "messages": [{"role": "user", "content": tool["prompt"]}]}).encode()
    request = urllib.request.Request(base + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + key})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


class HarnessAdapterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.env = dict(os.environ)
        self.env.update({"VEYSTRIX_HOME": str(self.home),
                         "VEYSTRIX_MODEL_ROOT": str(self.home / "models"),
                         "VEYSTRIX_NO_KEYCHAIN": "1"})
        self._patcher = patch.dict(os.environ, self.env, clear=False)
        self._patcher.start()
        self.port = _free_port()

    def tearDown(self):
        self._patcher.stop()
        self._tmp.cleanup()

    def test_every_adapter_cites_a_doc_url_and_date(self):
        self.assertEqual(len(NAMES), 11)
        for name in NAMES:
            url, date = DOC[name]
            self.assertTrue(url.startswith("https://"), name)
            # DeepSeek was verified later than the 2026-11-18 batch.
            self.assertEqual(date, "2026-11-20" if name == "deepseek" else "2026-11-18", name)

    def test_claude_code_base_url_is_the_origin(self):
        """Claude Code appends /v1/messages; the configured base must not include /v1."""
        self.assertEqual(anthropic_base_url(9000), "http://127.0.0.1:9000")
        env = run_environment("claude-code", 9000, MODEL_ID)
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:9000")
        self.assertNotIn("/v1", env["ANTHROPIC_BASE_URL"])
        self.assertNotIn("CLAUDE_CODE_MAX_CONTEXT_TOKENS", env)      # unknown window: Claude Code's own default
        env = run_environment("claude-code", 9000, MODEL_ID, context=262144)
        self.assertEqual(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "262144")
        self.assertNotIn("CLAUDE_CODE_MAX_CONTEXT_TOKENS", run_environment("codex", 9000, MODEL_ID, context=262144))

    def test_resolve_model_id_prefers_live_report(self):
        with patch("veystrix.tool.supervisor.health",
                   return_value={"installed_model_id": "veystrix-nemotron35-lightning"}):
            self.assertEqual(resolve_model_id(9000, "flashnext"),
                             "veystrix-nemotron35-lightning")
        # Explicit --model must win over the live report (Sonnet finding 4).
        with patch("veystrix.tool.supervisor.health",
                   return_value={"installed_model_id": "veystrix-nemotron35-lightning"}):
            self.assertEqual(resolve_model_id(9000, "flashnext", explicit=True),
                             "veystrix-flashnext")
        with patch("veystrix.tool.supervisor.health", return_value=None):
            self.assertEqual(resolve_model_id(9000, "flashnext"), "veystrix-flashnext")

    def test_connect_writes_configs_with_backup_and_idempotent_rewrite(self):
        for name in NAMES:
            with self.subTest(harness=name):
                first = configure(name, home=self.home, port=self.port, model_id=MODEL_ID)
                self.assertEqual(first["result"], "created")
                path = Path(first["path"])
                self.assertTrue(path.is_file(), path)
                second = configure(name, home=self.home, port=self.port, model_id=MODEL_ID)
                self.assertEqual(second["result"], "unchanged")
                # modify, then reconnect: a timestamped backup must survive
                original = path.read_text()
                try:
                    parsed = json.loads(original)
                    parsed["_veytest_marker"] = "user edit"
                    modified = json.dumps(parsed)
                except ValueError:                       # TOML / YAML / TypeScript files
                    modified = original + "\n# user edit\n"
                path.write_text(modified)
                third = configure(name, home=self.home, port=self.port, model_id=MODEL_ID)
                # Lossless TOML/YAML preserves comment-only edits, so the
                # regenerated content can equal the file: no rewrite needed.
                if third["result"] == "updated (backup created)":
                    backups = list(path.parent.glob(path.name + ".veystrix-backup-*"))
                    self.assertTrue(backups, f"no backup for {name}")
                else:
                    self.assertEqual(third["result"], "unchanged", (name, third))

    def test_connect_never_writes_outside_the_temporary_home(self):
        for name in NAMES:
            result = configure(name, home=self.home, port=self.port, model_id=MODEL_ID)
            self.assertTrue(str(Path(result["path"])).startswith(str(self.home)), result)

    def test_run_dry_run_routes_to_local_engine_without_config_writes(self):
        for name in NAMES:
            with self.subTest(harness=name):
                env = run_environment(name, self.port, MODEL_ID)
                joined = " ".join(env.values())
                self.assertIn("127.0.0.1", joined)
                self.assertIn(str(self.port), joined)

    def test_three_tool_call_task_against_mock_server_per_harness(self):
        tools = [
            {"name": "read_file", "arguments": {"path": "README.md"}, "prompt": "read the file"},
            {"name": "edit_file", "arguments": {"path": "README.md", "text": "hi"}, "prompt": "edit the file"},
            {"name": "run_shell", "arguments": {"command": "python -m unittest"}, "prompt": "run the tests"},
        ]
        with MockVeystrixServer(port=self.port) as mock:
            mock.queue(*[{"tool_calls": [t], "text": ""} for _ in NAMES for t in tools])
            base = mock.base_url                      # http://127.0.0.1:<port>/v1
            for name in NAMES:
                with self.subTest(harness=name):
                    env = run_environment(name, self.port, MODEL_ID)
                    key = env.get("VEYSTRIX_API_KEY") or env.get("OPENAI_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN") or "local"
                    # Verify each harness's configured endpoint matches the mock's loopback root.
                    configured = [v for v in env.values() if "127.0.0.1" in v]
                    if name == "claude-code":
                        # Claude Code appends /v1/messages to its base URL.
                        origin = base.rsplit("/v1", 1)[0]
                        self.assertTrue(any(v == origin for v in configured), (name, env))
                    else:
                        self.assertTrue(any(base in v for v in configured), (name, env))
                    for tool in tools:
                        reply = _chat(base, key, tool)
                        calls = reply["choices"][0]["message"].get("tool_calls") or []
                        self.assertEqual(calls[0]["function"]["name"], tool["name"])
                        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), tool["arguments"])
            self.assertEqual(len(mock.chat_requests()), 3 * len(NAMES))

    def test_cli_connect_and_harnesses_use_the_adapters(self):
        import subprocess
        out = subprocess.run([sys.executable, "-m", "veystrix", "harnesses"],
                             cwd=ROOT, env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        from veystrix.tool.harnesses import PUBLIC_HARNESS_CHOICES
        for name in PUBLIC_HARNESS_CHOICES:
            self.assertIn(name, out.stdout)
        for locked in ("claude-code", "codex", "aider", "cline"):
            self.assertNotIn(locked, out.stdout)
        out = subprocess.run([sys.executable, "-m", "veystrix", "connect", "opencode",
                              "--port", str(self.port)], cwd=ROOT, env=self.env,
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout)["harness"], "opencode")


if __name__ == "__main__":
    unittest.main()
