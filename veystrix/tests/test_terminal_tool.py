"""Offline tests for the terminal-tool front end and supervisor (M2).

Rules: temporary HOME (VEYSTRIX_HOME), mock server only, no real checkpoint, no real
harness config, no network beyond loopback. Every subprocess inherits the test env.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from veystrix.tool.commands import cmd_diagnostics, cmd_doctor, cmd_models
from veystrix.tool.config import Config, LocalKeyStore, home_dir
from veystrix.tool.engine import (bundle_integrity, cpp_refusal, manifest_path,
                                  report_engine, resolve_engine)
from veystrix.tool.sanitizer import clean
from veystrix.tool.supervisor import (MAX_RESTARTS, StartOptions, battery_refusal,
                                      build_command, guard_stop, health, memory_refusal,
                                      next_action, restart, start, state_path, status, stop)

ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TerminalToolTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.env = dict(os.environ)
        self.env.update({
            "VEYSTRIX_HOME": str(self.home),
            "VEYSTRIX_MODEL_ROOT": str(self.home / "models"),
            "VEYSTRIX_TEST_POWER": json.dumps({"on_ac": True, "percent": 100}),
            "VEYSTRIX_TEST_MEMORY": json.dumps({"available_bytes": 16 * 1024**3}),
            "VEYSTRIX_NO_KEYCHAIN": "1",
        })
        self._patcher = patch.dict(os.environ, self.env, clear=False)
        self._patcher.start()

    def tearDown(self):
        stop(self.home)
        self._patcher.stop()
        self._tmp.cleanup()

    def options(self, port: int, command: list[str] | None = None, **kwargs) -> StartOptions:
        return StartOptions(home=self.home, port=port,
                            command=command or [sys.executable, "-m",
                                                "integrations.hermes.mock_veystrix_server",
                                                "--port", str(port)],
                            timeout=30, **kwargs)

    # -- config / selector -------------------------------------------------
    def test_engine_precedence_flag_env_config_default(self):
        self.assertEqual(resolve_engine("python", {"VEYSTRIX_ENGINE": "cpp"}, {"engine": "cpp"}), "python")
        self.assertEqual(resolve_engine(None, {"VEYSTRIX_ENGINE": "cpp"}, {"engine": "python"}), "cpp")
        self.assertEqual(resolve_engine(None, {}, {"engine": "cpp"}), "cpp")
        self.assertEqual(resolve_engine(None, {}, {}), "python")
        with self.assertRaises(ValueError):
            resolve_engine("rust", {}, {})

    def test_config_roundtrip_aliases_and_permissions(self):
        config = Config(self.home)
        self.assertEqual(config.get("port"), 8080)
        config.set("minimum-battery-percent", "30")
        self.assertEqual(Config(self.home).get("minimum_battery_percent"), 30)
        self.assertEqual(config.path.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(ValueError):
            config.set("port", "99999")

    def test_key_store_fallback_and_sanitizer(self):
        store = LocalKeyStore(self.home)
        self.assertEqual(store.store("vx-test-secret-value-1234"), "file")
        self.assertEqual(store.load(), "vx-test-secret-value-1234")
        text = clean("key=vx-test-secret-value-1234 home=/Users/vincent/x sk-abcdefghijklmnopqrstuvwxyz",
                     home=str(self.home), extra_secrets=["vx-test-secret-value-1234"])
        self.assertNotIn("vx-test-secret-value-1234", text)
        self.assertNotIn("/Users/vincent", text)
        self.assertIn("[REDACTED]", text)

    # -- cpp clean refusal -------------------------------------------------
    def test_cpp_refusal_until_manifest_parity_pass(self):
        refusal = cpp_refusal({}, "flashnext")
        self.assertIn("no C++ engine", refusal)
        refusal = cpp_refusal({"engines": ["cpp"], "cpp_parity": {"flashnext:agent": "fail"}},
                              "flashnext")
        self.assertIn("cpp_parity: pass", refusal)
        self.assertIsNone(cpp_refusal({"engines": ["cpp"],
                                       "cpp_parity": {"flashnext:agent": "pass"}},
                                      "flashnext", "agent"))

    def test_cli_cpp_refuses_cleanly(self):
        result = subprocess.run([sys.executable, "-m", "veystrix", "start", "--engine", "cpp",
                                 "--port", str(_free_port())], cwd=ROOT, env=self.env,
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cpp", result.stderr.lower())
        self.assertIn("python", result.stderr.lower())

    def test_health_engine_field_wins_and_launch_is_loopback_only(self):
        self.assertEqual(report_engine({"engine": "cpp", "ready": True}, "python"), "cpp")
        self.assertEqual(report_engine({"ready": True}, "python"), "python")
        command = build_command(port=8080, model="flashnext", checkpoint="/tmp/model", environ={})
        self.assertIn("--host", command)
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertIn("--port", command)
        self.assertEqual(command[command.index("--port") + 1], "8080")

    def test_bundle_integrity_detects_missing_and_tampered_files(self):
        manifest = manifest_path(self.home)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        root = manifest.parent
        (root / "lib").mkdir(exist_ok=True)
        (root / "lib" / "engine.pyc").write_bytes(b"abc")
        import hashlib
        digest = hashlib.sha256(b"abc").hexdigest()
        manifest.write_text(json.dumps({"engines": ["python"], "files": {"lib/engine.pyc": digest}}))
        self.assertEqual(bundle_integrity(self.home), [])
        (root / "lib" / "engine.pyc").write_bytes(b"bad")
        self.assertTrue(any("checksum mismatch" in p for p in bundle_integrity(self.home)))

    # -- policy pure functions --------------------------------------------
    def test_battery_policy_below_25_refuses(self):
        self.assertIsNone(battery_refusal({"on_ac": True, "percent": 10}, 25))
        self.assertIn("25%", battery_refusal({"on_ac": False, "percent": 24}, 25) or "")
        self.assertIsNone(battery_refusal({"on_ac": False, "percent": 25}, 25))
        self.assertIsNotNone(battery_refusal({"on_ac": None, "percent": 80}, 25))
        self.assertIsNone(battery_refusal({"on_ac": False, "percent": 5}, 0))
        with self.assertRaises(SystemExit) as caught:
            with patch.dict(os.environ, {"VEYSTRIX_TEST_POWER": json.dumps({"on_ac": False, "percent": 20})}):
                start(self.options(_free_port()))
        self.assertIn("Battery", str(caught.exception))

    def test_memory_preflight_refuses(self):
        self.assertIn("memory", memory_refusal({"available_bytes": 1024}) or "")
        with self.assertRaises(SystemExit) as caught:
            with patch.dict(os.environ, {"VEYSTRIX_TEST_MEMORY": json.dumps({"available_bytes": 1024})}):
                start(self.options(_free_port()))
        self.assertIn("memory", str(caught.exception))

    def test_guard_stop_is_not_a_crash(self):
        self.assertTrue(guard_stop("memory_admission_refused: guard stop"))
        self.assertFalse(guard_stop("Listening on 127.0.0.1"))
        self.assertEqual(next_action("guard stop", 0, 0, True), "guard_stopped")
        self.assertEqual(next_action("guard stop", 0, 0, False), "guard_stopped")
        self.assertEqual(next_action("", 86, 0, True), "guard_stopped")
        self.assertEqual(next_action("", 1, 0, True), "restart")
        self.assertEqual(next_action("", 1, MAX_RESTARTS, True), "failed")
        self.assertEqual(next_action("", 1, 0, False), "failed")

    # -- watcher integration ----------------------------------------------
    def _write_watch_state(self, command: list[str], restarts: int = 0, supervised: bool = True):
        path = state_path(self.home)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"port": _free_port(), "engine": "python", "model": "flashnext",
                                    "command": command, "restarts": restarts, "pid": None,
                                    "engine_pid": None, "status": "starting", "supervised": supervised}))
        return path

    def test_watcher_never_restarts_after_guard_stop(self):
        self._write_watch_state([sys.executable, "-c", "print('guard stop'); raise SystemExit(86)"])
        result = subprocess.run([sys.executable, "-m", "veystrix.tool.supervisor", "_watch",
                                 "--home", str(self.home)], cwd=ROOT, env=self.env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        state = json.loads(state_path(self.home).read_text())
        self.assertEqual(state["status"], "guard_stopped")
        self.assertEqual(state["restarts"], 0)

    def test_watcher_bounds_crash_restarts_to_three(self):
        env = dict(self.env, VEYSTRIX_TEST_BACKOFF="0")
        self._write_watch_state([sys.executable, "-c", "import sys; sys.exit(3)"])
        result = subprocess.run([sys.executable, "-m", "veystrix.tool.supervisor", "_watch",
                                 "--home", str(self.home)], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        state = json.loads(state_path(self.home).read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["restarts"], MAX_RESTARTS)

    # -- live lifecycle against the mock server ---------------------------
    def test_start_status_logs_stop(self):
        port = _free_port()
        result = start(self.options(port))
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["engine"], "python")
        report = health(port)
        self.assertIsNotNone(report)
        self.assertTrue(report["ready"])
        self.assertEqual(status(self.home)["engine"], "python")
        self.assertIn("mock Veystrix", (self.home / ".veystrix" / "logs" / "engine.log").read_text())
        stopped = stop(self.home)
        self.assertEqual(stopped["status"], "stopped")
        self.assertIsNone(health(port))

    def test_restart_uses_new_command_and_reports_engine(self):
        port = _free_port()
        start(self.options(port))
        stopped = stop(self.home)
        self.assertEqual(stopped["status"], "stopped")
        restarted = restart(self.options(port))
        self.assertEqual(restarted["status"], "running")
        stop(self.home)

    def test_start_refuses_second_engine_on_same_port(self):
        port = _free_port()
        start(self.options(port))
        with self.assertRaises(SystemExit) as caught:
            start(self.options(port))
        self.assertIn("port", str(caught.exception))

    # -- other commands ----------------------------------------------------
    def test_models_add_validates_without_downloading(self):
        folder = self.home / "models" / "fake-inkling"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text(json.dumps({"model_type": "inkling2_loop"}))
        (folder / "tokenizer.json").write_text("{}")
        (folder / "tokenizer_config.json").write_text("{}")
        shard = "model-00001-of-00001.safetensors"
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": shard}}))
        (folder / shard).write_bytes(b"not real weights")
        with patch("veystrix.recipe.emit_recipe", return_value={}):
            self.assertEqual(cmd_models(["add", str(folder)], self.home, Config(self.home)), 0)

    def test_doctor_and_diagnostics_are_labeled_and_sanitized(self):
        doctor = subprocess.run([sys.executable, "-m", "veystrix", "doctor"], cwd=ROOT, env=self.env,
                                capture_output=True, text=True, timeout=30)
        self.assertIn("MEASURED", doctor.stdout)
        self.assertIn("CITED", doctor.stdout)
        import io
        with patch("sys.stdout", new=io.StringIO()):
            cmd_diagnostics([], self.home, Config(self.home), LocalKeyStore(self.home))
        written = sorted((self.home / ".veystrix" / "diagnostics").glob("diagnostics-*.txt"))[-1].read_text()
        self.assertNotIn(str(self.home), written)
        self.assertNotIn("vx-test", written)


class HarnessLaunchCommands(unittest.TestCase):
    def test_pi_and_hermes_launch_on_the_veystrix_model_and_profile(self):
        from veystrix.tool.harnesses import run_command
        self.assertEqual(run_command("pi", "veystrix-x"), ["pi", "--model", "veystrix/veystrix-x"])
        self.assertEqual(run_command("hermes", "veystrix-x"), ["hermes", "-p", "veystrix"])


class LockedHarnessTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_public_harness_inventory_only_includes_unlocked_and_direct(self):
        from veystrix.tool.tui import harness_inventory
        items = harness_inventory(self.home, active="opencode")
        keys = [item["key"] for item in items]
        self.assertEqual(keys, ["opencode", "pi", "hermes", "direct"])
        for locked in ("claude-code", "codex", "qwen-code", "deepseek", "kimi", "aider", "cline", "continue"):
            self.assertNotIn(locked, keys)

    def test_is_locked_helper(self):
        from veystrix.tool.harnesses import is_locked
        self.assertFalse(is_locked("opencode"))
        self.assertFalse(is_locked("pi"))
        self.assertFalse(is_locked("hermes"))
        self.assertTrue(is_locked("claude"))
        self.assertTrue(is_locked("claude-code"))
        self.assertTrue(is_locked("codex"))
        self.assertTrue(is_locked("aider"))
        self.assertTrue(is_locked("nonexistent"))

    def test_locked_harness_cli_rejection(self):
        from veystrix.tool.cli import dispatch_tool
        import io
        from contextlib import redirect_stderr
        for locked in ("codex", "claude", "claude-code", "aider"):
            buf = io.StringIO()
            with redirect_stderr(buf):
                with self.assertRaises(SystemExit) as ctx:
                    dispatch_tool(["run", locked, "--home", str(self.home)])
            err = str(ctx.exception) + "\n" + buf.getvalue()
            self.assertIn(f"harness '{locked}' is locked and unavailable in this release", err)
            self.assertIn("Available harnesses in this release: opencode, pi, hermes", err)

    def test_config_migrates_locked_harness_to_default(self):
        from veystrix.tool.config import Config
        c = Config(self.home)
        cfg_file = c.path
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        for locked in ("codex", "claude-code", "aider", "cline"):
            cfg_file.write_text(json.dumps({"harness": locked}))
            self.assertEqual(Config(self.home).get("harness"), "opencode")

        for unlocked in ("pi", "hermes", "direct"):
            cfg_file.write_text(json.dumps({"harness": unlocked}))
            self.assertEqual(Config(self.home).get("harness"), unlocked)


if __name__ == "__main__":
    unittest.main()
