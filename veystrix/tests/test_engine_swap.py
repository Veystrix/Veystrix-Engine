"""One-command engine swap, source-checkout C++ gating and engine auto-start (offline)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from veystrix.tool import engine as engine_mod
from veystrix.tool import supervisor, swap
from veystrix.tool.config import Config


class SwapBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name)
        env = {"VEYSTRIX_HOME": str(self.home), "VEYSTRIX_MODEL_ROOT": str(self.home / "models"), "VEYSTRIX_NO_KEYCHAIN": "1"}
        for name in ("VEYSTRIX_ENGINE", "VEYSTRIX_SOURCE_CPP"):
            os.environ.pop(name, None)
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        # A port nothing listens on: a developer's running engine on 8080 must not leak in.
        import socket
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            free = sock.getsockname()[1]
        config = Config(self.home)
        config.values["port"] = free
        config.values["model"] = "inkling2-mini"
        config.save()
        self.free_port = free
        self.fake_lib = self.home / "libveystrix_c.dylib"
        self.fake_lib.write_bytes(b"x")
        # Flash-Next runs on the generic runtime: its cpp gate is a passing port-check for this dylib.
        record = engine_mod.parity_record_path(self.home)
        record.parent.mkdir(parents=True, exist_ok=True)
        sha = engine_mod.file_sha256(self.fake_lib)
        record.write_text(json.dumps({t: {"status": "pass", "lib_sha256": sha} for t in ("qwen4_exp", "qwen3_5_moe", "inkling2_loop")}))

    def source_on(self, present=True):
        os.environ["VEYSTRIX_SOURCE_CPP"] = "1"
        return patch.object(engine_mod, "source_cpp_dylib", return_value=self.fake_lib if present else None)


class SourceModeGate(SwapBase):
    def test_redirected_home_is_bundle_only_by_default(self):
        with patch.object(engine_mod, "source_cpp_dylib", return_value=self.fake_lib):
            self.assertFalse(engine_mod.source_mode_enabled())
            with self.assertRaises(SystemExit):
                engine_mod.check_engine("cpp", self.home, "flashnext")

    def test_real_home_default_is_source_mode(self):
        self.assertTrue(engine_mod.source_mode_enabled({}))
        self.assertFalse(engine_mod.source_mode_enabled({"VEYSTRIX_HOME": "/tmp/x"}))
        self.assertTrue(engine_mod.source_mode_enabled({"VEYSTRIX_HOME": "/tmp/x", "VEYSTRIX_SOURCE_CPP": "1"}))
        self.assertFalse(engine_mod.source_mode_enabled({"VEYSTRIX_SOURCE_CPP": "0"}))

    def test_source_build_allows_cpp_without_bundle(self):
        with self.source_on():
            engine_mod.check_engine("cpp", self.home, "flashnext")      # no raise

    def test_missing_build_refuses_with_build_hint(self):
        with self.source_on(present=False):
            with self.assertRaises(SystemExit) as ctx:
                engine_mod.check_engine("cpp", self.home, "flashnext")
        self.assertIn("cmake", str(ctx.exception))

    def test_installed_bundle_wins_over_source_build(self):
        root = engine_mod.bundle_root(self.home)
        root.mkdir(parents=True)
        (root / "BUNDLE-MANIFEST.json").write_text(json.dumps({"engines": ["python"], "cpp_parity": {}}))
        with self.source_on():
            self.assertFalse(engine_mod.source_cpp_ready(self.home))
            with self.assertRaises(SystemExit):
                engine_mod.check_engine("cpp", self.home, "flashnext")

    def test_source_cpp_command_uses_router_with_cpp_env(self):
        with self.source_on():
            command = supervisor.build_command(port=8080, model="flashnext", checkpoint="",
                                               environ=dict(os.environ), engine="cpp", home=self.home)
            self.assertIn("veystrix.model_router", command)
            self.assertEqual(supervisor.engine_env_for("cpp", {}).get("VEYSTRIX_ENGINE"), "cpp")


class SwapFlow(SwapBase):
    def test_describe_reports_both_engines(self):
        with self.source_on():
            report = swap.describe(self.home)
        self.assertEqual(report["selected"], "cpp")          # C++ whenever it can serve the model
        self.assertFalse(report["available"]["python"]["available"])
        self.assertTrue(report["available"]["cpp"]["available"])

    def test_swap_saves_selection_when_not_running(self):
        with self.source_on():
            outcome = swap.swap(self.home, engine="cpp")
        self.assertEqual(outcome["engine"], "cpp")
        self.assertTrue(outcome["changed"])
        self.assertFalse(outcome["restarted"])
        self.assertEqual(Config(self.home).get("engine"), "cpp")

    def test_refused_swap_changes_nothing_and_stops_nothing(self):
        with self.source_on(present=False), patch.object(swap, "stop") as stop_mock:
            with self.assertRaises(SystemExit):
                swap.swap(self.home, engine="cpp")
        stop_mock.assert_not_called()
        self.assertEqual(Config(self.home).get("engine"), "python")

    def test_running_engine_is_restarted_on_the_new_engine(self):
        running = {"status": "running", "engine": "python", "configured_engine": "python",
                   "model": "flashnext", "port": 8123}
        with self.source_on(), patch.object(swap, "status", return_value=running), \
                patch.object(swap, "default_checkpoint", return_value="/models/fn"), \
                patch.object(swap, "stop") as stop_mock, patch.object(swap, "start") as start_mock:
            outcome = swap.swap(self.home, engine="cpp", timeout=5)
        stop_mock.assert_called_once()
        options = start_mock.call_args[0][0]
        self.assertEqual((options.engine, options.model, options.port), ("cpp", "inkling2-mini", 8123))
        self.assertEqual(options.checkpoint, "/models/fn")       # the router requires --checkpoint
        self.assertTrue(outcome["restarted"])

    def test_failed_restart_rolls_selection_back(self):
        running = {"status": "running", "engine": "python", "configured_engine": "python",
                   "model": "flashnext", "port": 8123}
        with self.source_on(), patch.object(swap, "status", return_value=running), \
                patch.object(swap, "default_checkpoint", return_value="/models/fn"), \
                patch.object(swap, "stop"), patch.object(swap, "start", side_effect=SystemExit("boom")):
            with self.assertRaises(SystemExit):
                swap.swap(self.home, engine="cpp", timeout=5)
        self.assertEqual(Config(self.home).get("engine"), "python")

    def test_no_restart_flag_leaves_running_engine(self):
        running = {"status": "running", "engine": "python", "configured_engine": "python",
                   "model": "flashnext", "port": 8123}
        with self.source_on(), patch.object(swap, "status", return_value=running), \
                patch.object(swap, "stop") as stop_mock:
            outcome = swap.swap(self.home, engine="cpp", restart=False)
        stop_mock.assert_not_called()
        self.assertFalse(outcome["restarted"])

    def test_model_swap_validates_name(self):
        with self.assertRaises(ValueError):
            swap.swap(self.home, model="gpt-9")
        with self.source_on(), patch.object(swap, "checkpoint_refusal", return_value=None):
            outcome = swap.swap(self.home, model="inkling2-mini")
        self.assertEqual(Config(self.home).get("model"), "inkling2-mini")
        self.assertEqual(outcome["model"], "inkling2-mini")

    def test_ensure_engine_noop_when_running(self):
        with patch.object(swap, "health", return_value={"engine": "python", "model": "m"}), \
                patch.object(swap, "start") as start_mock:
            result = swap.ensure_engine(self.home, port=9999)
        start_mock.assert_not_called()
        self.assertFalse(result["started"])

    def test_ensure_engine_starts_configured_engine(self):
        Config(self.home).set("engine", "cpp")
        with self.source_on(), patch.object(swap, "health", return_value=None), \
                patch.object(swap, "default_checkpoint", return_value="/models/fn"), \
                patch.object(swap, "start", return_value={"engine": "cpp", "model": "flashnext"}) as start_mock:
            result = swap.ensure_engine(self.home, port=9999)
        self.assertTrue(result["started"])
        self.assertEqual(start_mock.call_args[0][0].engine, "cpp")
        self.assertEqual(start_mock.call_args[0][0].checkpoint, "/models/fn")

    def test_ensure_engine_resolves_the_installed_checkpoint(self):
        models = self.home / "models"
        ckpt = models / "Inkling-2-Mini-oQ4e-mtp"
        ckpt.mkdir(parents=True)
        (ckpt / "config.json").write_text(json.dumps({"model_type": "inkling2_loop"}))
        for name in ("tokenizer.json", "tokenizer_config.json"):
            (ckpt / name).write_text("{}")
        (ckpt / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "m.safetensors"}}))
        (ckpt / "m.safetensors").write_bytes(b"0")
        with self.source_on(), patch.dict(os.environ, {"VEYSTRIX_MODEL_ROOT": str(models)}), \
                patch("veystrix.recipe.emit_recipe", return_value={}), \
                patch.object(swap, "health", return_value=None), \
                patch.object(swap, "start", return_value={"engine": "cpp", "model": "inkling2-mini"}) as start_mock:
            swap.ensure_engine(self.home, port=9999)
        self.assertEqual(start_mock.call_args[0][0].checkpoint, str(ckpt.resolve()))

    def test_ensure_engine_without_a_checkpoint_refuses_before_launch(self):
        with self.source_on(), patch.dict(os.environ, {"VEYSTRIX_MODEL_ROOT": str(self.home / "none")}), \
                patch.object(swap, "health", return_value=None), patch.object(swap, "start") as start_mock:
            with self.assertRaises(SystemExit) as ctx:
                swap.ensure_engine(self.home, port=9999)
        start_mock.assert_not_called()
        self.assertIn("no complete Inkling 2 Mini checkpoint", str(ctx.exception))


class CommandLine(SwapBase):
    def test_engine_command_prints_report_and_swaps(self):
        from veystrix.tool.cli import dispatch_tool
        import io
        from contextlib import redirect_stdout
        with self.source_on():
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                self.assertEqual(dispatch_tool(["engine"]), 0)
                self.assertEqual(dispatch_tool(["engine", "cpp", "--no-restart"]), 0)
        text = buffer.getvalue()
        self.assertIn("selected=cpp", text)
        self.assertIn("now using cpp", text)
        self.assertEqual(Config(self.home).get("engine"), "cpp")

    def test_slash_engine_inside_ui(self):
        from veystrix.tool.tui import slash
        with self.source_on():
            self.assertTrue(any("engine=" in line for line in slash("/engine", home=self.home, port=self.free_port)))
            lines = slash("/engine cpp", home=self.home, port=self.free_port)
        self.assertTrue(any("engine is now cpp" in line for line in lines))
        refused = slash("/engine fortran", home=self.home, port=self.free_port)
        self.assertTrue(any("cannot switch" in line for line in refused))

    def test_status_line(self):
        from veystrix.tool.cli import status_line
        self.assertEqual(status_line({"status": "stopped", "port": 8080}),
                         "veystrix: stopped (port 8080); start it with: veystrix start")
        line = status_line({"status": "running", "port": 8080, "engine": "cpp", "runtime": "generic",
                            "model": "veystrix-ornith", "kv": "bf16", "context_in_use": 512,
                            "context_ceiling": 262144})
        # The engine is deliberately not surfaced to users: one product, one line.
        self.assertEqual(line, "veystrix: running on 127.0.0.1:8080 | model veystrix-ornith"
                               " | kv bf16 | context 512/262144")
        self.assertNotIn("cpp", line)
        self.assertNotIn("engine", line)

    def test_run_refuses_a_model_id_the_endpoint_does_not_serve(self):
        from veystrix.tool import cli
        with patch.object(cli, "health", return_value={"ready": True, "installed_model_id": "veystrix-inkling2-mini"}), \
             patch.object(cli, "_served_ids", return_value=["veystrix-inkling2-mini"]), \
             patch.object(cli.os, "execvpe") as launched:
            with self.assertRaises(SystemExit) as ctx:
                cli.dispatch_tool(["run", "opencode", "--model", "inkling2", "--port", "8080"])
            self.assertIn("serves veystrix-inkling2-mini", str(ctx.exception))
            launched.assert_not_called()
            cli.dispatch_tool(["run", "opencode", "--model", "inkling2-mini", "--port", "8080"])
            launched.assert_called_once()



class CppWorkerEnv(unittest.TestCase):
    def test_cpp_worker_drops_the_malloc_large_cache(self):
        env = supervisor.apply_cpp_worker_env({"VEYSTRIX_ENGINE": "cpp"})
        self.assertEqual(env["MallocLargeCache"], "0")

    def test_user_value_wins_and_python_is_untouched(self):
        env = supervisor.apply_cpp_worker_env({"VEYSTRIX_ENGINE": "cpp", "MallocLargeCache": "1"})
        self.assertEqual(env["MallocLargeCache"], "1")
        self.assertNotIn("MallocLargeCache", supervisor.apply_cpp_worker_env({"VEYSTRIX_ENGINE": "python"}))
        self.assertNotIn("MallocLargeCache", supervisor.apply_cpp_worker_env({}))


class MemoryStateParsing(unittest.TestCase):
    def fake(self, page_line):
        text = (page_line + "\nPages free:      1000.\nPages inactive:   3000.\nPages purgeable:   0.\n")
        return lambda *a, **k: type("R", (), {"stdout": text})()

    def test_uses_vm_stat_page_size(self):
        with patch.object(supervisor, "sys_is_macos", return_value=True):
            state = supervisor.memory_state({}, runner=self.fake(
                "Mach Virtual Memory Statistics: (page size of 16384 bytes)"))
        self.assertEqual(state["available_bytes"], 4000 * 16384)

    def test_falls_back_to_4k_without_the_header(self):
        with patch.object(supervisor, "sys_is_macos", return_value=True):
            state = supervisor.memory_state({}, runner=self.fake("garbage"))
        self.assertEqual(state["available_bytes"], 4000 * 4096)


class ServerImportWithCpp(unittest.TestCase):
    def test_server_imports_with_cpp_selected(self):
        """Regression: server.py used to call install_for_server() without its module argument."""
        import subprocess
        import sys
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, "-c", "import veystrix.server"], cwd=root,
                                env={**os.environ, "VEYSTRIX_ENGINE": "cpp"},
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-800:])


class LauncherTests(unittest.TestCase):
    def test_launcher_wins_over_an_editable_install_from_any_cwd(self):
        """Regression: from another cwd the venv's editable install of a different checkout was imported."""
        import subprocess
        root = Path(__file__).resolve().parents[2]
        if not (root / ".venv" / "bin" / "python").exists():
            self.skipTest("no .venv in this checkout")
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([str(root / "scripts" / "install_cli.sh"), tmp], check=True, capture_output=True)
            result = subprocess.run([str(Path(tmp) / "veystrix"), "engine", "--json"], cwd="/",
                                    env={**os.environ, "VEYSTRIX_HOME": tmp, "VEYSTRIX_NO_KEYCHAIN": "1"},
                                    capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-500:])
        self.assertIn('"selected"', result.stdout)        # `engine` exists only in this checkout


class MissingModelEdgeCases(unittest.TestCase):
    def test_start_non_interactive_without_model_fails_cleanly(self):
        from veystrix.tool import cli
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"VEYSTRIX_HOME": tmp, "VEYSTRIX_MODEL_ROOT": tmp}), \
             patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(SystemExit) as ctx:
                cli.dispatch_tool(["start"])
            self.assertIn("veystrix models pull", str(ctx.exception))

    def test_start_falls_back_to_single_installed_public_model(self):
        from veystrix.tool import swap
        with patch.object(swap, "default_checkpoint", side_effect=[SystemExit("missing"), "/models/flash"]), \
             patch.object(swap, "_installed_public_models", return_value=[("flashnext", "/models/flash")]):
            model, ckpt = swap.resolve_model_and_checkpoint("inkling2-mini", explicit_model=False, interactive=False)
        self.assertEqual((model, ckpt), ("flashnext", "/models/flash"))

    def test_start_interactive_offers_download_and_proceeds(self):
        from veystrix.tool import swap
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"VEYSTRIX_MODEL_ROOT": tmp}), \
             patch.object(swap, "default_checkpoint", side_effect=[SystemExit("missing"), "/models/inkling2"]), \
             patch.object(swap, "_installed_public_models", return_value=[]), \
             patch("builtins.input", return_value="y"), \
             patch("veystrix.model_pull.pull", return_value={"files": 3}) as pulled:
            model, ckpt = swap.resolve_model_and_checkpoint("inkling2-mini", explicit_model=False, interactive=True)
        self.assertEqual((model, ckpt), ("inkling2-mini", "/models/inkling2"))
        pulled.assert_called_once()

    def test_run_checks_harness_binary_before_starting_engine(self):
        from veystrix.tool import cli
        with patch.object(cli, "health", return_value=None), \
             patch("shutil.which", return_value=None), \
             patch("veystrix.tool.swap.ensure_engine") as ensure:
            with self.assertRaises(SystemExit) as ctx:
                cli.dispatch_tool(["run", "opencode", "--port", "8080"])
            self.assertIn("not installed or not on PATH", str(ctx.exception))
            ensure.assert_not_called()

    def test_restart_does_not_stop_engine_when_no_checkpoint(self):
        from veystrix.tool import cli, swap
        with patch.object(cli, "status", return_value={"status": "running", "model": "inkling2-mini", "port": 8080}), \
             patch.object(cli, "stop") as stopper, \
             patch.object(swap, "resolve_model_and_checkpoint", side_effect=SystemExit("veystrix: no checkpoint")):
            with self.assertRaises(SystemExit):
                cli.dispatch_tool(["restart"])
            stopper.assert_not_called()


if __name__ == "__main__":
    unittest.main()
