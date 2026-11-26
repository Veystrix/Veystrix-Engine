"""Offline tests for the M4 bundle/protection/release pieces.

No bundle build, no minisign key, no GitHub call and no real checkpoint is used.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from veystrix.engine_cli import PROTOCOL_VERSION, version_payload
from veystrix.tool.commands import cmd_doctor, cmd_install, cmd_uninstall, cmd_update
from veystrix.tool.engine import check_protocol, cpp_refusal, engine_version
from veystrix.tool.config import Config, home_dir, resolve_default_model

ROOT = Path(__file__).resolve().parents[2]
LEAK_SCAN = ROOT / "packaging" / "engine" / "leak_scan.py"
GENERATE_LEGAL = ROOT / "packaging" / "legal" / "generate_legal.py"
COMPILE_PROTECTED = ROOT / "packaging" / "engine" / "compile_protected.sh"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class EngineContractTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_version_json_contract(self):
        payload = version_payload(self.home)
        self.assertIn("version", payload)
        self.assertEqual(payload["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(payload["engines"], ["python"])
        self.assertEqual(payload["cpp_parity"], {})
        manifest = self.home / ".veystrix" / "engine" / "BUNDLE-MANIFEST.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"engines": ["python", "cpp"],
                                        "cpp_parity": {"veystrix-flashnext": "pass"}}))
        payload = version_payload(self.home)
        self.assertEqual(payload["engines"], ["python", "cpp"])
        self.assertEqual(payload["cpp_parity"]["veystrix-flashnext"], "pass")

    def test_protocol_handshake_refuses_unknown_versions(self):
        manifest = self.home / ".veystrix" / "engine" / "BUNDLE-MANIFEST.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"protocol_version": PROTOCOL_VERSION}))
        check_protocol(self.home)
        manifest.write_text(json.dumps({"protocol_version": "999"}))
        with self.assertRaises(SystemExit) as caught:
            check_protocol(self.home)
        self.assertIn("not supported", str(caught.exception))
        manifest.unlink()
        self.assertEqual(engine_version(self.home), {})

    def test_install_uninstall_are_offline_dry_runs(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cmd_install(["3.8.2"], self.home, Config(self.home))
        self.assertEqual(code, 0)
        self.assertIn("veystrix-engine-3.8.2-arm64.tar.zst", buffer.getvalue())
        (self.home / ".veystrix" / "bin").mkdir(parents=True)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cmd_uninstall([], self.home)
        self.assertIn("keep", buffer.getvalue())
        self.assertTrue((self.home / ".veystrix" / "bin").is_dir())

    def test_engine_cli_json_output(self):
        result = subprocess.run([sys.executable, "-m", "veystrix.engine_cli", "--version", "--json",
                                 "--home", str(self.home)], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["protocol_version"], PROTOCOL_VERSION)


class LeakScanTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        (self.bundle / "BUNDLE-MANIFEST.json").write_text("{}")
        (self.bundle / "lib").mkdir()
        (self.bundle / "lib" / "engine.pyc").write_bytes(b"bytecode")

    def tearDown(self):
        self._tmp.cleanup()

    def run_scan(self, *args):
        return subprocess.run([sys.executable, str(LEAK_SCAN), "--bundle", str(self.bundle), *args],
                              cwd=ROOT, capture_output=True, text=True, timeout=30)

    def test_clean_bundle_passes(self):
        result = self.run_scan()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("leak scan OK", result.stdout)

    def test_source_private_path_and_token_are_rejected(self):
        (self.bundle / "lib" / "secret.py").write_text("print(1)")
        self.assertIn("source file", self.run_scan().stderr)
        (self.bundle / "lib" / "secret.py").unlink()
        (self.bundle / "notes.txt").write_text("see /Users/vincent/private")
        self.assertIn("/Users/", self.run_scan().stderr)
        (self.bundle / "notes.txt").unlink()
        (self.bundle / "notes.txt").write_text("ghp_" + "A" * 36)
        self.assertIn("credential-shaped", self.run_scan().stderr)

    def test_public_scan_ignores_git_but_rejects_source(self):
        public = self.root / "public"
        (public / ".git" / "objects").mkdir(parents=True)
        (public / ".git" / "HEAD").write_text("ref: refs/heads/main")
        (public / "README.md").write_text("docs only")
        result = self.run_scan("--public", str(public))
        self.assertEqual(result.returncode, 0, result.stderr)
        (public / "engine.py").write_text("print('leak')")
        self.assertIn("source file", self.run_scan("--public", str(public)).stderr)


class DefaultModelTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.config = Config(self.home)

    def tearDown(self):
        self._tmp.cleanup()

    def test_no_hardcoded_default_and_persisted_choice(self):
        with patch("veystrix.tool.config.installed_model_ids", return_value=["flashnext", "nemotron35-lightning"]):
            with self.assertRaises(SystemExit) as caught:
                resolve_default_model(self.config, interactive=False)
            self.assertIn("default model", str(caught.exception))
            chosen = resolve_default_model(self.config, explicit="flashnext", interactive=False)
            self.assertEqual(chosen, "flashnext")
        self.assertEqual(Config(self.home).get("default_model"), "flashnext")
        with patch("veystrix.tool.config.installed_model_ids", return_value=["flashnext", "nemotron35-lightning"]):
            self.assertEqual(resolve_default_model(self.config), "flashnext")

    def test_single_family_is_stored_without_prompt(self):
        with patch("veystrix.tool.config.installed_model_ids", return_value=["flashnext"]):
            self.assertEqual(resolve_default_model(self.config), "flashnext")
        self.assertEqual(Config(self.home).get("default_model"), "flashnext")


class ReleaseDocumentationTests(unittest.TestCase):
    def test_active_legal_generator_has_no_sparkle(self):
        module = _load("generate_legal_under_test", GENERATE_LEGAL)
        self.assertEqual(module.EXTRA_COMPONENTS, [])
        with tempfile.TemporaryDirectory() as out:
            module.write_notices([], Path(out), "2026-11-18", None)
            text = (Path(out) / "THIRD_PARTY_NOTICES.md").read_text()
            self.assertNotIn("Sparkle", text)
            self.assertNotIn("app bundle", text)

    def test_cpp_parity_gate_is_per_model_and_profile(self):
        # Only the canonical f"{model}:{profile}" key is accepted (looser shapes
        # would let one entry authorize every profile for a model).
        manifest = {"engines": ["python", "cpp"],
                    "cpp_parity": {"flashnext:agent": "pass"}}
        self.assertIsNone(cpp_refusal(manifest, "flashnext", "agent"))
        refusal = cpp_refusal(manifest, "flashnext", "plain")
        self.assertIn("cpp_parity: pass", refusal)
        self.assertIn("nemotron35-lightning", cpp_refusal(manifest, "nemotron35-lightning", "agent"))

    def test_bundle_script_emits_machine_manifest(self):
        text = (ROOT / "packaging" / "engine" / "build_engine_bundle.sh").read_text()
        self.assertIn("BUNDLE-MANIFEST.json", text)
        self.assertIn("cpp_parity", text)

    def test_tier2_script_documents_no_minus_oo_and_fallback(self):
        text = COMPILE_PROTECTED.read_text()
        self.assertIn("--require", text)
        self.assertIn("TIER-1 FALLBACK", text)
        self.assertNotIn("-OO", text.replace("do NOT use -OO", ""))

    def test_update_check_is_opt_in_and_offline_in_tests(self):
        home = Path(tempfile.mkdtemp())
        payload = {"tag_name": "v9.9.9", "published_at": "2026-11-18T00:00:00Z", "prerelease": True,
                   "assets": [{"name": "SHA256SUMS.txt", "browser_download_url": "https://x/SUMS"},
                              {"name": "veystrix-engine-9.9.9-arm64.tar.gz",
                               "browser_download_url": "https://x/a.tar.gz"}]}
        env = {"VEYSTRIX_TEST_RELEASES": json.dumps(payload)}
        with patch.dict(os.environ, env):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cmd_update(["--check"], home, Config(home))
        self.assertEqual(code, 0)
        self.assertIn("latest 9.9.9 (early build)", buffer.getvalue())
        self.assertIn("veystrix update", buffer.getvalue())

    def test_doctor_exits_zero_with_advisory_warnings(self):
        home = Path(tempfile.mkdtemp())
        buffer = io.StringIO()
        with patch.dict(os.environ, {"VEYSTRIX_MODEL_ROOT": str(home / "models")}), redirect_stdout(buffer):
            code = cmd_doctor([], home, Config(home), "python")
        self.assertEqual(code, 0)
        self.assertIn("CITED", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
