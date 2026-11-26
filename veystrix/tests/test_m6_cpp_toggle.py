"""M6: offline tests for the proven C ABI / --engine cpp toggle.

No real checkpoint, no real dylib, no harness config and no network beyond
loopback. The low-level binding is injected as a deterministic stub; the
launch tests use a tiny /health mock and a temporary tool HOME.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from veystrix.backends.base import GenerationCancelled
from veystrix.backends.cpp_engine import (
    verify_lib_sha256,
    CppEngineBackend,
    CppEngineError,
    SHADOW_ENV,
    VX_ERR_CANCELLED,
    VX_ERR_INTERNAL,
    default_lib_path,
    engine_name,
    install_for_server,
    resolve_lib_path,
)
from veystrix.tool import engine as engine_mod
from veystrix.tool.engine import check_engine, cpp_refusal, load_manifest, manifest_path, report_engine
from veystrix.tool.supervisor import StartOptions, build_command, start, state_path, status, stop

ROOT = Path(__file__).resolve().parents[2]


class FakeTokenizer:
    added_tokens_decoder = {}
    eos_token_id = 248044

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 248046, "<|endoftext|>": 248001}.get(token)

    def encode(self, text, add_special_tokens=False):
        return [(7 + len(text)) % 1000]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text, add_special_tokens)}

    def decode(self, ids):
        return "".join(f"<{t}>" for t in ids)

    def apply_chat_template(self, messages, **kwargs):
        return "<|im_start|>rendered<|im_end|>"


class StubBinding:
    """The same method surface as CtypesVxBinding, deterministic and CPU-only."""

    def __init__(self, tokens=(101, 102, 103)):
        self.tokens = list(tokens)
        self.prompt = []
        self.cursor = 0
        self.cancelled = False
        self.sampler_json = None
        self.engine = None
        self.session = None
        self.raise_on_step = None

    def abi_version(self):
        return 1

    def mode(self):
        return "fake"

    def open_engine(self, model_dir, profile_json):
        self.engine = (model_dir, profile_json)
        return self.engine

    def close_engine(self, engine):
        self.engine = None

    def open_session(self, engine):
        self.session = object()
        self.prompt = []
        self.cursor = 0
        self.cancelled = False
        return self.session

    def close_session(self, session):
        self.session = None

    def prefill(self, session, tokens):
        self.prompt = [int(t) for t in tokens]
        self.cursor = 0

    def step(self, session, sampler_json):
        self.sampler_json = sampler_json
        if self.raise_on_step is not None:
            raise self.raise_on_step
        if self.cancelled:
            raise CppEngineError("cancelled", VX_ERR_CANCELLED)
        token = self.tokens[self.cursor % len(self.tokens)]
        self.cursor += 1
        return token

    def save(self, session, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "stub.json").write_text(
            json.dumps({"prompt": self.prompt, "cursor": self.cursor}), encoding="utf-8")

    def restore(self, session, directory):
        state = json.loads((Path(directory) / "stub.json").read_text(encoding="utf-8"))
        self.prompt = list(state["prompt"])
        self.cursor = int(state["cursor"])

    def cancel(self, session):
        self.cancelled = True

    def stats(self, engine):
        return {"mode": "fake", "session_tokens": len(self.prompt) + self.cursor}


class CppBackendStubTest(unittest.TestCase):
    """CPU-only stub-binding tests for the C++ Backend protocol surface."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.binding = StubBinding()
        self.backend = CppEngineBackend(
            model_dir="/fake/model",
            profile_json='{"kv":"int8"}',
            binding=self.binding,
            tokenizer=FakeTokenizer(),
            session_dir=str(Path(self.tmp.name) / "sessions"),
        )
        self.addCleanup(self.backend.close)

    def test_missing_dylib_raises_without_fallback(self):
        bogus = str(Path(self.tmp.name) / "no-such-lib.dylib")
        with patch.dict(os.environ, {"VEYSTRIX_CPP_LIB": bogus}):
            with self.assertRaises(CppEngineError) as caught:
                CppEngineBackend(model_dir="/fake/model", tokenizer=FakeTokenizer())
        self.assertIn("VEYSTRIX_CPP_LIB", str(caught.exception))

    def test_generate_callbacks_stats_and_shadow_log(self):
        shadow = str(Path(self.tmp.name) / "shadow.jsonl")
        with patch.dict(os.environ, {SHADOW_ENV: shadow}):
            backend = CppEngineBackend(
                model_dir="/fake/model", profile_json={"kv": "int8"},
                binding=StubBinding(), tokenizer=FakeTokenizer(),
                session_dir=str(Path(self.tmp.name) / "shadow-sessions"),
            )
            self.addCleanup(backend.close)
            prompt_tokens = backend.open_conversation("system", "hello")
            self.assertGreater(prompt_tokens, 0)
            backend.append_user("next turn")
            prefilled = []
            decoded = []
            streamed = []
            text, stats = backend.generate(
                3,
                out=streamed.append,
                on_prefilled=lambda: prefilled.append(True),
                on_prefill_progress=lambda done, total: None,
                on_decode_token=lambda token, piece: decoded.append((token, piece)),
            )
        self.assertEqual(text, "<101><102><103>")
        self.assertEqual("".join(streamed), text)
        self.assertEqual([t for t, _ in decoded], [101, 102, 103])
        self.assertEqual(prefilled, [True])
        self.assertEqual(stats.tokens, 3)
        self.assertEqual(stats.prompt_tokens, prompt_tokens + 1)
        self.assertGreaterEqual(stats.prefill_seconds, 0.0)
        self.assertIsNotNone(backend.last_decode_rate)
        self.assertGreater(backend.last_decode_rate, 0.0)
        self.assertGreaterEqual(backend.last_prefill_seconds, 0.0)
        self.assertTrue(backend.check_invariant())

        lines = Path(shadow).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["schema"], 2)
        self.assertEqual(record["digest_source"], "token_id")  # StubBinding has no v2 ABI
        self.assertEqual(record["engine"], "cpp")
        self.assertEqual(record["mode"], "fake")
        self.assertEqual(record["profile"], {"kv": "int8", "digests": True})
        self.assertEqual(record["emitted_token_ids"], [101, 102, 103])
        self.assertEqual(len(record["step_digests"]), 3)
        self.assertEqual(len(record["prompt_token_ids"]), prompt_tokens + 1)

    def test_cancel_and_reset(self):
        self.backend.append_text("prompt")
        with self.assertRaises(GenerationCancelled):
            self.backend.generate(5, should_cancel=lambda: True)
        self.assertTrue(self.binding.cancelled)
        self.backend.reset()
        self.assertEqual(self.backend.tape, [])
        self.assertTrue(self.backend.check_invariant())

    def test_exception_propagation_never_falls_back(self):
        self.backend.append_text("prompt")
        self.binding.raise_on_step = CppEngineError("internal boom", VX_ERR_INTERNAL)
        with self.assertRaisesRegex(CppEngineError, "internal boom"):
            self.backend.generate(2)
        self.binding.raise_on_step = CppEngineError("cancelled", VX_ERR_CANCELLED)
        self.assertEqual(self.binding.raise_on_step.code, VX_ERR_CANCELLED)
        with self.assertRaises(GenerationCancelled):
            self.backend.generate(2)

    def test_session_persistence_and_settings(self):
        self.backend.append_text("alpha")
        self.backend.generate(1)
        self.backend.save_session("turn1")
        prior_tape = list(self.backend.tape)
        self.backend.reset()
        self.assertNotEqual(self.backend.tape, prior_tape)
        self.backend.load_session("turn1")
        self.assertEqual(self.backend.tape, prior_tape)
        self.assertIn("turn1", json.loads(self.backend.list_sessions()))
        self.assertEqual(json.loads(self.backend.delete_session("turn1"))["deleted"], "turn1")
        self.assertNotIn("turn1", json.loads(self.backend.list_sessions()))
        self.assertIn('"engine": "cpp"', self.backend.configure("status"))
        self.assertEqual(self.backend.configure("help"), "cpp engine: status | profile=<json> | help")
        self.assertIn('"engine": "cpp"', self.backend.settings_state())
        self.assertIn("context_window", self.backend.settings_registry().for_backend("cpp"))

    def test_sampler_and_batch_hooks(self):
        from types import SimpleNamespace

        self.backend.sampling = SimpleNamespace(greedy=True, seed=7, temperature=0.0)
        self.backend.append_text("x")
        self.backend.generate(1)
        sampler = json.loads(self.binding.sampler_json)
        self.assertTrue(sampler["greedy"])
        # P2: the seed moved out of sampler_json into vx_session_seed; a seed
        # key here would be refused by the strict C++ parser by name.
        self.assertNotIn("seed", sampler)
        self.assertIsNone(self.backend.fork_for_batch())
        with self.backend.batch_scope():
            self.assertEqual(self.backend.batch_context(), 262144)
            self.backend.batch_prepare(10)
            self.assertEqual(self.backend.batch_state_bytes(10), 10 * (27 * 1024 + 6 * 2 * 2 * 136))
            # Batched prefill is unsupported (C prefill resets state); it fails
            # closed rather than silently returning wrong tokens.
            from veystrix.backends.cpp_engine import CppEngineError
            with self.assertRaises(CppEngineError):
                self.backend.batch_prefill([1, 2, 3], 0, True, sampler)
            with self.assertRaises(CppEngineError):
                self.backend.batch_decode(101, sampler, False)
            self.assertEqual(self.backend.batch_prefix_plan([1], False, 4), (False, None))
            self.backend.batch_save_prefix()
            self.assertEqual(self.backend.batch_stats(tokens=1).tokens, 1)


class SelectorAndAbiTest(unittest.TestCase):
    def test_default_is_python_and_env_selects_cpp(self):
        with patch.dict(os.environ):
            os.environ.pop("VEYSTRIX_ENGINE", None)
            self.assertEqual(engine_name(), "python")
        with patch.dict(os.environ, {"VEYSTRIX_ENGINE": "cpp"}):
            self.assertEqual(engine_name(), "cpp")

    def test_selector_precedence_flag_env_config_default(self):
        resolve = engine_mod.resolve_engine
        self.assertEqual(resolve("python", {"VEYSTRIX_ENGINE": "cpp"}, {"engine": "cpp"}), "python")
        self.assertEqual(resolve(None, {"VEYSTRIX_ENGINE": "cpp"}, {"engine": "python"}), "cpp")
        self.assertEqual(resolve(None, {}, {"engine": "cpp"}), "cpp")
        self.assertEqual(resolve(None, {}, {}), "python")
        with self.assertRaises(ValueError):
            resolve("rust", {}, {})

    def test_cpp_refuses_cleanly_without_per_model_pass(self):
        self.assertIn("no C++ engine", cpp_refusal({}, "flashnext"))
        manifest = {"engines": ["python", "cpp"],
                    "cpp_parity": {"flashnext:agent": "pass"}}
        self.assertIn("cpp_parity: pass", cpp_refusal(manifest, "nemotron35-lightning"))
        self.assertIn("cpp_parity: pass", cpp_refusal(manifest, "flashnext", "plain"))
        # nemotron-only pass: flashnext must still refuse (asymmetry other direction)
        nemo_only = {"engines": ["python", "cpp"],
                     "cpp_parity": {"nemotron35-lightning:agent": "pass"}}
        self.assertIn("flashnext", cpp_refusal(nemo_only, "flashnext"))
        self.assertIsNone(cpp_refusal(nemo_only, "nemotron35-lightning"))
        # engines list without cpp must refuse even with both parity passes
        no_cpp_engine = {"engines": ["python"],
                         "cpp_parity": {"flashnext:agent": "pass",
                                        "nemotron35-lightning:agent": "pass"}}
        self.assertIsNotNone(cpp_refusal(no_cpp_engine, "flashnext"))
        self.assertIsNotNone(cpp_refusal(no_cpp_engine, "nemotron35-lightning"))
        # malformed non-dict parity must refuse cleanly, not crash
        malformed = {"engines": ["python", "cpp"], "cpp_parity": "pass"}
        self.assertIsNotNone(cpp_refusal(malformed, "flashnext"))
        self.assertIsNotNone(cpp_refusal(malformed, "nemotron35-lightning"))

    def test_cpp_allowed_for_both_proven_models(self):
        manifest = {"engines": ["python", "cpp"],
                    "cpp_parity": {"flashnext:agent": "pass",
                                   "nemotron35-lightning:agent": "pass"}}
        self.assertIsNone(cpp_refusal(manifest, "flashnext", "agent"))
        self.assertIsNone(cpp_refusal(manifest, "nemotron35-lightning", "agent"))

    def test_build_command_default_python_path_unchanged(self):
        environ = {}
        command = build_command(port=8080, model="flashnext", checkpoint="/tmp/model",
                                environ=environ, engine="python")
        self.assertEqual(environ, {})
        self.assertIn("-m", command)
        self.assertEqual(command[command.index("-m") + 1], "veystrix.model_router")
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--port") + 1], "8080")

    def test_build_command_cpp_sets_child_engine_selector(self):
        environ = {"VEYSTRIX_SOURCE_CPP": "0"}      # bundle contract, regardless of a local build
        command = build_command(port=9999, model="nemotron35-lightning", checkpoint="",
                                environ=environ, engine="cpp")
        self.assertEqual(environ, {"VEYSTRIX_SOURCE_CPP": "0"})  # build_command is pure
        self.assertEqual(command[command.index("--model") + 1], "nemotron35-lightning")
        self.assertEqual(command[-1], "cpp")  # --engine cpp is on the argv

    def test_build_command_cpp_launches_bundle_engine_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            environ = {}
            command = build_command(port=8080, model="flashnext", checkpoint="/fake/model",
                                    environ=environ, engine="cpp", home=home)
            expected = home / ".veystrix" / "engine" / "bin" / "veystrix-engine"
            self.assertEqual(command[0], str(expected))
            self.assertEqual(command[1], "serve")
            self.assertIn("--engine", command)
            self.assertEqual(command[command.index("--engine") + 1], "cpp")
            self.assertEqual(command[command.index("--port") + 1], "8080")
            self.assertEqual(command[command.index("--model") + 1], "flashnext")
            self.assertEqual(command[command.index("--checkpoint") + 1], "/fake/model")
            # build_command is pure: it does not mutate the caller's environ.
            self.assertEqual(environ, {})

    def test_default_lib_path_prefers_installed_bundle_binary(self):
        suffix = ".dylib" if os.name == "posix" and os.uname().sysname == "Darwin" else ".so"
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            lib = home / ".veystrix" / "engine" / f"libveystrix_c{suffix}"
            lib.parent.mkdir(parents=True)
            lib.write_bytes(b"fake")
            with patch.dict(os.environ, {"VEYSTRIX_HOME": str(home)}):
                self.assertEqual(default_lib_path(), lib)
                self.assertEqual(resolve_lib_path(), lib)

    def test_default_lib_path_finds_legacy_nested_layout(self):
        suffix = ".dylib" if os.name == "posix" and os.uname().sysname == "Darwin" else ".so"
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            lib = home / ".veystrix" / "engine" / "engine" / f"libveystrix_c{suffix}"
            lib.parent.mkdir(parents=True)
            lib.write_bytes(b"fake")
            with patch.dict(os.environ, {"VEYSTRIX_HOME": str(home)}):
                self.assertEqual(default_lib_path(), lib)

    def test_verify_lib_sha256_fails_closed(self):
        import hashlib
        from veystrix.backends.cpp_engine import CppEngineError
        with tempfile.TemporaryDirectory() as directory:
            lib = Path(directory) / "libveystrix_c.dylib"
            lib.write_bytes(b"the-real-attested-binary")
            manifest = Path(directory) / "BUNDLE-MANIFEST.json"
            good = hashlib.sha256(b"the-real-attested-binary").hexdigest()
            # No manifest -> source checkout / stub binding -> no-op.
            verify_lib_sha256(lib, None, None)
            # Manifest present with a matching hash -> loads.
            verify_lib_sha256(lib, good, manifest)
            # Mismatch -> refuse.
            with self.assertRaises(CppEngineError):
                verify_lib_sha256(lib, "0" * 64, manifest)
            # Malformed/missing hash with a manifest -> fail closed (no silent open).
            with self.assertRaises(CppEngineError):
                verify_lib_sha256(lib, None, manifest)

    def test_build_script_requires_explicit_parity_attestation(self):
        text = (ROOT / "packaging" / "engine" / "build_engine_bundle.sh").read_text()
        self.assertIn("VEYSTRIX_CPP_PARITY_ATTEST", text)
        self.assertIn("cpp_lib_sha256", text)
        self.assertIn('mode != "live"', text)  # live-mode smoke check
        self.assertIn("cpp_attested.sha256", text)  # committed provenance anchor
        self.assertIn('install_name_tool -add_rpath "$MLX_LIB_IN_BUNDLE"', text)  # relocatable rpath
        self.assertIn('MLX_LIB_IN_BUNDLE="@loader_path', text)

    def test_build_script_emits_engine_cli_launcher(self):
        text = (ROOT / "packaging" / "engine" / "build_engine_bundle.sh").read_text()
        self.assertIn('mkdir -p "$OUT/bin"', text)
        self.assertIn('cat > "$OUT/bin/veystrix-engine"', text)
        self.assertIn("-m veystrix.engine_cli", text)
        self.assertIn('chmod +x "$OUT/bin/veystrix-engine"', text)

    def test_c_abi_binding_covers_every_header_symbol(self):
        source = (ROOT / "veystrix" / "backends" / "cpp_engine.py").read_text()
        expected = {
            "vx_abi_version": 0, "vx_mode": 0, "vx_last_error": 0,
            "vx_engine_open": 2, "vx_engine_close": 1, "vx_session_open": 1,
            "vx_session_close": 1, "vx_session_prefill": 3, "vx_session_step": 3,
            "vx_session_step_batch": 4,  # optional multi-session ABI (feature-detected)
            "vx_snapshot_save": 2, "vx_snapshot_restore": 2, "vx_cancel": 1,
            "vx_engine_stats": 3, "vx_selftest_throw": 0,
        }
        for name, count in expected.items():
            self.assertIn(f"lib.{name}.argtypes = ", source, name)
            self.assertIn(f"lib.{name}.restype = ", source, name)
            line = next(line for line in source.splitlines() if f"lib.{name}.argtypes = " in line)
            expr = line.split("= [", 1)[1].rsplit("]", 1)[0]
            self.assertEqual(len([part for part in expr.split(",") if part.strip()]), count, name)

    def test_install_for_server_is_idempotent(self):
        import veystrix

        module = types.ModuleType("veystrix.session")
        # `import veystrix.session as m` resolves the package attribute first,
        # so patch both the module table and the parent package attribute.
        with patch.dict(sys.modules, {"veystrix.session": module}), \
             patch.object(veystrix, "session", module):
            install_for_server(module)
            installed = module.build_backend
            self.assertTrue(callable(installed))
            install_for_server(module)
            self.assertIs(module.build_backend, installed)

    def test_default_python_selector_does_not_touch_build_backend(self):
        import veystrix.session as session_module

        original = session_module.build_backend
        with patch.dict(os.environ):
            os.environ.pop("VEYSTRIX_ENGINE", None)
            session_module.install_cpp_selector()
        self.assertIs(session_module.build_backend, original)

    def test_cpp_factory_reaches_session_build_backend(self):
        """The real seam: install_for_server() makes session.build_backend return cpp."""
        import veystrix.session as session_module
        import veystrix.backends.cpp_engine as cpp_module

        original = session_module.build_backend
        had_flag = hasattr(session_module, "_veystrix_cpp_selector_installed")

        def cleanup():
            session_module.build_backend = original
            if not had_flag:
                session_module.__dict__.pop("_veystrix_cpp_selector_installed", None)

        self.addCleanup(cleanup)
        sentinel = object()
        fake_backend = lambda **kwargs: sentinel
        args = types.SimpleNamespace(model_path="/fake/model")
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "flat_config.txt"
            config.write_text("n_layers=1\n")
            env = {"VEYSTRIX_ENGINE": "cpp", "VEYSTRIX_CPP_CONFIG_PATH": str(config)}
            with patch.dict(os.environ, env), \
                 patch.object(cpp_module, "CppNemotronBackend", fake_backend), \
                 patch.object(cpp_module, "_default_tokenizer", lambda model_dir: FakeTokenizer()):
                session_module.install_cpp_selector()
                result = session_module.build_backend("nemotron35-lightning", args, {})
        self.assertIs(result, sentinel)

    def test_source_checkout_script_module_gets_cpp_factory(self):
        """cli.py runs session.py as __main__; the live copy must be rebound."""
        import veystrix
        import veystrix.session as session_module
        import veystrix.backends.cpp_engine as cpp_module

        live = types.ModuleType("__main__")
        live.build_backend = lambda *args: "python-backend"
        sentinel = object()
        saved_module = sys.modules.get("veystrix.session")
        saved_attr = getattr(veystrix, "session", None)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "flat_config.txt"
            config.write_text("n_layers=1\n")
            env = {"VEYSTRIX_ENGINE": "cpp", "VEYSTRIX_CPP_CONFIG_PATH": str(config)}
            with patch.dict(os.environ, env), \
                 patch.object(cpp_module, "CppNemotronBackend", lambda **kwargs: sentinel), \
                 patch.object(cpp_module, "_default_tokenizer", lambda model_dir: FakeTokenizer()):
                session_module.install_cpp_selector(target=live)
                args = types.SimpleNamespace(model_path="/fake/model")
                self.assertIs(live.build_backend("nemotron35-lightning", args, {}), sentinel)
        self.assertIs(sys.modules.get("veystrix.session"), saved_module)
        self.assertIs(getattr(veystrix, "session", None), saved_attr)

    def test_bundle_template_records_both_models(self):
        text = (ROOT / "packaging" / "engine" / "build_engine_bundle.sh").read_text()
        self.assertIn('"flashnext:agent": "pass"', text)
        self.assertIn('"nemotron35-lightning:agent": "pass"', text)

    def test_engine_cli_serve_consumes_engine_flag(self):
        from veystrix import engine_cli
        captured = {}

        def fake_router(argv):
            captured["argv"] = argv
            return 0

        with patch("veystrix.model_router.main", fake_router), \
             patch.dict(os.environ):
            os.environ.pop("VEYSTRIX_ENGINE", None)
            rc = engine_cli.serve(["--port", "8080", "--model", "flashnext", "--engine", "cpp"])
            self.assertEqual(rc, 0)
            self.assertEqual(captured["argv"],
                             ["--host", "127.0.0.1", "--port", "8080", "--model", "flashnext"])
            self.assertEqual(os.environ["VEYSTRIX_ENGINE"], "cpp")

    def test_engine_cli_main_forwards_serve_port(self):
        # Regression for the shipped 1.1.4 bug: argparse's REMAINDER does not
        # capture a remainder that starts with an option, so main() must split
        # `serve` off before parsing. Calling serve() directly (the old test)
        # bypasses that path and let the bug ship.
        from veystrix import engine_cli
        captured = {}

        def fake_router(argv):
            captured["argv"] = argv
            return 0

        cases = [
            (["serve", "--port", "8080", "--model", "flashnext", "--checkpoint", "/x", "--engine", "cpp"],
             ["--host", "127.0.0.1", "--port", "8080", "--model", "flashnext", "--checkpoint", "/x"]),
            (["--home", "/tmp/home", "serve", "--port", "9001", "--model", "ornith"],
             ["--host", "127.0.0.1", "--port", "9001", "--model", "ornith"]),
            (["--home=/tmp/home", "serve", "--port", "9002", "--model", "ornith"],
             ["--host", "127.0.0.1", "--port", "9002", "--model", "ornith"]),
        ]
        with patch("veystrix.model_router.main", fake_router), \
             patch.dict(os.environ):
            for argv, expected in cases:
                with self.subTest(argv=argv):
                    os.environ.pop("VEYSTRIX_ENGINE", None)
                    self.assertEqual(engine_cli.main(argv), 0)
                    self.assertEqual(captured["argv"], expected)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class CppLifecycleTest(unittest.TestCase):
    """Start the tool with cpp selected against a tiny /health mock server."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name)
        self.env = dict(os.environ)
        self.env.update({
            "VEYSTRIX_HOME": str(self.home),
            "VEYSTRIX_TEST_POWER": json.dumps({"on_ac": True, "percent": 100}),
            "VEYSTRIX_TEST_MEMORY": json.dumps({"available_bytes": 16 * 1024**3}),
            "VEYSTRIX_NO_KEYCHAIN": "1",
        })
        self._patcher = patch.dict(os.environ, self.env, clear=False)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        self.addCleanup(stop, self.home)

        manifest = manifest_path(self.home)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({
            "protocol_version": "1", "engines": ["python", "cpp"],
            "cpp_parity": {"flashnext:agent": "pass", "nemotron35-lightning:agent": "pass"},
        }))
        self.server = self.home / "health_cpp.py"
        self.server.write_text(
            "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
            "import json, os\n"
            "class H(BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        engine = os.environ.get('VEYSTRIX_ENGINE', 'python')\n"
            "        body = json.dumps({'ready': True, 'engine': engine, 'model': 'flashnext'}).encode()\n"
            "        self.send_response(200)\n"
            "        self.send_header('Content-Type', 'application/json')\n"
            "        self.send_header('Content-Length', str(len(body)))\n"
            "        self.end_headers()\n"
            "        self.wfile.write(body)\n"
            "    def log_message(self, *args):\n"
            "        pass\n"
            "ThreadingHTTPServer(('127.0.0.1', int(os.environ['VEYSTRIX_PORT'])), H).serve_forever()\n"
        )

    @unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true" and sys.platform == "darwin",
                     "detached watcher loopback probe times out on macOS GitHub runners")
    def test_check_engine_and_status_report_cpp(self):
        check_engine("cpp", self.home, "flashnext")
        check_engine("cpp", self.home, "nemotron35-lightning")
        port = _free_port()
        options = StartOptions(home=self.home, port=port, engine="cpp", model="flashnext",
                               command=[sys.executable, str(self.server)], timeout=30)
        result = start(options)
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["engine"], "cpp")
        self.assertEqual(status(self.home)["engine"], "cpp")
        state = json.loads(state_path(self.home).read_text())
        self.assertEqual(state["engine_env"].get("VEYSTRIX_ENGINE"), "cpp")
        self.assertEqual(stop(self.home)["status"], "stopped")

    def test_health_engine_field_wins_over_state(self):
        self.assertEqual(report_engine({"engine": "cpp", "ready": True}, "python"), "cpp")
        self.assertEqual(report_engine({"ready": True}, "python"), "python")
        self.assertEqual(report_engine({"engine": {"kv_format": "int8"}}, "python"), "python")

    def test_cpp_refuses_before_any_child_is_launched(self):
        manifest = manifest_path(self.home)
        manifest.write_text(json.dumps({"engines": ["python", "cpp"], "cpp_parity": {}}))
        with self.assertRaises(SystemExit) as caught:
            check_engine("cpp", self.home, "flashnext")
        self.assertIn("cpp_parity: pass", str(caught.exception))
        self.assertEqual(load_manifest(self.home)["cpp_parity"], {})


class CppCriticalRegressionTest(unittest.TestCase):
    """Tests for the critical defects found in Opus's M6 review.

    Each test would have caught a real end-to-end failure that the stub-only
    suite masked: selector ordering, missing config_path, unbounded generation,
    missing model_path, unknown-model acceptance and the broken configure().
    """

    def test_fresh_process_selector_ordering(self):
        """A fresh import with VEYSTRIX_ENGINE=cpp must bind the cpp factory."""
        import subprocess
        code = ("import veystrix.session as s; "
                "print(s.build_backend.__name__, s.build_backend.__module__)")
        env = dict(os.environ, VEYSTRIX_ENGINE="cpp", VEYSTRIX_C_CPP_FAKE="1")
        cpp = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env, cwd=str(ROOT))
        self.assertEqual(cpp.returncode, 0, cpp.stderr)
        self.assertIn("build_backend_cpp", cpp.stdout)
        self.assertIn("cpp_engine", cpp.stdout)
        env = dict(os.environ)
        env.pop("VEYSTRIX_ENGINE", None)
        py = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, env=env, cwd=str(ROOT))
        self.assertEqual(py.returncode, 0, py.stderr)
        self.assertIn("build_backend veystrix.session", py.stdout)

    def test_factory_builds_config_path_and_model_key(self):
        """The profile must carry config_path + explicit model (C runtime contract)."""
        import veystrix.backends.cpp_engine as cpp_module
        captured = {}

        def fake_backend(**kwargs):
            captured.update(kwargs)
            return "backend"

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "flat.txt"
            config.write_text("n_layers=1\n")
            env = {"VEYSTRIX_CPP_CONFIG_PATH": str(config)}
            args = types.SimpleNamespace(model_path="/fake/model", session_dir=None)
            with patch.dict(os.environ, env), \
                 patch.object(cpp_module, "CppEngineBackend", fake_backend), \
                 patch.object(cpp_module, "CppNemotronBackend", fake_backend), \
                 patch.object(cpp_module, "_default_tokenizer", lambda d: FakeTokenizer()):
                module = types.ModuleType("sess")
                cpp_module.install_for_server(module)
                module.build_backend("nemotron35-lightning", args, {})
        profile = json.loads(captured["profile_json"])
        self.assertEqual(profile["config_path"], str(config))
        self.assertEqual(profile["model"], "nemotron")
        self.assertTrue(captured["stop_tokens"])

    def test_preferences_reach_the_cpp_sampler(self):
        """Regression: the C++ backend lacked `sampling`, so mirror_preferences skipped it and every
        request ran greedy while Python used the configured sampler."""
        from veystrix.session import mirror_preferences
        backend = CppEngineBackend(model_dir="/fake/model", profile_json='{"kv":"int8"}',
                                   binding=StubBinding(), tokenizer=FakeTokenizer(),
                                   session_dir=tempfile.mkdtemp())
        self.addCleanup(backend.close)
        self.assertTrue(json.loads(backend._sampler_json())["greedy"])
        prefs = {"thinking_enabled": True, "temperature": 0.7, "top_p": 0.8, "top_k": 30, "min_p": 0.05,
                 "presence_penalty": 1.5, "effort": "high", "max_tokens": -1, "think_budget": 256,
                 "profile": "plain"}
        mirror_preferences(backend, prefs, "plain")
        sampler = json.loads(backend._sampler_json())
        self.assertFalse(sampler["greedy"])
        self.assertEqual((sampler["temperature"], sampler["top_k"], sampler["presence_penalty"]), (0.7, 30, 1.5))

    def test_nemotron_factory_defaults_to_chunked_prefill_but_respects_override(self):
        import veystrix.backends.cpp_engine as cpp_module

        def build(model, env):
            with tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / "flat.txt"
                config.write_text("n_layers=1\n")
                args = types.SimpleNamespace(model_path="/fake/model", session_dir=None)
                full = {"VEYSTRIX_CPP_CONFIG_PATH": str(config), **env}
                with patch.dict(os.environ, full), \
                     patch.object(cpp_module, "CppEngineBackend", lambda **k: "b"), \
                     patch.object(cpp_module, "CppNemotronBackend", lambda **k: "b"), \
                     patch.object(cpp_module, "_default_tokenizer", lambda d: FakeTokenizer()):
                    os.environ.pop("VEYSTRIX_CPP_NEMOTRON_CHUNKED", None) if "VEYSTRIX_CPP_NEMOTRON_CHUNKED" not in env else None
                    module = types.ModuleType("sess")
                    cpp_module.install_for_server(module)
                    module.build_backend(model, args, {})
                    return os.environ.get("VEYSTRIX_CPP_NEMOTRON_CHUNKED")

        try:
            self.assertEqual(build("nemotron35-lightning", {}), "1")
            os.environ.pop("VEYSTRIX_CPP_NEMOTRON_CHUNKED", None)
            self.assertEqual(build("nemotron35-lightning", {"VEYSTRIX_CPP_NEMOTRON_CHUNKED": "0"}), "0")
        finally:
            os.environ.pop("VEYSTRIX_CPP_NEMOTRON_CHUNKED", None)

    def test_custom_factory_has_no_qwen_runtime(self):
        """The hand-written Qwen runtime is retired: Flash-Next never reaches the custom factory."""
        import veystrix.backends.cpp_engine as cpp_module
        module = types.ModuleType("sess")
        cpp_module.install_for_server(module)
        with self.assertRaises(SystemExit) as ctx:
            module.build_backend("flashnext", types.SimpleNamespace(model_path="/m"), {})
        self.assertIn("generic runtime", str(ctx.exception))

    def test_factory_sets_measured_decode_defaults_with_escape_hatches(self):
        """The 20 Nov decode defaults are set per model and an explicit 0 wins."""
        import veystrix.backends.cpp_engine as cpp_module
        keys = ("VEYSTRIX_CPP_KEEPWARM", "VEYSTRIX_CPP_MLX_DEST", "VEYSTRIX_CPP_MLX_CACHE_BYTES",
                "VEYSTRIX_CPP_QWEN_SHARED_EARLY", "VEYSTRIX_CPP_NEMO_ONEBATCH",
                "VEYSTRIX_CPP_NEMO_SHARED_EARLY", "VEYSTRIX_CPP_QWEN_PIPE", "VEYSTRIX_CPP_QWEN_ASYNC",
                "VEYSTRIX_CPP_NEMO_PIPE", "VEYSTRIX_CPP_QWEN_SPEC", "VEYSTRIX_CPP_QWEN_SPEC_TOP",
                "VEYSTRIX_CPP_NEMO_PREFETCH", "VEYSTRIX_CPP_NEMO_PREFETCH_TOP", "VEYSTRIX_CPP_GATE_TAIL")

        def build(name, env):
            saved = {k: os.environ.pop(k, None) for k in keys}
            try:
                with tempfile.TemporaryDirectory() as directory:
                    config = Path(directory) / "flat.txt"
                    config.write_text("n_layers=1\n")
                    args = types.SimpleNamespace(model_path="/fake/model", session_dir=None)
                    with patch.dict(os.environ, {"VEYSTRIX_CPP_CONFIG_PATH": str(config), **env}), \
                         patch.object(cpp_module, "CppEngineBackend", lambda **k: "b"), \
                         patch.object(cpp_module, "CppNemotronBackend", lambda **k: "b"), \
                         patch.object(cpp_module, "_default_tokenizer", lambda d: FakeTokenizer()):
                        module = types.ModuleType("sess")
                        cpp_module.install_for_server(module)
                        module.build_backend(name, args, {})
                        return {k: os.environ.get(k) for k in keys}
            finally:
                for k, v in saved.items():
                    if v is not None:
                        os.environ[k] = v

        n = build("nemotron35-lightning", {})
        self.assertEqual(n["VEYSTRIX_CPP_NEMO_ONEBATCH"], "1")
        self.assertEqual(n["VEYSTRIX_CPP_NEMO_SHARED_EARLY"], "1")
        self.assertIsNone(n["VEYSTRIX_CPP_QWEN_SHARED_EARLY"])
        self.assertEqual(n["VEYSTRIX_CPP_NEMO_PIPE"], "1")
        self.assertIsNone(n["VEYSTRIX_CPP_QWEN_PIPE"])
        self.assertEqual(n["VEYSTRIX_CPP_NEMO_PREFETCH"], "1")
        self.assertEqual(n["VEYSTRIX_CPP_NEMO_PREFETCH_TOP"], "2")
        self.assertEqual(n["VEYSTRIX_CPP_GATE_TAIL"], "2")
        self.assertIsNone(n["VEYSTRIX_CPP_QWEN_SPEC"])
        off2 = build("nemotron35-lightning", {"VEYSTRIX_CPP_NEMO_PIPE": "0"})
        self.assertEqual(off2["VEYSTRIX_CPP_NEMO_PIPE"], "0")
        off = build("nemotron35-lightning", {"VEYSTRIX_CPP_KEEPWARM": "0",
                                             "VEYSTRIX_CPP_MLX_CACHE_BYTES": "0"})
        self.assertEqual(off["VEYSTRIX_CPP_KEEPWARM"], "0")
        self.assertEqual(off["VEYSTRIX_CPP_MLX_CACHE_BYTES"], "0")

    def test_factory_auto_generates_flat_config(self):
        """With no VEYSTRIX_CPP_CONFIG_PATH, the checkpoint dir is enough."""
        import veystrix.backends.cpp_engine as cpp_module
        captured = {}

        def fake_backend(**kwargs):
            captured.update(kwargs)
            return "backend"

        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "auto-flat.txt"
            generated.write_text("n_layers=1\n")
            # No VEYSTRIX_CPP_CONFIG_PATH in the env.
            env = {"VEYSTRIX_CPP_CONFIG_PATH": ""}
            args = types.SimpleNamespace(model_path="/fake/model", session_dir=None)
            with patch.dict(os.environ, env, clear=False), \
                 patch.object(cpp_module, "_auto_flat_config", return_value=generated), \
                 patch.object(cpp_module, "CppEngineBackend", fake_backend), \
                 patch.object(cpp_module, "CppNemotronBackend", fake_backend), \
                 patch.object(cpp_module, "_default_tokenizer", lambda d: FakeTokenizer()):
                os.environ.pop("VEYSTRIX_CPP_CONFIG_PATH", None)
                module = types.ModuleType("sess")
                cpp_module.install_for_server(module)
                module.build_backend("nemotron35-lightning", args, {})
        profile = json.loads(captured["profile_json"])
        self.assertEqual(profile["config_path"], str(generated))
        self.assertEqual(profile["model"], "nemotron")

    def test_translate_env_value_aware_and_kv_mapping(self):
        """Launcher defaults (MTP=off, COMPILE_GDN=1) must not block cpp;
        KV tuning flags must be mapped, not silently dropped."""
        import veystrix.backends.cpp_engine as cpp_module
        base = {"FLASHNEXT_NATIVE_MTP": "off", "FLASHNEXT_COMPILE_GDN": "1"}
        with patch.dict(os.environ, base, clear=False):
            cpp_module.translate_env()  # must not raise
        with patch.dict(os.environ, {"FLASHNEXT_NATIVE_MTP": "on"}, clear=False):
            with self.assertRaises(SystemExit):
                cpp_module.translate_env()
        with patch.dict(os.environ, {"FLASHNEXT_KV_GROUP_SIZE": "64",
                                     "FLASHNEXT_KV_RECENT_TOKENS": "512"}, clear=False):
            os.environ.pop("VEYSTRIX_CPP_KV_GROUP_SIZE", None)
            os.environ.pop("VEYSTRIX_CPP_KV_RECENT_TOKENS", None)
            cpp_module.translate_env()
            self.assertEqual(os.environ["VEYSTRIX_CPP_KV_GROUP_SIZE"], "64")
            self.assertEqual(os.environ["VEYSTRIX_CPP_KV_RECENT_TOKENS"], "512")
            os.environ.pop("VEYSTRIX_CPP_KV_GROUP_SIZE", None)
            os.environ.pop("VEYSTRIX_CPP_KV_RECENT_TOKENS", None)

    def test_auto_flat_config_fails_closed(self):
        """A checkpoint dir that cannot yield a config produces no config."""
        import veystrix.backends.cpp_engine as cpp_module
        with tempfile.TemporaryDirectory() as directory:
            result = cpp_module._auto_flat_config(directory, "nemotron")
            self.assertIsNone(result)  # no config.json/index -> cannot generate

    def test_factory_refuses_unknown_model(self):
        import veystrix.backends.cpp_engine as cpp_module
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "flat.txt"
            config.write_text("x=1\n")
            with patch.dict(os.environ, {"VEYSTRIX_CPP_CONFIG_PATH": str(config)}):
                module = types.ModuleType("sess")
                cpp_module.install_for_server(module)
                with self.assertRaises(SystemExit):
                    module.build_backend("mystery-model",
                                         types.SimpleNamespace(model_path="/m"), {})

    def test_second_turn_appends_only_new_tokens_on_a_live_session(self):
        """Regression: the live-continuation path called a method that exists only on the binding
        (`self._check`), so every second turn on the C++ engine failed with an AttributeError."""
        class V2Stub(StubBinding):
            has_v2 = True

            def __init__(self):
                super().__init__(tokens=(11, 12, 13, 14))
                self.appended = []

            def prefill_ex(self, session, tokens, flags):
                self.appended.append((list(tokens), flags))

        binding = V2Stub()
        backend = CppEngineBackend(model_dir="/fake/m", profile_json="{}", binding=binding,
                                   tokenizer=FakeTokenizer(), stop_tokens={999999})
        self.addCleanup(backend.close)
        backend.generate(2)
        self.assertEqual(binding.appended, [])                  # first turn is a cold prefill
        backend.pending.extend([21, 22, 23])
        backend.generate(2)
        self.assertEqual(binding.appended, [([21, 22, 23], 1)])  # second turn appends only the new tokens

    def test_generate_reports_length_and_opens_turn_on_horizon(self):
        """Hitting max_tokens must be finish=length / turn_closed=False."""
        binding = StubBinding(tokens=(11, 12, 13, 14))
        backend = CppEngineBackend(model_dir="/fake/m", profile_json="{}",
                                   binding=binding, tokenizer=FakeTokenizer(),
                                   stop_tokens={999999})
        self.addCleanup(backend.close)
        answer, stats = backend.generate(4)
        self.assertEqual(stats.finish, "length")
        self.assertFalse(backend.turn_closed)
        self.assertEqual(stats.tokens, 4)

    def test_generate_reports_stop_on_stop_token(self):
        binding = StubBinding(tokens=(11, 12, 999, 14))
        backend = CppEngineBackend(model_dir="/fake/m", profile_json="{}",
                                   binding=binding, tokenizer=FakeTokenizer(),
                                   stop_tokens={999})
        self.addCleanup(backend.close)
        _, stats = backend.generate(10)
        self.assertEqual(stats.finish, "stop")
        self.assertTrue(backend.turn_closed)
        # Stop tokens are protocol delimiters: on the tape, not in tokens/text.
        self.assertEqual(stats.tokens, 2)
        self.assertEqual(backend.tape[-1], 999)

    def test_stop_token_text_not_streamed_but_on_tape(self):
        """Python never yields stop ids; cpp must not either (Sonnet #1)."""
        pieces = []
        binding = StubBinding(tokens=(11, 12, 999, 14))
        backend = CppEngineBackend(model_dir="/fake/m", profile_json="{}",
                                   binding=binding, tokenizer=FakeTokenizer(),
                                   stop_tokens={999})
        self.addCleanup(backend.close)
        answer, stats = backend.generate(10, out=pieces.append)
        self.assertEqual(stats.tokens, 2)
        self.assertEqual(pieces, ["<11>", "<12>"])
        self.assertNotIn("<999>", answer)
        self.assertEqual(backend.tape[-1], 999)

    def test_utf8_split_across_tokens_is_not_corrupted(self):
        """stream_decode holds back partial multibyte characters (Sonnet #2)."""
        class Utf8Tokenizer(FakeTokenizer):
            def decode(self, ids):
                if list(ids) == [1]:
                    return "\ufffd"
                if list(ids) == [1, 2]:
                    return "😀"
                return super().decode(ids)

        pieces = []
        binding = StubBinding(tokens=(1, 2, 5))
        backend = CppEngineBackend(model_dir="/fake/m", profile_json="{}",
                                   binding=binding, tokenizer=Utf8Tokenizer(),
                                   stop_tokens=set())
        self.addCleanup(backend.close)
        answer, stats = backend.generate(3, out=pieces.append)
        self.assertEqual(answer, "😀<5>")
        self.assertNotIn("\ufffd", answer)
        self.assertEqual(stats.tokens, 3)

    def test_model_path_exposed_for_router_ready(self):
        backend = CppEngineBackend(model_dir="/my/checkpoint", profile_json="{}",
                                   binding=StubBinding(), tokenizer=FakeTokenizer())
        self.addCleanup(backend.close)
        self.assertEqual(backend.model_path, "/my/checkpoint")

    def test_configure_profile_closes_old_engine_and_session(self):
        binding = StubBinding()
        backend = CppEngineBackend(model_dir="/m", profile_json='{"a":1}',
                                   binding=binding, tokenizer=FakeTokenizer())
        self.addCleanup(backend.close)
        old_engine = backend._engine
        result = backend.configure('profile={"b":2}')
        self.assertIn("profile updated", result)
        self.assertIsNot(backend._engine, old_engine)
        self.assertEqual(json.loads(backend.profile_json), {"b": 2})
        self.assertIsNotNone(backend._session)

    def test_empty_prefill_is_noop(self):
        from veystrix.backends.cpp_engine import CtypesVxBinding
        # Patch resolve_lib_path so no real dylib is needed; patch _check to
        # detect any C call. Empty prefill must not reach vx_session_prefill.
        with patch("veystrix.backends.cpp_engine.resolve_lib_path", return_value="/fake"), \
             patch.object(CtypesVxBinding, "_check", side_effect=AssertionError("C called")):
            binding = CtypesVxBinding.__new__(CtypesVxBinding)
            binding.prefill(object(), [])  # no raise = no C call
        self.assertTrue(True)

    def test_non_greedy_sampling_accepted(self):
        """P2: THINKING/INSTRUCT sampling reaches the C++ runtime."""
        from types import SimpleNamespace
        backend = CppEngineBackend(model_dir="/m", profile_json="{}",
                                   binding=StubBinding(), tokenizer=FakeTokenizer())
        self.addCleanup(backend.close)
        backend.sampling = SimpleNamespace(greedy=False, temperature=0.8, top_p=0.9,
                                           top_k=20, min_p=0.0, presence_penalty=1.5)
        backend.generate(1)
        sampler = json.loads(backend._binding.sampler_json)
        self.assertFalse(sampler["greedy"])
        self.assertEqual(sampler["temperature"], 0.8)
        self.assertEqual(sampler["presence_penalty"], 1.5)
        self.assertNotIn("seed", sampler)


class BatchFlagOffRegistryTest(unittest.TestCase):
    """VEYSTRIX_CPP_BATCH unset/'0' must never touch the shared-engine registry.

    A temp HOME + VEYSTRIX_C_CPP_FAKE=1 + the deterministic StubBinding keep this
    fully offline: no real dylib and no checkpoint are ever loaded. The registry
    get-or-create function (_acquire_shared_engine) is wrapped to count calls.
    """

    @staticmethod
    def _make_backend(home, binding=None):
        return CppEngineBackend(
            model_dir="/fake/model", profile_json='{"kv":"int8"}',
            binding=binding if binding is not None else StubBinding(),
            tokenizer=FakeTokenizer(),
            session_dir=str(Path(home) / "sessions"),
        )

    def test_flag_off_never_creates_shared_engine(self):
        import veystrix.backends.cpp_engine as cpp_module

        calls = []
        real_acquire = cpp_module._acquire_shared_engine

        def counting_acquire(binding, model_dir, profile_json):
            calls.append((str(model_dir), str(profile_json)))
            return real_acquire(binding, model_dir, profile_json)

        with tempfile.TemporaryDirectory() as home:
            env = {"HOME": home, "VEYSTRIX_HOME": home, "VEYSTRIX_C_CPP_FAKE": "1"}
            with patch.dict(os.environ, env, clear=False), \
                 patch.object(cpp_module, "_acquire_shared_engine", counting_acquire):
                # --- flag UNSET: zero acquisitions, each backend owns its engine ---
                cpp_module._reset_shared_engines()
                os.environ.pop(cpp_module.BATCH_ENV, None)
                off_backends = [self._make_backend(home) for _ in range(2)]
                try:
                    self.assertEqual(calls, [],
                                     "flag off must not acquire a shared engine")
                    for backend in off_backends:
                        self.assertIsNone(backend.fork_for_batch())
                    self.assertIsNot(off_backends[0]._engine, off_backends[1]._engine)
                finally:
                    for backend in off_backends:
                        backend.close()

                # --- flag explicitly '0': still zero acquisitions ---
                calls.clear()
                cpp_module._reset_shared_engines()
                os.environ[cpp_module.BATCH_ENV] = "0"
                off_zero = self._make_backend(home)
                try:
                    self.assertEqual(calls, [],
                                     "BATCH_ENV=0 must not acquire a shared engine")
                    self.assertIsNone(off_zero.fork_for_batch())
                finally:
                    off_zero.close()

                # --- flag '1': every backend goes through the registry ---
                calls.clear()
                cpp_module._reset_shared_engines()
                os.environ[cpp_module.BATCH_ENV] = "1"
                on_backends = [self._make_backend(home) for _ in range(2)]
                try:
                    # The get-or-create function is called once per backend;
                    # the identical key dedups them onto one shared engine.
                    self.assertEqual(len(calls), 2,
                                     "flag on must acquire a shared engine per backend")
                    self.assertEqual(len(set(calls)), 1)
                    self.assertIs(on_backends[0]._engine, on_backends[1]._engine)
                finally:
                    for backend in on_backends:
                        backend.close()
                cpp_module._reset_shared_engines()


if __name__ == "__main__":
    unittest.main(verbosity=2)
