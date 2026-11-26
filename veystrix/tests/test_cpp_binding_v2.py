"""P1 gate: an old C++ dylib must fail closed for every ABI v2 symbol.

No hardware or real checkpoint: the dylib is a Python stand-in that exposes
only the pre-v2 surface. The binding must name the missing symbol rather than
assume it exists (same discipline as vx_session_step_batch).
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from veystrix.backends import cpp_engine as mod


BASE_SYMBOLS = (
    "vx_abi_version", "vx_mode", "vx_last_error", "vx_engine_open",
    "vx_engine_close", "vx_session_open", "vx_session_close",
    "vx_session_prefill", "vx_session_step", "vx_session_step_batch",
    "vx_snapshot_save", "vx_snapshot_restore", "vx_cancel",
    "vx_engine_stats", "vx_selftest_throw",
)

V2_SYMBOLS = (
    "vx_engine_capabilities", "vx_session_prefill_ex", "vx_session_seed",
    "vx_session_last_digest", "vx_session_fork", "vx_session_tokens",
)


class OldDylib:
    """Only the legacy surface; any v2 attribute raises AttributeError."""

    def __init__(self):
        for name in BASE_SYMBOLS:
            setattr(self, name, lambda *a, **k: 0)
        self.vx_mode = lambda: b"live"
        self.vx_last_error = lambda: b""
        self.vx_abi_version = lambda: 1

    def __getattr__(self, name):
        raise AttributeError(name)


class BindingV2Tests(unittest.TestCase):
    def _binding(self, lib):
        with patch.object(mod, "resolve_lib_path", return_value="/dummy/libveystrix_c.dylib"), \
             patch.object(mod, "_attested_lib_sha256", return_value=(None, None)), \
             patch.object(mod, "verify_lib_sha256", return_value=None), \
             patch.object(mod.ctypes, "CDLL", return_value=lib):
            return mod.CtypesVxBinding("/dummy/libveystrix_c.dylib")

    def test_old_dylib_detected_and_every_v2_call_fails_closed(self):
        binding = self._binding(OldDylib())
        self.assertFalse(binding.has_v2)
        for symbol in V2_SYMBOLS:
            with self.subTest(symbol=symbol):
                with self.assertRaises(mod.CppEngineError) as caught:
                    binding.require_v2(symbol)
                self.assertIn(symbol, str(caught.exception))
                self.assertIn("does not export", str(caught.exception))

    def test_chunk_plan_uses_python_planner_and_old_dylib_fails_closed(self):
        class FakeBinding:
            has_v2 = False
            def prefill(self, session, tokens):
                raise AssertionError("must not prefill when the chunk ABI is missing")

        backend = mod.CppEngineBackend.__new__(mod.CppEngineBackend)
        backend._binding = FakeBinding()
        backend._session = object()
        backend.tape = [1, 2, 3]
        with patch.dict("os.environ", {"FLASHNEXT_PREFILL_CHUNK_TOKENS": "64"}):
            spans = backend._chunk_spans(130, 3)
        # Python's planner must be the authority, not a reimplementation.
        self.assertIsNotNone(spans)
        self.assertEqual(sum(b - a for a, b in spans), 130)
        with patch.dict("os.environ", {"FLASHNEXT_PREFILL_CHUNK_TOKENS": "64"}):
            with self.assertRaises(mod.CppEngineError) as caught:
                backend._prefill_with_chunks(list(range(130)))
        self.assertIn("v2 APPEND prefill ABI", str(caught.exception))

    def test_no_chunk_env_keeps_single_call(self):
        class FakeBinding:
            has_v2 = True
            def __init__(self):
                self.calls = []
            def prefill(self, session, tokens):
                self.calls.append(("reset", len(tokens)))
            def prefill_ex(self, session, tokens, flags):
                self.calls.append(("append", len(tokens), flags))

        backend = mod.CppEngineBackend.__new__(mod.CppEngineBackend)
        backend._binding = FakeBinding()
        backend._session = object()
        backend.tape = []
        with patch.dict("os.environ"):
            os.environ.pop("FLASHNEXT_PREFILL_CHUNK_TOKENS", None)
            backend._prefill_with_chunks(list(range(100)))
        self.assertEqual(backend._binding.calls, [("reset", 100)])

    def test_prefix_cache_restores_longest_strict_prefix(self):
        class FakeBinding:
            has_v2 = True
            def __init__(self):
                self.calls = []
                self.saved = []
            def abi_version(self): return 2
            def prefill(self, session, tokens):
                self.calls.append(("reset", len(tokens)))
            def prefill_ex(self, session, tokens, flags):
                self.calls.append(("append", len(tokens), flags))
            def save(self, session, directory):
                self.saved.append(directory)
            def restore(self, session, directory):
                self.calls.append(("restore", directory))

        backend = mod.CppEngineBackend.__new__(mod.CppEngineBackend)
        backend._binding = FakeBinding()
        backend._session = object()
        backend.tape = []
        backend.session_dir = Path(tempfile.mkdtemp())
        backend.model_dir = "/m"
        backend.profile_json = "{}"
        mod._CPP_PREFIX_CACHE.clear()
        with patch.dict("os.environ", {mod.PREFIX_CACHE_ENV: "1"}):
            backend._prefix_store([1, 2, 3])
            backend._binding.calls.clear()
            backend._prefill_with_prefix([1, 2, 3, 4, 5])
        calls = backend._binding.calls
        self.assertEqual(calls[0][0], "restore")
        self.assertEqual(calls[1], ("append", 2, 1))
        self.assertEqual(len(backend._binding.saved), 1)
        mod._CPP_PREFIX_CACHE.clear()

    def test_prefix_cache_stores_snapshot_dir_and_eviction_is_bounded(self):
        """Restore reads <snapshot_dir>/meta.txt; a quota breach must not wipe
        the whole cache (Sonnet findings 8/9)."""
        class B:
            has_v2 = True
            def abi_version(self): return 2
            def save(self, session, directory):
                Path(directory).mkdir(parents=True, exist_ok=True)
                (Path(directory) / "meta.txt").write_bytes(b"x" * 200_000)
        backend = mod.CppEngineBackend.__new__(mod.CppEngineBackend)
        backend._binding = B()
        backend._session = object()
        backend.session_dir = Path(tempfile.mkdtemp())
        backend.model_dir = "/m"
        backend.profile_json = "{}"
        mod._CPP_PREFIX_CACHE.clear()
        with patch.dict("os.environ", {mod.PREFIX_CACHE_ENV: "1",
                                       mod.PREFIX_CACHE_QUOTA_ENV: "0.0001"}):
            backend._prefix_store([1, 2, 3])
        # The cached path must point at vx_snapshot (where meta.txt lives).
        for path in mod._CPP_PREFIX_CACHE.values():
            self.assertTrue(path.endswith("vx_snapshot"), path)
            self.assertTrue((Path(path) / "meta.txt").is_file(), path)
        # A tiny quota evicts the just-stored entry only; no crash, cache empty.
        self.assertEqual(len(mod._CPP_PREFIX_CACHE), 0)
        mod._CPP_PREFIX_CACHE.clear()

    def test_window_env_plumbs_into_profile_and_backend_attrs(self):
        class MinimalBinding:
            def abi_version(self): return 1
            def open_engine(self, model_dir, profile_json): return object()
            def open_session(self, engine): return object()
            def close_session(self, session): pass
            def close_engine(self, engine): pass
            has_v2 = False
            def prefill(self, session, tokens): pass
            def step(self, session, sampler_json): return 1
            def cancel(self, session): pass
            @property
            def last_error(self): return ""
            def save(self, session, directory): pass
            def restore(self, session, directory): pass
            def stats(self, engine): return "{}"
            def capabilities(self, engine): return {}
            def mode(self): return "fake"

        class Tokenizer:
            added_tokens_decoder = {}
            def decode(self, ids): return "t"
            def encode(self, text, add_special_tokens=False): return [1]

        with patch.dict("os.environ", {"VEYSTRIX_CONTEXT_LIMIT": "32768",
                                       "VEYSTRIX_MAX_COLD_PREFILL": "65536"}):
            backend = mod.CppEngineBackend(model_dir="/m", profile_json="{}",
                                           binding=MinimalBinding(), tokenizer=Tokenizer())
            try:
                self.assertEqual(backend.context_limit, 32768)
                self.assertEqual(backend.max_cold_prefill, 65536)
                profile = json.loads(backend.profile_json)
                self.assertEqual(profile["context_window"], 32768)
                self.assertEqual(profile["max_cold_prefill"], 65536)
            finally:
                backend.close()

    def test_shadow_v2_records_real_logit_digest(self):
        import tempfile
        class V2Binding:
            has_v2 = True
            def __init__(self): self.digests = []
            def abi_version(self): return 2
            def mode(self): return "live"
            def open_engine(self, d, p): return object()
            def open_session(self, e): return object()
            def close_session(self, s): pass
            def close_engine(self, e): pass
            def prefill(self, s, t): pass
            def step(self, s, sampler): return 7
            def seed(self, s, seed): pass
            def last_digest(self, s): return "a" * 64
            def save(self, s, d): pass
            def restore(self, s, d): pass
            def cancel(self, s): pass
            def stats(self, e): return "{}"
            @property
            def last_error(self): return ""
        class Tok:
            added_tokens_decoder = {}
            def decode(self, ids): return "t"
            def encode(self, text, add_special_tokens=False): return [1]

        log = Path(tempfile.mkdtemp()) / "shadow.jsonl"
        binding = V2Binding()
        with patch.dict("os.environ", {mod.SHADOW_ENV: str(log)}):
            backend = mod.CppEngineBackend(model_dir="/m", profile_json="{}",
                                           binding=binding, tokenizer=Tok())
            try:
                backend.append_text("x")
                backend.generate(1)
            finally:
                backend.close()
        profile = json.loads(backend.profile_json)
        self.assertTrue(profile["digests"])  # enabled by shadow
        rec = json.loads(log.read_text().strip())
        self.assertEqual(rec["schema"], 2)
        self.assertEqual(rec["digest_source"], "logit")
        self.assertEqual(rec["step_digests"], ["a" * 64])
        self.assertIn("seed", rec)
        self.assertIn("kv", rec)

    def test_complete_dylib_detects_all_v2_symbols(self):
        class Complete(OldDylib):
            def __init__(self):
                super().__init__()
                for name in V2_SYMBOLS:
                    setattr(self, name, lambda *a, **k: 0)

        binding = self._binding(Complete())
        self.assertTrue(binding.has_v2)
        for symbol in V2_SYMBOLS:
            binding.require_v2(symbol)  # must not raise

    def test_prefix_store_supersedes_turn_prefixes_and_keeps_system_anchor(self):
        import tempfile
        class FakeBinding:
            has_v2 = True
            def abi_version(self): return 2
            def save(self, session, directory):
                Path(directory).mkdir(parents=True, exist_ok=True)
                (Path(directory) / "meta.txt").write_text("ok\n")
                (Path(directory) / "generic_state.safetensors").write_bytes(b"x" * 1024)

        backend = mod.CppEngineBackend.__new__(mod.CppEngineBackend)
        backend._binding = FakeBinding()
        backend._session = object()
        backend.session_dir = Path(tempfile.mkdtemp())
        backend.model_dir = "/m"
        backend.profile_json = "{}"
        mod._CPP_PREFIX_CACHE.clear()
        mod._CPP_PREFIX_SYSTEM_ANCHORS.clear()
        try:
            with patch.dict("os.environ", {mod.PREFIX_CACHE_ENV: "1",
                                           mod.PREFIX_CACHE_QUOTA_ENV: "1.0"}):
                backend._prefix_store([1, 2, 3], system_anchor=True)
                backend._prefix_store([1, 2, 3, 4, 5])
                backend._prefix_store([1, 2, 3, 4, 5, 6, 7])
            # System anchor [1, 2, 3] and latest turn [1..7] remain; intermediate [1..5] was superseded.
            stored_tokens = {toks for (_ns, toks) in mod._CPP_PREFIX_CACHE}
            self.assertEqual(stored_tokens, {(1, 2, 3), (1, 2, 3, 4, 5, 6, 7)})
            on_disk = list((backend.session_dir / "prefix-cache").iterdir())
            self.assertEqual(len(on_disk), 2)
        finally:
            mod._CPP_PREFIX_CACHE.clear()
            mod._CPP_PREFIX_SYSTEM_ANCHORS.clear()


if __name__ == "__main__":
    unittest.main(verbosity=2)
