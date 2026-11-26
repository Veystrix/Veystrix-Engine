"""CPU-only ctypes round-trip for the C ABI (no checkpoint, no GPU, no MLX).

Run: python3 cpp/engine/capi/test_capi_smoke.py
It builds the standalone dylib with CMake, then exercises the exported
surface: handles, prefill/step, snapshots, stats, cancellation, the default
"not wired" stub, and the forced-C++-throw exception-safety test.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

VX_OK = 0
VX_ERR_INVALID_ARGUMENT = -1
VX_ERR_NOT_WIRED = -2
VX_ERR_CANCELLED = -3
VX_ERR_INTERNAL = -4
VX_ERR_CONTEXT_LIMIT = -5

APPROVED_TMP = Path("/private/var/folders/37/fn6fxqyd2rgfmvc4954dggbm0000gn/T/opencode")


def build_dylib():
    capi = Path(__file__).resolve().parent
    repo = capi.parents[2]
    work = Path(tempfile.mkdtemp(prefix="vx-capi-smoke-", dir=str(APPROVED_TMP) if APPROVED_TMP.is_dir() else None))
    build = work / "build"
    cfg = subprocess.run(
        ["cmake", "-S", str(capi), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
        text=True, capture_output=True,
    )
    if cfg.returncode:
        raise RuntimeError(f"cmake configure failed:\n{cfg.stdout}\n{cfg.stderr}")
    bld = subprocess.run(["cmake", "--build", str(build), "--parallel"], text=True, capture_output=True)
    if bld.returncode:
        raise RuntimeError(f"cmake build failed:\n{bld.stdout}\n{bld.stderr}")
    suffix = ".dylib" if os.uname().sysname == "Darwin" else ".so"
    return work, build / f"libveystrix_c{suffix}", cfg.stdout + bld.stdout + bld.stderr


class CapSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work, cls.dylib, cls.build_log = build_dylib()
        cls.repo = Path(__file__).resolve().parents[3]
        cls.lib = ctypes.CDLL(str(cls.dylib))
        lib = cls.lib
        lib.vx_abi_version.argtypes = []
        lib.vx_abi_version.restype = ctypes.c_int
        lib.vx_mode.argtypes = []
        lib.vx_mode.restype = ctypes.c_char_p
        lib.vx_last_error.argtypes = []
        lib.vx_last_error.restype = ctypes.c_char_p
        lib.vx_engine_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.vx_engine_open.restype = ctypes.c_void_p
        lib.vx_engine_close.argtypes = [ctypes.c_void_p]
        lib.vx_engine_close.restype = None
        lib.vx_session_open.argtypes = [ctypes.c_void_p]
        lib.vx_session_open.restype = ctypes.c_void_p
        lib.vx_session_close.argtypes = [ctypes.c_void_p]
        lib.vx_session_close.restype = None
        lib.vx_session_prefill.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int]
        lib.vx_session_prefill.restype = ctypes.c_int
        lib.vx_session_step.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32)]
        lib.vx_session_step.restype = ctypes.c_int
        lib.vx_snapshot_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.vx_snapshot_save.restype = ctypes.c_int
        lib.vx_snapshot_restore.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.vx_snapshot_restore.restype = ctypes.c_int
        lib.vx_cancel.argtypes = [ctypes.c_void_p]
        lib.vx_cancel.restype = ctypes.c_int
        lib.vx_engine_stats.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        lib.vx_engine_stats.restype = ctypes.c_int
        lib.vx_selftest_throw.argtypes = []
        lib.vx_selftest_throw.restype = ctypes.c_int
        # ABI v2 (optional symbols: old dylibs must fail closed in the binding)
        lib.vx_engine_capabilities.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        lib.vx_engine_capabilities.restype = ctypes.c_int
        lib.vx_session_prefill_ex.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
                                              ctypes.c_int, ctypes.c_uint32]
        lib.vx_session_prefill_ex.restype = ctypes.c_int
        lib.vx_session_seed.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.vx_session_seed.restype = ctypes.c_int
        lib.vx_session_last_digest.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        lib.vx_session_last_digest.restype = ctypes.c_int
        lib.vx_session_fork.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.vx_session_fork.restype = ctypes.c_int
        lib.vx_session_tokens.argtypes = [ctypes.c_void_p]
        lib.vx_session_tokens.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def error(self) -> str:
        value = self.lib.vx_last_error()
        return value.decode("utf-8", "replace") if value else ""

    def test_abi_version_and_mode(self):
        self.assertEqual(self.lib.vx_abi_version(), 2)
        with patch.dict(os.environ, {"VEYSTRIX_C_CPP_FAKE": "1"}):
            self.assertEqual(self.lib.vx_mode(), b"fake")

    def test_default_stub_reports_not_wired(self):
        with patch.dict(os.environ):
            os.environ.pop("VEYSTRIX_C_CPP_FAKE", None)
            self.assertEqual(self.lib.vx_mode(), b"stub")
            handle = self.lib.vx_engine_open(b"/no/checkpoint", b"{}")
            self.assertFalse(handle)
            self.assertIn("not wired", self.error())

    def test_engine_session_prefill_step_stats_cancel(self):
        with patch.dict(os.environ, {"VEYSTRIX_C_CPP_FAKE": "1"}):
            engine = self.lib.vx_engine_open(b"/tmp/fake-model", b'{"kv":"int8"}')
            self.assertTrue(engine)
            session = self.lib.vx_session_open(engine)
            self.assertTrue(session)
            try:
                ids = (ctypes.c_uint32 * 3)(7, 8, 9)
                self.assertEqual(self.lib.vx_session_prefill(session, ids, 3), VX_OK)
                out = ctypes.c_uint32()
                sampler = b'{"fake_tokens":[11,12,13]}'
                self.assertEqual(self.lib.vx_session_step(session, sampler, ctypes.byref(out)), VX_OK)
                self.assertEqual(out.value, 11)
                self.assertEqual(self.lib.vx_session_step(session, sampler, ctypes.byref(out)), VX_OK)
                self.assertEqual(out.value, 12)
                buf = ctypes.create_string_buffer(2048)
                self.assertEqual(self.lib.vx_engine_stats(engine, buf, len(buf)), VX_OK)
                stats = json.loads(buf.value.decode())
                self.assertEqual(stats["mode"], "fake")
                self.assertEqual(stats["session_tokens"], 5)  # 3 prompt + 2 emitted
                self.assertEqual(self.lib.vx_cancel(session), VX_OK)
                self.assertEqual(self.lib.vx_session_step(session, sampler, ctypes.byref(out)), VX_ERR_CANCELLED)
                self.assertIn("cancelled", self.error())
            finally:
                self.lib.vx_session_close(session)
                self.lib.vx_engine_close(engine)

    def test_snapshot_save_restore_roundtrip(self):
        with patch.dict(os.environ, {"VEYSTRIX_C_CPP_FAKE": "1"}):
            engine = self.lib.vx_engine_open(b"/tmp/fake-model", b"{}")
            session = self.lib.vx_session_open(engine)
            directory = str(Path(self.work) / "snapshot")
            try:
                ids = (ctypes.c_uint32 * 2)(4, 5)
                self.assertEqual(self.lib.vx_session_prefill(session, ids, 2), VX_OK)
                out = ctypes.c_uint32()
                sampler = b'{"fake_tokens":[21,22,23]}'
                self.assertEqual(self.lib.vx_session_step(session, sampler, ctypes.byref(out)), VX_OK)
                self.assertEqual(out.value, 21)
                self.assertEqual(self.lib.vx_snapshot_save(session, directory.encode()), VX_OK)
                self.assertEqual(self.lib.vx_session_step(session, sampler, ctypes.byref(out)), VX_OK)
                self.assertEqual(out.value, 22)
                self.assertEqual(self.lib.vx_snapshot_restore(session, directory.encode()), VX_OK)
                self.assertEqual(self.lib.vx_session_step(session, sampler, ctypes.byref(out)), VX_OK)
                self.assertEqual(out.value, 22)  # restored at step index 1
            finally:
                self.lib.vx_session_close(session)
                self.lib.vx_engine_close(engine)

    def test_backend_over_real_dylib(self):
        # End-to-end: Python backend -> real ctypes binding -> fake C++ engine.
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        from veystrix.backends.cpp_engine import CppEngineBackend

        class TinyTokenizer:
            added_tokens_decoder = {}

            def encode(self, text, add_special_tokens=False):
                return [3, 4]

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": self.encode(text, add_special_tokens)}

            def decode(self, ids):
                return "".join(f"t{t}" for t in ids)

        with patch.dict(os.environ, {"VEYSTRIX_C_CPP_FAKE": "1",
                                     "VEYSTRIX_CPP_ALLOW_FAKE": "1"}):
            backend = CppEngineBackend(
                model_dir="/tmp/fake-model", profile_json='{"kv":"int8"}',
                lib_path=str(self.dylib), tokenizer=TinyTokenizer(),
                session_dir=str(Path(self.work) / "sessions"),
            )
            try:
                backend.append_text("hello")
                emitted = []
                text, stats = backend.generate(2, out=emitted.append)
                self.assertEqual(stats.tokens, 2)
                self.assertEqual(len(emitted), 2)
                self.assertTrue(backend.check_invariant())
                self.assertEqual(stats.prompt_tokens, 2)
                backend.save_session("s1")
                tape = list(backend.tape)
                backend.reset()
                backend.load_session("s1")
                self.assertEqual(backend.tape, tape)
            finally:
                backend.close()

    def test_backend_stub_is_a_hard_error(self):
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        from veystrix.backends.cpp_engine import CppEngineBackend, CppEngineError

        class TinyTokenizer:
            added_tokens_decoder = {}

        with patch.dict(os.environ):
            os.environ.pop("VEYSTRIX_C_CPP_FAKE", None)
            with self.assertRaises(CppEngineError) as caught:
                CppEngineBackend(model_dir="/tmp/fake-model", lib_path=str(self.dylib),
                                 tokenizer=TinyTokenizer())
        self.assertTrue("not 'live'" in str(caught.exception) or
                        "cannot serve" in str(caught.exception))

    def test_exception_never_crosses_abi(self):
        # The deliberate throw is caught inside vx_selftest_throw: the ctypes
        # caller receives a status code plus a message, never a crash.
        self.assertEqual(self.lib.vx_selftest_throw(), VX_ERR_INTERNAL)
        self.assertIn("deliberate C++ exception", self.error())

    def test_context_window_1m_profile_is_admitted(self):
        with patch.dict(os.environ, {"VEYSTRIX_C_CPP_FAKE": "1"}):
            profile = json.dumps({"context_window": 1048576,
                                  "max_cold_prefill": 65536,
                                  "max_incremental_prefill": 65536})
            engine = self.lib.vx_engine_open(b"/tmp/fake-model", profile.encode())
            self.assertTrue(engine, self.error())
            buf = ctypes.create_string_buffer(2048)
            self.assertEqual(self.lib.vx_engine_stats(engine, buf, len(buf)), 0)
            stats = json.loads(buf.value.decode())
            self.assertEqual(stats["mode"], "fake")
            self.lib.vx_engine_close(engine)

    def test_invalid_arguments_return_codes(self):
        self.assertEqual(self.lib.vx_session_prefill(None, None, 0), VX_ERR_INVALID_ARGUMENT)
        self.assertEqual(self.lib.vx_session_step(None, None, None), VX_ERR_INVALID_ARGUMENT)
        self.assertEqual(self.lib.vx_cancel(None), VX_ERR_INVALID_ARGUMENT)

    # ---- ABI v2 (P1 gate) -------------------------------------------------
    def test_unknown_profile_key_is_refused_by_name(self):
        with patch.dict(os.environ, {"VEYSTRIX_C_CPP_FAKE": "1"}):
            handle = self.lib.vx_engine_open(b"/tmp/fake-model", b'{"bogus_key":1}')
            self.assertFalse(handle)
            self.assertIn("bogus_key", self.error())
            self.assertIn("not supported", self.error())

    def test_dynamic_memory_profile_keys_are_accepted_and_validated(self):
        # The four optional dynamic-memory keys pass the strict allowlist.
        with patch.dict(os.environ, {"VEYSTRIX_C_CPP_FAKE": "1"}):
            engine = self.lib.vx_engine_open(
                b"/tmp/fake-model",
                b'{"expert_pool_bytes":1073741824,"nemotron_slots":2944,'
                b'"resident":false,"max_sessions":2}')
            self.assertTrue(engine, self.error())
            self.lib.vx_engine_close(engine)
            # resident may also be 0/1 and true.
            for body in (b'{"resident":true}', b'{"resident":1}', b'{"resident":0}'):
                handle = self.lib.vx_engine_open(b"/tmp/fake-model", body)
                self.assertTrue(handle, f"{body!r}: {self.error()}")
                self.lib.vx_engine_close(handle)
            # Malformed values fail closed, naming the key.
            for body, key in ((b'{"resident":"maybe"}', "resident"),
                              (b'{"resident":2}', "resident"),
                              (b'{"expert_pool_bytes":-1}', "expert_pool_bytes"),
                              (b'{"nemotron_slots":"many"}', "nemotron_slots"),
                              (b'{"max_sessions":-3}', "max_sessions")):
                handle = self.lib.vx_engine_open(b"/tmp/fake-model", body)
                self.assertFalse(handle, body)
                self.assertIn(key, self.error())

    def test_max_sessions_profile_cap_is_enforced(self):
        with patch.dict(os.environ, {"VEYSTRIX_C_CPP_FAKE": "1"}):
            engine = self.lib.vx_engine_open(b"/tmp/fake-model", b'{"max_sessions":2}')
            self.assertTrue(engine, self.error())
            s1 = self.lib.vx_session_open(engine)
            s2 = self.lib.vx_session_open(engine)
            try:
                self.assertTrue(s1 and s2)
                self.assertFalse(self.lib.vx_session_open(engine))
                self.assertIn("max_sessions=2", self.error())
                self.lib.vx_session_close(s2)  # a freed slot is reusable
                s3 = self.lib.vx_session_open(engine)
                self.assertTrue(s3, self.error())
                self.lib.vx_session_close(s3)
            finally:
                self.lib.vx_session_close(s1)
                self.lib.vx_engine_close(engine)
            # No profile key => no cap (the env value is advisory, Python-enforced).
            with patch.dict(os.environ, {"VEYSTRIX_CPP_MAX_SESSIONS": "1"}):
                engine = self.lib.vx_engine_open(b"/tmp/fake-model", b"{}")
                a = self.lib.vx_session_open(engine)
                b = self.lib.vx_session_open(engine)
                self.assertTrue(a and b)
                self.lib.vx_session_close(a)
                self.lib.vx_session_close(b)
                self.lib.vx_engine_close(engine)

    def test_capabilities_are_json_and_honest(self):
        with patch.dict(os.environ, {"VEYSTRIX_C_CPP_FAKE": "1"}):
            engine = self.lib.vx_engine_open(
                b"/tmp/fake-model", b'{"context_window":32,"digests":true}')
            self.assertTrue(engine)
            try:
                buf = ctypes.create_string_buffer(4096)
                self.assertEqual(self.lib.vx_engine_capabilities(engine, buf, len(buf)), VX_OK)
                caps = json.loads(buf.value.decode())
                self.assertEqual(caps["abi"], 2)
                self.assertEqual(caps["mode"], "fake")
                self.assertEqual(caps["context_window"], 32)
                self.assertTrue(caps["logit_digest"])
                self.assertTrue(caps["prefill_append"])
                self.assertEqual(caps["sampling"], ["greedy", "non_greedy"])
                self.assertNotIn("non_greedy", caps["refused"])  # now wired
            finally:
                self.lib.vx_engine_close(engine)

    def test_context_window_and_max_prefill_refuse(self):
        with patch.dict(os.environ, {"VEYSTRIX_C_CPP_FAKE": "1"}):
            engine = self.lib.vx_engine_open(
                b"/tmp/fake-model",
                b'{"context_window":4,"max_cold_prefill":3,"max_incremental_prefill":2}')
            session = self.lib.vx_session_open(engine)
            try:
                ids5 = (ctypes.c_uint32 * 5)(1, 2, 3, 4, 5)
                self.assertEqual(self.lib.vx_session_prefill(session, ids5, 5),
                                 VX_ERR_CONTEXT_LIMIT)
                self.assertIn("context window exceeded", self.error())
                ids4 = (ctypes.c_uint32 * 4)(1, 2, 3, 4)
                self.assertEqual(self.lib.vx_session_prefill(session, ids4, 4),
                                 VX_ERR_CONTEXT_LIMIT)
                self.assertIn("max cold prefill exceeded", self.error())
                ids2 = (ctypes.c_uint32 * 2)(1, 2)
                self.assertEqual(self.lib.vx_session_prefill(session, ids2, 2), VX_OK)
                self.assertEqual(self.lib.vx_session_prefill_ex(
                    session, ids5, 3, 1), VX_ERR_CONTEXT_LIMIT)
                self.assertIn("max incremental prefill exceeded", self.error())
            finally:
                self.lib.vx_session_close(session)
                self.lib.vx_engine_close(engine)

    def test_prefill_ex_append_and_token_count(self):
        with patch.dict(os.environ, {"VEYSTRIX_C_CPP_FAKE": "1"}):
            engine = self.lib.vx_engine_open(b"/tmp/fake-model", b"{}")
            session = self.lib.vx_session_open(engine)
            try:
                a = (ctypes.c_uint32 * 2)(7, 8)
                b = (ctypes.c_uint32 * 3)(9, 10, 11)
                self.assertEqual(self.lib.vx_session_prefill_ex(session, a, 2, 0), VX_OK)
                self.assertEqual(self.lib.vx_session_tokens(session), 2)
                self.assertEqual(self.lib.vx_session_prefill_ex(session, b, 3, 1), VX_OK)
                self.assertEqual(self.lib.vx_session_tokens(session), 5)
                self.assertEqual(self.lib.vx_session_prefill_ex(session, b, 3, 99),
                                 VX_ERR_INVALID_ARGUMENT)
                self.assertIn("unknown flags", self.error())
            finally:
                self.lib.vx_session_close(session)
                self.lib.vx_engine_close(engine)

    def test_logit_digest_is_real_sha256(self):
        import hashlib
        import struct
        with patch.dict(os.environ, {"VEYSTRIX_C_CPP_FAKE": "1"}):
            engine = self.lib.vx_engine_open(b"/tmp/fake-model", b'{"digests":true}')
            session = self.lib.vx_session_open(engine)
            try:
                ids = (ctypes.c_uint32 * 1)(3)
                self.lib.vx_session_prefill(session, ids, 1)
                out = ctypes.c_uint32()
                self.lib.vx_session_step(session, b'{"fake_tokens":[42]}', ctypes.byref(out))
                buf = ctypes.create_string_buffer(128)
                self.assertEqual(self.lib.vx_session_last_digest(session, buf, len(buf)), VX_OK)
                expected = hashlib.sha256(struct.pack("<I", 42)).hexdigest()
                self.assertEqual(buf.value.decode(), expected)
            finally:
                self.lib.vx_session_close(session)
                self.lib.vx_engine_close(engine)
            # digests=false must fail closed
            engine = self.lib.vx_engine_open(b"/tmp/fake-model", b"{}")
            session = self.lib.vx_session_open(engine)
            try:
                buf = ctypes.create_string_buffer(128)
                self.assertEqual(self.lib.vx_session_last_digest(session, buf, len(buf)),
                                 VX_ERR_NOT_WIRED)
            finally:
                self.lib.vx_session_close(session)
                self.lib.vx_engine_close(engine)

    def test_seed_and_cow_fork_fake_mode(self):
        with patch.dict(os.environ, {"VEYSTRIX_C_CPP_FAKE": "1"}):
            engine = self.lib.vx_engine_open(b"/tmp/fake-model", b"{}")
            src = self.lib.vx_session_open(engine)
            dst = self.lib.vx_session_open(engine)
            try:
                ids = (ctypes.c_uint32 * 3)(1, 2, 3)
                self.lib.vx_session_prefill(src, ids, 3)
                self.assertEqual(self.lib.vx_session_seed(src, 20261119), VX_OK)
                self.assertEqual(self.lib.vx_session_fork(src, dst), VX_OK)
                self.assertEqual(self.lib.vx_session_tokens(dst), 3)
                out = ctypes.c_uint32()
                sampler = b'{"fake_tokens":[5,6]}'
                self.lib.vx_session_step(src, sampler, ctypes.byref(out))
                self.assertEqual(out.value, 5)
                self.lib.vx_session_step(dst, sampler, ctypes.byref(out))
                self.assertEqual(out.value, 5)  # fork is independent of src's later steps
                self.assertEqual(self.lib.vx_session_tokens(dst), 4)
            finally:
                self.lib.vx_session_close(src)
                self.lib.vx_session_close(dst)
                self.lib.vx_engine_close(engine)


if __name__ == "__main__":
    unittest.main(verbosity=2)
