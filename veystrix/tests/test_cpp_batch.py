"""Offline tests for the opt-in batched C ABI path (VEYSTRIX_CPP_BATCH=1).

No real dylib, checkpoint or network: the low-level binding is either a
deterministic stub or a fake ctypes CDLL. The default (flag off) path is
asserted byte-for-byte unchanged: no shared engine, fork returns None, and the
decode hooks fail closed.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from veystrix.backends import cpp_engine as cpp
from veystrix.backends.cpp_engine import (
    BATCH_ENV,
    CppEngineBackend,
    CppEngineError,
    VX_ERR_INTERNAL,
    VX_ERR_INVALID_ARGUMENT,
)
from veystrix.sampling import Sampling
from veystrix.tool.supervisor import engine_env_for


class FakeTokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [7 + len(text)]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text, add_special_tokens)}

    def decode(self, ids):
        return "".join(f"<{int(t)}>" for t in ids)

    def convert_tokens_to_ids(self, token):
        return None


class BatchStubBinding:
    """Deterministic stub with per-session state and the optional step_batch ABI.

    This is the existing StubBinding pattern extended with ``step_batch`` and
    per-session tracking so several backends can share one engine/session set.
    """

    def __init__(self, tokens=(201, 202, 203, 204), *, has_step_batch=True):
        self.tokens = list(tokens)
        self.has_step_batch = has_step_batch
        self._session_state = {}  # session -> {"engine", "prompt", "cursor"}
        self.opened_engines = []
        self.closed_engines = []
        self.step_calls = []
        self.batch_calls = []
        self.step_error = None

    def abi_version(self):
        return 1

    def mode(self):
        return "fake"

    def open_engine(self, model_dir, profile_json):
        engine = (str(model_dir), str(profile_json))
        self.opened_engines.append(engine)
        return engine

    def close_engine(self, engine):
        self.closed_engines.append(engine)

    def open_session(self, engine):
        session = object()
        self._session_state[session] = {"engine": engine, "prompt": [], "cursor": 0}
        return session

    def close_session(self, session):
        self._session_state.pop(session, None)

    def prefill(self, session, tokens):
        state = self._session_state[session]
        state["prompt"] = [int(t) for t in tokens]
        state["cursor"] = 0

    def step(self, session, sampler_json):
        if self.step_error is not None:
            raise self.step_error
        self.step_calls.append((session, str(sampler_json)))
        state = self._session_state[session]
        token = self.tokens[state["cursor"] % len(self.tokens)]
        state["cursor"] += 1
        return token

    def step_batch(self, sessions, sampler_jsons):
        self.batch_calls.append((list(sessions), list(sampler_jsons)))
        if self.step_error is not None:
            raise self.step_error
        return [self.step(session, sampler) for session, sampler in zip(sessions, sampler_jsons)]

    def save(self, session, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        state = self._session_state[session]
        (Path(directory) / "stub.json").write_text(
            json.dumps({"prompt": state["prompt"], "cursor": state["cursor"]}), encoding="utf-8")

    def restore(self, session, directory):
        state = json.loads((Path(directory) / "stub.json").read_text(encoding="utf-8"))
        self._session_state[session].update(state)

    def cancel(self, session):
        pass

    def stats(self, engine):
        return {"mode": "fake", "session_tokens": 0}


class _FakeFunc:
    """Callable stand-in for a ctypes CDLL function symbol."""

    def __init__(self, result=0):
        self.argtypes = None
        self.restype = None
        self._result = result

    def __call__(self, *args):
        return self._result(*args) if callable(self._result) else self._result


class _FakeCDLL:
    """Minimal CDLL surface; a missing symbol raises AttributeError (dlsym miss)."""

    def __init__(self, names, *, mode=b"live", error=b""):
        self._funcs = {name: _FakeFunc() for name in names}
        self._funcs["vx_mode"] = _FakeFunc(mode)
        self._funcs["vx_last_error"] = _FakeFunc(error)

    def __getattr__(self, name):
        try:
            return self._funcs[name]
        except KeyError:
            raise AttributeError(name) from None


_ABI_NAMES = [
    "vx_abi_version", "vx_mode", "vx_last_error", "vx_engine_open", "vx_engine_close",
    "vx_session_open", "vx_session_close", "vx_session_prefill", "vx_session_step",
    "vx_snapshot_save", "vx_snapshot_restore", "vx_cancel", "vx_engine_stats",
    "vx_selftest_throw",
]

# Exact prototype from cpp/engine/capi/veystrix_c.h on branch cpp-multi-abi.
# The C++ sources are private and are intentionally absent from cli-terminal-tool,
# so the authoritative declaration is pinned verbatim here (single source of
# truth for the parity check below):
#
#   VX_API int vx_session_step_batch(vx_session** sessions, int n, const char** sampler_jsons, uint32_t* out_tokens);
_VX_STEP_BATCH_PROTOTYPE = (
    "VX_API int vx_session_step_batch(vx_session** sessions, int n, "
    "const char** sampler_jsons, uint32_t* out_tokens);"
)

# C type token -> ctypes type. vx_session is an opaque handle, so a vx_session*
# maps to c_void_p and a vx_session** maps to POINTER(c_void_p).
_C_TYPE_TO_CTYPES = {
    "vx_session**": ctypes.POINTER(ctypes.c_void_p),
    "int": ctypes.c_int,
    "const char**": ctypes.POINTER(ctypes.c_char_p),
    "uint32_t*": ctypes.POINTER(ctypes.c_uint32),
}


def _parse_step_batch_prototype(prototype):
    """Return ``(restype, argtypes)`` parsed from the pinned C header line."""
    match = re.fullmatch(r"VX_API\s+(\S+)\s+\w+\((.*)\);", prototype.strip())
    assert match, f"unexpected vx_session_step_batch prototype: {prototype!r}"
    restype = _C_TYPE_TO_CTYPES[match.group(1)]
    argtypes = []
    for param in match.group(2).split(","):
        tokens = param.split()  # e.g. ["const", "char**", "sampler_jsons"]
        argtypes.append(_C_TYPE_TO_CTYPES[" ".join(tokens[:-1])])
    return restype, argtypes


class CtypesStepBatchTest(unittest.TestCase):
    """Low-level binding: argtypes, feature detection and buffer handling."""

    def _binding(self, names, *, error=b""):
        fake = _FakeCDLL(names, error=error)
        with patch.object(cpp, "resolve_lib_path", return_value="/fake/libveystrix_c.dylib"), \
             patch.object(cpp.ctypes, "CDLL", return_value=fake):
            binding = cpp.CtypesVxBinding()
        return binding, fake

    def test_argtypes_present_and_feature_detected(self):
        names = _ABI_NAMES + ["vx_session_step_batch"]
        binding, fake = self._binding(names)
        self.assertTrue(binding.has_step_batch)
        self.assertEqual(
            fake.vx_session_step_batch.argtypes,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
             ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_uint32)],
        )
        self.assertIs(fake.vx_session_step_batch.restype, ctypes.c_int)

        old, _ = self._binding(list(_ABI_NAMES))
        self.assertFalse(old.has_step_batch)
        with self.assertRaises(CppEngineError):
            old.step_batch([1], ["{}"])

    def test_step_batch_argtypes_match_header(self):
        """The ctypes signature pinned in CtypesVxBinding must match the C header.

        The header itself lives in the private cpp-multi-abi source tree, so the
        exact declaration is pinned in _VX_STEP_BATCH_PROTOTYPE above and parsed
        into the expected ctypes signature here. The binding is constructed
        against a fake CDLL that exports vx_session_step_batch, so this runs
        without a real dylib; __init__ sets the symbol's argtypes/restype when it
        feature-detects the symbol via hasattr.
        """
        expected_restype, expected_argtypes = _parse_step_batch_prototype(
            _VX_STEP_BATCH_PROTOTYPE)

        # Parity with the pinned header declaration (4 params, int return).
        self.assertEqual(len(expected_argtypes), 4)
        self.assertIs(expected_argtypes[0], ctypes.POINTER(ctypes.c_void_p))
        self.assertIs(expected_argtypes[1], ctypes.c_int)
        self.assertIs(expected_argtypes[2], ctypes.POINTER(ctypes.c_char_p))
        self.assertIs(expected_argtypes[3], ctypes.POINTER(ctypes.c_uint32))
        self.assertIs(expected_restype, ctypes.c_int)

        # The symbol exists on the (fake) lib and the binding wires it up to the
        # exact signature parsed from the header, and exposes a step_batch method.
        binding, fake = self._binding(_ABI_NAMES + ["vx_session_step_batch"])
        func = fake.vx_session_step_batch
        self.assertEqual(func.argtypes, expected_argtypes)
        self.assertIs(func.restype, expected_restype)
        self.assertTrue(binding.has_step_batch)
        self.assertTrue(callable(getattr(cpp.CtypesVxBinding, "step_batch", None)))

    def test_step_batch_buffers_and_reads_out_only_on_ok(self):
        captured = {}

        def step_impl(sessions, n, samplers, out):
            captured["n"] = n
            captured["sessions"] = [sessions[i] for i in range(n)]
            captured["samplers"] = [bytes(samplers[i]).decode("utf-8") for i in range(n)]
            for i in range(n):
                out[i] = 300 + i
            return captured.get("code", 0)

        fake = _FakeCDLL(_ABI_NAMES + ["vx_session_step_batch"], error=b"boom")
        fake._funcs["vx_session_step_batch"] = _FakeFunc(step_impl)
        binding = cpp.CtypesVxBinding.__new__(cpp.CtypesVxBinding)
        binding._lib = fake
        binding.has_step_batch = True

        self.assertEqual(binding.step_batch([11, 22], ['{"a": 1}', '{"b": 2}']), [300, 301])
        self.assertEqual(captured["n"], 2)
        self.assertEqual(captured["sessions"], [11, 22])
        self.assertEqual(captured["samplers"], ['{"a": 1}', '{"b": 2}'])

        captured["code"] = VX_ERR_INTERNAL
        with self.assertRaises(CppEngineError) as caught:
            binding.step_batch([11], ["{}"])
        self.assertEqual(caught.exception.code, VX_ERR_INTERNAL)
        self.assertIn("boom", str(caught.exception))

        captured["code"] = 0
        with self.assertRaises(CppEngineError) as mismatch:
            binding.step_batch([11, 22], ["{}"])
        self.assertEqual(mismatch.exception.code, VX_ERR_INVALID_ARGUMENT)

        self.assertEqual(binding.step_batch([], []), [])


class _BackendTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env = patch.dict(os.environ, {BATCH_ENV: "1"}, clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)
        cpp._reset_shared_engines()
        self.addCleanup(cpp._reset_shared_engines)

    def backend(self, binding=None, profile='{"kv":"int8"}', **kwargs):
        binding = binding if binding is not None else BatchStubBinding()
        backend = CppEngineBackend(
            model_dir="/fake/model", profile_json=profile, binding=binding,
            tokenizer=FakeTokenizer(),
            session_dir=str(Path(self._tmp.name) / "sessions"), **kwargs)
        self.addCleanup(backend.close)
        return backend


class DefaultPathUnchangedTest(_BackendTestBase):
    """Flag off: every backend owns its engine; no batching is advertised."""

    def setUp(self):
        super().setUp()
        os.environ.pop(BATCH_ENV, None)

    def test_default_fork_none_own_engines_and_fail_closed_hooks(self):
        first_binding = BatchStubBinding()
        second_binding = BatchStubBinding()
        first = self.backend(binding=first_binding)
        second = self.backend(binding=second_binding)

        self.assertIsNone(first.fork_for_batch())
        self.assertIsNot(first._engine, second._engine)
        self.assertEqual(first_binding.opened_engines, [("/fake/model", '{"kv":"int8"}')])
        self.assertEqual(second_binding.opened_engines, [("/fake/model", '{"kv":"int8"}')])

        for call in (
            lambda: first.batch_decode_group([(first, 1, None, True)]),
            lambda: first.batch_decode(1, None, False),
            lambda: first.batch_prefill([1], 0, True, None),
        ):
            with self.assertRaises(CppEngineError):
                call()


class SharedEngineTest(_BackendTestBase):
    def test_refcount_and_distinct_sessions(self):
        first_binding = BatchStubBinding()
        second_binding = BatchStubBinding()
        first = self.backend(binding=first_binding)
        second = self.backend(binding=second_binding)

        self.assertIs(first._engine, second._engine)
        self.assertEqual(first_binding.opened_engines, [("/fake/model", '{"kv":"int8"}')])
        self.assertEqual(second_binding.opened_engines, [])
        self.assertIsNot(first._session, second._session)
        self.assertIs(first.batch_group_key(), second.batch_group_key())
        self.assertEqual(first.batch_group_key(), ("cpp", id(first._engine)))

        other_binding = BatchStubBinding()
        other_profile = self.backend(binding=other_binding, profile='{"kv":"fp16"}')
        self.assertIsNot(other_profile._engine, first._engine)

        first.close()
        self.assertEqual(first_binding.closed_engines, [])
        second.close()
        self.assertEqual(first_binding.closed_engines, [first._engine])
        other_profile.close()
        self.assertEqual(other_binding.closed_engines, [other_profile._engine])

    def test_fork_holds_its_own_engine_reference(self):
        binding = BatchStubBinding()
        parent = self.backend(binding=binding)
        child = parent.fork_for_batch()
        self.assertIsNotNone(child)
        self.addCleanup(child.close)

        parent.close()
        self.assertEqual(binding.closed_engines, [])
        child.close()
        self.assertEqual(binding.closed_engines, [parent._engine])

    def test_configure_profile_does_not_close_engine_still_in_use(self):
        binding = BatchStubBinding()
        first = self.backend(binding=binding)
        other = self.backend(binding=BatchStubBinding())
        old_engine = first._engine

        self.assertIn("profile updated", first.configure('profile={"kv":"fp16"}'))
        # The old shared engine is still alive for `other`.
        self.assertEqual(binding.closed_engines, [])
        self.assertIsNot(first._engine, other._engine)
        first.close()  # drops the new (fp16) engine, opened by this binding
        self.assertEqual(binding.closed_engines, [first._engine])
        other.close()  # drops the last ref to the old (int8) engine
        self.assertEqual(binding.closed_engines, [first._engine, old_engine])


class BatchHooksTest(_BackendTestBase):
    def test_flag_on_fork_and_group_hooks(self):
        binding = BatchStubBinding()
        parent = self.backend(binding=binding)
        parent.append_text("hello")

        child = parent.fork_for_batch()
        self.assertIsNotNone(child)
        self.addCleanup(child.close)
        self.assertIs(child._engine, parent._engine)
        self.assertIsNot(child._session, parent._session)
        self.assertEqual(child.tape, [])
        self.assertEqual(child.pending, [])
        self.assertIs(child.batch_group_key(), parent.batch_group_key())

        # One binding instance mirrors production: separate CtypesVxBinding
        # objects still resolve to the same loaded dylib, so session handles
        # are valid across all backends sharing the engine.
        other = self.backend(binding=binding)
        other_child = other.fork_for_batch()
        self.assertIsNotNone(other_child)
        self.addCleanup(other_child.close)

        steps = [(child, 201, None, True), (other_child, 202, None, True)]
        # Each fresh session starts at its own cursor 0.
        self.assertEqual(child.batch_decode_group(steps), [201, 201])
        self.assertEqual(len(binding.batch_calls), 1)
        sessions, samplers = binding.batch_calls[0]
        self.assertEqual(sessions, [child._session, other_child._session])
        import json as _json
        self.assertEqual([_json.loads(x)["greedy"] for x in samplers], [True, True])  # default sampler is greedy

        self.assertEqual(child.batch_decode(201, None, False), 202)
        self.assertEqual(len(binding.step_calls), 3)  # 2 batched + 1 single

    def test_prefill_appends_quantums_immediately(self):
        binding = BatchStubBinding()
        backend = self.backend(binding=binding)
        child = backend.fork_for_batch()
        self.addCleanup(child.close)

        self.assertIsNone(child.batch_prefill([1, 2], 0, False, None))
        self.assertEqual(binding._session_state[child._session]["prompt"], [1, 2])
        first = child.batch_prefill([3, 4], 2, True, None)
        self.assertEqual(first, 201)
        self.assertEqual(binding._session_state[child._session]["prompt"], [3, 4])  # stub has no prefill_ex; fallback buffers then resets
        self.assertEqual(len(binding.step_calls), 1)

    def test_non_greedy_batch_allowed(self):
        # ABI v2: each session has its own PRNG key sequence, so non-greedy
        # batching is safe (Sonnet P0 parity finding).
        binding = BatchStubBinding()
        backend = self.backend(binding=binding)
        backend.sampling = Sampling(temperature=0.8)

        child = backend.fork_for_batch()
        self.assertIsNotNone(child)
        # Non-greedy fork is allowed; group decode uses step_batch when enabled.
        out = backend.batch_decode_group([(backend, 1, None, True)])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)

    def test_missing_step_batch_abi_refuses_batch_fork(self):
        binding = BatchStubBinding(has_step_batch=False)
        backend = self.backend(binding=binding)
        self.assertIsNone(backend.fork_for_batch())
        self.assertEqual(binding.opened_engines, [("/fake/model", '{"kv":"int8"}')])


class SupervisorEnvTest(unittest.TestCase):
    def test_flag_only_mirrored_for_cpp(self):
        with patch.dict(os.environ):
            os.environ.pop(BATCH_ENV, None)
            self.assertEqual(engine_env_for("python", {BATCH_ENV: "1"}), {})
            self.assertEqual(engine_env_for("cpp", {}), {"VEYSTRIX_ENGINE": "cpp"})
            self.assertEqual(
                engine_env_for("cpp", {BATCH_ENV: "1"}),
                {"VEYSTRIX_ENGINE": "cpp", BATCH_ENV: "1"},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
