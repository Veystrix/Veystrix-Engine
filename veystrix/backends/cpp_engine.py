"""Python Backend implementation over Veystrix's narrow C ABI (Track B M6).

Stdlib-only (ctypes) and importable without MLX; it is never imported on the
default ``VEYSTRIX_ENGINE=python`` path. The C ABI is proven live for both Qwen
and Nemotron on one driver thread in Track A (cpp-p1x-integrate). The shipped
dylib is binary-only and lives in the engine bundle, not this source tree.

The engine drives one session per backend (N sessions share an engine only on
the opt-in batched path). Greedy and Python's five-field non-greedy sampling
are both supported (plan P2); unknown sampler keys are refused by name. There
is no silent fallback:
a missing dylib, unsupported ABI, non-live mode or a "not wired" error always
raises ``CppEngineError``. Per-model+profile parity is gated by the bundle
manifest (see docs/engineering/m6_cpp_toggle.md).
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import concurrent.futures
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import threading
import time

from veystrix.backends.base import DecodeTimer, GenerationCancelled
from veystrix.text import response_format_for, stream_decode
from veystrix.conversation import Conversation

# Must match vx_status in cpp/engine/capi/veystrix_c.h.
VX_OK = 0
VX_ERR_INVALID_ARGUMENT = -1
VX_ERR_NOT_WIRED = -2
VX_ERR_CANCELLED = -3
VX_ERR_INTERNAL = -4
VX_ERR_CONTEXT_LIMIT = -5

SHADOW_ENV = "VEYSTRIX_CPP_SHADOW_LOG"
LIB_ENV = "VEYSTRIX_CPP_LIB"
ENGINE_ENV = "VEYSTRIX_ENGINE"
FAKE_ENV = "VEYSTRIX_C_CPP_FAKE"
# Opt-in, default OFF. Every measured batched variant was SLOWER than the
# page-cache-shared baseline, so this path is never taken unless the operator
# explicitly asks for it. Old dylibs lack vx_session_step_batch and must fail
# closed rather than silently serving a different (slower/wrong) path.
BATCH_ENV = "VEYSTRIX_CPP_BATCH"
# D3: opt-in automatic prefix reuse. Off by default (same as Python's
# fragment cache) so a request never silently changes its prefill shape.
_STEP_TIMING = os.environ.get("VEYSTRIX_CPP_STEP_TIMING") == "1"   # diagnostic: split decode time into in-step vs callbacks
# Parked sessions: a client that alternates two conversations (qwen-code's per-turn side request with its own
# system prompt, Claude Code's small-model calls) used to throw away the live C session each time and re-prefill
# the whole main conversation on the next turn. Up to PARKED_ENV sessions (default 1) holding at most
# PARKED_TOKENS_ENV tokens in all (default 16,384) stay open; a request whose tokens extend a parked session's
# history resumes it and prefills only the new tokens, exactly as live continuation does. 0 turns it off.
# A parked session keeps its KV and recurrent state (~20-24 KB per token on Ornith / Flash-Next), so nothing is
# parked, and what is parked is released, while available memory is under PARKED_MIN_AVAILABLE_ENV GiB (default 4).
PARKED_ENV = "VEYSTRIX_CPP_PARKED_SESSIONS"
PARKED_TOKENS_ENV = "VEYSTRIX_CPP_PARKED_TOKENS"
PARKED_MIN_AVAILABLE_ENV = "VEYSTRIX_CPP_PARKED_MIN_AVAILABLE_GIB"
PARKED_MIN_TOKENS = 256
PREFIX_CACHE_ENV = "VEYSTRIX_CPP_PREFIX_CACHE"
PREFIX_CACHE_QUOTA_ENV = "VEYSTRIX_CPP_PREFIX_QUOTA_GB"
_DEFAULT_PREFIX_QUOTA_GB = 4.0
# (namespace, token tuple) -> snapshot dir. Namespace includes engine, ABI,
# model dir and profile so C++ and Python snapshots are never shared.
_CPP_PREFIX_CACHE: "OrderedDict[tuple, str]" = OrderedDict()
_CPP_PREFIX_SYSTEM_ANCHORS: set[tuple] = set()
SHADOW_SCHEMA = 2

UNLIMITED_GENERATION_HORIZON = (1 << 31) - 1
_DYNAMIC_MEMORY_ENV = "VEYSTRIX_DYNAMIC_MEMORY"
_runtime_budget = None


# Python flags with a direct C++ equivalent are mapped (never silently dropped).
_MAPPED_PY_FLAGS = {
    "FLASHNEXT_KV_CACHE_DTYPE": "VEYSTRIX_CPP_KV_TIER",
    "FLASHNEXT_KV_GROUP_SIZE": "VEYSTRIX_CPP_KV_GROUP_SIZE",
    "FLASHNEXT_KV_RECENT_TOKENS": "VEYSTRIX_CPP_KV_RECENT_TOKENS",
}
# Python-only flags with NO C++ equivalent. Refuse only when actually ENABLED:
# the standard launcher sets e.g. FLASHNEXT_NATIVE_MTP=off and
# FLASHNEXT_COMPILE_GDN=1 (numerics-neutral), so presence alone must not block
# the cpp engine (Sonnet finding 7). FLASHNEXT_COMPILE_GDN is absent by design:
# it is a numerics-neutral host-glue flag with no C++ meaning.
_UNMAPPED_PY_FLAGS = (
    "FLASHNEXT_NATIVE_MTP", "VEYSTRIX_LOOKUP_SAMPLING",
    "FLASHNEXT_EXPERT_PREFETCH", "VEYSTRIX_NEMOTRON_PARALLEL_MAMBA",
)
_DISABLED_VALUES = {"", "0", "off", "false", "no", "none"}


def translate_env() -> None:
    """Map known Python flags to C++ equivalents; refuse enabled unmapped ones."""
    for py_flag, cpp_flag in _MAPPED_PY_FLAGS.items():
        if py_flag in os.environ:
            os.environ.setdefault(cpp_flag, os.environ[py_flag])
    for py_flag in _UNMAPPED_PY_FLAGS:
        value = os.environ.get(py_flag)
        if value is None or value.strip().lower() in _DISABLED_VALUES:
            continue
        raise SystemExit(
            f"veystrix: {py_flag}={value} is a Python-engine-only flag with no C++ "
            f"equivalent (measured dead end or unported); unset it or set it to "
            f"off, or use --engine python."
        )


def _is_nemotron(model_dir) -> bool:
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text())
    except (OSError, ValueError, TypeError):
        return False
    return str(config.get("model_type", "")).lower() == "nemotron_h"


def apply_runtime_memory_budget(model_dir=None):
    """Size the C++ expert pool / slots from live RAM and the 5 GiB swap ceiling.

    Runs once per process, before any engine opens. Values are exported as the
    env vars the C++ pools already read, so the profile_json passed to the ABI
    stays byte-identical to what the caller supplied (important for cached
    engine keys and shadow records). Fail-closed: on any error we leave the
    C++ defaults in place (which are conservative).
    """
    global _runtime_budget
    if _runtime_budget is not None:
        return _runtime_budget
    if os.environ.get(_DYNAMIC_MEMORY_ENV) == "0":
        _runtime_budget = None
        return None
    try:
        from veystrix.memory_budget_runtime import plan
        from veystrix.model_memory import probe_checkpoint
        probe = probe_checkpoint(model_dir) if model_dir else None
        extra = ({"dense_bytes": probe.resident_dense_bytes,
                  "routed_expert_bytes": probe.routed_expert_bytes,
                  "record_bytes": probe.record_bytes} if probe else {})
        budget = plan(**extra)
        if budget.pool_bytes:
            os.environ.setdefault("VEYSTRIX_CPP_POOL_BYTES", str(budget.pool_bytes))
        if budget.nemotron_slots:
            os.environ.setdefault("VEYSTRIX_CPP_NEMO_POOL_SLOTS", str(budget.nemotron_slots))
        os.environ.setdefault("VEYSTRIX_CPP_RESIDENT", "1" if budget.resident else "0")
        if not budget.nemotron_slots and _is_nemotron(model_dir):
            # Streaming Nemotron: the slot pool has a ~0% hit rate here (a token needs ~138 different
            # experts out of 2,944 and the pool holds 96), so it only adds a copy. Since expert rows are
            # read with the batched 8-lane reader straight into the final buffer, no pool is faster
            # (25.2 s vs 26.6 s per 121 tokens through the server, byte-identical text).
            # VEYSTRIX_CPP_NEMO_POOL=1 (+ _SLOTS) still selects it.
            os.environ.setdefault("VEYSTRIX_CPP_NEMO_POOL", "0")
        os.environ.setdefault("VEYSTRIX_CPP_MAX_SESSIONS", str(budget.max_sessions))
        _runtime_budget = budget
    except Exception:
        _runtime_budget = None
    return _runtime_budget


def batch_enabled() -> bool:
    """True only when the operator explicitly opts into the batched C ABI."""
    return os.environ.get(BATCH_ENV) == "1"


@dataclass
class _SharedEngine:
    """One vx_engine shared by all cpp backends for a (model_dir, profile).

    ``group_key`` is a single tuple *object* so the batch scheduler's identity
    comparison (``a.batch_group_key() is b.batch_group_key()``) groups every
    backend on this engine. ``binding`` is the opener, used to close the engine
    once the last reference drops.
    """

    engine: object
    binding: object
    refs: int = 0
    group_key: tuple = ()


# Module-level refcounted registry. Only consulted when BATCH_ENV=1; with the
# flag off every backend still opens and owns its own engine (byte-for-byte the
# original behavior).
_shared_engines: dict[tuple[str, str], _SharedEngine] = {}
_shared_lock = threading.RLock()


def _acquire_shared_engine(binding, model_dir, profile_json) -> tuple[tuple, _SharedEngine]:
    key = (str(model_dir), str(profile_json))
    with _shared_lock:
        entry = _shared_engines.get(key)
        if entry is None:
            engine = binding.open_engine(model_dir, profile_json)
            entry = _SharedEngine(engine=engine, binding=binding,
                                  group_key=("cpp", id(engine)))
            _shared_engines[key] = entry
        entry.refs += 1
        return key, entry


def _retain_shared_engine(key: tuple) -> None:
    """A fork keeps the engine alive independently of its parent."""
    with _shared_lock:
        entry = _shared_engines.get(key)
        if entry is None:
            raise CppEngineError("the shared C++ engine registry entry is gone")
        entry.refs += 1


def _release_shared_engine(key: tuple) -> None:
    """Drop one reference; close the engine only when the last ref goes away."""
    with _shared_lock:
        entry = _shared_engines.get(key)
        if entry is None:
            return
        entry.refs -= 1
        if entry.refs > 0:
            return
        _shared_engines.pop(key, None)
        engine, closer = entry.engine, entry.binding
    # Close outside the lock: another thread may acquire a different engine.
    closer.close_engine(engine)


def _reset_shared_engines() -> None:
    """Test-only helper: close and drop every shared engine in the registry."""
    with _shared_lock:
        entries = list(_shared_engines.values())
        _shared_engines.clear()
    for entry in entries:
        try:
            entry.binding.close_engine(entry.engine)
        except Exception:
            pass


def engine_name() -> str:
    """Selected engine. Default stays python until the drop gate passes."""
    return os.environ.get(ENGINE_ENV, "python").strip().lower()


def default_lib_path() -> Path:
    """Default dylib location (override with VEYSTRIX_CPP_LIB).

    A binary-only bundle ships the library under the tool home
    (``~/.veystrix/engine`` or ``VEYSTRIX_HOME/.veystrix/engine``); a source
    checkout builds it under ``cpp/engine/build/capi``. The bundle candidates
    win only when the file actually exists, so the source path stays the
    documented default until a C++ bundle is installed.
    """
    suffix = ".dylib" if os.name == "posix" and os.uname().sysname == "Darwin" else ".so"
    name = f"libveystrix_c{suffix}"
    roots = []
    tool_home = os.environ.get("VEYSTRIX_HOME")
    if tool_home:
        roots.append(Path(tool_home).expanduser() / ".veystrix" / "engine")
    roots.append(Path.home() / ".veystrix" / "engine")
    here = Path(__file__).resolve()
    if len(here.parents) >= 4 and here.parents[2].name == "lib":
        roots.append(here.parents[3])
    for root in roots:
        # ``root`` is the installed bundle dir (<home>/.veystrix/engine). The
        # canonical install is root/<name>; ``bin/`` and the legacy nested
        # ``engine/`` layout are also probed so an older/reshaped bundle is
        # still found without a manual VEYSTRIX_CPP_LIB override.
        for candidate in (root / name, root / "bin" / name, root / "engine" / name):
            if candidate.is_file():
                return candidate
    repo = Path(__file__).resolve().parents[2]
    # Live (MLX-linked) builds first; build/capi is the CPU-only stub and stays last.
    for directory in ("build", "build-live", "build/capi"):
        candidate = repo / "cpp" / "engine" / directory / name
        if candidate.is_file():
            return candidate
    return repo / "cpp" / "engine" / "build" / "capi" / name


def resolve_lib_path(path: str | os.PathLike[str] | None = None) -> Path:
    if path is None:
        path = os.environ.get(LIB_ENV) or str(default_lib_path())
    result = Path(path).expanduser()
    if not result.is_file():
        raise CppEngineError(
            f"C++ engine dylib not found at {result}. Reinstall the engine "
            f"bundle (veystrix install --apply) or set {LIB_ENV} to a live "
            f"libveystrix_c dylib."
        )
    return result


def _manifest_for_lib(lib_path: str | os.PathLike[str]) -> Path | None:
    """Return the BUNDLE-MANIFEST.json for the bundle root a lib was found under.

    ``default_lib_path()`` probes two roots: ``$VEYSTRIX_HOME/.veystrix/engine``
    and ``~/.veystrix/engine``. The manifest MUST be read from the same root the
    lib came from, otherwise a lib found under the home root could be loaded
    unchecked while an alternate root attests a different hash.
    """
    resolved = Path(lib_path).resolve()
    for parent in resolved.parents:
        if parent.name == "engine":
            candidate = parent / "BUNDLE-MANIFEST.json"
            if candidate.is_file():
                return candidate
    return None


def _attested_lib_sha256(lib_path: str | os.PathLike[str]) -> tuple[str | None, Path | None]:
    """Return (recorded sha256, manifest path) from the lib's own bundle root."""
    manifest = _manifest_for_lib(lib_path)
    if manifest is None:
        return None, None
    try:
        data = json.loads(manifest.read_text())
    except (OSError, ValueError, TypeError):
        return None, manifest
    if not isinstance(data, dict):
        return None, manifest
    value = data.get("cpp_lib_sha256")
    return (str(value) if value else None), manifest


def verify_lib_sha256(path: str | os.PathLike[str], expected: str | None,
                      manifest: Path | None = None) -> None:
    """Fail closed if a bundle cannot attest the dylib that is about to load.

    When ``manifest`` is given (i.e. the lib came from an installed bundle), a
    missing or malformed ``cpp_lib_sha256`` is a hard refusal, as is a hash
    mismatch. A source checkout or stub/injected binding (no manifest) is a
    no-op. This closes the "malformed manifest fails open" gap.
    """
    if manifest is None:
        return
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if not expected:
        raise CppEngineError(
            f"engine bundle manifest {manifest} does not record cpp_lib_sha256; "
            f"refusing to load {path!r} as an attested C++ engine. "
            f"Reinstall the bundle or set {LIB_ENV}."
        )
    if digest.lower() != str(expected).strip().lower():
        raise CppEngineError(
            f"C++ engine dylib sha256 mismatch: manifest attests {expected!r} but "
            f"{path!r} hashes to {digest!r}. Refusing to use a binary that was not "
            f"attested for cpp parity; reinstall the engine bundle or set {LIB_ENV}."
        )


class CppEngineError(RuntimeError):
    """A clear failure from the C ABI or its loader. Never falls back."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


@dataclass
class Stats:
    """Agent-loop stats, mirroring the fields of FlashNextBackend.Stats."""

    finish: str = "stop"
    tokens: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    prefill_seconds: float = 0.0
    tail_tokens: int = 0
    tail_seconds: float = 0.0
    ui_seconds: float = 0.0
    pinned_bytes: int = 0
    pinned_signature: str = ""
    host_free_gb: float | None = None
    swap_gb: float | None = None

    @property
    def rate(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0

    @property
    def prompt_rate(self) -> float:
        return self.prompt_tokens / self.prefill_seconds if self.prefill_seconds else 0.0


class CtypesVxBinding:
    """The real ctypes binding. Methods raise CppEngineError on any nonzero code."""

    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = resolve_lib_path(path)
        # Fail closed if an installed bundle attests a specific dylib sha256 and
        # a different binary was swapped in, or if the manifest is malformed.
        # Source checkouts / injected bindings have no manifest -> no-op.
        expected_sha, manifest = _attested_lib_sha256(self.path)
        verify_lib_sha256(self.path, expected_sha, manifest)
        lib = ctypes.CDLL(str(self.path))
        self._lib = lib
        engine_p = ctypes.c_void_p
        session_p = ctypes.c_void_p
        lib.vx_abi_version.argtypes = []
        lib.vx_abi_version.restype = ctypes.c_int
        lib.vx_mode.argtypes = []
        lib.vx_mode.restype = ctypes.c_char_p
        lib.vx_last_error.argtypes = []
        lib.vx_last_error.restype = ctypes.c_char_p
        lib.vx_engine_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.vx_engine_open.restype = engine_p
        lib.vx_engine_close.argtypes = [engine_p]
        lib.vx_engine_close.restype = None
        lib.vx_session_open.argtypes = [engine_p]
        lib.vx_session_open.restype = session_p
        lib.vx_session_close.argtypes = [session_p]
        lib.vx_session_close.restype = None
        lib.vx_session_prefill.argtypes = [session_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int]
        lib.vx_session_prefill.restype = ctypes.c_int
        lib.vx_session_step.argtypes = [session_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32)]
        lib.vx_session_step.restype = ctypes.c_int
        # Optional, additive multi-session ABI. Older dylibs do not export it;
        # feature-detect and fail closed rather than assuming the symbol exists.
        self.has_step_batch = hasattr(lib, "vx_session_step_batch")
        if self.has_step_batch:
            # vx_session** sessions, int n, const char** sampler_jsons, uint32_t* out_tokens
            lib.vx_session_step_batch.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_uint32)]
            lib.vx_session_step_batch.restype = ctypes.c_int
        # ABI v2 (additive). Every new symbol is feature-detected: an old dylib
        # must fail closed with its missing symbol named, never be assumed.
        self.v2_symbols = (
            "vx_engine_capabilities", "vx_session_prefill_ex", "vx_session_seed",
            "vx_session_last_digest", "vx_session_fork", "vx_session_tokens",
        )
        self.has_v2 = all(hasattr(lib, name) for name in self.v2_symbols)
        if self.has_v2:
            lib.vx_engine_capabilities.argtypes = [engine_p, ctypes.c_char_p, ctypes.c_int]
            lib.vx_engine_capabilities.restype = ctypes.c_int
            lib.vx_session_prefill_ex.argtypes = [session_p, ctypes.POINTER(ctypes.c_uint32),
                                                  ctypes.c_int, ctypes.c_uint32]
            lib.vx_session_prefill_ex.restype = ctypes.c_int
            lib.vx_session_seed.argtypes = [session_p, ctypes.c_uint64]
            lib.vx_session_seed.restype = ctypes.c_int
            lib.vx_session_last_digest.argtypes = [session_p, ctypes.c_char_p, ctypes.c_int]
            lib.vx_session_last_digest.restype = ctypes.c_int
            lib.vx_session_fork.argtypes = [session_p, session_p]
            lib.vx_session_fork.restype = ctypes.c_int
            lib.vx_session_tokens.argtypes = [session_p]
            lib.vx_session_tokens.restype = ctypes.c_int
            if hasattr(lib, "vx_session_truncate"):
                lib.vx_session_truncate.argtypes = [session_p, ctypes.c_uint32]
                lib.vx_session_truncate.restype = ctypes.c_int
        lib.vx_snapshot_save.argtypes = [session_p, ctypes.c_char_p]
        lib.vx_snapshot_save.restype = ctypes.c_int
        lib.vx_snapshot_restore.argtypes = [session_p, ctypes.c_char_p]
        lib.vx_snapshot_restore.restype = ctypes.c_int
        lib.vx_cancel.argtypes = [session_p]
        lib.vx_cancel.restype = ctypes.c_int
        lib.vx_engine_stats.argtypes = [engine_p, ctypes.c_char_p, ctypes.c_int]
        lib.vx_engine_stats.restype = ctypes.c_int
        lib.vx_selftest_throw.argtypes = []
        lib.vx_selftest_throw.restype = ctypes.c_int
        # A deterministic fake dylib (VEYSTRIX_C_CPP_FAKE=1 in the C library) is
        # a test-only fixture, never a production engine. Refuse it unless an
        # explicit separate opt-in is present; do not accept it silently.
        mode = self.mode()
        if mode != "live" and os.environ.get("VEYSTRIX_CPP_ALLOW_FAKE") != "1":
            raise CppEngineError(
                f"C++ engine dylib mode is {mode!r}, not 'live'. A fake/stub "
                f"engine cannot serve; set {LIB_ENV} to a live bundle dylib "
                f"(VEYSTRIX_CPP_ALLOW_FAKE=1 only for explicit tests)."
            )

    @property
    def last_error(self) -> str:
        value = self._lib.vx_last_error()
        return value.decode("utf-8", "replace") if value else ""

    def _check(self, code: int, what: str) -> int:
        if code != VX_OK:
            detail = self.last_error
            msg = f"{what}: {detail or 'error code ' + str(code)}"
            if code == VX_ERR_CONTEXT_LIMIT:
                # The HTTP layer maps ContextLimitError -> 400; a raw
                # CppEngineError would surface as a 500.
                from veystrix.server import ContextLimitError
                raise ContextLimitError(msg)
            raise CppEngineError(msg, code)
        return code

    def abi_version(self) -> int:
        return int(self._lib.vx_abi_version())

    def require_v2(self, symbol: str) -> None:
        """Fail closed with the exact missing symbol when an old dylib is used."""
        if not self.has_v2 or symbol not in self.v2_symbols or not hasattr(self._lib, symbol):
            raise CppEngineError(
                f"this C++ engine dylib does not export {symbol}; install a bundle "
                f"with the ABI v2 engine (the feature is unavailable)."
            )

    def capabilities(self, engine) -> dict:
        import json as _json
        self.require_v2("vx_engine_capabilities")
        buf = ctypes.create_string_buffer(4096)
        self._check(self._lib.vx_engine_capabilities(engine, buf, len(buf)),
                    "vx_engine_capabilities")
        return _json.loads(buf.value.decode("utf-8"))

    def prefill_ex(self, session, tokens, flags: int) -> None:
        self.require_v2("vx_session_prefill_ex")
        arr = (ctypes.c_uint32 * max(len(tokens), 1))(*tokens) if tokens else (ctypes.c_uint32 * 1)()
        self._check(self._lib.vx_session_prefill_ex(session, arr, len(tokens), flags),
                    "vx_session_prefill_ex")

    def seed(self, session, seed: int) -> None:
        self.require_v2("vx_session_seed")
        self._check(self._lib.vx_session_seed(session, int(seed)), "vx_session_seed")

    def last_digest(self, session) -> str:
        self.require_v2("vx_session_last_digest")
        buf = ctypes.create_string_buffer(128)
        self._check(self._lib.vx_session_last_digest(session, buf, len(buf)),
                    "vx_session_last_digest")
        return buf.value.decode("ascii")

    def fork(self, src, dst) -> None:
        self.require_v2("vx_session_fork")
        self._check(self._lib.vx_session_fork(src, dst), "vx_session_fork")

    def tokens(self, session) -> int:
        self.require_v2("vx_session_tokens")
        count = int(self._lib.vx_session_tokens(session))
        if count < 0:
            raise CppEngineError(f"vx_session_tokens: {self.last_error or 'failed (negative token count)'}")
        return count

    def mode(self) -> str:
        value = self._lib.vx_mode()
        return value.decode("ascii", "replace") if value else ""

    def open_engine(self, model_dir: str, profile_json: str):
        handle = self._lib.vx_engine_open(model_dir.encode("utf-8"), profile_json.encode("utf-8"))
        if not handle:
            raise CppEngineError(f"vx_engine_open({model_dir}) failed: {self.last_error or 'null handle'}")
        return handle

    def close_engine(self, engine) -> None:
        self._lib.vx_engine_close(engine)

    def open_session(self, engine):
        handle = self._lib.vx_session_open(engine)
        if not handle:
            raise CppEngineError(f"vx_session_open failed: {self.last_error or 'null handle'}")
        return handle

    def close_session(self, session) -> None:
        self._lib.vx_session_close(session)

    def prefill(self, session, tokens) -> None:
        if not tokens:
            # C vx_session_prefill rejects a NULL/empty token buffer; an empty
            # prefill is a no-op (nothing to prefill), so skip it.
            return
        ids = (ctypes.c_uint32 * len(tokens))(*[int(t) for t in tokens])
        self._check(self._lib.vx_session_prefill(session, ids, len(tokens)), "vx_session_prefill")

    def step(self, session, sampler_json: str) -> int:
        out = ctypes.c_uint32()
        self._check(
            self._lib.vx_session_step(session, sampler_json.encode("utf-8"), ctypes.byref(out)),
            "vx_session_step",
        )
        return int(out.value)

    def step_batch(self, sessions, sampler_jsons) -> list[int]:
        """One greedy step for n sessions that share one engine (multi-ABI).

        All buffers are kept alive for the duration of the C call: the encoded
        bytes back the ``c_char_p`` array, and the three ctypes arrays are held
        in locals until ``vx_session_step_batch`` returns. Out tokens are read
        only after ``_check`` accepts VX_OK -- they are unspecified on error.
        """
        if not getattr(self, "has_step_batch", False) or not hasattr(self._lib, "vx_session_step_batch"):
            raise CppEngineError(
                "this C++ engine dylib does not export vx_session_step_batch; "
                "batched decode is unavailable. Install a bundle with the "
                "multi-session ABI or unset " + BATCH_ENV + "."
            )
        session_list = list(sessions)
        sampler_list = list(sampler_jsons)
        n = len(session_list)
        if n == 0:
            return []
        if n != len(sampler_list):
            raise CppEngineError(
                f"vx_session_step_batch: {n} sessions but {len(sampler_list)} samplers",
                VX_ERR_INVALID_ARGUMENT,
            )
        encoded = [str(sampler).encode("utf-8") for sampler in sampler_list]
        session_array = (ctypes.c_void_p * n)(*session_list)
        sampler_array = (ctypes.c_char_p * n)(*encoded)
        out_array = (ctypes.c_uint32 * n)()
        self._check(
            self._lib.vx_session_step_batch(session_array, n, sampler_array, out_array),
            "vx_session_step_batch",
        )
        return [int(out_array[index]) for index in range(n)]

    def save(self, session, directory: str) -> None:
        self._check(self._lib.vx_snapshot_save(session, directory.encode("utf-8")), "vx_snapshot_save")

    def restore(self, session, directory: str) -> None:
        self._check(self._lib.vx_snapshot_restore(session, directory.encode("utf-8")), "vx_snapshot_restore")

    def truncate(self, session, target_tokens: int) -> None:
        if hasattr(self._lib, "vx_session_truncate"):
            self._check(self._lib.vx_session_truncate(session, ctypes.c_uint32(target_tokens)), "vx_session_truncate")

    def cancel(self, session) -> None:
        self._check(self._lib.vx_cancel(session), "vx_cancel")

    def stats(self, engine) -> dict:
        buf = ctypes.create_string_buffer(4096)
        self._check(self._lib.vx_engine_stats(engine, buf, len(buf)), "vx_engine_stats")
        return json.loads(buf.value.decode("utf-8"))


class CppPrefixInfo:
    """Minimal prefix-hit info for the server's cache diagnostics."""

    def __init__(self, hit: int, total: int):
        self.hit_tokens = int(hit)
        self.prefilled_tokens = int(total)
        self.prefill_seconds = 0.0
        self.snapshot_seconds = 0.0

    def as_dict(self) -> dict:
        return {"hit_tokens": self.hit_tokens,
                "prefilled_tokens": self.prefilled_tokens}


class _CppSettingsRegistry:
    """Read-only settings surface mirroring the C ABI profile keys.

    The C++ engine is configured via profile JSON/env (resolved at engine
    open), so settings are reported, not mutated. This gives the branded UI
    the same visibility it has on the Python engine.
    """

    KEYS = ("model", "kv", "context_window", "max_cold_prefill",
            "max_incremental_prefill", "digests", "resident",
            "expert_pool_bytes", "expert_pool_policy", "nemotron_slots",
            "max_sessions")

    def for_backend(self, name: str):
        return list(self.KEYS)

    def render(self, backend, name: str, include_research: bool = False) -> str:
        try:
            profile = json.loads(backend.profile_json)
        except (ValueError, TypeError):
            profile = {}
        state = {k: profile[k] for k in self.KEYS if k in profile}
        state.setdefault("engine", "cpp")
        state.setdefault("allocator_cache_bytes", 512 * 2**20)
        state.setdefault("swap_ceiling_mib", 5120)
        return json.dumps(state, sort_keys=True)


def _default_tokenizer(model_dir: str | None):
    """Load the tokenizer lazily: only the cpp path needs transformers."""
    if not model_dir:
        raise CppEngineError("a model_dir is required for the C++ engine")
    from transformers import AutoTokenizer

    # Same tokenizer construction as FlashNextBackend; parity depends on it.
    return AutoTokenizer.from_pretrained(model_dir)


class CppEngineBackend(Conversation):
    """`Backend` protocol over the C ABI dylib.

    `binding` is injectable for tests; production uses CtypesVxBinding.
    """

    def __init__(self, model_dir: str | None = None, profile_json: str | None = None,
                 *, lib_path=None, binding=None, tokenizer=None, session_dir=None,
                 stop_tokens=None, **_kwargs):
        if isinstance(profile_json, (dict, list)):
            profile_json = json.dumps(profile_json, sort_keys=True)
        self.model_dir = str(model_dir or os.environ.get("VEYSTRIX_MODEL_DIR") or "")
        # The router's ProcessDriver.ready() compares installed_model_id to the
        # configured model id; python backends expose `model_path` for this.
        self.model_path = self.model_dir
        # D4/P5a: publish the same limits the Python backends publish so the
        # server's engine-agnostic admission can tighten them. Values come from
        # the profile (authoritative) or the environment; the C ABI enforces
        # them defensively too.
        try:
            profile_obj = json.loads(profile_json) if (profile_json or "").lstrip().startswith("{") else {}
        except (TypeError, ValueError):
            profile_obj = {}
        env_for_key = {
            "context_window": "VEYSTRIX_CONTEXT_LIMIT",
            "max_cold_prefill": "VEYSTRIX_MAX_COLD_PREFILL",
            "max_incremental_prefill": "VEYSTRIX_MAX_INCREMENTAL_PREFILL",
        }
        changed = False
        for key in ("context_window", "max_cold_prefill", "max_incremental_prefill"):
            if key not in profile_obj:
                env_value = os.environ.get(env_for_key[key])
                if env_value and env_value.isdigit() and int(env_value) > 0:
                    profile_obj[key] = int(env_value)
                    changed = True
        # Shadow v2 needs REAL logit digests from the engine, which the ABI only
        # records when the profile enables them (one branch, zero cost otherwise).
        self._shadow_path = os.environ.get(SHADOW_ENV) or None
        if self._shadow_path and not profile_obj.get("digests"):
            profile_obj["digests"] = True
            changed = True
        if changed:
            self.profile_json = json.dumps(profile_obj, sort_keys=True)
        else:
            self.profile_json = profile_json if profile_json is not None else "{}"
        self.context_limit = int(profile_obj.get("context_window") or 0) or None
        self.max_cold_prefill = int(profile_obj.get("max_cold_prefill") or 0) or None
        self.max_incremental_prefill = int(profile_obj.get("max_incremental_prefill") or 0) or None
        self.session_dir = Path(session_dir or os.environ.get("VEYSTRIX_SESSION_DIR")
                                or "~/.cache/veystrix/cpp/sessions").expanduser()
        if binding is None:
            translate_env()
            apply_runtime_memory_budget(model_dir)
        self._binding = binding if binding is not None else CtypesVxBinding(lib_path)
        version = self._binding.abi_version()
        if version < 1:
            raise CppEngineError(f"unsupported C ABI version {version}; need >= 1")
        self._batch_mode = batch_enabled()
        self._shared_key = None
        self._shared_entry = None
        if self._batch_mode:
            # Opt-in only: backends for the same (model_dir, profile) share ONE
            # engine, so a single vx_session_step_batch call can drive all their
            # sessions on that engine's driver thread. Each backend still owns
            # its own C session below.
            self._shared_key, self._shared_entry = _acquire_shared_engine(
                self._binding, self.model_dir, self.profile_json)
            self._engine = self._shared_entry.engine
            # Stable identity object: the scheduler groups with `is`.
            self._batch_group_key = self._shared_entry.group_key
        else:
            self._engine = self._binding.open_engine(self.model_dir, self.profile_json)
            self._batch_group_key = ("cpp", id(self._engine))
        self._session = None
        try:
            self._session = self._binding.open_session(self._engine)
        except BaseException:
            if self._batch_mode:
                _release_shared_engine(self._shared_key)
            else:
                self._binding.close_engine(self._engine)
            raise
        self._batch_prefill_tokens: list[int] = []
        self._closed = False
        self.stop_tokens = set(int(t) for t in (stop_tokens or ()))
        self.last_prefix = None
        self.last_decode_rate = None
        self.last_prefill_seconds = 0.0
        self.last_memory_report = None
        self.last_memory_after = None
        self.thinking_enabled = True
        # Same surface the Python backends expose. session.mirror_preferences() only copies
        # preferences (sampler, effort, budgets) onto backends that already define `sampling`,
        # so without these the server and chat ran every C++ request greedy.
        from veystrix.sampling import Sampling
        self.sampling = Sampling.greedy_settings()
        self.reasoning_effort = "medium"
        self.think_budget = 0
        self.answer_budget = 0
        self._interactive_budgets = None  # (answer, think) from the chat CLI
        self.finish_reason = "stop"
        # None means "the C session has no usable cached history": the next
        # generate() must RESET-prefill. An int means the C session holds that
        # many tape tokens and the next turn may APPEND (Sonnet finding 2/3).
        self._live_at = None
        self._seed_applied = False  # cleared whenever the C session is replaced
        self._settings = _CppSettingsRegistry()
        if tokenizer is None:
            resolved_tok = _default_tokenizer(self.model_dir)
        elif hasattr(tokenizer, "result") and callable(tokenizer.result):
            resolved_tok = tokenizer.result()
        else:
            resolved_tok = tokenizer
        super().__init__(resolved_tok)
        self._bg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._hydrate_prefix_cache()

    # --- Backend protocol -------------------------------------------------
    def truncate(self, target_tokens: int) -> bool:
        """Exact prefix truncation of the live C++ engine session to target_tokens."""
        if not getattr(self._binding, "has_v2", False) or not self._session:
            return False
        try:
            self._binding.truncate(self._session, target_tokens)
            self.tape = list(self.tape[:target_tokens])
            del self.pending[:]
            self._live_at = target_tokens
            return True
        except Exception:
            self._live_at = None
            return False

    def decode_token(self, token: int) -> str:
        return self.tokenizer.decode([int(token)])

    def _sampler_json(self) -> str:
        """Exactly Python's five sampling fields plus `greedy` (P2).

        The C++ parser refuses any other key by name — adding stop strings,
        frequency/repetition penalty or a per-request seed here would create a
        silent difference from the Python engine.
        """
        sampling = getattr(self, "sampling", None)
        payload = {}
        if sampling is not None:
            for key in ("greedy", "temperature", "top_p", "top_k", "min_p",
                        "presence_penalty"):
                if hasattr(sampling, key):
                    value = getattr(sampling, key)
                    if key != "greedy":
                        value = float(value) if key != "top_k" else int(value)
                    payload[key] = value
            if "greedy" not in payload:
                payload["greedy"] = bool(getattr(sampling, "temperature", 1.0) <= 0)
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def _prefix_cache_enabled() -> bool:
        raw = os.environ.get(PREFIX_CACHE_ENV)
        if raw is None or not raw.strip():
            return True
        return raw.strip().lower() not in _DISABLED_VALUES

    def _prefix_namespace(self):
        return (self.model_dir, self.profile_json, self._binding.abi_version())

    def _hydrate_prefix_cache(self) -> None:
        """Load persisted SSD prompt-cache entries for this model/profile on startup."""
        if not self._prefix_cache_enabled() or not getattr(self._binding, "has_v2", False):
            return
        root = self.session_dir / "prefix-cache"
        if not root.is_dir():
            return
        ns = self._prefix_namespace()
        try:
            entries = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime)
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir():
                continue
            tok_file = entry / "tokens.json"
            snap_dir = entry / "vx_snapshot"
            if not tok_file.is_file() or not (snap_dir / "meta.txt").is_file():
                continue
            try:
                meta = json.loads(tok_file.read_text(encoding="utf-8"))
                if tuple(meta.get("namespace", ())) != ns:
                    continue
                toks = tuple(int(t) for t in meta.get("tokens", ()))
                if toks:
                    entry_key = (ns, toks)
                    _CPP_PREFIX_CACHE[entry_key] = str(snap_dir)
                    if meta.get("kind") == "system":
                        _CPP_PREFIX_SYSTEM_ANCHORS.add(entry_key)
            except (OSError, ValueError, TypeError):
                continue
        self._supersede_all_strict_prefixes(ns)
        from veystrix.memory_tiers import current_tier
        default_quota = current_tier().prefix_cache_quota_gb
        self._prefix_evict(float(os.environ.get(PREFIX_CACHE_QUOTA_ENV, default_quota)))

    def _prefix_lookup(self, tokens):
        """Longest stored prefix or exact match, or None."""
        if not self._prefix_cache_enabled():
            return None
        namespace = self._prefix_namespace()
        best = None
        for (ns, cached), directory in _CPP_PREFIX_CACHE.items():
            if ns != namespace or not (1 <= len(cached) <= len(tokens)):
                continue
            if tuple(tokens[:len(cached)]) != cached:
                continue
            if best is None or len(cached) > len(best[1]):
                best = (directory, cached)
        if best is not None:
            hit_key = next(k for k, v in _CPP_PREFIX_CACHE.items() if v == best[0])
            _CPP_PREFIX_CACHE.move_to_end(hit_key)
        return best

    def prefix_cache_reusable_tokens(self, tokens) -> int:
        """Reusable cached prefix length for server admission and usage reporting."""
        if not getattr(self._binding, "has_v2", False):
            return 0
        hit = self._prefix_lookup(tokens)
        return len(hit[1]) if hit is not None else 0

    def _system_prefix_boundary(self, tokens) -> int | None:
        """Return token index right after the static system+tools message when >= 512 tokens."""
        if len(tokens) < 512:
            return None
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        unk = getattr(self.tokenizer, "unk_token_id", None)
        markers: list[int] = []
        if callable(convert):
            for name in ("<|end_message|>", "<|im_end|>"):
                tid = convert(name)
                if tid is not None and tid != unk:
                    markers.append(int(tid))
        if not markers:
            return None
        marker_set = set(markers)
        for idx in range(512, len(tokens) - 1):
            if int(tokens[idx]) in marker_set:
                cut = idx + 1
                if 512 <= cut < len(tokens):
                    return cut
                break
        return None

    def _prefix_evict(self, quota_gb: float) -> None:
        """Bound the persistent prefix cache by total snapshot bytes (LRU)."""
        if not _CPP_PREFIX_CACHE:
            return
        quota = int(quota_gb * (2**30))
        def size_of(path: str) -> int:
            root = Path(path).parent
            try:
                return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            except OSError:
                return 0
        entries = [(k, p, size_of(p)) for k, p in _CPP_PREFIX_CACHE.items()]
        total = sum(sz for _, _, sz in entries)
        if total <= quota:
            return
        removed_any = False
        for key, path, sz in entries:
            if key not in _CPP_PREFIX_CACHE:
                continue
            del _CPP_PREFIX_CACHE[key]
            shutil.rmtree(Path(path).parent, ignore_errors=True)
            removed_any = True
            total -= sz
            if total <= quota:
                break
        if removed_any:
            try:
                from veystrix.tool.storage import gc_cas_pages
                gc_cas_pages(self.session_dir / "prefix-cache")
            except Exception:
                pass

    def _supersede_strict_prefixes(self, ns, tok_tuple: tuple[int, ...]) -> None:
        """Remove non-anchor snapshots that are strict prefixes of `tok_tuple` in `ns`.

        In a multi-turn conversation, every turn extends the previous prompt prefix.
        Keeping every intermediate turn snapshot causes O(N^2) SSD bloat; superseding
        intermediate turn snapshots while preserving `system` anchors keeps disk usage
        O(1) per conversation with zero quality or TTFT loss.
        """
        new_key = (ns, tok_tuple)
        new_len = len(tok_tuple)
        stale = [
            (k, p)
            for k, p in _CPP_PREFIX_CACHE.items()
            if k[0] == ns
            and k != new_key
            and k not in _CPP_PREFIX_SYSTEM_ANCHORS
            and len(k[1]) < new_len
            and tok_tuple[:len(k[1])] == k[1]
        ]
        for old_key, old_path in stale:
            _CPP_PREFIX_CACHE.pop(old_key, None)
            shutil.rmtree(Path(old_path).parent, ignore_errors=True)
        if stale:
            try:
                from veystrix.tool.storage import gc_cas_pages
                gc_cas_pages(self.session_dir / "prefix-cache")
            except Exception:
                pass

    def _supersede_all_strict_prefixes(self, ns) -> None:
        """Prune already-superseded non-anchor prefix snapshots for `ns` on startup."""
        keys = sorted(
            (k for k in _CPP_PREFIX_CACHE if k[0] == ns and k not in _CPP_PREFIX_SYSTEM_ANCHORS),
            key=lambda k: len(k[1]),
        )
        all_ns = [k[1] for k in _CPP_PREFIX_CACHE if k[0] == ns]
        for old_key in keys:
            old_toks = old_key[1]
            old_len = len(old_toks)
            if any(len(other) > old_len and other[:old_len] == old_toks for other in all_ns):
                old_path = _CPP_PREFIX_CACHE.pop(old_key, None)
                if old_path:
                    shutil.rmtree(Path(old_path).parent, ignore_errors=True)

    def _prefix_store(self, tokens, *, system_anchor: bool = False):
        if not self._prefix_cache_enabled():
            return
        if not getattr(self._binding, "has_v2", False) or len(tokens) < 1:
            return
        ns = self._prefix_namespace()
        tok_tuple = tuple(int(t) for t in tokens)
        key = (ns, tok_tuple)
        if system_anchor:
            _CPP_PREFIX_SYSTEM_ANCHORS.add(key)
        if key in _CPP_PREFIX_CACHE and Path(_CPP_PREFIX_CACHE[key]).is_dir():
            _CPP_PREFIX_CACHE.move_to_end(key)
            return
        digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:24]
        directory = self.session_dir / "prefix-cache" / digest
        snapshot_dir = directory / "vx_snapshot"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self._binding.save(self._session, str(snapshot_dir))
            tok_tmp = directory / "tokens.json.tmp"
            tok_tmp.write_text(
                json.dumps({
                    "namespace": list(ns),
                    "tokens": list(tok_tuple),
                    "kind": "system" if system_anchor else "turn",
                }),
                encoding="utf-8",
            )
            os.replace(tok_tmp, directory / "tokens.json")
            _CPP_PREFIX_CACHE[key] = str(snapshot_dir)
            if not system_anchor:
                self._supersede_strict_prefixes(ns, tok_tuple)
            from veystrix.memory_tiers import current_tier
            default_quota = current_tier().prefix_cache_quota_gb
            self._prefix_evict(float(os.environ.get(PREFIX_CACHE_QUOTA_ENV, default_quota)))
        except (CppEngineError, OSError):
            _CPP_PREFIX_CACHE.pop(key, None)

    def _prefill_with_prefix(self, tokens, on_prefill_progress=None,
                             should_cancel=None) -> None:
        self._last_prefix_restore = None
        self._last_chunk_spans = None
        hit = self._prefix_lookup(tokens)
        if hit is not None and getattr(self._binding, "has_v2", False):
            directory, cached = hit
            try:
                self._binding.restore(self._session, directory)
            except CppEngineError:
                _CPP_PREFIX_CACHE.pop((self._prefix_namespace(), cached), None)
                hit = None
            else:
                self._last_prefix_restore = len(cached)
                remainder = tokens[len(cached):]
                if remainder:
                    self._append_with_chunks(
                        cached_len=len(cached),
                        prefix_tokens=list(cached),
                        remainder=remainder,
                        on_prefill_progress=on_prefill_progress,
                        should_cancel=should_cancel,
                    )
                if on_prefill_progress is not None:
                    on_prefill_progress(len(tokens), len(tokens))
                return
        sys_cut = self._system_prefix_boundary(tokens)
        if sys_cut is not None and self._prefix_cache_enabled() and getattr(self._binding, "has_v2", False):
            # Anchor and persist the static system+tools prefix before the user turn so
            # subsequent messages, cancelled turns, and new conversations hit it in ~0.2s.
            self._prefill_with_chunks(
                tokens[:sys_cut],
                on_prefill_progress=(lambda d, _t: on_prefill_progress(d, len(tokens)))
                if on_prefill_progress is not None else None,
                should_cancel=should_cancel,
            )
            self._prefix_store(tokens[:sys_cut], system_anchor=True)
            self._append_with_chunks(
                cached_len=sys_cut,
                prefix_tokens=list(tokens[:sys_cut]),
                remainder=tokens[sys_cut:],
                on_prefill_progress=(lambda d, _t: on_prefill_progress(sys_cut + d, len(tokens)))
                if on_prefill_progress is not None else None,
                should_cancel=should_cancel,
            )
            return
        self._prefill_with_chunks(tokens, on_prefill_progress,
                                  should_cancel=should_cancel)

    def _append_with_chunks(self, *, cached_len: int, prefix_tokens: list[int], remainder: list[int],
                            on_prefill_progress=None, should_cancel=None) -> None:
        if not remainder:
            return
        spans = self._chunk_spans(len(remainder), cached_len, for_progress=(on_prefill_progress is not None))
        if not spans:
            if should_cancel is not None and should_cancel():
                self.tape = list(prefix_tokens)
                del self.pending[:]
                self._live_at = len(self.tape)
                self._binding.cancel(self._session)
                raise GenerationCancelled()
            self._binding.prefill_ex(self._session, remainder, 1)
            return
        done = 0
        for start, end in spans:
            if should_cancel is not None and should_cancel():
                self._cancel_prefill(list(prefix_tokens) + list(remainder[:done]))
            self._binding.prefill_ex(self._session, remainder[start:end], 1)
            done = end
            if on_prefill_progress is not None:
                on_prefill_progress(done, len(remainder))

    def _cancel_prefill(self, committed_tokens: list[int]) -> None:
        if committed_tokens:
            self.tape = committed_tokens
            del self.pending[:]
            self._live_at = len(self.tape)
            if len(self.tape) >= 512:
                self._prefix_store(self.tape)
        self._binding.cancel(self._session)
        raise GenerationCancelled()

    def _profile_prefill_chunk(self) -> int:
        try:
            obj = json.loads(self.profile_json) if self.profile_json else {}
            val = int(obj.get("prefill_chunk") or 0) if isinstance(obj, dict) else 0
            return val if val > 0 else 0
        except (ValueError, TypeError):
            return 0

    def _chunk_spans(self, total: int, offset: int, *, for_progress: bool = False):
        """Python's OWN anchored chunk plan, reused so C++ chunks match exactly.

        Lazy import keeps this module importable without MLX; if MLX or the
        planner is unavailable we return None and keep the single-call path.
        When ``for_progress`` is True and the profile configures ``prefill_chunk``
        (generic runtime), emit chunk spans of that exact size so live prefill
        percentage updates between chunks without altering chunk boundaries.
        """
        if total <= 0:
            return None
        chunk = int(os.environ.get("FLASHNEXT_PREFILL_CHUNK_TOKENS", "0") or 0)
        if chunk > 0:
            try:
                from models.flashnext.prefill import plan_chunks
                spans = list(plan_chunks(total, chunk, offset=offset))
                return spans if len(spans) > 1 else None
            except Exception:
                if total > chunk:
                    return [(off, min(total, off + chunk)) for off in range(0, total, chunk)]
                return None
        if for_progress and getattr(self._binding, "has_v2", False):
            pchunk = self._profile_prefill_chunk()
            if pchunk > 0 and total > pchunk:
                return [(off, min(total, off + pchunk)) for off in range(0, total, pchunk)]
        return None

    def _prefill_with_chunks(self, tokens, on_prefill_progress=None,
                             should_cancel=None) -> None:
        if on_prefill_progress is not None:
            on_prefill_progress(0, len(tokens))
        spans = self._chunk_spans(len(tokens), len(self.tape), for_progress=(on_prefill_progress is not None))
        self._last_chunk_spans = spans
        if not spans:
            self._binding.prefill(self._session, tokens)
            return
        if not getattr(self._binding, "has_v2", False):
            # Fail closed: an old dylib cannot honour the chunk plan.
            raise CppEngineError(
                "this C++ engine dylib lacks the v2 APPEND prefill ABI; "
                "unset FLASHNEXT_PREFILL_CHUNK_TOKENS or install an ABI v2 bundle"
            )
        first_end = spans[0][1]
        self._binding.prefill(self._session, tokens[:first_end])
        done = first_end
        if on_prefill_progress is not None:
            on_prefill_progress(done, len(tokens))
        for start, end in spans[1:]:
            if should_cancel is not None and should_cancel():
                self._cancel_prefill(list(tokens[:done]) if done > 0 else [])
            self._binding.prefill_ex(self._session, tokens[start:end], 1)
            done = end
            if on_prefill_progress is not None:
                on_prefill_progress(done, len(tokens))

    def _maybe_seed_session(self) -> None:
        """Seed once per session lifetime (re-seeding each turn would restart
        the key stream and diverge from Python's process-global key after turn 1)."""
        if getattr(self, "_seed_applied", False):
            return
        seed = None
        sampling = getattr(self, "sampling", None)
        for source in (sampling, self):
            if source is not None and getattr(source, "seed", None) is not None:
                seed = getattr(source, "seed")
                break
        if seed is None:
            seed = os.environ.get("VEYSTRIX_CPP_SEED")
        if seed is None:
            return
        if not getattr(self._binding, "has_v2", False):
            return  # old dylib stays on its existing greedy/refusal path
        self._binding.seed(self._session, int(seed))
        self._seed_applied = True

    def generate(
        self,
        max_tokens: int,
        out=None,
        on_prefilled=None,
        on_prefill_progress=None,
        on_decode_token=None,
        should_cancel=None,
    ):
        prompt_tokens = [int(t) for t in self.tape] + [int(t) for t in self.pending]
        self._maybe_seed_session()
        sampler_json = self._sampler_json()
        timer = DecodeTimer()
        prefill_began = time.perf_counter()
        # Live continuation: if the C session already holds exactly self.tape
        # tokens, APPEND only the new `pending` tokens. Re-prefilling the whole
        # history each turn changes chunk shapes and can exceed the cold limit.
        live = (getattr(self, "_live_at", None) == len(self.tape)
                and len(self.tape) > 0  # never APPEND into an empty C session
                and bool(self.pending)
                and getattr(self._binding, "has_v2", False))
        if not live and getattr(self, "_live_at", None) is not None and getattr(self._binding, "has_v2", False):
            shared = self.common_prefix(prompt_tokens)
            if shared >= 256 and shared < len(self.tape):
                if self.truncate(shared):
                    self.pending = list(prompt_tokens[shared:])
                    live = True
        if live and callable(getattr(self._binding, "tokens", None)):
            try:
                cnt = self._binding.tokens(self._session)
                if cnt is not None and cnt >= 0 and cnt != len(self.tape):
                    live = False
            except Exception:
                pass
        prefill_len = len(self.pending) if live else len(prompt_tokens)
        try:
            if live:
                if on_prefill_progress is not None:
                    on_prefill_progress(0, prefill_len)
                self._append_with_chunks(
                    cached_len=len(self.tape),
                    prefix_tokens=list(self.tape),
                    remainder=list(self.pending),
                    on_prefill_progress=on_prefill_progress,
                    should_cancel=should_cancel,
                )
                self._last_prefix_restore = len(self.tape)
                self._last_chunk_spans = None
            else:
                self._prefill_with_prefix(prompt_tokens, on_prefill_progress,
                                          should_cancel=should_cancel)
        except GenerationCancelled:
            raise
        except BaseException:
            # A failed APPEND leaves some chunks in the C session but `pending`
            # intact; re-appending them next turn would duplicate tokens (#5).
            self._live_at = None
            raise
        self.last_prefix = CppPrefixInfo(self._last_prefix_restore or 0,
                                         len(prompt_tokens))
        self._prefix_store(prompt_tokens)
        prefill_seconds = time.perf_counter() - prefill_began
        del self.pending[:]
        if on_prefilled is not None:
            on_prefilled()
        if on_prefill_progress is not None:
            on_prefill_progress(prefill_len, prefill_len)

        horizon = UNLIMITED_GENERATION_HORIZON if not max_tokens or max_tokens < 0 else int(max_tokens)
        # Stop cleanly at the admitted context ceiling instead of letting the
        # C ABI throw VX_ERR_CONTEXT_LIMIT mid-stream after tokens were already
        # emitted (Sonnet finding 12). Mirror flashnext.py:1093-1095.
        ceiling = getattr(self, "context_limit", None)
        if ceiling:
            horizon = min(horizon, max(0, int(ceiling) - len(prompt_tokens)))
        emitted: list[int] = []
        _in_step = 0.0
        _decode_began = time.perf_counter()
        digests: list[str] = []
        chunks: list[str] = []
        partial: list[int] = []  # UTF-8 bytes spanning tokens (Sonnet finding 2)
        finish = "length"  # until a stop token is actually emitted
        hit_stop = False
        # Commit the prompt to the tape BEFORE decoding so a cancel/error keeps
        # the user's question in the conversation (Sonnet finding 3). The turn
        # is not closed yet: a cancel mid-answer must still emit <|im_end|> on
        # the next user turn, mirroring flashnext.py:1099-1101 (Sonnet #3).
        self.tape = list(prompt_tokens)
        self.turn_closed = False
        # Interactive think/answer budgets. Mirror flashnext.py:1002-1029,
        # 1265-1291, 1354-1361 exactly: a phase switch happens when </think>
        # is emitted (forced or natural), and the closing token is counted in
        # the THINKING phase, not silently skipped (Sonnet finding 10).
        budgets = getattr(self, "_interactive_budgets", None)
        budget_answer = budget_think = None
        phase = "answer"
        think_count = answer_count = 0
        close_token = None
        if budgets is not None:
            raw_answer, raw_think = budgets
            budget_answer = (None if raw_answer is None or int(raw_answer) < 0
                             else max(0, int(raw_answer)))
            # Inkling closes thinking with <|end_message|> and opens the answer
            # with <|content_text|>; forcing Qwen's </think> token would inject
            # a foreign marker into its stream. Its own parser handles phases.
            inkling = response_format_for(getattr(self, "family", None)) == "inkling"
            phase = "answer" if inkling else ("thinking" if self.thinking_enabled else "answer")
            if not inkling and self.thinking_enabled and raw_think is not None and int(raw_think) >= 0:
                budget_think = max(0, int(raw_think))
                ids = self.encode("</think>")
                if len(ids) != 1:
                    raise ValueError(
                        "interactive reasoning closure must encode as one token")
                close_token = int(ids[0])
        began = time.perf_counter()
        try:
            for _ in range(horizon):
                if should_cancel is not None and should_cancel():
                    self._binding.cancel(self._session)
                    raise GenerationCancelled()
                step_json = sampler_json
                # Force the close token on the LAST thinking token, matching
                # Python's `thinking == budget_think - 1` condition.
                force_close = (
                    phase == "thinking" and budget_think is not None
                    and budget_think > 0 and think_count == budget_think - 1
                )
                if force_close:
                    obj = json.loads(sampler_json) if sampler_json else {}
                    obj["force"] = close_token
                    step_json = json.dumps(obj, sort_keys=True)
                try:
                    if _STEP_TIMING:
                        _t_step = time.perf_counter()
                    token = self._binding.step(self._session, step_json)
                    if _STEP_TIMING:
                        _in_step += time.perf_counter() - _t_step
                except CppEngineError as exc:
                    if exc.code == VX_ERR_CANCELLED:
                        raise GenerationCancelled() from exc
                    raise
                if force_close:
                    token = close_token  # replace the sampled token, as Python
                # The stop token is a protocol delimiter: Python returns before
                # yielding it. Keep it on the tape but never stream its text or
                # count it as a produced token (Sonnet finding 1).
                if token in self.stop_tokens:
                    self.tape.append(token)  # protocol delimiter stays on tape
                    finish = "stop"
                    hit_stop = True
                    break
                if phase == "answer" and budget_answer is not None \
                        and answer_count >= budget_answer:
                    # Python returns before yielding this token. The model has
                    # still consumed it, so the C cache is one ahead of tape:
                    # force a full re-prefill next turn.
                    self._live_at = None
                    break
                piece = stream_decode(self.tokenizer, partial, token)
                emitted.append(token)
                self.tape.append(token)
                # Count in the CURRENT phase, then switch phase if closed. The
                # closing token is a thinking token, exactly like Python.
                if phase == "thinking":
                    think_count += 1
                    # The close marker may span tokens (stream_decode holds
                    # partial UTF-8), so also compare the token id directly.
                    if force_close or token == close_token or "</think>" in piece:
                        phase = "answer"
                else:
                    answer_count += 1
                if piece:
                    chunks.append(piece)
                # Shadow v2: a REAL float32-LE logit digest from the ABO when
                # the profile enables it. Fall back to the legacy token-id
                # hash only for old dylibs/shadow-off so the answer is unchanged.
                if self._shadow_path and getattr(self._binding, "has_v2", False):
                    try:
                        digests.append(self._binding.last_digest(self._session))
                    except CppEngineError:
                        digests.append(hashlib.sha256(struct.pack("<I", token)).hexdigest())
                else:
                    digests.append(hashlib.sha256(struct.pack("<I", token)).hexdigest())
                if on_decode_token is not None:
                    with timer.emitting():
                        on_decode_token(token, piece)
                if piece and out is not None:
                    with timer.emitting():
                        out(piece)
        except GenerationCancelled:
            # If cancelled cleanly between decode steps, the C session matches
            # self.tape exactly — keep _live_at intact so the next turn reuses it.
            if getattr(self._binding, "has_v2", False):
                try:
                    if self._binding.tokens(self._session) == len(self.tape):
                        self._live_at = len(self.tape)
                    else:
                        self._live_at = None
                except CppEngineError:
                    self._live_at = None
            else:
                self._live_at = None
            raise
        except BaseException:
            # A context-limit/error mid-decode may leave the C session diverged
            # from self.tape. Force a full RESET re-prefill next turn.
            self._live_at = None
            raise
        if _STEP_TIMING:
            _total = time.perf_counter() - _decode_began
            print(f"[step-timing] tokens={len(emitted)} in_step={_in_step:.2f}s outside_step={_total - _in_step:.2f}s "
                  f"({(_total - _in_step) * 1000 / max(1, len(emitted)):.1f} ms/token outside, "
                  f"{_in_step * 1000 / max(1, len(emitted)):.1f} ms/token in step) sampler={sampler_json} "
                  f"pid={os.getpid()} nice={os.getpriority(os.PRIO_PROCESS, 0)}", file=sys.stderr, flush=True)
        # Flush a trailing incomplete UTF-8 sequence (Sonnet finding 14).
        if partial:
            tail = self.tokenizer.decode(partial)
            chunks.append(tail)
            if out is not None:
                out(tail)
        elapsed = timer.since((began, 0.0))
        self._live_at = len(self.tape)  # C session caches this history
        # A turn is closed only when a stop token ended it. Hitting max_tokens
        # leaves the turn open (the caller can continue) and reports length.
        self.turn_closed = hit_stop
        self.finish_reason = finish
        answer = "".join(chunks)

        if self._shadow_path:  # opt-in only: one branch when disabled
            self._write_shadow(prompt_tokens, sampler_json, emitted, digests, finish)

        stats = Stats(
            finish=finish,
            tokens=len(emitted),
            seconds=elapsed,
            prompt_tokens=len(prompt_tokens),
            prefill_seconds=prefill_seconds,
        )
        self.last_decode_rate = stats.rate if stats.tokens and stats.seconds else None
        self.last_prefill_seconds = prefill_seconds
        return answer, stats

    def _write_shadow(self, prompt, sampler, emitted, digests, finish) -> None:
        try:
            profile = json.loads(self.profile_json)
        except (ValueError, TypeError):
            profile = self.profile_json
        record = {
            "schema": SHADOW_SCHEMA,
            "engine": "cpp",
            "mode": self._binding.mode(),
            "model_dir": self.model_dir,
            "prompt_token_ids": prompt,
            "profile": profile,
            "sampler": sampler,
            "emitted_token_ids": emitted,
            "step_digests": digests,        # real logit digests in schema 2
            "digest_source": "logit" if self._shadow_path and
                             getattr(self._binding, "has_v2", False) else "token_id",
            "seed": getattr(getattr(self, "sampling", None), "seed", None),
            "kv": profile.get("kv") if isinstance(profile, dict) else None,
            "chunk_boundaries": getattr(self, "_last_chunk_spans", None),
            "prefix_restore_tokens": getattr(self, "_last_prefix_restore", None),
            "profile_key": getattr(self, "profile_label", None),
            "finish": finish,
        }
        try:
            with open(self._shadow_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            # Shadow logging must never change the generated answer.
            pass

    def check_invariant(self) -> bool:
        # Engine-wide session_tokens is shared by all sessions on one engine;
        # compare THIS session's count (ABI v2 vx_session_tokens) instead.
        if getattr(self._binding, "has_v2", False):
            try:
                return self._binding.tokens(self._session) == len(self.tape)
            except CppEngineError:
                return False
        stats = self.engine_stats()
        try:
            session_tokens = int(stats.get("session_tokens", -1))
        except (TypeError, ValueError):
            return False
        return session_tokens == len(self.tape)

    # --- parked sessions ---------------------------------------------------
    @staticmethod
    def _parked_limits() -> tuple[int, int]:
        from veystrix.memory_tiers import current_tier
        tier = current_tier()
        def number(name, default):
            try:
                return max(0, int(os.environ.get(name, default)))
            except ValueError:
                return default
        return number(PARKED_ENV, tier.parked_sessions), number(PARKED_TOKENS_ENV, tier.parked_tokens)

    @staticmethod
    def _parking_memory_ok() -> bool:
        try:
            floor = float(os.environ.get(PARKED_MIN_AVAILABLE_ENV, "3.5"))
        except ValueError:
            floor = 3.5
        if floor <= 0:
            return True
        try:
            from veystrix.bench.memory import sample
            available = sample()["available_bytes"]
        except Exception:
            return False  # unknown memory state: do not hold extra sessions
        return bool(available) and available >= floor * 2**30

    def _park_current(self) -> bool:
        """Keep the live C session open instead of closing it. True when it was parked."""
        count, budget = self._parked_limits()
        if getattr(self, "_parked", None) and not self._parking_memory_ok():
            self._close_parked()
        if (count == 0 or self._closed or getattr(self, "_batch_mode", False) or self._session is None
                or self.pending or len(self.tape) < PARKED_MIN_TOKENS or len(self.tape) > budget
                or getattr(self, "_live_at", None) != len(self.tape) or not self._parking_memory_ok()):
            return False
        parked = self.__dict__.setdefault("_parked", [])
        parked.append({"session": self._session, "tape": tuple(int(t) for t in self.tape),
                       "seed_applied": getattr(self, "_seed_applied", False)})
        while parked and (len(parked) > count or sum(len(p["tape"]) for p in parked) > budget):
            self._binding.close_session(parked.pop(0)["session"])
        return True

    def parked_prefix(self, ids) -> int:
        """Tokens of the longest parked history that ``ids`` matches or extends (0: none)."""
        best = 0
        for entry in getattr(self, "_parked", ()):
            n = len(entry["tape"])
            if best < n <= len(ids) and tuple(ids[:n]) == entry["tape"]:
                best = n
        return best

    def activate_parked(self, ids) -> int:
        """Swap the parked session that ``ids`` extends in as the live session; returns its token count."""
        parked = getattr(self, "_parked", [])
        n = self.parked_prefix(ids)
        if not n or self._closed:
            return 0
        index = next(i for i, e in enumerate(parked) if len(e["tape"]) == n and tuple(ids[:n]) == e["tape"])
        entry = parked.pop(index)
        if not self._park_current() and self._session is not None:
            self._binding.close_session(self._session)
        self._session = entry["session"]
        self.tape = list(entry["tape"])
        self.pending = []
        self._live_at = n
        self._seed_applied = entry["seed_applied"]
        self._replay_needed = False
        return n

    def _close_parked(self) -> None:
        for entry in getattr(self, "_parked", ()):
            try:
                self._binding.close_session(entry["session"])
            except Exception:
                pass
        self.__dict__["_parked"] = []

    def reset(self) -> None:
        if not self._closed:
            if not self._park_current():
                self._binding.close_session(self._session)
            try:
                self._session = self._binding.open_session(self._engine)
            except BaseException:
                # Do not leave a dangling handle the next call would reuse.
                self._session = None
                self._closed = True
                raise
        self.tape.clear()
        self.pending.clear()
        # A fresh C session has no cached history and no PRNG seeding.
        self._live_at = None
        self._seed_applied = False
        self._batch_prefill_tokens = []
        self.turn_closed = True
        self.finish_reason = "stop"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        bg = getattr(self, "_bg_executor", None)
        if bg is not None:
            try:
                bg.shutdown(wait=False)
            except Exception:
                pass
            self._bg_executor = None
        self._close_parked()
        session = getattr(self, "_session", None)
        if session is not None:
            self._binding.close_session(session)
            self._session = None
        if getattr(self, "_batch_mode", False):
            # Shared engines outlive the backend that last used them. Never
            # close one here unless this was the last reference.
            _release_shared_engine(self._shared_key)
        else:
            self._binding.close_engine(self._engine)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _session_path(self, name: str) -> Path:
        safe = Path(str(name)).name
        if not safe or safe in (".", ".."):
            raise ValueError("invalid session name")
        return self.session_dir / safe

    def save_session(self, name: str) -> str:
        path = self._session_path(name)
        path.mkdir(parents=True, exist_ok=True)
        self._binding.save(self._session, str(path / "vx_snapshot"))
        transcript = {
            "tape": list(self.tape),
            "pending": list(self.pending),
            "turn_closed": self.turn_closed,
            "profile": self.profile_json,
            "model_dir": self.model_dir,
        }
        (path / "transcript.json").write_text(json.dumps(transcript, sort_keys=True), encoding="utf-8")
        return str(path)

    def load_session(self, name: str) -> str:
        path = self._session_path(name)
        self._binding.restore(self._session, str(path / "vx_snapshot"))
        transcript = json.loads((path / "transcript.json").read_text(encoding="utf-8"))
        self.tape = [int(t) for t in transcript["tape"]]
        self.pending = [int(t) for t in transcript["pending"]]
        self.turn_closed = bool(transcript["turn_closed"])
        # The restored snapshot may hold only the prompt of the saved turn, not
        # the emitted tokens on the tape. Fail safe: force a RESET re-prefill.
        self._live_at = None
        self._seed_applied = False
        return str(path)

    def list_sessions(self) -> str:
        if not self.session_dir.is_dir():
            return json.dumps([])
        return json.dumps(sorted(p.name for p in self.session_dir.iterdir() if p.is_dir()))

    def delete_session(self, name: str) -> str:
        path = self._session_path(name)
        if path.is_dir():
            shutil.rmtree(path)
        return json.dumps({"deleted": path.name})

    def configure(self, argument: str) -> str:
        text = str(argument).strip()
        if text in ("", "help", "?"):
            return "cpp engine: status | profile=<json> | help"
        if text == "status":
            return json.dumps({"engine": "cpp", "stats": self.engine_stats()}, sort_keys=True)
        if text.startswith("profile="):
            new_profile = text[len("profile="):]
            # Close the old session AND drop this backend's claim on the old
            # engine first. A shared engine is released by refcount, never
            # closed outright here (other backends may still be using it).
            self._close_parked()
            self._binding.close_session(self._session)
            if self._batch_mode:
                _release_shared_engine(self._shared_key)
            else:
                self._binding.close_engine(self._engine)
            self.profile_json = new_profile
            if self._batch_mode:
                self._shared_key, self._shared_entry = _acquire_shared_engine(
                    self._binding, self.model_dir, self.profile_json)
                self._engine = self._shared_entry.engine
                self._batch_group_key = self._shared_entry.group_key
            else:
                self._engine = self._binding.open_engine(self.model_dir, self.profile_json)
                self._batch_group_key = ("cpp", id(self._engine))
            self._session = self._binding.open_session(self._engine)
            # New session: no cached history, no PRNG key.
            self._live_at = None
            self._seed_applied = False
            return "profile updated"
        raise ValueError(f"unknown cpp engine configuration: {text}")

    def settings_registry(self):
        return self._settings

    def settings_state(self, include_research: bool = False) -> str:
        return self._settings.render(self, "cpp", include_research)

    def engine_stats(self) -> dict:
        return self._binding.stats(self._engine)

    @property
    def cache_tokens(self) -> int:
        return len(self.tape)

    # --- batch_* hooks ----------------------------------------------------
    # Opt-in only (VEYSTRIX_CPP_BATCH=1). All measured batched variants were
    # SLOWER than the baseline, so the default path never batches: fork returns
    # None and the decode hooks fail closed. When enabled, backends for the same
    # model+profile share one engine, so vx_session_step_batch can legally drive
    # all their sessions on that engine's single driver thread.
    #
    # The C prefill RESETS session state, so chunked prefill would keep only the
    # last chunk: the opt-in path buffers every chunk and prefills once on the
    # final chunk. Each session has its own PRNG key sequence (ABI v2 seed),
    # so greedy and non-greedy sampling both work in batch mode.

    def _batch_greedy(self) -> bool:
        sampling = getattr(self, "sampling", None)
        if sampling is None:
            return True
        greedy = getattr(sampling, "greedy", True)
        temperature = getattr(sampling, "temperature", 0.0)
        return bool(greedy) and not (isinstance(temperature, (int, float)) and temperature > 0)

    def _require_batch_greedy(self) -> None:
        if not self._batch_greedy():
            raise CppEngineError(
                "batched decode is greedy-only; use greedy sampling or unset " + BATCH_ENV
            )

    def _require_batch_mode(self) -> None:
        if not getattr(self, "_batch_mode", False):
            self._batch_unsupported()

    def fork_for_batch(self):
        if not getattr(self, "_batch_mode", False) or self._closed:
            return None
        # Fail closed: no batching when the loaded dylib lacks the ABI or the
        # injected binding cannot serve it. The normal cooperative step path
        # (batch_decode -> vx_session_step) still works in that case.
        if not getattr(self._binding, "has_step_batch", True) or not hasattr(self._binding, "step_batch"):
            return None
        from copy import copy

        child = copy(self)
        child._closed = False
        # Distinct C session on the SAME shared engine: that is what makes the
        # batched ABI legal. A fresh session has an empty tape/pending.
        child._session = self._binding.open_session(self._engine)
        child.tape = []
        child.pending = []
        child._batch_prefill_tokens = []
        child.turn_closed = True
        child.finish_reason = "stop"
        # The fork is a brand-new C session: nothing cached, nothing seeded.
        child._live_at = None
        child._seed_applied = False
        # The child holds its own reference so the engine cannot be closed by
        # the parent while this fork is still decoding on it.
        _retain_shared_engine(child._shared_key)
        return child

    @contextmanager
    def batch_scope(self):
        yield

    def batch_context(self) -> int:
        # Nemotron admits 1M; Qwen's tiered4 window is 65k cold / 262k logical.
        model = getattr(self, "model_path", "") or ""
        if "nemotron" in str(model).lower():
            return 1048576
        return int(self.context_limit or 262144)

    def batch_prepare(self, tokens) -> None:
        return None

    @staticmethod
    def batch_state_bytes(tokens) -> int:
        # INT8 KV (6 layers * K/V * 2 heads * 136 B) + ~27 KiB/token of
        # recurrent state; use a conservative upper bound so admission does not
        # over-commit (Sonnet P0 finding 3).
        return int(tokens) * (27 * 1024 + 6 * 2 * 2 * 136)

    @staticmethod
    def batch_stats(**fields) -> Stats:
        return Stats(**fields)

    def _batch_unsupported(self):
        raise CppEngineError("batched decode is not supported by the C++ engine (set " + BATCH_ENV + "=1)")

    def batch_group_key(self):
        # Return the cached tuple OBJECT: BatchScheduler groups with `is`, and
        # every backend on one shared engine must yield the same object.
        return self._batch_group_key

    def batch_prefill(self, chunk, offset, final, sampler):
        """Prefill one 128-token quantum immediately via APPEND.

        Buffering all quanta until `final` caused head-of-line blocking: other
        sessions could not decode while one session's whole prompt was fed.
        The first chunk RESETs; subsequent chunks APPEND to the live cache.
        """
        self._require_batch_mode()
        # A forked batch child is a fresh C session; seed it once before its
        # first prefill (Sonnet finding 11).
        self._maybe_seed_session()
        part = [int(t) for t in chunk]
        if not part:
            return self._binding.step(self._session, self._sampler_json()) if final else None
        first = not getattr(self, "_batch_prefill_started", False)
        try:
            if first:
                self._binding.prefill(self._session, part)
                self._batch_prefill_started = True
            elif hasattr(self._binding, "prefill_ex"):
                self._binding.prefill_ex(self._session, part, 1)
            else:
                # Stub/old binding without APPEND: fall back to a single full
                # prefill on final (never serve silently wrong tokens).
                self._batch_prefill_tokens.extend(part)
                if final:
                    all_tokens = self._batch_prefill_tokens
                    self._batch_prefill_tokens = []
                    self._binding.prefill(self._session, all_tokens)
                return self._binding.step(self._session, self._sampler_json()) if final else None
        except BaseException:
            # A cancel/error mid-prefill must not leave the "started" flag set,
            # which would make the next request APPEND onto a fresh session.
            self._batch_prefill_started = False
            self._batch_prefill_tokens = []
            self._live_at = None
            raise
        self._batch_tokens_seen = getattr(self, "_batch_tokens_seen", 0) + len(part)
        if final:
            self._batch_prefill_started = False
            self._batch_tokens_seen = 0
        # cooperative.generate uses the first decode token returned here as the
        # first value in its decode loop, so produce it after the final quantum.
        return self._binding.step(self._session, self._sampler_json()) if final else None

    def batch_decode(self, value, sampler, want_logits):
        self._require_batch_mode()
        return self._binding.step(self._session, self._sampler_json())

    def batch_decode_group(self, steps):
        self._require_batch_mode()
        if not steps:
            return []
        backends = [step[0] for step in steps]
        # All steps in a group share one engine (that is how batch_group_key
        # groups them), so one step_batch call on any of their bindings is safe.
        sessions = [backend._session for backend in backends]
        samplers = [backend._sampler_json() for backend in backends]
        return list(self._binding.step_batch(sessions, samplers))

    def batch_prefix_plan(self, prompt, cached, width):
        return cached, None

    def batch_save_prefix(self) -> None:
        return None


# Terminal model names -> the C ABI's engine selector (see capi/veystrix_c.cpp
# engine_name_for_open). Only Nemotron has a hand-written runtime; every Qwen family
# (Flash-Next, Ornith) is built by veystrix.backends.cpp_generic before this factory.
_CPP_MODEL_FOR_NAME = {
    "nemotron35-lightning": "nemotron",
    "nemotron": "nemotron",
}
CPP_CONFIG_PATH_ENV = "VEYSTRIX_CPP_CONFIG_PATH"


_LAST_CONFIG_ERROR = ""  # set by _auto_flat_config for an actionable SystemExit


def _auto_flat_config(model_dir, cpp_model: str) -> Path | None:
    """Derive a flat config from a checkpoint (atomic writes + surfaced errors)."""
    global _LAST_CONFIG_ERROR
    _LAST_CONFIG_ERROR = ""
    if not model_dir or not Path(model_dir).is_dir():
        return None
    repo = Path(__file__).resolve().parents[2]
    gen = repo / "cpp/engine/tools" / "gen_nemotron_config.py"
    if not gen.is_file():
        return None
    # Fold the generator's own mtime into the key so a generator fix invalidates
    # cached configs (Sonnet finding 14).
    version = int(gen.stat().st_mtime)
    key = hashlib.sha256(
        f"{Path(model_dir).resolve()}|{cpp_model}|{version}".encode()).hexdigest()[:20]
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    out = cache / "veystrix" / "cpp-config" / key / "flat.txt"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        ckpt = Path(model_dir)
        required = [ckpt / "config.json", ckpt / "model.safetensors.index.json"]
        missing = [p.name for p in required if not p.is_file()]
        if missing:
            _LAST_CONFIG_ERROR = f"checkpoint missing {', '.join(missing)}"
            return None
        stamp = max(p.stat().st_mtime for p in required)
        stale = (not out.is_file()) or out.stat().st_mtime < stamp
        if stale:
            # Generate to a temp path then rename atomically so a killed or
            # concurrent run cannot leave a truncated config with a fresh mtime.
            tmp = out.with_suffix(".tmp")
            cp = subprocess.run(
                [sys.executable, str(gen), "--checkpoint", str(model_dir),
                 "--out", str(tmp)], capture_output=True, text=True, timeout=300)
            if cp.returncode != 0:
                _LAST_CONFIG_ERROR = cp.stderr.strip()[-500:]
                tmp.unlink(missing_ok=True)
                return None
            os.replace(tmp, out)
        return out if out.is_file() else None
    except Exception as exc:
        _LAST_CONFIG_ERROR = str(exc)
        return None


class CppNemotronBackend(CppEngineBackend):
    """Nemotron chat framing (chat_template.jinja): <think></think>, no blank
    line, and <tool_response> blocks. Mirrors nemotron35.py:294-319 without
    importing that module (which would pull in MLX)."""

    @staticmethod
    def _assistant_prefix(enable_thinking: bool = True) -> str:
        im_start = "<|im_start|>"
        return (f"{im_start}assistant\n<think>\n" if enable_thinking
                else f"{im_start}assistant\n<think></think>")

    def append_tool_results(self, results, enable_thinking: bool = True) -> int:
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        head = f"{self._close()}{self._separator()}{im_start}user\n"
        tail = f"{im_end}\n{self._assistant_prefix(enable_thinking)}"
        body = "".join(f"<tool_response>\n{r}\n</tool_response>\n" for r in results)
        return self.append_text(f"{head}{body}{tail}")


def install_for_server(module):
    """Bind ``module.build_backend`` to the C++ factory.

    Called by ``veystrix.session.install_cpp_selector()`` only when
    ``VEYSTRIX_ENGINE=cpp``. ``module`` is the live session module (may be
    ``__main__`` when session.py is run as a script). The default python path
    never reaches here, so its behavior is unchanged. Passing the module
    explicitly avoids the ``sys.modules`` swapping the earlier version needed.
    """
    if getattr(module, "_veystrix_cpp_selector_installed", False):
        return

    def build_backend_cpp(name, args, prefs):
        cpp_model = _CPP_MODEL_FOR_NAME.get(str(name).strip().lower())
        if cpp_model is None:
            raise SystemExit(
                f"veystrix: no hand-written C++ runtime for {name!r} (only nemotron35-lightning has one; "
                f"flashnext and ornith run on the generic runtime)."
            )
        model_dir = getattr(args, "model_path", None) or os.environ.get("VEYSTRIX_MODEL_DIR")
        if not model_dir:
            raise SystemExit("veystrix: a model/checkpoint path is required for the C++ engine")
        # Decode defaults measured on 20 Nov 2026 (docs/performance/ENGINE_AB_2026-11-20.md, "C++ >= 1.2x
        # Python"): every one keeps the seeded text byte-identical. Set any of them to 0 to turn it off.
        #  * KEEPWARM: small GPU spin kernels while expert reads are in flight, so the GPU clock does not
        #    drop between layers (the Python engine's FLASHNEXT_GPU_KEEPWARM).
        #  * MLX_DEST: expert rows are read into MLX's own recycled Metal buffers instead of fresh
        #    malloc'd memory that MLX has to wrap (and fault in) as a new buffer every layer.
        #  * MLX_CACHE_BYTES: cap MLX's free-buffer cache at 256 MiB; the RAM it frees becomes page cache,
        #    which cut Qwen's physical SSD reads from ~640 to ~520 MB per token.
        os.environ.setdefault("VEYSTRIX_CPP_KEEPWARM", "1")
        os.environ.setdefault("VEYSTRIX_CPP_MLX_DEST", "1")
        os.environ.setdefault("VEYSTRIX_CPP_MLX_CACHE_BYTES", str(256 << 20))
        if cpp_model == "nemotron":
            # Token-by-token prefill costs ~0.21 s per prompt token (a 10K-token agent prompt would
            # take ~35 minutes). Real 2048-token chunks with per-layer eval are 37x faster, guard-clean
            # to 32K+ tokens with zero swap-out, and produce the same tokens as Python (logit digests
            # differ in the last bits: different summation order, see docs/performance/P6_FINAL_VERDICT.md).
            # VEYSTRIX_CPP_NEMOTRON_CHUNKED=0 restores the exact chunk-1 path.
            os.environ.setdefault("VEYSTRIX_CPP_NEMOTRON_CHUNKED", "1")
            # All six expert tensors of a MoE layer in one 8-lane read batch (was six serial batches),
            # and the shared expert launched on the GPU while those rows are read (as Python does).
            os.environ.setdefault("VEYSTRIX_CPP_NEMO_ONEBATCH", "1")
            os.environ.setdefault("VEYSTRIX_CPP_NEMO_SHARED_EARLY", "1")
            # Per-expert pipelined reads (see QWEN_PIPE above): ~11.5 -> ~14.3 tok/s, text
            # byte-identical. VEYSTRIX_CPP_NEMO_PIPE=0 restores the one-batch read.
            os.environ.setdefault("VEYSTRIX_CPP_NEMO_PIPE", "1")
            # Same page-cache warm-up for the next MoE layer's 2 highest-scored guesses. 13.54 -> 14.28
            # tok/s pooled. VEYSTRIX_CPP_NEMO_PREFETCH=0 turns it off.
            os.environ.setdefault("VEYSTRIX_CPP_NEMO_PREFETCH", "1")
            os.environ.setdefault("VEYSTRIX_CPP_NEMO_PREFETCH_TOP", "2")
            # GPU tail gate on the pipelined MoE path (as for Qwen above, depth 2 here): 13.04 -> 13.28 and
            # 13.22 -> 13.51 tok/s (n=8 and n=6, C ABI), float32 logit digests identical at every step. Depth 3
            # gained nothing on Nemotron (ENGINE_AB_2026-11-20.md, "Nemotron tail gate"). VEYSTRIX_CPP_GATE_TAIL=0
            # turns it off.
            os.environ.setdefault("VEYSTRIX_CPP_GATE_TAIL", "2")

        # The C runtime reads a FLAT config (tensor->shard map), not config.json.
        # Require the operator to supply the guarded Track-A config path rather
        # than passing "{}" that fails cryptically inside vx_engine_open.
        config_path = os.environ.get(CPP_CONFIG_PATH_ENV) or getattr(args, "cpp_config_path", None)
        if config_path and not Path(config_path).is_file():
            # An explicitly supplied path that does not exist is an operator
            # error: never silently substitute a different config (Sonnet #15).
            raise SystemExit(
                f"veystrix: {CPP_CONFIG_PATH_ENV}={config_path!r} is not a file; "
                f"fix the path or unset the variable to auto-derive from the checkpoint."
            )
        if not config_path:
            # Auto-derive the flat config from the checkpoint when the operator
            # did not supply one (fail closed if generation is impossible).
            auto = _auto_flat_config(model_dir, cpp_model)
            if auto is None:
                cause = globals().get("_LAST_CONFIG_ERROR", "")
                raise SystemExit(
                    f"veystrix: could not derive a flat config for {cpp_model!r} "
                    f"from {model_dir!r}"
                    + (f": {cause}" if cause else "")
                    + f". Set {CPP_CONFIG_PATH_ENV} to a guarded flat config "
                      f"(cpp/engine/tools/gen_nemotron_config.py)."
                )
            config_path = auto
        profile_json = json.dumps({"config_path": str(config_path), "model": cpp_model},
                                  sort_keys=True)

        # Stop tokens mirror the python backends so a turn ends at EOS/<|im_end|>
        # instead of running to max_tokens and falsely reporting finish="stop".
        stop_tokens: set[int] = set()
        try:
            tokenizer = _default_tokenizer(model_dir)
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None:
                if isinstance(eos, (list, tuple)):
                    stop_tokens.update(int(t) for t in eos)
                else:
                    stop_tokens.add(int(eos))
            convert = getattr(tokenizer, "convert_tokens_to_ids", None)
            if convert is not None:
                markers = ("<|im_end|>", "<|endoftext|>")
                if cpp_model == "nemotron":
                    markers = ("<|im_end|>", "<|endoftext|>", "<|eos|>", "<|end|>")
                for marker in markers:
                    tid = convert(marker)
                    if tid is not None:
                        stop_tokens.add(int(tid))
            # Nemotron's config.json carries eos_token_id (can be a list).
            if cpp_model == "nemotron":
                cfg_path = Path(model_dir) / "config.json"
                if cfg_path.is_file():
                    cfg = json.loads(cfg_path.read_text())
                    eos_ids = cfg.get("eos_token_id")
                    if eos_ids is not None:
                        if isinstance(eos_ids, (list, tuple)):
                            stop_tokens.update(int(t) for t in eos_ids)
                        else:
                            stop_tokens.add(int(eos_ids))
        except Exception:  # tokenizer is optional in offline tests
            pass
        if not stop_tokens:
            raise SystemExit(
                "veystrix: could not determine stop tokens for the C++ engine "
                "from the checkpoint tokenizer; refusing to generate unbounded."
            )

        backend_cls = CppNemotronBackend if cpp_model == "nemotron" else CppEngineBackend
        backend = backend_cls(
            model_dir=model_dir,
            profile_json=profile_json,
            tokenizer=tokenizer,
            session_dir=getattr(args, "session_dir", None),
            stop_tokens=stop_tokens,
        )
        # Mirror preference state (answer budget, sampling, thinking) like the
        # python backends. Sampling now reaches the C++ runtime; preferences
        # also drive /settings.
        if prefs:
            from veystrix.session import mirror_preferences
            mirror_preferences(backend, prefs, prefs.get("profile", "default"))
        return backend

    module.build_backend = build_backend_cpp
    module._veystrix_cpp_selector_installed = True
