"""Engine selector and bundle integrity checks.

The default engine is ``python``. ``cpp`` is a clean refusal until an installed
bundle manifest records ``cpp_parity: pass`` for the selected model and profile.

Explicit cpp manifest contract (source of truth shared with the bundle build
script and the M6 docs, ``docs/engineering/m6_cpp_toggle.md``):

* ``engines`` must list ``"cpp"``, else the bundle has no C++ engine at all.
* ``cpp_parity`` must record ``pass`` for *both* proven models (Track A proved
  bit-exact Qwen + Nemotron on one driver thread): ``flashnext`` (Qwen) and
  ``nemotron35-lightning`` (Nemotron), using the short model names that the
  terminal tool actually passes to :func:`check_engine` (the ``--model`` choices
  in ``veystrix/tool/cli.py`` and config values) -- NOT the ``veystrix-*``
  OpenAI-facing model IDs. The ONLY accepted shape is the canonical flat key
  ``f"{model}:{profile}": "pass"`` (profile ``agent`` for the shipped bundle);
  looser shapes were removed so one entry cannot authorize every profile. The
  gate is per model+profile: a missing or non-``pass`` entry refuses cpp only for
  the model/profile selected, naming both. :func:`check_engine` resolves the
  active profile from saved preferences unless one is passed explicitly.
* A python-only bundle keeps ``cpp_parity`` empty, so cpp stays a clean refusal
  and the default python path is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ENGINE_CHOICES = ("python", "cpp")
DEFAULT_ENGINE = "python"
KNOWN_PROTOCOL_VERSION = "1"        # bump only together with version_payload() in engine_cli

# The two models a C++ bundle must prove (Track A). Keys are the short names
# check_engine() receives; profile is the default launch profile.
CPP_PROVEN_MODELS = ("inkling2-mini",)
CPP_DEFAULT_PROFILE = "agent"


def cpp_parity_key(model: str, profile: str = CPP_DEFAULT_PROFILE) -> str:
    """Canonical manifest key for a model/profile parity entry."""
    return f"{model}:{profile}"


def resolve_engine(explicit: str | None, environ=None, config=None) -> str:
    """Precedence: --engine, VEYSTRIX_ENGINE, veystrix config, default python."""
    environ = os.environ if environ is None else environ
    raw = explicit or environ.get("VEYSTRIX_ENGINE") or (config or {}).get("engine") or DEFAULT_ENGINE
    value = str(raw).strip().lower()
    if value not in ENGINE_CHOICES:
        raise ValueError(f"unknown engine {raw!r}; use: {', '.join(ENGINE_CHOICES)}")
    return value


SOURCE_ENV = "VEYSTRIX_SOURCE_CPP"


def source_root() -> Path:
    """Repository root when the tool runs from a checkout (not a packaged bundle)."""
    return Path(__file__).resolve().parents[2]


def source_cpp_dylib() -> Path | None:
    """The checkout's own C++ engine build, if one exists."""
    root = source_root()
    if not (root / "cpp" / "engine" / "CMakeLists.txt").is_file():
        return None
    # The live (MLX-linked) library; build/capi holds the CPU-only stub and is not a serving engine.
    for directory in ("build", "build-live"):
        for name in ("libveystrix_c.dylib", "libveystrix_c.so"):
            candidate = root / "cpp" / "engine" / directory / name
            if candidate.is_file():
                return candidate
    return None


def source_mode_enabled(environ=None, home: Path | None = None) -> bool:
    """Source-checkout C++ is for the developer's own machine.

    ``VEYSTRIX_SOURCE_CPP=1/0`` forces it on/off. By default it is on only for
    the real user home: a redirected ``VEYSTRIX_HOME`` (tests, sandboxes) keeps the
    strict bundle-only contract unless the variable says otherwise.
    """
    environ = os.environ if environ is None else environ
    flag = str(environ.get(SOURCE_ENV, "")).strip()
    if flag in ("0", "1"):
        return flag == "1"
    if environ.get("VEYSTRIX_HOME"):
        return False
    return home is None or Path(home).expanduser() == Path.home()


def bundle_installed(home: Path) -> bool:
    return manifest_path(home).is_file()


def source_cpp_ready(home: Path, environ=None) -> bool:
    """True when cpp can run from this checkout: no bundle installed + a built dylib."""
    return (source_mode_enabled(environ, home) and not bundle_installed(home)
            and source_cpp_dylib() is not None)


def bundle_root(home: Path) -> Path:
    return home / ".veystrix" / "engine"


def manifest_path(home: Path) -> Path:
    return bundle_root(home) / "BUNDLE-MANIFEST.json"


def load_manifest(home: Path) -> dict:
    try:
        data = json.loads(manifest_path(home).read_text())
    except (OSError, TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _parity_value(manifest: dict, model: str, profile: str) -> str:
    """Read the recorded parity status for one model/profile.

    Only the canonical ``f"{model}:{profile}": "pass"`` shape is accepted. The
    looser ``{model: "pass"}`` / ``{model: {profile: ...}}`` shapes were dropped:
    they let one entry authorize every profile for a model, weakening the
    per-model+profile gate. The bundle emits only the canonical key.
    Missing/malformed data never falls through to a pass.
    """
    table = manifest.get("cpp_parity")
    if not isinstance(table, dict):
        return "absent"
    entry = table.get(cpp_parity_key(model, profile))
    if isinstance(entry, dict):
        entry = entry.get("status", entry.get("result"))
    return str(entry).strip().lower() if entry is not None else "absent"


def cpp_refusal(manifest: dict, model: str, profile: str = CPP_DEFAULT_PROFILE) -> str | None:
    """None means cpp may be used; otherwise a clean, model-specific refusal."""
    engines = manifest.get("engines")
    if not isinstance(engines, list) or "cpp" not in [str(e).lower() for e in engines]:
        return ("--engine cpp is not available: the installed bundle has no C++ engine. "
                "Use --engine python (the default).")
    if _parity_value(manifest, model, profile) != "pass":
        return (f"--engine cpp is not available: bundle manifest does not record "
                f"cpp_parity: pass for model {model!r}, profile {profile!r}. Use --engine python.")
    return None


def check_engine(engine: str, home: Path, model: str, profile: str = CPP_DEFAULT_PROFILE) -> None:
    """Raise SystemExit with a clean refusal if the selected engine cannot run.

    ``profile`` is the cpp parity profile to gate; callers thread the actual
    launch profile (StartOptions.engine_profile) so the gate matches what the
    engine will use. It defaults to ``agent`` — the only profile Track A proved
    and the only key the bundle manifest records. Preferences profiles
    (``plain``/``agent``) are a separate chat-mode concept; cpp parity is keyed
    by the engine profile, so we must NOT read the chat profile here (that would
    gate ``plain`` users against a nonexistent key and break cpp entirely).
    """
    from veystrix import model_identity
    identity = model_identity.IDENTITIES.get(str(model or ""))
    if identity is not None and engine not in identity.engines:
        other = " or ".join(identity.engines)
        raise SystemExit(f"veystrix: {identity.display_name} has no {engine} engine; use --engine {other} "
                         f"(veystrix engine {identity.engines[0]}).")
    if engine == "python":
        return
    chosen = str(profile).strip().lower() or CPP_DEFAULT_PROFILE
    generic = identity is not None and identity.cpp_runtime == "generic"
    if source_cpp_ready(home):
        if generic:
            refusal = generic_parity_refusal(home, identity)
            if refusal:
                raise SystemExit("veystrix: " + refusal)
        return          # developer checkout with its own live build (the engine verifies mode=live)
    refusal = cpp_refusal(load_manifest(home), model, chosen)
    if refusal and identity is not None and "python" not in identity.engines:
        refusal = (refusal.replace(" Use --engine python (the default).", "").replace(" Use --engine python.", "")
                   + f" {identity.display_name} has no Python engine, so it cannot run until a C++ engine with "
                     f"a passing parity record is available.")
    if refusal:
        if source_mode_enabled() and not bundle_installed(home) and source_cpp_dylib() is None:
            refusal += (" To use the C++ engine from this checkout, build it: "
                        "cmake -S cpp/engine -B cpp/engine/build -DVEYSTRIX_BUILD_CAPI=ON "
                        "&& cmake --build cpp/engine/build -j")
        raise SystemExit("veystrix: " + refusal)


def parity_record_path(home: Path) -> Path:
    """Where ``veystrix models port-check`` records a passing generic-runtime conformance run."""
    return Path(home) / ".veystrix" / "parity" / "generic.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def generic_parity_refusal(home: Path, identity, lib: Path | None = None) -> str | None:
    """None when the current C++ build passed the port-check for this family's model_type.

    The record names the dylib's SHA-256, so a rebuilt engine must be port-checked again."""
    lib = lib or source_cpp_dylib()
    if lib is None:
        return "the C++ engine is not built"
    try:
        record = json.loads(parity_record_path(home).read_text())
    except (OSError, ValueError):
        record = {}
    entry = record.get(identity.model_type) if isinstance(record, dict) else None
    how = "run: veystrix models port-check   (offline, tiny synthetic model, seconds)"
    if not isinstance(entry, dict) or entry.get("status") != "pass":
        return (f"--engine cpp is not available for {identity.display_name}: no passing generic-runtime "
                f"port-check for {identity.model_type} is recorded; {how}")
    if entry.get("lib_sha256") != file_sha256(lib):
        return (f"--engine cpp is not available for {identity.display_name}: the C++ engine was rebuilt "
                f"since the last passing port-check; {how}")
    return None


def bundle_integrity(home: Path, *, read=None) -> list[str]:
    """Verify the installed bundle's manifest and file digests. Read-only, no model."""
    root = bundle_root(home)
    path = manifest_path(home)
    read = read or (lambda p: p.read_bytes())
    problems: list[str] = []
    try:
        raw = read(path)
        manifest = json.loads(raw)
    except (OSError, ValueError, TypeError):
        return ["bundle manifest is missing or unreadable: " + str(path)]
    if not isinstance(manifest, dict):
        return ["bundle manifest is not a JSON object: " + str(path)]
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        return ["bundle manifest 'files' is not an object"]
    for relative, expected in sorted(files.items()):
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            problems.append(f"unsafe path in bundle manifest: {relative!r}")
            continue
        try:
            actual = hashlib.sha256(read(root / relative)).hexdigest()
        except OSError:
            problems.append(f"missing bundle file: {relative}")
            continue
        if str(expected).lower() != actual:
            problems.append(f"checksum mismatch for bundle file: {relative}")
    return problems


def engine_version(home: Path) -> dict:
    """The installed bundle's machine-readable manifest (empty when running from source)."""
    try:
        data = json.loads((bundle_root(home) / "BUNDLE-MANIFEST.json").read_text())
    except (OSError, ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def check_protocol(home: Path) -> None:
    """Refuse an engine whose protocol version the front end does not know."""
    version = engine_version(home).get("protocol_version")
    if version is None:
        raise SystemExit("veystrix: installed bundle has no protocol_version in BUNDLE-MANIFEST.json")
    if str(version) != KNOWN_PROTOCOL_VERSION:
        raise SystemExit(f"veystrix: engine protocol {version} is not supported (front end knows "
                         f"{KNOWN_PROTOCOL_VERSION}); update the terminal tool")


def report_engine(health: dict | None, state_engine: str) -> str:
    """The active engine reported by /health wins; fall back to our own state."""
    if isinstance(health, dict):
        value = health.get("engine")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return state_engine
