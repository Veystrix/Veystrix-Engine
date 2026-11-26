"""Verify that a built app still has every protection guardrail.

Run after build_app.sh / make_dmg.sh. This is deliberately about the built
artifact, not the source tree: a future change that drops encryption, the
shield, integrity manifest or signing is caught here even if unit tests pass.
"""
from __future__ import annotations

import hashlib
import marshal
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dist" / "Veystrix.app"
RESOURCES = APP / "Contents" / "Resources"
LIB = RESOURCES / "engine" / "lib"
PYTHON = RESOURCES / "engine" / "python" / "bin" / "python3"
BOOTSTRAP = {"_protect.py", "sitecustomize.py"}

results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((bool(ok), name, detail))


def main() -> int:
    check("app bundle exists", APP.is_dir(), str(APP))
    if not APP.is_dir():
        return report()

    engine = RESOURCES / "engine"
    check("engine lib exists", LIB.is_dir(), str(LIB))
    encrypted = sorted(LIB.rglob("*.pyc.enc"))
    check("bytecode encrypted (209 modules)", len(encrypted) >= 100, f"{len(encrypted)} encrypted modules")
    plaintext = sorted(LIB.rglob("*.pyc"))
    check("no plaintext bytecode", not plaintext, f"{len(plaintext)} found")
    source = [p for p in LIB.rglob("*.py") if p.name not in BOOTSTRAP]
    check("no product source in bundle", not source, f"{len(source)} unexpected .py files")
    c_source = sorted(LIB.rglob("*.c"))
    check("no C source in bundle", not c_source, f"{len(c_source)} found")
    check("bootstrap files are the only readable code",
          all((LIB / name).is_file() for name in BOOTSTRAP))
    check("no bytecode caches in bundle", not list(LIB.rglob("__pycache__")))
    check("shield dylib present", (LIB / "_protect_key.dylib").is_file())
    check("precompiled native reader present",
          (LIB / "models" / "flashnext" / "native_read.bundled.dylib").is_file())

    manifest = RESOURCES / "BundleIntegrity.json"
    if manifest.is_file():
        import json
        data = json.loads(manifest.read_text())
        pinned = data.get("files", {})
        check("integrity manifest pins files", len(pinned) > 100, f"{len(pinned)} files pinned")
        # Recompute exactly as the app does, so a manifest that disagrees with
        # the signed bundle fails here rather than blocking launch on a user Mac.
        prefixes = ("engine/lib/", "Info.plist", "VERSION", "LICENSE.txt", "THIRD_PARTY_NOTICES.md")
        excluded_names = {"BundleIntegrity.json", "BUILD-CHANNEL"}
        excluded_suffixes = (".dylib", ".so")
        actual: dict[str, str] = {}
        for path in RESOURCES.rglob("*"):
            rel = path.relative_to(RESOURCES).as_posix()
            if rel in excluded_names or rel.endswith(excluded_suffixes):
                continue
            if not any(rel == p or rel.startswith(p) for p in prefixes):
                continue
            if path.is_file() and not path.is_symlink():
                actual[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        mismatched = [name for name, digest in actual.items() if pinned.get(name) != digest]
        unexpected = [name for name in actual if name not in pinned]
        check("integrity manifest matches signed bundle", not mismatched and not unexpected,
              f"{len(mismatched)} changed, {len(unexpected)} extra: {(mismatched + unexpected)[:4]}")
    else:
        check("integrity manifest exists", False)

    if encrypted:
        try:
            marshal.loads(encrypted[0].read_bytes()[16:])
            decompiled = True
        except Exception:
            decompiled = False
        check("naive decompile fails", not decompiled)

    if PYTHON.is_file() and LIB.is_dir():
        env = dict(os.environ, PYTHONPATH=str(LIB), PYTHONDONTWRITEBYTECODE="1")
        run = subprocess.run(
            [str(PYTHON), "-c",
             "from veystrix import model_identity as m; print(m.FLASHNEXT.context_limit)"],
            env=env, capture_output=True, text=True, timeout=90,
        )
        check("bundled engine loads cleanly", run.returncode == 0 and "262144" in run.stdout,
              (run.stderr or run.stdout)[-160:])

    signed = subprocess.run(["codesign", "-v", str(APP)], capture_output=True, text=True)
    check("code signature valid", signed.returncode == 0, signed.stderr.strip()[-160:])

    import re
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    dmg = ROOT / "dist" / f"Veystrix-{version}-arm64.dmg"
    check(f"DMG for version {version} exists", dmg.is_file(), str(dmg))
    if dmg.is_file():
        digest = hashlib.sha256(dmg.read_bytes()).hexdigest()
        print(f"DMG SHA-256: {digest}")

    # Source-level guardrail: the key must still be bound to runtime signals.
    builder = (ROOT / "packaging" / "engine" / "protect_engine.py").read_text()
    check("key still bound to shield digest", "derive(base, clean_signals(lib))" in builder)
    return report()


def report() -> int:
    for ok, name, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    failed = [name for ok, name, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} guardrails passed")
    if failed:
        print("FAILED: " + ", ".join(failed), file=sys.stderr)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
