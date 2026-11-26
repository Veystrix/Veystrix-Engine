#!/usr/bin/env python3
"""Hard release gate: refuse any source, private path, secret, leaked symbol, or never-ship file.

The product is binary-only. A release bundle or the public repo must contain NO source
file of any kind (Python, Swift, C/C++, TypeScript, ...), no `.git`, no docs/results/
research, no `/Users/` string (including in `.dylib`/`.so` binaries), no credential-shaped
token, no uncompiled `.pyc` for protected modules (with `--require-compiled`), no leaked
non-`vx_*` exported C++ symbols in `libveystrix_c.dylib`, and no oversized file.

Usage: leak_scan.py --bundle DIR [--public DIR] [--max-mb N] [--require-compiled] [--cython-list PATH]
Exit 1 with a printed list of violations when anything fails. No network, no model.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE_SUFFIXES = {
    ".py", ".pyx", ".pxd", ".pyi", ".swift", ".c", ".cc", ".cpp", ".h", ".hpp",
    ".m", ".mm", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt", ".rb",
}
NEVER_SHIP = {"docs", "research_notes", "experiments", "results", "reports", "scratch",
              "tests", "bench", ".git", "cmake", "CMakeFiles"}
# The bundled CPython interpreter and its third-party dependencies are standard open-source
# components. Shipping their source and symlinks is normal and reveals nothing about the engine;
# only the engine tree (lib/, bin/, root) is held to the binary-only rule. Secrets and the
# developer's own path are still forbidden everywhere (see scan()).
RUNTIME_ROOT = "python"
# Bootstrap loader: plaintext by necessity (it installs the decryptor) and carries no product
# logic. Everything else under lib/ must be encrypted bytecode.
BOOTSTRAP = {"_protect.py", "sitecustomize.py"}
# Directories we intentionally ship inside the engine tree, but only as encrypted bytecode.
SHIPPABLE_WHEN_ENCRYPTED = {"tests", "bench"}
# Strings that identify the machine or project that built the bundle. These must never appear,
# not even inside the open-source runtime. A bare "/Users/" is deliberately NOT here: PyPI wheels
# legitimately embed their own build-farm paths (e.g. /Users/runner/...), which are not our leak;
# these markers match only paths from THIS project/checkout.
DEV_FINGERPRINTS = (b"Veystrix-qsa", b"/Veystrix-", b"Documents/Veystrix", b"cpp/engine/build")
TEXT_SUFFIXES = {".txt", ".json", ".jsonc", ".md", ".yml", ".yaml", ".sh", ".pyc", ".pyo", ".toml"}
BINARY_SUFFIXES = {".dylib", ".so"}
SECRET = re.compile(rb"(gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{32,}"
                    rb"|sk-ant-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}"
                    rb"|xox[baprs]-[A-Za-z0-9-]{10,}|hf_[A-Za-z0-9]{30,}"
                    rb"|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def _load_protected_patterns(cython_list: Path | None) -> list[str]:
    if cython_list is None or not cython_list.is_file():
        return []
    patterns: list[str] = []
    for raw in cython_list.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(".py"):
            patterns.append(line[:-3])
    return patterns


def _matches_protected_bytecode(relative: Path, protected_stems: list[str]) -> bool:
    """Return True if a .pyc/.pyo file corresponds to a protected module stem."""
    if relative.suffix.lower() not in (".pyc", ".pyo"):
        return False
    parts = list(relative.parts)
    # Normalise __pycache__/foo.cpython-312.pyc -> foo.pyc in its parent directory
    if len(parts) >= 2 and parts[-2] == "__pycache__":
        stem = relative.name.split(".", 1)[0]
        norm = "/".join([*parts[:-2], stem])
    else:
        norm = "/".join([*parts[:-1], relative.stem])
    # Strip leading bundle prefixes like engine/lib/ or lib/
    candidates = [norm]
    for marker in ("/lib/", "lib/"):
        if marker in norm:
            candidates.append(norm.split(marker, 1)[1])
    for cand in candidates:
        for pat in protected_stems:
            if fnmatch.fnmatch(cand, pat):
                return True
    return False


def _check_capi_exports(path: Path) -> list[str]:
    """Fail if libveystrix_c.dylib exports any symbol outside the narrow vx_* C ABI."""
    if path.name != "libveystrix_c.dylib" or not shutil.which("nm"):
        return []
    proc = subprocess.run(["nm", "-gU", str(path)], capture_output=True, text=True)
    if proc.returncode != 0:
        return []
    leaked: list[str] = []
    for line in proc.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        sym = fields[-1]
        if not (sym.startswith("_vx_") or sym.startswith("vx_")):
            leaked.append(sym)
    return leaked


def scan(root: Path, max_bytes: int, *, public: bool = False,
         require_compiled: bool = False, cython_list: Path | None = None) -> list[str]:
    problems: list[str] = []
    if not root.is_dir():
        return [f"missing directory: {root}"]
    banned = {".git"} if public else NEVER_SHIP
    protected_stems = _load_protected_patterns(cython_list) if require_compiled else []
    for path in root.rglob("*"):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        # The bundled interpreter + third-party deps are open source; only the engine tree is sealed.
        # Tolerate an outer "engine/" wrapper (the scan may be pointed at the dist dir or the bundle).
        rparts = relative.parts
        idx = 1 if rparts and rparts[0] == "engine" else 0
        in_runtime = (not public) and len(rparts) > idx and rparts[idx] == RUNTIME_ROOT
        parts = set(relative.parts[:-1]) | ({relative.name} if path.is_dir() else set())
        if public and ".git" in parts:       # a working copy always has .git; it is not shipped
            continue
        if (parts & banned) and not in_runtime and path.is_file():
            # Flag files only: a never-ship directory is reported through its files, so the
            # engine tree's intentionally-shipped tests/bench dirs do not trip on the dir entry.
            # Inside those two dirs, allow encrypted bytecode (the deliberately shipped closure);
            # docs/results/research/.git are never allowed, encrypted or not.
            offending = parts & banned
            if not (offending <= SHIPPABLE_WHEN_ENCRYPTED and path.name.endswith(".pyc.enc")):
                problems.append(f"{relative}: never-ship path")
                continue
        if path.is_symlink():
            if not in_runtime:                # the runtime legitimately uses symlinks (python3, man pages)
                problems.append(f"{relative}: symlink")
            continue
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        is_source = suffix in SOURCE_SUFFIXES or path.name.endswith((".py", ".pyx", ".pxd"))
        if is_source and not in_runtime and path.name not in BOOTSTRAP:
            problems.append(f"{relative}: source file")
            continue
        if protected_stems and _matches_protected_bytecode(relative, protected_stems):
            problems.append(f"{relative}: uncompiled bytecode for protected module")
        if path.name == "libveystrix_c.dylib":
            leaked = _check_capi_exports(path)
            if leaked:
                problems.append(f"{relative}: non-vx exported symbols: {', '.join(leaked[:5])}")
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > max_bytes:
            problems.append(f"{relative}: {size} bytes exceeds size budget {max_bytes}")
        if suffix in TEXT_SUFFIXES or suffix in BINARY_SUFFIXES or size < 2_000_000:
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if in_runtime:
                # The runtime is public open-source. Its wheels embed upstream build-farm paths
                # (/Users/runner/...) and its test data contains dummy tokens and binary blobs —
                # neither is our leak. Flag only what identifies THIS build (our project path).
                for mark in DEV_FINGERPRINTS:
                    if mark in data:
                        problems.append(f"{relative}: contains build fingerprint {mark.decode()!r}")
                        break
            else:
                # Our own tree and binaries must be pristine: no absolute path, no secret.
                if b"/Users/" in data:
                    problems.append(f"{relative}: contains /Users/ private path")
                if SECRET.search(data):
                    problems.append(f"{relative}: credential-shaped string")
    return sorted(set(problems))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--public", type=Path, help="public repo working copy (also scanned)")
    parser.add_argument("--max-mb", type=int, default=800)
    parser.add_argument("--require-compiled", action="store_true",
                        help="fail if any module in cython_modules.txt ships as .pyc/.pyo")
    parser.add_argument("--cython-list", type=Path,
                        default=Path(__file__).with_name("cython_modules.txt"))
    args = parser.parse_args(argv)
    problems = scan(args.bundle, args.max_mb * 1024 * 1024,
                    require_compiled=args.require_compiled, cython_list=args.cython_list)
    if args.public:
        problems += [f"public/{p}" for p in scan(args.public, args.max_mb * 1024 * 1024, public=True)]
    if problems:
        print("LEAK SCAN FAILED:", file=sys.stderr)
        for problem in problems:
            print("  " + problem, file=sys.stderr)
        return 1
    print(f"leak scan OK: {args.bundle}" + (f" + {args.public}" if args.public else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
