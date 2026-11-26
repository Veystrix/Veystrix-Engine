"""Build-time bytecode protection with a runtime-derived key.

The encryption key is not stored directly. The compiled shield holds
``base XOR sha256(clean_signals)`` and recomputes the signal digest at runtime.
On a clean run the XOR cancels and the real key appears; a patched check or
modified bootstrap changes a signal and decryption fails with bad marshal data.
This is a cost-raiser, not unbreakable DRM: the dylib still runs on the user's Mac.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _protect import encrypt  # noqa: E402

SIGNAL_COUNT = 5
BOOTSTRAP = ("_protect.py", "sitecustomize.py")


def bootstrap_digest(lib: Path) -> bytes:
    digest = hashlib.sha256()
    for name in BOOTSTRAP:
        digest.update((lib / name).read_bytes())
    return digest.digest()


def clean_signals(lib: Path) -> bytes:
    """The exact digest input expected on an unmodified, untraced run."""
    return bytes(SIGNAL_COUNT) + bootstrap_digest(lib)


def derive(base: bytes, signals: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(base, hashlib.sha256(signals).digest()))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: protect_engine.py <engine-lib-dir>", file=sys.stderr)
        return 2
    lib = Path(sys.argv[1]).resolve()
    if not lib.is_dir():
        print(f"not a directory: {lib}", file=sys.stderr)
        return 2

    base = os.urandom(32)
    work = lib.parent / ".protect-build"
    work.mkdir(exist_ok=True)

    shutil.copy(HERE / "_protect.py", lib / "_protect.py")
    (lib / "sitecustomize.py").write_text("import _protect\n_protect.install()\n")

    count = 0
    for pyc in sorted(lib.rglob("*.pyc")):
        pyc.with_suffix(".pyc.enc").write_bytes(encrypt(pyc.read_bytes(), base))
        pyc.unlink()
        count += 1

    derived = derive(base, clean_signals(lib))
    config = work / "shield_config.h"
    config.write_text(
        "#ifndef VEYSTRIX_SHIELD_CONFIG_H\n#define VEYSTRIX_SHIELD_CONFIG_H\n"
        "static const unsigned char vx_derived_key[32] = {" +
        ",".join(str(b) for b in derived) + "};\n#endif\n"
    )
    dylib = lib / "_protect_key.dylib"
    subprocess.run(
        ["xcrun", "clang", "-O2", "-dynamiclib", "-arch", "arm64",
         # Relative install name: never bake the build machine's absolute path into the dylib.
         "-install_name", "@rpath/_protect_key.dylib",
         "-I", str(work), str(HERE / "shield.c"), "-o", str(dylib)],
        check=True, capture_output=True, text=True,
    )
    shutil.rmtree(work)

    # Build-time self test: a clean process must derive the real key.
    probe = ctypes.CDLL(str(dylib))
    probe.vx_shield_key.restype = ctypes.c_void_p
    effective = ctypes.string_at(probe.vx_shield_key(), 32)
    if effective != base:
        print("ERROR: shield derived the wrong key on a clean run", file=sys.stderr)
        return 1

    for cache in sorted(lib.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache)
    remaining = list(lib.rglob("*.pyc"))
    if remaining:
        print(f"ERROR: unencrypted bytecode remains: {remaining[:3]}", file=sys.stderr)
        return 1
    print(f"protected {count} modules; key bound to runtime shield digest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
