"""Build-time bytecode protection for the packaged engine.

Generates a per-build key dylib, installs a tiny import bootstrap, and encrypts
every compiled module so plain-text decompilers see ciphertext. This is a
cost-raiser, not unbreakable protection; the key ships in native code.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _protect import encrypt  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: protect_engine.py <engine-lib-dir>", file=sys.stderr)
        return 2
    lib = Path(sys.argv[1]).resolve()
    if not lib.is_dir():
        print(f"not a directory: {lib}", file=sys.stderr)
        return 2

    secret = os.urandom(32)
    work = lib.parent / ".protect-build"
    work.mkdir(exist_ok=True)
    source = work / "protect_key.c"
    source.write_text(
        "#include <stddef.h>\n"
        "static const unsigned char key[32] = {" +
        ",".join(str(byte) for byte in secret) + "};\n"
        "const unsigned char *vx_protect_key(void) { return key; }\n"
        "int vx_protect_key_length(void) { return 32; }\n"
    )
    dylib = lib / "_protect_key.dylib"
    subprocess.run(
        ["xcrun", "clang", "-O2", "-dynamiclib", "-arch", "arm64", str(source), "-o", str(dylib)],
        check=True, capture_output=True, text=True,
    )
    shutil.copy(HERE / "_protect.py", lib / "_protect.py")
    (lib / "sitecustomize.py").write_text(
        "import _protect\n_protect.install()\n"
    )
    shutil.rmtree(work)

    count = 0
    for pyc in sorted(lib.rglob("*.pyc")):
        encrypted = pyc.with_suffix(".pyc.enc")
        encrypted.write_bytes(encrypt(pyc.read_bytes(), secret))
        pyc.unlink()
        count += 1
    remaining = list(lib.rglob("*.pyc"))
    if remaining:
        print(f"ERROR: unencrypted bytecode remains: {remaining[:3]}", file=sys.stderr)
        return 1
    print(f"protected {count} modules; key compiled into {dylib.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
