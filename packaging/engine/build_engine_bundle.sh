#!/bin/bash
# Build the self-contained engine runtime shipped with the binary-only Veystrix terminal tool.
#   packaging/engine/build_engine_bundle.sh [OUT_DIR]        (default: dist/engine)
# Layout:  OUT/python/   pinned CPython (python-build-standalone) with the engine's pinned dependencies
#          OUT/lib/      veystrix/, models/{flashnext,nemotron35}/ as compiled .pyc, plus data files
#          OUT/licenses/ license texts for everything inside
#          OUT/BUNDLE-MANIFEST.txt
# Honest limits: .pyc files are an anti-clutter and anti-accident measure, NOT protection. They decompile
# easily; do not describe the bundle as obfuscated or protected. Dependencies are downloaded from PyPI with
# pinned versions (network needed, no model weights). Nothing here loads or imports a model.
# Excluded on purpose: opencv (LGPL FFmpeg build; see legal/COMMERCIALIZATION_READINESS.md), tests, docs,
# benchmarks, and every checkpoint.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${1:-$ROOT/dist/engine}"; WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
mkdir -p "$(dirname "$OUT")"; OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"   # absolute: later steps cd
source "$HERE/runtime.pin"
[[ "$(uname -m)" == "arm64" ]] || { echo "ERROR: Apple silicon build host required" >&2; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT/lib" "$OUT/licenses"

echo "==> Python runtime $PBS_FILE"
CACHE="${VEYSTRIX_DOWNLOAD_CACHE:-$ROOT/dist/downloads}"; mkdir -p "$CACHE"
if [[ ! -f "$CACHE/$PBS_FILE" ]]; then
  curl -fsSL -o "$CACHE/$PBS_FILE" "https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_TAG/${PBS_FILE//+/%2B}"
fi
echo "$PBS_SHA256  $CACHE/$PBS_FILE" | shasum -a 256 -c - || { echo "ERROR: runtime checksum mismatch" >&2; rm -f "$CACHE/$PBS_FILE"; exit 1; }
tar -xzf "$CACHE/$PBS_FILE" -C "$WORK" && mv "$WORK/python" "$OUT/python"
PY="$OUT/python/bin/python3"

echo "==> Pinned dependencies (opencv excluded)"
"$PY" - "$ROOT/pyproject.toml" > "$WORK/requirements.txt" <<'PYEOF'
import sys, tomllib
deps = tomllib.load(open(sys.argv[1], "rb"))["project"]["dependencies"]
bad = [d for d in deps if "==" not in d]
if bad: sys.exit(f"unpinned dependency in pyproject.toml: {bad}")
print("\n".join(d for d in deps if not d.lower().startswith(("opencv", "cv2"))))
PYEOF
"$PY" -m pip install --no-cache-dir --disable-pip-version-check --only-binary=:all: -r "$WORK/requirements.txt"
# opencv arrives transitively (mlx-vlm). Its wheels carry an LGPL FFmpeg build, so it is removed from the bundle and
# the smoke test asserts it is gone. Features that need it (video input) are unavailable in the packaged app.
"$PY" -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python >/dev/null 2>&1 || true
SP="$OUT/python/lib/python3.12/site-packages"
rm -rf "$SP/cv2"
# Text-only engine: mlx-vlm pulls in an audio stack (mlx-audio -> scipy, sounddevice, miniaudio)
# that no shipped runtime module imports. Drop it here; when adding a model that needs audio or
# SciPy, add its package back to this list and to smoke_import.py's import closure.
"$PY" -m pip uninstall -y scipy mlx-audio sounddevice miniaudio >/dev/null 2>&1 || true
rm -rf "$SP"/{scipy,mlx_audio,_sounddevice.py,_sounddevice_data,miniaudio.py,_miniaudio.abi3.so} \
       "$SP"/{scipy-*,mlx_audio-*,sounddevice-*,miniaudio-*}.dist-info 2>/dev/null || true
# C/Python headers and cmake files are build-time only; the compiled extensions link the dylibs.
rm -rf "$SP/mlx/include" "$SP/mlx/lib/cmake"
# Bytecode caches are only regenerated when PYTHONDONTWRITEBYTECODE is unset. LaunchPlan and the
# smoke test both set it, so the caches shipped by pip are dead weight in the bundle.
find "$SP" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
# The engine never opens a window: drop Tcl/Tk, IDLE and the stdlib test suite (smaller bundle, fewer files to sign).
rm -rf "$OUT"/python/lib/python3.12/{test,idlelib,tkinter,turtledemo,turtle.py} "$OUT"/python/lib/python3.12/lib-dynload/_tkinter*.so \
       "$OUT"/python/lib/{tcl*,tk*,itcl*,thread*,Tix*,sqlite3*} "$OUT"/python/bin/{idle*,2to3*,pydoc*} 2>/dev/null || true

echo "==> Engine sources -> compiled lib"
for pkg in veystrix; do
  mkdir -p "$OUT/lib/$(dirname "$pkg")" 2>/dev/null || true
  ditto "$ROOT/$pkg" "$OUT/lib/$pkg"
done
find "$OUT/lib" \( -name tests -o -name bench -o -name __pycache__ -o -name '*.md' \) -prune -exec rm -rf {} +
# The engine imports its telemetry and memory sampler at module load (veystrix.bench.telemetry is a no-op
# unless a benchmark activates it). Ship exactly that import closure, not the qualification harnesses.
mkdir -p "$OUT/lib/veystrix/bench"
cp "$ROOT/veystrix/bench/"{__init__.py,telemetry.py,memory.py,provenance.py} "$OUT/lib/veystrix/bench/"
# Compile next to the source (-b = legacy .pyc location, importable without the .py), then drop every .py file.
# -s "$OUT" strips the build path from the stored module filename, so no /Users/<name>/... leaks via co_filename.
"$PY" -m compileall -q -b -s "$OUT" "$OUT/lib"
find "$OUT/lib" -name '*.py' -delete
find "$OUT/lib" -name '*.c' -delete
[[ -z "$(find "$OUT/lib" -type f \( -name '*.py' -o -name '*.c' \) -print -quit)" ]] || { echo "ERROR: source files remain in bundle"; exit 1; }

# Encrypt all compiled modules. Decompilers then see ciphertext instead of
# bytecode; the per-build key lives in a native dylib, not in a Python file.
echo "==> Encrypting compiled modules"
"$PY" "$ROOT/packaging/engine/protect_engine.py" "$OUT/lib" || { echo "ERROR: bytecode protection failed"; exit 1; }
[[ -z "$(find "$OUT/lib" -type f -name '*.pyc' -print -quit)" ]] || { echo "ERROR: unencrypted bytecode remains"; exit 1; }

echo "==> License texts"
"$PY" "$ROOT/packaging/legal/generate_legal.py" --out "$WORK/legal" --emit-texts "$OUT/licenses" >/dev/null \
  || { cp "$ROOT/legal/THIRD_PARTY_NOTICES.md" "$OUT/licenses/"; cp -R "$ROOT/LICENSES/." "$OUT/licenses/"; echo "   (generate_legal --emit-texts unavailable: copied notices only)"; }

echo "==> C++ engine (binary only)"
# M6: a C++ bundle is one that actually ships the proven C ABI library. Detect it
# from the build output, or require it explicitly with VEYSTRIX_CPP_BUNDLE=1.
# No C/C++ source is ever copied into the bundle.
CPP_LIB=""
for candidate in "$ROOT/cpp/engine/build/libveystrix_c.dylib" \
                 "$ROOT/cpp/engine/build/capi/libveystrix_c.dylib" \
                 "$ROOT/cpp/engine/build/capi/libveystrix_c.so"; do
  if [[ -f "$candidate" ]]; then CPP_LIB="$candidate"; break; fi
done
CPP_ENGINE=0
CPP_LIB_SHA=""
if [[ "${VEYSTRIX_CPP_BUNDLE:-0}" == "1" ]] || [[ -n "$CPP_LIB" ]]; then
  [[ -n "$CPP_LIB" ]] || { echo "ERROR: VEYSTRIX_CPP_BUNDLE=1 but no libveystrix_c dylib was found" >&2; exit 1; }
  # cpp_parity:pass is NOT self-attested from mere binary presence. The release
  # engineer must explicitly assert Track-A parity for THIS dylib with
  # VEYSTRIX_CPP_PARITY_ATTEST=1. Without it we still ship the binary but emit a
  # python-only manifest, so --engine cpp cleanly refuses (no false "pass" claim).
  [[ "${VEYSTRIX_CPP_PARITY_ATTEST:-0}" == "1" ]] || {
    echo "WARNING: C++ dylib present but VEYSTRIX_CPP_PARITY_ATTEST!=1; emitting a python-only" >&2
    echo "         manifest. Set VEYSTRIX_CPP_PARITY_ATTEST=1 only after verifying this exact" >&2
    echo "         dylib is the Track-A-proven bit-exact build for both models." >&2
    CPP_LIB=""
  }
fi
if [[ -n "$CPP_LIB" ]]; then
  # Fail closed: the library must actually be the LIVE engine, not a fake/stub
  # that merely happens to sit at the build path (the stub build also emits
  # libveystrix_c.dylib and would otherwise inherit a false parity "pass").
  DYLD_LIBRARY_PATH="$SP/mlx/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" "$PY" - "$CPP_LIB" <<'PYEOF' || { echo "ERROR: C++ dylib is not the live engine" >&2; exit 1; }
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
lib.vx_abi_version.restype = ctypes.c_int
lib.vx_mode.restype = ctypes.c_char_p
if lib.vx_abi_version() < 1:
    raise SystemExit(f"ABI version {lib.vx_abi_version()} < 1")
mode = (lib.vx_mode() or b"").decode("ascii", "replace")
if mode != "live":
    raise SystemExit(f"mode={mode!r}, expected 'live'")
PYEOF
  CPP_ENGINE=1
  # Install at the bundle ROOT: OUT is installed as <home>/.veystrix/engine, and
  # cpp_engine.default_lib_path() probes <home>/.veystrix/engine/<name> first.
  # The old $OUT/engine/<name> layout nested it one level too deep and was never found.
  DEST="$OUT/$(basename "$CPP_LIB")"
  cp "$CPP_LIB" "$DEST"
  # The C ABI dylib is linked against @rpath/libmlx.dylib (and libjaccl) with a
  # build-host INSTALL_RPATH that does not exist on a user machine. In the
  # bundle, MLX's dylibs live under the bundled Python's site-packages; add a
  # relocatable @loader_path rpath so dlopen resolves them without the host path.
  MLX_LIB_IN_BUNDLE="@loader_path/python/lib/python3.12/site-packages/mlx/lib"
  install_name_tool -add_rpath "$MLX_LIB_IN_BUNDLE" "$DEST" 2>/dev/null || true
  # Delete every build-host rpath: these embed the developer's absolute path
  # (…/.venv/… and …/cpp/engine/…) and would leak it in the shipped binary.
  while IFS= read -r rp; do
    [[ -n "$rp" ]] && install_name_tool -delete_rpath "$rp" "$DEST" 2>/dev/null || true
  done < <(otool -l "$DEST" | awk '/cmd LC_RPATH/{g=1} g&&/ path /{print $2; g=0}' | grep '/Users/' || true)
  install_name_tool -id @rpath/libveystrix_c.dylib "$DEST" 2>/dev/null || true
  # install_name_tool invalidates the arm64 ad-hoc signature; re-sign the final bytes.
  codesign -f -s - "$DEST" 2>/dev/null || true
  # Attest the FINAL shipped bytes (post rpath rewrite), since that is what the
  # client hashes at load time. If a Track-A attested SHA is committed, require
  # an exact match; ATTEST alone cannot bless an arbitrary binary.
  ATTEST_SHA_FILE="$ROOT/packaging/engine/cpp_attested.sha256"
  if [[ -f "$ATTEST_SHA_FILE" ]]; then
    expected="$(tr -d ' \t\r\n' < "$ATTEST_SHA_FILE")"
    actual="$(shasum -a 256 "$DEST" | awk '{print $1}')"
    [[ "$expected" == "$actual" ]] || {
      echo "ERROR: shipped C++ dylib sha $actual != attested $expected" >&2
      exit 1
    }
  fi
  CPP_LIB_SHA="$(shasum -a 256 "$DEST" | awk '{print $1}')"
fi

echo "==> Engine CLI launcher (bin/veystrix-engine)"
# Documented front-end/engine contract: <home>/.veystrix/engine/bin/veystrix-engine
# serve --port ... --model ... --engine python|cpp, and --version --json. The
# bundle ships sources as .pyc under OUT/lib, so the launcher sets PYTHONPATH to
# OUT/lib and runs the bundled interpreter; no source is exposed. This is the
# executable supervisor.build_command() launches for the cpp engine.
mkdir -p "$OUT/bin"
cat > "$OUT/bin/veystrix-engine" <<'LAUNCHER'
#!/bin/sh
# Veystrix engine-side CLI (binary-only bundle). Resolves paths relative to the
# launcher so it works from any cwd after install.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$ROOT/lib${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
exec "$ROOT/python/bin/python3" -m veystrix.engine_cli "$@"
LAUNCHER
chmod +x "$OUT/bin/veystrix-engine"

# The user-facing command: the coding-agent TUI and every terminal-tool subcommand.
cat > "$OUT/bin/veystrix" <<'LAUNCHER'
#!/bin/sh
# Veystrix (binary-only bundle). Installed under ~/.veystrix/engine; resolves itself from any cwd.
set -e
HERE="$(cd "$(dirname "$0")" && pwd -P)"
ROOT="$(cd "$HERE/.." && pwd -P)"
export PYTHONPATH="$ROOT/lib"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
exec "$ROOT/python/bin/python3" -m veystrix "$@"
LAUNCHER
chmod +x "$OUT/bin/veystrix"

echo "==> Sanitize bundled interpreter (strip build path from stdlib .pyc and drop console scripts)"
# python-build-standalone's cached stdlib .pyc embed the absolute path of the machine that
# compiled them in co_filename. Recompile with stripdir="$OUT" so the stored path is
# bundle-relative, never /Users/<name>/... . Use the compile_dir API with a single worker:
# the CLI's parallel path (-j) silently drops stripdir, which would leave the path in.
find "$OUT/python/lib/python3.12" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
"$PY" - "$OUT" "$OUT/python/lib/python3.12" <<'PYEOF'
import compileall, sys
out, target = sys.argv[1], sys.argv[2]
compileall.compile_dir(target, stripdir=out, quiet=1, force=True)
PYEOF
# pip writes the absolute build-interpreter path into every console-script shebang
# (python/bin/mlx_lm, transformers, fastapi, ...). We launch the engine with
# `python3 -m veystrix`, never these, so remove them: they are dead weight and the only
# thing in python/bin that leaks the build path. Keep the interpreter itself.
find "$OUT/python/bin" -mindepth 1 -maxdepth 1 ! -name 'python3' ! -name 'python3.12' ! -name 'python' -delete 2>/dev/null || true

echo "==> Manifest and final trim"
# List what actually ships, using metadata rather than pip so pip can be removed before the smoke test.
{
  echo "Veystrix engine bundle, built $(date -u +%FT%TZ)"
  echo "runtime: $PBS_FILE"
  "$PY" - <<'PYEOF'
import importlib.metadata as md
skipped = {"pip", "setuptools", "wheel"}
for dist in sorted(md.distributions(), key=lambda d: (d.metadata.get("Name") or "").lower()):
    name = dist.metadata.get("Name") or ""
    if name.lower() not in skipped:
        print(f"{name}=={dist.version}")
PYEOF
} > "$OUT/BUNDLE-MANIFEST.txt"
# Machine-readable manifest consumed by the front end. A C++ bundle records BOTH
# proven models (Track A: bit-exact Qwen + Nemotron on one driver thread); the
# python-only bundle keeps the empty parity map, so --engine cpp stays a clean refusal.
"$PY" - "$ROOT/pyproject.toml" "$OUT/BUNDLE-MANIFEST.json" "$CPP_ENGINE" "$CPP_LIB_SHA" <<'PYEOF'
import json, sys, tomllib
version = tomllib.load(open(sys.argv[1], "rb"))["project"]["version"]
cpp = sys.argv[3] == "1"
lib_sha = sys.argv[4]
manifest = {
    "version": version,
    "protocol_version": "1",
    "engines": ["python", "cpp"] if cpp else ["python"],
    # Keys must match tool/engine.py::_parity_value's f"{model}:{profile}" lookup
    # using the model names the terminal tool passes to check_engine().
    # Both models are set to "pass" ONLY when VEYSTRIX_CPP_PARITY_ATTEST=1 was
    # asserted by the release engineer for the exact dylib below (never from
    # mere binary presence). The sha256 records which binary was attested.
    "cpp_parity": ({"flashnext:agent": "pass", "nemotron35-lightning:agent": "pass",
                    "inkling2-mini:agent": "pass"}
                   if cpp else {}),
}
if cpp:
    manifest["cpp_lib_sha256"] = lib_sha
with open(sys.argv[2], "w") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
PYEOF
# pip/setuptools/wheel are needed only to assemble the bundle and for the source-checkout `setup`
# command, which cannot run from a bundled engine. Remove them before the smoke test checks they are gone.
"$PY" -m pip uninstall -y pip setuptools wheel >/dev/null 2>&1 || true
rm -rf "$SP"/{pip,setuptools,wheel,pip-*,setuptools-*,wheel-*} 2>/dev/null || true

echo "==> Smoke test (imports only, no model)"
"$PY" "$HERE/smoke_import.py" "$OUT"

echo "==> Bundle ready: $OUT ($(du -sh "$OUT" | cut -f1)). Signing and leak scan are rebuilt for the terminal tool in milestone M4."
