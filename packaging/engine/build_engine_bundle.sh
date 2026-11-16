#!/bin/bash
# Build the self-contained engine runtime that Veystrix.app embeds in Contents/Resources/engine.
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
for pkg in veystrix models/__init__.py models/flashnext models/nemotron35 models/shared; do
  mkdir -p "$OUT/lib/$(dirname "$pkg")" 2>/dev/null || true
  ditto "$ROOT/$pkg" "$OUT/lib/$pkg"
done
# Nemotron ships its runtime, not its model-free tests.
find "$OUT/lib/models/nemotron35" -name 'test_*.py' -delete
find "$OUT/lib" \( -name tests -o -name bench -o -name __pycache__ -o -name '*.md' \) -prune -exec rm -rf {} +
# The engine imports its telemetry and memory sampler at module load (veystrix.bench.telemetry is a no-op
# unless a benchmark activates it). Ship exactly that import closure, not the qualification harnesses.
mkdir -p "$OUT/lib/veystrix/bench"
cp "$ROOT/veystrix/bench/"{__init__.py,telemetry.py,memory.py,provenance.py} "$OUT/lib/veystrix/bench/"
# The production safety guard and optional layer diagnostics currently share these
# benchmark utilities. Ship only their exact import closure, not the test suite.
mkdir -p "$OUT/lib/models/flashnext/tests/bench"
cp "$ROOT/models/flashnext/tests/__init__.py" "$OUT/lib/models/flashnext/tests/"
cp "$ROOT/models/flashnext/tests/bench/"{__init__.py,prefill_memory_harness.py,bench_prefill_memory.py,layer_gate_probe.py} "$OUT/lib/models/flashnext/tests/bench/"
# Precompile the native helper. Packaged builds then need no C source and no
# compiler on the user's Mac (a normal macOS install has no xcrun/clang).
echo "==> Precompiling native_read.dylib"
SRCDIR="$ROOT/models/flashnext"
xcrun clang -O2 -dynamiclib -arch "$(uname -m)" "$SRCDIR/native_read.c" \
  -o "$OUT/lib/models/flashnext/native_read.bundled.dylib"
[[ -f "$OUT/lib/models/flashnext/native_read.bundled.dylib" ]] || { echo "ERROR: native dylib missing"; exit 1; }

# Compile next to the source (-b = legacy .pyc location, importable without the .py), then drop every .py file.
# The settings registry now globs setting_*.pyc first, so no provider source is shipped.
"$PY" -m compileall -q -b -j 0 "$OUT/lib"
find "$OUT/lib" -name '*.py' -delete
find "$OUT/lib" -name '*.c' -delete
[[ -f "$OUT/lib/models/flashnext/settings/registry.pyc" ]] || { echo "ERROR: registry missing"; exit 1; }
[[ -z "$(find "$OUT/lib" -type f \( -name '*.py' -o -name '*.c' \) -print -quit)" ]] || { echo "ERROR: source files remain in bundle"; exit 1; }

echo "==> License texts"
"$PY" "$ROOT/packaging/legal/generate_legal.py" --out "$WORK/legal" --emit-texts "$OUT/licenses" >/dev/null \
  || { cp "$ROOT/legal/THIRD_PARTY_NOTICES.md" "$OUT/licenses/"; cp -R "$ROOT/LICENSES/." "$OUT/licenses/"; echo "   (generate_legal --emit-texts unavailable: copied notices only)"; }

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
# pip/setuptools/wheel are needed only to assemble the bundle and for the source-checkout `setup`
# command, which cannot run from a bundled engine. Remove them before the smoke test checks they are gone.
"$PY" -m pip uninstall -y pip setuptools wheel >/dev/null 2>&1 || true
rm -rf "$SP"/{pip,setuptools,wheel,pip-*,setuptools-*,wheel-*} 2>/dev/null || true

echo "==> Smoke test (imports only, no model)"
"$PY" "$HERE/smoke_import.py" "$OUT"

echo "==> Bundle ready: $OUT ($(du -sh "$OUT" | cut -f1)). Sign nested Mach-O files with packaging/macos/sign_app.sh."
