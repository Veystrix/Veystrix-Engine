#!/bin/bash
# Tier-2 protection: compile the know-how modules to native extensions with Cython.
# Usage: compile_protected.sh <BUNDLE_LIB_DIR> [--require] [--dry-run]
#
# The bundle is built first (tier 1: .pyc, no .py). This step replaces the .pyc for the
# listed modules with native .so files and removes every generated .c/.py source. Per the
# brief: do NOT use -OO (it strips asserts that may be safety checks). If Cython is absent,
# the default keeps tier 1 and prints a clear notice; --require makes absence fatal for a
# release build. Never ship source of any kind; leak_scan.py enforces that afterwards.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB="${1:?usage: compile_protected.sh LIB_DIR [--require] [--dry-run]}"; shift || true
REQUIRE=0; DRY=0
for arg in "$@"; do
  case "$arg" in
    --require) REQUIRE=1 ;;
    --dry-run) DRY=1 ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done
[[ -d "$LIB" ]] || { echo "ERROR: no bundle lib directory: $LIB" >&2; exit 1; }
PY="${VEYSTRIX_BUNDLE_PYTHON:-python3}"
if ! "$PY" -c 'import Cython' >/dev/null 2>&1; then
  if [[ $REQUIRE == 1 ]]; then
    echo "ERROR: Cython is required for a release bundle but is not installed" >&2; exit 1
  fi
  echo "==> TIER-1 ONLY: Cython not installed; keeping .pyc bundle (cost-raiser, not DRM)."
  exit 0
fi
count=0
while IFS= read -r pattern; do
  [[ -z "$pattern" || "$pattern" == \#* ]] && continue
  for source in $LIB/$pattern; do
    [[ -f "$source" ]] || continue
    if [[ $DRY == 1 ]]; then
      echo "would compile: ${source#$LIB/}"
    else
      echo "==> compiling ${source#$LIB/}"
      c_file="${source%.py}.c"
      so_file="${source%.py}.so"
      if ! "$PY" -m cython -3 --no-docstrings "$source" -o "$c_file"; then
        echo "   TIER-1 FALLBACK: ${source#$LIB/} could not be compiled by Cython; kept as .pyc" >&2
        continue
      fi
      cflags=$("$PY"-config --cflags 2>/dev/null || "$PY" -c "import sysconfig; print(sysconfig.get_config_var('CFLAGS') or '')")
      ldflags=$("$PY"-config --ldflags 2>/dev/null || "$PY" -c "import sysconfig; print(sysconfig.get_config_var('LDFLAGS') or '')")
      inc=$("$PY" -c "import sysconfig; print('-I' + (sysconfig.get_path('include') or ''))" 2>/dev/null || true)
      if clang -bundle -undefined dynamic_lookup -O3 $cflags $inc "$c_file" $ldflags -o "$so_file" >/dev/null 2>&1 && [[ -f "$so_file" ]]; then
        rm -f "$c_file" "$source" "${source%.py}.pyc"
        echo "   compiled: ${so_file#$LIB/}"
      else
        echo "   TIER-1 FALLBACK: clang compile failed for ${source#$LIB/}; kept as .pyc" >&2
        rm -f "$c_file"
        continue
      fi
    fi
    count=$((count + 1))
  done
done < "$HERE/cython_modules.txt"
echo "==> tier-2 targets: $count. Cython does not remove all bytecode: MLX/transformers stay dependencies."
