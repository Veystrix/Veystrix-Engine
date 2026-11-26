#!/bin/bash
# Build the binary-only terminal bundle and prepare a DRAFT public release.
# Default is --dry-run: it prints every step and runs the local gates only.
# The owner must pass --publish-draft explicitly to call `gh release create --draft`;
# this script NEVER publishes a release by itself. No Apple ID / notarization exists.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PUBLIC="${VEYSTRIX_PUBLIC_REPO:-/Users/vincent/Documents/Veystrix-Projects/Veystrix-Public}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3.12 || command -v python3)"
VERSION="$("$PY" -c 'import tomllib;print(tomllib.load(open("'"$ROOT"'/pyproject.toml","rb"))["project"]["version"])')"
DIST="$ROOT/dist/veystrix-$VERSION"
PUBLISH=0
UPLOAD=1
for arg in "$@"; do
  case "$arg" in
    --publish-draft) PUBLISH=1 ;;
    --build-only) PUBLISH=1; UPLOAD=0 ;;      # every local step for real, no GitHub call
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done

echo "==> version $VERSION -> $DIST"
echo "    (dry-run: NOTHING is executed and no release is created unless --publish-draft is passed)"
run() { echo "    + $*"; if [[ $PUBLISH == 1 ]]; then "$@"; fi; }
if [[ $PUBLISH == 1 ]]; then
  command -v minisign >/dev/null || { echo "ERROR: minisign is required to sign a release" >&2; exit 1; }
  [[ -n "${VEYSTRIX_MINISIGN_KEY:-}" && -f "${VEYSTRIX_MINISIGN_KEY}" ]] \
    || { echo "ERROR: VEYSTRIX_MINISIGN_KEY must point to the offline private key" >&2; exit 1; }
else
  echo "    note: minisign/Cython are only required when actually building a release;"
  echo "          this environment has neither installed; dry-run documents the steps."
fi

echo "==> 1. engine bundle (tier 1)"
run bash "$ROOT/packaging/engine/build_engine_bundle.sh" "$DIST/engine"
echo "==> 2. tier-2 Cython protection (fails closed for release)"
run bash "$ROOT/packaging/engine/compile_protected.sh" "$DIST/engine/lib" --require
echo "==> 3. engine CLI contract"
run "$PY" -m veystrix.engine_cli --version --json --home "$HOME/.veystrix"
echo "==> 4. leak scan (hard gate: no source, no private paths, no tokens, no uncompiled protected modules)"
run "$PY" "$ROOT/packaging/engine/leak_scan.py" --bundle "$DIST" --public "$PUBLIC" --require-compiled
ASSET="veystrix-engine-$VERSION-arm64.tar.gz"
OUTDIR="$ROOT/dist/release-$VERSION"
echo "==> 5. archive, checksums + minisign signature (private key stays offline)"
run mkdir -p "$OUTDIR"
run env COPYFILE_DISABLE=1 tar --no-mac-metadata -C "$DIST" -czf "$OUTDIR/$ASSET" engine
run sh -c "cd '$OUTDIR' && shasum -a 256 '$ASSET' > SHA256SUMS.txt"
run minisign -S -s "${VEYSTRIX_MINISIGN_KEY:-}" -m "$OUTDIR/SHA256SUMS.txt" -t "Veystrix $VERSION"
run sh -c "cd '$OUTDIR' && shasum -a 256 -c SHA256SUMS.txt"
echo "==> 6. Homebrew formula for the tap (joyboyreturns/homebrew-veystrix)"
run "$PY" "$ROOT/packaging/release/brew_formula.py" "$VERSION" "$OUTDIR/$ASSET" "$OUTDIR/veystrix.rb"
echo "==> 7. DRAFT release in the public repo (owner publishes by hand)"
NOTES_FILE="$ROOT/docs/release/RELEASE_NOTES.md"
[[ ! -f "$NOTES_FILE" ]] && NOTES_FILE="$ROOT/docs/release/EARLY_BUILD_NOTES.md"
[[ $UPLOAD == 1 ]] && run gh release create "v$VERSION" -R Veystrix/veystrix-engine --draft \
    --title "Veystrix $VERSION" --notes-file "$NOTES_FILE" \
    "$OUTDIR/$ASSET" "$OUTDIR/SHA256SUMS.txt" "$OUTDIR/SHA256SUMS.txt.minisig"

if [[ $PUBLISH == 0 ]]; then
  echo "DRY RUN COMPLETE. Nothing published. To create a DRAFT, re-run with --publish-draft."
else
  echo "DRAFT created. The owner must review and publish it by hand."
fi
