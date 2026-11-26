#!/bin/bash
# Veystrix installer for Apple-silicon Macs. Binary-only: no source code is downloaded.
#   curl -fsSL https://raw.githubusercontent.com/Veystrix/Veystrix-Engine/main/install.sh | bash
# Installs the newest Veystrix terminal build into ~/.veystrix/engine
# and a `veystrix` command into ~/.local/bin (override with VEYSTRIX_BIN_DIR).
# Later updates: `veystrix update` (it verifies the release signature before installing).
set -euo pipefail
REPO="Veystrix/Veystrix-Engine"
PUBKEY="RWRqvDsIRe4F5ZmA+iMcMf+cnRjiG+k6NDEQFm7m0BsyVNmP/p9IUceg"
HOME_DIR="$HOME/.veystrix"
BIN_DIR="${VEYSTRIX_BIN_DIR:-$HOME/.local/bin}"
die() { echo "veystrix install: $*" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || die "Veystrix needs a Mac with Apple silicon"

echo "==> finding the newest Veystrix build"
api="$(curl -fsSL -H 'Accept: application/vnd.github+json' "https://api.github.com/repos/$REPO/releases?per_page=30")" \
  || die "could not reach GitHub"
url="$(printf '%s' "$api" | grep -o '"browser_download_url": *"[^"]*/veystrix-engine-[^"/]*-arm64\.tar\.gz"' \
       | head -1 | sed 's/.*"\(https:[^"]*\)"$/\1/')"
[[ -n "$url" ]] || die "no Veystrix terminal build is published yet"
base="${url%/*}"; asset="${url##*/}"; version="${asset#veystrix-engine-}"; version="${version%-arm64.tar.gz}"

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
echo "==> downloading Veystrix $version"
curl -fL --progress-bar -o "$WORK/$asset" "$url"
curl -fsSL -o "$WORK/SHA256SUMS.txt" "$base/SHA256SUMS.txt"
curl -fsSL -o "$WORK/SHA256SUMS.txt.minisig" "$base/SHA256SUMS.txt.minisig"

if command -v minisign >/dev/null 2>&1; then
  minisign -Vm "$WORK/SHA256SUMS.txt" -P "$PUBKEY" >/dev/null || die "signature check FAILED; nothing was installed"
  echo "==> signature OK"
else
  echo "    (minisign not installed: checking the SHA-256 only; \`veystrix update\` checks signatures)"
fi
want="$(awk -v n="$asset" '$2==n {print $1}' "$WORK/SHA256SUMS.txt")"
have="$(shasum -a 256 "$WORK/$asset" | awk '{print $1}')"
[[ -n "$want" && "$want" == "$have" ]] || die "checksum mismatch for $asset; nothing was installed"
echo "==> checksum OK"

mkdir -p "$HOME_DIR"
stage="$(mktemp -d "$HOME_DIR/engine.new-XXXXXX")"
tar -xzf "$WORK/$asset" -C "$stage"
[[ -x "$stage/engine/bin/veystrix" ]] || { rm -rf "$stage"; die "the download is not a Veystrix bundle"; }
rm -rf "$HOME_DIR/engine.previous"
[[ -d "$HOME_DIR/engine" ]] && mv "$HOME_DIR/engine" "$HOME_DIR/engine.previous"
mv "$stage/engine" "$HOME_DIR/engine" && rmdir "$stage"
rm -rf "$HOME_DIR/engine.previous"

mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/veystrix" <<'SH'
#!/bin/sh
exec "$HOME/.veystrix/engine/bin/veystrix" "$@"
SH
chmod 755 "$BIN_DIR/veystrix"

echo "==> installed Veystrix $version"
case ":$PATH:" in
  *":$BIN_DIR:"*) echo "    run: veystrix" ;;
  *) echo "    add $BIN_DIR to your PATH, e.g.:  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.zshrc"
     echo "    then run: veystrix" ;;
esac
