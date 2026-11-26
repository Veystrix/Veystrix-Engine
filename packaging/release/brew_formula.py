"""Write the Homebrew formula for one release archive.

Homebrew rewrites install names and re-signs Mach-O files it installs, which would break the
bundle's pinned checksums. The formula therefore keeps the verified archive sealed in libexec
(`using: :nounzip`) and its `veystrix` launcher unpacks it into ~/.veystrix/engine the first
time a new version runs. `brew upgrade veystrix` ships the next archive the same way.

usage: brew_formula.py VERSION ARCHIVE OUT.rb
"""
import hashlib
import sys
from pathlib import Path

TEMPLATE = '''class Veystrix < Formula
  desc "Commercial local LLM inference engine & terminal coding agent for Apple Silicon"
  homepage "https://github.com/Veystrix/veystrix-engine"
  url "https://github.com/Veystrix/veystrix-engine/releases/download/v{version}/{asset}",
      using: :nounzip
  sha256 "{sha256}"
  version "{version}"
  license :cannot_represent

  depends_on arch: :arm64
  depends_on :macos

  def install
    libexec.install "{asset}" => "veystrix-engine.tar.gz"
    (bin/"veystrix").write <<~SH
      #!/bin/sh
      # Unpacks the sealed engine on first run of each version, then runs it.
      set -e
      ENGINE="$HOME/.veystrix/engine"
      if [ "$(cat "$ENGINE/.installed-version" 2>/dev/null)" != "{version}" ]; then
        echo "veystrix: setting up version {version} (first run only) ..." >&2
        mkdir -p "$HOME/.veystrix"
        STAGE="$(mktemp -d "$HOME/.veystrix/engine.new-XXXXXX")"
        tar -xzf "#{{libexec}}/veystrix-engine.tar.gz" -C "$STAGE"
        rm -rf "$HOME/.veystrix/engine.previous"
        [ -d "$ENGINE" ] && mv "$ENGINE" "$HOME/.veystrix/engine.previous"
        mv "$STAGE/engine" "$ENGINE" && rmdir "$STAGE"
        rm -rf "$HOME/.veystrix/engine.previous"
        echo "{version}" > "$ENGINE/.installed-version"
      fi
      VEYSTRIX_INSTALLER=brew exec "$ENGINE/bin/veystrix" "$@"
    SH
  end

  def caveats
    <<~EOS
      Veystrix is installed. Run `veystrix` in any project folder to start.
      Models are downloaded on request from inside Veystrix (/download).
    EOS
  end

  test do
    assert_match "veystrix", shell_output("#{{bin}}/veystrix --version")
  end
end
'''


def main() -> None:
    version, archive, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    out.write_text(TEMPLATE.format(version=version, asset=archive.name, sha256=sha))
    print(f"wrote {out} (sha256 {sha[:12]}...)")


if __name__ == "__main__":
    main()
