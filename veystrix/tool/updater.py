"""`veystrix update`: find, verify and install the newest Veystrix terminal build.

Trust chain: SHA256SUMS.txt is signed with the owner's minisign key; the public half is
compiled into veystrix/tool/signature.py. An archive is installed only when the signature
verifies AND the archive's SHA-256 is the one in that signed list. The new engine is
unpacked next to the old one and swapped in with renames, so an interrupted update leaves
the installed version working. Models, the cache and settings are never touched.

Homebrew installs are upgraded by Homebrew (`brew upgrade veystrix`), not by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO = "joyboyreturns/Veystrix"
RELEASES_API = f"https://api.github.com/repos/{REPO}/releases?per_page=30"
ASSET = re.compile(r"^veystrix-engine-(?P<version>[0-9][0-9A-Za-z.\-]*)-arm64\.tar\.gz$")
SUMS, SIG = "SHA256SUMS.txt", "SHA256SUMS.txt.minisig"


class UpdateError(RuntimeError):
    pass


def version_key(version: str) -> tuple:
    """Order 1.10.0 after 1.9.0, and a pre-release (1.2.0-early.1) before its release."""
    core, _, pre = str(version).lstrip("v").partition("-")
    nums = tuple(int(x) if x.isdigit() else 0 for x in core.split("."))
    return nums + ((1,) if not pre else (0, *[int(p) if p.isdigit() else p for p in re.split(r"[.\-]", pre)]))


def installed_version(home: Path) -> str:
    from veystrix.tool import engine as engine_mod
    version = str(engine_mod.engine_version(home).get("version") or "")
    if version:
        return version
    try:
        import tomllib
        root = Path(__file__).resolve().parents[2]
        return tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    except (OSError, KeyError, ValueError, TypeError):
        return ""


def _get(url: str, timeout: float = 30) -> bytes:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "veystrix-updater"})
    with urllib.request.urlopen(request, timeout=timeout) as response:   # noqa: S310 (https only)
        return response.read()


def releases(environ=None) -> list[dict]:
    """Newest-first releases; tests inject VEYSTRIX_TEST_RELEASES (a release or a list)."""
    environ = os.environ if environ is None else environ
    injected = environ.get("VEYSTRIX_TEST_RELEASES")
    data = json.loads(injected) if injected else json.loads(_get(RELEASES_API).decode("utf-8", "replace"))
    data = data if isinstance(data, list) else [data]
    return [r for r in data if isinstance(r, dict) and not r.get("draft")]


def latest_build(items: list[dict]) -> dict | None:
    """The newest release that carries a terminal-tool archive, with its asset URLs."""
    best = None
    for release in items:
        assets = {a.get("name"): a.get("browser_download_url") for a in release.get("assets") or []
                  if isinstance(a, dict)}
        for name, url in assets.items():
            match = ASSET.match(name or "")
            if not match or not url:
                continue
            found = {"version": match.group("version"), "tag": release.get("tag_name", ""),
                     "prerelease": bool(release.get("prerelease")), "asset": name, "url": url,
                     "sums": assets.get(SUMS), "sig": assets.get(SIG),
                     "published": release.get("published_at", "")}
            if best is None or version_key(found["version"]) > version_key(best["version"]):
                best = found
    return best


def verify(archive: Path, sums_text: bytes, signature_text: str, public_key: str | None = None) -> None:
    from veystrix.tool.signature import RELEASE_PUBLIC_KEY, SignatureError, verify_minisign
    try:
        verify_minisign(sums_text, signature_text, public_key or RELEASE_PUBLIC_KEY)
    except SignatureError as exc:
        raise UpdateError(f"the release signature is not valid ({exc}); nothing was installed") from None
    expected = None
    for line in sums_text.decode("utf-8", "replace").splitlines():
        digest, _, name = line.strip().partition("  ")
        if name.lstrip("*./") == archive.name:
            expected = digest.lower()
    if expected is None:
        raise UpdateError(f"{archive.name} is not listed in the signed checksums; nothing was installed")
    digest = hashlib.sha256()
    with open(archive, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise UpdateError(f"{archive.name} does not match its signed checksum; nothing was installed")


def install_archive(archive: Path, home: Path) -> Path:
    """Unpack a verified archive (top folder ``engine/``) and swap it in as ~/.veystrix/engine."""
    from veystrix.tool import engine as engine_mod
    target = engine_mod.bundle_root(home)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="engine.new-", dir=target.parent))
    try:
        with tarfile.open(archive, "r:gz") as tar:
            members = [m for m in tar.getmembers()
                       if not (Path(m.name).name.startswith("._") or Path(m.name).name == ".DS_Store")]
            for member in members:
                parts = Path(member.name).parts
                if not parts or parts[0] != "engine":
                    raise UpdateError(f"unexpected path in archive: {member.name!r}")
            # "data" refuses absolute paths, '..' escapes and links that leave the folder.
            tar.extractall(staging, members=members, filter="data")
        new = staging / "engine"
        if not (new / "BUNDLE-MANIFEST.json").is_file() or not (new / "bin" / "veystrix").is_file():
            raise UpdateError("the archive is not a Veystrix engine bundle; nothing was installed")
        old = target.parent / "engine.previous"
        shutil.rmtree(old, ignore_errors=True)
        if target.exists():
            target.rename(old)
        new.rename(target)
        shutil.rmtree(old, ignore_errors=True)
        return target
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def update(home: Path, *, check_only: bool = False, echo=print, environ=None) -> int:
    environ = os.environ if environ is None else environ
    current = installed_version(home)
    try:
        build = latest_build(releases(environ))
    except (OSError, ValueError) as exc:
        echo(f"update: could not reach GitHub Releases: {exc}")
        return 1
    if build is None:
        echo("update: no Veystrix terminal build is published yet")
        return 1
    newer = not current or version_key(build["version"]) > version_key(current)
    label = build["version"] + (" (early build)" if build["prerelease"] else "")
    echo(f"installed {current or 'unknown'} · latest {label}")
    if not newer:
        echo("Veystrix is up to date.")
        return 0
    if environ.get("VEYSTRIX_INSTALLER") == "brew":
        echo("Installed with Homebrew: run  brew update && brew upgrade veystrix")
        return 0
    if check_only:
        echo("run  veystrix update  to install it")
        return 0
    if not build["sums"] or not build["sig"]:
        echo("update: the release has no signed checksums; refusing to install it")
        return 1
    work = Path(tempfile.mkdtemp(prefix="veystrix-update-"))
    try:
        archive = work / build["asset"]
        echo(f"downloading {build['asset']} ...")
        archive.write_bytes(_get(build["url"], timeout=600))
        verify(archive, _get(build["sums"]), _get(build["sig"]).decode("utf-8", "replace"))
        echo("signature and checksum verified")
        install_archive(archive, home)
    except UpdateError as exc:
        echo(f"update: {exc}")
        return 1
    except OSError as exc:
        echo(f"update: download failed ({exc}); the installed version is unchanged")
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    echo(f"updated to {build['version']}. Restart the engine to use it:  veystrix stop && veystrix start")
    return 0
