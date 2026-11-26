"""Release signature check: a real minisign signature (throwaway test key) and tampering."""
import hashlib
import io
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from veystrix.tool import updater
from veystrix.tool.signature import RELEASE_PUBLIC_KEY, SignatureError, verify_minisign

PUBLIC = 'untrusted comment: minisign public key 3445DAA22E255B94\nRWSUWyUuotpFNGeEDKzDBjvEPBRW/lypOr/Q372k59icvqsujRZZBDGL\n'
SIGNATURE = 'untrusted comment: signature from minisign secret key\nRUSUWyUuotpFNMbpnbeoGCUX+RstcMrGQPyBNY5vCYxT91J4TzbN2xt5/lWhmH5KGAVSQoHsritLIav/4cbmN6UDHLLsUz2YSAo=\ntrusted comment: veystrix test\n8IjOy5aLFkBi1vMuwo6yxUO6L5ELbyOpVVQV1lgBTLxsXyzts5zWipuD9tQRbQ1Kr3fN/2qWeDmt4yctkTiPAw==\n'
DATA = 'abc123  veystrix-engine-1.1.0-arm64.tar.zst\n'.encode()


class MinisignVerify(unittest.TestCase):
    def test_valid_signature_returns_the_trusted_comment(self):
        self.assertEqual(verify_minisign(DATA, SIGNATURE, PUBLIC), "veystrix test")

    def test_tampering_is_rejected(self):
        for data, sig, key in ((DATA + b"x", SIGNATURE, PUBLIC),
                               (DATA, SIGNATURE.replace("veystrix test", "veystrix evil"), PUBLIC),
                               (DATA, SIGNATURE, RELEASE_PUBLIC_KEY),
                               (DATA, "garbage", PUBLIC)):
            with self.assertRaises(SignatureError):
                verify_minisign(data, sig, key)


def _engine_archive(path: Path, version: str, extra: dict | None = None) -> None:
    files = {"engine/BUNDLE-MANIFEST.json": f'{{"version": "{version}"}}'.encode(),
             "engine/bin/veystrix": b"#!/bin/sh\n", **(extra or {})}
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            tar.addfile(info, io.BytesIO(data))


@unittest.skipUnless(shutil.which("minisign"), "minisign is needed to sign the test release")
class UpdaterInstall(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        subprocess.run(["minisign", "-G", "-W", "-p", self.dir / "k.pub", "-s", self.dir / "k.key"],
                       check=True, capture_output=True)
        self.public = (self.dir / "k.pub").read_text()
        self.home = self.dir / "home"

    def release(self, archive: Path) -> tuple[bytes, str]:
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        sums = self.dir / "SHA256SUMS.txt"
        sums.write_text(f"{digest}  {archive.name}\n")
        subprocess.run(["minisign", "-S", "-s", self.dir / "k.key", "-m", sums], check=True, capture_output=True)
        return sums.read_bytes(), (self.dir / "SHA256SUMS.txt.minisig").read_text()

    def test_verified_archive_replaces_the_engine_and_keeps_everything_else(self):
        archive = self.dir / "veystrix-engine-1.2.0-arm64.tar.gz"
        _engine_archive(archive, "1.2.0")
        sums, sig = self.release(archive)
        old = self.home / ".veystrix" / "engine"
        old.mkdir(parents=True)
        (old / "BUNDLE-MANIFEST.json").write_text('{"version": "1.1.0"}')
        (self.home / ".veystrix" / "config.json").write_text("{}")
        updater.verify(archive, sums, sig, self.public)
        updater.install_archive(archive, self.home)
        self.assertEqual(updater.installed_version(self.home), "1.2.0")
        self.assertTrue((self.home / ".veystrix" / "config.json").exists())
        self.assertFalse((self.home / ".veystrix" / "engine.previous").exists())

    def test_tampered_archive_or_signature_installs_nothing(self):
        archive = self.dir / "veystrix-engine-1.2.0-arm64.tar.gz"
        _engine_archive(archive, "1.2.0")
        sums, sig = self.release(archive)
        with self.assertRaises(updater.UpdateError):
            updater.verify(archive, sums, sig)                        # signed by another key
        archive.write_bytes(archive.read_bytes() + b"x")
        with self.assertRaises(updater.UpdateError):
            updater.verify(archive, sums, sig, self.public)           # checksum no longer matches

    def test_archive_escaping_the_engine_folder_is_refused(self):
        archive = self.dir / "veystrix-engine-1.2.0-arm64.tar.gz"
        _engine_archive(archive, "1.2.0", {"evil.txt": b"x"})
        with self.assertRaises(updater.UpdateError):
            updater.install_archive(archive, self.home)
        self.assertFalse((self.home / ".veystrix" / "engine").exists())


class UpdaterChoice(unittest.TestCase):
    def test_newest_terminal_build_wins_and_versions_order(self):
        rel = lambda tag, name, pre=False: {"tag_name": tag, "prerelease": pre, "assets": [   # noqa: E731
            {"name": name, "browser_download_url": "u/" + name}]}
        build = updater.latest_build([rel("v1.0.0", "Veystrix-1.0.0-arm64.dmg"),
                                      rel("v1.1.0", "veystrix-engine-1.1.0-arm64.tar.gz", True),
                                      rel("v1.0.9", "veystrix-engine-1.0.9-arm64.tar.gz")])
        self.assertEqual(build["version"], "1.1.0")
        self.assertTrue(build["prerelease"])
        key = updater.version_key
        self.assertLess(key("1.9.0"), key("1.10.0"))
        self.assertLess(key("1.2.0-early.1"), key("1.2.0"))
        self.assertLess(key("1.2.0-early.1"), key("1.2.0-early.2"))


if __name__ == "__main__":
    unittest.main()
