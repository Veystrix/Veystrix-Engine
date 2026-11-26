import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from types import SimpleNamespace

from veystrix import universal_cache as U


class SharedCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.cache = U.CacheManager(self.directory, quota_bytes=U.DEFAULT_QUOTA_GB * U.GB, reserve_bytes=0)

    def snapshot(self, family, key, tokens, size, parent=None):
        fingerprint = "a" * 64
        path = self.cache.namespace(family, fingerprint)
        key = key * U.FORMATS[family][0]
        data = path / (key + ".safetensors")
        data.write_bytes(bytes(size))
        side = path / (key + U.FORMATS[family][1])
        if family == "flashnext":
            side.write_text(json.dumps({"key": key, "fingerprint": fingerprint, "parent": parent,
                                        "file_bytes": size, "n_tokens": tokens}))
        else:
            side.write_bytes(b"VXT1" + bytes(tokens * 4))
        return key, path, data.stat().st_size + side.stat().st_size

    def test_quota_preview_lease_and_ancestor(self):
        base, qwen, base_size = self.snapshot("flashnext", "b", 100, 150)
        leaf, _, child_size = self.snapshot("flashnext", "c", 10_000, 150, base)
        _, _, other_size = self.snapshot("nemotron35", "d", 100, 200)
        with self.cache.transaction() as tx:
            self.assertEqual(tx.quota, 80 * U.GB)
            self.assertEqual(tx.used, base_size + child_size + other_size)
            lease = tx.lease(qwen, leaf)
        try:
            with self.cache.transaction() as tx:
                preview = tx.reclaim(tx.used)
                self.assertEqual(preview["reclaimed_bytes"], other_size)
                self.assertEqual(preview["shortfall_bytes"], base_size + child_size)
                self.assertEqual(len(tx.entries), 3)
                blocked = tx.set_quota(100)
                self.assertGreater(blocked["shortfall_bytes"], 0)
                self.assertEqual(blocked["reclaimed_bytes"], 0)
                self.assertEqual(len(tx.entries), 3)
                result = tx.reclaim(other_size, apply=True)
                self.assertEqual(result["reclaimed_bytes"], other_size)
                self.assertFalse(tx.make_room(tx.quota))
        finally:
            os.close(lease)
        with self.cache.transaction() as tx:
            self.assertEqual(tx.set_quota(100)["shortfall_bytes"], 0)
            self.assertEqual(tx.used, 0)
        with U.CacheManager(self.directory).transaction() as tx:
            self.assertEqual(tx.quota, 100)

    def test_model_free_process_observes_committed_files_and_skips_unknown_data(self):
        _, path, size = self.snapshot("nemotron35", "a", 8, 200)
        user = path / "conversation.txt"
        user.write_text("preserve")
        orphan = path / ("b" * 24 + ".safetensors")
        orphan.write_bytes(b"interrupted")
        link = path / ("c" * 24 + ".safetensors")
        link.symlink_to(user)
        command = [sys.executable, "-m", "veystrix.universal_cache", "--directory", str(self.directory)]
        status = json.loads(subprocess.check_output(command + ["status"]))
        self.assertEqual(status["used_bytes"], size)
        self.assertFalse(orphan.exists())
        subprocess.check_output(command + ["clear"])
        self.assertEqual(user.read_text(), "preserve")
        self.assertTrue(link.is_symlink())

    def test_external_lease_and_pins_protect_the_same_registry(self):
        key, path, _ = self.snapshot("flashnext", "b", 1000, 100)
        other, _, other_size = self.snapshot("flashnext", "c", 500, 50)
        with self.cache.transaction() as tx:
            self.assertTrue(tx.set_pin(path, other, True))
        script = """
import os, sys
from pathlib import Path
from veystrix.universal_cache import CacheManager
with CacheManager(sys.argv[1]).transaction() as tx:
    fd = tx.lease(Path(sys.argv[2]), sys.argv[3])
print("ready", flush=True)
sys.stdin.readline()
os.close(fd)
"""
        with subprocess.Popen([sys.executable, "-c", script, str(self.directory), str(path), key],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as process:
            try:
                self.assertEqual(process.stdout.readline().strip(), "ready")
                with self.cache.transaction() as tx:
                    self.assertEqual(tx.reclaim(tx.used, apply=True)["snapshots"], 0)
            finally:
                process.communicate("\n", timeout=10)
        with self.cache.transaction() as tx:
            result = tx.reclaim(tx.used, apply=True)
            self.assertEqual(result["shortfall_bytes"], other_size)
            self.assertTrue(tx.set_pin(path, other, False))
            self.assertEqual(tx.reclaim(tx.used, apply=True)["shortfall_bytes"], 0)

    def test_reserve_refuses_growth_even_below_quota(self):
        cache = U.CacheManager(self.directory, reserve_bytes=10**18)
        with cache.transaction() as tx:
            self.assertFalse(tx.make_room(1))


class WatermarkAndArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.cache = U.CacheManager(self.directory, quota_bytes=400, reserve_bytes=0)

    def test_dynamic_quota_clamps_to_free_space_fraction(self):
        self.assertEqual(U.dynamic_quota_bytes(80 * U.GB, 1_000 * U.GB), 80 * U.GB)
        self.assertEqual(U.dynamic_quota_bytes(80 * U.GB, 50 * U.GB), int(50 * U.GB * U.WATERMARK_FRACTION))
        self.assertEqual(U.dynamic_quota_bytes(0, 100), 0)
        self.assertEqual(U.dynamic_quota_bytes(100, -5), 0)

    def test_make_room_uses_the_clamped_quota(self):
        with unittest.mock.patch("veystrix.universal_cache.shutil.disk_usage",
                                 return_value=SimpleNamespace(free=1000, total=2000, used=1000)):
            with self.cache.transaction() as tx:
                self.assertEqual(tx.effective_quota, 250)
                self.assertFalse(tx.make_room(300))
                self.assertTrue(tx.make_room(200))

    def test_artifact_tier_is_content_addressed_durable_and_verified(self):
        first = U.ArtifactTier(self.cache)
        record = first.put(b"def parse():\n    return 1\n", namespace="ast", meta={"path": "a.py"})
        digest = record["digest"]
        self.assertEqual(first.get(digest), b"def parse():\n    return 1\n")
        duplicate = first.put(b"def parse():\n    return 1\n")
        self.assertEqual(duplicate["digest"], digest)
        self.assertEqual(first.status()["artifacts"], 1)

        reopened = U.ArtifactTier(U.CacheManager(self.directory, quota_bytes=U.GB, reserve_bytes=0))
        self.assertEqual(reopened.get(digest), b"def parse():\n    return 1\n")
        self.assertTrue(reopened.verify(digest)["present"])
        data_path, _record_path = reopened._paths(digest)
        data_path.write_bytes(b"tampered")
        self.assertIsNone(reopened.get(digest))
        self.assertFalse(reopened.verify(digest)["present"])

    def test_compact_drops_stale_registry_records(self):
        with self.cache.transaction() as tx:
            tx.metadata["reuse"] = {"nemotron35/" + "a" * 64 + "/" + "b" * 24: {"hits": 3}}
            tx.metadata["pins"] = ["flashnext/" + "c" * 64 + "/" + "d" * 64]
            result = tx.compact()
        self.assertEqual(result["stale_reuse"], 1)
        self.assertEqual(result["stale_pins"], 1)
        self.assertEqual(tx.metadata["reuse"], {})
        self.assertEqual(tx.metadata["pins"], [])
