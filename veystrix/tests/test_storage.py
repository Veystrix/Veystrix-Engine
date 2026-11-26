"""Unit tests for `veystrix storage` SSD prompt-cache inspection and reclamation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from veystrix.tool.storage import claim_cache, format_bytes, parse_size, scan_cache, storage_report


def _make_snapshot(base: Path, family: str, digest: str, size_bytes: int, mtime: float) -> Path:
    snap = base / family / "sessions" / "prefix-cache" / digest
    (snap / "vx_snapshot").mkdir(parents=True, exist_ok=True)
    data = snap / "vx_snapshot" / "generic_state.safetensors"
    data.write_bytes(b"x" * size_bytes)
    (snap / "tokens.json").write_text(json.dumps({"tokens": [1, 2, 3]}), encoding="utf-8")
    os.utime(data, (mtime, mtime))
    os.utime(snap / "tokens.json", (mtime, mtime))
    os.utime(snap, (mtime, mtime))
    return snap


class StorageTests(unittest.TestCase):
    def test_parse_size(self):
        self.assertEqual(parse_size("500MB"), 500 * 1024 ** 2)
        self.assertEqual(parse_size("1.5GB"), int(1.5 * 1024 ** 3))
        self.assertEqual(parse_size("2"), 2 * 1024 ** 3)
        self.assertIsNone(parse_size("all"))
        with self.assertRaises(ValueError):
            parse_size("0GB")
        with self.assertRaises(ValueError):
            parse_size("abc")

    def test_scan_and_lru_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            s1 = _make_snapshot(base, "inkling2-mini", "aaa111", 2 * 1024 * 1024, 1000.0)
            s2 = _make_snapshot(base, "inkling2-mini", "bbb222", 3 * 1024 * 1024, 2000.0)
            s3 = _make_snapshot(base, "inkling2-mini", "ccc333", 4 * 1024 * 1024, 3000.0)
            # Named user session outside prefix-cache must never be touched
            named = base / "inkling2-mini" / "sessions" / "my-saved-chat"
            named.mkdir(parents=True)
            (named / "transcript.json").write_text("{}", encoding="utf-8")

            report = storage_report(base)
            self.assertEqual(report["snapshots"], 3)
            self.assertIn("inkling2-mini", report["by_model"])
            self.assertGreaterEqual(report["total_bytes"], 9 * 1024 * 1024)

            # Dry run leaves everything intact
            dry = claim_cache(base, 4 * 1024 * 1024, dry_run=True)
            self.assertEqual(dry["removed_snapshots"], 2)
            self.assertTrue(s1.is_dir() and s2.is_dir() and s3.is_dir())

            # Claim 4 MB -> evicts s1 (2 MB) + s2 (3 MB), keeps newest s3 (4 MB) and named session
            res = claim_cache(base, 4 * 1024 * 1024, dry_run=False)
            self.assertEqual(res["removed_snapshots"], 2)
            self.assertFalse(s1.exists())
            self.assertFalse(s2.exists())
            self.assertTrue(s3.is_dir())
            self.assertTrue((named / "transcript.json").is_file())

            # Claim all -> removes remaining prefix-cache snapshot, still keeps named session
            res_all = claim_cache(base, None, dry_run=False)
            self.assertEqual(res_all["removed_snapshots"], 1)
            self.assertEqual(res_all["remaining_bytes"], 0)
            self.assertEqual(len(scan_cache(base)), 0)
            self.assertTrue((named / "transcript.json").is_file())

    def test_format_bytes(self):
        self.assertEqual(format_bytes(0), "0 B")
        self.assertEqual(format_bytes(2 * 1024 ** 2), "2.0 MB")
        self.assertEqual(format_bytes(3 * 1024 ** 3), "3.00 GB")

    def test_vxkv2_cas_gc_and_phase2_kv_policy(self):
        import struct
        from veystrix.kv_policy import KVGeometry, MODES, kv_bytes
        from veystrix.tool.storage import gc_cas_pages

        # Verify Phase 2 ultra-low-bit modes shrink 1M KV footprint monotonically
        geom = KVGeometry(attention_layers=8, kv_heads=8, head_dim=128, sliding_layers=40, sliding_window=1024)
        self.assertIn("sparse-paged-k4v2", MODES)
        self.assertIn("sparse-paged-k2v2", MODES)
        b_int4 = kv_bytes(geom, "sparse-paged", 1_000_000)
        b_k4v2 = kv_bytes(geom, "sparse-paged-k4v2", 1_000_000)
        b_k2v2 = kv_bytes(geom, "sparse-paged-k2v2", 1_000_000)
        self.assertLess(b_k4v2, b_int4)
        self.assertLess(b_k2v2, b_k4v2)

        # Verify CAS page GC retains pages referenced by live VXKV2 manifests and removes orphans
        with tempfile.TemporaryDirectory() as tmp:
            pc_root = Path(tmp) / "inkling2-mini" / "sessions" / "prefix-cache"
            cas_dir = pc_root / ".cas_pages" / "aa"
            cas_dir.mkdir(parents=True)
            live_digest = "a" * 64
            orphan_digest = "b" * 64
            (cas_dir / f"{live_digest}.vxpg").write_bytes(b"live-page-bytes")
            (cas_dir / f"{orphan_digest}.vxpg").write_bytes(b"orphan-page-bytes")

            snap = pc_root / "snap1" / "vx_snapshot"
            snap.mkdir(parents=True)
            # Minimal valid VXKV2 manifest referencing `live_digest`
            mf = bytearray(b"VXKV2\x00\x00\x00")
            mf += struct.pack("<III", 2, 256, 0)  # version=2, page=256, n_meta=0
            mf += struct.pack("<I", 1)            # n_tensors=1
            tname = b"l0.middle.k0"
            mf += struct.pack("<I", len(tname)) + tname
            mf += struct.pack("<BB", 3, 4)        # UInt32, ndim=4
            mf += struct.pack("<qqqq", 1, 8, 256, 16)
            mf += struct.pack("<BI", 1, 1)        # paged=1, npages=1
            mf += struct.pack("<II", 0, 256) + live_digest.encode("ascii")
            (snap / "snapshot.vxkv2").write_bytes(bytes(mf))

            # scan_cache must ignore .cas_pages as a snapshot dir
            entries = scan_cache(Path(tmp))
            self.assertEqual(len(entries), 1)

            removed, reclaimed = gc_cas_pages(pc_root)
            self.assertEqual(removed, 1)
            self.assertEqual(reclaimed, len(b"orphan-page-bytes"))
            self.assertTrue((cas_dir / f"{live_digest}.vxpg").is_file())
            self.assertFalse((cas_dir / f"{orphan_digest}.vxpg").exists())


if __name__ == "__main__":
    unittest.main()
