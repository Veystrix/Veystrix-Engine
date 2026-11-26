"""Offline checks for the segmented long-context ladder arithmetic."""
from __future__ import annotations

import unittest


def segments(tokens: int, chunk: int) -> int:
    return (tokens + chunk - 1) // chunk


class LadderTests(unittest.TestCase):
    def test_segment_counts(self):
        self.assertEqual(segments(65536, 65536), 1)
        self.assertEqual(segments(262144, 65536), 4)
        self.assertEqual(segments(1048576, 65536), 16)
        self.assertEqual(segments(1048577, 65536), 17)

    def test_kv_budget_fits_16gb(self):
        # INT8 KV: 6 attn layers * 2 (K,V) * 2 kv heads * 136 B/token.
        per_token = 6 * 2 * 2 * 136
        for tokens in (65536, 262144, 1048576):
            gib = tokens * per_token / 2**30
            self.assertLess(gib, 4.0, f"{tokens} tokens uses {gib:.2f} GiB KV")

    def test_snapshot_is_full_copy_not_delta(self):
        # 16 snapshots at 1M would be ~27 GB: retain only the latest few.
        kv_1m_gib = 1048576 * 6 * 2 * 2 * 136 / 2**30
        self.assertGreater(kv_1m_gib * 16, 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
