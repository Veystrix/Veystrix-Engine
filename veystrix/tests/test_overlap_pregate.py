"""Offline test for the overlap pre-gate bound (no profile file needed)."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _mod():
    p = Path(__file__).resolve().parents[2] / "cpp/engine/tools/overlap_pregate.py"
    spec = importlib.util.spec_from_file_location("overlap_pregate", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class OverlapPregateTests(unittest.TestCase):
    def setUp(self):
        self.m = _mod()

    def test_host_work_can_hide_ssd_read_across_sessions(self):
        # G=3, S=90, H=60 -> sequential 153, overlapped max(90,63)=90 -> 1.7x
        self.assertAlmostEqual(self.m.bound(3, 90, 60), 153 / 90, places=3)
        self.assertGreaterEqual(self.m.bound(3, 90, 60), 1.15)

    def test_conservative_bound_is_lower(self):
        self.assertLess(self.m.conservative_bound(3, 90, 60),
                        self.m.bound(3, 90, 60))

    def test_gpu_bound_when_ssd_is_fast(self):
        # Fully resident: S=0, G dominates -> bound ~1 (no overlap benefit).
        self.assertAlmostEqual(self.m.bound(50, 0, 5), 1.0, places=3)

    def test_step_parts_from_profile_row(self):
        g, s, h = self.m.step_parts(dict(final_eval_ms=2, moe_eval_ms=1,
                                         read_wall_ms=90, cpu_build_ms=50,
                                         route_sync_ms=5, sample_ms=2, digest_ms=3))
        self.assertEqual((g, s, h), (3, 90, 60))


if __name__ == "__main__":
    unittest.main(verbosity=2)
