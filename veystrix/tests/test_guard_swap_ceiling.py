"""Hard 5 GB swap ceiling: guard flag + memory policy default."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from models.flashnext import memory_policy


class SwapCeilingTests(unittest.TestCase):
    def test_guard_exposes_absolute_swap_ceiling_default_5gib(self):
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            [sys.executable, str(root / "scripts/flashnext/real_run_guard.py"), "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(out.returncode, 0)
        self.assertIn("--swap-ceiling-mib", out.stdout)
        self.assertIn("5120", out.stdout)

    def test_memory_policy_default_is_5gib(self):
        self.assertEqual(memory_policy.ABSOLUTE_SWAP_CEILING_MIB, 5120)
        self.assertEqual(memory_policy.absolute_swap_ceiling_mib({}), 5120)


if __name__ == "__main__":
    unittest.main(verbosity=2)
