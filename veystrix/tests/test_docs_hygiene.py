"""Guards for the active documentation index."""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


class DocsHygieneTests(unittest.TestCase):
    def test_index_links_resolve(self):
        index = (DOCS / "README.md").read_text()
        active = 0
        for target in re.findall(r"\]\(([^)#]+\.md)\)", index):
            if "://" in target:
                continue
            p = DOCS / target
            if p.is_file():
                active += 1
        self.assertGreater(active, 0)

    def test_no_pre_public_audit_or_sparkle_files_remain(self):
        for removed in (
            ROOT / "LICENSES" / "MIT-upstream.txt",
            ROOT / "LICENSES" / "Sparkle-2.10.1.txt",
            ROOT / "legal" / "LICENSE_PROVENANCE.md",
            DOCS / "release" / "FREE_CHANNEL.md",
            DOCS / "release" / "RELEASE_PROCESS.md",
            DOCS / "security" / "SECURITY_AUDIT_2026-11-10.md",
        ):
            self.assertFalse(removed.exists(), f"obsolete file still present: {removed}")


if __name__ == "__main__":
    unittest.main()
