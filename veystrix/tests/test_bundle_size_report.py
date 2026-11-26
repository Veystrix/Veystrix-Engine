import tempfile
import unittest
from pathlib import Path

from scripts.packaging.bundle_size_report import directory_sizes, mib


class BundleSizeReportTests(unittest.TestCase):
    def test_sizes_are_summed_and_sorted_descending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "big").mkdir()
            (root / "small").mkdir()
            (root / "big" / "a.bin").write_bytes(b"x" * 2048)
            (root / "small" / "b.bin").write_bytes(b"x" * 10)
            sizes = directory_sizes(root)
        self.assertEqual([name for name, _ in sizes], ["big", "small"])
        self.assertEqual(dict(sizes)["big"], 2048)

    def test_missing_directory_yields_no_rows(self):
        self.assertEqual(directory_sizes(Path("/nonexistent/veystrix")), [])

    def test_mib_uses_binary_units(self):
        self.assertEqual(mib(1024 * 1024).strip(), "1.0 MiB")


if __name__ == "__main__":
    unittest.main()
