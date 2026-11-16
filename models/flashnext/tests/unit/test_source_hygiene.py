"""Packaged builds must not ship readable source, and must not need a compiler."""
from __future__ import annotations

import py_compile
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from models.flashnext import native_read
from models.flashnext.settings.registry import get_registry

ROOT = Path(__file__).resolve().parents[4]


class SourceHygieneTests(unittest.TestCase):
    def test_settings_registry_loads_pyc_only_providers(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = ROOT / "models" / "flashnext" / "settings" / "setting_runtime.py"
            shutil.copy(source, folder / source.name)
            py_compile.compile(str(folder / source.name),
                               cfile=str(folder / "setting_runtime.pyc"), doraise=True)
            (folder / source.name).unlink()
            registry = get_registry(folder)
        self.assertTrue(registry.settings)
        self.assertTrue(any(s.env_key for s in registry.settings))

    def test_bundled_native_library_is_used_without_a_compiler(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / "native_read.bundled.dylib"
            library.touch()
            with mock.patch.object(native_read, "BUNDLED_LIBRARY", library), \
                 mock.patch.object(native_read, "_open_library", return_value="handle") as open_library, \
                 mock.patch.object(native_read.subprocess, "run") as compile:
                self.assertEqual(native_read._load_library(), "handle")
            open_library.assert_called_once_with(library)
            compile.assert_not_called()

    def test_build_script_removes_every_py_and_c_from_the_bundle(self):
        script = (ROOT / "packaging" / "engine" / "build_engine_bundle.sh").read_text()
        self.assertIn('find "$OUT/lib" -name \'*.py\' -delete', script)
        self.assertIn('find "$OUT/lib" -name \'*.c\' -delete', script)
        self.assertIn("source files remain in bundle", script)
        self.assertNotIn("! -path '*/settings/setting_*.py'", script)
        self.assertIn("native_read.bundled.dylib", script)


if __name__ == "__main__":
    unittest.main()
