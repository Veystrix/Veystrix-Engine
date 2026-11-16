import unittest
from unittest import mock

from veystrix.tests import _linux


class LinuxDiscoveryTests(unittest.TestCase):
    def test_finds_shared_test_modules(self):
        names = _linux.test_module_names()
        self.assertIn("veystrix.tests.test_model_identity", names)
        self.assertIn("veystrix.tests.test_checkpoints", names)
        self.assertTrue(all(name.startswith("veystrix.tests.test_") for name in names))

    def test_mlx_import_error_is_a_skip(self):
        failure = ModuleNotFoundError("no mlx", name="mlx.core")
        with mock.patch.object(_linux.importlib, "import_module", side_effect=failure):
            module, reason = _linux.import_test_module("veystrix.tests.test_server")
        self.assertIsNone(module)
        self.assertIn("mlx.core", reason)

    def test_unrelated_import_error_is_reraised(self):
        failure = ModuleNotFoundError("no dep", name="some_other_package")
        with mock.patch.object(_linux.importlib, "import_module", side_effect=failure):
            with self.assertRaises(ModuleNotFoundError):
                _linux.import_test_module("veystrix.tests.test_server")

    def test_successful_import_returns_the_module(self):
        module, reason = _linux.import_test_module("veystrix.tests.test_model_identity")
        self.assertIsNotNone(module)
        self.assertIsNone(reason)

    def test_runtime_mlx_modules_are_declared_and_pure_ones_are_not(self):
        self.assertTrue(_linux.needs_mlx_at_runtime("veystrix.tests.test_session"))
        self.assertFalse(_linux.needs_mlx_at_runtime("veystrix.tests.test_model_identity"))
        self.assertTrue(all(name.startswith("veystrix.tests.")
                            for name in _linux.RUNTIME_MLX_MODULES))


if __name__ == "__main__":
    unittest.main()
