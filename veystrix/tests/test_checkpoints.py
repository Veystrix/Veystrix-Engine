from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from veystrix import checkpoints, model_identity
from veystrix.checkpoints import installed_inkling2_mini, resolve_inkling2_mini


def make_checkpoint(root: Path, name: str, model_type: str = "inkling2_loop") -> Path:
    path = root / name
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"weight": "model-00001-of-00001.safetensors"}
    }))
    (path / "model-00001-of-00001.safetensors").touch()
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        (path / filename).touch()
    return path


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._environment = patch.dict(os.environ, {}, clear=False)
        self._environment.start()
        for name in (
            "VEYSTRIX_INKLING2_MODEL",
            "VEYSTRIX_MODEL_ROOT",
        ):
            os.environ.pop(name, None)
        self.addCleanup(self._environment.stop)
        self._recipe = patch("veystrix.recipe.emit_recipe", return_value={})
        self._recipe.start()
        self.addCleanup(self._recipe.stop)

    def test_auto_selects_the_only_complete_compatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = make_checkpoint(root, "custom")
            make_checkpoint(root, "wrong", "other")
            with patch.dict(os.environ, {"VEYSTRIX_MODEL_ROOT": str(root)}, clear=False):
                self.assertEqual(installed_inkling2_mini(), [expected])
                self.assertEqual(resolve_inkling2_mini(), expected.resolve())

    def test_partial_download_is_not_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = make_checkpoint(root, "partial")
            (path / "model-00001-of-00001.safetensors").unlink()
            with patch.dict(os.environ, {"VEYSTRIX_MODEL_ROOT": str(root)}, clear=False):
                self.assertEqual(installed_inkling2_mini(), [])

    def test_partial_download_explains_how_to_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = make_checkpoint(root, "partial")
            (path / "model-00001-of-00001.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "model-00001-of-00001.safetensors"):
                resolve_inkling2_mini(path)

    def test_multiple_checkpoints_require_an_explicit_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_checkpoint(root, "first")
            make_checkpoint(root, "second")
            with patch.dict(os.environ, {"VEYSTRIX_MODEL_ROOT": str(root)}, clear=False):
                with self.assertRaisesRegex(ValueError, "choose one with --checkpoint"):
                    resolve_inkling2_mini()

    def test_alias_resolves_to_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = make_checkpoint(root, model_identity.INKLING2_MINI.checkpoint_directory)
            with patch.dict(os.environ, {"VEYSTRIX_MODEL_ROOT": str(root)}, clear=False):
                self.assertEqual(
                    resolve_inkling2_mini(model_identity.INKLING2_MINI.checkpoint_alias),
                    expected.resolve(),
                )

    def test_identity_coupling_for_day1_future_models(self):
        from dataclasses import replace

        custom = replace(
            model_identity.INKLING2_MINI,
            family="future-model",
            model_types=frozenset({"brand_new_type"}),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            future = make_checkpoint(root, "future", model_type="brand_new_type")
            legacy = make_checkpoint(root, "legacy", model_type="inkling2_loop")
            self.assertTrue(checkpoints.generic_compatible(future, custom))
            self.assertFalse(checkpoints.generic_compatible(legacy, custom))


if __name__ == "__main__":
    unittest.main()
