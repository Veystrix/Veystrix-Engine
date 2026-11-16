"""Malformed checkpoint metadata must be refused, never crash an inspector.

No weights, no MLX: these tests write small JSON files into a temp directory
and exercise every metadata reader with deterministic random shapes.
"""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from models.flashnext import checkpoint_compat, preflight
from veystrix import checkpoints

CONFIG_KEYS = ("model_type", "text_config", "llm_config", "num_hidden_layers",
               "hidden_size", "num_experts", "num_experts_per_tok", "moe_intermediate_size",
               "vocab_size", "layer_types", "quantization", "norm_convention",
               "ngram_vocab_size", "max_position_embeddings")
WEIGHT_KEYS = ("language_model.mtp.fc_embedding.weight", "weight", "../escape", "")


def _scalar(rng: random.Random):
    return rng.choice([None, True, False, 0, -1, 10 ** 30, 1.5, "", "qwen4_exp",
                       "qwen4_exp_text", "x" * 800, [], [1, 2], {}, {"a": "b"}])


def _value(rng: random.Random, depth: int = 0):
    if depth >= 2 or rng.random() < 0.5:
        return _scalar(rng)
    if rng.random() < 0.5:
        return {rng.choice(CONFIG_KEYS): _value(rng, depth + 1) for _ in range(rng.randrange(4))}
    return [_value(rng, depth + 1) for _ in range(rng.randrange(4))]


def _config(rng: random.Random) -> dict:
    return {key: _value(rng) for key in rng.sample(CONFIG_KEYS, rng.randrange(1, 7))}


def _index(rng: random.Random) -> dict:
    return {"weight_map": {rng.choice(WEIGHT_KEYS) + str(i): rng.choice(["a.safetensors", "../b", 5])
                           for i in range(rng.randrange(1, 5))}}


class ConfigFuzzTests(unittest.TestCase):
    def test_metadata_inspectors_survive_malformed_input(self):
        rng = random.Random(20261116)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ckpt"
            root.mkdir()
            for case in range(80):
                config, index = _config(rng), _index(rng)
                (root / "config.json").write_text(json.dumps(config))
                (root / "model.safetensors.index.json").write_text(json.dumps(index))
                for name, call in (
                    ("describe_checkpoint", lambda: checkpoint_compat.describe_checkpoint(root)),
                    ("describe_checkpoint_dict", lambda: preflight.describe_checkpoint_dict(config)),
                    ("check_checkpoint", lambda: preflight.check_checkpoint(root)),
                    ("flashnext_compatible", lambda: checkpoints.flashnext_compatible(root)),
                ):
                    with self.subTest(case=case, inspector=name):
                        call()  # any uncaught exception is a failure

    def test_valid_config_still_reports_compatible(self):
        config = {"model_type": "qwen4_exp", "text_config": {"model_type": "qwen4_exp_text"}}
        self.assertTrue(preflight.describe_checkpoint_dict(config)["compatible"])

    def test_identity_gate_follows_the_descriptor(self):
        from dataclasses import replace
        from unittest import mock
        from veystrix import model_identity

        future = replace(model_identity.FLASHNEXT,
                         model_types=frozenset({"qwen4_new"}))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({"model_type": "qwen4_new"}))
            with mock.patch.object(checkpoint_compat.model_identity, "FLASHNEXT", future):
                self.assertTrue(checkpoint_compat.describe_checkpoint(root).compatible)


if __name__ == "__main__":
    unittest.main()
