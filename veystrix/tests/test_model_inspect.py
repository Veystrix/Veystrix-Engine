"""`veystrix models inspect`: architecture needs, engine coverage and memory fit (offline)."""
from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from veystrix import model_inspect


def write_checkpoint(directory: Path, config: dict, tensors: dict | None = None):
    (directory / "config.json").write_text(json.dumps(config))
    if tensors:
        header, offset = {}, 0
        for name, (shape, size) in tensors.items():
            header[name] = {"dtype": "U32", "shape": shape, "data_offsets": [offset, offset + size]}
            offset += size
        raw = json.dumps(header).encode()
        (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


class InspectTests(unittest.TestCase):
    def inspect(self, config, tensors=None):
        with tempfile.TemporaryDirectory() as tmp:
            write_checkpoint(Path(tmp), config, tensors)
            return model_inspect.inspect(tmp)

    def test_known_family_is_ready_on_both_engines(self):
        r = self.inspect({"model_type": "nemotron_h", "num_hidden_layers": 4, "n_routed_experts": 8,
                          "mlp_hidden_act": "relu2", "layers_block_type": ["mamba", "moe", "attention", "moe"],
                          "ssm_state_size": 128, "num_attention_heads": 8, "num_key_value_heads": 2,
                          "head_dim": 64, "quantization": {"bits": 4, "group_size": 64, "mode": "affine"}})
        self.assertEqual(r.known_family, "nemotron35-lightning")
        self.assertTrue(r.cpp_ready)
        self.assertIn("mamba2-ssm", r.needs)
        self.assertIn("moe-relu2-affine-q4", r.needs)
        self.assertEqual(r.kv_bytes_per_token["attention_layers"], 1)

    def test_unknown_architecture_lists_the_ports_it_needs(self):
        r = self.inspect({"model_type": "deepseek_v9", "num_hidden_layers": 10, "num_experts": 64,
                          "hidden_act": "silu", "kv_lora_rank": 512, "scoring_func": "sigmoid",
                          "sliding_window": 4096, "num_attention_heads": 16,
                          "quantization": {"bits": 4, "group_size": 64, "mode": "mxfp4"}})
        self.assertIsNone(r.known_family)
        self.assertFalse(r.cpp_ready)
        for primitive in ("mla-attention", "sigmoid-router-bias", "sliding-window-attention", "mxfp4-nvfp4"):
            self.assertIn(primitive, r.missing_cpp)

    def test_unknown_but_fully_covered_is_flagged_as_needing_an_adapter(self):
        r = self.inspect({"model_type": "new_moe", "num_hidden_layers": 4, "num_experts": 8, "hidden_act": "silu",
                          "num_attention_heads": 8, "quantization": {"bits": 4, "group_size": 32, "mode": "affine"}})
        self.assertEqual(r.missing_cpp, [])
        self.assertTrue(any("adapter" in n for n in r.notes))

    def test_memory_fit_covers_every_ram_size(self):
        r = self.inspect({"model_type": "new_moe", "num_hidden_layers": 2, "num_experts": 4, "hidden_act": "silu",
                          "num_attention_heads": 4},
                         {"m.layers.0.mlp.switch_mlp.up.weight": ([4, 8], 400_000_000),
                          "m.layers.1.mlp.switch_mlp.up.weight": ([4, 8], 400_000_000),
                          "m.embed.weight": ([8, 8], 10_000_000)})
        self.assertEqual([row["ram_gib"] for row in r.fit], [16, 24, 32, 48, 64, 96, 128])
        self.assertEqual(r.memory["experts_per_layer"], 4)
        self.assertEqual(r.fit[-1]["mode"], "resident")
        self.assertIn("weights", model_inspect.render(r))

    def test_missing_config_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                model_inspect.inspect(tmp)

    def test_cli_json(self):
        import io
        from contextlib import redirect_stdout
        from veystrix.tool.commands import cmd_models
        with tempfile.TemporaryDirectory() as tmp:
            write_checkpoint(Path(tmp), {"model_type": "x", "num_hidden_layers": 1, "num_attention_heads": 2})
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(cmd_models(["inspect", tmp, "--json"], Path(tmp), None), 0)
        self.assertEqual(json.loads(out.getvalue())["model_type"], "x")


if __name__ == "__main__":
    unittest.main()
