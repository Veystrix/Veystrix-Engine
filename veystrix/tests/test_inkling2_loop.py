"""Unit tests for Inkling 2 Mini (inkling2_loop) integration in Veystrix.

Covers:
  1. Model identity registration (family, model_types, 1M context ceiling, generic C++ runtime).
  2. veystrix-recipe/3 generation from Inkling 2 Mini config + NVFP4/affine safetensors headers.
  3. 3-pass looped core geometry (24 physical layers -> 48 visits: 40 sliding-window + 8 full-attn).
  4. 1M-token KV policy budgeting: 1,024-token sliding-window cap (~160 MiB BF16) + sparse-paged
     mode fitting 1,048,576 tokens comfortably inside a 16 GB Mac KV budget.
"""
from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from veystrix import kv_policy, model_identity, recipe


def _write_fake_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    header = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset]}
    raw = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)


def _inkling2_config() -> dict:
    local_ids = [i for i in range(24) if i not in (5, 11, 17, 23)]
    quant = {"group_size": 64, "bits": 4, "mode": "affine",
             "model.embed": {"bits": 8, "group_size": 64, "mode": "affine"},
             "lm_head": {"bits": 8, "group_size": 64, "mode": "affine"}}
    for l in range(24):
        if l < 20:
            quant[f"model.layers.{l}.mlp.experts.w13"] = {"bits": 4, "group_size": 16, "mode": "nvfp4"}
            quant[f"model.layers.{l}.mlp.experts.w2"] = {"bits": 4, "group_size": 16, "mode": "nvfp4"}
    return {
        "model_type": "inkling2_loop",
        "text_config": {
            "model_type": "inkling2_loop_text",
            "model_max_length": 1_048_576,
            "hidden_size": 2304,
            "num_hidden_layers": 24,
            "vocab_size": 201024,
            "unpadded_vocab_size": 200058,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "d_rel": 16,
            "rel_extent": 1024,
            "sliding_window_size": 1024,
            "local_layer_ids": local_ids,
            "n_layers_per_attn_res_block": 6,
            "n_routed_experts": 128,
            "num_experts_per_tok": 8,
            "intermediate_size": 1152,
            "final_logit_softcapping": 30.0,
            "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64},
        },
        "mtp_config": {"num_hidden_layers": 3, "target_layer_ids": [5, 23, 41, 48]},
        "quantization": quant,
    }


def _inkling2_headers() -> dict[str, tuple[str, list[int]]]:
    t: dict[str, tuple[str, list[int]]] = {
        "model.embed.weight": ("U32", [201024, 576]),
        "model.embed.scales": ("BF16", [201024, 36]),
        "lm_head.weight": ("U32", [201024, 576]),
        "lm_head.scales": ("BF16", [201024, 36]),
        "model.norm.weight": ("BF16", [2304]),
    }
    for l in range(24):
        if l < 20:
            t[f"model.layers.{l}.mlp.experts.w13.weight"] = ("U32", [128, 2304, 288])
            t[f"model.layers.{l}.mlp.experts.w13.scales"] = ("U8", [128, 2304, 144])
            t[f"model.layers.{l}.mlp.experts.w13.global_scale"] = ("F32", [128])
            t[f"model.layers.{l}.mlp.experts.w2.weight"] = ("U32", [128, 2304, 144])
            t[f"model.layers.{l}.mlp.experts.w2.scales"] = ("U8", [128, 2304, 72])
            t[f"model.layers.{l}.mlp.experts.w2.global_scale"] = ("F32", [128])
        else:
            t[f"model.layers.{l}.mlp.experts.w13.weight"] = ("U32", [128, 2304, 288])
            t[f"model.layers.{l}.mlp.experts.w13.scales"] = ("BF16", [128, 2304, 36])
            t[f"model.layers.{l}.mlp.experts.w2.weight"] = ("U32", [128, 2304, 144])
            t[f"model.layers.{l}.mlp.experts.w2.scales"] = ("BF16", [128, 2304, 18])
    return t


class Inkling2LoopTests(unittest.TestCase):
    def test_identity_registered(self):
        ident = model_identity.for_model_type("inkling2_loop")
        self.assertIsNotNone(ident)
        assert ident is not None
        self.assertEqual(ident.family, "inkling2-mini")
        self.assertEqual(ident.context_limit, 1_048_576)
        self.assertEqual(ident.cpp_runtime, "generic")
        self.assertEqual(ident.download_repo, "mlx-community/Inkling-2-Mini-oQ4e-mtp")

    def test_recipe_emits_3_pass_48_visits_and_nvfp4(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(_inkling2_config()))
            _write_fake_safetensors(root / "model-00001-of-00001.safetensors", _inkling2_headers())
            rec = recipe.emit_recipe(root)

        self.assertEqual(rec["format"], "veystrix-recipe/3")
        self.assertEqual(rec["model_type"], "inkling2_loop")
        self.assertEqual(rec["loop"]["passes"], 3)
        self.assertEqual(rec["loop"]["core_start"], 6)
        self.assertEqual(rec["loop"]["core_end"], 18)
        self.assertEqual(rec["residual"]["type"], "block_attn_res")
        self.assertTrue(rec["mtp"]["present"])
        self.assertEqual(rec["mtp"]["draft_type"], "dspark")
        self.assertEqual(rec["mtp"]["num_hidden_layers"], 3)
        self.assertEqual(rec["mtp"]["block_size"], 8)
        self.assertEqual(rec["mtp"]["markov_rank"], 256)
        self.assertEqual(rec["mtp"]["target_layer_ids"], [5, 23, 41, 48])
        self.assertEqual(rec["blocks"]["attn_local"]["sliding_window"], 1024)
        self.assertEqual(rec["blocks"]["attn_full"]["sliding_window"], 0)
        self.assertEqual(rec["blocks"]["moe"]["type"], "moe.inkling2_sigmoid")

        # Verify KV geometry counts 8 full-attention visits and 40 sliding-window visits
        geom = kv_policy.KVGeometry.from_recipe(rec)
        self.assertEqual(geom.attention_layers, 8)
        self.assertEqual(geom.sliding_layers, 40)
        self.assertEqual(geom.sliding_window, 1024)

    def test_1m_context_fits_16gb_ram_with_sparse_paged(self):
        geom = kv_policy.KVGeometry(
            attention_layers=8, kv_heads=8, head_dim=128, sliding_layers=40, sliding_window=1024
        )
        # 16 GB Mac with ~2.5 GiB resident dense weights -> ~3.3 GiB KV budget
        budget_16g = kv_policy.kv_budget(16 * kv_policy.GIB, int(2.5 * kv_policy.GIB))
        # BF16 at 1M tokens is ~32.2 GiB (far over 16 GB)
        bf16_1m = kv_policy.kv_bytes(geom, "bf16", 1_048_576)
        self.assertGreater(bf16_1m, 30 * kv_policy.GIB)
        # sparse-paged keeps 1K exact prefix + 8K exact recent + 64 MiB page index + 64K hot LRU
        # + 160 MiB sliding-window cap -> ~1.1 GiB total RAM at 1M tokens!
        sp_1m = kv_policy.kv_bytes(geom, "sparse-paged", 1_048_576)
        self.assertLess(sp_1m, budget_16g)
        self.assertLess(sp_1m, int(1.5 * kv_policy.GIB))

        kv_plan = kv_policy.plan(geom, 1_048_576, budget_16g, int4_allowed=True)
        self.assertEqual(kv_plan.mode, "sparse-paged")
        self.assertLessEqual(kv_plan.bytes_at_context, budget_16g)


if __name__ == "__main__":
    unittest.main()
