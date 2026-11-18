"""CPU-only unit tests for the per-session state calculator.

No MLX, no benchmark, no checkpoint: every expected number below is hand
arithmetic on a tiny config, written out next to the assertion.
"""
from __future__ import annotations

import unittest

from models.shared.tests.bench.session_state_mb import (
    MB,
    PRODUCTION_CONFIGS,
    session_state_mb,
    state_breakdown,
)


TINY = {
    # 4 layers: 2 recurrent/GDN (or mamba) and 2 full-attention layers.
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention", "full_attention",
                    "linear_attention", "full_attention"],
    "layers_block_type": ["mamba", "attention", "mamba", "attention"],
    # Qwen GDN: Hk=1, Dk=4, Hv=2, Dv=4, conv kernel=3, 2 kv heads, head_dim=8.
    "linear_num_key_heads": 1,
    "linear_key_head_dim": 4,
    "linear_num_value_heads": 2,
    "linear_value_head_dim": 4,
    "linear_conv_kernel_dim": 3,
    "num_key_value_heads": 2,
    "head_dim": 8,
    # Nemotron Mamba: H=2, D=4, S=8, G=2, conv kernel=3; same kv heads/head_dim.
    "mamba_num_heads": 2,
    "mamba_head_dim": 4,
    "ssm_state_size": 8,
    "n_groups": 2,
    "conv_kernel": 3,
}
CTX = 4096


class SessionStateMbTests(unittest.TestCase):
    def test_qwen_tiny_matches_hand_arithmetic(self):
        # conv_dim = 2*Hk*Dk + Hv*Dv = 2*1*4 + 2*4 = 16
        # conv/layer = (3-1)*16*2 B (bf16) = 64
        # recurrent/layer = Hv*Dv*Dk*4 B (f32) = 2*4*4*4 = 128
        # GDN total = 2 layers * (64+128) = 384 B
        # KV/full layer = 2*4096*2*8*2 B (bf16) = 262_144 B; 2 layers => 524_288 B
        # total = 384 + 524_288 = 524_672 B = 0.524672 MB
        b = state_breakdown("qwen", CTX, TINY)
        self.assertEqual(b["conv_dim"], 16)
        self.assertEqual(b["gdn_conv_bytes"], 2 * 64)
        self.assertEqual(b["gdn_recurrent_bytes"], 2 * 128)
        self.assertEqual(b["fixed_bytes"], 384)
        self.assertEqual(b["kv_bytes"], 524_288)
        self.assertEqual(b["total_bytes"], 524_672)
        self.assertAlmostEqual(session_state_mb("qwen", CTX, TINY), 524_672 / MB)

    def test_nemotron_tiny_matches_hand_arithmetic(self):
        # conv_dim = H*D + 2*G*S = 2*4 + 2*2*8 = 40
        # conv/mamba = (3-1)*40*2 B (bf16 assumption) = 160
        # SSM/mamba = H*D*S*4 B (f32) = 2*4*8*4 = 256
        # Mamba total = 2 layers * (160+256) = 832 B
        # KV/attn layer = 2*4096*2*8*1 B (INT8) = 131_072 B; 2 layers => 262_144 B
        # total = 832 + 262_144 = 262_976 B = 0.262976 MB
        b = state_breakdown("nemotron", CTX, TINY)
        self.assertEqual(b["conv_dim"], 40)
        self.assertEqual(b["mamba_conv_bytes"], 2 * 160)
        self.assertEqual(b["mamba_ssm_bytes"], 2 * 256)
        self.assertEqual(b["fixed_bytes"], 832)
        self.assertEqual(b["kv_bytes"], 262_144)
        self.assertEqual(b["total_bytes"], 262_976)
        self.assertAlmostEqual(session_state_mb("nemotron", CTX, TINY), 262_976 / MB)

    def test_kv_part_grows_linearly_with_context(self):
        # The fixed recurrent state must not change; only KV scales with ctx.
        # Qwen KV bytes/token = 2 full layers * 2 (K+V) * 2 kv heads * 8 * 2 B = 128
        # Nemotron KV bytes/token = 2 attn layers * 2 (K+V) * 2 * 8 * 1 B = 64
        for model, per_token in (("qwen", 128), ("nemotron", 64)):
            small = state_breakdown(model, 4096, TINY)
            large = state_breakdown(model, 8192, TINY)
            self.assertEqual(small["fixed_bytes"], large["fixed_bytes"])
            self.assertEqual(large["kv_bytes"] - small["kv_bytes"], 4096 * per_token)
            delta_mb = session_state_mb(model, 8192, TINY) - session_state_mb(model, 4096, TINY)
            self.assertAlmostEqual(delta_mb * MB, 4096 * per_token)

    def test_ordering_at_4k_16k_64k_128k(self):
        for model, config in (("qwen", TINY), ("nemotron", TINY),
                              ("qwen", PRODUCTION_CONFIGS["qwen"]),
                              ("nemotron", PRODUCTION_CONFIGS["nemotron"])):
            sizes = [session_state_mb(model, ctx, config)
                     for ctx in (4096, 16_384, 65_536, 131_072)]
            self.assertEqual(sizes, sorted(sizes), f"{model}: {sizes}")
            self.assertTrue(all(a < b for a, b in zip(sizes, sizes[1:])))

    def test_production_qwen_fixed_state_matches_prefill_budget(self):
        # 36 layers * [(conv_kernel-1)*conv_dim*2 + Hv*Dv*Dk*4]
        # conv_dim = 2*16*128 + 48*128 = 10_240
        # = 36 * [3*10_240*2 + 48*128*128*4] = 36 * 3_207_168 = 115_458_048 B
        b = state_breakdown("qwen", 4096, PRODUCTION_CONFIGS["qwen"])
        self.assertEqual(b["gdn_conv_bytes"], 36 * 3 * 10_240 * 2)
        self.assertEqual(b["gdn_recurrent_bytes"], 36 * 48 * 128 * 128 * 4)
        self.assertEqual(b["fixed_bytes"], 115_458_048)

    def test_nemotron_int8_scale_bias_matches_repo_plan(self):
        # models/nemotron35/tests/unit/test_int8_attention.py:118:
        # 6 attn layers * rounded(1_048_576) * 2 * 2 kv heads *
        # (128 + 4*(128//64)) B = 6*1_048_576*2*2*136 = 3_422_552_064 B
        b = state_breakdown("nemotron", 1_048_576, PRODUCTION_CONFIGS["nemotron"],
                            int8_scale_bias=True)
        self.assertEqual(b["kv_bytes"], 3_422_552_064)

    def test_text_config_wrapper_is_equivalent(self):
        wrapped = {"text_config": TINY, "model_type": "qwen4_exp"}
        self.assertEqual(session_state_mb("qwen4_exp", CTX, wrapped),
                         session_state_mb("qwen", CTX, TINY))


if __name__ == "__main__":
    unittest.main()
