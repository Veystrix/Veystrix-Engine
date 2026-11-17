#!/usr/bin/env python3
"""Synthetic Python oracle for the BIT-EXACT batched prefill ctest.

Builds a TINY but REAL Qwen4Exp language stack (real mlx_vlm modules, seeded,
eval mode, M4 GPU) and feeds ALL S prompt tokens through each layer in ONE call
-- i.e. the production `language(ids, cache)` prefill path -- then dumps the
inputs, weights, initial caches and expected outputs as .npy.

WHAT IS REAL
------------
  * Qwen4ExpGatedDeltaNet   (full-sequence conv + fused Metal delta kernel)
  * Qwen4ExpAttention       (fused Metal MRoPE + dense causal SDPA, indexer=None)
  * Qwen4ExpGatedResidual   (attn hyper, mlp hyper, and final no-combine mixer)
  * Qwen3_5MoeSparseMoeBlock router + NON-ZERO dense shared expert
  * nn.Embedding + nn.QuantizedLinear lm_head
  * Qwen4ExpDecoderLayer.__call__ op order (language.py:744-772)

WHAT IS STUBBED (identically in Python and C++)
-----------------------------------------------
  * ROUTED experts: `mlp.switch_mlp(x, inds)` is replaced by a module returning
    EXACTLY ZERO [B,S,k,H].  Because the real router scores are normalized to
    sum to 1, `(0 * scores[...,None]).sum(-2)` is exactly 0, so the C++ MoE
    callback simply returns mx.zeros([S,H]).  The router itself and the shared
    expert are REAL and bit-exact; only the gather-QMM experts are stubbed.
  * PLE: absent (config.ple_layer_ids = []).  The C++ code wires PLE but the
    synthetic oracle does not exercise it.

RMSNORM CONVENTION
------------------
  patch_rmsnorm.apply() installs the one-centered gain (g = weight) for the
  hc_norms, matching C++ gated_residual's `w2d` gain.  The attention q/k norms
  are switched back to `1 + weight` (`_flashnext_one_centered = False`) to
  match C++ qsa_rms_norm.  GDN's gated norm uses mx.fast.rms_norm directly and
  is unaffected by the patch.

Usage:
    .venv/bin/python cpp/engine/tools/gen_prefill_batched_vectors.py \
        --out cpp/engine/tests/vectors_prefill_batched
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
VENV_SITE = "/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages"
for _p in (VENV_SITE, str(ROOT / "models" / "flashnext")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import patch_rmsnorm  # noqa: E402

patch_rmsnorm.apply()

from mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    Qwen4ExpDecoderLayer,
    Qwen4ExpGatedResidual,
    QSAKVCache,
)
from mlx_vlm.models.qwen3_5.language import ArraysCache  # noqa: E402

# --------------------------------------------------------------------------- #
# Geometry (kept deliberately tiny; GDN head dims must be multiples of 32 for
# the fused Metal delta kernel).
# --------------------------------------------------------------------------- #
H = 32
HC = 2
LOW = 16
S = 4
N_LAYERS = 2                  # layer 0 GDN, layer 1 full attention
Dk = Dv = 32
Hk, Hv = 1, 2
K = 4
E = 8
TOP_K = 2
GROUP = 32
BITS = 4
INTER = 32
VOCAB = 64
EPS = 1e-6

N_HEADS = 2
KV_HEADS = 1
HEAD_DIM = 16
ROPE_BASE = 10000.0
MROPE_SECTION = [6, 6, 4]
N_IDX_HEADS = 2
N_IDX_KV = 1
IDX_HEAD_DIM = 16
COMPRESS_RATIO = 2
INDEX_BUDGET = 64             # block_topk = 32 > S//2 -> indexer returns None for all S <= 64

KEY_DIM = Hk * Dk
VALUE_DIM = Hv * Dv
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
HC_HIDDEN = HC * H

LAYER_TYPES = ["linear_attention", "full_attention"]


def make_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=H,
        moe_intermediate_size=INTER,
        shared_expert_intermediate_size=INTER,
        num_experts=E,
        num_experts_per_tok=TOP_K,
        vocab_size=VOCAB,
        rms_norm_eps=EPS,
        hc_count=HC,
        hc_lowrank=LOW,
        linear_num_value_heads=Hv,
        linear_num_key_heads=Hk,
        linear_key_head_dim=Dk,
        linear_value_head_dim=Dv,
        linear_conv_kernel_dim=K,
        output_gate_type="sigmoid",
        hidden_act="silu",
        layer_types=list(LAYER_TYPES),
        ple_layer_ids=[],
        num_key_value_heads=KV_HEADS,
        num_attention_heads=N_HEADS,
        head_dim=HEAD_DIM,
        attention_bias=False,
        max_position_embeddings=2048,
        rope_parameters={
            "partial_rotary_factor": 1.0,
            "rope_theta": ROPE_BASE,
            "mrope_section": list(MROPE_SECTION),
        },
        indexer_n_heads=N_IDX_HEADS,
        indexer_kv_heads=N_IDX_KV,
        indexer_head_dim=IDX_HEAD_DIM,
        indexer_budget=INDEX_BUDGET,
        indexer_compress_ratio=COMPRESS_RATIO,
    )


class StubSwitchGLU(nn.Module):
    """Routed experts stub: exactly zero, regardless of input / indices."""

    def __init__(self, k: int, dim: int):
        super().__init__()
        self.k = k
        self.dim = dim

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        return mx.zeros((*x.shape[:-1], self.k, self.dim), dtype=x.dtype)


class StubMoeBlock:
    """Thin wrapper reproducing Qwen3_5MoeSparseMoeBlock.__call__ with a stubbed
    switch_mlp.  The router + shared expert remain the REAL modules."""

    def __init__(self, real_block):
        self.real = real_block
        real_block.switch_mlp = StubSwitchGLU(TOP_K, H)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.real.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-TOP_K, axis=-1)[..., -TOP_K:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        y = self.real.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        shared_y = self.real.shared_expert(x)
        shared_y = mx.sigmoid(self.real.shared_expert_gate(x)) * shared_y
        return y + shared_y


def make_weights(rng, cfg):
    """Seed every module with deterministic weights and return the stack."""
    def R(shape, scale=0.05, dtype=mx.float32):
        return mx.array(rng.standard_normal(shape) * scale).astype(dtype)

    layers = []
    for i, kind in enumerate(LAYER_TYPES):
        layer = Qwen4ExpDecoderLayer(cfg, i)
        # Hyper-connections: f32 weights (C++ gated_residual uses f32 matmul).
        for hc_name in ("attn_hyper_connection", "mlp_hyper_connection"):
            hc_mod = getattr(layer, hc_name)
            hc_mod.hc_norm.weight = R((HC_HIDDEN,), 0.02)
            hc_mod.input_mix_weight_down.weight = R((LOW, HC_HIDDEN))
            hc_mod.input_mix_weight_up.weight = R((HC_HIDDEN, LOW))
            hc_mod.block_inject_weight.weight = R((HC, HC_HIDDEN))
            # one-centered (g = weight), matching C++ gated_residual.
            hc_mod.hc_norm._flashnext_one_centered = True
        # GDN: bf16 weights, matching the proven gen_prefill_vectors.py recipe.
        if kind == "linear_attention":
            gdn = layer.linear_attn
            gdn.in_proj_qkv.weight = R((CONV_DIM, H), 0.1, mx.bfloat16)
            gdn.in_proj_z.weight = R((VALUE_DIM, H), 0.1, mx.bfloat16)
            gdn.in_proj_b.weight = R((Hv, H), 0.1, mx.bfloat16)
            gdn.in_proj_a.weight = R((Hv, H), 0.1, mx.bfloat16)
            gdn.conv1d.weight = R((CONV_DIM, K, 1), 0.1, mx.bfloat16)
            gdn.A_log = R((Hv,), 1.0)
            gdn.dt_bias = R((Hv,), 0.5)
            gdn.norm.weight = mx.array(np.ones(Dv, dtype=np.float32))
            gdn.out_proj.weight = R((H, VALUE_DIM), 0.1, mx.bfloat16)
        else:
            attn = layer.self_attn
            attn.q_proj.weight = R((N_HEADS * HEAD_DIM * 2, H))
            attn.k_proj.weight = R((KV_HEADS * HEAD_DIM, H))
            attn.v_proj.weight = R((KV_HEADS * HEAD_DIM, H))
            attn.o_proj.weight = R((H, N_HEADS * HEAD_DIM))
            attn.q_norm.weight = R((HEAD_DIM,), 0.02)
            attn.k_norm.weight = R((HEAD_DIM,), 0.02)
            # Production patches ALL Qwen4ExpRMSNorm to one-centered (g = weight).
            # Attention q_norm/k_norm are one-centered; C++ uses the one-centered form.
            attn.q_norm._flashnext_one_centered = True
            attn.k_norm._flashnext_one_centered = True
            idx = attn.indexer
            idx.index_qk_proj.weight = R(
                ((N_IDX_HEADS + N_IDX_KV) * IDX_HEAD_DIM, H))
            idx.q_layernorm.weight = R((IDX_HEAD_DIM,), 0.02)
            idx.k_layernorm.weight = R((IDX_HEAD_DIM,), 0.02)
            idx.q_layernorm._flashnext_one_centered = False
            idx.k_layernorm._flashnext_one_centered = False
        # MoE: real router + shared expert, stubbed routed experts.
        mlp = layer.mlp
        mlp.gate.weight = R((E, H))
        mlp.shared_expert.gate_proj.weight = R((INTER, H))
        mlp.shared_expert.up_proj.weight = R((INTER, H))
        mlp.shared_expert.down_proj.weight = R((H, INTER))
        mlp.shared_expert_gate.weight = R((1, H))
        layer.mlp = StubMoeBlock(mlp)
        layer.eval()
        layers.append(layer)

    mixer = Qwen4ExpGatedResidual(cfg, use_combine=False)
    mixer.hc_norm.weight = R((HC_HIDDEN,), 0.02)
    mixer.input_mix_weight_down.weight = R((LOW, HC_HIDDEN))
    mixer.input_mix_weight_up.weight = R((HC_HIDDEN, LOW))
    mixer.hc_norm._flashnext_one_centered = True
    mixer.eval()

    embed = nn.Embedding(VOCAB, H)
    embed.weight = R((VOCAB, H), 0.05)
    embed.eval()

    # Quantized lm_head (production Q4 path): H=32, group=32 -> 1 group.
    lm = nn.QuantizedLinear(H, VOCAB, group_size=GROUP, bits=BITS, bias=False)
    lm.eval()

    return layers, mixer, embed, lm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors_prefill_batched")
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--seq-len", type=int, default=S,
                    help="prompt length S fed through every layer in one call")
    args = ap.parse_args()
    seq_len = args.seq_len

    mx.set_default_device(mx.gpu)
    assert mx.metal.is_available(), "these vectors require the M4 GPU Metal path"
    assert 1 < seq_len <= 64, "indexer-None guarantee holds for S <= 64"

    rng = np.random.default_rng(args.seed)
    cfg = make_cfg()
    layers, mixer, embed, lm = make_weights(rng, cfg)

    tokens = mx.array(np.arange(seq_len, dtype=np.int32)[None, :] + 7)
    position_ids = mx.array(np.arange(seq_len, dtype=np.int32)[None, :])

    caches = []
    for kind in LAYER_TYPES:
        caches.append(ArraysCache(size=2) if kind == "linear_attention"
                      else QSAKVCache())

    # Production prefill: embed -> tile -> ONE call per layer over all S tokens.
    x = embed(tokens)  # [1,S,H] before tiling
    hidden = mx.tile(x, (1, 1, HC))
    expected_hidden = []
    expected_branch = []
    for i, layer in enumerate(layers):
        kind = LAYER_TYPES[i]
        mask = None
        hidden = layer(hidden, tokens, mask=mask, cache=caches[i],
                       position_ids=position_ids if kind == "full_attention" else None)
        mx.eval(hidden)
        expected_hidden.append(mx.reshape(hidden[:, -1, :], (1, HC_HIDDEN)))
    mixed = mixer(hidden)
    expected_mixed = mx.reshape(mixed[:, -1, :], (1, H))
    logits = lm(expected_mixed)  # [1, VOCAB]
    mx.eval(logits)
    expected_logits = mx.reshape(logits, (VOCAB,))

    assert bool(mx.all(mx.isfinite(expected_logits))), "non-finite logits"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def save(name: str, arr: mx.array) -> None:
        # mx.save handles bfloat16 (numpy's buffer protocol does not).
        mx.save(str(out / f"{name}.npy"), arr)

    save("tokens", tokens)
    save("position_ids", position_ids)
    save("x", x)
    save("embed_w", embed.weight)
    save("expected_logits", expected_logits)
    save("expected_mixed", expected_mixed)

    for i, layer in enumerate(layers):
        pre = f"L{i}"
        a_hc = layer.attn_hyper_connection
        m_hc = layer.mlp_hyper_connection
        save(f"{pre}_attn_norm_w", a_hc.hc_norm.weight)
        save(f"{pre}_attn_down_w", a_hc.input_mix_weight_down.weight)
        save(f"{pre}_attn_up_w", a_hc.input_mix_weight_up.weight)
        save(f"{pre}_attn_inject_w", a_hc.block_inject_weight.weight)
        save(f"{pre}_mlp_norm_w", m_hc.hc_norm.weight)
        save(f"{pre}_mlp_down_w", m_hc.input_mix_weight_down.weight)
        save(f"{pre}_mlp_up_w", m_hc.input_mix_weight_up.weight)
        save(f"{pre}_mlp_inject_w", m_hc.block_inject_weight.weight)
        save(f"{pre}_gate_w", layer.mlp.real.gate.weight)
        save(f"{pre}_shared_gw", layer.mlp.real.shared_expert.gate_proj.weight)
        save(f"{pre}_shared_uw", layer.mlp.real.shared_expert.up_proj.weight)
        save(f"{pre}_shared_dw", layer.mlp.real.shared_expert.down_proj.weight)
        save(f"{pre}_shared_glw", layer.mlp.real.shared_expert_gate.weight)
        save(f"{pre}_expected_hidden", expected_hidden[i])

        if LAYER_TYPES[i] == "linear_attention":
            g = layer.linear_attn
            for nm, w in (
                ("gdn_qkv", g.in_proj_qkv.weight),
                ("gdn_z", g.in_proj_z.weight),
                ("gdn_b", g.in_proj_b.weight),
                ("gdn_a", g.in_proj_a.weight),
                ("gdn_conv", g.conv1d.weight),
                ("gdn_A_log", g.A_log),
                ("gdn_dt_bias", g.dt_bias),
                ("gdn_norm", g.norm.weight),
                ("gdn_out", g.out_proj.weight),
            ):
                save(f"{pre}_{nm}", w)
            save(f"{pre}_gdn_conv_state_init",
                 mx.zeros((1, K - 1, CONV_DIM), mx.float32))
            save(f"{pre}_gdn_rec_state_init",
                 mx.zeros((1, Hv, Dv, Dk), mx.float32))
            save(f"{pre}_gdn_conv_state_final", caches[i][0])
            save(f"{pre}_gdn_rec_state_final", caches[i][1])
        else:
            attn = layer.self_attn
            for nm, w in (
                ("fa_q_w", attn.q_proj.weight),
                ("fa_k_w", attn.k_proj.weight),
                ("fa_v_w", attn.v_proj.weight),
                ("fa_o_w", attn.o_proj.weight),
                ("fa_q_norm_w", attn.q_norm.weight),
                ("fa_k_norm_w", attn.k_norm.weight),
                ("fa_index_qk_w", attn.indexer.index_qk_proj.weight),
                ("fa_index_q_norm_w", attn.indexer.q_layernorm.weight),
                ("fa_index_k_norm_w", attn.indexer.k_layernorm.weight),
            ):
                save(f"{pre}_{nm}", w)

    save("head_norm_w", mixer.hc_norm.weight)
    save("head_down_w", mixer.input_mix_weight_down.weight)
    save("head_up_w", mixer.input_mix_weight_up.weight)
    save("head_lm_w", lm.weight)
    save("head_lm_s", lm.scales)
    save("head_lm_b", lm.biases)

    meta = dict(
        generator="gen_prefill_batched_vectors.py",
        seed=args.seed,
        H=H, HC=HC, LOW=LOW, S=seq_len, n_layers=N_LAYERS,
        hk=Hk, hv=Hv, dk=Dk, dv=Dv, conv_kernel=K,
        n_heads=N_HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM,
        n_idx_heads=N_IDX_HEADS, n_idx_kv=N_IDX_KV, idx_head_dim=IDX_HEAD_DIM,
        compress_ratio=COMPRESS_RATIO, token_budget=INDEX_BUDGET,
        rope_dim=HEAD_DIM, rope_base=ROPE_BASE,
        rope_style="interleaved", rope_mrope_section=MROPE_SECTION,
        eps=EPS, top_k=TOP_K, group_size=GROUP, bits=BITS, vocab=VOCAB,
        layer_types=list(LAYER_TYPES),
        moe_routed="stubbed_to_zero",
        ple="absent",
        note="Python feeds all S tokens through each layer in ONE call; "
             "routed experts are stubbed to exact zero in both oracle and C++.",
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote {len(list(out.glob('*.npy')))} arrays to {out}")
    print("expected_logits[:8] =", np.asarray(expected_logits)[:8])


if __name__ == "__main__":
    main()
