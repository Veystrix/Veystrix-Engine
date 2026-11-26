#!/usr/bin/env python3
"""Bit-exact vectors for the EXTENDED decoder step (decoder_full).

Builds a TINY but REAL Qwen4-Exp language stack with a MIX of layers
(2 GDN linear layers + 1 QSA full-attention layer) and NON-ZERO shared experts:

  * Qwen4ExpGatedResidual          (hyper-connections, twice per layer + mixer)
  * Qwen4ExpGatedDeltaNet          (linear attention, layers 0 and 1)
  * Qwen4ExpAttention + QSAKVCache (full attention, layer 2)
  * Qwen3_5MoeSparseMoeBlock       (router + quantized SwitchGLU + dense shared)
  * nn.Embedding / lm_head

All weights are seeded deterministically, modules are in eval(), and RoPE is
forced to its eager (non-fused) branch so the C++ port (which uses eager
`qsa_apply_rope`) can reproduce it bit-for-bit.

As in gen_decoder_vectors.py, the generator runs the step TWICE: once through
the real modules (`expected_logits`) and once through a primitive-formula
reference that mirrors EXACTLY what decoder_full.cpp computes (explicit
sigmoid/silu, one-centered norms, `mx.fast.scaled_dot_product_attention`,
real indexer mask). It ASSERTs the two are byte-identical; generation aborts
otherwise. This makes the saved vectors a proof, not an approximation.

Outputs in --out (mx.save npy + meta.json). Usage:
  .venv/bin/python cpp/engine/tools/gen_decoder_full_vectors.py \
      --out cpp/engine/tests/vectors_decoder_full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".venv/lib/python3.12/site-packages"))
sys.path.insert(0, str(ROOT / "models/flashnext"))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import patch_rmsnorm  # noqa: E402

patch_rmsnorm.apply()  # one-centered gain convention (production)

from mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    Qwen4ExpGatedResidual,
    Qwen4ExpGatedDeltaNet,
    Qwen4ExpAttention,
    QSAKVCache,
)
from mlx_vlm.models.qwen3_5_moe.language import (  # noqa: E402
    Qwen3_5MoeSparseMoeBlock,
)
from mlx_vlm.models.switch_layers import QuantizedSwitchLinear  # noqa: E402
from mlx_vlm.models.qwen3_5.language import (  # noqa: E402
    _qwen3_5_decode_depthwise_conv,
)
from mlx_vlm.models.qwen3_5.gated_delta import (  # noqa: E402
    gated_delta_kernel,
    _compute_g_beta,
)


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

H = 32                 # hidden size
HC = 2                 # hyper-connection streams
LOW = 16               # hc lowrank
N_LAYERS = 3           # layers 0,1 = GDN; layer 2 = QSA full attention
FULL_INTERVAL = 3      # is_linear = (idx+1) % FULL_INTERVAL != 0

Dk = Dv = 32           # GDN head dims (multiples of 32 for the Metal kernel)
Hk, Hv = 1, 2          # GDN key / value heads
K = 4                  # GDN conv kernel
E = 8                  # routed experts
TOP_K = 2
GROUP = 32
INTER = 32             # expert intermediate width
VOCAB = 64
EPS = 1e-6

# Dense attention (Qwen4ExpAttention)
N_HEADS = 2            # attention query heads (n_heads * head_dim == H)
KV_HEADS = 1           # attention key/value heads (GQA)
HEAD_DIM = 16          # attention per-head dim (partial_rotary_factor = 1.0)
ROPE_BASE = 10000.0
MROPE_SECTION = [6, 6, 4]

# QSA indexer (Qwen4ExpQSAIndexer)
N_IDX_HEADS = 2
N_IDX_KV = 1
IDX_HEAD_DIM = 16
COMPRESS_RATIO = 2
INDEX_BUDGET = 4       # block_topk = budget / ratio = 2

PAST_LEN = 8           # prepopulated KV / indexer length (> budget -> mask)
TOKEN = 7

KEY_DIM = Hk * Dk
VALUE_DIM = Hv * Dv
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
HC_HIDDEN = HC * H


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
        full_attention_interval=FULL_INTERVAL,
        num_key_value_heads=KV_HEADS,
        num_attention_heads=N_HEADS,
        head_dim=HEAD_DIM,
        attention_bias=False,
        max_position_embeddings=2048,
        rope_parameters={
            "partial_rotary_factor": 1.0,
            "rope_theta": ROPE_BASE,
            "mrope_section": MROPE_SECTION,
        },
        indexer_n_heads=N_IDX_HEADS,
        indexer_kv_heads=N_IDX_KV,
        indexer_head_dim=IDX_HEAD_DIM,
        indexer_budget=INDEX_BUDGET,
        indexer_compress_ratio=COMPRESS_RATIO,
    )


# --------------------------------------------------------------------------- #
# Deterministic weight helpers                                                 #
# --------------------------------------------------------------------------- #

def make_projection(rng, out, inp):
    codes = rng.integers(0, 16, size=(E, out, inp), dtype=np.uint8)
    packed = np.zeros((E, out, inp // 8), dtype=np.uint32)
    for n in range(8):
        packed |= codes[..., n::8].astype(np.uint32) << (n * 4)
    scales = mx.array(
        (rng.random((E, out, inp // GROUP)).astype(np.float32) * 0.013 + 0.003)
    ).astype(mx.bfloat16)
    biases = mx.array(
        (rng.random((E, out, inp // GROUP)).astype(np.float32) * 0.0006 - 0.0003)
    ).astype(mx.bfloat16)
    return mx.array(packed), scales, biases


def seed_hc(hc, rng):
    hc.hc_norm.weight = mx.array(
        (rng.random((HC_HIDDEN,)).astype(np.float32) + 0.5))
    hc.input_mix_weight_down.weight = mx.array(
        (rng.standard_normal((LOW, HC_HIDDEN)).astype(np.float32) * 0.1))
    hc.input_mix_weight_up.weight = mx.array(
        (rng.standard_normal((HC_HIDDEN, LOW)).astype(np.float32) * 0.1))
    if "block_inject_weight" in hc:
        hc.block_inject_weight.weight = mx.array(
            (rng.standard_normal((HC, HC_HIDDEN)).astype(np.float32) * 0.1))


def seed_gdn(gdn, rng):
    def R(shape, scale, dtype=mx.float32):
        return mx.array(rng.standard_normal(shape) * scale).astype(dtype)

    gdn.in_proj_qkv.weight = R((CONV_DIM, H), 0.1, mx.bfloat16)
    gdn.in_proj_z.weight = R((VALUE_DIM, H), 0.1, mx.bfloat16)
    gdn.in_proj_b.weight = R((Hv, H), 0.1, mx.bfloat16)
    gdn.in_proj_a.weight = R((Hv, H), 0.1, mx.bfloat16)
    gdn.conv1d.weight = R((CONV_DIM, K, 1), 0.1, mx.bfloat16)
    gdn.A_log = R((Hv,), 1.0)
    gdn.dt_bias = R((Hv,), 0.5)
    gdn.norm.weight = mx.array(np.ones(Dv, dtype=np.float32))
    gdn.out_proj.weight = R((H, VALUE_DIM), 0.1, mx.bfloat16)


def seed_moe(moe, rng):
    """Seed router + quantized routed experts + NON-ZERO dense shared expert."""
    moe.gate.weight = mx.array(
        (rng.standard_normal((E, H)).astype(np.float32) * 0.1))
    gw, gs, gb = make_projection(rng, INTER, H)
    uw, us, ub = make_projection(rng, INTER, H)
    dw, ds, db = make_projection(rng, H, INTER)
    for proj, (w, s, b) in (
        ("gate_proj", (gw, gs, gb)),
        ("up_proj", (uw, us, ub)),
        ("down_proj", (dw, ds, db)),
    ):
        q = QuantizedSwitchLinear(
            H if proj != "down_proj" else INTER,
            INTER if proj != "down_proj" else H,
            E, bias=False, group_size=GROUP, bits=4, mode="affine")
        q.weight, q.scales, q.biases = w, s, b
        setattr(moe.switch_mlp, proj, q)
    # NON-ZERO shared expert (dense Qwen3_5MoeMLP). Small scales keep the
    # compiled nn.silu == explicit gate*sigmoid(gate) (see self-check).
    se = moe.shared_expert
    se.gate_proj.weight = mx.array(
        (rng.standard_normal((INTER, H)).astype(np.float32) * 0.03))
    se.up_proj.weight = mx.array(
        (rng.standard_normal((INTER, H)).astype(np.float32) * 0.03))
    se.down_proj.weight = mx.array(
        (rng.standard_normal((H, INTER)).astype(np.float32) * 0.03))
    moe.shared_expert_gate.weight = mx.array(
        (rng.standard_normal((1, H)).astype(np.float32) * 0.1))
    return {"gw": gw, "gs": gs, "gb": gb, "uw": uw, "us": us, "ub": ub,
            "dw": dw, "ds": ds, "db": db,
            "segw": se.gate_proj.weight, "seuw": se.up_proj.weight,
            "sedw": se.down_proj.weight,
            "seglw": moe.shared_expert_gate.weight}


def seed_attn(attn, rng):
    """Seed dense q/k/v/o and the one-centered q_norm/k_norm/indexer weights."""
    def R(shape, scale=0.1):
        return mx.array(rng.standard_normal(shape).astype(np.float32) * scale)

    attn.q_proj.weight = R((N_HEADS * HEAD_DIM * 2, H), 0.08)
    attn.k_proj.weight = R((KV_HEADS * HEAD_DIM, H), 0.08)
    attn.v_proj.weight = R((KV_HEADS * HEAD_DIM, H), 0.08)
    attn.o_proj.weight = R((H, N_HEADS * HEAD_DIM), 0.08)
    # Qwen4ExpRMSNorm weights are zero-centered gains (effective 1 + w).
    attn.q_norm.weight = mx.array(
        rng.standard_normal((HEAD_DIM,)).astype(np.float32) * 0.1)
    attn.k_norm.weight = mx.array(
        rng.standard_normal((HEAD_DIM,)).astype(np.float32) * 0.1)
    idx = attn.indexer
    idx.index_qk_proj.weight = R(
        ((N_IDX_HEADS + N_IDX_KV) * IDX_HEAD_DIM, H), 0.08)
    idx.q_layernorm.weight = mx.array(
        rng.standard_normal((IDX_HEAD_DIM,)).astype(np.float32) * 0.1)
    idx.k_layernorm.weight = mx.array(
        rng.standard_normal((IDX_HEAD_DIM,)).astype(np.float32) * 0.1)
    # The q/k norms use the ZERO-centered gain (1 + weight) that the verified
    # C++ QSA primitives implement (`qsa_rms_norm`). patch_rmsnorm.apply() made
    # Qwen4ExpRMSNorm one-centered globally for the hc/head norms (matching the
    # decoder's `gated_residual`); reconcile the q/k norms per-instance via the
    # patch's own `configure` attribute.
    for norm in (attn.q_norm, attn.k_norm,
                 idx.q_layernorm, idx.k_layernorm):
        norm._flashnext_one_centered = False
    # Force the eager RoPE branch (the C++ qsa_apply_rope reproduces eager).
    attn.rotary_emb.fused_apply = False
    return attn


def make_qsa_cache(rng):
    """Prepopulate a QSAKVCache beyond the indexer block budget."""
    cache = QSAKVCache()
    cache.keys = mx.array(
        rng.standard_normal((1, KV_HEADS, PAST_LEN, HEAD_DIM)).astype(np.float32)
        * 0.1)
    cache.values = mx.array(
        rng.standard_normal((1, KV_HEADS, PAST_LEN, HEAD_DIM)).astype(np.float32)
        * 0.1)
    cache.offset = PAST_LEN
    cache.index_keys = mx.array(
        rng.standard_normal((1, PAST_LEN, IDX_HEAD_DIM)).astype(np.float32)
        * 0.1)
    cache.index_position_ids = mx.array(
        np.arange(PAST_LEN, dtype=np.int32)[None, :])
    return cache


def clone_qsa_cache(c):
    nc = QSAKVCache()
    nc.keys, nc.values, nc.offset = c.keys, c.values, c.offset
    nc.index_keys, nc.index_position_ids = c.index_keys, c.index_position_ids
    return nc


# --------------------------------------------------------------------------- #
# Primitive-formula reference (mirrors decoder_full.cpp exactly)              #
# --------------------------------------------------------------------------- #

def bits_equal(a, b) -> bool:
    a = mx.view(mx.reshape(a, (-1,)), mx.uint32)
    b = mx.view(mx.reshape(b, (-1,)), mx.uint32)
    mx.eval(a, b)
    return bool(mx.all(a == b))


def ref_hc(hyper_input, nw, dw, uw, iw=None, use_inject=True):
    gs = (*hyper_input.shape[:-1], HC, H)
    w2d = mx.reshape(nw, (HC, H))
    yg = mx.reshape(hyper_input, gs).astype(mx.float32)
    yg = yg * mx.rsqrt(mx.mean(mx.square(yg), axis=-1, keepdims=True) + EPS)
    yg = yg * w2d.astype(mx.float32)
    normed = mx.reshape(yg, hyper_input.shape).astype(hyper_input.dtype)
    low = mx.matmul(normed, mx.transpose(dw)) / HC
    silu = low * mx.sigmoid(low)
    mix = mx.sigmoid(mx.matmul(silu, mx.transpose(uw)))
    mixed = mx.mean(mx.reshape(mix, gs) * mx.reshape(normed, gs), axis=-2)
    if not use_inject:
        return mixed
    inj = 2 * mx.sigmoid(mx.matmul(normed, mx.transpose(iw)) / HC)
    return mixed, inj


def ref_gdn(x, w, conv_state, rec_state):
    mixed_qkv = mx.matmul(x, mx.transpose(w["qkv"]))
    z = mx.matmul(x, mx.transpose(w["z"])).reshape(1, 1, Hv, Dv)
    b = mx.matmul(x, mx.transpose(w["b"]))
    a = mx.matmul(x, mx.transpose(w["a"]))
    conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
    new_conv = mx.contiguous(conv_input[:, -(K - 1):, :])
    prep_w = w["conv"][:, :, 0].T.astype(mx.float32)
    conv_pre = _qwen3_5_decode_depthwise_conv(conv_input, prep_w)
    conv_out = conv_pre * mx.sigmoid(conv_pre)
    q, k, v = [
        t.reshape(1, 1, h, d)
        for t, h, d in zip(mx.split(conv_out, [KEY_DIM, 2 * KEY_DIM], -1),
                           [Hk, Hk, Hv], [Dk, Dk, Dv])
    ]
    scale = q.shape[-1] ** -0.5
    qn = (q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-6)) * scale
    kn = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6)
    g, beta = _compute_g_beta(w["A_log"], a, b, w["dt_bias"])
    y, new_rec = gated_delta_kernel(qn, kn, v, g, beta, rec_state)
    ny = mx.fast.rms_norm(y, w["norm"], EPS).astype(mx.float32)
    gated = (ny * mx.sigmoid(z.astype(mx.float32))).astype(y.dtype)
    out = mx.matmul(gated.reshape(1, 1, VALUE_DIM), mx.transpose(w["out"]))
    return out, new_conv, new_rec


def ref_moe_with_shared(x2d, gate_w, ex):
    logits = mx.matmul(x2d, mx.transpose(gate_w))
    gates = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-TOP_K, axis=-1)[..., -TOP_K:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    x4 = mx.expand_dims(x2d, (-2, -3))

    def proj(w, s, b):
        return mx.gather_qmm(x4, w, s, b, rhs_indices=inds, transpose=True,
                             group_size=GROUP, bits=4, mode="affine")

    g = proj(ex["gw"], ex["gs"], ex["gb"])
    u = proj(ex["uw"], ex["us"], ex["ub"])
    act = u * (g * mx.sigmoid(g))
    d = mx.gather_qmm(act, ex["dw"], ex["ds"], ex["db"], rhs_indices=inds,
                      transpose=True, group_size=GROUP, bits=4, mode="affine")
    d = mx.squeeze(d, axis=-2)
    routed = (d * scores[..., None]).sum(axis=-2)

    # Dense shared expert (Qwen3_5MoeMLP) + sigmoid gate.
    sg = mx.sigmoid(mx.matmul(x2d, mx.transpose(ex["seglw"])))
    sg_gate = mx.matmul(x2d, mx.transpose(ex["segw"]))
    sg_up = mx.matmul(x2d, mx.transpose(ex["seuw"]))
    shared = mx.matmul(sg_up * (sg_gate * mx.sigmoid(sg_gate)),
                       mx.transpose(ex["sedw"]))
    return routed + sg * shared


def qsa_norm(x, weight, eps=EPS):
    dtype = x.dtype
    y = x.astype(mx.float32)
    y = y * mx.rsqrt(mx.mean(mx.square(y), axis=-1, keepdims=True) + eps)
    y = y * (1.0 + weight.astype(mx.float32))
    return mx.reshape(y, x.shape).astype(dtype)


def rope_apply(x, position_ids, dim=HEAD_DIM, base=ROPE_BASE):
    """Eager partial rotary, mirrors qsa_apply_rope (2-D text positions)."""
    inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    freqs = position_ids.astype(mx.float32)[..., None] * inv_freq  # [B,T,half]
    emb = mx.concatenate([freqs, freqs], axis=-1)                  # [B,T,dim]
    cos = mx.cos(emb).astype(x.dtype)
    sin = mx.sin(emb).astype(x.dtype)
    cos = mx.expand_dims(cos, 1)  # [B,1,T,dim]
    sin = mx.expand_dims(sin, 1)
    x1 = x[..., :dim // 2]
    x2 = x[..., dim // 2:dim]
    rot = mx.concatenate([-x2, x1], axis=-1)
    embed = x[..., :dim] * cos + rot * sin
    return mx.concatenate([embed, x[..., dim:]], axis=-1)


def ref_full_attention(x, w, cache_state, position_ids, mask):
    B, L, _ = x.shape
    Hq, Hkv, D = N_HEADS, KV_HEADS, HEAD_DIM
    q = mx.matmul(x, mx.transpose(w["q"]))
    k = mx.matmul(x, mx.transpose(w["k"]))
    v = mx.matmul(x, mx.transpose(w["v"]))
    q4 = mx.reshape(q, (B, L, Hq, 2 * D))
    queries, gate = mx.split(q4, 2, axis=-1)
    gate = mx.reshape(gate, (B, L, Hq * D))
    queries = mx.transpose(qsa_norm(queries, w["qn"]), (0, 2, 1, 3))
    keys = mx.transpose(qsa_norm(mx.reshape(k, (B, L, Hkv, D)), w["kn"]),
                        (0, 2, 1, 3))
    values = mx.transpose(mx.reshape(v, (B, L, Hkv, D)), (0, 2, 1, 3))
    queries = rope_apply(queries, position_ids)
    keys = rope_apply(keys, position_ids)
    full_k = mx.concatenate([cache_state["keys"], keys], axis=2)
    full_v = mx.concatenate([cache_state["values"], values], axis=2)
    kv_seq_len = L + cache_state["offset"] + 1
    use_mask = mask[..., :kv_seq_len]
    scale = D ** -0.5
    out = mx.fast.scaled_dot_product_attention(
        queries, full_k, full_v, scale=scale, mask=use_mask)
    out = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (B, L, Hq * D))
    return mx.matmul(out * mx.sigmoid(gate), mx.transpose(w["o"]))


def inject(hidden, branch, inj):
    return hidden + mx.reshape(branch[..., None, :] * inj[..., None],
                               hidden.shape)


# --------------------------------------------------------------------------- #
# Build + run                                                                  #
# --------------------------------------------------------------------------- #

def build(seed):
    rng = np.random.default_rng(seed)
    cfg = make_cfg()
    layers = []
    for idx in range(N_LAYERS):
        is_linear = (idx + 1) % FULL_INTERVAL != 0
        d = {
            "attn_hc": Qwen4ExpGatedResidual(cfg),
            "mlp_hc": Qwen4ExpGatedResidual(cfg),
            "moe": Qwen3_5MoeSparseMoeBlock(cfg),
            "is_linear": is_linear,
        }
        seed_hc(d["attn_hc"], rng)
        seed_hc(d["mlp_hc"], rng)
        d["ex"] = seed_moe(d["moe"], rng)
        if is_linear:
            d["gdn"] = Qwen4ExpGatedDeltaNet(cfg)
            seed_gdn(d["gdn"], rng)
            g = d["gdn"]
            d["gdn_w"] = {
                "qkv": g.in_proj_qkv.weight, "z": g.in_proj_z.weight,
                "b": g.in_proj_b.weight, "a": g.in_proj_a.weight,
                "conv": g.conv1d.weight, "A_log": g.A_log,
                "dt_bias": g.dt_bias, "norm": g.norm.weight,
                "out": g.out_proj.weight,
            }
            d["conv_state"] = mx.array(
                rng.standard_normal((1, K - 1, CONV_DIM)).astype(np.float32) * 0.05)
            d["rec_state"] = mx.array(
                rng.standard_normal((1, Hv, Dv, Dk)).astype(np.float32) * 0.02)
            for m in (d["gdn"],):
                m.eval()
        else:
            d["attn"] = Qwen4ExpAttention(cfg)
            seed_attn(d["attn"], rng)
            a = d["attn"]
            d["attn_w"] = {
                "q": a.q_proj.weight, "k": a.k_proj.weight,
                "v": a.v_proj.weight, "o": a.o_proj.weight,
                "qn": a.q_norm.weight, "kn": a.k_norm.weight,
            }
            d["cache"] = make_qsa_cache(rng)
            # run_real mutates the cache; keep a pristine clone for the
            # primitive reference (MLX arrays are immutable, so sharing the
            # stored arrays is safe — updates create new arrays).
            d["cache_pristine"] = clone_qsa_cache(d["cache"])
            a.eval()
        for m in (d["attn_hc"], d["mlp_hc"], d["moe"]):
            m.eval()
        layers.append(d)

    mixer = Qwen4ExpGatedResidual(cfg, use_combine=False)
    seed_hc(mixer, rng)
    mixer.eval()
    embed = nn.Embedding(VOCAB, H)
    embed.weight = mx.array(
        rng.standard_normal((VOCAB, H)).astype(np.float32) * 0.1)
    embed.eval()
    lm_head = nn.Linear(H, VOCAB, bias=False)
    lm_head.weight = mx.array(
        rng.standard_normal((VOCAB, H)).astype(np.float32) * 0.1)
    lm_head.eval()
    tokens = mx.array(np.array([[TOKEN]], dtype=np.int32))
    position_ids = mx.array(np.array([[PAST_LEN]], dtype=np.int32))
    return cfg, layers, mixer, embed, lm_head, tokens, position_ids


def run_real(layers, mixer, embed, lm_head, tokens, position_ids):
    hidden = mx.tile(embed(tokens), (1, 1, HC))
    debug = []
    for i, d in enumerate(layers):
        mixed, hyp, inj = d["attn_hc"](hidden)
        if d["is_linear"]:
            branch = d["gdn"](mixed, cache=[d["conv_state"], d["rec_state"]])
        else:
            branch = d["attn"](mixed, mask=None, cache=d["cache"],
                               position_ids=position_ids)
        hidden = inject(hidden, branch, inj)
        mixed2, hyp2, inj2 = d["mlp_hc"](hidden)
        moe_out = d["moe"](mixed2)
        hidden = inject(hidden, moe_out, inj2)
        debug.append({"branch": branch, "moe": moe_out, "hidden": hidden})
    final = mixer(hidden)
    logits = mx.matmul(final, mx.transpose(lm_head.weight))
    mx.eval(hidden, logits)
    for i in range(len(debug)):
        mx.eval(debug[i]["branch"], debug[i]["moe"], debug[i]["hidden"])
    return logits, debug


def run_primitive(layers, mixer, lm_head, tokens, embed_w, position_ids):
    e = mx.take(embed_w, mx.reshape(tokens, (-1,)), axis=0).reshape(1, 1, H)
    hidden = mx.tile(e, (1, 1, HC))
    debug = []
    for d in layers:
        mixed, inj = ref_hc(hidden, d["attn_hc"].hc_norm.weight,
                            d["attn_hc"].input_mix_weight_down.weight,
                            d["attn_hc"].input_mix_weight_up.weight,
                            d["attn_hc"].block_inject_weight.weight)
        if d["is_linear"]:
            branch, _, _ = ref_gdn(mixed, d["gdn_w"], d["conv_state"],
                                   d["rec_state"])
        else:
            # Mask from the REAL indexer (qsa_full.cpp is verified against it),
            # run on the PRISTINE cache so run_real's updates don't leak in.
            idx = d["attn"].indexer
            pristine = d["cache_pristine"]
            mask = idx(mixed, clone_qsa_cache(pristine), position_ids)
            if mask is None:
                raise SystemExit("indexer returned None: increase PAST_LEN")
            cache_state = {
                "keys": pristine.keys[..., :pristine.offset, :],
                "values": pristine.values[..., :pristine.offset, :],
                "offset": pristine.offset,
            }
            branch = ref_full_attention(mixed, d["attn_w"], cache_state,
                                        position_ids, mask)
            d["mask"] = mask
        hidden = inject(hidden, branch, inj)
        mixed2, inj2 = ref_hc(hidden, d["mlp_hc"].hc_norm.weight,
                              d["mlp_hc"].input_mix_weight_down.weight,
                              d["mlp_hc"].input_mix_weight_up.weight,
                              d["mlp_hc"].block_inject_weight.weight)
        moe_out = ref_moe_with_shared(mixed2.reshape(1, H),
                                      d["moe"].gate.weight, d["ex"])
        hidden = inject(hidden, moe_out, inj2)
        debug.append({"branch": branch, "moe": moe_out, "hidden": hidden})
    final = ref_hc(hidden, mixer.hc_norm.weight,
                   mixer.input_mix_weight_down.weight,
                   mixer.input_mix_weight_up.weight, None, use_inject=False)
    logits = mx.matmul(final, mx.transpose(lm_head.weight))
    mx.eval(hidden, logits)
    for i in range(len(debug)):
        mx.eval(debug[i]["branch"], debug[i]["moe"], debug[i]["hidden"])
    return logits, debug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors_decoder_full")
    ap.add_argument("--seed", type=int, default=20261117)
    args = ap.parse_args()

    cfg, layers, mixer, embed, lm_head, tokens, position_ids = build(args.seed)
    logits_real, real_dbg = run_real(layers, mixer, embed, lm_head, tokens,
                                     position_ids)
    logits_ref, ref_dbg = run_primitive(layers, mixer, lm_head, tokens,
                                        embed.weight, position_ids)

    assert logits_real.shape == logits_ref.shape
    if not bits_equal(logits_real, logits_ref):
        for i in range(len(real_dbg)):
            for key in ("branch", "moe", "hidden"):
                if not bits_equal(real_dbg[i][key], ref_dbg[i][key]):
                    raise SystemExit(
                        f"seed {args.seed}: real != primitive at layer {i} {key}")
        raise SystemExit(f"seed {args.seed}: final logits differ real vs primitive")
    print("self-check OK: real modules == C++ primitive formulas (bit-exact)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"config": {
        "hidden_size": H, "hc_count": HC, "hc_lowrank": LOW,
        "n_layers": N_LAYERS, "key_dim": KEY_DIM, "value_dim": VALUE_DIM,
        "num_k_heads": Hk, "num_v_heads": Hv, "head_k_dim": Dk,
        "head_v_dim": Dv, "conv_kernel": K, "n_experts": E, "top_k": TOP_K,
        "group_size": GROUP, "bits": 4, "moe_intermediate": INTER,
        "vocab": VOCAB, "eps": EPS, "output_gate_type": "sigmoid",
        "n_attn_heads": N_HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
        "n_idx_heads": N_IDX_HEADS, "n_idx_kv": N_IDX_KV,
        "idx_head_dim": IDX_HEAD_DIM, "compress_ratio": COMPRESS_RATIO,
        "token_budget": INDEX_BUDGET, "rope_dim": HEAD_DIM,
        "rope_base": ROPE_BASE, "mrope_section": MROPE_SECTION,
        "past_len": PAST_LEN,
    }, "layer_kinds": ["linear" if d["is_linear"] else "full" for d in layers],
        "seed": args.seed, "mlx_version": mx.__version__}

    def save(name, arr):
        mx.save(str(out / f"{name}.npy"), arr)
        meta[name] = {"shape": list(arr.shape),
                      "dtype": str(arr.dtype).split(".")[-1]}

    save("tokens", tokens)
    save("position_ids", position_ids)
    save("embeddings", embed(tokens))
    save("embed_w", embed.weight)
    save("lm_head_w", lm_head.weight)
    save("mix_norm_w", mixer.hc_norm.weight)
    save("mix_down_w", mixer.input_mix_weight_down.weight)
    save("mix_up_w", mixer.input_mix_weight_up.weight)
    save("expected_logits", logits_real)

    for i, d in enumerate(layers):
        a, m, moe = d["attn_hc"], d["mlp_hc"], d["moe"]
        save(f"L{i}_attn_norm_w", a.hc_norm.weight)
        save(f"L{i}_attn_down_w", a.input_mix_weight_down.weight)
        save(f"L{i}_attn_up_w", a.input_mix_weight_up.weight)
        save(f"L{i}_attn_inject_w", a.block_inject_weight.weight)
        save(f"L{i}_mlp_norm_w", m.hc_norm.weight)
        save(f"L{i}_mlp_down_w", m.input_mix_weight_down.weight)
        save(f"L{i}_mlp_up_w", m.input_mix_weight_up.weight)
        save(f"L{i}_mlp_inject_w", m.block_inject_weight.weight)
        save(f"L{i}_gate_w", moe.gate.weight)
        for key, arr in d["ex"].items():
            save(f"L{i}_expert_{key}", arr)
        if d["is_linear"]:
            for key, arr in d["gdn_w"].items():
                save(f"L{i}_gdn_{key}", arr)
            save(f"L{i}_conv_state", d["conv_state"])
            save(f"L{i}_rec_state", d["rec_state"])
        else:
            for key, arr in d["attn_w"].items():
                save(f"L{i}_attn_{key}", arr)
            idx = d["attn"].indexer
            pristine = d["cache_pristine"]
            save(f"L{i}_index_qk_w", idx.index_qk_proj.weight)
            save(f"L{i}_index_q_norm_w", idx.q_layernorm.weight)
            save(f"L{i}_index_k_norm_w", idx.k_layernorm.weight)
            save(f"L{i}_cache_keys", pristine.keys[..., :pristine.offset, :])
            save(f"L{i}_cache_values", pristine.values[..., :pristine.offset, :])
            save(f"L{i}_index_keys", pristine.index_keys)
            save(f"L{i}_index_pos", pristine.index_position_ids)
            save(f"L{i}_exp_mask", d["mask"])
        save(f"L{i}_exp_branch", real_dbg[i]["branch"])
        save(f"L{i}_exp_moe", real_dbg[i]["moe"])
        save(f"L{i}_exp_hidden", real_dbg[i]["hidden"])

    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote decoder_full vectors to {out}")
    print(f"logits shape={logits_real.shape}, first row head="
          f"{np.asarray(logits_real)[0, 0, :5]}")
    print(f"layer kinds = {meta['layer_kinds']}")


if __name__ == "__main__":
    main()
