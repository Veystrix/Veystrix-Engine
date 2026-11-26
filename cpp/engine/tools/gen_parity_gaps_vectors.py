#!/usr/bin/env python3
"""Bit-exact vectors for the two remaining parity gaps (parity_gaps.cpp).

Gap 1 -- QUANTIZED shared expert:
  Builds a REAL Qwen3_5MoeSparseMoeBlock whose routed experts are quantized
  QuantizedSwitchLinear layers AND whose shared expert's gate/up/down are
  nn.QuantizedLinear (the production WSB layout from
  models/flashnext/preflight.py). The sigmoid shared_expert_gate stays a dense
  nn.Linear, matching the C++ helper's dense gate input. Runs one decode step
  through the real module, through a gather_qmm primitive reference, asserts
  the two are byte-identical, then saves inputs + Q4 triplets + expected.

Gap 2 -- INDEXER-NONE dense fallback:
  Builds a REAL Qwen4ExpAttention with a prepopulated QSAKVCache whose indexer
  history is short enough that max_complete_blocks <= block_topk, so
  Qwen4ExpQSAIndexer.__call__ returns None and super().__call__ runs ordinary
  dense SDPA with mask=None. Runs the real attention, a primitive reference
  (exactly mirroring full_attention_dense_fallback), asserts bit-equality and
  saves the dense q/k/v/o weights, norms, cache, positions and expected output.

Usage:
  .venv/bin/python cpp/engine/tools/gen_parity_gaps_vectors.py \
      --out cpp/engine/tests/vectors_parity_gaps
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

from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock  # noqa: E402
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpAttention, QSAKVCache  # noqa: E402
from mlx_vlm.models.switch_layers import QuantizedSwitchLinear  # noqa: E402
from mlx_vlm.models.activations import swiglu  # noqa: E402


# --------------------------------------------------------------------------- #
# Geometry                                                                     #
# --------------------------------------------------------------------------- #
H = 32               # hidden size
E = 8                # routed experts
TOP_K = 2
GROUP = 32
INTER = 32           # routed + shared intermediate width
EPS = 1e-6

# Dense attention
N_HEADS = 2
KV_HEADS = 1
HEAD_DIM = 16
ROPE_BASE = 10000.0
MROPE_SECTION = [6, 6, 4]

# QSA indexer (kept small so it returns None)
N_IDX_HEADS = 2
N_IDX_KV = 1
IDX_HEAD_DIM = 16
COMPRESS_RATIO = 2
INDEX_BUDGET = 4     # block_topk = 4 // 2 = 2

PAST_LEN = 6         # dense KV cache length
IDX_PAST_LEN = 2     # indexer history -> key_len = 3 -> blocks = 1 <= 2 -> None
TOKEN = 7


def make_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=H,
        moe_intermediate_size=INTER,
        shared_expert_intermediate_size=INTER,
        num_experts=E,
        num_experts_per_tok=TOP_K,
        rms_norm_eps=EPS,
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
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def bits_equal(a, b) -> bool:
    a = mx.view(mx.reshape(a, (-1,)), mx.uint32)
    b = mx.view(mx.reshape(b, (-1,)), mx.uint32)
    mx.eval(a, b)
    return bool(mx.all(a == b))


def make_projection(rng, n_experts, out, inp, scale=0.03):
    """Q4 affine triplets [E, out, in/8], [E, out, in/G], [E, out, in/G]."""
    codes = rng.integers(0, 16, size=(n_experts, out, inp), dtype=np.uint8)
    packed = np.zeros((n_experts, out, inp // 8), dtype=np.uint32)
    for n in range(8):
        packed |= codes[..., n::8].astype(np.uint32) << (n * 4)
    scales = mx.array(
        (rng.random((n_experts, out, inp // GROUP)).astype(np.float32) * 0.013 + 0.003)
    ).astype(mx.bfloat16)
    biases = mx.array(
        (rng.random((n_experts, out, inp // GROUP)).astype(np.float32) * 0.0006 - 0.0003)
    ).astype(mx.bfloat16)
    return mx.array(packed), scales, biases


def quant_linear(inp, out, w, s, b, n_experts=None):
    """nn.QuantizedLinear for the shared expert, QuantizedSwitchLinear for routed."""
    if n_experts is None:
        lin = nn.QuantizedLinear(inp, out, group_size=GROUP, bits=4, bias=False)
    else:
        lin = QuantizedSwitchLinear(
            inp, out, n_experts, bias=False, group_size=GROUP, bits=4, mode="affine")
    lin.weight, lin.scales, lin.biases = w, s, b
    return lin


def qsa_norm(x, weight, eps=EPS):
    """One-centered Qwen4ExpRMSNorm (1 + weight), float32 compute."""
    dtype = x.dtype
    y = x.astype(mx.float32)
    y = y * mx.rsqrt(mx.mean(mx.square(y), axis=-1, keepdims=True) + eps)
    y = y * (1.0 + weight.astype(mx.float32))
    return mx.reshape(y, x.shape).astype(dtype)


def rope_apply(x, position_ids, dim=HEAD_DIM, base=ROPE_BASE):
    """Eager partial rotary (matches qsa_apply_rope for 2-D text positions)."""
    inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    freqs = position_ids.astype(mx.float32)[..., None] * inv_freq  # [B,T,half]
    emb = mx.concatenate([freqs, freqs], axis=-1)                  # [B,T,dim]
    cos = mx.cos(emb).astype(x.dtype)
    sin = mx.sin(emb).astype(x.dtype)
    cos = mx.expand_dims(cos, 1)  # [B,1,T,dim]
    sin = mx.expand_dims(sin, 1)
    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2:dim]
    rot = mx.concatenate([-x2, x1], axis=-1)
    embed = x[..., :dim] * cos + rot * sin
    return mx.concatenate([embed, x[..., dim:]], axis=-1)


# --------------------------------------------------------------------------- #
# Gap 1: quantized shared expert                                               #
# --------------------------------------------------------------------------- #
def build_quantized_moe(seed):
    rng = np.random.default_rng(seed)
    cfg = make_cfg()
    moe = Qwen3_5MoeSparseMoeBlock(cfg)

    # Router (production stores it unquantized).
    moe.gate.weight = mx.array(
        (rng.standard_normal((E, H)).astype(np.float32) * 0.1))

    # Routed experts: quantized SwitchGLU (as in gen_decoder_full_vectors.py).
    routed = {}
    for name, (out, inp) in {
        "gate_proj": (INTER, H), "up_proj": (INTER, H), "down_proj": (H, INTER)
    }.items():
        w, s, b = make_projection(rng, E, out, inp)
        setattr(moe.switch_mlp, name,
                quant_linear(inp, out, w, s, b, n_experts=E))
        routed[name] = (w, s, b)

    # Shared expert: quantized WSB triplets, single expert (leading dim 1).
    shared = {}
    for name, (out, inp) in {
        "gate_proj": (INTER, H), "up_proj": (INTER, H), "down_proj": (H, INTER)
    }.items():
        w, s, b = make_projection(rng, 1, out, inp)
        setattr(moe.shared_expert, name,
                quant_linear(inp, out, w, s, b))
        shared[name] = (w, s, b)

    # Sigmoid gate: dense nn.Linear (matches the C++ dense gate input).
    moe.shared_expert_gate.weight = mx.array(
        (rng.standard_normal((1, H)).astype(np.float32) * 0.1))

    # Decode-shaped input (bf16, like the mlp hyper-connection output).
    x = mx.array(
        (rng.standard_normal((1, H)).astype(np.float32) * 0.5)).astype(mx.bfloat16)
    for m in (moe,):
        m.eval()

    # ---- real production forward -------------------------------------------
    expected = moe(x)

    # ---- primitive reference (mirrors moe_block_shared_quantized) ----------
    gate_w = moe.gate.weight
    gate_w_shared = moe.shared_expert_gate.weight
    logits = mx.matmul(x, mx.transpose(gate_w))
    gates = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-TOP_K, axis=-1)[..., -TOP_K:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)

    x4 = mx.expand_dims(x, (-2, -3))

    def rproj(triplet, xin, indices):
        w, s, b = triplet
        return mx.gather_qmm(xin, w, s, b, rhs_indices=indices, transpose=True,
                             group_size=GROUP, bits=4, mode="affine")

    rg = rproj(routed["gate_proj"], x4, inds)
    ru = rproj(routed["up_proj"], x4, inds)
    ra = ru * (rg * mx.sigmoid(rg))
    rd = rproj(routed["down_proj"], ra, inds)
    rd = mx.squeeze(rd, axis=-2)
    routed_out = (rd * scores[..., None]).sum(axis=-2)

    zero_route = mx.zeros((1, 1), dtype=mx.int32)
    sg = rproj(shared["gate_proj"], x4, zero_route)
    su = rproj(shared["up_proj"], x4, zero_route)
    sa = su * (sg * mx.sigmoid(sg))
    sd = rproj(shared["down_proj"], sa, zero_route)
    shared_out = mx.squeeze(mx.squeeze(sd, axis=-2), axis=1)
    sg_gate = mx.sigmoid(mx.matmul(x, mx.transpose(gate_w_shared)))
    primitive = routed_out + sg_gate * shared_out

    mx.eval(expected, primitive)
    if not bits_equal(expected, primitive):
        raise SystemExit("gap 1 self-check FAILED: real != primitive (quantized shared)")

    return {
        "x": x,
        "routes": mx.astype(inds, mx.int32),
        "scores": scores,
        "gate_w": gate_w,
        "shared_gate_w": gate_w_shared,
        "routed": routed,
        "shared": shared,
        "expected": expected,
    }


# --------------------------------------------------------------------------- #
# Gap 2: indexer-None dense attention                                          #
# --------------------------------------------------------------------------- #
def build_indexer_none_attention(seed):
    rng = np.random.default_rng(seed)
    cfg = make_cfg()
    attn = Qwen4ExpAttention(cfg)

    def R(shape, scale=0.08):
        return mx.array(rng.standard_normal(shape).astype(np.float32) * scale)

    attn.q_proj.weight = R((N_HEADS * HEAD_DIM * 2, H))
    attn.k_proj.weight = R((KV_HEADS * HEAD_DIM, H))
    attn.v_proj.weight = R((KV_HEADS * HEAD_DIM, H))
    attn.o_proj.weight = R((H, N_HEADS * HEAD_DIM))
    attn.q_norm.weight = mx.array(
        rng.standard_normal((HEAD_DIM,)).astype(np.float32) * 0.1)
    attn.k_norm.weight = mx.array(
        rng.standard_normal((HEAD_DIM,)).astype(np.float32) * 0.1)

    idx = attn.indexer
    idx.index_qk_proj.weight = R(
        ((N_IDX_HEADS + N_IDX_KV) * IDX_HEAD_DIM, H))
    idx.q_layernorm.weight = mx.array(
        rng.standard_normal((IDX_HEAD_DIM,)).astype(np.float32) * 0.1)
    idx.k_layernorm.weight = mx.array(
        rng.standard_normal((IDX_HEAD_DIM,)).astype(np.float32) * 0.1)

    # One-centered gains for q/k norms; eager RoPE (C++ reproduces eager).
    for norm in (attn.q_norm, attn.k_norm, idx.q_layernorm, idx.k_layernorm):
        norm._flashnext_one_centered = False
    attn.rotary_emb.fused_apply = False

    # Prepopulated cache: short indexer history -> indexer returns None.
    cache = QSAKVCache()
    cache.keys = mx.array(
        rng.standard_normal((1, KV_HEADS, PAST_LEN, HEAD_DIM)).astype(np.float32) * 0.1)
    cache.values = mx.array(
        rng.standard_normal((1, KV_HEADS, PAST_LEN, HEAD_DIM)).astype(np.float32) * 0.1)
    cache.offset = PAST_LEN
    cache.index_keys = mx.array(
        rng.standard_normal((1, IDX_PAST_LEN, IDX_HEAD_DIM)).astype(np.float32) * 0.1)
    cache.index_position_ids = mx.array(
        np.arange(IDX_PAST_LEN, dtype=np.int32)[None, :])

    x = mx.array(
        (rng.standard_normal((1, 1, H)).astype(np.float32) * 0.5)).astype(mx.bfloat16)
    position_ids = mx.array(np.array([[TOKEN]], dtype=np.int32))
    attn.eval()

    # Sanity: indexer MUST return None in this configuration.
    probe = QSAKVCache()
    probe.index_keys = cache.index_keys
    probe.index_position_ids = cache.index_position_ids
    probe.offset = cache.offset
    if idx(x, probe, position_ids) is not None:
        raise SystemExit("gap 2 setup ERROR: indexer did not return None")

    # The real attention mutates `cache` via update_and_fetch (it allocates a
    # step buffer and writes the new k/v into it). Capture the PRE-update dense
    # KV tensors for the primitive reference and for the saved C++ vectors.
    pre_keys, pre_values = cache.keys, cache.values

    # ---- real production forward (mask=None -> dense SDPA) -----------------
    expected = attn(x, mask=None, cache=cache, position_ids=position_ids)

    # ---- primitive reference (mirrors full_attention_dense_fallback) -------
    w = {
        "q": attn.q_proj.weight, "k": attn.k_proj.weight,
        "v": attn.v_proj.weight, "o": attn.o_proj.weight,
        "qn": attn.q_norm.weight, "kn": attn.k_norm.weight,
    }
    B, L, _ = x.shape
    q = mx.matmul(x, mx.transpose(w["q"]))
    k = mx.matmul(x, mx.transpose(w["k"]))
    v = mx.matmul(x, mx.transpose(w["v"]))
    q4 = mx.reshape(q, (B, L, N_HEADS, 2 * HEAD_DIM))
    queries, gate = mx.split(q4, 2, axis=-1)
    gate = mx.reshape(gate, (B, L, N_HEADS * HEAD_DIM))
    queries = mx.transpose(qsa_norm(queries, w["qn"]), (0, 2, 1, 3))
    keys = mx.transpose(
        qsa_norm(mx.reshape(k, (B, L, KV_HEADS, HEAD_DIM)), w["kn"]),
        (0, 2, 1, 3))
    values = mx.transpose(
        mx.reshape(v, (B, L, KV_HEADS, HEAD_DIM)), (0, 2, 1, 3))
    queries = rope_apply(queries, position_ids)
    keys = rope_apply(keys, position_ids)
    full_k = mx.concatenate([pre_keys, keys], axis=2)
    full_v = mx.concatenate([pre_values, values], axis=2)
    scale = HEAD_DIM ** -0.5
    out = mx.fast.scaled_dot_product_attention(
        queries, full_k, full_v, scale=scale, mask=None)
    out = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (B, L, N_HEADS * HEAD_DIM))
    primitive = mx.matmul(out * mx.sigmoid(gate), mx.transpose(w["o"]))

    mx.eval(expected, primitive)
    if not bits_equal(expected, primitive):
        raise SystemExit("gap 2 self-check FAILED: real != primitive (dense fallback)")

    return {
        "x": x,
        "position_ids": position_ids,
        "w": w,
        "cache_keys": pre_keys,
        "cache_values": pre_values,
        "index_keys": cache.index_keys,
        "index_pos": cache.index_position_ids,
        "index_qk_w": idx.index_qk_proj.weight,
        "index_q_norm_w": idx.q_layernorm.weight,
        "index_k_norm_w": idx.k_layernorm.weight,
        "expected": expected,
    }


# --------------------------------------------------------------------------- #
# Save                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors_parity_gaps")
    ap.add_argument("--seed", type=int, default=20261117)
    args = ap.parse_args()

    moe = build_quantized_moe(args.seed)
    attn = build_indexer_none_attention(args.seed + 1)
    print("self-check OK: real modules == C++ primitive formulas (both gaps, bit-exact)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "seed": args.seed,
        "mlx_version": mx.__version__,
        "group_size": GROUP,
        "bits": 4,
        "n_experts": E,
        "top_k": TOP_K,
        "n_heads": N_HEADS,
        "kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "rope_dim": HEAD_DIM,
        "rope_base": ROPE_BASE,
        "mrope_section": MROPE_SECTION,
        "n_idx_heads": N_IDX_HEADS,
        "n_idx_kv": N_IDX_KV,
        "idx_head_dim": IDX_HEAD_DIM,
        "compress_ratio": COMPRESS_RATIO,
        "token_budget": INDEX_BUDGET,
        "past_len": PAST_LEN,
        "eps": EPS,
    }

    def save(name, arr):
        mx.save(str(out / f"{name}.npy"), arr)
        meta[name] = {"shape": list(arr.shape),
                      "dtype": str(arr.dtype).split(".")[-1]}

    # Gap 1 vectors.
    save("g_x", moe["x"])
    save("g_routes", moe["routes"])
    save("g_scores", moe["scores"])
    save("g_shared_gate_w", moe["shared_gate_w"])
    for grp, triplets in (("routed", moe["routed"]), ("shared", moe["shared"])):
        for proj, (w, s, b) in triplets.items():
            tag = {"gate_proj": "g", "up_proj": "u", "down_proj": "d"}[proj]
            save(f"g_{grp}_{tag}w", w)
            save(f"g_{grp}_{tag}s", s)
            save(f"g_{grp}_{tag}b", b)
    save("g_expected", moe["expected"])

    # Gap 2 vectors.
    save("fa_x", attn["x"])
    save("fa_pos", attn["position_ids"])
    for key, name in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "o"),
                      ("qn", "q_norm"), ("kn", "k_norm")):
        save(f"fa_{name}_w", attn["w"][key])
    save("fa_cache_keys", attn["cache_keys"])
    save("fa_cache_values", attn["cache_values"])
    save("fa_index_keys", attn["index_keys"])
    save("fa_index_pos", attn["index_pos"])
    save("fa_index_qk_w", attn["index_qk_w"])
    save("fa_index_q_norm_w", attn["index_q_norm_w"])
    save("fa_index_k_norm_w", attn["index_k_norm_w"])
    save("fa_expected", attn["expected"])

    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote parity-gaps vectors to {out}")
    print(f"wrote {len([k for k in meta if isinstance(meta[k], dict)])} arrays")


if __name__ == "__main__":
    main()
