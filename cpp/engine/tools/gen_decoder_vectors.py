#!/usr/bin/env python3
"""Generate bit-exact vectors for one full multi-layer Qwen4-Exp decode step.

This is the KEY integration vector for the C++ decoder loop.  It builds a TINY
but REAL Qwen4-Exp language-stack from the actual mlx_vlm modules:

  * Qwen4ExpGatedResidual          (hyper-connections, twice per layer + mixer)
  * Qwen4ExpGatedDeltaNet          (linear attention, every layer is linear)
  * Qwen3_5MoeSparseMoeBlock       (router + quantized SwitchGLU experts)
  * nn.Embedding / lm_head         (input embedding + final projection)

with a permissive synthetic config (hidden=32, 2 layers, hc_count=2, 8 experts,
top-2, group_size=32, vocab=64).  ALL weights are seeded deterministically and
the model is put in eval() so the GDN decode uses the fused gated-delta kernel.

The production RMSNorm convention is the "one-centered" gain from
``models/flashnext/patch_rmsnorm.py``; we apply that patch before constructing
the modules.  (The C++ rms_norm / gated_residual primitives implement that
convention.)

SATELITE CHECK -- why this is trustworthy
-----------------------------------------
The verified C++ primitives implement silu as ``x * sigmoid(x)`` (explicit),
whereas the real mlx_vlm modules call ``nn.silu(x)`` (fused, ~1 ULP different
for |x| >= 0.5).  To guarantee the "real module" reference equals the C++
primitive path, this generator ALSO computes the whole step with a *primitive
formula* reference (explicit sigmoid everywhere, the same one-centered norms,
``mx.gather_qmm``, the real fused gated-delta kernel) and ASSERTs that its
logits are byte-identical to the real-module logits.  If that assertion fails,
generation aborts and the seed/scales must be changed.  We keep every silu
input small so the two formulations coincide.

Outputs in ``--out`` (mx.save npy + meta.json):
  per layer l:
    L{l}_attn_norm_w, L{l}_attn_down_w, L{l}_attn_up_w, L{l}_attn_inject_w
    L{l}_gdn_w_qkv, L{l}_gdn_w_z, L{l}_gdn_w_b, L{l}_gdn_w_a,
    L{l}_gdn_conv_w, L{l}_gdn_A_log, L{l}_gdn_dt_bias, L{l}_gdn_norm_w,
    L{l}_gdn_w_out, L{l}_conv_state, L{l}_rec_state
    L{l}_mlp_norm_w, L{l}_mlp_down_w, L{l}_mlp_up_w, L{l}_mlp_inject_w
    L{l}_gate_w, L{l}_expert_{gw,gs,gb,uw,us,ub,dw,ds,db}
    L{l}_exp_gdn_out, L{l}_exp_moe_out, L{l}_exp_hidden  (debug intermediates)
  inputs/head:
    tokens, embeddings, embed_w, lm_head_w,
    mix_norm_w, mix_down_w, mix_up_w,          (final mixer, no inject)
    expected_logits, expected_logits_primitive
  meta.json

Usage:
  .venv/bin/python cpp/engine/tools/gen_decoder_vectors.py \
      --out cpp/engine/tests/vectors_decoder
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
N_LAYERS = 2           # tiny stack; all layers linear attention
Dk = Dv = 32           # GDN head dims (multiples of 32 for the Metal kernel)
Hk, Hv = 1, 2          # GDN key / value heads
K = 4                  # GDN conv kernel
E = 8                  # routed experts
TOP_K = 2
GROUP = 32
INTER = 32             # expert intermediate width
VOCAB = 64
EPS = 1e-6

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
        full_attention_interval=10**9,  # every layer is linear
    )


# --------------------------------------------------------------------------- #
# Deterministic weight helpers                                                 #
# --------------------------------------------------------------------------- #

def make_projection(rng, out, inp):
    """Q4 affine expert projection, same packing convention as gen_vectors.py.

    Scales/biases are kept SMALL so the dequantized gate/up activations land in
    the region where nn.silu == x*sigmoid(x) bit-for-bit (the C++ primitive
    uses the explicit formulation).
    """
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
        (rng.random((HC_HIDDEN,)).astype(np.float32) + 0.5)
    )
    hc.input_mix_weight_down.weight = mx.array(
        (rng.standard_normal((LOW, HC_HIDDEN)).astype(np.float32) * 0.1)
    )
    hc.input_mix_weight_up.weight = mx.array(
        (rng.standard_normal((HC_HIDDEN, LOW)).astype(np.float32) * 0.1)
    )
    if "block_inject_weight" in hc:
        hc.block_inject_weight.weight = mx.array(
            (rng.standard_normal((HC, HC_HIDDEN)).astype(np.float32) * 0.1)
        )


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
    moe.gate.weight = mx.array(
        (rng.standard_normal((E, H)).astype(np.float32) * 0.1)
    )
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
            E,
            bias=False,
            group_size=GROUP,
            bits=4,
            mode="affine",
        )
        q.weight, q.scales, q.biases = w, s, b
        setattr(moe.switch_mlp, proj, q)
    # Zero the shared expert so the real MoE == routed experts only (which is
    # exactly what the C++ moe_block primitive computes).
    for name in ("gate_proj", "up_proj", "down_proj"):
        p = getattr(moe.shared_expert, name)
        p.weight = mx.zeros_like(p.weight)
    moe.shared_expert_gate.weight = mx.zeros_like(moe.shared_expert_gate.weight)
    return {"gw": gw, "gs": gs, "gb": gb, "uw": uw, "us": us, "ub": ub,
            "dw": dw, "ds": ds, "db": db}


# --------------------------------------------------------------------------- #
# Primitive-formula reference (explicit sigmoid, matches the C++ primitives)  #
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
    silu = low * mx.sigmoid(low)                       # explicit (C++ primitive)
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
    conv_out = conv_pre * mx.sigmoid(conv_pre)          # explicit (C++ primitive)
    q, k, v = [
        t.reshape(1, 1, h, d)
        for t, h, d in zip(mx.split(conv_out, [KEY_DIM, 2 * KEY_DIM], -1),
                           [Hk, Hk, Hv], [Dk, Dk, Dv])
    ]
    # Qwen4ExpGatedDeltaNet._normalize_qk (matches gdn_normalize_qk in C++).
    scale = q.shape[-1] ** -0.5
    qn = (q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-6)) * scale
    kn = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6)
    g, beta = _compute_g_beta(w["A_log"], a, b, w["dt_bias"])
    y, new_rec = gated_delta_kernel(qn, kn, v, g, beta, rec_state)
    ny = mx.fast.rms_norm(y, w["norm"], EPS).astype(mx.float32)
    gated = (ny * mx.sigmoid(z.astype(mx.float32))).astype(y.dtype)
    out = mx.matmul(gated.reshape(1, 1, VALUE_DIM), mx.transpose(w["out"]))
    return out, new_conv, new_rec


def ref_moe(x2d, gate_w, ex):
    logits = mx.matmul(x2d, mx.transpose(gate_w))
    gates = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-TOP_K, axis=-1)[..., -TOP_K:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    x4 = mx.expand_dims(x2d, (-2, -3))
    route = inds
    def proj(w, s, b):
        return mx.gather_qmm(x4, w, s, b, rhs_indices=route, transpose=True,
                             group_size=GROUP, bits=4, mode="affine")
    g = proj(ex["gw"], ex["gs"], ex["gb"])
    u = proj(ex["uw"], ex["us"], ex["ub"])
    act = u * (g * mx.sigmoid(g))                      # explicit (C++ swiglu)
    d = mx.gather_qmm(act, ex["dw"], ex["ds"], ex["db"], rhs_indices=route,
                      transpose=True, group_size=GROUP, bits=4, mode="affine")
    d = mx.squeeze(d, axis=-2)
    out = (d * scores[..., None]).sum(axis=-2)
    return out.reshape(1, 1, H), inds, scores


def inject(hidden, branch, inj):
    return hidden + mx.reshape(
        branch[..., None, :] * inj[..., None], hidden.shape)


# --------------------------------------------------------------------------- #
# Build + run                                                                  #
# --------------------------------------------------------------------------- #

def build(seed):
    rng = np.random.default_rng(seed)
    cfg = make_cfg()
    layers = []
    for _ in range(N_LAYERS):
        d = {
            "attn_hc": Qwen4ExpGatedResidual(cfg),
            "gdn": Qwen4ExpGatedDeltaNet(cfg),
            "mlp_hc": Qwen4ExpGatedResidual(cfg),
            "moe": Qwen3_5MoeSparseMoeBlock(cfg),
        }
        seed_hc(d["attn_hc"], rng)
        seed_gdn(d["gdn"], rng)
        seed_hc(d["mlp_hc"], rng)
        d["ex"] = seed_moe(d["moe"], rng)

        # Extract the GDN weights into plain arrays for the reference + saving.
        g = d["gdn"]
        d["gdn_w"] = {
            "qkv": g.in_proj_qkv.weight, "z": g.in_proj_z.weight,
            "b": g.in_proj_b.weight, "a": g.in_proj_a.weight,
            "conv": g.conv1d.weight, "A_log": g.A_log,
            "dt_bias": g.dt_bias, "norm": g.norm.weight,
            "out": g.out_proj.weight,
        }
        d["conv_state"] = mx.array(
            (rng.standard_normal((1, K - 1, CONV_DIM)).astype(np.float32) * 0.05))
        d["rec_state"] = mx.array(
            (rng.standard_normal((1, Hv, Dv, Dk)).astype(np.float32) * 0.02))
        for m in (d["attn_hc"], g, d["mlp_hc"], d["moe"]):
            m.eval()
        layers.append(d)

    mixer = Qwen4ExpGatedResidual(cfg, use_combine=False)
    seed_hc(mixer, rng)
    mixer.eval()

    embed = nn.Embedding(VOCAB, H)
    embed.weight = mx.array(rng.standard_normal((VOCAB, H)).astype(np.float32) * 0.1)
    embed.eval()
    lm_head = nn.Linear(H, VOCAB, bias=False)
    lm_head.weight = mx.array(rng.standard_normal((VOCAB, H)).astype(np.float32) * 0.1)
    lm_head.eval()

    tokens = mx.array(np.array([[7]], dtype=np.int32))
    return cfg, layers, mixer, embed, lm_head, tokens


def run_real(layers, mixer, embed, lm_head, tokens):
    hidden = mx.tile(embed(tokens), (1, 1, HC))
    debug = []
    for i, d in enumerate(layers):
        cache = [d["conv_state"], d["rec_state"]]
        mixed, hyp, inj = d["attn_hc"](hidden)
        branch = d["gdn"](mixed, cache=cache)
        hidden = inject(hidden, branch, inj)
        h_after_attn = hidden
        mixed2, hyp2, inj2 = d["mlp_hc"](hidden)
        moe_out = d["moe"](mixed2)
        hidden = inject(hidden, moe_out, inj2)
        debug.append({"gdn": branch, "moe": moe_out,
                      "h_after_attn": h_after_attn, "hidden": hidden})
    final = mixer(hidden)
    logits = mx.matmul(final, mx.transpose(lm_head.weight))
    mx.eval(hidden, logits)
    for i in range(len(debug)):
        mx.eval(debug[i]["gdn"], debug[i]["moe"],
                debug[i]["h_after_attn"], debug[i]["hidden"])
    return logits, debug


def run_primitive(layers, mixer, lm_head, tokens, embed_w):
    # Embedding via the same op C++ uses (take + tile).
    e = mx.take(embed_w, mx.reshape(tokens, (-1,)), axis=0).reshape(1, 1, H)
    hidden = mx.tile(e, (1, 1, HC))
    debug = []
    for d in layers:
        mixed, inj = ref_hc(hidden, d["attn_hc"].hc_norm.weight,
                            d["attn_hc"].input_mix_weight_down.weight,
                            d["attn_hc"].input_mix_weight_up.weight,
                            d["attn_hc"].block_inject_weight.weight)
        branch, nc, nr = ref_gdn(mixed, d["gdn_w"], d["conv_state"], d["rec_state"])
        hidden = inject(hidden, branch, inj)
        h_after_attn = hidden
        mixed2, inj2 = ref_hc(hidden, d["mlp_hc"].hc_norm.weight,
                              d["mlp_hc"].input_mix_weight_down.weight,
                              d["mlp_hc"].input_mix_weight_up.weight,
                              d["mlp_hc"].block_inject_weight.weight)
        moe_out, inds, scores = ref_moe(mixed2.reshape(1, H),
                                        d["moe"].gate.weight, d["ex"])
        hidden = inject(hidden, moe_out, inj2)
        debug.append({"gdn": branch, "moe": moe_out,
                      "h_after_attn": h_after_attn, "hidden": hidden})
    final = ref_hc(hidden, mixer.hc_norm.weight,
                   mixer.input_mix_weight_down.weight,
                   mixer.input_mix_weight_up.weight, None, use_inject=False)
    logits = mx.matmul(final, mx.transpose(lm_head.weight))
    mx.eval(hidden, logits)
    for i in range(len(debug)):
        mx.eval(debug[i]["gdn"], debug[i]["moe"],
                debug[i]["h_after_attn"], debug[i]["hidden"])
    return logits, debug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors_decoder")
    ap.add_argument("--seed", type=int, default=20261117)
    args = ap.parse_args()

    cfg, layers, mixer, embed, lm_head, tokens = build(args.seed)
    logits_real, real_dbg = run_real(layers, mixer, embed, lm_head, tokens)
    logits_ref, ref_dbg = run_primitive(
        layers, mixer, lm_head, tokens, embed.weight)

    # ---- self-check: real modules must equal the C++ primitive formulas ----
    assert logits_real.shape == logits_ref.shape
    if not bits_equal(logits_real, logits_ref):
        for i in range(len(real_dbg)):
            for key in ("gdn", "moe", "h_after_attn", "hidden"):
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
    }, "seed": args.seed, "mlx_version": mx.__version__}

    def save(name, arr):
        mx.save(str(out / f"{name}.npy"), arr)
        meta[name] = {"shape": list(arr.shape),
                      "dtype": str(arr.dtype).split(".")[-1]}

    save("tokens", tokens)
    save("embeddings", embed(tokens))
    save("embed_w", embed.weight)
    save("lm_head_w", lm_head.weight)
    save("mix_norm_w", mixer.hc_norm.weight)
    save("mix_down_w", mixer.input_mix_weight_down.weight)
    save("mix_up_w", mixer.input_mix_weight_up.weight)
    save("expected_logits", logits_real)
    save("expected_logits_primitive", logits_ref)

    for i, d in enumerate(layers):
        a, g = d["attn_hc"], d["gdn"]
        m, moe = d["mlp_hc"], d["moe"]
        save(f"L{i}_attn_norm_w", a.hc_norm.weight)
        save(f"L{i}_attn_down_w", a.input_mix_weight_down.weight)
        save(f"L{i}_attn_up_w", a.input_mix_weight_up.weight)
        save(f"L{i}_attn_inject_w", a.block_inject_weight.weight)
        for key, arr in d["gdn_w"].items():
            save(f"L{i}_gdn_{key}", arr)
        save(f"L{i}_conv_state", d["conv_state"])
        save(f"L{i}_rec_state", d["rec_state"])
        save(f"L{i}_mlp_norm_w", m.hc_norm.weight)
        save(f"L{i}_mlp_down_w", m.input_mix_weight_down.weight)
        save(f"L{i}_mlp_up_w", m.input_mix_weight_up.weight)
        save(f"L{i}_mlp_inject_w", m.block_inject_weight.weight)
        save(f"L{i}_gate_w", moe.gate.weight)
        for key, arr in d["ex"].items():
            save(f"L{i}_expert_{key}", arr)
        save(f"L{i}_exp_gdn_out", real_dbg[i]["gdn"])
        save(f"L{i}_exp_moe_out", real_dbg[i]["moe"])
        save(f"L{i}_exp_hidden", real_dbg[i]["hidden"])

    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote decoder vectors to {out}")
    print(f"logits shape={logits_real.shape}, first row head="
          f"{np.asarray(logits_real)[0, 0, :5]}")


if __name__ == "__main__":
    main()
