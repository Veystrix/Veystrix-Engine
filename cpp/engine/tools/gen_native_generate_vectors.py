#!/usr/bin/env python3
"""Ground truth for the STANDALONE C++ greedy-generation executable.

This tool does two things (controlled by flags):

  (default) BUILD the vectors:
    Build a TINY but REAL Qwen4-Exp language stack from the actual mlx_vlm
    modules (all layers linear-attention, eval(), seeded), serialize EVERY
    weight to .npy files in <out>/weights, write a flat key=value config to
    <out>/config.txt, and run Python greedy decoding for N tokens from a prompt
    token.  For each step it (a) runs the REAL modules, (b) runs a primitive
    formula reference that mirrors the C++ engine bit-for-bit, and (c) ASSERTs
    the two logit rows are byte-identical.  It then records the REAL logits'
    greedy token id and SHA-256 (float32 little-endian, exactly the convention
    in models/shared/logit_reference.py) to <out>/expected.json.  It also saves
    <out>/expected_logits_step{k}.npy and <out>/expected_token_ids.npy for the
    C++ wrapper test.

  --check COMPARE the C++ output to the recorded ground truth:
    Runs the compiled veystrix_native_generate binary on the generated
    weights/config, parses its per-step `step=... in=... out=... sha256=...`
    lines, and requires token ids AND logit digests to match byte-for-byte.
    Prints `test_native_generate OK` on success.

Why all-linear (not the mixed GDN+QSA stack of gen_decoder_full_vectors.py):
    The full-attention QSA indexer only yields a mask when its cache is
    prepopulated past the block budget; a from-scratch generation must start
    from zeroed caches.  The all-linear stack is the exact configuration that
    `run_decoder_step` (decoder.cpp) is verified bit-exact for, and it supports
    zeroed GDN caches at step 0.  Everything else (hyper-connections, router,
    Q4 SwiGLU experts, embed/lm_head) is the real Qwen4-Exp implementation.

Usage:
    .venv/bin/python cpp/engine/tools/gen_native_generate_vectors.py \
        --out cpp/engine/tests/vectors_native_generate --steps 3
    .venv/bin/python cpp/engine/tools/gen_native_generate_vectors.py \
        --check --out cpp/engine/tests/vectors_native_generate \
        --binary /tmp/native_generate --steps 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
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
# Config (same tiny geometry as gen_decoder_vectors.py)                        #
# --------------------------------------------------------------------------- #

H = 32                 # hidden size
HC = 2                 # hyper-connection streams
LOW = 16               # hc lowrank
N_LAYERS = 2           # all layers linear attention
DK = DV = 32           # GDN head dims (multiples of 32 for the Metal kernel)
HK, HV = 1, 2          # GDN key / value heads
K = 4                  # GDN conv kernel
E = 8                  # routed experts
TOP_K = 2
GROUP = 32
INTER = 32             # expert intermediate width
VOCAB = 64
EPS = 1e-6
PROMPT_TOKEN = 7

KEY_DIM = HK * DK
VALUE_DIM = HV * DV
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
        linear_num_value_heads=HV,
        linear_num_key_heads=HK,
        linear_key_head_dim=DK,
        linear_value_head_dim=DV,
        linear_conv_kernel_dim=K,
        output_gate_type="sigmoid",
        hidden_act="silu",
        full_attention_interval=10**9,  # every layer is linear
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
    gdn.in_proj_b.weight = R((HV, H), 0.1, mx.bfloat16)
    gdn.in_proj_a.weight = R((HV, H), 0.1, mx.bfloat16)
    gdn.conv1d.weight = R((CONV_DIM, K, 1), 0.1, mx.bfloat16)
    gdn.A_log = R((HV,), 1.0)
    gdn.dt_bias = R((HV,), 0.5)
    gdn.norm.weight = mx.array(np.ones(DV, dtype=np.float32))
    gdn.out_proj.weight = R((H, VALUE_DIM), 0.1, mx.bfloat16)


def seed_moe(moe, rng):
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
    # Zeroed shared expert: real MoE == routed-only (what C++ moe_block does).
    for name in ("gate_proj", "up_proj", "down_proj"):
        p = getattr(moe.shared_expert, name)
        p.weight = mx.zeros_like(p.weight)
    moe.shared_expert_gate.weight = mx.zeros_like(
        moe.shared_expert_gate.weight)
    return {"gw": gw, "gs": gs, "gb": gb, "uw": uw, "us": us, "ub": ub,
            "dw": dw, "ds": ds, "db": db}


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
        g = d["gdn"]
        d["gdn_w"] = {
            "qkv": g.in_proj_qkv.weight, "z": g.in_proj_z.weight,
            "b": g.in_proj_b.weight, "a": g.in_proj_a.weight,
            "conv": g.conv1d.weight, "A_log": g.A_log,
            "dt_bias": g.dt_bias, "norm": g.norm.weight,
            "out": g.out_proj.weight,
        }
        for m in (d["attn_hc"], g, d["mlp_hc"], d["moe"]):
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
    return cfg, layers, mixer, embed, lm_head


# --------------------------------------------------------------------------- #
# Primitive-formula reference (mirrors the C++ engine exactly)                 #
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
    z = mx.matmul(x, mx.transpose(w["z"])).reshape(1, 1, HV, DV)
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
                           [HK, HK, HV], [DK, DK, DV])
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


def ref_moe(x2d, gate_w, ex):
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
    return (d * scores[..., None]).sum(axis=-2)


def inject(hidden, branch, inj):
    return hidden + mx.reshape(
        branch[..., None, :] * inj[..., None], hidden.shape)


def zero_caches():
    """One zeroed cache per layer: (conv_state, rec_state), float32."""
    return [
        [mx.zeros((1, K - 1, CONV_DIM)), mx.zeros((1, HV, DV, DK))]
        for _ in range(N_LAYERS)
    ]


def real_step(layers, mixer, embed, lm_head, caches, token):
    """One greedy step through the REAL mlx_vlm modules. Mutates caches."""
    tok = mx.array(np.array([[token]], dtype=np.int32))
    hidden = mx.tile(embed(tok), (1, 1, HC))
    for d, cache in zip(layers, caches):
        mixed, _hyp, inj = d["attn_hc"](hidden)
        branch = d["gdn"](mixed, cache=cache)       # updates cache in place
        hidden = inject(hidden, branch, inj)
        mixed2, _hyp2, inj2 = d["mlp_hc"](hidden)
        moe_out = d["moe"](mixed2)
        hidden = inject(hidden, moe_out, inj2)
    final = mixer(hidden)
    logits = mx.matmul(final, mx.transpose(lm_head.weight))
    mx.eval(logits)
    return logits


def primitive_step(layers, mixer, lm_head, embed_w, pstates, token):
    """One greedy step through the C++-mirror formulas. Returns (logits, states)."""
    e = mx.take(embed_w, mx.array([token], dtype=mx.int32), axis=0).reshape(1, 1, H)
    hidden = mx.tile(e, (1, 1, HC))
    new_states = []
    for d, (cs, rs) in zip(layers, pstates):
        mixed, inj = ref_hc(hidden, d["attn_hc"].hc_norm.weight,
                            d["attn_hc"].input_mix_weight_down.weight,
                            d["attn_hc"].input_mix_weight_up.weight,
                            d["attn_hc"].block_inject_weight.weight)
        branch, ncs, nrs = ref_gdn(mixed, d["gdn_w"], cs, rs)
        new_states.append((ncs, nrs))
        hidden = inject(hidden, branch, inj)
        mixed2, inj2 = ref_hc(hidden, d["mlp_hc"].hc_norm.weight,
                              d["mlp_hc"].input_mix_weight_down.weight,
                              d["mlp_hc"].input_mix_weight_up.weight,
                              d["mlp_hc"].block_inject_weight.weight)
        moe_out = ref_moe(mixed2.reshape(1, H), d["moe"].gate.weight, d["ex"])
        hidden = inject(hidden, moe_out.reshape(1, 1, H), inj2)
    final = ref_hc(hidden, mixer.hc_norm.weight,
                   mixer.input_mix_weight_down.weight,
                   mixer.input_mix_weight_up.weight, None, use_inject=False)
    logits = mx.matmul(final, mx.transpose(lm_head.weight))
    mx.eval(logits)
    return logits, new_states


def logit_sha256(logits) -> str:
    row = np.asarray(logits.reshape(-1).astype(mx.float32), dtype="<f4")
    return hashlib.sha256(row.tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Build mode                                                                   #
# --------------------------------------------------------------------------- #

def build_vectors(out: Path, steps: int, seed: int):
    cfg, layers, mixer, embed, lm_head = build(seed)

    weights = out / "weights"
    weights.mkdir(parents=True, exist_ok=True)

    def save(name, arr):
        mx.save(str(weights / f"{name}.npy"), arr)

    # Global weights.
    save("embed_w", embed.weight)
    save("lm_head_w", lm_head.weight)
    save("mix_norm_w", mixer.hc_norm.weight)
    save("mix_down_w", mixer.input_mix_weight_down.weight)
    save("mix_up_w", mixer.input_mix_weight_up.weight)

    # Per-layer weights.
    for i, d in enumerate(layers):
        a, g, m, moe = d["attn_hc"], d["gdn"], d["mlp_hc"], d["moe"]
        save(f"L{i}_attn_norm_w", a.hc_norm.weight)
        save(f"L{i}_attn_down_w", a.input_mix_weight_down.weight)
        save(f"L{i}_attn_up_w", a.input_mix_weight_up.weight)
        save(f"L{i}_attn_inject_w", a.block_inject_weight.weight)
        for key, arr in d["gdn_w"].items():
            save(f"L{i}_gdn_{key}", arr)
        save(f"L{i}_mlp_norm_w", m.hc_norm.weight)
        save(f"L{i}_mlp_down_w", m.input_mix_weight_down.weight)
        save(f"L{i}_mlp_up_w", m.input_mix_weight_up.weight)
        save(f"L{i}_mlp_inject_w", m.block_inject_weight.weight)
        save(f"L{i}_gate_w", moe.gate.weight)
        for key, arr in d["ex"].items():
            save(f"L{i}_expert_{key}", arr)

    # Flat config (the documented model_adapter key=value format plus the few
    # extra scalars the standalone executable needs).
    config_lines = [
        "model_family=QWEN4_EXP",
        f"hidden_size={H}",
        f"hc_count={HC}",
        f"hc_lowrank={LOW}",
        f"n_layers={N_LAYERS}",
        f"vocab={VOCAB}",
        f"eps={EPS}",
        f"n_experts={E}",
        f"top_k={TOP_K}",
        f"group_size={GROUP}",
        f"moe_intermediate={INTER}",
        "bits=4",
        f"key_dim={KEY_DIM}",
        f"value_dim={VALUE_DIM}",
        f"num_k_heads={HK}",
        f"num_v_heads={HV}",
        f"head_k_dim={DK}",
        f"head_v_dim={DV}",
        f"conv_kernel={K}",
        "output_gate_type=sigmoid",
        f"prompt_token={PROMPT_TOKEN}",
        "layer_types=" + ",".join(["linear_attention"] * N_LAYERS),
    ]
    (out / "config.txt").write_text("\n".join(config_lines) + "\n")

    # Greedy decode: real modules + primitive mirror, assert bit-exact per step.
    caches = zero_caches()
    pstates = [
        (mx.zeros((1, K - 1, CONV_DIM)), mx.zeros((1, HV, DV, DK)))
        for _ in range(N_LAYERS)
    ]
    token = PROMPT_TOKEN
    token_ids = [token]
    records = []
    for step in range(steps):
        logits = real_step(layers, mixer, embed, lm_head, caches, token)
        plogits, pstates = primitive_step(
            layers, mixer, lm_head, embed.weight, pstates, token)
        if not bits_equal(logits, plogits):
            raise SystemExit(
                f"step {step}: REAL modules != C++ primitive formulas "
                f"(bit-exact self-check failed)")
        next_tok = int(np.asarray(mx.argmax(logits.reshape(-1))).item())
        digest = logit_sha256(logits)
        records.append({
            "step": step,
            "token_in": token,
            "token_out": next_tok,
            "logit_sha256": digest,
        })
        mx.save(str(out / f"expected_logits_step{step}.npy"),
                logits.reshape(-1).astype(mx.float32))
        print(f"python step {step}: {token} -> {next_tok}  sha256={digest}")
        token = next_tok
        token_ids.append(token)

    mx.save(str(out / "expected_token_ids.npy"),
            mx.array(np.array(token_ids, dtype=np.int32)))
    (out / "expected.json").write_text(
        json.dumps({"seed": seed, "steps": steps, "records": records,
                    "token_ids": token_ids}, indent=2) + "\n")
    print(f"wrote native-generate vectors to {out} (weights in {weights})")
    return records


# --------------------------------------------------------------------------- #
# Check mode                                                                   #
# --------------------------------------------------------------------------- #

LINE_RE = re.compile(
    r"step=(\d+)\s+in=(\d+)\s+out=(\d+)\s+sha256=([0-9a-fA-F]{64})")


def check_output(out: Path, binary: str, steps: int):
    expected = json.loads((out / "expected.json").read_text())["records"]
    weights = out / "weights"
    config = out / "config.txt"
    proc = subprocess.run(
        [binary, str(weights), str(config), str(steps)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"C++ binary exited {proc.returncode}\nSTDOUT:\n{proc.stdout}"
            f"\nSTDERR:\n{proc.stderr}")
    got = {}
    for line in proc.stdout.splitlines():
        m = LINE_RE.search(line)
        if m:
            got[int(m.group(1))] = {
                "in": int(m.group(2)), "out": int(m.group(3)),
                "sha256": m.group(4).lower(),
            }
    if len(got) != steps:
        raise SystemExit(
            f"expected {steps} C++ step lines, parsed {len(got)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    for rec in expected:
        g = got[rec["step"]]
        if g["in"] != rec["token_in"] or g["out"] != rec["token_out"]:
            raise SystemExit(
                f"step {rec['step']} token ids differ: "
                f"C++ {g['in']}->{g['out']} vs Python "
                f"{rec['token_in']}->{rec['token_out']}")
        if g["sha256"] != rec["logit_sha256"]:
            raise SystemExit(
                f"step {rec['step']} logit digest differs: "
                f"C++ {g['sha256']} vs Python {rec['logit_sha256']}")
    print("test_native_generate OK "
          f"({steps} steps, token ids + logit digests bit-exact)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors_native_generate")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--binary", default="/tmp/native_generate")
    args = ap.parse_args()
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    if args.check:
        check_output(out, args.binary, args.steps)
    else:
        build_vectors(out, args.steps, args.seed)


if __name__ == "__main__":
    main()
