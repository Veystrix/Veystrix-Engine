#!/usr/bin/env python3
"""Bit-exact golden vectors for the MTP (multi-token prediction) foundation.

Everything here uses the REAL production modules and ops -- no reimplementation
of the MTP math -- with a TINY synthetic config and fixed seeds. No checkpoint
and no real benchmark data are used. The C++ primitive (cpp/engine/src/mtp.cpp)
is compared byte-for-byte against these vectors by test_mtp.cpp.

We emit:
  config.json   - tiny synthetic TextConfig values (hidden, heads, experts, eps,
                  rope, mrope_section, ...) consumed by the C++ MTPConfig.
  weights/*.npy - every weight of the real Qwen4ExpMTPModule + embedding, dumped
                  verbatim as float32 (nn.Linear [out,in], norms one-centered).
  fuse_inputs.npy / hidden.npy / embed.npy / next_tokens.npy
                - draft-head inputs and the real fuse_inputs() output.
  logits.npy    - the real mtp.forward() draft logits for (hidden, next_tokens).
  verify/caseN/ - rows.npy, drafts.npy, accepted, next_token, meta.json for the
                  real veystrix.sampling.Sampler.verify_point_mass() path.

Note on "one-centered" gains: Qwen4ExpRMSNorm multiplies by (1.0 + weight).
The dumped norm weights are the RAW zero-centered values; mtp.cpp adds 1.0.

Note on the draft step: mtp.forward() is called with a FRESH mtp cache
(make_cache) and position_ids=None, so the QSA indexer returns None and the
attention is a dense 1-token softmax (output == v). Rotary is evaluated at
position 0 (cos=1, sin=0 -> identity).

Usage (needs mlx 0.32.2 + mlx_vlm + the repo on PYTHONPATH):
  /Users/vincent/Documents/Veystrix/.venv/bin/python \
      cpp/engine/tools/gen_mtp_vectors.py --out cpp/engine/tests/vectors_mtp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

# The engine's venv is not checked into this worktree; fall back to the known
# developer venv that actually carries mlx 0.32.2 + mlx_vlm.
if __import__("importlib").util.find_spec("mlx_vlm") is None:
    _VENV = Path("/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages")
    if _VENV.is_dir():
        sys.path.insert(0, str(_VENV))

from mlx_vlm.models.qwen4_exp.config import TextConfig  # noqa: E402
from mlx_vlm.models.qwen4_exp.language import QSAKVCache  # noqa: E402

from models.flashnext import mtp as mtp_mod  # noqa: E402
from veystrix.sampling import Sampler, Sampling  # noqa: E402


# ---- tiny synthetic config -------------------------------------------------
H = 32                 # hidden_size
HC = 2                 # hc_count (Qwen4 requires > 1)
LOW = 16               # hc_lowrank
NH = 4                 # attention heads
NKV = 1                # kv heads
D = 8                  # head_dim  (rotary dim = D * 0.5 = 4)
NEX = 4                # routed experts
TOPK = 2               # experts per token
INTER = 32             # moe_intermediate_size
SINTER = 32            # shared_expert_intermediate_size
VOCAB = 64
EPS = 1e-6
NIDX = 4               # indexer query heads
NIDXKV = 1
IDXD = 16              # indexer_head_dim
BUDGET = 8
RATIO = 4
ROPE_BASE = 10000.0
MROPE_SECTION = [2, 1, 1]          # sums to rotary dim (D * 0.5)
PARTIAL = 0.5
DATA_SEED = 4242
FUSE_SEED = 9001
DRAFT_SEED = 9002


def build_config() -> TextConfig:
    return TextConfig(
        model_type="qwen4_exp",
        hidden_size=H,
        num_hidden_layers=48,        # mtp.py replace()s this to 1
        num_attention_heads=NH,
        num_key_value_heads=NKV,
        head_dim=D,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
        num_experts=NEX,
        num_experts_per_tok=TOPK,
        shared_expert_intermediate_size=SINTER,
        moe_intermediate_size=INTER,
        rms_norm_eps=EPS,
        vocab_size=VOCAB,
        max_position_embeddings=128,
        hc_count=HC,
        hc_lowrank=LOW,
        layer_types=["qwen_sparse_attention"] * 48,
        indexer_n_heads=NIDX,
        indexer_kv_heads=NIDXKV,
        indexer_head_dim=IDXD,
        indexer_budget=BUDGET,
        indexer_compress_ratio=RATIO,
        rope_parameters={
            "type": "default",
            "mrope_section": MROPE_SECTION,
            "rope_theta": ROPE_BASE,
            "partial_rotary_factor": PARTIAL,
        },
    )


def set_weights(module, rng: np.random.Generator, scale: float = 0.2) -> None:
    """Deterministically overwrite every float weight of a module tree."""
    for _, m in module.named_modules():
        d = vars(m)
        if "weight" in d and isinstance(d["weight"], mx.array):
            w = d["weight"]
            d["weight"] = mx.array(
                (rng.normal(size=w.shape) * scale).astype(np.float32)
            )


def save(d: Path, name: str, a: mx.array) -> None:
    # Preserve dtype: weights are float32, next_tokens must stay int32.
    mx.save(str(d / f"{name}.npy"), a)


def emit_draft(out: Path) -> dict:
    cfg = build_config()
    mtp = mtp_mod.Qwen4ExpMTPModule(cfg)
    embed = nn.Embedding(VOCAB, H)

    rng = np.random.default_rng(DATA_SEED)
    set_weights(mtp, rng, 0.2)
    set_weights(embed, rng, 0.2)

    wdir = out / "weights"
    wdir.mkdir(parents=True, exist_ok=True)

    L = mtp.layers[0]
    named = {
        "emb": embed.weight,
        "pre_fc_norm_embedding.weight": mtp.pre_fc_norm_embedding.weight,
        "fc_embedding.weight": mtp.fc_embedding.weight,
        "pre_fc_norm_hidden.weight": mtp.pre_fc_norm_hidden.weight,
        "fc_hidden.weight": mtp.fc_hidden.weight,
        "attn_hc_norm.weight": L.attn_hyper_connection.hc_norm.weight,
        "attn_down.weight": L.attn_hyper_connection.input_mix_weight_down.weight,
        "attn_up.weight": L.attn_hyper_connection.input_mix_weight_up.weight,
        "attn_inject.weight": L.attn_hyper_connection.block_inject_weight.weight,
        "q_proj.weight": L.self_attn.q_proj.weight,
        "k_proj.weight": L.self_attn.k_proj.weight,
        "v_proj.weight": L.self_attn.v_proj.weight,
        "o_proj.weight": L.self_attn.o_proj.weight,
        "q_norm.weight": L.self_attn.q_norm.weight,
        "k_norm.weight": L.self_attn.k_norm.weight,
        "mlp_hc_norm.weight": L.mlp_hyper_connection.hc_norm.weight,
        "mlp_down.weight": L.mlp_hyper_connection.input_mix_weight_down.weight,
        "mlp_up.weight": L.mlp_hyper_connection.input_mix_weight_up.weight,
        "mlp_inject.weight": L.mlp_hyper_connection.block_inject_weight.weight,
        "gate.weight": L.mlp.gate.weight,
        "sw_gate.weight": L.mlp.switch_mlp.gate_proj.weight,
        "sw_up.weight": L.mlp.switch_mlp.up_proj.weight,
        "sw_down.weight": L.mlp.switch_mlp.down_proj.weight,
        "shared_gate.weight": L.mlp.shared_expert.gate_proj.weight,
        "shared_up.weight": L.mlp.shared_expert.up_proj.weight,
        "shared_down.weight": L.mlp.shared_expert.down_proj.weight,
        "shared_gate_logits.weight": L.mlp.shared_expert_gate.weight,
        "mix_hc_norm.weight": mtp.hyper_connection_mixer.hc_norm.weight,
        "mix_down.weight": mtp.hyper_connection_mixer.input_mix_weight_down.weight,
        "mix_up.weight": mtp.hyper_connection_mixer.input_mix_weight_up.weight,
    }
    for name, w in named.items():
        save(wdir, name, w)

    # Draft-step inputs: one token, fresh cache, position 0.
    frng = np.random.default_rng(FUSE_SEED)
    hidden = mx.array(frng.normal(size=(1, 1, HC * H)).astype(np.float32))
    next_tokens = mx.array(frng.integers(0, VOCAB, size=(1, 1)).astype(np.int32))
    token_emb = embed(next_tokens)

    fused = mtp.fuse_inputs(token_emb, hidden)
    mx.eval(fused)
    save(out, "hidden", hidden)
    save(out, "next_tokens", next_tokens)
    save(out, "token_embed", token_emb)
    save(out, "fuse_inputs", fused)

    # Real end-to-end draft logits (fresh cache, position_ids=None).
    # Mirror mtp_mod.forward() but call the module directly (there is no full
    # language model here); tie_word_embeddings=True => embed.as_linear().
    mx.random.seed(DRAFT_SEED)
    cache = [QSAKVCache()]
    mtp_output, hc_hidden = mtp(hidden, next_tokens, embed, cache)
    mx.eval(mtp_output, hc_hidden)
    logits = embed.as_linear(mtp_output)
    mx.eval(logits)
    save(out, "logits", logits)

    meta = {
        "hidden_size": H, "hc_count": HC, "hc_lowrank": LOW,
        "n_heads": NH, "n_kv_heads": NKV, "head_dim": D,
        "n_experts": NEX, "top_k": TOPK, "inter": INTER, "shared_inter": SINTER,
        "vocab": VOCAB, "eps": EPS,
        "indexer_n_heads": NIDX, "indexer_head_dim": IDXD,
        "indexer_budget": BUDGET, "indexer_compress_ratio": RATIO,
        "rope_base": ROPE_BASE, "mrope_section": MROPE_SECTION,
        "partial_rotary_factor": PARTIAL,
        "data_seed": DATA_SEED, "draft_seed": DRAFT_SEED,
        "mlx_version": mx.__version__,
        "position": 0,
    }
    return meta


# ---- verify / accept-reject vectors ---------------------------------------
def emit_verify(out: Path) -> None:
    vdir = out / "verify"
    vdir.mkdir(parents=True, exist_ok=True)
    cases = []

    def emit(name, vocab, temps, topp, topk, minp, pen, drafts, seed,
             dominant=None, seen=(), drafts_from_argmax=False, boost=None):
        d = vdir / name
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed + 1000)
        rows = rng.normal(size=(len(drafts) + 1, vocab)).astype(np.float32)
        if dominant is not None:
            rows[0, dominant[0]] = dominant[1]
        if boost:
            for row_idx, (tok, val) in boost.items():
                rows[row_idx, tok] = val
        if drafts_from_argmax:
            # Drafts equal the target's top token for each row -> high accept
            # chance, so the accept path (and accept-then-reject) is exercised.
            drafts = [int(np.argmax(rows[i])) for i in range(len(drafts))]
        rows_mx = mx.array(rows)
        drafts_mx = mx.array(drafts, dtype=mx.int32)
        s = Sampler(Sampling(temperature=temps, top_p=topp, top_k=topk,
                             min_p=minp, presence_penalty=pen))
        for t in seen:
            s.observe(t)
        mx.random.seed(seed)
        accepted, next_token = s.verify_point_mass(rows_mx, drafts_mx)
        mx.save(str(d / "rows.npy"), rows_mx)
        mx.save(str(d / "drafts.npy"), drafts_mx)
        (d / "result.json").write_text(json.dumps({
            "accepted": int(accepted), "next_token": int(next_token),
            "vocab": vocab, "temperature": temps, "top_p": topp,
            "top_k": topk, "min_p": minp, "presence_penalty": pen,
            "seed": seed, "seen": list(seen),
        }, indent=2) + "\n")
        cases.append(name)

    # Full-vocabulary path, all drafts accepted / first rejected / later reject.
    emit("case0_accept_all", 32, 1.0, 1.0, 0, 0.0, 0.0, [3, 7, 11], 21)
    emit("case1_reject_first", 32, 1.0, 1.0, 0, 0.0, 0.0, [5, 9, 2], 22)
    emit("case2_reject_later", 32, 1.0, 1.0, 0, 0.0, 0.0, [4, 6, 20], 23)
    # Near point-mass draft (chance > 0.999999 -> auto accept).
    emit("case3_point_mass", 32, 1.0, 1.0, 0, 0.0, 0.0, [7, 7, 7], 24,
         dominant=(7, 20.0))
    # Top-k path (kept + remap) and nucleus/min-p path.
    emit("case4_topk", 64, 1.0, 1.0, 10, 0.0, 0.0, [12, 33, 5], 25)
    emit("case5_topp_minp", 64, 0.8, 0.9, 0, 0.05, 0.0, [8, 40, 17], 26)
    # Presence penalty: accepted drafts enter the seen set.
    emit("case6_presence", 64, 0.7, 0.8, 20, 0.0, 1.5, [3, 10, 27], 27,
         seen=[2, 7])
    # Drafts = target argmax: exercise the accept path / accept-then-reject.
    emit("case7_argmax_chain", 32, 1.0, 1.0, 0, 0.0, 0.0, [0, 0, 0], 28,
         drafts_from_argmax=True)
    emit("case8_argmax_topk", 64, 1.0, 0.9, 10, 0.0, 0.0, [0, 0, 0], 29,
         drafts_from_argmax=True)
    emit("case9_argmax_presence", 64, 0.7, 0.9, 20, 0.0, 1.5, [0, 0, 0], 30,
         drafts_from_argmax=True, seen=[5, 9])
    # First two rows peaked on their drafts (auto-accept), third random ->
    # reject at index 2 and redraw from the target.
    emit("case10_peaked_chain", 32, 1.0, 1.0, 0, 0.0, 0.0, [3, 5, 9], 31,
         boost={0: (3, 12.0), 1: (5, 12.0)})

    (vdir / "cases.txt").write_text("\n".join(cases) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors_mtp")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = emit_draft(out)
    meta["verify_cases"] = 11
    (out / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
    emit_verify(out)
    print(f"wrote MTP golden vectors to {out} (mlx {mx.__version__})")


if __name__ == "__main__":
    main()
