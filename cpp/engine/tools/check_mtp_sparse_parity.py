#!/usr/bin/env python3
"""Bit-exact Python <-> C++ check of the MTP SPARSE indexer mask path.

This validates the long-context MTP draft attention WITHOUT a checkpoint. It
drives the *real* Python modules (``Qwen4ExpMTPModule`` and its
``Qwen4ExpQSAIndexer``) on the tiny synthetic config from gen_mtp_vectors.py.
The synthetic indexer already selects blocks at short contexts
(``indexer_budget=8``, ``compress_ratio=4`` -> ``block_topk=2`` -> mask when
``key_len >= 12``), so no production constant needs patching: we simply warm a
QSAKVCache to ``past_len=24`` (key_len=25 -> 5 complete blocks > 2) and the
indexer returns a real sparse mask. The C++ side is driven with
``sparse_threshold=1`` via config (the opt-in env
``VEYSTRIX_MTP_FORCE_SPARSE=1`` is an alternative) so the sparse path is entered.

The comparison is raw-byte (uint32/int32 view) equality of:
  * the per-head SDPA output  [B, n_heads, S, head_dim]  (head-level parity),
  * the boolean QSA mask      [B, 1, S, key_len],
  * the full draft logits     [B, S, V].
No tolerance, no "close enough". The C++ driver is emitted into the output
directory and is NEVER committed; only the allowed files in the worktree are
modified by this task.

Usage (from anywhere; no checkpoint, no real_run_guard needed):
  .venv/bin/python <worktree>/cpp/engine/tools/check_mtp_sparse_parity.py \
      --worktree <worktree> --out /tmp/mtp_sparse_parity
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

_TOOLS = Path(__file__).resolve().parent
_REPO = _TOOLS.parents[2]
sys.path.insert(0, str(_TOOLS))
sys.path.insert(0, str(_REPO))
if __import__("importlib").util.find_spec("mlx_vlm") is None:
    _VENV = Path("/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages")
    if _VENV.is_dir():
        sys.path.insert(0, str(_VENV))

from gen_mtp_vectors import (  # noqa: E402
    build_config, H, HC, LOW, NH, NKV, D, NEX, TOPK, INTER,
    SINTER, VOCAB, EPS, NIDX, NIDXKV, IDXD, BUDGET, RATIO, ROPE_BASE,
    MROPE_SECTION, PARTIAL, DATA_SEED, DRAFT_SEED,
)
from mlx_vlm.models.qwen4_exp.language import QSAKVCache  # noqa: E402
from models.flashnext import mtp as mtp_mod  # noqa: E402

PAST_LEN = 24          # key_len = 25 -> 5 complete blocks > block_topk (=2)
WARM_SEED = 77001      # hidden streams / tokens used to warm the QSA cache
FINAL_SEED = 77002     # hidden stream / token for the compared draft step
SPARSE_THRESHOLD = 1   # C++ config gate: force the indexer attempt (opt-in)

# --------------------------------------------------------------------------- #
# C++ parity driver (emitted into --out at run time, never committed).
# --------------------------------------------------------------------------- #
CPP_DRIVER = r'''
#include "mtp.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

static std::string read_all(const std::string& p) {
  std::ifstream f(p); std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}
static int get_int(const std::string& s, const std::string& key, int def) {
  auto k = s.find(key + "=");
  if (k == std::string::npos) return def;
  auto e = s.find('\n', k);
  return std::stoi(s.substr(k + key.size() + 1, e == std::string::npos ? std::string::npos : e - k - key.size() - 1));
}
static float get_float(const std::string& s, const std::string& key, float def) {
  auto k = s.find(key + "=");
  if (k == std::string::npos) return def;
  auto e = s.find('\n', k);
  return std::stof(s.substr(k + key.size() + 1, e == std::string::npos ? std::string::npos : e - k - key.size() - 1));
}
static std::vector<int> get_int_array(const std::string& s, const std::string& key) {
  std::vector<int> out;
  auto k = s.find(key + "=");
  if (k == std::string::npos) return out;
  auto e = s.find('\n', k);
  std::string body = s.substr(k + key.size() + 1, e - k - key.size() - 1);
  std::string cur;
  for (char ch : body) {
    if (ch == ',' || ch == ' ') { if (!cur.empty()) { out.push_back(std::stoi(cur)); cur.clear(); } }
    else cur += ch;
  }
  if (!cur.empty()) out.push_back(std::stoi(cur));
  return out;
}
static void write_raw(const std::string& path, const mx::array& a) {
  mx::array f = mx::reshape(mx::astype(a, mx::float32), {-1});
  mx::eval(f);
  std::FILE* of = std::fopen(path.c_str(), "wb");
  if (!of) throw std::runtime_error("cannot write " + path);
  std::fwrite(f.data<float>(), sizeof(float), f.size(), of);
  std::fclose(of);
}
static void write_raw_i32(const std::string& path, const mx::array& a) {
  mx::array f = mx::reshape(mx::astype(a, mx::int32), {-1});
  mx::eval(f);
  std::FILE* of = std::fopen(path.c_str(), "wb");
  if (!of) throw std::runtime_error("cannot write " + path);
  std::fwrite(f.data<int32_t>(), sizeof(int32_t), f.size(), of);
  std::fclose(of);
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s <dir>\n", argv[0]); return 2; }
  const std::string dir = argv[1];
  try {
    const std::string cfg = read_all(dir + "/config.txt");
    MTPConfig c;
    c.hidden_size = get_int(cfg, "hidden_size", 0);
    c.hc_count = get_int(cfg, "hc_count", 0);
    c.hc_lowrank = get_int(cfg, "hc_lowrank", 0);
    c.n_heads = get_int(cfg, "n_heads", 0);
    c.n_kv_heads = get_int(cfg, "n_kv_heads", 0);
    c.head_dim = get_int(cfg, "head_dim", 0);
    c.n_experts = get_int(cfg, "n_experts", 0);
    c.top_k = get_int(cfg, "top_k", 0);
    c.inter = get_int(cfg, "inter", 0);
    c.shared_inter = get_int(cfg, "shared_inter", 0);
    c.vocab = get_int(cfg, "vocab", 0);
    c.eps = get_float(cfg, "eps", 1e-6f);
    c.rope_base = get_float(cfg, "rope_base", 10000.0f);
    c.partial_rotary_factor = get_float(cfg, "partial_rotary_factor", 0.5f);
    c.max_position_embeddings = get_int(cfg, "max_position_embeddings", 128);
    c.mrope_section = get_int_array(cfg, "mrope_section");
    c.indexer_n_heads = get_int(cfg, "indexer_n_heads", 0);
    c.indexer_kv_heads = get_int(cfg, "indexer_kv_heads", 1);
    c.indexer_head_dim = get_int(cfg, "indexer_head_dim", 0);
    c.indexer_budget = get_int(cfg, "indexer_budget", 0);
    c.indexer_compress_ratio = get_int(cfg, "indexer_compress_ratio", 1);
    c.sparse_threshold = get_int(cfg, "sparse_threshold", 2048);

    auto wl = [&](const char* n) { return mx::load(dir + "/weights/" + n + ".npy"); };
    MTPWeights w(
        wl("emb"),
        wl("pre_fc_norm_embedding.weight"), wl("fc_embedding.weight"),
        wl("pre_fc_norm_hidden.weight"), wl("fc_hidden.weight"),
        wl("attn_hc_norm.weight"), wl("attn_down.weight"),
        wl("attn_up.weight"), wl("attn_inject.weight"),
        wl("q_proj.weight"), wl("k_proj.weight"), wl("v_proj.weight"),
        wl("o_proj.weight"), wl("q_norm.weight"), wl("k_norm.weight"),
        wl("mlp_hc_norm.weight"), wl("mlp_down.weight"),
        wl("mlp_up.weight"), wl("mlp_inject.weight"),
        wl("gate.weight"), wl("sw_gate.weight"), wl("sw_up.weight"),
        wl("sw_down.weight"), wl("shared_gate.weight"),
        wl("shared_up.weight"), wl("shared_down.weight"),
        wl("shared_gate_logits.weight"),
        wl("mix_hc_norm.weight"), wl("mix_down.weight"), wl("mix_up.weight"));
    w.set_indexer_weights(wl("indexer.index_qk_proj.weight"),
                          wl("indexer.q_layernorm.weight"),
                          wl("indexer.k_layernorm.weight"));

    auto ld = [&](const char* n) { return mx::load(dir + "/" + n + ".npy"); };
    mx::array hidden = ld("hidden"), ids = ld("tokens");
    mx::array pk = ld("past_k"), pv = ld("past_v");
    mx::array pik = ld("past_index_keys"), pip = ld("past_index_positions");
    const int past_len = get_int(cfg, "past_len", 0);
    mx::eval(hidden, ids, pk, pv, pik, pip);

    MTPDraftCache cache;
    cache.past_keys = pk; cache.past_values = pv; cache.past_len = past_len;
    cache.indexer.index_keys = pik;
    cache.indexer.index_position_ids = pip;
    cache.indexer.offset = past_len;

    MTPSparseDebug dbg;
    mx::array logits = mtp_draft_forward(c, w, hidden, ids, cache, &dbg);
    mx::eval(logits);
    if (!dbg.qsa_mask.has_value() || !dbg.attn_out.has_value())
      throw std::runtime_error("sparse path did not produce mask / attn_out");
    mx::eval(*dbg.qsa_mask, *dbg.attn_out);

    write_raw(dir + "/cpp_logits.f32", logits);
    write_raw(dir + "/cpp_attn_out.f32", *dbg.attn_out);
    write_raw_i32(dir + "/cpp_mask.i32", *dbg.qsa_mask);
    std::printf("MTP_SPARSE_PARITY past_len=%d logits_elems=%zu attn_elems=%zu mask_elems=%zu\n",
                past_len, static_cast<size_t>(mx::reshape(mx::astype(logits, mx::float32), {-1}).size()),
                static_cast<size_t>(mx::reshape(mx::astype(*dbg.attn_out, mx::float32), {-1}).size()),
                static_cast<size_t>(mx::reshape(mx::astype(*dbg.qsa_mask, mx::int32), {-1}).size()));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "MTP_SPARSE_PARITY error: %s\n", e.what());
    return 1;
  }
}
'''


def set_weights_deterministic(module, rng: np.random.Generator,
                              scale: float = 0.2) -> None:
    """Overwrite every float weight with a fixed-seed value.

    gen_mtp_vectors.set_weights() writes through ``vars(module)`` which no longer
    exposes parameters in mlx 0.32.2 (it silently no-ops). Assign to the module
    attribute directly so the golden is reproducible across processes.
    """
    for _, m in module.named_modules():
        w = getattr(m, "weight", None)
        if isinstance(w, mx.array):
            m.weight = mx.array(
                (rng.normal(size=w.shape) * scale).astype(np.float32))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Optional REAL-checkpoint sparse check. It avoids the prohibitive 2050-token
# backbone prefill by injecting a synthetic PREPOPULATED MTP QSA cache (past K/V
# and raw indexer keys) directly into the real module's QSAKVCache. This still
# exercises the real quantized index_qk_proj, the real one-centered indexer
# norms and the real masked SDPA -- i.e. the whole sparse MTP path -- while only
# requiring the cost of a model load (the same as the position-7/11 regressions).
# --------------------------------------------------------------------------- #
REAL_PAST_LEN = 2052   # key_len=2053 -> 513 complete blocks > block_topk=512

REAL_CPP_DRIVER = r'''
#define main real_checkpoint_generate_main_disabled
#include "real_checkpoint_generate.cpp"
#undef main

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s <dir> <checkpoint>\n", argv[0]); return 2; }
  const std::string dir = argv[1], ckpt = argv[2];
  try {
    Config cfg = read_config(dir + "/config.txt");
    const int gs = get_int(cfg, "group_size", 32);
    const int bits = get_int(cfg, "bits", 4);
    DenseLoader dl(ckpt, cfg, gs, bits);

    MTPConfig c = mtp_real::config_from(cfg);
    c.indexer_n_heads = get_int(cfg, "indexer_n_heads");
    c.indexer_kv_heads = get_int(cfg, "indexer_kv_heads", 1);
    c.indexer_head_dim = get_int(cfg, "indexer_head_dim");
    c.indexer_budget = get_int(cfg, "indexer_budget");
    c.indexer_compress_ratio = get_int(cfg, "indexer_compress_ratio", 4);
    c.sparse_threshold = get_int(cfg, "sparse_threshold", 2048);
    MTPRealWeights w = mtp_real::load_weights(dl, c, gs, bits);
    const std::string ip = "language_model.mtp.layers.0.self_attn.indexer.";
    w.set_indexer_weights(
        dl.linear(ip + "index_qk_proj", c.hidden_size),
        dl.raw(ip + "q_layernorm.weight"),
        dl.raw(ip + "k_layernorm.weight"));
    if (!w.has_indexer())
      throw std::runtime_error("real MTP indexer weights not loaded");

    std::vector<float> hv = mtp_real::read_f32(dir + "/hidden.f32");
    mx::array hidden(hv.data(), mx::Shape{1, 1, c.hc_count * c.hidden_size},
                     mx::float32);
    hidden = mx::astype(hidden, mx::bfloat16);
    std::FILE* tf = std::fopen((dir + "/token.txt").c_str(), "r");
    if (!tf) throw std::runtime_error("cannot open token.txt");
    long token = 0;
    if (std::fscanf(tf, "%ld", &token) != 1) throw std::runtime_error("bad token");
    std::fclose(tf);
    std::vector<int32_t> ids{(int32_t)token};
    mx::array idarr(ids.data(), mx::Shape{1, 1}, mx::int32);

    const int past_len = get_int(cfg, "past_len", 0);
    std::vector<float> kv = mtp_real::read_f32(dir + "/past_k.f32");
    std::vector<float> vv = mtp_real::read_f32(dir + "/past_v.f32");
    std::vector<float> ik = mtp_real::read_f32(dir + "/past_index_keys.f32");
    std::vector<float> ipos = mtp_real::read_f32(dir + "/past_index_positions.f32");
    const size_t kwant = (size_t)c.n_kv_heads * past_len * c.head_dim;
    const size_t iwant = (size_t)past_len * c.indexer_head_dim;
    if (kv.size() != kwant || vv.size() != kwant || ik.size() != iwant ||
        ipos.size() != (size_t)past_len)
      throw std::runtime_error("real sparse cache input element count mismatch");
    mx::array pk(kv.data(), mx::Shape{1, c.n_kv_heads, past_len, c.head_dim}, mx::float32);
    mx::array pv(vv.data(), mx::Shape{1, c.n_kv_heads, past_len, c.head_dim}, mx::float32);
    mx::array pik(ik.data(), mx::Shape{1, past_len, c.indexer_head_dim}, mx::float32);
    mx::array pip(ipos.data(), mx::Shape{1, past_len}, mx::float32);
    pk = mx::astype(pk, mx::bfloat16); pv = mx::astype(pv, mx::bfloat16);
    pik = mx::astype(pik, mx::bfloat16);
    pip = mx::astype(pip, mx::int32);
    mx::eval(hidden, idarr, pk, pv, pik, pip);

    MTPDraftCache cache;
    cache.past_keys = pk; cache.past_values = pv; cache.past_len = past_len;
    cache.indexer.index_keys = pik;
    cache.indexer.index_position_ids = pip;
    cache.indexer.offset = past_len;

    MTPSparseDebug dbg;
    mx::array logits = mtp_draft_forward_real(c, w, hidden, idarr, cache, &dbg);
    mx::eval(logits);
    if (!dbg.qsa_mask.has_value() || !dbg.attn_out.has_value())
      throw std::runtime_error("real sparse path did not produce mask / attn_out");
    mx::eval(*dbg.qsa_mask, *dbg.attn_out);

    mx::array f = mx::reshape(mx::astype(logits, mx::float32), {-1});
    mx::array a = mx::reshape(mx::astype(*dbg.attn_out, mx::float32), {-1});
    mx::array m = mx::reshape(mx::astype(*dbg.qsa_mask, mx::int32), {-1});
    mx::eval(f, a, m);
    auto wr = [&](const char* n, const mx::array& arr) {
      std::FILE* of = std::fopen((dir + "/" + n).c_str(), "wb");
      if (!of) throw std::runtime_error("cannot write " + std::string(n));
      const float* p = arr.data<float>();
      std::fwrite(p, sizeof(float), arr.size(), of);  // int32 mask shares 4-byte width
      std::fclose(of);
    };
    wr("cpp_logits.f32", f); wr("cpp_attn_out.f32", a); wr("cpp_mask.i32", m);
    std::printf("MTP_SPARSE_REAL past_len=%d logits=%zu mask=%zu\n", past_len,
                (size_t)f.size(), (size_t)m.size());
    mem_telemetry("mtp-sparse-real");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "MTP_SPARSE_REAL error: %s\n", e.what());
    return 1;
  }
}
'''


def phase_gen_real(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    P = REAL_PAST_LEN
    ckpt = Path(args.checkpoint)

    from models.flashnext.loader import load_streaming

    model, config, _store = load_streaming(
        str(ckpt), verbose=False, keep_vision=False, use_mtp=True)
    language = model.language_model
    tc = config["text_config"]
    H = int(tc["hidden_size"]); HC = int(tc["hc_count"])
    D = int(tc["head_dim"]); NKV = int(tc["num_key_value_heads"])
    IDXD = int(tc["indexer_head_dim"])
    topk = int(tc["indexer_budget"]) // int(tc["indexer_compress_ratio"])
    blocks = (P + 1) // int(tc["indexer_compress_ratio"])
    if blocks <= topk:
        raise SystemExit(f"REAL_PAST_LEN {P} gives {blocks} <= topk {topk}; increase it")

    rot = language.mtp.layers[0].self_attn.rotary_emb
    rot.fused_apply = False
    rot._compiled_apply = None

    # Synthetic prepopulated QSA cache with the REAL checkpoint dtypes/shapes.
    rng = np.random.default_rng(77010)
    cache = [__import__("mlx_vlm.models.qwen4_exp.language", fromlist=["QSAKVCache"]).QSAKVCache()]
    entry = cache[0]
    entry.keys = mx.astype(mx.array(rng.normal(size=(1, NKV, P, D)) * 0.02,
                                    dtype=mx.float32), mx.bfloat16)
    entry.values = mx.astype(mx.array(rng.normal(size=(1, NKV, P, D)) * 0.02,
                                      dtype=mx.float32), mx.bfloat16)
    entry.offset = P
    entry.index_keys = mx.astype(mx.array(rng.normal(size=(1, P, IDXD)) * 0.02,
                                          dtype=mx.float32), mx.bfloat16)
    entry.index_position_ids = mx.array(
        np.broadcast_to(np.arange(P, dtype=np.int32)[None, :], (1, P)))

    import mlx_vlm.models.qwen3_5.language as q35
    orig_sdpa = q35.scaled_dot_product_attention
    capture = {"on": False, "out": None, "mask": None}

    def sdpa_hook(queries, keys, values, cache=None, scale=1.0, mask=None,
                  sinks=None):
        res = orig_sdpa(queries, keys, values, cache=cache, scale=scale,
                        mask=mask, sinks=sinks)
        if capture["on"]:
            capture["out"] = res
            capture["mask"] = mask
        return res

    q35.scaled_dot_product_attention = sdpa_hook
    try:
        frng = np.random.default_rng(77011)
        hidden = mx.astype(
            mx.array(frng.normal(size=(1, 1, HC * H)).astype(np.float32)),
            mx.bfloat16)
        tokens = mx.array(frng.integers(0, int(tc["vocab_size"]), size=(1, 1)),
                          dtype=mx.int32)
        capture["on"] = True
        logits, _hc = mtp_mod.forward(language, hidden, tokens, cache,
                                      return_hidden=True)
        mx.eval(logits)
        capture["on"] = False
    finally:
        q35.scaled_dot_product_attention = orig_sdpa
    mask, attn_out = capture["mask"], capture["out"]
    if not isinstance(mask, mx.array):
        raise SystemExit(f"real indexer returned None at past_len={P} "
                         f"(blocks={blocks}, topk={topk})")
    if attn_out is None:
        raise SystemExit("real SDPA hook captured nothing")
    mx.eval(mask, attn_out)

    # Snapshot only the PRE-final-step state: the final step's indexer call
    # appended one raw key (+ position) to the cache, which C++ re-appends.
    past_k = np.asarray(mx.astype(entry.keys[:, :, :P, :], mx.float32))
    past_v = np.asarray(mx.astype(entry.values[:, :, :P, :], mx.float32))
    past_ik = np.asarray(mx.astype(entry.index_keys[:, :P, :], mx.float32))
    past_ip = np.asarray(mx.astype(entry.index_position_ids[..., :P], mx.int32))

    (out / "hidden.f32").write_bytes(bytes(mx.astype(hidden, mx.float32)))
    (out / "token.txt").write_text(f"{int(tokens.item())}\n")
    (out / "past_k.f32").write_bytes(past_k.astype("<f4").tobytes())
    (out / "past_v.f32").write_bytes(past_v.astype("<f4").tobytes())
    (out / "past_index_keys.f32").write_bytes(past_ik.astype("<f4").tobytes())
    (out / "past_index_positions.f32").write_bytes(past_ip.astype("<f4").tobytes())
    (out / "py_logits.f32").write_bytes(bytes(mx.astype(logits, mx.float32)))
    (out / "py_attn_out.f32").write_bytes(bytes(mx.astype(attn_out, mx.float32)))
    (out / "py_mask.i32").write_bytes(bytes(mx.astype(mask, mx.int32)))

    rp = tc["rope_parameters"]
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]
    lines = []
    def kv(k, v): lines.append(f"{k}={v}")
    for k in ["hidden_size", "hc_count", "hc_lowrank"]:
        kv(k, tc[k])
    kv("n_heads", tc["num_attention_heads"]); kv("kv_heads", NKV)
    kv("head_dim", D); kv("n_experts", tc["num_experts"])
    kv("top_k", tc["num_experts_per_tok"])
    kv("inter", tc["moe_intermediate_size"])
    kv("shared_inter", tc["shared_expert_intermediate_size"])
    kv("vocab", tc["vocab_size"]); kv("eps", tc["rms_norm_eps"])
    quant = config.get("quantization", {})
    kv("group_size", quant.get("group_size", 32)); kv("bits", quant.get("bits", 4))
    kv("rope_base", rp["rope_theta"])
    kv("partial_rotary_factor", rp["partial_rotary_factor"])
    kv("rope_max_pos", tc["max_position_embeddings"])
    kv("rope_mrope", ",".join(str(x) for x in rp["mrope_section"]))
    kv("one_centered_norms", 1)
    kv("tie_word_embeddings", 1 if config.get("tie_word_embeddings", False) else 0)
    kv("indexer_n_heads", tc["indexer_n_heads"])
    kv("indexer_kv_heads", tc["indexer_kv_heads"])
    kv("indexer_head_dim", IDXD)
    kv("indexer_budget", tc["indexer_budget"])
    kv("indexer_compress_ratio", tc["indexer_compress_ratio"])
    kv("sparse_threshold", 2048)
    kv("past_len", P)
    for name in sorted(wmap):
        if name.startswith("language_model.mtp.") or \
           name.startswith("language_model.model.embed_tokens.") or \
           name.startswith("language_model.lm_head."):
            kv("shard." + name, wmap[name])
    (out / "config.txt").write_text("\n".join(lines) + "\n")

    meta = {
        "config": "real checkpoint; synthetic prepopulated MTP QSA cache",
        "checkpoint": str(ckpt), "past_len": P, "key_len": P + 1,
        "block_topk": topk, "max_complete_blocks": blocks,
        "indexer_selected_blocks": True, "mlx_version": mx.__version__,
        "python_logits_sha256": sha256(out / "py_logits.f32"),
        "python_attn_out_sha256": sha256(out / "py_attn_out.f32"),
        "python_mask_sha256": sha256(out / "py_mask.i32"),
    }
    (out / "python_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"real golden: past_len={P} key_len={P+1} blocks={blocks} > topk={topk}; "
          f"logits_sha={meta['python_logits_sha256'][:16]}", flush=True)
    return 0


def dump(out: Path, name: str, a: mx.array) -> None:
    mx.save(str(out / f"{name}.npy"), a)


def phase_gen(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights").mkdir(parents=True, exist_ok=True)
    P = PAST_LEN

    cfg = build_config()
    mtp = mtp_mod.Qwen4ExpMTPModule(cfg)
    embed = __import__("mlx.nn", fromlist=["Embedding"]).Embedding(VOCAB, H)
    rng = np.random.default_rng(DATA_SEED)
    set_weights_deterministic(mtp, rng, 0.2)
    set_weights_deterministic(embed, rng, 0.2)

    # Force the deterministic eager RoPE branch (same convention as the other
    # parity tools); the indexer shares this rotary object with the attention.
    rot = mtp.layers[0].self_attn.rotary_emb
    rot.fused_apply = False
    rot._compiled_apply = None

    # ---- warm the QSA cache for P steps with genuine module calls ----------
    cache = [QSAKVCache()]
    wrng = np.random.default_rng(WARM_SEED)
    for p in range(P):
        h = mx.array(wrng.normal(size=(1, 1, HC * H)).astype(np.float32))
        tok = mx.array(wrng.integers(0, VOCAB, size=(1, 1)).astype(np.int32))
        mtp(h, tok, embed, cache)
    mx.eval([e.state for e in cache])
    entry = cache[0]
    if int(entry.offset) != P:
        raise SystemExit(f"warm-up offset {entry.offset} != {P}")
    if entry.index_keys is None:
        raise SystemExit("warm-up did not populate indexer state")

    # Keep the batch dimension: [B, n_kv_heads, past_len, head_dim].
    past_k = np.asarray(mx.astype(entry.keys[:, :, :P, :], mx.float32))
    past_v = np.asarray(mx.astype(entry.values[:, :, :P, :], mx.float32))
    past_ik = np.asarray(mx.astype(entry.index_keys, mx.float32))
    past_ip = np.asarray(mx.astype(entry.index_position_ids, mx.int32))

    # ---- capture the per-head SDPA output + mask on the final step ----------
    import mlx_vlm.models.qwen3_5.language as q35
    orig_sdpa = q35.scaled_dot_product_attention
    capture = {"on": False, "out": None, "mask": None}

    def sdpa_hook(queries, keys, values, cache=None, scale=1.0, mask=None,
                  sinks=None):
        res = orig_sdpa(queries, keys, values, cache=cache, scale=scale,
                        mask=mask, sinks=sinks)
        if capture["on"]:
            capture["out"] = res
            capture["mask"] = mask
        return res

    q35.scaled_dot_product_attention = sdpa_hook
    try:
        frng = np.random.default_rng(FINAL_SEED)
        hidden = mx.array(frng.normal(size=(1, 1, HC * H)).astype(np.float32))
        tokens = mx.array(frng.integers(0, VOCAB, size=(1, 1)).astype(np.int32))
        capture["on"] = True
        mtp_output, _hc = mtp(hidden, tokens, embed, cache)
        mx.eval(mtp_output)
        logits = embed.as_linear(mtp_output)
        mx.eval(logits)
        capture["on"] = False
    finally:
        q35.scaled_dot_product_attention = orig_sdpa

    mask, attn_out = capture["mask"], capture["out"]
    if mask is None or not isinstance(mask, mx.array):
        raise SystemExit("indexer did NOT return a sparse mask at PAST_LEN "
                         f"{P}; increase PAST_LEN (need key_len//ratio > topk)")
    if attn_out is None:
        raise SystemExit("SDPA hook did not capture per-head attention output")
    mx.eval(mask, attn_out)

    # ---- dump all weights (real module tensors, verbatim) -------------------
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
        # QSA indexer weights (the sparse-path additions).
        "indexer.index_qk_proj.weight": L.self_attn.indexer.index_qk_proj.weight,
        "indexer.q_layernorm.weight": L.self_attn.indexer.q_layernorm.weight,
        "indexer.k_layernorm.weight": L.self_attn.indexer.k_layernorm.weight,
    }
    for name, w in named.items():
        dump(out / "weights", name, w)

    dump(out, "hidden", hidden)
    dump(out, "tokens", tokens)
    dump(out, "past_k", mx.array(past_k, mx.float32))
    dump(out, "past_v", mx.array(past_v, mx.float32))
    dump(out, "past_index_keys", mx.array(past_ik, mx.float32))
    dump(out, "past_index_positions", mx.array(past_ip, mx.int32))
    dump(out, "py_logits", logits)
    dump(out, "py_mask", mask)
    dump(out, "py_attn_out", attn_out)

    lines = [
        f"hidden_size={H}", f"hc_count={HC}", f"hc_lowrank={LOW}",
        f"n_heads={NH}", f"n_kv_heads={NKV}", f"head_dim={D}",
        f"n_experts={NEX}", f"top_k={TOPK}", f"inter={INTER}",
        f"shared_inter={SINTER}", f"vocab={VOCAB}", f"eps={EPS}",
        f"rope_base={ROPE_BASE}", f"partial_rotary_factor={PARTIAL}",
        f"max_position_embeddings=128", f"mrope_section={','.join(map(str, MROPE_SECTION))}",
        f"indexer_n_heads={NIDX}", f"indexer_kv_heads={NIDXKV}",
        f"indexer_head_dim={IDXD}", f"indexer_budget={BUDGET}",
        f"indexer_compress_ratio={RATIO}", f"sparse_threshold={SPARSE_THRESHOLD}",
        f"past_len={P}",
    ]
    (out / "config.txt").write_text("\n".join(lines) + "\n")

    (out / "py_logits.f32").write_bytes(bytes(mx.astype(logits, mx.float32)))
    (out / "py_attn_out.f32").write_bytes(bytes(mx.astype(attn_out, mx.float32)))
    (out / "py_mask.i32").write_bytes(bytes(mx.astype(mask, mx.int32)))
    meta = {
        "config": "tiny synthetic (gen_mtp_vectors.py constants); no checkpoint",
        "past_len": P, "key_len": P + 1, "block_topk": BUDGET // RATIO,
        "max_complete_blocks": (P + 1) // RATIO,
        "indexer_selected_blocks": True, "sparse_threshold": SPARSE_THRESHOLD,
        "warming_seed": WARM_SEED, "final_seed": FINAL_SEED,
        "data_seed": DATA_SEED,
        "logits_shape": list(logits.shape), "attn_out_shape": list(attn_out.shape),
        "mask_shape": list(mask.shape),
        "py_logits_sha256": sha256(out / "py_logits.f32"),
        "py_attn_out_sha256": sha256(out / "py_attn_out.f32"),
        "py_mask_sha256": sha256(out / "py_mask.i32"),
        "rope_branch": "eager (fused_apply=False)", "mlx_version": mx.__version__,
    }
    (out / "python_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"python golden: past_len={P} key_len={P+1} blocks={(P+1)//RATIO} "
          f"> topk={BUDGET//RATIO}; mask={list(mask.shape)}; "
          f"attn_out={list(attn_out.shape)}; logits_sha256={meta['py_logits_sha256'][:16]}",
          flush=True)
    return 0


def compile_driver(args) -> int:
    wt = Path(args.worktree).resolve()
    out = Path(args.out)
    (out / "mtp_sparse_driver.cpp").write_text(CPP_DRIVER)
    binary = out / "mtp_sparse_driver"
    mlxroot = Path(args.mlxroot)
    native_c = wt / "models/flashnext/native_read.c"
    native_o = out / "native_read.o"
    subprocess.run(["xcrun", "clang", "-O2", "-c", str(native_c),
                    "-o", str(native_o)], check=True)
    cmd = [
        "xcrun", "clang++", "-std=c++20", "-O2",
        "-I" + str(mlxroot / "include"),
        "-I" + str(wt / "cpp/engine/src"),
        "-I" + str(wt / "cpp/decode/src"),
    ] + [str(p) for p in sorted((wt / "cpp/engine/src").glob("*.cpp"))] + [
        str(wt / "cpp/decode/src/io_pager.cpp"),
        str(native_o),
        str(out / "mtp_sparse_driver.cpp"),
        "-L" + str(mlxroot / "lib"), "-lmlx",
        "-Wl,-rpath," + str(mlxroot / "lib"),
        "-o", str(binary),
    ]
    print("compiling C++ driver: " + " ".join(cmd[:7]) + " ...", flush=True)
    subprocess.run(cmd, check=True)
    print(f"wrote {binary}", flush=True)
    return 0


def compile_driver_real(args) -> int:
    wt = Path(args.worktree).resolve()
    out = Path(args.out)
    (out / "mtp_sparse_real_driver.cpp").write_text(REAL_CPP_DRIVER)
    binary = out / "mtp_sparse_real_driver"
    mlxroot = Path(args.mlxroot)
    native_c = wt / "models/flashnext/native_read.c"
    native_o = out / "native_read.o"
    subprocess.run(["xcrun", "clang", "-O2", "-c", str(native_c),
                    "-o", str(native_o)], check=True)
    cmd = [
        "xcrun", "clang++", "-std=c++20", "-O2",
        "-I" + str(mlxroot / "include"),
        "-I" + str(wt / "cpp/engine/src"),
        "-I" + str(wt / "cpp/decode/src"),
        "-I" + str(wt / "cpp/engine/tools"),
    ] + [str(p) for p in sorted((wt / "cpp/engine/src").glob("*.cpp"))] + [
        str(wt / "cpp/decode/src/io_pager.cpp"),
        str(native_o),
        str(out / "mtp_sparse_real_driver.cpp"),
        "-L" + str(mlxroot / "lib"), "-lmlx",
        "-Wl,-rpath," + str(mlxroot / "lib"),
        "-o", str(binary),
    ]
    print("compiling REAL C++ driver: " + " ".join(cmd[:7]) + " ...", flush=True)
    subprocess.run(cmd, check=True)
    print(f"wrote {binary}", flush=True)
    return 0


def _report_comparison(out: Path, label: str) -> int:
    report = {
        "attn_out": _cmp(out / "py_attn_out.f32", out / "cpp_attn_out.f32", "<f4"),
        "qsa_mask": _cmp(out / "py_mask.i32", out / "cpp_mask.i32", "<i4"),
        "logits": _cmp(out / "py_logits.f32", out / "cpp_logits.f32", "<f4"),
    }
    meta = json.loads((out / "python_meta.json").read_text())
    for k in ("past_len", "block_topk", "max_complete_blocks"):
        if k in meta:
            report[k] = meta[k]
    (out / f"report_{label}.json").write_text(json.dumps(report, indent=2) + "\n")
    print("=" * 72)
    all_ok = True
    for key, r in report.items():
        if not isinstance(r, dict):
            continue
        print(f"{key:10s}: python_sha={r['python_sha256'][:16]} "
              f"cpp_sha={r['cpp_sha256'][:16]} "
              f"diff={r['differing_elements']}/{r['elements'][0]} "
              f"bit_exact={r['bit_exact']}")
        all_ok = all_ok and r["bit_exact"]
    print(f"MTP SPARSE PARITY ({label}): {'BIT-EXACT PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


def _cmp(path_py: Path, path_cpp: Path, dtype) -> dict:
    a = np.fromfile(path_py, dtype=dtype)
    b = np.fromfile(path_cpp, dtype=dtype)
    sha_a = hashlib.sha256(path_py.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(path_cpp.read_bytes()).hexdigest()
    if a.size == b.size:
        diffs = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
    else:
        diffs = -1
    return {
        "python_sha256": sha_a, "cpp_sha256": sha_b,
        "elements": [int(a.size), int(b.size)], "differing_elements": diffs,
        "bit_exact": bool(a.size == b.size and sha_a == sha_b and diffs == 0),
    }


def phase_compare(args) -> int:
    out = Path(args.out)
    if compile_driver(args) != 0:
        return 1
    # This harness forces Python's eager RoPE (fused_apply=False); the C++ MTP
    # default is the fused Metal kernel, so pin the matching eager branch here.
    import os as _os
    _env = dict(_os.environ)
    _env["VEYSTRIX_CPP_MTP_EAGER_ROPE"] = "1"
    proc = subprocess.run([str(out / "mtp_sparse_driver"), str(out)],
                          capture_output=True, text=True, env=_env)
    print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", flush=True)
    if proc.returncode != 0:
        print(f"C++ driver failed: exit {proc.returncode}", file=sys.stderr)
        return proc.returncode

    report = {
        "attn_out": _cmp(out / "py_attn_out.f32", out / "cpp_attn_out.f32", "<f4"),
        "qsa_mask": _cmp(out / "py_mask.i32", out / "cpp_mask.i32", "<i4"),
        "logits": _cmp(out / "py_logits.f32", out / "cpp_logits.f32", "<f4"),
    }
    meta = json.loads((out / "python_meta.json").read_text())
    report["past_len"] = meta["past_len"]
    report["block_topk"] = meta["block_topk"]
    report["max_complete_blocks"] = meta["max_complete_blocks"]
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    print("=" * 72)
    all_ok = True
    for key, r in report.items():
        if not isinstance(r, dict):
            continue
        print(f"{key:10s}: python_sha={r['python_sha256'][:16]} "
              f"cpp_sha={r['cpp_sha256'][:16]} "
              f"diff={r['differing_elements']}/{r['elements'][0]} "
              f"bit_exact={r['bit_exact']}")
        all_ok = all_ok and r["bit_exact"]
    print(f"MTP SPARSE PARITY (synthetic): {'BIT-EXACT PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


def phase_compare_real(args) -> int:
    out = Path(args.out)
    if compile_driver_real(args) != 0:
        return 1
    import os as _os
    _env = dict(_os.environ)
    _env["VEYSTRIX_CPP_MTP_EAGER_ROPE"] = "1"  # golden forces fused_apply=False
    proc = subprocess.run(
        [str(out / "mtp_sparse_real_driver"), str(out),
         str(Path(args.checkpoint).resolve())],
        capture_output=True, text=True, env=_env)
    print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", flush=True)
    if proc.returncode != 0:
        print(f"REAL C++ driver failed: exit {proc.returncode}", file=sys.stderr)
        return proc.returncode
    return _report_comparison(out, "real")


def main() -> int:
    default_mlxroot = "/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages/mlx"
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mlxroot", default=default_mlxroot)
    ap.add_argument("--phase", choices=["gen", "compare"], default=None)
    ap.add_argument("--real", action="store_true",
                    help="use the real checkpoint with an injected prepopulated cache")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    if args.real and not args.checkpoint:
        raise SystemExit("--real requires --checkpoint")
    common = ["--worktree", args.worktree, "--out", args.out,
              "--mlxroot", args.mlxroot]
    if args.real:
        common += ["--real", "--checkpoint", args.checkpoint]
        if args.phase == "gen":
            return phase_gen_real(args)
        if args.phase == "compare":
            return phase_compare_real(args)
        r1 = subprocess.run([sys.executable, __file__, *common, "--phase", "gen"])
        if r1.returncode != 0:
            return r1.returncode
        return subprocess.run(
            [sys.executable, __file__, *common, "--phase", "compare"]
        ).returncode
    if args.phase == "gen":
        return phase_gen(args)
    if args.phase == "compare":
        return phase_compare(args)
    r1 = subprocess.run([sys.executable, __file__, *common, "--phase", "gen"])
    if r1.returncode != 0:
        return r1.returncode
    return subprocess.run(
        [sys.executable, __file__, *common, "--phase", "compare"]
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
