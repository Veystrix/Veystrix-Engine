// Foundation of native Qwen4 MTP (multi-token prediction) / speculative decode.
//
// Bit-exact C++ port of the *distinctive* MTP computation and the lossless
// verify/accept/reject logic, mirroring:
//   models/flashnext/mtp.py   Qwen4ExpMTPModule.fuse_inputs / __call__ / forward
//   models/flashnext/mtp.py   attach() -> one Qwen4ExpDecoderLayer, fresh cache
//   veystrix/sampling.py      Sampler.verify_point_mass
//
// Scope of this foundation (see docs/engineering/mtp_port.md):
//   * mtp_fuse_inputs   : embedding+hidden fusion (the MTP-specific primitive).
//   * mtp_draft_forward : one full draft step -- fusion -> one QSA decoder
//                         layer -> hyper-connection mixer -> tied LM head.
//                         The FRESH-cache single-token draft step (position 0)
//                         is the legacy path. The REAL-checkpoint path also
//                         accepts a nonzero position + a PREPOPULATED QSA KV
//                         cache (see MTPDraftCache / mtp_draft_forward_real):
//                         RoPE is evaluated at cache.offset and attention is a
//                         dense softmax over the past keys + the current token.
//                         The QSA indexer returns None while key_len <= 2048,
//                         so dense is correct for short contexts.
//   * mtp_verify        : point-mass accept/reject + redraw, lossless.
//
// The LM head here is TIED to embed_tokens (tie_word_embeddings=True, the path
// exercised by the synthetic golden vectors). For an untied checkpoint, add a
// separate lm_head weight and use it in place of `w.emb` in the final matmul —
// see mtp.py::forward, which branches on args.tie_word_embeddings.
//
// Everything is float32 and uses the same MLX 0.32.2 ops as Python, so the
// tests compare raw bytes against vectors produced by the REAL Python modules
// on a tiny synthetic config (tools/gen_mtp_vectors.py). No checkpoint needed.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "engine.hpp"
#include "qsa_full.hpp"
#include "sampling.hpp"

namespace mx = mlx::core;

namespace veystrix::engine {

// Model geometry for one MTP draft head. Mirrors fields of the synthetic
// TextConfig used by gen_mtp_vectors.py (a replace()'d copy drives the single
// Qwen4ExpDecoderLayer with layer_types=["qwen_sparse_attention"]).
struct MTPConfig {
  int hidden_size = 0;      // H
  int hc_count = 0;         // HC (Qwen4 hyper-connection streams)
  int hc_lowrank = 0;       // low rank of the gated-residual mix
  int n_heads = 0;          // attention query heads
  int n_kv_heads = 0;       // attention KV heads (GQA)
  int head_dim = 0;         // per-head dim D
  int n_experts = 0;        // routed experts
  int top_k = 0;            // experts per token
  int inter = 0;            // routed expert intermediate size
  int shared_inter = 0;     // shared expert intermediate size
  int vocab = 0;            // vocabulary size
  float eps = 1e-6f;        // RMSNorm epsilon
  float rope_base = 10000.0f;
  float partial_rotary_factor = 0.5f;
  std::vector<int> mrope_section;   // e.g. {2,1,1}, sums to rotary dim
  int max_position_embeddings = 2048;
  int position = 0;         // draft-step position (0 for the fresh-cache step)

  // QSA indexer geometry. Kept for full-integration parity; the dense
  // single-token draft path does not consume the indexer (it returns None).
  int indexer_n_heads = 0;
  int indexer_head_dim = 0;
  int indexer_budget = 0;
  int indexer_compress_ratio = 1;

  // Real-checkpoint conventions. The synthetic golden vectors use the
  // unpatched (1 + weight) gain and a tied LM head; the real sidecar has
  // one-centered gains (see models/flashnext/patch_rmsnorm.py) and an untied
  // lm_head (tie_word_embeddings=False). Defaults keep the synthetic path
  // byte-identical.
  bool one_centered_norms = false;
  bool tie_word_embeddings = true;
};

// Every weight of one MTP draft head, verbatim from the real module (float32).
// nn.Linear weights are [out, in] and applied as x @ W.T; norm weights are the
// RAW zero-centered gains (the Qwen4 RMSNorm multiplies by 1.0 + weight).
struct MTPWeights {
  mx::array emb;                    // [V, H]  embed_tokens.weight (tied LM head)
  mx::array pre_fc_norm_embedding;  // [H]
  mx::array fc_embedding;           // [H, H]
  mx::array pre_fc_norm_hidden;     // [HC*H]
  mx::array fc_hidden;              // [H, H]

  // layer.attn_hyper_connection
  mx::array attn_hc_norm;           // [HC*H]
  mx::array attn_down;              // [LOW, HC*H]
  mx::array attn_up;                // [HC*H, LOW]
  mx::array attn_inject;            // [HC, HC*H]
  // layer.self_attn
  mx::array q_proj;                 // [n_heads*head_dim*2, H]
  mx::array k_proj;                 // [n_kv_heads*head_dim, H]
  mx::array v_proj;                 // [n_kv_heads*head_dim, H]
  mx::array o_proj;                 // [H, n_heads*head_dim]
  mx::array q_norm;                 // [head_dim]
  mx::array k_norm;                 // [head_dim]
  // layer.mlp_hyper_connection
  mx::array mlp_hc_norm;            // [HC*H]
  mx::array mlp_down;               // [LOW, HC*H]
  mx::array mlp_up;                 // [HC*H, LOW]
  mx::array mlp_inject;             // [HC, HC*H]
  // layer.mlp (dense SwitchGLU + shared expert, tiny synthetic config)
  mx::array gate;                   // [n_experts, H]
  mx::array sw_gate;                // [n_experts, inter, H]
  mx::array sw_up;                  // [n_experts, inter, H]
  mx::array sw_down;                // [n_experts, H, inter]
  mx::array shared_gate;            // [shared_inter, H]
  mx::array shared_up;              // [shared_inter, H]
  mx::array shared_down;            // [H, shared_inter]
  mx::array shared_gate_logits;     // [1, H]
  // hyper_connection_mixer (use_combine=False: no inject weight)
  mx::array mix_hc_norm;            // [HC*H]
  mx::array mix_down;               // [LOW, HC*H]
  mx::array mix_up;                 // [HC*H, LOW]

  MTPWeights() = delete;  // mx::array has no default constructor
  MTPWeights(
      mx::array a_emb, mx::array a_pre_fc_norm_embedding, mx::array a_fc_embedding,
      mx::array a_pre_fc_norm_hidden, mx::array a_fc_hidden,
      mx::array a_attn_hc_norm, mx::array a_attn_down, mx::array a_attn_up,
      mx::array a_attn_inject, mx::array a_q_proj, mx::array a_k_proj,
      mx::array a_v_proj, mx::array a_o_proj, mx::array a_q_norm,
      mx::array a_k_norm, mx::array a_mlp_hc_norm, mx::array a_mlp_down,
      mx::array a_mlp_up, mx::array a_mlp_inject, mx::array a_gate,
      mx::array a_sw_gate, mx::array a_sw_up, mx::array a_sw_down,
      mx::array a_shared_gate, mx::array a_shared_up, mx::array a_shared_down,
      mx::array a_shared_gate_logits, mx::array a_mix_hc_norm,
      mx::array a_mix_down, mx::array a_mix_up)
      : emb(std::move(a_emb)),
        pre_fc_norm_embedding(std::move(a_pre_fc_norm_embedding)),
        fc_embedding(std::move(a_fc_embedding)),
        pre_fc_norm_hidden(std::move(a_pre_fc_norm_hidden)),
        fc_hidden(std::move(a_fc_hidden)),
        attn_hc_norm(std::move(a_attn_hc_norm)),
        attn_down(std::move(a_attn_down)),
        attn_up(std::move(a_attn_up)),
        attn_inject(std::move(a_attn_inject)),
        q_proj(std::move(a_q_proj)),
        k_proj(std::move(a_k_proj)),
        v_proj(std::move(a_v_proj)),
        o_proj(std::move(a_o_proj)),
        q_norm(std::move(a_q_norm)),
        k_norm(std::move(a_k_norm)),
        mlp_hc_norm(std::move(a_mlp_hc_norm)),
        mlp_down(std::move(a_mlp_down)),
        mlp_up(std::move(a_mlp_up)),
        mlp_inject(std::move(a_mlp_inject)),
        gate(std::move(a_gate)),
        sw_gate(std::move(a_sw_gate)),
        sw_up(std::move(a_sw_up)),
        sw_down(std::move(a_sw_down)),
        shared_gate(std::move(a_shared_gate)),
        shared_up(std::move(a_shared_up)),
        shared_down(std::move(a_shared_down)),
        shared_gate_logits(std::move(a_shared_gate_logits)),
        mix_hc_norm(std::move(a_mix_hc_norm)),
        mix_down(std::move(a_mix_down)),
        mix_up(std::move(a_mix_up)) {}
};

// Real-checkpoint weights for one MTP draft head. Unlike the synthetic
// MTPWeights above, these stay in the checkpoint's packed affine-Q4 layout:
// projections are `Linear` (dense or QPacked), the routed experts are an
// `ExpertWeights` triplet consumed by q4_projection/gather_qmm, and the
// tokenizer embedding / untied LM head are QPacked tables. Loading them with
// the tool's DenseLoader avoids materializing ~5 GiB of bf16 expert arrays and
// matches Python's QuantizedLinear / gather_qmm arithmetic exactly. Norm gains
// are the RAW checkpoint values; with cfg.one_centered_norms=true they are used
// as-is (the patched production convention).
struct MTPRealWeights {
  QPacked embed;                   // language_model.model.embed_tokens
  mx::array pre_fc_norm_embedding;
  Linear fc_embedding;
  mx::array pre_fc_norm_hidden;
  Linear fc_hidden;

  mx::array attn_hc_norm;
  Linear attn_down, attn_up, attn_inject;
  Linear q_proj, k_proj, v_proj, o_proj;
  mx::array q_norm, k_norm;

  mx::array mlp_hc_norm;
  Linear mlp_down, mlp_up, mlp_inject;
  Linear gate;                     // router (unquantized BF16 weight)
  ExpertWeights experts;           // mlp.switch_mlp (packed Q4 triplets)
  Linear shared_gate, shared_up, shared_down, shared_gate_logits;

  mx::array mix_hc_norm;
  Linear mix_down, mix_up;

  QPacked lm_head;                 // language_model.lm_head (untied)

  MTPRealWeights() = delete;
  MTPRealWeights(
      QPacked a_embed, mx::array a_pre_fc_norm_embedding, Linear a_fc_embedding,
      mx::array a_pre_fc_norm_hidden, Linear a_fc_hidden,
      mx::array a_attn_hc_norm, Linear a_attn_down, Linear a_attn_up,
      Linear a_attn_inject, Linear a_q_proj, Linear a_k_proj, Linear a_v_proj,
      Linear a_o_proj, mx::array a_q_norm, mx::array a_k_norm,
      mx::array a_mlp_hc_norm, Linear a_mlp_down, Linear a_mlp_up,
      Linear a_mlp_inject, Linear a_gate, ExpertWeights a_experts,
      Linear a_shared_gate, Linear a_shared_up, Linear a_shared_down,
      Linear a_shared_gate_logits, mx::array a_mix_hc_norm,
      Linear a_mix_down, Linear a_mix_up, QPacked a_lm_head)
      : embed(std::move(a_embed)),
        pre_fc_norm_embedding(std::move(a_pre_fc_norm_embedding)),
        fc_embedding(std::move(a_fc_embedding)),
        pre_fc_norm_hidden(std::move(a_pre_fc_norm_hidden)),
        fc_hidden(std::move(a_fc_hidden)),
        attn_hc_norm(std::move(a_attn_hc_norm)),
        attn_down(std::move(a_attn_down)), attn_up(std::move(a_attn_up)),
        attn_inject(std::move(a_attn_inject)),
        q_proj(std::move(a_q_proj)), k_proj(std::move(a_k_proj)),
        v_proj(std::move(a_v_proj)), o_proj(std::move(a_o_proj)),
        q_norm(std::move(a_q_norm)), k_norm(std::move(a_k_norm)),
        mlp_hc_norm(std::move(a_mlp_hc_norm)),
        mlp_down(std::move(a_mlp_down)), mlp_up(std::move(a_mlp_up)),
        mlp_inject(std::move(a_mlp_inject)), gate(std::move(a_gate)),
        experts(std::move(a_experts)),
        shared_gate(std::move(a_shared_gate)), shared_up(std::move(a_shared_up)),
        shared_down(std::move(a_shared_down)),
        shared_gate_logits(std::move(a_shared_gate_logits)),
        mix_hc_norm(std::move(a_mix_hc_norm)),
        mix_down(std::move(a_mix_down)), mix_up(std::move(a_mix_up)),
        lm_head(std::move(a_lm_head)) {}
};

// Qwen4ExpMTPModule.fuse_inputs:
//   fused = (fc_embedding(rmsnorm(emb))[...,None,:]
//            + fc_hidden(rmsnorm(hidden).reshape(...,HC,H))).reshape(...,HC*H)
// token_embeddings : [B, S, H]
// hidden_states    : [B, S, HC*H]
// returns          : [B, S, HC*H]
mx::array mtp_fuse_inputs(const MTPConfig& cfg, const MTPWeights& w,
                          const mx::array& token_embeddings,
                          const mx::array& hidden_states);

// Full fresh-cache single-token draft step. Returns draft logits [B, S, V].
// hidden_states   : [B, S, HC*H]  (previous hidden stream fed to fuse_inputs)
// next_token_ids  : [B, S] int32   (the just-sampled token ids)
mx::array mtp_draft_forward(const MTPConfig& cfg, const MTPWeights& w,
                            const mx::array& hidden_states,
                            const mx::array& next_token_ids);

// The dense attention state consumed by one MTP draft step when the QSA cache
// is prepopulated. Mirrors the subset of Python's QSAKVCache that the head's
// single-layer attention reads after `_prepare_projected_qkv` has run:
//
//   past_keys / past_values : [B, n_kv_heads, past_len, head_dim], already
//       RoPE-rotated at their own positions (Python stores keys AFTER rotary
//       via `cache.update_and_fetch(keys, values)`).
//   past_len : == cache.offset. It is both the number of cached tokens and the
//       RoPE start position for the current token (Python builds
//       `position_ids = arange(cache.offset, cache.offset + L)` when
//       position_ids=None). This is the nonzero "draft position".
//
// An empty cache reproduces the legacy fresh-cache S=1 position-0 path. Indexer
// state (index_keys / index_position_ids) is deliberately NOT carried: while
// `key_len <= 2048` the QSA indexer returns None and the mask is None, so dense
// attention is bit-correct (see docs/engineering/mtp_full_parity.md). A sparse
// indexer mask for `past_len >= 2048` is out of scope for this path.
struct MTPDraftCache {
  std::optional<mx::array> past_keys;
  std::optional<mx::array> past_values;
  int past_len = 0;
  bool empty() const { return past_len == 0 && !past_keys.has_value(); }
};

// Full single-token draft step against the REAL checkpoint's MTP sidecar
// (packed Q4 weights, one-centered norms, 512 streamed experts and an untied
// lm_head). Geometry/weights are the exact shard-22 tensors; the caller
// supplies the same hidden stream and token ids as Python's
// `mtp.forward(language, hidden, ids, mtp_cache)`.
// Returns draft logits [B, S, V].
//
// The 4-argument overload is the legacy fresh-cache path (position 0, no past
// KV) and is kept byte-identical for regression. The 5-argument overload takes
// a prepopulated cache: RoPE uses `cache.past_len` and attention runs densely
// over `past_keys/past_values` concatenated with the current key/value.
mx::array mtp_draft_forward_real(const MTPConfig& cfg, const MTPRealWeights& w,
                                  const mx::array& hidden_states,
                                  const mx::array& next_token_ids);
mx::array mtp_draft_forward_real(const MTPConfig& cfg, const MTPRealWeights& w,
                                  const mx::array& hidden_states,
                                  const mx::array& next_token_ids,
                                  const MTPDraftCache& cache);

// Result of mtp_verify, mirroring Sampler.verify_point_mass's return tuple.
struct MTPVerifyResult {
  int accepted = 0;        // how many leading drafts were kept
  int32_t next_token = 0;  // token following the accepted prefix
};

// Lossless point-mass speculative verification (Sampler.verify_point_mass).
// rows    : target logits, one row per verified input token; len == drafts+1.
// drafts  : the point-mass draft token ids.
// params  : the sampler settings used for every row's distribution.
// seed    : mx.random.seed(seed) is applied (Python and C++ share the PRNG).
// seen    : initial produced-token history (for presence penalty).
MTPVerifyResult mtp_verify(const std::vector<mx::array>& rows,
                           const std::vector<int32_t>& drafts,
                           const SamplingParams& params, uint64_t seed,
                           const std::vector<int32_t>& seen = {});

}  // namespace veystrix::engine
