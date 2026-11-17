// Extension of the verified `run_decoder_step` loop (decoder.hpp) that closes
// the two remaining gaps for a generic Qwen4-Exp decode step:
//
//   1. NON-ZERO shared expert in the MoE block
//      (adaptive_topk._moe_call / Qwen3_5MoeSparseMoeBlock.__call__:
//         y = routed_moe(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x))
//   2. FULL-ATTENTION (QSA sparse) branch composed into the layer loop, so a
//      mixed stack (GDN linear layers + full-attention layers) is handled.
//
// Everything here is built on the already bit-exact primitives:
//   gated_residual / router / moe_block / final_head  (engine.hpp)
//   gdn_layer_forward                                 (gdn_layer.hpp)
//   qsa_rms_norm / qsa_full_indexer / qsa_apply_rope  (qsa*.hpp)
//   mx::fast::scaled_dot_product_attention            (the SAME MLX kernel
//                                                     Python's mx.fast calls)
//
// SCOPE / HONESTY (see decoder_full_port.md for the full write-up):
// -----------------------------------------------------------------
// * `moe_block_with_shared` implements the DENSE shared expert as defined by
//   the real `Qwen3_5MoeMLP` (gate/up/down nn.Linear, silu activation). A
//   checkpoint that stores the shared expert as quantized WSB triplets must
//   substitute q4_projection for the three GLU matmuls; the combine math is
//   identical and is documented in the note.
// * The full-attention branch is bit-exact for a single decode token (B=1,S=1)
//   with 2-D text positions, eager (non-fused) RoPE, and a non-None QSA mask.
//   What remains for full production parity is listed in decoder_full_port.md
//   (fused-Metal RoPE, 3-D MRoPE positions, S>1 prefill, quantized KV cache).
//
// This header intentionally includes only decoder.hpp / qsa_full.hpp — it does
// NOT touch decoder.*, engine.*, qsa*, or CMakeLists.txt.
#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "decoder.hpp"
#include "qsa_full.hpp"

namespace veystrix::engine {

// ---------------------------------------------------------------------------
// 1. Shared expert (dense GLU) + routed MoE combine
// ---------------------------------------------------------------------------

// Weights of the dense shared expert `Qwen3_5MoeMLP` plus its sigmoid gate
// `nn.Linear(dim, 1, bias=False)`. All layouts match nn.Linear [out, in].
struct SharedExpertWeights {
  mx::array gate_w;        // [inter, H]   shared_expert.gate_proj.weight
  mx::array up_w;          // [inter, H]   shared_expert.up_proj.weight
  mx::array down_w;        // [H, inter]   shared_expert.down_proj.weight
  mx::array gate_logits_w; // [1, H]       shared_expert_gate.weight

  SharedExpertWeights(mx::array gw, mx::array uw, mx::array dw, mx::array glw)
      : gate_w(std::move(gw)), up_w(std::move(uw)), down_w(std::move(dw)),
        gate_logits_w(std::move(glw)) {}
};

// routed MoE (Q4 gather_qmm, already verified) + non-zero shared expert:
//   y  = moe_block(x, route, scores, routed)
//   sg = sigmoid(x @ shared_expert_gate.T)        [tokens, 1]
//   s  = W_down ( swiglu(W_gate x, W_up x) )      [tokens, H]
//   return y + sg * s
// `swiglu` is the engine's explicit up*(gate*sigmoid(gate)) formulation.
mx::array moe_block_with_shared(const mx::array& x, const mx::array& route,
                                const mx::array& scores,
                                const ExpertWeights& routed,
                                const SharedExpertWeights& shared);

// ---------------------------------------------------------------------------
// 2. Full-attention (QSA sparse) branch
// ---------------------------------------------------------------------------

// All weights of one `Qwen4ExpAttention` layer: the four dense projections,
// the two one-centered Qwen4ExpRMSNorm gains, and the QSA indexer weights.
// Every layout matches nn.Linear [out, in] / nn.RMSNorm [dim].
struct FullAttentionWeights {
  mx::array q_w;           // [n_heads*head_dim*2, H]  (queries + gate halves)
  mx::array k_w;           // [kv_heads*head_dim, H]
  mx::array v_w;           // [kv_heads*head_dim, H]
  mx::array o_w;           // [H, n_heads*head_dim]
  mx::array q_norm_w;      // [head_dim]  one-centered gain
  mx::array k_norm_w;      // [head_dim]  one-centered gain
  // QSA indexer (`Qwen4ExpQSAIndexer`)
  mx::array index_qk_w;    // [(n_idx_heads+n_idx_kv)*idx_head_dim, H]
  mx::array index_q_norm_w;// [idx_head_dim]
  mx::array index_k_norm_w;// [idx_head_dim]

  FullAttentionWeights(mx::array q, mx::array k, mx::array v, mx::array o,
                       mx::array qn, mx::array kn, mx::array iqk,
                       mx::array iqn, mx::array ikn)
      : q_w(std::move(q)), k_w(std::move(k)), v_w(std::move(v)),
        o_w(std::move(o)), q_norm_w(std::move(qn)), k_norm_w(std::move(kn)),
        index_qk_w(std::move(iqk)), index_q_norm_w(std::move(iqn)),
        index_k_norm_w(std::move(ikn)) {}
};

// Geometry shared by all full-attention layers of a model.
struct FullAttentionConfig {
  int n_heads = 0;         // attention query heads
  int kv_heads = 0;        // attention key/value heads (GQA)
  int head_dim = 0;        // attention per-head dim
  int n_idx_heads = 0;     // indexer query heads
  int n_idx_kv = 1;        // indexer key heads (Qwen uses 1)
  int idx_head_dim = 0;    // indexer per-head dim
  int compress_ratio = 1;  // tokens per compressed key block
  int token_budget = 0;    // indexer_budget
  float eps = 1e-6f;       // RMSNorm epsilon
  QSARotaryParams rope;    // attention + indexer share the SAME rotary_emb
};

// Per-layer mutable cache for a full-attention layer: the QSA indexer state
// (reuses QSACache) plus the dense attention KV cache.
struct FullAttentionCache {
  QSACache indexer;                        // index_keys / index_position_ids
  std::optional<mx::array> keys;           // [B, kv_heads, past_len, head_dim]
  std::optional<mx::array> values;         // [B, kv_heads, past_len, head_dim]
  int offset = 0;                          // production cache.offset (past_len)
};

struct FullAttentionOutput {
  mx::array output;       // [B, S, n_heads*head_dim]  (after o_proj)
  mx::array mask;         // [B, 1, S, key_len] booleans used for SDPA
};

// Compute one `Qwen4ExpAttention` branch for a decode step.
//   x            : [B, S, H]  (the attn hyper-connection `mixed` input)
//   position_ids : [B, S] 2-D text positions for the current token(s)
// Updates `cache` in place (both the indexer state and the KV cache).
// Returns the o_proj output and the exact boolean QSA mask used for SDPA.
FullAttentionOutput full_attention_forward(
    const mx::array& x, const FullAttentionWeights& w,
    const FullAttentionConfig& cfg, FullAttentionCache& cache,
    const mx::array& position_ids);

// ---------------------------------------------------------------------------
// 3. Combined decoder step (linear + full-attention layers, shared MoE expert)
// ---------------------------------------------------------------------------

enum class AttnKind { Linear, FullAttention };

// One layer of the extended loop: the verified base weights (hc + routed MoE
// + optional GDN), the non-zero shared expert, and (for full layers) the dense
// attention weights / kind.
struct DecoderFullLayer {
  DecoderLayerWeights base;
  SharedExpertWeights shared;
  AttnKind attn_kind = AttnKind::Linear;
  std::optional<FullAttentionWeights> full;

  DecoderFullLayer(DecoderLayerWeights b, SharedExpertWeights s,
                   AttnKind k, std::optional<FullAttentionWeights> f = std::nullopt)
      : base(std::move(b)), shared(std::move(s)), attn_kind(k),
        full(std::move(f)) {}
};

struct DecoderFullLayerCache {
  DecoderLayerCache base;  // carries GDN state for linear layers
  FullAttentionCache full; // used for full-attention layers

  DecoderFullLayerCache(DecoderLayerCache b, FullAttentionCache f = {})
      : base(std::move(b)), full(std::move(f)) {}
};

struct DecoderFullDebug {
  std::vector<mx::array> attn_out;   // branch output [B,S,H] per layer
  std::vector<mx::array> moe_out;    // MoE+shared output [B,S,H] per layer
  std::vector<mx::array> hidden;     // hidden after the layer
  std::vector<mx::array> qsa_mask;   // QSA mask for full layers (empty otherwise)
};

// Run ONE decode token through a mixed stack.
//   input_embeddings : [1, 1, H]  (pre-embedding, NOT tiled)
//   position_ids     : [1, 1] current token positions (2-D text positions)
//   caches           : size == cfg.n_layers; updated in place
mx::array run_decoder_step_full(
    const mx::array& input_embeddings,
    const std::vector<DecoderFullLayer>& layers,
    const DecoderHeadWeights& head,
    const DecoderConfig& cfg,
    const FullAttentionConfig& fa_cfg,
    const mx::array& position_ids,
    std::vector<DecoderFullLayerCache>& caches,
    DecoderFullDebug* debug = nullptr);

}  // namespace veystrix::engine
