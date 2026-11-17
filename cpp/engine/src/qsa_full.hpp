// Full QSA (Qwen Sparse Attention) indexer path for Qwen3.8-Flash-Next.
//
// Bit-exact C++ port of the REMAINING stages of `Qwen4ExpQSAIndexer.__call__`
// (the pre-RoPE projection/normalization path already lives in qsa.hpp/.cpp):
//
//   .venv/lib/python3.12/site-packages/mlx_vlm/models/qwen4_exp/language.py
//     Qwen4ExpQSAIndexer.__call__    lines 306-380
//       line 300-304  _apply_rope (MRoPE partial rotary)
//       line 324-327  cache.update_indexer
//       line 329-332  None early return
//       line 334-348  rope on query + mean-pooled keys
//       line 350-352  scores = query @ pooled.T ; sum(max(f32,0)) ; /sqrt(D)
//       line 354-380  causal valid_blocks mask, argpartition, selected_tokens
//     QSAKVCache.update_indexer        lines 54-63
//     _append_indexer_positions        lines 21-38
//   .venv/lib/python3.12/site-packages/mlx_vlm/models/rope_utils.py
//     MRoPERotaryEmbedding.apply_rotary / compute_inv_freq /
//     compute_mrope_frequencies / _apply_interleaved_rotary_pos_emb_axis1
//
// This file is self-contained: it only depends on qsa.hpp (QSAConfig,
// QSAIndexerOutput, qsa_rms_norm) and the MLX C++ API. It does NOT modify any
// existing file. See research_notes/Cerebras inference and IO splitting/
// qsa_full_port.md for verification, integration and caveat details.
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "qsa.hpp"

namespace veystrix::engine {

// Rotary parameters consumed by qsa_apply_rope. These mirror
// MRoPERotaryEmbedding's constructor + __call__:
//   dim              partial-rotary dimension (int(head_dim * partial_factor));
//                    only the first `dim` channels are rotated, the rest pass.
//   base             rope theta (inv_freq = 1 / base**(arange(0,dim,2)/dim)).
//   mrope_section    e.g. {11,11,10}; only used for 3-D (multimodal) positions.
//   style            "interleaved" (Qwen) or "chunked"; 3-D positions only.
//   attention_scaling multiplies cos/sin before the rotate.
struct QSARotaryParams {
  int dim = 0;
  float base = 10000.0f;
  std::vector<int> mrope_section;
  std::string style = "interleaved";
  float attention_scaling = 1.0f;
  int max_position_embeddings = 2048;
};

// Model-agnostic configuration bundle for the full indexer path.
struct QSAFullConfig {
  QSAConfig base;        // dims / heads / ratio / budget / topk / eps
  QSARotaryParams rope;  // rotary layout
};

// The indexer state carried by QSAKVCache (index_keys + index_position_ids).
// Mirrors QSAKVCache.__init__ / update_indexer / _append_indexer_positions.
struct QSACache {
  std::optional<mx::array> index_keys;          // [B, total_len, D]
  std::optional<mx::array> index_position_ids;  // [B, total_len] (2-D positions)
                                                // or [axis, B, total_len] (3-D)
  int offset = 0;  // production cache.offset; `__call__` reads past_len from it

  bool empty() const { return !index_keys.has_value(); }

  // Mirrors QSAKVCache.update_indexer lines 54-63:
  //   if index_keys is None: store as-is
  //   else: concatenate keys on axis=1 and append position ids on axis=-1
  //         (with 2-D <-> 3-D broadcasting from _append_indexer_positions).
  void update_indexer(const mx::array& keys, const mx::array& position_ids);
};

// inv_freq for the rotary embedding, shape [dim / 2], float32:
//   1.0 / (base ** (arange(0, dim, 2).astype(float32) / dim))
mx::array qsa_compute_inv_freq(const QSARotaryParams& rp);

// apply_rotary matching MRoPERotaryEmbedding.apply_rotary (the deterministic
// eager branch; see qsa_full_port.md for the fused-Metal caveat).
//   x            : [..., D]   (any float dtype); first `rp.dim` channels rotate
//   position_ids : [B, T] for text (2-D) or [axis, B, T] for MRoPE (3-D)
// Returns the same shape/dtype as x with partial rotary applied.
mx::array qsa_apply_rope(const mx::array& x, const mx::array& position_ids,
                         const QSARotaryParams& rp);

// The full QSA indexer block-selection stage (Qwen4ExpQSAIndexer.__call__
// lines 306-380). Updates `cache` in place (update_indexer runs before the
// None early-return, exactly as in production).
//
// Returns std::nullopt when max_complete_blocks <= block_topk (production
// returns None). Otherwise returns the boolean QSA mask [B, 1, S, key_len].
std::optional<mx::array> qsa_full_indexer(
    const mx::array& x,
    const Linear& proj_weight,
    const mx::array& q_weight,
    const mx::array& k_weight,
    const QSAFullConfig& cfg,
    QSACache& cache,
    const mx::array& position_ids);

}  // namespace veystrix::engine
