// QSA (Qwen Sparse Attention) indexer primitive for Qwen3.8-Flash-Next.
//
// Bit-exact C++ port of the *pre-RoPE* projection + normalization path of
// `Qwen4ExpQSAIndexer.__call__` in
//   .venv/lib/python3.12/site-packages/mlx_vlm/models/qwen4_exp/language.py
// (the model file backing `Qwen3.8-Flash-Next`). This is the part of production
// decode that turns the hidden states into the normalized indexer query and the
// normalized, mean-pooled indexer keys, right before RoPE / block scoring.
//
// Why pre-RoPE only: the following stages (`_apply_rope` -> MRoPE with
// mrope_section / fused Metal compiled kernels, and `argpartition` block
// selection) are out of scope for a small bit-exact primitive and carry real
// tie-breaking / device-fused-kernel parity risk. Everything in this file uses
// only deterministic, standard MLX ops whose op order is mirrored exactly.
//
// Op order mirrored (see qsa.cpp for the one-to-one mapping):
//   qk        = x @ W.T                         (nn.Linear, bias=False)
//   qk        = qk.reshape(B, S, Hq + Hk, D)
//   query     = qk[:, :, :Hq]                   [B, S, Hq, D]
//   raw_keys  = qk[:, :, Hq:].squeeze(2)        [B, S, D]   (Hk == 1)
//   query     = q_layernorm(query).transpose(0, 2, 1, 3)   [B, Hq, S, D]
//   blocks    = S // compress_ratio
//   pooled    = raw_keys[:, :blocks*ratio].reshape(B, blocks, ratio, D)
//   pooled    = mean(pooled.astype(float32), axis=2).astype(raw_keys.dtype)
//   pooled    = k_layernorm(pooled)             [B, blocks, D]
//   pooled_k  = expand_dims(pooled, 1)          [B, 1, blocks, D]
//
// `q_layernorm` / `k_layernorm` are `Qwen4ExpRMSNorm`:
//   y = x.astype(float32)
//   y = y * rsqrt(mean(square(y), axis=-1, keepdims=True) + eps)
//   y = y * (1.0 + weight.astype(float32))      // "one-centered" gain
//   return y.reshape(x.shape).astype(x.dtype)
#pragma once

#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace veystrix::engine {

// Generic RoPE parameters. Reserved for the RoPE/block-selection stages that
// follow this primitive (not consumed yet) and for future models that share
// the same RMSNorm/matmul backbone but use a different rotary layout.
// Empty `mrope_section` means plain (non-multimodal) RoPE.
struct QSARoPEConfig {
  float base = 10000.0f;
  int max_position_embeddings = 2048;
  std::vector<int> mrope_section;  // e.g. {8, 12, 12} for Qwen MRoPE
  std::string style = "interleaved";
  float attention_scaling = 1.0f;
};

// Model-agnostic configuration for the QSA indexer primitive. No Qwen-only
// constants are baked in: head counts/dims, block geometry, eps and RoPE are
// all supplied by the caller. This lets the same primitive be reused by
// Qwen3.8-Flash-Next, future QSA variants, or any model sharing the ops.
struct QSAConfig {
  int n_heads = 0;         // indexer query heads (Hq)
  int kv_heads = 1;        // indexer key/value heads (Hk); Qwen uses 1
  int head_dim = 0;        // per-head dimension (D)
  int compress_ratio = 1;  // tokens per compressed key block
  int token_budget = 0;    // indexer_budget (reserved for block selection)
  int block_topk = 0;      // token_budget / compress_ratio (reserved)
  float eps = 1e-6f;       // RMSNorm epsilon (config.rms_norm_eps)
  QSARoPEConfig rope;      // reserved for the RoPE stage / future models
};

// Exact Qwen4ExpRMSNorm (no group_size): float32 compute, one-centered weight.
//   x      : [..., D] of any float dtype
//   weight : [D] (gain is 1.0 + weight)
//   eps    : RMSNorm epsilon
// Returns the same shape and dtype as x.
mx::array qsa_rms_norm(const mx::array& x, const mx::array& weight, float eps);

// The pre-RoPE QSA indexer projection + normalization path.
//   x           : hidden states [B, S, H]
//   proj_weight : index_qk_proj weight [ (Hq + Hk) * D, H ]  (bias = False)
//   q_weight    : q_layernorm weight [D]
//   k_weight    : k_layernorm weight [D]
//   cfg         : model-agnostic QSA configuration (see QSAConfig)
// Returns a pair:
//   .first  = normalized query    [B, Hq, S, D]
//   .second = normalized pooled keys [B, 1, S // compress_ratio, D]
struct QSAIndexerOutput {
  mx::array query;       // [B, Hq, S, D]
  mx::array pooled_keys; // [B, 1, blocks, D]
};
QSAIndexerOutput qsa_indexer_pre_rope(
    const mx::array& x,
    const mx::array& proj_weight,
    const mx::array& q_weight,
    const mx::array& k_weight,
    const QSAConfig& cfg);

}  // namespace veystrix::engine
