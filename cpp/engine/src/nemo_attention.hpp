// Nemotron-H attention primitive for decode (S=1).
//
// Bit-exact port of the *bf16 KV* path of models/nemotron35/attention.py
// StreamedAttention.__call__ (the fallback used when VEYSTRIX_NEMOTRON_KV=bf16,
// i.e. when the backend keeps a plain mlx_lm KVCache instead of AttentionCache).
// Nemotron-H attention has NO RoPE and NO q_norm/k_norm:
//
//   q = q_proj(x).reshape(B,L,Hq,D).transpose(0,2,1,3)
//   k = k_proj(x).reshape(B,L,Hkv,D).transpose(0,2,1,3)
//   v = v_proj(x).reshape(B,L,Hkv,D).transpose(0,2,1,3)
//   cache.append(k, v)                       # KVCache.update_and_fetch
//   out = scaled_dot_product_attention(q, cache_keys, cache_values, scale)
//   out = o_proj(out.transpose(0,2,1,3).reshape(B,L,Hq*D))
//
// All dense projections use `linear()` (mx::matmul(x, w.T)); config has
// attention_bias=False so there is no bias. The SDPA is the same MLX fused
// kernel Python calls, with mask=None (decode) and sinks=None.
//
// STATUS: bf16 KV is bit-exact (see tests/test_nemo_attention.cpp). The default
// INT8 affine block KV path (FORMAT="affine-int8-g64") is NOT implemented here;
// it needs mx.quantize/dequantize block packing and is left as a TODO.
#pragma once

#include <optional>

#include "engine.hpp"

namespace veystrix::engine {

struct NemoAttentionConfig {
  int hidden_size = 0;
  int num_heads = 0;
  int num_key_value_heads = 0;
  int head_dim = 0;
  float scale = 0.0f;  // head_dim ** -0.5
};

struct NemoAttentionWeights {
  Linear q_proj, k_proj, v_proj, o_proj;  // dense bf16, no bias
};

// Logical KV cache state (matches KVCache.state, i.e. keys[..., :offset, :]).
struct NemoAttentionCache {
  std::optional<mx::array> keys;   // [B, Hkv, N, D], bf16
  std::optional<mx::array> values; // [B, Hkv, N, D], bf16
  int offset = 0;
};

// Optional per-step intermediates for isolating parity failures.
struct NemoAttentionDebug {
  std::optional<mx::array> q;        // [B, Hq, S, D] pre-cache projection
  std::optional<mx::array> k;        // [B, Hkv, S, D]
  std::optional<mx::array> v;        // [B, Hkv, S, D]
  std::optional<mx::array> attended; // [B, Hq, S, D] SDPA output before o_proj
};

// One decode step. x: [B, S(=1), hidden_size] bf16 -> [B, S, hidden_size] bf16.
mx::array nemo_attention_forward(const mx::array& x,
                                 const NemoAttentionWeights& w,
                                 const NemoAttentionConfig& cfg,
                                 NemoAttentionCache& cache,
                                 NemoAttentionDebug* debug = nullptr);

}  // namespace veystrix::engine
