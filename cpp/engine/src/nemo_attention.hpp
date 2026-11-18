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
// STATUS: bf16 KV is bit-exact (tests/test_nemo_attention.cpp). The production
// default INT8 affine block KV path (FORMAT="affine-int8-g64") is implemented
// by nemo_attention_int8_forward() below (tests/test_nemo_int8.cpp).
#pragma once

#include <optional>
#include <utility>
#include <vector>

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

// ---- Production-default INT8 affine block KV path ---------------------------
//
// Mirrors models/nemotron35/attention.py with a real AttentionCache:
//   keys = mx.quantize(k, group_size=64, bits=8)   (packed uint32 + bf16 s/b)
//   attend(): single block -> dequantize + mx.fast.SDPA
//             multi block  -> quantized_matmul running-softmax loop
// Constants (attention.py:12-15): FORMAT=affine-int8-g64, GROUP_SIZE=64,
// BLOCK_TOKENS=4096, QUERY_TOKENS=128.
//
// One packed block is the exact state of an mlx_lm QuantizedKVCache block:
//   packed  uint32 [B, Hkv, n, D/4]
//   scales  bf16   [B, Hkv, n, D/GROUP_SIZE]
//   biases  bf16   [B, Hkv, n, D/GROUP_SIZE]
// (n <= BLOCK_TOKENS; words*4/GROUP_SIZE == D/GROUP_SIZE scale columns.)
struct NemoInt8Block {
  mx::array packed_k, scales_k, biases_k;  // keys group  (packed, scales, biases)
  mx::array packed_v, scales_v, biases_v;  // values group
  int offset = 0;  // tokens currently stored in this block
  NemoInt8Block(mx::array pk, mx::array sk, mx::array bk,
                mx::array pv, mx::array sv, mx::array bv, int off)
      : packed_k(std::move(pk)), scales_k(std::move(sk)), biases_k(std::move(bk)),
        packed_v(std::move(pv)), scales_v(std::move(sv)), biases_v(std::move(bv)),
        offset(off) {}
};

struct NemoInt8Cache {
  std::vector<NemoInt8Block> blocks;
  int offset = 0;  // total tokens appended across all blocks
  // Production BLOCK_TOKENS=4096. Tests may lower this to exercise the
  // multi-block loop with a short context (the Python reference's global
  // BLOCK_TOKENS is monkeypatched to the same value in that case).
  int block_tokens = 4096;
};

// x: [B, S, hidden_size] bf16 -> [B, S, hidden_size] bf16. S=1 is the decode
// case (always single-block for prompts < 4096 tokens); S>1 prefill also works.
mx::array nemo_attention_int8_forward(const mx::array& x,
                                      const NemoAttentionWeights& w,
                                      const NemoAttentionConfig& cfg,
                                      NemoInt8Cache& cache,
                                      NemoAttentionDebug* debug = nullptr);

}  // namespace veystrix::engine
