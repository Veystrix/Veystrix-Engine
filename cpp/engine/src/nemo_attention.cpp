#include "nemo_attention.hpp"

#include <mlx/mlx.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace veystrix::engine {

namespace {
// models/nemotron35/attention.py:12-15 -- the affine INT8 block format.
constexpr int kGroupSize = 64;       // GROUP_SIZE
constexpr int kBits = 8;             // bits
// BLOCK_TOKENS is NemoInt8Cache::block_tokens (default 4096); tests lower it.
constexpr int kQueryTokens = 128;    // QUERY_TOKENS

// Slice the token axis (axis 2) of a packed/scales/biases array.
mx::array slice_tokens(const mx::array& a, int start, int end) {
  const int nd = static_cast<int>(a.ndim());
  mx::Shape s(nd, 0), e(nd), st(nd, 1);
  for (int d = 0; d < nd; ++d) e[d] = a.shape(d);
  s[2] = start;
  e[2] = end;
  return mx::slice(a, s, e, st);
}
}  // namespace

mx::array nemo_attention_forward(const mx::array& x,
                                 const NemoAttentionWeights& w,
                                 const NemoAttentionConfig& cfg,
                                 NemoAttentionCache& cache,
                                 NemoAttentionDebug* debug) {
  const int B = static_cast<int>(x.shape(0));
  const int S = static_cast<int>(x.shape(1));
  const int Hq = cfg.num_heads;
  const int Hkv = cfg.num_key_value_heads;
  const int D = cfg.head_dim;

  // ---- q / k / v projections --------------------------------------------
  // NemotronHAttention.__call__:
  //   self.q_proj(x).reshape(B,L,Hq,-1).transpose(0,2,1,3)
  // No norm, no RoPE. linear() == mx::matmul(x, w.T) for dense weights.
  mx::array q = mx::transpose(
      mx::reshape(linear(x, w.q_proj), mx::Shape{B, S, Hq, D}), {0, 2, 1, 3});
  mx::array k = mx::transpose(
      mx::reshape(linear(x, w.k_proj), mx::Shape{B, S, Hkv, D}), {0, 2, 1, 3});
  mx::array v = mx::transpose(
      mx::reshape(linear(x, w.v_proj), mx::Shape{B, S, Hkv, D}), {0, 2, 1, 3});

  if (debug) {
    debug->q = q;
    debug->k = k;
    debug->v = v;
  }

  // ---- KV cache append (KVCache.update_and_fetch logical state) -----------
  // KVCache returns keys[..., :offset, :]; for an all-decode run that is just
  // the concatenation of every appended k/v. Concatenating here yields the
  // exact same bf16 bits (the preallocation + slice copy in KVCache is a no-op
  // on values).
  if (!cache.keys.has_value()) {
    cache.keys = k;
    cache.values = v;
  } else {
    cache.keys = mx::concatenate({*cache.keys, k}, 2);
    cache.values = mx::concatenate({*cache.values, v}, 2);
  }
  cache.offset += S;

  // ---- scaled_dot_product_attention --------------------------------------
  // Python (mlx_lm base.scaled_dot_product_attention, KVCache has no `bits`):
  //   mx.fast.scaled_dot_product_attention(q, keys, values, scale=scale,
  //                                        mask=None, sinks=None)
  // Same MLX fast kernel; mask_mode="" (no mask) for S=1 decode.
  mx::array attended = mx::fast::scaled_dot_product_attention(
      q, *cache.keys, *cache.values, cfg.scale,
      /*mask_mode=*/"", /*mask_arr=*/std::nullopt, /*sinks=*/std::nullopt);
  if (debug) debug->attended = attended;

  // ---- output projection -------------------------------------------------
  // output.transpose(0,2,1,3).reshape(B,L,-1); then o_proj(...)
  mx::array out = mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                              mx::Shape{B, S, Hq * D});
  return linear(out, w.o_proj);
}

// ---------------------------------------------------------------------------
// Production-default INT8 affine block KV path (VEYSTRIX_NEMOTRON_KV=int8).
// Bit-exact port of models/nemotron35/attention.py: StreamedAttention.__call__
// + AttentionCache.append + attend().
// ---------------------------------------------------------------------------
mx::array nemo_attention_int8_forward(const mx::array& x,
                                      const NemoAttentionWeights& w,
                                      const NemoAttentionConfig& cfg,
                                      NemoInt8Cache& cache,
                                      NemoAttentionDebug* debug) {
  const int B = static_cast<int>(x.shape(0));
  const int S = static_cast<int>(x.shape(1));
  const int Hq = cfg.num_heads;
  const int Hkv = cfg.num_key_value_heads;
  const int D = cfg.head_dim;

  // ---- q / k / v projections (identical to the bf16 path) -----------------
  mx::array q = mx::transpose(
      mx::reshape(linear(x, w.q_proj), mx::Shape{B, S, Hq, D}), {0, 2, 1, 3});
  mx::array k = mx::transpose(
      mx::reshape(linear(x, w.k_proj), mx::Shape{B, S, Hkv, D}), {0, 2, 1, 3});
  mx::array v = mx::transpose(
      mx::reshape(linear(x, w.v_proj), mx::Shape{B, S, Hkv, D}), {0, 2, 1, 3});
  if (debug) {
    debug->q = q;
    debug->k = k;
    debug->v = v;
  }

  // ---- AttentionCache.append: quantize K/V then pack into blocks ----------
  // Python: each update_and_fetch calls mx.quantize(keys_slice, 64, 8, "affine")
  // and writes the (packed uint32, bf16 scales, bf16 biases) into the block.
  // quantize groups along the last axis independently per token, so quantizing
  // the whole S-chunk then slicing the token axis yields identical packed
  // words/scales/biases to quantizing each update_and_fetch slice separately.
  std::vector<mx::array> qk = mx::quantize(k, kGroupSize, kBits, "affine");
  std::vector<mx::array> qv = mx::quantize(v, kGroupSize, kBits, "affine");
  if (qk.size() != 3 || qv.size() != 3)
    throw std::runtime_error("nemo_attention_int8_forward: quantize layout");

  int consumed = 0;
  while (consumed < S) {
    const bool fresh = cache.blocks.empty() ||
                       cache.blocks.back().offset == cache.block_tokens;
    const int room = fresh ? cache.block_tokens
                           : (cache.block_tokens - cache.blocks.back().offset);
    const int take = std::min(S - consumed, room);
    mx::array pk = slice_tokens(qk[0], consumed, consumed + take);
    mx::array sk = slice_tokens(qk[1], consumed, consumed + take);
    mx::array bk = slice_tokens(qk[2], consumed, consumed + take);
    mx::array pv = slice_tokens(qv[0], consumed, consumed + take);
    mx::array sv = slice_tokens(qv[1], consumed, consumed + take);
    mx::array bv = slice_tokens(qv[2], consumed, consumed + take);
    if (fresh) {
      cache.blocks.emplace_back(std::move(pk), std::move(sk), std::move(bk),
                                std::move(pv), std::move(sv), std::move(bv),
                                take);
    } else {
      NemoInt8Block& block = cache.blocks.back();
      block.packed_k = mx::concatenate({block.packed_k, pk}, 2);
      block.scales_k = mx::concatenate({block.scales_k, sk}, 2);
      block.biases_k = mx::concatenate({block.biases_k, bk}, 2);
      block.packed_v = mx::concatenate({block.packed_v, pv}, 2);
      block.scales_v = mx::concatenate({block.scales_v, sv}, 2);
      block.biases_v = mx::concatenate({block.biases_v, bv}, 2);
      block.offset += take;
    }
    cache.offset += take;
    consumed += take;
  }

  // ---- attend() -----------------------------------------------------------
  mx::array attended = [&]() -> mx::array {
  if (cache.blocks.size() == 1) {
    // Single block: dequantize K/V and run the same fused MLX SDPA Python uses.
    // mask="causal" for S>1 prefill chunks, mask=None for S=1 decode.
    const NemoInt8Block& block = cache.blocks[0];
    mx::array dk = mx::dequantize(block.packed_k, block.scales_k,
                                  block.biases_k, kGroupSize, kBits, "affine");
    mx::array dv = mx::dequantize(block.packed_v, block.scales_v,
                                  block.biases_v, kGroupSize, kBits, "affine");
    return mx::fast::scaled_dot_product_attention(
        q, dk, dv, cfg.scale,
        /*mask_mode=*/S > 1 ? "causal" : "",
        /*mask_arr=*/std::nullopt, /*sinks=*/std::nullopt);
  } else {
    // Multi-block: quantized_matmul running-softmax loop with per-block causal
    // masking and exp/max correction (attention.py:117-157). Production diverts
    // S>1 multi-block prefill to a dequant+SDPA path when
    // VEYSTRIX_NEMOTRON_ATTENTION_PREFILL_SDPA=1; that variant is documented as
    // a separate path. This loop is the S=1 decode path for >4096-token context.
    const int G = Hq / Hkv;
    const int offset = cache.offset - S;
    std::vector<mx::array> outputs;
    const mx::array neg_inf = mx::array(-std::numeric_limits<float>::infinity());
    for (int start = 0; start < S; start += kQueryTokens) {
      const int end = std::min(S, start + kQueryTokens);
      const int qlen = end - start;
      mx::array qc = slice_tokens(q, start, end);            // [B,Hq,qlen,D]
      mx::array qf = mx::astype(qc, mx::float32);
      qf = mx::multiply(qf, mx::array(cfg.scale));           // float32 * scale
      qf = mx::reshape(qf, mx::Shape{B, Hkv, G, qlen, D});
      mx::array positions = mx::arange(offset + start, offset + end, 1, mx::int32);

      std::optional<mx::array> previous_max, normalizer, weighted;
      for (size_t bi = 0; bi < cache.blocks.size(); ++bi) {
        const NemoInt8Block& block = cache.blocks[bi];
        const int first = static_cast<int>(bi) * cache.block_tokens;
        if (first >= offset + end) break;
        // Insert a size-1 kv-head axis at -3 so MLX broadcasts over the G query
        // groups: [B,Hkv,n,W] -> [B,Hkv,1,n,W].
        mx::array pk = mx::expand_dims(block.packed_k, -3);
        mx::array sk = mx::expand_dims(block.scales_k, -3);
        mx::array bk = mx::expand_dims(block.biases_k, -3);
        mx::array pv = mx::expand_dims(block.packed_v, -3);
        mx::array sv = mx::expand_dims(block.scales_v, -3);
        mx::array bv = mx::expand_dims(block.biases_v, -3);

        mx::array scores = mx::quantized_matmul(
            qf, pk, sk, bk, /*transpose=*/true, kGroupSize, kBits, "affine");
        if (first + block.offset > offset + start + 1) {
          mx::array keypos =
              mx::arange(first, first + block.offset, 1, mx::int32);
          mx::array valid = mx::reshape(positions, mx::Shape{qlen, 1}) >=
                            mx::reshape(keypos, mx::Shape{1, block.offset});
          scores = mx::where(valid, scores, neg_inf);
        }
        mx::array maximum = mx::max(scores, -1, /*keepdims=*/true);
        if (previous_max) maximum = mx::maximum(maximum, *previous_max);
        mx::array weights = mx::exp(scores - maximum);
        mx::array total = mx::sum(weights, -1, /*keepdims=*/true);
        mx::array result = mx::quantized_matmul(
            weights, pv, sv, bv, /*transpose=*/false, kGroupSize, kBits, "affine");
        if (previous_max) {
          mx::array correction = mx::exp(*previous_max - maximum);
          total = total + *normalizer * correction;
          result = result + *weighted * correction;
        }
        previous_max = maximum;
        normalizer = total;
        weighted = result;
      }
      mx::array out = mx::astype(*weighted / *normalizer, q.dtype());
      outputs.push_back(mx::reshape(out, mx::Shape{B, Hq, qlen, D}));
    }
    return outputs.size() == 1 ? outputs[0] : mx::concatenate(outputs, 2);
  }
  }();

  if (debug) debug->attended = attended;

  // ---- output projection ---------------------------------------------------
  mx::array out = mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                               mx::Shape{B, S, Hq * D});
  return linear(out, w.o_proj);
}

}  // namespace veystrix::engine
