#include "nemo_attention.hpp"

#include <mlx/mlx.h>

#include <cmath>
#include <utility>

namespace veystrix::engine {

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

}  // namespace veystrix::engine
