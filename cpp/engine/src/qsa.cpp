#include "qsa.hpp"

namespace veystrix::engine {

mx::array qsa_rms_norm(const mx::array& x, const mx::array& weight, float eps) {
  // Mirrors Qwen4ExpRMSNorm.__call__ (group_size is None in the indexer):
  //   dtype = x.dtype
  //   y = x.astype(mx.float32)
  //   y = y * mx.rsqrt(mx.mean(mx.square(y), axis=-1, keepdims=True) + eps)
  //   y = y * (1.0 + weight.astype(mx.float32))
  //   return y.reshape(x.shape).astype(dtype)
  auto dtype = x.dtype();
  mx::array y = mx::astype(x, mx::float32);
  mx::array ms = mx::mean(mx::square(y), -1, /*keepdims=*/true);
  y = y * mx::rsqrt(ms + eps);
  mx::array gain = 1.0f + mx::astype(weight, mx::float32);
  y = y * gain;
  return mx::astype(mx::reshape(y, x.shape()), dtype);
}

QSAIndexerOutput qsa_indexer_pre_rope(
    const mx::array& x,
    const mx::array& proj_weight,
    const mx::array& q_weight,
    const mx::array& k_weight,
    const QSAConfig& cfg) {
  const int B = static_cast<int>(x.shape()[0]);
  const int S = static_cast<int>(x.shape()[1]);
  const int Hq = cfg.n_heads;
  const int Hk = cfg.kv_heads;
  const int D = cfg.head_dim;
  const int ratio = cfg.compress_ratio;

  // nn.Linear(x @ W.T), bias=False:
  //   qk = self.index_qk_proj(x).reshape(B, S, Hq + Hk, D)
  mx::array qk = mx::matmul(x, mx::transpose(proj_weight));
  qk = mx::reshape(qk, {B, S, Hq + Hk, D});

  // query = qk[:, :, :Hq]                         [B, S, Hq, D]
  mx::array query =
      mx::slice(qk, {0, 0, 0, 0}, {B, S, Hq, D}, {1, 1, 1, 1});
  // raw_keys = qk[:, :, Hq:].squeeze(2)           [B, S, D]  (Hk == 1)
  mx::array raw_keys = mx::squeeze(
      mx::slice(qk, {0, 0, Hq, 0}, {B, S, Hq + Hk, D}, {1, 1, 1, 1}), 2);

  // query = self.q_layernorm(query).transpose(0, 2, 1, 3)  [B, Hq, S, D]
  query = mx::transpose(qsa_rms_norm(query, q_weight, cfg.eps), {0, 2, 1, 3});

  // key_len = raw_keys.shape[1]  (== S, no cache in this primitive)
  // max_complete_blocks = key_len // compress_ratio
  // complete_key_len = max_complete_blocks * compress_ratio
  const int blocks = S / ratio;
  const int complete_len = blocks * ratio;

  // pooled_keys = raw_keys[:, :complete_key_len].reshape(
  //     B, max_complete_blocks, compress_ratio, head_dim)
  mx::array pooled =
      mx::slice(raw_keys, {0, 0, 0}, {B, complete_len, D}, {1, 1, 1});
  pooled = mx::reshape(pooled, {B, blocks, ratio, D});

  // pooled_keys = mean(pooled.astype(float32), axis=2).astype(raw_keys.dtype)
  pooled = mx::astype(mx::mean(mx::astype(pooled, mx::float32), 2),
                      raw_keys.dtype());

  // pooled_keys = self.k_layernorm(pooled)
  pooled = qsa_rms_norm(pooled, k_weight, cfg.eps);

  // pooled_keys = mx.expand_dims(pooled_keys, axis=1)  [B, 1, blocks, D]
  mx::array pooled_keys = mx::expand_dims(pooled, 1);

  return {std::move(query), std::move(pooled_keys)};
}

}  // namespace veystrix::engine
