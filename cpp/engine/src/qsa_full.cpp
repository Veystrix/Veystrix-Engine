#include "qsa_full.hpp"

#include <cmath>
#include <limits>

namespace veystrix::engine {

// ---------------------------------------------------------------------------
// QSAKVCache indexer state
// ---------------------------------------------------------------------------

void QSACache::update_indexer(const mx::array& keys,
                              const mx::array& position_ids) {
  // Mirrors QSAKVCache.update_indexer (language.py 54-63).
  if (!index_keys.has_value()) {
    index_keys = keys;
    index_position_ids = position_ids;
    return;
  }
  index_keys = mx::concatenate({*index_keys, keys}, 1);

  // _append_indexer_positions (language.py 21-38).
  mx::array cached = *index_position_ids;
  mx::array cur = position_ids;
  if (cached.ndim() == 3 && cur.ndim() == 2) {
    // position_ids = broadcast_to(position_ids[None],
    //                              (cached.shape[0], *position_ids.shape))
    cur = mx::broadcast_to(mx::expand_dims(cur, 0),
                           {cached.shape()[0], cur.shape()[0], cur.shape()[1]});
  } else if (cached.ndim() == 2 && cur.ndim() == 3) {
    // cached = broadcast_to(cached[None], (position_ids.shape[0], *cached.shape))
    cached = mx::broadcast_to(
        mx::expand_dims(cached, 0),
        {cur.shape()[0], cached.shape()[0], cached.shape()[1]});
  } else if (cached.ndim() != cur.ndim()) {
    throw std::invalid_argument(
        "QSA position IDs must be 2-D text positions or 3-D MRoPE positions");
  }
  index_position_ids = mx::concatenate({cached, cur}, -1);
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

mx::array qsa_compute_inv_freq(const QSARotaryParams& rp) {
  // compute_inv_freq (rope_utils.py 1043-1044):
  //   1.0 / (base ** (arange(0, dim, 2).astype(float32) / dim))
  mx::array ar = mx::astype(mx::arange(0, rp.dim, 2), mx::float32);
  mx::array ex = ar / static_cast<float>(rp.dim);
  mx::array base_arr(static_cast<float>(rp.base), mx::float32);
  mx::array pw = mx::power(base_arr, ex);
  mx::array one(1.0f, mx::float32);
  return one / pw;
}

mx::array qsa_apply_rope(const mx::array& x, const mx::array& position_ids,
                         const QSARotaryParams& rp) {
  const int dim = rp.dim;
  const int half = dim / 2;
  const auto dtype = x.dtype();

  mx::array inv_freq = qsa_compute_inv_freq(rp);  // [half] float32

  // compute_mrope_frequencies (rope_utils.py 1130-1155).
  mx::array freqs = [&]() -> mx::array {
  if (position_ids.ndim() == 2) {
    // Text-only: same scalar position on every MRoPE axis, so the section
    // layout collapses to `pos[..., None] * inv_freq`. [B, T, half]
    mx::array f = mx::expand_dims(mx::astype(position_ids, mx::float32), -1);
    return f * inv_freq;
  } else {
    // 3-D MRoPE [axis, B, T]: interleave/chunk selector picks the axis that
    // feeds each frequency (mrope_position_selector / _selected_mrope_freqs).
    std::vector<int> selector(inv_freq.shape()[0], 0);
    if (rp.style == "interleaved") {
      // _interleaved_position_selector (rope_utils.py 512-517).
      for (int axis = 1; axis <= 2; ++axis) {
        int offset = axis;  // (1, 2) for axes 1 and 2
        for (int idx = offset;
             idx < std::min(rp.mrope_section[axis] * 3,
                            static_cast<int>(inv_freq.shape()[0]));
             idx += 3) {
          selector[idx] = axis;
        }
      }
    } else {
      // _chunked_position_selector (rope_utils.py 520-527).
      int offset = rp.mrope_section[0];
      for (int axis = 1; axis < static_cast<int>(rp.mrope_section.size());
           ++axis) {
        int length = rp.mrope_section[axis];
        for (int idx = offset; idx < std::min(offset + length,
                                              static_cast<int>(inv_freq.shape()[0]));
             ++idx) {
          selector[idx] = axis;
        }
        offset += length;
      }
    }
    mx::array sel(selector.data(), {static_cast<int>(selector.size())}, mx::int32);
    mx::array positions = mx::take(position_ids, sel, 0);       // [half, B, T]
    positions = mx::transpose(positions, {1, 2, 0});            // [B, T, half]
    return mx::astype(positions, mx::float32) * inv_freq;
  }
  }();

  // MRoPERotaryEmbedding.__call__ (rope_utils.py 1228-1242):
  //   emb = concatenate([freqs, freqs], axis=-1)
  //   cos = cos(emb) * scaling ; sin = sin(emb) * scaling
  //   cast cos/sin to x.dtype (cast_output=True)
  mx::array emb = mx::concatenate({freqs, freqs}, -1);         // [B, T, dim]
  mx::array cos = mx::cos(emb) * rp.attention_scaling;
  mx::array sin = mx::sin(emb) * rp.attention_scaling;
  cos = mx::astype(cos, dtype);
  sin = mx::astype(sin, dtype);
  cos = mx::expand_dims(cos, 1);  // [B, 1, T, dim]
  sin = mx::expand_dims(sin, 1);

  // _apply_interleaved_rotary_pos_emb_axis1 (rope_utils.py 1338-1357),
  // which is what apply_multimodal_rotary_pos_emb takes for
  // style="interleaved", unsqueeze_dim=1, cast_output=True.
  mx::array x_rot = mx::slice(x, {0, 0, 0, 0},
                              {x.shape()[0], x.shape()[1], x.shape()[2], dim},
                              {1, 1, 1, 1});
  mx::array x_pass = mx::slice(x, {0, 0, 0, dim},
                               {x.shape()[0], x.shape()[1], x.shape()[2],
                                x.shape()[3]},
                               {1, 1, 1, 1});
  // rotate_half: concat([-x2, x1]) where x1 = first half, x2 = second half.
  mx::array x1 = mx::slice(x_rot, {0, 0, 0, 0},
                           {x.shape()[0], x.shape()[1], x.shape()[2], half},
                           {1, 1, 1, 1});
  mx::array x2 = mx::slice(x_rot, {0, 0, 0, half},
                           {x.shape()[0], x.shape()[1], x.shape()[2], dim},
                           {1, 1, 1, 1});
  mx::array rot = mx::concatenate({-x2, x1}, -1);
  mx::array embed = mx::astype(x_rot * cos + rot * sin, dtype);
  return mx::concatenate({embed, x_pass}, -1);
}

// ---------------------------------------------------------------------------
// Full indexer: scoring + selection
// ---------------------------------------------------------------------------

std::optional<mx::array> qsa_full_indexer(
    const mx::array& x,
    const mx::array& proj_weight,
    const mx::array& q_weight,
    const mx::array& k_weight,
    const QSAFullConfig& cfg,
    QSACache& cache,
    const mx::array& position_ids) {
  const int B = static_cast<int>(x.shape()[0]);
  const int S = static_cast<int>(x.shape()[1]);
  const int Hq = cfg.base.n_heads;
  const int Hk = cfg.base.kv_heads;
  const int D = cfg.base.head_dim;
  const int ratio = cfg.base.compress_ratio;
  const int topk = cfg.base.block_topk;
  const int past_len = cache.offset;

  // --- projection + normalization (same op order as qsa_indexer_pre_rope) ---
  // qk = index_qk_proj(x).reshape(B, S, Hq + Hk, D)
  mx::array qk = mx::matmul(x, mx::transpose(proj_weight));
  qk = mx::reshape(qk, {B, S, Hq + Hk, D});
  mx::array query = mx::slice(qk, {0, 0, 0, 0}, {B, S, Hq, D}, {1, 1, 1, 1});
  mx::array raw_keys = mx::squeeze(
      mx::slice(qk, {0, 0, Hq, 0}, {B, S, Hq + Hk, D}, {1, 1, 1, 1}), 2);
  // query = q_layernorm(query).transpose(0, 2, 1, 3)  [B, Hq, S, D]
  query = mx::transpose(qsa_rms_norm(query, q_weight, cfg.base.eps),
                        {0, 2, 1, 3});

  // --- cache.update_indexer runs BEFORE the None early return ---
  cache.update_indexer(raw_keys, position_ids);
  const mx::array& full_keys = *cache.index_keys;
  const mx::array& full_pos = *cache.index_position_ids;
  const int key_len = static_cast<int>(full_keys.shape()[1]);

  const int max_complete_blocks = key_len / ratio;
  if (max_complete_blocks <= topk) {
    return std::nullopt;  // production returns None
  }

  // query = self._apply_rope(query, position_ids)    [B, Hq, S, D]
  query = qsa_apply_rope(query, position_ids, cfg.rope);

  // complete_key_len / mean-pooled keys / k_layernorm / expand_dims
  const int complete_len = max_complete_blocks * ratio;
  mx::array pooled = mx::slice(full_keys, {0, 0, 0},
                               {B, complete_len, D}, {1, 1, 1});
  pooled = mx::reshape(pooled, {B, max_complete_blocks, ratio, D});
  pooled = mx::astype(mx::mean(mx::astype(pooled, mx::float32), 2),
                      full_keys.dtype());
  pooled = qsa_rms_norm(pooled, k_weight, cfg.base.eps);     // [B, blocks, D]
  pooled = mx::expand_dims(pooled, 1);                        // [B, 1, blocks, D]

  // block_starts = arange(blocks) * ratio
  // block_position_ids = full_position_ids[..., block_starts]
  mx::array block_starts = mx::arange(max_complete_blocks) * ratio;
  mx::array block_pos = mx::take(full_pos, block_starts, -1);  // [B, blocks]
  // pooled_keys = self._apply_rope(pooled_keys, block_position_ids)
  pooled = qsa_apply_rope(pooled, block_pos, cfg.rope);        // [B, 1, blocks, D]

  // scores = query @ pooled_keys.transpose(0, 1, 3, 2)        [B, Hq, S, blocks]
  mx::array scores = mx::matmul(query, mx::transpose(pooled, {0, 1, 3, 2}));
  // scores = sum(maximum(scores.astype(float32), 0), axis=1)  [B, S, blocks]
  mx::array zero(0.0f, mx::float32);
  scores = mx::sum(mx::maximum(mx::astype(scores, mx::float32), zero), 1);
  // scores = scores / math.sqrt(head_dim)  (float32 division by float32 scalar)
  double scale_d = std::sqrt(static_cast<double>(D));
  mx::array scale(static_cast<float>(scale_d), mx::float32);
  scores = scores / scale;

  // query_ends = past_len + arange(seq_len) + 1               [S]
  mx::array query_ends = mx::arange(past_len + 1, past_len + S + 1);
  // complete_counts = query_ends // compress_ratio             [S]
  mx::array complete_counts = query_ends / ratio;
  // valid_blocks = arange(blocks)[None,None,:] < complete_counts[None,:,None]
  mx::array valid = mx::less(
      mx::expand_dims(mx::expand_dims(mx::arange(max_complete_blocks), 0), 0),
      mx::expand_dims(complete_counts, {0, 2}));
  // scores = where(valid_blocks, scores, -inf)
  mx::array neg_inf(-std::numeric_limits<float>::infinity(), mx::float32);
  scores = mx::where(valid, scores, neg_inf);

  // selected_blocks = argpartition(scores, kth=-topk, axis=-1)[..., -topk:]
  mx::array sel_idx = mx::argpartition(scores, -topk, -1);
  mx::array selected_blocks = mx::slice(
      sel_idx, {0, 0, max_complete_blocks - topk},
      {B, S, max_complete_blocks}, {1, 1, 1});

  // token_indices = arange(key_len); token_blocks = token_indices // ratio
  mx::array token_indices = mx::arange(key_len);
  mx::array token_blocks = token_indices / ratio;

  // selected_tokens = any(token_blocks[None,None,None,:] ==
  //                        selected_blocks[..., None], axis=2)
  mx::array tb = mx::expand_dims(token_blocks, {0, 1, 2});   // [1,1,1,key_len]
  mx::array sb = mx::expand_dims(selected_blocks, -1);        // [B,S,topk,1]
  mx::array selected_tokens = mx::any(mx::equal(tb, sb), 2);  // [B,S,key_len]

  // tail_starts = complete_counts * ratio
  mx::array tail_starts = complete_counts * ratio;            // [S]
  mx::array ti = mx::expand_dims(token_indices, {0, 1});      // [1,1,key_len]
  mx::array ts = mx::expand_dims(tail_starts, {0, 2});        // [1,S,1]
  mx::array qe = mx::expand_dims(query_ends, {0, 2});         // [1,S,1]
  mx::array tail = mx::logical_and(mx::greater_equal(ti, ts),
                                   mx::less(ti, qe));          // [1,S,key_len]
  mx::array causal = mx::less(ti, qe);                         // [1,S,key_len]
  mx::array topk_arr(topk, complete_counts.dtype());
  mx::array use_sparse = mx::expand_dims(
      mx::greater(complete_counts, topk_arr), {0, 2});          // [1,S,1]
  selected_tokens = mx::where(use_sparse,
                              mx::logical_or(selected_tokens, tail), causal);

  // return selected_tokens[:, None]
  return mx::expand_dims(selected_tokens, 1);                  // [B,1,S,key_len]
}

}  // namespace veystrix::engine
