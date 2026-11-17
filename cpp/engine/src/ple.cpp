#include "ple.hpp"

#include "engine.hpp"  // hc_silu

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <unistd.h>

namespace veystrix::engine {

namespace {

mx::array i64_scalar(int64_t v) { return mx::array(v, mx::int64); }
mx::array f32_scalar(float v) { return mx::array(v, mx::float32); }

// Qwen4ExpRMSNorm with group_size: y = x.f32.reshape(...,-1,gs);
// y *= rsqrt(mean(y^2,-1,keepdims)+eps); y *= (1 + weight.f32).
mx::array ple_rms_norm(const mx::array& x, const mx::array& weight,
                       int group_size, float eps, bool one_centered) {
  auto dtype = x.dtype();
  mx::array y = mx::astype(x, mx::float32);
  mx::Shape shape = y.shape();
  mx::Shape grouped(shape);
  grouped.pop_back();
  grouped.push_back(-1);
  grouped.push_back(group_size);
  y = mx::reshape(y, grouped);
  mx::array w = mx::reshape(weight, mx::Shape{-1, group_size});
  mx::array mean = mx::mean(mx::square(y), -1, /*keepdims=*/true);
  y = mx::multiply(y, mx::rsqrt(mx::add(mean, f32_scalar(eps))));
  mx::array gain = mx::astype(w, mx::float32);
  if (!one_centered) gain = mx::add(f32_scalar(1.0f), gain);
  y = mx::multiply(y, gain);
  return mx::astype(mx::reshape(y, shape), dtype);
}

// Qwen4ExpNGramEmbedding._shift_right_ignore_eos, exactly.
mx::array shift_right_ignore_eos(const mx::array& token_ids, int shift,
                                 int64_t eos) {
  if (shift == 0) return token_ids;
  int B = token_ids.shape(0);
  int L = token_ids.shape(1);
  mx::array positions = mx::arange(0, L, 1, mx::int64);
  mx::array eos_i = i64_scalar(eos);
  mx::array minus_one = i64_scalar(-1);
  mx::array zero = i64_scalar(0);
  mx::array shift_i = i64_scalar(shift);

  mx::array eos_positions =
      mx::where(mx::equal(token_ids, eos_i), positions, minus_one);
  mx::array prev_incl = mx::cummax(eos_positions, /*axis=*/1);
  mx::array shifted_prev = mx::slice(prev_incl, mx::Shape{0, 0},
                                     mx::Shape{B, L - 1});
  mx::array previous_eos = mx::concatenate(
      {mx::full(mx::Shape{B, 1}, minus_one, mx::int64), shifted_prev}, 1);
  mx::array segment_start = mx::add(previous_eos, i64_scalar(1));
  mx::array position_in_segment =
      mx::subtract(mx::expand_dims(positions, 0), segment_start);
  mx::array source_positions = mx::subtract(positions, shift_i);
  mx::array gather_positions = mx::broadcast_to(
      mx::expand_dims(mx::maximum(source_positions, zero), 0),
      mx::Shape{B, L});
  mx::array shifted = mx::take_along_axis(token_ids, gather_positions, 1);
  mx::array valid = mx::logical_and(
      mx::greater_equal(position_in_segment, shift_i),
      mx::greater_equal(mx::expand_dims(source_positions, 0), zero));
  return mx::where(valid, shifted, eos_i);
}

// ShardedEmbedding.__call__ with QuantizedEmbedding shards: gather the PACKED
// rows then dequantize (this ordering is required for bit-exactness).
mx::array sharded_embedding(const mx::array& ngram_ids, const PLEWeights& w,
                            int head_dim) {
  mx::array flat = mx::reshape(ngram_ids, mx::Shape{-1});
  mx::array zero = i64_scalar(0);
  std::optional<mx::array> result;
  bool have_result = false;
  size_t nshards = w.emb_shards.size();
  for (size_t i = 0; i < nshards; ++i) {
    int64_t start = w.shard_offsets[i];
    int64_t end = w.shard_offsets[i + 1];
    mx::array start_i = i64_scalar(start);
    mx::array local = mx::clip(mx::subtract(flat, start_i), zero,
                               i64_scalar(end - start - 1));
    const auto& sh = w.emb_shards[i];
    mx::array wg = mx::take(sh.weight, local, /*axis=*/0);
    mx::array sg = mx::take(sh.scales, local, /*axis=*/0);
    mx::array bg = mx::take(sh.biases, local, /*axis=*/0);
    mx::array values = mx::dequantize(wg, sg, bg, w.emb_group_size,
                                      w.emb_bits, w.emb_mode);
    mx::array in_shard = mx::logical_and(
        mx::greater_equal(flat, start_i), mx::less(flat, i64_scalar(end)));
    if (!have_result) {
      result = std::move(values);
      have_result = true;
    } else {
      result = mx::where(mx::expand_dims(in_shard, 1), values, *result);
    }
  }
  mx::Shape ids_shape = ngram_ids.shape();
  mx::Shape emb_shape(ids_shape);
  emb_shape.push_back(head_dim);
  return mx::reshape(*result, emb_shape);
}

// ---- Positioned-read (streamed) embedding backend ------------------------
// Mirrors models/flashnext/ngram.py:
//   StreamingShardedEmbedding._groups / _direct_rows and
//   StreamingQuantizedEmbedding._rows (which calls store.rows -> rows_np).
// Only the rows actually requested are pread; only shards owning at least one
// requested row are touched.  The packed rows are gathered BEFORE dequantize,
// preserving the bit-exact ordering used by the in-memory backend.
mx::array read_streamed_rows(const PLERowTensorRef& ref,
                             const std::vector<int64_t>& rows) {
  size_t n = rows.size();
  size_t bytes = n * (size_t)ref.row_bytes;
  void* buf = std::malloc(bytes ? bytes : 1);
  if (buf == nullptr) throw std::runtime_error("PLE streamed: malloc failed");
  uint8_t* p = static_cast<uint8_t*>(buf);
  for (size_t i = 0; i < n; ++i) {
    off_t off = (off_t)(ref.data_offset + rows[i] * ref.row_bytes);
    ssize_t got = ::pread(ref.fd, p + i * ref.row_bytes,
                          (size_t)ref.row_bytes, off);
    if (got != ref.row_bytes) {
      std::free(buf);
      throw std::runtime_error("PLE streamed: short pread");
    }
  }
  mx::Shape shape{(int)n, (int)ref.elements_per_row};
  auto free_buf = [](void* q) { std::free(q); };
  switch (ref.elem) {
    case PLEElemType::U32:
      return mx::array(buf, shape, mx::uint32, free_buf);
    case PLEElemType::F32:
      return mx::array(buf, shape, mx::float32, free_buf);
    case PLEElemType::F16:
      return mx::array(buf, shape, mx::float16, free_buf);
    case PLEElemType::BF16:
      return mx::array(buf, shape, mx::bfloat16, free_buf);
  }
  std::free(buf);
  throw std::runtime_error("PLE streamed: unknown element type");
}

mx::array sharded_embedding_streamed(const mx::array& ngram_ids,
                                     const PLEWeights& w, int head_dim) {
  mx::array flat = mx::reshape(ngram_ids, mx::Shape{-1});
  mx::eval(flat);
  size_t n = flat.size();
  if (n == 0) return mx::zeros(mx::Shape{0, head_dim}, mx::float32);
  const int64_t* src = flat.data<int64_t>();
  std::vector<int64_t> global_rows(src, src + n);

  int nshards = (int)w.emb_streamed.size();
  int64_t total = w.shard_offsets.back();
  struct Group {
    std::vector<int> positions;
    std::vector<int64_t> rows;
  };
  std::vector<Group> groups(nshards);
  for (size_t pos = 0; pos < n; ++pos) {
    int64_t row = global_rows[pos];
    int si;
    int64_t local;
    if (row < 0 || row >= total) {
      si = 0;
      local = row;
    } else {
      si = (int)(std::upper_bound(w.shard_offsets.begin(),
                                  w.shard_offsets.end(), row)
                 - w.shard_offsets.begin()) -
           1;
      local = row - w.shard_offsets[si];
    }
    int64_t shard_size = w.shard_offsets[si + 1] - w.shard_offsets[si];
    local = std::min(std::max(local, (int64_t)0), shard_size - 1);
    groups[si].positions.push_back((int)pos);
    groups[si].rows.push_back(local);
  }

  std::vector<mx::array> blocks;
  std::vector<int> packed_positions;
  for (int si = 0; si < nshards; ++si) {
    if (groups[si].rows.empty()) continue;
    const auto& g = groups[si];
    const auto& ref = w.emb_streamed[si];
    mx::array wg = read_streamed_rows(ref.weight, g.rows);
    mx::array sg = read_streamed_rows(ref.scales, g.rows);
    mx::array bg = read_streamed_rows(ref.biases, g.rows);
    blocks.push_back(mx::dequantize(wg, sg, bg, w.emb_group_size,
                                    w.emb_bits, w.emb_mode));
    packed_positions.insert(packed_positions.end(), g.positions.begin(),
                            g.positions.end());
  }
  mx::array packed = blocks.size() == 1 ? blocks[0]
                                        : mx::concatenate(blocks, 0);
  std::vector<uint32_t> inverse(n, 0);
  for (size_t k = 0; k < packed_positions.size(); ++k)
    inverse[packed_positions[k]] = (uint32_t)k;
  mx::array idx(inverse.data(), mx::Shape{(int)n}, mx::uint32);
  mx::array values = mx::take(packed, idx, /*axis=*/0);
  mx::Shape emb_shape = ngram_ids.shape();
  emb_shape.push_back(head_dim);
  return mx::reshape(values, emb_shape);
}

// Qwen4ExpNGramEmbedding.__call__; writes the updated trailing context back to
// cache.context.
mx::array ple_ngram_embedding(const mx::array& input_ids,
                              const PLEWeights& w, PLECache& cache) {
  mx::array ids = mx::astype(input_ids, mx::int64);
  int B = ids.shape(0);
  int S = ids.shape(1);
  int ctx_len = w.ngram_size - 1;
  mx::array eos = i64_scalar(w.eos_token_id);
  mx::array previous_context =
      cache.context.has_value()
          ? *cache.context
          : mx::full(mx::Shape{B, ctx_len}, eos, mx::int64);
  mx::array token_history = mx::concatenate({previous_context, ids}, 1);
  int L = token_history.shape(1);
  cache.context = mx::contiguous(
      mx::slice(token_history, mx::Shape{0, L - ctx_len},
                mx::Shape{B, L}));

  int ngram_heads = (w.ngram_size - 1) * w.heads_per_ngram;
  std::vector<mx::array> shifted;
  shifted.reserve(w.ngram_size);
  for (int s = 0; s < w.ngram_size; ++s)
    shifted.push_back(shift_right_ignore_eos(token_history, s,
                                             w.eos_token_id));

  std::vector<mx::array> blocks;
  blocks.reserve(w.ngram_size - 1);
  for (int ngram = 2; ngram <= w.ngram_size; ++ngram) {
    int start = (ngram - 2) * w.heads_per_ngram;
    int end = start + w.heads_per_ngram;
    mx::array mult = mx::reshape(
        mx::slice(w.layer_multipliers, mx::Shape{0}, mx::Shape{ngram}),
        mx::Shape{ngram});
    mx::array mixed =
        mx::multiply(shifted[0], mx::expand_dims(mx::slice(mult, mx::Shape{0},
                                                          mx::Shape{1}), 0));
    for (int pos = 1; pos < ngram; ++pos) {
      mx::array m = mx::expand_dims(
          mx::slice(mult, mx::Shape{pos}, mx::Shape{pos + 1}), 0);
      mixed = mx::bitwise_xor(mixed,
                              mx::multiply(shifted[pos], m));
    }
    mx::array sizes = mx::reshape(
        mx::slice(w.head_vocab_sizes, mx::Shape{start}, mx::Shape{end}),
        mx::Shape{1, 1, w.heads_per_ngram});
    mx::array offsets = mx::reshape(
        mx::slice(w.head_offsets, mx::Shape{start}, mx::Shape{end}),
        mx::Shape{1, 1, w.heads_per_ngram});
    mx::array ngram_ids = mx::add(
        mx::remainder(mx::expand_dims(mixed, -1), sizes), offsets);
    blocks.push_back(ngram_ids);
  }
  mx::array ngram_ids_all = mx::concatenate(blocks, /*axis=*/-1);
  mx::array ngram_ids =
      mx::slice(ngram_ids_all, mx::Shape{0, L - S, 0},
                mx::Shape{B, L, ngram_heads});

  // ngram_embedding.dims = ple_embed_dim / ngram_heads.  Prefer the explicit
  // config value (required by the streamed backend); otherwise recover it from
  // stored scales[rows, dims/group] on the in-memory backend.
  int dims = 0;
  if (w.ple_embed_dim > 0)
    dims = w.ple_embed_dim / ngram_heads;
  else if (!w.emb_shards.empty())
    dims = (int)(w.emb_shards[0].scales.shape(1)) * w.emb_group_size;
  if (dims <= 0)
    throw std::runtime_error(
        "PLE: cannot determine head dim; set PLEWeights.ple_embed_dim");
  mx::array embeddings = w.emb_streamed.empty()
                             ? sharded_embedding(ngram_ids, w, dims)
                             : sharded_embedding_streamed(ngram_ids, w, dims);
  return mx::reshape(embeddings, mx::Shape{B, S, dims * ngram_heads});
}

// Qwen4ExpPLELayer._short_conv.
mx::array short_conv(const mx::array& x, const PLEWeights& w, PLECache& cache) {
  int B = x.shape(0);
  int C = x.shape(2);
  // Python _short_conv: state = mx.zeros(..., dtype=x.dtype) (bf16 on real ckpt).
  mx::array state =
      cache.conv_state.has_value()
          ? *cache.conv_state
          : mx::zeros(mx::Shape{B, w.short_conv_state_len, C}, x.dtype());
  mx::array conv_input = mx::concatenate({state, x}, 1);
  int total = conv_input.shape(1);
  cache.conv_state = mx::contiguous(
      mx::slice(conv_input, mx::Shape{0, total - w.short_conv_state_len, 0},
                mx::Shape{B, total, C}));
  mx::array y = mx::conv1d(conv_input, w.conv_w, /*stride=*/1,
                           /*padding=*/0, w.conv_dilation, /*groups=*/C);
  // Python uses nn.silu (compiled/fused) after the PLE conv.
  return hc_silu(y);  // nn.silu
}

}  // namespace

mx::array ple_forward(const mx::array& hidden_states,
                      const mx::array& input_ids,
                      const PLEWeights& w,
                      PLECache& cache,
                      const mx::array* mask) {
  mx::array embeddings = ple_ngram_embedding(input_ids, w, cache);

  mx::array key_lin = linear(embeddings, w.key_proj);
  mx::array keys = ple_rms_norm(key_lin, w.norm_key, w.hidden_size, w.eps,
                                w.norm_one_centered);
  mx::array values = linear(embeddings, w.value_proj);
  mx::array queries = ple_rms_norm(hidden_states, w.norm_query, w.hidden_size,
                                   w.eps, w.norm_one_centered);

  int B = hidden_states.shape(0);
  int S = hidden_states.shape(1);
  int H = w.hidden_size;
  int hc = w.hc_count;
  keys = mx::reshape(keys, mx::Shape{B, S, hc, H});
  queries = mx::reshape(queries, mx::Shape{B, S, hc, H});

  mx::array gate_sum =
      mx::sum(mx::multiply(keys, queries), -1, /*keepdims=*/true);
  // Python: `sum / math.sqrt(H)` and `max(abs(gate), 1e-6)` use Python floats,
  // which MLX keeps in the array's dtype (bf16 on the real checkpoint). C++
  // f32 scalars would upcast, so build scalars with gate_sum's dtype.
  mx::array sqrt_h = mx::astype(mx::array((float)std::sqrt((double)H)), gate_sum.dtype());
  mx::array eps6 = mx::astype(mx::array(1e-6f), gate_sum.dtype());
  mx::array gate = mx::divide(gate_sum, sqrt_h);
  gate = mx::multiply(mx::sign(gate), mx::sqrt(mx::maximum(mx::abs(gate), eps6)));

  mx::array gated =
      mx::reshape(mx::multiply(mx::sigmoid(gate), mx::expand_dims(values, 2)),
                  mx::Shape{B, S, hc * H});
  mx::array normed = ple_rms_norm(gated, w.norm_conv, H, w.eps,
                                  w.norm_one_centered);

  if (mask != nullptr && mask->ndim() == 2) {
    mx::array m3 = mx::expand_dims(*mask, -1);
    gated = mx::where(m3, gated, f32_scalar(0.0f));
    normed = mx::where(m3, normed, f32_scalar(0.0f));
  }
  return mx::add(gated, short_conv(normed, w, cache));
}

// ---------------------------------------------------------------------------
// Compute-path fallbacks (mirror the Python helpers bit-for-bit).
// ---------------------------------------------------------------------------
uint64_t ple_splitmix64(uint64_t value) {
  constexpr uint64_t MASK64 = std::numeric_limits<uint64_t>::max();
  constexpr uint64_t GAMMA = 0x9E3779B97F4A7C15ULL;
  constexpr uint64_t M1 = 0xBF58476D1CE4E5B9ULL;
  constexpr uint64_t M2 = 0x94D049BB133111EBULL;
  value = (value + GAMMA) & MASK64;
  value = (uint64_t)(((__uint128_t)(value ^ (value >> 30)) * M1) & MASK64);
  value = (uint64_t)(((__uint128_t)(value ^ (value >> 27)) * M2) & MASK64);
  return (value ^ (value >> 31)) & MASK64;
}

std::vector<int64_t> ple_build_layer_multipliers(
    int unigram_vocab_size, int ngram_size, int ple_layer_index, int64_t seed) {
  constexpr uint64_t MASK64 = std::numeric_limits<uint64_t>::max();
  constexpr uint64_t GAMMA = 0x9E3779B97F4A7C15ULL;
  constexpr int64_t PRIME_1 = 10007;
  int64_t max_long = std::numeric_limits<int64_t>::max();
  int64_t multiplier_max = max_long / std::max(unigram_vocab_size, 1);
  int64_t half_bound = std::max<int64_t>(1, multiplier_max / 2);
  uint64_t base_seed = (uint64_t)(seed + PRIME_1 * (int64_t)ple_layer_index);
  std::vector<int64_t> out;
  out.reserve(ngram_size);
  for (int index = 0; index < ngram_size; ++index) {
    uint64_t value =
        (base_seed + (uint64_t)((__uint128_t)GAMMA * (uint64_t)(index + 1))) &
        MASK64;
    uint64_t h = ple_splitmix64(value);
    out.push_back((int64_t)(2 * (h % (uint64_t)half_bound) + 1));
  }
  return out;
}

bool ple_is_prime(int64_t value) {
  if (value < 2) return false;
  if (value % 2 == 0) return value == 2;
  for (int64_t d = 3; d <= (int64_t)std::sqrt((double)value); d += 2)
    if (value % d == 0) return false;
  return true;
}

int64_t ple_find_nth_prime_after(int64_t start, int count) {
  int64_t prime = start;
  for (int i = 0; i < count; ++i) {
    ++prime;
    while (!ple_is_prime(prime)) ++prime;
  }
  return prime;
}

std::vector<int64_t> ple_head_vocab_sizes(int ngram_vocab_size_base,
                                          int ngram_heads,
                                          int ple_layer_index) {
  std::vector<int64_t> out;
  out.reserve(ngram_heads);
  for (int h = 0; h < ngram_heads; ++h) {
    int global_head = ple_layer_index * ngram_heads + h;
    out.push_back(ple_find_nth_prime_after(
        (int64_t)ngram_vocab_size_base - 1, global_head + 1));
  }
  return out;
}

}  // namespace veystrix::engine
