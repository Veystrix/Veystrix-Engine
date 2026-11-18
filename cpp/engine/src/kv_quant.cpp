#include "kv_quant.hpp"

#include <algorithm>
#include <stdexcept>

namespace veystrix::engine {

namespace {

// Axis carrying the token dimension in KV tensors [..., T, D] and in the
// packed/scales/biases arrays produced by mx::quantize (same leading axes).
int token_axis(const mx::array& a) { return a.ndim() - 2; }

mx::array slice_tokens(const mx::array& a, int start, int stop) {
  const int nd = a.ndim();
  const int axis = token_axis(a);
  mx::Shape lo(nd, 0), hi(a.shape()), step(nd, 1);
  lo[axis] = start;
  hi[axis] = stop;
  return mx::slice(a, lo, hi, step);
}

mx::array concat_tokens(const mx::array& a, const mx::array& b) {
  return mx::concatenate(std::vector<mx::array>{a, b}, token_axis(a));
}

KVPacked concat_packed(const KVPacked& a, const KVPacked& b) {
  return KVPacked(concat_tokens(a.data, b.data), concat_tokens(a.scales, b.scales),
                  concat_tokens(a.biases, b.biases), a.group_size, a.bits);
}

KVPacked slice_packed(const KVPacked& a, int start, int stop) {
  return KVPacked(slice_tokens(a.data, start, stop),
                  slice_tokens(a.scales, start, stop),
                  slice_tokens(a.biases, start, stop),
                  a.group_size, a.bits);
}

KVPacked drop_packed_rows(const KVPacked& a, int start) {
  return slice_packed(a, start, a.data.shape(token_axis(a.data)));
}

}  // namespace

KVPacked kv_quantize(const mx::array& x, int group_size, int bits,
                     const std::string& mode) {
  // mx::quantize(w, group_size, bits, mode) -> vector{data, scales, biases}.
  // Same numeric op as Python `mx.quantize(x, group_size=gs, bits=bits)` in
  // models/flashnext/quantized_kv.py / mixed_kv.py.
  std::vector<mx::array> parts = mx::quantize(x, group_size, bits, mode);
  if (parts.size() != 3) {
    throw std::runtime_error("kv_quantize: mx::quantize did not return 3 arrays");
  }
  return KVPacked(std::move(parts[0]), std::move(parts[1]), std::move(parts[2]),
                  group_size, bits);
}

mx::array kv_dequantize(const mx::array& data, const mx::array& scales,
                        const mx::array& biases, int group_size, int bits,
                        const std::string& mode) {
  // Same numeric op as Python `mx.dequantize(data, scales, biases,
  // group_size=gs, bits=bits)`. The affine result r = scale*q + bias is
  // computed in float and rounded once to the scales' dtype (bf16 here),
  // exactly as in MLX's dequantize kernel.
  return mx::dequantize(data, scales, biases, group_size, bits, mode);
}

mx::array kv_dequantize(const KVPacked& q) {
  return kv_dequantize(q.data, q.scales, q.biases, q.group_size, q.bits);
}

Tiered4KV::Tiered4KV(int group_size, int recent_tokens)
    : group_size_(group_size), recent_tokens_(recent_tokens) {
  if (group_size_ != 32 && group_size_ != 64) {
    throw std::invalid_argument("Tiered4KV: group_size must be 32 or 64");
  }
  if (recent_tokens_ < kPage || recent_tokens_ % kPage != 0) {
    throw std::invalid_argument("Tiered4KV: recent_tokens must be a positive multiple of 256");
  }
}

void Tiered4KV::append(const mx::array& keys, const mx::array& values) {
  // tiered4: history key_bits=4, value_bits=4; recent bits=8 for both.
  KVPacked hk = kv_quantize(keys, group_size_, /*bits=*/4);
  KVPacked hv = kv_quantize(values, group_size_, /*bits=*/4);
  KVPacked rk = kv_quantize(keys, group_size_, /*bits=*/8);
  KVPacked rv = kv_quantize(values, group_size_, /*bits=*/8);

  if (!history_keys_) {
    history_keys_.emplace(std::move(hk));
    history_values_.emplace(std::move(hv));
    recent_keys_.emplace(std::move(rk));
    recent_values_.emplace(std::move(rv));
  } else {
    history_keys_.emplace(concat_packed(*history_keys_, hk));
    history_values_.emplace(concat_packed(*history_values_, hv));
    recent_keys_.emplace(concat_packed(*recent_keys_, rk));
    recent_values_.emplace(concat_packed(*recent_values_, rv));
  }

  const int tokens = keys.shape(token_axis(keys));
  const int recent_len = (offset_ - history_tokens_) + tokens;
  const int aged = std::max(0, (recent_len - recent_tokens_) / kPage * kPage);
  if (aged > 0) {
    // Expired INT8 rows are dropped; their already-written INT4 rows become
    // visible history. No historical row is requantized.
    recent_keys_.emplace(drop_packed_rows(*recent_keys_, aged));
    recent_values_.emplace(drop_packed_rows(*recent_values_, aged));
    history_tokens_ += aged;
  }
  offset_ += tokens;
}

KVPacked Tiered4KV::history_tier(const KVPacked& full) const {
  // Only rows [0, history_tokens_) are the history tier consumed by attention;
  // the rest of `full` still overlaps with the recent tier.
  return slice_packed(full, /*start=*/0, /*stop=*/history_tokens_);
}

}  // namespace veystrix::engine
