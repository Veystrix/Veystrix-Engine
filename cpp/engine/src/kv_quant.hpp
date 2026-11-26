// KV-cache tier quantization PRIMITIVES (foundation stage).
//
// C++ mirror of the affine INT4/INT8 KV representation used by
//   models/flashnext/quantized_kv.py  (AffineQuantizedQSAKVCache: mx.quantize /
//                                      mx.dequantize, group_size 64, bits 4/8)
//   models/flashnext/mixed_kv.py      (MixedPrecisionQSAKVCache mode="tiered4":
//                                      INT4 history + INT8 recent window 8192)
//
// These are thin, typed wrappers over the SAME numeric MLX ops the Python path
// calls (mx::quantize / mx::dequantize on mlx 0.32.2). Bit-exactness against
// Python golden vectors is enforced by tests/test_kv_quant.cpp.
//
// Full prefill attention integration (gather + SDPA over packed rows) is NOT
// part of this stage -- see docs/engineering/kv_quant_port.md.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace veystrix::engine {

// Packed affine representation returned by mx::quantize (Python tuple order:
// (data, scales, biases)). For a bf16 input, scales/biases are bf16; `data` is
// uint32 with 32/bits elements packed per word. Shapes follow MLX:
//   data   [..., T, D/(32/bits)]
//   scales [..., T, D/group_size]
//   biases [..., T, D/group_size]
struct KVPacked {
  mx::array data;    // uint32 packed
  mx::array scales;  // bf16 for bf16 input
  mx::array biases;  // bf16 for bf16 input
  int group_size;
  int bits;

  KVPacked() = delete;
  KVPacked(mx::array d, mx::array s, mx::array b, int gs, int b_)
      : data(std::move(d)), scales(std::move(s)), biases(std::move(b)),
        group_size(gs), bits(b_) {}
};

// Quantize a bf16 K/V tensor (or any MLX array) with the affine scheme used by
// the Python caches. Mirrors `mx.quantize(x, group_size=gs, bits=bits)`.
KVPacked kv_quantize(const mx::array& x, int group_size, int bits,
                     const std::string& mode = "affine");

// Dequantize a packed triplet. Mirrors
// `mx.dequantize(data, scales, biases, group_size=gs, bits=bits)`.
mx::array kv_dequantize(const mx::array& data, const mx::array& scales,
                        const mx::array& biases, int group_size, int bits,
                        const std::string& mode = "affine");
mx::array kv_dequantize(const KVPacked& q);

// tiered4 container: INT4 history + INT8 recent window. Mirrors
// models/flashnext/mixed_kv.py::MixedPrecisionQSAKVCache(mode="tiered4",
// group_size, recent_tokens) with key_bits=4, value_bits=4, recent_bits=8,
// PAGE=256. The numerics (per-token quantize, per-group scales, window aging)
// match the Python reference; capacity growth (HISTORY_GROWTH=4096) is a
// pure allocation detail and is not reproduced -- arrays are stored at their
// logical token length.
class Tiered4KV {
 public:
  static constexpr int kPage = 256;
  static constexpr int kHistoryGrowth = 4096;  // documented, capacity only

  Tiered4KV(int group_size = 64, int recent_tokens = 8192);

  // Feed a chunk of keys/values shaped [..., T, D] (bf16). Both tiers are
  // written from the ORIGINAL activations (no INT8 -> INT4 cascade), matching
  // MixedPrecisionQSAKVCache.update_and_fetch.
  void append(const mx::array& keys, const mx::array& values);

  // History tier: rows [0, history_tokens) in INT4. Sliced views of the
  // stored packed arrays (attention only ever reads these rows).
  KVPacked history_keys() const { return history_tier(*history_keys_); }
  KVPacked history_values() const { return history_tier(*history_values_); }

  // Recent tier: rows [history_tokens, offset) in INT8.
  const KVPacked& recent_keys() const { return *recent_keys_; }
  const KVPacked& recent_values() const { return *recent_values_; }

  // Reference dequantization of each tier (verification path).
  mx::array dequantize_history_keys() const { return kv_dequantize(history_keys()); }
  mx::array dequantize_history_values() const { return kv_dequantize(history_values()); }
  mx::array dequantize_recent_keys() const { return kv_dequantize(*recent_keys_); }
  mx::array dequantize_recent_values() const { return kv_dequantize(*recent_values_); }

  int group_size() const { return group_size_; }
  int recent_tokens() const { return recent_tokens_; }
  int offset() const { return offset_; }
  int history_tokens() const { return history_tokens_; }

 private:
  KVPacked history_tier(const KVPacked& full) const;

  int group_size_;
  int recent_tokens_;
  int offset_ = 0;
  int history_tokens_ = 0;
  // History packed arrays hold every written token [0, offset_); the exposed
  // "history tier" is the slice [0, history_tokens_).
  std::optional<KVPacked> history_keys_;
  std::optional<KVPacked> history_values_;
  std::optional<KVPacked> recent_keys_;
  std::optional<KVPacked> recent_values_;
};

}  // namespace veystrix::engine
