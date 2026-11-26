#include "fc2_skip.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace veystrix::engine {

namespace {

int64_t prod_except_last(const mx::Shape& shape) {
  int64_t n = 1;
  for (size_t i = 0; i + 1 < shape.size(); ++i) n *= shape[i];
  return n;
}

bool is_dtype(const mx::array& a, mx::Dtype d) {
  return static_cast<mx::Dtype::Val>(a.dtype()) ==
         static_cast<mx::Dtype::Val>(d);
}

// Return a copy of `a` (last dim = cols) with `group_elems` consecutive
// elements zeroed at each requested group, for every row. Bytes are zeroed
// to 0x00: 0 for unsigned packed words, +0.0f for f32/f16/bf16 bit patterns.
// Dtype is preserved. This is the entire mechanism behind stage 2's
// bit-exactness: the stock kernel is untouched.
mx::array zero_group_copy(const mx::array& a, int group_elems,
                          const std::vector<int>& groups, int64_t rows) {
  if (a.ndim() < 2)
    throw std::runtime_error("fc2_skip: expected a 2+ dimensional tensor");
  mx::eval(a);
  const mx::Shape shape = a.shape();
  const int64_t cols = shape.back();
  const auto dt = a.dtype();

  auto zero_at = [&](auto& v) {
    for (int g : groups) {
      const int64_t base = static_cast<int64_t>(g) * group_elems;
      if (base < 0 || base + group_elems > cols) {
        throw std::runtime_error("fc2_skip: zero group out of range");
      }
      for (int64_t n = 0; n < rows; ++n) {
        for (int k = 0; k < group_elems; ++k) v[n * cols + base + k] = 0;
      }
    }
  };

  if (is_dtype(a, mx::uint32)) {
    std::vector<uint32_t> v(a.size());
    std::memcpy(v.data(), a.data<uint32_t>(), a.nbytes());
    zero_at(v);
    return mx::array(v.begin(), shape, mx::uint32);
  }
  if (a.itemsize() == 4) {  // float32 scales/biases (mx::quantize output)
    std::vector<float> v(a.size());
    std::memcpy(v.data(), a.data<float>(), a.nbytes());
    zero_at(v);
    return mx::array(v.begin(), shape, mx::float32);
  }
  if (a.itemsize() == 2) {  // bfloat16 / float16 scales/biases (checkpoint)
    std::vector<uint16_t> v(a.size());
    std::memcpy(v.data(), a.data<uint16_t>(), a.nbytes());
    zero_at(v);
    // Construct as uint16 then REINTERPRET: mx::array(iter, shape, bf16)
    // numerically converts uint16 -> bf16 and corrupts the bit patterns.
    mx::array u(v.begin(), shape, mx::uint16);
    return mx::view(u, dt);
  }
  throw std::runtime_error("fc2_skip: unsupported tensor dtype for zero-fill");
}

mx::array slice_kept(const mx::array& a, int group_elems,
                     const std::vector<int>& keep) {
  std::vector<mx::array> parts;
  parts.reserve(keep.size());
  const int axis = static_cast<int>(a.ndim()) - 1;
  for (int g : keep) {
    mx::Shape start(a.ndim(), 0), stop = a.shape(), step(a.ndim(), 1);
    start[axis] = g * group_elems;
    stop[axis] = (g + 1) * group_elems;
    parts.push_back(mx::slice(a, start, stop, step));
  }
  if (parts.empty()) throw std::runtime_error("fc2_skip: no groups kept");
  return mx::concatenate(parts, axis);
}

}  // namespace

// ------------------------------- Stage 1 ------------------------------------

ZeroFraction probe_zero_fraction(const mx::array& relu2_act, int group_size) {
  if (group_size <= 0)
    throw std::runtime_error("fc2_skip: group_size must be positive");
  if (relu2_act.ndim() < 2)
    throw std::runtime_error("fc2_skip: activations must be [rows, intermediate]");
  const int intermediate = static_cast<int>(relu2_act.shape().back());
  if (intermediate % group_size != 0)
    throw std::runtime_error("fc2_skip: intermediate not divisible by group_size");

  // Cast to float32 so exact-zero detection is uniform across bf16/f16/f32.
  mx::array f = mx::astype(relu2_act, mx::float32);
  mx::eval(f);
  const float* p = f.data<float>();
  const int64_t rows = prod_except_last(f.shape());
  const int64_t n_groups = intermediate / group_size;

  ZeroFraction z;
  z.total_activations = rows * intermediate;
  z.total_group_slots = rows * n_groups;
  z.total_groups = n_groups;

  // Per-group "zero in this row" flags; a group is skippable only if zero in
  // every row.
  std::vector<int> group_nonzero_in_row(static_cast<size_t>(n_groups), 0);
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = p + r * intermediate;
    for (int64_t g = 0; g < n_groups; ++g) {
      bool group_zero = true;
      for (int k = 0; k < group_size; ++k) {
        const float v = row[g * group_size + k];
        if (v == 0.0f) {
          ++z.zero_activations;
        } else {
          group_zero = false;
          group_nonzero_in_row[static_cast<size_t>(g)] = 1;
        }
      }
      if (group_zero) ++z.zero_group_slots;
    }
  }
  for (int64_t g = 0; g < n_groups; ++g) {
    if (group_nonzero_in_row[static_cast<size_t>(g)] == 0) {
      ++z.zero_groups;
      z.zero_group_ids.push_back(static_cast<int>(g));
    }
  }
  z.activation_zero_fraction =
      z.total_activations ? double(z.zero_activations) / z.total_activations : 0.0;
  z.group_slot_zero_fraction =
      z.total_group_slots ? double(z.zero_group_slots) / z.total_group_slots : 0.0;
  z.group_zero_fraction =
      z.total_groups ? double(z.zero_groups) / z.total_groups : 0.0;
  return z;
}

std::vector<int> fully_zero_groups(const mx::array& relu2_act, int group_size) {
  return probe_zero_fraction(relu2_act, group_size).zero_group_ids;
}

// ------------------------------- Stage 2 ------------------------------------

Fc2Compacted compact_fc2_kgroups(const mx::array& x, const mx::array& w,
                                 const mx::array& scales,
                                 const mx::array& biases,
                                 const std::vector<int>& zero_groups,
                                 int group_size, int bits) {
  const int n_groups = static_cast<int>(scales.shape().back());
  std::vector<int> keep;
  for (int g = 0; g < n_groups; ++g) {
    if (std::find(zero_groups.begin(), zero_groups.end(), g) == zero_groups.end())
      keep.push_back(g);
  }
  const int w_group_elems = group_size * bits / 32;  // 8 words for G64/b4
  Fc2Compacted out(slice_kept(x, group_size, keep),
                   slice_kept(w, w_group_elems, keep),
                   slice_kept(scales, 1, keep),
                   slice_kept(biases, 1, keep));
  out.kept_groups = keep;
  return out;
}

Fc2ZeroFilled zero_fill_skipped_groups(const mx::array& w,
                                       const mx::array& scales,
                                       const mx::array& biases,
                                       const std::vector<int>& zero_groups,
                                       int hidden, int /*intermediate*/,
                                       int group_size, int bits) {
  const int w_group_elems = group_size * bits / 32;
  return Fc2ZeroFilled(zero_group_copy(w, w_group_elems, zero_groups, hidden),
                       zero_group_copy(scales, 1, zero_groups, hidden),
                       zero_group_copy(biases, 1, zero_groups, hidden));
}

mx::array fc2_quantized_matmul_skip(const mx::array& relu2_act,
                                    const mx::array& w, const mx::array& scales,
                                    const mx::array& biases, int hidden,
                                    int intermediate, int group_size, int bits,
                                    Fc2SkipStats* stats) {
  const ZeroFraction zf = probe_zero_fraction(relu2_act, group_size);
  const Fc2ZeroFilled z = zero_fill_skipped_groups(
      w, scales, biases, zf.zero_group_ids, hidden, intermediate,
      group_size, bits);
  if (stats) {
    stats->total_groups = zf.total_groups;
    stats->skipped_groups = zf.zero_groups;
    stats->kept_groups = zf.total_groups - zf.zero_groups;
  }
  // Unchanged stock kernel, unchanged dense K layout, unchanged lane tree.
  return mx::quantized_matmul(relu2_act, z.w, z.scales, z.biases,
                              /*transpose=*/true, group_size, bits, "affine");
}

mx::array fc2_gather_qmm_skip(const mx::array& relu2_act,
                              const mx::array& w, const mx::array& scales,
                              const mx::array& biases,
                              const mx::array& rhs_indices,
                              int /*hidden*/, int /*intermediate*/,
                              int group_size, int bits, Fc2SkipStats* stats) {
  const ZeroFraction zf = probe_zero_fraction(relu2_act, group_size);
  const int64_t hidden = w.shape(w.ndim() - 2);
  const Fc2ZeroFilled z = zero_fill_skipped_groups(
      w, scales, biases, zf.zero_group_ids,
      static_cast<int>(hidden), 0, group_size, bits);
  if (stats) {
    stats->total_groups = zf.total_groups;
    stats->skipped_groups = zf.zero_groups;
    stats->kept_groups = zf.total_groups - zf.zero_groups;
  }
  return mx::gather_qmm(relu2_act, z.w, z.scales, z.biases,
                        /*lhs_indices=*/std::nullopt, rhs_indices,
                        /*transpose=*/true, group_size, bits, "affine",
                        /*sorted_indices=*/false);
}

// ------------------------------- Stage 3 ------------------------------------

Fc2ChunkLayout fc2_chunk_layout(int hidden, int intermediate, int group_size,
                                int bits, int scale_itemsize) {
  Fc2ChunkLayout L;
  L.hidden = hidden;
  L.intermediate = intermediate;
  L.group_size = group_size;
  L.bits = bits;
  L.n_groups = intermediate / group_size;
  L.words_per_group_weight = static_cast<int64_t>(group_size) * bits / 32;
  L.bytes_weight_per_group = static_cast<int64_t>(hidden) *
                             L.words_per_group_weight * 4;
  L.bytes_scale_per_group = static_cast<int64_t>(hidden) * scale_itemsize;
  L.bytes_bias_per_group = L.bytes_scale_per_group;
  L.chunk_bytes = L.bytes_weight_per_group + L.bytes_scale_per_group +
                  L.bytes_bias_per_group;
  L.total_bytes = L.chunk_bytes * L.n_groups;
  return L;
}

std::vector<uint8_t> repack_fc2_kgroup_major(const mx::array& w,
                                             const mx::array& scales,
                                             const mx::array& biases,
                                             int hidden, int intermediate,
                                             int group_size, int bits) {
  mx::eval(w, scales, biases);
  const Fc2ChunkLayout L =
      fc2_chunk_layout(hidden, intermediate, group_size, bits,
                       static_cast<int>(scales.itemsize()));
  const int n_groups = static_cast<int>(L.n_groups);
  const int words = static_cast<int>(L.words_per_group_weight);
  const size_t sb = scales.itemsize(), bb = biases.itemsize();
  const uint8_t* wp = reinterpret_cast<const uint8_t*>(w.data<uint32_t>());
  const uint8_t* sp = reinterpret_cast<const uint8_t*>(scales.data<uint16_t>());
  const uint8_t* bp = reinterpret_cast<const uint8_t*>(biases.data<uint16_t>());

  std::vector<uint8_t> out;
  out.reserve(static_cast<size_t>(L.total_bytes));
  for (int g = 0; g < n_groups; ++g) {
    for (int n = 0; n < hidden; ++n) {
      // weight: row n, group g -> 8 uint32 words (32 bytes at G64/b4)
      const uint8_t* wrow = wp + (static_cast<size_t>(n) * (intermediate / 8) +
                                  static_cast<size_t>(g) * words) * 4;
      out.insert(out.end(), wrow, wrow + static_cast<size_t>(words) * 4);
      // scale / bias: [n, g] in the source dtype (raw bytes preserved)
      const uint8_t* sc = sp + (static_cast<size_t>(n) * n_groups + g) * sb;
      out.insert(out.end(), sc, sc + sb);
      const uint8_t* bi = bp + (static_cast<size_t>(n) * n_groups + g) * bb;
      out.insert(out.end(), bi, bi + bb);
    }
  }
  if (static_cast<int64_t>(out.size()) != L.total_bytes)
    throw std::runtime_error("fc2_skip: repack size mismatch");
  return out;
}

Fc2Layout fc2_layout_from_env() {
  const char* v = std::getenv("VEYSTRIX_NEMOTRON_FC2_LAYOUT");
  if (v && (std::string(v) == "kgroup" || std::string(v) == "kgroup_major"))
    return Fc2Layout::KGroupMajor;
  return Fc2Layout::Dense;  // original format is the default fallback
}

// ------------------------------- Stage 4 ------------------------------------

Fc2SkipPolicy fc2_skip_policy_from_env() {
  Fc2SkipPolicy p;
  const char* en = std::getenv("VEYSTRIX_NEMOTRON_FC2_SKIP");
  p.enabled = en && std::string(en) == "1";  // default OFF; rollback = 0
  const char* th = std::getenv("VEYSTRIX_NEMOTRON_FC2_MIN_SKIP_FRACTION");
  if (th) p.min_skip_fraction = std::atof(th);
  const char* rt = std::getenv("VEYSTRIX_NEMOTRON_FC2_ROUNDTRIP_BYTES");
  if (rt) p.round_trip_bytes = std::atoll(rt);
  return p;
}

bool should_skip_fc2(const Fc2SkipPolicy& policy, const ZeroFraction& zf,
                     int64_t bytes_saved) {
  // NEEDS OWNER APPROVAL: a true per-layer decision needs the relu2
  // activations on the host, i.e. one GPU->host sync per layer.
  if (!policy.enabled) return false;
  if (zf.group_zero_fraction < policy.min_skip_fraction) return false;
  return bytes_saved > policy.round_trip_bytes;
}

}  // namespace veystrix::engine
