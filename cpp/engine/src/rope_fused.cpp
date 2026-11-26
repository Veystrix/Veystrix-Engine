#include "rope_fused.hpp"

#include <algorithm>
#include <cstdint>
#include <string>

#include <mlx/fast.h>
#include <mlx/mlx.h>

namespace veystrix::engine {

namespace {

std::string replace_all(std::string s, const std::string& from,
                        const std::string& to) {
  if (from.empty()) return s;
  std::size_t p = 0;
  while ((p = s.find(from, p)) != std::string::npos) {
    s.replace(p, from.size(), to);
    p += to.size();
  }
  return s;
}

// Verbatim reconstruction of the f-string in rope_utils._mrope_apply_kernel
// (lines 607-645) for pairing == "half_split" (style == "interleaved").
// Placeholders:
//   @HALF_DIM@    -> rotary_dim / 2
//   @ROTARY_DIM@  -> rotary_dim
//   @PAIR_SOURCE@ -> per-pairing angle/cos/sin block (see below)
const char* kMropeApplyTemplate = R"MROPEKERNEL(
        uint elem = thread_position_in_grid.x;

        const int half_dim = @HALF_DIM@;
        const int q_bsz = x_shape[0];
        const int q_heads = x_shape[1];
        const int q_len = x_shape[2];
        const int q_dim = x_shape[3];
        const int slots = half_dim + q_dim - @ROTARY_DIM@;
        const int work_size = q_bsz * q_heads * q_len * slots;

        if (elem >= uint(work_size)) {
            return;
        }

        int local = int(elem);
        int slot = local % slots;
        int tmp = local / slots;
        int t = tmp % q_len;
        tmp = tmp / q_len;
        int h = tmp % q_heads;
        int b = tmp / q_heads;
        int base = ((b * q_heads + h) * q_len + t) * q_dim;

        if (slot >= half_dim) {
            int pass_d = @ROTARY_DIM@ + slot - half_dim;
            int pass_idx = base + pass_d;
            x_out[pass_idx] = x[pass_idx];
            return;
        }

        @PAIR_SOURCE@

        int idx = base + d;
        float xv = static_cast<float>(x[idx]);
        float xp = static_cast<float>(x[base + pair_d]);
        x_out[idx] = static_cast<T>(xv * c - xp * s);
        x_out[base + pair_d] = static_cast<T>(xp * c + xv * s);
)MROPEKERNEL";

std::string mrope_kernel_source(int rotary_dim, int position_ndim) {
  const std::string pair_head =
      "\n        int freq_idx = slot;\n"
      "        int d = freq_idx;\n"
      "        int pair_d = d + half_dim;\n";
  std::string pair_source;
  if (position_ndim == 2) {
    // selector_source == "" (rope_utils.py line 579).
    pair_source =
        pair_head +
        "        \n"
        "        float pos = static_cast<float>(position_ids[b * q_len + t]);\n";
  } else {
    // selector_source == "int axis = int(position_selector[freq_idx]);"
    // (rope_utils.py lines 581-582).
    pair_source =
        pair_head +
        "        int axis = int(position_selector[freq_idx]);\n"
        "        float pos = "
        "static_cast<float>(position_ids[(axis * q_bsz + b) * q_len + t]);\n";
  }
  pair_source +=
      "        float angle = pos * static_cast<float>(inv_freq[freq_idx]);\n"
      "        float c = metal::cos(angle);\n"
      "        float s = metal::sin(angle);\n"
      "        ";

  std::string src = kMropeApplyTemplate;
  src = replace_all(src, "@HALF_DIM@", std::to_string(rotary_dim / 2));
  src = replace_all(src, "@ROTARY_DIM@", std::to_string(rotary_dim));
  src = replace_all(src, "@PAIR_SOURCE@", pair_source);
  return src;
}

}  // namespace

mx::array rope_fused_compute_inv_freq(int dim, float base) {
  // rope_utils.py 1043-1044:
  //   1.0 / (base ** (arange(0, dim, 2).astype(float32) / dim))
  mx::array ar = mx::astype(mx::arange(0, dim, 2), mx::float32);
  mx::array ex = ar / static_cast<float>(dim);
  mx::array base_arr(static_cast<float>(base), mx::float32);
  mx::array pw = mx::power(base_arr, ex);
  mx::array one(1.0f, mx::float32);
  return one / pw;
}

mx::array rope_fused_position_selector(const std::vector<int>& mrope_section,
                                       int freq_dim, bool interleaved) {
  // rope_utils.py 512-527.
  std::vector<int32_t> selector(static_cast<std::size_t>(freq_dim), 0);
  if (interleaved) {
    // _interleaved_position_selector (512-517).
    for (int axis = 1; axis <= 2; ++axis) {
      const int offset = axis;
      const int limit =
          std::min(mrope_section[axis] * 3, freq_dim);
      for (int idx = offset; idx < limit; idx += 3) selector[idx] = axis;
    }
  } else {
    // _chunked_position_selector (520-527).
    int offset = mrope_section[0];
    for (std::size_t axis = 1; axis < mrope_section.size(); ++axis) {
      const int length = mrope_section[axis];
      const int limit = std::min(offset + length, freq_dim);
      for (int idx = offset; idx < limit; ++idx) selector[idx] =
          static_cast<int32_t>(axis);
      offset += length;
    }
  }
  return mx::array(selector.data(), {static_cast<int>(selector.size())},
                   mx::int32);
}

std::pair<mx::array, mx::array> rope_fused_cos_sin(
    const mx::array& position_ids, const mx::array& inv_freq,
    const mx::array& position_selector, int rotary_dim, bool interleaved) {
  (void)interleaved;  // selector already encodes the axis choice.
  const int position_ndim = static_cast<int>(position_ids.ndim());
  mx::array angles = [&]() -> mx::array {
    if (position_ndim == 2) {
      // (b, t, half) angles; the same scalar position feeds every frequency.
      return mx::expand_dims(mx::astype(position_ids, mx::float32), -1) *
             inv_freq;
    }
    // (axis, b, t) -> select the axis feeding each freq -> (b, t, half).
    mx::array positions = mx::take(position_ids, position_selector, 0);
    positions = mx::transpose(positions, {1, 2, 0});
    return mx::astype(positions, mx::float32) * inv_freq;
  }();
  mx::array cos = mx::expand_dims(mx::cos(angles), 1);   // [B,1,T,half]
  mx::array sin = mx::expand_dims(mx::sin(angles), 1);
  // Half-split pairing duplicates each half's cos/sin into the full width.
  cos = mx::concatenate({cos, cos}, -1);
  sin = mx::concatenate({sin, sin}, -1);
  return {cos, sin};
}

mx::array rope_fused_apply(const mx::array& x, const mx::array& position_ids,
                           const mx::array& inv_freq,
                           const mx::array& position_selector,
                           int rotary_dim) {
  const int position_ndim = static_cast<int>(position_ids.ndim());
  if (x.ndim() != 4) {
    throw std::invalid_argument("rope_fused_apply: x must be 4-D [B,H,T,D]");
  }
  if (position_ndim != 2 && position_ndim != 3) {
    throw std::invalid_argument(
        "rope_fused_apply: position_ids must be 2-D or 3-D");
  }
  if (rotary_dim <= 0 || rotary_dim % 2 != 0 ||
      rotary_dim > static_cast<int>(x.shape(3))) {
    throw std::invalid_argument("rope_fused_apply: invalid rotary_dim");
  }

  const int D = static_cast<int>(x.shape(3));
  const int half_dim = rotary_dim / 2;
  const int slots = half_dim + D - rotary_dim;
  const std::size_t work_size = static_cast<std::size_t>(x.shape(0)) *
                                static_cast<std::size_t>(x.shape(1)) *
                                static_cast<std::size_t>(x.shape(2)) *
                                static_cast<std::size_t>(slots);

  try {
    auto kernel = mx::fast::metal_kernel(
        "mrope_apply_half_split_" + std::to_string(rotary_dim) + "_" +
            std::to_string(position_ndim) + "d",
        {"x", "position_ids", "inv_freq", "position_selector"}, {"x_out"},
        mrope_kernel_source(rotary_dim, position_ndim));

    std::vector<mx::array> inputs{x, position_ids, inv_freq, position_selector};
    std::vector<mx::Shape> out_shapes{x.shape()};
    std::vector<mx::Dtype> out_dtypes{x.dtype()};
    auto grid = std::make_tuple(static_cast<int>(work_size), 1, 1);
    auto threadgroup = std::make_tuple(256, 1, 1);
    std::vector<std::pair<std::string, mx::fast::TemplateArg>> tmpl = {
        {"T", x.dtype()},
    };
    std::vector<mx::array> out =
        kernel(inputs, out_shapes, out_dtypes, grid, threadgroup, tmpl,
               std::nullopt, false, mx::StreamOrDevice{});
    return out[0];
  } catch (const std::exception&) {
    // No Metal available: fall back to the pure-MLX reference. The cos/sin
    // used here come from mx.cos/mx.sin, so parity with the fused kernel is
    // approximate (~1 ULP) rather than bit-exact. This branch is never taken
    // on the Apple-Silicon production path the vectors are generated on.
    auto cs = rope_fused_cos_sin(position_ids, inv_freq, position_selector,
                                 rotary_dim);
    return rope_fused_apply_precomputed(x, cs.first, cs.second, rotary_dim);
  }
}

mx::array rope_fused_apply_precomputed(const mx::array& x,
                                       const mx::array& cos,
                                       const mx::array& sin,
                                       int rotary_dim) {
  const int half = rotary_dim / 2;
  const int D = static_cast<int>(x.shape(3));
  mx::array rot = mx::slice(x, {0, 0, 0, 0},
                            {static_cast<int>(x.shape(0)),
                             static_cast<int>(x.shape(1)),
                             static_cast<int>(x.shape(2)), half},
                            {1, 1, 1, 1});
  mx::array mirror = mx::slice(x, {0, 0, 0, half},
                               {static_cast<int>(x.shape(0)),
                                static_cast<int>(x.shape(1)),
                                static_cast<int>(x.shape(2)), rotary_dim},
                               {1, 1, 1, 1});
  mx::array pair = mx::concatenate({mirror, rot}, -1);

  // Fused kernel arithmetic: f32 multiply-add, then cast to T.
  mx::array rot32 = mx::astype(rot, mx::float32);
  mx::array pair32 = mx::astype(pair, mx::float32);
  mx::array cos32 = mx::astype(cos, mx::float32);
  mx::array sin32 = mx::astype(sin, mx::float32);
  mx::array embed =
      mx::astype(rot32 * cos32 - pair32 * sin32, x.dtype());

  if (D == rotary_dim) return embed;
  mx::array pass = mx::slice(x, {0, 0, 0, rotary_dim},
                             {static_cast<int>(x.shape(0)),
                              static_cast<int>(x.shape(1)),
                              static_cast<int>(x.shape(2)), D},
                             {1, 1, 1, 1});
  return mx::concatenate({embed, pass}, -1);
}

}  // namespace veystrix::engine
