// Bit-exact C++ port of PRODUCTION's *fused-Metal* MRoPE rotary path.
//
// Source of truth (mlx_vlm 0.32.2):
//   .venv/lib/python3.12/site-packages/mlx_vlm/models/rope_utils.py
//     MRoPERotaryEmbedding.apply_rotary              lines 1244-1287
//     _compiled_mrope_apply / _fast_mrope_apply      lines 1007-1025, 693-742
//     _mrope_apply_kernel                            lines 568-652
//     compute_inv_freq / mrope_position_selector     lines 1043, 536-539
//   .venv/lib/python3.12/site-packages/mlx_vlm/models/qwen3_5/language.py
//     Qwen3_5RotaryEmbedding (style="interleaved")    lines 27-42
//   .venv/lib/python3.12/site-packages/mlx_vlm/models/qwen4_exp/language.py
//     Qwen4ExpQSAIndexer._apply_rope                  lines 300-304
//
// WHY THIS IS NOT JUST A DIFFERENT cos/sin MULTIPLY-ADD ORDER
// -----------------------------------------------------------
// The eager RoPE path (qsa_full.cpp::qsa_apply_rope) builds a cos/sin array
// with mx.cos/mx.sin and then does `x*cos + rotate_half(x)*sin`.
//
// The FUSED path does NOT use a precomputed cos/sin array at all. Its Metal
// kernel computes the angle *per frequency slot inside the kernel*:
//
//     float pos   = static_cast<float>(position_ids[...]);
//     float angle = pos * static_cast<float>(inv_freq[freq_idx]);
//     float c = metal::cos(angle);
//     float s = metal::sin(angle);
//     x_out[idx]        = static_cast<T>(xv * c - xp * s);
//     x_out[base+pair_d]= static_cast<T>(xp * c + xv * s);
//
// `metal::cos` / `metal::sin` are Metal stdlib intrinsics and differ from
// MLX's mx.cos/mx.sin by ~1 float32 ULP. To be bit-exact we therefore call
// the EXACT kernel source via mx::fast::metal_kernel (the same technique used
// by gdn_kernel.cpp). The kernel source is assembled in rope_fused.cpp from
// the verbatim f-string in rope_utils.py.
//
// TWO PRODUCTION QUIRKS THAT ARE DELIBERATELY REPRODUCED:
//   1. The fused kernel IGNORES attention_scaling (there is no scaling input
//      in the kernel). QSA uses scaling == 1.0, so this has no impact there.
//   2. The rotate is done in float32 and cast to T at the end. For f16/bf16
//      this differs from the eager path (which casts cos/sin to T *first*).
//
// This file is self-contained and does not modify any existing file.
// See research_notes/Cerebras inference and IO splitting/rope_fused_port.md.
#pragma once

#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace veystrix::engine {

// Exact rope_utils.compute_inv_freq (lines 1043-1044), shape [dim/2], float32:
//   1.0 / (base ** (arange(0, dim, 2).astype(float32) / dim))
mx::array rope_fused_compute_inv_freq(int dim, float base);

// Exact rope_utils.mrope_position_selector (lines 536-539 / 512-527).
// Returns an int32 array of length `freq_dim` (= rotary_dim/2) giving the
// MRoPE axis (0,1,2) that feeds each frequency. For 2-D text position ids the
// kernel ignores this, but production still passes it, so we return it too.
//   interleaved == true  -> _interleaved_position_selector
//   interleaved == false -> _chunked_position_selector
mx::array rope_fused_position_selector(const std::vector<int>& mrope_section,
                                       int freq_dim, bool interleaved = true);

// Reference cos/sin using the SAME angle computation as the fused kernel
// (`angle = pos * inv_freq[freq_idx]`, per-axis selected for 3-D ids), but
// with mx.cos/mx.sin instead of metal::cos/metal::sin. This is useful for
// analysis only; it can differ from the kernel by ~1 ULP. For bit-exact
// fused parity use rope_fused_apply(). Returns [B, 1, T, rotary_dim] arrays.
std::pair<mx::array, mx::array> rope_fused_cos_sin(
    const mx::array& position_ids, const mx::array& inv_freq,
    const mx::array& position_selector, int rotary_dim,
    bool interleaved = true);

// Bit-exact reproduction of production's fused-Metal rotary apply.
// Mirrors `_fast_mrope_apply(kernel, x, x, position_ids, inv_freq,
// position_selector, pairing="half_split")` from rope_utils.py, i.e. exactly
// what Qwen4ExpQSAIndexer._apply_rope triggers when fused_apply is True.
//
//   x                : [B, H, T, D]  (any float dtype); first `rotary_dim`
//                      channels are rotated, the rest are copied through.
//   position_ids     : [B, T] (text) or [axis, B, T] (MRoPE), int32.
//   inv_freq         : [rotary_dim/2] float32 (rope_fused_compute_inv_freq).
//   position_selector: [rotary_dim/2] int32 (rope_fused_position_selector).
//   rotary_dim       : partial-rotary dimension (e.g. 32).
//
// Returns the same shape and dtype as `x`.
mx::array rope_fused_apply(const mx::array& x, const mx::array& position_ids,
                           const mx::array& inv_freq,
                           const mx::array& position_selector,
                           int rotary_dim);

// Convenience overload: apply PRECOMPUTED cos/sin in the fused kernel's exact
// arithmetic order (float32 multiply-add, then cast to x's dtype):
//   out[..., i]       = T(xv*c - xp*s)
//   out[..., i+half]  = T(xp*c + xv*s)
// This matches the kernel whenever `cos`/`sin` are the same values the kernel
// would compute internally. `cos`/`sin` are expected as [B, 1, T, rotary_dim]
// (or broadcastable to x's leading dims). Provided for callers that already
// have cos/sin (the "different multiply-add order" comparison case).
mx::array rope_fused_apply_precomputed(const mx::array& x,
                                       const mx::array& cos,
                                       const mx::array& sin,
                                       int rotary_dim);

}  // namespace veystrix::engine
