// Convenience wrapper around the bit-exact fused-Metal MRoPE in rope_fused.hpp.
//
// rope_fused_apply() takes the *kernel inputs* (inv_freq + position_selector)
// explicitly, which is ideal for tests but verbose at call sites that already
// have a QSARotaryParams / rotary config. This header builds those inputs with
// the exact production helpers and forwards to rope_fused_apply().
//
// Drop-in usage in the full-attention path (real_checkpoint_generate.cpp):
//
//   #include "rope_fused_apply.hpp"
//   ...
//   query = veystrix::engine::rope_apply_fused(query, position_ids, cfg.rope);
//   // instead of qsa_apply_rope(query, position_ids, cfg.rope)
//
// This file is additive only: it does not modify qsa_full.cpp or any shared
// file. The primary agent can wire it in after validating the parity test.
#pragma once

#include <vector>

#include "qsa_full.hpp"   // QSARotaryParams
#include "rope_fused.hpp"

namespace veystrix::engine {

// Build inv_freq + position selector from raw config and apply the fused kernel.
// `interleaved=false` selects the chunked selector (production's
// _chunked_position_selector); for this model it is always true.
inline mx::array rope_apply_fused(const mx::array& x,
                                  const mx::array& position_ids,
                                  int rotary_dim, float base,
                                  const std::vector<int>& mrope_section,
                                  bool interleaved = true) {
  mx::array inv_freq = rope_fused_compute_inv_freq(rotary_dim, base);
  mx::array selector =
      rope_fused_position_selector(mrope_section, rotary_dim / 2, interleaved);
  return rope_fused_apply(x, position_ids, inv_freq, selector, rotary_dim);
}

// Convenience overload matching qsa_apply_rope's signature exactly, so it can
// replace the eager call without touching surrounding code.
inline mx::array rope_apply_fused(const mx::array& x,
                                  const mx::array& position_ids,
                                  const QSARotaryParams& rp) {
  return rope_apply_fused(x, position_ids, rp.dim, rp.base,
                          rp.mrope_section, rp.style != "chunked");
}

}  // namespace veystrix::engine
