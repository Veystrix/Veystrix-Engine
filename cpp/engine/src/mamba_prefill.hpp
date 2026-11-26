#pragma once

// Bit-exact S > 1 Mamba-2 prefill for Nemotron 3.5 Lightning.
//
// Mirrors mlx_lm.models.nemotron_h.NemotronHMamba2Mixer.__call__ for a prompt
// chunk (full depthwise conv1d over the padded sequence + the SSD-SSM chunked
// scan from mlx_lm.models.ssm.ssm_attn), INCLUDING state carry across chunks.
// The S == 1 decode path (mamba_layer_forward / mamba_kernel) is untouched.
//
// This is the numeric port required by plan P5b (R2): every Nemotron parity
// result to date used chunk_tokens=1, which is NOT the production prefill.

#include "mamba_layer.hpp"

namespace veystrix::engine {

// One S > 1 Mamba mixer call. `cache` is the conv/ssm state from the previous
// chunk (or zeros); the returned state advances by S tokens. Output is
// [B, S, hidden_size].
MambaLayerOutput mamba_layer_prefill(const mx::array& x,
                                     const MambaWeights& w,
                                     const MambaCacheState& cache,
                                     const MambaConfig& cfg);

}  // namespace veystrix::engine
