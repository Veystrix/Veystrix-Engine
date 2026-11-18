// Bit-exact port of the Nemotron-H depthwise-conv1d DECODE chain from
// mlx_lm.models.nemotron_h.NemotronHMamba2Mixer._conv / nn.Conv1d, plus the
// conv-cache prep/tail update (plan item 3). Unlike the GDN conv, Mamba's
// nn.Conv1d carries a bias (use_conv_bias=True) and works on [B,K,C].
#pragma once
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace veystrix::engine {

// Result of preparing one cached-conv decode step.
struct MambaConvPrep {
  mx::array padded;     // [B, K, conv_dim]  (conv_state ++ conv_input)
  mx::array new_state;  // [B, K-1, conv_dim] (next cache[0])
};

// _conv state prep/tail (nemotron_h.py:143-159, S=1 path):
//   conv_state = conv_state.astype(conv_input.dtype)
//   padded = concatenate([conv_state, conv_input], axis=1)
//   cache[0] = contiguous(padded[:, -(K-1):, :])
MambaConvPrep mamba_conv_prep(const mx::array& conv_state,
                              const mx::array& conv_input, int K);

// Depthwise conv on an ALREADY-PREPPED weight [K, C] (float32) with bias [C].
// Mirrors MLX nn.Conv1d (bf16, depthwise, K=4, bias) bit-exactly:
//   out = astype(sum(astype(x,f32) * weight[None,:,:], axis=1), x.dtype)
//   out = out[:, None, :] + bias        (bias add in the activation dtype)
// Verified empirically against nn.Conv1d for C=64 and C=1536, K=4, bf16.
// bias may be an empty array to disable the add. Returns [B, 1, C].
mx::array mamba_decode_depthwise_conv(const mx::array& conv_input,
                                      const mx::array& weight,
                                      const mx::array& bias);

// Full decode chain from the RAW nn.Conv1d weight [C, K, 1] (bf16) and its
// bias [C]. Does the exact weight prep conv1d.weight[:, :, 0].T.astype(f32)
// and then calls mamba_decode_depthwise_conv. Returns the pre-silu conv output.
mx::array mamba_causal_conv_decode(const mx::array& conv_input,
                                   const mx::array& raw_weight,
                                   const mx::array& bias);

// Full chain including nn.silu (the activation _conv applies to the conv
// output): silu(x) = x * sigmoid(x), emitted via the same compiled graph.
mx::array mamba_causal_conv_decode_silu(const mx::array& conv_input,
                                        const mx::array& raw_weight,
                                        const mx::array& bias);

}  // namespace veystrix::engine
