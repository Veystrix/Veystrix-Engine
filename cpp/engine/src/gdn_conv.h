// Bit-exact port of the GDN causal-conv1d DECODE chain from
// mlx_vlm.models.qwen3_5.language:
//   - _qwen3_5_decode_depthwise_conv  (the depthwise conv + cast helper)
//   - Qwen3_5GatedDeltaNet._causal_conv1d_decode  (conv weight prep)
//   - nn.silu  (applied by Qwen3_5GatedDeltaNet.__call__)
#pragma once
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace veystrix::engine {

// Depthwise-conv helper on an ALREADY-PREPPED weight of shape [K, C] float32.
// Mirrors _qwen3_5_decode_depthwise_conv exactly:
//   out = sum(astype(conv_input, f32) * weight[None], axis=1)
//   return astype(out, conv_input.dtype)[:, None, :]
// conv_input: [B, K, C] (half dtype); weight: [K, C] float32;
// returns: [B, 1, C] in conv_input's dtype.
mx::array gdn_decode_depthwise_conv(const mx::array& conv_input,
                                    const mx::array& weight);

// Full decode chain from the RAW nn.Conv1d weight of shape [C, K, 1] (half
// dtype). Does the exact prep conv1d.weight[:, :, 0].T.astype(float32) and
// then calls gdn_decode_depthwise_conv. Returns the pre-silu conv output.
mx::array gdn_causal_conv_decode(const mx::array& conv_input,
                                 const mx::array& weight);

// Full chain including the fused activation applied on the production path:
//   silu(conv_out) = conv_out * sigmoid(conv_out)
// (nn.silu is mx.compile'd on the Python side; this mirrors its op sequence).
mx::array gdn_causal_conv_decode_silu(const mx::array& conv_input,
                                      const mx::array& weight);

}  // namespace veystrix::engine
