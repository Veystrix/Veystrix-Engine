#include "gdn_conv.h"

#include "engine.hpp"  // hc_silu

#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace veystrix::engine {

mx::array gdn_decode_depthwise_conv(const mx::array& conv_input,
                                    const mx::array& weight) {
  // Exact mirror of the Python helper:
  //   @partial(mx.compile, shapeless=True)
  //   def _qwen3_5_decode_depthwise_conv(conv_input, weight):
  //       out = mx.sum(conv_input.astype(mx.float32) * weight[None, :, :],
  //                    axis=1)
  //       return out.astype(conv_input.dtype)[:, None, :]
  mx::array in_f32 = mx::astype(conv_input, mx::float32);
  mx::array w = mx::expand_dims(weight, 0);  // weight[None, :, :] -> [1, K, C]
  mx::array prod = in_f32 * w;               // broadcast [B,K,C] * [1,K,C]
  mx::array summed = mx::sum(prod, 1);       // reduce time axis -> [B, C]
  mx::array casted = mx::astype(summed, conv_input.dtype());
  return mx::expand_dims(casted, 1);         // [:, None, :] -> [B, 1, C]
}

mx::array gdn_causal_conv_decode(const mx::array& conv_input,
                                 const mx::array& weight) {
  // Exact prep from Qwen3_5GatedDeltaNet._causal_conv1d_decode:
  //   weight = self.conv1d.weight[:, :, 0].T.astype(mx.float32)
  // raw conv1d.weight has shape [C, K, 1] (depthwise nn.Conv1d).
  mx::array w = mx::squeeze(weight, 2);  // [:, :, 0] -> [C, K] (axis 2 is size 1)
  w = mx::transpose(w, {1, 0});          // .T -> [K, C]
  w = mx::astype(w, mx::float32);                  // .astype(mx.float32)
  return gdn_decode_depthwise_conv(conv_input, w);
}

mx::array gdn_causal_conv_decode_silu(const mx::array& conv_input,
                                      const mx::array& weight) {
  // nn.silu(x) == x * mx.sigmoid(x); applied to the conv output on the
  // production path before the q/k/v split.
  mx::array conv_out = gdn_causal_conv_decode(conv_input, weight);
  // Python uses nn.silu (compiled/fused); use hc_silu to emit the same kernel.
  return hc_silu(conv_out);
}

}  // namespace veystrix::engine
