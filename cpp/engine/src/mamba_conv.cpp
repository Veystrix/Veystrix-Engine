#include "mamba_conv.h"

#include "engine.hpp"  // hc_silu

#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace veystrix::engine {

MambaConvPrep mamba_conv_prep(const mx::array& conv_state,
                              const mx::array& conv_input, int K) {
  // Python keeps conv_state in the activation dtype; cast defensively so the
  // concatenate cannot promote the chain to f32.
  mx::array cs = mx::astype(conv_state, conv_input.dtype());
  mx::array padded = mx::concatenate({cs, conv_input}, 1);  // [B, K-1+S, C]
  const int n_keep = K - 1;
  const int T = padded.shape(1);
  // cache[0] = contiguous(padded[:, -n_keep:, :]).
  mx::array new_state = mx::contiguous(mx::slice(
      padded, mx::Shape{0, T - n_keep, 0}, mx::Shape{padded.shape(0), T,
                                                    padded.shape(2)}));
  return {padded, new_state};
}

mx::array mamba_decode_depthwise_conv(const mx::array& conv_input,
                                      const mx::array& weight,
                                      const mx::array& bias) {
  // Confirmed bit-exact to MLX nn.Conv1d (depthwise, bf16, K=4, +bias):
  //   out = astype(sum(astype(x,f32) * weight[None,:,:], axis=1), x.dtype)
  //   out = out[:, None, :] + bias    (bias add in the activation dtype)
  mx::array in_f32 = mx::astype(conv_input, mx::float32);
  mx::array w = mx::expand_dims(weight, 0);  // [1, K, C]
  mx::array prod = in_f32 * w;               // broadcast [B,K,C] * [1,K,C]
  mx::array summed = mx::sum(prod, 1);       // reduce time axis -> [B, C]
  mx::array casted = mx::astype(summed, conv_input.dtype());
  mx::array out = mx::expand_dims(casted, 1);  // [B, 1, C]
  if (bias.size()) out = out + bias;           // bf16 + bf16 -> bf16
  return out;
}

mx::array mamba_causal_conv_decode(const mx::array& conv_input,
                                   const mx::array& raw_weight,
                                   const mx::array& bias) {
  // Raw nn.Conv1d.weight has shape [C, K, 1]. Prep: weight[:, :, 0].T [K, C]
  // cast to f32 (matches the gdn_conv prep, with bias added by the caller).
  mx::array w = mx::squeeze(raw_weight, 2);  // [:, :, 0] -> [C, K]
  w = mx::transpose(w, {1, 0});              // .T -> [K, C]
  w = mx::astype(w, mx::float32);
  return mamba_decode_depthwise_conv(conv_input, w, bias);
}

mx::array mamba_causal_conv_decode_silu(const mx::array& conv_input,
                                        const mx::array& raw_weight,
                                        const mx::array& bias) {
  // _conv returns nn.silu(conv1d(padded)); nn.silu is the compiled x*sigmoid(x).
  mx::array conv_out = mamba_causal_conv_decode(conv_input, raw_weight, bias);
  return hc_silu(conv_out);
}

}  // namespace veystrix::engine
