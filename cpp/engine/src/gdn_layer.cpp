#include "gdn_layer.hpp"

#include "engine.hpp"
#include "gdn_conv.h"

#include <mlx/mlx.h>
#include <mlx/fast.h>

#include <utility>

namespace veystrix::engine {

GDNLayerOutput gdn_layer_forward(const mx::array& x, const GDNWeights& w,
                                 const GDNCacheState& cache,
                                 const GDNConfig& cfg) {
  const int B = x.shape(0);
  const int S = x.shape(1);
  const int H = cfg.hidden_size;
  const int Hk = cfg.num_k_heads;
  const int Hv = cfg.num_v_heads;
  const int Dk = cfg.head_k_dim;
  const int Dv = cfg.head_v_dim;
  const int K = cfg.conv_kernel;
  const int conv_dim = 2 * cfg.key_dim + cfg.value_dim;
  const bool sigmoid = cfg.output_activation == GDNActivation::Sigmoid;

  mx::array conv_state = cache.conv_state;
  mx::array rec_state = cache.recurrent_state;
  std::vector<mx::array> steps;
  steps.reserve(S);

  for (int t = 0; t < S; ++t) {
    // One token: x_t [B, 1, H].
    mx::array xt = mx::slice(x, mx::Shape{0, t, 0}, mx::Shape{B, t + 1, H});

    // 1. Linear projections (nn.Linear: y = x @ W.T).
    mx::array mixed_qkv = linear(xt, w.in_proj_qkv);  // [B,1,conv_dim]
    mx::array z = linear(xt, w.in_proj_z);            // [B,1,value_dim]
    mx::array b = linear(xt, w.in_proj_b);            // [B,1,Hv]
    mx::array a = linear(xt, w.in_proj_a);            // [B,1,Hv]
    z = mx::reshape(z, mx::Shape{B, 1, Hv, Dv});

    // 2. Cached depthwise conv: conv_input = concat([conv_state, mixed_qkv]).
    mx::array conv_input = mx::concatenate({conv_state, mixed_qkv}, 1);  // [B,K,conv_dim]
    const int n_keep = K - 1;
    const int Tconv = conv_input.shape(1);
    // cache[0] = contiguous(conv_input[:, -n_keep:, :]).
    conv_state = mx::contiguous(mx::slice(
        conv_input, mx::Shape{0, Tconv - n_keep, 0},
        mx::Shape{B, Tconv, conv_dim}));

    // depthwise conv + silu (prep the raw [C,K,1] weight once per call).
    mx::array conv_out = gdn_causal_conv_decode_silu(conv_input, w.conv1d_weight);  // [B,1,C]

    // 3. Split conv_out into q, k, v and reshape to per-head views.
    mx::array q = mx::slice(conv_out, mx::Shape{0, 0, 0},
                            mx::Shape{B, 1, cfg.key_dim});
    mx::array k = mx::slice(conv_out, mx::Shape{0, 0, cfg.key_dim},
                            mx::Shape{B, 1, 2 * cfg.key_dim});
    mx::array v = mx::slice(conv_out, mx::Shape{0, 0, 2 * cfg.key_dim},
                            mx::Shape{B, 1, conv_dim});
    q = mx::reshape(q, mx::Shape{B, 1, Hk, Dk});
    k = mx::reshape(k, mx::Shape{B, 1, Hk, Dk});
    v = mx::reshape(v, mx::Shape{B, 1, Hv, Dv});

    // 4. Qwen4 _normalize_qk (L2 + query scale). The fused Metal kernel already
    //    maps the Hv value heads onto the Hk key heads via
    //    hk_idx = hv_idx / (Hv / Hk), so q/k stay at Hk heads, exactly as in
    //    Python's gated_delta_update -> gated_delta_kernel (no explicit repeat).
    auto qk = gdn_normalize_qk(q, k);

    // 5. g, beta and the gated-delta-rule recurrent step (kernel path).
    auto gb = compute_g_beta(w.A_log, a, b, w.dt_bias);
    auto step = gated_delta_step_kernel(qk.first, qk.second, v, gb.first,
                                        gb.second, rec_state);
    rec_state = step.state;
    // Python gated_delta_kernel returns y.astype(q.dtype) (bf16 on real ckpt).
    step.y = mx::astype(step.y, xt.dtype());

    // 6. Qwen4ExpRMSNormGated(output_gate_type) with the z projection as gate.
    mx::array normed = rms_norm_gated(step.y, w.norm_weight, z, cfg.eps, sigmoid);

    // 7. out_proj on the flattened per-head output.
    mx::array flat = mx::reshape(normed, mx::Shape{B, 1, cfg.value_dim});
    steps.push_back(linear(flat, w.out_proj));  // [B,1,H]
  }

  mx::array out = (S == 1) ? steps[0] : mx::concatenate(steps, 1);
  return {out, conv_state, rec_state};
}

}  // namespace veystrix::engine
