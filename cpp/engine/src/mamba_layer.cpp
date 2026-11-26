#include "mamba_layer.hpp"

#include "mamba_conv.h"
#include "mamba_kernel.h"

#include <mlx/mlx.h>
#include <mlx/fast.h>

#include <utility>
#include <vector>

namespace mx = mlx::core;

namespace veystrix::engine {

namespace {

// MambaRMSNormGated.__call__ (nemotron_h.py:83-88):
//   x = swiglu(gate, x)                       # nn.silu(gate) * x
//   x = mx.unflatten(x, -1, (-1, group_size)) # [B,1,ngroups,group_size]
//   x = mx.fast.rms_norm(x, weight=None, eps)
//   return self.weight * x.flatten(-2)
mx::array mamba_rms_norm_gated(const mx::array& y, const mx::array& gate,
                               const mx::array& weight, int n_groups,
                               float eps) {
  // swiglu(gate, y) == nn.silu(gate) * y (activations.py:10-11).
  mx::array x = swiglu(gate, y);
  const int B = x.shape(0);
  const int S = x.shape(1);
  const int I = x.shape(2);
  const int group_size = I / n_groups;
  x = mx::reshape(x, mx::Shape{B, S, n_groups, group_size});
  x = mx::fast::rms_norm(x, std::nullopt, eps);   // weight=None
  x = mx::reshape(x, mx::Shape{B, S, I});
  return weight * x;                              // broadcast [I] over [B,S,I]
}

}  // namespace

MambaLayerOutput mamba_layer_forward(const mx::array& x, const MambaWeights& w,
                                     const MambaCacheState& cache,
                                     const MambaConfig& cfg) {
  const int B = x.shape(0);
  const int S = x.shape(1);
  const int hidden = cfg.hidden_size;
  const int I = cfg.intermediate_size;
  const int conv_dim = cfg.conv_dim;
  const int H = cfg.num_heads;
  const int Dh = cfg.head_dim;
  const int G = cfg.n_groups;
  const int Ds = cfg.ssm_state_size;
  const int K = cfg.conv_kernel;

  mx::array conv_state = cache.conv_state;
  mx::array ssm_state = cache.ssm_state;
  std::vector<mx::array> steps;
  steps.reserve(S);

  for (int t = 0; t < S; ++t) {
    mx::array xt = mx::slice(x, mx::Shape{0, t, 0}, mx::Shape{B, t + 1, hidden});

    // 1. in_proj: [B,1,hidden] -> [B,1,I+conv_dim+H].
    mx::array projected = linear(xt, w.in_proj);

    // 2. split -> gate, conv_input, dt (all bf16).
    mx::array gate = mx::slice(projected, mx::Shape{0, 0, 0},
                               mx::Shape{B, 1, I});
    mx::array conv_input = mx::slice(projected, mx::Shape{0, 0, I},
                                     mx::Shape{B, 1, I + conv_dim});
    mx::array dt_raw = mx::slice(projected, mx::Shape{0, 0, I + conv_dim},
                                 mx::Shape{B, 1, I + conv_dim + H});

    // 3. cached depthwise conv state prep / tail update.
    MambaConvPrep prep = mamba_conv_prep(conv_state, conv_input, K);
    conv_state = prep.new_state;

    // 4. depthwise conv (+ bias) + silu: [B,1,conv_dim].
    mx::array conv_out =
        mamba_causal_conv_decode_silu(prep.padded, w.conv1d_weight, w.conv1d_bias);

    // 5. split conv_out -> hidden_ssm, B, C, then reshape to per-head views.
    mx::array hidden_ssm = mx::slice(conv_out, mx::Shape{0, 0, 0},
                                     mx::Shape{B, 1, I});
    mx::array Bv = mx::slice(conv_out, mx::Shape{0, 0, I},
                             mx::Shape{B, 1, I + G * Ds});
    mx::array Cv = mx::slice(conv_out, mx::Shape{0, 0, I + G * Ds},
                             mx::Shape{B, 1, I + 2 * G * Ds});
    hidden_ssm = mx::reshape(hidden_ssm, mx::Shape{B, 1, H, Dh});
    Bv = mx::reshape(Bv, mx::Shape{B, 1, G, Ds});
    Cv = mx::reshape(Cv, mx::Shape{B, 1, G, Ds});

    // 6-8. SSM step (fused Metal kernel). It computes dt = compute_dt(dt_raw,
    // dt_bias) internally, exactly like ssm_update_kernel. D is cast to the
    // hidden dtype by the mixer (nemotron_h.py:195).
    mx::array D = mx::astype(w.D, xt.dtype());
    SSMStepOutput step = mamba_ssm_step_kernel(
        hidden_ssm, w.A_log, Bv, Cv, D, dt_raw, w.dt_bias, ssm_state,
        cfg.time_step_min, cfg.time_step_max);
    ssm_state = step.state;

    // 9. y.reshape([B,1,I]).
    mx::array y = mx::reshape(step.y, mx::Shape{B, 1, I});

    // 10. MambaRMSNormGated(y, gate).
    mx::array normed = mamba_rms_norm_gated(y, gate, w.norm_weight, G, cfg.eps);

    // 11. out_proj.
    steps.push_back(linear(normed, w.out_proj));  // [B,1,hidden]
  }

  // 12. (block residual x + hidden is applied by the caller, not the mixer.)
  mx::array out = (S == 1) ? steps[0] : mx::concatenate(steps, 1);
  return {out, conv_state, ssm_state};
}

}  // namespace veystrix::engine
