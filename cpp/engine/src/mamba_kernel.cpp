#include "mamba_kernel.h"

#include <mlx/mlx.h>
#include <mlx/fast.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mx = mlx::core;

namespace veystrix::engine {

namespace {

// mlx_lm.models.ssm.compute_dt (ssm.py:7-11), which is @mx.compile'd in Python:
//   dt = dt.astype(f32); dt = nn.softplus(dt + dt_bias); clip(dt, tmin, tmax)
// nn.softplus(x) == mx.logaddexp(x, 0) (verified in the installed MLX 0.32.2).
// The operand here is already f32, so no bf16-promotion hazard applies.
mx::array compute_dt(const mx::array& dt_raw, const mx::array& dt_bias,
                     float tmin, float tmax) {
  // Python's mlx_lm compute_dt is @mx.compile around nn.softplus, which is
  // itself @mx.compile(shapeless=True). The fused kernel differs by ~1 ULP
  // from the unfused logaddexp/clip chain; compile the SAME graph so the real
  // decode is bit-identical (this only surfaces after several mamba steps).
  static auto fn = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array dt = mx::astype(in[0], mx::float32);
        dt = mx::logaddexp(dt + in[1], mx::array(0.0f));
        dt = mx::clip(dt, in[2], in[3]);
        return {dt};
      },
      /*shapeless=*/true);
  return fn({dt_raw, dt_bias, mx::array(tmin), mx::array(tmax)})[0];
}

// NON-BIT-EXACT ops reference, used only when Metal is unavailable. It expresses
// the same Mamba-2 recurrence with eager MLX ops. It does NOT reproduce the
// kernel's `fast::exp` approximate builtin, the per-lane 4-accumulate order or
// the 32-lane `simd_sum`, and it uses a generic reduction instead. Do NOT use
// this path for parity/bit-exactness checks — tests must exercise the kernel.
SSMStepOutput mamba_ssm_step_ops(const mx::array& X, const mx::array& A_log,
                                 const mx::array& Bv, const mx::array& Cv,
                                 const mx::array& Dv, const mx::array& dt_raw,
                                 const mx::array& dt_bias, const mx::array& state,
                                 float tmin, float tmax) {
  const int B = X.shape(0);
  const int H = X.shape(2);
  const int Dh = X.shape(3);
  const int G = Bv.shape(2);            // number of groups
  const int Ds = Bv.shape(3);
  const int repeats = H / G;            // kernel template arg G = h // hb

  mx::array dt = compute_dt(dt_raw, dt_bias, tmin, tmax);   // [B,1,H] f32

  // dA[h] = exp(-exp(A_log[h]) * dt[h])  [B,1,H] f32
  mx::array A = -mx::exp(A_log);                            // [H] f32
  mx::array dA = mx::exp(mx::reshape(A, mx::Shape{1, 1, H}) * dt);

  // Expand B,C from [B,1,G,Ds] to one row per head [B,1,H,Ds].
  auto expand_group = [&](const mx::array& v) {
    mx::array v4 = mx::reshape(v, mx::Shape{B, 1, G, 1, Ds});
    mx::array vb = mx::broadcast_to(v4, mx::Shape{B, 1, G, repeats, Ds});
    return mx::reshape(vb, mx::Shape{B, 1, H, Ds});
  };
  mx::array Bh = expand_group(Bv);
  mx::array Ch = expand_group(Cv);

  // Add trailing singleton dims so MLX broadcasting lines up with the kernel.
  mx::array x5  = mx::reshape(X,    mx::Shape{B, 1, H, Dh, 1});
  mx::array dt5 = mx::reshape(dt,   mx::Shape{B, 1, H, 1, 1});
  mx::array dA5 = mx::reshape(dA,   mx::Shape{B, 1, H, 1, 1});
  mx::array B5  = mx::reshape(Bh,   mx::Shape{B, 1, H, 1, Ds});
  mx::array C5  = mx::reshape(Ch,   mx::Shape{B, 1, H, 1, Ds});
  mx::array st5 = mx::reshape(state, mx::Shape{B, 1, H, Dh, Ds});

  mx::array xf = mx::astype(x5, mx::float32);
  mx::array Bf = mx::astype(B5, mx::float32);
  mx::array Cf = mx::astype(C5, mx::float32);

  // S' = dA * S + x * dt * B ; y = sum_s(S' * C) + x * D
  mx::array new_state = dA5 * st5 + xf * dt5 * Bf;          // [B,1,H,Dh,Ds] f32
  mx::array state_out = mx::reshape(new_state, mx::Shape(state.shape()));
  mx::array acc = mx::sum(new_state * Cf, std::vector<int>{-1}, false);
  mx::array D3 = mx::reshape(Dv, mx::Shape{1, 1, H});
  mx::array y = mx::astype(
      acc + mx::astype(X, mx::float32) * mx::astype(D3, mx::float32), X.dtype());

  return {y, state_out};
}

}  // namespace

SSMStepOutput mamba_ssm_step_kernel(const mx::array& hidden_states,
                                    const mx::array& A_log,
                                    const mx::array& B,
                                    const mx::array& C,
                                    const mx::array& D,
                                    const mx::array& dt_raw,
                                    const mx::array& dt_bias,
                                    const mx::array& state,
                                    float time_step_min,
                                    float time_step_max) {
  // Exact decode kernel: mirrors mlx_lm.models.ssm.ssm_update_kernel with the
  // SAME MSL source string the Python engine compiles (byte-identical, see
  // mamba_kernel.h). Launch geometry, template args, output shapes/dtypes and
  // reduction order are reproduced exactly.
  int B_batch = hidden_states.shape(0);
  int H = hidden_states.shape(2);
  int Dh = hidden_states.shape(3);
  int groups = B.shape(2);             // number of groups (hb in Python)
  int G = H / groups;                  // kernel template arg G = h // hb
  int Ds = state.shape(3);

  try {
    auto kernel = mx::fast::metal_kernel(
        "ssm_kernel",
        {"X", "A_log", "B", "C", "D", "dt", "state_in"},
        {"out", "state_out"},
        kSSMKernelSource);

    mx::array dt = compute_dt(dt_raw, dt_bias, time_step_min, time_step_max);
    std::vector<mx::array> inputs{hidden_states, A_log, B, C, D, dt, state};
    std::vector<mx::Shape> out_shapes{mx::Shape{B_batch, 1, H, Dh},
                                      mx::Shape(state.shape())};
    std::vector<mx::Dtype> out_dtypes{hidden_states.dtype(), state.dtype()};
    auto grid = std::make_tuple(32, Dh, H * B_batch);
    auto threadgroup = std::make_tuple(32, 8, 1);
    std::vector<std::pair<std::string, mx::fast::TemplateArg>> tmpl = {
        {"T", hidden_states.dtype()}, {"U", state.dtype()},
        {"Dh", Dh}, {"Ds", Ds}, {"H", H}, {"G", G},
    };
    std::vector<mx::array> out = kernel(inputs, out_shapes, out_dtypes,
                                        grid, threadgroup, tmpl,
                                        std::nullopt, false, mx::StreamOrDevice{});
    return {out[0], out[1]};
  } catch (const std::exception&) {
    // No Metal — fall back to the ops reference. NOT bit-exact.
    return mamba_ssm_step_ops(hidden_states, A_log, B, C, D, dt_raw, dt_bias,
                              state, time_step_min, time_step_max);
  }
}

}  // namespace veystrix::engine
