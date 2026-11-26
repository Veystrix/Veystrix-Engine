// AUTO-GENERATED from mlx_lm.models.ssm (ssm.py:17-55). The MSL source below
// is copied VERBATIM from the `source` string constant of make_ssm_kernel();
// do not edit by hand. Regenerate by extracting the literal from ssm.py.
#pragma once

#include <string>
#include <mlx/mlx.h>

namespace veystrix::engine {

// Verbatim MSL source for the fused SSM step kernel ("ssm_kernel").
inline const std::string kSSMKernelSource = R"MAMBAKERNEL(
        auto n = thread_position_in_grid.z;
        auto h_idx = n % H;
        auto g_idx = n / G;
        constexpr int n_per_t = Ds / 32;

        auto x = X + n * Dh;
        out += n * Dh;
        auto i_state = state_in + n * Dh * Ds;
        auto o_state = state_out + n * Dh * Ds;

        // C and B have shape [batch, group, state_dim]
        // C and B need to be offset by group size
        auto C_ = C + g_idx * Ds;
        auto B_ = B + g_idx * Ds;

        auto ds_idx = thread_position_in_threadgroup.x;
        auto d_idx = thread_position_in_grid.y;

        auto dt_ = static_cast<float>(dt[n]);
        auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
        auto dA = fast::exp(A * dt_);

        float acc = 0.0;
        auto x_ = static_cast<float>(x[d_idx]);

        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * ds_idx + i;
            auto idx = d_idx * Ds + s_idx;
            auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
            auto state = dA * i_state[idx] + dB_by_x;
            o_state[idx] = static_cast<U>(state);
            acc += state * C_[s_idx];
        }
        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
        }
    )MAMBAKERNEL";

// Output of one fused SSM decode step: y [B,1,H,Dh] (input dtype) and the
// updated SSM state [B,H,Dh,Ds] (state dtype).
struct SSMStepOutput {
  mlx::core::array y;
  mlx::core::array state;
};

// Bit-exact decode path: launches the exact fused Metal kernel the Python
// engine compiles (ssm_kernel), reproducing its reduction order (simd_sum,
// 4-accumulate loop) and float math (fast::exp). Computes dt = compute_dt(...)
// internally, mirroring mlx_lm.models.ssm.ssm_update_kernel.
//
//   hidden_states (X) : [B,1,H,Dh]   (T = hidden_states.dtype(), e.g. bf16)
//   A_log            : [H]          f32
//   B, C             : [B,1,G,Ds]   (G = number of groups)
//   D                : [H]          (cast to hidden dtype by the mixer)
//   dt_raw           : [B,1,H]      (raw dt, promoted to f32 in compute_dt)
//   dt_bias          : [H]          f32
//   state            : [B,H,Dh,Ds]  (U = state.dtype(), e.g. f32)
//   time_step_min/max: compute_dt clip range
//
// Falls back to a NON-BIT-EXACT eager-ops reference if Metal is unavailable.
SSMStepOutput mamba_ssm_step_kernel(
    const mlx::core::array& hidden_states,
    const mlx::core::array& A_log,
    const mlx::core::array& B,
    const mlx::core::array& C,
    const mlx::core::array& D,
    const mlx::core::array& dt_raw,
    const mlx::core::array& dt_bias,
    const mlx::core::array& state,
    float time_step_min,
    float time_step_max);

}  // namespace veystrix::engine
