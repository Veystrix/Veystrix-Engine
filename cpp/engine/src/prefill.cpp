#include "prefill.hpp"

#include "engine.hpp"
#include "gdn_kernel.h"

#include <mlx/mlx.h>
#include <mlx/fast.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace veystrix::engine {

// Exact masked variant of the fused gated-delta Metal kernel from
// mlx_vlm.models.qwen3_5.gated_delta._make_gated_delta_kernel(has_mask=True,
// vectorized=False). Identical to kGdnDeltaKernelSource except the per-token
// guard: masked positions emit y=0 and leave the recurrent state untouched.
// (This is the kernel production actually uses when an SSM/padding mask is
// passed on M4 GPU; the unmasked source in gdn_kernel.h is used when mask=None.)
inline const std::string kGdnDeltaMaskedKernelSource = R"GDNMASKED(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          if (mask[b_idx * T + t]) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y[dv_idx] = static_cast<InT>(out);
            }
          } else {
            y[dv_idx] = static_cast<InT>(0);
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
    )GDNMASKED";

namespace {

// Run the exact fused Metal delta kernel over the whole [B,T,..] sequence.
// With mask==nullptr this is byte-identical to gdn_kernel.cpp's
// gated_delta_step_kernel; with a mask it selects the masked source/inputs
// exactly as gated_delta_kernel -> _gated_delta_kernel_masked does in Python.
DeltaStepOutput run_delta_kernel(const mx::array& q, const mx::array& k,
                                 const mx::array& v, const mx::array& g,
                                 const mx::array& beta, const mx::array& state,
                                 const mx::array* mask) {
  const int B = q.shape(0), T = q.shape(1);
  const int Hk = q.shape(2), Dk = q.shape(3);
  const int Hv = v.shape(2), Dv = v.shape(3);

  try {
    const bool masked = mask != nullptr;
    const std::string& source =
        masked ? kGdnDeltaMaskedKernelSource : kGdnDeltaKernelSource;
    const char* name = masked ? "gated_delta_step_mask" : "gated_delta_step";
    std::vector<std::string> input_names = {
        "q", "k", "v", "g", "beta", "state_in", "T"};
    if (masked) input_names.push_back("mask");

    auto kernel = mx::fast::metal_kernel(
        name, input_names, {"y", "state_out"}, source);

    mx::array Tarr(static_cast<int32_t>(T));
    std::vector<mx::array> inputs{q, k, v, g, beta, state, Tarr};
    if (masked) inputs.push_back(*mask);

    std::vector<mx::Shape> out_shapes{mx::Shape{B, T, Hv, Dv},
                                      mx::Shape(state.shape())};
    std::vector<mx::Dtype> out_dtypes{q.dtype(), state.dtype()};
    auto grid = std::make_tuple(32, Dv, B * Hv);
    auto threadgroup = std::make_tuple(32, 4, 1);
    std::vector<std::pair<std::string, mx::fast::TemplateArg>> tmpl = {
        {"InT", q.dtype()}, {"StT", state.dtype()},
        {"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk}, {"Hv", Hv},
    };
    std::vector<mx::array> out = kernel(inputs, out_shapes, out_dtypes,
                                        grid, threadgroup, tmpl,
                                        std::nullopt, false,
                                        mx::StreamOrDevice{});
    return {out[0], out[1]};
  } catch (const std::exception&) {
    // No Metal: fall back to the ops reference loop (bit-similar, not exact).
    // Note the fallback cannot honour the kernel's mask semantics; use only on
    // CPU where production would take gated_delta_ops instead of the kernel.
    return gated_delta_step(q, k, v, g, beta, state);
  }
}

}  // namespace

GDNLayerOutput gdn_prefill(const mx::array& x, const GDNWeights& w,
                           const GDNCacheState& cache, const GDNConfig& cfg,
                           const mx::array* mask) {
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

  // 1. Full-sequence linear projections (nn.Linear: y = x @ W.T). Production
  //    projects all S tokens in one matmul; doing S separate matmuls (as the
  //    decode wrapper does) is NOT bit-exact to that, so keep the full matmul.
  mx::array mixed_qkv = mx::matmul(x, mx::transpose(w.in_proj_qkv));  // [B,S,C]
  mx::array z = mx::matmul(x, mx::transpose(w.in_proj_z));            // [B,S,Hv*Dv]
  mx::array b = mx::matmul(x, mx::transpose(w.in_proj_b));            // [B,S,Hv]
  mx::array a = mx::matmul(x, mx::transpose(w.in_proj_a));            // [B,S,Hv]
  z = mx::reshape(z, mx::Shape{B, S, Hv, Dv});

  // 2. Mask handling: production masks mixed_qkv BEFORE the conv, and passes
  //    the same mask into the fused delta kernel (handled in run_delta_kernel).
  //    a / b are never masked. mask.shape[0] != B would set mask=None in Python.
  if (mask != nullptr && mask->shape(0) == B) {
    mx::array cond = mx::expand_dims(*mask, std::vector<int>{-1});  // [B,S,1]
    mixed_qkv = mx::where(cond, mixed_qkv, mx::zeros_like(mixed_qkv));
  }

  // 3. conv_input = concat([conv_state, mixed_qkv]); advance conv cache.
  mx::array conv_input = mx::concatenate({cache.conv_state, mixed_qkv}, 1);
  const int n_keep = K - 1;
  const int Tconv = conv_input.shape(1);
  mx::array new_conv_state = mx::contiguous(mx::slice(
      conv_input, mx::Shape{0, Tconv - n_keep, 0},
      mx::Shape{B, Tconv, conv_dim}));

  // 4. FULL depthwise causal conv1d over the sequence (the decisive S>1
  //    difference from the decode path) + silu. mx::conv1d is the same primitive
  //    nn.Conv1d calls (groups=conv_dim, bias=False), so it is bit-exact.
  mx::array conv_out = mx::conv1d(conv_input, w.conv1d_weight,
                                  /*stride=*/1, /*padding=*/0,
                                  /*dilation=*/1, /*groups=*/conv_dim);  // [B,S,C]
  conv_out = conv_out * mx::sigmoid(conv_out);  // nn.silu

  // 5. Split into q, k, v and reshape to per-head views.
  mx::array q = mx::slice(conv_out, mx::Shape{0, 0, 0},
                          mx::Shape{B, S, cfg.key_dim});
  mx::array k = mx::slice(conv_out, mx::Shape{0, 0, cfg.key_dim},
                          mx::Shape{B, S, 2 * cfg.key_dim});
  mx::array v = mx::slice(conv_out, mx::Shape{0, 0, 2 * cfg.key_dim},
                          mx::Shape{B, S, conv_dim});
  q = mx::reshape(q, mx::Shape{B, S, Hk, Dk});
  k = mx::reshape(k, mx::Shape{B, S, Hk, Dk});
  v = mx::reshape(v, mx::Shape{B, S, Hv, Dv});

  // 6. Qwen4Exp _normalize_qk (L2 + query scale), f32.
  auto qk = gdn_normalize_qk(q, k);

  // 7. g, beta and the full-sequence fused delta kernel. Hv value heads map
  //    onto Hk key heads inside the kernel (hk_idx = hv_idx / (Hv/Hk)), so
  //    q/k stay at Hk heads exactly as in production (no explicit repeat).
  auto gb = compute_g_beta(w.A_log, a, b, w.dt_bias);
  DeltaStepOutput step = run_delta_kernel(qk.first, qk.second, v, gb.first,
                                          gb.second, cache.recurrent_state,
                                          mask);

  // 8. Gated RMSNorm with the z projection, then out_proj over all S tokens.
  mx::array normed = rms_norm_gated(step.y, w.norm_weight, z, cfg.eps, sigmoid);
  mx::array flat = mx::reshape(normed, mx::Shape{B, S, cfg.value_dim});
  mx::array out = mx::matmul(flat, mx::transpose(w.out_proj));  // [B,S,H]

  return {out, new_conv_state, step.state};
}

}  // namespace veystrix::engine
