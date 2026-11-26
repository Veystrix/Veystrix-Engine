// Full single Mamba-2 layer forward — NemotronHMamba2Mixer.__call__ DECODE
// step (S == 1), reproducing plan §2 steps 1-12 from the bit-exact primitives
// in engine.hpp / mamba_conv.h / mamba_kernel.h.
//
// Python reference (mlx_lm 0.32.2, nemotron_h.py:207-234):
//   in_proj -> split(gate, conv_input, dt) -> _conv (depthwise + bias + silu)
//   -> split(hidden_ssm, B, C) -> _ssm (fused ssm_kernel) -> reshape
//   -> MambaRMSNormGated(y, gate) -> out_proj.
//
// The caller supplies the two-cache state (conv_state [B,K-1,conv_dim],
// ssm_state [B,H,Dh,Ds] f32). Updated caches are returned with the output.
#pragma once
#include <mlx/mlx.h>

#include "engine.hpp"

namespace mx = mlx::core;

namespace veystrix::engine {

// Geometry of one Mamba-2 layer (mirrors the ModelArgs fields the mixer reads).
struct MambaConfig {
  int hidden_size;        // input / output width
  int num_heads;          // H
  int head_dim;           // Dh
  int n_groups;           // number of B/C groups
  int ssm_state_size;     // Ds
  int conv_kernel;        // K
  int intermediate_size;  // I = H * Dh
  int conv_dim;           // I + 2 * n_groups * Ds
  float eps;              // MambaRMSNormGated eps
  float time_step_min;    // compute_dt clip lower bound
  float time_step_max;    // compute_dt clip upper bound
};

// All weights of one Mamba-2 layer. Linear weights are [out, in] (forward is
// x @ W.T); the depthwise conv weight is the raw [conv_dim, K, 1] tensor.
struct MambaWeights {
  Linear in_proj;           // [I + conv_dim + H, hidden_size]
  mx::array conv1d_weight;  // [conv_dim, K, 1] (bf16)
  mx::array conv1d_bias;    // [conv_dim] (bf16; empty if use_conv_bias=False)
  mx::array A_log;          // [H] f32
  mx::array dt_bias;        // [H] f32
  mx::array D;              // [H] f32 (cast to hidden dtype in the layer)
  mx::array norm_weight;    // [I] (checkpoint-cast bf16)
  Linear out_proj;          // [hidden_size, I]

  MambaWeights() = delete;
  MambaWeights(Linear in, mx::array conv_w, mx::array conv_b, mx::array alog,
               mx::array dtb, mx::array d, mx::array nw, Linear out)
      : in_proj(std::move(in)), conv1d_weight(std::move(conv_w)),
        conv1d_bias(std::move(conv_b)), A_log(std::move(alog)),
        dt_bias(std::move(dtb)), D(std::move(d)), norm_weight(std::move(nw)),
        out_proj(std::move(out)) {}
};

// Decode cache carried between steps.
struct MambaCacheState {
  mx::array conv_state;   // [B, K-1, conv_dim] (activation dtype)
  mx::array ssm_state;    // [B, H, Dh, Ds] f32

  MambaCacheState() = delete;
  MambaCacheState(mx::array c, mx::array s)
      : conv_state(std::move(c)), ssm_state(std::move(s)) {}
};

struct MambaLayerOutput {
  mx::array output;       // [B, S, hidden_size]
  mx::array conv_state;   // new cache[0]  [B, K-1, conv_dim]
  mx::array ssm_state;    // new cache[1]  [B, H, Dh, Ds]
};

// One decode step. For S > 1 it iterates token-by-token through the same
// decode path (the Python prefill uses a different full-conv/chunked-scan
// path, so S > 1 is not bit-exact to prefill).
MambaLayerOutput mamba_layer_forward(const mx::array& x, const MambaWeights& w,
                                     const MambaCacheState& cache,
                                     const MambaConfig& cfg);

}  // namespace veystrix::engine
