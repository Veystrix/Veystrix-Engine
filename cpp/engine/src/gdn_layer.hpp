// Full single GDN (gated-delta-net) linear-attention layer forward — the
// Qwen4ExpGatedDeltaNet.__call__ DECODE step (S == 1), composed entirely from
// the already-verified bit-exact primitives in engine.hpp / gdn_conv.h.
//
// Python reference (mlx_vlm 0.32.2):
//   Qwen3_5GatedDeltaNet.__call__   (qwen3_5/language.py:1109-1196)
//   Qwen4ExpGatedDeltaNet          (qwen4_exp/language.py): overrides
//       _normalize_qk (L2 + query scale) and self.norm (RMSNormGated+sigmoid).
//   gated_delta_update / gated_delta_kernel (qwen3_5/gated_delta.py).
//
// Bit-exactness note: the Python layer is generic over the input dtype. With a
// bf16 input, _normalize_qk works in bf16 and the recurrent kernel sees bf16
// q/k; the verified C++ primitive gdn_normalize_qk is f32-only. This layer is
// therefore exercised/verified with an f32 input x (and f32 conv_state), in
// which case the whole post-projection path — conv, q/k norm, delta kernel,
// gated RMSNorm, out_proj — is f32 and every primitive is bit-exact. Linear /
// conv / out_proj weights remain bf16 (as in production); MLX promotes the
// matmul to f32 exactly as nn.Linear does. See gen_gdn_layer_vectors.py.
#pragma once
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace veystrix::engine {

enum class GDNActivation { Sigmoid, Silu };

// Geometry of one GDN layer. Mirrors the config fields consumed by
// Qwen3_5GatedDeltaNet.__init__.
struct GDNConfig {
  int hidden_size;   // H
  int key_dim;       // Hk * Dk
  int value_dim;     // Hv * Dv
  int num_k_heads;   // Hk
  int num_v_heads;   // Hv  (must be a multiple of Hk)
  int head_k_dim;    // Dk  (multiple of 32 for the Metal kernel)
  int head_v_dim;    // Dv
  int conv_kernel;   // K
  float eps;                                          // RMSNorm eps
  GDNActivation output_activation = GDNActivation::Sigmoid;  // Qwen4Exp default
};

// All weights of one GDN layer, stored exactly as nn.Linear / nn.Conv1d store
// them: Linear weights are [out, in] (the forward computes x @ W.T), the
// depthwise conv weight is the raw [conv_dim, K, 1] tensor.
struct GDNWeights {
  mx::array in_proj_qkv;  // [conv_dim, hidden_size]
  mx::array in_proj_z;    // [value_dim, hidden_size]
  mx::array in_proj_b;    // [Hv, hidden_size]
  mx::array in_proj_a;    // [Hv, hidden_size]
  mx::array conv1d_weight;// [conv_dim, K, 1] (raw nn.Conv1d.weight, half dtype)
  mx::array A_log;        // [Hv] float32
  mx::array dt_bias;      // [Hv] float32
  mx::array norm_weight;  // [head_v_dim] float32 (Qwen4ExpRMSNormGated gain)
  mx::array out_proj;     // [hidden_size, value_dim]

  GDNWeights() = delete;
  GDNWeights(mx::array qkv, mx::array z, mx::array b, mx::array a,
             mx::array conv_w, mx::array alog, mx::array dtb,
             mx::array nw, mx::array out)
      : in_proj_qkv(std::move(qkv)), in_proj_z(std::move(z)),
        in_proj_b(std::move(b)), in_proj_a(std::move(a)),
        conv1d_weight(std::move(conv_w)), A_log(std::move(alog)),
        dt_bias(std::move(dtb)), norm_weight(std::move(nw)),
        out_proj(std::move(out)) {}
};

// Decode cache carried between steps: the tail of the conv input and the
// recurrent state. Both are returned updated by gdn_layer_forward.
struct GDNCacheState {
  mx::array conv_state;       // [B, K-1, conv_dim]
  mx::array recurrent_state;  // [B, Hv, Dv, Dk] float32

  GDNCacheState() = delete;
  GDNCacheState(mx::array c, mx::array r)
      : conv_state(std::move(c)), recurrent_state(std::move(r)) {}
};

struct GDNLayerOutput {
  mx::array output;           // [B, S, hidden_size]
  mx::array conv_state;       // [B, K-1, conv_dim]  (new cache[0])
  mx::array recurrent_state;  // [B, Hv, Dv, Dk]     (new cache[1])
};

// One decode step of Qwen4ExpGatedDeltaNet.__call__. For S > 1 it iterates
// token-by-token through the same decode path (the Python prefill uses a
// different full-conv + chunked-scan path, so S > 1 is not bit-exact to that).
// Returns the layer output together with the updated cache.
GDNLayerOutput gdn_layer_forward(const mx::array& x, const GDNWeights& weights,
                                 const GDNCacheState& cache,
                                 const GDNConfig& cfg);

}  // namespace veystrix::engine
