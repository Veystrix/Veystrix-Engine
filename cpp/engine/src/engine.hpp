// Veystrix native decode engine (C++ port of the MoE decode compute path).
//
// Goal: bit-exact parity with the Python engine's per-token MoE block, using
// the SAME MLX 0.32.2 build so gather_qmm / softmax / argpartition reduce in
// the same order. Every function here mirrors a specific Python function:
//
//   router()        <-> adaptive_topk._moe_call  (softmax, argpartition, renorm)
//   q4_projection() <-> StreamingSwitchLinear.__call__ (mx.gather_qmm)
//   swiglu()        <-> activation in expert_cache (up * (gate*sigmoid(gate)))
//   weighted_combine() <-> metal_runtime.weighted_combine
//   moe_block()     <-> the composition, one MoE layer for one decode token
//
// Off by default; research port. Bit-exactness is enforced by tests that
// compare against vectors produced by the real Python path (tools/gen_vectors.py).
#pragma once
#include <array>
#include <utility>
#include <cstdint>
#include <optional>
#include <vector>

#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace veystrix::engine {

struct Q4Projection {
  mx::array weight;   // uint32 packed, shape (n_experts, out, in/8)
  mx::array scales;   // bf16/f16, shape (n_experts, out, in/group)
  mx::array biases;   // bf16/f16, same shape as scales
  Q4Projection() = delete;  // mx::array has no default constructor
  Q4Projection(mx::array w, mx::array s, mx::array b)
      : weight(std::move(w)), scales(std::move(s)), biases(std::move(b)) {}
};

struct ExpertWeights {
  Q4Projection gate;
  Q4Projection up;
  Q4Projection down;
  int group_size = 32;
  int bits = 4;
  ExpertWeights() = delete;
  ExpertWeights(Q4Projection g, Q4Projection u, Q4Projection d,
                int gs = 32, int b = 4)
      : gate(std::move(g)), up(std::move(u)), down(std::move(d)),
        group_size(gs), bits(b) {}
};

// Router: exact top-k of a softmax over logits, with the production renorm.
// Returns (indices [tokens, width], scores [tokens, width]) matching
// adaptive_topk._moe_call at threshold 1.0 (all k slots kept).
//   logits   : [tokens, n_experts] float
//   top_k    : number of routed experts
//   renorm_blend : _RENORM_BLEND from adaptive_topk (0 disables the blend).
struct RouterOutput {
  mx::array indices;  // int32 [tokens, k]
  mx::array scores;   // float  [tokens, k]
};
RouterOutput router(const mx::array& logits, int top_k, float renorm_blend);

// A single Q4 affine gather matmul: y[t,s,:] = W[route[t,s]] @ x[t,:].
// Mirrors mx.gather_qmm(x, w, scales, biases, rhs_indices=route,
//                       transpose=True, group_size, bits, mode="affine").
// x: [tokens, in] or [tokens, slots, in] when slot_input.
// route: [tokens, slots] uint32.
mx::array q4_projection(const mx::array& x, const Q4Projection& p,
                        const mx::array& route, int group_size, int bits,
                        bool slot_input = false, bool sorted_indices = false);

// SwiGLU activation: out = up * (gate * sigmoid(gate)).
mx::array swiglu(const mx::array& gate, const mx::array& up);

// Weighted combine of expert outputs: (out * scores[...,None]).sum(-2).
mx::array weighted_combine(const mx::array& expert_out, const mx::array& scores);

// Full MoE block for one decode step (routed experts only, no shared expert):
//   routes/scores are EXTERNAL (already computed by the router) so the same
//   function can serve both the Python-parity test and the real decode path.
// Returns [tokens, hidden] float output.
mx::array moe_block(const mx::array& x, const mx::array& route,
                    const mx::array& scores, const ExpertWeights& w);

// Convenience: compute router then moe_block (matches _moe_call at thr 1.0).
mx::array moe_block_from_logits(const mx::array& x, const mx::array& logits,
                                int top_k, float renorm_blend,
                                const ExpertWeights& w);

// Nemotron-style relu2 MoE block (no SwiGLU gate): up projection -> relu2 ->
// down projection -> score combine. Mirrors streamed_experts._gather / _expert.
mx::array moe_block_relu2(const mx::array& x, const mx::array& route,
                          const mx::array& scores, const Q4Projection& up,
                          const Q4Projection& down);

// RMSNorm with the production one-centered convention (patch_rmsnorm.py):
//   y = x.f32 * rsqrt(mean(x.f32^2, -1, keepdims) + eps) * weight.f32
// `weight` is the checkpoint gain (float32; one-centered means no +1).
mx::array rms_norm(const mx::array& x, const mx::array& weight, float eps);

// Greedy sampler: argmax over the last axis -> uint32 scalar-ish array,
// matching veystrix.sampling.Sampler.__call__ for greedy decoding.
mx::array sample_greedy(const mx::array& logits);

// GDN q/k L2 normalization (compile_gdn._normalize_qk):
//   scale = last_dim^-0.5; v = v*rsqrt(sum(v^2,-1,keepdims)+1e-6); return v*scale, k
std::pair<mx::array, mx::array> gdn_normalize_qk(const mx::array& q, const mx::array& k);

// GDN gate chains (compile_gdn._gate_sigmoid / _gate_silu):
//   f32(y) * sigmoid(f32(gate))  or  f32(y) * silu(f32(gate)).
mx::array gdn_gate_sigmoid(const mx::array& y, const mx::array& gate);
mx::array gdn_gate_silu(const mx::array& y, const mx::array& gate);

// Qwen4 gated residual / hyper-connection (Qwen4ExpGatedResidual.__call__).
// Returns (mixed_input, hyper_input, injection_weights). The hc_norm is applied
// with the one-centered convention via rms_norm with group_size (passed as the
// pre-shaped weight). Weights: down (hc_hidden -> lowrank), up (lowrank -> hc_hidden),
// block_inject (hc_hidden -> hc_count). No bias on any Linear.
struct GatedResidualOutput {
  mx::array mixed_input;
  mx::array injection_weights;
};
GatedResidualOutput gated_residual(
    const mx::array& hyper_input,
    const mx::array& norm_weight,   // hc_norm gain, shape [hc_hidden]
    const mx::array& down_w, const mx::array& up_w,
    const mx::array& inject_w,
    int hc_count, int hidden_size, float eps);

}  // namespace veystrix::engine
