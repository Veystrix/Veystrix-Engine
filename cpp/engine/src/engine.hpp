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

}  // namespace veystrix::engine
