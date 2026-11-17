#include "engine.hpp"

#include <mlx/mlx.h>

namespace veystrix::engine {

RouterOutput router(const mx::array& logits, int top_k, float /*renorm_blend*/) {
  // Mirrors adaptive_topk._moe_call at threshold 1.0 (all k slots kept):
  //   gates = softmax(logits, -1, precise=True)
  //   inds  = argpartition(gates, kth=-k, axis=-1)[..., -k:]
  //   scores= take_along_axis(gates, inds, -1)
  //   (selected_mass == topk_mass, so renorm divides by topk_mass -> sums to 1)
  int ndim = static_cast<int>(logits.shape().size());
  int last = ndim - 1;
  mx::array gates = mx::softmax(logits, last, /*precise=*/true);
  mx::array inds_all = mx::argpartition(gates, /*kth=*/-top_k, last);
  // Slice [..., -k:] on the last axis (mx::slice takes Shape = vector<int>).
  mx::Shape start(ndim, 0), stop(ndim), strides(ndim, 1);
  for (int d = 0; d < ndim; ++d) stop[d] = inds_all.shape(d);
  start[last] = inds_all.shape(last) - top_k;
  mx::array inds = mx::slice(inds_all, start, stop, strides);
  mx::array scores = mx::take_along_axis(gates, inds, last);
  mx::array topk_mass = mx::sum(scores, last, /*keepdims=*/true);
  scores = scores / topk_mass;
  return RouterOutput{inds, scores};
}

mx::array q4_projection(const mx::array& x, const Q4Projection& p,
                        const mx::array& route, int group_size, int bits,
                        bool slot_input, bool sorted_indices) {
  // rhs_indices must be int32; Python passes uint32 indices from the route.
  mx::array rhs = mx::astype(route, mx::int32);
  return mx::gather_qmm(x, p.weight, p.scales, p.biases,
                        /*lhs_indices=*/std::nullopt,
                        /*rhs_indices=*/rhs,
                        /*transpose=*/true,
                        /*group_size=*/group_size,
                        /*bits=*/bits,
                        /*mode=*/"affine",
                        /*sorted_indices=*/sorted_indices);
}

mx::array swiglu(const mx::array& gate, const mx::array& up) {
  // Production: activation(gate, up) = up * (gate * sigmoid(gate)).
  // The Python engine uses mlx-vlm's SwiGLU; verify exact op order via parity.
  return up * (gate * mx::sigmoid(gate));
}

mx::array weighted_combine(const mx::array& expert_out, const mx::array& scores) {
  // (expert_out [t, slots, hidden] * scores[...,None]).sum(axis=-2)
  mx::array weighted = expert_out * mx::expand_dims(scores, -1);
  return mx::sum(weighted, /*axis=*/static_cast<int>(expert_out.shape().size()) - 2,
                 /*keepdims=*/false);
}

mx::array moe_block(const mx::array& x, const mx::array& route,
                    const mx::array& scores, const ExpertWeights& w) {
  // Match StreamingSwitchGLU.__call__: x[tokens,hidden] is expanded by
  // (-2,-3) before gather_qmm, so x is [tokens,1,1,hidden]. gather_qmm with
  // rhs_indices=route[tokens,k] yields [tokens,k,1,out]; squeeze(-2) then
  // gives [tokens,k,out] before the score combine over axis -2.
  mx::array x4 = mx::expand_dims(x, std::vector<int>{-2, -3});
  mx::array g = q4_projection(x4, w.gate, route, w.group_size, w.bits);
  mx::array u = q4_projection(x4, w.up, route, w.group_size, w.bits);
  mx::array a = swiglu(g, u);                       // [tokens,k,1,inter]
  mx::array d = q4_projection(a, w.down, route, w.group_size, w.bits);
  d = mx::squeeze(d, -2);                           // [tokens,k,hidden]
  return weighted_combine(d, scores);              // [tokens,hidden]
}

mx::array moe_block_from_logits(const mx::array& x, const mx::array& logits,
                                int top_k, float renorm_blend,
                                const ExpertWeights& w) {
  RouterOutput r = router(logits, top_k, renorm_blend);
  return moe_block(x, r.indices, r.scores, w);
}

mx::array rms_norm(const mx::array& x, const mx::array& weight, float eps) {
  // Mirrors patch_rmsnorm._rms_norm (one-centered, non-compiled path):
  // f32 cast, rsqrt(mean(square)+eps), multiply by f32 gain, cast back.
  auto dtype = x.dtype();
  mx::array y = mx::astype(x, mx::float32);
  mx::array mean = mx::mean(mx::square(y), -1, /*keepdims=*/true);
  y = y * mx::rsqrt(mean + eps);
  y = y * mx::astype(weight, mx::float32);
  y = mx::reshape(y, x.shape());
  return mx::astype(y, dtype);
}

mx::array sample_greedy(const mx::array& logits) {
  // veystrix.sampling.Sampler.__call__ (greedy): logits.reshape(-1),
  // argmax(axis=-1).reshape(1).astype(uint32).
  mx::array row = mx::reshape(logits, {-1});
  mx::array idx = mx::argmax(row, -1);
  return mx::astype(mx::reshape(idx, {1}), mx::uint32);
}

}  // namespace veystrix::engine
