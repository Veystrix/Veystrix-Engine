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
  // gate/up/down via gather_qmm with the SAME routed indices.
  mx::array g = q4_projection(x, w.gate, route, w.group_size, w.bits);
  mx::array u = q4_projection(x, w.up, route, w.group_size, w.bits);
  mx::array a = swiglu(g, u);
  // down projection consumes slot activations: [t, slots, inter].
  mx::array d = q4_projection(a, w.down, route, w.group_size, w.bits,
                              /*slot_input=*/true);
  return weighted_combine(d, scores);
}

mx::array moe_block_from_logits(const mx::array& x, const mx::array& logits,
                                int top_k, float renorm_blend,
                                const ExpertWeights& w) {
  RouterOutput r = router(logits, top_k, renorm_blend);
  return moe_block(x, r.indices, r.scores, w);
}

}  // namespace veystrix::engine
