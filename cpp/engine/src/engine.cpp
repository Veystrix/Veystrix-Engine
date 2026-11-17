#include "engine.hpp"

#include <mlx/mlx.h>

#include <cmath>
#include <cstdlib>

namespace veystrix::engine {

mx::array linear(const mx::array& x, const Linear& L) {
  mx::array out = [&]() -> mx::array {
    if (L.q) {
      const QPacked& q = *L.q;
      // Python QuantizedLinear.__call__:
      //   mx.quantized_matmul(x, w, scales, biases, transpose=True, gs, bits, mode)
      std::optional<mx::array> biases =
          q.biases.size() ? std::optional<mx::array>(q.biases) : std::nullopt;
      return mx::quantized_matmul(x, q.w, q.scales, biases, /*transpose=*/true,
                                  q.group_size, q.bits, "affine");
    } else if (L.dense) {
      // Original path, byte-for-byte identical to every existing ctest.
      return mx::matmul(x, mx::transpose(*L.dense));
    }
    throw std::runtime_error("linear(): neither dense nor q is set");
  }();
  if (L.bias) out = out + *L.bias;
  return out;
}

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
  // Production: activation(gate, up) = nn.silu(gate) * up. Python's nn.silu is
  // a compiled/fused kernel; use hc_silu so the same fused kernel is emitted
  // with VEYSTRIX_CPP_COMPILED_SILU=1 (avoids 1-ULP rounding differences).
  return up * hc_silu(gate);
}

mx::array hc_silu(const mx::array& x) {
  // Python's nn.silu is `@partial(mx.compile, shapeless=True) x*sigmoid(x)` —
  // one fused JIT kernel. Compile the identical graph by default so the C++
  // engine is bit-exact to the Python reference out of the box (real-checkpoint
  // gate (b)). Opt out with VEYSTRIX_CPP_UNFUSED_SILU only for legacy synthetic
  // golden vectors captured before the compiled-silu convention.
  static const bool use_unfused =
      std::getenv("VEYSTRIX_CPP_UNFUSED_SILU") != nullptr;
  if (use_unfused) return x * mx::sigmoid(x);
  static auto fn = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        return {in[0] * mx::sigmoid(in[0])};
      },
      /*shapeless=*/true);
  mx::array out = fn({x})[0];
  // Force the fused silu kernel to complete before returning so the shared
  // shapeless compiled graph is never invoked concurrently by MLX's lazy
  // scheduler across the many layers (a nondeterminism source on gate (a)).
  mx::eval(out);
  return out;
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

mx::array moe_block_relu2(const mx::array& x, const mx::array& route,
                          const mx::array& scores, const Q4Projection& up,
                          const Q4Projection& down) {
  // Nemotron relu2 MLP: y = W_down . relu2(W_up . x). Single up projection,
  // no SwiGLU gate. Scales/biases are Q4 affine (G64 for Lightning).
  mx::array x4 = mx::expand_dims(x, std::vector<int>{-2, -3});
  mx::array u = q4_projection(x4, up, route, 64, 4);
  mx::array a = mx::square(mx::maximum(u, mx::array(0.0f)));  // relu2
  mx::array d = q4_projection(a, down, route, 64, 4);
  d = mx::squeeze(d, -2);
  return weighted_combine(d, scores);
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

DeltaStepOutput gated_delta_step(const mx::array& q, const mx::array& k,
                                 const mx::array& v, const mx::array& g,
                                 const mx::array& beta, const mx::array& state) {
  // Mirrors _gated_delta_step_ops (scalar gate g of shape [B,H]).
  // state: [B,H,Dv,Dk]; k,q: [B,H,Dk]; v: [B,H,Dv].
  mx::array decay = mx::expand_dims(g, std::vector<int>{-1, -1});  // [B,H,1,1]
  mx::array new_state = state * decay;
  mx::array kv_mem = mx::sum(new_state * mx::expand_dims(k, -1), -1);  // [B,H,Dv]
  mx::array delta = (v - kv_mem) * mx::expand_dims(beta, -1);          // [B,H,Dv]
  new_state = new_state + mx::expand_dims(k, -1) * mx::expand_dims(delta, -1);
  mx::array y = mx::sum(new_state * mx::expand_dims(q, -1), -1);  // [B,H,Dv]
  return {mx::astype(y, q.dtype()), new_state};
}

std::pair<mx::array, mx::array> compute_g_beta(const mx::array& A_log,
                                               const mx::array& a,
                                               const mx::array& b,
                                               const mx::array& dt_bias) {
  // compute_g = exp(-exp(A_log) * softplus(a + dt_bias)); beta = sigmoid(b).
  // Python uses bf16 nn.softplus(a+dt_bias). A f32 zero in logaddexp would
  // promote bf16 -> f32 (verified: differs by up to 8e-3); use a same-dtype
  // zero so softplus stays in the input dtype. Dtype-generic (f32 path unchanged).
  mx::array A = mx::exp(mx::astype(A_log, mx::float32));
  mx::array ad = a + dt_bias;
  mx::array sp = mx::logaddexp(ad, mx::astype(mx::array(0.0f), ad.dtype()));
  mx::array g = mx::exp(-A * sp);
  return {g, mx::sigmoid(b)};
}

GatedResidualOutput gated_residual(
    const mx::array& hyper_input,
    const mx::array& norm_weight,
    const Linear& down_w, const Linear& up_w,
    const Linear& inject_w,
    int hc_count, int hidden_size, float eps) {
  // Qwen4ExpGatedResidual.__call__ (one-centered hc_norm via fast rms_norm).
  // norm_weight gain is [hc_count, hidden_size]; grouped norm == rms_norm over
  // the last axis of the flattened [..., hc_count*hidden_size] with that 2D gain.
  int hc_hidden = hc_count * hidden_size;
  mx::array w2d = mx::reshape(norm_weight, mx::Shape{hc_count, hidden_size});
  // Manual grouped RMSNorm, one-centered gain, matching patch_rmsnorm._rms_norm.
  // Reshape [..., hc_hidden] -> [..., hc_count, hidden_size], normalize last
  // axis, multiply by 2D gain, reshape back.
  mx::Shape gs(hyper_input.shape().begin(), hyper_input.shape().end() - 1);
  gs.push_back(hc_count);
  gs.push_back(hidden_size);
  mx::array yg = mx::astype(mx::reshape(hyper_input, gs), mx::float32);
  yg = yg * mx::rsqrt(mx::mean(mx::square(yg), -1, /*keepdims=*/true) + eps);
  yg = yg * mx::astype(w2d, mx::float32);
  mx::array normed = mx::astype(mx::reshape(yg, hyper_input.shape()),
                                hyper_input.dtype());
  // Linear weights are stored [out, in]; compute x @ W.T (nn.Linear).
  // mix = sigmoid(up(silu(down(normed)/hc_count)))
  // Divide by an int32 ARRAY (not a float scalar): C++ array/float upcasts
  // bf16 -> f32, while Python's `x / self.hc_count` (int) keeps bf16. This is
  // essential for bit-exactness on the real bf16 checkpoint.
  mx::array hc_arr = mx::array(hc_count);  // int32 scalar array
  mx::array low = linear(normed, down_w) / hc_arr;
  mx::array silu_low = hc_silu(low);
  mx::array mix = mx::sigmoid(linear(silu_low, up_w));
  mx::array mixg = mx::reshape(mix, gs);
  mx::array normg = mx::reshape(normed, gs);
  mx::array mixed_input = mx::mean(mixg * normg, -2);
  mx::array two = mx::array(2);  // int32 scalar: keeps bf16
  mx::array inj = two * mx::sigmoid(linear(normed, inject_w) / hc_arr);
  return {mixed_input, inj};
}

mx::array final_head(const mx::array& hidden,
                     const mx::array& norm_weight,
                     const Linear& down_w, const Linear& up_w,
                     const Linear& lm_head_w,
                     int hc_count, int hidden_size, float eps) {
  // gated_residual with use_combine=False -> mixed_input only. Reuse the same
  // math but skip the injection-weight projection (no block_inject_weight).
  int hc_hidden = hc_count * hidden_size;
  mx::array w2d = mx::reshape(norm_weight, mx::Shape{hc_count, hidden_size});
  mx::Shape gs(hidden.shape().begin(), hidden.shape().end() - 1);
  gs.push_back(hc_count);
  gs.push_back(hidden_size);
  mx::array yg = mx::astype(mx::reshape(hidden, gs), mx::float32);
  yg = yg * mx::rsqrt(mx::mean(mx::square(yg), -1, /*keepdims=*/true) + eps);
  yg = yg * mx::astype(w2d, mx::float32);
  mx::array normed = mx::astype(mx::reshape(yg, hidden.shape()), hidden.dtype());
  mx::array hc_arr = mx::array(hc_count);  // int32 array: keeps bf16
  mx::array low = linear(normed, down_w) / hc_arr;
  mx::array mix = mx::sigmoid(linear(hc_silu(low), up_w));
  mx::array mixed_input = mx::mean(mx::reshape(mix, gs) * mx::reshape(normed, gs), -2);
  // lm_head: nn.Linear(hidden, vocab, bias=False) -> matmul(x, W.T).
  return linear(mixed_input, lm_head_w);
}

mx::array rms_norm_gated(const mx::array& x, const mx::array& weight,
                         const mx::array& gate, float eps, bool sigmoid_act) {
  auto dtype = x.dtype();
  mx::array y = mx::fast::rms_norm(x, weight, eps);
  y = mx::astype(y, mx::float32);
  mx::array g = mx::astype(gate, mx::float32);
  // Python: `gate = nn.silu(gate)` (compiled/fused) in the non-sigmoid case.
  mx::array act = sigmoid_act ? mx::sigmoid(g) : hc_silu(g);
  return mx::astype(y * act, dtype);
}

std::pair<mx::array, mx::array> gdn_normalize_qk(const mx::array& q, const mx::array& k) {
  // Python's _normalize_qk keeps the input dtype (bf16 on the real checkpoint):
  //   scale = q.shape[-1] ** -0.5
  //   q = q * rsqrt(sum(square(q), -1, keepdims=True) + 1e-6)
  //   k = k * rsqrt(sum(square(k), -1, keepdims=True) + 1e-6)
  //   return q * scale, k
  // Casting to f32 or using f32 scalars promotes and diverges (verified up to
  // 5e-3). Use same-dtype scalars so the whole chain stays in the input dtype.
  // Dtype-generic: f32 synthetic inputs are unchanged.
  int Dk = q.shape().back();
  double scale = std::pow(static_cast<double>(Dk), -0.5);
  mx::array eps = mx::astype(mx::array(1e-6f), q.dtype());
  mx::array scale_arr = mx::astype(mx::array(static_cast<float>(scale)), q.dtype());
  mx::array qn = q * mx::rsqrt(mx::sum(mx::square(q), -1, /*keepdims=*/true) + eps);
  mx::array kn = k * mx::rsqrt(mx::sum(mx::square(k), -1, /*keepdims=*/true) + eps);
  return {qn * scale_arr, kn};
}

mx::array gdn_gate_sigmoid(const mx::array& y, const mx::array& gate) {
  return mx::astype(y, mx::float32) * mx::sigmoid(mx::astype(gate, mx::float32));
}

mx::array gdn_gate_silu(const mx::array& y, const mx::array& gate) {
  mx::array g = mx::astype(gate, mx::float32);
  mx::array silu = g * mx::sigmoid(g);
  return mx::astype(y, mx::float32) * silu;
}

mx::array sample_greedy(const mx::array& logits) {
  // veystrix.sampling.Sampler.__call__ (greedy): logits.reshape(-1),
  // argmax(axis=-1).reshape(1).astype(uint32).
  mx::array row = mx::reshape(logits, {-1});
  mx::array idx = mx::argmax(row, -1);
  return mx::astype(mx::reshape(idx, {1}), mx::uint32);
}

}  // namespace veystrix::engine
