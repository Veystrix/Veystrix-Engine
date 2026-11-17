#include "parity_gaps.hpp"

#include <mlx/mlx.h>

#include <cmath>
#include <utility>

namespace veystrix::engine {

namespace {

// Wrap a [tokens, H] activation with the two singleton dims gathered_qmm
// expects (matching engine.cpp::moe_block: x4 = expand_dims(x, {-2, -3})).
mx::array as_gather_input(const mx::array& x) {
  return mx::expand_dims(x, std::vector<int>{-2, -3});
}

// A single "expert" route (index 0) of shape [tokens, 1]. One shared expert
// means every token gathers the same (only) weight matrix.
mx::array zero_route(int tokens) {
  return mx::zeros(mx::Shape{tokens, 1}, mx::int32);
}

// One Q4 affine GLU projection of the shared expert: returns [tokens, H].
mx::array shared_q4_glu(const mx::array& x, const ExpertWeights& w,
                        const mx::array& route, const SharedQuantConfig& cfg) {
  mx::array x4 = as_gather_input(x);
  mx::array g = q4_projection(x4, w.gate, route, cfg.group_size, cfg.bits);
  mx::array u = q4_projection(x4, w.up, route, cfg.group_size, cfg.bits);
  mx::array a = swiglu(g, u);  // [t, 1, 1, inter]
  mx::array d = q4_projection(a, w.down, route, cfg.group_size, cfg.bits);
  d = mx::squeeze(d, -2);  // [t, 1, H]
  return mx::squeeze(d, 1);  // [t, H]
}

// The DENSE shared expert, kept for the `shared_quantized == false` case.
// Mirrors decoder_full.cpp::moe_block_with_shared's GLU exactly.
mx::array shared_dense_glu(const mx::array& x, const Linear& gate_w,
                           const Linear& up_w, const Linear& down_w) {
  mx::array g = linear(x, gate_w);
  mx::array u = linear(x, up_w);
  return linear(swiglu(g, u), down_w);
}

}  // namespace

mx::array moe_block_shared_quantized(
    const mx::array& x, const mx::array& route, const mx::array& scores,
    const ExpertWeights& routed_weights,
    const ExpertWeights& shared_q4_weights,
    const Linear& shared_gate_logits_w, const SharedQuantConfig& config) {
  // Routed experts: the already bit-verified Q4 gather_qmm MoE block.
  mx::array routed = moe_block(x, route, scores, routed_weights);

  int tokens = static_cast<int>(x.shape(0));
  // Dense fallback: the caller supplies dequantized dense gate/up/down weights
  // through the same ExpertWeights struct (weight is then a dense [out, in]
  // array; scales/biases are ignored).
  mx::array s = config.shared_quantized
                    ? shared_q4_glu(x, shared_q4_weights,
                                    zero_route(tokens), config)
                    : shared_dense_glu(x, shared_q4_weights.gate.weight,
                                       shared_q4_weights.up.weight,
                                       shared_q4_weights.down.weight);

  // adaptive_topk._moe_call / Qwen3_5MoeSparseMoeBlock.__call__:
  //   return routed + sigmoid(shared_expert_gate(x)) * shared_expert(x)
  mx::array gate_logits = linear(x, shared_gate_logits_w);  // [t, 1]
  mx::array sg = mx::sigmoid(gate_logits);
  return routed + sg * s;
}

mx::array full_attention_dense_fallback(const mx::array& x,
                                        const FullAttentionWeights& w,
                                        const FullAttentionConfig& cfg,
                                        FullAttentionCache& cache,
                                        const mx::array& position_ids) {
  // Invariant check: this helper is only for the indexer-None case. Evaluate
  // the indexer on a COPY of the indexer cache so the dense KV cache (and the
  // caller's indexer state) is left untouched.
  {
    QSACache probe = cache.indexer;
    probe.offset = cache.offset;
    QSAFullConfig qcfg;
    qcfg.base.n_heads = cfg.n_idx_heads;
    qcfg.base.kv_heads = cfg.n_idx_kv;
    qcfg.base.head_dim = cfg.idx_head_dim;
    qcfg.base.compress_ratio = cfg.compress_ratio;
    qcfg.base.token_budget = cfg.token_budget;
    qcfg.base.block_topk = cfg.token_budget / cfg.compress_ratio;
    qcfg.base.eps = cfg.eps;
    qcfg.rope = cfg.rope;
    if (qsa_full_indexer(x, w.index_qk_w, w.index_q_norm_w, w.index_k_norm_w,
                         qcfg, probe, position_ids)
            .has_value()) {
      throw std::runtime_error(
          "full_attention_dense_fallback: QSA indexer produced a mask; the "
          "dense fallback is only valid when the indexer returns None");
    }
  }

  const int B = static_cast<int>(x.shape(0));
  const int S = static_cast<int>(x.shape(1));
  const int Hq = cfg.n_heads;
  const int Hkv = cfg.kv_heads;
  const int D = cfg.head_dim;

  // Dense q/k/v projections (Qwen3_5Attention._prepare_projected_qkv).
  mx::array qproj = linear(x, w.q_w);  // [B,S,Hq*D*2]
  mx::array kproj = linear(x, w.k_w);  // [B,S,Hkv*D]
  mx::array vproj = linear(x, w.v_w);

  mx::array q4 = mx::reshape(qproj, mx::Shape{B, S, Hq, 2 * D});
  std::vector<mx::array> halves = mx::split(q4, 2, -1);
  mx::array q = halves[0];     // [B, S, Hq, D]
  mx::array gate = halves[1];  // [B, S, Hq, D]
  gate = mx::reshape(gate, mx::Shape{B, S, Hq * D});

  // Qwen4ExpAttention uses the one-centered Qwen4ExpRMSNorm for q/k.
  q = mx::transpose(qsa_rms_norm(q, w.q_norm_w, cfg.eps), {0, 2, 1, 3});
  mx::array k = mx::transpose(
      qsa_rms_norm(mx::reshape(kproj, mx::Shape{B, S, Hkv, D}), w.k_norm_w,
                   cfg.eps),
      {0, 2, 1, 3});
  mx::array v = mx::transpose(
      mx::reshape(vproj, mx::Shape{B, S, Hkv, D}), {0, 2, 1, 3});

  // RoPE (eager branch; same primitive as the sparse path).
  q = qsa_apply_rope(q, position_ids, cfg.rope);
  k = qsa_apply_rope(k, position_ids, cfg.rope);

  // KV cache update_and_fetch: concatenate on axis 2.
  if (!cache.keys.has_value()) {
    cache.keys = k;
    cache.values = v;
  } else {
    cache.keys = mx::concatenate({*cache.keys, k}, 2);
    cache.values = mx::concatenate({*cache.values, v}, 2);
  }

  // No QSA mask. Python's mask is None for a single decode token (N == 1,
  // see mlx_vlm/models/base.py::create_attention_mask), so MLX fused SDPA is
  // called with no mask at all. For S > 1 the production mask is "causal".
  const float scale = static_cast<float>(std::pow(static_cast<double>(D), -0.5));
  mx::array out =
      (S == 1)
          ? mx::fast::scaled_dot_product_attention(
                q, *cache.keys, *cache.values, scale, /*mask_mode=*/"",
                /*mask_arr=*/std::nullopt, /*sinks=*/std::nullopt)
          : mx::fast::scaled_dot_product_attention(
                q, *cache.keys, *cache.values, scale, /*mask_mode=*/"causal",
                /*mask_arr=*/std::nullopt, /*sinks=*/std::nullopt);

  out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}),
                    mx::Shape{B, S, Hq * D});
  mx::array result = linear(out * mx::sigmoid(gate), w.o_w);

  cache.offset += S;  // KVCache.update_and_fetch advances offset by L
  return result;
}

}  // namespace veystrix::engine
