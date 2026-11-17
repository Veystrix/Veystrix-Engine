#include "prefill_batched.hpp"

#include <mlx/fast.h>
#include <mlx/mlx.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qsa.hpp"               // qsa_rms_norm
#include "rope_fused_apply.hpp"  // rope_apply_fused (bit-exact fused Metal MRoPE)

namespace veystrix::engine {

namespace {

// --------------------------------------------------------------------------- //
// Dense (non-sparse) full-attention branch, replicated VERBATIM from
// tools/real_checkpoint_generate.cpp:501-551 (full_attention_dense).  The QSA
// indexer returns None for <2048 past tokens, so production falls back to
// dense causal SDPA; this is that fallback, bit-exact for arbitrary S.
// --------------------------------------------------------------------------- //
FullAttentionOutput full_attention_dense_batched(
    const mx::array& x, const FullAttentionWeights& w,
    const FullAttentionConfig& cfg, FullAttentionCache& cache,
    const mx::array& position_ids) {
  const int B = static_cast<int>(x.shape(0));
  const int S = static_cast<int>(x.shape(1));
  const int Hq = cfg.n_heads, Hkv = cfg.kv_heads, D = cfg.head_dim;

  mx::array qproj = linear(x, w.q_w);
  mx::array kproj = linear(x, w.k_w);
  mx::array vproj = linear(x, w.v_w);

  mx::array q4 = mx::reshape(qproj, mx::Shape{B, S, Hq, 2 * D});
  std::vector<mx::array> halves = mx::split(q4, 2, -1);
  mx::array q = halves[0];
  mx::array gate = mx::reshape(halves[1], mx::Shape{B, S, Hq * D});

  // Production patches ALL Qwen4ExpRMSNorm (incl. attention q_norm/k_norm) to
  // one-centered gains: gain = weight (NO +1.0). qsa_rms_norm adds +1.0 (the
  // unpatched vector convention), so use the one-centered form here.
  auto qnorm = [&](const mx::array& a, const mx::array& gain) {
    auto d = a.dtype();
    mx::array y = mx::astype(a, mx::float32);
    y = y * mx::rsqrt(mx::mean(mx::square(y), -1, true) + cfg.eps);
    y = y * mx::astype(gain, mx::float32);
    return mx::astype(y, d);
  };
  q = mx::transpose(qnorm(q, w.q_norm_w), {0, 2, 1, 3});
  mx::array k = mx::transpose(
      qnorm(mx::reshape(kproj, mx::Shape{B, S, Hkv, D}), w.k_norm_w),
      {0, 2, 1, 3});
  mx::array v = mx::transpose(
      mx::reshape(vproj, mx::Shape{B, S, Hkv, D}), {0, 2, 1, 3});

  q = rope_apply_fused(q, position_ids, cfg.rope);
  k = rope_apply_fused(k, position_ids, cfg.rope);

  if (!cache.keys.has_value()) {
    cache.keys = k;
    cache.values = v;
  } else {
    cache.keys = mx::concatenate({*cache.keys, k}, 2);
    cache.values = mx::concatenate({*cache.values, v}, 2);
  }

  const float scale = static_cast<float>(std::pow(static_cast<double>(D), -0.5));
  mx::array out = mx::fast::scaled_dot_product_attention(
      q, *cache.keys, *cache.values, scale, /*mask_mode=*/"causal",
      /*mask_arr=*/std::nullopt, /*sinks=*/std::nullopt);
  out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), mx::Shape{B, S, Hq * D});
  out = linear(out * mx::sigmoid(gate), w.o_w);

  cache.offset += S;
  // Placeholder mask: dense (indexer=None) fallback produces no sparse mask.
  std::vector<int32_t> dummy{0};
  mx::array mask(dummy.begin(), mx::Shape{1}, mx::int32);
  return {out, mask};
}

// --------------------------------------------------------------------------- //
// Qwen4ExpGatedResidual with use_combine=False (final head).  Replicated from
// real_checkpoint_generate.cpp:814-825, returning only `mixed_input` [B,S,H].
// --------------------------------------------------------------------------- //
mx::array final_mixed_no_combine(const mx::array& hidden,
                                 const mx::array& norm_w, const Linear& down_w,
                                 const Linear& up_w, int hc, int H, float eps) {
  const int hc_hidden = hc * H;
  mx::Shape gs{static_cast<int>(hidden.shape(0)),
               static_cast<int>(hidden.shape(1)), hc, H};
  mx::array w2d = mx::reshape(norm_w, mx::Shape{hc, H});
  mx::array yg = mx::astype(mx::reshape(hidden, gs), mx::float32);
  yg = yg * mx::rsqrt(mx::mean(mx::square(yg), -1, true) + eps);
  yg = yg * mx::astype(w2d, mx::float32);
  mx::array normed = mx::astype(mx::reshape(yg, hidden.shape()), hidden.dtype());
  mx::array low = linear(normed, down_w) / static_cast<float>(hc);
  mx::array mix = mx::sigmoid(linear(low * mx::sigmoid(low), up_w));
  return mx::mean(mx::reshape(mix, gs) * mx::reshape(normed, gs), -2);
}

inline mx::array inject_batched(const mx::array& hidden,
                                const mx::array& branch,
                                const mx::array& inj) {
  mx::array b4 = mx::expand_dims(branch, -2);
  mx::array i4 = mx::expand_dims(inj, -1);
  return hidden + mx::reshape(b4 * i4, hidden.shape());
}

}  // namespace

mx::array run_prefill_batched(
    const mx::array& input_ids,
    const std::function<mx::array(const mx::array&)>& embed,
    std::vector<BatchedLayer>& layers, const BatchedHead& head,
    const DecoderConfig& cfg, const FullAttentionConfig& fa,
    std::vector<DecoderFullLayerCache>& caches, int group_size, int bits,
    BatchedPrefillDebug* debug) {
  const int B = static_cast<int>(input_ids.shape(0));
  const int S = static_cast<int>(input_ids.shape(1));
  if (B != 1)
    throw std::runtime_error("run_prefill_batched: currently B=1 only");
  const int hc = cfg.hc_count, H = cfg.hidden_size;
  if (static_cast<int>(layers.size()) != cfg.n_layers ||
      static_cast<int>(caches.size()) != cfg.n_layers)
    throw std::runtime_error("run_prefill_batched: layers/caches size mismatch");

  // 1. embed [1,S,H] -> tile to [1,S,hc*H], exactly like
  //    Qwen4ExpModel.__call__ (language.py:804-807) and the tool's run_step.
  mx::array hidden = mx::tile(embed(input_ids), std::vector<int>{1, 1, hc});

  for (int l = 0; l < cfg.n_layers; ++l) {
    BatchedLayer& L = layers[l];

    // 2. PLE (upstream of the attention block; S>1, no mask for one prompt).
    //    PLE is intentionally NOT exercised by the synthetic oracle (the tiny
    //    config has no PLE layer); it is wired here for the real stack.
    if (L.ple) {
      mx::array ids64 = mx::astype(input_ids, mx::int64);
      hidden = hidden + ple_forward(hidden, ids64, *L.ple, L.ple_cache, nullptr);
    }

    // 3. attention hyper-connection (gated residual with combine).
    GatedResidualOutput ar =
        gated_residual(hidden, L.attn_norm_w, L.attn_down_w, L.attn_up_w,
                       L.attn_inject_w, hc, H, cfg.eps);

    // 4. attention branch over ALL S tokens.
    mx::array branch = [&]() -> mx::array {
      if (L.kind == AttnKind::Linear) {
        if (!L.gdn) throw std::runtime_error("batched layer: missing GDN weights");
        GDNLayerOutput g = gdn_prefill(ar.mixed_input, *L.gdn,
                                       caches[l].base.gdn, cfg.gdn, nullptr);
        caches[l].base.gdn = GDNCacheState(g.conv_state, g.recurrent_state);
        return g.output;
      }
      if (!L.full)
        throw std::runtime_error("batched layer: missing full-attn weights");
      std::vector<int32_t> pos(S);
      for (int t = 0; t < S; ++t) pos[t] = caches[l].full.offset + t;
      mx::array position_ids(pos.data(), mx::Shape{1, S}, mx::int32);
      return full_attention_dense_batched(ar.mixed_input, *L.full, fa,
                                          caches[l].full, position_ids)
          .output;
    }();

    // 5. hyper-connection inject.
    hidden = inject_batched(hidden, branch, ar.injection_weights);

    // 6. MLP hyper-connection.
    GatedResidualOutput mr =
        gated_residual(hidden, L.mlp_norm_w, L.mlp_down_w, L.mlp_up_w,
                       L.mlp_inject_w, hc, H, cfg.eps);

    // 7. router over [S,H].
    mx::array m2d = mx::reshape(mr.mixed_input, mx::Shape{S, H});
    mx::array gate_logits = linear(m2d, L.gate_w);
    RouterOutput routes = router(gate_logits, L.top_k, 0.0f);

    // 8. routed experts via the injected callback -> [S,H].
    if (!L.moe_forward)
      throw std::runtime_error("batched layer: missing MoE forward callback");
    mx::array moe2d = L.moe_forward(m2d, routes.indices, routes.scores);

    // 9. non-zero dense shared expert: sg * s, added to the routed output.
    mx::array sgl = linear(m2d, L.shared_glw);
    mx::array sg = mx::sigmoid(sgl);
    mx::array g = linear(m2d, L.shared_gw);
    mx::array u = linear(m2d, L.shared_uw);
    mx::array s = linear(swiglu(g, u), L.shared_dw);
    moe2d = moe2d + sg * s;

    // 10. MLP inject: reshape [S,H] -> [1,S,H].
    mx::array moe = mx::reshape(moe2d, mx::Shape{1, S, H});
    hidden = inject_batched(hidden, moe, mr.injection_weights);

    if (debug) {
      mx::array last = mx::reshape(
          mx::slice(hidden, mx::Shape{0, S - 1, 0}, mx::Shape{1, S, hc * H}),
          mx::Shape{1, hc * H});
      debug->hidden_after_layer.push_back(last);
      mx::array blast = mx::reshape(
          mx::slice(branch, mx::Shape{0, S - 1, 0}, mx::Shape{1, S, H}),
          mx::Shape{1, H});
      debug->branch.push_back(blast);
    }
  }

  // 11. final head: gated residual (use_combine=False) -> last token -> lm_head.
  mx::array mixed = final_mixed_no_combine(hidden, head.norm_w, head.down_w,
                                           head.up_w, hc, H, cfg.eps);
  mx::array last = mx::reshape(
      mx::slice(mixed, mx::Shape{0, S - 1, 0}, mx::Shape{1, S, H}),
      mx::Shape{1, H});

  Linear lm;
  if (head.lm_quant) {
    const BatchedLMTriplet& q = *head.lm_quant;
    lm = Linear::quantized(q.w, q.s, q.b, group_size, bits);
  } else if (head.lm_dense) {
    lm = *head.lm_dense;
  } else {
    throw std::runtime_error("batched head: no lm_head weights");
  }
  return mx::astype(linear(last, lm), mx::float32);
}

}  // namespace veystrix::engine
