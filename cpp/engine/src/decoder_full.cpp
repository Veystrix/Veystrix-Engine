#include "decoder_full.hpp"

#include <mlx/mlx.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace veystrix::engine {

namespace {

// Qwen4ExpDecoderLayer injection (same as decoder.cpp):
//   hidden = hyper_input + (branch[...,None,:] * inj[...,None]).reshape(...)
mx::array inject_branch(const mx::array& hidden, const mx::array& branch,
                        const mx::array& inj) {
  mx::array b4 = mx::expand_dims(branch, -2);  // [B, S, 1, H]
  mx::array i4 = mx::expand_dims(inj, -1);     // [B, S, hc_count, 1]
  return hidden + mx::reshape(b4 * i4, hidden.shape());
}

// Convert FullAttentionConfig -> QSAFullConfig for the indexer. The indexer and
// the dense attention share the same rotary embedding instance in production.
QSAFullConfig make_qsa_config(const FullAttentionConfig& cfg) {
  QSAFullConfig q;
  q.base.n_heads = cfg.n_idx_heads;
  q.base.kv_heads = cfg.n_idx_kv;
  q.base.head_dim = cfg.idx_head_dim;
  q.base.compress_ratio = cfg.compress_ratio;
  q.base.token_budget = cfg.token_budget;
  q.base.block_topk = cfg.token_budget / cfg.compress_ratio;
  q.base.eps = cfg.eps;
  q.rope = cfg.rope;
  return q;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Routed MoE + non-zero dense shared expert
// ---------------------------------------------------------------------------

mx::array moe_block_with_shared(const mx::array& x, const mx::array& route,
                                const mx::array& scores,
                                const ExpertWeights& routed,
                                const SharedExpertWeights& shared) {
  // Routed experts: the already-verified Q4 gather_qmm MoE block.
  mx::array y = moe_block(x, route, scores, routed);

  // Qwen3_5MoeSparseMoeBlock / adaptive_topk._moe_call:
  //   shared      = self.shared_expert(x)                    dense GLU MLP
  //   shared_gate = sigmoid(self.shared_expert_gate(x))      [tokens, 1]
  //   return y + shared_gate * shared
  mx::array gate_logits =
      mx::matmul(x, mx::transpose(shared.gate_logits_w));      // [t, 1]
  mx::array shared_gate = mx::sigmoid(gate_logits);

  mx::array g = mx::matmul(x, mx::transpose(shared.gate_w));   // [t, inter]
  mx::array u = mx::matmul(x, mx::transpose(shared.up_w));     // [t, inter]
  mx::array act = swiglu(g, u);                                // explicit silu
  mx::array s = mx::matmul(act, mx::transpose(shared.down_w)); // [t, H]

  return y + shared_gate * s;
}

// ---------------------------------------------------------------------------
// 2. Full-attention (QSA sparse) branch
// ---------------------------------------------------------------------------

FullAttentionOutput full_attention_forward(
    const mx::array& x, const FullAttentionWeights& w,
    const FullAttentionConfig& cfg, FullAttentionCache& cache,
    const mx::array& position_ids) {
  const int B = static_cast<int>(x.shape(0));
  const int S = static_cast<int>(x.shape(1));
  const int Hq = cfg.n_heads;
  const int Hkv = cfg.kv_heads;
  const int D = cfg.head_dim;

  // ---- QSA indexer -> sparse mask ---------------------------------------
  // The indexer reads past_len from cache.offset; keep it in sync (the dense
  // KV cache offset is the authoritative one).
  cache.indexer.offset = cache.offset;
  QSAFullConfig qcfg = make_qsa_config(cfg);
  std::optional<mx::array> mask_opt = qsa_full_indexer(
      x, w.index_qk_w, w.index_q_norm_w, w.index_k_norm_w, qcfg,
      cache.indexer, position_ids);
  if (!mask_opt.has_value()) {
    // Production returns None (dense attention, no sparse mask). A full layer
    // in a decode step is only composed here when the indexer yields a mask;
    // callers must prepopulate the indexer cache past the top-k block budget.
    throw std::runtime_error(
        "full_attention_forward: QSA indexer returned None (dense path not "
        "composed); prepopulate the indexer cache beyond the block budget");
  }
  mx::array mask = *mask_opt;  // [B, 1, S, key_len]

  // ---- dense q/k/v projections (Qwen3_5Attention._prepare_projected_qkv) -
  mx::array qproj = mx::matmul(x, mx::transpose(w.q_w));  // [B,S,Hq*D*2]
  mx::array kproj = mx::matmul(x, mx::transpose(w.k_w));  // [B,S,Hkv*D]
  mx::array vproj = mx::matmul(x, mx::transpose(w.v_w));

  mx::array q4 = mx::reshape(qproj, mx::Shape{B, S, Hq, 2 * D});
  std::vector<mx::array> halves = mx::split(q4, 2, -1);
  mx::array q = halves[0];        // [B, S, Hq, D]
  mx::array gate = halves[1];     // [B, S, Hq, D]
  gate = mx::reshape(gate, mx::Shape{B, S, Hq * D});

  // Qwen4ExpAttention overrides q_norm/k_norm with the ONE-CENTERED
  // Qwen4ExpRMSNorm (qsa_rms_norm), not nn.RMSNorm.
  q = mx::transpose(qsa_rms_norm(q, w.q_norm_w, cfg.eps), {0, 2, 1, 3});
  mx::array k = mx::transpose(
      qsa_rms_norm(mx::reshape(kproj, mx::Shape{B, S, Hkv, D}), w.k_norm_w,
                   cfg.eps),
      {0, 2, 1, 3});
  mx::array v = mx::transpose(
      mx::reshape(vproj, mx::Shape{B, S, Hkv, D}), {0, 2, 1, 3});

  // ---- RoPE (eager branch; see decoder_full_port.md for the fused caveat) -
  q = qsa_apply_rope(q, position_ids, cfg.rope);
  k = qsa_apply_rope(k, position_ids, cfg.rope);

  // ---- KV cache update_and_fetch: concatenate on axis 2 ------------------
  if (!cache.keys.has_value()) {
    cache.keys = k;
    cache.values = v;
  } else {
    cache.keys = mx::concatenate({*cache.keys, k}, 2);
    cache.values = mx::concatenate({*cache.values, v}, 2);
  }
  const mx::array& full_k = *cache.keys;
  const mx::array& full_v = *cache.values;

  // Python _prepare_projected_qkv truncates the mask to kv_seq_len:
  //   kv_seq_len = S + cache.offset + 1  (position_ids supplied explicitly)
  const int kv_seq_len = S + cache.offset + 1;
  const int mask_w = static_cast<int>(mask.shape(3));
  mx::array use_mask = mask;
  if (kv_seq_len < mask_w) {
    use_mask = mx::slice(mask, {0, 0, 0, 0}, {B, 1, S, kv_seq_len},
                         {1, 1, 1, 1});
  }

  // ---- scaled_dot_product_attention: the SAME MLX fused kernel Python calls
  const float scale = static_cast<float>(std::pow(static_cast<double>(D), -0.5));
  mx::array out = mx::fast::scaled_dot_product_attention(
      q, full_k, full_v, scale, /*mask_mode=*/"", /*mask_arr=*/use_mask,
      /*sinks=*/std::nullopt);

  // output.transpose(0,2,1,3).reshape(B,L,-1); then o_proj(out * sigmoid(gate))
  out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), mx::Shape{B, S, Hq * D});
  out = mx::matmul(out * mx::sigmoid(gate), mx::transpose(w.o_w));

  cache.offset += S;  // KVCache.update_and_fetch advances offset by L
  return {out, mask};
}

// ---------------------------------------------------------------------------
// 3. Combined decoder step
// ---------------------------------------------------------------------------

mx::array run_decoder_step_full(
    const mx::array& input_embeddings,
    const std::vector<DecoderFullLayer>& layers,
    const DecoderHeadWeights& head,
    const DecoderConfig& cfg,
    const FullAttentionConfig& fa_cfg,
    const mx::array& position_ids,
    std::vector<DecoderFullLayerCache>& caches,
    DecoderFullDebug* debug) {
  if (static_cast<int>(layers.size()) != cfg.n_layers ||
      static_cast<int>(caches.size()) != cfg.n_layers) {
    throw std::invalid_argument(
        "run_decoder_step_full: layers/caches size != n_layers");
  }

  mx::array hidden =
      mx::tile(input_embeddings, std::vector<int>{1, 1, cfg.hc_count});

  for (int l = 0; l < cfg.n_layers; ++l) {
    const DecoderFullLayer& L = layers[l];

    // ---- attention hyper-connection --------------------------------------
    GatedResidualOutput ar = gated_residual(
        hidden, L.base.attn_norm_w, L.base.attn_down_w, L.base.attn_up_w,
        L.base.attn_inject_w, cfg.hc_count, cfg.hidden_size, cfg.eps);

    std::optional<mx::array> branch;
    if (L.attn_kind == AttnKind::Linear) {
      // GDN linear-attention branch (already bit-exact).
      GDNLayerOutput g = gdn_layer_forward(ar.mixed_input, L.base.gdn,
                                           caches[l].base.gdn, cfg.gdn);
      caches[l].base.gdn = GDNCacheState(g.conv_state, g.recurrent_state);
      branch = g.output;
    } else {
      // Full-attention (QSA sparse) branch.
      if (!L.full.has_value()) {
        throw std::invalid_argument(
            "run_decoder_step_full: full layer missing attention weights");
      }
      FullAttentionOutput f = full_attention_forward(
          ar.mixed_input, *L.full, fa_cfg, caches[l].full, position_ids);
      branch = f.output;
      if (debug) debug->qsa_mask.push_back(f.mask);
    }

    hidden = inject_branch(hidden, *branch, ar.injection_weights);
    if (debug) debug->attn_out.push_back(*branch);

    // ---- mlp hyper-connection --------------------------------------------
    GatedResidualOutput mr = gated_residual(
        hidden, L.base.mlp_norm_w, L.base.mlp_down_w, L.base.mlp_up_w,
        L.base.mlp_inject_w, cfg.hc_count, cfg.hidden_size, cfg.eps);

    // ---- MoE with non-zero shared expert ---------------------------------
    mx::array m2d = mx::reshape(mr.mixed_input, mx::Shape{1, cfg.hidden_size});
    mx::array gate_logits = mx::matmul(m2d, mx::transpose(L.base.gate_w));
    RouterOutput routes =
        router(gate_logits, L.base.top_k, L.base.renorm_blend);
    ExpertWeights ew(L.base.expert_gate, L.base.expert_up,
                     L.base.expert_down, L.base.group_size, L.base.bits);
    mx::array moe2d =
        moe_block_with_shared(m2d, routes.indices, routes.scores, ew,
                             L.shared);
    mx::array moe = mx::reshape(moe2d, mx::Shape{1, 1, cfg.hidden_size});

    hidden = inject_branch(hidden, moe, mr.injection_weights);
    if (debug) {
      debug->moe_out.push_back(moe);
      debug->hidden.push_back(hidden);
    }
  }

  return final_head(hidden, head.norm_w, head.down_w, head.up_w,
                    head.lm_head_w, cfg.hc_count, cfg.hidden_size, cfg.eps);
}

}  // namespace veystrix::engine
