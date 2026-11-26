#include "decoder.hpp"

#include <mlx/mlx.h>

#include <stdexcept>
#include <utility>

namespace veystrix::engine {

namespace {

// Qwen4ExpDecoderLayer injection:
//   hidden = hyper_input + (branch[...,None,:] * inj[...,None]).reshape(...)
// `hidden` is the hyper_input (the gated-residual returns it unchanged), so we
// add straight back onto the value we passed into gated_residual.
mx::array inject_branch(const mx::array& hidden, const mx::array& branch,
                        const mx::array& inj) {
  // branch : [B, S, H]   inj : [B, S, hc_count]
  mx::array b4 = mx::expand_dims(branch, -2);  // [B, S, 1, H]
  mx::array i4 = mx::expand_dims(inj, -1);     // [B, S, hc_count, 1]
  return hidden + mx::reshape(b4 * i4, hidden.shape());
}

}  // namespace

mx::array run_decoder_step(
    const mx::array& input_embeddings,
    const std::vector<DecoderLayerWeights>& layers,
    const DecoderHeadWeights& head,
    const DecoderConfig& cfg,
    std::vector<DecoderLayerCache>& caches,
    DecoderStepDebug* debug) {
  if (static_cast<int>(layers.size()) != cfg.n_layers ||
      static_cast<int>(caches.size()) != cfg.n_layers) {
    throw std::invalid_argument("run_decoder_step: layers/caches size != n_layers");
  }

  // Qwen4ExpModel.__call__: embed_tokens -> tile by hc_count.
  mx::array hidden =
      mx::tile(input_embeddings, std::vector<int>{1, 1, cfg.hc_count});

  for (int l = 0; l < cfg.n_layers; ++l) {
    const DecoderLayerWeights& w = layers[l];

    // ---- attention hyper-connection ------------------------------------
    GatedResidualOutput ar = gated_residual(
        hidden, w.attn_norm_w, w.attn_down_w, w.attn_up_w,
        w.attn_inject_w, cfg.hc_count, cfg.hidden_size, cfg.eps);

    // ---- attention branch: GDN linear attention ------------------------
    // Every layer in the validated config is linear, so this is the exact
    // `self.linear_attn(mixed, mask, cache)` call of the Python layer.
    GDNLayerOutput branch =
        gdn_layer_forward(ar.mixed_input, w.gdn, caches[l].gdn, cfg.gdn);
    caches[l].gdn = GDNCacheState(branch.conv_state, branch.recurrent_state);

    // ---- inject attention branch into the hyper-connection -------------
    hidden = inject_branch(hidden, branch.output, ar.injection_weights);
    if (debug) debug->gdn_out.push_back(branch.output);

    // ---- mlp hyper-connection ------------------------------------------
    GatedResidualOutput mr = gated_residual(
        hidden, w.mlp_norm_w, w.mlp_down_w, w.mlp_up_w,
        w.mlp_inject_w, cfg.hc_count, cfg.hidden_size, cfg.eps);

    // ---- MoE branch: EXACT router from the gate weights ----------------
    // mlx_vlm Qwen3_5MoeSparseMoeBlock.__call__:
    //   gate = nn.Linear(dim, n_experts)  ->  softmax(precise)  ->
    //   argpartition(top_k)  ->  scores / sum
    mx::array m2d = mx::reshape(mr.mixed_input, mx::Shape{1, cfg.hidden_size});
    mx::array gate_logits = linear(m2d, w.gate_w);
    RouterOutput routes = router(gate_logits, w.top_k, w.renorm_blend);
    ExpertWeights ew(w.expert_gate, w.expert_up, w.expert_down,
                     w.group_size, w.bits);
    mx::array moe2d = moe_block(m2d, routes.indices, routes.scores, ew);
    mx::array moe = mx::reshape(moe2d, mx::Shape{1, 1, cfg.hidden_size});

    // ---- inject MoE branch into the hyper-connection -------------------
    hidden = inject_branch(hidden, moe, mr.injection_weights);
    if (debug) {
      debug->moe_out.push_back(moe);
      debug->hidden.push_back(hidden);
    }
  }

  // ---- final mixer (use_combine=False) -> lm_head ----------------------
  return final_head(hidden, head.norm_w, head.down_w, head.up_w,
                    head.lm_head_w, cfg.hc_count, cfg.hidden_size, cfg.eps);
}

mx::array run_decoder_step_from_tokens(
    const mx::array& tokens,
    const mx::array& embed_w,
    const std::vector<DecoderLayerWeights>& layers,
    const DecoderHeadWeights& head,
    const DecoderConfig& cfg,
    std::vector<DecoderLayerCache>& caches,
    DecoderStepDebug* debug) {
  mx::array ids = mx::reshape(tokens, mx::Shape{-1});
  mx::array e = mx::take(embed_w, ids, 0);  // [tokens, hidden]
  mx::array e3 = mx::reshape(e, mx::Shape{1, 1, cfg.hidden_size});
  return run_decoder_step(e3, layers, head, cfg, caches, debug);
}

}  // namespace veystrix::engine
