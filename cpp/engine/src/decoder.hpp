// Full multi-layer decoder forward loop — assemble the already-verified
// bit-exact primitives into ONE decode token (batch=1, seq=1), matching
// Qwen4ExpModel.__call__ / Qwen4ExpDecoderLayer.__call__ from mlx_vlm.
//
// Python reference (qwen4_exp/language.py):
//   Qwen4ExpModel.__call__           lines 792-829
//     embed_tokens -> tile(1,1,hc_count) -> per-layer loop ->
//     hyper_connection_mixer (gated residual, use_combine=False) -> lm_head
//   Qwen4ExpDecoderLayer.__call__    lines 744-772
//     attn hyper-connection -> linear_attn/self_attn ->
//     hidden = hyper_input + branch[...,None,:]*inj[...,None]
//     mlp hyper-connection -> MoE -> same injection
//   Qwen4ExpGatedResidual            lines 420-454
//
// Composition (all pieces verified independently in STATUS.md):
//   gated_residual()  -> gdn_layer_forward()  -> router()+moe_block()
//   -> final_head()
//
// SCOPE / HONESTY
// ---------------
// The attention branch is handled by gdn_layer_forward for every layer. This
// is EXACTLY what the validated tiny config uses (all layers are linear
// attention, so QSA full attention is never invoked). A generic full-attention
// hook (QSACache) is carried in DecoderLayerCache for future integration but
// the full-QSA dense branch is NOT composed here — see decoder_loop_port.md.
//
// NOTE ON model_adapter.hpp: it defines a SECOND `GDNConfig` in this namespace,
// which collides with gdn_layer.hpp's `GDNConfig`. To keep both translation
// units valid, this header intentionally does NOT include model_adapter.hpp;
// callers holding a ModelSpec must copy the scalar fields into DecoderConfig
// explicitly (see the integration example in decoder_loop_port.md).
#pragma once

#include <vector>

#include "engine.hpp"       // router, gated_residual, moe_block, final_head, ...
#include "gdn_layer.hpp"    // GDNConfig, GDNWeights, GDNCacheState, gdn_layer_forward
#include "qsa_full.hpp"     // QSACache (carried for future full-attention layers)

namespace veystrix::engine {

// All weights needed by one decoder layer. Weight layout matches the real
// mlx_vlm modules / nn.Linear convention (out, in), so a caller can load them
// straight from a checkpoint via ExpertStore / mx::load.
struct DecoderLayerWeights {
  // attn hyper-connection (Qwen4ExpGatedResidual, use_combine=True)
  mx::array attn_norm_w;     // [hc_hidden]        hc_norm gain (one-centered)
  mx::array attn_down_w;     // [hc_lowrank, hc_hidden]
  mx::array attn_up_w;       // [hc_hidden, hc_lowrank]
  mx::array attn_inject_w;   // [hc_count, hc_hidden]

  // linear-attention branch (Qwen4ExpGatedDeltaNet)
  GDNWeights gdn;

  // mlp hyper-connection (Qwen4ExpGatedResidual, use_combine=True)
  mx::array mlp_norm_w;
  mx::array mlp_down_w;
  mx::array mlp_up_w;
  mx::array mlp_inject_w;

  // MoE branch (Qwen3_5MoeSparseMoeBlock, routed experts + zeroed shared)
  mx::array gate_w;          // [n_experts, hidden]  (nn.Linear, no bias)
  Q4Projection expert_gate;  // gate_proj  (Q4 affine)
  Q4Projection expert_up;    // up_proj    (Q4 affine)
  Q4Projection expert_down;  // down_proj  (Q4 affine)
  int top_k = 0;
  float renorm_blend = 0.0f; // 0 -> exact softmax/argpartition/renorm
  int group_size = 32;
  int bits = 4;

  DecoderLayerWeights(
      mx::array anw, mx::array adw, mx::array auw, mx::array aiw,
      GDNWeights gw,
      mx::array mnw, mx::array mdw, mx::array muw, mx::array miw,
      mx::array gate,
      Q4Projection eg, Q4Projection eu, Q4Projection ed,
      int tk, float rb, int gs, int b)
      : attn_norm_w(std::move(anw)), attn_down_w(std::move(adw)),
        attn_up_w(std::move(auw)), attn_inject_w(std::move(aiw)),
        gdn(std::move(gw)),
        mlp_norm_w(std::move(mnw)), mlp_down_w(std::move(mdw)),
        mlp_up_w(std::move(muw)), mlp_inject_w(std::move(miw)),
        gate_w(std::move(gate)),
        expert_gate(std::move(eg)), expert_up(std::move(eu)),
        expert_down(std::move(ed)),
        top_k(tk), renorm_blend(rb), group_size(gs), bits(b) {}
};

// Final mixer (Qwen4ExpGatedResidual, use_combine=False) + lm_head.
struct DecoderHeadWeights {
  mx::array norm_w;      // [hc_hidden]
  mx::array down_w;      // [hc_lowrank, hc_hidden]
  mx::array up_w;        // [hc_hidden, hc_lowrank]
  mx::array lm_head_w;   // [vocab, hidden]

  DecoderHeadWeights(mx::array nw, mx::array dw, mx::array uw, mx::array lw)
      : norm_w(std::move(nw)), down_w(std::move(dw)),
        up_w(std::move(uw)), lm_head_w(std::move(lw)) {}
};

// Geometry / hyper-parameters of the decode step.
struct DecoderConfig {
  int hidden_size = 0;    // H (one stream)
  int hc_count = 0;       // number of hyper-connection streams
  int hc_lowrank = 0;     // hc lowrank
  int n_layers = 0;       // number of decoder layers
  int vocab = 0;          // lm_head output width
  float eps = 1e-6f;      // RMSNorm epsilon
  GDNConfig gdn;          // geometry for the (linear) attention branch
};

// Mutable per-layer cache carried between decode steps.
struct DecoderLayerCache {
  GDNCacheState gdn;            // required for every (linear) layer here
  QSACache qsa;                 // reserved for future full-attention layers
  DecoderLayerCache(GDNCacheState g, QSACache q = {})
      : gdn(std::move(g)), qsa(std::move(q)) {}
};

// Optional debug intermediates, one entry per layer, for isolating a failing
// primitive when comparing against generated vectors.
struct DecoderStepDebug {
  std::vector<mx::array> gdn_out;   // attention branch output [1,1,H]
  std::vector<mx::array> moe_out;   // MoE output [1,1,H]
  std::vector<mx::array> hidden;    // hidden state after the layer [1,1,hc_hidden]
};

// Run ONE decode token through all layers.
//   input_embeddings : [1, 1, hidden]  (pre-embedding, NOT tiled)
//   caches           : size == cfg.n_layers; updated in place (conv_state +
//                      recurrent_state per layer) for the next token.
//   debug            : optional out-param for per-layer intermediates.
// Returns logits [1, 1, vocab].
mx::array run_decoder_step(
    const mx::array& input_embeddings,
    const std::vector<DecoderLayerWeights>& layers,
    const DecoderHeadWeights& head,
    const DecoderConfig& cfg,
    std::vector<DecoderLayerCache>& caches,
    DecoderStepDebug* debug = nullptr);

// Convenience: token ids -> embedding -> tile -> run_decoder_step.
//   tokens    : [1, 1] int32
//   embed_w   : [vocab, hidden] nn.Embedding weight
mx::array run_decoder_step_from_tokens(
    const mx::array& tokens,
    const mx::array& embed_w,
    const std::vector<DecoderLayerWeights>& layers,
    const DecoderHeadWeights& head,
    const DecoderConfig& cfg,
    std::vector<DecoderLayerCache>& caches,
    DecoderStepDebug* debug = nullptr);

}  // namespace veystrix::engine
