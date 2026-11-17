// Bit-exact BATCHED prefill (S > 1) for the real-checkpoint Qwen4Exp stack.
//
// WHY THIS EXISTS
// ---------------
// cpp/engine/tools/real_checkpoint_generate.cpp prefills the prompt with S
// sequential S==1 `run_step` calls.  Python's production `language(ids, cache)`
// feeds ALL S prompt tokens through every layer in ONE call.  Those are not the
// same computation: e.g. the GDN full-sequence conv + fused delta kernel,
// full-sequence Linear projections and dense causal SDPA reduce in a different
// order than S separate decode steps.  The difference is ~1 ULP at layer 0 but
// is amplified by MoE routing through 48 layers (L47 corr ~0.91 in the
// instrumented run).
//
// This file composes the ALREADY BIT-EXACT batched primitives
//   * gdn_prefill                          (prefill.hpp / prefill.cpp)
//   * full_attention_dense_batched         (replicated below from the tool)
//   * gated_residual / linear / router / swiglu / quantized_matmul (engine.hpp)
//   * ple_forward                          (ple.hpp, S>1)
// in the exact op order of Qwen4ExpDecoderLayer.__call__
// (mlx_vlm/models/qwen4_exp/language.py:744-772) followed by the final
// Qwen4ExpGatedResidual(use_combine=False) -> lm_head
// (real_checkpoint_generate.cpp:803-841).
//
// This file is ADDITIVE: it does not edit or replace any shared file.  The
// primary agent maps RealLayer -> BatchedLayer at integration time (see the
// comment next to BatchedLayer and the report).
//
// MOE: the routed-expert forward is injected as a std::function callback, so
// this file never depends on StreamingMoE / CrossShardMoE.  The synthetic
// oracle STUBS routed experts to zero (identically in Python and C++); the
// router and the non-zero dense shared expert are real in both.
#pragma once

#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

#include "decoder_full.hpp"  // DecoderConfig, FullAttentionConfig, DecoderFullLayerCache, AttnKind
#include "engine.hpp"        // Linear, gated_residual, router, swiglu, linear
#include "gdn_layer.hpp"     // GDNWeights / GDNConfig / GDNCacheState
#include "ple.hpp"           // PLEWeights / PLECache / ple_forward
#include "prefill.hpp"       // gdn_prefill

namespace mx = mlx::core;

namespace veystrix::engine {

// Callback for the routed-MoE experts.  Mirrors
//   L.stream->forward(m2d, routes.indices, routes.scores)  (or L.cross->forward)
// from real_checkpoint_generate.cpp:744-746, with `x` already reshaped to
// [S, H] (2-D, arbitrary token count).
//   x       : [S, H]   the MLP hyper-connection `mixed` input
//   indices : [S, k]   int32 router indices
//   scores  : [S, k]   f32 normalized router scores
//   returns : [S, H]   the routed-expert output (shared expert added by caller)
using BatchedMoEForward = std::function<mx::array(
    const mx::array& x, const mx::array& indices, const mx::array& scores)>;

// Minimal, self-contained per-layer weights.  The primary agent maps
// RealLayer (tools/real_checkpoint_generate.cpp:455) onto this 1:1:
//   attn_norm_w / attn_down_w / attn_up_w / attn_inject_w  <-- same fields
//   mlp_norm_w  / mlp_down_w  / mlp_up_w  / mlp_inject_w   <-- same fields
//   gate_w / shared_{gw,uw,dw,glw}                         <-- same fields
//   gdn / full / ple / ple_cache / top_k / kind            <-- same fields
//   moe_forward = [&L](const mx::array& x, const mx::array& i,
//                      const mx::array& s) {
//                   return L.stream ? L.stream->forward(x, i, s).output
//                                   : L.cross->forward(x, i, s).output;
//                 }
struct BatchedLayer {
  mx::array attn_norm_w;
  Linear attn_down_w, attn_up_w, attn_inject_w;
  mx::array mlp_norm_w;
  Linear mlp_down_w, mlp_up_w, mlp_inject_w;
  Linear gate_w;
  Linear shared_gw, shared_uw, shared_dw, shared_glw;
  std::optional<GDNWeights> gdn;
  std::optional<FullAttentionWeights> full;
  BatchedMoEForward moe_forward;
  std::optional<PLEWeights> ple;
  PLECache ple_cache;
  int top_k = 0;
  AttnKind kind = AttnKind::Linear;

  BatchedLayer(mx::array an, Linear ad, Linear au, Linear ai,
               mx::array mn, Linear md, Linear mu, Linear mi,
               Linear gw, Linear sgw, Linear suw, Linear sdw, Linear sglw,
               int tk, AttnKind k)
      : attn_norm_w(std::move(an)), attn_down_w(std::move(ad)),
        attn_up_w(std::move(au)), attn_inject_w(std::move(ai)),
        mlp_norm_w(std::move(mn)), mlp_down_w(std::move(md)),
        mlp_up_w(std::move(mu)), mlp_inject_w(std::move(mi)),
        gate_w(std::move(gw)), shared_gw(std::move(sgw)),
        shared_uw(std::move(suw)), shared_dw(std::move(sdw)),
        shared_glw(std::move(sglw)), top_k(tk), kind(k) {}
};

// A packed affine-Q4 lm_head triplet (DenseLoader::QTriplet equivalent).  The
// head is applied with mx::quantized_matmul(..., transpose=true, gs, bits,
// "affine") exactly like the tool; a dense Linear is also accepted for tests.
struct BatchedLMTriplet {
  mx::array w, s, b;
  BatchedLMTriplet(mx::array w_, mx::array s_, mx::array b_)
      : w(std::move(w_)), s(std::move(s_)), b(std::move(b_)) {}
};

struct BatchedHead {
  mx::array norm_w;
  Linear down_w, up_w;
  std::optional<BatchedLMTriplet> lm_quant;  // production path (Q4 lm_head)
  std::optional<Linear> lm_dense;            // optional dense lm_head (tests)

  BatchedHead(mx::array nw, Linear dw, Linear uw, BatchedLMTriplet lm)
      : norm_w(std::move(nw)), down_w(std::move(dw)), up_w(std::move(uw)),
        lm_quant(std::move(lm)) {}
  BatchedHead(mx::array nw, Linear dw, Linear uw, Linear lm)
      : norm_w(std::move(nw)), down_w(std::move(dw)), up_w(std::move(uw)),
        lm_dense(std::move(lm)) {}
};

// Optional instrumentation.  Per-layer hidden states are the last-token slice
// [1, H] (matching dump_python_layers.py's layer_L##.f32) so a failed parity
// can be localized to the first divergent layer.
struct BatchedPrefillDebug {
  std::vector<mx::array> hidden_after_layer;  // [1, H] per layer
  std::vector<mx::array> branch;              // attention branch [1, H] per layer
  std::vector<mx::array> m2d;                 // MLP gated-residual output [S,H] per layer
  std::vector<mx::array> gate_logits;         // router logits [S,experts] per layer
  std::vector<mx::array> moe2d;               // final MoE branch [S,H] per layer
  std::vector<mx::array> mlp_inj;             // MLP injection weights [S,hc] per layer
  std::vector<mx::array> gdn_in;              // attention gated-residual mixed_input [S,H] per layer
};

// Feed all S prompt tokens through every layer in ONE call and return the
// logits for the LAST prompt token, shape [1, vocab], float32.
//
//   input_ids : [1, S] int32 prompt tokens (B=1, as production).
//   embed     : callable input_ids -> [1, S, H] embeddings (f32).  In the real
//               engine this gathers all S packed rows and dequantizes them in
//               one shot (equivalent to S embed_token calls, elementwise).
//   layers    : size == cfg.n_layers, passed by reference because a PLE layer
//               carries a mutable PLECache; all other weight fields are read-only
//               and per-layer GDN/attention caches live in `caches`.
//   cfg       : DecoderConfig (hidden_size, hc_count, eps, n_layers, ...).
//   fa        : FullAttentionConfig geometry (used by full-attention layers).
//   caches    : size == cfg.n_layers, updated in place (GDN conv/recurrent and
//               full-attention KV) for the subsequent decode steps.
//   group_size, bits : affine-Q4 parameters for `head.lm_quant`.
mx::array run_prefill_batched(
    const mx::array& input_ids,
    const std::function<mx::array(const mx::array&)>& embed,
    std::vector<BatchedLayer>& layers,
    const BatchedHead& head,
    const DecoderConfig& cfg,
    const FullAttentionConfig& fa,
    std::vector<DecoderFullLayerCache>& caches,
    int group_size, int bits,
    BatchedPrefillDebug* debug = nullptr);

}  // namespace veystrix::engine
