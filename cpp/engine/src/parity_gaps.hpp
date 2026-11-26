// Two small, self-contained bit-exactness gaps left by decoder_full.cpp:
//
//   1. QUANTIZED shared expert. `moe_block_with_shared` (decoder_full.cpp)
//      computes the shared expert as a DENSE GLU MLP. In the real deployed
//      checkpoint (`models/flashnext/preflight.py` inventory) the shared
//      expert's gate/up/down projections are stored as WSB quantized triplets
//      and loaded as `nn.QuantizedLinear`, so production uses gather_qmm /
//      quantized_matmul rather than dense matmul. `moe_block_shared_quantized`
//      below reproduces that path with the already-verified `q4_projection`.
//
//   2. INDEXER-NONE dense fallback. `full_attention_forward` throws when
//      `qsa_full_indexer` returns std::nullopt. Production
//      (`Qwen4ExpAttention.__call__`, qwen4_exp/language.py 398-417) simply
//      leaves `mask` untouched and calls `super().__call__`; with mask=None /
//      "causal" that is ordinary (unmasked-for-S=1 / causal-for-S>1) SDPA.
//      `full_attention_dense_fallback` runs that dense path.
//
// This header includes (but does not modify) engine.hpp / decoder_full.hpp.
// All implementations live in parity_gaps.cpp and depend only on the symbols
// available from engine.cpp + qsa.cpp + qsa_full.cpp (see the standalone
// compile command in tests/test_parity_gaps.cpp).
#pragma once

#include <stdexcept>

#include "decoder_full.hpp"
#include "engine.hpp"

namespace veystrix::engine {

// Configuration for the quantized shared-expert helper.
struct SharedQuantConfig {
  int group_size = 32;   // affine quantization group size
  int bits = 4;          // bits per weight
  // Production stores the shared GLU as quantized WSB triplets (true).
  // Set false to run the DENSE shared expert instead (same combine math,
  // matching decoder_full.cpp::moe_block_with_shared) for checkpoints whose
  // shared expert is not quantized.
  bool shared_quantized = true;
};

// Routed MoE (Q4 gather_qmm, via moe_block) PLUS a shared expert whose
// gate/up/down projections are single-expert Q4 affine gathers:
//
//   routed = moe_block(x, route, scores, routed_weights)         [t, H]
//   g, u   = Q4_gate(x), Q4_up(x)                                [t, inter]
//   s      = Q4_down(swiglu(g, u))                               [t, H]
//   sg     = sigmoid(x @ shared_gate_logits_w.T)                 [t, 1]
//   return routed + sg * s
//
// `shared_q4_weights.gate/up/down` each hold a ONE-expert Q4Projection:
//   weight [1, out, in/8], scales [1, out, in/group], biases [1, out, in/group].
// `shared_gate_logits_w` is the dense [1, H] sigmoid-gate weight. (The real
// checkpoint also quantizes this gate; see the parity note for why it is kept
// dense here and how to extend the helper if byte-exactness of the gate
// matters.)
mx::array moe_block_shared_quantized(
    const mx::array& x,
    const mx::array& route,
    const mx::array& scores,
    const ExpertWeights& routed_weights,
    const ExpertWeights& shared_q4_weights,
    const Linear& shared_gate_logits_w,
    const SharedQuantConfig& config);

// Dense full-attention path used when the QSA indexer returns None
// (max_complete_blocks <= block_topk). Mirrors Qwen4ExpAttention.__call__ ->
// Qwen3_5Attention.__call__ with mask=None / "causal": the same q/k/v proj,
// one-centered q/k norms, RoPE and KV-cache update as full_attention_forward,
// but SDPA runs WITHOUT a sparse QSA mask.
//
// For a decode step (S == 1) Python passes mask=None, so the MLX fused SDPA
// is called with no mask at all. For S > 1 the caller's "causal" mask mode is
// reproduced by using mask_mode="causal" (the MLX fused causal kernel).
//
// Throws if the indexer (evaluated on a copy of cache.indexer) actually yields
// a mask, because this helper is only valid for the indexer-None case.
mx::array full_attention_dense_fallback(
    const mx::array& x,
    const FullAttentionWeights& w,
    const FullAttentionConfig& cfg,
    FullAttentionCache& cache,
    const mx::array& position_ids);

}  // namespace veystrix::engine
