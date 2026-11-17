// Bit-exact PREFILL (S > 1) for the GDN linear-attention layer — the
// Qwen4ExpGatedDeltaNet.__call__ full-sequence path on the M4 (GPU, Metal
// kernel available) production configuration.
//
// WHY THIS EXISTS
// ---------------
// The app always prefills a prompt before decode. For S > 1 the Python layer
// does NOT run the S==1 decode conv; it runs a different path:
//
//   Qwen3_5GatedDeltaNet.__call__   (qwen3_5/language.py:1109-1196)
//     mixed_qkv/z/b/a = full projections over [B,S,H]
//     if mask: mixed_qkv = where(mask[...,None], mixed_qkv, 0)
//     conv_input = concat([conv_state, mixed_qkv], 1)   // [B,K-1+S,C]
//     cache[0]   = contiguous(conv_input[:, -(K-1):, :])
//     conv_out   = silu(self.conv1d(conv_input))         // FULL depthwise conv
//     q,k,v = split/reshape(conv_out)
//     q,k = _normalize_qk(q,k)                           // Qwen4Exp override
//     gated_delta_update(..., mask, use_kernel=not training)
//     out = self.norm(out, z); return out_proj(out.reshape(B,S,-1))
//
//   gated_delta_update (qwen3_5/gated_delta.py:382-408)
//     On M4 GPU with the Metal kernel available and eval mode, use_kernel=True
//     so the `not use_kernel or device!=gpu or !metal` branch is FALSE and it
//     calls gated_delta_kernel DIRECTLY (VERIFIED by instrumenting the real
//     model: S in {2,5,16} -> ['kernel'], never 'chunked'/'ops'). The Metal
//     kernel's own `for (t=0; t<T; ++t)` loop performs the per-token delta-rule
//     recurrence. So production prefill is a SEQUENCE LOOP INSIDE THE FUSED
//     KERNEL, mathematically identical to repeated S==1 steps, NOT the
//     chunked scan (gated_delta_chunked is only for backends without a kernel).
//
// This function mirrors that path op-for-op with the SAME MLX 0.32.2 primitives
// (full matmul, mx::conv1d, fused delta kernel with the exact Metal source),
// which is what makes it bit-exact rather than merely close.
//
// DTYPE: as with gdn_layer.cpp, exercise with f32 x / f32 cache. Linear/conv/
// out_proj weights stay bf16 (production dtype); MLX promotes the matmul and
// conv to f32 exactly as nn.Linear / nn.Conv1d do. See gen_prefill_vectors.py.
#pragma once
#include "gdn_layer.hpp"  // GDNWeights / GDNConfig / GDNCacheState / GDNLayerOutput

namespace veystrix::engine {

// One full-sequence prefill of Qwen4ExpGatedDeltaNet.__call__ for S > 1.
//   x    : [B, S, H]  (f32)
//   cache: conv_state [B,K-1,C] and recurrent_state [B,Hv,Dv,Dk] (f32)
//   mask : optional bool [B,S] SSM/padding mask, exactly as production passes
//          it to __call__ (nullptr == no mask, the normal single-prompt case).
// Returns output [B,S,H] together with the updated conv/recurrent cache.
// Falls back to the ops reference (no Metal) only for the delta kernel.
GDNLayerOutput gdn_prefill(const mx::array& x, const GDNWeights& weights,
                           const GDNCacheState& cache, const GDNConfig& cfg,
                           const mx::array* mask = nullptr);

// NOTE on MoE: Qwen3_5MoeSparseMoeBlock.__call__(x) takes NO mask and processes
// [B,S,H] uniformly (router softmax/argpartition, SwitchGLU gather, shared
// expert). There is no S>1-specific mask handling in the MoE path, so no
// separate `moe_prefill` is required — the existing S==1 MoE primitives
// (moe_block / q4_projection) already accept an arbitrary token count. Only
// the GDN (and PLE) layers consume an SSM mask, and PLE is upstream of the
// attention block. See research_notes/.../prefill_port.md.

}  // namespace veystrix::engine
