// Phase F — Nemotron per-session decode runtime (both-model parity).
//
// Qwen's Phase F seam lives in qwen_runtime.{hpp,cpp}: a per-layer cursor driver
// over real_checkpoint_generate.cpp plus per-session PLE/cache staging consumed
// by MsScheduler. This header is the Nemotron counterpart.
//
// WHY A WHOLE-TOKEN PATH FOR MULTI-SESSION (honest scope):
// Nemotron's `run_step` (nemotron_generate.cpp) does all of its expert reads
// with synchronous `pread` through ExpertReader — there is NO async ticket /
// `plan_pin_submit` seam and therefore no safe point at which to yield a
// session mid-layer. A per-layer layer_pre/layer_post split CAN be written (and
// is, below, as `run_step_cursor` for structural parity with qwen_runtime), but
// interleaving sessions at that split would give no overlap benefit: the reads
// are already synchronous and the single GPU-driving thread must wait for them
// anyway. The multi-session driver therefore advances one WHOLE `run_step` per
// session per scheduler step, with each session's mutable state isolated in its
// own SessionCacheState. `run_step` is reused VERBATIM (include-with-renamed-
// main), so the N=2/3 gate is a strict token+digest comparison against the solo
// binary. See docs/engineering/phase_f_nemotron.md for what remains.
//
// C1 hard rules (unchanged): one driver thread, one mutex + one CV (owned by
// the existing pager), atomic tickets, the existing width-8 user-interactive
// dispatch queue where applicable, no MLX in any notify callback, and NO cross-
// session row batching. Nemotron's expert reads add no thread, mutex or CV.
#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include <mlx/mlx.h>

#include "mamba_layer.hpp"
#include "nemo_attention.hpp"

namespace veystrix::engine::nemotron_runtime {

// P5-overlap: drive N sessions in layer lockstep. All MoE expert reads are
// submitted (async) before any session waits, so one session's host build + GPU
// work overlaps another session's SSD reads. `model`/`reader` are the tool's
// Model* / ExpertReader* (opaque here); the callbacks swap each session's
// cache into the shared Model around its layer work. Returns one logits row per
// session. Numeric ops and their order are identical to the solo run_step.
using CacheSwapFn = std::function<void(int /*session_index*/)>;

// S>1 chunked prefill for Nemotron, composing the bit-exact P5b primitives
// (mamba_layer_prefill, S>1 INT8 attention, sorted relu2 MoE). `tokens` is one
// Python-anchored chunk. Returns the logits row for the LAST token.
mx::array run_prefill_chunk(void* model, void* reader,
                            const std::vector<int32_t>& tokens);

std::vector<mx::array> run_step_pipelined(
    void* model, void* reader, const std::vector<int32_t>& tokens,
    const CacheSwapFn& swap_in, const CacheSwapFn& swap_out);

// True iff VEYSTRIX_CPP_PHASE_F_CURSOR=1. When set, the N=1 gate binary drives
// decode through the split layer_pre/layer_post `run_step_cursor`; otherwise it
// calls the original `run_step` (identical CLI to nemotron_generate).
bool phase_f_cursor_enabled();

// Number of dispatch-group completion callbacks observed (atomic load). Nemotron
// uses synchronous expert pread, so this is structurally always 0; it exists so
// the gate/diagnostics share the same shape as qwen_runtime.
int cursor_notify_count();

// N=1 per-layer cursor gate entry point (same CLI as nemotron_generate).
int nemotron_gate_main(int argc, char** argv);

// N-session driver entry point:
//   nemotron_runtime multi <checkpoint-dir> <num_sessions> <tokens> <config0> ...
int multisession_main(int argc, char** argv);

// Per-session MUTABLE decode state — the Nemotron analogue of Qwen's per-session
// caches + PLE state. The read-only weights live in one shared Model and are
// never written; only these objects change as a session advances:
//   * mamba_caches  — Mamba-2 conv (bf16) + SSM (f32) recurrence per Mamba layer
//   * attn_int8     — INT8 affine-block KV cache per attention layer (default)
//   * attn_f32      — bf16 KV cache per attention layer (VEYSTRIX_NEMOTRON_KV=bf16)
//
// MLX arrays are immutable: mamba_layer_forward returns NEW cache arrays and
// nemo_attention_*_forward only appends blocks / bumps integer offsets. A copy
// of this struct therefore shares the harmless read-only array handles while
// every mutable field (vector structure, offsets, and array REPLACEMENTS) stays
// strictly per-session.
struct SessionCacheState {
  std::vector<MambaCacheState> mamba_caches;
  std::vector<NemoAttentionCache> attn_f32;
  std::vector<NemoInt8Cache> attn_int8;

  // Build `n_mamba` zero Mamba caches and `n_attn` empty attention caches with
  // exactly the geometry/state that nemotron_generate.cpp::load_model builds.
  // `num_heads` is the Mamba head count (SSM state dim 1), NOT hidden_size.
  void init(int n_mamba, int conv_kernel, int conv_dim,
            int num_heads, int head_dim, int ssm_state_size,
            int n_attn, int block_tokens);
};

// Stage 1 batched shared-read decode (opt-in VEYSTRIX_CPP_NEMO_BATCHED=1).
// Runs N sessions in strict layer lockstep on the single driver thread. Mamba /
// attention layers stay per-session (each session's caches are swapped in), but
// every MoE layer is batched: routes for all N sessions are unioned, the expert
// rows are read ONCE, and one moe_block_relu2 runs over the [N,K] batch. Stage 0b
// (probe_nemo_moe_batch) proved moe_block_relu2 bit-invariant at T=N. This runs
// the same primitive as the solo path with T=N instead of T=1, so tokens/digests
// stay bit-identical. Returns 0 on success.
int batched_multisession_main(int argc, char** argv);

}  // namespace veystrix::engine::nemotron_runtime
