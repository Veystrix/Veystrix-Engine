// Phase G — long-lived Nemotron C-ABI runtime with ONE GPU-driving thread.
//
// Live counterpart of tools/nemotron_generate.cpp.
// Plain C++ header: no MLX types and no STL object crosses the C ABI; the C ABI
// holds an opaque pointer and calls these methods. The class owns a single
// worker thread; EVERY MLX call made on behalf of the C ABI runs on that thread.
//
// Why one thread: MLX default streams are per-thread (mlx 0.32.2
// mlx/stream.cpp:20). A ctypes request thread driving MLX directly would get a
// different stream and re-introduce the P1 nondeterminism hazard.
//
// Numeric path: nemotron_engine_runtime.cpp includes
// tools/nemotron_generate.cpp with main() renamed, so the flat-config loader,
// load_model, ExpertReader, NemoPool assembly, run_step and sample_greedy are
// reused verbatim. No engine primitive is reimplemented and no tool file is
// edited. The split here is purely lifecycle: load once (open), prefill on
// demand, decode one token per step, cancel at a step boundary.
//
// Snapshot: nemotron_generate.cpp has NO prefix-cache/snapshot primitive (the
// Qwen runtime's prefix_capture_with_logits is Qwen-specific). snapshot_save /
// snapshot_restore therefore fail closed with a clear error; they are not
// silently faked.
//
// Scope (Phase G): one live session per engine, greedy decode only.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// The CancelledError type the C ABI maps to VX_ERR_CANCELLED.
#include "cancelled_error.hpp"

namespace veystrix::engine::capi_live {

class NemotronEngineRuntime {
 public:
  NemotronEngineRuntime();
  ~NemotronEngineRuntime();
  NemotronEngineRuntime(const NemotronEngineRuntime&) = delete;
  NemotronEngineRuntime& operator=(const NemotronEngineRuntime&) = delete;

  // Load weights + build caches. Runs on the driver thread; blocks the caller.
  void open(const std::string& model_dir, const std::string& profile_json);

  // Session lifecycle. One live session per engine in Phase G.
  int new_session();
  void close_session(int sid);

  // Prompt tokens -> logits for the next token. Nemotron prefill is the CLI's
  // sequential chunk_tokens=1 path: one run_step per prompt token.
  void prefill(int sid, const std::vector<uint32_t>& tokens);

  // Append one caller-supplied chunk (ABI v2 VX_PREFILL_APPEND) without reset.
  void prefill_append(int sid, const std::vector<uint32_t>& tokens);

  // One decode step. Samples from the current logits, returns the token id.
  // Non-greedy sampler_json fails closed (Phase G supports greedy only).
  uint32_t step(int sid, const std::string& sampler_json);

  // Multi-session batched step: N sessions on ONE driver thread. Each session's
  // Mamba/attention cache is swapped into the shared Model around its step, so
  // tokens are bit-identical to N sequential single-session steps. Returns one
  // token per session in `sids` order. Same sampler/engine constraints as step().
  std::vector<uint32_t> step_batch(const std::vector<int>& sids,
                                   const std::vector<std::string>& sampler_jsons);

  // In-RAM copy-on-write fork of a session's cache state.
  void fork_session(int src_sid, int dst_sid);

  // Per-session PRNG seed for non-greedy sampling (ABI v2 vx_session_seed).
  void seed(int sid, uint64_t seed);

  // SHA-256 hex of the logits row from the last step (ABI v2 last_digest).
  std::string last_digest(int sid);

  // Cooperative cancel: atomic flag, no MLX, no job posting. Checked at the
  // next step boundary (the following step throws CancelledError).
  void cancel_session(int sid);
  void cancel();
  bool consume_cancel();

  // Not wired: nemotron_generate.cpp has no snapshot primitive. Fails closed.
  void snapshot_save(int sid, const std::string& dir);
  void snapshot_restore(int sid, const std::string& dir);

  // Plain-text stats fragment (already escaped values), no MLX calls.
  std::string driver_stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace veystrix::engine::capi_live
