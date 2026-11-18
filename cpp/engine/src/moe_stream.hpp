// Streaming MoE: load expert rows from a REAL safetensors checkpoint into a
// persistent slot pool, remap the router route to those slots, and run the
// exact moe_block (gather_qmm) path on the slot-resident weights.
//
// This closes the I/O gap: the C++ decode step no longer needs Python / .npy
// vectors to feed expert weights. The slot bookkeeping implements the
// bit-exact rules in docs/performance/OPUS_E0_E1_SPEC.md (E0/E1): dedup misses,
// pin slots with a refcount until the command buffers complete (mx::eval), and
// combine expert outputs in router-rank order (no re-sorting of the route).
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engine.hpp"

namespace veystrix::engine {

struct StreamForwardResult {
  mx::array output;               // [tokens, hidden]
  std::vector<int32_t> slots;     // remapped route (router-rank order, flat T*K)
  std::vector<int32_t> misses;    // expert ids loaded from disk this call
};

// Persistent streaming MoE for one MoE layer of a Qwen4Exp checkpoint.
// Reads raw expert rows straight from the shard (via the native pager) into a
// slot pool. Resident experts keep their slot across forward() calls.
class StreamingMoE {
 public:
  // checkpoint_dir: directory with model.safetensors.index.json + shards.
  // layer: MoE layer index. num_slots: persistent slot pool size.
  // prefix: e.g. "language_model.model.layers.0.mlp.switch_mlp"; empty uses
  //         the Qwen4Exp default for `layer`.
  StreamingMoE(const std::string& checkpoint_dir, int layer, int num_slots,
               int group_size = 32, int bits = 4, int read_width = 8,
               const std::string& prefix = "");
  ~StreamingMoE();

  StreamingMoE(const StreamingMoE&) = delete;
  StreamingMoE& operator=(const StreamingMoE&) = delete;

  // x: [tokens, hidden] float; route: [tokens, k] expert ids (uint32/int32);
  // scores: [tokens, k] float. Throws std::runtime_error on fail-closed
  // conditions (pool exhausted, short read, missing tensor).
  StreamForwardResult forward(const mx::array& x, const mx::array& route,
                              const mx::array& scores);

  // Bit-exact batched prefill: build a compact first-occurrence expert stack
  // (distinct routed experts in first-occurrence order), remap routes to
  // stacked-row indices, and run one sorted gather_qmm over the whole prompt.
  // Matches Python SwitchGLU when indices.size >= 64. route = [tokens,k] global
  // expert ids. Reads only the distinct expert rows (low memory).
  mx::array forward_stacked(const mx::array& x, const mx::array& route,
                            const mx::array& scores);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Convenience: construct a pool, run one step, return only the output.
mx::array moe_stream_forward(const std::string& checkpoint_dir, int layer,
                             const mx::array& x, const mx::array& route,
                             const mx::array& scores, int num_slots = 16,
                             int group_size = 32, int bits = 4,
                             int read_width = 8,
                             const std::string& prefix = "");

}  // namespace veystrix::engine
