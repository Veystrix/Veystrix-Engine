// Generic (recipe-driven) runtime behind the narrow C ABI.
//
// Same contract as NemotronEngineRuntime (one MLX driver
// thread, N sessions, APPEND prefill, batched step, fork, snapshots with
// post-prefix logits, seeded non-greedy sampling, per-session cancel, logit
// digests), but the model is a GenericModel built from a recipe. Selected by
// vx_engine_open when the checkpoint's model_type has a recipe (profile key
// "recipe") or when the profile says {"engine":"generic"}.
//
// Profile keys (flat JSON): recipe (path or inline JSON, required), kv
// (bf16|int8|k8v4|int4|tiered|tiered4|tiered4+|tiered-k8v4), kv_window,
// kv_prefix, kv_hist_k_bits, kv_hist_v_bits, resident (experts in RAM),
// prefill_chunk (default 2048), digests.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace veystrix::engine::capi_live {

class GenericEngineRuntime {
 public:
  GenericEngineRuntime();
  ~GenericEngineRuntime();
  GenericEngineRuntime(const GenericEngineRuntime&) = delete;
  GenericEngineRuntime& operator=(const GenericEngineRuntime&) = delete;

  void open(const std::string& model_dir, const std::string& profile_json);
  int new_session();
  void close_session(int sid);
  void prefill(int sid, const std::vector<uint32_t>& tokens);
  void prefill_append(int sid, const std::vector<uint32_t>& tokens);
  uint32_t step(int sid, const std::string& sampler_json);
  std::vector<uint32_t> step_batch(const std::vector<int>& sids,
                                   const std::vector<std::string>& sampler_jsons);
  void fork_session(int src_sid, int dst_sid);
  void seed(int sid, uint64_t seed);
  int session_tokens(int sid);
  void session_truncate(int sid, uint32_t target_tokens);
  std::string last_digest(int sid);
  void cancel_session(int sid);
  void cancel();
  bool consume_cancel();
  void snapshot_save(int sid, const std::string& dir);
  void snapshot_restore(int sid, const std::string& dir);
  std::string driver_stats() const;
  std::string capabilities_extra() const;  // ,"engine":"generic",... fragment

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// True when `model_dir/config.json` declares a model_type the generic runtime
// serves (qwen3_5_moe today). Cheap: reads config.json only.
bool generic_model_type_supported(const std::string& model_dir, std::string* model_type = nullptr);

}  // namespace veystrix::engine::capi_live
