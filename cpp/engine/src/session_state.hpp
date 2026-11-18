// Multi-session Phase 1 — per-session, exclusively-owned state (contract §1.1).
//
// This is a NON-NUMERIC container: it holds opaque model-cache handles, a
// frozen numerics profile, a prefix-cache handle, an absolute position counter
// and an INDEPENDENT PRNG stream. Nothing here is ever shared with, or advanced
// by, another session. Bit-identity (contract §3) depends on that isolation.
//
// Labels: the splitmix64 stream and the yield handshake are [idea] design
// choices; the bit-identity requirement itself is a hard contract rule [idea].
// No real checkpoint, no MLX dependency, no measurement in this module.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace veystrix::engine {

// Numerics profile frozen at admission (contract §1.1, §3 rule 3). Once frozen
// it is immutable for the session's lifetime; memory pressure refuses new
// sessions rather than mutating an admitted one.
struct NumericsProfile {
  int prefill_chunk = 512;             // [idea] policy knob, frozen at admission
  std::string kv_tier = "int8";        // [idea] e.g. "bf16", "int8"
  std::string kv_mode = "quantized";   // [idea] e.g. "quantized", "full"
  int adaptive_top_k = 20;             // [idea] frozen top-k

  void freeze() noexcept { frozen = true; }
  bool is_frozen() const noexcept { return frozen; }

 private:
  bool frozen = false;
};

// Prefix-cache handle with solo-equivalent restoration semantics (§1.1, §3
// rule 4). A config whose prefix hit is not bit-identical must leave `enabled`
// false; this module only carries the handle, it performs no cache lookup.
struct PrefixCacheHandle {
  bool enabled = false;
  std::string chain_key;   // SHA-256 chain over fingerprint + token ids
  int prefix_len = 0;
};

class SessionState {
 public:
  SessionState(int session_id, unsigned long long prng_key,
               NumericsProfile profile, PrefixCacheHandle prefix = {});

  // Exclusive ownership: copying would duplicate the PRNG stream and the
  // opaque cache handles, which is forbidden by the bit-identity contract.
  SessionState(const SessionState&) = delete;
  SessionState& operator=(const SessionState&) = delete;
  SessionState(SessionState&&) = default;
  SessionState& operator=(SessionState&&) = default;
  ~SessionState() = default;

  int id() const noexcept { return id_; }

  // The seed/key is constant for the session's lifetime; only `state_` moves,
  // and only when THIS session calls draw_u64().
  unsigned long long prng_key() const noexcept { return key_; }
  unsigned long long prng_state() const noexcept { return state_; }
  unsigned long long draw_u64() noexcept;

  int position() const noexcept { return position_; }
  void advance_position(int n = 1) noexcept { position_ += n; }

  const NumericsProfile& profile() const noexcept { return profile_; }
  const PrefixCacheHandle& prefix() const noexcept { return prefix_; }

  // Opaque model cache vector (KV / GDN / Mamba / PLE layers). Exclusively
  // owned; the scheduler never shares these objects between sessions. The
  // Phase 1 CPU fixtures store opaque per-session implementation state here.
  std::vector<std::shared_ptr<void>>& model_cache() noexcept { return cache_; }
  const std::vector<std::shared_ptr<void>>& model_cache() const noexcept { return cache_; }

  // Explicit yield state (§1.3): a decode step that needs expert reads at
  // layer L records the request here; the scheduler consumes it after the step
  // returns. No other session may set or clear this session's request.
  void request_yield(int layer) noexcept {
    yield_requested_ = true;
    yield_layer_ = layer;
  }
  bool consume_yield(int* layer_out) noexcept;

 private:
  int id_;
  unsigned long long key_;
  unsigned long long state_;
  NumericsProfile profile_;
  PrefixCacheHandle prefix_;
  std::vector<std::shared_ptr<void>> cache_;
  int position_ = 0;
  bool yield_requested_ = false;
  int yield_layer_ = -1;
};

}  // namespace veystrix::engine
