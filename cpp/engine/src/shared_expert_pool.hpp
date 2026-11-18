// Phase 1 SharedExpertPool — one process-wide pool keyed by (layer, expert),
// layered on top of the ported NemoPool. See the design contract:
//   docs/performance/multi-session/architecture.md §1.2 (§4 for failure isolation)
// and the implementation note docs/engineering/shared_expert_pool_phase1.md.
//
// Only the FETCH is shared. Compute stays single-row / per-session; no rows
// from different sessions are ever batched into one matmul (§3 rule 1).
//
// Sync / thread / worker accounting (non-negotiable):
//   * All pin and fence bookkeeping lives under the pool's EXISTING mutex,
//     obtained via NemoPool::mutex(). There is NO new mutex here.
//   * There is NO new thread and NO new I/O worker; the I/O worker count is
//     unchanged. Fences are released explicitly by the existing driver, not by
//     a dedicated completion thread.
//   * NEEDS OWNER APPROVAL: any dedicated fence/completion thread, additional
//     mutex, or I/O worker-count change. None of those are added here.
#pragma once

#include "nemo_pool.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace veystrix::engine {

// Wraps NemoPool and adds multi-session pinning + per-layer union fetch.
// The NemoPool behaviours (in-flight shared futures, cached/uncached fd maps,
// LRU, fail-closed invalidating fills) are inherited unchanged.
class SharedExpertPool {
 public:
  // The contract's shorthand pin(layer, expert) pins under this session id.
  static constexpr int kDefaultSessionId = 0;

  // One distinct (layer, expert) key resolved to its resident slot.
  struct SlotMapping {
    int layer = -1;
    int expert = -1;
    int slot = -1;
  };

  struct PinState;  // defined in the .cpp; no mutex of its own (see above)

  SharedExpertPool(int num_slots, int64_t record_bytes, NemoPool::Resolver resolver,
                   bool nocache, NemoPool::ReadFn read_fn = {});
  static SharedExpertPool from_env(int num_slots, int64_t record_bytes,
                                   NemoPool::Resolver resolver,
                                   NemoPool::ReadFn read_fn = {});

  SharedExpertPool(const SharedExpertPool&) = delete;
  SharedExpertPool& operator=(const SharedExpertPool&) = delete;
  SharedExpertPool(SharedExpertPool&&) noexcept = default;
  SharedExpertPool& operator=(SharedExpertPool&&) noexcept = default;

  // Record that session `session_id` has queued GPU work which will read the
  // record (layer, expert). Increments that slot's pin count; the slot is from
  // then on ineligible as an LRU victim. If the key is not resident, this first
  // acquires it (a hit or a single coalesced fill). Never released when a read
  // returns — only by release_fence() for this session and layer.
  void pin(int session_id, int layer, int expert);
  // Literal contract shorthand: pin(layer, expert) == pin(kDefaultSessionId, ...).
  void pin(int layer, int expert) { pin(kDefaultSessionId, layer, expert); }

  // Acquire + pin in one call. Returns the slot index.
  int acquire_and_pin(int session_id, int layer, int expert);

  // Completion fence. Releases ONLY `session_id`'s pin references recorded at
  // this layer; every other session's pins (including pins on the same slot)
  // remain. Returns the number of references released. This is the sole pin
  // release path (not a read return, not another session's fence).
  int release_fence(int session_id, int layer);

  // Per-layer union fetch. `experts` is the multiset of expert ids needed at
  // `layer` by all sessions waiting there. Issues exactly one pool acquire per
  // DISTINCT key, in first-seen order; NemoPool's shared futures coalesce
  // concurrent same-key acquires into one physical read. Returns one mapping
  // per distinct key. This is a union ACROSS SESSIONS at one layer, never
  // across rows (§1.2). It does not pin — pin the keys the caller will consume.
  std::vector<SlotMapping> fetch_union(int layer, const std::vector<int>& experts);

  // Failure isolation (§4): NemoPool wakes ALL coalesced waiters with an
  // error and invalidates the failed slot back to the free list; SharedExpert
  // Pool does not swallow or widen that error. RETRY-ONCE-THEN-FAIL-SESSION is
  // the CALLER's responsibility: on a thrown acquire the caller retries once
  // via NemoPool::read_direct (its own read), and a SECOND failure fails only
  // that session. Sibling sessions' pins and in-flight reads are untouched;
  // their pins are released through their own fences. No dedicated retry or
  // cancellation mechanism is added here.

  // Introspection (guarded by the pool's existing mutex).
  int pin_count(int slot) const;
  int pinned_slot_count() const;
  bool is_pinned(int slot) const;

  NemoPool& pool() { return pool_; }
  const NemoPool& pool() const { return pool_; }
  NemoPoolStats stats() const { return pool_.stats(); }

 private:
  // NOTE: no new mutex — state_ is guarded entirely by pool_.mutex().
  NemoPool pool_;
  std::shared_ptr<PinState> state_;
};

}  // namespace veystrix::engine
