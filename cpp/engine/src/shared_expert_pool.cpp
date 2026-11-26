#include "shared_expert_pool.hpp"

#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace veystrix::engine {

// Pin bookkeeping. Deliberately contains NO std::mutex: every access is made
// while holding NemoPool::mutex(), the existing pool lock. Any new mutex would
// be NEEDS OWNER APPROVAL (architecture.md §1.2).
struct SharedExpertPool::PinState {
  std::unordered_map<int, int> by_slot;              // slot -> pin references
  std::unordered_map<int64_t, std::vector<int>> by_fence;  // (session,layer) -> slots
};

namespace {

// Combine (session_id, layer) into one fence key. Both are 32-bit ids.
int64_t fence_key(int session_id, int layer) {
  return (static_cast<int64_t>(static_cast<uint32_t>(session_id)) << 32) |
         static_cast<uint32_t>(layer);
}

}  // namespace

SharedExpertPool::SharedExpertPool(int num_slots, int64_t record_bytes,
                                   NemoPool::Resolver resolver, bool nocache,
                                   NemoPool::ReadFn read_fn)
    : pool_(num_slots, record_bytes, std::move(resolver), nocache, std::move(read_fn)),
      state_(std::make_shared<PinState>()) {
  // Veto is held by the pool and called with the pool mutex HELD. It reads
  // state_ without taking a lock — safe because state_ is itself guarded by
  // that same lock. No new lock, thread, or worker is introduced.
  pool_.set_eviction_veto([st = state_](int slot) {
    auto it = st->by_slot.find(slot);
    return it != st->by_slot.end() && it->second > 0;
  });
}

SharedExpertPool SharedExpertPool::from_env(int num_slots, int64_t record_bytes,
                                            NemoPool::Resolver resolver,
                                            NemoPool::ReadFn read_fn) {
  return SharedExpertPool(num_slots, record_bytes, std::move(resolver),
                          nemo_pool_nocache_from_env(), std::move(read_fn));
}

int SharedExpertPool::acquire_and_pin(int session_id, int layer, int expert) {
  // Residency is established first by the pool (one physical read on a miss,
  // coalesced via shared futures if another session is already filling it).
  int slot = pool_.acquire(layer, expert);
  // Then record the pin under the pool's existing lock. In the soft-lockstep
  // scheduler (§1.3) the driver is the single thread issuing layer work, so
  // there is no competing evictor in this acquire->pin window; a future
  // multi-threaded driver would need to hold pool_.mutex() across both steps.
  std::lock_guard<std::mutex> lk(pool_.mutex());
  ++state_->by_slot[slot];
  state_->by_fence[fence_key(session_id, layer)].push_back(slot);
  return slot;
}

void SharedExpertPool::pin(int session_id, int layer, int expert) {
  (void)acquire_and_pin(session_id, layer, expert);
}

int SharedExpertPool::release_fence(int session_id, int layer) {
  std::lock_guard<std::mutex> lk(pool_.mutex());  // pool's existing lock
  PinState& st = *state_;
  auto it = st.by_fence.find(fence_key(session_id, layer));
  if (it == st.by_fence.end()) return 0;
  int released = 0;
  for (int slot : it->second) {
    auto pc = st.by_slot.find(slot);
    if (pc == st.by_slot.end()) continue;
    if (--pc->second == 0) st.by_slot.erase(pc);
    ++released;
  }
  // Only this session's fence key is erased; other sessions' pins on the same
  // slot (and their own fence keys) are untouched.
  st.by_fence.erase(it);
  return released;
}

std::vector<SharedExpertPool::SlotMapping>
SharedExpertPool::fetch_union(int layer, const std::vector<int>& experts) {
  std::vector<SlotMapping> out;
  std::unordered_set<int> seen;
  out.reserve(experts.size());
  for (int expert : experts) {
    if (!seen.insert(expert).second) continue;  // one acquire per distinct key
    int slot = pool_.acquire(layer, expert);     // coalesces in-flight fills
    out.push_back(SlotMapping{layer, expert, slot});
  }
  return out;
}

int SharedExpertPool::pin_count(int slot) const {
  std::lock_guard<std::mutex> lk(pool_.mutex());
  auto it = state_->by_slot.find(slot);
  return it == state_->by_slot.end() ? 0 : it->second;
}

int SharedExpertPool::pinned_slot_count() const {
  std::lock_guard<std::mutex> lk(pool_.mutex());
  return static_cast<int>(state_->by_slot.size());
}

bool SharedExpertPool::is_pinned(int slot) const { return pin_count(slot) > 0; }

}  // namespace veystrix::engine
