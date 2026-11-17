#include "slot_plan.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <unistd.h>

namespace veystrix::engine {

SlotStep plan_slots(const std::vector<int32_t>& route_ids,
                    std::vector<int32_t>& slot_expert,
                    const std::vector<int>& slot_ref,
                    const std::vector<uint64_t>& slot_last,
                    std::vector<int32_t>& free_slots, int num_slots) {
  if (num_slots < 1)
    throw std::runtime_error("plan_slots: num_slots must be >= 1");
  if (static_cast<int>(slot_expert.size()) != num_slots ||
      static_cast<int>(slot_ref.size()) != num_slots ||
      static_cast<int>(slot_last.size()) != num_slots)
    throw std::runtime_error("plan_slots: bookkeeping vector size mismatch");

  SlotStep step;

  // Correct resident map: slot -> expert, indexed ONCE, never by searching a
  // flattened [expert, slot, ...] list (Bug 1). Keys are real expert ids only.
  std::unordered_map<int32_t, int32_t> resident;
  resident.reserve(static_cast<size_t>(num_slots));
  for (int s = 0; s < num_slots; ++s)
    if (slot_expert[s] >= 0) resident[slot_expert[s]] = s;

  std::unordered_set<int32_t> needed(route_ids.begin(), route_ids.end());

  // Distinct misses in first-appearance route order.
  std::unordered_set<int32_t> seen;
  seen.reserve(route_ids.size() * 2 + 1);
  for (int32_t e : route_ids)
    if (resident.find(e) == resident.end() && seen.insert(e).second)
      step.miss_e.push_back(e);

  // LRU eviction: never a slot with ref > 0, never an expert this route needs.
  if (step.miss_e.size() > free_slots.size()) {
    std::vector<int> cand;
    cand.reserve(static_cast<size_t>(num_slots));
    for (int s = 0; s < num_slots; ++s) {
      if (slot_expert[s] < 0 || slot_ref[s] > 0) continue;
      if (needed.count(slot_expert[s])) continue;
      cand.push_back(s);
    }
    std::sort(cand.begin(), cand.end(),
              [&](int a, int b) { return slot_last[a] < slot_last[b]; });
    for (int s : cand) {
      if (step.miss_e.size() <= free_slots.size()) break;
      slot_expert[s] = -1;
      free_slots.push_back(s);
      step.evicted.push_back(s);
    }
  }
  if (step.miss_e.size() > free_slots.size())
    throw std::runtime_error("plan_slots: slot pool exhausted");

  // Assign misses from the front of free_slots, first-appearance order —
  // exactly the order plan_remap() uses. Peek (do NOT erase): the caller erases
  // after the reads are issued, and the un-erased snapshot is what plan_remap
  // consumes so the two slot plans agree.
  std::unordered_map<int32_t, int32_t> miss_slot;
  miss_slot.reserve(step.miss_e.size() * 2 + 1);
  for (size_t i = 0; i < step.miss_e.size(); ++i) {
    int32_t s = free_slots[i];
    if (s < 0 || s >= num_slots)
      throw std::runtime_error("plan_slots: free slot index out of range");
    step.miss_s.push_back(s);
    miss_slot[step.miss_e[i]] = s;
  }

  // Per-route-position slots. Residents keep their slot; misses use the slot
  // just assigned. Nothing here can be -1.
  step.slots.reserve(route_ids.size());
  for (int32_t e : route_ids) {
    auto r = resident.find(e);
    int32_t s;
    if (r != resident.end()) {
      s = r->second;  // never evicted: eviction protects route experts
    } else {
      auto m = miss_slot.find(e);
      if (m == miss_slot.end())
        throw std::runtime_error("plan_slots: internal miss lookup failed");
      s = m->second;
    }
    if (s < 0 || s >= num_slots)
      throw std::runtime_error("plan_slots: assigned slot out of range");
    step.slots.push_back(s);
  }
  return step;
}

bool verify_slots_enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("VEYSTRIX_CPP_VERIFY_SLOTS");
    return v != nullptr && std::string(v) == "1";
  }();
  return enabled;
}

void verify_slot_fill(bool enabled,
                      const std::vector<int32_t>& route_ids,
                      const std::vector<int32_t>& slots,
                      const std::vector<int32_t>& slot_expert,
                      const std::vector<int32_t>& miss_e,
                      const std::vector<int32_t>& miss_s,
                      const std::vector<int32_t>& fds,
                      const std::vector<int64_t>& tensor_off,
                      const std::vector<int64_t>& row_bytes,
                      const std::vector<uint8_t*>& bufs) {
  if (!enabled) return;  // the only cost when disabled: one branch

  if (slots.size() != route_ids.size())
    throw std::runtime_error("verify_slot_fill: slots/route size mismatch");
  for (size_t j = 0; j < route_ids.size(); ++j) {
    int32_t s = slots[j];
    if (s < 0 || s >= static_cast<int32_t>(slot_expert.size()) ||
        slot_expert[s] != route_ids[j])
      throw std::runtime_error(
          "verify_slot_fill: slot_expert[slots[j]] != route_ids[j]");
  }

  const size_t nt = fds.size();
  if (nt != tensor_off.size() || nt != row_bytes.size() || nt != bufs.size() ||
      miss_e.size() != miss_s.size())
    throw std::runtime_error("verify_slot_fill: tensor param size mismatch");

  for (size_t m = 0; m < miss_e.size(); ++m) {
    const int32_t e = miss_e[m], s = miss_s[m];
    for (size_t t = 0; t < nt; ++t) {
      const int64_t rb = row_bytes[t];
      if (rb <= 0) continue;
      std::vector<uint8_t> expected(static_cast<size_t>(rb));
      const int64_t off = tensor_off[t] + static_cast<int64_t>(e) * rb;
      const ssize_t got =
          ::pread(fds[t], expected.data(), static_cast<size_t>(rb), off);
      if (got != static_cast<ssize_t>(rb))
        throw std::runtime_error("verify_slot_fill: short pread");
      if (std::memcmp(expected.data(),
                      bufs[t] + static_cast<size_t>(s) * rb,
                      static_cast<size_t>(rb)) != 0)
        throw std::runtime_error(
            "verify_slot_fill: slot bytes differ from expected expert row");
    }
  }
}

}  // namespace veystrix::engine
