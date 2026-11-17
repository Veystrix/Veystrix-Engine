#include "io_pager.hpp"

#include <algorithm>
#include <unordered_map>

// The existing batched reader: libdispatch concurrent positioned reads with
// GIL-released semantics. Identical struct layout to PagerRead (fd, pad,
// offset, length, dst-as-uintptr). We link its source directly to stay bit-exact
// with the Python engine's reader.
extern "C" int flashnext_batch_pread(const void* reads, int32_t count, int32_t width);

namespace veystrix {

PagerPlan plan_remap(const std::vector<int32_t>& route,
                     const std::vector<int32_t>& expert_slot_pairs,
                     const std::vector<int32_t>& free_slots,
                     int32_t fd, int64_t header, int64_t row_bytes,
                     uint8_t* slot_base, int64_t slot_stride) {
  PagerPlan plan;
  plan.slots.reserve(route.size());
  std::unordered_map<int32_t, int32_t> resident;
  resident.reserve(expert_slot_pairs.size() / 2);
  // Resident list is (expert, slot) pairs.
  for (size_t i = 0; i + 1 < expert_slot_pairs.size(); i += 2) {
    resident[expert_slot_pairs[i]] = expert_slot_pairs[i + 1];
  }

  size_t next_free = 0;
  // Deduplicate misses: the same expert routed twice in one layer needs one read.
  std::unordered_map<int32_t, int32_t> assigned;
  for (int32_t expert : route) {
    auto r = resident.find(expert);
    if (r != resident.end()) {
      plan.slots.push_back(r->second);
      continue;
    }
    auto a = assigned.find(expert);
    if (a != assigned.end()) {
      plan.slots.push_back(a->second);
      continue;
    }
    // Need a fresh slot. Fail closed with -1 if the pool is exhausted; the
    // caller must fall back to the exact Python path (no partial mapping).
    if (next_free >= free_slots.size()) {
      plan.slots.push_back(-1);
      continue;
    }
    int32_t slot = free_slots[next_free++];
    assigned[expert] = slot;
    plan.slots.push_back(slot);
    plan.reads.push_back({fd, 0, header + static_cast<int64_t>(expert) * row_bytes,
                          row_bytes, reinterpret_cast<uint64_t>(slot_base + slot * slot_stride)});
  }
  return plan;
}

int run_batched_reads(const std::vector<PagerRead>& reads, int width) {
  // PagerRead must be ABI-compatible with flashnext_read:
  //   int32 fd; int32 pad; int64 offset; int64 length; uint64 dst.
  static_assert(sizeof(PagerRead) == 32, "PagerRead ABI mismatch with native_read.c");
  return flashnext_batch_pread(reads.data(), static_cast<int32_t>(reads.size()), width);
}

}  // namespace veystrix
