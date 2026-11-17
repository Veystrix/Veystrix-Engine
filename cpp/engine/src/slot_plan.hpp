// Pure-CPU slot bookkeeping shared by the streaming MoE paths.
//
// This is the single source of truth for deciding WHICH SLOT each routed
// expert occupies for one decode step. It contains no MLX calls and no pager
// I/O; the caller still uses plan_remap() to emit the actual reads. The point
// is that the (buggy) callers used to (1) count misses by searching a FLATTENED
// [expert, slot, ...] list, so a small expert id matched a slot number, and
// (2) derive (expert, slot) misses from read offsets across all 9 tensors, so
// a tensor-aligned offset in another shard produced an out-of-range slot.
// Both are replaced by this deterministic, fail-closed planner.
#pragma once
#include <cstdint>
#include <vector>

namespace veystrix::engine {

struct SlotStep {
  std::vector<int32_t> slots;    // one per route position, router-rank order
  std::vector<int32_t> miss_e;   // distinct missed experts, first-appearance order
  std::vector<int32_t> miss_s;   // their assigned slots (parallel to miss_e)
  std::vector<int32_t> evicted;  // slots evicted this step
};

// Plan one step. Mutates slot_expert (evictions set -1) and free_slots
// (evictions are appended); miss assignment is LEFT to the caller via
// step.miss_e / step.miss_s so the reads can be issued first. Uses the same
// LRU eviction rule as StreamingMoE: never evict a slot with ref > 0 or one
// whose expert is needed by this route. Throws std::runtime_error when the
// pool cannot hold the step. Never returns -1 or an out-of-range slot.
SlotStep plan_slots(const std::vector<int32_t>& route_ids,
                    std::vector<int32_t>& slot_expert,
                    const std::vector<int>& slot_ref,
                    const std::vector<uint64_t>& slot_last,
                    std::vector<int32_t>& free_slots, int num_slots);

// Debug verifier gate: true iff VEYSTRIX_CPP_VERIFY_SLOTS=1. Cached in a
// function-local static, so the hot path is a single branch.
bool verify_slots_enabled();

// Debug-only end-to-end verification. No-op (single early-return branch) when
// `enabled` is false. When enabled it (a) checks slot_expert[slots[j]] ==
// route_ids[j] for every route position, and (b) for each filled slot re-reads
// the expected expert row of every tensor and memcmps it against the slot
// bytes. Throws std::runtime_error on any mismatch. fds/tensor_off/row_bytes/
// bufs are parallel, one entry per tensor (all 9 expert tensors).
void verify_slot_fill(bool enabled,
                      const std::vector<int32_t>& route_ids,
                      const std::vector<int32_t>& slots,
                      const std::vector<int32_t>& slot_expert,
                      const std::vector<int32_t>& miss_e,
                      const std::vector<int32_t>& miss_s,
                      const std::vector<int32_t>& fds,
                      const std::vector<int64_t>& tensor_off,
                      const std::vector<int64_t>& row_bytes,
                      const std::vector<uint8_t*>& bufs);

}  // namespace veystrix::engine
