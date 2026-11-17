// Portable part of Method #11: a C++ expert pager that maps router-selected
// expert ids to contiguous cache slots, issues batched positioned reads via the
// existing native_read.c, and hands the bytes to MLX zero-copy.
//
// Deliberately EXCLUDES the two mechanisms already rejected on our 16 GB M4:
//   - O_DIRECT / F_NOCACHE page-cache bypass  (measured 8% slower)
//   - KICK/WAIT early overlap of shared compute with reads (measured -8.2%)
// See docs/performance/DO_NOT_IMPLEMENT.md and PORTS_SEQUENCE.md.
#pragma once
#include <cstdint>
#include <functional>
#include <vector>

namespace veystrix {

// A single expert-record read: read `length` bytes at `offset` into `dst`.
struct PagerRead {
  int32_t fd;
  int32_t pad;      // ABI padding to match flashnext_read in native_read.c
  int64_t offset;
  int64_t length;
  uint64_t dst;     // uintptr of destination buffer
};

// Result of a remap: the cache slots assigned to each routed expert (in the
// caller's route order), plus the list of records that must be read.
struct PagerPlan {
  std::vector<int32_t> slots;                 // one per route, in route order
  std::vector<PagerRead> reads;               // deduplicated misses
};

// Map routed expert ids to cache slots. `route` holds expert ids in route order.
// `resident` maps expert id -> slot for already-cached experts. For each id not
// resident, a fresh slot from `free_slots` is assigned and a read is emitted.
// Pure and deterministic: same inputs -> same plan. No GPU, no I/O.
PagerPlan plan_remap(const std::vector<int32_t>& route,
                     const std::vector<int32_t>& expert_slot_pairs,
                     const std::vector<int32_t>& free_slots,
                     int32_t fd, int64_t header, int64_t row_bytes,
                     uint8_t* slot_base, int64_t slot_stride);

// Run all reads in `plan` through the existing native batched pread (libdispatch,
// GIL-released). returns the number of failed reads (0 on success). Provided by
// native_read.c through a small C shim declared here.
int run_batched_reads(const std::vector<PagerRead>& reads, int width);

}  // namespace veystrix
