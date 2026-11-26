// Decode expert-row fetch with per-expert completion (Nemotron, VEYSTRIX_CPP_NEMO_PIPE=1).
//
// Two changes against the one-batch pread path (`rows6`), both exact by
// construction (the destination bytes are the same file bytes):
//
// 1. Rows already in the OS page cache are copied with memcpy from a read-only
//    shared mmap of the shard instead of pread. A kernel pread of cached pages
//    runs at ~29 GB/s over 8 lanes here, a user-space memcpy from mapped pages at
//    ~49 GB/s (micro-benchmark, 775 MB of Nemotron rows per token: 26.5 vs 15.7 ms).
//    Residency is sampled with mincore on the first, middle and last page of a
//    row (a full-row mincore costs ~0.45 us per page, ~15 ms per token). A row
//    sampled as not resident is read with pread, which on misses is ~2x faster
//    than page faults through the mapping (3.2 vs 1.5 GB/s). A wrong guess only
//    changes the speed of that row, never its bytes.
// 2. Every row belongs to a group (one routed expert). The caller can take each
//    group as soon as its rows have landed and launch that expert's GPU work
//    while the remaining rows (typically the page-cache misses) are still
//    being read, instead of waiting for the whole layer.
//
// Lanes run on the libdispatch user-interactive global queue, like the existing
// batched reader. While the caller waits it can keep the GPU warm
// (keepwarm::spin_once) at the usual period.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace veystrix::engine::rowfetch {

// Read-only shared mapping of the whole file behind `fd` (cached per fd for
// the life of the process). nullptr if the file cannot be mapped.
const uint8_t* shard_map(int fd);

// Hint the OS to release clean cached file pages for all mapped shards from RAM.
// Safe to call while mappings remain open.
void evict_cached_files();

// Whether resident rows are copied from the mapping (VEYSTRIX_CPP_ROWS_MMAP,
// default on; =0 reads every row with pread).
bool mmap_enabled();

// Read lanes per layer fetch: VEYSTRIX_CPP_ROWS_LANES (default 8; "auto" =
// performance-core count from hw.perflevel0.logicalcpu).
int lanes();
// VEYSTRIX_CPP_ROWS_PREAD_LANES=P (< lanes): at most P lanes in pread at once;
// a lane that meets a cold row while P preads are in flight puts it aside and
// keeps copying resident rows, then the put-aside rows are read at the end.
// 0 / unset = no cap (any lane may pread).
int pread_lanes();

struct Row {
  int fd = -1;
  const uint8_t* map = nullptr;  // shard mapping, or nullptr to always pread
  int64_t offset = 0;
  int64_t length = 0;
  uint8_t* dst = nullptr;
  int group = 0;
};

struct Stats {
  int memcpy_rows = 0;
  int pread_rows = 0;
};

// Speculative page-cache warm-up: for each row (dst ignored) whose sampled
// pages are not resident, pread it into a private scratch buffer so the bytes
// land in the OS page cache. Fire-and-forget on `width` lanes; returns false
// (and does nothing) if the previous prefetch is still running. Never touches
// model buffers, so it cannot change any value.
bool prefetch_cold(std::vector<Row> rows, int width);
struct PrefetchStats {
  long long rows_checked = 0, rows_read = 0, bytes_read = 0, skipped_busy = 0;
};
PrefetchStats prefetch_stats();

class Fetch {
 public:
  // Starts the reads immediately, in the given order, on `width` lanes.
  Fetch(std::vector<Row> rows, int groups, int width);
  ~Fetch();  // waits for every lane to finish
  Fetch(const Fetch&) = delete;
  Fetch& operator=(const Fetch&) = delete;

  // Blocks until a group whose rows have all landed has not been returned yet,
  // and returns it; returns -1 once every group was returned. While waiting it
  // queues a keep-warm spin each period when keep-warm is enabled.
  int next_ready();
  // True when every read succeeded (meaningful once next_ready returned -1).
  bool ok() const;
  // Exact OS/offset context for the first failed read, or empty when ok().
  std::string error_detail() const;
  Stats stats() const;
  // Runs fn(ok) on a background queue once every lane has finished (the GPU gate's
  // release). The rows' destination buffers must stay alive until then.
  void on_done(std::function<void(bool)> fn);

 private:
  struct State;
  std::shared_ptr<State> st_;
};

}  // namespace veystrix::engine::rowfetch
