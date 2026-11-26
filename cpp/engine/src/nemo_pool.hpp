// Nemotron expert slot POOL for the C++ decode engine (research, opt-in).
//
// Goal: keep a bounded set of *whole* expert records in one anonymous,
// fixed-stride region, so a routed expert that is already resident costs no
// read and no copy. A miss fills one slot from the shard; with
// VEYSTRIX_NEMOTRON_POOL_NOCACHE=1 that fill goes through an F_NOCACHE
// descriptor, so the file cache does not keep a second copy of the bytes.
// Reads that are NOT pool fills (e.g. the direct baseline / the prefill ring)
// always use a separate, normal cached descriptor.
//
// The bytes and their order are identical in both modes; only whether macOS
// keeps a cached copy differs. This is the C++ counterpart of
// models/flashnext/expert_pool.py (the _fd / _fill_fd pair) and store.py
// (set_uncached_reads).
//
// Two properties make this safe under concurrent decode layers:
//
//   1. In-flight coalescing. A second request for the same (layer, expert)
//      while its fill is outstanding waits on the SAME shared future instead
//      of issuing a second pread. One physical read, many waiters.
//   2. Slot pinning is the caller's job (mirrors StreamingMoE): a slot stays
//      valid until the caller is done with it; acquire() never evicts a slot
//      whose key is currently in flight.
//
// Deliberately NOT implemented here (see the design note in research_notes):
//   - mlock() of the pool region (recorded problems: first-touch cost, and a
//     bf16 scale/bias part that is not 4-byte aligned gets a wrong uint32 view).
//   - Any change to existing pool sizing defaults.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>       // additive Phase 1 hook: expose the existing pool lock
#include <string>
#include <sys/types.h>  // ssize_t

namespace veystrix::engine {

// Opt-in flag. "1" enables uncached pool fills; "0" or unset keeps the
// existing cached behaviour. Any other value throws when read via
// nemo_pool_nocache_from_env().
inline constexpr const char* kNemoPoolNocacheEnv = "VEYSTRIX_NEMOTRON_POOL_NOCACHE";

// Read the opt-in flag from the environment. Default OFF. Explicit rollback
// is "0". Throws std::runtime_error on an unrecognised value (fail closed).
bool nemo_pool_nocache_from_env();

// Counters returned by stats(). Bit-exact tests assert these; the owner uses
// physical_bytes_read / physical_reads for physical-MB-per-token comparisons.
struct NemoPoolStats {
  uint64_t hits = 0;               // acquire() found the (layer,expert) resident
  uint64_t misses = 0;             // acquire() issued a fresh fill
  uint64_t coalesced_reads = 0;    // acquire() waited on another thread's fill
  uint64_t physical_reads = 0;     // fills issued (leader reads only)
  uint64_t physical_bytes_read = 0;
  uint64_t evictions = 0;
  int resident = 0;
  int slots = 0;
  bool nocache_enabled = false;    // current mode (reflects set_uncached_reads)
};

// A bounded pool of fixed-stride expert records. Not thread-hostile: acquire,
// read_direct, slot_for and stats may be called from many threads at once.
class NemoPool {
 public:
  // Where one expert record lives on disk.
  struct Record {
    std::string shard;  // path to the shard file
    int64_t offset = 0; // byte offset of the record within the shard
    int64_t length = 0; // must equal record_bytes
  };

  // Resolves (layer, expert) -> Record. Pure; called outside the pool lock.
  using Resolver = std::function<Record(int layer, int expert)>;

  // Physical read primitive. Default calls ::pread (with EINTR / short-read
  // retry in the pool). Injectable so tests can count/wrap pread, and so the
  // owner can later route fills through native_read.c's batched reader; the
  // descriptor handed to it is already the correct per-mode one.
  //   returns bytes read (== len) on success, negative on failure.
  using ReadFn = std::function<ssize_t(int fd, void* dst, size_t len, int64_t offset)>;

  NemoPool(int num_slots, int64_t record_bytes, Resolver resolver,
           bool nocache, ReadFn read_fn = {});
  ~NemoPool();

  NemoPool(const NemoPool&) = delete;
  NemoPool& operator=(const NemoPool&) = delete;
  NemoPool(NemoPool&&) noexcept;
  NemoPool& operator=(NemoPool&&) noexcept;

  // Convenience: construct with the mode taken from kNemoPoolNocacheEnv.
  static NemoPool from_env(int num_slots, int64_t record_bytes,
                           Resolver resolver, ReadFn read_fn = {});

  // Return the slot holding the record. On a miss, issue the fill (or, if a
  // fill is already in flight for this key, wait on it). Throws on I/O error.
  int acquire(int layer, int expert);

  // -1 if the key is not currently resident. Does not trigger a fill.
  int slot_for(int layer, int expert) const;

  // Byte access to a slot's record-sized backing memory. Valid until the slot
  // is evicted by a later acquire(); the caller must keep its own reference.
  uint8_t* slot_data(int slot);
  const uint8_t* slot_data(int slot) const;

  // Read a record DIRECTLY from the shard through the normal cached
  // descriptor. No pool lookup, no coalescing, no counters. This is the
  // bit-exact baseline used by the tests and the "non-pool reads stay cached"
  // path from the spec.
  void read_direct(int layer, int expert, void* dst);

  NemoPoolStats stats() const;

  // Runtime rollback / admission knob, mirroring store.py set_uncached_reads.
  // Flips which fd subsequent fills use; already-open descriptors keep their
  // flags (we keep two separate maps instead of mutating one).
  void set_uncached_reads(bool enabled);

  // Test/introspection accessors. These lazily open the requested descriptor.
  // uncached_fd() opens an O_RDONLY fd with F_NOCACHE=1, F_RDAHEAD=0;
  // cached_fd() opens a plain O_RDONLY fd.
  int cached_fd(const std::string& shard);
  int uncached_fd(const std::string& shard);

  // -----------------------------------------------------------------------
  // Additive Phase 1 hooks for SharedExpertPool. These are ADDITIONS ONLY:
  // every ported behaviour above (in-flight futures, two fd maps, LRU,
  // fail-closed) is unchanged. See docs/engineering/shared_expert_pool_phase1.md.
  //
  // Returns the pool's EXISTING bookkeeping mutex. SharedExpertPool guards its
  // pin/fence bookkeeping with this same lock; NO new mutex is introduced.
  // NEEDS OWNER APPROVAL for any additional mutex, thread, or I/O worker.
  std::mutex& mutex() const;

  // Optional eviction veto. Called with the pool mutex HELD, so the callback
  // must NOT call back into the pool or take mutex() again (it must only read
  // externally-owned state that is itself guarded by that same lock). Shared-
  // ExpertPool uses this to make pinned slots ineligible as LRU victims.
  using EvictionVeto = std::function<bool(int slot)>;  // true => keep slot
  void set_eviction_veto(EvictionVeto veto);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace veystrix::engine
