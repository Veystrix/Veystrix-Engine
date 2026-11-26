#include "nemo_pool.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <future>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace veystrix::engine {

namespace {

// Default physical read: positioned read with EINTR / short-read retry, like
// native_read.c's per-record loop. Returns `len` on success, -1 on failure.
ssize_t default_pread(int fd, void* dst, size_t len, int64_t offset) {
  uint8_t* p = static_cast<uint8_t*>(dst);
  size_t done = 0;
  while (done < len) {
    ssize_t got = ::pread(fd, p + done, len - done,
                          static_cast<off_t>(offset + static_cast<int64_t>(done)));
    if (got > 0) {
      done += static_cast<size_t>(got);
    } else if (got < 0 && errno == EINTR) {
      continue;
    } else {
      return -1;
    }
  }
  return static_cast<ssize_t>(len);
}

int64_t encode_key(int layer, int expert) {
  return (static_cast<int64_t>(layer) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(expert));
}

}  // namespace

bool nemo_pool_nocache_from_env() {
  const char* v = std::getenv(kNemoPoolNocacheEnv);
  if (v == nullptr || std::strcmp(v, "0") == 0) return false;
  if (std::strcmp(v, "1") == 0) return true;
  throw std::runtime_error(std::string(kNemoPoolNocacheEnv) + " must be 0 or 1");
}

// ---------------------------------------------------------------------------
struct NemoPool::Impl {
  int num_slots;
  int64_t record_bytes;
  Resolver resolver;
  ReadFn read_fn;
  bool nocache = false;

  // Two fd maps, mirroring expert_pool.py's _fds (uncached) and the store's
  // cached descriptors. Fills use one, direct/non-pool reads always use the
  // other. Keeping them separate means a cached read of a reused expert is
  // never contaminated by F_NOCACHE.
  std::mutex fd_mu;
  std::unordered_map<std::string, int> cached_fds;
  std::unordered_map<std::string, int> uncached_fds;

  // One fixed-stride buffer; slot s's record lives at buf + s*record_bytes.
  std::shared_ptr<uint8_t> buf;

  // Bookkeeping guarded by mu.
  std::mutex mu;
  std::unordered_map<int64_t, int> slot_of;         // key -> slot
  std::vector<int64_t> key_at;                      // slot -> key (-1 = free)
  std::vector<uint64_t> last_used;                  // slot -> LRU tick
  std::vector<int> free_slots;
  uint64_t tick = 0;

  // Additive Phase 1 hook. Guarded by mu; invoked only from allocate_slot_locked
  // while mu is held (callback must not re-enter the pool or take mu again).
  EvictionVeto eviction_veto;

  struct InFlight {
    std::shared_future<int> future;
  };
  std::unordered_map<int64_t, std::shared_ptr<InFlight>> inflight;

  // Counters.
  uint64_t hits = 0, misses = 0, coalesced = 0;
  uint64_t physical_reads = 0, physical_bytes = 0, evictions = 0;
  bool last_fill_used_uncached = false;

  Impl(int slots, int64_t rb, Resolver r, bool nc, ReadFn rf)
      : num_slots(slots), record_bytes(rb), resolver(std::move(r)),
        read_fn(std::move(rf)), nocache(nc) {
    if (num_slots < 1) throw std::runtime_error("nemo pool needs >= 1 slot");
    if (record_bytes < 4 || record_bytes % 4 != 0)
      throw std::runtime_error("nemo pool record_bytes must be a positive multiple of 4");
    if (!resolver) throw std::runtime_error("nemo pool needs a record resolver");
    if (!read_fn) read_fn = default_pread;

    // Anonymous private mapping: zero-filled by the kernel (same contents the
    // old posix_memalign+memset produced) but pages are committed lazily on
    // first fill. The memset touched every page up front, which for a big
    // (resident) pool pre-faulted the whole region and spiked RSS/swap before a
    // single expert was read. Page-aligned, so the 64 B alignment still holds.
    const size_t bytes =
        static_cast<size_t>(num_slots) * static_cast<size_t>(record_bytes);
    void* raw = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                       MAP_ANON | MAP_PRIVATE, -1, 0);
    if (raw == MAP_FAILED)
      throw std::runtime_error("mmap failed for nemo pool buffer");
    buf = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(raw),
                                   [bytes](uint8_t* p) { ::munmap(p, bytes); });

    key_at.assign(num_slots, -1);
    last_used.assign(num_slots, 0);
    for (int s = 0; s < num_slots; ++s) free_slots.push_back(s);
  }

  ~Impl() {
    for (auto& kv : cached_fds)
      if (kv.second >= 0) ::close(kv.second);
    for (auto& kv : uncached_fds)
      if (kv.second >= 0) ::close(kv.second);
  }

  int open_fd(const std::string& shard, bool uncached_mode) {
    int fd = ::open(shard.c_str(), O_RDONLY);
    if (fd < 0)
      throw std::runtime_error("cannot open shard for nemo pool: " + shard);
    if (uncached_mode) {
#ifdef F_NOCACHE
      if (::fcntl(fd, F_NOCACHE, 1) != 0) {
        ::close(fd);
        throw std::runtime_error("F_NOCACHE failed on shard: " + shard);
      }
#else
      ::close(fd);
      throw std::runtime_error("F_NOCACHE unavailable on this platform");
#endif
#ifdef F_RDAHEAD
      // Best effort, matching expert_pool.py (it tolerates the fcntl failing).
      ::fcntl(fd, F_RDAHEAD, 0);
#endif
    }
    return fd;
  }

  int get_cached_fd(const std::string& shard) {
    std::lock_guard<std::mutex> lk(fd_mu);
    auto it = cached_fds.find(shard);
    if (it != cached_fds.end()) return it->second;
    int fd = open_fd(shard, /*uncached_mode=*/false);
    cached_fds.emplace(shard, fd);
    return fd;
  }

  int get_uncached_fd(const std::string& shard) {
    std::lock_guard<std::mutex> lk(fd_mu);
    auto it = uncached_fds.find(shard);
    if (it != uncached_fds.end()) return it->second;
    int fd = open_fd(shard, /*uncached_mode=*/true);
    uncached_fds.emplace(shard, fd);
    return fd;
  }

  // Allocate a free (or LRU victim) slot. Caller holds mu.
  int allocate_slot_locked(int64_t key) {
    if (!free_slots.empty()) {
      int s = free_slots.back();
      free_slots.pop_back();
      return s;
    }
    int victim = -1;
    uint64_t oldest = ~uint64_t{0};
    for (int s = 0; s < num_slots; ++s) {
      if (last_used[s] >= oldest) continue;
      // Never evict a slot whose fill is still in flight.
      if (inflight.find(key_at[s]) != inflight.end()) continue;
      // Additive Phase 1: never evict a pinned slot. The veto is called with
      // mu held and reads pin state guarded by that same lock (no new mutex).
      if (eviction_veto && eviction_veto(s)) continue;
      oldest = last_used[s];
      victim = s;
    }
    if (victim < 0) throw std::runtime_error("nemo pool exhausted; refusing partial remap");
    slot_of.erase(key_at[victim]);
    key_at[victim] = -1;
    ++evictions;
    return victim;
  }

  // Loop the read primitive until the whole record is in `dst`.
  void read_record_into(int fd, const Record& rec, uint8_t* dst) {
    size_t done = 0;
    while (done < static_cast<size_t>(rec.length)) {
      errno = 0;
      ssize_t got = read_fn(fd, dst + done,
                            static_cast<size_t>(rec.length) - done,
                            rec.offset + static_cast<int64_t>(done));
      if (got > 0) {
        done += static_cast<size_t>(got);
      } else if (got < 0 && errno == EINTR) {
        continue;
      } else {
        throw std::runtime_error("short/failed nemo pool read for shard: " + rec.shard);
      }
    }
  }
};

// ---------------------------------------------------------------------------
NemoPool::NemoPool(int num_slots, int64_t record_bytes, Resolver resolver,
                   bool nocache, ReadFn read_fn)
    : impl_(std::make_unique<Impl>(num_slots, record_bytes, std::move(resolver),
                                   nocache, std::move(read_fn))) {}

NemoPool::~NemoPool() = default;
NemoPool::NemoPool(NemoPool&&) noexcept = default;
NemoPool& NemoPool::operator=(NemoPool&&) noexcept = default;

NemoPool NemoPool::from_env(int num_slots, int64_t record_bytes,
                            Resolver resolver, ReadFn read_fn) {
  return NemoPool(num_slots, record_bytes, std::move(resolver),
                  nemo_pool_nocache_from_env(), std::move(read_fn));
}

int NemoPool::acquire(int layer, int expert) {
  Impl& s = *impl_;
  const int64_t key = encode_key(layer, expert);

  std::promise<int> prom;
  std::shared_future<int> fut;
  int my_slot = -1;
  Record rec;
  {
    std::lock_guard<std::mutex> lk(s.mu);

    // 1. In-flight check MUST precede the resident check: a key is inserted
    //    into slot_of at fill start, so a same-key request arriving mid-fill
    //    must coalesce rather than be mistaken for a hit on half-written data.
    auto ii = s.inflight.find(key);
    if (ii != s.inflight.end()) {
      fut = ii->second->future;  // copy the shared future; wait outside the lock
      ++s.coalesced;
    } else {
      auto it = s.slot_of.find(key);
      if (it != s.slot_of.end()) {
        ++s.hits;
        s.last_used[it->second] = ++s.tick;
        return it->second;
      }
      // New miss: become the leader.
      rec = s.resolver(layer, expert);
      if (rec.length != s.record_bytes)
        throw std::runtime_error("nemo pool record length mismatch");
      my_slot = s.allocate_slot_locked(key);
      s.slot_of[key] = my_slot;
      s.key_at[my_slot] = key;
      s.last_used[my_slot] = ++s.tick;
      ++s.misses;

      fut = prom.get_future().share();
      auto inf = std::make_shared<Impl::InFlight>();
      inf->future = fut;
      s.inflight[key] = std::move(inf);
    }
  }

  if (my_slot < 0) {
    // Coalesced waiter: blocks on the leader's single physical read.
    int rc = fut.get();
    if (rc != 0) throw std::runtime_error("nemo pool fill failed (waited)");
    std::lock_guard<std::mutex> lk(s.mu);
    auto it = s.slot_of.find(key);
    if (it == s.slot_of.end()) throw std::runtime_error("nemo pool fill lost its slot");
    s.last_used[it->second] = ++s.tick;
    return it->second;
  }

  // Leader: perform the one physical read, then publish to waiters.
  int rc = 0;
  bool uncached;
  {
    std::lock_guard<std::mutex> lk(s.mu);
    uncached = s.nocache;
  }
  try {
    int fd = uncached ? s.get_uncached_fd(rec.shard) : s.get_cached_fd(rec.shard);
    s.read_record_into(fd, rec, s.buf.get() + static_cast<size_t>(my_slot) * s.record_bytes);
  } catch (const std::exception&) {
    rc = -1;
  }
  {
    std::lock_guard<std::mutex> lk(s.mu);
    if (rc == 0) {
      ++s.physical_reads;
      s.physical_bytes += static_cast<uint64_t>(rec.length);
      s.last_fill_used_uncached = uncached;
    } else {
      // Invalidate exactly like expert_pool.invalidate(): a failed fill's slot
      // returns to the free list, never to the hit path.
      auto it = s.slot_of.find(key);
      if (it != s.slot_of.end() && it->second == my_slot) {
        s.slot_of.erase(it);
        s.key_at[my_slot] = -1;
        s.free_slots.push_back(my_slot);
      }
    }
    s.inflight.erase(key);
  }
  prom.set_value(rc);  // release every coalesced waiter
  if (rc != 0) throw std::runtime_error("nemo pool fill failed for shard: " + rec.shard);
  return my_slot;
}

int NemoPool::slot_for(int layer, int expert) const {
  Impl& s = *impl_;
  std::lock_guard<std::mutex> lk(s.mu);
  auto it = s.slot_of.find(encode_key(layer, expert));
  return it == s.slot_of.end() ? -1 : it->second;
}

uint8_t* NemoPool::slot_data(int slot) {
  return impl_->buf.get() + static_cast<size_t>(slot) * impl_->record_bytes;
}

const uint8_t* NemoPool::slot_data(int slot) const {
  return impl_->buf.get() + static_cast<size_t>(slot) * impl_->record_bytes;
}

void NemoPool::read_direct(int layer, int expert, void* dst) {
  Impl& s = *impl_;
  Record rec = s.resolver(layer, expert);
  if (rec.length != s.record_bytes)
    throw std::runtime_error("nemo pool record length mismatch (direct)");
  int fd = s.get_cached_fd(rec.shard);  // non-pool reads are always cached
  s.read_record_into(fd, rec, static_cast<uint8_t*>(dst));
}

NemoPoolStats NemoPool::stats() const {
  Impl& s = *impl_;
  NemoPoolStats out;
  std::lock_guard<std::mutex> lk(s.mu);
  out.hits = s.hits;
  out.misses = s.misses;
  out.coalesced_reads = s.coalesced;
  out.physical_reads = s.physical_reads;
  out.physical_bytes_read = s.physical_bytes;
  out.evictions = s.evictions;
  out.resident = static_cast<int>(s.slot_of.size());
  out.slots = s.num_slots;
  out.nocache_enabled = s.nocache;
  return out;
}

void NemoPool::set_uncached_reads(bool enabled) {
  Impl& s = *impl_;
  std::lock_guard<std::mutex> lk(s.mu);
  s.nocache = enabled;
}

int NemoPool::cached_fd(const std::string& shard) { return impl_->get_cached_fd(shard); }
int NemoPool::uncached_fd(const std::string& shard) { return impl_->get_uncached_fd(shard); }

// ---- Additive Phase 1 hooks (see header for the no-new-mutex contract) ----
std::mutex& NemoPool::mutex() const { return impl_->mu; }

void NemoPool::set_eviction_veto(EvictionVeto veto) {
  Impl& s = *impl_;
  std::lock_guard<std::mutex> lk(s.mu);
  s.eviction_veto = std::move(veto);
}

}  // namespace veystrix::engine
