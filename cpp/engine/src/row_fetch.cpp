#include "row_fetch.hpp"

#include <dispatch/dispatch.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include "gpu_keepwarm.hpp"

namespace veystrix::engine::rowfetch {

namespace {
constexpr uintptr_t kPage = 16384;

struct MapRegistry {
  std::mutex mu;
  std::map<int, std::tuple<dev_t, ino_t, off_t, const uint8_t*>> maps;
};
MapRegistry& map_registry() {
  static MapRegistry r;
  return r;
}

bool page_resident(const uint8_t* p) {
  char v = 0;
  void* a = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(p) & ~(kPage - 1));
  return ::mincore(a, 1, &v) == 0 && (v & 1);
}

// First, middle and last page of the row.
bool sampled_resident(const uint8_t* p, int64_t n) {
  return page_resident(p) && page_resident(p + n / 2) && page_resident(p + n - 1);
}

bool pread_all(int fd, uint8_t* dst, int64_t n, int64_t off, int* err_out = nullptr) {
  int64_t done = 0;
  while (done < n) {
    const ssize_t got = ::pread(fd, dst + done, static_cast<size_t>(n - done), static_cast<off_t>(off + done));
    if (got > 0) done += got;
    else if (got < 0 && errno == EINTR) continue;
    else {
      if (err_out) *err_out = (got == 0) ? EIO : errno;
      return false;
    }
  }
  return true;
}
}  // namespace

const uint8_t* shard_map(int fd) {
  // Cached per fd AND file identity: a process that closes a checkpoint and opens
  // another can get the same fd number for a different file (the generic port-check
  // does), and a stale mapping would copy the old file's bytes. A replaced mapping
  // is left mapped (never unmapped under a reader); it only costs address space.
  struct stat st {};
  if (::fstat(fd, &st) != 0) return nullptr;
  auto& reg = map_registry();
  std::lock_guard<std::mutex> lk(reg.mu);
  auto it = reg.maps.find(fd);
  if (it != reg.maps.end() && std::get<0>(it->second) == st.st_dev &&
      std::get<1>(it->second) == st.st_ino && std::get<2>(it->second) == st.st_size)
    return std::get<3>(it->second);
  const uint8_t* m = nullptr;
  if (st.st_size > 0) {
    void* p = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
    if (p != MAP_FAILED) m = static_cast<const uint8_t*>(p);
  }
  reg.maps[fd] = std::make_tuple(st.st_dev, st.st_ino, st.st_size, m);
  return m;
}

void evict_cached_files() {
  std::vector<std::pair<const uint8_t*, off_t>> entries;
  {
    auto& reg = map_registry();
    std::lock_guard<std::mutex> lk(reg.mu);
    entries.reserve(reg.maps.size());
    for (const auto& kv : reg.maps) {
      const uint8_t* m = std::get<3>(kv.second);
      const off_t sz = std::get<2>(kv.second);
      if (m != nullptr && sz > 0) entries.emplace_back(m, sz);
    }
  }
  for (const auto& e : entries) {
    (void)::madvise(const_cast<uint8_t*>(e.first), static_cast<size_t>(e.second), MADV_DONTNEED);
  }
}

int lanes() {
  static const int n = [] {
    const char* e = std::getenv("VEYSTRIX_CPP_ROWS_LANES");
    if (e && std::strcmp(e, "auto") == 0) {
      int v = 0;
      size_t len = sizeof(v);
      if (::sysctlbyname("hw.perflevel0.logicalcpu", &v, &len, nullptr, 0) == 0 && v > 0) return v;
      return 8;
    }
    const int v = e ? std::atoi(e) : 0;
    return v > 0 ? v : 8;
  }();
  return n;
}

int pread_lanes() {
  static const int n = [] {
    const char* e = std::getenv("VEYSTRIX_CPP_ROWS_PREAD_LANES");
    const int v = e ? std::atoi(e) : 0;
    return v > 0 ? v : 0;
  }();
  return n;
}

bool mmap_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VEYSTRIX_CPP_ROWS_MMAP");
    return !(e && std::strcmp(e, "0") == 0);
  }();
  return on;
}

namespace {
struct PrefetchState {
  std::atomic<bool> busy{false};
  std::atomic<long long> checked{0}, read{0}, bytes{0}, skipped{0};
};
PrefetchState& pstate() {
  static PrefetchState s;
  return s;
}
}  // namespace

bool prefetch_cold(std::vector<Row> rows, int width) {
  PrefetchState& ps = pstate();
  if (rows.empty()) return true;
  bool expected = false;
  if (!ps.busy.compare_exchange_strong(expected, true)) {
    ps.skipped.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  struct Batch {
    std::vector<Row> rows;
    std::atomic<int> next{0};
    std::atomic<int> lanes_left{0};
  };
  auto b = std::make_shared<Batch>();
  b->rows = std::move(rows);
  if (width < 1) width = 1;
  if (width > static_cast<int>(b->rows.size())) width = static_cast<int>(b->rows.size());
  b->lanes_left.store(width);
  dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
  for (int lane = 0; lane < width; ++lane) {
    dispatch_async(q, ^{
      Batch& bb = *b;
      PrefetchState& st = pstate();
      static thread_local std::vector<uint8_t> scratch;
      for (;;) {
        const int i = bb.next.fetch_add(1, std::memory_order_relaxed);
        if (i >= static_cast<int>(bb.rows.size())) break;
        const Row& r = bb.rows[static_cast<size_t>(i)];
        st.checked.fetch_add(1, std::memory_order_relaxed);
        if (r.map && sampled_resident(r.map + r.offset, r.length)) continue;
        if (scratch.size() < static_cast<size_t>(r.length)) scratch.resize(static_cast<size_t>(r.length));
        if (pread_all(r.fd, scratch.data(), r.length, r.offset)) {
          st.read.fetch_add(1, std::memory_order_relaxed);
          st.bytes.fetch_add(r.length, std::memory_order_relaxed);
        }
      }
      if (bb.lanes_left.fetch_sub(1) == 1) st.busy.store(false, std::memory_order_release);
    });
  }
  return true;
}

PrefetchStats prefetch_stats() {
  PrefetchState& ps = pstate();
  return {ps.checked.load(), ps.read.load(), ps.bytes.load(), ps.skipped.load()};
}

struct Fetch::State {
  std::vector<Row> rows;
  std::vector<std::atomic<int>> left;  // rows still in flight per group
  std::atomic<int> next{0};
  std::atomic<int> errors{0};
  std::mutex err_mu;
  int first_errno = 0;
  int failed_fd = -1;
  int64_t failed_offset = 0;
  int64_t failed_length = 0;
  std::atomic<int> memcpy_rows{0};
  std::atomic<int> pread_rows{0};
  std::atomic<int> preads_in_flight{0};
  int pread_cap = 0;               // 0 = no cap
  std::mutex defer_mu;
  std::vector<int> deferred;       // cold rows put aside while the pread cap was full
  std::mutex mu;
  std::condition_variable cv;
  std::vector<int> ready;   // groups completed, in completion order (under mu)
  size_t handed = 0;        // groups returned by next_ready (driver only)
  int groups = 0;
  dispatch_group_t lanes = nullptr;
  explicit State(int g) : left(static_cast<size_t>(g)), groups(g) {}
};

Fetch::Fetch(std::vector<Row> rows, int groups, int width) : st_(std::make_shared<State>(groups)) {
  State& s = *st_;
  s.rows = std::move(rows);
  for (auto& a : s.left) a.store(0, std::memory_order_relaxed);
  for (const Row& r : s.rows) s.left[static_cast<size_t>(r.group)].fetch_add(1, std::memory_order_relaxed);
  // Groups with no rows are ready at once.
  for (int g = 0; g < groups; ++g)
    if (s.left[static_cast<size_t>(g)].load(std::memory_order_relaxed) == 0) s.ready.push_back(g);
  if (width < 1) width = 1;
  if (width > static_cast<int>(s.rows.size())) width = static_cast<int>(s.rows.size());
  s.pread_cap = pread_lanes() < width ? pread_lanes() : 0;
  s.lanes = dispatch_group_create();
  dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
  std::shared_ptr<State> keep = st_;
  for (int lane = 0; lane < width; ++lane) {
    dispatch_group_async(s.lanes, q, ^{
      State& b = *keep;
      bool draining = false;
      for (;;) {
        int i;
        if (!draining) {
          i = b.next.fetch_add(1, std::memory_order_relaxed);
          if (i >= static_cast<int>(b.rows.size())) {
            draining = true;
            continue;
          }
        } else {
          // VEYSTRIX_CPP_ROWS_PREAD_LANES: cold rows deferred while the cap
          // was full are read once no unclassified rows are left.
          std::lock_guard<std::mutex> lk(b.defer_mu);
          if (b.deferred.empty()) break;
          i = b.deferred.back();
          b.deferred.pop_back();
        }
        const Row& r = b.rows[static_cast<size_t>(i)];
        bool good = true;
        if (!draining && r.map && sampled_resident(r.map + r.offset, r.length)) {
          std::memcpy(r.dst, r.map + r.offset, static_cast<size_t>(r.length));
          b.memcpy_rows.fetch_add(1, std::memory_order_relaxed);
        } else {
          if (!draining && b.pread_cap > 0) {
            if (b.preads_in_flight.fetch_add(1, std::memory_order_acq_rel) >= b.pread_cap) {
              b.preads_in_flight.fetch_sub(1, std::memory_order_acq_rel);
              std::lock_guard<std::mutex> lk(b.defer_mu);
              b.deferred.push_back(i);
              continue;
            }
          }
          int read_err = 0;
          good = pread_all(r.fd, r.dst, r.length, r.offset, &read_err);
          if (!draining && b.pread_cap > 0) b.preads_in_flight.fetch_sub(1, std::memory_order_acq_rel);
          b.pread_rows.fetch_add(1, std::memory_order_relaxed);
          if (!good) {
            std::lock_guard<std::mutex> lk(b.err_mu);
            if (b.first_errno == 0) {
              b.first_errno = read_err ? read_err : EIO;
              b.failed_fd = r.fd;
              b.failed_offset = r.offset;
              b.failed_length = r.length;
            }
          }
        }
        if (!good) b.errors.fetch_add(1, std::memory_order_relaxed);
        if (b.left[static_cast<size_t>(r.group)].fetch_sub(1, std::memory_order_acq_rel) == 1) {
          {
            std::lock_guard<std::mutex> lk(b.mu);  // release edge for the row bytes
            b.ready.push_back(r.group);
          }
          b.cv.notify_one();
        }
      }
    });
  }
}

Fetch::~Fetch() {
  if (st_ && st_->lanes) {
    dispatch_group_wait(st_->lanes, DISPATCH_TIME_FOREVER);
    dispatch_release(st_->lanes);
    st_->lanes = nullptr;
  }
}

int Fetch::next_ready() {
  State& s = *st_;
  if (s.handed >= static_cast<size_t>(s.groups)) return -1;
  static const long period_us = [] {
    const char* e = std::getenv("VEYSTRIX_CPP_KEEPWARM_PERIOD_US");
    const long v = e ? std::atol(e) : 500;
    return v > 0 ? v : 500;
  }();
  const bool warm = keepwarm::enabled();
  // Same rhythm as keepwarm::wait_ticket_warm: if nothing is ready, queue one
  // spin, then wait up to one period, and repeat.
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(s.mu);
      if (s.ready.size() > s.handed) return s.ready[s.handed++];
    }
    if (warm) keepwarm::spin_once();
    std::unique_lock<std::mutex> lk(s.mu);
    if (s.cv.wait_for(lk, std::chrono::microseconds(period_us), [&] { return s.ready.size() > s.handed; }))
      return s.ready[s.handed++];
  }
}

void Fetch::on_done(std::function<void(bool)> fn) {
  std::shared_ptr<State> keep = st_;
  dispatch_group_notify(st_->lanes, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
    fn(keep->errors.load(std::memory_order_acquire) == 0);
  });
}

bool Fetch::ok() const { return st_->errors.load(std::memory_order_acquire) == 0; }

std::string Fetch::error_detail() const {
  const int count = st_->errors.load(std::memory_order_acquire);
  if (count == 0) return "";
  std::lock_guard<std::mutex> lk(st_->err_mu);
  return "pread failed on fd=" + std::to_string(st_->failed_fd) +
         " offset=" + std::to_string(st_->failed_offset) +
         " len=" + std::to_string(st_->failed_length) +
         " errno=" + std::to_string(st_->first_errno) +
         " (" + std::strerror(st_->first_errno) + "), failures=" + std::to_string(count);
}

Stats Fetch::stats() const {
  return {st_->memcpy_rows.load(std::memory_order_relaxed), st_->pread_rows.load(std::memory_order_relaxed)};
}

}  // namespace veystrix::engine::rowfetch
