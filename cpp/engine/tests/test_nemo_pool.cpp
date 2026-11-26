// Bit-exact unit test for the Nemotron expert slot pool (no checkpoint, no
// SSD benchmark, no model run). The test writes a tiny deterministic shard
// into a temp dir and checks:
//   (a) gathered output through the pool == direct-read baseline, byte-exact
//       via mx::view -> uint32 (the repo standard), with F_NOCACHE applied;
//   (b) N concurrent requests for one (layer,expert) issue ONE physical pread
//       and every waiter receives identical bytes (in-flight coalescing);
//   (c) counters (hits / misses / coalesced / physical bytes) are exact;
//   (d) flag off uses the normal cached descriptor, not the F_NOCACHE one.
#include "nemo_pool.hpp"

#include <mlx/mlx.h>

#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace mx = mlx::core;
using namespace veystrix::engine;

namespace {

constexpr int kRecordBytes = 1024;   // multiple of 4, so uint32 view is exact
constexpr int kLayers = 2;
constexpr int kExperts = 8;
constexpr uint64_t kSeed = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t kMul = 6364136223846793005ULL;
constexpr uint64_t kAdd = 1442695040888963407ULL;

int g_failures = 0;
void check(bool ok, const char* what) {
  if (ok) {
    std::printf("  ok: %s\n", what);
  } else {
    std::fprintf(stderr, "  FAIL: %s\n", what);
    ++g_failures;
  }
}

// Plain positioned read for injected ReadFns (records are small, one pread).
ssize_t plain_pread(int fd, void* dst, size_t len, int64_t off) {
  return ::pread(fd, dst, len, static_cast<off_t>(off));
}

std::filesystem::path make_temp_dir() {
  std::string tmpl = (std::filesystem::temp_directory_path() / "nemo_pool_XXXXXX").string();
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  char* d = ::mkdtemp(buf.data());
  if (d == nullptr) throw std::runtime_error("mkdtemp failed");
  return std::filesystem::path(d);
}

// Same LCG as tools/gen_nemo_pool_vectors.py. Byte for byte deterministic.
void write_shard(const std::string& path) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot create shard: " + path);
  uint64_t st = kSeed;
  std::vector<uint8_t> rec(kRecordBytes);
  for (int l = 0; l < kLayers; ++l) {
    for (int e = 0; e < kExperts; ++e) {
      for (int i = 0; i < kRecordBytes; ++i) {
        st = st * kMul + kAdd;
        rec[i] = static_cast<uint8_t>((st >> 33) & 0xff);
      }
      f.write(reinterpret_cast<const char*>(rec.data()), kRecordBytes);
    }
  }
  f.close();
}

NemoPool::Resolver make_resolver(const std::string& shard) {
  return [shard](int layer, int expert) {
    return NemoPool::Record{shard,
                            static_cast<int64_t>(layer * kExperts + expert) * kRecordBytes,
                            kRecordBytes};
  };
}

// True byte-for-byte compare using the repo-standard mx::view -> uint32 trick.
bool bit_equal_bytes(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  if (a.size() != b.size()) return false;
  if (a.size() % 4 != 0) return false;
  mx::array aa(a.data(), mx::Shape{static_cast<int>(a.size())}, mx::uint8);
  mx::array bb(b.data(), mx::Shape{static_cast<int>(b.size())}, mx::uint8);
  mx::array av = mx::view(mx::reshape(aa, {-1}), mx::uint32);
  mx::array bv = mx::view(mx::reshape(bb, {-1}), mx::uint32);
  mx::eval(av, bv);
  if (av.size() != bv.size()) return false;
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  for (size_t i = 0; i < av.size(); ++i)
    if (pa[i] != pb[i]) return false;
  return true;
}

std::vector<uint8_t> direct_records(NemoPool& pool,
                                    const std::vector<std::pair<int, int>>& routes) {
  std::vector<uint8_t> out(routes.size() * kRecordBytes);
  for (size_t i = 0; i < routes.size(); ++i)
    pool.read_direct(routes[i].first, routes[i].second, out.data() + i * kRecordBytes);
  return out;
}

std::vector<uint8_t> pooled_records(NemoPool& pool,
                                    const std::vector<std::pair<int, int>>& routes) {
  std::vector<uint8_t> out(routes.size() * kRecordBytes);
  for (size_t i = 0; i < routes.size(); ++i) {
    int slot = pool.acquire(routes[i].first, routes[i].second);
    std::memcpy(out.data() + i * kRecordBytes, pool.slot_data(slot), kRecordBytes);
  }
  return out;
}

// (a) Bit-exact gather with F_NOCACHE enabled, plus proof the two fd modes exist.
void test_gather_bit_exact(const std::string& shard) {
  std::printf("[a] bit-exact gather with VEYSTRIX_NEMOTRON_POOL_NOCACHE=1\n");
  auto resolver = make_resolver(shard);
  NemoPool pool(8, kRecordBytes, resolver, /*nocache=*/true);
  std::vector<std::pair<int, int>> routes = {
      {0, 0}, {1, 7}, {0, 4}, {1, 1}, {0, 0}, {0, 3}, {1, 7}, {1, 2}};

  std::vector<uint8_t> got = pooled_records(pool, routes);
  std::vector<uint8_t> want = direct_records(pool, routes);
  check(bit_equal_bytes(got, want), "pooled gather == direct baseline (uint32 view)");

  int ufd = pool.uncached_fd(shard);
  int cfd = pool.cached_fd(shard);
  check(ufd >= 0 && cfd >= 0 && ufd != cfd, "two distinct per-mode fds opened");
#ifdef F_NOCACHE
  // Darwin's F_NOCACHE fcntl is a setter that returns 0 on success; there is
  // no getter. open_fd() already threw if applying F_NOCACHE=1 failed, so a
  // successful uncached_fd() plus a successful re-apply is the proof here.
  check(::fcntl(ufd, F_NOCACHE, 1) == 0, "fill fd accepts F_NOCACHE=1 (applied at open)");
  check(::fcntl(cfd, F_NOCACHE, 0) == 0, "cached fd never had F_NOCACHE applied");
#endif
  NemoPoolStats st = pool.stats();
  check(st.nocache_enabled, "stats reports nocache mode on");
}

// (b) N concurrent requests for one record: one pread, identical bytes.
void test_coalescing(const std::string& shard) {
  std::printf("[b] in-flight read coalescing (N concurrent identical requests)\n");
  constexpr int N = 8;
  std::atomic<int> calls{0};
  NemoPool* poolp = nullptr;  // set after construction; the ReadFn runs later

  // The leader's read blocks until every other thread has registered as a
  // coalesced waiter. This removes the "arrived after the fill finished" race
  // so the test is deterministic rather than timing-dependent. A timeout in
  // the spin keeps a broken pool from hanging the test.
  NemoPool::ReadFn counted = [&](int fd, void* dst, size_t len, int64_t off) -> ssize_t {
    calls.fetch_add(1);
    if (calls.load() == 1) {
      auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (poolp->stats().coalesced_reads < static_cast<uint64_t>(N - 1) &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
      }
    }
    return plain_pread(fd, dst, len, off);
  };

  auto resolver = make_resolver(shard);
  NemoPool pool(8, kRecordBytes, resolver, /*nocache=*/true, counted);
  poolp = &pool;
  std::barrier sync(N);
  std::vector<int> slots(N, -2);
  std::vector<std::thread> threads;
  for (int i = 0; i < N; ++i) {
    threads.emplace_back([&, i] {
      sync.arrive_and_wait();
      slots[i] = pool.acquire(1, 5);
    });
  }
  for (auto& t : threads) t.join();

  check(calls.load() == 1, "underlying pread invoked exactly once");
  bool same_slot = true;
  for (int i = 1; i < N; ++i) if (slots[i] != slots[0]) same_slot = false;
  check(same_slot && slots[0] >= 0, "all N threads got the same slot");

  bool same_bytes = true;
  std::vector<uint8_t> want(kRecordBytes);
  NemoPool baseline(1, kRecordBytes, resolver, /*nocache=*/false);
  baseline.read_direct(1, 5, want.data());
  for (int i = 1; i < N; ++i) {
    if (std::memcmp(pool.slot_data(slots[0]), pool.slot_data(slots[i]), kRecordBytes) != 0)
      same_bytes = false;
  }
  check(same_bytes && std::memcmp(pool.slot_data(slots[0]), want.data(), kRecordBytes) == 0,
        "all waiters see identical, bit-exact bytes");

  NemoPoolStats st = pool.stats();
  check(st.misses == 1, "misses == 1");
  check(st.coalesced_reads == static_cast<uint64_t>(N - 1), "coalesced_reads == N-1");
  check(st.physical_reads == 1, "physical_reads == 1");
  check(st.physical_bytes_read == static_cast<uint64_t>(kRecordBytes),
        "physical_bytes_read == one record");
  check(st.hits == 0, "hits == 0 (no false hit during in-flight fill)");
}

// (c)+(d) Exact counters with the flag off, and the cached-fd path.
void test_counters_and_flag_off(const std::string& shard) {
  std::printf("[c/d] counters + flag-off cached path\n");
  auto resolver = make_resolver(shard);
  std::vector<int> used_fds;
  NemoPool::ReadFn record_fd = [&](int fd, void* dst, size_t len, int64_t off) -> ssize_t {
    used_fds.push_back(fd);
    return plain_pread(fd, dst, len, off);
  };
  NemoPool pool(8, kRecordBytes, resolver, /*nocache=*/false, record_fd);
  std::vector<std::pair<int, int>> routes = {
      {0, 3}, {1, 5}, {0, 3}, {1, 2}, {0, 7}};

  std::vector<uint8_t> want = direct_records(pool, routes);
  std::vector<uint8_t> got = pooled_records(pool, routes);
  check(bit_equal_bytes(got, want), "cached-path gather == direct baseline (uint32 view)");

  NemoPoolStats st = pool.stats();
  check(st.hits == 1, "hits == 1 (one repeated expert)");
  check(st.misses == 4, "misses == 4 distinct experts");
  check(st.coalesced_reads == 0, "coalesced_reads == 0 (single-threaded)");
  check(st.physical_reads == 4, "physical_reads == 4");
  check(st.physical_bytes_read == 4ULL * kRecordBytes, "physical_bytes_read == 4 records");
  check(st.resident == 4 && st.evictions == 0, "resident == 4, evictions == 0");
  check(!st.nocache_enabled, "stats reports nocache mode off");

  check(!used_fds.empty(), "fills recorded their fd");
  int cfd = pool.cached_fd(shard);
  bool all_cached = true;
  for (int fd : used_fds) if (fd != cfd) all_cached = false;
  check(all_cached, "all flag-off fills used the normal cached descriptor");
#ifdef F_NOCACHE
  check(::fcntl(cfd, F_NOCACHE, 0) == 0, "flag-off fill fd has F_NOCACHE clear");
#endif
}

// Env flag parsing: default off, explicit 1/0, fail-closed on junk.
void test_env_flag() {
  std::printf("[env] VEYSTRIX_NEMOTRON_POOL_NOCACHE parsing\n");
  ::unsetenv(kNemoPoolNocacheEnv);
  check(!nemo_pool_nocache_from_env(), "unset => off");
  ::setenv(kNemoPoolNocacheEnv, "0", 1);
  check(!nemo_pool_nocache_from_env(), "\"0\" => off (explicit rollback)");
  ::setenv(kNemoPoolNocacheEnv, "1", 1);
  check(nemo_pool_nocache_from_env(), "\"1\" => on");
  ::setenv(kNemoPoolNocacheEnv, "junk", 1);
  bool threw = false;
  try { (void)nemo_pool_nocache_from_env(); } catch (const std::runtime_error&) { threw = true; }
  check(threw, "invalid value fails closed");
  ::unsetenv(kNemoPoolNocacheEnv);
}

}  // namespace

int main() {
  try {
    std::filesystem::path dir = make_temp_dir();
    std::string shard = (dir / "pool_shard.bin").string();
    write_shard(shard);
    std::printf("synthetic shard: %s (%d bytes)\n", shard.c_str(),
                kLayers * kExperts * kRecordBytes);

    test_env_flag();
    test_gather_bit_exact(shard);
    test_coalescing(shard);
    test_counters_and_flag_off(shard);

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 2;
  }
  if (g_failures) {
    std::fprintf(stderr, "test_nemo_pool FAILED (%d check(s))\n", g_failures);
    return 1;
  }
  std::printf("test_nemo_pool OK (all checks passed)\n");
  return 0;
}
