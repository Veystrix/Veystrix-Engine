// Phase 1 SharedExpertPool unit test — CPU-only synthetic fixtures.
//
// NO real checkpoint, NO SSD/benchmark, NO model run, NO MLX dependency: the
// resolver maps to a tiny temp file whose bytes are produced by the same LCG
// as tools/gen_shared_expert_pool_vectors.py, and every physical read goes
// through an injectable fake ReadFn that copies from that in-memory buffer.
//
// Test-only std::thread / std::barrier below are harness scaffolding used to
// create concurrent sessions; they are NOT new engine threads, mutexes, or
// I/O workers (those would be NEEDS OWNER APPROVAL and are not added).
//
// Checks:
//   (a) two concurrent sessions requesting one key do ONE physical read
//   (b) a pinned slot is not evicted until its fence
//   (c) a fence releases only that session's pins; other sessions' pins stay
//   (d) per-layer union fetch issues one acquire per DISTINCT key
//   (e) a failed read wakes every waiter with an error and leaves no stub
// Byte-exactness of payload bytes is asserted in (a), (b), (d), (e).
#include "shared_expert_pool.hpp"

#include <atomic>
#include <barrier>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>   // mkdtemp
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>  // mkdtemp on Darwin

using namespace veystrix::engine;

namespace {

constexpr int kRecordBytes = 256;   // multiple of 4 -> uint32 word view is exact
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

// Byte-for-byte comparison. The extra 32-bit word pass mirrors the repo's
// mx::view -> uint32 bit-exact convention without pulling in MLX (keeps this
// test CPU-only and self-contained).
bool bit_equal(const uint8_t* a, const uint8_t* b, size_t n) {
  if (n % 4 != 0) return false;
  if (std::memcmp(a, b, n) != 0) return false;
  for (size_t i = 0; i < n / 4; ++i) {
    uint32_t wa = 0, wb = 0;
    std::memcpy(&wa, a + i * 4, 4);
    std::memcpy(&wb, b + i * 4, 4);
    if (wa != wb) return false;
  }
  return true;
}

std::vector<uint8_t> make_golden() {
  std::vector<uint8_t> out(kLayers * kExperts * kRecordBytes);
  uint64_t st = kSeed;
  for (size_t i = 0; i < out.size(); ++i) {
    st = st * kMul + kAdd;
    out[i] = static_cast<uint8_t>((st >> 33) & 0xff);
  }
  return out;
}

const uint8_t* rec(const std::vector<uint8_t>& golden, int layer, int expert) {
  return golden.data() + static_cast<size_t>(layer * kExperts + expert) * kRecordBytes;
}

std::filesystem::path make_temp_dir() {
  std::string tmpl = (std::filesystem::temp_directory_path() / "sep_pool_XXXXXX").string();
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  char* d = ::mkdtemp(buf.data());
  if (d == nullptr) throw std::runtime_error("mkdtemp failed");
  return std::filesystem::path(d);
}

NemoPool::Resolver make_resolver(const std::string& shard) {
  return [shard](int layer, int expert) {
    return NemoPool::Record{
        shard, static_cast<int64_t>(layer * kExperts + expert) * kRecordBytes,
        kRecordBytes};
  };
}

// Injectable physical read. Counts calls, can fail, and can hold the leader
// until all other sessions have registered as coalesced waiters so tests (a)
// and (e) are deterministic rather than timing-dependent.
struct FakeIO {
  const std::vector<uint8_t>* data = nullptr;
  std::atomic<int> calls{0};
  std::atomic<bool> fail{false};
  bool block_for_waiters = false;
  int expected_waiters = 0;
  NemoPool* pool = nullptr;

  ssize_t read(int, void* dst, size_t len, int64_t off) {
    const int n = calls.fetch_add(1);
    if (n == 0 && block_for_waiters && pool != nullptr) {
      auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (pool->stats().coalesced_reads < static_cast<uint64_t>(expected_waiters) &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
      }
    }
    if (fail.load()) {
      errno = EIO;
      return -1;
    }
    std::memcpy(dst, data->data() + static_cast<size_t>(off), len);
    return static_cast<ssize_t>(len);
  }
};

NemoPool::ReadFn fake_read_fn(FakeIO& io) {
  return [&io](int fd, void* dst, size_t len, int64_t off) -> ssize_t {
    return io.read(fd, dst, len, off);
  };
}

// (a) Two concurrent sessions, one key, one physical read.
void test_coalescing(const std::string& shard, const std::vector<uint8_t>& golden) {
  std::printf("[a] in-flight coalescing: concurrent sessions share one read\n");
  constexpr int kSessions = 2;
  FakeIO io;
  io.data = &golden;
  io.block_for_waiters = true;
  io.expected_waiters = kSessions - 1;
  SharedExpertPool sep(8, kRecordBytes, make_resolver(shard),
                       /*nocache=*/false, fake_read_fn(io));
  io.pool = &sep.pool();

  std::barrier sync(kSessions);
  std::vector<int> slots(kSessions, -2);
  std::vector<std::thread> threads;
  for (int i = 0; i < kSessions; ++i) {
    threads.emplace_back([&, i] {
      sync.arrive_and_wait();
      slots[i] = sep.fetch_union(1, {5})[0].slot;  // each is its own "session"
    });
  }
  for (auto& t : threads) t.join();

  check(io.calls.load() == 1, "one physical read for the shared key");
  check(slots[0] >= 0 && slots[0] == slots[1], "both sessions got the same slot");
  check(bit_equal(sep.pool().slot_data(slots[0]), rec(golden, 1, 5), kRecordBytes),
        "coalesced payload is byte-exact");
  NemoPoolStats st = sep.stats();
  check(st.misses == 1, "misses == 1");
  check(st.coalesced_reads == kSessions - 1, "coalesced_reads == sessions-1");
  check(st.physical_reads == 1, "physical_reads == 1");
  check(st.physical_bytes_read == static_cast<uint64_t>(kRecordBytes),
        "physical_bytes_read == one record");
  check(st.hits == 0, "hits == 0 (no false hit on half-written data)");
}

// (b) Pin prevents eviction until the owning session's fence.
void test_pin_blocks_eviction(const std::string& shard,
                              const std::vector<uint8_t>& golden) {
  std::printf("[b] pin prevents eviction until fence\n");
  FakeIO io;
  io.data = &golden;
  SharedExpertPool sep(2, kRecordBytes, make_resolver(shard),
                       /*nocache=*/false, fake_read_fn(io));

  const int a = sep.acquire_and_pin(11, 0, 0);        // pinned by session 11
  (void)sep.pool().acquire(0, 1);                    // second slot, unpinned
  check(sep.pin_count(a) == 1, "pinned slot has one pin");

  const int c = sep.fetch_union(0, {2})[0].slot;      // 2 slots full -> victim needed
  check(sep.pool().slot_for(0, 0) == a,
        "pinned slot survives LRU pressure (victim was the unpinned slot)");
  check(sep.pool().slot_for(0, 2) == c && sep.pool().slot_for(0, 1) == -1,
        "unpinned slot was evicted instead");
  check(bit_equal(sep.pool().slot_data(a), rec(golden, 0, 0), kRecordBytes),
        "pinned payload still byte-exact after pressure");
  check(sep.stats().evictions == 1, "exactly one eviction occurred");

  check(sep.release_fence(11, 0) == 1, "fence released the pin");
  (void)sep.pool().acquire(0, 3);                    // now the old slot is eligible
  check(sep.pool().slot_for(0, 0) == -1, "slot becomes evictable after fence");
}

// (c) Fence releases only that session's pins; other sessions' pins intact.
void test_fence_isolation(const std::string& shard, const std::vector<uint8_t>& golden) {
  std::printf("[c] fence releases only the calling session's pins\n");
  FakeIO io;
  io.data = &golden;
  SharedExpertPool sep(3, kRecordBytes, make_resolver(shard),
                       /*nocache=*/false, fake_read_fn(io));

  const int a = sep.acquire_and_pin(1, 0, 0);   // session 1 pin on slot a
  sep.pin(2, 0, 0);                              // session 2 pin on the SAME slot
  const int b = sep.acquire_and_pin(2, 0, 1);   // session 2 pin on slot b
  (void)sep.pool().acquire(0, 2);               // unpinned third slot
  check(sep.pin_count(a) == 2 && sep.pin_count(b) == 1,
        "pin counts are per slot (a=2, b=1)");

  check(sep.release_fence(1, 0) == 1, "session 1 fence released one reference");
  check(sep.pin_count(a) == 1, "slot a still pinned by session 2");
  check(sep.pin_count(b) == 1, "slot b (session 2) untouched by session 1 fence");

  (void)sep.fetch_union(0, {3});                 // evicts the only unpinned slot
  check(sep.pool().slot_for(0, 0) == a && sep.pool().slot_for(0, 1) == b,
        "both session-2-pinned slots survive after session 1 fence");
  check(bit_equal(sep.pool().slot_data(a), rec(golden, 0, 0), kRecordBytes),
        "still-pinned payload is byte-exact");

  check(sep.release_fence(2, 0) == 2, "session 2 fence released both references");
  check(sep.pin_count(a) == 0 && sep.pin_count(b) == 0,
        "all pins cleared only after their owner's fence");
  (void)sep.pool().acquire(0, 4);
  (void)sep.pool().acquire(0, 5);
  check(sep.pool().slot_for(0, 0) == -1 && sep.pool().slot_for(0, 1) == -1,
        "previously pinned slots are evictable once all fences drain");
}

// (d) Per-layer union fetch deduplicates keys.
void test_union_dedup(const std::string& shard, const std::vector<uint8_t>& golden) {
  std::printf("[d] per-layer union fetch dedups distinct keys\n");
  FakeIO io;
  io.data = &golden;
  SharedExpertPool sep(8, kRecordBytes, make_resolver(shard),
                       /*nocache=*/false, fake_read_fn(io));

  const std::vector<int> requested{0, 1, 0, 2, 1, 3, 2};  // 7 keys, 4 distinct
  const auto maps = sep.fetch_union(0, requested);
  check(maps.size() == 4, "one mapping per distinct key");
  check(maps[0].expert == 0 && maps[1].expert == 1 && maps[2].expert == 2 &&
            maps[3].expert == 3,
        "first-seen order preserved");
  check(io.calls.load() == 4 && sep.stats().physical_reads == 4,
        "one physical read per distinct key");
  bool bytes_ok = true;
  for (const auto& m : maps) {
    if (m.slot < 0 || !bit_equal(sep.pool().slot_data(m.slot), rec(golden, 0, m.expert),
                                 kRecordBytes))
      bytes_ok = false;
  }
  check(bytes_ok, "every union payload is byte-exact");

  const auto again = sep.fetch_union(0, {3, 0});
  check(again.size() == 2 && io.calls.load() == 4,
        "repeat keys are pure hits (no extra reads)");
  check(sep.stats().hits == 2, "hits == 2 on the second fetch");
}

// (e) A failed read wakes all waiters with an error; no waiter is left blocked.
void test_failed_read_wakes_waiters(const std::string& shard,
                                    const std::vector<uint8_t>& golden) {
  std::printf("[e] failed shared read wakes every waiter with an error\n");
  constexpr int kSessions = 4;
  FakeIO io;
  io.data = &golden;
  io.fail = true;
  io.block_for_waiters = true;
  io.expected_waiters = kSessions - 1;
  SharedExpertPool sep(4, kRecordBytes, make_resolver(shard),
                       /*nocache=*/false, fake_read_fn(io));
  io.pool = &sep.pool();

  std::atomic<int> threw{0};
  std::barrier sync(kSessions);
  std::vector<std::thread> threads;
  for (int i = 0; i < kSessions; ++i) {
    threads.emplace_back([&] {
      sync.arrive_and_wait();
      try {
        (void)sep.fetch_union(1, {6});
      } catch (const std::exception&) {
        threw.fetch_add(1);
      }
    });
  }
  for (auto& t : threads) t.join();

  check(threw.load() == kSessions, "all sessions woke with an error (no hang)");
  check(io.calls.load() == 1, "only the leader issued a read");
  NemoPoolStats st = sep.stats();
  check(st.physical_reads == 0 && st.resident == 0,
        "failed fill returned its slot to the free list");
  check(sep.pool().slot_for(1, 6) == -1, "failed key is not on the hit path");

  // Recovery proves no in-flight entry was left stuck after the failure.
  io.fail = false;
  int slot = -1;
  try {
    slot = sep.fetch_union(1, {6})[0].slot;
  } catch (const std::exception&) {
  }
  check(slot >= 0 && bit_equal(sep.pool().slot_data(slot), rec(golden, 1, 6), kRecordBytes),
        "retry after failure succeeds and is byte-exact");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::vector<uint8_t> golden = make_golden();
    const std::filesystem::path dir = make_temp_dir();
    const std::string shard = (dir / "synthetic.bin").string();
    {
      std::ofstream f(shard, std::ios::binary);
      f.write(reinterpret_cast<const char*>(golden.data()),
              static_cast<std::streamsize>(golden.size()));
    }
    std::printf("synthetic shard: %s (%zu bytes, no checkpoint)\n", shard.c_str(),
                golden.size());

    // Optional cross-language golden check against the generator's records.bin.
    if (argc >= 2) {
      const std::string path = std::string(argv[1]) + "/records.bin";
      std::ifstream f(path, std::ios::binary);
      const std::vector<uint8_t> from_disk(
          (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      check(from_disk == golden,
            "generator records.bin matches in-test LCG bytes (cross-language)");
    }

    test_coalescing(shard, golden);
    test_pin_blocks_eviction(shard, golden);
    test_fence_isolation(shard, golden);
    test_union_dedup(shard, golden);
    test_failed_read_wakes_waiters(shard, golden);

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 2;
  }
  if (g_failures) {
    std::fprintf(stderr, "test_shared_expert_pool FAILED (%d check(s))\n", g_failures);
    return 1;
  }
  std::printf("test_shared_expert_pool OK (all checks passed)\n");
  return 0;
}
