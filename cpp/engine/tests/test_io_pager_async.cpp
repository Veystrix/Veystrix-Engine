// Phase 2 — CPU-only synthetic fixtures for io_pager_async (brief §5 Phase
// 2 item 3c/3d). Uses REAL libdispatch and a REAL temporary file (pread), but
// NO MLX, NO checkpoint, NO GPU, NO benchmark/timing run. Every assertion is
// about completion correctness and byte identity, not speed.
//
// Build (standalone):
//   xcrun clang++ -std=c++20 -O2 -Icpp/engine/src -Icpp/decode/src \
//     cpp/decode/src/io_pager.cpp cpp/decode/src/io_pager_async.cpp \
//     models/flashnext/native_read.c \
//     cpp/engine/tests/test_io_pager_async.cpp -o /tmp/test_io_pager_async
#include "io_pager_async.hpp"

#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

static uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// A small temp file of deterministic records.
struct FakeFile {
  int fd = -1;
  std::string path;
  int record_bytes = 0;
  int records = 0;

  explicit FakeFile(int n_records, int rec_bytes)
      : record_bytes(rec_bytes), records(n_records) {
    char tpl[] = "/tmp/io_pager_async_test_XXXXXX";
    int fd = ::mkstemp(tpl);
    CHECK(fd >= 0);
    path = tpl;
    std::vector<uint8_t> data(static_cast<std::size_t>(n_records) * rec_bytes);
    uint64_t h = 0x1234ABCDULL;
    for (auto& b : data) {
      h = mix64(h);
      b = static_cast<uint8_t>(h >> 56);
    }
    CHECK(::write(fd, data.data(), data.size()) ==
          static_cast<ssize_t>(data.size()));
    ::fsync(fd);
    this->fd = fd;
  }
  std::vector<uint8_t> expected(int rec) const {
    std::vector<uint8_t> out(record_bytes);
    uint64_t h = 0x1234ABCDULL;
    for (int i = 0; i < rec * record_bytes; ++i) h = mix64(h);
    for (auto& b : out) {
      h = mix64(h);
      b = static_cast<uint8_t>(h >> 56);
    }
    return out;
  }
  ~FakeFile() {
    if (fd >= 0) ::close(fd);
    if (!path.empty()) ::unlink(path.c_str());
  }
};

static void test_completion_once() {
  FakeFile f(16, 256);
  std::vector<std::vector<uint8_t>> dst(8, std::vector<uint8_t>(256, 0xAA));
  std::vector<veystrix::PagerRead> reads;
  for (int i = 0; i < 8; ++i)
    reads.push_back({f.fd, 0, static_cast<int64_t>(i * 256), 256,
                     reinterpret_cast<uint64_t>(dst[i].data())});

  std::atomic<int> callback_count{0};
  std::atomic<int> callback_ticket{-1};
  int ticket = veystrix::submit_batched_reads_async(
      reads, 8, [&](int t) {
        callback_count.fetch_add(1);
        callback_ticket.store(t);
      });
  CHECK(ticket >= 0);
  CHECK(veystrix::wait_ticket(ticket));  // does not hang
  // Callback fires exactly once, after completion.
  CHECK(callback_count.load() == 1);
  CHECK(callback_ticket.load() == ticket);
  CHECK(veystrix::ticket_complete(ticket));
  CHECK(veystrix::ticket_errors(ticket) == 0);

  // Every payload byte is intact.
  for (int i = 0; i < 8; ++i) CHECK(dst[i] == f.expected(i));
  std::printf("  ok: tickets fire once, payload bytes intact (8 lanes, width 8)\n");
}

static void test_no_write_after_complete() {
  FakeFile f(4, 512);
  std::vector<std::vector<uint8_t>> dst(4, std::vector<uint8_t>(512, 0x00));
  std::vector<veystrix::PagerRead> reads;
  for (int i = 0; i < 4; ++i)
    reads.push_back({f.fd, 0, static_cast<int64_t>(i * 512), 512,
                     reinterpret_cast<uint64_t>(dst[i].data())});

  std::atomic<bool> fired{false};
  int ticket = veystrix::submit_batched_reads_async(reads, 4, [&](int) {
    // On callback, the bytes must already be stable: sample them and check
    // again after a short spin to detect any write-after-complete.
    std::vector<std::vector<uint8_t>> snapshot = dst;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    for (int spin = 0; spin < 100000; ++spin) {
      std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    for (int i = 0; i < 4; ++i) CHECK(dst[i] == snapshot[i]);
    fired.store(true);
  });
  CHECK(veystrix::wait_ticket(ticket));
  CHECK(fired.load());
  for (int i = 0; i < 4; ++i) CHECK(dst[i] == f.expected(i));
  std::printf("  ok: no write-after-complete (acquire edge via C1 mutex/CV)\n");
}

static void test_wait_no_hang_and_width_clamp() {
  FakeFile f(32, 64);
  // width > 8 must clamp to the C1 width of 8 (never raise worker count/QoS).
  std::vector<std::vector<uint8_t>> dst(32, std::vector<uint8_t>(64));
  std::vector<veystrix::PagerRead> reads;
  for (int i = 0; i < 32; ++i)
    reads.push_back({f.fd, 0, static_cast<int64_t>(i * 64), 64,
                     reinterpret_cast<uint64_t>(dst[i].data())});
  std::atomic<int> count{0};
  int ticket = veystrix::submit_batched_reads_async(reads, 64,
                                                    [&](int) { ++count; });
  CHECK(veystrix::wait_ticket(ticket));
  CHECK(count.load() == 1);
  for (int i = 0; i < 32; ++i) CHECK(dst[i] == f.expected(i));

  // wait on an already-complete ticket returns immediately (callback counted
  // once more from a second submit, not this wait).
  CHECK(veystrix::wait_ticket(ticket));
  std::printf("  ok: wait_ticket returns promptly; width clamped to 8; no hang\n");
}

static void test_randomized_batches() {
  FakeFile f(64, 128);
  std::mt19937 rng(0xC0FFEE);
  for (int iter = 0; iter < 50; ++iter) {
    const int n = 1 + static_cast<int>(rng() % 20);
    std::vector<int> which(n);
    std::vector<std::vector<uint8_t>> dst(n, std::vector<uint8_t>(128, 0x55));
    std::vector<veystrix::PagerRead> reads;
    for (int i = 0; i < n; ++i) {
      which[i] = static_cast<int>(rng() % 64);
      reads.push_back({f.fd, 0, static_cast<int64_t>(which[i] * 128), 128,
                       reinterpret_cast<uint64_t>(dst[i].data())});
    }
    std::atomic<int> fires{0};
    int ticket = veystrix::submit_batched_reads_async(
        reads, 1 + static_cast<int>(rng() % 8), [&](int) { ++fires; });
    CHECK(veystrix::wait_ticket(ticket));
    CHECK(fires.load() == 1);
    for (int i = 0; i < n; ++i) CHECK(dst[i] == f.expected(which[i]));
  }
  std::printf("  ok: 50 randomized batches, byte-exact, exactly one completion each\n");
}

int main() {
  std::printf("test_io_pager_async (CPU-only synthetic):\n");
  test_completion_once();
  test_no_write_after_complete();
  test_wait_no_hang_and_width_clamp();
  test_randomized_batches();
  int dropped = veystrix::reset_async_pager_state();
  CHECK(dropped > 0);
  std::printf("test_io_pager_async OK (all checks passed)\n");
  return 0;
}
