// Phase F steps 1-4 — CPU-only contract test for the split-layer path.
//
// Exercises the REAL LayerCursor state machine + the REAL io_pager_async on a
// temporary file (real libdispatch + real pread), with NO MLX, NO checkpoint,
// NO GPU and NO benchmark timing. It models one decode token as L layers, each
// split into plan_pin_submit (async read submit) -> await_reads -> wait/build,
// and gates the N=1 cursor driver's per-step digest against a synchronous solo
// (run_batched_reads) path. Also checks the C1 dispatch-notify discipline: the
// callback fires exactly once per ticket, bumps only an atomic, and never
// writes to the destination after completion.
//
// Build (standalone):
//   xcrun clang++ -std=c++20 -O2 -Icpp/engine/src -Icpp/decode/src \
//     cpp/engine/src/layer_cursor.cpp \
//     cpp/decode/src/io_pager.cpp cpp/decode/src/io_pager_async.cpp \
//     models/flashnext/native_read.c \
//     cpp/engine/tests/test_phase_f_contract.cpp -o /tmp/test_phase_f_contract
#include "io_pager.hpp"
#include "io_pager_async.hpp"
#include "layer_cursor.hpp"

#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace veystrix;
using namespace veystrix::engine;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

static constexpr int kRows = 64;
static constexpr int kRowBytes = 64;

static uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// Deterministic payload for (session, layer, routed expert, tensor).
static std::vector<uint8_t> payload(int s, int layer, int expert, int tensor) {
  uint64_t h = mix64(static_cast<uint64_t>(s) << 40 ^
                     (static_cast<uint64_t>(layer) << 24) ^
                     (static_cast<uint64_t>(expert) << 8) ^ tensor);
  std::vector<uint8_t> out(kRowBytes);
  for (auto& b : out) { h = mix64(h + 1); b = static_cast<uint8_t>(h >> 56); }
  return out;
}

struct FakeFile {
  int fd = -1;
  std::string path;
  FakeFile() {
    char tpl[] = "/tmp/phase_f_contract_XXXXXX";
    fd = ::mkstemp(tpl);
    CHECK(fd >= 0);
    path = tpl;
    std::vector<uint8_t> data;
    for (int e = 0; e < kRows; ++e) {
      auto p = payload(0, 0, e, 0);  // canonical rows keyed by expert id
      data.insert(data.end(), p.begin(), p.end());
    }
    CHECK(::write(fd, data.data(), data.size()) == static_cast<ssize_t>(data.size()));
    ::fsync(fd);
  }
  ~FakeFile() { if (fd >= 0) ::close(fd); if (!path.empty()) ::unlink(path.c_str()); }
};

// Route for one (session, layer): three distinct expert ids.
static std::vector<int> route(int session, int layer) {
  uint64_t h = mix64(0xC0FFEEULL ^ (static_cast<uint64_t>(session) << 16) ^ layer);
  std::vector<int> out;
  while (out.size() < 3) {
    h = mix64(h);
    int e = static_cast<int>(h % kRows);
    bool seen = false;
    for (int x : out) if (x == e) seen = true;
    if (!seen) out.push_back(e);
  }
  return out;
}

// Fold one layer's materialized expert rows into a digest (a pure function of
// the bytes, independent of completion order). This stands in for the MoE
// graph build: bit-identity here means identical gathered bytes + order.
static uint64_t fold(uint64_t acc, const std::vector<std::vector<uint8_t>>& rows,
                     const std::vector<int>& slots) {
  for (int slot : slots)
    for (uint8_t b : rows[slot]) acc = mix64(acc ^ b);
  return acc;
}

static uint64_t solo_step(FakeFile& f, int session, int n_layers) {
  uint64_t h = mix64(static_cast<uint64_t>(session));
  for (int l = 0; l < n_layers; ++l) {
    auto r = route(session, l);
    std::vector<std::vector<uint8_t>> dst(r.size(), std::vector<uint8_t>(kRowBytes));
    std::vector<PagerRead> reads;
    for (size_t i = 0; i < r.size(); ++i)
      reads.push_back({f.fd, 0, static_cast<int64_t>(r[i] * kRowBytes),
                       kRowBytes, reinterpret_cast<uint64_t>(dst[i].data())});
    int errs = run_batched_reads(reads, 8);
    CHECK(errs == 0);
    h = fold(h, dst, {0, 1, 2});  // router-rank slot order
  }
  return h;
}

static uint64_t cursor_step(FakeFile& f, int session, int n_layers,
                            std::atomic<int>* notify) {
  uint64_t h = mix64(static_cast<uint64_t>(session));
  LayerCursor<std::vector<uint8_t>> cur;  // plain, non-MLX Hidden
  cur.begin(0, std::vector<uint8_t>{});
  for (int l = 0; l < n_layers; ++l) {
    CHECK(cur.layer == l);
    CHECK(cur.phase == LayerPhase::PreRoute);
    // --- plan_pin_submit: submit async reads for this layer's route ---------
    auto r = route(session, l);
    std::vector<std::vector<uint8_t>> dst(r.size(), std::vector<uint8_t>(kRowBytes, 0xAA));
    std::vector<PagerRead> reads;
    for (size_t i = 0; i < r.size(); ++i)
      reads.push_back({f.fd, 0, static_cast<int64_t>(r[i] * kRowBytes),
                       kRowBytes, reinterpret_cast<uint64_t>(dst[i].data())});
    int ticket = submit_batched_reads_async(
        reads, 8, [notify](int /*t*/) {
          // C1: atomic-only callback; NEVER builds a graph / calls MLX.
          notify->fetch_add(1, std::memory_order_relaxed);
        });
    cur.await_reads({ticket});
    CHECK(cur.phase == LayerPhase::AwaitReads && !cur.ready());

    // --- N=1 driver: no other session runnable -> wait here ------------------
    CHECK(wait_ticket(ticket));
    CHECK(ticket_complete(ticket));
    CHECK(ticket_errors(ticket) == 0);
    CHECK(cur.on_ticket_done(ticket));   // idempotent per C2
    CHECK(!cur.on_ticket_done(ticket));  // second removal returns false
    CHECK(cur.ready());
    cur.enter_post_moe();

    // --- build: bytes are materialized; fold in router-rank order ----------
    h = fold(h, dst, {0, 1, 2});
    cur.finish(std::vector<uint8_t>{});
    CHECK(cur.layer == l + 1 && cur.phase == LayerPhase::PreRoute);
  }
  return h;
}

static void test_n1_bit_exact() {
  FakeFile f;
  std::atomic<int> notify{0};
  constexpr int kLayers = 12;
  for (int s = 0; s < 5; ++s) {
    uint64_t solo = solo_step(f, s, kLayers);
    uint64_t cur = cursor_step(f, s, kLayers, &notify);
    CHECK(solo == cur);  // hard bit-exactness gate
  }
  CHECK(notify.load() == 5 * kLayers);  // exactly one completion per submit
  std::printf("  ok: N=1 cursor digest == solo digest for 5 sessions x 12 layers\n");
  std::printf("  ok: dispatch notify fired exactly %d times (atomic-only, once/ticket)\n",
              notify.load());
}

static void test_no_write_after_complete() {
  FakeFile f;
  std::vector<std::vector<uint8_t>> dst(4, std::vector<uint8_t>(kRowBytes, 0x00));
  std::vector<PagerRead> reads;
  for (int i = 0; i < 4; ++i)
    reads.push_back({f.fd, 0, static_cast<int64_t>(i * kRowBytes),
                     kRowBytes, reinterpret_cast<uint64_t>(dst[i].data())});
  std::atomic<int> n{0};
  int t = submit_batched_reads_async(reads, 8, [&](int) { ++n; });
  CHECK(wait_ticket(t));
  auto snapshot = dst;
  for (int spin = 0; spin < 200000; ++spin) std::atomic_signal_fence(std::memory_order_seq_cst);
  for (int i = 0; i < 4; ++i) CHECK(dst[i] == snapshot[i]);
  CHECK(n.load() == 1);
  std::printf("  ok: no write-after-complete (C1 mutex/CV acquire edge)\n");
}

int main() {
  std::printf("test_phase_f_contract (CPU-only synthetic):\n");
  test_n1_bit_exact();
  test_no_write_after_complete();
  int dropped = reset_async_pager_state();
  CHECK(dropped > 0);
  std::printf("test_phase_f_contract OK (all checks passed)\n");
  return 0;
}
