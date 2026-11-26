// Offline gate for the dynamic-memory contract (src/memory_budget.*).
// No checkpoint, no real/guarded run. Proves:
//   1. Profile keys / env resolve with precedence profile > env > default, and
//      garbage fails closed.
//   2. Env-driven sizes are honored: slot counts become exact capacities.
//   3. Resident mode FAILS CLOSED when the full expert set does not fit the
//      byte budget or the RAM headroom (real Qwen + Nemotron sizes, on THIS
//      machine's RAM and on simulated 16/24/32/48 GB tiers).
//   4. When it does fit and the pool holds every key, a second full pass over
//      all (layer, expert) keys performs ZERO storage reads (Nemotron NemoPool
//      with real preads).
//   5. Cache management: set_cache_limit applied, clear_cache wrapper works.
#include "memory_budget.hpp"
#include "nemo_pool.hpp"

#include <mlx/mlx.h>
#include <mlx/memory.h>

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;
namespace mb = veystrix::engine::membudget;

static int g_failures = 0;
#define CHECK(cond, msg)                                                      \
  do {                                                                        \
    if (cond) {                                                               \
      std::printf("  ok   %s\n", (msg));                                      \
    } else {                                                                  \
      std::printf("  FAIL %s  (line %d)\n", (msg), __LINE__);                 \
      ++g_failures;                                                           \
    }                                                                         \
  } while (0)

template <class F>
static bool throws_with(F&& f, const std::string& needle) {
  try {
    f();
  } catch (const std::exception& e) {
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

static void setenv_(const char* k, const char* v) { ::setenv(k, v, 1); }
static void unsetenv_(const char* k) { ::unsetenv(k); }

static void clear_env() {
  for (const char* k :
       {"VEYSTRIX_CPP_POOL_BYTES", "VEYSTRIX_CPP_NEMO_POOL_SLOTS", "VEYSTRIX_CPP_RESIDENT",
        "VEYSTRIX_CPP_MAX_SESSIONS", "VEYSTRIX_CPP_RAM_BYTES",
        "VEYSTRIX_CPP_RESIDENT_RESERVE_BYTES", "VEYSTRIX_CPP_MLX_CACHE_BYTES",
        "VEYSTRIX_CPP_MLX_MEMORY_BYTES", "VEYSTRIX_CPP_IDLE_CLEAR_MS"})
    unsetenv_(k);
}

constexpr size_t GiB = size_t{1} << 30;
// Real Nemotron-3.5-Lightning-30B-A3B geometry (6 tensor kinds per expert; see
// veystrix/memory_budget_runtime.py: 5,612,544 B/expert x 23 MoE layers x 128).
constexpr size_t kNemoRecordBytes = 5612544;
constexpr size_t kNemoKeys = 23 * 128;
// Real Qwen3.8-Flash-Next geometry (safetensors header of layer 0):
// 3 x 819,200 (weights) + 6 x 102,400 (scales/biases) = 3,072,000 B/expert.
constexpr size_t kQwenRecordBytes = 3'072'000;
constexpr size_t kQwenKeys = 48 * 512;

static void test_profile_resolution() {
  std::puts("[profile] precedence + strictness");
  clear_env();
  mb::MemoryProfile p = mb::resolve_memory_profile("{\"model\":\"x\"}");
  CHECK(p.pool_bytes == 0 && p.nemo_slots == 0 && !p.resident && p.max_sessions == 0,
        "all unset -> defaults");

  setenv_("VEYSTRIX_CPP_POOL_BYTES", "123456789");
  setenv_("VEYSTRIX_CPP_NEMO_POOL_SLOTS", "512");
  setenv_("VEYSTRIX_CPP_RESIDENT", "1");
  setenv_("VEYSTRIX_CPP_MAX_SESSIONS", "3");
  p = mb::resolve_memory_profile("{}");
  CHECK(p.pool_bytes == 123456789, "env VEYSTRIX_CPP_POOL_BYTES honored");
  CHECK(p.nemo_slots == 512, "env VEYSTRIX_CPP_NEMO_POOL_SLOTS honored");
  CHECK(p.resident, "env VEYSTRIX_CPP_RESIDENT=1 honored");
  CHECK(p.max_sessions == 3, "env VEYSTRIX_CPP_MAX_SESSIONS honored");

  p = mb::resolve_memory_profile(
      "{\"expert_pool_bytes\": 999, \"nemotron_slots\": 77, \"resident\": false, "
      "\"max_sessions\": 2}");
  CHECK(p.pool_bytes == 999 && p.nemo_slots == 77 && !p.resident && p.max_sessions == 2,
        "profile keys override env");
  p = mb::resolve_memory_profile("{\"resident\":true}");
  CHECK(p.resident, "profile resident=true");
  p = mb::resolve_memory_profile("{\"resident\":1}");
  CHECK(p.resident, "profile resident=1");

  CHECK(throws_with([] { mb::resolve_memory_profile("{\"resident\":\"maybe\"}"); },
                    "resident must be"),
        "garbage resident fails closed");
  setenv_("VEYSTRIX_CPP_POOL_BYTES", "12x");
  CHECK(throws_with([] { mb::resolve_memory_profile("{}"); }, "not an unsigned integer"),
        "garbage pool bytes fails closed");
  clear_env();
}

static void test_decision_and_refusal() {
  std::puts("[resident] fail-closed decision");
  // Pure decision.
  auto d = mb::decide_resident(10 * GiB, 0, 20 * GiB);
  CHECK(d.fits && d.reason.empty(), "fits when under RAM headroom and no budget");
  d = mb::decide_resident(10 * GiB, 8 * GiB, 20 * GiB);
  CHECK(!d.fits && d.reason.find("pool budget") != std::string::npos,
        "refuses when over explicit pool budget");
  d = mb::decide_resident(30 * GiB, 0, 20 * GiB);
  CHECK(!d.fits && d.reason.find("RAM headroom") != std::string::npos,
        "refuses when over RAM headroom");
  d = mb::decide_resident(0, 0, 20 * GiB);
  CHECK(!d.fits, "refuses an empty expert set");

  // Simulated tiers (VEYSTRIX_CPP_RAM_BYTES), real sizes, 0 active bytes.
  const size_t nemo_need = kNemoKeys * kNemoRecordBytes;   // 15.39 GiB
  const size_t qwen_need = kQwenKeys * kQwenRecordBytes;   // 70.3 GiB
  std::printf("  nemotron resident need = %.2f GiB, qwen = %.2f GiB\n",
              nemo_need / double(GiB), qwen_need / double(GiB));
  struct Tier { const char* name; size_t ram; bool nemo_fits; bool qwen_fits; };
  const Tier tiers[] = {
      {"16GB", 16 * GiB, false, false}, {"24GB", 24 * GiB, true, false},
      {"32GB", 32 * GiB, true, false},  {"48GB", 48 * GiB, true, false},
      {"64GB", 64 * GiB, true, false},  {"128GB", 128 * GiB, true, true},
  };
  for (const Tier& t : tiers) {
    setenv_("VEYSTRIX_CPP_RAM_BYTES", std::to_string(t.ram).c_str());
    const bool nf = !throws_with([&] { mb::require_resident_fit("nemotron", nemo_need, 0, 0); },
                                 "resident refused");
    const bool qf = !throws_with([&] { mb::require_resident_fit("qwen", qwen_need, 0, 0); },
                                 "resident refused");
    char m1[96], m2[96];
    std::snprintf(m1, sizeof m1, "%s tier: nemotron resident %s", t.name,
                  t.nemo_fits ? "fits" : "REFUSED");
    std::snprintf(m2, sizeof m2, "%s tier: qwen resident %s", t.name,
                  t.qwen_fits ? "fits" : "REFUSED");
    CHECK(nf == t.nemo_fits, m1);
    CHECK(qf == t.qwen_fits, m2);
  }
  // Active memory (dense weights already loaded) shrinks the headroom.
  setenv_("VEYSTRIX_CPP_RAM_BYTES", std::to_string(24 * GiB).c_str());
  CHECK(throws_with([&] { mb::require_resident_fit("nemotron", nemo_need, 0, 4 * GiB); },
                    "RAM headroom"),
        "24GB tier with 4 GiB already active refuses nemotron");
  CHECK(throws_with([&] { mb::require_ram_headroom("nemotron", 30 * GiB, 0); },
                    "exceeds RAM headroom"),
        "non-resident pool larger than RAM headroom is refused");
  clear_env();

  // THIS machine (no override): whatever hw.memsize says, a 15.4 GiB resident
  // Nemotron must be refused when RAM <= 18.4 GiB (16 GB dev machine).
  const size_t phys = mb::physical_ram_bytes();
  std::printf("  this machine: %.2f GiB physical\n", phys / double(GiB));
  if (phys <= 18 * GiB) {
    CHECK(throws_with([&] { mb::require_resident_fit("nemotron", nemo_need, 0, 0); },
                      "resident refused"),
          "REAL machine (<=18 GiB): nemotron resident is REFUSED -> streams");
    CHECK(throws_with([&] { mb::require_resident_fit("qwen", qwen_need, 0, 0); },
                      "resident refused"),
          "REAL machine (<=18 GiB): qwen resident is REFUSED -> streams");
  } else {
    std::puts("  (machine >18 GiB: real-machine refusal check not applicable)");
  }
}

static void test_nemo_pool_resident_zero_reads() {
  std::puts("[nemotron] slots>=keys -> zero SSD reads after warm-up");
  clear_env();
  namespace fs = std::filesystem;
  std::string tmpl = (fs::temp_directory_path() / "membudget_XXXXXX").string();
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  const char* dir = ::mkdtemp(buf.data());
  if (!dir) { CHECK(false, "mkdtemp"); return; }
  const std::string shard = std::string(dir) + "/shard.bin";
  constexpr int L = 4, E = 12;
  constexpr int RB = 256;
  {
    std::ofstream f(shard, std::ios::binary);
    std::vector<uint8_t> rec(RB);
    for (int k = 0; k < L * E; ++k) {
      for (int i = 0; i < RB; ++i) rec[i] = static_cast<uint8_t>(k * 31 + i);
      f.write(reinterpret_cast<const char*>(rec.data()), RB);
    }
  }
  auto resolver = [&](int l, int e) {
    return NemoPool::Record{shard, static_cast<int64_t>(l * E + e) * RB, RB};
  };
  auto run = [&](int slots) {
    NemoPool pool(slots, RB, resolver, /*nocache=*/false);
    for (int l = 0; l < L; ++l)
      for (int e = 0; e < E; ++e) pool.acquire(l, e);
    const uint64_t warm = pool.stats().physical_reads;
    for (int pass = 0; pass < 3; ++pass)
      for (int l = 0; l < L; ++l)
        for (int e = 0; e < E; ++e) {
          int s = pool.acquire(l, e);
          const uint8_t* p = pool.slot_data(s);
          const int k = l * E + e;
          if (p[0] != static_cast<uint8_t>(k * 31) ||
              p[RB - 1] != static_cast<uint8_t>(k * 31 + RB - 1)) {
            CHECK(false, "resident slot bytes match shard bytes");
            return std::pair<uint64_t, uint64_t>{0, 0};
          }
        }
    return std::pair<uint64_t, uint64_t>{warm, pool.stats().physical_reads};
  };
  auto full = run(L * E);
  CHECK(full.first == uint64_t(L * E) && full.second == full.first,
        "slots == all keys: 48 warm-up reads, then ZERO more reads over 3 passes");
  auto part = run(L * E / 2);
  CHECK(part.second > part.first, "slots < keys: reads continue (streaming)");
  fs::remove_all(dir);

  // Lazily-zeroed slot memory must still read as zero before the first fill.
  NemoPool z(8, RB, resolver, false);
  const uint8_t* p0 = z.slot_data(0);
  bool zero = true;
  for (int i = 0; i < 8 * RB; ++i) zero = zero && p0[i] == 0;
  CHECK(zero, "unfilled slot memory is zero");
}

static void test_mlx_cache_management() {
  std::puts("[mlx] cache limit + clear");
  clear_env();
  CHECK(mb::apply_mlx_limits_at_load("{}") == 0,
        "no dynamic budget + no env: MLX defaults left untouched");
  setenv_("VEYSTRIX_CPP_MLX_CACHE_BYTES", "268435456");
  CHECK(mb::apply_mlx_limits_at_load("{}") == 268435456, "env cache limit applied");
  const size_t prev = mx::set_cache_limit(268435456);
  CHECK(prev == 268435456, "mx::set_cache_limit reflects the applied value");
  unsetenv_("VEYSTRIX_CPP_MLX_CACHE_BYTES");
  const size_t dyn = mb::apply_mlx_limits_at_load("{\"resident\": false}");
  CHECK(dyn > 0 && dyn <= 2 * GiB, "dynamic budget -> default cache cap <= 2 GiB");
  mx::array a = mx::ones({1024, 1024});
  mx::eval(mx::sum(a));
  mb::clear_mlx_cache();
  CHECK(mx::get_cache_memory() == 0, "clear_mlx_cache empties the allocator cache");
  CHECK(mb::idle_clear_ms() == 2000, "idle clear default 2000 ms");
  setenv_("VEYSTRIX_CPP_IDLE_CLEAR_MS", "0");
  CHECK(mb::idle_clear_ms() == 0, "idle clear disabled with 0");
  setenv_("VEYSTRIX_CPP_IDLE_CLEAR_MS", "zzz");
  CHECK(mb::idle_clear_ms() == 2000, "garbage idle ms -> default");
  clear_env();
}

static void test_inkling_residency_across_mac_tiers() {
  std::puts("[inkling] dynamic residency across 16..128 GB Macs");
  clear_env();
  // Higher RAM Macs (24..128 GB): full 24-layer residency when RAM is available, even at pressure=2.
  for (size_t gib : {24ULL, 32ULL, 48ULL, 64ULL, 96ULL, 128ULL}) {
    auto p = mb::plan_inkling_residency(gib * GiB, (gib - 6) * GiB, 0, 2);
    CHECK(p.core == 12 && p.extra == 12,
          ("full 24-layer residency on " + std::to_string(gib) + "GB at pressure=2").c_str());
  }
  // 24 GB Mac under severe low-RAM (< 3.6 GiB avail) must drop to 0 core layers (never force 4).
  auto low24 = mb::plan_inkling_residency(24 * GiB, 2 * GiB, 0, 1);
  CHECK(low24.core == 0 && low24.extra == 0, "24GB low avail drops core to 0");
  // Critical pressure (4) on any Mac tier limits residency to <= 2 core layers.
  auto crit64 = mb::plan_inkling_residency(64 * GiB, 48 * GiB, 0, 4);
  CHECK(crit64.core <= 2 && crit64.extra == 0, "64GB critical pressure=4 caps residency");
  // Dynamic prefill chunk grows with RAM tier when headroom allows.
  CHECK(mb::dynamic_prefill_chunk(2048, 64 * GiB, 32 * GiB, 4 * GiB, 1) == 8192,
        "64GB tier scales prefill chunk up to 8192");
  CHECK(mb::dynamic_prefill_chunk(2048, 32 * GiB, 16 * GiB, 4 * GiB, 1) == 4096,
        "32GB tier scales prefill chunk up to 4096");
}

int main() {
  try {
    test_profile_resolution();
    test_decision_and_refusal();
    test_nemo_pool_resident_zero_reads();
    test_mlx_cache_management();
    test_inkling_residency_across_mac_tiers();
  } catch (const std::exception& e) {
    std::printf("  FAIL exception: %s\n", e.what());
    ++g_failures;
  }
  std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "PASS", g_failures,
              g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
