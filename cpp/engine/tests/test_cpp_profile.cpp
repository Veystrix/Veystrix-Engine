// CPU-only unit test for the Phase 0 opt-in profiler (cpp_profile.hpp).
// No MLX headers, no checkpoint, no link to veystrix_engine: the profiler is
// header-only. Build standalone:
//   xcrun clang++ -std=c++20 -O2 -Icpp/engine/src \
//       cpp/engine/tests/test_cpp_profile.cpp -o /tmp/test_cpp_profile
#include "cpp_profile.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace veystrix::engine;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// (c) instrument getenv to count reads of VEYSTRIX_CPP_NO_DIGEST. init() is
// the only place allowed to read it; the hot path must never call getenv.
// ---------------------------------------------------------------------------
static int g_no_digest_getenv_reads = 0;
using GetenvFn = char* (*)(const char*);
static GetenvFn g_real_getenv = nullptr;

extern "C" char* getenv(const char* name) {
  if (!g_real_getenv)
    g_real_getenv = reinterpret_cast<GetenvFn>(dlsym(RTLD_NEXT, "getenv"));
  if (name && std::strcmp(name, "VEYSTRIX_CPP_NO_DIGEST") == 0)
    ++g_no_digest_getenv_reads;
  return g_real_getenv(name);
}

static std::string make_temp_dir() {
  char tpl[] = "/tmp/cpp_profile_test_XXXXXX";
  char* d = ::mkdtemp(tpl);
  CHECK(d != nullptr);
  return std::string(d);
}

static bool file_exists(const std::string& p) {
  std::ifstream f(p);
  return f.good();
}

// Call every public profiler helper. `write` controls whether write_row() is
// invoked (it is, in both scenarios: the disabled case must still be a no-op).
static void call_all_helpers(cppprof::Profiler& pf, bool write) {
  auto t0 = pf.now();              // single branch when disabled
  (void)cppprof::ms_since(t0);     // free function
  (void)cppprof::physical_disk_bytes();
  (void)cppprof::mem_snapshot();   // dead stub, still callable
  pf.begin_step();
  pf.add_build(1.25);
  pf.add_route_sync(0, 2.5);
  pf.add_route_sync(2, 0.5);
  pf.add_read(3.0, 4096);
  pf.add_final_eval(0.75);
  pf.add_digest(0.125);
  pf.add_layer_cache(0, 7, 3, 2);
  if (write) {
    pf.step = 1;
    pf.write_row(1.0, 2.0, 3.0);
  }
}

static bool has(const std::string& s, const std::string& needle) {
  return s.find(needle) != std::string::npos;
}

int main() {
  // Start from a known environment regardless of the caller's shell.
  unsetenv("VEYSTRIX_CPP_PROFILE");
  unsetenv("VEYSTRIX_CPP_PROFILE_DIR");
  unsetenv("VEYSTRIX_CPP_NO_DIGEST");

  auto& pf = cppprof::profiler();

  // ---- (a) disabled: helpers must not create any profile.jsonl -------------
  const std::string dir_off = make_temp_dir();
  setenv("VEYSTRIX_CPP_PROFILE_DIR", dir_off.c_str(), 1);
  pf.init();  // VEYSTRIX_CPP_PROFILE is unset
  CHECK(!pf.enabled);
  CHECK(pf.dir == dir_off);
  call_all_helpers(pf, /*write=*/true);
  CHECK(!file_exists(dir_off + "/profile.jsonl"));
  std::fprintf(stderr, "PASS (a) disabled: no profile.jsonl created\n");

  // ---- (b) enabled: exactly one JSONL row, all B1 fields, correct values ---
  const std::string dir_on = make_temp_dir();
  setenv("VEYSTRIX_CPP_PROFILE", "1", 1);
  setenv("VEYSTRIX_CPP_PROFILE_DIR", dir_on.c_str(), 1);
  pf.init();
  CHECK(pf.enabled);
  CHECK(pf.dir == dir_on);
  call_all_helpers(pf, /*write=*/true);

  const std::string path = dir_on + "/profile.jsonl";
  CHECK(file_exists(path));
  std::ifstream in(path);
  std::string line;
  CHECK(static_cast<bool>(std::getline(in, line)));
  CHECK(!has(line, "FAIL"));  // guard against an empty/garbage row

  // Scalar fields (format is %.4f / %llu in write_row).
  CHECK(has(line, "\"step\":1"));
  CHECK(has(line, "\"build_ms\":1.2500"));
  CHECK(has(line, "\"route_sync_ms\":3.0000"));
  CHECK(has(line, "\"read_wall_ms\":3.0000"));
  CHECK(has(line, "\"bytes_requested\":4096"));
  CHECK(has(line, "\"bytes_physical\":"));  // >=0 delta from proc_pid_rusage
  CHECK(has(line, "\"final_eval_ms\":0.7500"));
  CHECK(has(line, "\"digest_ms\":0.1250"));
  CHECK(has(line, "\"active_gib\":1.0000"));
  CHECK(has(line, "\"cache_gib\":2.0000"));
  CHECK(has(line, "\"peak_gib\":3.0000"));

  // Per-layer route_sync + hits/misses/coalesced.
  CHECK(has(line, "\"route_sync_per_layer\":["));
  CHECK(has(line, "{\"l\":0,\"ms\":2.5000,\"h\":7,\"m\":3,\"c\":2}"));
  CHECK(has(line, "{\"l\":1,\"ms\":0.0000,\"h\":0,\"m\":0,\"c\":0}"));
  CHECK(has(line, "{\"l\":2,\"ms\":0.5000,\"h\":0,\"m\":0,\"c\":0}"));

  // Exactly one row.
  std::string extra;
  CHECK(!static_cast<bool>(std::getline(in, extra)));
  std::fprintf(stderr, "PASS (b) enabled: one JSONL row, all fields correct\n");

  // ---- (c) VEYSTRIX_CPP_NO_DIGEST read exactly once by init() --------------
  unsetenv("VEYSTRIX_CPP_PROFILE");
  setenv("VEYSTRIX_CPP_NO_DIGEST", "1", 1);
  g_no_digest_getenv_reads = 0;
  pf.init();
  CHECK(pf.no_digest);
  CHECK(g_no_digest_getenv_reads == 1);
  for (int i = 0; i < 100; ++i) call_all_helpers(pf, /*write=*/false);
  CHECK(g_no_digest_getenv_reads == 1);
  unsetenv("VEYSTRIX_CPP_NO_DIGEST");
  std::fprintf(stderr, "PASS (c) VEYSTRIX_CPP_NO_DIGEST read once, hot path never\n");

  std::fprintf(stderr, "ALL CPP PROFILE TESTS PASSED\n");
  return 0;
}
