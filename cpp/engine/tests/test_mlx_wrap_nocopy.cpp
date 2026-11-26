// Phase A3 (Opus plan): decide whether MLX 0.32.2's array(void*, shape, dtype,
// deleter) constructor wraps raw buffers WITHOUT a copy, or silently copies.
// This decides whether the per-call expert buffer wrapping in moe_stream.cpp /
// cross_shard_moe.cpp / nemo_pool.cpp is a hidden per-token copy cost.
//
// Build: xcrun clang++ -std=c++20 -O2 -I$MLX/include <this> -L$MLX/lib -lmlx -o /tmp/test_mlx_wrap
// Run:   /tmp/test_mlx_wrap   (needs Metal; runs on the default GPU stream)
#include <mlx/mlx.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

namespace mx = mlx::core;

struct Case { const char* name; size_t rows; int dim0, dim1; mx::Dtype dt; size_t elem; };

static bool check(const Case& c, void* raw, size_t bytes, bool* copied_out) {
  mx::Shape shape{(int)c.rows, c.dim0, c.dim1};
  // deleter sets a flag if/when it is called with the ORIGINAL pointer (no-copy);
  // MLX calls it with its own copy pointer in the copy case.
  bool deleter_called_with_raw = false;
  uint8_t* expected = static_cast<uint8_t*>(raw);
  auto deleter = [&](void* p) { if (p == expected) deleter_called_with_raw = true; };
  mx::array a(raw, shape, c.dt, deleter);
  mx::eval(a);  // force materialization / any deferred copy
  const uint8_t* dp = a.data<uint8_t>();
  bool same_ptr = (dp == expected);
  *copied_out = !same_ptr;
  return same_ptr;
}

int main() {
  // Representative Qwen/Nemotron expert tensor geometries.
  std::vector<Case> cases = {
    {"qwen_gate_w",   10, 640, 320, mx::uint32, 4},   // packed Q4 G32
    {"qwen_gate_s",   10, 640,  80, mx::bfloat16, 2},
    {"nemo_fc1_w",    16, 2688, 336, mx::uint32, 4},  // Q4 G64
    {"nemo_fc1_s",    16, 2688,  42, mx::bfloat16, 2},
  };
  std::vector<size_t> rowsets = {10, 16, 64, 256, 349};
  int failures = 0, copies = 0, nocopies = 0;

  for (const auto& base : cases) {
    for (size_t rows : rowsets) {
      size_t bytes = rows * (size_t)base.dim0 * (size_t)base.dim1 * base.elem;
      // Allocator A: posix_memalign 64 (what the engine uses today).
      void* p1 = nullptr;
      if (::posix_memalign(&p1, 64, bytes) != 0) { std::fprintf(stderr, "memalign failed\n"); return 2; }
      std::memset(p1, 0xAB, bytes);
      bool copied1 = false;
      bool ok1 = check({base.name, rows, base.dim0, base.dim1, base.dt, base.elem}, p1, bytes, &copied1);
      // Allocator B: page-rounded mmap MAP_ANON (what Python expert_pool uses).
      size_t page = (size_t)sysconf(_SC_PAGESIZE);
      size_t mapped = ((bytes + page - 1) / page) * page;
      void* p2 = ::mmap(nullptr, mapped, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
      if (p2 == MAP_FAILED) { std::fprintf(stderr, "mmap failed\n"); std::free(p1); return 2; }
      std::memset(p2, 0xCD, mapped);
      bool copied2 = false;
      bool ok2 = check({base.name, rows, base.dim0, base.dim1, base.dt, base.elem}, p2, mapped, &copied2);
      (void)ok1; (void)ok2;
      if (copied1) copies++; else nocopies++;
      std::printf("%-14s rows=%-4zu bytes=%-10zu memalign=%s  mmap=%s\n",
                  base.name, rows, bytes,
                  copied1 ? "COPIED" : "no-copy",
                  copied2 ? "COPIED" : "no-copy");
      ::munmap(p2, mapped);
      std::free(p1);
    }
  }
  std::printf("\nSUMMARY: no-copy cases=%d, copied cases=%d\n", nocopies, copies);
  std::printf("VERDICT: %s\n", copies > 0
      ? "array(void*,...) COPIES for at least some buffers -> hidden per-call copy cost is REAL"
      : "array(void*,...) wraps WITHOUT copy for all tested buffers -> no hidden copy cost");
  return failures ? 1 : 0;
}
