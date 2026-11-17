// Bit-exact test for the native streaming MoE path.
//
//   /tmp/test_moe_stream <vectordir> <checkpoint-dir>
//
// The vector dir contains only x/route/scores/expected (+ meta.json) produced
// by tools/gen_moe_stream_vectors.py. Expert weights are read from the REAL
// Qwen shard inside StreamingMoE via plan_remap + run_batched_reads.
#include "moe_stream.hpp"

#include <mlx/mlx.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "io_pager.hpp"

namespace mx = mlx::core;
using namespace veystrix;
using namespace veystrix::engine;

static bool bit_equal(const mx::array& a, const mx::array& b) {
  if (a.shape() != b.shape()) return false;
  mx::array av = mx::view(mx::reshape(mx::astype(a, mx::float32), {-1}), mx::uint32);
  mx::array bv = mx::view(mx::reshape(mx::astype(b, mx::float32), {-1}), mx::uint32);
  mx::eval(av, bv);
  if (av.size() != bv.size()) return false;
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  for (size_t i = 0; i < av.size(); ++i)
    if (pa[i] != pb[i]) return false;
  return true;
}

// Minimal meta.json parse (flat ints/strings only).
static std::string json_get(const std::string& s, const std::string& key) {
  std::string pat = "\"" + key + "\"";
  size_t p = s.find(pat);
  if (p == std::string::npos) return "";
  size_t colon = s.find(':', p + pat.size());
  size_t b = s.find_first_not_of(" \t\n\r", colon + 1);
  if (b == std::string::npos) return "";
  if (s[b] == '"') {
    size_t e = s.find('"', b + 1);
    return s.substr(b + 1, e - b - 1);
  }
  size_t e = s.find_first_of(",}\n", b);
  return s.substr(b, e - b);
}

// Verify the pager's pure remap semantics independently of I/O (dedup misses,
// resident reuse, fail-closed on pool exhaustion, router-rank slot order).
static bool check_pager_remap() {
  std::vector<int32_t> route = {5, 3, 5, 7};
  std::vector<int32_t> pairs = {3, 0};            // expert 3 already in slot 0
  std::vector<int32_t> free_slots = {1, 2, 3};
  uint8_t base[64] = {};
  auto plan = plan_remap(route, pairs, free_slots, /*fd=*/-1, /*header=*/0,
                         /*row_bytes=*/8, base, /*slot_stride=*/8);
  // First appearance order, dedup: 5->1, 3->0(resident), 5->1, 7->2.
  std::vector<int32_t> want = {1, 0, 1, 2};
  if (plan.slots != want || plan.reads.size() != 2) return false;
  if (plan.reads[0].offset != 5 * 8 || plan.reads[1].offset != 7 * 8) return false;

  auto closed = plan_remap(route, pairs, {1}, -1, 0, 8, base, 8);
  if (closed.reads.size() != 1) return false;  // only expert 5 fits, 7 -> -1
  if (closed.slots[3] != -1) return false;
  return true;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <vectordir> <checkpoint-dir>\n", argv[0]);
    return 2;
  }
  std::string dir = argv[1], ckpt = argv[2];

  if (!check_pager_remap()) {
    std::fprintf(stderr, "plan_remap semantics FAILED\n");
    return 1;
  }

  std::ifstream mf(dir + "/meta.json");
  if (!mf) { std::fprintf(stderr, "meta.json missing\n"); return 2; }
  std::stringstream ms;
  ms << mf.rdbuf();
  std::string meta = ms.str();
  int layer = std::stoi(json_get(meta, "layer"));
  int group = std::stoi(json_get(meta, "G"));
  int bits = std::stoi(json_get(meta, "bits"));
  std::string prefix = json_get(meta, "prefix");

  auto load = [&](const char* n) { return mx::load(dir + "/" + n + ".npy"); };
  mx::array x = load("x");
  mx::array route = load("route");
  mx::array scores = load("scores");
  mx::array expected = load("expected");

  // Pool sized generously above top-k; dedup means a single miss is read once.
  StreamingMoE moe(ckpt, layer, /*num_slots=*/16, group, bits, /*read_width=*/8,
                   prefix);
  StreamForwardResult res = moe.forward(x, route, scores);
  mx::eval(res.output, expected);

  if (!bit_equal(res.output, expected)) {
    mx::array ov = mx::view(mx::reshape(mx::astype(res.output, mx::float32), {-1}), mx::uint32);
    mx::array ev = mx::view(mx::reshape(mx::astype(expected, mx::float32), {-1}), mx::uint32);
    mx::eval(ov, ev);
    size_t n = ov.size() < ev.size() ? ov.size() : ev.size();
    const uint32_t* a = ov.data<uint32_t>();
    const uint32_t* b = ev.data<uint32_t>();
    size_t diff = n;
    for (size_t i = 0; i < n; ++i) if (a[i] != b[i]) { diff = i; break; }
    std::fprintf(stderr,
                 "streaming MoE differs: out=%zu exp=%zu first diff %zu "
                 "(0x%08x vs 0x%08x)\n", res.output.size(), expected.size(),
                 diff, diff < n ? a[diff] : 0, diff < n ? b[diff] : 0);
    return 1;
  }

  // Second forward with the same route must be bit-identical too, and must hit
  // the resident slots (no disk reads): exercises the refcount / residency path.
  StreamForwardResult again = moe.forward(x, route, scores);
  mx::eval(again.output);
  if (!again.misses.empty() || !bit_equal(again.output, expected)) {
    std::fprintf(stderr, "resident replay differs or re-read experts\n");
    return 1;
  }

  // Convenience free function on a fresh pool must also be bit-exact.
  mx::array one = moe_stream_forward(ckpt, layer, x, route, scores,
                                     /*num_slots=*/16, group, bits, 8, prefix);
  mx::eval(one);
  if (!bit_equal(one, expected)) {
    std::fprintf(stderr, "moe_stream_forward differs\n");
    return 1;
  }

  std::printf("test_moe_stream OK\n");
  return 0;
}
