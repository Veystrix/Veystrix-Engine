// Bit-exact parity test for the GDN delta-rule STEP KERNEL (production decode).
//
// The ops path (gated_delta_step) and the fused Metal kernel path
// (gated_delta_step_kernel) are NOT bit-identical — production decode uses the
// kernel. So the reference here is mlx_vlm's gated_delta_kernel (the exact
// scalar-gate, unmasked kernel the Python engine compiles), and we compare
// C++ kernel output byte-for-byte (uint32 view), not with a tolerance.
//
// Vectors are produced by cpp/engine/tools/gen_vectors.py with the same
// MLX 0.32.2 build and stored under each case dir as gdn_delta_*.npy.
#include "engine.hpp"

#include <mlx/mlx.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

static bool bit_equal_report(const mx::array& a, const mx::array& b,
                             const char* label, const std::string& name) {
  if (a.shape() != b.shape()) {
    std::fprintf(stderr, "[%s] %s shape differs\n", name.c_str(), label);
    return false;
  }
  mx::array av = mx::astype(mx::reshape(a, {-1}), mx::uint32);
  mx::array bv = mx::astype(mx::reshape(b, {-1}), mx::uint32);
  mx::eval(av, bv);
  if (av.size() != bv.size()) {
    std::fprintf(stderr, "[%s] %s size differs (%zu vs %zu)\n", name.c_str(),
                 label, av.size(), bv.size());
    return false;
  }
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  size_t diff = av.size();
  for (size_t i = 0; i < av.size(); ++i) {
    if (pa[i] != pb[i]) { diff = i; break; }
  }
  if (diff != av.size()) {
    std::fprintf(stderr,
                 "[%s] %s differs: first diff at word %zu/%zu "
                 "(0x%08x vs 0x%08x)\n",
                 name.c_str(), label, diff, av.size(), pa[diff], pb[diff]);
    return false;
  }
  return true;
}

static int run_case(const std::string& root, const std::string& name) {
  std::string dir = root + "/" + name;
  auto load = [&](const char* n) { return mx::load(dir + "/gdn_delta_" + n + ".npy"); };

  mx::array q = load("q"), k = load("k"), v = load("v");
  mx::array g = load("g"), beta = load("beta");
  mx::array state = load("state");
  mx::array ey = load("y"), estate = load("state_out");

  // Run the exact fused Metal kernel path (same one production decode uses).
  DeltaStepOutput out = gated_delta_step_kernel(q, k, v, g, beta, state);
  mx::eval(out.y, out.state);

  bool y_ok = bit_equal_report(out.y, ey, "y", name);
  bool s_ok = bit_equal_report(out.state, estate, "state_out", name);
  if (!y_ok || !s_ok) return 1;

  std::printf("  [%s] OK (delta kernel y + state_out bit-identical to Python kernel)\n",
              name.c_str());
  return 0;
}

int main(int argc, char** argv) {
  std::string root = argc > 1 ? argv[1] : "tests/vectors";

  // Read cases.json for case dir names (same minimal parse as test_engine_parity).
  std::ifstream f(root + "/cases.json");
  if (!f) {
    std::fprintf(stderr, "cases.json missing; run gen_vectors.py\n");
    return 2;
  }
  std::vector<std::string> cases;
  std::string line, cur;
  while (std::getline(f, line)) {
    if (line.find("\"dir\"") != std::string::npos) {
      auto q1 = line.find('"', line.find(':') + 1);
      auto q2 = line.find('"', q1 + 1);
      cur = line.substr(q1 + 1, q2 - q1 - 1);
    }
    if (!cur.empty() && line.find("}") != std::string::npos) {
      cases.push_back(cur);
      cur.clear();
    }
  }
  if (cases.empty()) {
    std::fprintf(stderr, "no cases parsed from cases.json\n");
    return 2;
  }

  int failed = 0;
  for (const auto& c : cases) failed += run_case(root, c);
  if (failed) {
    std::fprintf(stderr, "%d case(s) FAILED\n", failed);
    return 1;
  }
  std::printf("test_gdn_delta OK (%zu cases bit-identical to Python gated_delta_kernel)\n",
              cases.size());
  return 0;
}
