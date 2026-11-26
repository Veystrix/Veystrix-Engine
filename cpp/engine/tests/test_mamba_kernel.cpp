// Bit-exact parity test for the fused Mamba/SSM decode STEP KERNEL.
//
// The ops fallback (mamba_ssm_step_ops) is NOT bit-identical to the fused Metal
// kernel — production decode uses the kernel. The reference here is the real
// stock Python kernel `mlx_lm.models.ssm.ssm_update_kernel` (the S=1 decode
// path), and we compare C++ kernel output byte-for-byte (uint32 view) for BOTH
// `y` and `state_out`, not with a tolerance.
//
// Vectors are produced by cpp/engine/tools/gen_mamba_vectors.py with the same
// MLX 0.32.2 build and stored under tests/vectors_mamba_kernel/.
#include "mamba_kernel.h"

#include <mlx/mlx.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

// Copy of bit_equal_report from test_gdn_delta.cpp:25-53 (zero tolerance).
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

static std::string read_all(const std::string& path) {
  std::ifstream f(path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Parse a JSON string value like "tmin": "0.001" or "inf" from meta.json.
static float meta_float(const std::string& txt, const std::string& key,
                        const std::string& name) {
  std::string k = "\"" + key + "\":";
  auto p = txt.find(k);
  if (p == std::string::npos) {
    std::fprintf(stderr, "[%s] meta.json missing %s\n", name.c_str(), key.c_str());
    return std::nanf("");
  }
  auto q1 = txt.find('"', p + k.size());
  auto q2 = txt.find('"', q1 + 1);
  std::string v = txt.substr(q1 + 1, q2 - q1 - 1);
  if (v == "inf") return std::numeric_limits<float>::infinity();
  if (v == "-inf") return -std::numeric_limits<float>::infinity();
  return std::stof(v);
}

static int run_case(const std::string& root, const std::string& name) {
  std::string dir = root + "/" + name;
  auto load = [&](const char* n) { return mx::load(dir + "/" + n + ".npy"); };

  mx::array X = load("X"), A_log = load("A_log");
  mx::array B = load("B"), C = load("C"), D = load("D");
  mx::array dt_raw = load("dt_raw"), dt_bias = load("dt_bias");
  mx::array state = load("state");
  mx::array ey = load("y"), estate = load("state_out");

  std::string meta = read_all(dir + "/meta.json");
  float tmin = meta_float(meta, "tmin", name);
  float tmax = meta_float(meta, "tmax", name);
  if (std::isnan(tmin) || std::isnan(tmax)) return 1;

  // Run the exact fused Metal kernel path (same one production decode uses).
  SSMStepOutput out = mamba_ssm_step_kernel(
      X, A_log, B, C, D, dt_raw, dt_bias, state, tmin, tmax);
  mx::eval(out.y, out.state);

  bool y_ok = bit_equal_report(out.y, ey, "y", name);
  bool s_ok = bit_equal_report(out.state, estate, "state_out", name);
  if (!y_ok || !s_ok) return 1;

  std::printf("  [%s] OK (SSM kernel y + state_out bit-identical to Python kernel)\n",
              name.c_str());
  return 0;
}

// Extract every "dir": "..." value from cases.json.
static std::vector<std::string> read_cases(const std::string& root) {
  std::vector<std::string> cases;
  std::string txt = read_all(root + "/cases.json");
  const std::string key = "\"dir\":";
  size_t pos = 0;
  while ((pos = txt.find(key, pos)) != std::string::npos) {
    auto q1 = txt.find('"', pos + key.size());
    auto q2 = txt.find('"', q1 + 1);
    if (q1 != std::string::npos && q2 != std::string::npos)
      cases.push_back(txt.substr(q1 + 1, q2 - q1 - 1));
    pos = q2 + 1;
  }
  return cases;
}

int main(int argc, char** argv) {
  std::string root = argc > 1 ? argv[1] : "tests/vectors_mamba_kernel";

  std::vector<std::string> cases = read_cases(root);
  if (cases.empty()) {
    std::fprintf(stderr, "no cases parsed from cases.json (run gen_mamba_vectors.py)\n");
    return 2;
  }

  int failed = 0;
  for (const auto& c : cases) failed += run_case(root, c);
  if (failed) {
    std::fprintf(stderr, "%d case(s) FAILED\n", failed);
    return 1;
  }
  std::printf("test_mamba_kernel OK (%zu cases bit-identical to Python ssm_update_kernel)\n",
              cases.size());
  return 0;
}
