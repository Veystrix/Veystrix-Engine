// Bit-exact parity test for the S>1 Mamba prefill path (plan P5b) against
// vectors from tools/gen_mamba_prefill_vectors.py (real mlx_lm primitives).
#include "mamba_prefill.hpp"

#include <mlx/mlx.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

static bool bit_equal(const mx::array& a, const mx::array& b) {
  if (a.shape() != b.shape() || a.dtype() != b.dtype()) return false;
  mx::array av = mx::view(mx::reshape(a, {-1}), mx::uint32);
  mx::array bv = mx::view(mx::reshape(b, {-1}), mx::uint32);
  mx::eval(av, bv);
  if (av.size() != bv.size()) return false;
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  for (size_t i = 0; i < av.size(); ++i)
    if (pa[i] != pb[i]) return false;
  return true;
}

static double get_num(const std::string& s, const std::string& key) {
  auto at = s.find("\"" + key + "\"");
  if (at == std::string::npos) return 0;
  auto colon = s.find(':', at);
  size_t i = colon + 1;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\n')) ++i;
  std::string n;
  for (; i < s.size() && s[i] != ',' && s[i] != '\n' && s[i] != '}'; ++i) n += s[i];
  return std::stod(n);
}

static int run_case(const std::string& root) {
  std::ifstream mf(root + "/meta.json");
  std::string meta((std::istreambuf_iterator<char>(mf)),
                   std::istreambuf_iterator<char>());
  MambaConfig cfg;
  cfg.num_heads = static_cast<int>(get_num(meta, "H"));
  cfg.head_dim = static_cast<int>(get_num(meta, "Dh"));
  cfg.n_groups = static_cast<int>(get_num(meta, "G"));
  cfg.ssm_state_size = static_cast<int>(get_num(meta, "Ds"));
  cfg.conv_kernel = static_cast<int>(get_num(meta, "K"));
  cfg.intermediate_size = static_cast<int>(get_num(meta, "I"));
  cfg.conv_dim = static_cast<int>(get_num(meta, "CD"));
  cfg.hidden_size = static_cast<int>(get_num(meta, "hidden"));
  cfg.eps = static_cast<float>(get_num(meta, "eps"));
  cfg.time_step_min = static_cast<float>(get_num(meta, "time_step_min"));
  cfg.time_step_max = static_cast<float>(get_num(meta, "time_step_max"));

  auto load = [&](const std::string& n) { return mx::load(root + "/" + n + ".npy"); };
  MambaWeights w(
      Linear(load("in_w")), load("conv_w"), load("conv_b"), load("A_log"),
      load("dt_bias"), load("D"), load("norm_w"), Linear(load("out_w")));

  const int chunks = static_cast<int>(get_num(meta, "chunks"));
  std::optional<mx::array> conv_state, ssm_state;
  int failures = 0;
  for (int c = 0; c < chunks; ++c) {
    mx::array cs = conv_state.value_or(mx::zeros(
        mx::Shape{1, cfg.conv_kernel - 1, cfg.conv_dim}, mx::float32));
    mx::array ss = ssm_state.value_or(mx::zeros(
        mx::Shape{1, cfg.num_heads, cfg.head_dim, cfg.ssm_state_size}, mx::float32));
    MambaCacheState cache(cs, ss);
    MambaLayerOutput got = mamba_layer_prefill(load("x" + std::to_string(c)),
                                               w, cache, cfg);
    conv_state = got.conv_state;
    ssm_state = got.ssm_state;
    if (!bit_equal(got.output, load("expected_out" + std::to_string(c)))) {
      std::cerr << "FAIL " << root << " chunk " << c << ": output\n";
      ++failures;
    }
    if (!bit_equal(got.conv_state, load("expected_conv_state" + std::to_string(c)))) {
      std::cerr << "FAIL " << root << " chunk " << c << ": conv_state\n";
      ++failures;
    }
    if (!bit_equal(got.ssm_state, load("expected_ssm_state" + std::to_string(c)))) {
      std::cerr << "FAIL " << root << " chunk " << c << ": ssm_state\n";
      ++failures;
    }
  }
  return failures;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "usage: test_mamba_prefill <vectors-dir>\n"; return 2; }
  int failures = 0;
  for (const char* name : {"s2", "s7", "s64", "s128"})
    failures += run_case(std::string(argv[1]) + "/" + name);
  if (failures == 0)
    std::cout << "mamba_prefill: all S>1 cases bit-exact vs mlx_lm\n";
  return failures == 0 ? 0 : 1;
}
