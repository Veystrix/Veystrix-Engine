// Sorted Nemotron relu2 MoE parity (plan P5b) vs real mlx_lm SwitchMLP
// sorted-branch vectors (tools/gen_relu2_sorted_vectors.py).
#include "engine.hpp"
#include <mlx/mlx.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <string>
namespace mx = mlx::core;
using namespace veystrix::engine;

static bool bit_equal(const mx::array& a, const mx::array& b) {
  if (a.shape() != b.shape()) return false;
  mx::array av = mx::view(mx::reshape(a, {-1}), mx::uint32);
  mx::array bv = mx::view(mx::reshape(b, {-1}), mx::uint32);
  mx::eval(av, bv);
  if (av.size() != bv.size()) return false;
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  for (size_t i = 0; i < av.size(); ++i) if (pa[i] != pb[i]) return false;
  return true;
}

int main(int argc, char** argv) {
  std::string root = argc > 1 ? argv[1] : "cpp/engine/tests/vectors_relu2_sorted";
  int failures = 0;
  for (const char* name : {"t16", "t64", "t128"}) {
    std::string dir = root + "/" + name;
    auto load = [&](const char* n) { return mx::load(dir + "/" + n + ".npy"); };
    Q4Projection up(load("up_w"), load("up_s"), load("up_b"));
    Q4Projection down(load("dn_w"), load("dn_s"), load("dn_b"));
    mx::array out = moe_block_relu2_sorted(
        load("x"), load("route"), load("scores"), up, down);
    mx::eval(out);
    if (!bit_equal(out, load("expected"))) {
      std::cerr << "FAIL sorted relu2 " << name << "\n";
      ++failures;
    }
  }
  if (failures == 0)
    std::cout << "relu2_sorted: all cases bit-exact vs mlx_lm sorted branch\n";
  return failures == 0 ? 0 : 1;
}
