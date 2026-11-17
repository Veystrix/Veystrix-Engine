// Bit-exact parity for the Nemotron relu2 MoE block vs Python vectors.
#include "engine.hpp"
#include <mlx/mlx.h>
#include <cstdio>
#include <cstdint>
#include <string>
namespace mx = mlx::core;
using namespace veystrix::engine;

static bool bit_equal(const mx::array& a, const mx::array& b) {
  if (a.shape() != b.shape()) return false;
  mx::array av = mx::astype(mx::reshape(a, {-1}), mx::uint32);
  mx::array bv = mx::astype(mx::reshape(b, {-1}), mx::uint32);
  mx::eval(av, bv);
  if (av.size() != bv.size()) return false;
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  for (size_t i = 0; i < av.size(); ++i) if (pa[i] != pb[i]) return false;
  return true;
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "tests/vectors/case3_nemotron_relu2";
  auto load = [&](const char* n) { return mx::load(dir + "/" + n + ".npy"); };
  mx::array x = load("x"), route = load("route"), scores = load("scores");
  mx::array uw = load("up_w"), us = load("up_s"), ub = load("up_b");
  mx::array dw = load("dn_w"), ds = load("dn_s"), db = load("dn_b");
  mx::array expected = load("expected");
  Q4Projection up_proj(uw, us, ub), down_proj(dw, ds, db);
  mx::array out = moe_block_relu2(x, route, scores, up_proj, down_proj);
  mx::eval(out);
  if (!bit_equal(out, expected)) {
    std::fprintf(stderr, "test_relu2_parity FAIL\n");
    return 1;
  }
  std::printf("test_relu2_parity OK (Nemotron relu2 MoE bit-identical to Python)\n");
  return 0;
}
