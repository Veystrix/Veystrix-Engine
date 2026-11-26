// Bit-exact parity for the Qwen4 final head (gated_residual no-combine ->
// lm_head) against vectors from the real mlx_vlm op path.
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
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  for (size_t i = 0; i < av.size(); ++i) if (pa[i] != pb[i]) return false;
  return true;
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "cpp/engine/tests/vectors_final_head";
  auto load = [&](const char* n) { return mx::load(dir + "/" + n + ".npy"); };
  mx::array hidden = load("hidden"), nw = load("norm_weight");
  mx::array dw = load("down_w"), uw = load("up_w"), lw = load("lm_head_w");
  mx::array expected = load("expected");
  mx::array logits = final_head(hidden, nw, dw, uw, lw, 4, 32, 1e-6f);
  mx::eval(logits);
  if (!bit_equal(logits, expected)) {
    mx::array lv = mx::astype(mx::reshape(logits, {-1}), mx::uint32);
    mx::array ev = mx::astype(mx::reshape(expected, {-1}), mx::uint32);
    mx::eval(lv, ev);
    size_t n = lv.size();
    const uint32_t* a = lv.data<uint32_t>();
    const uint32_t* b = ev.data<uint32_t>();
    size_t d = n;
    for (size_t i = 0; i < n; ++i) if (a[i] != b[i]) { d = i; break; }
    std::fprintf(stderr, "final_head FAIL first diff %zu (0x%08x vs 0x%08x)\n",
                 d, d < n ? a[d] : 0, d < n ? b[d] : 0);
    return 1;
  }
  std::printf("test_final_head OK\n");
  return 0;
}
