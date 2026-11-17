// Smoke test: the C++ engine compiles, links against MLX 0.32.2, and the
// router + MoE compute produce correctly-shaped finite outputs. Bit-exact
// parity against the Python path is covered by test_engine_parity.cpp.
#include "engine.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

int main() {
  // Router: 2 tokens, 8 experts, top-3.
  std::vector<float> logits(2 * 8);
  for (int t = 0; t < 2; ++t)
    for (int e = 0; e < 8; ++e)
      logits[t * 8 + e] = std::sin(float(t * 8 + e) * 0.7f);
  mx::array lg(logits.begin(), {2, 8}, mx::float32);
  RouterOutput r = router(lg, 3, 0.0f);
  assert(r.indices.shape() == mx::Shape({2, 3}));
  assert(r.scores.shape() == mx::Shape({2, 3}));
  mx::eval(r.indices, r.scores);
  const int32_t* ids = r.indices.data<int32_t>();
  for (int i = 0; i < 6; ++i) assert(ids[i] >= 0 && ids[i] < 8);
  const float* sc = r.scores.data<float>();
  for (int t = 0; t < 2; ++t) {
    float s = sc[t * 3] + sc[t * 3 + 1] + sc[t * 3 + 2];
    assert(std::fabs(s - 1.0f) < 1e-5f);  // renorm -> sums to 1
  }

  // Tiny MoE: 4 experts, hidden=inter=32, group=32 (one group per row).
  // Weight shape (n_experts, out, in/8) = (E, out, 4) uint32 (8 codes/word).
  // scales/biases shape (E, out, in/group) = (E, out, 1).
  constexpr int E = 4, H = 32, I = 32, G = 32, K = 2;
  const int WORDS = H / 8;
  std::vector<uint32_t> w(E * H * WORDS, 0);
  for (int e = 0; e < E; ++e)
    for (int row = 0; row < H; ++row)
      for (int word = 0; word < WORDS; ++word) {
        uint32_t v = 0;
        for (int nib = 0; nib < 8; ++nib)
          v |= uint32_t((e + row + nib) % 16) << (nib * 4);
        w[(e * H + row) * WORDS + word] = v;
      }
  std::vector<float> s(E * H, 0.5f), b(E * H, 0.01f);
  auto mkproj = [&](int out) {
    return Q4Projection(
        mx::array(w.begin(), {E, out, WORDS}, mx::uint32),
        mx::array(s.begin(), {E, out, 1}, mx::float32),
        mx::array(b.begin(), {E, out, 1}, mx::float32));
  };
  ExpertWeights ew(mkproj(H), mkproj(I), mkproj(H), G, 4);

  std::vector<float> x(1 * H, 0.25f);
  mx::array xa(x.begin(), {1, H}, mx::float32);
  std::vector<int32_t> route{0, 1};
  mx::array rt(route.begin(), {1, K}, mx::int32);
  std::vector<float> score{0.6f, 0.4f};
  mx::array sc2(score.begin(), {1, K}, mx::float32);

  mx::array out = moe_block(xa, rt, sc2, ew);
  assert(out.shape() == mx::Shape({1, H}));
  mx::eval(out);
  const float* o = out.data<float>();
  for (int i = 0; i < H; ++i) assert(std::isfinite(o[i]));

  std::printf("test_engine_smoke OK\n");
  return 0;
}
