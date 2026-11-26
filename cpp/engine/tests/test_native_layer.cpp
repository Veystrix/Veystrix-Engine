// End-to-end native test: load a real MoE layer's weights from the checkpoint
// ENTIRELY IN C++ (no Python feeding weights), then run moe_block and compare
// against the Python-computed expected vector bit-for-bit.
//
// Usage: test_native_layer <shard> <prefix> <vectordir>
//   shard: ...model-00001-of-00022.safetensors
//   prefix: language_model.model.layers.0.mlp.switch_mlp
//   vectordir: dir with x.npy route.npy scores.npy expected.npy (from gen_real_vectors.py)
#include "expert_store.hpp"
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
  if (argc < 4) { std::printf("usage: test_native_layer <shard> <prefix> <vectordir>\n"); return 2; }
  std::string shard = argv[1], prefix = argv[2], vdir = argv[3];
  auto load = [&](const char* n) { return mx::load(vdir + "/" + n + ".npy"); };

  // Load the full real layer natively (all 512 experts) from the shard.
  ExpertWeights w = load_layer(shard, prefix, 32, 4);
  mx::array x = load("x"), route = load("route"), scores = load("scores");
  mx::array expected = load("expected");
  mx::array out = moe_block(x, route, scores, w);
  mx::eval(out);
  if (!bit_equal(out, expected)) {
    std::fprintf(stderr, "test_native_layer FAIL: output differs from Python reference\n");
    return 1;
  }
  std::printf("test_native_layer OK (C++-loaded real layer bit-identical to Python)\n");
  return 0;
}
