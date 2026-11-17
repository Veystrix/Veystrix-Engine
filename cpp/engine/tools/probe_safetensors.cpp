// Probe: can MLX C++ load_safetensors read real expert tensors bit-exactly?
#include <mlx/mlx.h>
#include <cstdio>
#include <cstdint>
namespace mx = mlx::core;

int main(int argc, char** argv) {
  if (argc < 2) { std::printf("usage: probe_safetensors <shard>\n"); return 2; }
  auto [weights, meta] = mx::load_safetensors(std::string(argv[1]));
  std::printf("loaded %zu tensors\n", weights.size());
  // Find one Q4 G32 gate weight and report shape/dtype.
  for (auto& [name, arr] : weights) {
    if (name.find("switch_mlp.gate_proj.weight") != std::string::npos) {
      std::printf("%s shape=(", name.c_str());
      for (auto d : arr.shape()) std::printf("%d,", d);
      std::printf(") dtype=%d\n", static_cast<int>(static_cast<mx::Dtype::Val>(arr.dtype())));
      mx::eval(arr);
      const uint32_t* p = arr.data<uint32_t>();
      std::printf("  first words: %08x %08x\n", p[0], p[1]);
      break;
    }
  }
  return 0;
}
