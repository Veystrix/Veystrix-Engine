#include "expert_store.hpp"

#include <mlx/mlx.h>

#include <stdexcept>

namespace veystrix::engine {

static mx::array view_bf16_if_needed(mx::array a) {
  // Scales/biases are stored as uint16 holding bfloat16 bit patterns.
  // Dtype::Val: bfloat16 has a distinct enum value; uint16 loads as uint16.
  if (static_cast<mx::Dtype::Val>(a.dtype()) == static_cast<mx::Dtype::Val>(mx::uint16)) {
    return mx::view(a, mx::bfloat16);  // reinterprets the same bytes/shape
  }
  return a;
}

Q4Projection load_projection(const std::string& shard_path,
                             const std::string& prefix,
                             const std::string& proj,
                             int group_size) {
  auto [weights, meta] = mx::load_safetensors(shard_path);
  auto get = [&](const std::string& suffix) -> mx::array {
    std::string key = prefix + "." + proj + "." + suffix;
    auto it = weights.find(key);
    if (it == weights.end())
      throw std::runtime_error("missing tensor in shard: " + key);
    return it->second;
  };
  mx::array w = get("weight");
  mx::array s = view_bf16_if_needed(get("scales"));
  mx::array b = view_bf16_if_needed(get("biases"));
  (void)group_size;  // shape is self-describing from the checkpoint
  return Q4Projection(w, s, b);
}

ExpertWeights load_layer(const std::string& shard_path,
                         const std::string& prefix,
                         int group_size, int bits) {
  return ExpertWeights(
      load_projection(shard_path, prefix, "gate_proj", group_size),
      load_projection(shard_path, prefix, "up_proj", group_size),
      load_projection(shard_path, prefix, "down_proj", group_size),
      group_size, bits);
}

}  // namespace veystrix::engine
