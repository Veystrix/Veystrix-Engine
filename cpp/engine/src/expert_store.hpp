// Native safetensors expert loader: loads a full MoE layer's Q4 affine
// gate/up/down projections directly from the checkpoint via MLX C++
// load_safetensors, so the C++ engine does not need Python to feed weights.
#pragma once
#include <string>

#include "engine.hpp"

namespace veystrix::engine {

// Load all experts of one MoE layer's projection from a shard.
// shard_path: absolute path to the .safetensors file containing this layer.
// prefix: e.g. "language_model.model.layers.0.mlp.switch_mlp"
// scales/biases are stored as uint16 (bf16 bit patterns); they are viewed as
// bfloat16 exactly as store.to_mx does.
Q4Projection load_projection(const std::string& shard_path,
                             const std::string& prefix,
                             const std::string& proj,  // "gate_proj"/"up_proj"/"down_proj"
                             int group_size);

// Load gate/up/down for one layer into an ExpertWeights bundle.
ExpertWeights load_layer(const std::string& shard_path,
                         const std::string& prefix,
                         int group_size = 32, int bits = 4);

}  // namespace veystrix::engine
