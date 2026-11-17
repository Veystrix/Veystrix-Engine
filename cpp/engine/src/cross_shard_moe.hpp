// Cross-shard streaming MoE: the variant of StreamingMoE whose 9 expert
// tensors live across more than one safetensors shard (layers 5/8 in the
// 10-layer reduced config, 12 of 48 layers in the full model). Moved out of
// tools/real_checkpoint_generate.cpp so it can be tested against a synthetic
// 2-shard checkpoint. Slot bookkeeping is delegated to slot_plan.cpp, the same
// fixed code path StreamingMoE uses.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "engine.hpp"
#include "moe_stream.hpp"

namespace veystrix::engine {

class CrossShardMoE {
 public:
  // `cfg` maps "shard.<tensor.key>" -> shard filename and carries the prompt.
  CrossShardMoE(const std::string& dir,
                const std::map<std::string, std::string>& cfg,
                const std::string& prefix, int num_slots, int group_size,
                int bits, bool deep_copy = false);
  ~CrossShardMoE();

  CrossShardMoE(const CrossShardMoE&) = delete;
  CrossShardMoE& operator=(const CrossShardMoE&) = delete;

  StreamForwardResult forward(const mx::array& x, const mx::array& route,
                              const mx::array& scores);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace veystrix::engine
