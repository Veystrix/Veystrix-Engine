#pragma once

// Bit-exact C++ port of the distribution-based sampling path in
// veystrix/sampling.py (Sampler.distribution / Sampler._mask /
// Sampler.__call__ non-greedy branch).  The greedy path already lives in
// engine.cpp as sample_greedy() and is intentionally not duplicated here.
//
// Both MLX Python and MLX C++ share the same PRNG implementation: seeding with
// mx.random.seed(seed) (Python) and mlx::core::random::seed(seed) (C++) yields
// bit-identical categorical draws for an identical logit row and seed.  So the
// full sampler — including the final draw — is reproducible across languages.

#include <mlx/mlx.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace mx = mlx::core;

namespace veystrix::engine {

// One decode's sampling settings. Mirrors veystrix.sampling.Sampling.
struct SamplingParams {
  float temperature = 1.0f;
  float top_p = 0.95f;
  int top_k = 20;
  float min_p = 0.0f;
  float presence_penalty = 0.0f;
};

// Result of sampling_distribution(), mirroring the (ids, masked) tuple
// returned by Sampler.distribution. When has_ids is false (the full-vocabulary
// path) `ids` is empty and `masked` spans the vocabulary; otherwise `ids`
// maps positions of `masked` back to token ids (the top-k survivor set).
struct Distribution {
  mx::array ids;
  mx::array masked;
  bool has_ids = false;
};

// Mirrors Sampler.distribution: presence penalty, temperature scaling,
// top-k argpartition/floor, then min-p and top-p masking.
// `seen` is the sorted set of already-produced token ids (presence penalty).
Distribution sampling_distribution(const mx::array& logits,
                                   const SamplingParams& params,
                                   const std::vector<int32_t>& seen = {});

// Mirrors Sampler._mask: apply min-p then top-p to a 1-D logit row, returning
// the row with excluded entries set to -inf.
mx::array sampling_mask(const mx::array& logits, float min_p, float top_p);

// Non-greedy branch of Sampler.__call__: distribution -> categorical draw ->
// top-k remap -> shape (1,) uint32 token. If `key` is provided it is used for
// the categorical draw (fully reproducible); otherwise the process-wide
// default key seeded by random::seed() is used, exactly like Python's
// mx.random.categorical without an explicit key.
mx::array sample_non_greedy(const mx::array& logits,
                            const SamplingParams& params,
                            const std::vector<int32_t>& seen = {},
                            const std::optional<mx::array>& key = std::nullopt);

}  // namespace veystrix::engine
