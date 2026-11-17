#include "sampling.hpp"

#include <limits>

namespace veystrix::engine {
namespace {

// -inf as a float32 scalar array, matching Python's -mx.inf under MLX's weak
// scalar promotion against a float32 logit row.
mx::array neg_inf32() {
  return mx::array(-std::numeric_limits<float>::infinity(), mx::float32);
}

}  // namespace

mx::array sampling_mask(const mx::array& logits, float min_p, float top_p) {
  // veystrix.sampling.Sampler._mask
  mx::array small = logits;
  mx::array probabilities = mx::softmax(small, -1);

  if (min_p > 0.0f) {
    // floor = min_p * max(probabilities); small = where(p < floor, -inf, small)
    mx::array floor = mx::array(min_p, mx::float32) * mx::max(probabilities);
    small = mx::where(probabilities < floor, neg_inf32(), small);
    probabilities = mx::softmax(small, -1);
  }

  if (0.0f < top_p && top_p < 1.0f) {
    // order = argsort(-p); ordered = p[order]; carried = cumsum(ordered)
    // keep = carried - ordered < top_p  (first token crossing the nucleus)
    // allowed = zeros(bool); allowed.at[order].add(keep)
    // small = where(allowed, small, -inf)
    mx::array order = mx::argsort(-probabilities, -1);
    mx::array ordered = mx::take(probabilities, order, -1);
    mx::array carried = mx::cumsum(ordered, -1);
    mx::array keep = (carried - ordered) < top_p;
    // Python does allowed = zeros(bool); allowed.at[order].add(keep), i.e.
    // allowed[order[i]] = keep[i]. `order` is a permutation, so its inverse
    // (argsort(order)) gives allowed = take(keep, inverse) bit-for-bit.
    mx::array allowed = mx::take(keep, mx::argsort(order, -1));
    small = mx::where(allowed, small, neg_inf32());
  }
  return small;
}

Distribution sampling_distribution(const mx::array& logits,
                                   const SamplingParams& s,
                                   const std::vector<int32_t>& seen) {
  // veystrix.sampling.Sampler.distribution
  mx::array row = mx::reshape(logits, {-1});

  // Presence penalty: row = row - hits, where hits[seen] = presence_penalty.
  // Python builds seen as a sorted int32 array and uses hits.at[seen].add(...).
  if (s.presence_penalty != 0.0f && !seen.empty()) {
    int n = static_cast<int>(row.shape(0));
    int k = static_cast<int>(seen.size());
    mx::array idx(seen.begin(), mx::Shape{k}, mx::int32);
    // Python: hits = zeros(n, dtype=row.dtype); hits.at[seen].add(penalty),
    // i.e. hits[seen[i]] = penalty. `seen` is a set (unique ids), so an
    // equality mask gives the same values bit-for-bit.
    mx::array positions = mx::reshape(mx::arange(n), {n, 1});
    mx::array eq = mx::equal(positions, mx::reshape(idx, {1, k}));
    mx::array is_seen = mx::any(eq, std::vector<int>{-1});
    mx::array pen(s.presence_penalty, row.dtype());
    mx::array zero(0.0f, row.dtype());
    mx::array hits = mx::where(is_seen, pen, zero);
    row = row - hits;
  }

  row = mx::astype(row, mx::float32) / s.temperature;

  if (s.top_k > 0 && s.top_k < static_cast<int>(row.shape(0))) {
    mx::array part = mx::argpartition(row, -s.top_k, -1);
    int n = static_cast<int>(part.shape(0));
    mx::array kept =
        mx::slice(part, mx::Shape{n - s.top_k}, mx::Shape{n}, mx::Shape{1});
    mx::array floor = mx::min(mx::take(row, kept, -1));
    row = mx::where(row < floor, neg_inf32(), row);
    // Keep the survivor indices sorted (the nucleus reduction assumes this).
    kept = mx::sort(kept, -1);
    return Distribution{kept,
                        sampling_mask(mx::take(row, kept, -1), s.min_p, s.top_p),
                        true};
  }
  return Distribution{mx::zeros({0}, mx::int32),
                      sampling_mask(row, s.min_p, s.top_p), false};
}

mx::array sample_non_greedy(const mx::array& logits,
                            const SamplingParams& s,
                            const std::vector<int32_t>& seen,
                            const std::optional<mx::array>& key) {
  // veystrix.sampling.Sampler.__call__ (non-greedy branch)
  Distribution dist = sampling_distribution(logits, s, seen);
  mx::array drawn = key ? mx::random::categorical(dist.masked, -1, *key)
                        : mx::random::categorical(dist.masked);
  mx::array token = dist.has_ids ? mx::take(dist.ids, drawn, -1) : drawn;
  // Shape (1,) not scalar: the decode loop feeds token[None] to the model.
  return mx::astype(mx::reshape(token, {1}), mx::uint32);
}

}  // namespace veystrix::engine
