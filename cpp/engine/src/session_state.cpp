#include "session_state.hpp"

#include <utility>

namespace veystrix::engine {

SessionState::SessionState(int session_id, unsigned long long prng_key,
                           NumericsProfile profile, PrefixCacheHandle prefix)
    : id_(session_id),
      key_(prng_key),
      state_(prng_key),
      profile_(std::move(profile)),
      prefix_(std::move(prefix)) {}

// splitmix64: a deterministic, independent per-session stream [idea]. The
// state advance is a pure function of THIS session's previous state; no global
// RNG, no cross-session seeding, and no other session ever calls this object's
// draw_u64(). That is what keeps interleaved sampling bit-identical to solo.
unsigned long long SessionState::draw_u64() noexcept {
  unsigned long long z = (state_ += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

bool SessionState::consume_yield(int* layer_out) noexcept {
  if (!yield_requested_) return false;
  if (layer_out) *layer_out = yield_layer_;
  yield_requested_ = false;
  return true;
}

}  // namespace veystrix::engine
