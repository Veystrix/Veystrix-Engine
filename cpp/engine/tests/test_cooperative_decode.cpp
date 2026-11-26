// CPU-only bit-exactness test for cooperative decode: interleaving N sessions
// on one thread must produce EXACTLY the same per-session token stream as
// running each session alone (the hard multi-session / bit-identity rule).
#include "cooperative_decode.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace veystrix::engine;

// A tiny deterministic session: token[step] = mix(seed, step), finished after
// `length` steps. Pure per-session state; no cross-session interaction.
struct TinySession {
  unsigned long long seed;
  int length;
  int produced = 0;
  std::vector<int64_t> tokens;
};

static bool tiny_step(void* p, int step, unsigned long long seed) {
  auto* s = static_cast<TinySession*>(p);
  // Deterministic function of ONLY this session's seed and step index.
  unsigned long long x = seed ^ (0x9e3779b97f4a7c15ULL * (unsigned long long)(step + 1));
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 27;
  s->tokens.push_back(static_cast<int64_t>(x & 0x3ff));
  ++s->produced;
  return s->produced >= s->length;
}

static bool equal_streams(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  return a == b;
}

int main() {
  const int N = 4;
  const int LEN = 6;
  const unsigned long long seeds[N] = {11ULL, 22ULL, 33ULL, 44ULL};

  // Solo reference: run each session to completion in isolation.
  std::vector<std::vector<int64_t>> solo(N);
  for (int i = 0; i < N; ++i) {
    TinySession s{seeds[i], LEN};
    CoopSession c{&s, tiny_step, seeds[i]};
    std::vector<CoopSession> one{c};
    cooperative_run(one);
    solo[i] = s.tokens;
  }

  // Cooperative: run all N interleaved on one thread.
  std::vector<TinySession> states;
  std::vector<CoopSession> coop;
  states.reserve(N);  // pointers into `states` must stay valid
  coop.reserve(N);
  for (int i = 0; i < N; ++i) {
    states.push_back(TinySession{seeds[i], LEN});
    coop.push_back(CoopSession{&states[i], tiny_step, seeds[i]});
  }
  int executed = cooperative_run(coop);
  if (executed != N * LEN) { std::fprintf(stderr, "FAIL: step count %d != %d\n", executed, N*LEN); return 1; }

  for (int i = 0; i < N; ++i) {
    if (!equal_streams(solo[i], states[i].tokens)) {
      std::fprintf(stderr, "FAIL: session %d stream differs from solo\n", i);
      return 1;
    }
    if (coop[i].steps_done != LEN) {
      std::fprintf(stderr, "FAIL: session %d step count\n", i);
      return 1;
    }
  }
  std::printf("test_cooperative_decode OK (%d sessions interleaved == solo, %d steps)\n", N, executed);
  return 0;
}
