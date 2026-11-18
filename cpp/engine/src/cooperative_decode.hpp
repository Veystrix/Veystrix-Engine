// Cooperative single-thread decode scheduler (foundation of continuous
// batching and the multi-session runtime). NON-NUMERIC: it never batches rows
// from different sessions into one matmul; it only interleaves independent
// single-row decode steps on ONE GPU-driving thread. Each session's output is
// therefore bit-identical to running it alone.
//
// The SessionState is a state machine that yields between decode steps. There
// are NO new threads, locks or worker-count changes here (those would need
// owner approval); this is pure sequential interleaving of opaque callbacks.
#pragma once

#include <cstddef>
#include <functional>
#include <vector>

namespace veystrix::engine {

// One cooperative decode step: given a session's opaque state, advance it by
// one token and return whether it is finished. `step` is the decode step index
// for that session (0-based), `seed` a per-session value (e.g. PRNG key).
using DecodeStepFn = std::function<bool(/*state=*/void*, /*step=*/int,
                                        /*session_seed=*/unsigned long long)>;

struct CoopSession {
  void* state = nullptr;
  DecodeStepFn step_fn;
  unsigned long long seed = 0;
  int steps_done = 0;
  bool done = false;
};

// Run all sessions cooperatively, round-robin, until every session is done.
// `max_steps` bounds a single session to avoid infinite loops (0 = unbounded).
// Returns the number of decode steps actually executed.
inline int cooperative_run(std::vector<CoopSession>& sessions, int max_steps = 0) {
  int executed = 0;
  bool any = true;
  while (any) {
    any = false;
    for (auto& s : sessions) {
      if (s.done) continue;
      if (max_steps > 0 && s.steps_done >= max_steps) { s.done = true; continue; }
      s.done = s.step_fn(s.state, s.steps_done, s.seed);
      ++s.steps_done;
      ++executed;
      any = true;
    }
  }
  return executed;
}

}  // namespace veystrix::engine
