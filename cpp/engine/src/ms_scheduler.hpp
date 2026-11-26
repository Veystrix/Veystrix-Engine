// Multi-session Phase 1 — one GPU-driving-thread scheduler (contract §1.3).
//
// Sessions are state machines that yield at "waiting for expert reads at layer
// L". The driver advances runnable sessions, skips waiting ones, and is
// work-conserving with bounded skew (soft lockstep, §2). Prefill is exclusive
// in v1 (§3 rule 5).
//
// BUILT ON cooperative_run: each scheduled step is executed by a one-shot
// proxy passed to cooperative_run(), so the round-robin/step semantics stay
// exactly those of the existing single-thread scheduler. Rows are never
// batched across sessions — each step is an independent single-row callback.
//
// THREADING: there is NO new thread, lock, worker count, or I/O worker here.
// drive() runs synchronously on the CALLING thread (the one GPU-driving
// thread). NEEDS OWNER APPROVAL: any future dedicated completion/fence thread,
// additional mutex, or I/O worker-count change — do NOT add one without it.
//
// Labels: soft lockstep, bounded skew and prefill exclusivity are [idea]
// policy choices from the contract; bit-identity is a hard contract rule.
#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "cooperative_decode.hpp"
#include "session_state.hpp"

namespace veystrix::engine {

enum class SessionPhase {
  Ready,
  Prefill,                // exclusive in v1 (§3 rule 5)
  WaitingForExpertReads,  // explicit yield at wait_layer (§1.3)
  Done,
};

// One session handed to the scheduler. `state` is opaque and is passed to
// step_fn exactly as with CoopSession. `session` is optional but, when set, is
// used to consume the explicit yield request after each step.
struct ScheduledSession {
  int id = 0;                       // reported to the reads-ready callback
  void* state = nullptr;
  SessionState* session = nullptr;
  DecodeStepFn step_fn;
  unsigned long long seed = 0;
  int prefill_steps = 0;            // exclusive prefill work left (v1)
};

// Phase F step 6: a per-session progress probe (absolute layer index) used for
// layer-level soft lockstep and furthest-behind-first ordering. The scheduler
// only *reads* it; the state itself stays exclusively owned by the session.
using LayerProgressFn = std::function<int(/*state=*/void*)>;

struct SchedulerConfig {
  // Bounded skew [idea]: a session may not lead the round by more than this
  // many decode steps while another runnable session is behind it.
  int max_skew_steps = 1;
  bool prefill_exclusive = true;    // v1 rule [idea]

  // Phase F step 6 (N>1 layer-cursor driver). When max_skew_layers > 0 the
  // skew bound and the dispatch ordering use `layer_progress` (absolute layer
  // index) instead of completed whole steps. max_skew_layers is swept over
  // {2,4,8}. furthest_behind_first orders runnable sessions least-advanced
  // first; it is a pure ordering policy, never a numeric change.
  int max_skew_layers = 0;
  LayerProgressFn layer_progress;
  bool furthest_behind_first = false;
};

// Callback reporting that the expert reads (and pins) for (session, layer) are
// satisfied. Supplied by the caller; no I/O worker is created here.
using ReadsReadyFn = std::function<bool(/*session_id=*/int, /*layer=*/int)>;

class MsScheduler {
 public:
  explicit MsScheduler(SchedulerConfig cfg = {}) : cfg_(cfg) {}

  // Register a session. Do not call while drive() is running (slots_ must not
  // be reallocated mid-round). Returns the slot index.
  int add(const ScheduledSession& s);

  // Drive rounds on the calling thread. Returns steps executed. `max_rounds`
  // bounds the call (0 = until done or no runnable session). `reads_ready`
  // wakes yielded sessions whose reads have completed.
  int drive(int max_rounds = 0, const ReadsReadyFn& reads_ready = {});

  int slot_count() const noexcept { return static_cast<int>(slots_.size()); }
  int rounds() const noexcept { return rounds_; }
  SessionPhase phase(int slot) const;
  int steps_done(int slot) const;
  int wait_layer(int slot) const;

 private:
  struct StepProxy {
    void* slot = nullptr;  // Slot*, kept opaque to the proxy callback
  };

  struct Slot {
    ScheduledSession scheduled;
    CoopSession coop;
    SessionPhase phase = SessionPhase::Ready;
    int wait_layer = -1;
  };

  // Executes exactly one step of `slot` via cooperative_run (see .cpp).
  void run_one_step(Slot& slot);
  // One-shot callback: delegates one real decode step, then records the
  // yield/done/prefill transitions. Returns true so cooperative_run's loop
  // terminates after this single pass.
  static bool one_step_fn(void* proxy, int step, unsigned long long seed);

  std::vector<Slot> slots_;
  SchedulerConfig cfg_;
  int rounds_ = 0;
};

}  // namespace veystrix::engine
