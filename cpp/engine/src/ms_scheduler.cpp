#include "ms_scheduler.hpp"

#include <algorithm>
#include <utility>

namespace veystrix::engine {

bool MsScheduler::one_step_fn(void* p, int /*step*/, unsigned long long /*seed*/) {
  auto* proxy = static_cast<StepProxy*>(p);
  auto* slot = static_cast<Slot*>(proxy->slot);

  // The real step uses this session's own step index and seed, exactly as in
  // cooperative_run. No cross-session state is touched by this callback.
  const unsigned long long seed = slot->coop.seed;
  const int step_index = slot->coop.steps_done;
  bool done = slot->scheduled.step_fn(slot->scheduled.state, step_index, seed);
  ++slot->coop.steps_done;
  slot->coop.done = done;

  // Consume an explicit yield request recorded by this session's step. Only
  // this session's SessionState is inspected, and only after its step returns.
  if (slot->scheduled.session) {
    int layer = -1;
    if (slot->scheduled.session->consume_yield(&layer)) {
      slot->phase = SessionPhase::WaitingForExpertReads;
      slot->wait_layer = layer;
    }
  }

  // v1: a session in prefill stays in the Prefill phase until its exclusive
  // prefill work is consumed; no decode of another session interleaves it.
  if (slot->scheduled.prefill_steps > 0) {
    --slot->scheduled.prefill_steps;
    if (slot->scheduled.prefill_steps == 0 && slot->phase == SessionPhase::Prefill) {
      slot->phase = SessionPhase::Ready;
    }
  }
  if (done) slot->phase = SessionPhase::Done;

  // One step per cooperative_run() pass (the proxy is now "done").
  return true;
}

int MsScheduler::add(const ScheduledSession& s) {
  Slot slot;
  slot.scheduled = s;
  slot.coop.state = s.state;
  slot.coop.step_fn = s.step_fn;
  slot.coop.seed = s.seed;
  slot.phase = (s.prefill_steps > 0) ? SessionPhase::Prefill : SessionPhase::Ready;
  slots_.push_back(std::move(slot));
  return static_cast<int>(slots_.size()) - 1;
}

void MsScheduler::run_one_step(Slot& slot) {
  StepProxy proxy{&slot};
  CoopSession one{&proxy, &MsScheduler::one_step_fn, 0};
  std::vector<CoopSession> round{one};
  cooperative_run(round);  // single-thread, single-row, one step
}

int MsScheduler::drive(int max_rounds, const ReadsReadyFn& reads_ready) {
  int executed = 0;
  int local_rounds = 0;
  const bool unbounded = (max_rounds <= 0);

  while (unbounded || local_rounds < max_rounds) {
    ++local_rounds;
    ++rounds_;
    // 1. Wake yielded sessions whose expert reads are satisfied. The callback
    //    is supplied by the caller; no I/O thread is created or counted here.
    for (auto& slot : slots_) {
      if (slot.phase == SessionPhase::WaitingForExpertReads && reads_ready) {
        if (reads_ready(slot.scheduled.id, slot.wait_layer)) {
          slot.phase = SessionPhase::Ready;
          slot.wait_layer = -1;
        }
      }
    }

    // 2. Select runnable sessions. Waiting and done sessions are skipped;
    //    that is what makes the driver work-conserving (§1.3).
    const bool prefill_active =
        cfg_.prefill_exclusive &&
        std::any_of(slots_.begin(), slots_.end(), [](const Slot& s) {
          return s.phase == SessionPhase::Prefill;
        });

    std::vector<int> runnable;
    if (prefill_active) {
      // No prefill overlaps another prefill, and no decode interleaves inside
      // one: only the FIRST prefill session advances this round [idea].
      for (int i = 0; i < static_cast<int>(slots_.size()); ++i) {
        if (slots_[i].phase == SessionPhase::Prefill) {
          runnable.push_back(i);
          break;
        }
      }
    } else {
      for (int i = 0; i < static_cast<int>(slots_.size()); ++i) {
        if (slots_[i].phase == SessionPhase::Ready) runnable.push_back(i);
      }
    }
    if (runnable.empty()) break;  // all done or all waiting: do not spin

    // 3. Bounded skew [idea]: skip a session that leads by more than the
    //    configured bound while another runnable session is behind. Never
    //    skip if that would leave the round empty (stay work-conserving).
    int max_steps = 0;
    for (const auto& slot : slots_) {
      max_steps = std::max(max_steps, slot.coop.steps_done);
    }
    std::vector<int> bounded;
    for (int i : runnable) {
      if (max_steps - slots_[i].coop.steps_done <= cfg_.max_skew_steps) {
        bounded.push_back(i);
      }
    }
    if (!bounded.empty()) runnable = std::move(bounded);

    // 4. Advance each runnable session once, via cooperative_run.
    for (int i : runnable) {
      run_one_step(slots_[i]);
      ++executed;
    }
  }
  return executed;
}

SessionPhase MsScheduler::phase(int slot) const {
  return slots_.at(static_cast<std::size_t>(slot)).phase;
}

int MsScheduler::steps_done(int slot) const {
  return slots_.at(static_cast<std::size_t>(slot)).coop.steps_done;
}

int MsScheduler::wait_layer(int slot) const {
  return slots_.at(static_cast<std::size_t>(slot)).wait_layer;
}

}  // namespace veystrix::engine
