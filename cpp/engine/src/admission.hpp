// Multi-session Phase 1 — admission and memory budget (contract §1.4, §5).
//
// Admission is the ONLY place a new session may be refused. It never changes an
// admitted session's numerics profile: it checks the prospective N+1 session
// against a budget and refuses if the page-cache floor or the standing 3 GiB
// available-memory rule would be violated. All inputs come from args; there is
// no real measurement in this module.
//
// Labels: the budget formula and refusal policy are [idea] (contract §5); the
// 3 GiB rule is [cited] (docs/performance/measurement-standard.md);
// the page-cache anchors are [estimate] from the Phase 0 simulator.
#pragma once

#include <string>

#include "session_state.hpp"

namespace veystrix::engine {

// All memory values are SI MB (1 MB = 1e6 B), matching the convention used by
// models/shared/tests/bench/multisession_sim.py (branch multi-session-phase0).
struct AdmissionArgs {
  double capacity_mb = 0.0;         // RAM available to the engine (from args)
  double model_resident_mb = 0.0;   // model weights resident (from args)
  double expert_pool_mb = 0.0;      // NOMINAL shared expert pool (from args)
  double pinned_pool_mb = 0.0;      // pinned, un-evictable part of that pool [idea]
  double page_cache_floor_mb = 0.0; // target floor (from args)
  double session_state_mb = 0.0;    // per-session state (from args)
  double min_available_mb = 3072.0; // standing 3 GiB rule [cited]
};

struct AdmissionDecision {
  bool admitted = false;
  int resulting_sessions = 0;      // N+1 when evaluated
  double projected_page_cache_mb = 0.0;
  double projected_available_mb = 0.0;
  double effective_expert_pool_mb = 0.0;
  std::string reason;              // "ok" or the refusal reason
};

// [estimate] Page-cache floor anchors from the Phase 0 simulator
// (models/shared/tests/bench/multisession_sim.py, branch multi-session-phase0;
// contract §2/§5): Nemotron ≈190–200 MB per AGGREGATE token at N=4,
// arrangement B, LRU, 6.6 GB pool; Qwen ≈0.9 GB per aggregate token at N=3,
// arrangement C, 349-slot pool. Both are simulator anchors, not measurements.
inline constexpr double kNemotronFloorMbPerAggToken = 195.0;  // ~190-200 MB [estimate]
inline constexpr double kQwenFloorMbPerAggToken = 900.0;      // ~0.9 GB [estimate]

// Convenience anchor: floor = per-aggregate-token anchor x target token count.
// Still [estimate]; callers should normally pass the floor explicitly in args.
double page_cache_floor_from_anchor(const std::string& model,
                                    int target_aggregate_tokens);

// Evaluate admission for one prospective session. `frozen_profile` must already
// be frozen (§3 rule 3); an unfrozen profile is refused rather than mutated.
AdmissionDecision evaluate_admission(const AdmissionArgs& args,
                                     int current_sessions,
                                     const NumericsProfile& frozen_profile);

}  // namespace veystrix::engine
