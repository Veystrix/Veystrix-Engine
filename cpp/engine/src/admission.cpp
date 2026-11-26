#include "admission.hpp"

#include <algorithm>

namespace veystrix::engine {

double page_cache_floor_from_anchor(const std::string& model,
                                    int target_aggregate_tokens) {
  // [estimate] both anchors are simulator projections, never measurements.
  double per_token = 0.0;
  if (model == "nemotron" || model == "nemotron35" || model == "nemotron_h") {
    per_token = kNemotronFloorMbPerAggToken;
  } else if (model == "qwen" || model == "qwen4_exp" || model == "flashnext") {
    per_token = kQwenFloorMbPerAggToken;
  }
  return per_token * static_cast<double>(std::max(0, target_aggregate_tokens));
}

AdmissionDecision evaluate_admission(const AdmissionArgs& args,
                                     int current_sessions,
                                     const NumericsProfile& frozen_profile) {
  AdmissionDecision d;
  d.resulting_sessions = current_sessions + 1;
  d.effective_expert_pool_mb =
      std::max(0.0, args.expert_pool_mb - args.pinned_pool_mb);

  // Contract §5: total = model resident + effective expert pool
  //                + page-cache floor + N x per-session state.
  // "Effective" pool capacity excludes pinned slots: already-queued GPU work
  // makes them un-evictable, so they are unavailable to new admissions [idea].
  const double used = args.model_resident_mb + d.effective_expert_pool_mb +
                      static_cast<double>(d.resulting_sessions) * args.session_state_mb;
  d.projected_page_cache_mb = args.capacity_mb - used;
  d.projected_available_mb = d.projected_page_cache_mb;

  if (!frozen_profile.is_frozen()) {
    d.reason = "numerics profile not frozen";
    return d;  // §3 rule 3: refuse, never mutate
  }
  if (d.projected_page_cache_mb < args.page_cache_floor_mb) {
    d.reason = "page cache floor";
    return d;
  }
  if (d.projected_available_mb < args.min_available_mb) {
    d.reason = "3 GiB available-memory rule";
    return d;
  }

  d.admitted = true;
  d.reason = "ok";
  return d;
}

}  // namespace veystrix::engine
