// Phase 2 — resumable per-layer decode state machine (offline/synthetic only).
//
// This is a PLAIN STRUCT state machine, deliberately NOT a C++20 coroutine, so
// that a single decode step can be suspended at a layer boundary and resumed by
// a later scheduler without any hidden control flow. It owns no mutex, no
// condition variable and no MLX object: it is a bookkeeping shell that a later
// phase will drive session-by-session. No production caller is changed here —
// `run_step` is NOT rewired in this phase.
//
// How run_step's layer loop maps onto this cursor (documentation, not code):
//
//   run_step (real_checkpoint_generate.cpp:1178)
//     for (l = 0; l < cfg.n_layers; ++l) {
//       PLE + gated_residual + attention          <- PreRoute work up to router
//       gate_logits = linear(m2d, L.gate_w);
//       routes = router(gate_logits, ...);        <- route sync (mx::eval)
//       StreamForwardResult sf = stream->forward(...)  <- plan + submit reads
//       ... build MoE graph, mx::eval, inject     <- PostMoE work
//     }
//
//   maps to, per session:
//     cursor.begin(l, hidden)                 // phase = PreRoute
//       ...PLE/attn/gated_residual/router...
//       tickets = pool.plan_and_submit(route)  // pin from plan time
//       cursor.await_reads(tickets)            // phase = AwaitReads, pending = tickets
//     // ---- suspend: driver may run another session here ----
//     while (!cursor.ready()) {
//       t = driver.next_completed_ticket();    // acquire-loaded from pager
//       cursor.on_ticket_done(t);              // removes from wait-set
//     }
//     cursor.enter_post_moe();                 // phase = PostMoE (all bytes materialized)
//       ...build MoE graph + mx::eval + inject...
//       next_hidden = ...
//       pool.release_fence(session, l)         // after this session's eval returns
//     cursor.finish(next_hidden);              // hidden = next, l++, phase = PreRoute
//
// Bit-exactness contract (C1 invariants): the op order / shapes / dtypes /
// gather order inside one layer are unchanged; only the *interleaving* of
// independent evaluations moves. This cursor never touches numeric data —
// `Hidden` is carried opaquely and is only moved between phases.
//
// Labels: the three-phase split and the ticket wait-set are [idea] design
// choices from the Opus brief §5/§C1/C2. No measurement, no MLX, no checkpoint.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace veystrix::engine {

// Ticket id used by the async pager callback (`void(int ticket)`). Kept as a
// plain int to match `submit_batched_reads_async`'s public signature.
using ReadTicket = int;
inline constexpr ReadTicket kNoTicket = -1;

// The three resumable phases of one layer of one session.
enum class LayerPhase : int {
  PreRoute = 0,   // route not yet computed / reads not yet submitted
  AwaitReads = 1, // reads submitted; pending wait-set non-empty (may be parked)
  PostMoE = 2,    // all tickets complete; graph build + eval + inject still to do
};

// Stable, allocation-free name for diagnostics/tests.
const char* phase_name(LayerPhase p) noexcept;

// Per-layer decode cursor. `Hidden` is a template parameter so the exact same
// state machine can be exercised with a fake array in CPU-only tests and later
// instantiated as `LayerCursor<mx::array>` in the engine without dragging MLX
// into this header.
template <class Hidden>
struct LayerCursor {
  int layer = -1;
  Hidden hidden{};
  std::vector<ReadTicket> pending{};   // the (session, layer) wait-set
  LayerPhase phase = LayerPhase::PreRoute;

  // Start/reset a layer with its incoming hidden state.
  void begin(int new_layer, Hidden h) {
    layer = new_layer;
    hidden = std::move(h);
    pending.clear();
    phase = LayerPhase::PreRoute;
  }

  // PreRoute -> AwaitReads. Install the tickets produced by route+plan+submit.
  // Tickets are deduplicated and sorted so the wait-set is a canonical set.
  void await_reads(std::vector<ReadTicket> tickets) {
    phase = LayerPhase::AwaitReads;
    pending = std::move(tickets);
    std::sort(pending.begin(), pending.end());
    pending.erase(std::unique(pending.begin(), pending.end()), pending.end());
  }

  // Ready iff the wait-set is empty (C2: "Ready when the set is empty").
  bool ready() const noexcept { return pending.empty(); }

  // Record that one in-flight ticket completed. Idempotent: returns true only
  // the first time a given ticket is removed, which is the debug-build guard
  // for "release_fence runs exactly once" (C2).
  bool on_ticket_done(ReadTicket t) {
    auto it = std::find(pending.begin(), pending.end(), t);
    if (it == pending.end()) return false;
    pending.erase(it);
    return true;
  }

  // AwaitReads -> PostMoE. Legal only when all tickets have completed, which
  // is the structural form of invariant 2 (no graph over still-writing bytes).
  void enter_post_moe() { phase = LayerPhase::PostMoE; }

  // PostMoE -> PreRoute of the next layer. Install the produced hidden state,
  // drop the wait-set and advance the layer index.
  void finish(Hidden next) {
    hidden = std::move(next);
    ++layer;
    pending.clear();
    phase = LayerPhase::PreRoute;
  }
};

}  // namespace veystrix::engine
