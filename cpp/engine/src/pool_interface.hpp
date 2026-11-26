// Phase 2 — the single shared pool contract for BOTH model families.
//
// C3 requires one pool abstraction exposing the same seven-stage lifecycle so
// that a scheduler can drive Qwen's `StreamingMoE` and Nemotron's
// `SharedExpertPool` interchangeably. This header DEFINES the contract only.
// Neither implementation is modified in this phase (the brief forbids it), and
// nothing in the engine is rewired to call it yet.
//
//   plan -> pin -> submit -> wait -> build -> eval -> release
//
// Ownership / bit-exactness rules that an implementation MUST honour:
//   * pin happens atomically with the residency decision, under the pool's
//     existing mutex, from plan time (invariant 3; closes F7's window);
//   * eviction never selects a pinned or in-flight slot; exhaustion parks the
//     session (backpressure), it does not throw (invariant 4);
//   * build is legal only after wait has observed every ticket with acquire
//     ordering; no lazy Load nodes (invariant 2);
//   * release runs exactly once, after this session's eval returns, and frees
//     only this (session, layer)'s pins (C2 idempotence guard).
//
// Mapping to today's code (documentation, NOT done here):
//
//   stage      StreamingMoE (Qwen, slot_plan LRU)      SharedExpertPool (Nemotron)
//   ---------  --------------------------------------  -------------------------------------------
//   plan       plan_remap / slot_plan LRU              fetch_union dedup (first-seen order)
//   pin        NEW: pin slots under the slot mutex     acquire_and_pin under pool_.mutex()
//   submit     submit_batched_reads_async              pool_.acquire (shared-future coalescing)
//   wait       wait_ticket(t) (mutex+CV)               shared_future::get() (acquire edge)
//   build      StreamingMoE::forward MoE graph         compact-unique/first-occurrence stack from slots
//   eval       mx::eval + inject on driver thread      mx::eval on driver thread
//   release    NEW: per (session,layer) fence          release_fence(session, layer)
//
// The two "NEW" cells are the work that full Phase 2 integration (a later
// phase) must add to StreamingMoE; the Nemotron side already has pin/fence and
// only needs the async submit adapter (wrap its shared_future as a ticket).
//
// Labels: [idea] design contract from Opus brief §C1/C2/C3 and §5 Phase 2.
// No measurement, no MLX, no checkpoint, no new thread/mutex in this header.
#pragma once

#include <cstdint>
#include <vector>

namespace veystrix::engine {

// Ticket id; matches `ReadTicket` / pager `AsyncTicket` (int).
using PoolTicket = int;

// What a session wants at one layer: its session id, layer index and the
// router's expert ids in route order (repeats allowed — plan dedups misses).
struct PoolRequest {
  int session = -1;
  int layer = -1;
  std::vector<int> experts{};
};

// One physical read produced by plan. Kept self-contained (no io_pager include)
// so this header stays dependency-free and CPU-testable.
struct PoolRead {
  int fd = -1;
  std::int64_t offset = 0;
  std::int64_t length = 0;
  std::uint64_t dst = 0;    // uintptr of destination slot
  int expert = -1;
  int slot = -1;
};

// Result of plan: the slot assigned to each routed expert (route order), the
// deduplicated misses to submit, and a `parked` flag for pool exhaustion.
struct PoolPlan {
  std::vector<int> slots{};
  std::vector<PoolRead> reads{};
  bool parked = false;      // backpressure: retry later, never throw
};

// Identity of a fence. `release` is keyed by this so pins from other sessions
// on the same slot are untouched.
struct PoolFence {
  int session = -1;
  int layer = -1;
  bool valid = false;
};

// The common lifecycle both pools will expose. Templated on the array types so
// a CPU test can instantiate it with fake arrays and the engine can later use
// `mx::array` without this header depending on MLX.
template <class HiddenArray, class ExpertArray>
class IPoolContract {
 public:
  virtual ~IPoolContract() = default;

  // 1. plan — choose victim-free slots, dedup misses; no I/O, no mutation.
  virtual PoolPlan plan(const PoolRequest& req) = 0;

  // 2. pin — under the pool mutex, establish residency and pin every slot in
  //    the plan. Returns false and sets plan.parked on exhaustion.
  virtual bool pin(const PoolRequest& req, const PoolPlan& plan) = 0;

  // 3. submit — issue plan.reads asynchronously; return one ticket covering them.
  virtual PoolTicket submit(const PoolRequest& req, const PoolPlan& plan) = 0;

  // 4. wait — acquire-complete the ticket. No graph build before this returns.
  virtual bool wait(PoolTicket ticket) = 0;

  // 5. build — AFTER wait, build the MoE graph from materialized bytes.
  virtual ExpertArray build(const PoolRequest& req, const PoolPlan& plan,
                            const HiddenArray& hidden) = 0;

  // 6. eval — materialize the MoE output on the single driver thread; returns
  //    the next hidden state (residual/inject applied by the caller).
  virtual HiddenArray eval(const PoolRequest& req, const ExpertArray& moe,
                           const HiddenArray& hidden) = 0;

  // 7. release — run this (session, layer)'s fence exactly once, after eval.
  //    Returns the number of pins released. Idempotent in debug builds.
  virtual int release(const PoolRequest& req) = 0;
};

}  // namespace veystrix::engine
