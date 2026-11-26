// Phase 2 — asynchronous batched expert reads (offline/synthetic only).
//
// This is the C1 async variant of `run_batched_reads`. Expert reads are issued
// on the EXISTING libdispatch user-interactive global queue at the SAME width
// (8). One `dispatch_group_async` block per lane pulls reads from a shared
// atomic counter; a single `dispatch_group_notify` block publishes completion
// and fires the caller's callback exactly once.
//
// SYNCHRONIZATION BUDGET (C1 hard rule — do not exceed):
//   - 1 engine driver thread (unchanged; this file never creates a thread)
//   - 1 std::mutex + 1 std::condition_variable  -> the completion ready-queue
//     below (a single process-wide pair, shared by every batch)
//   - atomic tickets (ticket allocator + per-batch error counter)
//   - libdispatch group/queue (the existing user-interactive queue; no new
//     worker count and no QoS change)
// There is deliberately NO second mutex, NO second CV and NO fence thread.
// If a design ever needs one, STOP and report per brief §4.
//
// The driver calls `wait_ticket` ONLY when no session is runnable, so the
// condition variable never blocks work that could proceed elsewhere.
//
// Labels: [idea] design from Opus brief §C1/§5 Phase 2. No measurement, no
// checkpoint, no MLX dependency here.
#pragma once

#include "io_pager.hpp"

#include <functional>

namespace veystrix {

// Ticket id matching the callback signature `void(int ticket)`.
using AsyncTicket = int;
inline constexpr AsyncTicket kInvalidTicket = -1;

// Submit `reads` asynchronously. Returns a ticket immediately. `on_done` is
// called exactly once, from the dispatch_group_notify block, after every
// read's destination bytes have been written and completion has been published
// with release ordering. `width` is clamped to [1, 8]; the QoS is always
// QOS_CLASS_USER_INTERACTIVE (C1: worker count / QoS are NOT changed here).
AsyncTicket submit_batched_reads_async(
    const std::vector<PagerRead>& reads, int width,
    std::function<void(int ticket)> on_done);

// Block until `ticket` is complete (driver-only; call when nothing runnable).
// Returns true if all reads succeeded, false if any read failed.
bool wait_ticket(AsyncTicket ticket);

// Wait at most `timeout_us` microseconds for `ticket` on the same C1 mutex/CV.
// Returns true once complete (then *ok holds whether every read succeeded).
bool wait_ticket_for(AsyncTicket ticket, long timeout_us, bool* ok);

// Drop a completed ticket's bookkeeping (completed tickets are otherwise kept
// for the life of the process). Call only after the ticket was waited on.
void release_ticket(AsyncTicket ticket);

// Non-blocking completion query (acquire load of the published flag).
bool ticket_complete(AsyncTicket ticket);

// Failed-read count for a ticket. Meaningful only once ticket_complete() is
// true; returns 0 for an unknown/in-flight ticket.
int ticket_errors(AsyncTicket ticket);

// Clear all completion state. Tests / engine shutdown only; callers must
// ensure no batch is still in flight. Returns the number of entries dropped.
int reset_async_pager_state();

}  // namespace veystrix
