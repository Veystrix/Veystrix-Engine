// GPU keep-warm while the driver waits for SSD expert reads (port of the Python
// engine's FLASHNEXT_GPU_KEEPWARM, models/flashnext/expert_cache.py).
//
// Decode leaves the GPU idle while a layer's routed experts are read from the
// SSD; the GPU governor then drops to low performance states and the next
// command buffers run slower. While waiting, the driver keeps short ALU-only
// spin kernels queued on a SEPARATE GPU stream (one threadgroup of 32 lanes,
// no memory reads, its output is discarded), so the GPU never looks idle.
// No model value is touched: exact by construction.
//
// VEYSTRIX_CPP_KEEPWARM=1 enables it (=0 disables), _ITERS sets the spin
// length (default 60000 fma), _PERIOD_US the re-queue period (default 500 us).
#pragma once

#include <vector>

#include "io_pager.hpp"
#include "io_pager_async.hpp"

namespace veystrix::engine::keepwarm {

// Whether keep-warm is on (read once from the environment).
bool enabled();

// Queue one spin kernel on the keep-warm stream (driver thread only).
void spin_once();

// Block until `ticket` completes, queuing one spin per period while waiting.
// Returns true when every read of the ticket succeeded. Releases the ticket.
bool wait_ticket_warm(veystrix::AsyncTicket ticket);

}  // namespace veystrix::engine::keepwarm

namespace veystrix::engine {
// Run a decode layer's expert reads: synchronous 8-lane batch by default; with
// VEYSTRIX_CPP_KEEPWARM=1 the same batch runs on the async pager while the
// driver keeps the GPU clock up. Returns the number of failed reads (0 = ok).
int run_expert_reads(const std::vector<veystrix::PagerRead>& reads, int width);
}  // namespace veystrix::engine
