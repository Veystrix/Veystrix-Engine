// Shared by the live C-ABI runtimes (NemotronEngineRuntime, GenericEngineRuntime).
#pragma once

#include <stdexcept>

namespace veystrix::engine::capi_live {

// Thrown at a step boundary when vx_cancel() was called. The C ABI maps this
// exception to VX_ERR_CANCELLED; it never crosses the C boundary.
struct CancelledError : std::runtime_error {
  CancelledError() : std::runtime_error("vx_session_step: cancelled") {}
};

}  // namespace veystrix::engine::capi_live
