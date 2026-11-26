#include "gpu_gate.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>

#include <Metal/Metal.hpp>
#include <mlx/backend/metal/device.h>
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace veystrix::engine::gpu_gate {
namespace {

struct Gate {
  MTL::SharedEvent* event = nullptr;
  std::atomic<uint64_t> next{0};
};

Gate& gate() {
  static Gate* g = [] {
    Gate* x = new Gate();  // process lifetime
    x->event = mx::metal::device(mx::Device::gpu).mtl_device()->newSharedEvent();
    return x;
  }();
  return *g;
}

}  // namespace

bool enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VEYSTRIX_GENERIC_GATE");
    return e && std::strcmp(e, "1") == 0;
  }();
  return on;
}

uint64_t arm() {
  Gate& g = gate();
  if (!g.event) return 0;
  mx::metal::CommandEncoder& enc = mx::metal::get_command_encoder(mx::default_stream(mx::Device::gpu));
  MTL::CommandBuffer* cb = enc.get_command_buffer();
  if (!cb) return 0;
  enc.end_encoding();  // a command-buffer wait goes between compute encoders
  const uint64_t v = g.next.fetch_add(1, std::memory_order_relaxed) + 1;
  cb->encodeWait(g.event, v);
  return v;
}

void signal(uint64_t value) {
  gate().event->setSignaledValue(value);
}

int tail() {
  static const int n = [] {
    const char* e = std::getenv("VEYSTRIX_CPP_GATE_TAIL");
    const int v = e ? std::atoi(e) : 0;
    return v > 0 ? v : 0;
  }();
  return n;
}

namespace {
std::atomic<bool> g_failed{false};
}
void mark_failed() { g_failed.store(true, std::memory_order_release); }
bool failed() { return g_failed.load(std::memory_order_acquire); }

}  // namespace veystrix::engine::gpu_gate
