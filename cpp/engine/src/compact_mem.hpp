// Compact expert-stack buffers for the Qwen MoE decode paths.
#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

#include <mlx/mlx.h>

namespace veystrix::engine {
namespace mx = mlx::core;

// VEYSTRIX_CPP_MLX_DEST=1: compact expert stacks live in buffers from MLX's
// own allocator (recycled Metal buffers that are already GPU-resident) rather
// than fresh posix_memalign memory MLX must wrap as a new no-copy Metal buffer
// for every tensor of every layer. Same bytes, same kernels (exact).
inline bool mlx_dest_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VEYSTRIX_CPP_MLX_DEST");
    return e && std::strcmp(e, "1") == 0;
  }();
  return on;
}

struct CompactMem {
  std::shared_ptr<uint8_t> mem;
  mx::allocator::Buffer mbuf{nullptr};
};

inline CompactMem alloc_compact(size_t bytes, size_t align) {
  CompactMem c;
  if (mlx_dest_enabled()) {
    mx::allocator::Buffer b = mx::allocator::malloc(bytes ? bytes : 1);
    c.mbuf = b;
    c.mem = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(b.raw_ptr()),
                                     [b](uint8_t*) { mx::allocator::free(b); });
    return c;
  }
  void* raw = nullptr;
  if (::posix_memalign(&raw, align, bytes ? bytes : 1) != 0)
    throw std::runtime_error("compact: posix_memalign failed");
  c.mem = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(raw),
                                   [](uint8_t* p) { std::free(p); });
  return c;
}

inline mx::array wrap_compact(const CompactMem& c, mx::Shape shape, mx::Dtype dt) {
  auto keep = c.mem;
  if (c.mbuf.ptr())
    return mx::array(c.mbuf, std::move(shape), dt, [keep](mx::allocator::Buffer) {});
  return mx::array(c.mem.get(), std::move(shape), dt, [keep](void*) {});
}

}  // namespace veystrix::engine
