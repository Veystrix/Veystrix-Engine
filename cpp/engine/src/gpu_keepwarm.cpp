#include "gpu_keepwarm.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <mlx/fast.h>
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace veystrix::engine::keepwarm {
namespace {

long env_long(const char* name, long def) {
  const char* e = std::getenv(name);
  if (!e || !*e) return def;
  const long v = std::atol(e);
  return v > 0 ? v : def;
}

struct State {
  mx::fast::CustomKernelFunction kernel;
  std::optional<mx::Stream> stream;
  std::optional<mx::array> iterations;
};

State& state() {
  // One per thread: an MLX (0.32) stream belongs to the thread that created it, and every engine runs on its
  // own driver thread. A process-wide static made a second engine in the same process fail with
  // "There is no Stream(gpu, N) in current thread" (the first engine's driver thread had made the stream).
  thread_local State s;
  if (!s.stream) {
    s.kernel = mx::fast::metal_kernel(
        "veystrix_gpu_keepwarm", {"iterations"}, {"out"},
        R"(
    uint lane = thread_position_in_grid.x;
    float value = float(lane);
    int loops = iterations[0];
    for (int i = 0; i < loops; ++i) {
        value = fma(value, 0.5f, 0.25f);
    }
    out[lane] = value;
)");
    s.stream = mx::new_stream(mx::Device::gpu);
    std::vector<int32_t> it{static_cast<int32_t>(env_long("VEYSTRIX_CPP_KEEPWARM_ITERS", 60000))};
    s.iterations = mx::array(it.begin(), mx::Shape{1}, mx::int32);
    mx::eval(*s.iterations);
  }
  return s;
}

}  // namespace

bool enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VEYSTRIX_CPP_KEEPWARM");
    return e && std::strcmp(e, "1") == 0;
  }();
  return on;
}

void spin_once() {
  State& s = state();
  std::vector<mx::array> out = s.kernel(
      {*s.iterations}, {mx::Shape{32}}, {mx::float32}, {32, 1, 1}, {32, 1, 1}, {},
      std::nullopt, false, *s.stream);
  mx::async_eval(out);
}

bool wait_ticket_warm(veystrix::AsyncTicket ticket) {
  static const long period_us = env_long("VEYSTRIX_CPP_KEEPWARM_PERIOD_US", 500);
  bool ok = true;
  while (!veystrix::wait_ticket_for(ticket, 0, &ok)) {
    spin_once();
    if (veystrix::wait_ticket_for(ticket, period_us, &ok)) break;
  }
  veystrix::release_ticket(ticket);
  return ok;
}

}  // namespace veystrix::engine::keepwarm

namespace veystrix::engine {
int run_expert_reads(const std::vector<veystrix::PagerRead>& reads, int width) {
  if (reads.empty()) return 0;
  // VEYSTRIX_CPP_READ_WIDTH overrides the lane count of decode expert reads.
  static const int forced = [] {
    const char* e = std::getenv("VEYSTRIX_CPP_READ_WIDTH");
    const int v = e ? std::atoi(e) : 0;
    return v >= 1 && v <= 64 ? v : 0;
  }();
  if (forced) width = forced;
  // VEYSTRIX_CPP_READ_CHUNK=<bytes>: split every read longer than this into
  // pieces of that size (rounded to 16 KiB) so a large row that misses the
  // page cache is fetched by several lanes at once instead of one lane at
  // queue depth 1. Same bytes, same destinations.
  static const int64_t chunk = [] {
    const char* e = std::getenv("VEYSTRIX_CPP_READ_CHUNK");
    long long v = e ? std::atoll(e) : 0;
    if (v <= 0) return int64_t{0};
    v = (v + 16383) / 16384 * 16384;
    return static_cast<int64_t>(v);
  }();
  std::vector<veystrix::PagerRead> split;
  const std::vector<veystrix::PagerRead>* use = &reads;
  if (chunk > 0) {
    split.reserve(reads.size() * 2);
    for (const auto& r : reads) {
      if (r.length <= chunk) { split.push_back(r); continue; }
      for (int64_t off = 0; off < r.length; off += chunk) {
        veystrix::PagerRead p = r;
        p.offset = r.offset + off;
        p.length = std::min<int64_t>(chunk, r.length - off);
        p.dst = r.dst + static_cast<uint64_t>(off);
        split.push_back(p);
      }
    }
    use = &split;
  }
  if (!keepwarm::enabled()) return veystrix::run_batched_reads(*use, width);
  const veystrix::AsyncTicket t =
      veystrix::submit_batched_reads_async(*use, width, [](int) {});
  return keepwarm::wait_ticket_warm(t) ? 0 : 1;
}
}  // namespace veystrix::engine
