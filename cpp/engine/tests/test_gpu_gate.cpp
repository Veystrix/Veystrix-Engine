// GPU gate: work encoded and committed before its input bytes exist must wait for
// signal(), then see the bytes the CPU wrote after the commit.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include <mlx/mlx.h>

#include "gpu_gate.hpp"

namespace mx = mlx::core;
namespace gate = veystrix::engine::gpu_gate;

int main() {
  int fails = 0;
  for (int round = 0; round < 3; ++round) {
    const int n = 1 << 16;
    mx::allocator::Buffer buf = mx::allocator::malloc(sizeof(float) * n);
    float* p = static_cast<float*>(buf.raw_ptr());
    for (int i = 0; i < n; ++i) p[i] = -1.0f;  // stale bytes the GPU must not see
    mx::array x(buf, mx::Shape{n}, mx::float32);
    mx::array y = mx::sum(x * 2.0f);
    const uint64_t v = gate::arm();
    if (v == 0) {
      std::printf("FAIL: arm returned 0\n");
      return 1;
    }
    mx::async_eval(y);  // encoded and committed behind the wait
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    for (int i = 0; i < n; ++i) p[i] = static_cast<float>(i % 7);
    std::thread t([v] { gate::signal(v); });  // released from another thread, as the read lanes do
    t.join();
    mx::eval(y);
    double want = 0;
    for (int i = 0; i < n; ++i) want += 2.0 * (i % 7);
    const float got = y.item<float>();
    const bool ok = std::abs(got - want) < 1e-3 * want;
    std::printf("round %d: sum %.1f want %.1f %s\n", round, got, want, ok ? "ok" : "FAIL");
    fails += ok ? 0 : 1;
  }
  std::printf(fails ? "FAILED\n" : "PASSED\n");
  return fails ? 1 : 0;
}
