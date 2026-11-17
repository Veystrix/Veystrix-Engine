// Minimal MLX C++ build/runtime probe for the Veystrix decode port.
#include <mlx/mlx.h>
#include <cstdio>

namespace mx = mlx::core;

int main() {
  auto a = mx::array({1.0f, 2.0f, 3.0f, 4.0f});
  auto b = mx::array({5.0f, 6.0f, 7.0f, 8.0f});
  auto c = a + b;
  mx::eval(c);
  const float* out = c.data<float>();
  for (size_t i = 0; i < c.size(); ++i) std::printf("%g ", out[i]);
  std::printf("\nmlx %s, metal=%d, mlx_probe OK\n",
              mx::version(), (int)mx::metal::is_available());
  return 0;
}
