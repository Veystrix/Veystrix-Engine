// probe_quantized -- standalone MLX-only probe (no engine linkage).
//
// Compares, for random packed affine Q4 weights of the shapes used by
// Flash-Next's 2560-wide projections, two ways of computing a matmul:
//
//   A = mx::quantized_matmul(x, w, s, b, /*transpose=*/true, gs, bits, "affine")
//   B = mx::matmul(x, mx::transpose(mx::dequantize(w, s, b, gs, bits, "affine")))
//
// and reports, for each case, the max float32 ULP distance between A and B and
// the number of elements whose float32 bit patterns differ. This is diagnostic
// only: it does NOT assert pass/fail. It is the empirical counterpart to the
// "quantized_matmul vs dequantize+matmul" parity gap documented in
// docs/performance/CPP_ENGINE_REAL_CROSSCHECK.md.
//
// Build (repo root, absolute MLX paths because this worktree has no .venv):
//   MLXROOT=/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages/mlx
//   xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" \
//     cpp/engine/tools/probe_quantized.cpp \
//     -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$MLXROOT/lib" -o /tmp/probe_quantized
// Run: /tmp/probe_quantized
#include <mlx/mlx.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace mx = mlx::core;

// Deterministic host PRNG so the probe is reproducible without depending on
// MLX's random key machinery (which differs between builds).
static uint64_t g_state = 0x9e3779b97f4a7c15ULL;
static float next_unit() {
  g_state = g_state * 6364136223846793005ULL + 1442695040888963407ULL;
  // Map the high 32 bits to [-1, 1).
  return static_cast<float>(static_cast<int32_t>(g_state >> 32)) / 2147483648.0f;
}

// Map a float32 bit pattern to a monotonic unsigned integer so that the
// numeric ULP distance is simply |ordered(a) - ordered(b)|.
static uint32_t f32_ordered(uint32_t bits) {
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

static uint32_t f32_ulp(uint32_t a_bits, uint32_t b_bits) {
  uint32_t oa = f32_ordered(a_bits);
  uint32_t ob = f32_ordered(b_bits);
  return oa >= ob ? oa - ob : ob - oa;
}

static void run_case(int m, int in, int out, int group_size, int bits) {
  // Random float weight [out, in]; in=2560 for the model's dense projections.
  std::vector<float> wf(static_cast<size_t>(out) * in);
  for (auto& v : wf) v = next_unit();
  mx::array w_f32(wf.begin(), {static_cast<int>(out), in}, mx::float32);

  auto q = mx::quantize(w_f32, group_size, bits, "affine");
  mx::array w = q[0], s = q[1], b = q[2];

  // Random activation [M, in].
  std::vector<float> xf(static_cast<size_t>(m) * in);
  for (auto& v : xf) v = next_unit();
  mx::array x(xf.begin(), {m, in}, mx::float32);

  mx::array a = mx::quantized_matmul(
      x, w, s, b, /*transpose=*/true, group_size, bits, "affine");
  mx::array dq = mx::dequantize(w, s, b, group_size, bits, "affine");
  mx::array bmat = mx::matmul(x, mx::transpose(dq));

  // Compare float32 bit patterns. Keep the dequantize/matmul dtypes as MLX
  // produced them but view both results as float32 for the ULP comparison.
  a = mx::astype(a, mx::float32);
  bmat = mx::astype(bmat, mx::float32);
  mx::eval(a, bmat);

  const float* pa = a.data<float>();
  const float* pb = bmat.data<float>();
  const size_t n = a.size();
  uint32_t max_ulp = 0;
  size_t differing = 0;
  for (size_t i = 0; i < n; ++i) {
    uint32_t ba, bb;
    std::memcpy(&ba, &pa[i], sizeof(ba));
    std::memcpy(&bb, &pb[i], sizeof(bb));
    if (ba != bb) ++differing;
    uint32_t u = f32_ulp(ba, bb);
    if (u > max_ulp) max_ulp = u;
  }

  std::printf(
      "M=%2d in=%d out=%d gs=%2d bits=%d : "
      "max_ulp=%u  differing=%zu/%zu\n",
      m, in, out, group_size, bits, max_ulp, differing, n);
}

int main() {
  std::printf(
      "probe_quantized: quantized_matmul vs dequantize+matmul "
      "(float32 bit-pattern comparison)\n");
  // 2560-wide projections, M=1 and M=16, group sizes 32 and 64, Q4 affine.
  for (int m : {1, 16}) {
    for (int gs : {32, 64}) {
      run_case(m, /*in=*/2560, /*out=*/2560, gs, /*bits=*/4);
    }
  }
  return 0;
}
