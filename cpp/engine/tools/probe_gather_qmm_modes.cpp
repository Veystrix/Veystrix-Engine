// probe_gather_qmm_modes -- standalone MLX-only probe (no engine linkage).
//
// QUESTION this probe answers
// ---------------------------
// Does `mx::gather_qmm` with a BATCHED activation operand (M_total = T rows,
// produced by expand_dims(x[T,in], (-2,-3)) -> x[T,1,1,in]) produce the SAME
// float32 bit results as T separate gather_qmm calls with one row each
// (x[t:t+1] -> [1,1,1,in])?
//
// This is the exact fork in the C++ tool's integrated batched prefill: the
// tool processes MoE ONE TOKEN AT A TIME (per-token chunking via mx::take, to
// stay inside the 16-slot resident pool), while Python production runs the
// whole 16-token prefill batch through gather_qmm at once. If the two are not
// bit-exact (different kernel dispatch -- GEMV-like M=1 vs GEMM-like M=16),
// per-token chunking itself is a divergence source.
//
// Two cases, both with real projection shapes:
//   (1) single-expert stack: weight [1, 2560, 320] u32 (packed in=2560),
//       scales/biases [1, 2560, 80] bf16, rhs_indices all 0, x [16, 2560].
//   (2) multi-expert stack:  weight [64, 2560, 320] u32,
//       scales/biases [64, 2560, 80] bf16, rhs_indices random in [0,64),
//       x [16, 2560].
//
// For each case:
//   A = gather_qmm(x4[T,1,1,2560], w, s, b, rhs_indices=route[T,K], ...)
//   B = concatenate([ gather_qmm(x4[t:t+1], w, s, b,
//                               rhs_indices=route[t:t+1], ...) for t in 0..T ])
// then report A-vs-B: element count, #bit-differing elements, max float32 ULP,
// max abs diff. Also reports an A-vs-A recomputation as a zero-baseline.
//
// Standalone build (absolute MLX paths so the worktree needs no .venv symlink):
//   MLXROOT=/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages/mlx
//   xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" \
//     cpp/engine/tools/probe_gather_qmm_modes.cpp \
//     -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$MLXROOT/lib" -o /tmp/probe_gather_qmm_modes
// Run: /tmp/probe_gather_qmm_modes
#include <mlx/mlx.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace mx = mlx::core;

// ------------------------------ tiny PRNG ------------------------------------
static uint64_t g_state = 0x9e3779b97f4a7c15ULL;
static uint32_t next_u32() {
  g_state = g_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<uint32_t>(g_state >> 32);
}
static float next_unit() {  // uniform [-1, 1)
  return static_cast<float>(static_cast<int32_t>(next_u32())) / 2147483648.0f;
}

// -------------------------- bitwise / ULP comparison -------------------------
static uint32_t f32_ordered(uint32_t bits) {
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
static uint32_t f32_ulp(uint32_t a, uint32_t b) {
  uint32_t oa = f32_ordered(a), ob = f32_ordered(b);
  return oa >= ob ? oa - ob : ob - oa;
}

struct Diff {
  size_t total = 0;
  size_t differing = 0;   // strict float32 bit-pattern mismatches
  uint32_t max_ulp = 0;
  float max_abs = 0.0f;
};

static Diff compare(const mx::array& a_in, const mx::array& b_in) {
  mx::array a = mx::astype(mx::reshape(a_in, {-1}), mx::float32);
  mx::array b = mx::astype(mx::reshape(b_in, {-1}), mx::float32);
  mx::eval(a, b);
  Diff d;
  d.total = a.size();
  const float* pa = a.data<float>();
  const float* pb = b.data<float>();
  for (size_t i = 0; i < d.total; ++i) {
    uint32_t ba, bb;
    std::memcpy(&ba, &pa[i], sizeof(ba));
    std::memcpy(&bb, &pb[i], sizeof(bb));
    if (ba != bb) ++d.differing;
    uint32_t u = f32_ulp(ba, bb);
    if (u > d.max_ulp) d.max_ulp = u;
    float ad = std::fabs(pa[i] - pb[i]);
    if (ad > d.max_abs) d.max_abs = ad;
  }
  return d;
}

static void print_diff(const char* label, const Diff& d) {
  std::printf(
      "  %-28s total=%zu  differing=%zu (%.4f%%)  max_ulp=%u  max_abs=%.9g\n",
      label, d.total, d.differing,
      d.total ? 100.0 * static_cast<double>(d.differing) / d.total : 0.0,
      d.max_ulp, d.max_abs);
}

// ------------------------------- the probe ----------------------------------
// in = 2560, out = 2560, group_size = 32, bits = 4.
// weight logical [out, in] packed along last axis -> [out, in/8=u32] u32.
// scales/biases [out, in/gs] bf16.
static void run_case(const char* name, int n_experts, int T, int K,
                     bool sorted = false, bool use_take = false) {
  constexpr int IN = 2560;
  constexpr int OUT = 2560;
  constexpr int GS = 32;
  constexpr int BITS = 4;
  constexpr int IN_PACK = IN / 8;  // 320 u32 words
  constexpr int N_GROUPS = IN / GS; // 80

  std::printf("\n=== %s : experts=%d T=%d K=%d sorted=%d take=%d ===\n",
              name, n_experts, T, K, static_cast<int>(sorted),
              static_cast<int>(use_take));

  // --- random packed Q4 weight [E, OUT, IN_PACK] u32 ---
  const size_t nw = static_cast<size_t>(n_experts) * OUT * IN_PACK;
  std::vector<uint32_t> wb(nw);
  for (auto& v : wb) v = next_u32();
  mx::array w(wb.begin(), {n_experts, OUT, IN_PACK}, mx::uint32);

  // --- scales/biases [E, OUT, N_GROUPS] bf16, from realistic f32 values ---
  // (positive scales ~U[0.5,1.5], biases ~U[-0.1,0.1], matching mlx quantize)
  const size_t ns = static_cast<size_t>(n_experts) * OUT * N_GROUPS;
  std::vector<float> sbf(ns), bbf(ns);
  for (size_t i = 0; i < ns; ++i) {
    sbf[i] = 0.5f + next_unit() * 0.5f;
    bbf[i] = next_unit() * 0.1f;
  }
  mx::array s = mx::astype(mx::array(sbf.begin(), {n_experts, OUT, N_GROUPS}, mx::float32), mx::bfloat16);
  mx::array b = mx::astype(mx::array(bbf.begin(), {n_experts, OUT, N_GROUPS}, mx::float32), mx::bfloat16);

  // --- activation x [T, IN] f32 ---
  std::vector<float> xf(static_cast<size_t>(T) * IN);
  for (auto& v : xf) v = next_unit() * 0.3f;
  mx::array x(xf.begin(), {T, IN}, mx::float32);
  mx::array x4 = mx::expand_dims(x, std::vector<int>{-2, -3});  // [T,1,1,IN]

  // --- rhs_indices [T, K] int32 ---
  std::vector<int32_t> rf(static_cast<size_t>(T) * K);
  for (auto& v : rf)
    v = (n_experts == 1) ? 0 : static_cast<int32_t>(next_u32() % n_experts);
  mx::array route(rf.begin(), {T, K}, mx::int32);

  auto qmm = [&](const mx::array& xb, const mx::array& ridx) {
    return mx::gather_qmm(xb, w, s, b,
                          /*lhs_indices=*/std::nullopt,
                          /*rhs_indices=*/ridx,
                          /*transpose=*/true,
                          /*group_size=*/GS,
                          /*bits=*/BITS,
                          /*mode=*/"affine",
                          /*sorted_indices=*/sorted);
  };

  // A: one batched call, M_total = T rows.
  mx::array A = qmm(x4, route);

  // B: T separate M=1 calls, concatenated along token axis.
  std::vector<mx::array> rows;
  rows.reserve(T);
  for (int t = 0; t < T; ++t) {
    // mx::array has no default ctor, so build xb in a lambda.
    auto row_x = [&]() {
      if (use_take) {
        // Mirrors the tool's actual per-token chunking via mx::take.
        mx::array ti = mx::array({t}, {1}, mx::int32);
        return mx::take(x4, ti, 0);                // [1,1,1,IN]
      }
      return mx::slice(x4, {t, 0, 0, 0}, {t + 1, 1, 1, IN}, {1, 1, 1, 1});
    };
    mx::array rb = mx::slice(route, {t, 0}, {t + 1, K}, {1, 1});
    rows.push_back(qmm(row_x(), rb));
  }
  mx::array B = mx::concatenate(rows, 0);

  std::printf("  A shape: [");
  for (size_t i = 0; i < A.shape().size(); ++i)
    std::printf("%s%d", i ? "," : "", A.shape(i));
  std::printf("]  B shape: [");
  for (size_t i = 0; i < B.shape().size(); ++i)
    std::printf("%s%d", i ? "," : "", B.shape(i));
  std::printf("]\n");

  print_diff("A(batched M=16) vs B(16x M=1)", compare(A, B));
  // Sanity baseline: recompute A and compare; a correct harness yields 0.
  mx::array A2 = qmm(x4, route);
  print_diff("A vs A2 (determinism baseline)", compare(A, A2));
}

int main() {
  std::printf(
      "probe_gather_qmm_modes: batched gather_qmm vs per-token gather_qmm\n"
      "  affine Q4, group_size=32, bits=4, x[T,1,1,2560] f32, "
      "rhs_indices[T,10] int32, out=2560\n");

  // Single-expert stack: rhs_indices all 0; isolates kernel dispatch only.
  run_case("single-expert stack", /*n_experts=*/1, /*T=*/16, /*K=*/10);

  // Real MoE multi-expert stack with random routing.
  run_case("multi-expert stack", /*n_experts=*/64, /*T=*/16, /*K=*/10);

  // Robustness: sorted_indices=true (Python uses it for >=64 gathered indices).
  run_case("multi-expert sorted=true", 64, 16, 10, /*sorted=*/true);

  // Robustness: per-token rows obtained via mx::take (the tool's real path).
  run_case("multi-expert take-sliced", 64, 16, 10, /*sorted=*/false,
           /*use_take=*/true);

  std::printf("\nDone.\n");
  return 0;
}
