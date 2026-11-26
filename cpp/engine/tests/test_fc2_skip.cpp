// Stage 1-4 validation for the bit-exact fc2 all-zero K-group skip.
// All data is synthetic and built in-process. No checkpoint / SSD / model run.
// Build with the standalone command in research_notes/.../fc2_skip.md.
#include "fc2_skip.hpp"

#include <mlx/mlx.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

static int g_fail = 0;

static void check(bool ok, const char* what) {
  if (ok) {
    std::printf("  [ OK ] %s\n", what);
  } else {
    std::printf("  [FAIL] %s\n", what);
    ++g_fail;
  }
}

// Byte-for-byte compare via mx::view -> uint32 (same convention as
// test_engine_parity.cpp). Returns the number of differing words.
static size_t diff_words(const mx::array& a, const mx::array& b) {
  mx::array av = mx::view(mx::reshape(mx::astype(a, mx::float32), {-1}), mx::uint32);
  mx::array bv = mx::view(mx::reshape(mx::astype(b, mx::float32), {-1}), mx::uint32);
  mx::eval(av, bv);
  if (av.size() != bv.size()) return static_cast<size_t>(-1);
  size_t d = 0;
  for (size_t i = 0; i < av.size(); ++i)
    if (av.data<uint32_t>()[i] != bv.data<uint32_t>()[i]) ++d;
  return d;
}

struct Weights {
  mx::array w, scales, biases;
};

static Weights make_weights(int N, int K, int G, int B, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> wf(static_cast<size_t>(N) * K);
  for (auto& v : wf) v = nd(rng) * 0.1f;
  mx::array dense = mx::reshape(
      mx::array(wf.begin(), {static_cast<int>(wf.size())}), {N, K});
  auto q = mx::quantize(dense, G, B, "affine");
  return Weights{q[0], q[1], q[2]};
}

// relu2-like sparse activations [rows, K].
//   zero_all : groups zero in EVERY row (skippable)
//   zero_first_row_only : groups zero only in row 0 (NOT skippable)
static mx::array make_activations(int rows, int K, int G,
                                  const std::vector<int>& zero_all,
                                  const std::vector<int>& zero_first_row_only,
                                  unsigned seed, bool bf16) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int NG = K / G;
  std::vector<float> a(static_cast<size_t>(rows) * K);
  for (auto& v : a) {
    const float h = nd(rng);
    v = h > 0.f ? h * h : 0.f;  // relu2, exact zeros for h<=0
  }
  for (int g : zero_all) {
    for (int r = 0; r < rows; ++r)
      for (int k = 0; k < G; ++k) a[static_cast<size_t>(r) * K + g * G + k] = 0.f;
  }
  for (int g : zero_first_row_only) {
    if (rows > 0)
      for (int k = 0; k < G; ++k) a[static_cast<size_t>(g) * G + k] = 0.f;
  }
  (void)NG;
  mx::array out = mx::reshape(
      mx::array(a.begin(), {static_cast<int>(a.size())}), {rows, K});
  return bf16 ? mx::astype(out, mx::bfloat16) : out;
}

// --------------------------------- Stage 1 ----------------------------------

static void test_probe() {
  std::printf("[Stage 1] zero-fraction probe (synthetic)\n");
  const int K = 1856, G = 64, rows = 2;
  const std::vector<int> zero_all = {0, 3, 7, 11, 15, 19, 23, 27};  // 8 groups
  const std::vector<int> zero_first = {5};  // zero in row 0 only
  mx::array a = make_activations(rows, K, G, zero_all, zero_first, 1, true);
  ZeroFraction z = probe_zero_fraction(a, G);

  check(z.total_activations == rows * K, "total_activations = rows*intermediate");
  check(z.total_groups == K / G, "total_groups = intermediate/group_size (29)");
  // Compare against a naive reference count over the same (bf16-rounded)
  // activations: relu2 naturally zeroes ~half of the values.
  mx::array af = mx::astype(a, mx::float32);
  mx::eval(af);
  int64_t naive = 0;
  const float* ap = af.data<float>();
  for (size_t i = 0; i < af.size(); ++i) if (ap[i] == 0.0f) ++naive;
  check(z.zero_activations == naive &&
            z.zero_activations >= static_cast<int64_t>(zero_all.size()) * G * rows + G,
        "zero_activations matches naive exact-zero count (includes forced zeros)");
  check(z.zero_groups == static_cast<int64_t>(zero_all.size()),
        "zero_groups = groups zero in EVERY row (8; row-only group excluded)");
  check(z.zero_group_slots == static_cast<int64_t>(zero_all.size()) * rows + 1,
        "zero_group_slots counts (row,group) pairs (17)");
  check(std::fabs(z.group_zero_fraction - 8.0 / 29.0) < 1e-12,
        "group_zero_fraction = 8/29");
  check(z.zero_group_ids == zero_all, "zero_group_ids exactly the 8 skippable groups");
}

// --------------------------------- Stage 2 ----------------------------------

static void test_compaction_experiment() {
  std::printf("[Stage 2] K-compaction experiment (candidate; expected to FAIL bit-identity)\n");
  const int M = 1, K = 1856, N = 2688, G = 64, B = 4;
  const std::vector<int> zero = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27};
  Weights wt = make_weights(N, K, G, B, 11);
  mx::array x = make_activations(M, K, G, zero, {}, 12, true);

  mx::array dense = mx::quantized_matmul(x, wt.w, wt.scales, wt.biases,
                                         true, G, B, "affine");
  Fc2Compacted c = compact_fc2_kgroups(x, wt.w, wt.scales, wt.biases,
                                       zero, G, B);
  mx::array comp = mx::quantized_matmul(c.x, c.w, c.scales, c.biases,
                                        true, G, B, "affine");
  mx::eval(dense, comp);
  const size_t d = diff_words(dense, comp);
  // Diagnostic only: this is the negative result that motivates zero-fill.
  std::printf("    compaction vs dense: %zu differing uint32 words out of %zu -> %s\n",
              d, static_cast<size_t>(dense.size()),
              d == 0 ? "BIT-IDENTICAL (unexpected)" : "DIFFERENT (expected)");
  check(c.kept_groups.size() == static_cast<size_t>(K / G - static_cast<int>(zero.size())),
        "compaction kept the right number of groups");
}

static void run_qmm_case(int M, bool bf16, bool bf16_scales) {
  const int K = 1856, N = 2688, G = 64, B = 4;
  const std::vector<int> zero = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28};
  Weights wt = make_weights(N, K, G, B, 21 + M);
  mx::array x = make_activations(M, K, G, zero, {}, 31 + M, bf16);

  mx::array w = wt.w, s = wt.scales, b = wt.biases;
  if (bf16_scales) {  // checkpoint convention: scales/biases are bf16
    s = mx::astype(s, mx::bfloat16);
    b = mx::astype(b, mx::bfloat16);
  }
  mx::array dense = mx::quantized_matmul(x, w, s, b, true, G, B, "affine");

  Fc2SkipStats stats;
  mx::array skipped = fc2_quantized_matmul_skip(
      x, w, s, b, /*hidden=*/N, /*intermediate=*/K, G, B, &stats);
  mx::eval(dense, skipped);
  const size_t d = diff_words(dense, skipped);
  char name[128];
  std::snprintf(name, sizeof(name),
                "fc2 qmm bit-exact: M=%d %s scales=%s (diff=%zu/%zu, skip=%lld/%lld)",
                M, bf16 ? "bf16" : "f32", bf16_scales ? "bf16" : "f32",
                d, static_cast<size_t>(dense.size()),
                (long long)stats.skipped_groups, (long long)stats.total_groups);
  check(d == 0, name);
  check(stats.skipped_groups == static_cast<int64_t>(zero.size()),
        "stats.skipped_groups matches the probe");
}

static void run_gather_case(int M) {
  const int K = 1856, N = 2688, G = 64, B = 4;
  const std::vector<int> zero = {0, 1, 2, 3, 4, 20, 21, 22, 23, 24, 25, 26, 27, 28};
  Weights wt = make_weights(N, K, G, B, 41 + M);
  mx::array x2 = make_activations(M, K, G, zero, {}, 51 + M, false);
  mx::array x = mx::expand_dims(mx::expand_dims(x2, -2), -3);  // [M,1,1,K]

  mx::array w = mx::expand_dims(wt.w, 0);
  mx::array s = mx::expand_dims(wt.scales, 0);
  mx::array b = mx::expand_dims(wt.biases, 0);
  mx::array idx = mx::zeros({M, 1}, mx::int32);

  mx::array dense = mx::gather_qmm(x, w, s, b, std::nullopt, idx,
                                   true, G, B, "affine", false);
  Fc2SkipStats stats;
  mx::array skipped = fc2_gather_qmm_skip(
      x, w, s, b, idx, /*hidden=*/N, /*intermediate=*/K, G, B, &stats);
  mx::eval(dense, skipped);
  const size_t d = diff_words(dense, skipped);
  char name[96];
  std::snprintf(name, sizeof(name),
                "fc2 gather_qmm bit-exact: M=%d f32 (diff=%zu/%zu)", M, d,
                static_cast<size_t>(dense.size()));
  check(d == 0, name);
}

static void test_control() {
  std::printf("[Stage 2] control: zeroing a NON-zero group must change output\n");
  const int M = 1, K = 1856, N = 2688, G = 64, B = 4;
  Weights wt = make_weights(N, K, G, B, 71);
  mx::array x = make_activations(M, K, G, {}, {}, 72, false);  // no zero groups

  // Manually zero-fill group 5, which has non-zero activations.
  Fc2ZeroFilled z = zero_fill_skipped_groups(
      wt.w, wt.scales, wt.biases, {5}, N, K, G, B);
  mx::array dense = mx::quantized_matmul(x, wt.w, wt.scales, wt.biases, true, G, B, "affine");
  mx::array altered = mx::quantized_matmul(x, z.w, z.scales, z.biases, true, G, B, "affine");
  mx::eval(dense, altered);
  const size_t d = diff_words(dense, altered);
  check(d > 0, "group indexing is correct (zeroing a live group changed the result)");
}

static void test_zero_fill_bit_exact() {
  std::printf("[Stage 2] zero-fill skip vs dense (byte-for-byte gate)\n");
  run_qmm_case(1, false, false);
  run_qmm_case(1, true, false);
  run_qmm_case(1, true, true);  // real checkpoint convention: bf16 x + bf16 scales
  run_qmm_case(4, false, false);
  run_qmm_case(4, true, false);
  run_gather_case(1);
  run_gather_case(4);
}

// --------------------------------- Stage 3 ----------------------------------

static void test_layout_and_repack() {
  std::printf("[Stage 3] chunk layout + K-group-major repack\n");
  const int N = 2688, K = 1856, G = 64, B = 4;
  Fc2ChunkLayout L = fc2_chunk_layout(N, K, G, B, 4);
  check(L.n_groups == 29, "n_groups = 1856/64 = 29");
  check(L.words_per_group_weight == 8, "8 packed uint32 words per group (G64/b4)");
  check(L.bytes_weight_per_group == 86016, "weight bytes/group = 2688*8*4");
  check(L.bytes_scale_per_group == 10752, "scale bytes/group = 2688*4");
  check(L.chunk_bytes == 107520, "chunk bytes = 86016+10752+10752");
  check(L.total_bytes == 3118080, "total fc2 bytes/expert = chunk*29");

  // Repack size and a byte-level spot check.
  Weights wt = make_weights(8, 128, G, B, 81);  // small: hidden=8, K=128 -> 2 groups
  Fc2ChunkLayout S = fc2_chunk_layout(8, 128, G, B, 4);
  std::vector<uint8_t> bytes = repack_fc2_kgroup_major(
      wt.w, wt.scales, wt.biases, 8, 128, G, B);
  check(static_cast<int64_t>(bytes.size()) == S.total_bytes,
        "repack produced exactly chunk_bytes*n_groups bytes");
  mx::eval(wt.w, wt.scales, wt.biases);
  // Group 1, row 0: first weight word must equal source w[0, 8] (little-endian).
  const uint32_t* srcw = wt.w.data<uint32_t>();
  uint32_t got = 0;
  std::memcpy(&got, bytes.data() + S.chunk_bytes + 0, 4);
  check(got == srcw[0 * (128 / 8) + 1 * 8],
        "repack group1/row0 first weight word matches source");
  const float* srcs = wt.scales.data<float>();
  float gs = 0.f;
  std::memcpy(&gs, bytes.data() + S.chunk_bytes + 32, 4);
  check(gs == srcs[0 * 2 + 1], "repack group1/row0 scale matches source");
}

// --------------------------------- Stage 4 ----------------------------------

static void test_policy() {
  std::printf("[Stage 4] env policy + break-even rule\n");
  Fc2SkipPolicy p = fc2_skip_policy_from_env();
  check(!p.enabled, "VEYSTRIX_NEMOTRON_FC2_SKIP defaults OFF in this test process");
  check(fc2_layout_from_env() == Fc2Layout::Dense,
        "original dense layout is the default fallback");

  ZeroFraction zf;
  zf.total_groups = 29;
  zf.zero_groups = 20;
  zf.group_zero_fraction = 20.0 / 29.0;
  Fc2SkipPolicy off;
  off.enabled = false;
  check(!should_skip_fc2(off, zf, 10 * 1000000),
        "disabled policy never skips (rollback = 0)");
  Fc2SkipPolicy on;
  on.enabled = true;
  on.min_skip_fraction = 0.5;
  on.round_trip_bytes = 3 * 1000000;
  check(should_skip_fc2(on, zf, 5 * 1000000),
        "enabled: skip when fraction>=threshold and saved>round-trip");
  check(!should_skip_fc2(on, zf, 2 * 1000000),
        "enabled: no skip when saved does not pay for round trip");
  zf.group_zero_fraction = 0.3;
  check(!should_skip_fc2(on, zf, 5 * 1000000),
        "enabled: no skip below the zero-fraction threshold");
}

int main() {
  std::printf("=== fc2_skip: stage 1-4 validation (synthetic, byte-exact) ===\n");
  try {
    test_probe();
    test_compaction_experiment();
    test_zero_fill_bit_exact();
    test_control();
    test_layout_and_repack();
    test_policy();
  } catch (const std::exception& e) {
    std::printf("EXCEPTION: %s\n", e.what());
    return 2;
  }
  if (g_fail) {
    std::printf("=== FAILED: %d check(s) ===\n", g_fail);
    return 1;
  }
  std::printf("=== ALL fc2_skip CHECKS PASSED ===\n");
  return 0;
}
