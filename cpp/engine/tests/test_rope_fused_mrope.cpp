// Synthetic Python-oracle ctest for the FULL-ATTENTION RoPE config
// (dim=64, base=1e7, mrope_section=[11,11,10], interleaved, scaling=1.0).
//
// Loads vectors produced by tools/gen_rope_mrope_vectors.py, which drives the
// REAL mlx_vlm MRoPERotaryEmbedding with fused_apply=True (the production
// Metal MRoPE kernel used by Qwen4ExpQSAIndexer._apply_rope). It calls BOTH:
//
//   * the eager path  qsa_apply_rope      (what full_attention_dense uses), and
//   * the fused path  rope_fused_apply    (the bit-exact production port),
//
// reports raw-bit / ULP differences against the Python oracle, and asserts
// that the FUSED path is bit-identical. Eager path differences are reported as
// the key finding (and asserted by default; define REQUIRE_EAGER_DIFF=0 to
// disable that assertion if a future MLX build makes eager match too).
//
// Standalone build (no CMake edit required; primary agent wires the ctest):
//   MLXROOT=/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages/mlx
//   xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
//     cpp/engine/tests/test_rope_fused_mrope.cpp \
//     cpp/engine/src/qsa_full.cpp cpp/engine/src/qsa.cpp \
//     cpp/engine/src/rope_fused.cpp cpp/engine/src/engine.cpp \
//     -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$MLXROOT/lib" \
//     -o /tmp/test_rope_fused_mrope
//   /tmp/test_rope_fused_mrope cpp/engine/tests/vectors_rope_mrope
#include "qsa_full.hpp"
#include "rope_fused_apply.hpp"

#include <mlx/mlx.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifndef REQUIRE_EAGER_DIFF
#define REQUIRE_EAGER_DIFF 1
#endif

namespace mx = mlx::core;
using namespace veystrix::engine;

// ------------------------------ tiny JSON helpers -----------------------------
static std::string get_str(const std::string& s, const std::string& key) {
  auto pos = s.find("\"" + key + "\"");
  if (pos == std::string::npos) return "";
  auto colon = s.find(':', pos);
  if (colon == std::string::npos) return "";
  std::size_t i = colon + 1;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n')) ++i;
  if (i >= s.size() || s[i] != '"') return "";
  ++i;
  std::string out;
  for (; i < s.size() && s[i] != '"'; ++i) out += s[i];
  return out;
}
static double get_num(const std::string& s, const std::string& key) {
  auto pos = s.find("\"" + key + "\"");
  if (pos == std::string::npos) return 0.0;
  auto colon = s.find(':', pos);
  std::size_t i = colon + 1;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n')) ++i;
  std::string num;
  for (; i < s.size(); ++i) {
    char c = s[i];
    if (c == ',' || c == '\n' || c == '}' || c == ' ' || c == '\t' ||
        c == '"') break;
    num += c;
  }
  return std::stod(num);
}
static std::vector<int> get_int_list(const std::string& s,
                                     const std::string& key) {
  std::vector<int> out;
  auto pos = s.find("\"" + key + "\"");
  if (pos == std::string::npos) return out;
  auto lb = s.find('[', pos);
  auto rb = s.find(']', lb);
  if (lb == std::string::npos || rb == std::string::npos) return out;
  for (std::size_t i = lb + 1; i < rb;) {
    if (s[i] == '-' || (s[i] >= '0' && s[i] <= '9')) {
      std::size_t j = i;
      if (s[j] == '-') ++j;
      while (j < rb && s[j] >= '0' && s[j] <= '9') ++j;
      out.push_back(std::stoi(s.substr(i, j - i)));
      i = j;
    } else {
      ++i;
    }
  }
  return out;
}

// -------------------------- bitwise / ULP comparison --------------------------
struct Diff {
  std::size_t total = 0;
  std::size_t differing = 0;   // raw-bit mismatches
  std::uint32_t max_ulp = 0;
  float max_abs = 0.0f;
};
static std::int32_t ordered(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return static_cast<std::int32_t>((u & 0x80000000u) ? ~u
                                                   : (u | 0x80000000u));
}
static Diff compare(const mx::array& got, const mx::array& ref) {
  mx::array g = mx::reshape(got, {-1});
  mx::array r = mx::reshape(ref, {-1});
  mx::eval(g, r);
  Diff d;
  d.total = g.size();
  const float* pg = g.data<float>();
  const float* pr = r.data<float>();
  for (std::size_t i = 0; i < d.total; ++i) {
    std::uint32_t bg, br;
    std::memcpy(&bg, &pg[i], sizeof(bg));
    std::memcpy(&br, &pr[i], sizeof(br));
    if (bg != br) ++d.differing;
    std::int64_t u = static_cast<std::int64_t>(ordered(pg[i])) -
                     static_cast<std::int64_t>(ordered(pr[i]));
    std::uint32_t ulp = static_cast<std::uint32_t>(u < 0 ? -u : u);
    if (ulp > d.max_ulp) d.max_ulp = ulp;
    float ad = std::fabs(pg[i] - pr[i]);
    if (ad > d.max_abs) d.max_abs = ad;
  }
  return d;
}
static bool bit_equal(const mx::array& a, const mx::array& b) {
  if (a.shape() != b.shape() || a.dtype() != b.dtype()) return false;
  // Both float32 and int32 are 4 bytes: view raw bits as uint32.
  mx::array av = mx::view(mx::reshape(a, {-1}), mx::uint32);
  mx::array bv = mx::view(mx::reshape(b, {-1}), mx::uint32);
  mx::eval(av, bv);
  if (av.size() != bv.size()) return false;
  const std::uint32_t* pa = av.data<std::uint32_t>();
  const std::uint32_t* pb = bv.data<std::uint32_t>();
  for (std::size_t i = 0; i < av.size(); ++i)
    if (pa[i] != pb[i]) return false;
  return true;
}

struct Totals {
  std::size_t elements = 0;
  std::size_t eager_differing = 0;
  std::size_t fused_differing = 0;
  std::uint32_t eager_max_ulp = 0;
  std::uint32_t fused_max_ulp = 0;
};

static int run_case(const std::string& root, const std::string& name,
                    Totals& totals) {
  const std::string dir = root + "/" + name;
  auto load = [&](const char* n) { return mx::load(dir + "/" + n + ".npy"); };
  std::ifstream mf(dir + "/meta.json");
  std::string meta((std::istreambuf_iterator<char>(mf)),
                   std::istreambuf_iterator<char>());

  mx::array q = load("q"), k = load("k");
  mx::array q_out = load("q_out"), k_out = load("k_out");
  mx::array positions = load("position_ids");
  mx::array inv_freq = load("inv_freq");
  mx::array selector = load("position_selector");
  mx::eval(q, k, q_out, k_out, positions, inv_freq, selector);

  QSARotaryParams rp;
  rp.dim = static_cast<int>(get_num(meta, "rotary_dim"));
  rp.base = static_cast<float>(get_num(meta, "base"));
  rp.mrope_section = get_int_list(meta, "mrope_section");
  rp.style = get_str(meta, "style");
  rp.attention_scaling = static_cast<float>(get_num(meta, "attention_scaling"));
  rp.max_position_embeddings =
      static_cast<int>(get_num(meta, "max_position_embeddings"));

  // Verify the helper inputs are themselves bit-identical to production.
  mx::array inv_freq_cpp = rope_fused_compute_inv_freq(rp.dim, rp.base);
  mx::array selector_cpp =
      rope_fused_position_selector(rp.mrope_section, rp.dim / 2,
                                   rp.style != "chunked");
  mx::eval(inv_freq_cpp, selector_cpp);
  if (!bit_equal(inv_freq_cpp, inv_freq)) {
    std::fprintf(stderr, "[%s] FAIL: inv_freq helper differs from Python\n",
                 name.c_str());
    return 1;
  }
  if (!bit_equal(selector_cpp, selector)) {
    std::fprintf(stderr,
                 "[%s] FAIL: position_selector helper differs from Python\n",
                 name.c_str());
    return 1;
  }

  // --- Eager path (current full_attention_dense). ---
  mx::array q_eager = qsa_apply_rope(q, positions, rp);
  mx::array k_eager = qsa_apply_rope(k, positions, rp);
  // --- Fused path (production's compiled Metal kernel port). ---
  // Exact API call sequence (kernel inputs come straight from Python):
  //   rope_fused_apply(x, position_ids, inv_freq, position_selector,
  //                    rotary_dim)
  mx::array q_fused =
      rope_fused_apply(q, positions, inv_freq, selector, rp.dim);
  mx::array k_fused =
      rope_fused_apply(k, positions, inv_freq, selector, rp.dim);
  // Also exercise the convenience wrapper (it builds those two inputs itself).
  mx::array q_wrapped = rope_apply_fused(q, positions, rp);
  mx::array k_wrapped = rope_apply_fused(k, positions, rp);
  mx::eval(q_eager, k_eager, q_fused, k_fused, q_wrapped, k_wrapped);

  Diff eq = compare(q_eager, q_out), ek = compare(k_eager, k_out);
  Diff fq = compare(q_fused, q_out), fk = compare(k_fused, k_out);
  totals.elements += q.size() + k.size();
  totals.eager_differing += eq.differing + ek.differing;
  totals.fused_differing += fq.differing + fk.differing;
  totals.eager_max_ulp = std::max(totals.eager_max_ulp,
                                  std::max(eq.max_ulp, ek.max_ulp));
  totals.fused_max_ulp = std::max(totals.fused_max_ulp,
                                  std::max(fq.max_ulp, fk.max_ulp));

  std::printf("\n[%s]  S=%d  dim=%d  base=%.0f  section=[%d,%d,%d]  %s\n",
              name.c_str(), static_cast<int>(get_num(meta, "S")), rp.dim,
              static_cast<double>(rp.base),
              rp.mrope_section.size() > 0 ? rp.mrope_section[0] : -1,
              rp.mrope_section.size() > 1 ? rp.mrope_section[1] : -1,
              rp.mrope_section.size() > 2 ? rp.mrope_section[2] : -1,
              rp.style.c_str());
  std::printf("  eager  q: differing=%zu/%zu max_ulp=%u max_abs=%.3e\n",
              eq.differing, eq.total, eq.max_ulp, eq.max_abs);
  std::printf("  eager  k: differing=%zu/%zu max_ulp=%u max_abs=%.3e\n",
              ek.differing, ek.total, ek.max_ulp, ek.max_abs);
  std::printf("  fused  q: differing=%zu/%zu max_ulp=%u max_abs=%.3e\n",
              fq.differing, fq.total, fq.max_ulp, fq.max_abs);
  std::printf("  fused  k: differing=%zu/%zu max_ulp=%u max_abs=%.3e\n",
              fk.differing, fk.total, fk.max_ulp, fk.max_abs);

  int rc = 0;
  if (fq.differing || fk.differing || fq.max_ulp || fk.max_ulp) {
    std::fprintf(stderr, "[%s] FAIL: fused path is NOT bit-identical to "
                         "Python oracle\n", name.c_str());
    rc = 1;
  }
  if (!bit_equal(q_wrapped, q_out) || !bit_equal(k_wrapped, k_out)) {
    std::fprintf(stderr,
                 "[%s] FAIL: rope_apply_fused wrapper is not bit-identical\n",
                 name.c_str());
    rc = 1;
  }
#if REQUIRE_EAGER_DIFF
  if (!eq.differing && !ek.differing) {
    std::fprintf(stderr,
                 "[%s] NOTE: eager path unexpectedly matches the oracle; "
                 "REQUIRE_EAGER_DIFF=0 to allow\n",
                 name.c_str());
    rc = 1;
  }
#endif
  return rc;
}

int main(int argc, char** argv) {
  const std::string root =
      argc > 1 ? argv[1] : "cpp/engine/tests/vectors_rope_mrope";
  std::ifstream f(root + "/cases.json");
  if (!f) {
    std::fprintf(stderr, "cases.json missing; run gen_rope_mrope_vectors.py\n");
    return 2;
  }
  std::string text((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
  std::vector<std::string> cases;
  std::size_t pos = 0;
  while ((pos = text.find("\"dir\"", pos)) != std::string::npos) {
    auto q1 = text.find('"', text.find(':', pos) + 1);
    auto q2 = text.find('"', q1 + 1);
    if (q1 != std::string::npos && q2 != std::string::npos)
      cases.push_back(text.substr(q1 + 1, q2 - q1 - 1));
    pos = q2;
  }
  if (cases.empty()) {
    std::fprintf(stderr, "no cases parsed\n");
    return 2;
  }

  Totals totals;
  int failed = 0;
  for (const auto& c : cases) failed += run_case(root, c, totals);

  std::printf("\n--- KEY FINDING ---\n");
  std::printf("eager qsa_apply_rope vs Python fused-Metal oracle: "
              "%zu/%zu elements differ (max %u ULP)\n",
              totals.eager_differing, totals.elements, totals.eager_max_ulp);
  std::printf("fused rope_fused_apply vs Python fused-Metal oracle: "
              "%zu/%zu elements differ (max %u ULP)\n",
              totals.fused_differing, totals.elements, totals.fused_max_ulp);
  if (failed) {
    std::fprintf(stderr, "test_rope_fused_mrope FAILED (%d case(s))\n", failed);
    return 1;
  }
  std::printf("test_rope_fused_mrope OK: fused path is bit-identical to "
              "production; eager path is not.\n");
  return 0;
}
