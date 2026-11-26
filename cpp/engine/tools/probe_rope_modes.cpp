// Diagnostic: eager full-attention RoPE (qsa_apply_rope) vs production's
// fused-Metal MRoPE (rope_fused_apply), both checked against the Python oracle
// vectors from gen_rope_mrope_vectors.py, at THIS model's config:
//   dim=64, base=1e7, mrope_section=[11,11,10], style=interleaved, scaling=1.0.
//
// Standalone build:
//   MLXROOT=/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages/mlx
//   xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
//     cpp/engine/tools/probe_rope_modes.cpp \
//     cpp/engine/src/qsa_full.cpp cpp/engine/src/qsa.cpp \
//     cpp/engine/src/rope_fused.cpp cpp/engine/src/engine.cpp \
//     -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$MLXROOT/lib" -o /tmp/probe_rope_modes
//   /tmp/probe_rope_modes cpp/engine/tests/vectors_rope_mrope
#include "qsa_full.hpp"
#include "rope_fused_apply.hpp"

#include <mlx/mlx.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

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
  std::size_t differing = 0;   // raw bit mismatches (strictest)
  std::uint32_t max_ulp = 0;   // max ULP distance among float values
  float max_abs = 0.0f;
};

static std::int32_t ordered(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return static_cast<std::int32_t>(
      (u & 0x80000000u) ? ~u : (u | 0x80000000u));
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

static void print_diff(const char* label, const Diff& d) {
  std::printf("%s: differing=%zu/%zu (%.2f%%), max_ulp=%u, max_abs=%.3e",
              label, d.differing, d.total,
              d.total ? 100.0 * static_cast<double>(d.differing) / d.total
                      : 0.0,
              d.max_ulp, d.max_abs);
}

static int run_case(const std::string& root, const std::string& name) {
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

  // --- Eager path (what full_attention_dense currently calls). ---
  mx::array q_eager = qsa_apply_rope(q, positions, rp);
  mx::array k_eager = qsa_apply_rope(k, positions, rp);

  // --- Fused path (production's compiled Metal MRoPE kernel). ---
  // Exact API: rope_fused_apply(x, position_ids, inv_freq, position_selector,
  //                             rotary_dim). The wrapper rope_apply_fused()
  // builds inv_freq + selector with the production helpers and calls this.
  mx::array q_fused =
      rope_fused_apply(q, positions, inv_freq, selector, rp.dim);
  mx::array k_fused =
      rope_fused_apply(k, positions, inv_freq, selector, rp.dim);
  mx::eval(q_eager, k_eager, q_fused, k_fused);

  Diff eq = compare(q_eager, q_out), ek = compare(k_eager, k_out);
  Diff fq = compare(q_fused, q_out), fk = compare(k_fused, k_out);
  Diff qa = compare(q_eager, q_fused);

  std::printf("\n[%s]  S=%d dtype=%s style=%s dim=%d base=%.0f\n",
              name.c_str(), static_cast<int>(get_num(meta, "S")),
              get_str(meta, "dtype").c_str(), rp.style.c_str(), rp.dim,
              static_cast<double>(rp.base));
  std::printf("  mrope_section=[%d,%d,%d]  scaling=%.1f  oracle=fused-Metal\n",
              rp.mrope_section.size() > 0 ? rp.mrope_section[0] : -1,
              rp.mrope_section.size() > 1 ? rp.mrope_section[1] : -1,
              rp.mrope_section.size() > 2 ? rp.mrope_section[2] : -1,
              static_cast<double>(rp.attention_scaling));
  std::printf("  "); print_diff("eager q vs oracle", eq); std::printf("\n");
  std::printf("  "); print_diff("eager k vs oracle", ek); std::printf("\n");
  std::printf("  "); print_diff("fused q vs oracle", fq); std::printf("\n");
  std::printf("  "); print_diff("fused k vs oracle", fk); std::printf("\n");
  std::printf("  "); print_diff("eager q vs fused q", qa); std::printf("\n");
  return 0;
}

int main(int argc, char** argv) {
  const std::string root =
      argc > 1 ? argv[1] : "cpp/engine/tests/vectors_rope_mrope";
  std::ifstream f(root + "/cases.json");
  if (!f) {
    std::fprintf(stderr, "cases.json missing: run gen_rope_mrope_vectors.py\n");
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
  std::printf("probe_rope_modes: comparing eager qsa_apply_rope vs fused "
              "rope_fused_apply against Python fused-Metal oracle\n");
  int failed = 0;
  for (const auto& c : cases) failed += run_case(root, c);
  return failed ? 1 : 0;
}
