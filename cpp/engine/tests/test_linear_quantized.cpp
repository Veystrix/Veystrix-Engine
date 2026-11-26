// Bit-exact ctest for the packed linear projection path.
//   test_linear_quantized <vectors_quant_linear-dir>
// For every case produced by gen_quant_linear_vectors.py, C++ linear(x, L)
// with a packed QPacked weight must be byte-identical to Python's
// nn.QuantizedLinear.__call__ output. The dense branch is also checked to be
// byte-identical to matrix transpose-matmul (regression for the old path).
#include "engine.hpp"

#include <mlx/mlx.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

static int failures = 0;
#define CHECK(cond, msg)                                                     \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);         \
      ++failures;                                                            \
    }                                                                        \
  } while (0)

static bool bit_equal(const mx::array& a, const mx::array& b) {
  if (a.shape() != b.shape()) return false;
  mx::array av = mx::view(mx::reshape(mx::astype(a, mx::float32), mx::Shape{-1}), mx::uint32);
  mx::array bv = mx::view(mx::reshape(mx::astype(b, mx::float32), mx::Shape{-1}), mx::uint32);
  mx::eval(av, bv);
  if (av.size() != bv.size()) return false;
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  for (size_t i = 0; i < av.size(); ++i)
    if (pa[i] != pb[i]) return false;
  return true;
}

static std::string json_get(const std::string& s, const std::string& key) {
  std::string pat = "\"" + key + "\"";
  size_t p = s.find(pat);
  if (p == std::string::npos) return "";
  size_t colon = s.find(':', p + pat.size());
  size_t b = s.find_first_not_of(" \t\n\r", colon + 1);
  if (b == std::string::npos) return "";
  if (s[b] == '"') { size_t e = s.find('"', b + 1); return s.substr(b + 1, e - b - 1); }
  size_t e = s.find_first_of(",}\n", b);
  return s.substr(b, e - b);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <vectors_quant_linear-dir>\n", argv[0]);
    return 2;
  }
  std::string dir = argv[1];
  std::ifstream mf(dir + "/meta.json");
  if (!mf) { std::fprintf(stderr, "meta.json missing\n"); return 2; }
  std::stringstream ms; ms << mf.rdbuf();
  std::string meta = ms.str();
  int gs = std::stoi(json_get(meta, "group_size"));
  int bits = std::stoi(json_get(meta, "bits"));

  // Parse the list of case npy-dirs by scanning "npy_dir": "..." entries.
  std::vector<std::string> cases;
  size_t pos = 0;
  while ((pos = meta.find("\"npy_dir\"", pos)) != std::string::npos) {
    size_t colon = meta.find(':', pos);
    size_t b = meta.find('"', colon);
    size_t e = meta.find('"', b + 1);
    cases.push_back(meta.substr(b + 1, e - b - 1));
    pos = e + 1;
  }
  if (cases.empty()) { std::fprintf(stderr, "no cases in meta.json\n"); return 2; }

  for (const auto& c : cases) {
    std::string cd = dir + "/" + c;
    auto load = [&](const char* n) { return mx::load(cd + "/" + std::string(n) + ".npy"); };
    mx::array x = load("x"), w = load("weight"), s = load("scales"),
              b = load("biases"), expected = load("expected");
    bool has_bias = std::ifstream(cd + "/bias.npy").good();

    std::optional<mx::array> bias;
    if (has_bias) bias = load("bias");
    Linear L = Linear::quantized(w, s, b, gs, bits, bias);
    mx::array out = linear(x, L);
    mx::eval(out, expected);
    if (!bit_equal(out, expected)) {
      mx::array ov = mx::view(mx::reshape(mx::astype(out, mx::float32), mx::Shape{-1}), mx::uint32);
      mx::array ev = mx::view(mx::reshape(mx::astype(expected, mx::float32), mx::Shape{-1}), mx::uint32);
      mx::eval(ov, ev);
      size_t n = ov.size() < ev.size() ? ov.size() : ev.size();
      const uint32_t* pa = ov.data<uint32_t>();
      const uint32_t* pb = ev.data<uint32_t>();
      size_t diff = n;
      for (size_t i = 0; i < n; ++i) if (pa[i] != pb[i]) { diff = i; break; }
      std::fprintf(stderr, "  %s: first diff %zu (0x%08x vs 0x%08x)\n",
                   c.c_str(), diff, diff < n ? pa[diff] : 0, diff < n ? pb[diff] : 0);
    }
    CHECK(bit_equal(out, expected), ("packed linear output: " + c).c_str());

    // Dense branch of linear() must equal the old explicit expression.
    mx::array dense_w = mx::dequantize(w, s, b, gs, bits, "affine");
    mx::array dense_out = linear(x, Linear(dense_w));
    mx::array old_out = mx::matmul(x, mx::transpose(dense_w));
    mx::eval(dense_out, old_out);
    CHECK(bit_equal(dense_out, old_out), ("dense branch == old matmul: " + c).c_str());
    std::printf("  %s: OK\n", c.c_str());
  }

  if (failures) { std::fprintf(stderr, "test_linear_quantized FAILED (%d)\n", failures); return 1; }
  std::printf("test_linear_quantized OK (%zu cases)\n", cases.size());
  return 0;
}
