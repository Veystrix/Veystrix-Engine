// Bit-exact ctest for the KV tier-quant primitives.
//   test_kv_quant <vectors_kv_quant-dir>
//
// Golden vectors come from tools/gen_kv_quant_vectors.py -- the REAL Python
// mx.quantize / mx.dequantize ops used by models/flashnext/quantized_kv.py and
// the REAL MixedPrecisionQSAKVCache(mode="tiered4") from mixed_kv.py.
// Every comparison is byte-for-byte on the raw uint32 view of each array
// (bf16 scales/biases and dequant outputs included).
//
//   cd cpp/engine/build && ./test_kv_quant ../tests/vectors_kv_quant
#include "kv_quant.hpp"

#include <mlx/mlx.h>

#include <cstdint>
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

static std::string meta_text(const std::string& path) {
  std::ifstream f(path);
  if (!f) { std::fprintf(stderr, "missing %s\n", path.c_str()); return ""; }
  std::stringstream ss; ss << f.rdbuf();
  return ss.str();
}

static int meta_int(const std::string& txt, const std::string& key) {
  auto p = txt.find("\"" + key + "\": ");
  if (p == std::string::npos) return -1;
  p += key.size() + 4;
  return std::stoi(txt.substr(p));
}

static std::vector<int> meta_int_array(const std::string& txt,
                                       const std::string& key) {
  std::vector<int> out;
  auto p = txt.find("\"" + key + "\": [");
  if (p == std::string::npos) return out;
  p = txt.find('[', p) + 1;
  auto e = txt.find(']', p);
  std::string body = txt.substr(p, e - p);
  size_t pos = 0;
  while ((pos = body.find_first_of("-0123456789", pos)) != std::string::npos) {
    size_t end = body.find_first_not_of("-0123456789", pos);
    out.push_back(std::stoi(body.substr(pos, end - pos)));
    pos = end == std::string::npos ? end : end + 1;
  }
  return out;
}

// Raw bit view: reshape flat then view as uint32. All vectors have an even
// number of bf16 elements (or are uint32 already), so this is lossless.
static mx::array raw(const mx::array& a) {
  return mx::view(mx::reshape(a, mx::Shape{-1}), mx::uint32);
}

static bool bit_equal(const mx::array& a, const mx::array& b) {
  if (a.shape() != b.shape() || a.dtype() != b.dtype()) return false;
  mx::array av = raw(a), bv = raw(b);
  mx::eval(av, bv);
  if (av.size() != bv.size()) return false;
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  for (size_t i = 0; i < av.size(); ++i)
    if (pa[i] != pb[i]) return false;
  return true;
}

static void check_bits(const char* label, const mx::array& got,
                       const mx::array& expected) {
  if (bit_equal(got, expected)) return;
  mx::array av = raw(got), bv = raw(expected);
  mx::eval(av, bv);
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  size_t n = av.size() < bv.size() ? av.size() : bv.size();
  size_t diff = n;
  for (size_t i = 0; i < n; ++i)
    if (pa[i] != pb[i]) { diff = i; break; }
  std::fprintf(stderr, "  %s: MISMATCH first diff word %zu "
                       "(got 0x%08x expected 0x%08x), sizes %zu/%zu\n",
               label, diff, diff < n ? pa[diff] : 0, diff < n ? pb[diff] : 0,
               av.size(), bv.size());
  ++failures;
}

static void check_packed(const char* prefix, const KVPacked& got,
                         const mx::array& exp_data, const mx::array& exp_scales,
                         const mx::array& exp_biases) {
  std::string l = prefix;
  check_bits((l + ".data").c_str(), got.data, exp_data);
  check_bits((l + ".scales").c_str(), got.scales, exp_scales);
  check_bits((l + ".biases").c_str(), got.biases, exp_biases);
}

int main(int argc, char** argv) {
  // The golden vectors were generated with mx.set_default_device(mx.cpu) and
  // MLX's CPU/GPU quantize kernels differ by 1 ULP on some values. Use the same
  // device so the comparison isolates the op, not the backend.
  mx::set_default_device(mx::Device(mx::Device::DeviceType::cpu));
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <vectors_kv_quant-dir>\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  auto load = [&](const std::string& cd, const char* n) {
    return mx::load(cd + "/" + std::string(n) + ".npy");
  };

  // ---- affine INT4 / INT8 primitives ---------------------------------------
  const std::vector<std::string> affine = {"affine4", "affine8", "affine4_wide"};
  for (const auto& name : affine) {
    const std::string cd = dir + "/" + name;
    const std::string meta = meta_text(cd + "/meta.json");
    const int gs = meta_int(meta, "group_size");
    const int bits = meta_int(meta, "bits");
    mx::array x = load(cd, "x");
    KVPacked q = kv_quantize(x, gs, bits);
    check_packed((name + " packed").c_str(), q, load(cd, "packed"),
                 load(cd, "scales"), load(cd, "biases"));
    mx::array dq = kv_dequantize(q);
    check_bits((name + " dequant").c_str(), dq, load(cd, "dequant"));
    std::printf("  %s: gs=%d bits=%d packed/scales/biases/dequant OK\n",
                name.c_str(), gs, bits);
  }

  // ---- tiered4 split: INT4 history + INT8 recent --------------------------
  {
    const std::string cd = dir + "/tiered4";
    const std::string meta = meta_text(cd + "/meta.json");
    const int gs = meta_int(meta, "group_size");
    const int recent_tokens = meta_int(meta, "recent_tokens");
    const int offset = meta_int(meta, "offset");
    const int history = meta_int(meta, "history_tokens");
    const std::vector<int> chunks = meta_int_array(meta, "chunks");

    Tiered4KV cache(gs, recent_tokens);
    for (size_t i = 0; i < chunks.size(); ++i) {
      cache.append(load(cd, ("chunk_k_" + std::to_string(i)).c_str()),
                   load(cd, ("chunk_v_" + std::to_string(i)).c_str()));
    }
    CHECK(cache.offset() == offset, "tiered4 offset");
    CHECK(cache.history_tokens() == history, "tiered4 history_tokens");
    if (cache.offset() != offset)
      std::fprintf(stderr, "  offset: got %d expected %d\n", cache.offset(), offset);
    if (cache.history_tokens() != history)
      std::fprintf(stderr, "  history_tokens: got %d expected %d\n",
                   cache.history_tokens(), history);

    check_packed("tiered4 hist_keys", cache.history_keys(),
                 load(cd, "hist_keys_data"), load(cd, "hist_keys_scales"),
                 load(cd, "hist_keys_biases"));
    check_packed("tiered4 hist_values", cache.history_values(),
                 load(cd, "hist_values_data"), load(cd, "hist_values_scales"),
                 load(cd, "hist_values_biases"));
    check_bits("tiered4 hist_keys_dequant", cache.dequantize_history_keys(),
               load(cd, "hist_keys_dequant"));
    check_bits("tiered4 hist_values_dequant", cache.dequantize_history_values(),
               load(cd, "hist_values_dequant"));

    check_packed("tiered4 recent_keys", cache.recent_keys(),
                 load(cd, "recent_keys_data"), load(cd, "recent_keys_scales"),
                 load(cd, "recent_keys_biases"));
    check_packed("tiered4 recent_values", cache.recent_values(),
                 load(cd, "recent_values_data"), load(cd, "recent_values_scales"),
                 load(cd, "recent_values_biases"));
    check_bits("tiered4 recent_keys_dequant", cache.dequantize_recent_keys(),
               load(cd, "recent_keys_dequant"));
    check_bits("tiered4 recent_values_dequant", cache.dequantize_recent_values(),
               load(cd, "recent_values_dequant"));

    std::printf("  tiered4: chunks=%zu offset=%d history_tokens=%d "
                "(INT4 history + INT8 recent) OK\n",
                chunks.size(), cache.offset(), cache.history_tokens());
  }

  if (failures) {
    std::fprintf(stderr, "test_kv_quant FAILED (%d mismatches)\n", failures);
    return 1;
  }
  std::printf("test_kv_quant: BIT-EXACT PASS (affine4/affine8/affine4_wide "
              "pack+dequant, tiered4 history/recent split)\n");
  return 0;
}
