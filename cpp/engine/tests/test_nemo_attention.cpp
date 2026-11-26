// Bit-exact parity test for the Nemotron attention decode primitive (bf16 KV).
//
// Golden vectors: tools/gen_nemo_attention_vectors.py -- the REAL
// models.nemotron35.attention.StreamedAttention with a plain KVCache (the
// VEYSTRIX_NEMOTRON_KV=bf16 fallback, == stock NemotronHAttention).
// All comparisons reinterpret floats as raw uint16 (bf16) / uint32 bits.
//
//   cd cpp/engine/build && ./test_nemo_attention ../tests/vectors_nemo_attention
#include "../src/nemo_attention.hpp"

#include <mlx/mlx.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

static bool bit_equal(const mx::array& a, const mx::array& b) {
  if (a.shape() != b.shape() || a.dtype() != b.dtype()) return false;
  mx::array av = mx::view(mx::reshape(a, {-1}), mx::uint32);
  mx::array bv = mx::view(mx::reshape(b, {-1}), mx::uint32);
  mx::eval(av, bv);
  if (av.size() != bv.size()) return false;
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  for (size_t i = 0; i < av.size(); ++i)
    if (pa[i] != pb[i]) return false;
  return true;
}

struct Check { const char* name; mx::array got; mx::array exp; };

static std::string shape_str(const mx::Shape& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) out += ",";
    out += std::to_string(s[i]);
  }
  return out + "]";
}

static void dump_first_diff(const char* label, const mx::array& a,
                            const mx::array& b) {
  mx::array av = mx::view(mx::reshape(a, {-1}), mx::uint32);
  mx::array bv = mx::view(mx::reshape(b, {-1}), mx::uint32);
  mx::eval(av, bv);
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  size_t n = av.size() < bv.size() ? av.size() : bv.size();
  for (size_t i = 0; i < n; ++i) {
    if (pa[i] != pb[i]) {
      std::fprintf(stderr, "    %s first diff at %zu: got 0x%08x expected 0x%08x\n",
                   label, i, pa[i], pb[i]);
      return;
    }
  }
  std::fprintf(stderr, "    %s shapes %s vs %s\n", label,
               shape_str(a.shape()).c_str(), shape_str(b.shape()).c_str());
}

static std::string meta_text(const std::string& dir) {
  std::ifstream f(dir + "/meta.json");
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
static int meta_int(const std::string& txt, const std::string& key) {
  auto p = txt.find("\"" + key + "\": ");
  if (p == std::string::npos) return -1;
  p += key.size() + 4;
  return std::stoi(txt.substr(p));
}
static float meta_float(const std::string& txt, const std::string& key) {
  auto p = txt.find("\"" + key + "\": ");
  if (p == std::string::npos) return -1.0f;
  p += key.size() + 4;
  return std::stof(txt.substr(p));
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <vectordir>\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  auto load = [&](const std::string& n) { return mx::load(dir + "/" + n + ".npy"); };
  const std::string meta = meta_text(dir);

  NemoAttentionConfig cfg;
  cfg.hidden_size = meta_int(meta, "hidden_size");
  cfg.num_heads = meta_int(meta, "num_heads");
  cfg.num_key_value_heads = meta_int(meta, "num_key_value_heads");
  cfg.head_dim = meta_int(meta, "head_dim");
  cfg.scale = meta_float(meta, "scale");
  const int steps = meta_int(meta, "steps");

  NemoAttentionWeights w{load("q_w"), load("k_w"), load("v_w"), load("o_w")};
  NemoAttentionCache cache;
  bool ok = true;

  for (int s = 0; s < steps; ++s) {
    mx::array x = load("x_" + std::to_string(s));
    NemoAttentionDebug dbg;
    mx::array got = nemo_attention_forward(x, w, cfg, cache, &dbg);
    mx::eval(got, *dbg.q, *dbg.k, *dbg.v, *dbg.attended);

    const std::vector<Check> checks = {
        {"output",       got,          load("expected_" + std::to_string(s))},
        {"q",            *dbg.q,       load("attn_q_" + std::to_string(s))},
        {"k",            *dbg.k,       load("attn_k_" + std::to_string(s))},
        {"v",            *dbg.v,       load("attn_v_" + std::to_string(s))},
        {"attended",     *dbg.attended, load("attended_" + std::to_string(s))},
    };
    for (const auto& c : checks) {
      if (!bit_equal(c.got, c.exp)) {
        ok = false;
        std::fprintf(stderr, "[step %d] %s: MISMATCH\n", s, c.name);
        dump_first_diff(c.name, c.got, c.exp);
      }
    }
  }

  // Compare final logical cache state (KVCache.state returned by generator).
  const std::vector<Check> cache_checks = {
      {"cache_keys",   *cache.keys,   load("cache_keys_final")},
      {"cache_values", *cache.values, load("cache_values_final")},
  };
  for (const auto& c : cache_checks) {
    if (!bit_equal(c.got, c.exp)) {
      ok = false;
      std::fprintf(stderr, "[cache] %s: MISMATCH\n", c.name);
      dump_first_diff(c.name, c.got, c.exp);
    }
  }

  if (ok) {
    std::printf("nemo_attention (bf16 KV): BIT-EXACT PASS "
                "(%d decode steps; output + q/k/v + attended + cache state)\n",
                steps);
    return 0;
  }
  std::fprintf(stderr, "nemo_attention (bf16 KV): FAIL\n");
  return 1;
}
