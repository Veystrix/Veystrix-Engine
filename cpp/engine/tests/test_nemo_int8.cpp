// Bit-exact parity test for the production-default INT8 KV Nemotron attention.
//
// Golden vectors: tools/gen_nemo_int8_vectors.py -- the REAL
// models.nemotron35.attention.StreamedAttention with a REAL AttentionCache
// (QuantizedKVCache blocks, affine-int8-g64). All comparisons reinterpret
// floats as raw uint16 (bf16) / uint32 (packed/scales/biases) bits.
//
//   cd cpp/engine/build && ./test_nemo_int8 ../tests/vectors_nemo_int8
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

static std::string meta_text(const std::string& path) {
  std::ifstream f(path);
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
  const std::string meta = meta_text(dir + "/meta.json");

  NemoAttentionConfig cfg;
  cfg.hidden_size = meta_int(meta, "hidden_size");
  cfg.num_heads = meta_int(meta, "num_heads");
  cfg.num_key_value_heads = meta_int(meta, "num_key_value_heads");
  cfg.head_dim = meta_int(meta, "head_dim");
  cfg.scale = meta_float(meta, "scale");

  auto load = [&](const std::string& n) { return mx::load(dir + "/" + n + ".npy"); };
  NemoAttentionWeights w{load("common/q_w"), load("common/k_w"),
                         load("common/v_w"), load("common/o_w")};

  bool ok = true;
  const std::vector<std::string> cases = {"decode", "prompt", "multiblock"};
  for (const auto& cname : cases) {
    const std::string cd = dir + "/" + cname;
    const std::string cmeta = meta_text(cd + "/meta.json");
    const int steps = meta_int(cmeta, "steps");
    const int expected_blocks = meta_int(cmeta, "blocks");
    auto cload = [&](const std::string& n) { return mx::load(cd + "/" + n + ".npy"); };

    NemoInt8Cache cache;
    cache.block_tokens = meta_int(cmeta, "block_tokens");
    for (int s = 0; s < steps; ++s) {
      mx::array x = cload("x_" + std::to_string(s));
      NemoAttentionDebug dbg;
      mx::array got = nemo_attention_int8_forward(x, w, cfg, cache, &dbg);
      mx::eval(got, *dbg.q, *dbg.k, *dbg.v, *dbg.attended);

      const std::vector<Check> checks = {
          {"output",   got,           cload("expected_" + std::to_string(s))},
          {"q",        *dbg.q,        cload("attn_q_" + std::to_string(s))},
          {"k",        *dbg.k,        cload("attn_k_" + std::to_string(s))},
          {"v",        *dbg.v,        cload("attn_v_" + std::to_string(s))},
          {"attended", *dbg.attended, cload("attended_" + std::to_string(s))},
      };
      for (const auto& c : checks) {
        if (!bit_equal(c.got, c.exp)) {
          ok = false;
          std::fprintf(stderr, "[%s step %d] %s: MISMATCH\n", cname.c_str(), s, c.name);
          dump_first_diff(c.name, c.got, c.exp);
        }
      }
    }

    // Final packed INT8 cache state for every block: packed/scales/biases for
    // K then V, exact layout of QuantizedKVCache.state / AttentionCache.state.
    if (static_cast<int>(cache.blocks.size()) != expected_blocks) {
      ok = false;
      std::fprintf(stderr, "[%s] expected %d blocks, got %zu\n",
                   cname.c_str(), expected_blocks, cache.blocks.size());
      continue;
    }
    for (size_t bi = 0; bi < cache.blocks.size(); ++bi) {
      const NemoInt8Block& b = cache.blocks[bi];
      const std::vector<Check> cache_checks = {
          {"packed_k", b.packed_k, cload("cache_block_state_" + std::to_string(bi * 6 + 0))},
          {"scales_k", b.scales_k, cload("cache_block_state_" + std::to_string(bi * 6 + 1))},
          {"biases_k", b.biases_k, cload("cache_block_state_" + std::to_string(bi * 6 + 2))},
          {"packed_v", b.packed_v, cload("cache_block_state_" + std::to_string(bi * 6 + 3))},
          {"scales_v", b.scales_v, cload("cache_block_state_" + std::to_string(bi * 6 + 4))},
          {"biases_v", b.biases_v, cload("cache_block_state_" + std::to_string(bi * 6 + 5))},
      };
      for (const auto& c : cache_checks) {
        if (!bit_equal(c.got, c.exp)) {
          ok = false;
          std::fprintf(stderr, "[%s cache block %zu] %s: MISMATCH\n",
                       cname.c_str(), bi, c.name);
          dump_first_diff(c.name, c.got, c.exp);
        }
      }
    }
  }

  if (ok) {
    std::printf("nemo_attention_int8: BIT-EXACT PASS "
                "(decode + prompt single-block + multiblock loop cases; "
                "output + q/k/v + attended + packed cache state)\n");
    return 0;
  }
  std::fprintf(stderr, "nemo_attention_int8: FAIL\n");
  return 1;
}
