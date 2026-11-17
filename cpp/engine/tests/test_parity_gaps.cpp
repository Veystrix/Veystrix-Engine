// Bit-exact parity test for the two remaining gaps (parity_gaps.cpp):
//   1. quantized shared expert (q4_projection gate/up/down + swiglu + down)
//   2. INDEXER-NONE dense full-attention fallback (no sparse QSA mask)
//
// Vectors are produced by the REAL mlx_vlm modules via
//   tools/gen_parity_gaps_vectors.py  (self-check: real == primitive).
// All comparisons reinterpret floats as raw uint32 bits.
//
//   cd /Users/vincent/Documents/Veystrix && MLXROOT=.venv/lib/python3.12/site-packages/mlx && \
//   xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
//     cpp/engine/src/engine.cpp cpp/engine/src/qsa.cpp cpp/engine/src/qsa_full.cpp \
//     cpp/engine/src/parity_gaps.cpp cpp/engine/tests/test_parity_gaps.cpp \
//     -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$(pwd)/$MLXROOT/lib" -o /tmp/test_pg && \
//   /tmp/test_pg cpp/engine/tests/vectors_parity_gaps
#include "../src/parity_gaps.hpp"

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

static void dump_first_diff(const char* label, const mx::array& a,
                            const mx::array& b) {
  mx::array av = mx::view(mx::reshape(a, {-1}), mx::uint32);
  mx::array bv = mx::view(mx::reshape(b, {-1}), mx::uint32);
  mx::eval(av, bv);
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  for (size_t i = 0; i < av.size(); ++i) {
    if (pa[i] != pb[i]) {
      std::fprintf(stderr, "  %s first diff at %zu: got 0x%08x expected 0x%08x\n",
                   label, i, pa[i], pb[i]);
      return;
    }
  }
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
static std::vector<int> meta_int_array(const std::string& txt,
                                       const std::string& key) {
  std::vector<int> out;
  auto p = txt.find("\"" + key + "\": [");
  if (p == std::string::npos) return out;
  p += key.size() + 5;
  while (p < txt.size() && txt[p] != ']') {
    if (txt[p] >= '0' && txt[p] <= '9') {
      out.push_back(std::stoi(txt.substr(p)));
      while (p < txt.size() && txt[p] != ',' && txt[p] != ']') ++p;
    }
    ++p;
  }
  return out;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <vectordir>\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  auto load = [&](const std::string& n) { return mx::load(dir + "/" + n + ".npy"); };
  const std::string meta = meta_text(dir);

  const int group_size = meta_int(meta, "group_size");
  const int bits = meta_int(meta, "bits");
  const float eps = meta_float(meta, "eps");
  bool ok = true;

  // -------------------------------------------------------------------------
  // Gap 1: quantized shared expert.
  // -------------------------------------------------------------------------
  {
    mx::array x = load("g_x");
    mx::array route = load("g_routes");
    mx::array scores = load("g_scores");
    mx::array shared_gate_w = load("g_shared_gate_w");
    mx::array expected = load("g_expected");

    ExpertWeights routed(
        Q4Projection(load("g_routed_gw"), load("g_routed_gs"), load("g_routed_gb")),
        Q4Projection(load("g_routed_uw"), load("g_routed_us"), load("g_routed_ub")),
        Q4Projection(load("g_routed_dw"), load("g_routed_ds"), load("g_routed_db")),
        group_size, bits);
    ExpertWeights shared(
        Q4Projection(load("g_shared_gw"), load("g_shared_gs"), load("g_shared_gb")),
        Q4Projection(load("g_shared_uw"), load("g_shared_us"), load("g_shared_ub")),
        Q4Projection(load("g_shared_dw"), load("g_shared_ds"), load("g_shared_db")),
        group_size, bits);

    SharedQuantConfig cfg;
    cfg.group_size = group_size;
    cfg.bits = bits;
    cfg.shared_quantized = true;

    mx::array out = moe_block_shared_quantized(
        x, route, scores, routed, shared, shared_gate_w, cfg);
    mx::eval(out);
    if (!bit_equal(out, expected)) {
      std::fprintf(stderr, "FAIL gap1: quantized shared expert output differs\n");
      dump_first_diff("gap1", out, expected);
      ok = false;
    }
  }

  // -------------------------------------------------------------------------
  // Gap 2: INDEXER-NONE dense full-attention fallback.
  // -------------------------------------------------------------------------
  {
    mx::array x = load("fa_x");
    mx::array position_ids = load("fa_pos");
    mx::array expected = load("fa_expected");

    FullAttentionWeights w(
        load("fa_q_w"), load("fa_k_w"), load("fa_v_w"), load("fa_o_w"),
        load("fa_q_norm_w"), load("fa_k_norm_w"),
        load("fa_index_qk_w"), load("fa_index_q_norm_w"),
        load("fa_index_k_norm_w"));

    FullAttentionConfig cfg;
    cfg.n_heads = meta_int(meta, "n_heads");
    cfg.kv_heads = meta_int(meta, "kv_heads");
    cfg.head_dim = meta_int(meta, "head_dim");
    cfg.n_idx_heads = meta_int(meta, "n_idx_heads");
    cfg.n_idx_kv = meta_int(meta, "n_idx_kv");
    cfg.idx_head_dim = meta_int(meta, "idx_head_dim");
    cfg.compress_ratio = meta_int(meta, "compress_ratio");
    cfg.token_budget = meta_int(meta, "token_budget");
    cfg.eps = eps;
    cfg.rope.dim = meta_int(meta, "rope_dim");
    cfg.rope.base = meta_float(meta, "rope_base");
    cfg.rope.mrope_section = meta_int_array(meta, "mrope_section");
    cfg.rope.style = "interleaved";
    cfg.rope.attention_scaling = 1.0f;

    FullAttentionCache cache;
    cache.keys = load("fa_cache_keys");
    cache.values = load("fa_cache_values");
    cache.indexer.index_keys = load("fa_index_keys");
    cache.indexer.index_position_ids = load("fa_index_pos");
    cache.offset = meta_int(meta, "past_len");
    cache.indexer.offset = cache.offset;

    mx::array out = full_attention_dense_fallback(x, w, cfg, cache, position_ids);
    mx::eval(out);
    if (!bit_equal(out, expected)) {
      std::fprintf(stderr, "FAIL gap2: indexer-None dense attention differs\n");
      dump_first_diff("gap2", out, expected);
      ok = false;
    }
  }

  if (!ok) return 1;
  std::printf("test_parity_gaps OK\n");
  return 0;
}
