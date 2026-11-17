// Standalone C++ parity test for veystrix_native_generate.
//
// Runs the SAME in-process greedy loop as the executable (weights from disk,
// zeroed caches, run_decoder_step + sample_greedy, token fed back) and compares
// every step's logits BIT-EXACTLY to expected_logits_step{k}.npy plus the
// generated token ids to expected_token_ids.npy, both produced by
// tools/gen_native_generate_vectors.py from the real mlx_vlm modules.
//
// Build + run (from repo root):
//   MLXROOT=.venv/lib/python3.12/site-packages/mlx && \
//   xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
//     cpp/engine/src/engine.cpp cpp/engine/src/gdn_kernel.cpp \
//     cpp/engine/src/gdn_conv.cpp cpp/engine/src/gdn_layer.cpp \
//     cpp/engine/src/qsa.cpp cpp/engine/src/qsa_full.cpp \
//     cpp/engine/src/decoder.cpp cpp/engine/src/decoder_full.cpp \
//     cpp/engine/tests/test_native_generate.cpp \
//     -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$(pwd)/$MLXROOT/lib" \
//     -o /tmp/test_native_generate && \
//   /tmp/test_native_generate cpp/engine/tests/vectors_native_generate
#include "../src/decoder.hpp"

#include <mlx/mlx.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
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

static std::map<std::string, std::string> read_config(const std::string& p) {
  std::ifstream f(p);
  std::map<std::string, std::string> c;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto eq = line.find('=');
    if (eq != std::string::npos) c[line.substr(0, eq)] = line.substr(eq + 1);
  }
  return c;
}
static int gi(const std::map<std::string, std::string>& c, const std::string& k,
              int d = 0) {
  auto it = c.find(k);
  return it == c.end() ? d : std::stoi(it->second);
}
static float gf(const std::map<std::string, std::string>& c, const std::string& k,
                float d = 1e-6f) {
  auto it = c.find(k);
  return it == c.end() ? d : std::stof(it->second);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <vector-dir>\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  const std::string wdir = dir + "/weights";
  auto load = [&](const std::string& n) { return mx::load(wdir + "/" + n + ".npy"); };
  const auto cfg = read_config(dir + "/config.txt");

  const int n_layers = gi(cfg, "n_layers");
  const int H = gi(cfg, "hidden_size");
  const int hc = gi(cfg, "hc_count");
  const int low = gi(cfg, "hc_lowrank");
  const int vocab = gi(cfg, "vocab");
  const int top_k = gi(cfg, "top_k");
  const int group = gi(cfg, "group_size", 32);
  const int bits = gi(cfg, "bits", 4);
  const float eps = gf(cfg, "eps");
  const int prompt_token = gi(cfg, "prompt_token", 0);

  DecoderConfig dcfg;
  dcfg.hidden_size = H; dcfg.hc_count = hc; dcfg.hc_lowrank = low;
  dcfg.n_layers = n_layers; dcfg.vocab = vocab; dcfg.eps = eps;
  dcfg.gdn = GDNConfig{H, gi(cfg, "key_dim"), gi(cfg, "value_dim"),
                       gi(cfg, "num_k_heads"), gi(cfg, "num_v_heads"),
                       gi(cfg, "head_k_dim"), gi(cfg, "head_v_dim"),
                       gi(cfg, "conv_kernel"), eps, GDNActivation::Sigmoid};

  DecoderHeadWeights head(load("mix_norm_w"), load("mix_down_w"),
                          load("mix_up_w"), load("lm_head_w"));
  mx::array embed_w = load("embed_w");

  std::vector<DecoderLayerWeights> layers;
  std::vector<DecoderLayerCache> caches;
  const int conv_dim = 2 * dcfg.gdn.key_dim + dcfg.gdn.value_dim;
  for (int l = 0; l < n_layers; ++l) {
    const std::string P = "L" + std::to_string(l) + "_";
    layers.emplace_back(
        load(P + "attn_norm_w"), load(P + "attn_down_w"),
        load(P + "attn_up_w"), load(P + "attn_inject_w"),
        GDNWeights(load(P + "gdn_qkv"), load(P + "gdn_z"), load(P + "gdn_b"),
                   load(P + "gdn_a"), load(P + "gdn_conv"), load(P + "gdn_A_log"),
                   load(P + "gdn_dt_bias"), load(P + "gdn_norm"), load(P + "gdn_out")),
        load(P + "mlp_norm_w"), load(P + "mlp_down_w"), load(P + "mlp_up_w"),
        load(P + "mlp_inject_w"), load(P + "gate_w"),
        Q4Projection(load(P + "expert_gw"), load(P + "expert_gs"), load(P + "expert_gb")),
        Q4Projection(load(P + "expert_uw"), load(P + "expert_us"), load(P + "expert_ub")),
        Q4Projection(load(P + "expert_dw"), load(P + "expert_ds"), load(P + "expert_db")),
        top_k, 0.0f, group, bits);
    caches.emplace_back(GDNCacheState(
        mx::zeros(mx::Shape{1, dcfg.gdn.conv_kernel - 1, conv_dim}, mx::float32),
        mx::zeros(mx::Shape{1, dcfg.gdn.num_v_heads, dcfg.gdn.head_v_dim,
                            dcfg.gdn.head_k_dim}, mx::float32)));
  }

  mx::array expected_ids =
      mx::load(dir + "/expected_token_ids.npy");  // [steps+1] int32
  mx::eval(expected_ids);
  const int steps = static_cast<int>(expected_ids.shape(0)) - 1;
  const int32_t* ids = expected_ids.data<int32_t>();

  int token = prompt_token;
  if (ids[0] != token) {
    std::fprintf(stderr, "FAIL: prompt token mismatch\n");
    return 1;
  }
  for (int step = 0; step < steps; ++step) {
    std::vector<int32_t> td{static_cast<int32_t>(token)};
    mx::array tokens(td.begin(), mx::Shape{1, 1}, mx::int32);
    mx::array logits = run_decoder_step_from_tokens(
        tokens, embed_w, layers, head, dcfg, caches);
    mx::array exp_logits = mx::load(
        dir + "/expected_logits_step" + std::to_string(step) + ".npy");
    // Generator saved logits.reshape(-1).astype(float32).
    logits = mx::reshape(mx::astype(logits, mx::float32), mx::Shape{-1});
    if (!bit_equal(logits, exp_logits)) {
      std::fprintf(stderr, "FAIL: step %d logits differ\n", step);
      return 1;
    }
    mx::array next = sample_greedy(logits);
    mx::eval(next);
    token = static_cast<int>(next.data<uint32_t>()[0]);
    if (token != ids[step + 1]) {
      std::fprintf(stderr, "FAIL: step %d token id %d != expected %d\n",
                   step, token, ids[step + 1]);
      return 1;
    }
  }
  std::printf("test_native_generate OK (bit-exact, %d steps, %d layers)\n",
              steps, n_layers);
  return 0;
}
