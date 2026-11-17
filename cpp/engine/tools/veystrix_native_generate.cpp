// veystrix_native_generate -- STANDALONE C++ greedy token generation.
//
// Decisive "the engine works" evidence: a real Qwen4-Exp (tiny, all-linear)
// model's weights are loaded from disk as npy via mx::load, a flat config file
// is parsed into DecoderConfig / GDNConfig, one zeroed cache is initialized per
// layer, and for N tokens the full decoder loop (run_decoder_step) runs in C++:
//   token id -> embedding -> hyper-connections -> GDN -> routed MoE -> lm_head
//   -> sample_greedy -> next token id (fed back for the next step).
// No Python runs in the generation loop.  Per-step logit SHA-256 uses the same
// float32-little-endian convention as models/shared/logit_reference.py.
//
// Build + run (from repo root):
//   MLXROOT=.venv/lib/python3.12/site-packages/mlx && \
//   xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
//     cpp/engine/src/engine.cpp cpp/engine/src/gdn_kernel.cpp \
//     cpp/engine/src/gdn_conv.cpp cpp/engine/src/gdn_layer.cpp \
//     cpp/engine/src/qsa.cpp cpp/engine/src/qsa_full.cpp \
//     cpp/engine/src/decoder.cpp cpp/engine/src/decoder_full.cpp \
//     cpp/engine/tools/veystrix_native_generate.cpp \
//     -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$(pwd)/$MLXROOT/lib" \
//     -o /tmp/native_generate && \
//   /tmp/native_generate <weights-dir> <config-file> <num_tokens>
//
// Output (one line per generated token, easy to diff with the Python ground
// truth produced by gen_native_generate_vectors.py):
//   step=<i> in=<token_in> out=<token_out> sha256=<64 hex>
#include "../src/decoder.hpp"

#include <mlx/mlx.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

// --------------------------------------------------------------------------- //
// Self-contained SHA-256 (no OpenSSL/CommonCrypto linking needed).            //
// --------------------------------------------------------------------------- //
namespace {

struct Sha256 {
  uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

  static uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
  }

  static void compress(uint32_t h[8], const uint8_t block[64]) {
    static const uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t(block[4 * i]) << 24) |
             (uint32_t(block[4 * i + 1]) << 16) |
             (uint32_t(block[4 * i + 2]) << 8) | uint32_t(block[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      uint32_t ch = (e & f) ^ ((~e) & g);
      uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = S0 + maj;
      hh = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
};

std::string sha256_hex(const void* data, size_t len) {
  Sha256 s;
  const size_t total = ((len + 9 + 63) / 64) * 64;
  std::vector<uint8_t> buf(total, 0);
  const uint8_t* in = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < len; ++i) buf[i] = in[i];
  buf[len] = 0x80;
  const uint64_t bits = uint64_t(len) * 8;
  for (int i = 0; i < 8; ++i) buf[total - 1 - i] = uint8_t(bits >> (8 * i));
  for (size_t i = 0; i < total; i += 64) Sha256::compress(s.h, buf.data() + i);
  static const char* HD = "0123456789abcdef";
  std::string out(64, '0');
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 4; ++j) {
      out[i * 8 + j * 2] = HD[(s.h[i] >> (28 - 8 * j)) & 0xf];
      out[i * 8 + j * 2 + 1] = HD[(s.h[i] >> (24 - 8 * j)) & 0xf];
    }
  }
  return out;
}

// --------------------------------------------------------------------------- //
// Flat key=value config parsing                                               //
// --------------------------------------------------------------------------- //
using Config = std::map<std::string, std::string>;

Config read_config(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open config file: " + path);
  Config cfg;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = line.substr(0, eq);
    std::string val = line.substr(eq + 1);
    while (!val.empty() && (val.back() == '\r' || val.back() == '\n')) val.pop_back();
    cfg[key] = val;
  }
  return cfg;
}

int get_int(const Config& c, const std::string& key, int def = 0) {
  auto it = c.find(key);
  if (it == c.end()) return def;
  return std::stoi(it->second);
}

float get_float(const Config& c, const std::string& key, float def = 1e-6f) {
  auto it = c.find(key);
  if (it == c.end()) return def;
  return std::stof(it->second);
}

// --------------------------------------------------------------------------- //
// Generation                                                                  //
// --------------------------------------------------------------------------- //
struct StepRecord {
  int step;
  int token_in;
  int token_out;
  std::string digest;
};

std::vector<StepRecord> run_generation(const std::string& wdir,
                                       const std::string& config_path,
                                       int num_tokens) {
  const Config cfg = read_config(config_path);
  auto load = [&](const std::string& name) {
    return mx::load(wdir + "/" + name + ".npy");
  };

  const int n_layers = get_int(cfg, "n_layers");
  const int H = get_int(cfg, "hidden_size");
  const int hc_count = get_int(cfg, "hc_count");
  const int hc_lowrank = get_int(cfg, "hc_lowrank");
  const int vocab = get_int(cfg, "vocab");
  const int top_k = get_int(cfg, "top_k");
  const int group_size = get_int(cfg, "group_size", 32);
  const int bits = get_int(cfg, "bits", 4);
  const float eps = get_float(cfg, "eps", 1e-6f);
  const int prompt_token = get_int(cfg, "prompt_token", 0);

  DecoderConfig dcfg;
  dcfg.hidden_size = H;
  dcfg.hc_count = hc_count;
  dcfg.hc_lowrank = hc_lowrank;
  dcfg.n_layers = n_layers;
  dcfg.vocab = vocab;
  dcfg.eps = eps;
  dcfg.gdn = GDNConfig{
      H,
      get_int(cfg, "key_dim"),
      get_int(cfg, "value_dim"),
      get_int(cfg, "num_k_heads"),
      get_int(cfg, "num_v_heads"),
      get_int(cfg, "head_k_dim"),
      get_int(cfg, "head_v_dim"),
      get_int(cfg, "conv_kernel"),
      eps,
      GDNActivation::Sigmoid};

  // Head (final mixer + lm_head).
  DecoderHeadWeights head(load("mix_norm_w"), load("mix_down_w"),
                          load("mix_up_w"), load("lm_head_w"));
  mx::array embed_w = load("embed_w");

  // Per-layer weights + one ZEROED cache per layer.
  std::vector<DecoderLayerWeights> layers;
  std::vector<DecoderLayerCache> caches;
  layers.reserve(n_layers);
  caches.reserve(n_layers);
  const int conv_dim = 2 * dcfg.gdn.key_dim + dcfg.gdn.value_dim;
  for (int l = 0; l < n_layers; ++l) {
    const std::string P = "L" + std::to_string(l) + "_";
    layers.emplace_back(
        load(P + "attn_norm_w"), load(P + "attn_down_w"),
        load(P + "attn_up_w"), load(P + "attn_inject_w"),
        GDNWeights(load(P + "gdn_qkv"), load(P + "gdn_z"),
                   load(P + "gdn_b"), load(P + "gdn_a"),
                   load(P + "gdn_conv"), load(P + "gdn_A_log"),
                   load(P + "gdn_dt_bias"), load(P + "gdn_norm"),
                   load(P + "gdn_out")),
        load(P + "mlp_norm_w"), load(P + "mlp_down_w"),
        load(P + "mlp_up_w"), load(P + "mlp_inject_w"),
        load(P + "gate_w"),
        Q4Projection(load(P + "expert_gw"), load(P + "expert_gs"),
                     load(P + "expert_gb")),
        Q4Projection(load(P + "expert_uw"), load(P + "expert_us"),
                     load(P + "expert_ub")),
        Q4Projection(load(P + "expert_dw"), load(P + "expert_ds"),
                     load(P + "expert_db")),
        top_k, 0.0f, group_size, bits);

    GDNCacheState zero_cache(
        mx::zeros(mx::Shape{1, dcfg.gdn.conv_kernel - 1, conv_dim}, mx::float32),
        mx::zeros(mx::Shape{1, dcfg.gdn.num_v_heads, dcfg.gdn.head_v_dim,
                            dcfg.gdn.head_k_dim}, mx::float32));
    caches.emplace_back(std::move(zero_cache));
  }

  std::vector<StepRecord> records;
  int token = prompt_token;
  for (int step = 0; step < num_tokens; ++step) {
    std::vector<int32_t> tok_data{static_cast<int32_t>(token)};
    mx::array tokens(tok_data.begin(), mx::Shape{1, 1}, mx::int32);
    mx::array logits = run_decoder_step_from_tokens(
        tokens, embed_w, layers, head, dcfg, caches);

    // sample_greedy: argmax over the flattened logits -> uint32 token id.
    mx::array next = sample_greedy(logits);
    mx::eval(next);
    uint32_t next_id = next.data<uint32_t>()[0];

    // Digest over the full logit row, cast to float32 little-endian (matches
    // logit_reference.py / logit_sha256 in the generator).
    mx::array row = mx::reshape(mx::astype(logits, mx::float32), mx::Shape{-1});
    mx::eval(row);
    const float* fdata = row.data<float>();
    std::string digest = sha256_hex(
        fdata, static_cast<size_t>(row.size()) * sizeof(float));

    records.push_back({step, token, static_cast<int>(next_id), digest});
    token = static_cast<int>(next_id);
  }
  return records;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <weights-dir> <config-file> [<num_tokens>]\n",
                 argv[0]);
    return 2;
  }
  const std::string weights_dir = argv[1];
  const std::string config_file = argv[2];
  const int num_tokens = (argc >= 4) ? std::atoi(argv[3]) : 1;
  try {
    std::vector<StepRecord> records =
        run_generation(weights_dir, config_file, num_tokens);
    for (const auto& r : records) {
      std::printf("step=%d in=%d out=%d sha256=%s\n",
                  r.step, r.token_in, r.token_out, r.digest.c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "native_generate failed: %s\n", e.what());
    return 1;
  }
}
