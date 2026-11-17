// Synthetic 2-shard cross-shard MoE test. Usage:
//   test_crossshard_moe <checkpoint-dir>
// The checkpoint comes from tools/gen_crossshard_checkpoint.py. It contains a
// layer whose 9 expert tensors are split across two shards, with one shard-B
// tensor placed at an absolute offset that triggers the old offset-derived-miss
// bug. Each step's output SHA must equal an all-experts-resident reference;
// the test prints every step SHA so a wrapper can compare two processes.
#include "cross_shard_moe.hpp"

#include <mlx/mlx.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

static int failures = 0;
#define CHECK(cond, msg)                                                    \
  do {                                                                      \
    if (!(cond)) {                                                          \
      std::fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);        \
      ++failures;                                                           \
    }                                                                       \
  } while (0)

// ---- self-contained SHA-256 (same convention as the rest of the engine) ----
namespace {
struct Sha256 {
  uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  static void compress(uint32_t h[8], const uint8_t b[64]) {
    static const uint32_t K[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0cb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(b[4*i])<<24)|(uint32_t(b[4*i+1])<<16)|(uint32_t(b[4*i+2])<<8)|uint32_t(b[4*i+3]);
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
      uint32_t s1 = rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
      w[i] = w[i-16]+s0+w[i-7]+s1;
    }
    uint32_t a=h[0],bb=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t S1=rotr(e,6)^rotr(e,11)^rotr(e,25), ch=(e&f)^((~e)&g), t1=hh+S1+ch+K[i]+w[i];
      uint32_t S0=rotr(a,2)^rotr(a,13)^rotr(a,22), maj=(a&bb)^(a&c)^(bb&c), t2=S0+maj;
      hh=g; g=f; f=e; e=d+t1; d=c; c=bb; bb=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=bb; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
  }
};
std::string sha256_hex(const void* data, size_t len) {
  Sha256 s;
  size_t total = ((len + 9 + 63) / 64) * 64;
  std::vector<uint8_t> buf(total, 0);
  const uint8_t* in = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < len; ++i) buf[i] = in[i];
  buf[len] = 0x80;
  uint64_t bits = uint64_t(len) * 8;
  for (int i = 0; i < 8; ++i) buf[total-1-i] = uint8_t(bits >> (8*i));
  for (size_t i = 0; i < total; i += 64) Sha256::compress(s.h, buf.data()+i);
  static const char* HD = "0123456789abcdef";
  std::string out(64, '0');
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j) {
      out[i*8+j*2] = HD[(s.h[i]>>(28-8*j))&0xf];
      out[i*8+j*2+1] = HD[(s.h[i]>>(24-8*j))&0xf];
    }
  return out;
}
}  // namespace

static std::string json_get(const std::string& s, const std::string& key) {
  std::string pat = "\"" + key + "\"";
  size_t p = s.find(pat);
  if (p == std::string::npos) return "";
  size_t colon = s.find(':', p + pat.size());
  size_t b = s.find_first_not_of(" \t\n\r", colon + 1);
  if (b == std::string::npos) return "";
  if (s[b] == '"') { size_t e = s.find('"', b+1); return s.substr(b+1, e-b-1); }
  size_t e = s.find_first_of(",}\n", b);
  return s.substr(b, e-b);
}

static std::string output_sha(const mx::array& a) {
  mx::array f = mx::reshape(mx::astype(a, mx::float32), mx::Shape{-1});
  mx::array u = mx::view(f, mx::uint32);
  mx::eval(u);
  return sha256_hex(u.data<uint32_t>(), static_cast<size_t>(u.size()) * sizeof(uint32_t));
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <checkpoint-dir>\n", argv[0]);
    return 2;
  }
  std::string dir = argv[1];
  std::ifstream mf(dir + "/meta.json");
  if (!mf) { std::fprintf(stderr, "meta.json missing\n"); return 2; }
  std::stringstream ms; ms << mf.rdbuf();
  std::string meta = ms.str();
  int E = std::stoi(json_get(meta, "E"));
  int HIDDEN = std::stoi(json_get(meta, "HIDDEN"));
  int INTER = std::stoi(json_get(meta, "INTER"));
  int GROUP = std::stoi(json_get(meta, "GROUP"));
  int BITS = std::stoi(json_get(meta, "BITS"));
  std::string PREFIX = json_get(meta, "PREFIX");

  const char* names[9] = {
      "gate_proj.weight","gate_proj.scales","gate_proj.biases",
      "up_proj.weight","up_proj.scales","up_proj.biases",
      "down_proj.weight","down_proj.scales","down_proj.biases"};

  std::ifstream rf(dir + "/all_tensors.bin", std::ios::binary);
  if (!rf) { std::fprintf(stderr, "all_tensors.bin missing\n"); return 2; }
  std::vector<uint8_t> raw((std::istreambuf_iterator<char>(rf)),
                           std::istreambuf_iterator<char>());

  std::vector<int> rows(9);
  for (int i = 0; i < 9; ++i) {
    int64_t n = 1;
    int per = (i % 3 == 0) ? 4 : 2;
    int dim2 = (i < 6) ? HIDDEN : INTER;
    // weight: dim2/8 elements; scales/biases: dim2/GROUP elements
    int64_t cols = (i % 3 == 0) ? dim2 / 8 : dim2 / GROUP;
    n = static_cast<int64_t>(E) * dim2 * cols;
    rows[i] = static_cast<int>(n * per);
  }

  auto make_arrays = [&](int stride_experts, size_t base_off) {
    std::vector<mx::array> a;
    a.reserve(9);
    size_t off = base_off;
    for (int i = 0; i < 9; ++i) {
      int dim1 = (i < 6) ? INTER : HIDDEN;
      int64_t cols = (i % 3 == 0) ? ((i < 6 ? HIDDEN : INTER) / 8)
                                  : ((i < 6 ? HIDDEN : INTER) / GROUP);
      mx::Shape shape{stride_experts, dim1, static_cast<int>(cols)};
      if (i % 3 == 0) {
        a.emplace_back(reinterpret_cast<const uint32_t*>(raw.data() + off),
                       shape, mx::uint32);
      } else {
        mx::array u16(reinterpret_cast<const uint16_t*>(raw.data() + off),
                      shape, mx::uint16);
        a.emplace_back(mx::view(u16, mx::bfloat16));
      }
      off += static_cast<size_t>(rows[i]);
    }
    return a;
  };

  // All-experts-resident reference weights (num_experts = E).
  auto ref = make_arrays(E, 0);
  ExpertWeights full(Q4Projection(ref[0],ref[1],ref[2]),
                     Q4Projection(ref[3],ref[4],ref[5]),
                     Q4Projection(ref[6],ref[7],ref[8]), GROUP, BITS);

  // Cross-shard pool with 16 slots; one tensor lives in shard B.
  std::map<std::string, std::string> cfg;
  for (int i = 0; i < 9; ++i)
    cfg["shard." + PREFIX + "." + names[i]] =
        (i == 8) ? "shardB.safetensors" : "shardA.safetensors";
  CrossShardMoE cross(dir, cfg, PREFIX, /*num_slots=*/16, GROUP, BITS);

  std::mt19937 rng(0x5EED1234u);
  std::uniform_int_distribution<int> pick(0, E - 1);
  const int K = 4;
  std::vector<float> xd(HIDDEN);
  for (int i = 0; i < HIDDEN; ++i) xd[i] = ((rng() % 2000) - 1000) / 1000.0f;
  mx::array x(xd.data(), mx::Shape{1, HIDDEN}, mx::float32);

  for (int step = 0; step < 40; ++step) {
    std::vector<int32_t> ids(K);
    std::vector<float> sc(K);
    float sum = 0;
    for (int j = 0; j < K; ++j) {
      ids[j] = pick(rng);
      sc[j] = 0.1f + (rng() % 100) / 1000.0f;
      sum += sc[j];
    }
    for (int j = 0; j < K; ++j) sc[j] /= sum;
    mx::array route(ids.data(), mx::Shape{1, K}, mx::int32);
    mx::array scores(sc.data(), mx::Shape{1, K}, mx::float32);

    mx::array expected = moe_block(x, route, scores, full);
    StreamForwardResult res = cross.forward(x, route, scores);
    std::string se = output_sha(expected), sr = output_sha(res.output);
    CHECK(se == sr, "cross-shard output must match all-resident reference");
    if (se != sr)
      std::fprintf(stderr, "  step %d: expected %s got %s\n", step,
                   se.c_str(), sr.c_str());
    std::printf("step=%d sha=%s\n", step, sr.c_str());
    if (failures) break;
  }

  if (failures) { std::fprintf(stderr, "test_crossshard_moe FAILED\n"); return 1; }
  std::printf("test_crossshard_moe OK (all 40 steps bit-exact)\n");
  return 0;
}
