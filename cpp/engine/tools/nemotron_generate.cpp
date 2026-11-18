// nemotron_generate -- standalone C++ greedy decode from the REAL
// NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit checkpoint, no Python in the loop.
//
// The 52-layer loop is composed ONLY from already-validated bit-exact
// primitives:
//   * mx::fast::rms_norm          (nn.RMSNorm block / final norm)
//   * mamba_layer_forward()       (S=1 Mamba-2 layer, mamba_layer.hpp)
//   * nemo_attention_int8_forward() (INT8 affine block KV S=1 attention, default)
//   * nemo_attention_forward()    (bf16 KV S=1 attention, VEYSTRIX_NEMOTRON_KV=bf16)
//   * moe_block_relu2()           (top-6 of 128 relu2 Q4 G64 experts)
//   * linear()                    (QPacked G64 dense / fused quantized_matmul)
//   * sample_greedy()             (greedy argmax)
// The routed experts are streamed row-by-row straight out of the safetensors
// shards (only the selected experts are materialized per MoE step), so the
// ~15 GB expert stack is never held resident.
//
// Per-step logit SHA-256 uses the float32-little-endian convention of
// capture_real_reference.py / gen_nemotron_reference.py.
//
// USAGE
//   nemotron_generate <checkpoint-dir> <reference.json> <num_tokens> [flat-config]
// The 4th argument defaults to <reference-dir>/nemotron_config.txt, produced
// by gen_nemotron_config.py. Output (stdout), one line per generated token:
//   step=<i> in=<token_in> out=<token_out> sha256=<64 hex>
#include "engine.hpp"
#include "mamba_layer.hpp"
#include "nemo_attention.hpp"
#include "model_adapter.hpp"

#include <mlx/mlx.h>
#include <mlx/memory.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

// --------------------------------------------------------------------------- //
// Self-contained SHA-256 (identical to real_checkpoint_generate.cpp).           //
// --------------------------------------------------------------------------- //
namespace {

struct Sha256 {
  uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
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
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(block[4 * i]) << 24) | (uint32_t(block[4 * i + 1]) << 16) |
             (uint32_t(block[4 * i + 2]) << 8) | uint32_t(block[4 * i + 3]);
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
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j) {
      out[i * 8 + j * 2] = HD[(s.h[i] >> (28 - 8 * j)) & 0xf];
      out[i * 8 + j * 2 + 1] = HD[(s.h[i] >> (24 - 8 * j)) & 0xf];
    }
  return out;
}

std::string digest_array(const mx::array& a) {
  mx::array f = mx::reshape(mx::astype(a, mx::float32), mx::Shape{-1});
  mx::eval(f);
  return sha256_hex(f.data<float>(), static_cast<size_t>(f.size()) * sizeof(float));
}

// mx::array has no default constructor; use an empty f32 array as a placeholder
// for members that are assigned before first use.
mx::array empty_f32() {
  return mx::array(std::initializer_list<int>{}, mx::float32);
}

static void mem_telemetry(const char* where) {
  const double GIB = 1024.0 * 1024.0 * 1024.0;
  std::fprintf(stderr, "[mem %s] active=%.2f GiB cache=%.2f GiB peak=%.2f GiB\n",
               where, mx::get_active_memory() / GIB, mx::get_cache_memory() / GIB,
               mx::get_peak_memory() / GIB);
}

// --------------------------------------------------------------------------- //
// Flat config (produced by gen_nemotron_config.py).                              //
// --------------------------------------------------------------------------- //
using Config = std::map<std::string, std::string>;

Config read_config(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open flat config: " + path);
  Config cfg;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string val = line.substr(eq + 1);
    while (!val.empty() && (val.back() == '\r' || val.back() == '\n')) val.pop_back();
    cfg[line.substr(0, eq)] = val;
  }
  return cfg;
}

int get_int(const Config& c, const std::string& key, int def = 0) {
  auto it = c.find(key);
  return it == c.end() ? def : std::stoi(it->second);
}
float get_float(const Config& c, const std::string& key, float def = 0.f) {
  auto it = c.find(key);
  return it == c.end() ? def : std::stof(it->second);
}
std::string get_str(const Config& c, const std::string& key,
                    const std::string& def = "") {
  auto it = c.find(key);
  return it == c.end() ? def : it->second;
}
std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep)) {
    while (!item.empty() && item.back() == '\r') item.pop_back();
    if (!item.empty()) out.push_back(item);
  }
  return out;
}
std::vector<int> split_ints(const std::string& s) {
  std::vector<int> out;
  for (const auto& x : split(s, ',')) out.push_back(std::stoi(x));
  return out;
}

// --------------------------------------------------------------------------- //
// Dense weight loader (adapted from real_checkpoint_generate.cpp DenseLoader).  //
// --------------------------------------------------------------------------- //
class DenseLoader {
 public:
  DenseLoader(const std::string& dir, const Config& cfg, int group_size, int bits)
      : dir_(dir), group_size_(group_size), bits_(bits) {
    for (const auto& [key, shard] : cfg)
      if (key.rfind("shard.", 0) == 0) shard_of_[key.substr(6)] = shard;
  }

  mx::array raw(const std::string& name) {
    auto it = shard_of_.find(name);
    if (it == shard_of_.end())
      throw std::runtime_error("tensor not in shard map: " + name);
    const std::string& shard = it->second;
    auto sit = shards_.find(shard);
    if (sit == shards_.end()) {
      auto [weights, meta] = mx::load_safetensors(dir_ + "/" + shard);
      sit = shards_.emplace(shard, std::move(weights)).first;
      std::fprintf(stderr, "[load] shard %s (%zu tensors)\n", shard.c_str(),
                   sit->second.size());
    }
    auto tit = sit->second.find(name);
    if (tit == sit->second.end())
      throw std::runtime_error("tensor missing from shard: " + name);
    mx::eval(tit->second);  // Python evals all params right after load_weights.
    return tit->second;
  }

  mx::array dense(const std::string& name) { return raw(name); }

  Linear linear(const std::string& name, int logical_in) {
    const std::string base =
        name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0
            ? name.substr(0, name.size() - 7)
            : name;
    if (shard_of_.count(base + ".scales") == 0)
      return Linear(raw(name.find(".weight") != std::string::npos
                            ? name
                            : base + ".weight"));
    mx::array w = raw(base + ".weight");
    mx::array s = as_bf16(raw(base + ".scales"));
    mx::array b = as_bf16(raw(base + ".biases"));
    int packed_in = static_cast<int>(w.shape(w.shape().size() - 1));
    int n_groups = static_cast<int>(s.shape(s.shape().size() - 1));
    int bits = bits_, gs = group_size_;
    if (logical_in > 0) {
      if (packed_in * 32 % logical_in != 0 || logical_in % n_groups != 0)
        throw std::runtime_error("linear(): bad shapes for " + base);
      bits = packed_in * 32 / logical_in;
      gs = logical_in / n_groups;
    }
    if (bits < 1 || bits > 8 || gs < 1 ||
        packed_in * 32 != gs * n_groups * bits)
      throw std::runtime_error("linear(): inconsistent gs/bits for " + base);
    return Linear::quantized(std::move(w), std::move(s), std::move(b), gs, bits);
  }

  struct QTriplet { mx::array w, s, b; };
  QTriplet quantized(const std::string& name) {
    return {raw(name + ".weight"), as_bf16(raw(name + ".scales")),
            as_bf16(raw(name + ".biases"))};
  }

 private:
  mx::array as_bf16(mx::array a) const {
    if (static_cast<mx::Dtype::Val>(a.dtype()) == mx::uint16)
      return mx::view(a, mx::bfloat16);
    return a;
  }

  std::string dir_;
  int group_size_, bits_;
  std::map<std::string, std::string> shard_of_;
  std::map<std::string, std::unordered_map<std::string, mx::array>> shards_;
};

// --------------------------------------------------------------------------- //
// Routed-expert row reader: pread ONLY selected rows out of the stacked tensors. //
// --------------------------------------------------------------------------- //
class ExpertReader {
 public:
  ExpertReader(const std::string& dir, const Config& cfg) : dir_(dir), cfg_(cfg) {}

  mx::array rows_u32(const std::string& name, const std::vector<int>& ids) {
    Info inf = info(name);
    if (inf.dtype != "U32")
      throw std::runtime_error("rows_u32: tensor is not U32: " + name);
    const size_t per = static_cast<size_t>(inf.shape[1] * inf.shape[2]);
    std::vector<uint32_t> buf(per * ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
      off_t off = inf.data_offset + static_cast<off_t>(ids[i]) * per * 4;
      if (::pread(inf.fd, buf.data() + i * per, per * 4, off) !=
          static_cast<ssize_t>(per * 4))
        throw std::runtime_error("short expert row read: " + name);
    }
    return mx::array(buf.begin(), mx::Shape{static_cast<int>(ids.size()),
                                            static_cast<int>(inf.shape[1]),
                                            static_cast<int>(inf.shape[2])},
                     mx::uint32);
  }

  mx::array rows_bf16(const std::string& name, const std::vector<int>& ids) {
    Info inf = info(name);
    if (inf.dtype != "BF16")
      throw std::runtime_error("rows_bf16: tensor is not BF16: " + name);
    const size_t per = static_cast<size_t>(inf.shape[1] * inf.shape[2]);
    std::vector<uint16_t> buf(per * ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
      off_t off = inf.data_offset + static_cast<off_t>(ids[i]) * per * 2;
      if (::pread(inf.fd, buf.data() + i * per, per * 2, off) !=
          static_cast<ssize_t>(per * 2))
        throw std::runtime_error("short expert row read: " + name);
    }
    mx::array a(buf.begin(), mx::Shape{static_cast<int>(ids.size()),
                                       static_cast<int>(inf.shape[1]),
                                       static_cast<int>(inf.shape[2])},
                mx::uint16);
    return mx::view(a, mx::bfloat16);
  }

 private:
  struct Info {
    int fd = -1;
    int64_t data_offset = 0;
    std::vector<int64_t> shape;
    std::string dtype;
  };

  Info info(const std::string& name) {
    auto ci = cache_.find(name);
    if (ci != cache_.end()) return ci->second;
    auto sit = cfg_.find("shard." + name);
    if (sit == cfg_.end())
      throw std::runtime_error("missing shard map for expert tensor: " + name);
    std::string path = dir_ + "/" + sit->second;
    int fd = -1;
    auto fi = fds_.find(path);
    if (fi != fds_.end()) fd = fi->second;
    else {
      fd = ::open(path.c_str(), O_RDONLY);
      if (fd < 0) throw std::runtime_error("cannot open shard: " + path);
      fds_[path] = fd;
    }
    unsigned char lenbuf[8];
    if (::pread(fd, lenbuf, 8, 0) != 8)
      throw std::runtime_error("short safetensors header length: " + path);
    uint64_t nn = 0;
    for (int i = 0; i < 8; ++i) nn |= static_cast<uint64_t>(lenbuf[i]) << (8 * i);
    std::string header(nn, '\0');
    if (::pread(fd, header.data(), nn, 8) != static_cast<ssize_t>(nn))
      throw std::runtime_error("short safetensors header: " + path);
    const std::string pat = "\"" + name + "\"";
    size_t p = header.find(pat);
    if (p == std::string::npos)
      throw std::runtime_error("tensor missing from shard header: " + name);
    size_t obj = header.find('{', p), close = header.find('}', obj);
    if (obj == std::string::npos || close == std::string::npos)
      throw std::runtime_error("bad safetensors header entry: " + name);
    std::string b = header.substr(obj, close - obj + 1);
    size_t sp = b.find("\"shape\""), sb = b.find('[', sp), se = b.find(']', sb);
    if (sp == std::string::npos || sb == std::string::npos ||
        se == std::string::npos)
      throw std::runtime_error("bad shape for " + name);
    std::string arr = b.substr(sb + 1, se - sb - 1);
    std::vector<int64_t> shape;
    size_t q = 0;
    while ((q = arr.find_first_of("0123456789-", q)) != std::string::npos) {
      size_t e = q;
      if (arr[e] == '-') ++e;
      while (e < arr.size() && arr[e] >= '0' && arr[e] <= '9') ++e;
      shape.push_back(std::strtoll(arr.substr(q, e - q).c_str(), nullptr, 10));
      q = e;
    }
    if (shape.size() != 3)
      throw std::runtime_error("expected 3-D expert tensor: " + name);
    size_t dp = b.find("\"data_offsets\""), db = b.find('[', dp),
           dc = b.find(',', db);
    if (dp == std::string::npos || db == std::string::npos ||
        dc == std::string::npos)
      throw std::runtime_error("bad data_offsets for " + name);
    int64_t off0 = std::strtoll(b.substr(db + 1, dc - db - 1).c_str(),
                                nullptr, 10);
    size_t yp = b.find("\"dtype\""), yq = b.find('"', yp + 8);
    if (yp == std::string::npos || yq == std::string::npos)
      throw std::runtime_error("bad dtype for " + name);
    size_t ye = b.find('"', yq + 1);
    std::string dtype = b.substr(yq + 1, ye - yq - 1);
    Info inf{fd, static_cast<int64_t>(8 + nn) + off0, shape, dtype};
    cache_.emplace(name, inf);
    return inf;
  }

  std::string dir_;
  const Config& cfg_;
  std::map<std::string, Info> cache_;
  std::map<std::string, int> fds_;
};

// --------------------------------------------------------------------------- //
// Per-layer weights / caches and the whole model state.                          //
// --------------------------------------------------------------------------- //
struct MambaLayer {
  mx::array norm_w;
  MambaWeights weights;
  int cache_idx = -1;
};
struct AttnLayer {
  mx::array norm_w = empty_f32();
  NemoAttentionWeights weights;
  NemoAttentionConfig cfg;
  NemoAttentionCache cache;      // bf16 KV (VEYSTRIX_NEMOTRON_KV=bf16)
  NemoInt8Cache int8_cache;      // INT8 affine block KV (production default)
  int cache_idx = -1;
};
struct MoeLayer {
  mx::array norm_w = empty_f32();
  Linear gate;
  mx::array gate_bias = empty_f32();  // e_score_correction_bias, f32 [n_experts]
  Linear shared_up, shared_down;
  std::string fc1_base, fc2_base;
};

struct StepRecord {
  int step = 0, token_in = 0, token_out = 0;
  std::string digest;
};

struct Model {
  int H = 0, n_layers = 0, top_k = 0, n_experts = 0;
  int group_size = 64, bits = 4;
  float eps = 1e-5f, routed_scale = 2.5f;
  ModelSpec spec;
  std::vector<MambaLayer> mamba_layers;
  std::vector<AttnLayer> attn_layers;
  std::vector<MoeLayer> moe_layers;
  std::vector<MambaCacheState> mamba_caches;
  std::vector<LayerKind> kinds;
  mx::array embed_w = empty_f32(), embed_s = empty_f32(), embed_b = empty_f32();
  mx::array norm_f_w = empty_f32();
  Linear lm_head;
  bool dump_layers = false;
  // Production default is INT8 affine block KV (FORMAT="affine-int8-g64").
  // Opt into the legacy bf16 KV path with VEYSTRIX_NEMOTRON_KV=bf16.
  bool use_int8_kv = true;
};

mx::array embed_token(const Model& m, int token) {
  std::vector<int32_t> ids{static_cast<int32_t>(token)};
  mx::array idx(ids.begin(), mx::Shape{1}, mx::int32);
  mx::array w = mx::take(m.embed_w, idx, 0);
  mx::array s = mx::take(m.embed_s, idx, 0);
  mx::array b = mx::take(m.embed_b, idx, 0);
  mx::array row = mx::dequantize(w, s, b, m.group_size, m.bits, "affine");
  return mx::reshape(row, mx::Shape{1, 1, m.H});
}

// Nemotron group_expert_select (mlx_lm nemotron_h.py), with n_group==1 and
// topk_group==1 so the group-masking branch is skipped.
struct NemotronRoute {
  mx::array indices;  // [1, top_k] int32
  mx::array scores;   // [1, top_k] f32
};
NemotronRoute nemotron_route(const mx::array& x, const MoeLayer& w, int top_k,
                             float routed_scale) {
  // Gate logits are computed outside @mx.compile in Python (MoEGate.__call__),
  // so keep the dense matmul unfused here too. Everything from sigmoid through
  // the score normalization MUST be one compiled graph: Python's
  // group_expert_select is @mx.compile and the fused kernel differs by ~1 ULP
  // from the eager op chain (enough to change bf16 routed-expert outputs).
  mx::array gates = linear(x, w.gate);  // [1,1,E] bf16
  const int last = static_cast<int>(gates.shape().size()) - 1;
  mx::array slots = mx::reshape(mx::arange(0, top_k, 1, mx::int32),
                                mx::Shape{1, 1, top_k});  // static shape, same ndim
  static const float cached_scale = routed_scale;
  static auto fn = mx::compile(
      [last, slots, k = top_k](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array orig = mx::sigmoid(mx::astype(in[0], mx::float32));
        mx::array scores = orig + in[1];
        mx::array inds_all = mx::argpartition(mx::negative(scores), k - 1, last);
        mx::array inds = mx::take_along_axis(inds_all, slots, last);
        mx::array top = mx::take_along_axis(orig, inds, last);
        mx::array denom = mx::sum(top, last, /*keepdims=*/true) + mx::array(1e-20f);
        top = top / denom;
        top = top * mx::array(cached_scale);
        return {inds, top};
      },
      /*shapeless=*/true);
  std::vector<mx::array> out = fn({gates, w.gate_bias});
  return {mx::reshape(out[0], mx::Shape{1, top_k}),
          mx::reshape(out[1], mx::Shape{1, top_k})};
}

// --------------------------------------------------------------------------- //
// Load every dense weight; build caches.                                         //
// --------------------------------------------------------------------------- //
Model load_model(const std::string& ckpt, const Config& cfg) {
  Model m;
  m.H = get_int(cfg, "hidden_size");
  m.n_layers = get_int(cfg, "n_layers");
  m.top_k = get_int(cfg, "top_k");
  m.n_experts = get_int(cfg, "n_experts");
  m.group_size = get_int(cfg, "group_size", 64);
  m.bits = get_int(cfg, "bits", 4);
  m.eps = get_float(cfg, "norm_eps", 1e-5f);
  m.routed_scale = get_float(cfg, "routed_scaling_factor", 2.5f);
  m.dump_layers = std::getenv("NEMOTRON_DUMP_LAYERS") != nullptr;
  // Default to the production INT8 KV path; bf16 is an explicit opt-in.
  const char* kv = std::getenv("VEYSTRIX_NEMOTRON_KV");
  m.use_int8_kv = !(kv && std::string(kv) == "bf16");
  std::fprintf(stderr, "[load] VEYSTRIX_NEMOTRON_KV=%s -> %s KV attention\n",
               kv ? kv : "(unset)", m.use_int8_kv ? "int8" : "bf16");

  std::vector<std::string> layer_types = split(get_str(cfg, "layer_types"), ',');
  if (static_cast<int>(layer_types.size()) != m.n_layers)
    throw std::runtime_error("layer_types size != n_layers");

  m.spec = from_nemotron_config(
      m.H, m.n_layers, layer_types, m.n_experts, m.top_k, m.group_size,
      get_int(cfg, "moe_intermediate"), m.eps,
      get_int(cfg, "mamba_num_heads"), get_int(cfg, "mamba_head_dim"),
      get_int(cfg, "mamba_n_groups"), get_int(cfg, "mamba_ssm_state_size"),
      get_int(cfg, "mamba_conv_kernel"), get_int(cfg, "mamba_intermediate_size"),
      get_int(cfg, "mamba_conv_dim"),
      get_float(cfg, "mamba_time_step_min", 0.f),
      get_float(cfg, "mamba_time_step_max",
                std::numeric_limits<float>::infinity()));

  DenseLoader dl(ckpt, cfg, m.group_size, m.bits);
  ExpertReader er(ckpt, cfg);

  const int mH = get_int(cfg, "mamba_num_heads");
  const int mDh = get_int(cfg, "mamba_head_dim");
  const int mG = get_int(cfg, "mamba_n_groups");
  const int mDs = get_int(cfg, "mamba_ssm_state_size");
  const int mK = get_int(cfg, "mamba_conv_kernel");
  const int mI = get_int(cfg, "mamba_intermediate_size");
  const int mConv = get_int(cfg, "mamba_conv_dim");

  NemoAttentionConfig ac;
  ac.hidden_size = m.H;
  ac.num_heads = get_int(cfg, "n_heads");
  ac.num_key_value_heads = get_int(cfg, "kv_heads");
  ac.head_dim = get_int(cfg, "head_dim");
  ac.scale = get_float(cfg, "attn_scale", 1.f / std::sqrt(float(ac.head_dim)));

  m.kinds.reserve(m.n_layers);
  for (int l = 0; l < m.n_layers; ++l) {
    LayerDispatch d = dispatch_layer(m.spec, l);
    m.kinds.push_back(d.kind);
    const std::string b = "backbone.layers." + std::to_string(l) + ".";
    mx::array norm_w = dl.dense(b + "norm.weight");

    if (d.kind == LayerKind::MAMBA_SSM) {
      const std::string p = b + "mixer.";
      Linear in = dl.linear(p + "in_proj", 0);
      mx::array conv_w = dl.dense(p + "conv1d.weight");
      mx::array conv_b = dl.dense(p + "conv1d.bias");
      // Python loads bf16 checkpoint values into the f32 A_log/dt_bias/D
      // parameters; the fused kernel casts each value to float anyway, so
      // casting here is bit-identical.
      mx::array A_log = mx::astype(dl.dense(p + "A_log"), mx::float32);
      mx::array dt_bias = mx::astype(dl.dense(p + "dt_bias"), mx::float32);
      mx::array D = mx::astype(dl.dense(p + "D"), mx::float32);
      mx::array nw = dl.dense(p + "norm.weight");
      Linear out = dl.linear(p + "out_proj", 0);
      MambaWeights w(std::move(in), std::move(conv_w), std::move(conv_b),
                     std::move(A_log), std::move(dt_bias), std::move(D),
                     std::move(nw), std::move(out));
      MambaCacheState cache(
          mx::zeros(mx::Shape{1, mK - 1, mConv}, mx::bfloat16),
          mx::zeros(mx::Shape{1, mH, mDh, mDs}, mx::float32));
      int idx = static_cast<int>(m.mamba_caches.size());
      m.mamba_caches.push_back(std::move(cache));
      m.mamba_layers.push_back({std::move(norm_w), std::move(w), idx});
    } else if (d.kind == LayerKind::FULL_ATTN_QSA) {
      const std::string p = b + "mixer.";
      NemoAttentionWeights w{
          dl.linear(p + "q_proj", 0), dl.linear(p + "k_proj", 0),
          dl.linear(p + "v_proj", 0), dl.linear(p + "o_proj", 0)};
      AttnLayer al;
      al.norm_w = std::move(norm_w);
      al.weights = std::move(w);
      al.cfg = ac;
      al.cache_idx = static_cast<int>(m.attn_layers.size());
      m.attn_layers.push_back(std::move(al));
    } else if (d.kind == LayerKind::MOE) {
      const std::string p = b + "mixer.";
      MoeLayer ml;
      ml.norm_w = std::move(norm_w);
      ml.gate = dl.linear(p + "gate.weight", 0);
      ml.gate_bias = mx::astype(dl.dense(p + "gate.e_score_correction_bias"),
                                mx::float32);
      ml.shared_up = dl.linear(p + "shared_experts.up_proj", 0);
      ml.shared_down = dl.linear(p + "shared_experts.down_proj", 0);
      ml.fc1_base = p + "switch_mlp.fc1";
      ml.fc2_base = p + "switch_mlp.fc2";
      (void)er;  // expert rows are read in run_step
      m.moe_layers.push_back(std::move(ml));
    } else {
      throw std::runtime_error("unsupported layer kind at layer " +
                               std::to_string(l));
    }
    if ((l + 1) % 8 == 0 || l == m.n_layers - 1)
      std::fprintf(stderr, "[load] prepared layer %d/%d\n", l + 1, m.n_layers);
  }

  DenseLoader::QTriplet embed = dl.quantized("backbone.embeddings");
  m.embed_w = std::move(embed.w);
  m.embed_s = std::move(embed.s);
  m.embed_b = std::move(embed.b);
  m.norm_f_w = dl.dense("backbone.norm_f.weight");
  DenseLoader::QTriplet head = dl.quantized("lm_head");
  m.lm_head = Linear::quantized(std::move(head.w), std::move(head.s),
                                std::move(head.b), m.group_size, m.bits);
  return m;
}

// --------------------------------------------------------------------------- //
// One decode step: token embedding -> 52 layers -> final norm + lm_head.          //
// --------------------------------------------------------------------------- //
mx::array run_step(Model& m, ExpertReader& er, int token) {
  mx::array h = embed_token(m, token);  // [1,1,H] bf16
  size_t mamba_i = 0, attn_i = 0, moe_i = 0;

  for (int l = 0; l < m.n_layers; ++l) {
    mx::array xn = mx::fast::rms_norm(h, m.kinds[l] == LayerKind::MAMBA_SSM
                                            ? m.mamba_layers[mamba_i].norm_w
                                            : (m.kinds[l] == LayerKind::FULL_ATTN_QSA
                                                   ? m.attn_layers[attn_i].norm_w
                                                   : m.moe_layers[moe_i].norm_w),
                                      m.eps);

    mx::array delta = empty_f32();
    if (m.kinds[l] == LayerKind::MAMBA_SSM) {
      MambaLayer& ml = m.mamba_layers[mamba_i++];
      MambaConfig mc;
      mc.hidden_size = m.H;
      mc.num_heads = m.spec.mamba_num_heads;
      mc.head_dim = m.spec.mamba_head_dim;
      mc.n_groups = m.spec.mamba_n_groups;
      mc.ssm_state_size = m.spec.mamba_ssm_state_size;
      mc.conv_kernel = m.spec.mamba_conv_kernel;
      mc.intermediate_size = m.spec.mamba_intermediate_size;
      mc.conv_dim = m.spec.mamba_conv_dim;
      mc.eps = m.eps;
      mc.time_step_min = m.spec.mamba_time_step_min;
      mc.time_step_max = m.spec.mamba_time_step_max;
      MambaCacheState& cache = m.mamba_caches[ml.cache_idx];
      MambaLayerOutput out = mamba_layer_forward(xn, ml.weights, cache, mc);
      cache = MambaCacheState(std::move(out.conv_state), std::move(out.ssm_state));
      delta = std::move(out.output);
    } else if (m.kinds[l] == LayerKind::FULL_ATTN_QSA) {
      AttnLayer& al = m.attn_layers[attn_i++];
      if (m.use_int8_kv)
        delta = nemo_attention_int8_forward(xn, al.weights, al.cfg, al.int8_cache);
      else
        delta = nemo_attention_forward(xn, al.weights, al.cfg, al.cache);
    } else {
      MoeLayer& ml = m.moe_layers[moe_i++];
      NemotronRoute route = nemotron_route(xn, ml, m.top_k, m.routed_scale);

      // Read the selected expert rows (dedup + sorted, like Python _gather).
      mx::array ids_i = mx::astype(route.indices, mx::int32);
      mx::eval(ids_i);
      const int32_t* ip = ids_i.data<int32_t>();
      std::vector<int> ids(ip, ip + m.top_k);
      std::vector<int> unique = ids;
      std::sort(unique.begin(), unique.end());
      unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
      std::vector<int> local(m.top_k);
      for (int i = 0; i < m.top_k; ++i)
        local[i] = static_cast<int>(
            std::lower_bound(unique.begin(), unique.end(), ids[i]) -
            unique.begin());
      mx::array local_idx(local.begin(), mx::Shape{1, m.top_k}, mx::int32);

      Q4Projection up(er.rows_u32(ml.fc1_base + ".weight", unique),
                      er.rows_bf16(ml.fc1_base + ".scales", unique),
                      er.rows_bf16(ml.fc1_base + ".biases", unique));
      Q4Projection down(er.rows_u32(ml.fc2_base + ".weight", unique),
                        er.rows_bf16(ml.fc2_base + ".scales", unique),
                        er.rows_bf16(ml.fc2_base + ".biases", unique));

      mx::array x2 = mx::reshape(xn, mx::Shape{1, m.H});
      mx::array routed = moe_block_relu2(x2, local_idx, route.scores, up, down);
      // Python: (y * scores).sum(-2).astype(y.dtype) -> bf16.
      routed = mx::astype(routed, mx::bfloat16);
      routed = mx::reshape(routed, mx::Shape{1, 1, m.H});

      // Shared expert: down_proj(relu2(up_proj(x))).
      mx::array su = linear(xn, ml.shared_up);
      // Mirror Python nn.relu2 (weak scalar 0): keep su's dtype (bf16), do
      // not promote bf16 -> f32 with a strongly-typed f32 zero array.
      mx::array sa = mx::square(
          mx::maximum(su, mx::astype(mx::array(0.0f), su.dtype())));
      mx::array sd = linear(sa, ml.shared_down);
      delta = routed + sd;
    }

    h = h + delta;  // NemotronHBlock: return x + mixer(norm(x))
    if (m.dump_layers)
      std::fprintf(stderr, "[NEMOLAYER step L=%02d] %s\n", l,
                   digest_array(h).c_str());
  }

  mx::array fused = mx::fast::rms_norm(h, m.norm_f_w, m.eps);
  mx::array logits = linear(fused, m.lm_head);  // [1,1,V]
  return mx::reshape(logits, mx::Shape{1, logits.shape(logits.shape().size() - 1)});
}

std::vector<StepRecord> run_generation(const std::string& ckpt, const Config& cfg,
                                       int num_tokens) {
  Model m = load_model(ckpt, cfg);
  if (std::getenv("NEMOTRON_LOAD_ONLY")) {
    std::fprintf(stderr, "[load] LOAD-ONLY: all weights loaded, no decode\n");
    mem_telemetry("load-only");
    return {};
  }
  ExpertReader er(ckpt, cfg);
  const std::vector<int> prompt = split_ints(get_str(cfg, "prompt_ids"));
  if (prompt.empty()) throw std::runtime_error("empty prompt_ids in config");
  std::fprintf(stderr, "[load] all weights ready; starting generation\n");
  mem_telemetry("post-load");

  std::vector<StepRecord> records;
  records.reserve(num_tokens);

  // Sequential single-token prefill (matches Python chunk_tokens=1). The
  // logits after the last prompt token produce the first generated token.
  int pos = 0;
  std::optional<mx::array> logits;
  for (size_t i = 0; i < prompt.size(); ++i) {
    logits = run_step(m, er, prompt[i]);
    ++pos;
    if ((i + 1) % 4 == 0 || i + 1 == prompt.size())
      mem_telemetry("prefill-step");
  }
  if (!logits) throw std::runtime_error("no prefill logits");
  int token = prompt.back();
  (void)pos;

  for (int step = 0; step < num_tokens; ++step) {
    if (step > 0) {
      logits = run_step(m, er, token);
      mem_telemetry("decode-step");
    }
    mx::array next = sample_greedy(*logits);
    mx::eval(next);
    int next_id = static_cast<int>(next.data<uint32_t>()[0]);
    std::string digest = digest_array(*logits);
    records.push_back({step, token, next_id, digest});
    std::fprintf(stderr, "[step %d] in=%d out=%d sha=%s\n", step, token,
                 next_id, digest.c_str());
    token = next_id;
  }
  return records;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <checkpoint-dir> <reference.json> <num_tokens> "
                 "[flat-config]\n", argv[0]);
    return 2;
  }
  const std::string ckpt = argv[1];
  const std::string reference = argv[2];
  const int num_tokens = std::atoi(argv[3]);
  std::string config_path = argc >= 5 ? argv[4] : "";
  if (config_path.empty()) {
    size_t slash = reference.find_last_of('/');
    config_path = (slash == std::string::npos ? std::string(".")
                                              : reference.substr(0, slash)) +
                  "/nemotron_config.txt";
  }
  try {
    Config cfg = read_config(config_path);
    if (cfg.empty())
      throw std::runtime_error("empty flat config; run gen_nemotron_config.py first: " +
                               config_path);
    if (get_int(cfg, "prompt_tokens", 0) == 0)
      throw std::runtime_error("flat config has no prompt_ids");
    std::vector<StepRecord> records = run_generation(ckpt, cfg, num_tokens);
    for (const auto& r : records)
      std::printf("step=%d in=%d out=%d sha256=%s\n", r.step, r.token_in,
                  r.token_out, r.digest.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "nemotron_generate failed: %s\n", e.what());
    return 1;
  }
}
