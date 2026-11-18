// real_checkpoint_generate -- standalone C++ greedy generation from the REAL
// Qwen3.8-Flash-Next MLX-4bit checkpoint, with no Python in the generation loop.
//
// WHAT THIS IS
// ------------
// Decisive integration evidence for the C++ engine rewrite: the 113 GB Q4
// checkpoint's NON-expert dense weights are read straight from the safetensors
// shards with mx::load_safetensors + mx::dequantize, every MoE layer is served
// by a persistent StreamingMoE (real expert rows via the native pager), all 48
// layers (GDN linear + QSA full attention, hyper-connections, shared experts)
// are composed by the SAME primitives as `run_decoder_step_full`, and greedy
// decoding runs token-by-token in C++. Per-step logit SHA-256 uses the exact
// float32-little-endian convention of capture_real_reference.py.
//
// HONEST SCOPE (see the note in research_notes/Cerebras inference and IO
// splitting/real_checkpoint_generate.md for the full write-up):
//  * The decoder loop is a byte-for-byte adaptation of run_decoder_step_full
//    with ONE substitution: the routed experts come from StreamingMoE::forward
//    (slot-remapped gather_qmm) instead of a fully-resident ExpertWeights.
//    run_decoder_step_full itself cannot accept streamed experts because it
//    re-computes the raw router indices internally and would gather stale
//    slots. Every other primitive call is identical.
//  * Prefill is fed as sequential single-token steps (S=1). run_decoder_step's
//    MoE reshape is [1,H]-only, so S>1 is not a supported shape; the GDN path
//    is token-by-token anyway. Python's chunked prefill is a different path.
//  * QSA indexer returns None for <2048 past tokens, so production uses dense
//    causal attention. That is the fallback implemented here (the library's
//    full_attention_forward throws on None). KV is dense, not int8 quantized.
//  * PLE (layer 2), fused-Metal RoPE, quantized dense matmul, quantized shared
//    expert, prompt prefix cache and MTP are NOT reproduced. Bit-exact parity
//    with the Python reference is NOT expected; the parity checker reports the
//    exact first divergent step without weakening the gate.
//
// USAGE
// -----
//   /tmp/real_gen <checkpoint-dir> <reference.json> <num_tokens> [flat-config]
//
// The 4th argument is optional. When omitted, the flat config is looked up at
// <reference-dir>/real_config.txt (written by gen_real_config.py). That flat
// config carries the tensor->shard map and the deterministic prompt ids.
//
// Build (from repo root): see the note / the handoff report.
// Output: one line per generated token,
//   step=<i> in=<token_in> out=<token_out> sha256=<64 hex>
#include "decoder_full.hpp"
#include "expert_store.hpp"
#include "moe_stream.hpp"
#include "cross_shard_moe.hpp"
#include "dense_manifest.hpp"
#include "rope_fused_apply.hpp"
#include "ple.hpp"
#include "prefill_batched.hpp"
#include "io_pager.hpp"
#include "gdn_layer.hpp"
#include "gdn_conv.h"

#include <mlx/mlx.h>
#include <mlx/memory.h>

#include <algorithm>
#include <array>
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
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
// Opt-in localization instrumentation (REALGEN_DEBUG_HASHES=1). Off by default.
// Captures the hidden state after every layer (plus branch / MoE / final mixed
// / final logits) on the LAST prefill step, then hashes + NaN/Inf-counts them.
// Used only by the determinism localization run; no effect when disabled.
// --------------------------------------------------------------------------- //
namespace debug_capture {
struct Capture {
  bool enabled = false;
  bool use_local_gdn = false;
  bool copy_cross_shard = false;
  bool moe_hash = false;
  int current_step = -1;
  std::vector<std::string> names;
  std::vector<mx::array> arrays;
  void add(const char* name, mx::array a) {
    if (!enabled) return;
    names.emplace_back(name);
    arrays.push_back(std::move(a));
  }
  void clear() { names.clear(); arrays.clear(); }
};
Capture& get() { static Capture c; return c; }
}  // namespace debug_capture

// --------------------------------------------------------------------------- //
// Self-contained SHA-256 (no OpenSSL/CommonCrypto linking needed).             //
// Identical to veystrix_native_generate.cpp.                                   //
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

// Opt-in: SHA-256 the float32 GDN cache state AFTER each prefill step. These
// are the tensors fed into the next step, so the first step/layer where the two
// runs' hashes diverge localizes the responsible op.
void dump_gdn_states(const std::vector<DecoderFullLayerCache>& caches,
                     int step, int n_layers) {
  for (int l = 0; l < n_layers; ++l) {
    mx::array cv = mx::reshape(
        mx::astype(caches[l].base.gdn.conv_state, mx::float32), mx::Shape{-1});
    mx::eval(cv);
    mx::array rs = mx::reshape(
        mx::astype(caches[l].base.gdn.recurrent_state, mx::float32), mx::Shape{-1});
    mx::eval(rs);
    std::fprintf(stderr,
                 "[GDNSTATE step=%02d L=%02d conv=%s rec=%s\n", step, l,
                 sha256_hex(cv.data<float>(), static_cast<size_t>(cv.size()) * sizeof(float)).c_str(),
                 sha256_hex(rs.data<float>(), static_cast<size_t>(rs.size()) * sizeof(float)).c_str());
  }
}

// Opt-in MoE diagnostics: hash raw expert bytes, routes, and MoE I/O.
void dbg_emit(const char* tag, const void* data, size_t len) {
  std::fprintf(stderr, "[MOEHASH] %s %s\n", tag,
               sha256_hex(data, len).c_str());
}
void dbg_emit_array(const char* tag, const mx::array& a) {
  mx::array f = mx::reshape(mx::astype(a, mx::float32), mx::Shape{-1});
  mx::eval(f);
  dbg_emit(tag, f.data<float>(), static_cast<size_t>(f.size()) * sizeof(float));
}
void dbg_emit_ints(const char* tag, const std::vector<int32_t>& v) {
  dbg_emit(tag, v.data(), v.size() * sizeof(int32_t));
}
bool dbg_should_hash() {
  return debug_capture::get().moe_hash && debug_capture::get().current_step <= 2;
}

// Always-on memory telemetry (no numeric effect): active/cache/peak GiB.
static void mem_telemetry(const char* where) {
  const double GIB = 1024.0 * 1024.0 * 1024.0;
  std::fprintf(stderr,
               "[mem %s] active=%.2f GiB cache=%.2f GiB peak=%.2f GiB\n",
               where, mx::get_active_memory() / GIB, mx::get_cache_memory() / GIB,
               mx::get_peak_memory() / GIB);
}

// --------------------------------------------------------------------------- //
// Flat config parsing (produced by gen_real_config.py).                        //
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

float get_float(const Config& c, const std::string& key, float def = 1e-6f) {
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
// Dense weight loader: only NON-expert tensors, grouped by shard.               //
// --------------------------------------------------------------------------- //

class DenseLoader {
 public:
  DenseLoader(const std::string& dir, const Config& cfg, int group_size, int bits)
      : dir_(dir), group_size_(group_size), bits_(bits),
        lazy_dense_(std::getenv("VEYSTRIX_CPP_LAZY_DENSE") != nullptr) {
    for (const auto& [key, shard] : cfg) {
      if (key.rfind("shard.", 0) == 0) shard_of_[key.substr(6)] = shard;
    }
  }

  mx::array raw(const std::string& name) {
    auto it = shard_of_.find(name);
    if (it == shard_of_.end())
      throw std::runtime_error("dense tensor not in shard map: " + name);
    auto sit = shards_.find(it->second);
    if (sit == shards_.end()) {
      auto [weights, meta] = mx::load_safetensors(dir_ + "/" + it->second);
      sit = shards_.emplace(it->second, std::move(weights)).first;
      std::fprintf(stderr, "[load] shard %s (%zu tensors)\n",
                   it->second.c_str(), sit->second.size());
    }
    auto tit = sit->second.find(name);
    if (tit == sit->second.end())
      throw std::runtime_error("tensor missing from shard: " + name);
    // Python evaluates ALL model parameters right after loading
    // (models/flashnext/loader.py: mx.eval(model.parameters())). C++ used to
    // return the lazy disk-read array here, so the read was deferred into the
    // prefill GPU graph and raced the background IO thread (nondeterministic
    // gate (b), large ~1.89 maxdiff at the first full-attention layer L11).
    // Materialize each requested tensor now, matching Python's load pattern.
    // A/B opt-out: VEYSTRIX_CPP_LAZY_DENSE=1 restores the old lazy behaviour.
    if (!lazy_dense_) mx::eval(tit->second);
    return tit->second;
  }

  // Raw (non-quantized) dense tensor: norm gains, A_log, conv weight, etc.
  // These have no `.scales` companion. Throws if `name` IS a quantized triplet,
  // because eager dequantization is exactly the memory/parity bug we removed.
  mx::array dense(const std::string& name, bool allow_dequantize = false) {
    const std::string base =
        name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0
            ? name.substr(0, name.size() - 7)
            : name;
    bool quantized = shard_of_.count(base + ".scales") != 0;
    if (!quantized) return raw(name);
    if (!allow_dequantize)
      throw std::runtime_error(
          "dense() called on a quantized tensor (use linear()): " + name);
    mx::array w = raw(base + ".weight");
    mx::array s = as_bf16(raw(base + ".scales"));
    mx::array b = as_bf16(raw(base + ".biases"));
    mx::array out = mx::dequantize(w, s, b, group_size_, bits_, "affine");
    mx::eval(out);
    return out;
  }

  // A dense projection, kept PACKED when the checkpoint has a `.scales`
  // companion (Python's nn.QuantizedLinear path) so no bf16 copy is
  // materialized and the arithmetic matches production. `logical_in` is the
  // module's logical input dim (nn.Linear.weight.shape[-1]); group_size and
  // bits are derived exactly like loader.py's class_predicate:
  //   bits = packed_in*32 // logical_in ; group_size = logical_in // n_groups
  // and fail closed if they do not divide evenly. When a manifest was loaded,
  // its recorded (kind, group_size, bits, has_bias) is cross-checked.
  Linear linear(const std::string& name, int logical_in) {
    // Accept either the triplet base ("...mlp.gate") or a name ending in
    // ".weight" ("...mlp.gate.weight"); normalize to the base, exactly as
    // dense() does. Manifest keys use module paths without ".weight".
    const std::string base =
        name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0
            ? name.substr(0, name.size() - 7)
            : name;
    if (shard_of_.count(base + ".scales") == 0) {
      // Plain Linear / raw tensor: keep the original dense path.
      if (!manifest_.empty()) {
        auto it = manifest_.find(base);
        if (it != manifest_.end() &&
            (it->second.kind == "quantized_linear" ||
             it->second.kind == "quantized_embedding"))
          throw std::runtime_error("manifest says quantized but checkpoint has no scales: " + base);
      }
      // No `.scales`: a plain nn.Linear (e.g. the router gate) or raw gain.
      // The shard map is keyed by the full tensor name (with ".weight").
      return Linear(raw(name.find(".weight") != std::string::npos
                            ? name
                            : base + ".weight"));
    }
    mx::array w = raw(base + ".weight");
    mx::array s = as_bf16(raw(base + ".scales"));
    mx::array b = as_bf16(raw(base + ".biases"));
    int packed_in = static_cast<int>(w.shape(w.shape().size() - 1));
    int n_groups = static_cast<int>(s.shape(s.shape().size() - 1));
    if (packed_in <= 0 || n_groups <= 0)
      throw std::runtime_error("linear(): invalid shapes for " + base);

    // Resolve (bits, group_size). The manifest, when loaded, is authoritative
    // (it is generated from the exact Python model); otherwise derive from the
    // global config bits, exactly the inverse of loader.py's class_predicate.
    int bits = 0, gs = 0;
    const dmanifest::MInfo* mp = nullptr;
    if (!manifest_.empty()) {
      auto it = manifest_.find(base);
      if (it == manifest_.end())
        throw std::runtime_error("manifest missing quantized tensor: " + base);
      mp = &it->second;
      if (mp->kind != "quantized_linear" && mp->kind != "quantized_embedding")
        throw std::runtime_error("manifest says non-quantized but checkpoint has scales: " + base);
      bits = mp->bits;
      gs = mp->group_size;
    } else if (logical_in > 0) {
      if (packed_in * 32 % logical_in != 0)
        throw std::runtime_error("linear(): bits do not divide evenly for " + base);
      bits = packed_in * 32 / logical_in;
      if (logical_in % n_groups != 0)
        throw std::runtime_error("linear(): group size does not divide evenly for " + base);
      gs = logical_in / n_groups;
    } else {
      bits = bits_;
      if (packed_in * 32 % bits != 0)
        throw std::runtime_error("linear(): global bits do not divide packed_in for " + base);
      int logical = packed_in * 32 / bits;
      if (logical % n_groups != 0)
        throw std::runtime_error("linear(): group size does not divide evenly for " + base);
      gs = logical / n_groups;
    }
    if (bits < 1 || bits > 8 || gs < 1)
      throw std::runtime_error("linear(): implausible gs/bits for " + base);
    // Shape consistency check (always): packed_in*32/bits must equal gs*n_groups.
    if (packed_in * 32 != gs * n_groups * bits)
      throw std::runtime_error("linear(): gs/bits inconsistent with shapes for " + base);
    return Linear::quantized(std::move(w), std::move(s), std::move(b), gs, bits);
  }

  // Load a dense manifest produced by tools/dump_dense_manifest.py. Empty path
  // disables the cross-check (used by synthetic fixtures without a manifest).
  void load_manifest(const std::string& path) {
    if (path.empty()) return;
    std::ifstream probe(path);
    if (!probe) throw std::runtime_error("cannot open dense manifest: " + path);
    manifest_ = dmanifest::load_file(path);
    if (manifest_.empty())
      throw std::runtime_error("dense manifest parsed to zero entries: " + path);
    std::fprintf(stderr, "[load] dense manifest: %zu entries (%s)\n",
                 manifest_.size(), path.c_str());
  }

  // Quantized triplet kept in packed form (embed table + lm_head): dequantize
  // only the gathered embedding row / use quantized_matmul for the head. This
  // both saves ~2.5 GB and matches Python's nn.quantize'd Embedding/Linear.
  struct QTriplet {
    mx::array w, s, b;
  };
  QTriplet quantized(const std::string& name) {
    return {raw(name + ".weight"), as_bf16(raw(name + ".scales")),
            as_bf16(raw(name + ".biases"))};
  }

 private:
  mx::array as_bf16(mx::array a) const {
    if (static_cast<mx::Dtype::Val>(a.dtype()) == mx::uint16) {
      mx::array v = mx::view(a, mx::bfloat16);
      // Resolve the byte reinterpret now so scales/biases are materialized at
      // load time (same rationale as raw()); a view's source is already eval'd.
      if (!lazy_dense_) mx::eval(v);
      return v;
    }
    return a;
  }

  std::string dir_;
  int group_size_, bits_;
  bool lazy_dense_;
  std::map<std::string, std::string> shard_of_;
  std::map<std::string, std::unordered_map<std::string, mx::array>> shards_;
  std::map<std::string, dmanifest::MInfo> manifest_;
};


// --------------------------------------------------------------------------- //
// Per-layer real weights.                                                      //
// --------------------------------------------------------------------------- //
struct RealLayer {
  mx::array attn_norm_w;
  Linear attn_down_w, attn_up_w, attn_inject_w;
  mx::array mlp_norm_w;
  Linear mlp_down_w, mlp_up_w, mlp_inject_w;
  Linear gate_w;
  Linear shared_gw, shared_uw, shared_dw, shared_glw;
  std::optional<GDNWeights> gdn;
  std::optional<FullAttentionWeights> full;
  std::unique_ptr<StreamingMoE> stream;       // single-shard layers
  std::unique_ptr<CrossShardMoE> cross;       // layers split across two shards
  int layer_idx = 0;
  bool split_layer = false;
  std::string expert_prefix;
  std::string expert_shard_path;  // shard path for full-layer expert load (prefill)
  std::optional<PLEWeights> ple;              // set only on PLE layers (index 1)
  PLECache ple_cache;                         // mutable n-gram/conv cache for that layer
  int top_k = 0;
  AttnKind kind = AttnKind::Linear;

  RealLayer(mx::array an, Linear ad, Linear au, Linear ai,
            mx::array mn, Linear md, Linear mu, Linear mi,
            Linear gw, Linear sg, Linear su, Linear sd,
            Linear sgl, std::optional<GDNWeights> g,
            std::optional<FullAttentionWeights> f,
            std::unique_ptr<StreamingMoE> st, std::unique_ptr<CrossShardMoE> cr,
            std::optional<PLEWeights> p,
            int tk, AttnKind k)
      : attn_norm_w(std::move(an)), attn_down_w(std::move(ad)),
        attn_up_w(std::move(au)), attn_inject_w(std::move(ai)),
        mlp_norm_w(std::move(mn)), mlp_down_w(std::move(md)),
        mlp_up_w(std::move(mu)), mlp_inject_w(std::move(mi)),
        gate_w(std::move(gw)), shared_gw(std::move(sg)),
        shared_uw(std::move(su)), shared_dw(std::move(sd)),
        shared_glw(std::move(sgl)), gdn(std::move(g)), full(std::move(f)),
        stream(std::move(st)), cross(std::move(cr)),
        ple(std::move(p)), top_k(tk), kind(k) {}
};

struct RealHead {
  mx::array norm_w;
  Linear down_w, up_w;
  DenseLoader::QTriplet lm;
};

// --------------------------------------------------------------------------- //
// Dense (non-sparse) full-attention branch. The QSA indexer returns None for   //
// <2048 past tokens; production then uses a causal mask. This mirrors          //
// full_attention_forward() with that None path composed.                       //
// --------------------------------------------------------------------------- //
FullAttentionOutput full_attention_dense(
    const mx::array& x, const FullAttentionWeights& w,
    const FullAttentionConfig& cfg, FullAttentionCache& cache,
    const mx::array& position_ids) {
  const int B = static_cast<int>(x.shape(0));
  const int S = static_cast<int>(x.shape(1));
  const int Hq = cfg.n_heads, Hkv = cfg.kv_heads, D = cfg.head_dim;

  mx::array qproj = linear(x, w.q_w);
  mx::array kproj = linear(x, w.k_w);
  mx::array vproj = linear(x, w.v_w);

  mx::array q4 = mx::reshape(qproj, mx::Shape{B, S, Hq, 2 * D});
  std::vector<mx::array> halves = mx::split(q4, 2, -1);
  mx::array q = halves[0];
  mx::array gate = mx::reshape(halves[1], mx::Shape{B, S, Hq * D});

  // Production patches ALL Qwen4ExpRMSNorm (incl. attention q_norm/k_norm) to
  // one-centered gains: gain = weight (NO +1.0). qsa_rms_norm adds +1.0 (the
  // unpatched vector convention), so use the one-centered form here.
  auto qnorm = [&](const mx::array& a, const mx::array& gain) {
    auto d = a.dtype();
    mx::array y = mx::astype(a, mx::float32);
    y = y * mx::rsqrt(mx::mean(mx::square(y), -1, true) + cfg.eps);
    y = y * mx::astype(gain, mx::float32);
    return mx::astype(y, d);
  };
  q = mx::transpose(qnorm(q, w.q_norm_w), {0, 2, 1, 3});
  mx::array k = mx::transpose(
      qnorm(mx::reshape(kproj, mx::Shape{B, S, Hkv, D}), w.k_norm_w),
      {0, 2, 1, 3});
  mx::array v = mx::transpose(
      mx::reshape(vproj, mx::Shape{B, S, Hkv, D}), {0, 2, 1, 3});

  q = rope_apply_fused(q, position_ids, cfg.rope);
  k = rope_apply_fused(k, position_ids, cfg.rope);

  if (!cache.keys.has_value()) {
    cache.keys = k;
    cache.values = v;
  } else {
    cache.keys = mx::concatenate({*cache.keys, k}, 2);
    cache.values = mx::concatenate({*cache.values, v}, 2);
  }

  const float scale = static_cast<float>(std::pow(static_cast<double>(D), -0.5));
  mx::array out = mx::fast::scaled_dot_product_attention(
      q, *cache.keys, *cache.values, scale, /*mask_mode=*/"causal",
      /*mask_arr=*/std::nullopt, /*sinks=*/std::nullopt);
  out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), mx::Shape{B, S, Hq * D});
  out = linear(out * mx::sigmoid(gate), w.o_w);

  cache.offset += S;
  // Placeholder mask: this branch is the dense (indexer=None) fallback, so no
  // sparse mask is produced. Only `.output` is consumed by the caller.
  std::vector<int32_t> dummy{0};
  mx::array mask(dummy.begin(), mx::Shape{1}, mx::int32);
  return {out, mask};
}

inline mx::array inject(const mx::array& hidden, const mx::array& branch,
                        const mx::array& inj) {
  mx::array b4 = mx::expand_dims(branch, -2);
  mx::array i4 = mx::expand_dims(inj, -1);
  return hidden + mx::reshape(b4 * i4, hidden.shape());
}

// --------------------------------------------------------------------------- //
// TOOL-LOCAL GDN forward for the opt-in determinism verification. Identical to
// gdn_layer_forward except: (1) the fused mx::fast::metal_kernel delta step is
// replaced by an ops-based delta step (the shared engine.cpp version uses
// mx::expand_dims({-1,-1}) which this MLX rejects, and expects Hv-head q/k);
// (2) conv_state / recurrent_state are mx::eval'd at every step boundary so the
// next step consumes a materialized buffer instead of a 16-deep lazy graph.
// No shared primitive is modified.
// --------------------------------------------------------------------------- //
DeltaStepOutput gated_delta_step_local(const mx::array& q, const mx::array& k,
                                       const mx::array& v, const mx::array& g,
                                       const mx::array& beta,
                                       const mx::array& state) {
  // q,k: [B,T,Hv,Dk]; v: [B,T,Hv,Dv]; g,beta: [B,T,Hv]; state: [B,Hv,Dv,Dk].
  const int B = g.shape(0), T = g.shape(1), Hv = g.shape(2);
  mx::array decay = mx::reshape(g, mx::Shape{B, T, Hv, 1, 1});
  mx::array new_state = state * decay;                                  // [B,T,Hv,Dv,Dk]
  mx::array kv_mem = mx::sum(new_state * mx::expand_dims(k, -1), -1);   // [B,T,Hv,Dv]
  mx::array delta = (v - kv_mem) * mx::expand_dims(beta, -1);           // [B,T,Hv,Dv]
  new_state = new_state + mx::expand_dims(k, -1) * mx::expand_dims(delta, -1);
  mx::array y = mx::sum(new_state * mx::expand_dims(q, -1), -1);        // [B,T,Hv,Dv]
  return {mx::astype(y, q.dtype()), new_state};
}

GDNLayerOutput gdn_layer_forward_local(const mx::array& x, const GDNWeights& w,
                                       const GDNCacheState& cache,
                                       const GDNConfig& cfg) {
  const int B = x.shape(0), S = x.shape(1);
  const int H = cfg.hidden_size;
  const int Hk = cfg.num_k_heads, Hv = cfg.num_v_heads;
  const int Dk = cfg.head_k_dim, Dv = cfg.head_v_dim;
  const int K = cfg.conv_kernel;
  const int conv_dim = 2 * cfg.key_dim + cfg.value_dim;
  const bool sigmoid = cfg.output_activation == GDNActivation::Sigmoid;

  mx::array conv_state = cache.conv_state;
  mx::array rec_state = cache.recurrent_state;
  std::vector<mx::array> steps;
  steps.reserve(S);

  for (int t = 0; t < S; ++t) {
    mx::array xt = mx::slice(x, mx::Shape{0, t, 0}, mx::Shape{B, t + 1, H});
    mx::array mixed_qkv = linear(xt, w.in_proj_qkv);
    mx::array z = linear(xt, w.in_proj_z);
    mx::array b = linear(xt, w.in_proj_b);
    mx::array a = linear(xt, w.in_proj_a);
    z = mx::reshape(z, mx::Shape{B, 1, Hv, Dv});

    mx::array conv_input = mx::concatenate({conv_state, mixed_qkv}, 1);
    const int n_keep = K - 1;
    const int Tconv = conv_input.shape(1);
    conv_state = mx::contiguous(mx::slice(
        conv_input, mx::Shape{0, Tconv - n_keep, 0}, mx::Shape{B, Tconv, conv_dim}));
    mx::array conv_out = gdn_causal_conv_decode_silu(conv_input, w.conv1d_weight);

    mx::array q = mx::slice(conv_out, mx::Shape{0, 0, 0}, mx::Shape{B, 1, cfg.key_dim});
    mx::array k = mx::slice(conv_out, mx::Shape{0, 0, cfg.key_dim},
                            mx::Shape{B, 1, 2 * cfg.key_dim});
    mx::array v = mx::slice(conv_out, mx::Shape{0, 0, 2 * cfg.key_dim},
                            mx::Shape{B, 1, conv_dim});
    q = mx::reshape(q, mx::Shape{B, 1, Hk, Dk});
    k = mx::reshape(k, mx::Shape{B, 1, Hk, Dk});
    v = mx::reshape(v, mx::Shape{B, 1, Hv, Dv});

    auto qk = gdn_normalize_qk(q, k);
    auto gb = compute_g_beta(w.A_log, a, b, w.dt_bias);
    // The fused kernel maps Hv value heads onto Hk key heads internally; the
    // ops-based path needs explicit Hk->Hv head replication before the step.
    std::vector<int32_t> hk_idx;
    for (int h = 0; h < Hk; ++h)
      for (int r = 0; r < Hv / Hk; ++r) hk_idx.push_back(h);
    mx::array hidx(hk_idx.data(), mx::Shape{Hv}, mx::int32);
    mx::array qv = mx::take(qk.first, hidx, 2);
    mx::array kv = mx::take(qk.second, hidx, 2);
    auto step = gated_delta_step_local(qv, kv, v, gb.first, gb.second,
                                       rec_state);
    // Materialization barriers: both state tensors are concrete before the next
    // token/layer reads them (breaks the lazy alias chain across prefill steps).
    mx::eval(step.state);
    rec_state = step.state;
    mx::eval(conv_state);

    mx::array normed = rms_norm_gated(step.y, w.norm_weight, z, cfg.eps, sigmoid);
    mx::array flat = mx::reshape(normed, mx::Shape{B, 1, cfg.value_dim});
    steps.push_back(linear(flat, w.out_proj));
  }
  mx::array out = (S == 1) ? steps[0] : mx::concatenate(steps, 1);
  return {out, conv_state, rec_state};
}

// --------------------------------------------------------------------------- //
// One decode step: run_decoder_step_full with the routed MoE served by          //
// StreamingMoE. All other primitive calls and their order are identical.       //
// --------------------------------------------------------------------------- //
mx::array run_step(
    const mx::array& input_embeddings,
    int32_t input_token,  // raw token id for this step (needed by PLE n-gram hash)
    std::vector<RealLayer>& layers,
    const RealHead& head,
    const DecoderConfig& cfg,
    const FullAttentionConfig& fa_cfg,
    const mx::array& position_ids,
    std::vector<DecoderFullLayerCache>& caches,
    int group_size, int bits) {
  mx::array hidden =
      mx::tile(input_embeddings, std::vector<int>{1, 1, cfg.hc_count});
  if (const char* ed = std::getenv("REALGEN_DUMP_EMBED")) {
    if (debug_capture::get().current_step == 15) {
      mx::array ef = mx::reshape(mx::astype(hidden, mx::float32), mx::Shape{-1});
      mx::eval(ef);
      std::FILE* f = std::fopen(ed, "wb");
      if (f) { std::fwrite(ef.data<float>(), sizeof(float), ef.size(), f); std::fclose(f); }
    }
  }

  for (int l = 0; l < cfg.n_layers; ++l) {
    RealLayer& L = layers[l];

    // Production decoder: hidden_states = hidden_states + ple(hidden_states,
    // input_ids, ple_cache). PLE is present only on the configured PLE layer.
    if (L.ple) {
      std::vector<int64_t> tid{static_cast<int64_t>(input_token)};
      mx::array tok(tid.data(), mx::Shape{1, 1}, mx::int64);
      mx::array ple_out =
          ple_forward(hidden, tok, *L.ple, L.ple_cache, /*mask=*/nullptr);
      hidden = hidden + ple_out;
    }

    GatedResidualOutput ar = gated_residual(
        hidden, L.attn_norm_w, L.attn_down_w, L.attn_up_w, L.attn_inject_w,
        cfg.hc_count, cfg.hidden_size, cfg.eps);

    mx::array branch = [&]() -> mx::array {
      if (L.kind == AttnKind::Linear) {
        GDNLayerOutput g =
            debug_capture::get().use_local_gdn
                ? gdn_layer_forward_local(ar.mixed_input, *L.gdn,
                                          caches[l].base.gdn, cfg.gdn)
                : gdn_layer_forward(ar.mixed_input, *L.gdn,
                                    caches[l].base.gdn, cfg.gdn);
        caches[l].base.gdn = GDNCacheState(g.conv_state, g.recurrent_state);
        return g.output;
      }
      return full_attention_dense(ar.mixed_input, *L.full, fa_cfg,
                                  caches[l].full, position_ids)
          .output;
    }();
    {
      char nm[64];
      std::snprintf(nm, sizeof(nm), "L%02d_branch_%s", l,
                    L.kind == AttnKind::Linear ? "gdn" : "qsa");
      debug_capture::get().add(nm, branch);
    }
    hidden = inject(hidden, branch, ar.injection_weights);
    if (const char* hd = std::getenv("REALGEN_DUMP_LAYER_HIDDEN")) {
      // Dump hidden after attention for the LAST prompt step only (step==n_layers-1 index in prefill).
      if (debug_capture::get().current_step == 15) {
        mx::array hf = mx::reshape(mx::astype(hidden, mx::float32), mx::Shape{-1});
        mx::eval(hf);
        char pth[512];
        std::snprintf(pth, sizeof(pth), "%s.attn.L%02d.f32", hd, l);
        std::FILE* f = std::fopen(pth, "wb");
        if (f) { std::fwrite(hf.data<float>(), sizeof(float), hf.size(), f); std::fclose(f); }
      }
    }
    if (dbg_should_hash()) {
      char tag[160];
      std::snprintf(tag, sizeof(tag), "step%d L%02d hidden_after_attn",
                    debug_capture::get().current_step, l);
      dbg_emit_array(tag, hidden);
    }
    {
      char nm[64];
      std::snprintf(nm, sizeof(nm), "L%02d_hidden_attn", l);
      debug_capture::get().add(nm, hidden);
    }

    GatedResidualOutput mr = gated_residual(
        hidden, L.mlp_norm_w, L.mlp_down_w, L.mlp_up_w, L.mlp_inject_w,
        cfg.hc_count, cfg.hidden_size, cfg.eps);

    mx::array m2d = mx::reshape(mr.mixed_input, mx::Shape{1, cfg.hidden_size});
    mx::array gate_logits = linear(m2d, L.gate_w);
    RouterOutput routes = router(gate_logits, L.top_k, 0.0f);
    StreamForwardResult sf =
        L.stream ? L.stream->forward(m2d, routes.indices, routes.scores)
                 : L.cross->forward(m2d, routes.indices, routes.scores);

    // Non-zero dense shared expert (Qwen3_5MoeSparseMoeBlock combine).
    mx::array sgl = linear(m2d, L.shared_glw);
    mx::array sg = mx::sigmoid(sgl);
    mx::array g = linear(m2d, L.shared_gw);
    mx::array u = linear(m2d, L.shared_uw);
    mx::array s = linear(swiglu(g, u), L.shared_dw);
    mx::array moe2d = sf.output + sg * s;
    mx::array moe = mx::reshape(moe2d, mx::Shape{1, 1, cfg.hidden_size});
    {
      char nm[64];
      std::snprintf(nm, sizeof(nm), "L%02d_moe", l);
      debug_capture::get().add(nm, moe);
    }

    hidden = inject(hidden, moe, mr.injection_weights);
    if (const char* hd = std::getenv("REALGEN_DUMP_LAYER_HIDDEN")) {
      if (debug_capture::get().current_step == 15) {
        mx::array hf = mx::reshape(mx::astype(hidden, mx::float32), mx::Shape{-1});
        mx::eval(hf);
        char pth[512];
        std::snprintf(pth, sizeof(pth), "%s.layer.L%02d.f32", hd, l);
        std::FILE* f = std::fopen(pth, "wb");
        if (f) { std::fwrite(hf.data<float>(), sizeof(float), hf.size(), f); std::fclose(f); }
      }
    }
    if (dbg_should_hash()) {
      const int st = debug_capture::get().current_step;
      const char* kind = L.stream ? "stream" : "cross";
      char tag[160];
      std::snprintf(tag, sizeof(tag), "step%d L%02d %s moe_in_m2d", st, l, kind);
      dbg_emit_array(tag, m2d);
      std::snprintf(tag, sizeof(tag), "step%d L%02d %s gate_logits", st, l, kind);
      dbg_emit_array(tag, gate_logits);
      std::snprintf(tag, sizeof(tag), "step%d L%02d %s route_indices", st, l, kind);
      dbg_emit_array(tag, routes.indices);
      std::snprintf(tag, sizeof(tag), "step%d L%02d %s route_scores", st, l, kind);
      dbg_emit_array(tag, routes.scores);
      std::snprintf(tag, sizeof(tag), "step%d L%02d %s slots", st, l, kind);
      dbg_emit_ints(tag, sf.slots);
      std::snprintf(tag, sizeof(tag), "step%d L%02d %s misses", st, l, kind);
      dbg_emit_ints(tag, sf.misses);
      std::snprintf(tag, sizeof(tag), "step%d L%02d %s moe_out", st, l, kind);
      dbg_emit_array(tag, sf.output);
      std::snprintf(tag, sizeof(tag), "step%d L%02d %s moe2d", st, l, kind);
      dbg_emit_array(tag, moe2d);
      std::snprintf(tag, sizeof(tag), "step%d L%02d %s hidden_after_layer", st, l, kind);
      dbg_emit_array(tag, hidden);
    }
    {
      char nm[64];
      std::snprintf(nm, sizeof(nm), "L%02d_hidden", l);
      debug_capture::get().add(nm, hidden);
    }
  }

  // final_head with a quantized lm_head (matches nn.Linear after nn.quantize).
  const int hc = cfg.hc_count, H = cfg.hidden_size;
  if (const char* hi = std::getenv("REALGEN_DUMP_HIDDEN")) {
    mx::array hf = mx::reshape(mx::astype(hidden, mx::float32), mx::Shape{-1});
    mx::eval(hf);
    char pth[512];
    std::snprintf(pth, sizeof(pth), "%s.step%d.f32", hi,
                  debug_capture::get().current_step);
    std::FILE* f = std::fopen(pth, "wb");
    if (f) { std::fwrite(hf.data<float>(), sizeof(float), hf.size(), f); std::fclose(f); }
  }
  mx::Shape gs{static_cast<int>(hidden.shape(0)),
               static_cast<int>(hidden.shape(1)), hc, H};
  mx::array w2d = mx::reshape(head.norm_w, mx::Shape{hc, H});
  mx::array yg = mx::astype(mx::reshape(hidden, gs), mx::float32);
  yg = yg * mx::rsqrt(mx::mean(mx::square(yg), -1, true) + cfg.eps);
  yg = yg * mx::astype(w2d, mx::float32);
  mx::array normed = mx::astype(mx::reshape(yg, hidden.shape()), hidden.dtype());
  mx::array hc_arr = mx::array(hc);  // int32 array: keeps bf16
  mx::array low = linear(normed, head.down_w) / hc_arr;
  mx::array mix = mx::sigmoid(
      linear(hc_silu(low), head.up_w));
  mx::array mixed = mx::mean(mx::reshape(mix, gs) * mx::reshape(normed, gs), -2);
  debug_capture::get().add("final_mixed", mixed);
  if (const char* mf = std::getenv("REALGEN_DUMP_MIXED")) {
    mx::array mf32 = mx::reshape(mx::astype(mixed, mx::float32), mx::Shape{-1});
    mx::eval(mf32);
    char pth[512];
    std::snprintf(pth, sizeof(pth), "%s.step%d.f32", mf,
                  debug_capture::get().current_step);
    std::FILE* f = std::fopen(pth, "wb");
    if (f) { std::fwrite(mf32.data<float>(), sizeof(float), mf32.size(), f); std::fclose(f); }
  }

  mx::array logits =
      mx::quantized_matmul(mixed, head.lm.w, head.lm.s, head.lm.b,
                           /*transpose=*/true, group_size, bits, "affine");
  debug_capture::get().add("final_logits", logits);
  return logits;
}

// --------------------------------------------------------------------------- //
// Generation driver.                                                           //
// --------------------------------------------------------------------------- //
struct StepRecord {
  int step;
  int token_in;
  int token_out;
  std::string digest;
};

}  // namespace

namespace {

mx::array embed_token(const DenseLoader::QTriplet& embed, int token,
                      int group_size, int bits, int hidden) {
  std::vector<int32_t> ids{static_cast<int32_t>(token)};
  mx::array idx(ids.begin(), mx::Shape{1}, mx::int32);
  mx::array w = mx::take(embed.w, idx, 0);
  mx::array s = mx::take(embed.s, idx, 0);
  mx::array b = mx::take(embed.b, idx, 0);
  if (const char* ed = std::getenv("REALGEN_DUMP_EMBEDTENSOR") ; ed && token == 93460) {
    for (auto [tag, arr] : {std::pair<const char*, mx::array>{"w", w}, {"s", s}, {"b", b}}) {
      mx::array f = mx::reshape(mx::astype(arr, mx::float32), mx::Shape{-1});
      mx::eval(f);
      char pth[512]; std::snprintf(pth, sizeof(pth), "%s.%s.f32", ed, tag);
      FILE* fp=fopen(pth,"wb"); if(fp){fwrite(f.data<float>(),sizeof(float),f.size(),fp);fclose(fp);}
    }
  }
  mx::array row = mx::dequantize(w, s, b, group_size, bits, "affine");
  return mx::reshape(row, mx::Shape{1, 1, hidden});
}

// ---- minimal safetensors header reader for streamed PLE row descriptors ----
struct StTensorRef {
  int fd = -1;
  int64_t data_offset = 0;      // absolute offset of element [0,0]
  int64_t row_bytes = 0;
  int64_t elements_per_row = 0; // prod(shape[1:])
  PLEElemType elem = PLEElemType::U32;
};
// fds are borrowed by PLEWeights for the whole run; keep them open here.
static std::vector<int> g_ple_fds;

// Parse one tensor's data_offsets[0] and shape from a flat safetensors header.
static bool st_meta(const std::string& header, const std::string& name,
                    int64_t& offset0, std::vector<int64_t>& shape,
                    std::string& dtype) {
  std::string pat = "\"" + name + "\"";
  size_t p = header.find(pat);
  if (p == std::string::npos) return false;
  size_t obj = header.find('{', p), close = header.find('}', obj);
  if (obj == std::string::npos || close == std::string::npos) return false;
  std::string b = header.substr(obj, close - obj + 1);
  // shape
  size_t sp = b.find("\"shape\""), sb = b.find('[', sp), se = b.find(']', sb);
  shape.clear();
  if (sp != std::string::npos && sb != std::string::npos) {
    std::string arr = b.substr(sb + 1, se - sb - 1);
    size_t q = 0;
    while ((q = arr.find_first_of("0123456789-", q)) != std::string::npos) {
      size_t e = q;
      if (arr[e] == '-') ++e;
      while (e < arr.size() && arr[e] >= '0' && arr[e] <= '9') ++e;
      shape.push_back(std::strtoll(arr.substr(q, e - q).c_str(), nullptr, 10));
      q = e;
    }
  }
  // data_offsets[0]
  size_t dp = b.find("\"data_offsets\""), db = b.find('[', dp), de = b.find(',', db);
  offset0 = -1;
  if (dp != std::string::npos && db != std::string::npos && de != std::string::npos)
    offset0 = std::strtoll(b.substr(db + 1, de - db - 1).c_str(), nullptr, 10);
  // dtype
  size_t yp = b.find("\"dtype\""), yq = b.find('"', yp + 8);
  if (yp != std::string::npos && yq != std::string::npos) {
    size_t ye = b.find('"', yq + 1);
    dtype = b.substr(yq + 1, ye - yq - 1);
  }
  return offset0 >= 0 && shape.size() >= 2;
}

static StTensorRef st_ref(const std::string& dir, const Config& cfg,
                          const std::string& key) {
  auto it = cfg.find("shard." + key);
  if (it == cfg.end()) throw std::runtime_error("missing shard map for PLE tensor: " + key);
  std::string path = dir + "/" + it->second;
  // Reuse an already-open fd for this path if present.
  int fd = -1;
  // (open a fresh one per call; count is small — 128 shards x 3 = 384, but
  //  we cache by path below)
  static std::map<std::string, int> fd_cache;
  auto fc = fd_cache.find(path);
  if (fc != fd_cache.end()) fd = fc->second;
  else {
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open PLE shard: " + path);
    g_ple_fds.push_back(fd);
    fd_cache[path] = fd;
  }
  std::string header;
  // Read header from this fd (data_start needed); do NOT push a second fd.
  unsigned char lenbuf[8];
  if (::pread(fd, lenbuf, 8, 0) != 8)
    throw std::runtime_error("short PLE shard header length: " + path);
  uint64_t nn = 0;
  for (int i = 0; i < 8; ++i) nn |= static_cast<uint64_t>(lenbuf[i]) << (8 * i);
  header.resize(nn);
  ::pread(fd, header.data(), nn, 8);
  int64_t data_start = 8 + static_cast<int64_t>(nn);

  StTensorRef r;
  r.fd = fd;
  std::vector<int64_t> shape;
  std::string dtype;
  int64_t off = -1;
  if (!st_meta(header, key, off, shape, dtype))
    throw std::runtime_error("cannot parse PLE tensor: " + key);
  r.data_offset = data_start + off;
  int64_t per = 1;
  for (size_t d = 1; d < shape.size(); ++d) per *= shape[d];
  r.elements_per_row = per;
  int itemsize = (dtype == "U32") ? 4 : (dtype == "BF16" || dtype == "F16") ? 2 : 4;
  r.row_bytes = per * itemsize;
  r.elem = (dtype == "U32") ? PLEElemType::U32
           : (dtype == "F32") ? PLEElemType::F32
           : (dtype == "F16") ? PLEElemType::F16
           : PLEElemType::BF16;
  return r;
}

// Row count (shape[0]) of a tensor from its shard header.
static int64_t st_rows(const std::string& dir, const Config& cfg,
                       const std::string& key) {
  auto it = cfg.find("shard." + key);
  if (it == cfg.end()) return 0;
  int fd = ::open((dir + "/" + it->second).c_str(), O_RDONLY);
  if (fd < 0) return 0;
  unsigned char lenbuf[8];
  if (::pread(fd, lenbuf, 8, 0) != 8) { ::close(fd); return 0; }
  uint64_t nn = 0;
  for (int i = 0; i < 8; ++i) nn |= static_cast<uint64_t>(lenbuf[i]) << (8 * i);
  std::string header(nn, '\0');
  ::pread(fd, header.data(), nn, 8);
  ::close(fd);
  // Find the shape array after the tensor key.
  std::string pat = "\"" + key + "\"";
  size_t p = header.find(pat);
  if (p == std::string::npos) return 0;
  size_t sp = header.find("\"shape\"", p);
  size_t sb = sp == std::string::npos ? std::string::npos : header.find('[', sp);
  size_t se = sb == std::string::npos ? std::string::npos : header.find(']', sb);
  if (sb == std::string::npos || se == std::string::npos) return 0;
  std::string arr = header.substr(sb + 1, se - sb - 1);
  size_t q = arr.find_first_of("0123456789");
  return q == std::string::npos ? 0 : std::strtoll(arr.c_str() + q, nullptr, 10);
}

// Build the production PLE layer (layer index 1) from the flat config's shard
// map. All PLE tensors are present there (gen_real_config.py emits them).
static PLEWeights load_ple(DenseLoader& dl, const Config& cfg,
                           const std::string& ckpt_dir,
                           int group_size, int bits, float eps) {
  const std::string pb = "language_model.model.layers." +
                         std::to_string(get_int(cfg, "ple_layer_ids", 2) - 1) +
                         ".ple.";
  const int ns = get_int(cfg, "ngram_size", 3);
  const int hpn = get_int(cfg, "heads_per_ngram", 8);
  const int hc = get_int(cfg, "hc_count", 4);
  const int hs = get_int(cfg, "hidden_size", 2560);
  const int ped = get_int(cfg, "ple_embed_dim", 2560);
  const int ck = get_int(cfg, "ple_conv_kernel_size", 4);
  const int nparts = get_int(cfg, "split_ngram_parts", 128);
  const int eos = get_int(cfg, "eos_token_id", 248044);
  const int cd = ns;  // conv_dilation == ngram_size in production

  Linear key_proj = dl.linear(pb + "key_proj", ped);
  Linear value_proj = dl.linear(pb + "value_proj", ped);
  mx::array nk = dl.raw(pb + "norm_key.weight");
  mx::array nq = dl.raw(pb + "norm_query.weight");
  mx::array nc = dl.raw(pb + "norm_conv.weight");
  mx::array cw = dl.raw(pb + "conv1d.weight");
  mx::array lm = dl.raw(pb + "ple_embedding.layer_multipliers");
  mx::array ho = dl.raw(pb + "ple_embedding.ngram_heads_offsets");
  mx::array hvs = dl.raw(pb + "ple_embedding.ngram_heads_vocab_sizes");

  // Streamed n-gram embedding: read only the ~16 needed rows per lookup from
  // disk (mirrors production StreamingShardedEmbedding). Never materialize the
  // full 24 GiB table. Three different absolute offsets per shard tensor.
  std::vector<PLEStreamedShard> streamed;
  std::vector<int64_t> offsets;
  offsets.push_back(0);
  streamed.reserve(nparts);
  for (int s = 0; s < nparts; ++s) {
    std::string base = pb + "ple_embedding.ngram_embedding.shard_" +
                       std::to_string(s);
    StTensorRef wref = st_ref(ckpt_dir, cfg, base + ".weight");
    StTensorRef sref = st_ref(ckpt_dir, cfg, base + ".scales");
    StTensorRef bref = st_ref(ckpt_dir, cfg, base + ".biases");
    PLEStreamedShard d;
    d.weight = {wref.fd, wref.data_offset, wref.row_bytes, wref.elements_per_row, wref.elem};
    d.scales = {sref.fd, sref.data_offset, sref.row_bytes, sref.elements_per_row, sref.elem};
    d.biases = {bref.fd, bref.data_offset, bref.row_bytes, bref.elements_per_row, bref.elem};
    streamed.push_back(d);
  }
  // shard_offsets: all shards have equal row counts in this checkpoint.
  // Read the row count from the first shard header via its element count.
  int64_t rows0 = streamed.empty() ? 0 : st_rows(ckpt_dir, cfg,
                       pb + "ple_embedding.ngram_embedding.shard_0.weight");
  for (int s = 0; s < nparts; ++s) offsets.push_back((s + 1) * rows0);
  std::fprintf(stderr, "[load] PLE: %zu streamed embedding shards, %lld rows/shard\n",
               streamed.size(), (long long)rows0);
  int sl = (ck - 1) * cd;
  PLEWeights w(std::move(key_proj), std::move(value_proj),
               std::move(nk), std::move(nq), std::move(nc),
               std::move(cw), std::move(lm), std::move(ho), std::move(hvs),
               /*shards=*/{}, std::move(offsets),
               group_size, bits, ns, hpn, hc, hs, ped, eos, eps,
               ck, cd, sl);
  w.set_streamed_shards(std::move(streamed));
  return w;
}

std::vector<StepRecord> run_generation(const std::string& ckpt,
                                       const Config& cfg, int num_tokens) {
  const int n_layers = get_int(cfg, "n_layers");
  const int H = get_int(cfg, "hidden_size");
  const int hc_count = get_int(cfg, "hc_count");
  const int hc_lowrank = get_int(cfg, "hc_lowrank");
  const int vocab = get_int(cfg, "vocab");
  const int top_k = get_int(cfg, "top_k");
  const int group_size = get_int(cfg, "group_size", 32);
  const int bits = get_int(cfg, "bits", 4);
  int num_slots = get_int(cfg, "num_slots", 16);
  // Optional: shrink the expert slot pool to top_k (saves ~0.8 GiB anonymous).
  // Bit-exact as long as PR #2's plan_slots bookkeeping holds. Default stays 16.
  {
    const char* e = std::getenv("VEYSTRIX_CPP_EXPERT_SLOTS");
    if (e) {
      int v = std::atoi(e);
      if (v >= top_k && v <= num_slots) num_slots = v;
    }
  }
  const float eps = get_float(cfg, "eps", 1e-6f);
  const std::vector<int> prompt = split_ints(get_str(cfg, "prompt_ids"));
  const std::vector<std::string> layer_types =
      split(get_str(cfg, "layer_types"), ',');
  if (static_cast<int>(layer_types.size()) != n_layers)
    throw std::runtime_error("layer_types size != n_layers");

  DecoderConfig dcfg;
  dcfg.hidden_size = H;
  dcfg.hc_count = hc_count;
  dcfg.hc_lowrank = hc_lowrank;
  dcfg.n_layers = n_layers;
  dcfg.vocab = vocab;
  dcfg.eps = eps;
  dcfg.gdn = GDNConfig{H, get_int(cfg, "key_dim"),
                       get_int(cfg, "value_dim"), get_int(cfg, "num_k_heads"),
                       get_int(cfg, "num_v_heads"), get_int(cfg, "head_k_dim"),
                       get_int(cfg, "head_v_dim"), get_int(cfg, "conv_kernel"),
                       eps, GDNActivation::Sigmoid};

  FullAttentionConfig fa;
  fa.n_heads = get_int(cfg, "n_heads");
  fa.kv_heads = get_int(cfg, "kv_heads");
  fa.head_dim = get_int(cfg, "head_dim");
  fa.n_idx_heads = get_int(cfg, "n_idx_heads");
  fa.n_idx_kv = get_int(cfg, "n_idx_kv", 1);
  fa.idx_head_dim = get_int(cfg, "idx_head_dim");
  fa.compress_ratio = get_int(cfg, "compress_ratio", 4);
  fa.token_budget = get_int(cfg, "token_budget");
  fa.eps = eps;
  fa.rope.dim = get_int(cfg, "rope_dim");
  fa.rope.base = get_float(cfg, "rope_base", 10000.0f);
  fa.rope.style = get_str(cfg, "rope_style", "interleaved");
  fa.rope.attention_scaling = get_float(cfg, "rope_scaling", 1.0f);
  fa.rope.max_position_embeddings = get_int(cfg, "rope_max_pos", 2048);
  {
    auto sec = split(get_str(cfg, "rope_mrope"), ',');
    for (auto& x : sec) fa.rope.mrope_section.push_back(std::stoi(x));
  }

  DenseLoader dl(ckpt, cfg, group_size, bits);
  {
    const char* em = std::getenv("VEYSTRIX_CPP_DENSE_MANIFEST");
    std::string mp = em ? std::string(em) : get_str(cfg, "dense_manifest", "");
    if (!mp.empty()) dl.load_manifest(mp);
  }
  const std::string prefix_fmt = get_str(
      cfg, "expert_prefix",
      "language_model.model.layers.{L}.mlp.switch_mlp");
  auto layer_prefix = [&](int l) {
    std::string p = prefix_fmt;
    size_t pos = p.find("{L}");
    p.replace(pos, 3, std::to_string(l));
    return p;
  };

  std::vector<RealLayer> layers;
  std::vector<DecoderFullLayerCache> caches;
  layers.reserve(n_layers);
  caches.reserve(n_layers);

  const int conv_dim = 2 * dcfg.gdn.key_dim + dcfg.gdn.value_dim;
  const int ple_layer = get_int(cfg, "ple_layer_ids", 2) - 1;
  const bool use_ple = std::getenv("VEYSTRIX_CPP_NO_PLE") == nullptr &&
                       get_int(cfg, "ple_layer_ids", 0) != 0;
  std::optional<PLEWeights> ple_weights;
  if (use_ple) ple_weights = load_ple(dl, cfg, ckpt, group_size, bits, eps);

  for (int l = 0; l < n_layers; ++l) {
    const std::string b = "language_model.model.layers." + std::to_string(l) + ".";
    const bool linear = layer_types[l] == "linear_attention";

    const int hc_hidden = hc_count * H;
    // Norm gains / conv / bias / A_log have no `.scales` companion -> raw.
    mx::array anw = dl.dense(b + "attn_hyper_connection.hc_norm.weight");
    mx::array mnw = dl.dense(b + "mlp_hyper_connection.hc_norm.weight");
    // Dense projections -> keep packed (Python nn.QuantizedLinear path).
    // logical_in is the module's input dim; 0 means "derive from global bits",
    // with the manifest (when loaded) authoritative.
    Linear adw = dl.linear(b + "attn_hyper_connection.input_mix_weight_down", hc_hidden);
    Linear auw = dl.linear(b + "attn_hyper_connection.input_mix_weight_up", hc_lowrank);
    Linear aiw = dl.linear(b + "attn_hyper_connection.block_inject_weight", hc_hidden);
    Linear mdw = dl.linear(b + "mlp_hyper_connection.input_mix_weight_down", hc_hidden);
    Linear muw = dl.linear(b + "mlp_hyper_connection.input_mix_weight_up", hc_lowrank);
    Linear miw = dl.linear(b + "mlp_hyper_connection.block_inject_weight", hc_hidden);
    Linear gate = dl.linear(b + "mlp.gate.weight", 0);
    Linear sgw = dl.linear(b + "mlp.shared_expert.gate_proj", H);
    Linear suw = dl.linear(b + "mlp.shared_expert.up_proj", H);
    Linear sdw = dl.linear(b + "mlp.shared_expert.down_proj", 0);  // in = inter (shape-derived)
    Linear sglw = dl.linear(b + "mlp.shared_expert_gate.weight", H);

    std::optional<GDNWeights> gdn;
    std::optional<FullAttentionWeights> full;
    if (linear) {
      const std::string a = b + "linear_attn.";
      gdn.emplace(dl.linear(a + "in_proj_qkv", H), dl.linear(a + "in_proj_z", H),
                  dl.linear(a + "in_proj_b", H), dl.linear(a + "in_proj_a", H),
                  dl.dense(a + "conv1d.weight"), dl.dense(a + "A_log"),
                  dl.dense(a + "dt_bias"), dl.dense(a + "norm.weight"),
                  dl.linear(a + "out_proj", dcfg.gdn.value_dim));
    } else {
      const std::string a = b + "self_attn.";
      const int qdim = fa.n_heads * fa.head_dim * 2;
      const int kvdim = fa.kv_heads * fa.head_dim;
      const int idxdim = (fa.n_idx_heads + fa.n_idx_kv) * fa.idx_head_dim;
      full.emplace(dl.linear(a + "q_proj", H), dl.linear(a + "k_proj", H),
                   dl.linear(a + "v_proj", H), dl.linear(a + "o_proj", qdim / 2),
                   dl.dense(a + "q_norm.weight"), dl.dense(a + "k_norm.weight"),
                   dl.linear(a + "indexer.index_qk_proj", H),
                   dl.dense(a + "indexer.q_layernorm.weight"),
                   dl.dense(a + "indexer.k_layernorm.weight"));
    }

    // Decide whether this layer's 9 expert tensors all live in one shard.
    std::string ep = layer_prefix(l);
    std::string first_shard;
    bool split = false;
    for (const char* part : {"gate_proj.weight", "gate_proj.scales",
                             "gate_proj.biases", "up_proj.weight",
                             "up_proj.scales", "up_proj.biases",
                             "down_proj.weight", "down_proj.scales",
                             "down_proj.biases"}) {
      auto it = cfg.find("shard." + ep + "." + part);
      if (it == cfg.end()) {
        split = true;
        break;
      }
      if (first_shard.empty()) first_shard = it->second;
      else if (it->second != first_shard) split = true;
    }

    std::unique_ptr<StreamingMoE> stream;
    std::unique_ptr<CrossShardMoE> cross;
    if (split) {
      cross = std::make_unique<CrossShardMoE>(
          ckpt, cfg, ep, num_slots, group_size, bits,
          debug_capture::get().copy_cross_shard);
    } else {
      stream = std::make_unique<StreamingMoE>(ckpt, l, num_slots, group_size,
                                              bits, /*read_width=*/8, ep);
    }

    if ((l + 1) % 8 == 0 || l == n_layers - 1)
      std::fprintf(stderr, "[load] prepared layer %d/%d%s\n", l + 1, n_layers,
                   split ? " (cross-shard experts)" : "");

    layers.emplace_back(
        std::move(anw), std::move(adw), std::move(auw), std::move(aiw),
        std::move(mnw), std::move(mdw), std::move(muw), std::move(miw),
        std::move(gate), std::move(sgw), std::move(suw), std::move(sdw),
        std::move(sglw), std::move(gdn), std::move(full), std::move(stream),
        std::move(cross), std::nullopt, top_k,
        linear ? AttnKind::Linear : AttnKind::FullAttention);
    layers.back().layer_idx = l;
    layers.back().split_layer = split;
    layers.back().expert_prefix = ep;
    layers.back().expert_shard_path = ckpt + "/" + first_shard;
    if (l == ple_layer && ple_weights)
      layers.back().ple = std::move(ple_weights);

    // Python's Qwen3.5 GatedDeltaNet creates conv_state with dtype=inputs.dtype
    // (bf16 on the real checkpoint); recurrent/init state stays f32. A f32
    // conv_state promotes the whole conv->qkv->delta chain to f32 and breaks
    // bit-exactness. Keep conv_state in the activation dtype.
    GDNCacheState zero(
        mx::zeros(mx::Shape{1, dcfg.gdn.conv_kernel - 1, conv_dim}, mx::bfloat16),
        mx::zeros(mx::Shape{1, dcfg.gdn.num_v_heads, dcfg.gdn.head_v_dim,
                            dcfg.gdn.head_k_dim}, mx::float32));
    caches.emplace_back(DecoderLayerCache(std::move(zero)));
  }

  const int head_hc_hidden = hc_count * H;
  RealHead head{
      dl.dense("language_model.model.hyper_connection_mixer.hc_norm.weight"),
      dl.linear("language_model.model.hyper_connection_mixer.input_mix_weight_down", head_hc_hidden),
      dl.linear("language_model.model.hyper_connection_mixer.input_mix_weight_up", hc_lowrank),
      dl.quantized("language_model.lm_head")};
  DenseLoader::QTriplet embed = dl.quantized("language_model.model.embed_tokens");
  std::fprintf(stderr, "[load] all weights ready; starting generation\n");
  mem_telemetry("post-load");
  {
    const char* emax = std::getenv("VEYSTRIX_CPP_MAX_ACTIVE_GIB");
    double limit = emax ? std::atof(emax)
                        : std::atof(get_str(cfg, "max_active_gib", "6.0").c_str());
    if (limit > 0) {
      double active_gib = mx::get_active_memory() / (1024.0 * 1024.0 * 1024.0);
      if (active_gib > limit) {
        throw std::runtime_error(
            "[load] ABORT: post-load active " + std::to_string(active_gib) +
            " GiB exceeds --max-active-gib " + std::to_string(limit) +
            " GiB (bad build / packed path not active)");
      }
    }
  }
  // Load-only mode (gate 2): stop after weights are loaded + telemetry, before
  // any decode. Lets the owner measure post-load active memory under the guard
  // without the (now unnecessary) multi-minute generation.
  if (std::getenv("VEYSTRIX_CPP_LOAD_ONLY")) {
    std::fprintf(stderr, "[load] LOAD-ONLY: weights loaded, no decode run\n");
    return {};
  }

  std::vector<StepRecord> records;
  records.reserve(num_tokens);

  // Prefill: sequential single-token steps through the real stack. The logits
  // after the last prompt token produce the first generated token.
  const bool dbg_requested = std::getenv("REALGEN_DEBUG_HASHES") != nullptr;
  // Opt-in fix: tool-local GDN path (ops delta step + state materialization)
  // for every prefill/decode step when the debug flag is set.
  debug_capture::get().use_local_gdn = dbg_requested;
  // Opt-in fix: deep-copy cross-shard expert rows into fresh owned arrays each
  // step (mirrors Python's fresh per-step buffer) instead of reusing buf_[i].
  debug_capture::get().copy_cross_shard =
      std::getenv("REALGEN_CROSS_SHARD_COPY") != nullptr;
  debug_capture::get().moe_hash = std::getenv("REALGEN_DEBUG_MOE") != nullptr;
  int pos = 0;
  std::optional<mx::array> logits;
  const bool seq_prefill = std::getenv("VEYSTRIX_CPP_SEQ_PREFILL") != nullptr;
  if (!seq_prefill && prompt.size() > 1) {
    // Batched prefill: ONE call feeds all S prompt tokens through every layer,
    // matching production language(ids, cache). Eliminates the sequential-S=1
    // ULP drift that amplifies through MoE routing. (Synthetic bit-exact vs Python.)
    std::vector<BatchedLayer> bl;
    bl.reserve(n_layers);
    for (int l = 0; l < n_layers; ++l) {
      RealLayer& R = layers[l];
      bl.emplace_back(R.attn_norm_w, R.attn_down_w, R.attn_up_w, R.attn_inject_w,
                      R.mlp_norm_w, R.mlp_down_w, R.mlp_up_w, R.mlp_inject_w,
                      R.gate_w, R.shared_gw, R.shared_uw, R.shared_dw,
                      R.shared_glw, R.top_k, R.kind);
      BatchedLayer& B = bl.back();
      if (R.gdn) B.gdn = R.gdn;
      if (R.full) B.full = R.full;
      if (R.ple) { B.ple = R.ple; B.ple_cache = R.ple_cache; }
      B.moe_forward = [&R, &ckpt, &cfg, group_size, bits, top_k](const mx::array& x, const mx::array& idx,
                           const mx::array& sc) {
        // MoE is per-token independent (gather_qmm gathers each row's experts
        // separately), so process one token at a time to stay within the
        // num_slots=16 slot pool. This is bit-exact regardless of batching.
        int S = static_cast<int>(x.shape(0));
        // Bit-exact prefill: Python sorts all S*K indices and runs one batched
        // sorted gather_qmm, which rounds differently from M=1 per-token calls
        // at large S. When enabled, build a large resident pool for THIS layer
        // on the fly (created and destroyed within the call, so only one layer's
        // experts are resident at a time ~= 1.3 GiB), process the whole prompt
        // at once so StreamingMoE::forward picks moe_block_sorted, then eval.
        // Bit-exact batched prefill uses the sorted first-occurrence expert
        // stack by default (Python SwitchGLU behavior at indices.size>=64).
        // Opt out with VEYSTRIX_CPP_UNSORTED_PREFILL for A/B.
        static const bool unsorted_prefill =
            std::getenv("VEYSTRIX_CPP_UNSORTED_PREFILL") != nullptr;
        if (S > 1 && !unsorted_prefill) {
          // Build a compact first-occurrence expert stack (only the distinct
          // routed rows are read), exactly matching Python SwitchGLU's weight
          // stack + sorted gather_qmm. Handles both single- and cross-shard.
          if (R.split_layer) {
            CrossShardMoE pf(ckpt, cfg, R.expert_prefix, top_k, group_size,
                             bits, false);
            return pf.forward_stacked(x, idx, sc);
          }
          StreamingMoE pf(ckpt, R.layer_idx, top_k, group_size, bits,
                          /*read_width=*/8, R.expert_prefix);
          return pf.forward_stacked(x, idx, sc);
        }
        if (S <= 1)
          return R.stream ? R.stream->forward(x, idx, sc).output
                          : R.cross->forward(x, idx, sc).output;
        std::vector<mx::array> outs;
        outs.reserve(S);
        for (int t = 0; t < S; ++t) {
          std::vector<int32_t> ti{t};
          mx::array rowidx(ti.data(), mx::Shape{1}, mx::int32);
          mx::array xt = mx::take(x, rowidx, 0);
          mx::array it = mx::take(idx, rowidx, 0);
          mx::array st = mx::take(sc, rowidx, 0);
          outs.push_back(R.stream ? R.stream->forward(xt, it, st).output
                                  : R.cross->forward(xt, it, st).output);
        }
        return mx::concatenate(outs, 0);
      };
    }
    BatchedHead bh(head.norm_w, head.down_w, head.up_w,
                   BatchedLMTriplet(head.lm.w, head.lm.s, head.lm.b));
    std::vector<int32_t> ids(prompt.begin(), prompt.end());
    mx::array ids_arr(ids.data(), mx::Shape{1, static_cast<int>(ids.size())}, mx::int32);
    auto embed_all = [&](const mx::array& ida) {
      mx::array f = mx::reshape(ida, mx::Shape{-1});
      mx::array w = mx::take(embed.w, f, 0), s = mx::take(embed.s, f, 0),
                b = mx::take(embed.b, f, 0);
      mx::array row = mx::dequantize(w, s, b, group_size, bits, "affine");
      // Materialize the gathered+dequantized embeddings before the prefill
      // graph consumes them (avoids a lazy dequantize read racing the GPU at
      // long context; matches Python's eagerly-evaluated parameters).
      mx::eval(row);
      static bool printed = false;
      if (!printed) {
        std::fprintf(stderr, "[embed] w=%ld s=%ld b=%ld dequant=%ld (bf16=%ld f32=%ld)\n",
                     (long)static_cast<mx::Dtype::Val>(w.dtype()),
                     (long)static_cast<mx::Dtype::Val>(s.dtype()),
                     (long)static_cast<mx::Dtype::Val>(b.dtype()),
                     (long)static_cast<mx::Dtype::Val>(row.dtype()),
                     (long)static_cast<mx::Dtype::Val>(mx::bfloat16),
                     (long)static_cast<mx::Dtype::Val>(mx::float32));
        printed = true;
      }
      return mx::reshape(row, mx::Shape{1, static_cast<int>(f.size()), H});
    };
    BatchedPrefillDebug bpdbg;
    // Always collect per-layer hidden states and force them through an f32
    // materialization barrier. MLX's lazy scheduler otherwise reorders the
    // compiled-silu graphs across the deep 48-layer chain nondeterministically
    // (gate (a)); evaluating each layer's hidden in order serializes the graph
    // and makes the prefill bit-exact. This is always on (not just for dumps).
    logits = run_prefill_batched(ids_arr, embed_all, bl, bh, dcfg, fa, caches,
                                 group_size, bits, &bpdbg);
    for (size_t l = 0; l < bpdbg.hidden_after_layer.size(); ++l) {
      mx::eval(mx::astype(bpdbg.hidden_after_layer[l], mx::float32));
      if (l < bpdbg.branch.size())
        mx::eval(mx::astype(bpdbg.branch[l], mx::float32));
      if (l < bpdbg.m2d.size())
        mx::eval(mx::astype(bpdbg.m2d[l], mx::float32));
      if (l < bpdbg.gate_logits.size())
        mx::eval(mx::astype(bpdbg.gate_logits[l], mx::float32));
      if (l < bpdbg.moe2d.size())
        mx::eval(mx::astype(bpdbg.moe2d[l], mx::float32));
      if (l < bpdbg.mlp_inj.size())
        mx::eval(mx::astype(bpdbg.mlp_inj[l], mx::float32));
      if (l < bpdbg.gdn_in.size())
        mx::eval(mx::astype(bpdbg.gdn_in[l], mx::float32));
    }
    if (const char* hd = std::getenv("REALGEN_DUMP_BATCH_LAYERS")) {
      for (size_t l = 0; l < bpdbg.hidden_after_layer.size(); ++l) {
        mx::array hf = mx::reshape(mx::astype(bpdbg.hidden_after_layer[l], mx::float32), mx::Shape{-1});
        mx::eval(hf);
        char pth[512]; std::snprintf(pth, sizeof(pth), "%s.L%02zu.f32", hd, l);
        FILE* fp=fopen(pth,"wb"); if(fp){fwrite(hf.data<float>(),sizeof(float),hf.size(),fp);fclose(fp);}
        if (l < bpdbg.branch.size()) {
          mx::array bf = mx::reshape(mx::astype(bpdbg.branch[l], mx::float32), mx::Shape{-1});
          mx::eval(bf);
          std::snprintf(pth, sizeof(pth), "%s.branch.L%02zu.f32", hd, l);
          FILE* bfp=fopen(pth,"wb"); if(bfp){fwrite(bf.data<float>(),sizeof(float),bf.size(),bfp);fclose(bfp);}
        }
        if (l < bpdbg.m2d.size()) {
          mx::array mf = mx::reshape(mx::astype(bpdbg.m2d[l], mx::float32), mx::Shape{-1});
          mx::eval(mf);
          std::snprintf(pth, sizeof(pth), "%s.m2d.L%02zu.f32", hd, l);
          FILE* mfp=fopen(pth,"wb"); if(mfp){fwrite(mf.data<float>(),sizeof(float),mf.size(),mfp);fclose(mfp);}
        }
        if (l < bpdbg.gate_logits.size()) {
          mx::array gf = mx::reshape(mx::astype(bpdbg.gate_logits[l], mx::float32), mx::Shape{-1});
          mx::eval(gf);
          std::snprintf(pth, sizeof(pth), "%s.gate.L%02zu.f32", hd, l);
          FILE* gfp=fopen(pth,"wb"); if(gfp){fwrite(gf.data<float>(),sizeof(float),gf.size(),gfp);fclose(gfp);}
        }
        if (l < bpdbg.moe2d.size()) {
          mx::array mf = mx::reshape(mx::astype(bpdbg.moe2d[l], mx::float32), mx::Shape{-1});
          mx::eval(mf);
          std::snprintf(pth, sizeof(pth), "%s.moe2d.L%02zu.f32", hd, l);
          FILE* mfp=fopen(pth,"wb"); if(mfp){fwrite(mf.data<float>(),sizeof(float),mf.size(),mfp);fclose(mfp);}
        }
        if (l < bpdbg.mlp_inj.size()) {
          mx::array mf = mx::reshape(mx::astype(bpdbg.mlp_inj[l], mx::float32), mx::Shape{-1});
          mx::eval(mf);
          std::snprintf(pth, sizeof(pth), "%s.mlpinj.L%02zu.f32", hd, l);
          FILE* mfp=fopen(pth,"wb"); if(mfp){fwrite(mf.data<float>(),sizeof(float),mf.size(),mfp);fclose(mfp);}
        }
        if (l < bpdbg.gdn_in.size()) {
          mx::array gf = mx::reshape(mx::astype(bpdbg.gdn_in[l], mx::float32), mx::Shape{-1});
          mx::eval(gf);
          std::snprintf(pth, sizeof(pth), "%s.gdnin.L%02zu.f32", hd, l);
          FILE* gfp=fopen(pth,"wb"); if(gfp){fwrite(gf.data<float>(),sizeof(float),gf.size(),gfp);fclose(gfp);}
        }
      }
    }
    if (std::getenv("REALGEN_DUMP_FA")) {
      const char* fad = std::getenv("REALGEN_DUMP_FA_DIR");
      if (fad) {
        FADump f = get_fa_dump();
        if (f.ok) {
          const char* names[] = {"qproj", "qnorm", "qrope", "krope", "sdpa"};
          const std::optional<mx::array> arrs[] = {f.qproj, f.qnorm, f.qrope, f.krope, f.sdpa};
          for (int i = 0; i < 5; ++i) {
            if (!arrs[i]) continue;
            mx::array af = mx::reshape(mx::astype(*arrs[i], mx::float32), mx::Shape{-1});
            mx::eval(af);
            char pth[512]; std::snprintf(pth, sizeof(pth), "%s/%s.f32", fad, names[i]);
            FILE* fp=fopen(pth,"wb"); if(fp){fwrite(af.data<float>(),sizeof(float),af.size(),fp);fclose(fp);}
          }
        }
      }
    }
    // copy back mutable PLE caches
    for (int l = 0; l < n_layers; ++l)
      if (bl[l].ple) layers[l].ple_cache = bl[l].ple_cache;
    pos = static_cast<int>(prompt.size());
    mem_telemetry("prefill-batched");
  } else {
    for (size_t i = 0; i < prompt.size(); ++i) {
      std::vector<int32_t> pd{static_cast<int32_t>(pos++)};
      mx::array pos_ids(pd.begin(), mx::Shape{1, 1}, mx::int32);
      debug_capture::get().current_step = static_cast<int>(i);
      logits = run_step(embed_token(embed, prompt[i], group_size, bits, H),
                        static_cast<int32_t>(prompt[i]),
                        layers, head, dcfg, fa, pos_ids, caches, group_size, bits);
      mem_telemetry("prefill-step");
      if (dbg_requested) dump_gdn_states(caches, static_cast<int>(i), n_layers);
    }
  }
  if (dbg_requested) {
    auto& dbg = debug_capture::get();
    std::fprintf(stderr, "[DBG] captured %zu arrays\n", dbg.arrays.size());
    for (size_t k = 0; k < dbg.arrays.size(); ++k) {
      mx::array f = mx::reshape(mx::astype(dbg.arrays[k], mx::float32), mx::Shape{-1});
      mx::eval(f);
      const float* p = f.data<float>();
      size_t nan = 0, inf = 0;
      for (size_t j = 0; j < f.size(); ++j) {
        if (std::isnan(p[j])) ++nan;
        else if (std::isinf(p[j])) ++inf;
      }
      std::fprintf(stderr, "[DBG %03zu] %-20s sha=%s nan=%zu inf=%zu\n", k,
                   dbg.names[k].c_str(),
                   sha256_hex(p, f.size() * sizeof(float)).c_str(), nan, inf);
    }
    dbg.clear();
  }
  if (!logits.has_value())
    throw std::runtime_error("empty prompt: no prefill logits");
  int token = prompt.empty() ? 0 : prompt.back();

  // Greedy decode steps. The prefill logits already give us step 0.
  for (int step = 0; step < num_tokens; ++step) {
    debug_capture::get().current_step = 100 + step;  // never collides with prefill 0..15
    if (step > 0) {
      std::vector<int32_t> pd{static_cast<int32_t>(pos++)};
      mx::array pos_ids(pd.begin(), mx::Shape{1, 1}, mx::int32);
      logits = run_step(embed_token(embed, token, group_size, bits, H),
                        static_cast<int32_t>(token), layers,
                        head, dcfg, fa, pos_ids, caches, group_size, bits);
      mem_telemetry("decode-step");
    }

    mx::array next = sample_greedy(*logits);
    mx::eval(next);
    int next_id = static_cast<int>(next.data<uint32_t>()[0]);

    mx::array row = mx::reshape(mx::astype(*logits, mx::float32), mx::Shape{-1});
    mx::eval(row);
    std::string digest = sha256_hex(
        row.data<float>(), static_cast<size_t>(row.size()) * sizeof(float));

    // Opt-in full fp32 logit dump (numeric comparison vs Python). No effect off.
    if (const char* df = std::getenv("REALGEN_DUMP_LOGITS")) {
      char pth[512];
      std::snprintf(pth, sizeof(pth), "%s.step%d.f32", df, step);
      std::FILE* f = std::fopen(pth, "wb");
      if (f) {
        std::fwrite(row.data<float>(), sizeof(float), row.size(), f);
        std::fclose(f);
      }
    }

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
    // Derive <reference-dir>/real_config.txt (written by gen_real_config.py).
    size_t slash = reference.find_last_of('/');
    config_path = (slash == std::string::npos ? std::string(".")
                                              : reference.substr(0, slash)) +
                  "/real_config.txt";
  }
  try {
    Config cfg = read_config(config_path);
    if (cfg.empty())
      throw std::runtime_error(
          "empty flat config; run gen_real_config.py first: " + config_path);
    if (get_int(cfg, "prompt_tokens", 0) == 0)
      throw std::runtime_error("flat config has no prompt_ids");
    std::vector<StepRecord> records = run_generation(ckpt, cfg, num_tokens);
    for (const auto& r : records)
      std::printf("step=%d in=%d out=%d sha256=%s\n", r.step, r.token_in,
                  r.token_out, r.digest.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "real_checkpoint_generate failed: %s\n", e.what());
    return 1;
  }
}
