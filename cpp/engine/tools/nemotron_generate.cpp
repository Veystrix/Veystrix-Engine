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
#include "fd_hints.hpp"
#include "gpu_keepwarm.hpp"
#include "gpu_gate.hpp"
#include "row_fetch.hpp"     // VEYSTRIX_CPP_NEMO_PIPE=1 decode row fetch
#include "engine.hpp"
#include "mamba_layer.hpp"
#include "nemo_attention.hpp"
#include "model_adapter.hpp"
#include "cpp_profile.hpp"  // phase 0 profiler (single branch when off)
#include "nemo_pool.hpp"     // Phase 1.1 opt-in expert pool (VEYSTRIX_CPP_NEMO_POOL=1)
#include "memory_budget.hpp" // dynamic-memory gate (C ABI runtime only)
// io_pager.hpp lives outside cpp/engine/src; include it by relative path so no
// CMake change (and no new binary) is needed. It provides the existing 8-lane
// batched pread used to fill pool slots.
#include "../../decode/src/io_pager.hpp"
#include "../../decode/src/io_pager_async.hpp"

#include <mlx/mlx.h>
#include <mlx/memory.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <exception>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
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
// Phase E-N (Nemotron TTFT) opt-in flags. All default off; rollback = unset.     //
//                                                                               //
// E1-N  VEYSTRIX_CPP_NEMOTRON_SHARD_INDEX=1                                      //
// E2-N  VEYSTRIX_CPP_NEMOTRON_REUSE_PREFILL=1                                     //
//                                                                               //
// NOTE (honest scope): Qwen's E1 caches `model.safetensors.index.json`'s          //
// tensor->shard weight_map because Qwen *discovers* shards by scanning headers.   //
// Nemotron's flat config already carries that map (`shard.<name>=<file>`, see      //
// `gen_nemotron_config.py`), so there is no index.json to parse. The redundant     //
// work here is different: `ExpertReader::info()` re-reads and re-scans the WHOLE    //
// safetensors header of a shard for every distinct tensor in it (~6 expert tensors //
// per MoE layer, ~40 MoE layers, only 4 shards). E1-N caches the parsed header     //
// once per SHARD instead. Same header bytes + same parser => bit-exact.           //
//                                                                               //
// E2-N: the Nemotron ExpertReader is ALREADY one long-lived object shared by all   //
// run_step calls (prefill + decode), so Qwen-E2's "rebuild a reader per prefill"    //
// bug does not exist here. The only per-read allocation is the temporary row       //
// buffer, so E2-N reuses that buffer across prefill steps (see rows_* below).      //
// --------------------------------------------------------------------------- //
bool nemo_shard_index_from_env() {
  const char* v = std::getenv("VEYSTRIX_CPP_NEMOTRON_SHARD_INDEX");
  if (v == nullptr || std::strcmp(v, "0") == 0) return false;
  if (std::strcmp(v, "1") == 0) return true;
  throw std::runtime_error("VEYSTRIX_CPP_NEMOTRON_SHARD_INDEX must be 0 or 1");
}
bool nemo_reuse_prefill_from_env() {
  const char* v = std::getenv("VEYSTRIX_CPP_NEMOTRON_REUSE_PREFILL");
  if (v == nullptr || std::strcmp(v, "0") == 0) return false;
  if (std::strcmp(v, "1") == 0) return true;
  throw std::runtime_error("VEYSTRIX_CPP_NEMOTRON_REUSE_PREFILL must be 0 or 1");
}
bool nemo_shard_index_enabled() {
  static const bool v = nemo_shard_index_from_env();
  return v;
}

// E1-N: parsed safetensors header, cached once per shard file (process-wide).
// The generator is single-threaded and the pool preloads all headers in its
// constructor, so no lock is needed: lookups become read-only after load.
struct ParsedShardHeader {
  std::string header;  // raw header JSON bytes (the [8, 8+nn) region)
  uint64_t nn = 0;     // header length from the 8-byte little-endian prefix
};
const ParsedShardHeader& parsed_shard_header(const std::string& path) {
  static std::map<std::string, ParsedShardHeader> cache;
  auto found = cache.find(path);
  if (found != cache.end()) return found->second;
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open shard header: " + path);
  unsigned char lenbuf[8];
  if (::pread(fd, lenbuf, 8, 0) != 8) {
    ::close(fd);
    throw std::runtime_error("short safetensors header length: " + path);
  }
  uint64_t nn = 0;
  for (int i = 0; i < 8; ++i) nn |= static_cast<uint64_t>(lenbuf[i]) << (8 * i);
  ParsedShardHeader sh;
  sh.nn = nn;
  sh.header.resize(nn);
  if (nn && ::pread(fd, sh.header.data(), nn, 8) != static_cast<ssize_t>(nn)) {
    ::close(fd);
    throw std::runtime_error("short safetensors header: " + path);
  }
  ::close(fd);
  auto res = cache.emplace(path, std::move(sh));
  return res.first->second;
}

// --------------------------------------------------------------------------- //
// Routed-expert row reader: pread ONLY selected rows out of the stacked tensors. //
// --------------------------------------------------------------------------- //
class ExpertReader {
 public:
  ExpertReader(const std::string& dir, const Config& cfg)
      : dir_(dir), cfg_(cfg), shard_index_(nemo_shard_index_enabled()) {}

  // E2-N: reuse the per-read row buffer across prefill steps instead of
  // allocating/freeing one for every one of a layer's six tensors.
  void set_reuse_scratch(bool v) { reuse_scratch_ = v; }

  // Prefill / baseline row fetch. Default: ONE batched read of every requested
  // expert row (the same 8-lane pager the pool fills use) straight into a
  // page-aligned buffer that MLX then owns. The serial path below issued one
  // pread per row on a single thread, zero-filled the destination and copied it
  // again into MLX; in the server's chunked prefill that was ~72% of the time
  // (profile 20 Nov). Bytes are identical. VEYSTRIX_CPP_NEMO_ROWS=serial
  // restores the previous implementation.
  static bool rows_batched() {
    static const bool on = [] {
      const char* e = std::getenv("VEYSTRIX_CPP_NEMO_ROWS");
      return !(e && std::strcmp(e, "serial") == 0);
    }();
    return on;
  }

  mx::array rows_batched_owned(int fd, int64_t data_offset, int64_t d1, int64_t d2,
                               const std::string& name,
                               const std::vector<int>& ids, size_t elem,
                               mx::Dtype wire) {
    const size_t per = static_cast<size_t>(d1 * d2);
    const size_t row_bytes = per * elem;
    const size_t bytes = row_bytes * ids.size();
    void* raw = nullptr;
    if (::posix_memalign(&raw, 16384, bytes ? bytes : 1) != 0)
      throw std::runtime_error("rows: posix_memalign failed: " + name);
    auto& _pf = cppprof::profiler();
    auto _t = _pf.now();
    // Consecutive expert ids are adjacent in the file AND in the destination
    // buffer, so they merge into one larger sequential read (capped so a few
    // long runs still spread over all lanes). VEYSTRIX_CPP_NEMO_ROWS_COALESCE=0
    // keeps one read per row; VEYSTRIX_CPP_NEMO_ROWS_WIDTH sets the lane count.
    static const size_t max_run = [] {
      const char* e = std::getenv("VEYSTRIX_CPP_NEMO_ROWS_COALESCE");
      if (e && std::strcmp(e, "0") == 0) return size_t{0};
      const long mib = e ? std::atol(e) : 8;       // value = cap in MiB (1 = "on", treated as 8)
      return static_cast<size_t>(mib > 1 ? mib : 8) << 20;
    }();
    static const int width = [] {
      const char* e = std::getenv("VEYSTRIX_CPP_NEMO_ROWS_WIDTH");
      const int w = e ? std::atoi(e) : 8;
      return w >= 1 && w <= 64 ? w : 8;
    }();
    std::vector<veystrix::PagerRead> reads;
    reads.reserve(ids.size());
    size_t i = 0;
    while (i < ids.size()) {
      size_t j = i + 1;
      if (max_run > 0)
        while (j < ids.size() && ids[j] == ids[j - 1] + 1 &&
               (j + 1 - i) * row_bytes <= max_run)
          ++j;
      veystrix::PagerRead pr;
      pr.fd = fd;
      pr.pad = 0;
      pr.offset = data_offset + static_cast<int64_t>(ids[i]) * static_cast<int64_t>(row_bytes);
      pr.length = static_cast<int64_t>((j - i) * row_bytes);
      pr.dst = reinterpret_cast<uint64_t>(static_cast<uint8_t*>(raw) + i * row_bytes);
      reads.push_back(pr);
      i = j;
    }
    if (!reads.empty() && veystrix::run_batched_reads(reads, width) != 0) {
      std::free(raw);
      throw std::runtime_error("short expert row read: " + name);
    }
    _pf.add_read(cppprof::ms_since(_t), static_cast<uint64_t>(bytes));
    return mx::array(raw, mx::Shape{static_cast<int>(ids.size()),
                                    static_cast<int>(d1), static_cast<int>(d2)},
                     wire, [](void* p) { std::free(p); });
  }

  mx::array rows_u32(const std::string& name, const std::vector<int>& ids) {
    Info inf = info(name);
    if (inf.dtype != "U32")
      throw std::runtime_error("rows_u32: tensor is not U32: " + name);
    if (rows_batched()) return rows_batched_owned(inf.fd, inf.data_offset, inf.shape[1], inf.shape[2], name, ids, 4, mx::uint32);
    const size_t per = static_cast<size_t>(inf.shape[1] * inf.shape[2]);
    auto& _pf = cppprof::profiler();
    auto _t = _pf.now();
    std::vector<uint32_t> local_buf;  // used only when !reuse_scratch_
    std::vector<uint32_t>& buf = reuse_scratch_ ? scratch32_ : local_buf;
    buf.assign(per * ids.size(), 0u);
    for (size_t i = 0; i < ids.size(); ++i) {
      off_t off = inf.data_offset + static_cast<off_t>(ids[i]) * per * 4;
      if (::pread(inf.fd, buf.data() + i * per, per * 4, off) !=
          static_cast<ssize_t>(per * 4))
        throw std::runtime_error("short expert row read: " + name);
    }
    _pf.add_read(cppprof::ms_since(_t), (uint64_t)(per * ids.size() * 4));
    return mx::array(buf.begin(), mx::Shape{static_cast<int>(ids.size()),
                                            static_cast<int>(inf.shape[1]),
                                            static_cast<int>(inf.shape[2])},
                     mx::uint32);
  }

  mx::array rows_bf16(const std::string& name, const std::vector<int>& ids) {
    Info inf = info(name);
    if (inf.dtype != "BF16")
      throw std::runtime_error("rows_bf16: tensor is not BF16: " + name);
    if (rows_batched()) {
      mx::array a = rows_batched_owned(inf.fd, inf.data_offset, inf.shape[1], inf.shape[2], name, ids, 2, mx::uint16);
      return mx::view(a, mx::bfloat16);
    }
    const size_t per = static_cast<size_t>(inf.shape[1] * inf.shape[2]);
    auto& _pf = cppprof::profiler();
    auto _t = _pf.now();
    std::vector<uint16_t> local_buf;  // used only when !reuse_scratch_
    std::vector<uint16_t>& buf = reuse_scratch_ ? scratch16_ : local_buf;
    buf.assign(per * ids.size(), 0u);
    for (size_t i = 0; i < ids.size(); ++i) {
      off_t off = inf.data_offset + static_cast<off_t>(ids[i]) * per * 2;
      if (::pread(inf.fd, buf.data() + i * per, per * 2, off) !=
          static_cast<ssize_t>(per * 2))
        throw std::runtime_error("short expert row read: " + name);
    }
    _pf.add_read(cppprof::ms_since(_t), (uint64_t)(per * ids.size() * 2));
    mx::array a(buf.begin(), mx::Shape{static_cast<int>(ids.size()),
                                       static_cast<int>(inf.shape[1]),
                                       static_cast<int>(inf.shape[2])},
                mx::uint16);
    return mx::view(a, mx::bfloat16);
  }

  // ---- Decode: all six tensors of a layer's routed experts in ONE batch ------
  // The default decode path fetched fc1/fc2 x weight/scales/biases with six
  // consecutive synchronous batches (six round trips per MoE layer, the two
  // large weight rounds each using only 6 of 8 lanes). Here every row of every
  // tensor is one read in a single 8-lane batch (consecutive ids coalesced),
  // optionally with the GPU kept warm while it waits. VEYSTRIX_CPP_NEMO_ONEBATCH=1.
  //
  // VEYSTRIX_CPP_MLX_DEST=1: destinations come from MLX's own allocator
  // (recycled, already GPU-resident Metal buffers) instead of fresh
  // posix_memalign memory that MLX must wrap as a new no-copy Metal buffer on
  // every layer. Same bytes either way.
  static bool one_batch() {
    static const bool on = [] {
      const char* e = std::getenv("VEYSTRIX_CPP_NEMO_ONEBATCH");
      return e && std::strcmp(e, "1") == 0;
    }();
    return on;
  }
  static bool mlx_dest() {
    static const bool on = [] {
      const char* e = std::getenv("VEYSTRIX_CPP_MLX_DEST");
      return e && std::strcmp(e, "1") == 0;
    }();
    return on;
  }

  std::array<mx::array, 6> rows6(const std::array<std::string, 6>& names,
                                 const std::vector<int>& ids) {
    auto& _pf = cppprof::profiler();
    const bool use_mlx = mlx_dest();
    struct Dest {
      mx::allocator::Buffer mbuf{nullptr};
      void* raw = nullptr;
      int64_t d1 = 0, d2 = 0;
      bool u32 = true;
    };
    std::array<Dest, 6> dst;
    std::vector<veystrix::PagerRead> reads;
    reads.reserve(36);
    size_t total = 0;
    try {
      for (int k = 0; k < 6; ++k) {
        Info inf = info(names[k]);
        Dest& d = dst[k];
        d.u32 = inf.dtype == "U32";
        if (!d.u32 && inf.dtype != "BF16")
          throw std::runtime_error("rows6: unexpected dtype for " + names[k]);
        d.d1 = inf.shape[1];
        d.d2 = inf.shape[2];
        const size_t row_bytes = static_cast<size_t>(d.d1 * d.d2) * (d.u32 ? 4 : 2);
        const size_t bytes = row_bytes * ids.size();
        if (use_mlx) {
          d.mbuf = mx::allocator::malloc(bytes ? bytes : 1);
          d.raw = d.mbuf.raw_ptr();
        } else if (::posix_memalign(&d.raw, 16384, bytes ? bytes : 1) != 0) {
          d.raw = nullptr;
          throw std::runtime_error("rows6: posix_memalign failed");
        }
        size_t i = 0;
        while (i < ids.size()) {
          size_t j = i + 1;
          while (j < ids.size() && ids[j] == ids[j - 1] + 1) ++j;
          veystrix::PagerRead pr;
          pr.fd = inf.fd;
          pr.pad = 0;
          pr.offset = inf.data_offset + static_cast<int64_t>(ids[i]) * static_cast<int64_t>(row_bytes);
          pr.length = static_cast<int64_t>((j - i) * row_bytes);
          pr.dst = reinterpret_cast<uint64_t>(static_cast<uint8_t*>(d.raw) + i * row_bytes);
          reads.push_back(pr);
          i = j;
        }
        total += bytes;
      }
      auto _t = _pf.now();
      if (run_expert_reads(reads, 8) != 0)
        throw std::runtime_error("rows6: short expert row read");
      _pf.add_read(cppprof::ms_since(_t), static_cast<uint64_t>(total));
    } catch (...) {
      for (auto& d : dst) {
        if (use_mlx) { if (d.mbuf.ptr()) mx::allocator::free(d.mbuf); }
        else if (d.raw) std::free(d.raw);
      }
      throw;
    }
    auto make = [&](int k) {
      Dest& d = dst[k];
      const mx::Shape shape{static_cast<int>(ids.size()), static_cast<int>(d.d1),
                            static_cast<int>(d.d2)};
      const mx::Dtype wire = d.u32 ? mx::uint32 : mx::uint16;
      mx::array a = use_mlx ? mx::array(d.mbuf, shape, wire)
                            : mx::array(d.raw, shape, wire, [](void* p) { std::free(p); });
      return d.u32 ? a : mx::view(a, mx::bfloat16);
    };
    return {make(0), make(1), make(2), make(3), make(4), make(5)};
  }

  struct Info {
    int fd = -1;
    int64_t data_offset = 0;
    std::vector<int64_t> shape;
    std::string dtype;
  };

  // ---- Decode: pipelined routed experts (VEYSTRIX_CPP_NEMO_PIPE=1) ----------
  // Same rows and the same stacked MLX buffers as rows6(), but the reads run on
  // rowfetch::Fetch (page-cache hits copied from a shared mmap, misses pread)
  // with one completion group per routed expert. Each expert's fc1 -> relu2 ->
  // fc2 is launched on the GPU as soon as its six rows have landed, while the
  // other experts' rows (usually the SSD misses) are still being read. The
  // per-expert gather_qmm over a one-row slice computes the same values as the
  // stacked gather_qmm (bitwise, checked on the real checkpoint); the outputs
  // are put back in route order and combined exactly as moe_block_relu2 does.
  static bool pipe_enabled() {
    static const bool on = [] {
      const char* e = std::getenv("VEYSTRIX_CPP_NEMO_PIPE");
      return e && std::strcmp(e, "1") == 0;
    }();
    return on;
  }

  mx::array pipelined_moe(const std::array<std::string, 6>& names, const std::vector<int>& unique,
                          const std::vector<int>& local, const mx::array& x2, const mx::array& scores) {
    auto& _pf = cppprof::profiler();
    auto _t = _pf.now();
    const int n = static_cast<int>(unique.size());
    std::vector<veystrix::engine::rowfetch::Row> rows;
    rows.reserve(static_cast<size_t>(6 * n));
    std::vector<mx::array> stack;  // owns the six destination buffers
    stack.reserve(6);
    size_t total = 0;
    const bool use_map = veystrix::engine::rowfetch::mmap_enabled();
    // Tensor-major order (all fc1 weight rows first, ...), like rows6: the big
    // rows start first, so the misses among them start reading at once.
    for (int k = 0; k < 6; ++k) {
      const Info inf = info(names[k]);
      const bool u32 = inf.dtype == "U32";
      if (!u32 && inf.dtype != "BF16")
        throw std::runtime_error("pipelined_moe: unexpected dtype for " + names[k]);
      const size_t row_bytes = static_cast<size_t>(inf.shape[1] * inf.shape[2]) * (u32 ? 4 : 2);
      mx::allocator::Buffer b = mx::allocator::malloc(row_bytes * static_cast<size_t>(n));
      uint8_t* base = static_cast<uint8_t*>(b.raw_ptr());
      mx::array a(b, mx::Shape{n, static_cast<int>(inf.shape[1]), static_cast<int>(inf.shape[2])},
                  u32 ? mx::uint32 : mx::uint16);
      stack.push_back(u32 ? a : mx::view(a, mx::bfloat16));
      const uint8_t* map = use_map ? veystrix::engine::rowfetch::shard_map(inf.fd) : nullptr;
      for (int i = 0; i < n; ++i) {
        veystrix::engine::rowfetch::Row r;
        r.fd = inf.fd;
        r.map = map;
        r.offset = inf.data_offset + static_cast<int64_t>(unique[i]) * static_cast<int64_t>(row_bytes);
        r.length = static_cast<int64_t>(row_bytes);
        r.dst = base + static_cast<size_t>(i) * row_bytes;
        r.group = i;
        rows.push_back(r);
      }
      total += row_bytes * static_cast<size_t>(n);
    }
    mx::array x4 = mx::expand_dims(x2, std::vector<int>{-2, -3});
    const std::vector<int32_t> zero{0};
    mx::array r0(zero.begin(), mx::Shape{1, 1}, mx::int32);
    std::vector<std::optional<mx::array>> d(static_cast<size_t>(n));
    {
      namespace rf = veystrix::engine::rowfetch;
      namespace gg = veystrix::engine::gpu_gate;
      auto fetch = std::make_shared<rf::Fetch>(std::move(rows), n, rf::lanes());
      auto build = [&](int g) {
        auto sl = [&](const mx::array& a) {
          return mx::slice(a, mx::Shape{g, 0, 0}, mx::Shape{g + 1, a.shape(1), a.shape(2)});
        };
        Q4Projection up(sl(stack[0]), sl(stack[1]), sl(stack[2]));
        Q4Projection dn(sl(stack[3]), sl(stack[4]), sl(stack[5]));
        mx::array u = q4_projection(x4, up, r0, 64, 4);
        mx::array a = mx::square(mx::maximum(u, mx::astype(mx::array(0.0f), u.dtype())));  // relu2
        d[static_cast<size_t>(g)].emplace(q4_projection(a, dn, r0, 64, 4));             // [1,1,1,H]
      };
      // VEYSTRIX_CPP_GATE_TAIL=N: the last N experts' compute goes behind a GPU gate
      // (gpu_gate.hpp) and the next layer is encoded while their rows land.
      if (gg::failed()) throw std::runtime_error("pipelined_moe: gated expert read failed");
      const int tail = gg::tail();
      int launched = 0;
      bool gated = false;
      for (;;) {
        if (tail > 0 && n - launched <= tail) {
          for (int g = 0; g < n; ++g)
            if (!d[static_cast<size_t>(g)]) build(g);
          const uint64_t v = gg::arm();
          if (v != 0) {
            std::vector<mx::array> keep(stack);
            fetch->on_done([v, keep = std::move(keep), fetch](bool ok) {
              if (!ok) gg::mark_failed();
              gg::signal(v);
            });
            gated = true;
          } else {
            while (fetch->next_ready() >= 0) {
            }
          }
          break;
        }
        const int g = fetch->next_ready();
        if (g < 0) break;
        build(g);
        mx::async_eval(*d[static_cast<size_t>(g)]);
        ++launched;
      }
      if (!gated && !fetch->ok()) throw std::runtime_error("pipelined_moe: short expert row read");
      const auto s = gated ? rf::Stats{} : fetch->stats();
      _pf.add_read(cppprof::ms_since(_t), static_cast<uint64_t>(total));
      pipe_last_memcpy_ = s.memcpy_rows;
      pipe_last_pread_ = s.pread_rows;
    }
    std::vector<mx::array> ordered;
    ordered.reserve(local.size());
    for (int j : local) ordered.push_back(*d[static_cast<size_t>(j)]);
    mx::array dcat = mx::squeeze(mx::concatenate(ordered, 1), -2);  // [1,K,H]
    return weighted_combine(dcat, scores);
  }
  int pipe_last_memcpy_ = 0, pipe_last_pread_ = 0;  // rows of the last pipelined layer

  // ---- Speculative next-layer row prefetch (decode, VEYSTRIX_CPP_NEMO_SPEC=1) --
  // After a MoE layer's route sync, the NEXT MoE layer's router is applied to
  // the same normed input to guess its experts. Their rows are read
  // asynchronously into MLX buffers with room for top_k extra rows while the
  // GPU computes the layers in between. When the real route arrives, rows that
  // were guessed are used in place and only the misses are read, into the
  // spare rows. The compute always uses the real route and real bytes, so the
  // output is unchanged; only where the bytes come from differs.
  static bool spec_enabled() {
    static const bool on = [] {
      const char* e = std::getenv("VEYSTRIX_CPP_NEMO_SPEC");
      return e && std::strcmp(e, "1") == 0;
    }();
    return on;
  }
  struct SpecRows {
    static constexpr int kN = 6;
    std::array<mx::allocator::Buffer, kN> mbuf{mx::allocator::Buffer{nullptr}, mx::allocator::Buffer{nullptr},
                                               mx::allocator::Buffer{nullptr}, mx::allocator::Buffer{nullptr},
                                               mx::allocator::Buffer{nullptr}, mx::allocator::Buffer{nullptr}};
    std::array<Info, kN> inf;
    std::array<size_t, kN> row_bytes{};
    std::array<bool, kN> handed{};
    std::vector<int> ids;  // guessed experts, sorted unique, rows 0..n-1
    size_t cap_rows = 0;
    veystrix::AsyncTicket ticket = veystrix::kInvalidTicket;
    SpecRows() = default;
    SpecRows(const SpecRows&) = delete;
    SpecRows& operator=(const SpecRows&) = delete;
    void wait() {
      if (ticket == veystrix::kInvalidTicket) return;
      const bool ok = keepwarm::wait_ticket_warm(ticket);
      ticket = veystrix::kInvalidTicket;
      if (!ok) throw std::runtime_error("spec rows: async read failed");
    }
    ~SpecRows() {
      if (ticket != veystrix::kInvalidTicket) {
        veystrix::wait_ticket(ticket);
        veystrix::release_ticket(ticket);
      }
      for (int k = 0; k < kN; ++k)
        if (!handed[k] && mbuf[k].ptr()) mx::allocator::free(mbuf[k]);
    }
  };

  std::unique_ptr<SpecRows> spec_issue(const std::array<std::string, 6>& names,
                                       const std::vector<int>& guess, int extra_rows) {
    auto sp = std::make_unique<SpecRows>();
    sp->ids = guess;
    sp->cap_rows = guess.size() + static_cast<size_t>(extra_rows);
    std::vector<veystrix::PagerRead> reads;
    for (int k = 0; k < SpecRows::kN; ++k) {
      sp->inf[k] = info(names[k]);
      const Info& inf = sp->inf[k];
      const bool u32 = inf.dtype == "U32";
      if (!u32 && inf.dtype != "BF16")
        throw std::runtime_error("spec rows: unexpected dtype for " + names[k]);
      sp->row_bytes[k] = static_cast<size_t>(inf.shape[1] * inf.shape[2]) * (u32 ? 4 : 2);
      sp->mbuf[k] = mx::allocator::malloc(sp->cap_rows * sp->row_bytes[k]);
      uint8_t* base = static_cast<uint8_t*>(sp->mbuf[k].raw_ptr());
      append_row_reads(reads, inf.fd, inf.data_offset, sp->row_bytes[k], guess, 0, base);
    }
    // VEYSTRIX_CPP_NEMO_SPEC_WIDTH: lanes for the guessed rows (default 8).
    // Fewer lanes leave more memory bandwidth to the GPU work they overlap.
    static const int spec_width = [] {
      const char* e = std::getenv("VEYSTRIX_CPP_NEMO_SPEC_WIDTH");
      const int v = e ? std::atoi(e) : 8;
      return v >= 1 && v <= 8 ? v : 8;
    }();
    if (!reads.empty())
      sp->ticket = veystrix::submit_batched_reads_async(reads, spec_width, nullptr);
    return sp;
  }

  // Coalesce consecutive ids into one read each; row r of `ids` lands at
  // dst row (first_row + r).
  static void append_row_reads(std::vector<veystrix::PagerRead>& reads, int fd,
                               int64_t data_offset, size_t row_bytes,
                               const std::vector<int>& ids, size_t first_row,
                               uint8_t* base) {
    size_t i = 0;
    while (i < ids.size()) {
      size_t j = i + 1;
      while (j < ids.size() && ids[j] == ids[j - 1] + 1) ++j;
      veystrix::PagerRead pr;
      pr.fd = fd;
      pr.pad = 0;
      pr.offset = data_offset + static_cast<int64_t>(ids[i]) * static_cast<int64_t>(row_bytes);
      pr.length = static_cast<int64_t>((j - i) * row_bytes);
      pr.dst = reinterpret_cast<uint64_t>(base + (first_row + i) * row_bytes);
      reads.push_back(pr);
      i = j;
    }
  }

  // Finish a speculative stack for the real `unique` experts. Fills `row_of`
  // (row index in the returned stacks for each entry of `unique`).
  std::array<mx::array, 6> spec_complete(SpecRows& sp, const std::vector<int>& unique,
                                         std::vector<int>& row_of, int& hits) {
    auto& _pf = cppprof::profiler();
    auto _t = _pf.now();
    std::vector<int> miss;
    row_of.assign(unique.size(), -1);
    hits = 0;
    for (size_t u = 0; u < unique.size(); ++u) {
      auto it = std::lower_bound(sp.ids.begin(), sp.ids.end(), unique[u]);
      if (it != sp.ids.end() && *it == unique[u]) {
        row_of[u] = static_cast<int>(it - sp.ids.begin());
        ++hits;
      } else {
        row_of[u] = static_cast<int>(sp.ids.size() + miss.size());
        miss.push_back(unique[u]);
      }
    }
    if (sp.ids.size() + miss.size() > sp.cap_rows)
      throw std::runtime_error("spec rows: capacity exceeded");
    std::vector<veystrix::PagerRead> reads;
    size_t total = 0;
    for (int k = 0; k < SpecRows::kN; ++k) {
      append_row_reads(reads, sp.inf[k].fd, sp.inf[k].data_offset, sp.row_bytes[k], miss,
                       sp.ids.size(), static_cast<uint8_t*>(sp.mbuf[k].raw_ptr()));
      total += sp.row_bytes[k] * unique.size();
    }
    if (!reads.empty() && run_expert_reads(reads, 8) != 0)
      throw std::runtime_error("spec rows: short miss read");
    sp.wait();
    _pf.add_read(cppprof::ms_since(_t), static_cast<uint64_t>(total));
    const int rows = static_cast<int>(sp.ids.size() + miss.size());
    auto make = [&](int k) {
      const Info& inf = sp.inf[k];
      const bool u32 = inf.dtype == "U32";
      const mx::Shape shape{rows, static_cast<int>(inf.shape[1]), static_cast<int>(inf.shape[2])};
      mx::array a(sp.mbuf[k], shape, u32 ? mx::uint32 : mx::uint16);
      sp.handed[k] = true;
      return u32 ? a : mx::view(a, mx::bfloat16);
    };
    return {make(0), make(1), make(2), make(3), make(4), make(5)};
  }

  // ---- Whole-layer asynchronous prefetch (prefill read/compute overlap) -----
  // All experts of one MoE layer's six tensors (fc1/fc2 x weight/scales/biases)
  // are read on the existing 8-lane dispatch queue into page-aligned buffers
  // while the GPU computes earlier layers. Used only when a prefill chunk is
  // long enough that nearly every expert is routed to anyway, so the extra
  // bytes read for unused experts are small and fully hidden behind compute.
  struct FullRows {
    static constexpr int kN = 6;
    void* raw[kN] = {};
    int64_t d1[kN] = {}, d2[kN] = {};
    int64_t experts = 0;
    bool u32[kN] = {};
    veystrix::AsyncTicket ticket = veystrix::kInvalidTicket;
    FullRows() = default;
    FullRows(const FullRows&) = delete;
    FullRows& operator=(const FullRows&) = delete;
    ~FullRows() { release(); }
    // Wait for any reads still writing into the buffers, then free what is left.
    void release() {
      if (ticket != veystrix::kInvalidTicket) {
        veystrix::wait_ticket(ticket);
        ticket = veystrix::kInvalidTicket;
      }
      for (int k = 0; k < kN; ++k) {
        if (raw[k]) std::free(raw[k]);
        raw[k] = nullptr;
      }
    }
    // Block until every read landed; throws if any failed.
    void wait() {
      if (ticket == veystrix::kInvalidTicket) return;
      const veystrix::AsyncTicket t = ticket;
      ticket = veystrix::kInvalidTicket;
      if (!veystrix::wait_ticket(t))
        throw std::runtime_error("nemotron prefill-ahead: async expert read failed");
    }
    // Hand tensor k's buffer to MLX (it frees it). Call only after wait().
    mx::array take(int k) {
      void* p = raw[k];
      raw[k] = nullptr;
      mx::array a(p, mx::Shape{static_cast<int>(experts), static_cast<int>(d1[k]),
                               static_cast<int>(d2[k])},
                  u32[k] ? mx::uint32 : mx::uint16, [](void* q) { std::free(q); });
      return u32[k] ? a : mx::view(a, mx::bfloat16);
    }
  };

  std::unique_ptr<FullRows> submit_full_layer(const std::array<std::string, 6>& names) {
    auto fr = std::make_unique<FullRows>();
    std::vector<veystrix::PagerRead> reads;
    static const size_t piece = size_t{8} << 20;
    for (int k = 0; k < FullRows::kN; ++k) {
      Info inf = info(names[k]);
      const bool is_u32 = (inf.dtype == "U32");
      if (!is_u32 && inf.dtype != "BF16")
        throw std::runtime_error("prefill-ahead: unexpected dtype for " + names[k]);
      fr->u32[k] = is_u32;
      fr->experts = inf.shape[0];
      fr->d1[k] = inf.shape[1];
      fr->d2[k] = inf.shape[2];
      const size_t bytes = static_cast<size_t>(inf.shape[0]) * static_cast<size_t>(inf.shape[1]) *
                           static_cast<size_t>(inf.shape[2]) * (is_u32 ? 4 : 2);
      if (::posix_memalign(&fr->raw[k], 16384, bytes ? bytes : 1) != 0)
        throw std::runtime_error("prefill-ahead: posix_memalign failed");
      for (size_t off = 0; off < bytes; off += piece) {
        veystrix::PagerRead pr;
        pr.fd = inf.fd;
        pr.pad = 0;
        pr.offset = inf.data_offset + static_cast<int64_t>(off);
        pr.length = static_cast<int64_t>(std::min(piece, bytes - off));
        pr.dst = reinterpret_cast<uint64_t>(static_cast<uint8_t*>(fr->raw[k]) + off);
        reads.push_back(pr);
      }
    }
    fr->ticket = veystrix::submit_batched_reads_async(reads, /*width=*/8, [](int) {});
    return fr;
  }

 public:
  // Phase 1.1: exposed so the pool resolver reuses the verbatim safetensors
  // header parser (no second parser, no change to the parse itself).
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
      apply_shard_read_hints(fd);
      fds_[path] = fd;
    }
    uint64_t nn = 0;
    std::string local_header;
    const std::string* header_src = nullptr;
    if (shard_index_) {
      // E1-N: one parsed header per shard, shared by every ExpertReader
      // (including the NemoPool resolver's reader). The bytes parsed below
      // are identical to the baseline's fresh pread of the same shard header.
      const ParsedShardHeader& sh = parsed_shard_header(path);
      nn = sh.nn;
      header_src = &sh.header;
    } else {
      unsigned char lenbuf[8];
      if (::pread(fd, lenbuf, 8, 0) != 8)
        throw std::runtime_error("short safetensors header length: " + path);
      for (int i = 0; i < 8; ++i)
        nn |= static_cast<uint64_t>(lenbuf[i]) << (8 * i);
      local_header.resize(nn);
      if (nn && ::pread(fd, local_header.data(), nn, 8) != static_cast<ssize_t>(nn))
        throw std::runtime_error("short safetensors header: " + path);
      header_src = &local_header;
    }
    const std::string& header = *header_src;
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

 private:
  std::string dir_;
  const Config& cfg_;
  std::map<std::string, Info> cache_;
  std::map<std::string, int> fds_;
  bool shard_index_ = false;   // E1-N
  bool reuse_scratch_ = false; // E2-N
  std::vector<uint32_t> scratch32_;
  std::vector<uint16_t> scratch16_;
};

// --------------------------------------------------------------------------- //
// Phase 1.1: opt-in NemoPool-backed expert fetch (VEYSTRIX_CPP_NEMO_POOL=1).       //
//                                                                               //
// A Nemotron expert is SIX safetensors tensors (fc1/fc2 x weight/scales/biases).  //
// NemoPool has a fixed record stride, so we use one pool per tensor kind. All    //
// six pools receive the IDENTICAL acquire sequence and slot count, so NemoPool's //
// freelist/LRU bookkeeping stays in lockstep and the slot index of a given       //
// (layer, expert) is the same in all six pools. That is what lets ONE gather     //
// index array (gather_qmm takes a single route for w/s/b) index all six arrays   //
// wrapped zero-copy over the pool buffers.                                       //
//                                                                               //
// Fills go through the existing 8-lane batched pread (io_pager::run_batched_reads)//
// via an injected no-op NemoPool::ReadFn: acquire() reserves the slot and the    //
// whole layer's misses are issued as one batched read immediately afterwards.   //
// Cached descriptors are used by default (F_NOCACHE measured slower, see        //
// DO_NOT_IMPLEMENT); VEYSTRIX_NEMOTRON_POOL_NOCACHE=1 selects uncached fills.     //
//                                                                               //
// Rollback: VEYSTRIX_CPP_NEMO_POOL!=1 keeps the original compact-unique reader.  //
// Digest gate: both gathers are bit-identical to the baseline (verified 3/3).    //
// "compact" (DEFAULT) copies first-occurrence rows out of the pool slots into a   //
// small contiguous buffer (the brief's Qwen-P5-style fallback) so gather_qmm sees//
// the same small 0..n-1 buffer as the baseline; it is measurably the FAST path.   //
// "pooled" wraps the whole slot buffer zero-copy but is slower (gather_qmm over a //
// 64-row array); opt in with VEYSTRIX_CPP_NEMO_POOL_GATHER=pooled.                //
// --------------------------------------------------------------------------- //
enum NemoKind : int { kF1W = 0, kF1S = 1, kF1B = 2, kF2W = 3, kF2S = 4, kF2B = 5 };
static constexpr int kNemoKinds = 6;

// Memory gate applied BEFORE any pool memory is allocated. Only the C ABI
// runtime passes an enabled gate (dynamic-memory contract, memory_budget.hpp);
// the standalone CLIs keep their historical un-gated behaviour.
struct NemoPoolGate {
  bool enabled = false;
  bool resident = false;        // hold every (layer, expert) record
  size_t explicit_budget = 0;   // VEYSTRIX_CPP_POOL_BYTES / expert_pool_bytes
  size_t active_bytes = 0;      // MLX active memory after the dense load
};

class PooledExpertTensors {
 public:
  PooledExpertTensors(const std::string& dir, const Config& cfg, int n_layers,
                      int top_k, int num_slots, bool compact_gather, bool nocache,
                      NemoPoolGate gate = {})
      : dir_(dir), cfg_(cfg), reader_(dir, cfg), top_k_(top_k),
        num_slots_(num_slots), compact_gather_(compact_gather), nocache_(nocache) {
    // Preload every header the resolver may need: resolver runs under the pool
    // lock, so it must never do first-touch header I/O.
    for (int l = 0; l < n_layers; ++l) {
      const std::string base = moe_base(l);
      if (cfg_.find("shard." + base + ".weight") == cfg_.end()) continue;
      moe_layers_.push_back(l);
      for (int k = 0; k < kNemoKinds; ++k) reader_.info(tensor_name(l, k));
    }
    if (moe_layers_.empty())
      throw std::runtime_error("NemoPool: no MoE layers found in config");

    // Size each record from the checkpoint's actual expert geometry.
    const int l0 = moe_layers_.front();
    for (int k = 0; k < kNemoKinds; ++k) {
      const auto inf = reader_.info(tensor_name(l0, k));
      if (inf.shape.size() != 3)
        throw std::runtime_error("NemoPool: expert tensor is not 3-D");
      geom_[k].dim1 = static_cast<int>(inf.shape[1]);
      geom_[k].dim2 = static_cast<int>(inf.shape[2]);
      geom_[k].u32 = (k == kF1W || k == kF2W);
      if (geom_[k].u32 && inf.dtype != "U32")
        throw std::runtime_error("NemoPool: expected U32 expert weight");
      if (!geom_[k].u32 && inf.dtype != "BF16")
        throw std::runtime_error("NemoPool: expected BF16 expert scale/bias");
      geom_[k].record = static_cast<size_t>(geom_[k].dim1) *
                        static_cast<size_t>(geom_[k].dim2) * (geom_[k].u32 ? 4 : 2);
      if (geom_[k].record == 0 || geom_[k].record % 4 != 0)
        throw std::runtime_error("NemoPool: bad record size");
    }

    // Gate BEFORE the NemoPool buffers exist (each NemoPool allocates and
    // zeroes slots*record bytes at construction). Total pool footprint is the
    // six tensor kinds x slots.
    size_t rec_sum = 0;
    for (int k = 0; k < kNemoKinds; ++k) rec_sum += geom_[k].record;
    const int n_experts_per_layer = static_cast<int>(
        reader_.info(tensor_name(l0, kF1W)).shape[0]);
    all_keys_ = moe_layers_.size() * static_cast<size_t>(n_experts_per_layer);
    if (gate.enabled) {
      if (gate.resident) {
        if (static_cast<size_t>(num_slots_) < all_keys_)
          throw std::runtime_error(
              "nemotron: resident refused: " + std::to_string(num_slots_) +
              " pool slots < " + std::to_string(all_keys_) + " (layer,expert) keys" +
              " (reopen with resident=0 to stream from SSD)");
        num_slots_ = static_cast<int>(all_keys_);  // never allocate beyond need
        veystrix::engine::membudget::require_resident_fit(
            "nemotron", all_keys_ * rec_sum, gate.explicit_budget, gate.active_bytes);
      } else {
        veystrix::engine::membudget::require_ram_headroom(
            "nemotron", static_cast<size_t>(num_slots_) * rec_sum, gate.active_bytes);
      }
    }
    resident_ = gate.enabled && gate.resident;

    // See the class comment: the injected ReadFn only RESERVES; the real read is
    // issued by run_fills() through the 8-lane batched reader.
    NemoPool::ReadFn reserve_only = [](int, void*, size_t len, int64_t) -> ssize_t {
      return static_cast<ssize_t>(len);
    };
    for (int k = 0; k < kNemoKinds; ++k) {
      NemoPool::Resolver resolver = [this, k](int layer, int expert) {
        const std::string name = tensor_name(layer, k);
        const auto inf = reader_.info(name);  // preloaded: map is read-only now
        auto sit = cfg_.find("shard." + name);
        if (sit == cfg_.end())
          throw std::runtime_error("NemoPool: missing shard map for " + name);
        NemoPool::Record rec;
        rec.shard = dir_ + "/" + sit->second;
        rec.offset = inf.data_offset +
                     static_cast<int64_t>(expert) * static_cast<int64_t>(record_bytes(k));
        rec.length = static_cast<int64_t>(record_bytes(k));
        return rec;
      };
      pools_[k] = std::make_unique<NemoPool>(num_slots_, record_bytes(k),
                                             std::move(resolver), nocache_,
                                             reserve_only);
    }
  }

  size_t record_bytes(int k) const { return geom_[k].record; }

  NemoPool& pool(int k) { return *pools_[k]; }

  // ---- Resident mode (VEYSTRIX_CPP_RESIDENT=1) ---------------------------
  bool resident() const { return resident_; }
  size_t all_keys() const { return all_keys_; }
  int pool_slots() const { return num_slots_; }
  size_t resident_bytes() const {
    size_t s = 0;
    for (int k = 0; k < kNemoKinds; ++k) s += geom_[k].record;
    return s * static_cast<size_t>(num_slots_);
  }
  // Warm-up: read every (layer, expert) record exactly once through the same
  // 8-lane batched fill path decode uses. Because num_slots_ == all_keys_ no
  // acquire() can ever evict afterwards, so decode performs ZERO SSD reads.
  void preload_all() {
    const int E = static_cast<int>(all_keys_ / moe_layers_.size());
    if (E < top_k_) throw std::runtime_error("NemoPool: fewer experts than top_k");
    std::vector<int> unique(E), local(top_k_);
    std::iota(unique.begin(), unique.end(), 0);
    std::iota(local.begin(), local.end(), 0);
    std::vector<veystrix::PagerRead> reads;
    for (int l : moe_layers_) {
      begin_layer(l, unique, local, reads);
      run_fills(reads);
    }
  }

  // Reserve this layer's slots for every distinct expert and every tensor kind,
  // and collect the batched fills. Must be called once per MoE layer before
  // make_array()/route_indices(). `local` maps the top_k route positions to
  // first-occurrence indices in `unique` (exactly the baseline compact path).
  void begin_layer(int layer, const std::vector<int>& unique,
                   const std::vector<int>& local,
                   std::vector<veystrix::PagerRead>& reads) {
    reads.clear();
    local_ = local;
    const NemoPoolStats s0 = pools_[0]->stats();
    slots_by_unique_.assign(kNemoKinds, std::vector<int>(unique.size(), -1));
    for (int k = 0; k < kNemoKinds; ++k) {
      for (size_t j = 0; j < unique.size(); ++j) {
        const bool resident = pools_[k]->slot_for(layer, unique[j]) >= 0;
        const int slot = pools_[k]->acquire(layer, unique[j]);
        slots_by_unique_[k][j] = slot;
        if (!resident) {
          const auto rec = resolve(layer, unique[j], k);
          const int fd = nocache_ ? pools_[k]->uncached_fd(rec.shard)
                                  : pools_[k]->cached_fd(rec.shard);
          veystrix::PagerRead pr;
          pr.fd = fd;
          pr.pad = 0;
          pr.offset = rec.offset;
          pr.length = rec.length;
          pr.dst = reinterpret_cast<uint64_t>(pools_[k]->slot_data(slot));
          reads.push_back(pr);
        }
      }
    }
    // Fail closed if the lockstep invariant ever breaks: one route must index
    // all six arrays. (Single-driver decode never exercises an eviction race.)
    for (int k = 1; k < kNemoKinds; ++k)
      if (slots_by_unique_[k] != slots_by_unique_[0])
        throw std::runtime_error("NemoPool: slot indices diverged across tensor kinds");

    route_slots_.assign(top_k_, -1);
    for (int i = 0; i < top_k_; ++i)
      route_slots_[i] = slots_by_unique_[0][local_[i]];

    const NemoPoolStats s1 = pools_[0]->stats();
    last_hits_ = static_cast<int>(s1.hits - s0.hits);
    last_misses_ = static_cast<int>(s1.misses - s0.misses);
    last_coalesced_ = static_cast<int>(s1.coalesced_reads - s0.coalesced_reads);
  }

  int last_hits() const { return last_hits_; }
  int last_misses() const { return last_misses_; }
  int last_coalesced() const { return last_coalesced_; }

  // Issue the whole layer's misses through the existing 8-lane batched reader.
  void run_fills(const std::vector<veystrix::PagerRead>& reads) {
    if (reads.empty()) return;
    auto& pf = cppprof::profiler();
    const auto t0 = pf.now();
    uint64_t bytes = 0;
    for (const auto& r : reads) bytes += static_cast<uint64_t>(r.length);
    const int err = veystrix::run_batched_reads(reads, /*width=*/8);
    pf.add_read(cppprof::ms_since(t0), bytes);
    if (err != 0) throw std::runtime_error("NemoPool: batched fill failed");
  }

  // P5-overlap: async counterpart of run_fills. The reads go to the existing
  // libdispatch 8-lane queue; the driver can run ANOTHER session's host build
  // and GPU work while these are in flight. No new thread/mutex/QoS.
  veystrix::AsyncTicket submit_fills(const std::vector<veystrix::PagerRead>& reads) {
    if (reads.empty()) return veystrix::kInvalidTicket;
    auto& pf = cppprof::profiler();
    const auto t0 = pf.now();
    uint64_t bytes = 0;
    for (const auto& r : reads) bytes += static_cast<uint64_t>(r.length);
    // The pager's completion callback reports failures through ticket_errors.
    const veystrix::AsyncTicket ticket = veystrix::submit_batched_reads_async(
        reads, /*width=*/8, [](int) {});
    pf.add_read(cppprof::ms_since(t0), bytes);
    return ticket;
  }
  static bool fills_ready(veystrix::AsyncTicket t) {
    return t == veystrix::kInvalidTicket || veystrix::ticket_complete(t);
  }
  static void wait_fill(veystrix::AsyncTicket t) {
    if (t == veystrix::kInvalidTicket) return;
    if (!veystrix::wait_ticket(t))
      throw std::runtime_error("NemoPool: async batched fill failed");
  }

  // Gather indices for moe_block_relu2: real pool slots for zero-copy, or the
  // unchanged 0..n-1 first-occurrence indices for the compact fallback.
  const std::vector<int>& route_indices() const {
    return compact_gather_ ? local_ : route_slots_;
  }

  // Build the mx::array the MoE graph consumes for one tensor kind.
  mx::array make_array(int k) {
    const auto& g = geom_[k];
    if (!compact_gather_) {
      // Zero-copy over the pool's slot memory: no iterator copy. The pool object
      // outlives the graph evaluation (kept alive by Model), so a no-op deleter
      // is safe. Most rows are stale/garbage but gather only touches our slots.
      mx::array a(pools_[k]->slot_data(0),
                  mx::Shape{num_slots_, g.dim1, g.dim2},
                  g.u32 ? mx::uint32 : mx::uint16,
                  [](void*) {});
      return g.u32 ? a : mx::view(a, mx::bfloat16);
    }
    // Compact first-occurrence stack FROM pool slots (brief's fallback). The
    // bytes are identical to the baseline compact-unique stack, so the digest
    // is identical by construction; gather indices stay 0..n-1.
    //
    // One pass: rows are memcpy'd straight into an aligned buffer that MLX then
    // owns (deleter frees it). The previous version zero-filled a std::vector,
    // memcpy'd into it, and let mx::array(iterator, ...) copy it a second time:
    // three passes over ~34 MB per layer, which was most of the 61 ms/token of
    // CPU-side "build" in the decode profile. VEYSTRIX_CPP_NEMO_WRAP=copy restores it.
    const size_t n = slots_by_unique_[k].size();
    const size_t per = static_cast<size_t>(g.dim1) * static_cast<size_t>(g.dim2);
    const size_t elem = g.u32 ? 4 : 2;
    const mx::Shape shape{static_cast<int>(n), g.dim1, g.dim2};
    static const bool wrap_owned = [] {
      const char* e = std::getenv("VEYSTRIX_CPP_NEMO_WRAP");
      return !(e && std::strcmp(e, "copy") == 0);
    }();
    if (wrap_owned) {
      static const size_t align = [] {
        const char* e = std::getenv("VEYSTRIX_CPP_COMPACT_PAGE_ALIGN");
        return (e && std::strcmp(e, "0") == 0) ? size_t{64} : size_t{16384};
      }();
      void* raw = nullptr;
      const size_t bytes = per * n * elem;
      if (::posix_memalign(&raw, align, bytes ? bytes : 1) != 0)
        throw std::runtime_error("NemoPool: posix_memalign failed for compact stack");
      uint8_t* dst = static_cast<uint8_t*>(raw);
      for (size_t j = 0; j < n; ++j)
        std::memcpy(dst + j * per * elem,
                    pools_[k]->slot_data(slots_by_unique_[k][j]), per * elem);
      mx::array a(raw, shape, g.u32 ? mx::uint32 : mx::uint16,
                  [](void* p) { std::free(p); });
      return g.u32 ? a : mx::view(a, mx::bfloat16);
    }
    if (g.u32) {
      std::vector<uint32_t> buf(per * n);
      for (size_t j = 0; j < n; ++j)
        std::memcpy(buf.data() + j * per,
                    pools_[k]->slot_data(slots_by_unique_[k][j]), per * 4);
      return mx::array(buf.begin(), shape, mx::uint32);
    }
    std::vector<uint16_t> buf(per * n);
    for (size_t j = 0; j < n; ++j)
      std::memcpy(buf.data() + j * per,
                  pools_[k]->slot_data(slots_by_unique_[k][j]), per * 2);
    mx::array a(buf.begin(), shape, mx::uint16);
    return mx::view(a, mx::bfloat16);
  }

 private:
  std::string moe_base(int layer) const {
    return "backbone.layers." + std::to_string(layer) +
           ".mixer.switch_mlp.fc1";
  }
  std::string tensor_name(int layer, int kind) const {
    static const char* suffix[3] = {".weight", ".scales", ".biases"};
    std::string b = "backbone.layers." + std::to_string(layer) +
                    ".mixer.switch_mlp." + (kind < 3 ? "fc1" : "fc2");
    return b + suffix[kind % 3];
  }
  NemoPool::Record resolve(int layer, int expert, int kind) {
    const std::string name = tensor_name(layer, kind);
    const auto inf = reader_.info(name);
    auto sit = cfg_.find("shard." + name);
    if (sit == cfg_.end())
      throw std::runtime_error("NemoPool: missing shard map for " + name);
    NemoPool::Record rec;
    rec.shard = dir_ + "/" + sit->second;
    rec.offset = inf.data_offset +
                 static_cast<int64_t>(expert) * static_cast<int64_t>(record_bytes(kind));
    rec.length = static_cast<int64_t>(record_bytes(kind));
    return rec;
  }

  struct Geom {
    int dim1 = 0, dim2 = 0;
    bool u32 = true;
    size_t record = 0;
  };

  std::string dir_;
  const Config& cfg_;
  ExpertReader reader_;
  int top_k_, num_slots_;
  bool compact_gather_, nocache_;
  bool resident_ = false;
  size_t all_keys_ = 0;
  std::vector<int> moe_layers_;
  std::array<Geom, kNemoKinds> geom_;
  std::array<std::unique_ptr<NemoPool>, kNemoKinds> pools_;
  std::vector<std::vector<int>> slots_by_unique_;
  std::vector<int> local_, route_slots_;
  int last_hits_ = 0, last_misses_ = 0, last_coalesced_ = 0;
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

  // Phase 1.1 opt-in expert pool (VEYSTRIX_CPP_NEMO_POOL=1). Null => baseline
  // compact-unique path (rollback, default). Kept alive for the whole run so
  // the zero-copy mx::arrays remain valid.
  std::unique_ptr<PooledExpertTensors> expert_pool;
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
// Diagnostic only (VEYSTRIX_CPP_LAYER_PROF=1): syncs after every layer so GPU time can be
// attributed per layer kind. Changes the sync pattern, never the arithmetic.
struct LayerProf {
  bool on = false;
  double ms[5] = {0, 0, 0, 0, 0};  // mamba, attn, moe router, moe read+build, moe compute
  int steps = 0, layers_seen = 0;
  LayerProf() {
    const char* e = std::getenv("VEYSTRIX_CPP_LAYER_PROF");
    on = e && std::strcmp(e, "1") == 0;
  }
  static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void report() {
    if (!on || steps == 0 || steps % 25) return;
    std::fprintf(stderr,
                 "[LPROF] steps=%d per-step ms: mamba %.1f attn %.1f moe-route %.1f moe-read+build %.1f "
                 "moe-compute %.1f total %.1f\n",
                 steps, ms[0] / steps, ms[1] / steps, ms[2] / steps, ms[3] / steps, ms[4] / steps,
                 (ms[0] + ms[1] + ms[2] + ms[3] + ms[4]) / steps);
  }
};
static LayerProf g_lprof;

mx::array run_step(Model& m, ExpertReader& er, int token) {
  mx::array h = embed_token(m, token);  // [1,1,H] bf16
  // Speculative rows for the next MoE layer (VEYSTRIX_CPP_NEMO_SPEC=1).
  const bool spec_on = ExpertReader::spec_enabled() && !m.expert_pool;
  std::unique_ptr<ExpertReader::SpecRows> spec;
  size_t mamba_i = 0, attn_i = 0, moe_i = 0;
  const bool lp = g_lprof.on;
  if (lp) mx::eval(h);
  double lp_t = lp ? LayerProf::now_ms() : 0.0;
  auto lp_lap = [&](int bucket) {
    if (!lp) return;
    const double t = LayerProf::now_ms();
    g_lprof.ms[bucket] += t - lp_t;
    lp_t = t;
  };

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
      if (lp) { mx::eval(delta, cache.conv_state, cache.ssm_state); lp_lap(0); }
    } else if (m.kinds[l] == LayerKind::FULL_ATTN_QSA) {
      AttnLayer& al = m.attn_layers[attn_i++];
      if (m.use_int8_kv)
        delta = nemo_attention_int8_forward(xn, al.weights, al.cfg, al.int8_cache);
      else
        delta = nemo_attention_forward(xn, al.weights, al.cfg, al.cache);
      if (lp) { mx::eval(delta); lp_lap(1); }
    } else {
      MoeLayer& ml = m.moe_layers[moe_i++];
      NemotronRoute route = nemotron_route(xn, ml, m.top_k, m.routed_scale);

      // Read the selected expert rows (dedup + sorted, like Python _gather).
      mx::array ids_i = mx::astype(route.indices, mx::int32);
      // VEYSTRIX_CPP_NEMO_PREFETCH=1 (with NEMO_PIPE): the same next-layer guess,
      // used only to warm the page cache with the guessed rows that are not
      // resident (rowfetch::prefetch_cold) once this layer's rows have landed.
      // VEYSTRIX_CPP_NEMO_PREFETCH_TOP=Q keeps the Q highest-scored guesses.
      static const bool nemo_prefetch = [] {
        const char* e = std::getenv("VEYSTRIX_CPP_NEMO_PREFETCH");
        return e && std::strcmp(e, "1") == 0;
      }();
      const bool pf_on = !spec_on && nemo_prefetch && ExpertReader::pipe_enabled() && !m.expert_pool;
      MoeLayer* next_ml = ((spec_on || pf_on) && moe_i < m.moe_layers.size()) ? &m.moe_layers[moe_i] : nullptr;
      std::optional<mx::array> guess_i, guess_s;
      if (next_ml) {
        NemotronRoute gr = nemotron_route(xn, *next_ml, m.top_k, m.routed_scale);
        guess_i.emplace(mx::astype(gr.indices, mx::int32));
        if (pf_on) guess_s.emplace(mx::astype(gr.scores, mx::float32));
      }
      {
        auto _ts = cppprof::profiler().now();
        if (guess_s) mx::eval({ids_i, *guess_i, *guess_s});
        else if (guess_i) mx::eval(ids_i, *guess_i);
        else mx::eval(ids_i);
        cppprof::profiler().add_route_sync(l, cppprof::ms_since(_ts));
      }
      lp_lap(2);
      // VEYSTRIX_CPP_NEMO_SHARED_EARLY=1: build the shared expert now and
      // launch it on the GPU so it runs while the routed rows are read (as the
      // Python engine does). Same ops and inputs, so the values are identical.
      static const bool shared_early = [] {
        const char* e = std::getenv("VEYSTRIX_CPP_NEMO_SHARED_EARLY");
        return e && std::strcmp(e, "1") == 0;
      }();
      auto shared_expert = [&]() {
        mx::array su = linear(xn, ml.shared_up);
        // Mirror Python nn.relu2 (weak scalar 0): keep su's dtype (bf16), do
        // not promote bf16 -> f32 with a strongly-typed f32 zero array.
        mx::array sa = mx::square(
            mx::maximum(su, mx::astype(mx::array(0.0f), su.dtype())));
        return linear(sa, ml.shared_down);
      };
      std::optional<mx::array> sd_early;
      if (shared_early) {
        sd_early.emplace(shared_expert());
        mx::async_eval(*sd_early);
      }
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

      // Phase 1.1 opt-in NemoPool fetch. When enabled, slots are reserved and
      // filled with one 8-lane batched read, then wrapped zero-copy (or copied
      // into a compact first-occurrence stack in the digest-fallback mode).
      // The baseline compact-unique path is unchanged when the flag is off.
      std::optional<Q4Projection> up, down;
      mx::array moe_route = local_idx;
      // VEYSTRIX_CPP_NEMO_PIPE=1: per-expert pipelined reads + GPU launches.
      const bool pipe = ExpertReader::pipe_enabled() && !m.expert_pool && !spec_on;
      std::optional<mx::array> piped;
      if (pipe) {
        piped.emplace(er.pipelined_moe(
            {ml.fc1_base + ".weight", ml.fc1_base + ".scales", ml.fc1_base + ".biases",
             ml.fc2_base + ".weight", ml.fc2_base + ".scales", ml.fc2_base + ".biases"},
            unique, local, mx::reshape(xn, mx::Shape{1, m.H}), route.scores));
        // "hit" in STEPPROF = rows copied from the page cache via the mapping.
        cppprof::profiler().add_layer_cache(l, er.pipe_last_memcpy_, er.pipe_last_pread_, 0);
      } else if (PooledExpertTensors* P = m.expert_pool.get()) {
        std::vector<veystrix::PagerRead> reads;
        P->begin_layer(l, unique, local, reads);
        P->run_fills(reads);
        up.emplace(P->make_array(kF1W), P->make_array(kF1S), P->make_array(kF1B));
        down.emplace(P->make_array(kF2W), P->make_array(kF2S), P->make_array(kF2B));
        const std::vector<int>& ri = P->route_indices();
        moe_route = mx::array(ri.begin(), mx::Shape{1, m.top_k}, mx::int32);
        cppprof::profiler().add_layer_cache(l, P->last_hits(), P->last_misses(),
                                            P->last_coalesced());
      } else if (spec) {
        std::vector<int> row_of;
        int hits = 0;
        std::array<mx::array, 6> r = er.spec_complete(*spec, unique, row_of, hits);
        spec.reset();
        for (int i = 0; i < m.top_k; ++i) local[i] = row_of[local[i]];
        moe_route = mx::array(local.begin(), mx::Shape{1, m.top_k}, mx::int32);
        cppprof::profiler().add_layer_cache(l, hits, static_cast<int>(unique.size()) - hits, 0);
        up.emplace(r[0], r[1], r[2]);
        down.emplace(r[3], r[4], r[5]);
      } else if (ExpertReader::one_batch() || spec_on) {
        std::array<mx::array, 6> r = er.rows6(
            {ml.fc1_base + ".weight", ml.fc1_base + ".scales", ml.fc1_base + ".biases",
             ml.fc2_base + ".weight", ml.fc2_base + ".scales", ml.fc2_base + ".biases"},
            unique);
        up.emplace(r[0], r[1], r[2]);
        down.emplace(r[3], r[4], r[5]);
      } else {
        up.emplace(er.rows_u32(ml.fc1_base + ".weight", unique),
                   er.rows_bf16(ml.fc1_base + ".scales", unique),
                   er.rows_bf16(ml.fc1_base + ".biases", unique));
        down.emplace(er.rows_u32(ml.fc2_base + ".weight", unique),
                     er.rows_bf16(ml.fc2_base + ".scales", unique),
                     er.rows_bf16(ml.fc2_base + ".biases", unique));
      }

      // Launch the guessed rows of the next MoE layer; they load while the GPU
      // runs this layer's experts and the layers up to the next route sync.
      if (next_ml && pf_on) {
        static const int top_q = [] {
          const char* e = std::getenv("VEYSTRIX_CPP_NEMO_PREFETCH_TOP");
          const int v = e ? std::atoi(e) : 0;
          return v > 0 ? v : 0;
        }();
        const int32_t* gp = guess_i->data<int32_t>();
        const float* gs = guess_s->data<float>();
        std::vector<int> ord(static_cast<size_t>(m.top_k));
        for (int i = 0; i < m.top_k; ++i) ord[static_cast<size_t>(i)] = i;
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return gs[a] > gs[b]; });
        if (top_q > 0 && top_q < m.top_k) ord.resize(static_cast<size_t>(top_q));
        std::vector<veystrix::engine::rowfetch::Row> rows;
        for (const std::string& nm :
             {next_ml->fc1_base + ".weight", next_ml->fc2_base + ".weight", next_ml->fc1_base + ".scales",
              next_ml->fc1_base + ".biases", next_ml->fc2_base + ".scales", next_ml->fc2_base + ".biases"}) {
          const ExpertReader::Info inf = er.info(nm);
          const int64_t rb = inf.shape[1] * inf.shape[2] * (inf.dtype == "U32" ? 4 : 2);
          const uint8_t* map = veystrix::engine::rowfetch::shard_map(inf.fd);
          for (int i : ord) {
            veystrix::engine::rowfetch::Row r;
            r.fd = inf.fd;
            r.map = map;
            r.offset = inf.data_offset + static_cast<int64_t>(gp[i]) * rb;
            r.length = rb;
            rows.push_back(r);
          }
        }
        veystrix::engine::rowfetch::prefetch_cold(std::move(rows), 8);
      } else if (next_ml) {
        const int32_t* gp = guess_i->data<int32_t>();
        std::vector<int> g(gp, gp + m.top_k);
        std::sort(g.begin(), g.end());
        g.erase(std::unique(g.begin(), g.end()), g.end());
        spec = er.spec_issue(
            {next_ml->fc1_base + ".weight", next_ml->fc1_base + ".scales", next_ml->fc1_base + ".biases",
             next_ml->fc2_base + ".weight", next_ml->fc2_base + ".scales", next_ml->fc2_base + ".biases"},
            g, m.top_k);
      }

      lp_lap(3);
      mx::array x2 = mx::reshape(xn, mx::Shape{1, m.H});
      mx::array routed = piped ? *piped : moe_block_relu2(x2, moe_route, route.scores, *up, *down);
      // Python: (y * scores).sum(-2).astype(y.dtype) -> bf16.
      routed = mx::astype(routed, mx::bfloat16);
      routed = mx::reshape(routed, mx::Shape{1, 1, m.H});

      // Shared expert: down_proj(relu2(up_proj(x))).
      mx::array sd = sd_early ? *sd_early : shared_expert();
      delta = routed + sd;
      if (lp) { mx::eval(delta); lp_lap(4); }
    }

    h = h + delta;  // NemotronHBlock: return x + mixer(norm(x))
    if (m.dump_layers)
      std::fprintf(stderr, "[NEMOLAYER step L=%02d] %s\n", l,
                   digest_array(h).c_str());
  }

  mx::array fused = mx::fast::rms_norm(h, m.norm_f_w, m.eps);
  mx::array logits = linear(fused, m.lm_head);  // [1,1,V]
  if (lp) { mx::eval(logits); lp_lap(0); ++g_lprof.steps; g_lprof.report(); }
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
  // E2-N (opt-in): reuse the row scratch buffer across the 1K chunk-1 prefill.
  // E1-N is applied inside every ExpertReader constructor (incl. the pool's).
  const bool reuse_prefill = nemo_reuse_prefill_from_env();
  er.set_reuse_scratch(reuse_prefill);
  if (nemo_shard_index_enabled() || reuse_prefill)
    std::fprintf(stderr, "[phase-e-n] shard_index=%d reuse_prefill=%d\n",
                 (int)nemo_shard_index_enabled(), (int)reuse_prefill);
  const std::vector<int> prompt = split_ints(get_str(cfg, "prompt_ids"));
  if (prompt.empty()) throw std::runtime_error("empty prompt_ids in config");
  std::fprintf(stderr, "[load] all weights ready; starting generation\n");
  // Phase A: start the profiler clock HERE (after load, before prefill) so row 1
  // records prefill bytes only; decode rows stay clean (see begin_step in the
  // loop below, which now runs only for step>0).
  cppprof::profiler().begin_step();
  mem_telemetry("post-load");

  // Phase 1.1: opt-in NemoPool. Unset / "0" leaves the baseline path unchanged.
  {
    const char* en = std::getenv("VEYSTRIX_CPP_NEMO_POOL");
    if (en && std::strcmp(en, "1") == 0) {
      const char* gm = std::getenv("VEYSTRIX_CPP_NEMO_POOL_GATHER");
      // compact-from-slots is the default: bit-identical AND measurably faster.
      const bool compact = !(gm && std::strcmp(gm, "pooled") == 0);
      if (gm && std::strcmp(gm, "compact") != 0 && std::strcmp(gm, "pooled") != 0)
        throw std::runtime_error(
            "VEYSTRIX_CPP_NEMO_POOL_GATHER must be 'pooled' or 'compact'");
      int slots = 64;
      const char* ns = std::getenv("VEYSTRIX_CPP_NEMO_POOL_SLOTS");
      if (ns) slots = std::atoi(ns);
      if (slots < m.top_k || slots > 4096)
        throw std::runtime_error("VEYSTRIX_CPP_NEMO_POOL_SLOTS out of range");
      const bool nocache = nemo_pool_nocache_from_env();  // fail-closed on bad value
      m.expert_pool = std::make_unique<PooledExpertTensors>(
          ckpt, cfg, m.n_layers, m.top_k, slots, compact, nocache);
      std::fprintf(stderr, "[nemo-pool] enabled gather=%s slots=%d nocache=%d\n",
                   compact ? "compact" : "pooled", slots, (int)nocache);
    } else if (en && std::strcmp(en, "0") != 0) {
      throw std::runtime_error("VEYSTRIX_CPP_NEMO_POOL must be 0 or 1");
    }
  }

  std::vector<StepRecord> records;
  records.reserve(num_tokens);

  // Sequential single-token prefill (matches Python chunk_tokens=1). The
  // logits after the last prompt token produce the first generated token.
  const auto t_gen0 = cppprof::Clock::now();
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

  const auto t_decode0 = cppprof::Clock::now();
  auto& pf = cppprof::profiler();
  for (int step = 0; step < num_tokens; ++step) {
    // step 0 uses prefill logits: its profiler window was opened at
    // "starting generation" above (prefill only). Decode steps reopen it.
    if (step > 0) pf.begin_step();
    if (step > 0) {
      auto tb = pf.now();
      logits = run_step(m, er, token);
      pf.add_build(cppprof::ms_since(tb));
      if (pf.enabled) { auto te = pf.now(); mx::eval(*logits); pf.add_final_eval(cppprof::ms_since(te)); }
      mem_telemetry("decode-step");
    }
    auto tsamp = pf.now();
    mx::array next = sample_greedy(*logits);
    mx::eval(next);
    pf.add_sample(cppprof::ms_since(tsamp));
    int next_id = static_cast<int>(next.data<uint32_t>()[0]);
    std::string digest;
    const bool want_digest = !pf.no_digest;
    if (want_digest) {
      auto td = pf.now();
      digest = digest_array(*logits);
      pf.add_digest(cppprof::ms_since(td));
    }
    records.push_back({step, token, next_id, digest});
    std::fprintf(stderr, "[step %d] in=%d out=%d sha=%s\n", step, token,
                 next_id, want_digest ? digest.c_str() : "NO_DIGEST");
    if (pf.enabled) {
      const double GIB = 1024.0*1024.0*1024.0;
      pf.step++;
      pf.write_row(mx::get_active_memory()/GIB, mx::get_cache_memory()/GIB, mx::get_peak_memory()/GIB);
    }
    token = next_id;
  }

  if (pf.enabled) {
    const double prefill_s = std::chrono::duration<double>(
                                 t_decode0 - t_gen0).count();
    const double decode_s = std::chrono::duration<double>(
                                cppprof::Clock::now() - t_decode0).count();
    std::fprintf(stderr,
                 "[timing] steps=%d prefill_s=%.4f decode_s=%.4f "
                 "decode_tok_s=%.4f (VEYSTRIX_CPP_NO_DIGEST=%d)\n",
                 num_tokens, prefill_s, decode_s,
                 decode_s > 0 ? num_tokens / decode_s : 0.0, (int)pf.no_digest);
  }

  if (m.expert_pool) {
    uint64_t hits = 0, misses = 0, coalesced = 0, fills = 0, bytes = 0, evict = 0;
    for (int k = 0; k < kNemoKinds; ++k) {
      const NemoPoolStats s = m.expert_pool->pool(k).stats();
      hits += s.hits; misses += s.misses; coalesced += s.coalesced_reads;
      fills += s.physical_reads; bytes += s.physical_bytes_read;
      evict += s.evictions;
    }
    const double hit_rate = (hits + misses) ? double(hits) / double(hits + misses) : 0.0;
    std::fprintf(stderr,
                 "[nemo-pool] hits=%llu misses=%llu coalesced=%llu fills=%llu "
                 "fill_bytes=%llu evictions=%llu hit_rate=%.4f\n",
                 (unsigned long long)hits, (unsigned long long)misses,
                 (unsigned long long)coalesced, (unsigned long long)fills,
                 (unsigned long long)bytes, (unsigned long long)evict, hit_rate);
  }
  return records;
}

}  // namespace

int main(int argc, char** argv) {
  cppprof::profiler().init();
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
    std::vector<StepRecord> records;
    {
      // Diagnostic: VEYSTRIX_CPP_GEN_THREAD=1 runs generation on a worker thread, as the C ABI
      // runtime's driver does, to separate thread effects from server effects.
      const char* gt = std::getenv("VEYSTRIX_CPP_GEN_THREAD");
      if (gt && std::strcmp(gt, "1") == 0) {
        std::exception_ptr err;
        std::thread worker([&] {
          try { records = run_generation(ckpt, cfg, num_tokens); }
          catch (...) { err = std::current_exception(); }
        });
        worker.join();
        if (err) std::rethrow_exception(err);
      } else {
        records = run_generation(ckpt, cfg, num_tokens);
      }
    }
    for (const auto& r : records)
      std::printf("step=%d in=%d out=%d sha256=%s\n", r.step, r.token_in,
                  r.token_out, r.digest.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "nemotron_generate failed: %s\n", e.what());
    return 1;
  }
}
