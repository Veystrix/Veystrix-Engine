#include "prefix_cache.hpp"

#include <mlx/mlx.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace veystrix::engine {
namespace {

// Minimal self-contained SHA-256 (FIPS 180-4). Used for the chain key so this
// module has no external dependency.
struct Sha256 {
  uint32_t h[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                   0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
  static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  static void compress(uint32_t h[8], const uint8_t block[64]) {
    static const uint32_t K[64] = {
      0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
      0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
      0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
      0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
      0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
      0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
      0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
      0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(block[4*i])<<24)|(uint32_t(block[4*i+1])<<16)|
             (uint32_t(block[4*i+2])<<8)|uint32_t(block[4*i+3]);
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
      uint32_t s1 = rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
      w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t S1 = rotr(e,6)^rotr(e,11)^rotr(e,25);
      uint32_t ch = (e&f)^(~e&g);
      uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      uint32_t S0 = rotr(a,2)^rotr(a,13)^rotr(a,22);
      uint32_t maj = (a&b)^(a&c)^(b&c);
      uint32_t t2 = S0 + maj;
      hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
  }
  std::string hex(const std::vector<uint8_t>& msg) {
    std::vector<uint8_t> data = msg;
    uint64_t bitlen = uint64_t(msg.size()) * 8;
    data.push_back(0x80);
    while (data.size() % 64 != 56) data.push_back(0);
    for (int i = 7; i >= 0; --i) data.push_back(uint8_t(bitlen >> (8*i)));
    for (size_t i = 0; i < data.size(); i += 64) compress(h, data.data()+i);
    static const char* d = "0123456789abcdef";
    std::string out;
    for (uint32_t v : h) for (int b = 3; b >= 0; --b) out += d[(v>>(8*b))&0xf];
    return out;
  }
};

size_t dtype_bytes(mx::Dtype::Val v) {
  switch (v) {
    case mx::Dtype::Val::float32: case mx::Dtype::Val::int32: case mx::Dtype::Val::uint32: return 4;
    case mx::Dtype::Val::bfloat16: case mx::Dtype::Val::float16:
    case mx::Dtype::Val::uint16: case mx::Dtype::Val::int16: return 2;
    case mx::Dtype::Val::int64: case mx::Dtype::Val::uint64: return 8;
    default: throw std::runtime_error("prefix_cache: unsupported dtype");
  }
}

TensorBlob to_blob(const mx::array& a) {
  // Force a contiguous materialization: cache tensors can be strided views
  // (e.g. transposed KV), and raw-data memcpy requires contiguous storage.
  mx::array ac = mx::contiguous(a);
  mx::eval(ac);
  TensorBlob b;
  b.shape.assign(a.shape().begin(), a.shape().end());
  auto v = static_cast<mx::Dtype::Val>(a.dtype());
  b.dtype = static_cast<int>(v);
  size_t n = a.size(), bytes = n * dtype_bytes(v);
  b.bytes.resize(bytes);
  switch (v) {
    case mx::Dtype::Val::float32: std::memcpy(b.bytes.data(), ac.data<float>(), bytes); break;
    case mx::Dtype::Val::bfloat16: std::memcpy(b.bytes.data(), ac.data<mx::bfloat16_t>(), bytes); break;
    case mx::Dtype::Val::float16: std::memcpy(b.bytes.data(), ac.data<mx::float16_t>(), bytes); break;
    case mx::Dtype::Val::int32: std::memcpy(b.bytes.data(), ac.data<int32_t>(), bytes); break;
    case mx::Dtype::Val::uint32: std::memcpy(b.bytes.data(), ac.data<uint32_t>(), bytes); break;
    case mx::Dtype::Val::int64: std::memcpy(b.bytes.data(), ac.data<int64_t>(), bytes); break;
    default: throw std::runtime_error("prefix_cache: to_blob dtype");
  }
  return b;
}

mx::array from_blob(const TensorBlob& b) {
  size_t n = 1;
  for (int d : b.shape) n *= size_t(d);
  auto v = static_cast<mx::Dtype::Val>(b.dtype);
  if (b.bytes.size() != n * dtype_bytes(v))
    throw std::runtime_error("prefix_cache: corrupt blob size");
  void* p = const_cast<uint8_t*>(b.bytes.data());
  mx::Shape shape(b.shape.begin(), b.shape.end());
  switch (v) {
    case mx::Dtype::Val::float32: return mx::array(static_cast<float*>(p), shape, mx::float32);
    case mx::Dtype::Val::bfloat16: return mx::array(static_cast<mx::bfloat16_t*>(p), shape, mx::bfloat16);
    case mx::Dtype::Val::float16: return mx::array(static_cast<mx::float16_t*>(p), shape, mx::float16);
    case mx::Dtype::Val::int32: return mx::array(static_cast<int32_t*>(p), shape, mx::int32);
    case mx::Dtype::Val::uint32: return mx::array(static_cast<uint32_t*>(p), shape, mx::uint32);
    case mx::Dtype::Val::int64: return mx::array(static_cast<int64_t*>(p), shape, mx::int64);
    default: throw std::runtime_error("prefix_cache: from_blob dtype");
  }
}

// Copy bytes into a fresh MLX array that OWNS its buffer (copy-on-write:
// restore must not alias the snapshot's stored bytes, and must outlive it).
mx::array clone_blob(const TensorBlob& b) {
  auto v = static_cast<mx::Dtype::Val>(b.dtype);
  size_t nbytes = b.bytes.size();
  void* raw = nullptr;
  if (::posix_memalign(&raw, 64, nbytes ? nbytes : 1) != 0)
    throw std::runtime_error("prefix_cache: posix_memalign failed");
  std::memcpy(raw, b.bytes.data(), nbytes);
  auto deleter = [](void* p) { std::free(p); };
  mx::Shape shape(b.shape.begin(), b.shape.end());
  switch (v) {
    case mx::Dtype::Val::float32: return mx::array(static_cast<float*>(raw), shape, mx::float32, deleter);
    case mx::Dtype::Val::bfloat16: return mx::array(static_cast<mx::bfloat16_t*>(raw), shape, mx::bfloat16, deleter);
    case mx::Dtype::Val::float16: return mx::array(static_cast<mx::float16_t*>(raw), shape, mx::float16, deleter);
    case mx::Dtype::Val::int32: return mx::array(static_cast<int32_t*>(raw), shape, mx::int32, deleter);
    case mx::Dtype::Val::uint32: return mx::array(static_cast<uint32_t*>(raw), shape, mx::uint32, deleter);
    case mx::Dtype::Val::int64: return mx::array(static_cast<int64_t*>(raw), shape, mx::int64, deleter);
    default: throw std::runtime_error("prefix_cache: clone_blob dtype");
  }
}

}  // namespace

std::string prefix_chain_key(const std::string& fingerprint,
                             const std::vector<int32_t>& tokens) {
  std::ostringstream os;
  os << fingerprint << "|";
  for (int32_t t : tokens) {
    uint32_t u = static_cast<uint32_t>(t);
    os << char((u>>24)&0xff) << char((u>>16)&0xff) << char((u>>8)&0xff) << char(u&0xff);
  }
  std::string s = os.str();
  std::vector<uint8_t> msg(s.begin(), s.end());
  Sha256 sha;
  return sha.hex(msg);
}

PrefixSnapshot prefix_capture(const std::vector<DecoderFullLayerCache>& caches,
                              const std::vector<int32_t>& tokens,
                              const std::string& fingerprint) {
  PrefixSnapshot snap;
  snap.tokens = tokens;
  snap.chain_key = prefix_chain_key(fingerprint, tokens);
  for (const auto& c : caches) {
    LayerSnapshot l;
    l.has_gdn = true;
    l.gdn_conv = to_blob(c.base.gdn.conv_state);
    l.gdn_recurrent = to_blob(c.base.gdn.recurrent_state);
    l.has_qsa = !c.base.qsa.empty();
    if (l.has_qsa) {
      l.qsa_keys = to_blob(*c.base.qsa.index_keys);
      l.qsa_positions = to_blob(*c.base.qsa.index_position_ids);
      l.qsa_offset = c.base.qsa.offset;
    }
    l.has_full = c.full.keys.has_value();
    if (l.has_full) {
      l.full_keys = to_blob(*c.full.keys);
      l.full_values = to_blob(*c.full.values);
      l.full_offset = c.full.offset;
    }
    snap.layers.push_back(std::move(l));
  }
  return snap;
}

PrefixSnapshot prefix_capture_with_logits(
    const std::vector<DecoderFullLayerCache>& caches,
    const std::vector<int32_t>& tokens,
    const std::string& fingerprint,
    const mx::array& logits,
    const std::vector<PLECache>& ple_caches) {
  PrefixSnapshot snap = prefix_capture(caches, tokens, fingerprint);
  snap.has_logits = true;
  snap.logits = to_blob(logits);
  for (size_t i = 0; i < snap.layers.size() && i < ple_caches.size(); ++i) {
    const auto& pc = ple_caches[i];
    if (pc.context.has_value() || pc.conv_state.has_value()) {
      snap.layers[i].has_ple = true;
      if (pc.context.has_value()) snap.layers[i].ple_context = to_blob(*pc.context);
      if (pc.conv_state.has_value()) snap.layers[i].ple_conv = to_blob(*pc.conv_state);
    }
  }
  return snap;
}

std::vector<DecoderFullLayerCache> prefix_restore(const PrefixSnapshot& snap,
                                                  int n_layers,
                                                  std::vector<PLECache>* ple_out) {
  if (static_cast<int>(snap.layers.size()) != n_layers)
    throw std::runtime_error("prefix_cache: layer count mismatch");
  if (ple_out) { ple_out->clear(); ple_out->resize(n_layers); }
  std::vector<DecoderFullLayerCache> out;
  out.reserve(n_layers);
  for (int li = 0; li < n_layers; ++li) {
    const auto& l = snap.layers[li];
    GDNCacheState gdn(clone_blob(l.gdn_conv), clone_blob(l.gdn_recurrent));
    QSACache qsa;
    if (l.has_qsa) {
      qsa.index_keys = clone_blob(l.qsa_keys);
      qsa.index_position_ids = clone_blob(l.qsa_positions);
      qsa.offset = l.qsa_offset;
    }
    DecoderLayerCache base(std::move(gdn), std::move(qsa));
    FullAttentionCache full;
    if (l.has_full) {
      full.keys = clone_blob(l.full_keys);
      full.values = clone_blob(l.full_values);
      full.offset = l.full_offset;
    }
    if (ple_out && l.has_ple) {
      if (!l.ple_context.bytes.empty()) (*ple_out)[li].context = clone_blob(l.ple_context);
      if (!l.ple_conv.bytes.empty()) (*ple_out)[li].conv_state = clone_blob(l.ple_conv);
    }
    out.emplace_back(std::move(base), std::move(full));
  }
  return out;
}

void prefix_save(const PrefixSnapshot& snap, const std::string& directory) {
  std::string cmd = "mkdir -p '" + directory + "'";
  if (std::system(cmd.c_str()) != 0)
    throw std::runtime_error("prefix_cache: mkdir failed");
  std::ofstream meta(directory + "/meta.txt");
  meta << "chain_key " << snap.chain_key << "\n";
  meta << "n_tokens " << snap.tokens.size() << "\n";
  meta << "n_layers " << snap.layers.size() << "\n";
  meta << "has_logits " << (snap.has_logits ? 1 : 0) << "\n";
  for (size_t i = 0; i < snap.tokens.size(); ++i) meta << "tok " << snap.tokens[i] << "\n";
  if (snap.has_logits) {
    std::ofstream f(directory + "/logits", std::ios::binary);
    for (int d : snap.logits.shape) f << d << " ";
    f << "\n" << snap.logits.dtype << "\n";
    f.write(reinterpret_cast<const char*>(snap.logits.bytes.data()), snap.logits.bytes.size());
  }
  for (size_t li = 0; li < snap.layers.size(); ++li) {
    const auto& l = snap.layers[li];
    std::string p = directory + "/L" + std::to_string(li);
    auto write_blob = [&](const std::string& suffix, const TensorBlob& b) {
      std::ofstream f(p + suffix, std::ios::binary);
      for (int d : b.shape) f << d << " ";
      f << "\n" << b.dtype << "\n";
      f.write(reinterpret_cast<const char*>(b.bytes.data()), b.bytes.size());
    };
    write_blob(".gdn_conv", l.gdn_conv);
    write_blob(".gdn_recur", l.gdn_recurrent);
    if (l.has_qsa) { write_blob(".qsa_keys", l.qsa_keys); write_blob(".qsa_pos", l.qsa_positions); }
    if (l.has_full) { write_blob(".full_keys", l.full_keys); write_blob(".full_vals", l.full_values); }
    if (l.has_ple) { write_blob(".ple_ctx", l.ple_context); write_blob(".ple_conv", l.ple_conv); }
    { std::ofstream f(p + ".offsets"); f << l.full_offset << " " << l.qsa_offset << "\n"; }
  }
}

PrefixSnapshot prefix_load(const std::string& directory,
                           const std::string& fingerprint,
                           const std::vector<int32_t>& tokens) {
  PrefixSnapshot snap;
  std::ifstream meta(directory + "/meta.txt");
  if (!meta) throw std::runtime_error("prefix_cache: meta missing");
  std::string key; meta >> key >> snap.chain_key;
  size_t nt = 0, nl = 0;
  int has_logits = 0;
  meta >> key >> nt; meta >> key >> nl; meta >> key >> has_logits;
  snap.tokens.resize(nt);
  for (size_t i = 0; i < nt; ++i) { meta >> key; meta >> snap.tokens[i]; }
  // Fail closed: verify fingerprint+token chain and exact token ids.
  if (snap.chain_key != prefix_chain_key(fingerprint, tokens) || snap.tokens != tokens)
    throw std::runtime_error("prefix_cache: chain/token mismatch");
  snap.layers.resize(nl);
  for (size_t li = 0; li < nl; ++li) {
    auto& l = snap.layers[li];
    std::string p = directory + "/L" + std::to_string(li);
    auto read_blob = [&](const std::string& suffix, TensorBlob& b) {
      std::ifstream f(p + suffix, std::ios::binary);
      if (!f) throw std::runtime_error("prefix_cache: blob missing " + suffix);
      std::string line; std::getline(f, line);
      std::istringstream ss(line); int d;
      while (ss >> d) b.shape.push_back(d);
      f >> b.dtype; f.ignore(1);
      size_t n = 1; for (int dd : b.shape) n *= size_t(dd);
      b.bytes.resize(n * dtype_bytes(static_cast<mx::Dtype::Val>(b.dtype)));
      f.read(reinterpret_cast<char*>(b.bytes.data()), b.bytes.size());
      if (!f) throw std::runtime_error("prefix_cache: truncated blob");
    };
    l.has_gdn = true;
    read_blob(".gdn_conv", l.gdn_conv);
    read_blob(".gdn_recur", l.gdn_recurrent);
    l.has_qsa = std::ifstream(p + ".qsa_keys").good();
    if (l.has_qsa) { read_blob(".qsa_keys", l.qsa_keys); read_blob(".qsa_pos", l.qsa_positions); }
    l.has_full = std::ifstream(p + ".full_keys").good();
    if (l.has_full) { read_blob(".full_keys", l.full_keys); read_blob(".full_vals", l.full_values); }
    l.has_ple = std::ifstream(p + ".ple_ctx").good();
    if (l.has_ple) { read_blob(".ple_ctx", l.ple_context); read_blob(".ple_conv", l.ple_conv); }
    { std::ifstream f(p + ".offsets"); if (f) f >> l.full_offset >> l.qsa_offset; }
  }
  snap.has_logits = (has_logits != 0);
  if (snap.has_logits) {
    std::ifstream f(directory + "/logits", std::ios::binary);
    if (!f) throw std::runtime_error("prefix_cache: logits missing");
    std::string line; std::getline(f, line);
    std::istringstream ss(line); int d;
    while (ss >> d) snap.logits.shape.push_back(d);
    f >> snap.logits.dtype; f.ignore(1);
    size_t n = 1; for (int dd : snap.logits.shape) n *= size_t(dd);
    snap.logits.bytes.resize(n * dtype_bytes(static_cast<mx::Dtype::Val>(snap.logits.dtype)));
    f.read(reinterpret_cast<char*>(snap.logits.bytes.data()), snap.logits.bytes.size());
    if (!f) throw std::runtime_error("prefix_cache: truncated logits");
  }
  return snap;
}

}  // namespace veystrix::engine
