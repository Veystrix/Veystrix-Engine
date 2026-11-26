#include "nemo_snapshot.hpp"
#include "prefix_cache.hpp"

#include <cstring>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace veystrix::engine {
namespace {

void write_blob(std::ofstream& f, const TensorBlob& b) {
  for (int d : b.shape) f << d << " ";
  f << "\n" << b.dtype << "\n";
  f.write(reinterpret_cast<const char*>(b.bytes.data()),
          static_cast<std::streamsize>(b.bytes.size()));
}
TensorBlob read_blob(std::ifstream& f) {
  TensorBlob b;
  std::string line;
  if (!std::getline(f, line)) throw std::runtime_error("nemo_snapshot: shape line missing");
  std::istringstream ss(line);
  int d = 0;
  while (ss >> d) b.shape.push_back(d);
  if (!(f >> b.dtype)) throw std::runtime_error("nemo_snapshot: dtype missing");
  f.get();  // consume newline
  if (b.shape.empty()) throw std::runtime_error("nemo_snapshot: empty shape");
  size_t n = 1;
  for (int dim : b.shape) {
    if (dim <= 0) throw std::runtime_error("nemo_snapshot: bad shape dim");
    n *= static_cast<size_t>(dim);
  }
  int bytes_per = 4;
  if (b.dtype == static_cast<int>(mx::Dtype::Val::bfloat16) ||
      b.dtype == static_cast<int>(mx::Dtype::Val::float16)) bytes_per = 2;
  else if (b.dtype == static_cast<int>(mx::Dtype::Val::uint32) ||
           b.dtype == static_cast<int>(mx::Dtype::Val::int32) ||
           b.dtype == static_cast<int>(mx::Dtype::Val::float32)) bytes_per = 4;
  else throw std::runtime_error("nemo_snapshot: unsupported dtype");
  b.bytes.resize(n * bytes_per);
  f.read(reinterpret_cast<char*>(b.bytes.data()),
         static_cast<std::streamsize>(b.bytes.size()));
  if (static_cast<size_t>(f.gcount()) != b.bytes.size())
    throw std::runtime_error("nemo_snapshot: truncated tensor");
  return b;
}

std::string expect_key(std::ifstream& f, const char* key) {
  std::string got;
  if (!(f >> got) || got != key)
    throw std::runtime_error(std::string("nemo_snapshot: expected ") + key);
  std::string value;
  if (!(f >> value)) throw std::runtime_error("nemo_snapshot: missing value");
  return value;
}

}  // namespace

NemoSnapshot nemo_snapshot_capture(const nemotron_runtime::SessionCacheState& state,
                                   const std::vector<int32_t>& tokens,
                                   const std::string& fingerprint,
                                   bool int8) {
  NemoSnapshot snap;
  snap.tokens = tokens;
  snap.int8 = int8;
  snap.fingerprint = fingerprint;
  snap.chain_key = prefix_chain_key(fingerprint, tokens);
  for (const auto& mc : state.mamba_caches) {
    NemoMambaSnapshot m;
    m.conv = to_blob(mc.conv_state);
    m.ssm = to_blob(mc.ssm_state);
    snap.mamba.push_back(std::move(m));
  }
  if (int8) {
    for (const auto& ac : state.attn_int8) {
      NemoAttnSnapshot a;
      a.int8 = true;
      a.offset = ac.offset;
      a.block_tokens = ac.block_tokens;
      for (const auto& b : ac.blocks) {
        NemoAttnSnapshot::Block blk;
        blk.packed_k = to_blob(b.packed_k);
        blk.scales_k = to_blob(b.scales_k);
        blk.biases_k = to_blob(b.biases_k);
        blk.packed_v = to_blob(b.packed_v);
        blk.scales_v = to_blob(b.scales_v);
        blk.biases_v = to_blob(b.biases_v);
        blk.offset = b.offset;
        a.blocks.push_back(std::move(blk));
      }
      snap.attn.push_back(std::move(a));
    }
  } else {
    for (const auto& ac : state.attn_f32) {
      NemoAttnSnapshot a;
      if (!ac.keys || !ac.values)
        throw std::runtime_error("nemo_snapshot: bf16 cache has no K/V");
      a.keys = to_blob(*ac.keys);
      a.values = to_blob(*ac.values);
      a.offset = ac.offset;
      snap.attn.push_back(std::move(a));
    }
  }
  return snap;
}

void nemo_snapshot_save(const NemoSnapshot& snap, const std::string& directory) {
  std::string cmd = "mkdir -p '" + directory + "'";
  if (std::system(cmd.c_str()) != 0)
    throw std::runtime_error("nemo_snapshot: mkdir failed");
  std::ofstream meta(directory + "/meta.txt");
  meta << "chain_key " << snap.chain_key << "\n";
  meta << "n_tokens " << snap.tokens.size() << "\n";
  meta << "n_mamba " << snap.mamba.size() << "\n";
  meta << "n_attn " << snap.attn.size() << "\n";
  meta << "int8 " << (snap.int8 ? 1 : 0) << "\n";
  for (size_t i = 0; i < snap.tokens.size(); ++i) meta << "tok " << snap.tokens[i] << "\n";
  for (size_t i = 0; i < snap.mamba.size(); ++i) {
    std::ofstream f(directory + "/M" + std::to_string(i), std::ios::binary);
    write_blob(f, snap.mamba[i].conv);
    write_blob(f, snap.mamba[i].ssm);
  }
  for (size_t i = 0; i < snap.attn.size(); ++i) {
    const auto& a = snap.attn[i];
    std::ofstream f(directory + "/A" + std::to_string(i), std::ios::binary);
    f << a.offset << " " << a.block_tokens << " " << a.blocks.size() << "\n";
    if (a.int8) {
      for (const auto& b : a.blocks) {
        write_blob(f, b.packed_k); write_blob(f, b.scales_k); write_blob(f, b.biases_k);
        write_blob(f, b.packed_v); write_blob(f, b.scales_v); write_blob(f, b.biases_v);
        f << b.offset << "\n";
      }
    } else {
      write_blob(f, a.keys);
      write_blob(f, a.values);
    }
  }
}

nemotron_runtime::SessionCacheState nemo_snapshot_restore(const std::string& directory,
                                        const std::string& fingerprint,
                                        const std::vector<int32_t>& tokens) {
  std::ifstream meta(directory + "/meta.txt");
  if (!meta) throw std::runtime_error("nemo_snapshot: meta missing");
  const std::string chain = expect_key(meta, "chain_key");
  const size_t nt = static_cast<size_t>(std::stoul(expect_key(meta, "n_tokens")));
  const size_t nm = static_cast<size_t>(std::stoul(expect_key(meta, "n_mamba")));
  const size_t na = static_cast<size_t>(std::stoul(expect_key(meta, "n_attn")));
  const bool int8 = std::stoi(expect_key(meta, "int8")) != 0;
  std::vector<int32_t> saved(nt);
  for (size_t i = 0; i < nt; ++i) {
    std::string k;
    if (!(meta >> k >> saved[i]) || k != "tok")
      throw std::runtime_error("nemo_snapshot: bad token line");
  }
  if (saved != tokens || chain != prefix_chain_key(fingerprint, tokens))
    throw std::runtime_error("nemo_snapshot: chain/token mismatch (fail closed)");

  // Build the state directly from restored blobs; no zero-init needed (and
  // nemotron_runtime.cpp cannot be linked here — it owns its own main()).
  int block_tokens = 4096;
  if (na) {
    std::ifstream af(directory + "/A0", std::ios::binary);
    if (!af) throw std::runtime_error("nemo_snapshot: A0 missing");
    int offset = 0, nblocks = 0;
    if (!(af >> offset >> block_tokens >> nblocks))
      throw std::runtime_error("nemo_snapshot: bad attention header");
    af.get();  // consume header newline before read_blob's getline
  }
  nemotron_runtime::SessionCacheState state;
  // swap_in_cache reads BOTH arrays, so both must be sized even in int8 mode.
  state.attn_int8.resize(na);
  state.attn_f32.resize(na);
  for (size_t i = 0; i < nm; ++i) {
    std::ifstream f(directory + "/M" + std::to_string(i), std::ios::binary);
    if (!f) throw std::runtime_error("nemo_snapshot: mamba file missing");
    TensorBlob c = read_blob(f), s = read_blob(f);
    state.mamba_caches.push_back(MambaCacheState(clone_blob(c), clone_blob(s)));
  }
  for (size_t i = 0; i < na; ++i) {
    std::ifstream f(directory + "/A" + std::to_string(i), std::ios::binary);
    if (!f) throw std::runtime_error("nemo_snapshot: attention file missing");
    int offset = 0, nblocks = 0;
    if (!(f >> offset >> block_tokens >> nblocks))
      throw std::runtime_error("nemo_snapshot: bad attention header");
    f.get();  // consume header newline
    if (int8) {
      NemoInt8Cache& ac = state.attn_int8[i];
      ac.offset = offset;
      ac.block_tokens = block_tokens;
      ac.blocks.clear();
      for (int b = 0; b < nblocks; ++b) {
        TensorBlob pk = read_blob(f), sk = read_blob(f), bk = read_blob(f);
        TensorBlob pv = read_blob(f), sv = read_blob(f), bv = read_blob(f);
        int bo = 0;
        if (!(f >> bo)) throw std::runtime_error("nemo_snapshot: block offset missing");
        f.get();  // consume block-offset newline
        ac.blocks.emplace_back(clone_blob(pk), clone_blob(sk), clone_blob(bk),
                               clone_blob(pv), clone_blob(sv), clone_blob(bv), bo);
      }
    } else {
      TensorBlob k = read_blob(f), v = read_blob(f);
      state.attn_f32[i].keys = clone_blob(k);
      state.attn_f32[i].values = clone_blob(v);
      state.attn_f32[i].offset = offset;
    }
  }
  return state;
}

}  // namespace veystrix::engine
