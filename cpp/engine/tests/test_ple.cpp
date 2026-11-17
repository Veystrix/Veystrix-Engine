// Bit-exact ctest for the production Qwen4ExpPLELayer (layer index 1).
//   test_ple <vectors_ple-dir>
// For every case produced by gen_ple_vectors.py, C++ ple_forward must be
// byte-identical to the Python layer output (and both cache updates).
#include "ple.hpp"

#include <mlx/mlx.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

static int failures = 0;
#define CHECK(cond, msg)                                                     \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);         \
      ++failures;                                                            \
    }                                                                        \
  } while (0)

static bool bit_equal_f32(const mx::array& a, const mx::array& b) {
  if (a.shape() != b.shape()) return false;
  mx::array av = mx::view(mx::reshape(mx::astype(a, mx::float32), mx::Shape{-1}), mx::uint32);
  mx::array bv = mx::view(mx::reshape(mx::astype(b, mx::float32), mx::Shape{-1}), mx::uint32);
  mx::eval(av, bv);
  if (av.size() != bv.size()) return false;
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  for (size_t i = 0; i < av.size(); ++i)
    if (pa[i] != pb[i]) return false;
  return true;
}

static bool bit_equal_i64(const mx::array& a, const mx::array& b) {
  if (a.shape() != b.shape()) return false;
  mx::array av = mx::view(mx::reshape(mx::astype(a, mx::int64), mx::Shape{-1}), mx::int64);
  mx::array bv = mx::view(mx::reshape(mx::astype(b, mx::int64), mx::Shape{-1}), mx::int64);
  mx::eval(av, bv);
  if (av.size() != bv.size()) return false;
  const int64_t* pa = av.data<int64_t>();
  const int64_t* pb = bv.data<int64_t>();
  for (size_t i = 0; i < av.size(); ++i)
    if (pa[i] != pb[i]) return false;
  return true;
}

static std::string json_get(const std::string& s, const std::string& key) {
  std::string pat = "\"" + key + "\"";
  size_t p = s.find(pat);
  if (p == std::string::npos) return "";
  size_t colon = s.find(':', p + pat.size());
  size_t b = s.find_first_not_of(" \t\n\r", colon + 1);
  if (b == std::string::npos) return "";
  if (s[b] == '"') { size_t e = s.find('"', b + 1); return s.substr(b + 1, e - b - 1); }
  size_t e = s.find_first_of(",}\n", b);
  return s.substr(b, e - b);
}

static int json_int(const std::string& s, const std::string& key, int fallback = 0) {
  std::string v = json_get(s, key);
  return v.empty() ? fallback : std::stoi(v);
}

// Write a fixture shard tensor as raw little-endian rows and return a
// positioned-read descriptor pointing at it (the same shape as a safetensors
// tensor ref: absolute data offset + per-row byte stride).
static PLERowTensorRef write_rows_file(const std::string& path,
                                       const mx::array& a) {
  mx::eval(a);
  PLERowTensorRef r;
  r.data_offset = 0;
  r.elements_per_row = a.shape(1);
  size_t count = a.size();
  FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) throw std::runtime_error("cannot create " + path);
  if (a.dtype() == mx::uint32) {
    r.elem = PLEElemType::U32;
    r.row_bytes = a.shape(1) * 4;
    std::fwrite(a.data<uint32_t>(), 4, count, f);
  } else if (a.dtype() == mx::float32) {
    r.elem = PLEElemType::F32;
    r.row_bytes = a.shape(1) * 4;
    std::fwrite(a.data<float>(), 4, count, f);
  } else if (a.dtype() == mx::bfloat16) {
    r.elem = PLEElemType::BF16;
    r.row_bytes = a.shape(1) * 2;
    std::fwrite(a.data<mx::bfloat16_t>(), 2, count, f);
  } else if (a.dtype() == mx::float16) {
    r.elem = PLEElemType::F16;
    r.row_bytes = a.shape(1) * 2;
    std::fwrite(a.data<mx::float16_t>(), 2, count, f);
  } else {
    std::fclose(f);
    throw std::runtime_error("unsupported dtype in streamed fixture");
  }
  std::fclose(f);
  r.fd = ::open(path.c_str(), O_RDONLY);
  if (r.fd < 0) throw std::runtime_error("cannot open " + path);
  return r;
}

static std::string ple_tmp_root() {
  namespace fs = std::filesystem;
  return (fs::temp_directory_path() /
          ("ple_stream_" + std::to_string(::getpid()))).string();
}

static void report_diff(const char* label, const mx::array& a, const mx::array& b) {
  mx::array av = mx::view(mx::reshape(mx::astype(a, mx::float32), mx::Shape{-1}), mx::uint32);
  mx::array bv = mx::view(mx::reshape(mx::astype(b, mx::float32), mx::Shape{-1}), mx::uint32);
  mx::eval(av, bv);
  size_t n = av.size() < bv.size() ? av.size() : bv.size();
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  size_t diff = n;
  for (size_t i = 0; i < n; ++i) if (pa[i] != pb[i]) { diff = i; break; }
  std::fprintf(stderr, "  %s: first diff %zu (0x%08x vs 0x%08x)\n",
               label, diff, diff < n ? pa[diff] : 0, diff < n ? pb[diff] : 0);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <vectors_ple-dir>\n", argv[0]);
    return 2;
  }
  std::string dir = argv[1];
  std::ifstream mf(dir + "/meta.json");
  if (!mf) { std::fprintf(stderr, "meta.json missing\n"); return 2; }
  std::stringstream ms; ms << mf.rdbuf();
  std::string meta = ms.str();

  int nshards = json_int(meta, "split_ngram_parts", 4);
  int gs = json_int(meta, "quant_group_size", 32);
  int bits = json_int(meta, "quant_bits", 4);

  // Parse case directories.
  std::vector<std::string> cases;
  size_t pos = 0;
  while ((pos = meta.find("\"npy_dir\"", pos)) != std::string::npos) {
    size_t colon = meta.find(':', pos);
    size_t b = meta.find('"', colon);
    size_t e = meta.find('"', b + 1);
    cases.push_back(meta.substr(b + 1, e - b - 1));
    pos = e + 1;
  }
  if (cases.empty()) { std::fprintf(stderr, "no cases in meta.json\n"); return 2; }

  // ---- Compute-path fallback checks (need one case's config constants) ----
  {
    std::string cd = dir + "/" + cases[0];
    auto load = [&](const char* n) { return mx::load(cd + "/" + std::string(n) + ".npy"); };
    int vocab = json_int(meta, "vocab_size", 248320);
    int ns = json_int(meta, "ngram_size", 3);
    int hpn = json_int(meta, "heads_per_ngram", 8);
    int ple_idx = json_int(meta, "ple_layer_index", 0);
    int seed = json_int(meta, "config_seed", 1234);
    int base = json_int(meta, "ngram_vocab_size_base", 20000000);
    int nheads = (ns - 1) * hpn;
    auto mults = ple_build_layer_multipliers(vocab, ns, ple_idx, seed);
    auto sizes = ple_head_vocab_sizes(base, nheads, ple_idx);
    mx::array lm = load("layer_multipliers");
    mx::array hvs = load("head_vocab_sizes");
    std::vector<int64_t> lm_ref(lm.size()), hvs_ref(hvs.size());
    {
      mx::array lmi = mx::astype(mx::reshape(lm, mx::Shape{-1}), mx::int64);
      mx::array hvsi = mx::astype(mx::reshape(hvs, mx::Shape{-1}), mx::int64);
      mx::eval(lmi, hvsi);
      const int64_t* p = lmi.data<int64_t>();
      for (size_t i = 0; i < lm.size(); ++i) lm_ref[i] = p[i];
      p = hvsi.data<int64_t>();
      for (size_t i = 0; i < hvs.size(); ++i) hvs_ref[i] = p[i];
    }
    CHECK(mults == lm_ref, "computed layer_multipliers == fixture");
    CHECK(sizes == hvs_ref, "computed head_vocab_sizes == fixture");
    if (mults != lm_ref) {
      std::fprintf(stderr, "  mults computed:");
      for (auto v : mults) std::fprintf(stderr, " %lld", (long long)v);
      std::fprintf(stderr, "\n  mults fixture :");
      for (auto v : lm_ref) std::fprintf(stderr, " %lld", (long long)v);
      std::fprintf(stderr, "\n");
    }
    if (sizes != hvs_ref) {
      std::fprintf(stderr, "  sizes computed:");
      for (auto v : sizes) std::fprintf(stderr, " %lld", (long long)v);
      std::fprintf(stderr, "\n  sizes fixture :");
      for (auto v : hvs_ref) std::fprintf(stderr, " %lld", (long long)v);
      std::fprintf(stderr, "\n");
    }
  }

  for (const auto& c : cases) {
    std::string cd = dir + "/" + c;
    auto load = [&](const char* n) { return mx::load(cd + "/" + std::string(n) + ".npy"); };

    mx::array hidden = load("hidden_states"), ids = load("input_ids");
    mx::array kw = load("key_w"), ks = load("key_s"), kb = load("key_b");
    mx::array vw = load("val_w"), vs = load("val_s"), vb = load("val_b");
    mx::array nk = load("norm_key"), nq = load("norm_query"), nc = load("norm_conv");
    mx::array conv_w = load("conv_w");
    mx::array lm = load("layer_multipliers"), ho = load("head_offsets"),
              hvs = load("head_vocab_sizes"), shard_offsets = load("shard_offsets");
    mx::array cc0 = load("cache_conv"), ctx0 = load("cache_context");
    mx::array expected = load("expected");
    mx::array exp_cc = load("expected_cache_conv"), exp_ctx = load("expected_cache_context");
    bool has_mask = std::ifstream(cd + "/mask.npy").good();
    bool bf16_shards = std::ifstream(cd + "/shard_bf16.flag").good();
    std::optional<mx::array> mask;
    if (has_mask) mask = load("mask");

    // Build BOTH backends from the same fixture shard arrays.  The streamed
    // backend writes each shard's rows to a raw temp file so the C++ reader
    // exercises the real pread path (no full-shard mx::array materialization).
    namespace fs = std::filesystem;
    static const std::string tmp_root = ple_tmp_root();
    fs::create_directories(tmp_root);

    std::vector<PLEEmbShard> shards;
    std::vector<PLEStreamedShard> streamed;
    std::vector<int> open_fds;
    shards.reserve(nshards);
    streamed.reserve(nshards);
    for (int i = 0; i < nshards; ++i) {
      std::string pw = "shard_" + std::to_string(i) + "_w";
      std::string ps = "shard_" + std::to_string(i) + "_s";
      std::string pb_ = "shard_" + std::to_string(i) + "_b";
      mx::array sw = load(pw.c_str()), ss = load(ps.c_str()),
                sb = load(pb_.c_str());
      if (bf16_shards) {
        // Fixture stores BF16 rows as raw uint16 bits.
        ss = mx::view(ss, mx::bfloat16);
        sb = mx::view(sb, mx::bfloat16);
      }
      PLEStreamedShard st;
      st.weight = write_rows_file(tmp_root + "/s" + std::to_string(i) + "_w.bin", sw);
      st.scales = write_rows_file(tmp_root + "/s" + std::to_string(i) + "_s.bin", ss);
      st.biases = write_rows_file(tmp_root + "/s" + std::to_string(i) + "_b.bin", sb);
      open_fds.push_back(st.weight.fd);
      open_fds.push_back(st.scales.fd);
      open_fds.push_back(st.biases.fd);
      streamed.push_back(std::move(st));
      shards.emplace_back(std::move(sw), std::move(ss), std::move(sb));
    }

    std::vector<int64_t> offsets(shard_offsets.size());
    {
      mx::array o = mx::astype(mx::reshape(shard_offsets, mx::Shape{-1}), mx::int64);
      mx::eval(o);
      const int64_t* p = o.data<int64_t>();
      for (size_t i = 0; i < offsets.size(); ++i) offsets[i] = p[i];
    }

    Linear key_proj = Linear::quantized(kw, ks, kb, gs, bits, std::nullopt);
    Linear val_proj = Linear::quantized(vw, vs, vb, gs, bits, std::nullopt);

    PLEWeights w(
        key_proj, val_proj, nk, nq, nc, conv_w, lm, ho, hvs,
        std::move(shards), std::move(offsets),
        /*emb_group_size=*/gs, /*emb_bits=*/bits,
        /*ngram_size=*/json_int(meta, "ngram_size", 3),
        /*heads_per_ngram=*/json_int(meta, "heads_per_ngram", 8),
        /*hc_count=*/json_int(meta, "hc_count", 4),
        /*hidden_size=*/json_int(meta, "hidden_size", 2560),
        /*ple_embed_dim=*/json_int(meta, "ple_embed_dim", 0),
        /*eos_token_id=*/json_int(meta, "eos_token_id", 248044),
        /*eps=*/1e-6f,
        /*conv_kernel=*/json_int(meta, "ple_conv_kernel_size", 4),
        /*conv_dilation=*/json_int(meta, "conv_dilation", 3),
        /*short_conv_state_len=*/json_int(meta, "short_conv_state_len", 9),
        /*mode=*/"affine");

    // The synthetic oracle uses the unpatched language.py RMSNorm (1.0+w).
    w.norm_one_centered = false;

    PLECache cache(ctx0, cc0);  // (context, conv_state)
    mx::array out = ple_forward(hidden, ids, w, cache,
                                has_mask ? &*mask : nullptr);
    mx::eval(out, *cache.conv_state, *cache.context, expected, exp_cc, exp_ctx);

    if (!bit_equal_f32(out, expected)) report_diff("ple output", out, expected);
    CHECK(bit_equal_f32(out, expected), ("ple output: " + c).c_str());
    if (!bit_equal_f32(*cache.conv_state, exp_cc))
      report_diff("cache_conv", *cache.conv_state, exp_cc);
    CHECK(bit_equal_f32(*cache.conv_state, exp_cc), ("cache_conv: " + c).c_str());
    if (!bit_equal_i64(*cache.context, exp_ctx))
      CHECK(false, ("cache_context: " + c).c_str());
    else
      std::printf("  %s: cache_context OK\n", c.c_str());
    std::printf("  %s: %s\n", c.c_str(),
                bit_equal_f32(out, expected) ? "OK" : "MISMATCH");

    // ---- streamed backend: same output from positioned row reads -----------
    w.set_streamed_shards(std::move(streamed));
    PLECache cache2(ctx0, cc0);  // (context, conv_state)
    mx::array out2 = ple_forward(hidden, ids, w, cache2,
                                 has_mask ? &*mask : nullptr);
    mx::eval(out2, *cache2.conv_state, *cache2.context);
    if (!bit_equal_f32(out2, expected))
      report_diff("streamed ple output", out2, expected);
    CHECK(bit_equal_f32(out2, expected),
          ("streamed ple output: " + c).c_str());
    CHECK(bit_equal_f32(*cache2.conv_state, exp_cc),
          ("streamed cache_conv: " + c).c_str());
    CHECK(bit_equal_i64(*cache2.context, exp_ctx),
          ("streamed cache_context: " + c).c_str());
    std::printf("  %s streamed: %s\n", c.c_str(),
                bit_equal_f32(out2, expected) ? "OK" : "MISMATCH");
    for (int fd : open_fds) ::close(fd);
  }

  std::error_code ec;
  std::filesystem::remove_all(ple_tmp_root(), ec);

  if (failures) {
    std::fprintf(stderr, "test_ple FAILED (%d)\n", failures);
    return 1;
  }
  std::printf("test_ple OK (%zu cases)\n", cases.size());
  return 0;
}
