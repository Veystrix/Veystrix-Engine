#include "generic/kv_snapshot_store.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>

#ifdef __APPLE__
#include <compression.h>
#endif

#include "generic/sha256.hpp"

namespace veystrix::engine::generic {

namespace fs = std::filesystem;

namespace {

constexpr char kManifestMagic[8] = {'V', 'X', 'K', 'V', '2', 0, 0, 0};
constexpr char kPageMagic[8] = {'V', 'X', 'P', 'G', '2', 0, 0, 0};
constexpr uint32_t kFormatVersion = 2;
constexpr size_t kMinPagedBytes = 4096;

WireDtype to_wire_dtype(mx::Dtype dt) {
  if (dt == mx::bool_) return WireDtype::Bool;
  if (dt == mx::uint8) return WireDtype::UInt8;
  if (dt == mx::uint16) return WireDtype::UInt16;
  if (dt == mx::uint32) return WireDtype::UInt32;
  if (dt == mx::uint64) return WireDtype::UInt64;
  if (dt == mx::int8) return WireDtype::Int8;
  if (dt == mx::int16) return WireDtype::Int16;
  if (dt == mx::int32) return WireDtype::Int32;
  if (dt == mx::int64) return WireDtype::Int64;
  if (dt == mx::float16) return WireDtype::Float16;
  if (dt == mx::float32) return WireDtype::Float32;
  if (dt == mx::bfloat16) return WireDtype::BFloat16;
  throw std::runtime_error("vxkv2: unsupported array dtype");
}

mx::Dtype from_wire_dtype(WireDtype w) {
  switch (w) {
    case WireDtype::Bool: return mx::bool_;
    case WireDtype::UInt8: return mx::uint8;
    case WireDtype::UInt16: return mx::uint16;
    case WireDtype::UInt32: return mx::uint32;
    case WireDtype::UInt64: return mx::uint64;
    case WireDtype::Int8: return mx::int8;
    case WireDtype::Int16: return mx::int16;
    case WireDtype::Int32: return mx::int32;
    case WireDtype::Int64: return mx::int64;
    case WireDtype::Float16: return mx::float16;
    case WireDtype::Float32: return mx::float32;
    case WireDtype::BFloat16: return mx::bfloat16;
  }
  throw std::runtime_error("vxkv2: unknown wire dtype");
}

size_t wire_dtype_itemsize(WireDtype w) {
  switch (w) {
    case WireDtype::Bool:
    case WireDtype::UInt8:
    case WireDtype::Int8:
      return 1;
    case WireDtype::UInt16:
    case WireDtype::Int16:
    case WireDtype::Float16:
    case WireDtype::BFloat16:
      return 2;
    case WireDtype::UInt32:
    case WireDtype::Int32:
    case WireDtype::Float32:
      return 4;
    case WireDtype::UInt64:
    case WireDtype::Int64:
      return 8;
  }
  throw std::runtime_error("vxkv2: unknown wire dtype");
}

bool ends_with(const std::string& s, const char* suffix) {
  const size_t slen = std::strlen(suffix);
  return s.size() >= slen && s.compare(s.size() - slen, slen, suffix) == 0;
}

SnapshotTransform choose_transform(const std::string& name, WireDtype dt, size_t nbytes) {
  if ((dt == WireDtype::BFloat16 || dt == WireDtype::Float16 || dt == WireDtype::Int16 ||
       dt == WireDtype::UInt16) &&
      (nbytes % 2 == 0)) {
    return SnapshotTransform::BytePlane2;
  }
  if (dt == WireDtype::UInt32 && (ends_with(name, ".k0") || ends_with(name, ".v0")) &&
      (nbytes % 4 == 0)) {
    return SnapshotTransform::BitPlane4;
  }
  if ((dt == WireDtype::Float32 || dt == WireDtype::Int32 || dt == WireDtype::UInt32) &&
      (nbytes % 4 == 0)) {
    return SnapshotTransform::BytePlane4;
  }
  return SnapshotTransform::None;
}

void apply_transform(SnapshotTransform tf, const uint8_t* src, uint8_t* dst, size_t nbytes) {
  switch (tf) {
    case SnapshotTransform::None:
      std::memcpy(dst, src, nbytes);
      break;
    case SnapshotTransform::BytePlane2:
      byte_plane_2_encode(src, dst, nbytes);
      break;
    case SnapshotTransform::BytePlane4:
      byte_plane_4_encode(src, dst, nbytes);
      break;
    case SnapshotTransform::BitPlane4:
      bit_plane_4_encode(src, dst, nbytes);
      break;
  }
}

void undo_transform(SnapshotTransform tf, const uint8_t* src, uint8_t* dst, size_t nbytes) {
  switch (tf) {
    case SnapshotTransform::None:
      std::memcpy(dst, src, nbytes);
      break;
    case SnapshotTransform::BytePlane2:
      byte_plane_2_decode(src, dst, nbytes);
      break;
    case SnapshotTransform::BytePlane4:
      byte_plane_4_decode(src, dst, nbytes);
      break;
    case SnapshotTransform::BitPlane4:
      bit_plane_4_decode(src, dst, nbytes);
      break;
  }
}

size_t compress_buffer(SnapshotCodec codec, const uint8_t* src, size_t src_len,
                       uint8_t* dst, size_t dst_cap) {
  if (codec == SnapshotCodec::None || src_len == 0) return 0;
#ifdef __APPLE__
  compression_algorithm alg =
      (codec == SnapshotCodec::LZFSE) ? COMPRESSION_LZFSE : COMPRESSION_LZ4;
  return compression_encode_buffer(dst, dst_cap, src, src_len, nullptr, alg);
#else
  (void)src;
  (void)src_len;
  (void)dst;
  (void)dst_cap;
  return 0;
#endif
}

size_t decompress_buffer(SnapshotCodec codec, const uint8_t* src, size_t src_len,
                         uint8_t* dst, size_t dst_cap) {
  if (codec == SnapshotCodec::None) {
    if (src_len != dst_cap) return 0;
    std::memcpy(dst, src, src_len);
    return src_len;
  }
#ifdef __APPLE__
  compression_algorithm alg =
      (codec == SnapshotCodec::LZFSE) ? COMPRESSION_LZFSE : COMPRESSION_LZ4;
  return compression_decode_buffer(dst, dst_cap, src, src_len, nullptr, alg);
#else
  (void)src;
  (void)src_len;
  (void)dst;
  (void)dst_cap;
  throw std::runtime_error("vxkv2: compressed snapshot requires Apple libcompression");
#endif
}

std::vector<uint8_t> encode_page_blob(const uint8_t* raw, size_t raw_len,
                                      SnapshotCodec want_codec,
                                      SnapshotTransform preferred_tf) {
  std::vector<uint8_t> best_payload;
  SnapshotCodec chosen_codec = SnapshotCodec::None;
  SnapshotTransform chosen_tf = SnapshotTransform::None;

  if (want_codec != SnapshotCodec::None && raw_len >= 64) {
    std::vector<uint8_t> transformed;
    const uint8_t* comp_in = raw;
    if (preferred_tf != SnapshotTransform::None) {
      transformed.resize(raw_len);
      apply_transform(preferred_tf, raw, transformed.data(), raw_len);
      comp_in = transformed.data();
    }
    std::vector<uint8_t> comp_buf(raw_len);
    size_t csz = compress_buffer(want_codec, comp_in, raw_len, comp_buf.data(), comp_buf.size());
    if (csz > 0 && csz < (raw_len * 98) / 100) {
      comp_buf.resize(csz);
      best_payload = std::move(comp_buf);
      chosen_codec = want_codec;
      chosen_tf = preferred_tf;
    } else if (preferred_tf == SnapshotTransform::BitPlane4) {
      size_t csz_raw = compress_buffer(want_codec, raw, raw_len, comp_buf.data(), comp_buf.size());
      if (csz_raw > 0 && csz_raw < (raw_len * 98) / 100) {
        comp_buf.resize(csz_raw);
        best_payload = std::move(comp_buf);
        chosen_codec = want_codec;
        chosen_tf = SnapshotTransform::None;
      }
    }
  }

  const size_t comp_len =
      (chosen_codec == SnapshotCodec::None) ? raw_len : best_payload.size();
  std::vector<uint8_t> out(24 + comp_len);
  std::memcpy(out.data(), kPageMagic, 8);
  out[8] = static_cast<uint8_t>(chosen_codec);
  out[9] = static_cast<uint8_t>(chosen_tf);
  out[10] = 0;
  out[11] = 0;
  uint32_t raw32 = static_cast<uint32_t>(raw_len);
  uint32_t comp32 = static_cast<uint32_t>(comp_len);
  uint32_t flags = 0;
  std::memcpy(out.data() + 12, &raw32, 4);
  std::memcpy(out.data() + 16, &comp32, 4);
  std::memcpy(out.data() + 20, &flags, 4);
  if (chosen_codec == SnapshotCodec::None) {
    std::memcpy(out.data() + 24, raw, raw_len);
  } else {
    std::memcpy(out.data() + 24, best_payload.data(), comp_len);
  }
  return out;
}

void decode_page_blob(const uint8_t* blob, size_t blob_len, uint8_t* dst, size_t expected_raw) {
  if (blob_len < 24 || std::memcmp(blob, kPageMagic, 8) != 0)
    throw std::runtime_error("vxkv2: corrupt CAS page header");
  const auto codec = static_cast<SnapshotCodec>(blob[8]);
  const auto tf = static_cast<SnapshotTransform>(blob[9]);
  uint32_t raw32 = 0, comp32 = 0;
  std::memcpy(&raw32, blob + 12, 4);
  std::memcpy(&comp32, blob + 16, 4);
  if (raw32 != expected_raw || 24 + static_cast<size_t>(comp32) != blob_len)
    throw std::runtime_error("vxkv2: CAS page size mismatch");
  if (tf == SnapshotTransform::None) {
    size_t dsz = decompress_buffer(codec, blob + 24, comp32, dst, expected_raw);
    if (dsz != expected_raw) throw std::runtime_error("vxkv2: CAS page decompress failed");
  } else {
    std::vector<uint8_t> tmp(expected_raw);
    size_t dsz = decompress_buffer(codec, blob + 24, comp32, tmp.data(), expected_raw);
    if (dsz != expected_raw) throw std::runtime_error("vxkv2: CAS page decompress failed");
    undo_transform(tf, tmp.data(), dst, expected_raw);
  }
}

void write_u8(std::vector<uint8_t>& b, uint8_t v) { b.push_back(v); }
void write_u32(std::vector<uint8_t>& b, uint32_t v) {
  const size_t p = b.size();
  b.resize(p + 4);
  std::memcpy(b.data() + p, &v, 4);
}
void write_u64(std::vector<uint8_t>& b, uint64_t v) {
  const size_t p = b.size();
  b.resize(p + 8);
  std::memcpy(b.data() + p, &v, 8);
}
void write_i64(std::vector<uint8_t>& b, int64_t v) {
  const size_t p = b.size();
  b.resize(p + 8);
  std::memcpy(b.data() + p, &v, 8);
}
void write_str(std::vector<uint8_t>& b, const std::string& s) {
  write_u32(b, static_cast<uint32_t>(s.size()));
  b.insert(b.end(), s.begin(), s.end());
}

struct Reader {
  const uint8_t* data;
  size_t len;
  size_t pos = 0;

  void need(size_t n) const {
    if (pos + n > len) throw std::runtime_error("vxkv2: truncated manifest");
  }
  uint8_t u8() {
    need(1);
    return data[pos++];
  }
  uint32_t u32() {
    need(4);
    uint32_t v = 0;
    std::memcpy(&v, data + pos, 4);
    pos += 4;
    return v;
  }
  uint64_t u64() {
    need(8);
    uint64_t v = 0;
    std::memcpy(&v, data + pos, 8);
    pos += 8;
    return v;
  }
  int64_t i64() {
    need(8);
    int64_t v = 0;
    std::memcpy(&v, data + pos, 8);
    pos += 8;
    return v;
  }
  std::string str() {
    uint32_t n = u32();
    need(n);
    std::string s(reinterpret_cast<const char*>(data + pos), n);
    pos += n;
    return s;
  }
  std::string fixed_str(size_t n) {
    need(n);
    std::string s(reinterpret_cast<const char*>(data + pos), n);
    pos += n;
    return s;
  }
};

std::string page_path_for(const std::string& cas_dir, const std::string& digest_hex) {
  return cas_dir + "/" + digest_hex.substr(0, 2) + "/" + digest_hex + ".vxpg";
}

void extract_4d_token_page(const uint8_t* base, int64_t B, int64_t H, int64_t T, int64_t D,
                           size_t elem_sz, int64_t t_lo, int64_t t_hi, uint8_t* out) {
  const size_t row_bytes = static_cast<size_t>(D) * elem_sz;
  const size_t slice_rows = static_cast<size_t>(t_hi - t_lo);
  const size_t slice_bytes = slice_rows * row_bytes;
  const size_t head_stride = static_cast<size_t>(T) * row_bytes;
  const size_t t_off = static_cast<size_t>(t_lo) * row_bytes;
  const int64_t bh = B * H;
  for (int64_t i = 0; i < bh; ++i) {
    std::memcpy(out + static_cast<size_t>(i) * slice_bytes,
                base + static_cast<size_t>(i) * head_stride + t_off, slice_bytes);
  }
}

void scatter_4d_token_page(const uint8_t* in, int64_t B, int64_t H, int64_t T, int64_t D,
                           size_t elem_sz, int64_t t_lo, int64_t t_hi, uint8_t* base) {
  const size_t row_bytes = static_cast<size_t>(D) * elem_sz;
  const size_t slice_rows = static_cast<size_t>(t_hi - t_lo);
  const size_t slice_bytes = slice_rows * row_bytes;
  const size_t head_stride = static_cast<size_t>(T) * row_bytes;
  const size_t t_off = static_cast<size_t>(t_lo) * row_bytes;
  const int64_t bh = B * H;
  for (int64_t i = 0; i < bh; ++i) {
    std::memcpy(base + static_cast<size_t>(i) * head_stride + t_off,
                in + static_cast<size_t>(i) * slice_bytes, slice_bytes);
  }
}

}  // namespace

void byte_plane_2_encode(const uint8_t* src, uint8_t* dst, size_t nbytes) {
  const size_t half = nbytes / 2;
  uint8_t* hi = dst;
  uint8_t* lo = dst + half;
  for (size_t i = 0; i < half; ++i) {
    lo[i] = src[2 * i];
    hi[i] = src[2 * i + 1];
  }
}

void byte_plane_2_decode(const uint8_t* src, uint8_t* dst, size_t nbytes) {
  const size_t half = nbytes / 2;
  const uint8_t* hi = src;
  const uint8_t* lo = src + half;
  for (size_t i = 0; i < half; ++i) {
    dst[2 * i] = lo[i];
    dst[2 * i + 1] = hi[i];
  }
}

void byte_plane_4_encode(const uint8_t* src, uint8_t* dst, size_t nbytes) {
  const size_t q = nbytes / 4;
  uint8_t* p0 = dst;
  uint8_t* p1 = dst + q;
  uint8_t* p2 = dst + 2 * q;
  uint8_t* p3 = dst + 3 * q;
  for (size_t i = 0; i < q; ++i) {
    p3[i] = src[4 * i];
    p2[i] = src[4 * i + 1];
    p1[i] = src[4 * i + 2];
    p0[i] = src[4 * i + 3];
  }
}

void byte_plane_4_decode(const uint8_t* src, uint8_t* dst, size_t nbytes) {
  const size_t q = nbytes / 4;
  const uint8_t* p0 = src;
  const uint8_t* p1 = src + q;
  const uint8_t* p2 = src + 2 * q;
  const uint8_t* p3 = src + 3 * q;
  for (size_t i = 0; i < q; ++i) {
    dst[4 * i] = p3[i];
    dst[4 * i + 1] = p2[i];
    dst[4 * i + 2] = p1[i];
    dst[4 * i + 3] = p0[i];
  }
}

void bit_plane_4_encode(const uint8_t* src, uint8_t* dst, size_t nbytes) {
  const size_t plane_bytes = nbytes / 4;
  uint8_t* p0 = dst;
  uint8_t* p1 = dst + plane_bytes;
  uint8_t* p2 = dst + 2 * plane_bytes;
  uint8_t* p3 = dst + 3 * plane_bytes;
  for (size_t g = 0; g < plane_bytes; ++g) {
    uint8_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    for (int k = 0; k < 4; ++k) {
      const uint8_t v = src[4 * g + k];
      const uint8_t lo = v & 0x0Fu;
      const uint8_t hi = (v >> 4) & 0x0Fu;
      const int shift_lo = 2 * k;
      const int shift_hi = 2 * k + 1;
      b0 |= static_cast<uint8_t>(((lo >> 0) & 1u) << shift_lo);
      b1 |= static_cast<uint8_t>(((lo >> 1) & 1u) << shift_lo);
      b2 |= static_cast<uint8_t>(((lo >> 2) & 1u) << shift_lo);
      b3 |= static_cast<uint8_t>(((lo >> 3) & 1u) << shift_lo);
      b0 |= static_cast<uint8_t>(((hi >> 0) & 1u) << shift_hi);
      b1 |= static_cast<uint8_t>(((hi >> 1) & 1u) << shift_hi);
      b2 |= static_cast<uint8_t>(((hi >> 2) & 1u) << shift_hi);
      b3 |= static_cast<uint8_t>(((hi >> 3) & 1u) << shift_hi);
    }
    p0[g] = b0;
    p1[g] = b1;
    p2[g] = b2;
    p3[g] = b3;
  }
}

void bit_plane_4_decode(const uint8_t* src, uint8_t* dst, size_t nbytes) {
  const size_t plane_bytes = nbytes / 4;
  const uint8_t* p0 = src;
  const uint8_t* p1 = src + plane_bytes;
  const uint8_t* p2 = src + 2 * plane_bytes;
  const uint8_t* p3 = src + 3 * plane_bytes;
  for (size_t g = 0; g < plane_bytes; ++g) {
    const uint8_t b0 = p0[g], b1 = p1[g], b2 = p2[g], b3 = p3[g];
    for (int k = 0; k < 4; ++k) {
      const int shift_lo = 2 * k;
      const int shift_hi = 2 * k + 1;
      const uint8_t lo = static_cast<uint8_t>(
          (((b0 >> shift_lo) & 1u) << 0) | (((b1 >> shift_lo) & 1u) << 1) |
          (((b2 >> shift_lo) & 1u) << 2) | (((b3 >> shift_lo) & 1u) << 3));
      const uint8_t hi = static_cast<uint8_t>(
          (((b0 >> shift_hi) & 1u) << 0) | (((b1 >> shift_hi) & 1u) << 1) |
          (((b2 >> shift_hi) & 1u) << 2) | (((b3 >> shift_hi) & 1u) << 3));
      dst[4 * g + k] = static_cast<uint8_t>(lo | (hi << 4));
    }
  }
}

std::string resolve_cas_dir(const std::string& snapshot_dir, const std::string& override_dir) {
  if (!override_dir.empty()) return override_dir;
  if (const char* env = std::getenv("VEYSTRIX_SNAPSHOT_CAS_DIR"); env && *env) {
    return std::string(env);
  }
  fs::path p(snapshot_dir);
  if (p.filename() == "vx_snapshot" && p.has_parent_path()) {
    fs::path parent = p.parent_path();
    if (parent.has_parent_path()) {
      fs::path grand = parent.parent_path();
      if (grand.filename() == "prefix-cache" || grand.filename() == "sessions") {
        return (grand / ".cas_pages").string();
      }
    }
  }
  return (p / ".cas_pages").string();
}

SnapshotCodec default_snapshot_codec() {
  const char* env = std::getenv("VEYSTRIX_SNAPSHOT_CODEC");
  if (!env || !*env) return SnapshotCodec::LZFSE;
  std::string s(env);
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (s == "lz4") return SnapshotCodec::LZ4;
  if (s == "none" || s == "raw" || s == "0") return SnapshotCodec::None;
  return SnapshotCodec::LZFSE;
}

bool is_vxkv2_snapshot(const std::string& snapshot_dir) {
  std::error_code ec;
  return fs::is_regular_file(fs::path(snapshot_dir) / "snapshot.vxkv2", ec);
}

SnapshotSaveStats save_vxkv2_raw(
    const std::string& snapshot_dir,
    const std::vector<HostTensor>& tensors,
    const std::unordered_map<std::string, std::string>& meta,
    const SnapshotSaveOptions& opts) {
  auto t0 = std::chrono::steady_clock::now();
  SnapshotSaveStats stats;
  fs::create_directories(snapshot_dir);
  const std::string cas_dir = resolve_cas_dir(snapshot_dir, opts.cas_dir);
  fs::create_directories(cas_dir);

  // Deterministic tensor ordering by name.
  std::vector<size_t> order(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return tensors[a].name < tensors[b].name;
  });

  struct PageTask {
    size_t entry_idx;
    int64_t t_lo;
    int64_t t_hi;
    std::string digest;
    uint64_t comp_size = 0;
    bool written = false;
  };
  struct TensorPlan {
    size_t src_idx = 0;
    bool paged = false;
    std::vector<uint8_t> inline_blob;
    std::vector<PageTask> pages;
  };

  std::vector<TensorPlan> plans(order.size());
  std::vector<std::pair<size_t, size_t>> flat_tasks;

  const int64_t pg = std::max<int64_t>(1, opts.page_tokens);
  for (size_t i = 0; i < order.size(); ++i) {
    const HostTensor& ht = tensors[order[i]];
    TensorPlan& tp = plans[i];
    tp.src_idx = order[i];
    stats.raw_bytes += ht.data.size();

    const bool can_page = (ht.shape.size() == 4 && ht.shape[2] > 0 && ht.data.size() >= kMinPagedBytes);
    if (can_page) {
      tp.paged = true;
      const int64_t T = ht.shape[2];
      for (int64_t t_lo = 0; t_lo < T; t_lo += pg) {
        const int64_t t_hi = std::min<int64_t>(T, t_lo + pg);
        tp.pages.push_back(PageTask{i, t_lo, t_hi, "", 0, false});
        flat_tasks.emplace_back(i, tp.pages.size() - 1);
      }
    } else {
      tp.paged = false;
      auto tf = choose_transform(ht.name, ht.dtype, ht.data.size());
      tp.inline_blob = encode_page_blob(ht.data.data(), ht.data.size(), opts.codec, tf);
    }
  }

  const int n_threads = std::clamp(opts.io_threads, 1, 4);
  std::atomic<size_t> next_task{0};
  std::mutex dir_mu;
  std::unordered_set<std::string> created_shards;
  std::exception_ptr worker_err;
  std::mutex err_mu;

  auto worker = [&]() {
    try {
      std::vector<uint8_t> page_buf;
      while (true) {
        const size_t idx = next_task.fetch_add(1);
        if (idx >= flat_tasks.size()) break;
        const auto [ei, pi] = flat_tasks[idx];
        TensorPlan& tp = plans[ei];
        PageTask& pt = tp.pages[pi];
        const HostTensor& ht = tensors[tp.src_idx];
        const int64_t B = ht.shape[0], H = ht.shape[1], T = ht.shape[2], D = ht.shape[3];
        const size_t elem_sz = wire_dtype_itemsize(ht.dtype);
        const size_t slice_bytes =
            static_cast<size_t>(B) * static_cast<size_t>(H) *
            static_cast<size_t>(pt.t_hi - pt.t_lo) * static_cast<size_t>(D) * elem_sz;
        page_buf.resize(slice_bytes);
        extract_4d_token_page(ht.data.data(), B, H, T, D, elem_sz, pt.t_lo, pt.t_hi, page_buf.data());

        uint32_t hdr[5] = {
            static_cast<uint32_t>(ht.dtype),
            static_cast<uint32_t>(B),
            static_cast<uint32_t>(H),
            static_cast<uint32_t>(pt.t_hi - pt.t_lo),
            static_cast<uint32_t>(D),
        };
        std::vector<uint8_t> hash_in(sizeof(hdr) + slice_bytes);
        std::memcpy(hash_in.data(), hdr, sizeof(hdr));
        std::memcpy(hash_in.data() + sizeof(hdr), page_buf.data(), slice_bytes);
        pt.digest = sha256_hex(hash_in.data(), hash_in.size());

        const std::string shard = pt.digest.substr(0, 2);
        const std::string final_path = page_path_for(cas_dir, pt.digest);
        std::error_code ec;
        if (fs::is_regular_file(final_path, ec)) {
          pt.comp_size = fs::file_size(final_path, ec);
          pt.written = false;
          continue;
        }
        {
          std::lock_guard<std::mutex> lk(dir_mu);
          if (created_shards.insert(shard).second) {
            fs::create_directories(cas_dir + "/" + shard, ec);
          }
        }
        auto tf = choose_transform(ht.name, ht.dtype, slice_bytes);
        std::vector<uint8_t> blob = encode_page_blob(page_buf.data(), slice_bytes, opts.codec, tf);
        pt.comp_size = blob.size();
        const std::string tmp_path =
            cas_dir + "/" + shard + "/." + pt.digest + "." +
            std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) + ".tmp";
        {
          std::ofstream wf(tmp_path, std::ios::binary);
          wf.write(reinterpret_cast<const char*>(blob.data()),
                   static_cast<std::streamsize>(blob.size()));
          wf.close();
          if (!wf) {
            fs::remove(tmp_path, ec);
            throw std::runtime_error("vxkv2: failed writing CAS page " + final_path);
          }
        }
        if (fs::is_regular_file(final_path, ec)) {
          fs::remove(tmp_path, ec);
          pt.written = false;
        } else {
          fs::rename(tmp_path, final_path, ec);
          if (ec) {
            fs::remove(tmp_path, ec);
            if (!fs::is_regular_file(final_path, ec)) {
              throw std::runtime_error("vxkv2: failed committing CAS page " + final_path);
            }
            pt.written = false;
          } else {
            pt.written = true;
          }
        }
      }
    } catch (...) {
      std::lock_guard<std::mutex> lk(err_mu);
      if (!worker_err) worker_err = std::current_exception();
    }
  };

  if (n_threads > 1 && flat_tasks.size() > 1) {
    std::vector<std::thread> pool;
    const int k = std::min<int>(n_threads, static_cast<int>(flat_tasks.size()));
    pool.reserve(static_cast<size_t>(k));
    for (int t = 0; t < k; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
  } else {
    worker();
  }
  if (worker_err) std::rethrow_exception(worker_err);

  std::vector<uint8_t> mf;
  mf.insert(mf.end(), kManifestMagic, kManifestMagic + 8);
  write_u32(mf, kFormatVersion);
  write_u32(mf, static_cast<uint32_t>(pg));
  std::vector<std::pair<std::string, std::string>> sorted_meta(meta.begin(), meta.end());
  std::sort(sorted_meta.begin(), sorted_meta.end());
  write_u32(mf, static_cast<uint32_t>(sorted_meta.size()));
  for (const auto& [k, v] : sorted_meta) {
    write_str(mf, k);
    write_str(mf, v);
  }
  write_u32(mf, static_cast<uint32_t>(plans.size()));
  for (const auto& tp : plans) {
    const HostTensor& ht = tensors[tp.src_idx];
    write_str(mf, ht.name);
    write_u8(mf, static_cast<uint8_t>(ht.dtype));
    write_u8(mf, static_cast<uint8_t>(ht.shape.size()));
    for (int64_t d : ht.shape) write_i64(mf, d);
    write_u8(mf, tp.paged ? 1 : 0);
    if (tp.paged) {
      write_u32(mf, static_cast<uint32_t>(tp.pages.size()));
      for (const auto& pt : tp.pages) {
        write_u32(mf, static_cast<uint32_t>(pt.t_lo));
        write_u32(mf, static_cast<uint32_t>(pt.t_hi));
        mf.insert(mf.end(), pt.digest.begin(), pt.digest.end());
        stats.pages_total += 1;
        stats.logical_comp_bytes += pt.comp_size;
        if (pt.written) {
          stats.pages_written += 1;
          stats.disk_bytes += pt.comp_size;
        } else {
          stats.pages_deduped += 1;
        }
      }
    } else {
      write_u64(mf, static_cast<uint64_t>(tp.inline_blob.size()));
      mf.insert(mf.end(), tp.inline_blob.begin(), tp.inline_blob.end());
    }
  }

  const std::string mf_path = snapshot_dir + "/snapshot.vxkv2";
  const std::string mf_tmp = mf_path + ".tmp";
  {
    std::ofstream f(mf_tmp, std::ios::binary);
    f.write(reinterpret_cast<const char*>(mf.data()), static_cast<std::streamsize>(mf.size()));
    f.close();
    if (!f) throw std::runtime_error("vxkv2: failed writing manifest " + mf_tmp);
  }
  std::error_code ec;
  fs::rename(mf_tmp, mf_path, ec);
  if (ec) {
    fs::remove(mf_tmp, ec);
    throw std::runtime_error("vxkv2: failed committing manifest " + mf_path);
  }
  fs::remove(snapshot_dir + "/generic_state.safetensors", ec);
  fs::remove(snapshot_dir + "/generic_state.safetensors.lz4", ec);

  stats.disk_bytes += mf.size();
  stats.logical_comp_bytes += mf.size();
  stats.elapsed_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
  return stats;
}

std::pair<std::vector<HostTensor>, std::unordered_map<std::string, std::string>>
load_vxkv2_raw(const std::string& snapshot_dir,
               const std::string& cas_dir_override,
               int io_threads,
               SnapshotRestoreStats* stats_out) {
  auto t0 = std::chrono::steady_clock::now();
  SnapshotRestoreStats local_stats;
  const std::string mf_path = snapshot_dir + "/snapshot.vxkv2";
  std::ifstream mf_in(mf_path, std::ios::binary | std::ios::ate);
  if (!mf_in) throw std::runtime_error("vxkv2: missing manifest " + mf_path);
  const auto mf_size = mf_in.tellg();
  if (mf_size < 16) throw std::runtime_error("vxkv2: corrupt manifest " + mf_path);
  mf_in.seekg(0);
  std::vector<uint8_t> mf(static_cast<size_t>(mf_size));
  if (!mf_in.read(reinterpret_cast<char*>(mf.data()), mf_size))
    throw std::runtime_error("vxkv2: failed reading manifest " + mf_path);
  local_stats.disk_bytes_read += static_cast<uint64_t>(mf_size);

  if (std::memcmp(mf.data(), kManifestMagic, 8) != 0)
    throw std::runtime_error("vxkv2: bad manifest magic");

  Reader rd{mf.data(), mf.size(), 8};
  const uint32_t version = rd.u32();
  if (version != kFormatVersion)
    throw std::runtime_error("vxkv2: unsupported manifest version " + std::to_string(version));
  (void)rd.u32();

  std::unordered_map<std::string, std::string> meta;
  const uint32_t n_meta = rd.u32();
  for (uint32_t i = 0; i < n_meta; ++i) {
    std::string k = rd.str();
    std::string v = rd.str();
    meta.emplace(std::move(k), std::move(v));
  }

  struct RestorePage {
    size_t tensor_idx;
    int64_t t_lo;
    int64_t t_hi;
    std::string digest;
  };

  const uint32_t n_tensors = rd.u32();
  std::vector<HostTensor> tensors(n_tensors);
  std::vector<RestorePage> pages;
  const std::string cas_dir = resolve_cas_dir(snapshot_dir, cas_dir_override);

  for (uint32_t i = 0; i < n_tensors; ++i) {
    HostTensor& ht = tensors[i];
    ht.name = rd.str();
    ht.dtype = static_cast<WireDtype>(rd.u8());
    const uint8_t ndim = rd.u8();
    ht.shape.resize(ndim);
    size_t count = 1;
    for (uint8_t d = 0; d < ndim; ++d) {
      int64_t dim = rd.i64();
      ht.shape[d] = dim;
      count *= static_cast<size_t>(dim);
    }
    const size_t nbytes = count * wire_dtype_itemsize(ht.dtype);
    local_stats.raw_bytes += nbytes;
    ht.data.resize(nbytes);
    const bool paged = (rd.u8() != 0);
    if (paged) {
      const uint32_t np = rd.u32();
      for (uint32_t p = 0; p < np; ++p) {
        const int64_t t_lo = static_cast<int64_t>(rd.u32());
        const int64_t t_hi = static_cast<int64_t>(rd.u32());
        std::string digest = rd.fixed_str(64);
        pages.push_back(RestorePage{i, t_lo, t_hi, std::move(digest)});
      }
    } else {
      const uint64_t blen = rd.u64();
      rd.need(static_cast<size_t>(blen));
      decode_page_blob(rd.data + rd.pos, static_cast<size_t>(blen), ht.data.data(), nbytes);
      rd.pos += static_cast<size_t>(blen);
    }
  }

  const int n_threads = std::clamp(io_threads, 1, 4);
  std::atomic<size_t> next_page{0};
  std::atomic<uint64_t> bytes_read{0};
  std::exception_ptr worker_err;
  std::mutex err_mu;

  auto worker = [&]() {
    try {
      std::vector<uint8_t> blob;
      std::vector<uint8_t> slice_buf;
      while (true) {
        const size_t idx = next_page.fetch_add(1);
        if (idx >= pages.size()) break;
        const RestorePage& rp = pages[idx];
        HostTensor& ht = tensors[rp.tensor_idx];
        const std::string ppath = page_path_for(cas_dir, rp.digest);
        std::ifstream pf(ppath, std::ios::binary | std::ios::ate);
        if (!pf) throw std::runtime_error("vxkv2: missing CAS page " + ppath);
        const auto fsz = pf.tellg();
        if (fsz < 24) throw std::runtime_error("vxkv2: corrupt CAS page " + ppath);
        pf.seekg(0);
        blob.resize(static_cast<size_t>(fsz));
        if (!pf.read(reinterpret_cast<char*>(blob.data()), fsz))
          throw std::runtime_error("vxkv2: failed reading CAS page " + ppath);
        bytes_read.fetch_add(static_cast<uint64_t>(fsz), std::memory_order_relaxed);

        const int64_t B = ht.shape[0], H = ht.shape[1], T = ht.shape[2], D = ht.shape[3];
        const size_t elem_sz = wire_dtype_itemsize(ht.dtype);
        const size_t slice_bytes =
            static_cast<size_t>(B) * static_cast<size_t>(H) *
            static_cast<size_t>(rp.t_hi - rp.t_lo) * static_cast<size_t>(D) * elem_sz;
        slice_buf.resize(slice_bytes);
        decode_page_blob(blob.data(), blob.size(), slice_buf.data(), slice_bytes);
        scatter_4d_token_page(slice_buf.data(), B, H, T, D, elem_sz, rp.t_lo, rp.t_hi,
                              ht.data.data());
      }
    } catch (...) {
      std::lock_guard<std::mutex> lk(err_mu);
      if (!worker_err) worker_err = std::current_exception();
    }
  };

  if (n_threads > 1 && pages.size() > 1) {
    std::vector<std::thread> pool;
    const int k = std::min<int>(n_threads, static_cast<int>(pages.size()));
    pool.reserve(static_cast<size_t>(k));
    for (int t = 0; t < k; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
  } else {
    worker();
  }
  if (worker_err) std::rethrow_exception(worker_err);

  local_stats.pages_read = pages.size();
  local_stats.disk_bytes_read += bytes_read.load();
  local_stats.elapsed_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0)
                               .count();
  if (stats_out) *stats_out = local_stats;
  return {std::move(tensors), std::move(meta)};
}

SnapshotSaveStats save_vxkv2_snapshot(
    const std::string& snapshot_dir,
    const std::unordered_map<std::string, mx::array>& arrays,
    const std::unordered_map<std::string, std::string>& meta,
    const SnapshotSaveOptions& opts) {
  std::vector<std::string> names;
  names.reserve(arrays.size());
  for (const auto& kv : arrays) names.push_back(kv.first);
  std::sort(names.begin(), names.end());

  std::vector<mx::array> eval_list;
  std::vector<mx::array> need_eval;
  eval_list.reserve(names.size());
  for (const auto& name : names) {
    const mx::array& src = arrays.at(name);
    if (src.flags().row_contiguous && src.is_available()) {
      eval_list.push_back(src);
    } else {
      mx::array c = mx::contiguous(src);
      eval_list.push_back(c);
      need_eval.push_back(c);
    }
  }
  if (!need_eval.empty()) mx::eval(need_eval);

  std::vector<HostTensor> host_tensors;
  host_tensors.reserve(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    const mx::array& a = eval_list[i];
    HostTensor ht;
    ht.name = names[i];
    ht.dtype = to_wire_dtype(a.dtype());
    ht.shape.reserve(a.ndim());
    for (int d : a.shape()) ht.shape.push_back(static_cast<int64_t>(d));
    ht.data.resize(a.nbytes());
    std::memcpy(ht.data.data(), a.data<char>(), a.nbytes());
    host_tensors.push_back(std::move(ht));
  }
  return save_vxkv2_raw(snapshot_dir, host_tensors, meta, opts);
}

std::pair<std::unordered_map<std::string, mx::array>, std::unordered_map<std::string, std::string>>
load_vxkv2_snapshot(const std::string& snapshot_dir,
                    const std::string& cas_dir_override,
                    int io_threads,
                    SnapshotRestoreStats* stats_out) {
  auto [host_tensors, meta] = load_vxkv2_raw(snapshot_dir, cas_dir_override, io_threads, stats_out);
  std::unordered_map<std::string, mx::array> arrays;
  arrays.reserve(host_tensors.size());
  for (const auto& ht : host_tensors) {
    mx::Shape sh;
    sh.reserve(ht.shape.size());
    for (int64_t d : ht.shape) sh.push_back(static_cast<int>(d));
    mx::array arr(ht.data.data(), sh, from_wire_dtype(ht.dtype));
    arrays.emplace(ht.name, std::move(arr));
  }
  return {std::move(arrays), std::move(meta)};
}

std::pair<uint64_t, uint64_t> gc_cas_pages(const std::string& prefix_cache_root,
                                           const std::string& cas_dir_in) {
  const std::string cas_dir =
      cas_dir_in.empty() ? (fs::path(prefix_cache_root) / ".cas_pages").string() : cas_dir_in;
  std::error_code ec;
  if (!fs::is_directory(cas_dir, ec)) return {0, 0};

  std::unordered_set<std::string> live;
  if (fs::is_directory(prefix_cache_root, ec)) {
    for (const auto& entry : fs::recursive_directory_iterator(prefix_cache_root, ec)) {
      if (!entry.is_regular_file(ec) || entry.path().filename() != "snapshot.vxkv2") continue;
      try {
        std::ifstream f(entry.path(), std::ios::binary | std::ios::ate);
        auto sz = f.tellg();
        if (sz < 16) continue;
        f.seekg(0);
        std::vector<uint8_t> buf(static_cast<size_t>(sz));
        if (!f.read(reinterpret_cast<char*>(buf.data()), sz)) continue;
        if (std::memcmp(buf.data(), kManifestMagic, 8) != 0) continue;
        Reader rd{buf.data(), buf.size(), 8};
        if (rd.u32() != kFormatVersion) continue;
        (void)rd.u32();
        uint32_t nm = rd.u32();
        for (uint32_t i = 0; i < nm; ++i) {
          (void)rd.str();
          (void)rd.str();
        }
        uint32_t nt = rd.u32();
        for (uint32_t i = 0; i < nt; ++i) {
          (void)rd.str();
          (void)rd.u8();
          uint8_t ndim = rd.u8();
          for (uint8_t d = 0; d < ndim; ++d) (void)rd.i64();
          bool paged = (rd.u8() != 0);
          if (paged) {
            uint32_t np = rd.u32();
            for (uint32_t p = 0; p < np; ++p) {
              (void)rd.u32();
              (void)rd.u32();
              live.insert(rd.fixed_str(64));
            }
          } else {
            uint64_t blen = rd.u64();
            rd.need(static_cast<size_t>(blen));
            rd.pos += static_cast<size_t>(blen);
          }
        }
      } catch (...) {
      }
    }
  }

  uint64_t removed_files = 0, reclaimed_bytes = 0;
  for (const auto& entry : fs::recursive_directory_iterator(cas_dir, ec)) {
    if (!entry.is_regular_file(ec) || entry.path().extension() != ".vxpg") continue;
    const std::string stem = entry.path().stem().string();
    if (live.find(stem) == live.end()) {
      const uint64_t fsz = entry.file_size(ec);
      if (fs::remove(entry.path(), ec)) {
        ++removed_files;
        reclaimed_bytes += fsz;
      }
    }
  }
  return {removed_files, reclaimed_bytes};
}

}  // namespace veystrix::engine::generic
