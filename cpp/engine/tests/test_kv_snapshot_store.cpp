// Unit tests + Apple Silicon compression/latency benchmark for VXKV2
// (Page-level Content-Addressed Storage + Byte-Plane/Bit-Plane Shuffled LZFSE/LZ4)
// and Phase 2 ultra-low-bit KV modes (k4v2 / k2v2).
#include "generic/kv_snapshot_store.hpp"
#include "generic/kv_store.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#include <compression.h>
#endif

namespace fs = std::filesystem;
using namespace veystrix::engine::generic;

namespace {

int g_failures = 0;

void check(bool cond, const char* msg) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    ++g_failures;
  }
}

uint16_t float_to_bf16(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

float bf16_to_float(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f = 0.0f;
  std::memcpy(&f, &u, 4);
  return f;
}

HostTensor make_realistic_bf16_kv(const std::string& name, int H, int T, int D, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> chan_scale(static_cast<size_t>(H * D));
  for (size_t i = 0; i < chan_scale.size(); ++i) {
    chan_scale[i] = ((i % 50) == 0) ? 3.5f : (0.15f + 0.05f * static_cast<float>(i % 5));
  }
  HostTensor ht;
  ht.name = name;
  ht.dtype = WireDtype::BFloat16;
  ht.shape = {1, H, T, D};
  const size_t n = static_cast<size_t>(H) * static_cast<size_t>(T) * static_cast<size_t>(D);
  ht.data.resize(n * 2);
  auto* dst = reinterpret_cast<uint16_t*>(ht.data.data());
  for (int h = 0; h < H; ++h) {
    for (int t = 0; t < T; ++t) {
      for (int d = 0; d < D; ++d) {
        const size_t idx = (static_cast<size_t>(h) * T + t) * D + d;
        dst[idx] = float_to_bf16(dist(rng) * chan_scale[static_cast<size_t>(h) * D + d]);
      }
    }
  }
  return ht;
}

std::vector<HostTensor> host_affine_quantize(const std::string& base_name, const HostTensor& bf16_t,
                                             int gs, int bits) {
  const int H = static_cast<int>(bf16_t.shape[1]);
  const int T = static_cast<int>(bf16_t.shape[2]);
  const int D = static_cast<int>(bf16_t.shape[3]);
  const int vals_per_u32 = 32 / bits;
  const int D_packed = D / vals_per_u32;
  const int D_groups = D / gs;
  const uint32_t qmax = (1u << bits) - 1u;
  const auto* src = reinterpret_cast<const uint16_t*>(bf16_t.data.data());

  HostTensor q{base_name + "0", WireDtype::UInt32, {1, H, T, D_packed},
               std::vector<uint8_t>(static_cast<size_t>(H) * T * D_packed * 4, 0)};
  HostTensor s{base_name + "1", WireDtype::BFloat16, {1, H, T, D_groups},
               std::vector<uint8_t>(static_cast<size_t>(H) * T * D_groups * 2, 0)};
  HostTensor b{base_name + "2", WireDtype::BFloat16, {1, H, T, D_groups},
               std::vector<uint8_t>(static_cast<size_t>(H) * T * D_groups * 2, 0)};

  auto* qbuf = reinterpret_cast<uint32_t*>(q.data.data());
  auto* sbuf = reinterpret_cast<uint16_t*>(s.data.data());
  auto* bbuf = reinterpret_cast<uint16_t*>(b.data.data());

  for (int h = 0; h < H; ++h) {
    for (int t = 0; t < T; ++t) {
      const size_t row_base = (static_cast<size_t>(h) * T + t) * D;
      const size_t q_base = (static_cast<size_t>(h) * T + t) * D_packed;
      const size_t g_base = (static_cast<size_t>(h) * T + t) * D_groups;
      for (int g = 0; g < D_groups; ++g) {
        float vmin = 1e30f, vmax = -1e30f;
        for (int i = 0; i < gs; ++i) {
          float v = bf16_to_float(src[row_base + g * gs + i]);
          if (v < vmin) vmin = v;
          if (v > vmax) vmax = v;
        }
        float scale = (vmax > vmin) ? ((vmax - vmin) / static_cast<float>(qmax)) : 1.0f;
        sbuf[g_base + g] = float_to_bf16(scale);
        bbuf[g_base + g] = float_to_bf16(vmin);
        float inv_s = 1.0f / scale;
        for (int i = 0; i < gs; ++i) {
          float v = bf16_to_float(src[row_base + g * gs + i]);
          int qv = static_cast<int>(std::lround((v - vmin) * inv_s));
          qv = std::clamp(qv, 0, static_cast<int>(qmax));
          const int d = g * gs + i;
          const int word = d / vals_per_u32;
          const int shift = (d % vals_per_u32) * bits;
          qbuf[q_base + word] |= (static_cast<uint32_t>(qv) << shift);
        }
      }
    }
  }
  return {std::move(q), std::move(s), std::move(b)};
}

HostTensor host_concat_axis2(const HostTensor& a, const HostTensor& b) {
  const int64_t H = a.shape[1], Ta = a.shape[2], Tb = b.shape[2], D = a.shape[3];
  const size_t esz = a.data.size() / static_cast<size_t>(H * Ta * D);
  const size_t row_bytes = static_cast<size_t>(D) * esz;
  HostTensor out;
  out.name = a.name;
  out.dtype = a.dtype;
  out.shape = {1, H, Ta + Tb, D};
  out.data.resize(static_cast<size_t>(H) * static_cast<size_t>(Ta + Tb) * row_bytes);
  for (int64_t h = 0; h < H; ++h) {
    uint8_t* dst = out.data.data() + static_cast<size_t>(h) * static_cast<size_t>(Ta + Tb) * row_bytes;
    std::memcpy(dst, a.data.data() + static_cast<size_t>(h) * static_cast<size_t>(Ta) * row_bytes,
                static_cast<size_t>(Ta) * row_bytes);
    std::memcpy(dst + static_cast<size_t>(Ta) * row_bytes,
                b.data.data() + static_cast<size_t>(h) * static_cast<size_t>(Tb) * row_bytes,
                static_cast<size_t>(Tb) * row_bytes);
  }
  return out;
}

const HostTensor& find_tensor(const std::vector<HostTensor>& vec, const std::string& name) {
  for (const auto& t : vec)
    if (t.name == name) return t;
  throw std::runtime_error("missing tensor: " + name);
}

bool tensors_bit_exact(const HostTensor& a, const HostTensor& b) {
  return a.dtype == b.dtype && a.shape == b.shape && a.data == b.data;
}

void test_transforms_roundtrip() {
  std::mt19937 rng(42);
  {
    std::vector<uint8_t> src(4096), enc(4096), dec(4096);
    for (auto& b : src) b = static_cast<uint8_t>(rng() & 0xFF);
    byte_plane_2_encode(src.data(), enc.data(), src.size());
    byte_plane_2_decode(enc.data(), dec.data(), enc.size());
    check(src == dec, "byte_plane_2 round-trip exact");
  }
  {
    std::vector<uint8_t> src(4096), enc(4096), dec(4096);
    for (auto& b : src) b = static_cast<uint8_t>(rng() & 0xFF);
    byte_plane_4_encode(src.data(), enc.data(), src.size());
    byte_plane_4_decode(enc.data(), dec.data(), enc.size());
    check(src == dec, "byte_plane_4 round-trip exact");
  }
  {
    std::vector<uint8_t> src(4096), enc(4096), dec(4096);
    for (auto& b : src) b = static_cast<uint8_t>(rng() & 0xFF);
    bit_plane_4_encode(src.data(), enc.data(), src.size());
    bit_plane_4_decode(enc.data(), dec.data(), enc.size());
    check(src == dec, "bit_plane_4 round-trip exact");
  }
  std::printf("[PASS] test_transforms_roundtrip\n");
}

void test_vxkv2_bit_exact_and_cas_dedup() {
  fs::path tmp = fs::temp_directory_path() / "veystrix_test_vxkv2_dedup";
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::path pc_root = tmp / "prefix-cache";
  fs::path snap1 = pc_root / "turn1" / "vx_snapshot";
  fs::path snap2 = pc_root / "turn2" / "vx_snapshot";
  fs::path snap_branch = pc_root / "branch_b" / "vx_snapshot";

  auto k_pre = make_realistic_bf16_kv("l0.prefix.k0", 8, 1024, 128, 101);
  auto v_pre = make_realistic_bf16_kv("l0.prefix.v0", 8, 1024, 128, 102);
  auto k_mid_bf = make_realistic_bf16_kv("tmp", 8, 1024, 128, 103);
  auto v_mid_bf = make_realistic_bf16_kv("tmp", 8, 1024, 128, 104);
  auto km = host_affine_quantize("l0.middle.k", k_mid_bf, 64, 4);
  auto vm = host_affine_quantize("l0.middle.v", v_mid_bf, 64, 4);
  auto k_rec1 = make_realistic_bf16_kv("l0.recent.k0", 8, 512, 128, 105);
  auto v_rec1 = make_realistic_bf16_kv("l0.recent.v0", 8, 512, 128, 106);

  HostTensor chain_t{"chain", WireDtype::Int32, {2560}, std::vector<uint8_t>(2560 * 4, 7)};
  HostTensor logits_t = make_realistic_bf16_kv("logits", 1, 1, 1024, 107);

  std::vector<HostTensor> a1{k_pre, v_pre, km[0], km[1], km[2], vm[0], vm[1], vm[2],
                             k_rec1, v_rec1, chain_t, logits_t};
  std::unordered_map<std::string, std::string> meta1{
      {"fingerprint", "test-fp-1"}, {"n_tokens", "2560"},
  };

  for (SnapshotCodec codec : {SnapshotCodec::LZFSE, SnapshotCodec::LZ4, SnapshotCodec::None}) {
    fs::remove_all(tmp, ec);
    SnapshotSaveOptions opts;
    opts.codec = codec;
    opts.page_tokens = 256;
    opts.io_threads = 4;

    auto s1 = save_vxkv2_raw(snap1.string(), a1, meta1, opts);
    check(is_vxkv2_snapshot(snap1.string()), "is_vxkv2_snapshot true");
    check(s1.pages_total > 0 && s1.pages_written == s1.pages_total, "turn1 writes all pages");

    SnapshotRestoreStats r1;
    auto [loaded1, lmeta1] = load_vxkv2_raw(snap1.string(), "", 4, &r1);
    check(lmeta1["fingerprint"] == "test-fp-1", "meta fingerprint exact");
    check(loaded1.size() == a1.size(), "restored tensor count matches");
    for (const auto& orig : a1) {
      check(tensors_bit_exact(orig, find_tensor(loaded1, orig.name)),
            ("bit-exact restore: " + orig.name).c_str());
    }

    // Turn 2: identical prefix + middle + recent extended by 256 tokens
    auto k_ext = make_realistic_bf16_kv("l0.recent.k0", 8, 256, 128, 205);
    auto v_ext = make_realistic_bf16_kv("l0.recent.v0", 8, 256, 128, 206);
    auto k_rec2 = host_concat_axis2(k_rec1, k_ext);
    auto v_rec2 = host_concat_axis2(v_rec1, v_ext);
    auto a2 = a1;
    for (auto& t : a2) {
      if (t.name == "l0.recent.k0") t = k_rec2;
      if (t.name == "l0.recent.v0") t = v_rec2;
    }
    auto s2 = save_vxkv2_raw(snap2.string(), a2, {{"fingerprint", "test-fp-1"}, {"n_tokens", "2816"}}, opts);
    check(s2.pages_deduped == s1.pages_total, "turn2 deduplicates 100% of turn1 pages");
    check(s2.pages_written == 2, "turn2 writes only the 2 new delta pages");

    // Branch B: forks from turn 1 with different 256-token continuation
    auto k_b = make_realistic_bf16_kv("l0.recent.k0", 8, 256, 128, 305);
    auto v_b = make_realistic_bf16_kv("l0.recent.v0", 8, 256, 128, 306);
    auto ab = a1;
    for (auto& t : ab) {
      if (t.name == "l0.recent.k0") t = host_concat_axis2(k_rec1, k_b);
      if (t.name == "l0.recent.v0") t = host_concat_axis2(v_rec1, v_b);
    }
    auto sb = save_vxkv2_raw(snap_branch.string(), ab, {{"fingerprint", "test-fp-1"}, {"n_tokens", "2816"}}, opts);
    check(sb.pages_deduped == s1.pages_total, "branch_b deduplicates 100% of shared pages");
    check(sb.pages_written == 2, "branch_b writes only 2 branch-unique pages");

    auto [loaded2, _m2] = load_vxkv2_raw(snap2.string(), "", 4);
    check(tensors_bit_exact(k_rec2, find_tensor(loaded2, "l0.recent.k0")), "turn2 recent.k0 bit-exact");
    auto [loadedb, _mb] = load_vxkv2_raw(snap_branch.string(), "", 4);
    check(tensors_bit_exact(find_tensor(ab, "l0.recent.k0"), find_tensor(loadedb, "l0.recent.k0")),
          "branch_b recent.k0 bit-exact");

    // Delete turn2 and branch_b, then run GC -> should delete the 4 delta pages and keep turn1 intact
    fs::remove_all(pc_root / "turn2", ec);
    fs::remove_all(pc_root / "branch_b", ec);
    auto [gc_removed, gc_bytes] = gc_cas_pages(pc_root.string());
    check(gc_removed == 4, "gc removes exactly 4 orphaned delta pages");
    check(gc_bytes > 0, "gc reclaims positive bytes");
    auto [loaded1_after, _m1a] = load_vxkv2_raw(snap1.string(), "", 4);
    check(tensors_bit_exact(k_pre, find_tensor(loaded1_after, "l0.prefix.k0")),
          "turn1 still bit-exact after GC");
  }

  fs::remove_all(tmp, ec);
  std::printf("[PASS] test_vxkv2_bit_exact_and_cas_dedup (LZFSE, LZ4, None — 0 ULP bit-exact)\n");
}

void test_phase2_low_bit_kv_policy() {
  auto p_k4v2 = KVPolicy::from_mode("sparse-paged-k4v2");
  check(p_k4v2.hist_k_bits == 4 && p_k4v2.hist_v_bits == 2 && p_k4v2.sparse_paged, "sparse-paged-k4v2 parsed");
  auto p_k2v2 = KVPolicy::from_mode("sparse-paged-k2v2");
  check(p_k2v2.hist_k_bits == 2 && p_k2v2.hist_v_bits == 2 && p_k2v2.sparse_paged, "sparse-paged-k2v2 parsed");
  auto p_k3v2 = KVPolicy::from_mode("k3v2");
  check(p_k3v2.hist_k_bits == 3 && p_k3v2.hist_v_bits == 2, "k3v2 parsed");
  double bpv_int4 = KVPolicy::from_mode("sparse-paged").bytes_per_value(1000000);
  double bpv_k4v2 = p_k4v2.bytes_per_value(1000000);
  double bpv_k2v2 = p_k2v2.bytes_per_value(1000000);
  check(bpv_k4v2 < bpv_int4 && bpv_k2v2 < bpv_k4v2, "monotonic bytes_per_value reduction");
  std::printf("[PASS] test_phase2_low_bit_kv_policy (1M bpv: int4=%.4f, k4v2=%.4f, k2v2=%.4f)\n",
              bpv_int4, bpv_k4v2, bpv_k2v2);
}

void run_benchmark() {
  fs::path tmp = fs::temp_directory_path() / "veystrix_bench_vxkv2";
  std::error_code ec;
  fs::remove_all(tmp, ec);

  // Exact Inkling 2 Mini Looped MoE geometry: 8 full-attn visits + 40 SW visits (1024 cap), 8 KV heads, D=128
  const int n_full = 8;
  const int n_sw = 40;
  const int H = 8, D = 128;
  const int pre_T = 1024;
  const int rec_T = 2048;
  const int mid_T = 4096;  // 16 pages of aged middle history per full-attention visit
  const int sw_T = 1024;

  std::vector<HostTensor> arrays_int4, arrays_k4v2, arrays_k2v2;

  for (int l = 0; l < n_full; ++l) {
    std::string pfx = "f" + std::to_string(l);
    auto kp = make_realistic_bf16_kv(pfx + ".prefix.k0", H, pre_T, D, 1000 + l * 10 + 1);
    auto vp = make_realistic_bf16_kv(pfx + ".prefix.v0", H, pre_T, D, 1000 + l * 10 + 2);
    auto kr = make_realistic_bf16_kv(pfx + ".recent.k0", H, rec_T, D, 1000 + l * 10 + 3);
    auto vr = make_realistic_bf16_kv(pfx + ".recent.v0", H, rec_T, D, 1000 + l * 10 + 4);
    auto km_bf = make_realistic_bf16_kv("tmp", H, mid_T, D, 1000 + l * 10 + 5);
    auto vm_bf = make_realistic_bf16_kv("tmp", H, mid_T, D, 1000 + l * 10 + 6);
    auto km4 = host_affine_quantize(pfx + ".middle.k", km_bf, 64, 4);
    auto vm4 = host_affine_quantize(pfx + ".middle.v", vm_bf, 64, 4);
    auto km2 = host_affine_quantize(pfx + ".middle.k", km_bf, 64, 2);
    auto vm2 = host_affine_quantize(pfx + ".middle.v", vm_bf, 64, 2);

    auto push_common = [&](std::vector<HostTensor>& dst) {
      dst.push_back(kp); dst.push_back(vp); dst.push_back(kr); dst.push_back(vr);
    };
    push_common(arrays_int4);
    for (const auto& t : km4) arrays_int4.push_back(t);
    for (const auto& t : vm4) arrays_int4.push_back(t);

    push_common(arrays_k4v2);
    for (const auto& t : km4) arrays_k4v2.push_back(t);
    for (const auto& t : vm2) arrays_k4v2.push_back(t);

    push_common(arrays_k2v2);
    for (const auto& t : km2) arrays_k2v2.push_back(t);
    for (const auto& t : vm2) arrays_k2v2.push_back(t);
  }
  for (int l = 0; l < n_sw; ++l) {
    std::string pfx = "sw" + std::to_string(l);
    auto ks = make_realistic_bf16_kv(pfx + ".recent.k0", H, sw_T, D, 2000 + l * 2);
    auto vs = make_realistic_bf16_kv(pfx + ".recent.v0", H, sw_T, D, 2000 + l * 2 + 1);
    arrays_int4.push_back(ks); arrays_int4.push_back(vs);
    arrays_k4v2.push_back(ks); arrays_k4v2.push_back(vs);
    arrays_k2v2.push_back(ks); arrays_k2v2.push_back(vs);
  }

  std::unordered_map<std::string, std::string> meta{{"fingerprint", "inkling2-bench"}, {"n_tokens", "7168"}};

  // 1. Legacy monolithic buffer + raw LZ4 baseline
  uint64_t leg_raw_sz = 0;
  for (const auto& t : arrays_int4) leg_raw_sz += t.data.size();
  std::vector<uint8_t> mono(leg_raw_sz);
  size_t off = 0;
  for (const auto& t : arrays_int4) {
    std::memcpy(mono.data() + off, t.data.data(), t.data.size());
    off += t.data.size();
  }
  fs::path leg_dir = tmp / "legacy";
  fs::create_directories(leg_dir);
  std::string leg_path = (leg_dir / "generic_state.safetensors.lz4").string();
  auto t0 = std::chrono::steady_clock::now();
  uint64_t leg_disk_sz = leg_raw_sz;
#ifdef __APPLE__
  {
    std::vector<uint8_t> dst(leg_raw_sz);
    size_t csz = compression_encode_buffer(dst.data(), dst.size(), mono.data(), mono.size(), nullptr, COMPRESSION_LZ4);
    if (csz > 0 && csz + 16 < (leg_raw_sz * 95) / 100) {
      std::ofstream lf(leg_path, std::ios::binary);
      lf.write("VXLZ4\0\0\0", 8);
      lf.write(reinterpret_cast<const char*>(&leg_raw_sz), 8);
      lf.write(reinterpret_cast<const char*>(dst.data()), static_cast<std::streamsize>(csz));
      leg_disk_sz = csz + 16;
    } else {
      std::ofstream rf(leg_path, std::ios::binary);
      rf.write(reinterpret_cast<const char*>(mono.data()), static_cast<std::streamsize>(mono.size()));
    }
  }
#endif
  double leg_save_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  auto tr0 = std::chrono::steady_clock::now();
  {
    std::ifstream rf(leg_path, std::ios::binary | std::ios::ate);
    auto fsz = rf.tellg();
    rf.seekg(0);
    std::vector<uint8_t> src(static_cast<size_t>(fsz));
    rf.read(reinterpret_cast<char*>(src.data()), fsz);
    if (leg_disk_sz < leg_raw_sz) {
      std::vector<uint8_t> dst(leg_raw_sz);
#ifdef __APPLE__
      compression_decode_buffer(dst.data(), dst.size(), src.data() + 16, src.size() - 16, nullptr, COMPRESSION_LZ4);
#endif
      std::string tmp_st = (leg_dir / ".restore.safetensors.tmp").string();
      std::ofstream tf(tmp_st, std::ios::binary);
      tf.write(reinterpret_cast<const char*>(dst.data()), static_cast<std::streamsize>(dst.size()));
      tf.close();
      fs::remove(tmp_st, ec);
    }
  }
  double leg_load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr0).count();

  std::printf("\n=== Inkling 2 Mini Looped MoE KV Snapshot Benchmark (8 full + 40 SW visits, 7,168 tok) ===\n");
  std::printf("%-36s %9s %9s %7s %9s %9s\n", "Format / Mode", "Raw MiB", "Disk MiB", "Ratio", "Save ms", "Load ms");
  std::printf("%-36s %9.2f %9.2f %6.2fx %9.1f %9.1f\n",
              "Legacy monolithic LZ4 (INT4)", leg_raw_sz / 1048576.0, leg_disk_sz / 1048576.0,
              static_cast<double>(leg_raw_sz) / leg_disk_sz, leg_save_ms, leg_load_ms);

  auto bench_vxkv2 = [&](const char* label, const std::vector<HostTensor>& arrs,
                         SnapshotCodec codec) {
    fs::path d = tmp / label / "prefix-cache" / "turn1" / "vx_snapshot";
    SnapshotSaveOptions o;
    o.codec = codec;
    o.page_tokens = 256;
    o.io_threads = 4;
    auto s = save_vxkv2_raw(d.string(), arrs, meta, o);
    SnapshotRestoreStats r;
    auto [loaded, _m] = load_vxkv2_raw(d.string(), "", 4, &r);
    std::printf("%-36s %9.2f %9.2f %6.2fx %9.1f %9.1f\n",
                label, s.raw_bytes / 1048576.0, s.logical_comp_bytes / 1048576.0,
                static_cast<double>(s.raw_bytes) / s.logical_comp_bytes, s.elapsed_ms, r.elapsed_ms);

    if (std::strcmp(label, "VXKV2 CAS+BytePlane LZFSE (INT4)") == 0) {
      auto arrs2 = arrs;
      for (int l = 0; l < n_full; ++l) {
        std::string pfx = "f" + std::to_string(l);
        auto ke = make_realistic_bf16_kv(pfx + ".recent.k0", H, 256, D, 9000 + l * 2);
        auto ve = make_realistic_bf16_kv(pfx + ".recent.v0", H, 256, D, 9000 + l * 2 + 1);
        for (auto& t : arrs2) {
          if (t.name == pfx + ".recent.k0") t = host_concat_axis2(t, ke);
          if (t.name == pfx + ".recent.v0") t = host_concat_axis2(t, ve);
        }
      }
      fs::path d2 = tmp / label / "prefix-cache" / "turn2" / "vx_snapshot";
      auto s2 = save_vxkv2_raw(d2.string(), arrs2, meta, o);
      std::printf("  -> Turn 2 delta (+256 tok):        raw=%6.2f MiB  new_disk=%5.2f MiB  dedup=%llu/%llu pages  save=%.1f ms\n",
                  s2.raw_bytes / 1048576.0, s2.disk_bytes / 1048576.0,
                  static_cast<unsigned long long>(s2.pages_deduped),
                  static_cast<unsigned long long>(s2.pages_total), s2.elapsed_ms);
    }
  };

  bench_vxkv2("VXKV2 CAS+BytePlane LZ4   (INT4)", arrays_int4, SnapshotCodec::LZ4);
  bench_vxkv2("VXKV2 CAS+BytePlane LZFSE (INT4)", arrays_int4, SnapshotCodec::LZFSE);
  bench_vxkv2("VXKV2 CAS+BytePlane LZFSE (K4V2)", arrays_k4v2, SnapshotCodec::LZFSE);
  bench_vxkv2("VXKV2 CAS+BytePlane LZFSE (K2V2)", arrays_k2v2, SnapshotCodec::LZFSE);

  fs::remove_all(tmp, ec);
}

}  // namespace

int main() {
  test_transforms_roundtrip();
  test_vxkv2_bit_exact_and_cas_dedup();
  test_phase2_low_bit_kv_policy();
  run_benchmark();
  if (g_failures > 0) {
    std::fprintf(stderr, "\nFAILED: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("\nALL TESTS PASSED (0 failures)\n");
  return 0;
}
