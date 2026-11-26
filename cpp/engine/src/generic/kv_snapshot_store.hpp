// Page-level Content-Addressed Storage (CAS) + byte-plane/bit-plane shuffled
// lossless snapshot persistence (VXKV2) for Veystrix generic KV caches.
//
// Phase 1 (zero quality loss / bit-exact):
//   - Splits 4D KV tensors [B, H, T, D] along axis 2 into immutable 256-token
//     pages stored once in a shared content-addressed page pool (.cas_pages/)
//     keyed by SHA-256 of canonical page bytes.
//   - Applies SIMD-friendly lossless byte-plane shuffling on BF16/FP16/FP32
//     tensors (separating low-entropy exponent bytes from mantissa bytes) and
//     4-bit-plane shuffling on packed INT4/INT2 uint32 tensors before Apple
//     libcompression (LZFSE / LZ4), automatically falling back to raw if a
//     page is incompressible.
//   - Restores pages in parallel (up to 4 I/O+decode threads) directly into
//     contiguous host buffers and constructs MLX arrays in memory with zero
//     temporary files on SSD.
//   - Garbage-collects unreferenced CAS pages when snapshots are pruned.
#pragma once

#include <mlx/mlx.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace veystrix::engine::generic {

namespace mx = mlx::core;

enum class SnapshotCodec : uint8_t {
  None = 0,
  LZ4 = 1,
  LZFSE = 2,
};

enum class SnapshotTransform : uint8_t {
  None = 0,
  BytePlane2 = 1,  // 16-bit elements (BF16 / FP16 / INT16 / UINT16): [hi_bytes | lo_bytes]
  BytePlane4 = 2,  // 32-bit elements (FP32 / INT32 / UINT32): [b3 | b2 | b1 | b0]
  BitPlane4 = 3,   // Packed 4-bit nibbles (INT4): 4 bit-planes across uint32 lanes
};

enum class WireDtype : uint8_t {
  Bool = 0,
  UInt8 = 1,
  UInt16 = 2,
  UInt32 = 3,
  UInt64 = 4,
  Int8 = 5,
  Int16 = 6,
  Int32 = 7,
  Int64 = 8,
  Float16 = 9,
  Float32 = 10,
  BFloat16 = 11,
};

struct HostTensor {
  std::string name;
  WireDtype dtype = WireDtype::Float32;
  std::vector<int64_t> shape;
  std::vector<uint8_t> data;
};

struct SnapshotSaveOptions {
  SnapshotCodec codec = SnapshotCodec::LZFSE;
  int page_tokens = 256;
  int io_threads = 4;
  std::string cas_dir;  // optional explicit CAS root; auto-resolved if empty
};

struct SnapshotSaveStats {
  uint64_t raw_bytes = 0;           // total uncompressed tensor payload bytes
  uint64_t disk_bytes = 0;          // manifest + newly written CAS page bytes
  uint64_t logical_comp_bytes = 0;  // manifest + sum of all referenced CAS page sizes
  uint64_t pages_total = 0;
  uint64_t pages_written = 0;
  uint64_t pages_deduped = 0;
  double elapsed_ms = 0.0;
};

struct SnapshotRestoreStats {
  uint64_t raw_bytes = 0;
  uint64_t disk_bytes_read = 0;
  uint64_t pages_read = 0;
  double elapsed_ms = 0.0;
};

// Bit-exact lossless byte-plane and bit-plane transforms (exposed for unit tests).
void byte_plane_2_encode(const uint8_t* src, uint8_t* dst, size_t nbytes);
void byte_plane_2_decode(const uint8_t* src, uint8_t* dst, size_t nbytes);
void byte_plane_4_encode(const uint8_t* src, uint8_t* dst, size_t nbytes);
void byte_plane_4_decode(const uint8_t* src, uint8_t* dst, size_t nbytes);
void bit_plane_4_encode(const uint8_t* src, uint8_t* dst, size_t nbytes);
void bit_plane_4_decode(const uint8_t* src, uint8_t* dst, size_t nbytes);

// Resolve the shared CAS page directory for a snapshot directory.
// For `.../prefix-cache/<digest>/vx_snapshot`, returns `.../prefix-cache/.cas_pages`.
// Otherwise returns `<snapshot_dir>/.cas_pages`.
std::string resolve_cas_dir(const std::string& snapshot_dir, const std::string& override_dir = "");

// Parse codec from env var `VEYSTRIX_SNAPSHOT_CODEC` ("lzfse" | "lz4" | "none").
SnapshotCodec default_snapshot_codec();

// True if `snapshot_dir` contains a VXKV2 manifest (`snapshot.vxkv2`).
bool is_vxkv2_snapshot(const std::string& snapshot_dir);

// Device-independent raw host-buffer save/restore (used by both MLX wrapper and CPU tests).
SnapshotSaveStats save_vxkv2_raw(
    const std::string& snapshot_dir,
    const std::vector<HostTensor>& tensors,
    const std::unordered_map<std::string, std::string>& meta,
    const SnapshotSaveOptions& opts = {});

std::pair<std::vector<HostTensor>, std::unordered_map<std::string, std::string>>
load_vxkv2_raw(const std::string& snapshot_dir,
               const std::string& cas_dir_override = "",
               int io_threads = 4,
               SnapshotRestoreStats* stats = nullptr);

// Save named MLX arrays + string metadata into `snapshot_dir/snapshot.vxkv2` + shared CAS pool.
SnapshotSaveStats save_vxkv2_snapshot(
    const std::string& snapshot_dir,
    const std::unordered_map<std::string, mx::array>& arrays,
    const std::unordered_map<std::string, std::string>& meta,
    const SnapshotSaveOptions& opts = {});

// Restore named MLX arrays + string metadata directly into RAM from `snapshot_dir/snapshot.vxkv2`.
std::pair<std::unordered_map<std::string, mx::array>, std::unordered_map<std::string, std::string>>
load_vxkv2_snapshot(const std::string& snapshot_dir,
                    const std::string& cas_dir_override = "",
                    int io_threads = 4,
                    SnapshotRestoreStats* stats = nullptr);

// Garbage-collect unreferenced `.vxpg` files under `cas_dir` given all live snapshot dirs
// under `prefix_cache_root`. Returns number of deleted page files and reclaimed bytes.
std::pair<uint64_t, uint64_t> gc_cas_pages(const std::string& prefix_cache_root,
                                           const std::string& cas_dir = "");

}  // namespace veystrix::engine::generic
