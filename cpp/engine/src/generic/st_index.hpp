// Safetensors index for the generic runtime: every tensor of every shard in a
// checkpoint directory, read from the headers only (no payload).
//
// Used for (a) loading resident ("dense") tensors by name into MLX buffers with
// pread, and (b) locating routed-expert rows (absolute file offset + per-expert
// row bytes) so the expert streamer can read single experts without loading the
// stacked tensor. Byte-exact: an array built here holds the file's bytes.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace veystrix::engine::generic {

namespace mx = mlx::core;

struct TensorInfo {
  std::string name;
  int shard = -1;            // index into StIndex::shards()
  int fd = -1;               // open read-only descriptor of the shard
  int nocache_fd = -1;       // F_NOCACHE descriptor for one-shot loads & 16 GB prefill
  std::string dtype;         // safetensors dtype string ("BF16", "U32", ...)
  std::vector<int64_t> shape;
  int64_t offset = 0;        // absolute file offset of the first byte
  int64_t nbytes = 0;
};

mx::Dtype dtype_from_st(const std::string& st);

class StIndex {
 public:
  explicit StIndex(const std::string& dir);
  ~StIndex();
  StIndex(const StIndex&) = delete;
  StIndex& operator=(const StIndex&) = delete;

  const TensorInfo* find(const std::string& name) const;
  const TensorInfo& at(const std::string& name) const;
  bool has(const std::string& name) const { return find(name) != nullptr; }
  const std::map<std::string, TensorInfo>& all() const { return tensors_; }
  const std::vector<std::string>& shards() const { return shard_paths_; }

  // Read a whole tensor into a fresh MLX buffer (pread, exact bytes).
  mx::array load(const std::string& name) const;

  // Return the F_NOCACHE descriptor for a shard (falling back to the regular fd)
  // for one-shot resident loads so macOS does not duplicate weights in page cache.
  int load_fd(int shard, int fallback_fd) const;

  // Release one-shot F_NOCACHE load descriptors once resident loading finishes.
  void close_load_fds();

  // Bytes loaded so far through load() (resident dense accounting).
  int64_t loaded_bytes() const { return loaded_bytes_; }

 private:
  std::map<std::string, TensorInfo> tensors_;
  std::vector<std::string> shard_paths_;
  std::vector<int> fds_;
  std::vector<int> nocache_fds_;
  mutable int64_t loaded_bytes_ = 0;
};

// pread the whole range [off, off+len) into dst (retrying short reads).
void pread_full(int fd, void* dst, int64_t len, int64_t off);

}  // namespace veystrix::engine::generic
