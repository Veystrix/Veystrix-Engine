#include "generic/st_index.hpp"

#include "fd_hints.hpp"
#include "generic/json_lite.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace veystrix::engine::generic {

mx::Dtype dtype_from_st(const std::string& st) {
  if (st == "BF16") return mx::bfloat16;
  if (st == "F16") return mx::float16;
  if (st == "F32") return mx::float32;
  if (st == "U32") return mx::uint32;
  if (st == "I32") return mx::int32;
  if (st == "U8") return mx::uint8;
  if (st == "I8") return mx::int8;
  if (st == "U16") return mx::uint16;
  if (st == "I16") return mx::int16;
  if (st == "I64") return mx::int64;
  if (st == "U64") return mx::uint64;
  if (st == "BOOL") return mx::bool_;
  throw std::runtime_error("safetensors: unsupported dtype " + st);
}

void pread_full(int fd, void* dst, int64_t len, int64_t off) {
  auto* p = static_cast<uint8_t*>(dst);
  while (len > 0) {
    ssize_t got = ::pread(fd, p, static_cast<size_t>(len), static_cast<off_t>(off));
    if (got < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("pread failed: ") + std::strerror(errno));
    }
    if (got == 0) throw std::runtime_error("pread: unexpected end of file");
    p += got;
    len -= got;
    off += got;
  }
}

StIndex::StIndex(const std::string& dir) {
  namespace fs = std::filesystem;
  std::vector<std::string> files;
  for (const auto& e : fs::directory_iterator(dir)) {
    if (!e.is_regular_file() && !e.is_symlink()) continue;
    const std::string n = e.path().filename().string();
    if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0 && n[0] != '.')
      files.push_back(e.path().string());
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) throw std::runtime_error("generic: no *.safetensors in " + dir);
  for (const std::string& path : files) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("generic: cannot open " + path);
    apply_shard_read_hints(fd);
    int nfd = ::open(path.c_str(), O_RDONLY);
    if (nfd >= 0) {
#ifdef F_NOCACHE
      (void)::fcntl(nfd, F_NOCACHE, 1);
#endif
#ifdef F_RDAHEAD
      (void)::fcntl(nfd, F_RDAHEAD, 0);
#endif
    }
    const int shard = static_cast<int>(shard_paths_.size());
    shard_paths_.push_back(path);
    fds_.push_back(fd);
    nocache_fds_.push_back(nfd);
    const int hfd = nfd >= 0 ? nfd : fd;
    uint64_t hlen = 0;
    pread_full(hfd, &hlen, 8, 0);
    if (hlen == 0 || hlen > (512ull << 20)) throw std::runtime_error("generic: bad header in " + path);
    std::string header(static_cast<size_t>(hlen), '\0');
    pread_full(hfd, header.data(), static_cast<int64_t>(hlen), 8);
    Json h = Json::parse(header);
    const int64_t base = 8 + static_cast<int64_t>(hlen);
    for (const auto& kv : h.obj("safetensors header")) {
      if (kv.first == "__metadata__") continue;
      TensorInfo ti;
      ti.name = kv.first;
      ti.shard = shard;
      ti.fd = fd;
      ti.nocache_fd = nfd >= 0 ? nfd : fd;
      ti.dtype = kv.second.at("dtype").str("dtype");
      for (const Json& d : kv.second.at("shape").arr("shape")) ti.shape.push_back(d.i64("dim"));
      const auto& offs = kv.second.at("data_offsets").arr("data_offsets");
      ti.offset = base + offs.at(0).i64("offset");
      ti.nbytes = offs.at(1).i64("offset") - offs.at(0).i64("offset");
      tensors_[ti.name] = std::move(ti);
    }
  }
}

void StIndex::close_load_fds() {
  for (int& fd : nocache_fds_) {
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
  }
}

int StIndex::load_fd(int shard, int fallback_fd) const {
  if (shard >= 0 && static_cast<size_t>(shard) < nocache_fds_.size() &&
      nocache_fds_[static_cast<size_t>(shard)] >= 0) {
    return nocache_fds_[static_cast<size_t>(shard)];
  }
  return fallback_fd;
}

StIndex::~StIndex() {
  close_load_fds();
  for (int fd : fds_) ::close(fd);
}

const TensorInfo* StIndex::find(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const TensorInfo& StIndex::at(const std::string& name) const {
  const TensorInfo* t = find(name);
  if (!t) throw std::runtime_error("generic: tensor not in checkpoint: " + name);
  return *t;
}

mx::array StIndex::load(const std::string& name) const {
  const TensorInfo& t = at(name);
  mx::Shape shape;
  int64_t count = 1;
  for (int64_t d : t.shape) {
    shape.push_back(static_cast<int>(d));
    count *= d;
  }
  const mx::Dtype dt = dtype_from_st(t.dtype);
  if (count * static_cast<int64_t>(mx::size_of(dt)) != t.nbytes)
    throw std::runtime_error("generic: size mismatch for " + name);
  mx::allocator::Buffer b = mx::allocator::malloc(static_cast<size_t>(t.nbytes ? t.nbytes : 1));
  try {
    pread_full(load_fd(t.shard, t.fd), b.raw_ptr(), t.nbytes, t.offset);
  } catch (...) {
    mx::allocator::free(b);
    throw;
  }
  loaded_bytes_ += t.nbytes;
  return mx::array(b, std::move(shape), dt);
}

}  // namespace veystrix::engine::generic
