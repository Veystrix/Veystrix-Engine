#include "generic/expert_stream.hpp"

#include "generic/st_index.hpp"
#include "memory_budget.hpp"

#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace veystrix::engine::generic {

int read_lanes() {
  static const int n = [] {
    const char* e = std::getenv("VEYSTRIX_GENERIC_READ_LANES");
    int v = e ? std::atoi(e) : 8;
    return v < 1 ? 1 : (v > 64 ? 64 : v);
  }();
  return n;
}

int prefill_read_lanes() {
  static const int n = [] {
    const char* ep = std::getenv("VEYSTRIX_GENERIC_PREFILL_READ_LANES");
    const char* e = (ep && *ep) ? ep : std::getenv("VEYSTRIX_GENERIC_READ_LANES");
    int v = (e && *e) ? std::atoi(e) : 16;
    return v < 1 ? 1 : (v > 64 ? 64 : v);
  }();
  return n;
}

// VEYSTRIX_GENERIC_ROW_ORDER=tensor (or 1; recipe knob "tensor_rows"): issue one-token decode expert rows
// tensor-major, the custom Qwen runtime's order (Flash-Next qo2: see ENGINE_AB_2026-11-20.md).
bool tensor_major_rows() {
  static const bool on = [] {
    const char* e = std::getenv("VEYSTRIX_GENERIC_ROW_ORDER");
    return e && (std::strcmp(e, "tensor") == 0 || std::strcmp(e, "1") == 0);
  }();
  return on;
}

bool rows_enabled() {
  // Default on (Ornith 12.11 -> 13.01 tok/s, identical text); VEYSTRIX_GENERIC_ROWS=0 restores pread runs.
  static const bool on = [] {
    const char* e = std::getenv("VEYSTRIX_GENERIC_ROWS");
    return !(e && std::strcmp(e, "0") == 0);
  }();
  return on;
}

namespace {

ExpertTensorLoc locate(const StIndex& idx, const std::string& name) {
  const TensorInfo& t = idx.at(name);
  if (t.shape.size() < 2) throw std::runtime_error("generic: expert tensor is not stacked: " + name);
  ExpertTensorLoc l;
  l.name = name;
  l.fd = t.fd;
  l.nocache_fd = t.nocache_fd >= 0 ? t.nocache_fd : t.fd;
  l.base = t.offset;
  l.experts = t.shape[0];
  l.row_bytes = t.nbytes / t.shape[0];
  if (l.row_bytes * t.shape[0] != t.nbytes)
    throw std::runtime_error("generic: uneven expert rows in " + name);
  for (size_t i = 1; i < t.shape.size(); ++i) l.row_shape.push_back(static_cast<int>(t.shape[i]));
  l.dtype = dtype_from_st(t.dtype);
  return l;
}

struct ReadJob {
  int fd;
  int64_t off, len;
  uint8_t* dst;
};

}  // namespace

ExpertLayer::ExpertLayer(const StIndex& idx, const std::vector<std::string>& proj_paths,
                         const std::vector<std::pair<int, int>>& quant, bool resident,
                         const std::vector<std::string>& modes)
    : resident_(resident), initially_resident_(resident) {
  if (proj_paths.size() != quant.size()) throw std::runtime_error("generic: expert quant size");
  for (size_t p = 0; p < proj_paths.size(); ++p) {
    const std::string& path = proj_paths[p];
    ExpertProjLoc pl;
    pl.w = locate(idx, path + ".weight");
    pl.s = locate(idx, path + ".scales");
    pl.has_b = idx.has(path + ".biases");
    if (pl.has_b) pl.b = locate(idx, path + ".biases");
    pl.group_size = quant[p].first;
    pl.bits = quant[p].second;
    pl.mode = (p < modes.size() && !modes[p].empty()) ? modes[p] : "affine";
    pl.has_gs = idx.has(path + ".global_scale");
    if (pl.has_gs) {
      // NVFP4 checkpoints store per-expert global_amax in global_scale; MLX's nvfp4
      // dequantize/gather_qmm produces e2m1 * e4m3fn (max 6 * 448 = 2688.0), so the
      // effective weight multiplier is global_scale / 2688.0.
      mx::array gs = mx::astype(idx.load(path + ".global_scale"), mx::float32);
      if (pl.mode == "nvfp4") gs = mx::divide(gs, mx::array(2688.0f, mx::float32));
      pl.global_scale = gs;
      mx::eval(pl.global_scale);
    }
    if (experts_ == 0) experts_ = static_cast<int>(pl.w.experts);
    if (pl.w.experts != experts_ || pl.s.experts != experts_ || (pl.has_b && pl.b.experts != experts_))
      throw std::runtime_error("generic: expert count mismatch at " + path);
    // Packed in-dim must agree with scales: w[-1]*elem_bits/bits == s[-1]*gs.
    const int elem_bits = static_cast<int>(mx::size_of(pl.w.dtype)) * 8;
    const int64_t in_w = static_cast<int64_t>(pl.w.row_shape.back()) * elem_bits / pl.bits;
    const int64_t in_s = static_cast<int64_t>(pl.s.row_shape.back()) * pl.group_size;
    if (in_w != in_s)
      throw std::runtime_error("generic: quantization of " + path + " does not match the recipe (" +
                               std::to_string(in_w) + " vs " + std::to_string(in_s) + ")");
    projs_.push_back(pl);
  }
  if (resident_) {
    for (const auto& pl : projs_) {
      res_w_.push_back(idx.load(pl.w.name));
      res_s_.push_back(idx.load(pl.s.name));
      if (pl.has_b) res_b_.push_back(idx.load(pl.b.name));
    }
    std::vector<mx::array> all(res_w_);
    all.insert(all.end(), res_s_.begin(), res_s_.end());
    all.insert(all.end(), res_b_.begin(), res_b_.end());
    mx::eval(all);
    mx::clear_cache();
  }
}

int64_t ExpertLayer::bytes_per_expert() const {
  int64_t t = 0;
  for (const auto& p : projs_) t += p.w.row_bytes + p.s.row_bytes + (p.has_b ? p.b.row_bytes : 0);
  return t;
}

ExpertSet ExpertLayer::plan(const std::vector<int>& wanted, std::vector<rowfetch::Row>& rows, bool decode) const {
  if (resident_) throw std::runtime_error("generic: plan() on resident experts");
  if (wanted.empty()) throw std::runtime_error("generic: empty expert fetch");
  ExpertSet set;
  set.wanted = wanted;
  const int n = static_cast<int>(wanted.size());
  // Full-layer prefill reads (>=64 cold experts) bypass mmap and the OS page cache
  // via F_NOCACHE on every Mac tier so streaming multi-GiB cold weights never bloats
  // RAM or evicts hot decode/app pages into swap. Decode keeps mmap+memcpy speed.
  const bool nocache_prefill = !decode && n >= 64;
  const bool use_map = !nocache_prefill && rowfetch::mmap_enabled();
  std::vector<const ExpertTensorLoc*> locs;
  std::vector<uint8_t*> bases;
  auto alloc = [&](const ExpertTensorLoc& l) -> mx::array {
    const int64_t bytes = l.row_bytes * n;
    mx::allocator::Buffer buf = mx::allocator::malloc(static_cast<size_t>(bytes));
    locs.push_back(&l);
    bases.push_back(static_cast<uint8_t*>(buf.raw_ptr()));
    set.bytes += bytes;
    mx::Shape shape{n};
    shape.insert(shape.end(), l.row_shape.begin(), l.row_shape.end());
    return mx::array(buf, shape, l.dtype);
  };
  for (const auto& p : projs_) {
    set.w.push_back(alloc(p.w));
    set.s.push_back(alloc(p.s));
    if (p.has_b) set.b.push_back(alloc(p.b));
  }
  rows.reserve(rows.size() + locs.size() * static_cast<size_t>(n));
  for (int j = 0; j < n; ++j) {
    for (size_t t = 0; t < locs.size(); ++t) {
      const ExpertTensorLoc& l = *locs[t];
      rowfetch::Row r;
      r.fd = nocache_prefill ? l.nocache_fd : l.fd;
      r.map = use_map ? rowfetch::shard_map(l.fd) : nullptr;
      r.offset = l.base + static_cast<int64_t>(wanted[static_cast<size_t>(j)]) * l.row_bytes;
      r.length = l.row_bytes;
      r.dst = bases[t] + static_cast<int64_t>(j) * l.row_bytes;
      r.group = j;
      rows.push_back(r);
    }
  }
  if (decode && tensor_major_rows()) {
    const size_t T = locs.size(), first = rows.size() - T * static_cast<size_t>(n);
    std::vector<rowfetch::Row> tm;
    tm.reserve(T * static_cast<size_t>(n));
    for (size_t t = 0; t < T; ++t)
      for (int j = 0; j < n; ++j) tm.push_back(rows[first + static_cast<size_t>(j) * T + t]);
    std::copy(tm.begin(), tm.end(), rows.begin() + static_cast<std::ptrdiff_t>(first));
  }
  std::vector<uint32_t> remap(static_cast<size_t>(experts_), 0u);
  for (int i = 0; i < n; ++i) remap[static_cast<size_t>(wanted[i])] = static_cast<uint32_t>(i);
  set.remap = mx::array(remap.data(), mx::Shape{experts_}, mx::uint32);
  return set;
}

ExpertSet ExpertLayer::alloc_stacks(int capacity) const {
  if (resident_) throw std::runtime_error("generic: alloc_stacks() on resident experts");
  ExpertSet set;
  auto alloc = [&](const ExpertTensorLoc& l) -> mx::array {
    mx::allocator::Buffer buf = mx::allocator::malloc(static_cast<size_t>(l.row_bytes * capacity));
    set.bases.push_back(static_cast<uint8_t*>(buf.raw_ptr()));
    mx::Shape shape{capacity};
    shape.insert(shape.end(), l.row_shape.begin(), l.row_shape.end());
    return mx::array(buf, shape, l.dtype);
  };
  for (const auto& p : projs_) {
    set.w.push_back(alloc(p.w));
    set.s.push_back(alloc(p.s));
    if (p.has_b) set.b.push_back(alloc(p.b));
  }
  return set;
}

void ExpertLayer::add_rows(const ExpertSet& set, const std::vector<int>& experts, int first_row, int group_base,
                           std::vector<rowfetch::Row>& rows) const {
  const bool use_map = rowfetch::mmap_enabled();
  size_t t = 0;
  auto one = [&](const ExpertTensorLoc& l) {
    uint8_t* base = set.bases.at(t++);
    const uint8_t* map = use_map ? rowfetch::shard_map(l.fd) : nullptr;
    for (size_t i = 0; i < experts.size(); ++i) {
      rowfetch::Row r;
      r.fd = l.fd;
      r.map = map;
      r.offset = l.base + static_cast<int64_t>(experts[i]) * l.row_bytes;
      r.length = l.row_bytes;
      r.dst = base + static_cast<int64_t>(first_row + static_cast<int>(i)) * l.row_bytes;
      r.group = group_base + static_cast<int>(i);
      rows.push_back(r);
    }
  };
  for (const auto& p : projs_) {
    one(p.w);
    one(p.s);
    if (p.has_b) one(p.b);
  }
}

std::vector<rowfetch::Row> ExpertLayer::prefetch_rows(const std::vector<int>& experts) const {
  std::vector<rowfetch::Row> rows;
  if (resident_) return rows;
  for (const auto& p : projs_) {
    for (const ExpertTensorLoc* l : {&p.w, &p.s, p.has_b ? &p.b : nullptr}) {
      if (!l) continue;
      const uint8_t* map = rowfetch::shard_map(l->fd);
      for (int e : experts) {
        rowfetch::Row r;
        r.fd = l->fd;
        r.map = map;
        r.offset = l->base + static_cast<int64_t>(e) * l->row_bytes;
        r.length = l->row_bytes;
        rows.push_back(r);
      }
    }
  }
  return rows;
}

ExpertSet ExpertLayer::fetch(const std::vector<int>& wanted, bool decode) const {
  ExpertSet set;
  if (resident_) {
    set.w = res_w_;
    set.s = res_s_;
    set.b = res_b_;
    return set;
  }
  if (wanted.empty()) throw std::runtime_error("generic: empty expert fetch");
  if (rows_enabled()) {
    std::vector<rowfetch::Row> rows;
    set = plan(wanted, rows, decode);
    rowfetch::Fetch f(std::move(rows), static_cast<int>(wanted.size()), read_lanes());
    while (f.next_ready() >= 0) {
    }
    if (!f.ok()) {
      const std::string layer_name = projs_.empty() ? "unknown" : projs_[0].w.name;
      throw std::runtime_error("generic: expert read failed (" + layer_name + "): " + f.error_detail());
    }
    const rowfetch::Stats st = f.stats();
    set.rows_memcpy = st.memcpy_rows;
    set.rows_pread = st.pread_rows;
    return set;
  }
  set.wanted = wanted;
  const int n = static_cast<int>(wanted.size());
  // Runs of consecutive expert ids -> one read per tensor per run.
  std::vector<std::pair<int, int>> runs;  // (first expert, count) ; compact row = sum of previous counts
  for (int i = 0; i < n;) {
    int j = i + 1;
    while (j < n && wanted[j] == wanted[j - 1] + 1) ++j;
    runs.emplace_back(i, j - i);
    i = j;
  }
  std::vector<ReadJob> jobs;
  auto plan = [&](const ExpertTensorLoc& l) -> mx::array {
    const int64_t bytes = l.row_bytes * n;
    mx::allocator::Buffer buf = mx::allocator::malloc(static_cast<size_t>(bytes));
    auto* dst = static_cast<uint8_t*>(buf.raw_ptr());
    for (const auto& r : runs) {
      const int64_t first = wanted[r.first];
      // Split large runs so lanes share the work.
      const int64_t total = l.row_bytes * r.second;
      const int64_t chunk = std::max<int64_t>(l.row_bytes, 4 << 20);
      for (int64_t at = 0; at < total; at += chunk)
        jobs.push_back({l.fd, l.base + first * l.row_bytes + at, std::min(chunk, total - at),
                        dst + static_cast<int64_t>(r.first) * l.row_bytes + at});
    }
    set.bytes += bytes;
    mx::Shape shape{n};
    shape.insert(shape.end(), l.row_shape.begin(), l.row_shape.end());
    return mx::array(buf, shape, l.dtype);
  };
  for (const auto& p : projs_) {
    set.w.push_back(plan(p.w));
    set.s.push_back(plan(p.s));
    if (p.has_b) set.b.push_back(plan(p.b));
  }
  // Largest first, so the long reads start early.
  std::sort(jobs.begin(), jobs.end(), [](const ReadJob& a, const ReadJob& b) { return a.len > b.len; });
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  std::mutex err_mu;
  std::string err_msg;
  const size_t lanes = std::min<size_t>(static_cast<size_t>(read_lanes()), jobs.size());
  dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
  std::atomic<size_t>* nextp = &next;
  std::atomic<bool>* failedp = &failed;
  std::mutex* err_mup = &err_mu;
  std::string* err_msgp = &err_msg;
  const ReadJob* jp = jobs.data();
  const size_t nj = jobs.size();
  dispatch_apply(lanes, q, ^(size_t) {
    for (;;) {
      size_t i = nextp->fetch_add(1);
      if (i >= nj) break;
      try {
        pread_full(jp[i].fd, jp[i].dst, jp[i].len, jp[i].off);
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(*err_mup);
        if (err_msgp->empty()) {
          *err_msgp = std::string(e.what()) + " (fd=" + std::to_string(jp[i].fd) +
                      " off=" + std::to_string(jp[i].off) + " len=" + std::to_string(jp[i].len) + ")";
        }
        failedp->store(true);
      } catch (...) {
        failedp->store(true);
      }
    }
  });
  if (failed.load()) {
    const std::string layer_name = projs_.empty() ? "unknown" : projs_[0].w.name;
    throw std::runtime_error("generic: expert read failed (" + layer_name + "): " +
                             (err_msg.empty() ? "unknown I/O error" : err_msg));
  }
  // remap: expert id -> compact row.
  std::vector<uint32_t> remap(static_cast<size_t>(experts_), 0u);
  for (int i = 0; i < n; ++i) remap[static_cast<size_t>(wanted[i])] = static_cast<uint32_t>(i);
  set.remap = mx::array(remap.data(), mx::Shape{experts_}, mx::uint32);
  return set;
}

void ExpertLayer::advise_all() const {
  if (resident_) return;
  for (const auto& p : projs_) {
    for (const ExpertTensorLoc* l : {&p.w, &p.s, p.has_b ? &p.b : nullptr}) {
      if (!l) continue;
      struct radvisory ra;
      ra.ra_offset = static_cast<off_t>(l->base);
      ra.ra_count = static_cast<int>(std::min<int64_t>(l->row_bytes * l->experts, INT32_MAX));
      (void)::fcntl(l->fd, F_RDADVISE, &ra);
    }
  }
}

void ExpertLayer::load_resident() {
  if (resident_) return;
  std::vector<int> all(static_cast<size_t>(experts_));
  for (int i = 0; i < experts_; ++i) all[static_cast<size_t>(i)] = i;
  ExpertSet set = fetch(all, false);
  res_w_ = std::move(set.w);
  res_s_ = std::move(set.s);
  res_b_ = std::move(set.b);
  resident_ = true;
  std::vector<mx::array> to_eval(res_w_);
  to_eval.insert(to_eval.end(), res_s_.begin(), res_s_.end());
  to_eval.insert(to_eval.end(), res_b_.begin(), res_b_.end());
  mx::eval(to_eval);
  mx::clear_cache();
}

void ExpertLayer::release_resident() {
  if (initially_resident_) return;
  res_w_.clear();
  res_s_.clear();
  res_b_.clear();
  resident_ = false;
  mx::clear_cache();
}

}  // namespace veystrix::engine::generic
