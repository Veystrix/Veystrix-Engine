#include "moe_stream.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

#include "io_pager.hpp"
#include "slot_plan.hpp"

namespace mx = mlx::core;
namespace veystrix::engine {

namespace {

// ---- minimal safetensors header reader -------------------------------------
// We only need a handful of tensors, so a targeted string scan is enough:
// the safetensors header is a flat JSON object {name: {dtype, shape,
// data_offsets, ...}, ...}. Tensor names contain no JSON escapes.

bool find_value(const std::string& s, const std::string& key, size_t from,
                size_t& value_begin, size_t& value_end) {
  std::string pat = "\"" + key + "\"";
  size_t p = s.find(pat, from);
  if (p == std::string::npos) return false;
  size_t colon = s.find(':', p + pat.size());
  if (colon == std::string::npos) return false;
  size_t b = colon + 1;
  while (b < s.size() && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n' || s[b] == '\r')) ++b;
  value_begin = b;
  return true;
}

bool parse_int_array(const std::string& obj, const std::string& key,
                     std::vector<int64_t>& out) {
  size_t b, e;
  if (!find_value(obj, key, 0, b, e)) return false;
  if (b >= obj.size() || obj[b] != '[') return false;
  size_t p = b + 1;
  out.clear();
  while (p < obj.size() && obj[p] != ']') {
    if (obj[p] == '-' || (obj[p] >= '0' && obj[p] <= '9')) {
      size_t q = p;
      if (obj[q] == '-') ++q;
      while (q < obj.size() && obj[q] >= '0' && obj[q] <= '9') ++q;
      out.push_back(std::strtoll(obj.substr(p, q - p).c_str(), nullptr, 10));
      p = q;
    } else {
      ++p;
    }
  }
  return !out.empty();
}

struct TensorMeta {
  std::vector<int> shape;
  int64_t offset = -1;  // absolute byte offset in the shard file
  int64_t row_bytes = 0;
};

bool parse_tensor(const std::string& header, const std::string& name, TensorMeta& out) {
  std::string pat = "\"" + name + "\"";
  size_t p = header.find(pat);
  if (p == std::string::npos) return false;
  size_t colon = header.find(':', p + pat.size());
  size_t obj = header.find('{', colon);
  if (obj == std::string::npos) return false;
  size_t close = header.find('}', obj);
  if (close == std::string::npos) return false;
  std::string body = header.substr(obj, close - obj + 1);

  std::vector<int64_t> shape, offs;
  if (!parse_int_array(body, "shape", shape)) return false;
  if (!parse_int_array(body, "data_offsets", offs) || offs.empty()) return false;
  out.shape.assign(shape.begin(), shape.end());
  out.offset = offs[0];
  return true;
}

int64_t read_header(const std::string& path, std::string& header) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open shard: " + path);
  unsigned char lenbuf[8];
  ssize_t got = ::pread(fd, lenbuf, 8, 0);
  if (got != 8) {
    ::close(fd);
    throw std::runtime_error("short safetensors header length: " + path);
  }
  uint64_t n = 0;
  for (int i = 0; i < 8; ++i) n |= static_cast<uint64_t>(lenbuf[i]) << (8 * i);
  header.resize(n);
  got = ::pread(fd, header.data(), n, 8);
  ::close(fd);
  if (got != static_cast<ssize_t>(n))
    throw std::runtime_error("short safetensors header: " + path);
  return static_cast<int64_t>(8 + n);  // data_start
}

// Find the shard containing `key` by scanning shard headers. For layer 0 this
// is the first shard, so the scan is cheap.
std::string find_shard(const std::string& dir, const std::string& key) {
  DIR* d = ::opendir(dir.c_str());
  if (!d) throw std::runtime_error("cannot open checkpoint dir: " + dir);
  std::vector<std::string> names;
  while (struct dirent* e = ::readdir(d)) {
    std::string n(e->d_name);
    const std::string ext = ".safetensors";
    if (n.size() > ext.size() &&
        n.substr(n.size() - ext.size()) == ext) names.push_back(n);
  }
  ::closedir(d);
  std::sort(names.begin(), names.end());
  std::string header;
  for (const auto& n : names) {
    read_header(dir + "/" + n, header);
    if (header.find("\"" + key + "\"") != std::string::npos) return n;
  }
  throw std::runtime_error("tensor not found in any shard: " + key);
}

}  // namespace

// ---------------------------------------------------------------------------
struct StreamingMoE::Impl {
  std::string dir;
  int layer = 0;
  int num_slots = 0;
  int group_size = 32;
  int bits = 4;
  int read_width = 8;
  std::string prefix;

  int shard_fd = -1;
  int64_t data_start = 0;

  // Tensor order: gw, gs, gb, uw, us, ub, dw, ds, db.
  static constexpr int kTensors = 9;
  std::vector<TensorMeta> t;
  std::vector<int64_t> tensor_off;   // absolute file offset
  std::vector<bool> is_weight;

  // One persistent buffer per tensor, sized num_slots * row_bytes. Slot s's
  // row lives at buffer + s * row_bytes (matches pager slot_stride).
  std::vector<std::shared_ptr<uint8_t>> buf;

  // Slot bookkeeping. slot_expert[s] == -1 means free.
  std::vector<int32_t> slot_expert;
  std::vector<int> slot_refcount;
  std::vector<uint64_t> slot_lastused;
  std::vector<int32_t> free_slots;
  uint64_t tick = 0;

  Impl(const std::string& checkpoint_dir, int layer_, int slots, int gs,
       int b, int width, const std::string& prefix_)
      : dir(checkpoint_dir), layer(layer_), num_slots(slots), group_size(gs),
        bits(b), read_width(width) {
    if (num_slots < 1) throw std::runtime_error("num_slots must be >= 1");
    prefix = prefix_.empty()
                 ? "language_model.model.layers." + std::to_string(layer) +
                       ".mlp.switch_mlp"
                 : prefix_;

    const char* names[kTensors] = {
        "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
        "up_proj.weight",   "up_proj.scales",   "up_proj.biases",
        "down_proj.weight", "down_proj.scales", "down_proj.biases",
    };
    const bool weight[kTensors] = {true, false, false, true, false, false,
                                   true, false, false};
    t.resize(kTensors);
    tensor_off.resize(kTensors);
    is_weight.assign(weight, weight + kTensors);
    std::string header;
    std::string first_shard;
    data_start = 0;
    for (int i = 0; i < kTensors; ++i) {
      std::string key = prefix + "." + names[i];
      std::string shard = find_shard(dir, key);
      if (data_start == 0) {
        first_shard = shard;
        std::string h;
        data_start = read_header(dir + "/" + shard, h);
        header = std::move(h);
      } else if (shard != first_shard) {
        // The pager reads from one fd with one data_start; a layer split across
        // shards needs per-tensor fds and is not supported by this port.
        throw std::runtime_error("MoE layer spans multiple shards: " + key);
      }
      if (!parse_tensor(header, key, t[i]))
        throw std::runtime_error("cannot parse tensor: " + key);
      int64_t itemsize = weight[i] ? 4 : 2;
      int64_t row = 1;
      for (size_t d = 1; d < t[i].shape.size(); ++d) row *= t[i].shape[d];
      t[i].row_bytes = row * itemsize;
      tensor_off[i] = data_start + t[i].offset;
    }

    // All nine layer-0 tensors are in the same shard for this checkpoint.
    std::string gate_key = prefix + "." + names[0];
    std::string shard = find_shard(dir, gate_key);
    shard_fd = ::open((dir + "/" + shard).c_str(), O_RDONLY);
    if (shard_fd < 0) throw std::runtime_error("cannot open shard for reading: " + shard);

    buf.resize(kTensors);
    for (int i = 0; i < kTensors; ++i) {
      size_t bytes = static_cast<size_t>(num_slots) * static_cast<size_t>(t[i].row_bytes);
      void* raw = nullptr;
      if (::posix_memalign(&raw, 64, bytes) != 0)
        throw std::runtime_error("posix_memalign failed for slot buffer");
      std::memset(raw, 0, bytes);
      buf[i] = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(raw),
                                        [](uint8_t* p) { std::free(p); });
    }

    slot_expert.assign(num_slots, -1);
    slot_refcount.assign(num_slots, 0);
    slot_lastused.assign(num_slots, 0);
    for (int s = 0; s < num_slots; ++s) free_slots.push_back(s);
  }

  ~Impl() {
    if (shard_fd >= 0) ::close(shard_fd);
  }

  mx::array wrap(int i) {
    mx::Shape shape(t[i].shape.begin(), t[i].shape.end());
    shape[0] = num_slots;  // slot pool is axis 0, not the checkpoint's E
    auto keep = buf[i];    // deleter captures a ref so the buffer outlives the array
    mx::Dtype dtype = is_weight[i] ? mx::uint32 : mx::bfloat16;
    return mx::array(buf[i].get(), std::move(shape), dtype,
                     [keep](void*) {});
  }
};

StreamingMoE::StreamingMoE(const std::string& checkpoint_dir, int layer,
                           int num_slots, int group_size, int bits,
                           int read_width, const std::string& prefix)
    : impl_(std::make_unique<Impl>(checkpoint_dir, layer, num_slots, group_size,
                                   bits, read_width, prefix)) {}

StreamingMoE::~StreamingMoE() = default;

StreamForwardResult StreamingMoE::forward(const mx::array& x,
                                          const mx::array& route,
                                          const mx::array& scores) {
  Impl& s = *impl_;

  // Flatten route (expert ids) to int32 in router-rank order.
  mx::array r = mx::astype(route, mx::int32);
  mx::eval(r);
  const int32_t* rp = r.data<int32_t>();
  std::vector<int32_t> route_ids(rp, rp + r.size());
  if (route_ids.empty()) throw std::runtime_error("empty route");
  size_t T = route.shape().size() >= 2 ? static_cast<size_t>(route.shape(0)) : 1;
  size_t K = route_ids.size() / T;
  if (T * K != route_ids.size()) throw std::runtime_error("bad route shape");

  // Single source of truth for slot bookkeeping. plan_slots performs the LRU
  // eviction (mutating slot_expert/free_slots) and returns the exact slot plan.
  // The old code counted misses by searching a flattened [expert, slot, ...]
  // list and derived (expert, slot) from read offsets across all 9 tensors;
  // both are gone. See cpp/engine/src/slot_plan.hpp.
  SlotStep step = plan_slots(route_ids, s.slot_expert, s.slot_refcount,
                             s.slot_lastused, s.free_slots, s.num_slots);

  // Pairs for plan_remap come from the POST-eviction resident state. Slot_expert
  // is now authoritative: evicted experts are -1, so they cannot be "resident".
  std::vector<int32_t> pairs;
  for (int slot = 0; slot < s.num_slots; ++slot)
    if (s.slot_expert[slot] >= 0) {
      pairs.push_back(s.slot_expert[slot]);
      pairs.push_back(slot);
    }
  std::vector<int32_t> free_snapshot = s.free_slots;

  // One plan per tensor geometry; all must agree with the shared planner.
  std::vector<veystrix::PagerRead> all_reads;
  std::vector<int32_t> slots;
  for (int i = 0; i < Impl::kTensors; ++i) {
    veystrix::PagerPlan plan = veystrix::plan_remap(
        route_ids, pairs, free_snapshot, s.shard_fd,
        s.tensor_off[i], s.t[i].row_bytes,
        s.buf[i].get(), s.t[i].row_bytes);
    if (i == 0) slots = plan.slots;
    else if (plan.slots != slots)
      throw std::runtime_error("inconsistent slot plan across tensors");
    if (plan.slots != step.slots)
      throw std::runtime_error("plan_remap slot plan disagrees with plan_slots");
    if (plan.reads.size() != step.miss_e.size())
      throw std::runtime_error("plan_remap read count disagrees with plan_slots");
    for (auto& read : plan.reads) all_reads.push_back(read);
  }
  // Fail-closed range check: no -1, no out-of-range slot may reach gather_qmm.
  for (int32_t slot : slots)
    if (slot < 0 || slot >= s.num_slots)
      throw std::runtime_error("fail-closed: slot out of range");

  // Commit miss assignments from the planner (NOT from read offsets).
  std::vector<int32_t> used = slots;
  std::sort(used.begin(), used.end());
  used.erase(std::unique(used.begin(), used.end()), used.end());
  for (int32_t slot : used) ++s.slot_refcount[slot];
  for (size_t i = 0; i < step.miss_e.size(); ++i) {
    s.slot_expert[step.miss_s[i]] = step.miss_e[i];
    auto it = std::find(s.free_slots.begin(), s.free_slots.end(), step.miss_s[i]);
    if (it != s.free_slots.end()) s.free_slots.erase(it);
  }

  // Batched positioned reads of all missing rows straight from the shard.
  if (!all_reads.empty()) {
    int errors = veystrix::run_batched_reads(all_reads, s.read_width);
    if (errors != 0)
      throw std::runtime_error("pager batched read failed: " + std::to_string(errors));
  }

  // Debug verifier (single early-return branch when disabled).
  if (verify_slots_enabled()) {
    std::vector<int32_t> fds(Impl::kTensors, s.shard_fd);
    std::vector<int64_t> row_bytes(Impl::kTensors);
    std::vector<uint8_t*> bufs(Impl::kTensors);
    for (int i = 0; i < Impl::kTensors; ++i) {
      row_bytes[i] = s.t[i].row_bytes;
      bufs[i] = s.buf[i].get();
    }
    verify_slot_fill(true, route_ids, slots, s.slot_expert,
                     step.miss_e, step.miss_s, fds, s.tensor_off,
                     row_bytes, bufs);
  }

  // Wrap the slot buffers as mx arrays (bf16 scales/biases are a byte reinterpret).
  // mx::array has no default constructor, so build the bundle incrementally.
  std::vector<mx::array> a;
  a.reserve(Impl::kTensors);
  for (int i = 0; i < Impl::kTensors; ++i) a.push_back(s.wrap(i));
  ExpertWeights w(Q4Projection(a[0], a[1], a[2]),
                  Q4Projection(a[3], a[4], a[5]),
                  Q4Projection(a[6], a[7], a[8]),
                  s.group_size, s.bits);

  // Route slots, in router-rank order, are the gather indices. Scores stay in
  // the same order so the weighted combine reduces in router-rank order.
  mx::array route_slots(slots.begin(),
                        mx::Shape{static_cast<int>(T), static_cast<int>(K)},
                        mx::int32);
  // Python SwitchGLU.__call__ sorts whenever indices.size >= 64 and calls
  // gather_qmm(sorted_indices=True). At prefill (T*K >= 64) this kernel differs
  // from the unsorted one by ~1 ULP, so use the matching sorted path.
  mx::array out = (T * K >= 64)
                      ? moe_block_sorted(x, route_slots, scores, w)
                      : moe_block(x, route_slots, scores, w);
  mx::eval(out);  // command buffers complete -> safe to unpin slots

  for (int32_t slot : used) --s.slot_refcount[slot];
  for (int32_t slot : used) s.slot_lastused[slot] = ++s.tick;

  return StreamForwardResult{out, slots, step.miss_e};
}

mx::array StreamingMoE::forward_stacked(const mx::array& x,
                                        const mx::array& route,
                                        const mx::array& scores) {
  Impl& s = *impl_;
  mx::array r = mx::astype(route, mx::int32);
  mx::eval(r);
  const int32_t* rp = r.data<int32_t>();
  std::vector<int32_t> flat(rp, rp + r.size());
  if (flat.empty()) throw std::runtime_error("empty route");
  size_t T = route.shape().size() >= 2 ? static_cast<size_t>(route.shape(0)) : 1;
  size_t K = flat.size() / T;
  int E = static_cast<int>(s.t[0].shape[0]);  // total experts in checkpoint

  // First-occurrence distinct experts + remap to stacked-row indices.
  std::vector<int32_t> wanted;
  std::vector<int32_t> local(flat.size());
  std::vector<int32_t> seen(E, -1);
  for (size_t i = 0; i < flat.size(); ++i) {
    int32_t e = flat[i];
    if (e < 0) e = 0;
    if (e >= E) e = E - 1;
    if (seen[e] < 0) { seen[e] = (int32_t)wanted.size(); wanted.push_back(e); }
    local[i] = seen[e];
  }
  int n = static_cast<int>(wanted.size());

  // Read ONLY the wanted expert rows for all 9 tensors into compact buffers.
  std::vector<std::shared_ptr<uint8_t>> bufs(Impl::kTensors);
  std::vector<veystrix::PagerRead> reads;
  for (int i = 0; i < Impl::kTensors; ++i) {
    size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(s.t[i].row_bytes);
    void* raw = nullptr;
    if (::posix_memalign(&raw, 64, bytes ? bytes : 1) != 0)
      throw std::runtime_error("stacked: posix_memalign failed");
    bufs[i] = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(raw),
                                       [](uint8_t* p) { std::free(p); });
    for (int j = 0; j < n; ++j) {
      reads.push_back({s.shard_fd, 0,
                       s.tensor_off[i] + static_cast<int64_t>(wanted[j]) * s.t[i].row_bytes,
                       static_cast<int64_t>(s.t[i].row_bytes),
                       reinterpret_cast<uint64_t>(bufs[i].get() + (size_t)j * s.t[i].row_bytes)});
    }
  }
  if (!reads.empty()) {
    int err = veystrix::run_batched_reads(reads, s.read_width);
    if (err != 0) throw std::runtime_error("stacked: pager read failed");
  }

  auto wrap = [&](int i) {
    mx::Shape shape(s.t[i].shape.begin(), s.t[i].shape.end());
    shape[0] = n;
    auto keep = bufs[i];
    mx::Dtype dt = (i % 3 == 0) ? mx::uint32 : mx::bfloat16;
    return mx::array(bufs[i].get(), std::move(shape), dt, [keep](void*) {});
  };
  ExpertWeights ew(Q4Projection(wrap(0), wrap(1), wrap(2)),
                   Q4Projection(wrap(3), wrap(4), wrap(5)),
                   Q4Projection(wrap(6), wrap(7), wrap(8)),
                   s.group_size, s.bits);
  mx::array local_arr(local.data(), mx::Shape{(int)T, (int)K}, mx::int32);
  mx::array out = moe_block_sorted(x, local_arr, scores, ew);
  mx::eval(out);  // buffers must outlive the gather_qmm command buffers
  return out;
}

mx::array moe_stream_forward(const std::string& checkpoint_dir, int layer,
                             const mx::array& x, const mx::array& route,
                             const mx::array& scores, int num_slots,
                             int group_size, int bits, int read_width,
                             const std::string& prefix) {
  StreamingMoE m(checkpoint_dir, layer, num_slots, group_size, bits, read_width,
                 prefix);
  return m.forward(x, route, scores).output;
}

}  // namespace veystrix::engine
