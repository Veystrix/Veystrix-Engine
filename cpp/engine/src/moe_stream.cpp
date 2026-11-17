#include "moe_stream.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

#include "io_pager.hpp"

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

  // Distinct experts needed by this step.
  std::unordered_set<int32_t> needed(route_ids.begin(), route_ids.end());

  // Resident set (expert -> slot) and the caller's free-slot list.
  std::vector<int32_t> pairs;
  for (int slot = 0; slot < s.num_slots; ++slot)
    if (s.slot_expert[slot] >= 0) {
      pairs.push_back(s.slot_expert[slot]);
      pairs.push_back(slot);
    }
  // Count experts in `needed` that are not already resident.
  size_t misses = 0;
  for (int32_t e : needed) {
    bool found = false;
    for (size_t i = 0; i + 1 < pairs.size(); i += 2)
      if (pairs[i] == e) { found = true; break; }
    if (!found) ++misses;
  }

  // Evict LRU slots (never a slot whose refcount > 0 or that this route needs)
  // until there are enough free slots for the misses. Fail closed otherwise.
  if (misses > s.free_slots.size()) {
    std::vector<int> candidates;
    for (int slot = 0; slot < s.num_slots; ++slot) {
      if (s.slot_expert[slot] < 0 || s.slot_refcount[slot] > 0) continue;
      if (needed.count(s.slot_expert[slot])) continue;  // needed this step
      candidates.push_back(slot);
    }
    std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
      return s.slot_lastused[a] < s.slot_lastused[b];
    });
    for (int slot : candidates) {
      if (misses <= s.free_slots.size()) break;
      s.slot_expert[slot] = -1;
      s.free_slots.push_back(slot);
    }
  }
  if (misses > s.free_slots.size())
    throw std::runtime_error("slot pool exhausted; refusing partial remap");

  std::vector<int32_t> free_snapshot = s.free_slots;

  // One plan per tensor geometry; all must agree on the slot mapping. Collect
  // the batched reads across all nine tensors into a single pager call.
  std::vector<veystrix::PagerRead> all_reads;
  std::vector<int32_t> slots;
  for (int i = 0; i < Impl::kTensors; ++i) {
    veystrix::PagerPlan plan = veystrix::plan_remap(
        route_ids, pairs, free_snapshot, s.shard_fd,
        s.tensor_off[i], s.t[i].row_bytes,
        s.buf[i].get(), s.t[i].row_bytes);
    if (i == 0) {
      slots = plan.slots;
    } else if (plan.slots != slots) {
      throw std::runtime_error("inconsistent slot plan across tensors");
    }
    for (auto& read : plan.reads) all_reads.push_back(read);
  }
  if (std::find(slots.begin(), slots.end(), -1) != slots.end())
    throw std::runtime_error("fail-closed: plan_remap returned slot -1");

  // Pin every slot used this step; release only after mx::eval (below),
  // i.e. after all command buffers reading those buffers have completed.
  std::vector<int32_t> used = slots;
  std::sort(used.begin(), used.end());
  used.erase(std::unique(used.begin(), used.end()), used.end());
  std::vector<int32_t> misses_experts;
  std::vector<int32_t> misses_slots;
  {
    // Derive the (deduped) missed experts from the first tensor's reads.
    int64_t off0 = s.tensor_off[0], rb0 = s.t[0].row_bytes;
    uint8_t* base0 = s.buf[0].get();
    for (const auto& read : all_reads) {
      if (read.offset < off0 || (read.offset - off0) % rb0 != 0) continue;
      int32_t e = static_cast<int32_t>((read.offset - off0) / rb0);
      int32_t slot = static_cast<int32_t>(
          (reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(read.dst)) - base0) / rb0);
      misses_experts.push_back(e);
      misses_slots.push_back(slot);
    }
  }
  for (int32_t slot : used) ++s.slot_refcount[slot];
  for (size_t i = 0; i < misses_experts.size(); ++i) {
    s.slot_expert[misses_slots[i]] = misses_experts[i];
    auto it = std::find(s.free_slots.begin(), s.free_slots.end(), misses_slots[i]);
    if (it != s.free_slots.end()) s.free_slots.erase(it);
  }

  // Batched positioned reads of all missing rows straight from the shard.
  if (!all_reads.empty()) {
    int errors = veystrix::run_batched_reads(all_reads, s.read_width);
    if (errors != 0)
      throw std::runtime_error("pager batched read failed: " + std::to_string(errors));
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
  mx::array out = moe_block(x, route_slots, scores, w);
  mx::eval(out);  // command buffers complete -> safe to unpin slots

  for (int32_t slot : used) --s.slot_refcount[slot];
  for (int32_t slot : used) s.slot_lastused[slot] = ++s.tick;

  return StreamForwardResult{out, slots, misses_experts};
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
