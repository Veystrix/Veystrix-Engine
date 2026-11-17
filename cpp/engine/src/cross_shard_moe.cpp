#include "cross_shard_moe.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "io_pager.hpp"
#include "slot_plan.hpp"

namespace mx = mlx::core;
namespace veystrix::engine {

namespace xsmoe {

struct TensorMeta {
  std::vector<int> shape;
  int64_t offset = -1;
  int64_t row_bytes = 0;
};

namespace {

bool find_value(const std::string& s, const std::string& key, size_t from,
                size_t& value_begin) {
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
  size_t b;
  if (!find_value(obj, key, 0, b)) return false;
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

}  // namespace

bool parse_tensor(const std::string& header, const std::string& name,
                  TensorMeta& out) {
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
    throw std::runtime_error("short safetensors length: " + path);
  }
  uint64_t n = 0;
  for (int i = 0; i < 8; ++i) n |= static_cast<uint64_t>(lenbuf[i]) << (8 * i);
  header.resize(n);
  got = ::pread(fd, header.data(), n, 8);
  ::close(fd);
  if (got != static_cast<ssize_t>(n))
    throw std::runtime_error("short safetensors header: " + path);
  return static_cast<int64_t>(8 + n);
}

}  // namespace xsmoe

struct CrossShardMoE::Impl {
  static constexpr int kTensors = 9;
  int num_slots, group_size, bits;
  bool deep_copy;
  std::map<std::string, int> fds;
  std::vector<int> fd_of;
  std::vector<xsmoe::TensorMeta> t;
  std::vector<int64_t> absolute;
  std::vector<std::shared_ptr<uint8_t>> buf;
  std::vector<int32_t> slot_expert, free_slots;
  std::vector<int> slot_ref;
  std::vector<uint64_t> slot_last;
  uint64_t tick = 0;

  Impl(const std::string& dir, const std::map<std::string, std::string>& cfg,
       const std::string& prefix, int slots, int gs, int b, bool copy)
      : num_slots(slots), group_size(gs), bits(b), deep_copy(copy) {
    static const char* names[kTensors] = {
        "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
        "up_proj.weight",   "up_proj.scales",   "up_proj.biases",
        "down_proj.weight", "down_proj.scales", "down_proj.biases"};
    static const bool is_weight[kTensors] = {true, false, false, true, false,
                                             false, true, false, false};
    t.resize(kTensors);
    absolute.resize(kTensors);
    for (int i = 0; i < kTensors; ++i) {
      std::string key = prefix + "." + names[i];
      auto it = cfg.find("shard." + key);
      if (it == cfg.end())
        throw std::runtime_error("no shard map for expert tensor: " + key);
      std::string shard = it->second;
      auto fit = fds.find(shard);
      if (fit == fds.end()) {
        int fd = ::open((dir + "/" + shard).c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open expert shard: " + shard);
        fit = fds.emplace(shard, fd).first;
      }
      std::string header;
      int64_t data_start = xsmoe::read_header(dir + "/" + shard, header);
      if (!xsmoe::parse_tensor(header, key, t[i]))
        throw std::runtime_error("cannot parse expert tensor: " + key);
      int64_t itemsize = is_weight[i] ? 4 : 2;
      int64_t row = 1;
      for (size_t d = 1; d < t[i].shape.size(); ++d) row *= t[i].shape[d];
      t[i].row_bytes = row * itemsize;
      absolute[i] = data_start + t[i].offset;
      fd_of.push_back(fit->second);
    }
    buf.resize(kTensors);
    for (int i = 0; i < kTensors; ++i) {
      size_t bytes = static_cast<size_t>(num_slots) *
                     static_cast<size_t>(t[i].row_bytes);
      void* raw = nullptr;
      if (::posix_memalign(&raw, 64, bytes) != 0)
        throw std::runtime_error("posix_memalign failed (cross-shard MoE)");
      std::memset(raw, 0, bytes);
      buf[i] = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(raw),
                                        [](uint8_t* p) { std::free(p); });
    }
    slot_expert.assign(num_slots, -1);
    slot_ref.assign(num_slots, 0);
    slot_last.assign(num_slots, 0);
    for (int s = 0; s < num_slots; ++s) free_slots.push_back(s);
  }

  ~Impl() {
    for (auto& [shard, fd] : fds)
      if (fd >= 0) ::close(fd);
  }
};

CrossShardMoE::CrossShardMoE(const std::string& dir,
                             const std::map<std::string, std::string>& cfg,
                             const std::string& prefix, int num_slots,
                             int group_size, int bits, bool deep_copy)
    : impl_(std::make_unique<Impl>(dir, cfg, prefix, num_slots, group_size,
                                   bits, deep_copy)) {}

CrossShardMoE::~CrossShardMoE() = default;

StreamForwardResult CrossShardMoE::forward(const mx::array& x,
                                           const mx::array& route,
                                           const mx::array& scores) {
  Impl& s = *impl_;
  mx::array r = mx::astype(route, mx::int32);
  mx::eval(r);
  const int32_t* rp = r.data<int32_t>();
  std::vector<int32_t> ids(rp, rp + r.size());
  if (ids.empty()) throw std::runtime_error("empty route (cross-shard MoE)");
  size_t T = route.shape().size() >= 2 ? static_cast<size_t>(route.shape(0)) : 1;
  size_t K = ids.size() / T;

  // Single source of truth for slot bookkeeping (same as StreamingMoE).
  SlotStep step = plan_slots(ids, s.slot_expert, s.slot_ref, s.slot_last,
                             s.free_slots, s.num_slots);

  // Pairs from POST-eviction resident state.
  std::vector<int32_t> pairs;
  for (int slot = 0; slot < s.num_slots; ++slot)
    if (s.slot_expert[slot] >= 0) {
      pairs.push_back(s.slot_expert[slot]);
      pairs.push_back(slot);
    }
  std::vector<int32_t> free_snapshot = s.free_slots;

  std::vector<veystrix::PagerRead> reads;
  std::vector<int32_t> slots;
  for (int i = 0; i < Impl::kTensors; ++i) {
    veystrix::PagerPlan plan = veystrix::plan_remap(
        ids, pairs, free_snapshot, s.fd_of[i], s.absolute[i], s.t[i].row_bytes,
        s.buf[i].get(), s.t[i].row_bytes);
    if (i == 0) slots = plan.slots;
    else if (plan.slots != slots)
      throw std::runtime_error("inconsistent slot plan across expert tensors");
    if (plan.slots != step.slots)
      throw std::runtime_error("plan_remap slot plan disagrees with plan_slots");
    if (plan.reads.size() != step.miss_e.size())
      throw std::runtime_error("plan_remap read count disagrees with plan_slots");
    for (auto& rd : plan.reads) reads.push_back(rd);
  }
  // Fail-closed range check (the old cross-shard path had none at all).
  for (int32_t slot : slots)
    if (slot < 0 || slot >= s.num_slots)
      throw std::runtime_error("fail-closed: slot out of range (cross-shard)");

  std::vector<int32_t> used = slots;
  std::sort(used.begin(), used.end());
  used.erase(std::unique(used.begin(), used.end()), used.end());
  for (int32_t slot : used) ++s.slot_ref[slot];
  for (size_t i = 0; i < step.miss_e.size(); ++i) {
    s.slot_expert[step.miss_s[i]] = step.miss_e[i];
    auto it = std::find(s.free_slots.begin(), s.free_slots.end(), step.miss_s[i]);
    if (it != s.free_slots.end()) s.free_slots.erase(it);
  }
  if (!reads.empty() && veystrix::run_batched_reads(reads, 8) != 0)
    throw std::runtime_error("cross-shard batched read failed");

  // Debug verifier (single early-return branch when disabled).
  if (verify_slots_enabled()) {
    std::vector<int64_t> row_bytes(Impl::kTensors);
    std::vector<uint8_t*> bufs(Impl::kTensors);
    for (int i = 0; i < Impl::kTensors; ++i) {
      row_bytes[i] = s.t[i].row_bytes;
      bufs[i] = s.buf[i].get();
    }
    verify_slot_fill(true, ids, slots, s.slot_expert, step.miss_e, step.miss_s,
                     s.fd_of, s.absolute, row_bytes, bufs);
  }

  std::vector<mx::array> a;
  a.reserve(Impl::kTensors);
  for (int i = 0; i < Impl::kTensors; ++i) {
    mx::Shape shape(s.t[i].shape.begin(), s.t[i].shape.end());
    shape[0] = s.num_slots;
    mx::Dtype dtype = (i % 3 == 0) ? mx::uint32 : mx::bfloat16;
    if (s.deep_copy) {
      if (dtype == mx::uint32) {
        a.emplace_back(reinterpret_cast<uint32_t*>(s.buf[i].get()),
                       std::move(shape), dtype);
      } else {
        a.emplace_back(reinterpret_cast<mx::bfloat16_t*>(s.buf[i].get()),
                       std::move(shape), dtype);
      }
    } else {
      auto keep = s.buf[i];
      a.emplace_back(s.buf[i].get(), std::move(shape), dtype,
                     [keep](void*) {});
    }
  }
  ExpertWeights ew(Q4Projection(a[0], a[1], a[2]),
                   Q4Projection(a[3], a[4], a[5]),
                   Q4Projection(a[6], a[7], a[8]),
                   s.group_size, s.bits);
  mx::array route_slots(slots.begin(),
                        mx::Shape{static_cast<int>(T), static_cast<int>(K)},
                        mx::int32);
  mx::array out = moe_block(x, route_slots, scores, ew);
  mx::eval(out);
  for (int32_t slot : used) --s.slot_ref[slot];
  for (int32_t slot : used) s.slot_last[slot] = ++s.tick;
  return {out, slots, step.miss_e};
}

}  // namespace veystrix::engine
