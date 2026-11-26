#include "memory_budget.hpp"

#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/types.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace veystrix::engine::membudget {

namespace {

bool parse_u64(const std::string& s, unsigned long long& out) {
  if (s.empty()) return false;
  for (char c : s)
    if (!std::isdigit(static_cast<unsigned char>(c))) return false;
  char* end = nullptr;
  out = std::strtoull(s.c_str(), &end, 10);
  return end && *end == '\0';
}

// Flat-object scalar lookup: returns the raw token after `"key"` `:`, empty
// when the key is absent. The ABI refuses nested profiles, so a flat scan is
// sufficient and keeps this file free of a JSON dependency.
std::string profile_scalar(const std::string& json, const char* key) {
  const std::string needle = std::string("\"") + key + "\"";
  std::size_t at = 0;
  while ((at = json.find(needle, at)) != std::string::npos) {
    std::size_t p = at + needle.size();
    while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) ++p;
    if (p < json.size() && json[p] == ':') {
      ++p;
      while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) ++p;
      std::size_t q = p;
      while (q < json.size() && json[q] != ',' && json[q] != '}' &&
             !std::isspace(static_cast<unsigned char>(json[q])))
        ++q;
      return json.substr(p, q - p);
    }
    at += needle.size();
  }
  return {};
}

size_t strict_bytes(const std::string& what, const std::string& raw) {
  unsigned long long v = 0;
  if (!parse_u64(raw, v))
    throw std::runtime_error("memory profile: " + what + " is not an unsigned integer: '" +
                             raw + "'");
  return static_cast<size_t>(v);
}

}  // namespace

size_t env_bytes(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return 0;
  return strict_bytes(name, v);
}

size_t physical_ram_bytes() {
  if (size_t o = env_bytes("VEYSTRIX_CPP_RAM_BYTES")) return o;
  uint64_t mem = 0;
  size_t len = sizeof(mem);
  if (::sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0 || mem == 0)
    throw std::runtime_error("memory budget: cannot read hw.memsize");
  return static_cast<size_t>(mem);
}

size_t available_ram_bytes() {
  if (size_t o = env_bytes("VEYSTRIX_CPP_AVAIL_RAM_BYTES")) return o;
  vm_size_t page_size = 0;
  if (::host_page_size(::mach_host_self(), &page_size) != KERN_SUCCESS || page_size == 0)
    return physical_ram_bytes();
  vm_statistics64_data_t vm{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (::host_statistics64(::mach_host_self(), HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vm), &count) != KERN_SUCCESS)
    return physical_ram_bytes();
  const uint64_t avail_pages =
      static_cast<uint64_t>(vm.free_count) +
      static_cast<uint64_t>(vm.inactive_count) +
      static_cast<uint64_t>(vm.speculative_count);
  return static_cast<size_t>(avail_pages * static_cast<uint64_t>(page_size));
}

size_t swap_used_bytes() {
  if (size_t o = env_bytes("VEYSTRIX_CPP_SWAP_USED_BYTES")) return o;
  struct xsw_usage xsu{};
  size_t len = sizeof(xsu);
  if (::sysctlbyname("vm.swapusage", &xsu, &len, nullptr, 0) != 0) return 0;
  return static_cast<size_t>(xsu.xsu_used);
}

int memory_pressure_level() {
  if (const char* e = std::getenv("VEYSTRIX_CPP_PRESSURE_LEVEL")) {
    if (*e) return std::max(1, std::atoi(e));
  }
  int level = 1;
  size_t len = sizeof(level);
  if (::sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &len, nullptr, 0) != 0)
    return 1;
  return level > 0 ? level : 1;
}

size_t tier_mlx_cache_bytes() {
  const size_t phys = physical_ram_bytes();
  if (phys >= (size_t{80} << 30)) return size_t{2} << 30;    // 96-128+ GB Macs: 2 GiB
  if (phys >= (size_t{43} << 30)) return size_t{1} << 30;    // 48-64 GB Macs: 1 GiB
  if (phys >= (size_t{20} << 30)) return size_t{512} << 20;  // 24-32 GB Macs: 512 MiB
  return size_t{256} << 20;                                  // 16 GB Macs: 256 MiB
}

InklingResidencyPlan plan_inkling_residency(size_t phys_bytes, size_t avail_bytes,
                                            size_t swap_bytes, int pressure) {
  InklingResidencyPlan plan;
  constexpr size_t kGiB = size_t{1} << 30;
  constexpr size_t kPerLayer = size_t{550} << 20;  // ~534 MiB weights + allocator alignment

  // < 16 GB (e.g. 8 GB Mac): stream all MoE layers.
  if (phys_bytes < 16 * kGiB) return plan;

  // Critical VM pressure (level 4) on ANY Mac: keep residency minimal so macOS recovers.
  if (pressure >= 4) {
    plan.core = avail_bytes >= 6 * kGiB ? 2 : 0;
    return plan;
  }

  const size_t swap_tax =
      swap_bytes > 2 * kGiB
          ? std::min<size_t>(swap_bytes - 2 * kGiB, 2 * kGiB)
          : 0;
  const size_t effective_avail = avail_bytes > swap_tax ? (avail_bytes - swap_tax) : 0;

  // 32 / 48 / 64 / 96 / 128+ GB Macs: full 24-layer residency (15.10 GiB) whenever RAM allows.
  if (phys_bytes >= 28 * kGiB) {
    if (effective_avail >= (size_t{33} << 29)) {  // >= 16.5 GiB effective available
      plan.core = 12;
      plan.extra = 12;
      return plan;
    }
    if (effective_avail >= (size_t{21} << 29)) {  // >= 10.5 GiB: all 12 core + extra that fit
      plan.core = 12;
      const size_t rem = effective_avail - (size_t{21} << 29);
      plan.extra = static_cast<int>(std::min<size_t>(rem / kPerLayer, 12));
      return plan;
    }
    const size_t base = 4 * kGiB;
    plan.core = effective_avail > base ? static_cast<int>(std::min<size_t>((effective_avail - base) / kPerLayer, 12)) : 0;
    plan.extra = 0;
    return plan;
  }

  // 24 GB Macs: full 24-layer residency when >= 17 GiB available, otherwise 12 core + extra.
  if (phys_bytes >= 20 * kGiB) {
    if (effective_avail >= 17 * kGiB) {
      plan.core = 12;
      plan.extra = 12;
      return plan;
    }
    if (effective_avail >= (size_t{21} << 29)) {  // >= 10.5 GiB
      plan.core = 12;
      const size_t rem = effective_avail - (size_t{21} << 29);
      plan.extra = static_cast<int>(std::min<size_t>(rem / kPerLayer, 12));
      return plan;
    }
    const size_t base = (size_t{36} << 30) / 10;
    plan.core = effective_avail > base ? static_cast<int>(std::min<size_t>((effective_avail - base) / kPerLayer, 12)) : 0;
    plan.extra = 0;
    return plan;
  }

  // 16 GB Macs: keep ~2.0 GiB OS/KV/prefill headroom and cap core residency at 9 layers
  // (~7.1 GiB total resident) so 2k+ prefill scratch + KV stays below the macOS paging cliff.
  const size_t base = 2 * kGiB;
  const int fit = effective_avail > base ? static_cast<int>((effective_avail - base) / kPerLayer) : 0;
  plan.core = std::clamp(fit, 0, 9);
  plan.extra = 0;
  return plan;
}

int dynamic_prefill_chunk(int base_chunk, size_t phys_bytes, size_t avail_bytes,
                          size_t active_bytes, int pressure) {
  if (base_chunk > 0 && base_chunk < 256) return base_chunk;
  constexpr size_t kGiB = size_t{1} << 30;
  const int base = std::clamp(base_chunk, 256, 16384);
  if (pressure >= 4) return std::min(base, 256);
  // On 16 GB Macs, 2-slice streamed MoE prefill keeps per-layer expert scratch at
  // ~273 MiB (half of 547 MiB) and attention is tiled at qb <= 512, so 1024-token
  // chunks stay out of swap while cutting 10K SSD reads by 4x vs 256-token chunks.
  // Step down to 512 under warning pressure or < 1.5 GiB available, 256 on critical.
  if (phys_bytes < 20 * kGiB) {
    if (pressure >= 2 || avail_bytes < (size_t{3} << 29)) return std::min(base, 512);
    return std::min(base, 1024);
  }
  const size_t metal_budget = (phys_bytes * 3) / 4;
  const size_t metal_headroom = metal_budget > active_bytes ? (metal_budget - active_bytes) : 0;
  if (pressure >= 2 || avail_bytes < (size_t{18} << 30) / 10 || metal_headroom < (size_t{18} << 30) / 10)
    return std::min(base, 512);
  if (avail_bytes < (size_t{32} << 30) / 10 || metal_headroom < (size_t{28} << 30) / 10)
    return std::min(base, 1024);
  if (phys_bytes >= 56 * kGiB && avail_bytes >= 16 * kGiB) return std::max(base, 8192);
  if (phys_bytes >= 28 * kGiB && avail_bytes >= 8 * kGiB) return std::max(base, 4096);
  return base;
}

size_t ram_headroom_bytes(size_t active_bytes) {
  const size_t phys = physical_ram_bytes();
  // Same rule as veystrix/memory_budget_runtime.py: max(3 GiB, RAM/4).
  size_t reserve = std::max(kDefaultResidentReserveBytes, phys / 4);
  if (const char* r = std::getenv("VEYSTRIX_CPP_RESIDENT_RESERVE_BYTES"))
    if (*r) reserve = strict_bytes("VEYSTRIX_CPP_RESIDENT_RESERVE_BYTES", r);
  const size_t used = active_bytes + reserve;
  return phys > used ? phys - used : 0;
}

MemoryProfile resolve_memory_profile(const std::string& profile_json) {
  MemoryProfile p;
  // pool bytes
  {
    std::string t = profile_scalar(profile_json, "expert_pool_bytes");
    p.pool_bytes = t.empty() ? env_bytes("VEYSTRIX_CPP_POOL_BYTES")
                             : strict_bytes("expert_pool_bytes", t);
  }
  // nemotron slots
  {
    std::string t = profile_scalar(profile_json, "nemotron_slots");
    p.nemo_slots = static_cast<long long>(
        t.empty() ? env_bytes("VEYSTRIX_CPP_NEMO_POOL_SLOTS")
                  : strict_bytes("nemotron_slots", t));
  }
  // resident: true/false/0/1
  {
    std::string t = profile_scalar(profile_json, "resident");
    if (t.empty()) {
      const char* e = std::getenv("VEYSTRIX_CPP_RESIDENT");
      if (e != nullptr && *e != '\0') t = e;
    }
    if (t.empty() || t == "0" || t == "false") {
      p.resident = false;
    } else if (t == "1" || t == "true") {
      p.resident = true;
    } else {
      throw std::runtime_error("memory profile: resident must be 0/1/true/false, got '" +
                               t + "'");
    }
  }
  // max sessions
  {
    std::string t = profile_scalar(profile_json, "max_sessions");
    p.max_sessions = t.empty() ? env_bytes("VEYSTRIX_CPP_MAX_SESSIONS")
                               : strict_bytes("max_sessions", t);
  }
  return p;
}

ResidentDecision decide_resident(size_t needed_bytes, size_t explicit_budget,
                                 size_t ram_headroom) {
  ResidentDecision d;
  d.needed = needed_bytes;
  d.explicit_budget = explicit_budget;
  d.ram_headroom = ram_headroom;
  if (needed_bytes == 0) {
    d.reason = "no experts to hold";
  } else if (explicit_budget != 0 && needed_bytes > explicit_budget) {
    d.reason = "all experts need " + std::to_string(needed_bytes) +
               " B but the pool budget is " + std::to_string(explicit_budget) + " B";
  } else if (needed_bytes > ram_headroom) {
    d.reason = "all experts need " + std::to_string(needed_bytes) +
               " B but only " + std::to_string(ram_headroom) +
               " B of RAM headroom is available";
  } else {
    d.fits = true;
  }
  return d;
}

void require_resident_fit(const std::string& who, size_t needed_bytes,
                          size_t explicit_budget, size_t active_bytes) {
  const ResidentDecision d =
      decide_resident(needed_bytes, explicit_budget, ram_headroom_bytes(active_bytes));
  if (!d.fits)
    throw std::runtime_error(who + ": resident refused: " + d.reason +
                             " (reopen with resident=0 to stream from SSD)");
}

void require_ram_headroom(const std::string& who, size_t needed_bytes,
                          size_t active_bytes) {
  const size_t head = ram_headroom_bytes(active_bytes);
  if (needed_bytes > head)
    throw std::runtime_error(who + ": pool of " + std::to_string(needed_bytes) +
                             " B exceeds RAM headroom of " + std::to_string(head) + " B");
}

size_t apply_mlx_limits_at_load(const std::string& profile_json) {
  size_t cache = env_bytes("VEYSTRIX_CPP_MLX_CACHE_BYTES");
  if (const size_t mem = env_bytes("VEYSTRIX_CPP_MLX_MEMORY_BYTES")) mx::set_memory_limit(mem);
  if (cache == 0) {
    const bool dynamic =
        !profile_scalar(profile_json, "resident").empty() ||
        std::getenv("VEYSTRIX_CPP_RESIDENT") != nullptr;
    if (dynamic) cache = tier_mlx_cache_bytes();
  }
  if (cache != 0) mx::set_cache_limit(cache);
  return cache;
}

size_t clear_mlx_cache() {
  const size_t before = mx::get_cache_memory();
  mx::clear_cache();
  return before;
}

int idle_clear_ms() {
  const char* v = std::getenv("VEYSTRIX_CPP_IDLE_CLEAR_MS");
  if (v == nullptr || *v == '\0') return 2000;
  unsigned long long ms = 0;
  // Called from the driver loop: never throw there. Garbage -> the default
  // (the idle clear is cache hygiene only, so a lenient fallback is safe).
  if (!parse_u64(v, ms) || ms > 3600000ULL) return 2000;
  return static_cast<int>(ms);
}

}  // namespace veystrix::engine::membudget
