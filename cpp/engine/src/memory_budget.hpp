// Dynamic-memory contract for the C++ engine (cache / residency management
// ONLY — no numeric path reads or writes anything here).
//
// The Python side (veystrix/memory_budget_runtime.py, exported by
// veystrix/backends/cpp_engine.py::apply_runtime_memory_budget) decides how much
// RAM the expert pools may use and whether the model is fully resident. It
// reaches C++ two ways, resolved here with ONE precedence rule
// (profile key > environment variable > default):
//
//   profile key         env var                        meaning
//   expert_pool_bytes   VEYSTRIX_CPP_POOL_BYTES        byte budget of the pool
//   nemotron_slots      VEYSTRIX_CPP_NEMO_POOL_SLOTS   Nemotron pool slot count
//   resident            VEYSTRIX_CPP_RESIDENT          1 = hold ALL experts
//   max_sessions        VEYSTRIX_CPP_MAX_SESSIONS      session cap (C ABI only)
//
// Resident mode is FAIL CLOSED: if every (layer, expert) record does not fit
// the stated byte budget AND the machine's RAM headroom, engine open throws
// (vx_engine_open returns NULL + vx_last_error). The caller then reopens with
// resident=0, which streams from SSD as before. We never silently stream while
// claiming residency and never allocate more than the RAM gate allows.
#pragma once
#include <cstddef>
#include <string>

namespace veystrix::engine::membudget {

// Floor of the headroom kept free for the OS / working set when deciding
// whether a resident pool may be allocated. The effective reserve is
// max(3 GiB, physical/4) -- identical to memory_budget_runtime.py -- and can be
// overridden (bytes) with VEYSTRIX_CPP_RESIDENT_RESERVE_BYTES.
constexpr size_t kDefaultResidentReserveBytes = size_t{3} << 30;

// Strict unsigned parse of an environment variable. Returns 0 when unset or
// empty; throws std::runtime_error on non-numeric text (fail closed).
size_t env_bytes(const char* name);

// Physical RAM in bytes (sysctl hw.memsize). VEYSTRIX_CPP_RAM_BYTES overrides
// it; that override exists so the offline gates can simulate other tiers.
size_t physical_ram_bytes();

// Currently available (free + inactive + speculative) RAM in bytes via
// host_statistics64(HOST_VM_INFO64). VEYSTRIX_CPP_AVAIL_RAM_BYTES overrides it
// for tests; falls back to physical_ram_bytes() if the Mach query fails.
size_t available_ram_bytes();

// Currently used swap in bytes via sysctl vm.swapusage.
// VEYSTRIX_CPP_SWAP_USED_BYTES overrides it for tests; returns 0 if unavailable.
size_t swap_used_bytes();

// macOS VM pressure level via sysctl kern.memorystatus_vm_pressure_level:
// 1 = normal, 2 = warning, 4 = critical. VEYSTRIX_CPP_PRESSURE_LEVEL overrides
// it for tests; defaults to 1 if unavailable.
int memory_pressure_level();

// Tier-appropriate MLX free-buffer cache cap (bytes):
//   16 GB -> 256 MiB, 24-32 GB -> 512 MiB, 48-64 GB -> 1 GiB, 96-128+ GB -> 2 GiB.
size_t tier_mlx_cache_bytes();

// Dynamic Inkling 2 Mini residency plan (looped core layers 6..17 + extra
// prelude/epilogue layers 0..5, 18..23) sized from physical RAM, live available
// RAM, swap used, and VM pressure across 16/24/32/48/64/96/128 GB Macs.
struct InklingResidencyPlan {
  int core = 0;   // 0..12 looped core layers (each runs 3x per token)
  int extra = 0;  // 0..12 prelude/epilogue layers (0..5 then 18..23)
};
InklingResidencyPlan plan_inkling_residency(size_t phys_bytes, size_t avail_bytes,
                                            size_t swap_bytes, int pressure);

// Dynamic prefill chunk size based on live available RAM, Metal active memory,
// physical RAM tier, and VM pressure (shrinks under pressure, scales up on
// 32-128 GB Macs when plenty of RAM is free).
int dynamic_prefill_chunk(int base_chunk, size_t phys_bytes, size_t avail_bytes,
                          size_t active_bytes, int pressure);

// RAM a pool may still claim: physical - active_bytes - reserve (clamped >= 0).
size_t ram_headroom_bytes(size_t active_bytes);

struct MemoryProfile {
  size_t pool_bytes = 0;  // 0 = unset
  long long nemo_slots = 0;  // 0 = unset
  bool resident = false;
  size_t max_sessions = 0;  // 0 = unset
};

// Resolve the four knobs from the flat profile JSON (profile key wins) and the
// environment. `resident` may be given as true/false/0/1. Garbage throws.
MemoryProfile resolve_memory_profile(const std::string& profile_json);

// Pure decision (no allocation): fits iff needed <= explicit_budget (when
// non-zero) AND needed <= ram_headroom. `reason` is empty when it fits.
struct ResidentDecision {
  bool fits = false;
  size_t needed = 0;
  size_t explicit_budget = 0;
  size_t ram_headroom = 0;
  std::string reason;
};
ResidentDecision decide_resident(size_t needed_bytes, size_t explicit_budget,
                                 size_t ram_headroom);

// Throws std::runtime_error("<who>: resident refused: ...") when it does not fit.
void require_resident_fit(const std::string& who, size_t needed_bytes,
                          size_t explicit_budget, size_t active_bytes);

// RAM gate for a NON-resident pool: refuses any allocation larger than the
// current headroom so a mis-sized pool cannot push the machine into swap.
void require_ram_headroom(const std::string& who, size_t needed_bytes,
                          size_t active_bytes);

// MLX allocator cache management (never changes tensor values).
//  * VEYSTRIX_CPP_MLX_CACHE_BYTES      -> mx::set_cache_limit(value)
//  * VEYSTRIX_CPP_MLX_MEMORY_BYTES     -> mx::set_memory_limit(value)
//  * unset but a dynamic budget is active (resident knob present) ->
//    cache limit = min(2 GiB, physical/8).
//  * nothing set and no dynamic budget -> MLX defaults untouched.
// Returns the cache limit applied (0 = left at the MLX default).
size_t apply_mlx_limits_at_load(const std::string& profile_json);

// mx::clear_cache() wrapper (driver thread only). Returns bytes released.
size_t clear_mlx_cache();

// Driver idle timeout before the cache is cleared (ms). Default 2000,
// VEYSTRIX_CPP_IDLE_CLEAR_MS=0 disables the idle clear; garbage -> default.
int idle_clear_ms();

}  // namespace veystrix::engine::membudget
