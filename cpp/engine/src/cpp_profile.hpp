// Phase 0 opt-in profiler for the C++ engine (Opus brief B1).
//
// Every disabled diagnostic must reduce to a single branch on the hot path:
// no allocations, timers, closures or strings when VEYSTRIX_CPP_PROFILE is off.
// All getenv() calls happen once in init() at startup; the hot path only reads
// the `enabled` bool (a plain member, checked by an inline branch).
//
// Per decode step the profiler writes one JSONL row to <dir>/profile.jsonl with
// build_ms, route_sync_ms (sum + per layer), read_wall_ms, bytes_requested,
// bytes_physical (proc_pid_rusage ri_diskio_bytesread delta), hits/misses/
// coalesced per layer, final_eval_ms, digest_ms, and MLX active/cache/peak.
//
// Phase A diagnostics (all opt-in; single branch when disabled):
//   moe_eval_ms          time spent in mx::eval(out) MoE output waits still present
//   wrap_ms              time (re)wrapping expert buffers as mx::array
//   ple_ms               time in ple_forward (PLE layer)
//   xshard_route_sync_ms time in cross-shard route mx::eval(r) waits
//   cpu_build_ms         = build - syncs - reads - evals (unattributed CPU time)
//   peak_phys_footprint_bytes  max proc_pid_rusage ri_phys_footprint over the run
//
// NOTE on byte accounting: begin_step() must be called at "starting generation"
// (after load, before prefill) so that step 1 = prefill only. consume_target()
// calls it again for each decode step, keeping decode rows clean. Legacy
// artifacts where begin_step() was only inside consume_target() have a step-1
// row holding ~37.5 GB (load+prefill); those rows must be excluded.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>
#if defined(__APPLE__)
#include <libproc.h>
#include <unistd.h>
#endif

namespace veystrix::engine {
namespace cppprof {

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
  // Default-constructed sentinel (returned by now() when disabled) short-
  // circuits WITHOUT calling Clock::now() — disabled diagnostics cost no timer.
  if (t0 == Clock::time_point{}) return 0.0;
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Physical disk bytes read by this process (macOS). Returns 0 on other OSes.
inline uint64_t physical_disk_bytes() {
#if defined(__APPLE__)
  struct rusage_info_v2 ri;
  if (proc_pid_rusage(getpid(), RUSAGE_INFO_V2, (rusage_info_t*)&ri) == 0)
    return (uint64_t)ri.ri_diskio_bytesread;
#endif
  return 0;
}

// Resident physical footprint (macOS ri_phys_footprint), in bytes. Unlike
// ri_resident_size this includes compressed/wired memory and is the number the
// owner's guard reports as "phys_footprint". Returns 0 on other OSes.
inline uint64_t phys_footprint_bytes() {
#if defined(__APPLE__)
  struct rusage_info_v2 ri;
  if (proc_pid_rusage(getpid(), RUSAGE_INFO_V2, (rusage_info_t*)&ri) == 0)
    return (uint64_t)ri.ri_phys_footprint;
#endif
  return 0;
}

// MLX memory snapshot.
struct MemSnap { double active_gib = 0, cache_gib = 0, peak_gib = 0; };
inline MemSnap mem_snapshot() {
  // Declared weak to avoid a hard dependency: tools link mlx anyway.
  return MemSnap{0, 0, 0};
}

struct LayerCounters {
  double route_sync_ms = 0;
  int hits = 0, misses = 0, coalesced = 0;
};

// The single profiler instance. init() is called once at startup.
struct Profiler {
  bool enabled = false;
  bool no_digest = false;
  std::string dir;
  int step = 0;
  Clock::time_point step_start;
  double build_ms = 0, read_wall_ms = 0, final_eval_ms = 0, digest_ms = 0, sample_ms = 0;
  double route_sync_total_ms = 0;
  // Phase A diagnostics.
  double moe_eval_ms = 0, wrap_ms = 0, ple_ms = 0, xshard_route_sync_ms = 0;
  uint64_t peak_phys_footprint_bytes = 0;  // run-wide max, NOT reset per step
  uint64_t bytes_requested = 0, bytes_physical_before = 0;
  std::vector<LayerCounters> layers;
  std::mutex mu;

  void init() {
    const char* p = std::getenv("VEYSTRIX_CPP_PROFILE");
    enabled = (p && *p && std::strcmp(p, "0") != 0);
    const char* d = std::getenv("VEYSTRIX_CPP_PROFILE_DIR");
    if (d && *d) dir = d; else dir = "results/cpp/profile";
    const char* nd = std::getenv("VEYSTRIX_CPP_NO_DIGEST");
    no_digest = (nd && *nd && std::strcmp(nd, "0") != 0);
    const char* pl = std::getenv("VEYSTRIX_CPP_PROFILE_LAYERS");
    per_layer = (pl && std::strcmp(pl, "1") == 0);
  }
  bool per_layer = false;               // VEYSTRIX_CPP_PROFILE_LAYERS=1: mean route sync per layer in STEPPROF
  std::vector<double> acc_layer_sync;

  // Hot-path helpers: each is a single branch when disabled.
  inline void begin_step() {
    if (!enabled) return;
    step_start = Clock::now();
    build_ms = read_wall_ms = final_eval_ms = digest_ms = sample_ms = 0;
    route_sync_total_ms = 0;
    moe_eval_ms = wrap_ms = ple_ms = xshard_route_sync_ms = 0;
    copy_ms = 0;
    bytes_requested = 0;
    bytes_physical_before = physical_disk_bytes();
    layers.clear();
  }
  inline Clock::time_point now() const { return enabled ? Clock::now() : Clock::time_point{}; }
  inline void add_build(double ms) { if (enabled) build_ms += ms; }
  inline void add_route_sync(int layer, double ms) {
    if (!enabled) return;
    route_sync_total_ms += ms;
    if ((int)layers.size() <= layer) layers.resize(layer + 1);
    layers[layer].route_sync_ms += ms;
  }
  inline void add_read(double ms, uint64_t bytes) {
    if (!enabled) return;
    read_wall_ms += ms; bytes_requested += bytes;
  }
  inline void add_final_eval(double ms) { if (enabled) final_eval_ms += ms; }
  inline void add_digest(double ms) { if (enabled) digest_ms += ms; }
  inline void add_sample(double ms) { if (enabled) sample_ms += ms; }
  // Phase A: each is a single branch when disabled.
  inline void add_moe_eval(double ms) { if (enabled) moe_eval_ms += ms; }
  inline void add_wrap(double ms) { if (enabled) wrap_ms += ms; }
  inline void add_ple(double ms) { if (enabled) ple_ms += ms; }
  inline void add_xshard_route_sync(double ms) {
    if (enabled) xshard_route_sync_ms += ms;
  }
  inline void add_layer_cache(int layer, int hits_, int misses_, int coalesced_) {
    if (!enabled) return;
    if ((int)layers.size() <= layer) layers.resize(layer + 1);
    layers[layer].hits += hits_;
    layers[layer].misses += misses_;
    layers[layer].coalesced += coalesced_;
  }
  // Opt-in running summary for long-lived runtimes (the C ABI never writes
  // rows): call once per decode step with the step's wall time; prints the
  // mean per-step buckets to stderr every `every` steps. No fork, no files.
  double acc_step = 0, acc_sync = 0, acc_read = 0, acc_moe = 0, acc_wrap = 0,
         acc_copy = 0, acc_final = 0;
  uint64_t acc_req = 0, acc_phys = 0;
  long acc_hits = 0, acc_miss = 0;
  int acc_n = 0;
  double copy_ms = 0;
  inline void add_copy(double ms) { if (enabled) copy_ms += ms; }
  void end_step_summary(double step_ms, const char* tag, int every = 25) {
    if (!enabled) return;
    acc_step += step_ms; acc_sync += route_sync_total_ms + xshard_route_sync_ms;
    acc_read += read_wall_ms; acc_moe += moe_eval_ms; acc_wrap += wrap_ms;
    acc_copy += copy_ms; acc_final += final_eval_ms;
    acc_req += bytes_requested;
    const uint64_t phys_now = physical_disk_bytes();
    acc_phys += phys_now > bytes_physical_before ? phys_now - bytes_physical_before : 0;
    for (auto& l : layers) { acc_hits += l.hits; acc_miss += l.misses; }
    if (per_layer) {
      if (acc_layer_sync.size() < layers.size()) acc_layer_sync.resize(layers.size(), 0.0);
      for (size_t i = 0; i < layers.size(); ++i) acc_layer_sync[i] += layers[i].route_sync_ms;
    }
    ++acc_n;
    if (acc_n % every == 0) {
      const double n = acc_n;
      std::fprintf(stderr,
                   "[STEPPROF %s] steps=%d per-step ms: total %.1f sync %.1f read %.1f moe_eval %.1f "
                   "copy %.1f wrap %.1f final %.1f other %.1f | MB req %.0f phys %.0f | hit %.1f%%\n",
                   tag, acc_n, acc_step / n, acc_sync / n, acc_read / n, acc_moe / n, acc_copy / n,
                   acc_wrap / n, acc_final / n,
                   (acc_step - acc_sync - acc_read - acc_moe - acc_copy - acc_wrap - acc_final) / n,
                   acc_req / n / 1e6, acc_phys / n / 1e6,
                   (acc_hits + acc_miss) ? 100.0 * acc_hits / (acc_hits + acc_miss) : 0.0);
      if (per_layer) {
        std::fprintf(stderr, "[LAYERSYNC %s] steps=%d ms/step:", tag, acc_n);
        for (double v : acc_layer_sync) std::fprintf(stderr, " %.2f", v / n);
        std::fprintf(stderr, "\n");
      }
    }
    copy_ms = 0;
  }

  // One-off breakdown of the current window (e.g. a whole prefill).
  void print_window(const char* tag, double total_ms) {
    if (!enabled) return;
    const uint64_t phys_now = physical_disk_bytes();
    std::fprintf(stderr,
                 "[WINPROF %s] total %.1f ms: sync %.1f read %.1f moe_eval %.1f copy %.1f wrap %.1f "
                 "| MB req %.0f phys %.0f\n",
                 tag, total_ms, route_sync_total_ms + xshard_route_sync_ms, read_wall_ms,
                 moe_eval_ms, copy_ms, wrap_ms, bytes_requested / 1e6,
                 (phys_now > bytes_physical_before ? phys_now - bytes_physical_before : 0) / 1e6);
  }

  void write_row(double active_gib, double cache_gib, double peak_gib) {
    if (!enabled) return;
    std::lock_guard<std::mutex> lk(mu);
    if (std::system(("mkdir -p '" + dir + "'").c_str()) != 0) return;
    std::FILE* f = std::fopen((dir + "/profile.jsonl").c_str(), "a");
    if (!f) return;
    const uint64_t phys_now = physical_disk_bytes();  // single syscall per row
    const uint64_t footprint = phys_footprint_bytes();  // one more syscall per row
    if (footprint > peak_phys_footprint_bytes) peak_phys_footprint_bytes = footprint;
    // Unattributed CPU time inside build_ms: build wraps the whole run_step,
    // so subtract the read/sync/eval waits that were booked inside it.
    double syncs = route_sync_total_ms + xshard_route_sync_ms;
    double evals = final_eval_ms + moe_eval_ms;
    double cpu_build = build_ms - syncs - read_wall_ms - evals;
    if (cpu_build < 0) cpu_build = 0;
    std::fprintf(f,
      "{\"step\":%d,\"build_ms\":%.4f,\"route_sync_ms\":%.4f,\"read_wall_ms\":%.4f,"
      "\"bytes_requested\":%llu,\"bytes_physical\":%llu,\"final_eval_ms\":%.4f,"
      "\"sample_ms\":%.4f,\"digest_ms\":%.4f,\"active_gib\":%.4f,\"cache_gib\":%.4f,\"peak_gib\":%.4f,"
      "\"moe_eval_ms\":%.4f,\"wrap_ms\":%.4f,\"ple_ms\":%.4f,\"xshard_route_sync_ms\":%.4f,"
      "\"cpu_build_ms\":%.4f,\"peak_phys_footprint_bytes\":%llu,"
      "\"route_sync_per_layer\":[",
      step, build_ms, route_sync_total_ms, read_wall_ms,
      (unsigned long long)bytes_requested,
      (unsigned long long)(phys_now > bytes_physical_before
                            ? phys_now - bytes_physical_before : 0),
      final_eval_ms, sample_ms, digest_ms, active_gib, cache_gib, peak_gib,
      moe_eval_ms, wrap_ms, ple_ms, xshard_route_sync_ms, cpu_build,
      (unsigned long long)peak_phys_footprint_bytes);
    for (size_t i = 0; i < layers.size(); ++i) {
      if (i) std::fputc(',', f);
      std::fprintf(f, "{\"l\":%zu,\"ms\":%.4f,\"h\":%d,\"m\":%d,\"c\":%d}",
                   i, layers[i].route_sync_ms, layers[i].hits,
                   layers[i].misses, layers[i].coalesced);
    }
    std::fprintf(f, "]}\n");
    std::fclose(f);
  }
};

inline Profiler& profiler() {
  static Profiler p;
  return p;
}

}  // namespace cppprof
}  // namespace veystrix::engine
