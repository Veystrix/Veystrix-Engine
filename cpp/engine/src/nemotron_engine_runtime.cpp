// Phase G — long-lived Nemotron runtime behind the narrow C ABI.
//
// WHY THE INCLUDE: run_generation() in nemotron_generate.cpp owns the carefully
// ordered flat-config loader, load_model, ExpertReader, NemoPool assembly,
// sequential (chunk_tokens=1) prefill, run_step and sample_greedy path. We include
// that TU with main() renamed so EVERY numeric primitive is reused verbatim; no
// engine primitive is reimplemented and no tool file is edited. The split here
// is purely lifecycle: load once (open), prefill on demand, decode one token per
// step, cancel at a step boundary.
//
// C1 THREADING: this TU owns ONE driver thread and a single mutex+CV. Every MLX
// call (load, prefill, step) runs on that thread. ABI caller threads post a
// std::function and block; ctypes releases the GIL while blocked. MLX default
// streams are per-thread, so a single driver gives bit-exact streams.
//
// Verification hooks (same names as the Qwen runtime, off by default):
//   VEYSTRIX_CPP_CAPI_STEP_LOG=<path> append CLI-format step records with the
//       logit SHA-256, so a ctypes run can be gated against nemotron_generate.
//   VEYSTRIX_CPP_CAPI_ASSERT_THREAD=1 abort/throw if a job runs off the driver.
#include "nemotron_engine_runtime.hpp"
#include "memory_budget.hpp"
#include "nemotron_runtime.hpp"  // SessionCacheState (per-session Mamba/attn cache)

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#define main nemogen_capi_main_disabled
#include "../tools/nemotron_generate.cpp"
#undef main

#include "sampler_json.hpp"
#include "sampling.hpp"
#include "nemo_snapshot.hpp"
#include <mlx/memory.h>

namespace mx = mlx::core;
using namespace veystrix::engine;

namespace veystrix::engine::capi_live {

namespace {

// Minimal extractor for the Phase G profile: {"config_path": "/path/to/
// nemotron_config.txt"} or a bare path to a flat config file. All other keys
// are ignored (the flat config is authoritative for the shard map).
std::string config_path_from_profile(const std::string& profile_json) {
  if (profile_json.empty())
    throw std::runtime_error("nemotron runtime: empty profile_json; pass {\"config_path\":\"...\"}");
  if (profile_json[0] != '{') return profile_json;  // bare flat-config path
  const std::string key = "\"config_path\"";
  std::size_t at = profile_json.find(key);
  if (at == std::string::npos)
    throw std::runtime_error("nemotron runtime: profile_json needs a config_path key");
  std::size_t colon = profile_json.find(':', at + key.size());
  std::size_t q1 = profile_json.find('"', colon == std::string::npos ? at : colon);
  if (q1 == std::string::npos)
    throw std::runtime_error("nemotron runtime: bad config_path value");
  std::size_t q2 = profile_json.find('"', q1 + 1);
  if (q2 == std::string::npos)
    throw std::runtime_error("nemotron runtime: unterminated config_path value");
  std::string path = profile_json.substr(q1 + 1, q2 - q1 - 1);
  if (path.empty()) throw std::runtime_error("nemotron runtime: empty config_path");
  return path;
}

// Greedy-only sampler flag. Non-greedy fails closed instead of silently
// producing different tokens than the requested sampler.
void check_sampler_json(const std::string& sampler_json) {
  if (sampler_json.empty()) return;
  std::size_t at = sampler_json.find("\"greedy\"");
  if (at == std::string::npos) return;
  std::size_t colon = sampler_json.find(':', at + 8);
  if (colon == std::string::npos) return;
  std::size_t f = sampler_json.find("false", colon);
  std::size_t t = sampler_json.find("true", colon);
  if (f != std::string::npos && (t == std::string::npos || f < t))
    throw std::runtime_error(
        "nemotron runtime: non-greedy sampling is not wired in Phase G C ABI");
}

bool thread_assert_enabled() {
  const char* e = std::getenv("VEYSTRIX_CPP_CAPI_ASSERT_THREAD");
  return e && std::strcmp(e, "1") == 0;
}

std::string step_log_path() {
  const char* e = std::getenv("VEYSTRIX_CPP_CAPI_STEP_LOG");
  return e ? std::string(e) : std::string();
}

struct Session {
  int id = 0;
  std::vector<int32_t> chain;       // prompt + emitted tokens
  std::optional<mx::array> logits;  // logits for the NEXT token
  // Per-session Mamba/attention cache. run_step mutates the shared Model's
  // caches in place; this is swapped in/out around prefill/step so sessions
  // stay isolated (same mechanism as the multi-session driver).
  veystrix::engine::nemotron_runtime::SessionCacheState cache;
  int pos = 0;
  int token = 0;                    // last consumed token
  int emitted = 0;
  bool logits_ready = false;
  std::set<int32_t> seen;
  std::optional<mx::random::KeySequence> rng;
  std::string last_digest;
  bool cancel_requested = false;  // per-session; touched only on driver thread
};

}  // namespace

struct NemotronEngineRuntime::Impl {
  std::mutex mu;
  std::condition_variable cv;
  std::thread driver;
  bool stop = false;
  bool has_job = false;
  bool job_done = false;
  std::function<void()> job;
  std::exception_ptr error;
  bool started = false;
  int next_sid = 1;
  // Multi-session: sessions share the read-only Model; each owns its Mamba/
  // attention cache (swapped in/out around each step) — C1 single driver thread.
  std::unordered_map<int, Session> sessions;
  std::atomic<bool> cancel_flag{false};
  std::atomic<unsigned long long> prefill_tokens{0};
  std::atomic<unsigned long long> steps{0};
  std::atomic<unsigned long long> session_tokens{0};
  std::thread::id driver_id;
  int driver_stream_index = -1;

  // Config must outlive ExpertReader: it holds a const Config&.
  Config cfg;
  std::unique_ptr<ExpertReader> er;
  std::unique_ptr<Model> model;
  std::FILE* step_log = nullptr;
  std::string fingerprint;  // snapshot namespace: model dir + profile + kv mode

  ~Impl() {
    {
      std::lock_guard<std::mutex> lk(mu);
      stop = true;
    }
    cv.notify_all();
    if (driver.joinable()) driver.join();
    if (step_log) std::fclose(step_log);
  }

  void start() {
    if (started) return;
    started = true;
    driver = std::thread([this] { loop(); });
  }

  void loop() {
    driver_id = std::this_thread::get_id();
    driver_stream_index = mx::default_stream(mx::Device::gpu).index;
    const int idle_ms = veystrix::engine::membudget::idle_clear_ms();
    bool idle_cleared = false;
    for (;;) {
      std::function<void()> fn;
      {
        std::unique_lock<std::mutex> lk(mu);
        // Idle hygiene (memory management only): if no job arrives within the
        // idle window, drop MLX's allocator cache on this (the driver) thread.
        // idle_ms==0 keeps the original unbounded wait.
        if (idle_ms > 0) {
          while (!cv.wait_for(lk, std::chrono::milliseconds(idle_ms),
                              [this] { return stop || has_job; })) {
            if (!idle_cleared) {
              lk.unlock();
              veystrix::engine::membudget::clear_mlx_cache();
              idle_cleared = true;
              lk.lock();
            }
          }
        } else {
          cv.wait(lk, [this] { return stop || has_job; });
        }
        idle_cleared = false;
        if (stop) break;
        fn = std::move(job);
        has_job = false;
        job_done = false;
      }
      std::exception_ptr err;
      try {
        fn();
      } catch (...) {
        err = std::current_exception();
      }
      {
        std::lock_guard<std::mutex> lk(mu);
        error = err;
        job_done = true;
      }
      cv.notify_all();
    }
  }

  void post(std::function<void()> fn) {
    std::unique_lock<std::mutex> lk(mu);
    if (stop || !started) throw std::runtime_error("nemotron runtime: driver not running");
    cv.wait(lk, [this] { return !has_job; });
    job = std::move(fn);
    has_job = true;
    job_done = false;
    error = nullptr;
    cv.notify_all();
    cv.wait(lk, [this] { return job_done; });
    std::exception_ptr err = error;
    lk.unlock();
    if (err) std::rethrow_exception(err);
  }

  void assert_driver() const {
    if (thread_assert_enabled() && std::this_thread::get_id() != driver_id)
      throw std::runtime_error("nemotron runtime: MLX job ran OFF the driver thread");
  }

  // -------------------------------------------------------------------------
  // Load: verbatim setup order of nemotron_generate's run_generation + main.
  // -------------------------------------------------------------------------
  void load(const std::string& model_dir, const std::string& profile_json) {
    assert_driver();
    if (model) throw std::runtime_error("nemotron runtime: already open");
    if (std::getenv("VEYSTRIX_CPP_MTP"))
      throw std::runtime_error(
          "nemotron runtime: Phase G C ABI supports greedy decode only (MTP/non-greedy off)");
    if (std::getenv("NEMOTRON_LOAD_ONLY"))
      throw std::runtime_error(
          "nemotron runtime: NEMOTRON_LOAD_ONLY is not supported by the live C ABI");

    cppprof::profiler().init();
    membudget::apply_mlx_limits_at_load(profile_json);
    cfg = read_config(config_path_from_profile(profile_json));
    if (cfg.empty())
      throw std::runtime_error("nemotron runtime: empty flat config (run gen_nemotron_config.py first)");
    model = std::make_unique<Model>(load_model(model_dir, cfg));
    er = std::make_unique<ExpertReader>(model_dir, cfg);
    Model& m = *model;
    // A snapshot is only valid for the SAME prefill path. Mixing a chunk-1
    // snapshot into a 2048-chunk session would silently change kernel shapes.
    static const bool chunked = [] {
      const char* e = std::getenv("VEYSTRIX_CPP_NEMOTRON_CHUNKED");
      return e && std::strcmp(e, "1") == 0;
    }();
    fingerprint = model_dir + "|" + profile_json + "|kv=" +
                  (m.use_int8_kv ? "int8" : "bf16") +
                  "|chunked=" + (chunked ? "2048" : "1");

    // Phase 1.1 opt-in NemoPool. Unset / "0" leaves the baseline path unchanged.
    // Copied verbatim from run_generation: the pool is lifecycle glue, not a
    // numeric primitive, and must be assembled before the first run_step.
    // Dynamic-memory contract (memory_budget.hpp): profile keys / env decide the
    // slot count and resident mode. RESIDENT=1 implies the pool (there is no
    // other way to hold every expert) and is FAIL CLOSED: engine open throws if
    // the slot count does not cover every (layer, expert) key or the records
    // do not fit the byte budget / RAM headroom. The gate runs inside the
    // PooledExpertTensors ctor, before any pool buffer is allocated.
    {
      const membudget::MemoryProfile mp = membudget::resolve_memory_profile(profile_json);
      const char* en = std::getenv("VEYSTRIX_CPP_NEMO_POOL");
      if (en && std::strcmp(en, "0") != 0 && std::strcmp(en, "1") != 0)
        throw std::runtime_error("VEYSTRIX_CPP_NEMO_POOL must be 0 or 1");
      const bool pool_on = (en && std::strcmp(en, "1") == 0) || mp.resident;
      if (mp.resident && en && std::strcmp(en, "0") == 0)
        throw std::runtime_error(
            "nemotron: resident=1 conflicts with VEYSTRIX_CPP_NEMO_POOL=0");
      if (pool_on) {
        const char* gm = std::getenv("VEYSTRIX_CPP_NEMO_POOL_GATHER");
        const bool compact = !(gm && std::strcmp(gm, "pooled") == 0);
        if (gm && std::strcmp(gm, "compact") != 0 && std::strcmp(gm, "pooled") != 0)
          throw std::runtime_error(
              "VEYSTRIX_CPP_NEMO_POOL_GATHER must be 'pooled' or 'compact'");
        // Slots: profile nemotron_slots > VEYSTRIX_CPP_NEMO_POOL_SLOTS > 64.
        long long slots = mp.nemo_slots > 0 ? mp.nemo_slots : 64;
        // Resident mode may legitimately exceed the 4096 streaming cap; the
        // ctor gate (not this range check) bounds it by bytes and RAM.
        if (slots < m.top_k || (!mp.resident && slots > 4096) || slots > (1 << 20))
          throw std::runtime_error("VEYSTRIX_CPP_NEMO_POOL_SLOTS out of range");
        const bool nocache = nemo_pool_nocache_from_env();
        NemoPoolGate gate;
        gate.enabled = true;
        gate.resident = mp.resident;
        gate.explicit_budget = mp.pool_bytes;
        gate.active_bytes = mx::get_active_memory();
        m.expert_pool = std::make_unique<PooledExpertTensors>(
            model_dir, cfg, m.n_layers, m.top_k, static_cast<int>(slots), compact,
            nocache, gate);
        std::fprintf(stderr,
                     "[nemo-pool] enabled gather=%s slots=%d nocache=%d resident=%d "
                     "pool_bytes=%zu\n",
                     compact ? "compact" : "pooled", m.expert_pool->pool_slots(),
                     (int)nocache, (int)m.expert_pool->resident(),
                     m.expert_pool->resident_bytes());
        if (m.expert_pool->resident()) {
          m.expert_pool->preload_all();
          std::fprintf(stderr, "[nemo-pool] resident warm-up done: %zu keys\n",
                       m.expert_pool->all_keys());
        }
      }
    }

    cppprof::profiler().begin_step();  // after load, before prefill (CLI parity)
    mem_telemetry("post-load");
    const std::string log = step_log_path();
    if (!log.empty()) {
      step_log = std::fopen(log.c_str(), "a");
      if (!step_log) throw std::runtime_error("nemotron runtime: cannot open step log");
    }
    std::fprintf(stderr, "[phase-g] all weights ready; driver=%s stream=%d\n",
                 "one-thread", driver_stream_index);
  }

  // Swap a session's Mamba/attention cache into the shared Model, and back out.
  void swap_in_cache(Session& s) {
    Model& m = *model;
    auto& sc = s.cache;
    m.mamba_caches = sc.mamba_caches;
    for (std::size_t k = 0; k < m.attn_layers.size() && k < sc.attn_int8.size(); ++k) {
      m.attn_layers[k].int8_cache = sc.attn_int8[k];
      m.attn_layers[k].cache = sc.attn_f32[k];
    }
  }
  void swap_out_cache(Session& s) {
    Model& m = *model;
    auto& sc = s.cache;
    sc.mamba_caches = m.mamba_caches;
    sc.attn_int8.resize(m.attn_layers.size());
    sc.attn_f32.resize(m.attn_layers.size());
    for (std::size_t k = 0; k < m.attn_layers.size(); ++k) {
      sc.attn_int8[k] = m.attn_layers[k].int8_cache;
      sc.attn_f32[k] = m.attn_layers[k].cache;
    }
  }

  // Fresh per-session cache state, identical to the zeros built in load_model.
  void reset_state(Session& s) {
    assert_driver();
    Model& m = *model;
    const int mK = m.spec.mamba_conv_kernel;
    const int mConv = m.spec.mamba_conv_dim;
    const int mH = m.spec.mamba_num_heads;
    const int mDh = m.spec.mamba_head_dim;
    const int mDs = m.spec.mamba_ssm_state_size;
    for (auto& c : m.mamba_caches)
      c = MambaCacheState(mx::zeros(mx::Shape{1, mK - 1, mConv}, mx::bfloat16),
                          mx::zeros(mx::Shape{1, mH, mDh, mDs}, mx::float32));
    for (auto& al : m.attn_layers) {
      al.cache = NemoAttentionCache{};
      al.int8_cache = NemoInt8Cache{};
    }
    // Seed this session's cache store with the fresh zero state just written.
    swap_out_cache(s);
    s.chain.clear();
    s.logits.reset();
    s.pos = 0;
    s.token = 0;
    s.emitted = 0;
    s.logits_ready = false;
    cancel_flag.store(false);
    s.cancel_requested = false;  // a cancel must not leak into the next turn
  }

  // -------------------------------------------------------------------------
  // Sequential single-token prefill: verbatim copy of run_generation's loop.
  // Each run_step advances the cache; the last prompt token's logits produce
  // the first generated token (same as the CLI / Python chunk_tokens=1 path).
  // -------------------------------------------------------------------------
  void prefill_state(Session& s, const std::vector<uint32_t>& tokens,
                     bool append = false) {
    assert_driver();
    if (tokens.empty()) throw std::runtime_error("nemotron runtime: empty prefill");
    if (append && s.chain.empty())
      throw std::runtime_error("nemotron runtime: APPEND before a reset prefill");
    Model& m = *model;
    if (append) s.cancel_requested = false;  // clear a cancel with no step
    if (append && !s.logits_ready) {
      // Commit the last emitted token to the caches before appending, or the
      // continuation starts one position early (Sonnet finding 1).
      s.logits = run_step(*model, *er, s.token);
      mx::eval(*s.logits);
      s.logits_ready = true;
    }
    if (!append) reset_state(s);
    std::vector<int> prompt;
    prompt.reserve(tokens.size());
    for (uint32_t t : tokens) prompt.push_back(static_cast<int>(t));

    cppprof::profiler().begin_step();
    static const bool chunked = [] {
      const char* e = std::getenv("VEYSTRIX_CPP_NEMOTRON_CHUNKED");
      return e && std::strcmp(e, "1") == 0;
    }();
    if (chunked && prompt.size() > 1) {
      // Split into Python-sized 2048-token chunks. A single S=8192 forward
      // retains ~6 GB of lazy activation graphs; per-chunk eval bounds it.
      static const size_t kChunk = 2048;
      for (std::size_t off = 0; off < prompt.size(); off += kChunk) {
        std::vector<int32_t> part(prompt.begin() + off,
                                  prompt.begin() + std::min(off + kChunk, prompt.size()));
        mx::array lg = nemotron_runtime::run_prefill_chunk(model.get(), er.get(), part);
        mx::eval(lg);
        if (off + part.size() == prompt.size()) s.logits = lg;
        // non-final chunk logits release when lg goes out of scope
        mx::clear_cache();      // bound peak anonymous memory across chunks
      }
      if (append) s.chain.insert(s.chain.end(), prompt.begin(), prompt.end());
      else s.chain = prompt;
      s.pos += static_cast<int>(prompt.size());
      s.token = prompt.back();
      s.logits_ready = true;
      s.emitted = 0;
      s.seen.clear();
      prefill_tokens.fetch_add(prompt.size());
      session_tokens.fetch_add(prompt.size());
      return;
    }
    std::optional<mx::array> logits;
    for (std::size_t i = 0; i < prompt.size(); ++i) {
      logits = run_step(m, *er, prompt[i]);
      ++s.pos;
      if ((i + 1) % 4 == 0 || i + 1 == prompt.size()) mem_telemetry("prefill-step");
    }
    if (!logits) throw std::runtime_error("nemotron runtime: no prefill logits");
    // Materialize NOW: the graph reads shared expert-pool slot memory, which a
    // later session's APPEND may overwrite before this session is sampled.
    mx::eval(*logits);
    s.logits = *logits;
    if (append) s.chain.insert(s.chain.end(), prompt.begin(), prompt.end());
    else s.chain = prompt;
    s.token = prompt.back();
    s.logits_ready = true;
    s.emitted = 0;
    s.seen.clear();  // Python makes a new Sampler per generate() call
    prefill_tokens.fetch_add(prompt.size());
    session_tokens.fetch_add(prompt.size());
  }

  // Sampling + bookkeeping shared by the solo step and the pipelined batch.
  // Operates on s.logits (already computed) so both paths use byte-identical ops.
  uint32_t sample_from_logits(Session& s, const std::string& sampler_json) {
    SamplerSettings settings = parse_sampler_json(sampler_json);
    auto& pf = cppprof::profiler();
    auto tsamp = pf.now();
    std::optional<mx::array> next;
    if (settings.force_token >= 0) {
      std::vector<uint32_t> forced{static_cast<uint32_t>(settings.force_token)};
      next = mx::array(forced.begin(), mx::Shape{1}, mx::uint32);
    } else if (settings.greedy) {
      next = sample_greedy(*s.logits);
    } else {
      SamplingParams params;
      params.temperature = settings.temperature;
      params.top_p = settings.top_p;
      params.top_k = settings.top_k;
      params.min_p = settings.min_p;
      params.presence_penalty = settings.presence_penalty;
      std::vector<int32_t> seen(s.seen.begin(), s.seen.end());
      std::optional<mx::array> key;
      if (s.rng) key = s.rng->next();
      next = sample_non_greedy(*s.logits, params, seen, key);
    }
    mx::eval(*next);
    pf.add_sample(cppprof::ms_since(tsamp));
    int next_id = static_cast<int>(next->data<uint32_t>()[0]);
    if (!settings.greedy && settings.presence_penalty > 0.0f)
      s.seen.insert(static_cast<int32_t>(next_id));
    std::string digest;
    if (!pf.no_digest) {
      auto td = pf.now();
      digest = digest_array(*s.logits);
      s.last_digest = digest;
      pf.add_digest(cppprof::ms_since(td));
    }
    if (step_log)
      std::fprintf(step_log, "step=%d in=%d out=%d sha256=%s\n",
                   s.emitted, s.token, next_id, digest.c_str());
    s.token = next_id;
    s.chain.push_back(next_id);
    s.logits_ready = false;
    ++s.emitted;
    steps.fetch_add(1);
    session_tokens.fetch_add(1);
    return static_cast<uint32_t>(next_id);
  }

  uint32_t step_state(Session& s, const std::string& sampler_json) {
    assert_driver();
    if (s.cancel_requested && (s.cancel_requested = false, true)) throw CancelledError();
    if (s.chain.empty()) throw std::runtime_error("nemotron runtime: step before prefill");

    // CLI opens a fresh profiler window only for step>0; step 0 reuses the
    // window opened before prefill. Profiler is diagnostics-only, but mirror
    // the ordering exactly.
    if (s.emitted > 0) cppprof::profiler().begin_step();
    if (!s.logits_ready) {
      auto& pf = cppprof::profiler();
      auto tb = pf.now();
      s.logits = run_step(*model, *er, s.token);
      pf.add_build(cppprof::ms_since(tb));
      {
        auto te = pf.now();
        mx::eval(*s.logits);  // materialize before pool slots can be reused
        pf.add_final_eval(cppprof::ms_since(te));
      }
      pf.end_step_summary(cppprof::ms_since(tb), "nemotron");
      mem_telemetry("decode-step");
      s.logits_ready = true;
    }

    return sample_from_logits(s, sampler_json);
  }
};

NemotronEngineRuntime::NemotronEngineRuntime() : impl_(std::make_unique<Impl>()) {}

NemotronEngineRuntime::~NemotronEngineRuntime() = default;

void NemotronEngineRuntime::open(const std::string& model_dir,
                                 const std::string& profile_json) {
  impl_->start();
  impl_->post([this, model_dir, profile_json] { impl_->load(model_dir, profile_json); });
}

int NemotronEngineRuntime::new_session() {
  int sid = 0;
  impl_->post([this, &sid] {
    if (!impl_->model)
      throw std::runtime_error("nemotron runtime: open() before new_session()");
    sid = impl_->next_sid++;
    Session s;
    s.id = sid;
    impl_->reset_state(s);
    impl_->sessions.emplace(sid, std::move(s));
  });
  return sid;
}

void NemotronEngineRuntime::close_session(int sid) {
  impl_->post([this, sid] {
    impl_->sessions.erase(sid);
    // Session caches are the big transient buffers: hand them back now.
    veystrix::engine::membudget::clear_mlx_cache();
  });
}

void NemotronEngineRuntime::prefill(int sid, const std::vector<uint32_t>& tokens) {
  impl_->post([this, sid, tokens] {
    auto it = impl_->sessions.find(sid);
    if (it == impl_->sessions.end())
      throw std::runtime_error("nemotron runtime: bad session id");
    impl_->swap_in_cache(it->second);
    impl_->prefill_state(it->second, tokens);
    impl_->swap_out_cache(it->second);
  });
}

void NemotronEngineRuntime::prefill_append(int sid, const std::vector<uint32_t>& tokens) {
  impl_->post([this, sid, tokens] {
    auto it = impl_->sessions.find(sid);
    if (it == impl_->sessions.end())
      throw std::runtime_error("nemotron runtime: bad session id");
    // Must swap THIS session's mutable cache into the shared Model exactly as
    // prefill()/step() do. Without this, APPEND runs on the previous session's
    // cache — the multi-session fork/append path silently diverged.
    impl_->swap_in_cache(it->second);
    impl_->prefill_state(it->second, tokens, /*append=*/true);
    impl_->swap_out_cache(it->second);
  });
}

uint32_t NemotronEngineRuntime::step(int sid, const std::string& sampler_json) {
  uint32_t out = 0;
  impl_->post([this, sid, &sampler_json, &out] {
    auto it = impl_->sessions.find(sid);
    if (it == impl_->sessions.end())
      throw std::runtime_error("nemotron runtime: bad session id");
    impl_->swap_in_cache(it->second);
    out = impl_->step_state(it->second, sampler_json);
    impl_->swap_out_cache(it->second);
  });
  return out;
}

std::vector<uint32_t> NemotronEngineRuntime::step_batch(
    const std::vector<int>& sids, const std::vector<std::string>& sampler_jsons) {
  std::vector<uint32_t> out(sids.size(), 0);
  if (sids.size() != sampler_jsons.size())
    throw std::runtime_error("nemotron runtime: step_batch size mismatch");
  const bool overlap = [] {
    const char* e = std::getenv("VEYSTRIX_CPP_OVERLAP");
    return e && std::strcmp(e, "1") == 0;
  }();
  impl_->post([this, &sids, &sampler_jsons, &out, overlap] {
    if (overlap && sids.size() >= 2 && impl_->model && impl_->model->expert_pool) {
      // Match step_state exactly: a session with logits_ready (just prefilled)
      // samples from its cached logits; only the others run a decode step.
      std::vector<std::size_t> need;
      for (std::size_t i = 0; i < sids.size(); ++i) {
        auto it = impl_->sessions.find(sids[i]);
        if (it == impl_->sessions.end())
          throw std::runtime_error("nemotron runtime: bad session id in batch");
        if (it->second.logits_ready) {
          out[i] = impl_->sample_from_logits(it->second, sampler_jsons[i]);
        } else {
          need.push_back(i);
        }
      }
      if (!need.empty()) {
        std::vector<int32_t> toks;
        for (auto i : need) toks.push_back(static_cast<int32_t>(
            impl_->sessions.find(sids[i])->second.token));
        std::vector<int> batch_sids;
        for (auto i : need) batch_sids.push_back(sids[i]);
        auto swap_in = [this, &batch_sids](int j) {
          impl_->swap_in_cache(impl_->sessions.find(batch_sids[j])->second);
        };
        auto swap_out = [this, &batch_sids](int j) {
          impl_->swap_out_cache(impl_->sessions.find(batch_sids[j])->second);
        };
        std::vector<mx::array> lg = nemotron_runtime::run_step_pipelined(
            impl_->model.get(), impl_->er.get(), toks, swap_in, swap_out);
        for (std::size_t j = 0; j < need.size(); ++j) {
          Session& s = impl_->sessions.find(batch_sids[j])->second;
          s.logits = lg[j];
          s.logits_ready = true;
          out[need[j]] = impl_->sample_from_logits(s, sampler_jsons[need[j]]);
        }
      }
      return;
    }
    for (size_t i = 0; i < sids.size(); ++i) {
      auto it = impl_->sessions.find(sids[i]);
      if (it == impl_->sessions.end())
        throw std::runtime_error("nemotron runtime: bad session id in batch");
      impl_->swap_in_cache(it->second);
      out[i] = impl_->step_state(it->second, sampler_jsons[i]);
      impl_->swap_out_cache(it->second);
    }
  });
  return out;
}

void NemotronEngineRuntime::fork_session(int src_sid, int dst_sid) {
  impl_->post([this, src_sid, dst_sid] {
    auto src = impl_->sessions.find(src_sid);
    auto dst = impl_->sessions.find(dst_sid);
    if (src == impl_->sessions.end() || dst == impl_->sessions.end())
      throw std::runtime_error("nemotron runtime: bad session id in fork");
    if (src == dst)
      throw std::runtime_error("nemotron runtime: fork needs two distinct sessions");
    dst->second = src->second;
    dst->second.id = dst_sid;
  });
}

void NemotronEngineRuntime::seed(int sid, uint64_t seed) {
  impl_->post([this, sid, seed] {
    auto it = impl_->sessions.find(sid);
    if (it == impl_->sessions.end())
      throw std::runtime_error("nemotron runtime: bad session id");
    it->second.rng.emplace(seed);
  });
}

std::string NemotronEngineRuntime::last_digest(int sid) {
  std::string out;
  impl_->post([this, sid, &out] {
    auto it = impl_->sessions.find(sid);
    if (it == impl_->sessions.end())
      throw std::runtime_error("nemotron runtime: bad session id");
    out = it->second.last_digest;
  });
  return out;
}

void NemotronEngineRuntime::cancel_session(int sid) {
  impl_->post([this, sid] {
    auto it = impl_->sessions.find(sid);
    if (it != impl_->sessions.end()) it->second.cancel_requested = true;
  });
}

void NemotronEngineRuntime::cancel() { impl_->cancel_flag.store(true); }

bool NemotronEngineRuntime::consume_cancel() { return impl_->cancel_flag.exchange(false); }

void NemotronEngineRuntime::snapshot_save(int sid, const std::string& dir) {
  impl_->post([this, sid, dir] {
    auto it = impl_->sessions.find(sid);
    if (it == impl_->sessions.end())
      throw std::runtime_error("nemotron runtime: bad session id");
    NemoSnapshot snap = nemo_snapshot_capture(it->second.cache, it->second.chain,
                                              impl_->fingerprint,
                                              impl_->model->use_int8_kv);
    nemo_snapshot_save(snap, dir);
  });
}

void NemotronEngineRuntime::snapshot_restore(int sid, const std::string& dir) {
  impl_->post([this, sid, dir] {
    auto it = impl_->sessions.find(sid);
    if (it == impl_->sessions.end())
      throw std::runtime_error("nemotron runtime: bad session id");
    Session& s = it->second;
      std::vector<int32_t> tokens;
    {
      std::ifstream meta(dir + "/meta.txt");
      if (!meta) throw std::runtime_error("nemotron runtime: snapshot meta missing");
      std::string key;
      std::size_t nt = 0;
      // Skip: chain_key <hex>, then read n_tokens and the fixed header keys.
      if (!(meta >> key >> key) || !(meta >> key >> nt))
        throw std::runtime_error("nemotron runtime: bad snapshot meta");
      std::string ignore;
      std::size_t extra = 0;
      meta >> ignore >> extra >> ignore >> extra >> ignore >> extra;  // n_mamba, n_attn, int8
      tokens.resize(nt);
      for (std::size_t i = 0; i < nt; ++i) {
        if (!(meta >> key >> tokens[i]) || key != "tok")
          throw std::runtime_error("nemotron runtime: bad snapshot tokens");
      }
    }
    // Strict-prefix rule: at least one new token must still be fed.
    // A fresh session has an empty chain: the snapshot's own token ids become
    // its history. On an existing session the snapshot must be a prefix.
    if (!s.chain.empty()) {
      if (tokens.size() > s.chain.size())
        throw std::runtime_error("nemotron runtime: snapshot tokens are not a strict prefix");
      if (!std::equal(tokens.begin(), tokens.end(), s.chain.begin()))
        throw std::runtime_error("nemotron runtime: snapshot token mismatch");
    }
      s.cache = nemo_snapshot_restore(dir, impl_->fingerprint, tokens);
      s.chain = tokens;
    s.pos = static_cast<int>(tokens.size());
    s.token = tokens.back();
    s.logits_ready = false;  // Nemotron does NOT store logits: caller must feed 1 token
    s.emitted = 0;
    s.seen.clear();
    impl_->cancel_flag.store(false);
  });
}

std::string NemotronEngineRuntime::driver_stats() const {
  std::ostringstream out;
  out << ",\"engine\":\"nemotron\""
      << ",\"driver_stream\":" << impl_->driver_stream_index;
  return out.str();
}

}  // namespace veystrix::engine::capi_live
