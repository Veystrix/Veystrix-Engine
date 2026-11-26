#include "generic/generic_engine_runtime.hpp"

#include <mlx/mlx.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#ifdef __APPLE__
#include <compression.h>
#endif

#include "cpp_profile.hpp"
#include "engine.hpp"            // sample_greedy
#include "generic/generic_model.hpp"
#include "generic/inkling_dspark.hpp"
#include "generic/json_lite.hpp"
#include "generic/kv_snapshot_store.hpp"
#include "row_fetch.hpp"
#include "generic/sha256.hpp"
#include "gpu_keepwarm.hpp"
#include "memory_budget.hpp"
#include "cancelled_error.hpp"
#include "sampler_json.hpp"
#include "sampling.hpp"

namespace mx = mlx::core;
using namespace veystrix::engine;
using namespace veystrix::engine::generic;

namespace veystrix::engine::capi_live {

namespace {

const std::set<std::string>& generic_types() {
  static const std::set<std::string> t = {
      "qwen3_5_moe", "qwen3_5_moe_text", "inkling2_loop", "inkling2_loop_text"};
  return t;
}

std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return "";
  std::ostringstream s;
  s << f.rdbuf();
  return s.str();
}

int env_int(const char* name, int def) {
  const char* e = std::getenv(name);
  if (!e || !*e) return def;
  return std::atoi(e);
}

std::string digest_logits(const mx::array& logits) {
  mx::array f = mx::reshape(mx::astype(logits, mx::float32), {-1});
  mx::eval(f);
  return sha256_hex(f.data<float>(), static_cast<size_t>(f.size()) * sizeof(float));
}

struct Session {
  int id = 0;
  SessionState state;
  DSparkState dspark_state;
  std::deque<int32_t> spec_queue;   // verified speculative tokens waiting to be returned
  int dspark_miss_streak = 0;       // consecutive s1 skips for adaptive cooldown
  int dspark_skip_left = 0;         // remaining S=1 steps before next DSpark probe
  std::vector<int32_t> chain;       // tokens whose effect is in `state` + pending
  std::optional<mx::array> logits;  // next-token logits (when logits_ready)
  int token = 0;                    // last token (pending when !logits_ready)
  bool logits_ready = false;
  int emitted = 0;
  std::set<int32_t> seen;
  std::optional<mx::random::KeySequence> rng;
  std::string last_digest;
  bool cancel_requested = false;
};

}  // namespace

bool generic_model_type_supported(const std::string& model_dir, std::string* model_type) {
  const std::string text = read_file(model_dir + "/config.json");
  if (text.empty()) return false;
  try {
    Json cfg = Json::parse(text);
    std::string mt = cfg.get_str("model_type", "");
    if (const Json* tc = cfg.find("text_config"); tc && tc->is_object() && mt.empty())
      mt = tc->get_str("model_type", "");
    if (model_type) *model_type = mt;
    return generic_types().count(mt) > 0;
  } catch (...) {
    return false;
  }
}

struct GenericEngineRuntime::Impl {
  std::mutex mu;
  std::condition_variable cv;
  std::thread driver;
  bool stop = false, has_job = false, job_done = false, started = false;
  std::function<void()> job;
  std::exception_ptr error;
  std::thread::id driver_id;
  int driver_stream_index = -1;

  std::unordered_map<int, Session> sessions;
  int next_sid = 1;
  std::atomic<bool> cancel_flag{false};
  std::atomic<unsigned long long> prefill_tokens{0}, steps{0};
  std::atomic<unsigned long long> spec_rounds{0}, spec_drafted{0}, spec_accepted{0};

  std::unique_ptr<GenericModel> model;
  std::string model_dir, fingerprint, kv_desc;
  std::string speed_desc;  // active per-model speed knobs ("name=value,...") and where each came from
  int prefill_chunk = 2048;
  int dspark_draft = 3;
  bool digests = true;
  double load_seconds = 0;
  unsigned long long expert_bytes_total = 0;

  ~Impl() {
    {
      std::lock_guard<std::mutex> lk(mu);
      stop = true;
    }
    cv.notify_all();
    if (driver.joinable()) driver.join();
  }

  void start() {
    if (started) return;
    started = true;
    driver = std::thread([this] { loop(); });
  }

  void loop() {
    driver_id = std::this_thread::get_id();
    driver_stream_index = mx::default_stream(mx::Device::gpu).index;
    const int idle_ms = membudget::idle_clear_ms();
    bool idle_cleared = false;
    for (;;) {
      std::function<void()> fn;
      {
        std::unique_lock<std::mutex> lk(mu);
        if (idle_ms > 0) {
          while (!cv.wait_for(lk, std::chrono::milliseconds(idle_ms), [this] { return stop || has_job; })) {
            if (!idle_cleared) {
              lk.unlock();
              membudget::clear_mlx_cache();
              rowfetch::evict_cached_files();
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
    if (stop || !started) throw std::runtime_error("generic runtime: driver not running");
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

  Session& get(int sid) {
    auto it = sessions.find(sid);
    if (it == sessions.end()) throw std::runtime_error("generic runtime: bad session id");
    return it->second;
  }

  // Per-model speed knobs live in the recipe ("speed": {...}), not in scattered env defaults. Each knob is
  // exact (same bytes, same kernels) and maps to the env switch the blocks read; an env value the operator
  // set wins (A/B screening), the recipe value is the model's measured default, the code default applies
  // otherwise. Applied before the model is built: the blocks read their switches once.
  void apply_speed_knobs(const Json& recipe_json) {
    static const std::pair<const char*, const char*> knobs[] = {
        {"rows", "VEYSTRIX_GENERIC_ROWS"},           {"pipe", "VEYSTRIX_GENERIC_PIPE"},
        {"spec_fetch", "VEYSTRIX_GENERIC_SPEC_FETCH"}, {"spec_top", "VEYSTRIX_GENERIC_SPEC_TOP"},
        {"prefill_pipe", "VEYSTRIX_GENERIC_PREFILL_PIPE"}, {"prefill_advise", "VEYSTRIX_GENERIC_PREFILL_ADVISE"},
        {"read_lanes", "VEYSTRIX_GENERIC_READ_LANES"}, {"keepwarm", "VEYSTRIX_CPP_KEEPWARM"},
        {"tensor_rows", "VEYSTRIX_GENERIC_ROW_ORDER"}, {"gate", "VEYSTRIX_GENERIC_GATE"},
        {"gate_tail", "VEYSTRIX_CPP_GATE_TAIL"},
    };
    const Json* sp = recipe_json.find("speed");
    speed_desc.clear();
    for (const auto& [key, env] : knobs) {
      const Json* v = (sp && sp->is_object()) ? sp->find(key) : nullptr;
      const char* set = std::getenv(env);
      std::string value, from;
      if (set && *set) {
        value = set;
        from = "env";
      } else if (v && !v->is_null()) {
        value = v->is_bool() ? (v->boolean(key) ? "1" : "0") : std::to_string(v->i64(key));
        ::setenv(env, value.c_str(), 0);
        from = "recipe";
      } else {
        continue;
      }
      speed_desc += (speed_desc.empty() ? "" : ",") + std::string(key) + "=" + value + "(" + from + ")";
    }
    if (sp && sp->is_object())
      for (const auto& kv : sp->obj("speed")) {
        bool known = false;
        for (const auto& k : knobs) known = known || kv.first == k.first;
        if (!known && kv.first != "note")
          throw std::runtime_error("generic runtime: unknown recipe speed knob '" + kv.first + "'");
      }
  }

  void load(const std::string& dir, const std::string& profile_json) {
    auto t0 = std::chrono::steady_clock::now();
    if (model) throw std::runtime_error("generic runtime: already open");
    cppprof::profiler().init();
    // MLX free-buffer cache: capped at 256 MiB unless VEYSTRIX_CPP_MLX_CACHE_BYTES says otherwise. A
    // streamed prefill frees ~0.4 GB of expert rows per layer; an uncapped cache keeps all of it and
    // the page cache (the expert cache) loses that much (the first real Ornith run hit the RAM floor).
    if (membudget::apply_mlx_limits_at_load(profile_json) == 0) mx::set_cache_limit(membudget::tier_mlx_cache_bytes());
    Json prof = (!profile_json.empty() && profile_json[0] == '{') ? Json::parse(profile_json) : Json();
    std::string recipe = prof.get_str("recipe", "");
    if (recipe.empty()) {
      const char* e = std::getenv("VEYSTRIX_GENERIC_RECIPE");
      if (e) recipe = e;
    }
    if (recipe.empty())
      throw std::runtime_error(
          "generic runtime: no recipe (pass profile {\"recipe\": <path>}; generate one with "
          "`veystrix models inspect <ckpt> --emit-recipe <path>`)");
    // KV policy: profile > recipe cache.kv > bf16.
    std::string mode = "bf16";
    int64_t window = -2;
    int prefix = -1, hk = -1, hv = -1;
    Json recipe_json;
    {
      const std::string text = recipe[0] == '{' ? recipe : read_file(recipe);
      if (text.empty()) throw std::runtime_error("generic runtime: cannot read recipe " + recipe);
      recipe_json = Json::parse(text);
      if (const Json* c = recipe_json.find("cache"))
        if (const Json* kv = c->find("kv"); kv && kv->is_object()) {
          mode = kv->get_str("mode", mode);
          window = kv->get_i64("window", window);
          prefix = static_cast<int>(kv->get_i64("prefix", prefix));
          hk = static_cast<int>(kv->get_i64("hist_k_bits", hk));
          hv = static_cast<int>(kv->get_i64("hist_v_bits", hv));
        }
    }
    apply_speed_knobs(recipe_json);
    mode = prof.get_str("kv", mode);
    if (mode.empty()) mode = "bf16";
    window = prof.get_i64("kv_window", window);
    prefix = static_cast<int>(prof.get_i64("kv_prefix", prefix));
    hk = static_cast<int>(prof.get_i64("kv_hist_k_bits", hk));
    hv = static_cast<int>(prof.get_i64("kv_hist_v_bits", hv));
    ModelOptions opt;
    opt.kv = KVPolicy::from_mode(mode, window, prefix, hk, hv);
    const membudget::MemoryProfile mp = membudget::resolve_memory_profile(profile_json);
    opt.resident_experts = mp.resident;
    const int env_ds = env_int("VEYSTRIX_INKLING_DSPARK", -1);
    // DSpark is enabled by default for Inkling 2 (Q4 drafter + confidence-gated dynamic stopping
    // + adaptive cooldown delivers +18% decode on 16 GB streamed MoE and ~3.5x on 24 GB+ resident).
    // Set VEYSTRIX_INKLING_DSPARK=0 or profile {"dspark": false} to disable.
    opt.enable_dspark = (env_ds >= 0) ? (env_ds > 0) : prof.get_bool("dspark", true);
    const bool is_inkling_streamed = (!opt.resident_experts && recipe_json.get_str("model_type", "").find("inkling") != std::string::npos);
    if (is_inkling_streamed) {
      // Prevent macOS aggressive file read-ahead from bloating RAM and pushing swap > 3 GB.
      ::setenv("VEYSTRIX_CPP_RDAHEAD", "0", 0);
    }
    const bool explicit_draft = prof.find("dspark_draft") != nullptr || env_int("VEYSTRIX_INKLING_DSPARK_DRAFT", -1) >= 0;
    const int default_draft = opt.resident_experts ? 7 : 5;
    dspark_draft = std::clamp(
        static_cast<int>(prof.get_i64("dspark_draft", env_int("VEYSTRIX_INKLING_DSPARK_DRAFT", default_draft))), 1,
        7);
    const int default_chunk = 2048;
    prefill_chunk = static_cast<int>(prof.get_i64("prefill_chunk", env_int("VEYSTRIX_GENERIC_PREFILL_CHUNK", default_chunk)));
    if (prefill_chunk < 1) throw std::runtime_error("generic runtime: prefill_chunk must be >= 1");
    digests = !cppprof::profiler().no_digest;
    model = std::make_unique<GenericModel>(dir, recipe_json.dump(), opt);
    if (!explicit_draft && model->resident_experts()) dspark_draft = 7;
    model_dir = dir;
    kv_desc = opt.kv.describe();
    fingerprint = dir + "|recipe=" + model->recipe_digest() + "|kv=" + kv_desc +
                  "|chunk=" + std::to_string(prefill_chunk) +
                  "|resident=" + (opt.resident_experts ? "1" : "0");
    load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::string ds_info;
    if (model->dspark()) {
      char dbuf[64];
      std::snprintf(dbuf, sizeof dbuf, " dspark=on(draft=%d,%.2fGiB)", dspark_draft,
                    model->dspark()->weight_bytes() / 1073741824.0);
      ds_info = dbuf;
    }
    std::fprintf(stderr,
                 "[generic] %s ready: layers=%d resident=%.2f GiB streamed=%.2f GiB kv=%s chunk=%d "
                 "speed=[%s]%s (%.1fs)\n",
                 model->model_type().c_str(), model->n_layers(),
                 model->resident_bytes() / 1073741824.0, model->streamed_bytes() / 1073741824.0,
                 kv_desc.c_str(), prefill_chunk, speed_desc.c_str(), ds_info.c_str(), load_seconds);
  }

  void reset_session(Session& s) {
    model->reset(s.state);
    if (model->dspark()) model->dspark()->reset(s.dspark_state);
    s.spec_queue.clear();
    s.dspark_miss_streak = 0;
    s.dspark_skip_left = 0;
    s.chain.clear();
    s.logits.reset();
    s.token = 0;
    s.logits_ready = false;
    s.emitted = 0;
    s.seen.clear();
    s.cancel_requested = false;
    cancel_flag.store(false);
  }

  void eval_state(Session& s, const mx::array& logits) {
    std::vector<mx::array> outs{logits};
    for (auto& ls : s.state) {
      for (auto& a : ls.arrays) outs.push_back(a);
      for (auto& a : ls.pre) outs.push_back(a);
      if (ls.kv) ls.kv->append_state_arrays(outs);
    }
    mx::eval(outs);
  }

  // Feed `toks` in dynamically sized chunks based on live RAM headroom and pressure.
  mx::array run_prefill(Session& s, const std::vector<int32_t>& toks) {
    const size_t phys = membudget::physical_ram_bytes();
    const bool tight_16g = phys < (size_t{20} << 30);
    const size_t normal_cache = membudget::env_bytes("VEYSTRIX_CPP_MLX_CACHE_BYTES")
                                    ? membudget::env_bytes("VEYSTRIX_CPP_MLX_CACHE_BYTES")
                                    : membudget::tier_mlx_cache_bytes();

    if (s.chain.empty() && toks.size() >= 512 && model->is_looped() && !model->resident_experts()) {
      if (tight_16g) {
        mx::set_cache_limit(0);
        mx::clear_cache();
      }
      ForwardStats fs;
      VerifyResult vr = model->prefill_stage_major(toks, s.state, &fs, [this, &s]() {
        return cancel_flag.load() || s.cancel_requested;
      });
      if (model->dspark() && vr.features) {
        model->dspark()->commit_context(s.dspark_state, *vr.features);
      }
      expert_bytes_total += static_cast<unsigned long long>(fs.expert_bytes);
      mx::set_cache_limit(normal_cache);
      return vr.logits;
    }

    std::optional<mx::array> last;
    for (size_t off = 0; off < toks.size();) {
      const int pressure = membudget::memory_pressure_level();
      const size_t avail = membudget::available_ram_bytes();
      const size_t active = mx::get_active_memory();
      const int chunk = membudget::dynamic_prefill_chunk(prefill_chunk, phys, avail, active, pressure);
      if (tight_16g) {
        mx::set_cache_limit(0);
        mx::clear_cache();
      } else if (pressure >= 2) {
        mx::set_cache_limit(size_t{64} << 20);
        mx::clear_cache();
      } else {
        mx::set_cache_limit(normal_cache);
      }
      const size_t take = std::min(toks.size() - off, static_cast<size_t>(chunk));
      std::vector<int32_t> part(toks.begin() + off, toks.begin() + off + take);
      off += take;
      // On 16 GB Macs, skip double-buffered 547 MiB prefill prefetch so two streamed layers
      // are never resident in Metal scratch simultaneously during prefill.
      const bool allow_prefetch = (part.size() >= 32) && !tight_16g;
      ForwardStats fs;
      mx::array lg0 = mx::array(0.0f);
      if (model->dspark()) {
        VerifyResult vr = model->forward_with_features(part, s.state, /*all_logits=*/false,
                                                       /*save_verify_history=*/false, &fs,
                                                       /*prefetch_next_layer=*/allow_prefetch);
        lg0 = vr.logits;
        if (vr.features) model->dspark()->commit_context(s.dspark_state, *vr.features);
      } else {
        std::vector<mx::array> lg = model->forward({part}, {&s.state}, &fs, /*prefetch_next_layer=*/allow_prefetch);
        lg0 = lg[0];
      }
      expert_bytes_total += static_cast<unsigned long long>(fs.expert_bytes);
      // Bound the lazy graph: materialise logits and every state + KV array.
      std::vector<mx::array> outs{lg0};
      for (auto& ls : s.state) {
        for (auto& a : ls.arrays) outs.push_back(a);
        for (auto& a : ls.pre) outs.push_back(a);
        if (ls.kv) ls.kv->append_state_arrays(outs);
      }
      mx::eval(outs);
      last = lg0;
      if (tight_16g || off < toks.size() || part.size() >= 256) mx::clear_cache();
    }
    mx::set_cache_limit(normal_cache);
    return *last;
  }

  void commit_pending(Session& s) {
    if (s.logits_ready || s.chain.empty()) return;
    s.spec_queue.clear();
    if (model->dspark()) {
      VerifyResult vr = model->forward_with_features({s.token}, s.state, /*all_logits=*/false,
                                                     /*save_verify_history=*/false);
      if (vr.features) model->dspark()->commit_context(s.dspark_state, *vr.features);
      mx::eval(vr.logits);
      s.logits = vr.logits;
    } else {
      std::vector<mx::array> lg = model->forward({{s.token}}, {&s.state});
      mx::eval(lg[0]);
      s.logits = lg[0];
    }
    s.logits_ready = true;
  }

  void prefill_state(Session& s, const std::vector<uint32_t>& tokens, bool append) {
    if (tokens.empty()) throw std::runtime_error("generic runtime: empty prefill");
    if (append && s.chain.empty()) throw std::runtime_error("generic runtime: APPEND before a reset prefill");
    std::vector<int32_t> toks(tokens.begin(), tokens.end());
    for (int32_t t : toks)
      if (t < 0 || t >= model->vocab()) throw std::runtime_error("generic runtime: token id out of range");
    if (append) {
      s.cancel_requested = false;
      commit_pending(s);  // the last emitted token enters the caches first
      s.spec_queue.clear();
      if (!s.state.empty() && s.state[0].tokens != static_cast<int64_t>(s.chain.size())) {
        if (!model->truncate(s.state, static_cast<int64_t>(s.chain.size()))) {
          throw std::runtime_error("generic runtime: state alignment truncate failed");
        }
        if (model->dspark()) {
          model->dspark()->truncate(s.dspark_state, static_cast<int64_t>(s.chain.size()));
        }
      }
    } else {
      reset_session(s);
    }
    cppprof::profiler().begin_step();
    auto t_prefill = cppprof::profiler().now();
    mx::array lg = run_prefill(s, toks);
    if (toks.size() >= 32) {
      char tag[64];
      std::snprintf(tag, sizeof tag, "generic-prefill %zu tok", toks.size());
      cppprof::profiler().print_window(tag, cppprof::ms_since(t_prefill));
    }
    s.logits = lg;
    s.chain.insert(s.chain.end(), toks.begin(), toks.end());
    s.token = toks.back();
    s.logits_ready = true;
    s.emitted = 0;
    s.seen.clear();
    prefill_tokens.fetch_add(toks.size());
  }

  uint32_t sample(Session& s, const std::string& sampler_json) {
    SamplerSettings settings = parse_sampler_json(sampler_json);
    auto& pf = cppprof::profiler();
    auto ts = pf.now();
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
    int probed = -1;
    std::string probe_lp;
    if (settings.probe) {
      mx::array g = sample_greedy(*s.logits);
      mx::eval(g, *next);
      probed = static_cast<int>(g.data<uint32_t>()[0]);
      if (settings.logprob) {
        mx::array lf = mx::reshape(mx::astype(*s.logits, mx::float32), mx::Shape{-1});
        mx::array ls = mx::subtract(lf, mx::logsumexp(lf, 0, false));
        mx::array pick = mx::take(ls, mx::array({settings.force_token, probed}, mx::Shape{2}, mx::int32), 0);
        mx::eval(pick);
        const float* v = pick.data<float>();
        char buf[96];
        std::snprintf(buf, sizeof buf, "lp=%.6f top=%.6f", static_cast<double>(v[0]), static_cast<double>(v[1]));
        probe_lp = buf;
      }
    }
    mx::eval(*next);
    pf.add_sample(cppprof::ms_since(ts));
    const int id = static_cast<int>(next->data<uint32_t>()[0]);
    if (!settings.greedy && settings.presence_penalty > 0.0f) s.seen.insert(id);
    if (digests) {
      auto td = pf.now();
      s.last_digest = digest_logits(*s.logits);
      pf.add_digest(cppprof::ms_since(td));
    }
    if (!probe_lp.empty()) s.last_digest = probe_lp;
    s.token = id;
    s.chain.push_back(id);
    s.logits_ready = false;
    ++s.emitted;
    steps.fetch_add(1);
    return static_cast<uint32_t>(probed >= 0 ? probed : id);
  }

  void check_cancel(Session& s) {
    if (s.cancel_requested) {
      s.cancel_requested = false;
      throw CancelledError();
    }
    if (s.chain.empty()) throw std::runtime_error("generic runtime: step before prefill");
  }

  // Decode the sessions that need a forward as ONE lane batch (shared expert
  // reads), then sample each.
  std::vector<uint32_t> step_many(const std::vector<int>& sids, const std::vector<std::string>& samplers) {
    std::vector<Session*> ss;
    for (int sid : sids) {
      Session& s = get(sid);
      check_cancel(s);
      ss.push_back(&s);
    }
    std::set<Session*> uniq(ss.begin(), ss.end());
    if (uniq.size() != ss.size()) throw std::runtime_error("generic runtime: duplicate session in batch");
    if (ss.size() == 1 && model->dspark() != nullptr) {
      Session& s = *ss[0];
      SamplerSettings settings = parse_sampler_json(samplers[0]);
      if (settings.force_token < 0 && !settings.probe) {
        if (!s.spec_queue.empty()) {
          const int32_t tok = s.spec_queue.front();
          s.spec_queue.pop_front();
          s.chain.push_back(tok);
          if (!settings.greedy && settings.presence_penalty > 0.0f) s.seen.insert(tok);
          ++s.emitted;
          steps.fetch_add(1);
          return {static_cast<uint32_t>(tok)};
        }
        if (!s.logits_ready) {
          const Inkling2DSpark* ds = model->dspark();
          static int decode_pressure = 1;
          if ((s.emitted & 15) == 0) {
            decode_pressure = membudget::memory_pressure_level();
            if (decode_pressure >= 4) {
              mx::set_cache_limit(size_t{64} << 20);
              mx::clear_cache();
              rowfetch::evict_cached_files();
            } else {
              const size_t c = membudget::env_bytes("VEYSTRIX_CPP_MLX_CACHE_BYTES")
                                   ? membudget::env_bytes("VEYSTRIX_CPP_MLX_CACHE_BYTES")
                                   : membudget::tier_mlx_cache_bytes();
              mx::set_cache_limit(c);
            }
          }
          // Only throttle DSpark draft depth on critical pressure (>= 4); warning level (2)
          // is normal on 16 GB Macs with an 8.7 GiB resident core and must keep full draft=5 speed.
          const int eff_draft = (decode_pressure >= 4 && !model->resident_experts()) ? std::min(dspark_draft, 3) : dspark_draft;
          const int K = std::min(eff_draft, ds->max_draft_tokens());
          static const bool ds_timing = [] {
            const char* e = std::getenv("VEYSTRIX_INKLING_DSPARK_TIMING");
            return e && std::strcmp(e, "1") == 0;
          }();
          static const float ds_tau_env = [] {
            const char* e = std::getenv("VEYSTRIX_INKLING_DSPARK_TAU");
            return (e && *e) ? static_cast<float>(std::atof(e)) : -1.0f;
          }();
          // Measured on this Mac: skipping the confidence probe after a miss loses
          // roughly half the accepted draft tokens with no latency saving.
          static const int ds_max_backoff = env_int("VEYSTRIX_INKLING_DSPARK_BACKOFF", 0);
          const float min_cum_p = (ds_tau_env >= 0.0f) ? ds_tau_env : (model->resident_experts() ? 0.0f : 0.50f);
          auto t_d0 = std::chrono::steady_clock::now();
          std::vector<int32_t> drafts;
          const bool skipped_probe = (!model->resident_experts() && s.dspark_skip_left > 0);
          if (skipped_probe) {
            --s.dspark_skip_left;
          } else {
            auto embed_fn = [this](const mx::array& ids) { return model->embed_normalized(ids); };
            auto head_fn = [this](const mx::array& x) { return model->lm_head_raw(x); };
            drafts = ds->draft_block(s.dspark_state, s.token, K, embed_fn, head_fn,
                                     model->final_logit_softcap(), model->unpadded_vocab(), min_cum_p);
          }
          // Zero-cost 4-gram suffix lookup fallback when DSpark is uncertain or in cooldown.
          if (drafts.empty() && s.chain.size() >= 8) {
            const size_t cn = s.chain.size();
            const size_t min_pos = (cn > 4096) ? (cn - 4096) : 0;
            const int32_t n0 = s.chain[cn - 4], n1 = s.chain[cn - 3], n2 = s.chain[cn - 2], n3 = s.chain[cn - 1];
            for (size_t pos = cn - 5; pos >= min_pos; --pos) {
              if (s.chain[pos] == n0 && s.chain[pos + 1] == n1 && s.chain[pos + 2] == n2 && s.chain[pos + 3] == n3) {
                const int take = std::min(std::min(K, 4), static_cast<int>(cn - 4 - pos));
                if (take >= 2) {
                  drafts.assign(s.chain.begin() + static_cast<std::ptrdiff_t>(pos + 4),
                                s.chain.begin() + static_cast<std::ptrdiff_t>(pos + 4 + take));
                }
                break;
              }
              if (pos == min_pos) break;
            }
          }
          auto t_d1 = std::chrono::steady_clock::now();
          const int n_draft = static_cast<int>(drafts.size());
          auto sample_row = [&](const mx::array& row_1v, int32_t exclude_id) -> int32_t {
            if (settings.greedy) {
              mx::array g = sample_greedy(row_1v);
              mx::eval(g);
              return static_cast<int32_t>(g.data<uint32_t>()[0]);
            }
            mx::array lg = row_1v;
            if (exclude_id >= 0) {
              mx::array mask = mx::equal(mx::arange(lg.shape(-1), mx::int32), mx::array(exclude_id, mx::int32));
              lg = mx::where(mask, mx::array(-1e9f, lg.dtype()), lg);
            }
            SamplingParams params;
            params.temperature = settings.temperature;
            params.top_p = settings.top_p;
            params.top_k = settings.top_k;
            params.min_p = settings.min_p;
            params.presence_penalty = settings.presence_penalty;
            std::vector<int32_t> seen(s.seen.begin(), s.seen.end());
            std::optional<mx::array> key;
            if (s.rng) key = s.rng->next();
            mx::array tok = sample_non_greedy(lg, params, seen, key);
            mx::eval(tok);
            return static_cast<int32_t>(tok.data<uint32_t>()[0]);
          };
          if (n_draft > 0) {
            std::vector<int32_t> verify_toks;
            verify_toks.reserve(drafts.size() + 1);
            verify_toks.push_back(s.token);
            verify_toks.insert(verify_toks.end(), drafts.begin(), drafts.end());
            const int S = static_cast<int>(verify_toks.size());
            // Compaction is deferred during verify (save_verify_history=true), so trim_verify_state
            // is guaranteed to succeed without holding a second SessionState copy (avoiding a 400+ MiB
            // MLX copy-on-write duplication of every KV buffer on every speculative verify step).
            const int64_t old_n = s.state[0].tokens;
            ForwardStats fs;
            VerifyResult vr = model->forward_with_features(verify_toks, s.state, /*all_logits=*/true,
                                                           /*save_verify_history=*/true, &fs);
            expert_bytes_total += static_cast<unsigned long long>(fs.expert_bytes);
            int accepted = 0;
            int32_t bonus = 0;
            if (settings.greedy) {
              mx::array pred = mx::argmax(vr.logits, -1);
              mx::eval(pred);
              const uint32_t* p = pred.data<uint32_t>();
              while (accepted < n_draft && static_cast<int32_t>(p[accepted]) == drafts[static_cast<size_t>(accepted)]) {
                ++accepted;
              }
              bonus = static_cast<int32_t>(p[accepted]);
            } else {
              // Exact lossless speculative rejection sampling matching sample_non_greedy
              // (honoring temperature, top_k, top_p, min_p, and presence_penalty) for
              // deterministic draft proposal q(x*)=1: accept x* with probability p_target(x*);
              // on reject, sample from the exact masked target distribution with x* excluded.
              SamplingParams params;
              params.temperature = settings.temperature;
              params.top_p = settings.top_p;
              params.top_k = settings.top_k;
              params.min_p = settings.min_p;
              params.presence_penalty = settings.presence_penalty;
              std::vector<int32_t> seen_live(s.seen.begin(), s.seen.end());
              const int V = vr.logits.shape(-1);
              if (params.presence_penalty == 0.0f && params.top_k > 0 && params.top_k < V) {
                // Fast batched path (Inkling default: T=0.6, top_p=0.95, top_k=20, presence_penalty=0):
                // evaluate all S = n_draft + 1 top_k distributions + uniforms in ONE single GPU sync,
                // then perform exact rejection + residual bonus sampling on CPU over <= top_k floats.
                std::vector<Distribution> dists;
                std::vector<mx::array> probs_vec;
                std::vector<mx::array> to_eval;
                dists.reserve(static_cast<size_t>(S));
                probs_vec.reserve(static_cast<size_t>(S));
                to_eval.reserve(static_cast<size_t>(2 * S + 1));
                for (int r_i = 0; r_i < S; ++r_i) {
                  mx::array row = mx::slice(vr.logits, {r_i, 0}, {r_i + 1, V});
                  dists.push_back(sampling_distribution(row, params, seen_live));
                  probs_vec.push_back(mx::softmax(dists.back().masked, -1, /*precise=*/true));
                  to_eval.push_back(dists.back().ids);
                  to_eval.push_back(probs_vec.back());
                }
                mx::array u = s.rng ? mx::random::uniform({S}, mx::float32, s.rng->next())
                                    : mx::random::uniform({S}, mx::float32);
                to_eval.push_back(u);
                mx::eval(to_eval);
                const float* uv = u.data<float>();
                for (int d_i = 0; d_i < n_draft; ++d_i) {
                  const int32_t target_id = drafts[static_cast<size_t>(d_i)];
                  const auto& ids = dists[static_cast<size_t>(d_i)].ids;
                  const int32_t* ids_ptr = ids.data<int32_t>();
                  const float* pr_ptr = probs_vec[static_cast<size_t>(d_i)].data<float>();
                  float p_tok = 0.0f;
                  for (size_t k = 0; k < ids.size(); ++k) {
                    if (ids_ptr[k] == target_id) {
                      p_tok = pr_ptr[k];
                      break;
                    }
                  }
                  if (uv[d_i] < p_tok) {
                    ++accepted;
                  } else {
                    break;
                  }
                }
                const int32_t reject_id = (accepted < n_draft) ? drafts[static_cast<size_t>(accepted)] : -1;
                const auto& b_ids = dists[static_cast<size_t>(accepted)].ids;
                const int32_t* b_ids_ptr = b_ids.data<int32_t>();
                const float* b_pr_ptr = probs_vec[static_cast<size_t>(accepted)].data<float>();
                const size_t bn = b_ids.size();
                double total_p = 0.0;
                for (size_t k = 0; k < bn; ++k) {
                  if (b_ids_ptr[k] != reject_id && b_pr_ptr[k] > 0.0f) total_p += static_cast<double>(b_pr_ptr[k]);
                }
                if (total_p > 0.0) {
                  const double draw = static_cast<double>(uv[n_draft]) * total_p;
                  double acc_p = 0.0;
                  bonus = b_ids_ptr[0];
                  for (size_t k = 0; k < bn; ++k) {
                    if (b_ids_ptr[k] == reject_id || b_pr_ptr[k] <= 0.0f) continue;
                    acc_p += static_cast<double>(b_pr_ptr[k]);
                    bonus = b_ids_ptr[k];
                    if (draw < acc_p) break;
                  }
                } else {
                  mx::array row = mx::slice(vr.logits, {accepted, 0}, {accepted + 1, V});
                  bonus = sample_row(row, reject_id);
                }
              } else {
                mx::array u = s.rng ? mx::random::uniform({n_draft}, mx::float32, s.rng->next())
                                    : mx::random::uniform({n_draft}, mx::float32);
                mx::eval(u);
                const float* uv = u.data<float>();
                for (int d_i = 0; d_i < n_draft; ++d_i) {
                  mx::array row = mx::slice(vr.logits, {d_i, 0}, {d_i + 1, V});
                  Distribution dist = sampling_distribution(row, params, seen_live);
                  mx::array probs = mx::softmax(dist.masked, -1, /*precise=*/true);
                  const int32_t target_id = drafts[static_cast<size_t>(d_i)];
                  float p_tok = 0.0f;
                  if (dist.has_ids) {
                    mx::eval(std::vector<mx::array>{dist.ids, probs});
                    const int32_t* ids_ptr = dist.ids.data<int32_t>();
                    const float* pr_ptr = probs.data<float>();
                    for (size_t k = 0; k < dist.ids.size(); ++k) {
                      if (ids_ptr[k] == target_id) {
                        p_tok = pr_ptr[k];
                        break;
                      }
                    }
                  } else {
                    mx::array pt = mx::slice(probs, {target_id}, {target_id + 1});
                    mx::eval(pt);
                    p_tok = pt.data<float>()[0];
                  }
                  if (uv[d_i] < p_tok) {
                    ++accepted;
                    if (params.presence_penalty > 0.0f &&
                        std::find(seen_live.begin(), seen_live.end(), target_id) == seen_live.end()) {
                      seen_live.push_back(target_id);
                    }
                  } else {
                    break;
                  }
                }
                mx::array row = mx::slice(vr.logits, {accepted, 0}, {accepted + 1, V});
                if (accepted < n_draft) {
                  const int32_t reject_id = drafts[static_cast<size_t>(accepted)];
                  Distribution dist = sampling_distribution(row, params, seen_live);
                  mx::array masked = dist.masked;
                  if (dist.has_ids) {
                    mx::array eq = mx::equal(dist.ids, mx::array(reject_id, mx::int32));
                    masked = mx::where(eq, mx::array(-std::numeric_limits<float>::infinity(), mx::float32), masked);
                  } else {
                    mx::array eq = mx::equal(mx::arange(V, mx::int32), mx::array(reject_id, mx::int32));
                    masked = mx::where(eq, mx::array(-std::numeric_limits<float>::infinity(), mx::float32), masked);
                  }
                  std::optional<mx::array> key;
                  if (s.rng) key = s.rng->next();
                  mx::array drawn = key ? mx::random::categorical(masked, -1, *key)
                                        : mx::random::categorical(masked);
                  mx::array tok = dist.has_ids ? mx::take(dist.ids, drawn, -1) : drawn;
                  mx::eval(tok);
                  bonus = static_cast<int32_t>(tok.data<uint32_t>()[0]);
                } else {
                  bonus = sample_row(row, -1);
                }
              }
            }
            auto t_v1 = std::chrono::steady_clock::now();
            const int keep = accepted + 1;
            if (!model->trim_verify_state(s.state, old_n, keep, S)) {
              throw std::runtime_error("generic runtime: speculative verify rollback failed");
            }
            if (vr.features) {
              mx::array f_keep = (keep < S)
                                     ? mx::slice(*vr.features, {0, 0, 0},
                                                 {keep, vr.features->shape(1), vr.features->shape(2)})
                                     : *vr.features;
              ds->commit_context(s.dspark_state, f_keep);
            }
            auto t_c1 = std::chrono::steady_clock::now();
            if (ds_timing) {
              auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
              std::fprintf(stderr,
                           "[DSPARK_ROUND] draft=%.1fms verify=%.1fms (exp=%.0fMB) commit=%.1fms acc=%d/%d\n",
                           ms(t_d0, t_d1), ms(t_d1, t_v1), fs.expert_bytes / 1048576.0, ms(t_v1, t_c1),
                           accepted, n_draft);
            }
            if (accepted > 0) {
              s.dspark_miss_streak = 0;
              s.dspark_skip_left = 0;
            } else if (ds_max_backoff > 0) {
              ++s.dspark_miss_streak;
              s.dspark_skip_left = std::min(s.dspark_miss_streak, ds_max_backoff);
            }
            spec_rounds.fetch_add(1);
            spec_drafted.fetch_add(static_cast<unsigned long long>(n_draft));
            spec_accepted.fetch_add(static_cast<unsigned long long>(accepted));
            for (int i = 0; i < accepted; ++i) s.spec_queue.push_back(drafts[static_cast<size_t>(i)]);
            s.spec_queue.push_back(bonus);
            s.token = bonus;
            s.logits_ready = false;
            const int32_t first_tok = s.spec_queue.front();
            s.spec_queue.pop_front();
            s.chain.push_back(first_tok);
            if (!settings.greedy && settings.presence_penalty > 0.0f) s.seen.insert(first_tok);
            ++s.emitted;
            steps.fetch_add(1);
            return {static_cast<uint32_t>(first_tok)};
          } else {
            // Low-confidence step or cooldown: run exact S=1 decode while keeping DSpark context synced.
            ForwardStats fs;
            VerifyResult vr = model->forward_with_features({s.token}, s.state, /*all_logits=*/false,
                                                           /*save_verify_history=*/false, &fs);
            expert_bytes_total += static_cast<unsigned long long>(fs.expert_bytes);
            const int32_t tok = sample_row(vr.logits, -1);
            auto t_v1 = std::chrono::steady_clock::now();
            if (vr.features) ds->commit_context(s.dspark_state, *vr.features);
            auto t_c1 = std::chrono::steady_clock::now();
            if (ds_timing) {
              auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
              std::fprintf(stderr,
                           "[DSPARK_ROUND] draft=%.1fms verify=%.1fms (exp=%.0fMB) commit=%.1fms acc=0/0 (%s)\n",
                           ms(t_d0, t_d1), ms(t_d1, t_v1), fs.expert_bytes / 1048576.0, ms(t_v1, t_c1),
                           skipped_probe ? "s1-cd" : "s1");
            }
            if (!skipped_probe && ds_max_backoff > 0) {
              ++s.dspark_miss_streak;
              if (s.dspark_miss_streak >= 2) {
                s.dspark_skip_left = std::min(s.dspark_miss_streak - 1, ds_max_backoff);
              }
            }
            spec_rounds.fetch_add(1);
            s.token = tok;
            s.logits_ready = false;
            s.chain.push_back(tok);
            if (!settings.greedy && settings.presence_penalty > 0.0f) s.seen.insert(tok);
            ++s.emitted;
            steps.fetch_add(1);
            return {static_cast<uint32_t>(tok)};
          }
        }
      }
    }
    auto& pf = cppprof::profiler();
    std::vector<std::vector<int32_t>> toks;
    std::vector<SessionState*> states;
    std::vector<Session*> need;
    for (Session* s : ss)
      if (!s->logits_ready) {
        need.push_back(s);
        toks.push_back({s->token});
        states.push_back(&s->state);
      }
    if (!need.empty()) {
      if (ss[0]->emitted > 0) pf.begin_step();
      auto tb = pf.now();
      ForwardStats fs;
      std::vector<mx::array> lg = model->forward(toks, states, &fs);
      pf.add_build(cppprof::ms_since(tb));
      auto te = pf.now();
      mx::eval(lg);
      pf.add_final_eval(cppprof::ms_since(te));
      pf.end_step_summary(cppprof::ms_since(tb), "generic");
      expert_bytes_total += static_cast<unsigned long long>(fs.expert_bytes);
      for (size_t i = 0; i < need.size(); ++i) {
        need[i]->logits = lg[i];
        need[i]->logits_ready = true;
      }
    }
    std::vector<uint32_t> out;
    for (size_t i = 0; i < ss.size(); ++i) out.push_back(sample(*ss[i], samplers[i]));
    return out;
  }

  void snapshot_save(Session& s, const std::string& dir) {
    if (s.chain.empty()) throw std::runtime_error("generic runtime: snapshot of an empty session");
    commit_pending(s);  // state + logits now cover the whole chain
    namespace fs = std::filesystem;
    fs::create_directories(dir);
    std::unordered_map<std::string, mx::array> arrays;
    for (auto& kv : model->save(s.state)) arrays.emplace(kv.first, mx::contiguous(kv.second));
    if (model->dspark()) {
      for (auto& kv : model->dspark()->save(s.dspark_state, "dspark")) {
        arrays.emplace(kv.first, mx::contiguous(kv.second));
      }
    }
    arrays.emplace("logits", *s.logits);
    arrays.emplace("chain", mx::array(s.chain.data(), mx::Shape{static_cast<int>(s.chain.size())}, mx::int32));
    std::unordered_map<std::string, std::string> meta{{"fingerprint", fingerprint},
                                                      {"n_tokens", std::to_string(s.chain.size())}};
    const char* fmt_env = std::getenv("VEYSTRIX_SNAPSHOT_FORMAT");
    const bool use_legacy = (fmt_env && std::strcmp(fmt_env, "legacy") == 0);
    if (!use_legacy) {
      SnapshotSaveOptions sopts;
      sopts.codec = default_snapshot_codec();
      sopts.page_tokens = 256;
      sopts.io_threads = 4;
      save_vxkv2_snapshot(dir, arrays, meta, sopts);
    } else {
      const std::string raw_tmp = dir + "/generic_state.safetensors.tmp";
      const std::string raw_path = dir + "/generic_state.safetensors";
      const std::string lz4_path = dir + "/generic_state.safetensors.lz4";
      mx::save_safetensors(raw_tmp, arrays, meta);
      bool wrote_lz4 = false;
#ifdef __APPLE__
      {
        std::error_code ec;
        auto raw_size = fs::file_size(raw_tmp, ec);
        if (!ec && raw_size > 4096 && raw_size <= static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
          std::ifstream rf(raw_tmp, std::ios::binary);
          std::vector<uint8_t> raw_buf(static_cast<size_t>(raw_size));
          if (rf.read(reinterpret_cast<char*>(raw_buf.data()), static_cast<std::streamsize>(raw_size))) {
            std::vector<uint8_t> dst(raw_buf.size());
            size_t csz = compression_encode_buffer(
                dst.data(), dst.size(), raw_buf.data(), raw_buf.size(), nullptr, COMPRESSION_LZ4);
            if (csz > 0 && csz + 16 < (raw_buf.size() * 95) / 100) {
              const std::string lz4_tmp = lz4_path + ".tmp";
              std::ofstream lf(lz4_tmp, std::ios::binary);
              static constexpr char kMagic[8] = {'V', 'X', 'L', 'Z', '4', 0, 0, 0};
              uint64_t ulen = static_cast<uint64_t>(raw_buf.size());
              lf.write(kMagic, 8);
              lf.write(reinterpret_cast<const char*>(&ulen), sizeof(ulen));
              lf.write(reinterpret_cast<const char*>(dst.data()), static_cast<std::streamsize>(csz));
              lf.close();
              if (lf) {
                fs::rename(lz4_tmp, lz4_path, ec);
                if (!ec) {
                  fs::remove(raw_tmp, ec);
                  fs::remove(raw_path, ec);
                  wrote_lz4 = true;
                }
              }
              if (!wrote_lz4) fs::remove(lz4_tmp, ec);
            }
          }
        }
      }
#endif
      if (!wrote_lz4) {
        std::error_code ec;
        fs::remove(lz4_path, ec);
        fs::rename(raw_tmp, raw_path);
      }
    }
    std::ofstream m(dir + "/meta.txt");
    m << "fingerprint " << fingerprint << "\nn_tokens " << s.chain.size() << "\n";
    if (!m) throw std::runtime_error("generic runtime: snapshot meta write failed");
  }

  void snapshot_restore(Session& s, const std::string& dir) {
    namespace fs = std::filesystem;
    std::unordered_map<std::string, mx::array> arrays;
    std::unordered_map<std::string, std::string> meta;
    if (is_vxkv2_snapshot(dir)) {
      auto loaded = load_vxkv2_snapshot(dir, "", 4);
      arrays = std::move(loaded.first);
      meta = std::move(loaded.second);
    } else {
      const std::string lz4_path = dir + "/generic_state.safetensors.lz4";
      const std::string raw_path = dir + "/generic_state.safetensors";
      std::string load_path = raw_path;
      std::string cleanup_tmp;
      struct TmpGuard {
        std::string* p;
        ~TmpGuard() {
          if (p && !p->empty()) {
            std::error_code ec;
            fs::remove(*p, ec);
          }
        }
      } guard{&cleanup_tmp};
      if (fs::is_regular_file(lz4_path)) {
#ifdef __APPLE__
        std::ifstream lf(lz4_path, std::ios::binary | std::ios::ate);
        auto fsize = lf.tellg();
        if (fsize < 16) throw std::runtime_error("generic runtime: corrupt lz4 snapshot");
        lf.seekg(0);
        std::vector<uint8_t> src(static_cast<size_t>(fsize));
        if (!lf.read(reinterpret_cast<char*>(src.data()), fsize))
          throw std::runtime_error("generic runtime: failed reading lz4 snapshot");
        static constexpr char kMagic[8] = {'V', 'X', 'L', 'Z', '4', 0, 0, 0};
        if (std::memcmp(src.data(), kMagic, 8) != 0)
          throw std::runtime_error("generic runtime: invalid lz4 snapshot magic");
        uint64_t ulen = 0;
        std::memcpy(&ulen, src.data() + 8, sizeof(ulen));
        std::vector<uint8_t> dst(static_cast<size_t>(ulen));
        size_t dsz = compression_decode_buffer(
            dst.data(), dst.size(), src.data() + 16, src.size() - 16, nullptr, COMPRESSION_LZ4);
        if (dsz != ulen) throw std::runtime_error("generic runtime: lz4 snapshot decode size mismatch");
        cleanup_tmp = dir + "/.restore.safetensors.tmp";
        std::ofstream rf(cleanup_tmp, std::ios::binary);
        rf.write(reinterpret_cast<const char*>(dst.data()), static_cast<std::streamsize>(dsz));
        rf.close();
        if (!rf) throw std::runtime_error("generic runtime: failed writing restore temp file");
        load_path = cleanup_tmp;
#else
        throw std::runtime_error("generic runtime: lz4 snapshot requires Apple libcompression");
#endif
      }
      auto loaded = mx::load_safetensors(load_path);
      arrays = std::move(loaded.first);
      meta = std::move(loaded.second);
    }
    auto fp = meta.find("fingerprint");
    if (fp == meta.end() || fp->second != fingerprint)
      throw std::runtime_error("generic runtime: snapshot was taken with a different model/recipe/KV plan");
    auto ch = arrays.find("chain");
    auto lg = arrays.find("logits");
    if (ch == arrays.end() || lg == arrays.end()) throw std::runtime_error("generic runtime: bad snapshot");
    mx::array chain = mx::astype(ch->second, mx::int32);
    mx::eval(chain);
    std::vector<int32_t> tokens(chain.data<int32_t>(), chain.data<int32_t>() + chain.size());
    if (tokens.empty()) throw std::runtime_error("generic runtime: empty snapshot");
    reset_session(s);
    NamedArrays named;
    std::vector<mx::array> to_eval;
    for (auto& kv : arrays) {
      if (kv.first == "chain" || kv.first == "logits") continue;
      named.emplace_back(kv.first, kv.second);
      to_eval.push_back(kv.second);
    }
    to_eval.push_back(lg->second);
    mx::eval(to_eval);
    model->load(s.state, named, static_cast<int64_t>(tokens.size()));
    if (model->dspark()) {
      model->dspark()->load(s.dspark_state, named, "dspark");
    }
    s.chain = tokens;
    s.logits = lg->second;
    s.logits_ready = true;
    s.token = tokens.back();
    s.emitted = 0;
    s.seen.clear();
    s.cancel_requested = false;
    cancel_flag.store(false);
  }

  int session_tokens(Session& s) {
    commit_pending(s);
    return static_cast<int>(s.chain.size());
  }

  void session_truncate(Session& s, uint32_t target_tokens) {
    commit_pending(s);
    if (target_tokens >= s.chain.size()) return;
    if (target_tokens == 0) {
      reset_session(s);
      return;
    }
    if (!model->truncate(s.state, static_cast<int64_t>(target_tokens))) {
      throw std::runtime_error("generic runtime: session truncate failed (history aged out of sliding window)");
    }
    s.chain.resize(target_tokens);
    s.token = s.chain.back();
    s.logits_ready = true;
    s.spec_queue.clear();
    s.emitted = 0;
    s.seen.clear();
    if (model->dspark()) {
      model->dspark()->truncate(s.dspark_state, static_cast<int64_t>(target_tokens));
    }
  }
};

GenericEngineRuntime::GenericEngineRuntime() : impl_(std::make_unique<Impl>()) {}
GenericEngineRuntime::~GenericEngineRuntime() = default;

void GenericEngineRuntime::open(const std::string& model_dir, const std::string& profile_json) {
  impl_->start();
  impl_->post([this, model_dir, profile_json] { impl_->load(model_dir, profile_json); });
}

int GenericEngineRuntime::new_session() {
  int sid = 0;
  impl_->post([this, &sid] {
    if (!impl_->model) throw std::runtime_error("generic runtime: open() before new_session()");
    sid = impl_->next_sid++;
    Session s;
    s.id = sid;
    impl_->reset_session(s);
    impl_->sessions.emplace(sid, std::move(s));
  });
  return sid;
}

void GenericEngineRuntime::close_session(int sid) {
  impl_->post([this, sid] {
    impl_->sessions.erase(sid);
    membudget::clear_mlx_cache();
    rowfetch::evict_cached_files();
  });
}

void GenericEngineRuntime::prefill(int sid, const std::vector<uint32_t>& tokens) {
  impl_->post([this, sid, &tokens] { impl_->prefill_state(impl_->get(sid), tokens, false); });
}

void GenericEngineRuntime::prefill_append(int sid, const std::vector<uint32_t>& tokens) {
  impl_->post([this, sid, &tokens] { impl_->prefill_state(impl_->get(sid), tokens, true); });
}

uint32_t GenericEngineRuntime::step(int sid, const std::string& sampler_json) {
  uint32_t out = 0;
  impl_->post([this, sid, &sampler_json, &out] { out = impl_->step_many({sid}, {sampler_json})[0]; });
  return out;
}

std::vector<uint32_t> GenericEngineRuntime::step_batch(const std::vector<int>& sids,
                                                       const std::vector<std::string>& sampler_jsons) {
  if (sids.size() != sampler_jsons.size()) throw std::runtime_error("generic runtime: step_batch size mismatch");
  std::vector<uint32_t> out;
  impl_->post([this, &sids, &sampler_jsons, &out] { out = impl_->step_many(sids, sampler_jsons); });
  return out;
}

void GenericEngineRuntime::fork_session(int src_sid, int dst_sid) {
  impl_->post([this, src_sid, dst_sid] {
    if (src_sid == dst_sid) throw std::runtime_error("generic runtime: fork needs two distinct sessions");
    Session& src = impl_->get(src_sid);
    Session& dst = impl_->get(dst_sid);
    dst = src;  // MLX arrays are immutable handles: copy-on-write
    dst.id = dst_sid;
    dst.cancel_requested = false;
  });
}

void GenericEngineRuntime::seed(int sid, uint64_t seed) {
  impl_->post([this, sid, seed] { impl_->get(sid).rng.emplace(seed); });
}

std::string GenericEngineRuntime::last_digest(int sid) {
  std::string out;
  impl_->post([this, sid, &out] { out = impl_->get(sid).last_digest; });
  return out;
}

void GenericEngineRuntime::cancel_session(int sid) {
  impl_->post([this, sid] {
    auto it = impl_->sessions.find(sid);
    if (it != impl_->sessions.end()) it->second.cancel_requested = true;
  });
}

int GenericEngineRuntime::session_tokens(int sid) {
  int result = 0;
  impl_->post([this, sid, &result] { result = impl_->session_tokens(impl_->get(sid)); });
  return result;
}

void GenericEngineRuntime::session_truncate(int sid, uint32_t target_tokens) {
  impl_->post([this, sid, target_tokens] { impl_->session_truncate(impl_->get(sid), target_tokens); });
}

void GenericEngineRuntime::cancel() { impl_->cancel_flag.store(true); }
bool GenericEngineRuntime::consume_cancel() { return impl_->cancel_flag.exchange(false); }

void GenericEngineRuntime::snapshot_save(int sid, const std::string& dir) {
  impl_->post([this, sid, dir] { impl_->snapshot_save(impl_->get(sid), dir); });
}

void GenericEngineRuntime::snapshot_restore(int sid, const std::string& dir) {
  impl_->post([this, sid, dir] { impl_->snapshot_restore(impl_->get(sid), dir); });
}

std::string GenericEngineRuntime::driver_stats() const {
  std::ostringstream o;
  o << ",\"engine\":\"generic\",\"driver_stream\":" << impl_->driver_stream_index;
  if (impl_->model) {
    const auto& m = *impl_->model;
    o << ",\"model_type\":\"" << m.model_type() << "\",\"recipe_digest\":\"" << m.recipe_digest()
      << "\",\"kv\":\"" << impl_->kv_desc << "\",\"resident_bytes\":" << m.resident_bytes()
      << ",\"streamed_bytes\":" << m.streamed_bytes()
      << ",\"expert_bytes_read\":" << impl_->expert_bytes_total
      << ",\"kv_values_per_token\":" << m.kv_values_per_token()
      << ",\"prefill_chunk\":" << impl_->prefill_chunk << ",\"load_seconds\":" << impl_->load_seconds
      << ",\"spec_rounds\":" << impl_->spec_rounds.load()
      << ",\"spec_drafted\":" << impl_->spec_drafted.load()
      << ",\"spec_accepted\":" << impl_->spec_accepted.load()
      << ",\"blocks\":{" << m.describe_blocks() << "}";
  }
  return o.str();
}

std::string GenericEngineRuntime::capabilities_extra() const {
  std::ostringstream o;
  o << ",\"engine\":\"generic\"";
  if (impl_->model) {
    const auto& m = *impl_->model;
    const KVPolicy& p = m.kv_policy();
    o << ",\"model_type\":\"" << m.model_type() << "\",\"kv\":{\"mode\":\"" << p.name << "\",\"plan\":\""
      << p.describe() << "\",\"exact\":" << (p.exact() ? "true" : "false")
      << ",\"window\":" << p.recent_tokens << ",\"prefix\":" << p.prefix_tokens
      << ",\"recent_bits\":" << p.recent_bits << ",\"hist_k_bits\":" << p.hist_k_bits
      << ",\"hist_v_bits\":" << p.hist_v_bits << "},\"experts\":\""
      << (m.resident_experts() ? "resident" : "streamed") << "\",\"speed\":\"" << impl_->speed_desc
      << "\",\"fork\":true,\"step_batch\":true";
  }
  return o.str();
}

}  // namespace veystrix::engine::capi_live
