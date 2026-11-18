// multisession_real.cpp -- OWNER-RUN real-checkpoint concurrent-session parity tool.
//
// WHAT THIS IS
// ------------
// End-to-end validation of multi-session bit-identity on the REAL 113 GB Q4
// checkpoint. N sessions with DISTINCT prompts share ONE read-only copy of the
// model weights (DenseLoader / RealLayer / RealHead / embed loaded once), but
// each session owns its OWN mutable decode state:
//
//   * std::vector<DecoderFullLayerCache>  (GDN conv/recurrent + full-attn KV)
//   * std::vector<PLECache>               (n-gram history + conv state)
//   * position counter, PRNG key/state (SessionState), step counter, logits
//
// Prefill is exclusive (v1 rule): each session's prompt is prefilled into its
// own caches on the calling thread, one session at a time, using the SAME
// batched prefill path as real_checkpoint_generate (the solo reference). Then
// decode steps are interleaved by MsScheduler (which drives cooperative_run on
// the SAME single GPU-driving thread). Rows are NEVER batched across sessions:
// every step is one independent single-row run_step() call, and its logits are
// fully materialized (mx::eval) before the next session's step runs.
//
// The per-step digest is the exact float32-little-endian SHA-256 convention of
// real_checkpoint_generate / capture_real_reference.py. Emitted line format:
//   session=<i> step=<s> token=<t> sha256=<64 hex>
//
// REUSE: this file does NOT reimplement any numeric primitive. It includes
// real_checkpoint_generate.cpp with main() renamed (the include-with-renamed-main
// pattern of probe_embed.cpp) and calls DenseLoader, load_ple, run_step,
// embed_token, run_prefill_batched, sample_greedy and sha256_hex directly. The
// weight-loading / prefill orchestration is copied verbatim from run_generation
// so the SOLO and CONCURRENT paths are the same computation.
//
// USAGE (all real runs must go through scripts/flashnext/real_run_guard.py):
//   multisession_real <checkpoint-dir> <num_sessions> <tokens> <config0> <config1> ...
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ---- Reuse the exact real-checkpoint implementation (main disabled) ---------
#define main realgen_main_disabled
#include "real_checkpoint_generate.cpp"
#undef main

#include "cooperative_decode.hpp"
#include "ms_scheduler.hpp"
#include "session_state.hpp"

namespace mx = mlx::core;
using namespace veystrix::engine;

// The shared, read-only model. Loaded ONCE for all sessions.
struct SharedModel {
  Config cfg;
  std::string ckpt;
  int group_size = 32;
  int bits = 4;
  int n_layers = 0;
  int H = 0;
  int top_k = 0;
  DecoderConfig dcfg;
  FullAttentionConfig fa;
  std::vector<RealLayer> layers;
  // RealHead / QTriplet have no default constructor (mx::array is non-default).
  std::optional<RealHead> head;
  std::optional<DenseLoader::QTriplet> embed;
  phase1::KVTierConfig kv_cfg;
  int conv_dim = 0;
};

// Per-session, exclusively owned mutable state. The weights in SharedModel are
// never written; only these objects change as a session advances.
struct SessionCtx {
  int id = 0;
  std::vector<int> prompt;
  int num_tokens = 0;
  int position = 0;
  int produced = 0;
  int token = 0;
  std::vector<DecoderFullLayerCache> caches;
  std::vector<PLECache> ple_caches;
  std::vector<phase1::KVCacheWriter> kv_writers;
  std::optional<mx::array> logits;
  bool logits_ready = false;
  SessionState sstate;  // independent PRNG key/state + position (bit-identity)
  std::vector<StepRecord> records;

  SessionCtx(int id_, unsigned long long prng_key, NumericsProfile profile)
      : id(id_),
        sstate(id_, prng_key, profile) {}
};

// Single driver thread -> a single pointer to the shared model is sufficient.
// MsScheduler::drive() is synchronous on this thread; no locks are introduced.
static SharedModel* g_model = nullptr;

// --------------------------------------------------------------------------- //
// Weight loading -- verbatim orchestration from run_generation() in
// real_checkpoint_generate.cpp (lines ~1385-1594), minus the prompt-dependent
// part. Every DenseLoader / StreamingMoE / CrossShardMoE / PLE call is identical.
// --------------------------------------------------------------------------- //
static void load_shared(const std::string& ckpt, const Config& cfg, SharedModel& m) {
  m.cfg = cfg;
  m.ckpt = ckpt;
  const int n_layers = get_int(cfg, "n_layers");
  const int H = get_int(cfg, "hidden_size");
  const int hc_count = get_int(cfg, "hc_count");
  const int hc_lowrank = get_int(cfg, "hc_lowrank");
  const int vocab = get_int(cfg, "vocab");
  const int top_k = get_int(cfg, "top_k");
  const int group_size = get_int(cfg, "group_size", 32);
  const int bits = get_int(cfg, "bits", 4);
  const float eps = get_float(cfg, "eps", 1e-6f);
  int num_slots = get_int(cfg, "num_slots", 16);
  {
    const char* e = std::getenv("VEYSTRIX_CPP_EXPERT_SLOTS");
    if (e) {
      int v = std::atoi(e);
      if (v >= top_k && v <= num_slots) num_slots = v;
    }
  }
  const std::vector<std::string> layer_types = split(get_str(cfg, "layer_types"), ',');
  if (static_cast<int>(layer_types.size()) != n_layers)
    throw std::runtime_error("layer_types size != n_layers");

  m.n_layers = n_layers;
  m.H = H;
  m.top_k = top_k;
  m.group_size = group_size;
  m.bits = bits;

  DecoderConfig dcfg;
  dcfg.hidden_size = H;
  dcfg.hc_count = hc_count;
  dcfg.hc_lowrank = hc_lowrank;
  dcfg.n_layers = n_layers;
  dcfg.vocab = vocab;
  dcfg.eps = eps;
  dcfg.gdn = GDNConfig{H, get_int(cfg, "key_dim"), get_int(cfg, "value_dim"),
                       get_int(cfg, "num_k_heads"), get_int(cfg, "num_v_heads"),
                       get_int(cfg, "head_k_dim"), get_int(cfg, "head_v_dim"),
                       get_int(cfg, "conv_kernel"), eps, GDNActivation::Sigmoid};

  FullAttentionConfig fa;
  fa.n_heads = get_int(cfg, "n_heads");
  fa.kv_heads = get_int(cfg, "kv_heads");
  fa.head_dim = get_int(cfg, "head_dim");
  fa.n_idx_heads = get_int(cfg, "n_idx_heads");
  fa.n_idx_kv = get_int(cfg, "n_idx_kv", 1);
  fa.idx_head_dim = get_int(cfg, "idx_head_dim");
  fa.compress_ratio = get_int(cfg, "compress_ratio", 4);
  fa.token_budget = get_int(cfg, "token_budget");
  fa.eps = eps;
  fa.rope.dim = get_int(cfg, "rope_dim");
  fa.rope.base = get_float(cfg, "rope_base", 10000.0f);
  fa.rope.style = get_str(cfg, "rope_style", "interleaved");
  fa.rope.attention_scaling = get_float(cfg, "rope_scaling", 1.0f);
  fa.rope.max_position_embeddings = get_int(cfg, "rope_max_pos", 2048);
  {
    auto sec = split(get_str(cfg, "rope_mrope"), ',');
    for (auto& x : sec) fa.rope.mrope_section.push_back(std::stoi(x));
  }
  m.dcfg = dcfg;
  m.fa = fa;

  DenseLoader dl(ckpt, cfg, group_size, bits);
  {
    const char* em = std::getenv("VEYSTRIX_CPP_DENSE_MANIFEST");
    std::string mp = em ? std::string(em) : get_str(cfg, "dense_manifest", "");
    if (!mp.empty()) dl.load_manifest(mp);
  }
  const std::string prefix_fmt =
      get_str(cfg, "expert_prefix", "language_model.model.layers.{L}.mlp.switch_mlp");
  auto layer_prefix = [&](int l) {
    std::string p = prefix_fmt;
    size_t pos = p.find("{L}");
    p.replace(pos, 3, std::to_string(l));
    return p;
  };

  std::vector<RealLayer>& layers = m.layers;
  layers.reserve(n_layers);

  const phase1::KVTierConfig kv_cfg = phase1::KVTierConfig::from_env();
  m.kv_cfg = kv_cfg;
  if (kv_cfg.enabled())
    std::fprintf(stderr, "[kv-tier] enabled mode=%d group_size=%d recent_tokens=%d\n",
                 static_cast<int>(kv_cfg.mode), kv_cfg.group_size, kv_cfg.recent_tokens);

  const int conv_dim = 2 * dcfg.gdn.key_dim + dcfg.gdn.value_dim;
  m.conv_dim = conv_dim;
  const int ple_layer = get_int(cfg, "ple_layer_ids", 2) - 1;
  const bool use_ple = std::getenv("VEYSTRIX_CPP_NO_PLE") == nullptr &&
                       get_int(cfg, "ple_layer_ids", 0) != 0;
  std::optional<PLEWeights> ple_weights;
  if (use_ple) ple_weights = load_ple(dl, cfg, ckpt, group_size, bits, eps);

  for (int l = 0; l < n_layers; ++l) {
    const std::string b = "language_model.model.layers." + std::to_string(l) + ".";
    const bool linear = layer_types[l] == "linear_attention";

    const int hc_hidden = hc_count * H;
    mx::array anw = dl.dense(b + "attn_hyper_connection.hc_norm.weight");
    mx::array mnw = dl.dense(b + "mlp_hyper_connection.hc_norm.weight");
    Linear adw = dl.linear(b + "attn_hyper_connection.input_mix_weight_down", hc_hidden);
    Linear auw = dl.linear(b + "attn_hyper_connection.input_mix_weight_up", hc_lowrank);
    Linear aiw = dl.linear(b + "attn_hyper_connection.block_inject_weight", hc_hidden);
    Linear mdw = dl.linear(b + "mlp_hyper_connection.input_mix_weight_down", hc_hidden);
    Linear muw = dl.linear(b + "mlp_hyper_connection.input_mix_weight_up", hc_lowrank);
    Linear miw = dl.linear(b + "mlp_hyper_connection.block_inject_weight", hc_hidden);
    Linear gate = dl.linear(b + "mlp.gate.weight", 0);
    Linear sgw = dl.linear(b + "mlp.shared_expert.gate_proj", H);
    Linear suw = dl.linear(b + "mlp.shared_expert.up_proj", H);
    Linear sdw = dl.linear(b + "mlp.shared_expert.down_proj", 0);
    Linear sglw = dl.linear(b + "mlp.shared_expert_gate.weight", H);

    std::optional<GDNWeights> gdn;
    std::optional<FullAttentionWeights> full;
    if (linear) {
      const std::string a = b + "linear_attn.";
      gdn.emplace(dl.linear(a + "in_proj_qkv", H), dl.linear(a + "in_proj_z", H),
                  dl.linear(a + "in_proj_b", H), dl.linear(a + "in_proj_a", H),
                  dl.dense(a + "conv1d.weight"), dl.dense(a + "A_log"),
                  dl.dense(a + "dt_bias"), dl.dense(a + "norm.weight"),
                  dl.linear(a + "out_proj", dcfg.gdn.value_dim));
    } else {
      const std::string a = b + "self_attn.";
      const int qdim = fa.n_heads * fa.head_dim * 2;
      const int idxdim = (fa.n_idx_heads + fa.n_idx_kv) * fa.idx_head_dim;
      full.emplace(dl.linear(a + "q_proj", H), dl.linear(a + "k_proj", H),
                   dl.linear(a + "v_proj", H), dl.linear(a + "o_proj", qdim / 2),
                   dl.dense(a + "q_norm.weight"), dl.dense(a + "k_norm.weight"),
                   dl.linear(a + "indexer.index_qk_proj", H),
                   dl.dense(a + "indexer.q_layernorm.weight"),
                   dl.dense(a + "indexer.k_layernorm.weight"));
    }

    std::string ep = layer_prefix(l);
    std::string first_shard;
    bool split = false;
    for (const char* part : {"gate_proj.weight", "gate_proj.scales",
                             "gate_proj.biases", "up_proj.weight",
                             "up_proj.scales", "up_proj.biases",
                             "down_proj.weight", "down_proj.scales",
                             "down_proj.biases"}) {
      auto it = cfg.find("shard." + ep + "." + part);
      if (it == cfg.end()) {
        split = true;
        break;
      }
      if (first_shard.empty()) first_shard = it->second;
      else if (it->second != first_shard) split = true;
    }

    std::unique_ptr<StreamingMoE> stream;
    std::unique_ptr<CrossShardMoE> cross;
    if (split) {
      cross = std::make_unique<CrossShardMoE>(
          ckpt, cfg, ep, num_slots, group_size, bits,
          debug_capture::get().copy_cross_shard);
    } else {
      stream = std::make_unique<StreamingMoE>(ckpt, l, num_slots, group_size,
                                              bits, /*read_width=*/8, ep);
    }

    if ((l + 1) % 8 == 0 || l == n_layers - 1)
      std::fprintf(stderr, "[load] prepared layer %d/%d%s\n", l + 1, n_layers,
                   split ? " (cross-shard experts)" : "");

    layers.emplace_back(
        std::move(anw), std::move(adw), std::move(auw), std::move(aiw),
        std::move(mnw), std::move(mdw), std::move(muw), std::move(miw),
        std::move(gate), std::move(sgw), std::move(suw), std::move(sdw),
        std::move(sglw), std::move(gdn), std::move(full), std::move(stream),
        std::move(cross), std::nullopt, top_k,
        linear ? AttnKind::Linear : AttnKind::FullAttention);
    layers.back().layer_idx = l;
    layers.back().split_layer = split;
    layers.back().expert_prefix = ep;
    layers.back().expert_shard_path = ckpt + "/" + first_shard;
    if (l == ple_layer && ple_weights)
      layers.back().ple = std::move(ple_weights);
  }

  const int head_hc_hidden = hc_count * H;
  m.head = RealHead{
      dl.dense("language_model.model.hyper_connection_mixer.hc_norm.weight"),
      dl.linear("language_model.model.hyper_connection_mixer.input_mix_weight_down",
                head_hc_hidden),
      dl.linear("language_model.model.hyper_connection_mixer.input_mix_weight_up",
                hc_lowrank),
      dl.quantized("language_model.lm_head")};
  m.embed = dl.quantized("language_model.model.embed_tokens");
  std::fprintf(stderr, "[load] all weights ready (shared across sessions)\n");
  mem_telemetry("post-load");
  {
    const char* emax = std::getenv("VEYSTRIX_CPP_MAX_ACTIVE_GIB");
    double limit = emax ? std::atof(emax)
                        : std::atof(get_str(cfg, "max_active_gib", "6.0").c_str());
    if (limit > 0) {
      double active_gib = mx::get_active_memory() / (1024.0 * 1024.0 * 1024.0);
      if (active_gib > limit)
        throw std::runtime_error(
            "[load] ABORT: post-load active " + std::to_string(active_gib) +
            " GiB exceeds --max-active-gib " + std::to_string(limit) + " GiB");
    }
  }
}

// --------------------------------------------------------------------------- //
// Per-session cache initialization (verbatim from run_generation, ~line 1579).
// --------------------------------------------------------------------------- //
static void init_session_caches(SharedModel& m, SessionCtx& s) {
  const int n_layers = m.n_layers;
  const int conv_dim = m.conv_dim;
  s.caches.reserve(n_layers);
  s.ple_caches.assign(n_layers, PLECache{});
  s.kv_writers.reserve(n_layers);
  for (int l = 0; l < n_layers; ++l) {
    GDNCacheState zero(
        mx::zeros(mx::Shape{1, m.dcfg.gdn.conv_kernel - 1, conv_dim}, mx::bfloat16),
        mx::zeros(mx::Shape{1, m.dcfg.gdn.num_v_heads, m.dcfg.gdn.head_v_dim,
                            m.dcfg.gdn.head_k_dim}, mx::float32));
    s.caches.emplace_back(DecoderLayerCache(std::move(zero)));
    s.kv_writers.emplace_back(m.kv_cfg);
  }
}

// --------------------------------------------------------------------------- //
// Exclusive batched prefill for ONE session. Verbatim orchestration from
// run_generation (lines ~1678-1874). Uses the session's own caches / PLE / KV
// writers; keeps the always-on f32 materialization barriers (they change the
// eval order and therefore the cache bytes).
// --------------------------------------------------------------------------- //
static mx::array prefill_session(SharedModel& m, SessionCtx& s) {
  const Config& cfg = m.cfg;
  const int n_layers = m.n_layers;
  const int H = m.H;
  const int group_size = m.group_size;
  const int bits = m.bits;
  const int top_k = m.top_k;
  const std::string& ckpt = m.ckpt;

  std::vector<BatchedLayer> bl;
  bl.reserve(n_layers);
  for (int l = 0; l < n_layers; ++l) {
    RealLayer& R = m.layers[l];
    bl.emplace_back(R.attn_norm_w, R.attn_down_w, R.attn_up_w, R.attn_inject_w,
                    R.mlp_norm_w, R.mlp_down_w, R.mlp_up_w, R.mlp_inject_w,
                    R.gate_w, R.shared_gw, R.shared_uw, R.shared_dw,
                    R.shared_glw, R.top_k, R.kind);
    BatchedLayer& B = bl.back();
    if (R.gdn) B.gdn = R.gdn;
    if (R.full) B.full = R.full;
    if (R.ple) { B.ple = R.ple; B.ple_cache = s.ple_caches[l]; }
    B.moe_forward = [&R, &ckpt, &cfg, group_size, bits, top_k](
                        const mx::array& x, const mx::array& idx,
                        const mx::array& sc) {
      int S = static_cast<int>(x.shape(0));
      static const bool unsorted_prefill =
          std::getenv("VEYSTRIX_CPP_UNSORTED_PREFILL") != nullptr;
      if (S > 1 && !unsorted_prefill) {
        if (R.split_layer) {
          CrossShardMoE pf(ckpt, cfg, R.expert_prefix, top_k, group_size,
                           bits, false);
          return pf.forward_stacked(x, idx, sc);
        }
        StreamingMoE pf(ckpt, R.layer_idx, top_k, group_size, bits,
                        /*read_width=*/8, R.expert_prefix);
        return pf.forward_stacked(x, idx, sc);
      }
      if (S <= 1)
        return R.stream ? R.stream->forward(x, idx, sc).output
                        : R.cross->forward(x, idx, sc).output;
      std::vector<mx::array> outs;
      outs.reserve(S);
      for (int t = 0; t < S; ++t) {
        std::vector<int32_t> ti{t};
        mx::array rowidx(ti.data(), mx::Shape{1}, mx::int32);
        mx::array xt = mx::take(x, rowidx, 0);
        mx::array it = mx::take(idx, rowidx, 0);
        mx::array st = mx::take(sc, rowidx, 0);
        outs.push_back(R.stream ? R.stream->forward(xt, it, st).output
                                : R.cross->forward(xt, it, st).output);
      }
      return mx::concatenate(outs, 0);
    };
  }
  BatchedHead bh(m.head->norm_w, m.head->down_w, m.head->up_w,
                 BatchedLMTriplet(m.head->lm.w, m.head->lm.s, m.head->lm.b));
  std::vector<int32_t> ids(s.prompt.begin(), s.prompt.end());
  mx::array ids_arr(ids.data(), mx::Shape{1, static_cast<int>(ids.size())}, mx::int32);
  auto embed_all = [&](const mx::array& ida) {
    mx::array f = mx::reshape(ida, mx::Shape{-1});
    mx::array w = mx::take(m.embed->w, f, 0), sm = mx::take(m.embed->s, f, 0),
               b = mx::take(m.embed->b, f, 0);
    mx::array row = mx::dequantize(w, sm, b, group_size, bits, "affine");
    mx::eval(row);
    return mx::reshape(row, mx::Shape{1, static_cast<int>(f.size()), H});
  };
  BatchedPrefillDebug bpdbg;
  mx::array logits = run_prefill_batched(ids_arr, embed_all, bl, bh, m.dcfg,
                                         m.fa, s.caches, group_size, bits, &bpdbg);
  // Always-on f32 materialization barriers: serialize the deep 48-layer graph
  // so the prefill is bit-identical to the solo path. No numeric change here,
  // but the eval order matters.
  for (size_t l = 0; l < bpdbg.hidden_after_layer.size(); ++l) {
    mx::eval(mx::astype(bpdbg.hidden_after_layer[l], mx::float32));
    if (l < bpdbg.branch.size())
      mx::eval(mx::astype(bpdbg.branch[l], mx::float32));
    if (l < bpdbg.m2d.size())
      mx::eval(mx::astype(bpdbg.m2d[l], mx::float32));
    if (l < bpdbg.gate_logits.size())
      mx::eval(mx::astype(bpdbg.gate_logits[l], mx::float32));
    if (l < bpdbg.moe2d.size())
      mx::eval(mx::astype(bpdbg.moe2d[l], mx::float32));
    if (l < bpdbg.mlp_inj.size())
      mx::eval(mx::astype(bpdbg.mlp_inj[l], mx::float32));
    if (l < bpdbg.gdn_in.size())
      mx::eval(mx::astype(bpdbg.gdn_in[l], mx::float32));
  }
  for (int l = 0; l < n_layers; ++l)
    if (bl[l].ple) s.ple_caches[l] = bl[l].ple_cache;
  if (m.kv_cfg.enabled())
    for (int l = 0; l < n_layers; ++l) s.kv_writers[l].prime_full(s.caches[l].full);
  s.position = static_cast<int>(s.prompt.size());
  s.sstate.advance_position(s.position);
  mem_telemetry("prefill-batched");
  return logits;
}

// Swap a session's private PLE caches into the shared (read-only-except-PLE)
// layer objects immediately before a run_step, and copy them back after. On a
// single driver thread with one step fully evaluated before the next, this is
// safe and keeps PLE state strictly per-session.
static void ple_swap_in(SharedModel& m, SessionCtx& s) {
  for (int l = 0; l < m.n_layers; ++l)
    if (m.layers[l].ple) m.layers[l].ple_cache = s.ple_caches[l];
}
static void ple_swap_out(SharedModel& m, SessionCtx& s) {
  for (int l = 0; l < m.n_layers; ++l)
    if (m.layers[l].ple) s.ple_caches[l] = m.layers[l].ple_cache;
}

static void emit_record(SessionCtx& s, int in_tok, int out_tok,
                        const mx::array& decided) {
  mx::array row = mx::reshape(mx::astype(decided, mx::float32), mx::Shape{-1});
  mx::eval(row);  // force the whole step graph BEFORE another session's step
  std::string digest = sha256_hex(row.data<float>(),
                                  static_cast<size_t>(row.size()) * sizeof(float));
  s.records.push_back({s.produced, in_tok, out_tok, digest});
  std::fprintf(stderr, "session=%d step=%d token=%d sha256=%s\n", s.id,
               s.produced, out_tok, digest.c_str());
  ++s.produced;
}

static void consume_target(SharedModel& m, SessionCtx& s, int tok) {
  std::vector<int32_t> pd{static_cast<int32_t>(s.position++)};
  mx::array pos_ids(pd.begin(), mx::Shape{1, 1}, mx::int32);
  ple_swap_in(m, s);
  s.logits = run_step(embed_token(*m.embed, tok, m.group_size, m.bits, m.H),
                      static_cast<int32_t>(tok), m.layers, *m.head, m.dcfg,
                      m.fa, pos_ids, s.caches, m.group_size, m.bits,
                      s.kv_writers);
  ple_swap_out(m, s);
  s.sstate.advance_position(1);
  s.logits_ready = true;
  mem_telemetry("decode-step");
}

// One cooperative decode step for one session; invoked by MsScheduler on the
// single driver thread. It reads/writes ONLY this session's mutable state.
static bool decode_step(void* state, int /*step*/, unsigned long long /*seed*/) {
  SessionCtx* s = static_cast<SessionCtx*>(state);
  SharedModel* m = g_model;
  if (!s->logits_ready) consume_target(*m, *s, s->token);

  mx::array decided = *s->logits;
  (void)s->sstate.draw_u64();  // advance THIS session's PRNG stream only
  mx::array next = sample_greedy(decided);
  mx::eval(next);
  int next_id = static_cast<int>(next.data<uint32_t>()[0]);
  emit_record(*s, s->token, next_id, decided);
  s->token = next_id;
  s->logits_ready = false;
  return s->produced >= s->num_tokens;
}

static void check_same_geometry(const Config& a, const Config& b) {
  for (const char* k : {"n_layers", "hidden_size", "hc_count", "hc_lowrank",
                        "vocab", "top_k", "group_size", "bits", "layer_types"}) {
    std::string va = get_str(a, k), vb = get_str(b, k);
    if (va != vb)
      throw std::runtime_error(std::string("config geometry mismatch on ") + k +
                               ": " + va + " != " + vb);
  }
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <checkpoint-dir> <num_sessions> <tokens> "
                 "<config0> <config1> ...\n", argv[0]);
    return 2;
  }
  const std::string ckpt = argv[1];
  const int num_sessions = std::atoi(argv[2]);
  const int tokens = std::atoi(argv[3]);
  if (num_sessions < 1 || tokens < 1 || argc != 4 + num_sessions) {
    std::fprintf(stderr, "invalid num_sessions/tokens/config count\n");
    return 2;
  }
  try {
    std::vector<Config> configs;
    for (int i = 0; i < num_sessions; ++i) {
      Config c = read_config(argv[4 + i]);
      if (c.empty()) throw std::runtime_error("empty config: " + std::string(argv[4 + i]));
      if (get_int(c, "prompt_tokens", 0) == 0)
        throw std::runtime_error("config has no prompt_ids: " + std::string(argv[4 + i]));
      if (i > 0) check_same_geometry(configs[0], c);
      configs.push_back(std::move(c));
    }

    // Load the real weights ONCE. All sessions share this read-only model.
    SharedModel model;
    load_shared(ckpt, configs[0], model);
    g_model = &model;

    // Per-session caches / PLE / KV writers / PRNG; exclusive prefill.
    std::vector<SessionCtx> sessions;
    sessions.reserve(num_sessions);
    for (int i = 0; i < num_sessions; ++i) {
      NumericsProfile profile;
      profile.prefill_chunk = get_int(configs[i], "prefill_chunk", 512);
      profile.freeze();
      sessions.emplace_back(i, /*prng_key=*/0x9E3779B97F4A7C15ULL +
                                    static_cast<unsigned long long>(i),
                            profile);
      SessionCtx& s = sessions.back();
      s.prompt = split_ints(get_str(configs[i], "prompt_ids"));
      s.num_tokens = tokens;
      s.token = s.prompt.empty() ? 0 : s.prompt.back();
      init_session_caches(model, s);
      mx::array logits = prefill_session(model, s);
      if (s.prompt.empty()) throw std::runtime_error("empty prompt");
      s.logits = logits;
      s.logits_ready = true;
      std::fprintf(stderr, "[prefill] session=%d prompt_tokens=%zu\n",
                   s.id, s.prompt.size());
    }

    // Interleave decode steps via MsScheduler on this ONE thread. Prefill was
    // already exclusive, so all slots start Ready (prefill_steps=0).
    SchedulerConfig sched_cfg;
    sched_cfg.prefill_exclusive = true;
    sched_cfg.max_skew_steps = 1;
    MsScheduler scheduler(sched_cfg);
    for (int i = 0; i < num_sessions; ++i) {
      ScheduledSession ss;
      ss.id = i;
      ss.state = &sessions[i];
      ss.session = &sessions[i].sstate;
      ss.step_fn = &decode_step;
      ss.seed = sessions[i].sstate.prng_key();
      ss.prefill_steps = 0;
      scheduler.add(ss);
    }
    scheduler.drive(tokens + num_sessions + 4);

    int total = 0;
    for (const auto& s : sessions) {
      total += s.produced;
      if (s.produced != tokens)
        std::fprintf(stderr,
                     "[warn] session=%d produced %d/%d steps\n",
                     s.id, s.produced, tokens);
    }
    std::fprintf(stderr, "[done] sessions=%d tokens_each=%d steps=%d rounds=%d\n",
                 num_sessions, tokens, total, scheduler.rounds());
    return total == num_sessions * tokens ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "multisession_real failed: %s\n", e.what());
    return 1;
  }
}
