// Phase F — Nemotron per-session decode runtime + per-layer cursor seam.
//
// WHY THE INCLUDE: nemotron_generate.cpp owns the 52-layer run_step() whose
// ordering (rms_norm -> mamba/attn/moe -> residual add, plus the fused router
// and INT8 KV cache) was bit-gated against the Python reference. Re-typing any
// primitive risks a silent bit divergence. We include that TU with main()
// renamed so EVERY numeric primitive is reused verbatim; only the driver changes.
//
// The multi-session path advances one WHOLE run_step() per session per
// scheduler step, with each session's mutable caches isolated in a
// SessionCacheState. Nemotron's expert reads are synchronous pread (no async
// ticket seam), so there is no safe mid-layer yield and the whole-token path is
// the bit-exact one. run_step_cursor() below is the structural per-layer
// layer_pre/layer_post driver (N=1 gate) mirroring qwen_runtime; the hard N=2/3
// gate uses run_step verbatim. See docs/engineering/phase_f_nemotron.md.
//
// C1: one driver thread (this one), no new thread/mutex/CV/worker. Expert reads
// use the existing synchronous pread; g_notify_count is a single atomic that is
// never touched by MLX (and stays 0 because Nemotron has no async dispatch).
#include <atomic>
#include <memory>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <utility>

#define main nemotron_main_disabled
#include "../tools/nemotron_generate.cpp"
#undef main

#include "nemotron_runtime.hpp"
#include "ms_scheduler.hpp"
#include "mamba_prefill.hpp"
#include <mlx/memory.h>

namespace mx = mlx::core;
using namespace veystrix::engine;

namespace veystrix::engine::nemotron_runtime {

// C1: a single atomic counter; the (nonexistent) dispatch callback would only
// bump it and never call MLX. Nemotron uses synchronous pread so it stays 0.
static std::atomic<int> g_notify_count{0};
static bool g_cursor_enabled = false;

bool phase_f_cursor_enabled() { return g_cursor_enabled; }
int cursor_notify_count() { return g_notify_count.load(std::memory_order_relaxed); }

void SessionCacheState::init(int n_mamba, int conv_kernel, int conv_dim,
                             int num_heads, int head_dim, int ssm_state_size,
                             int n_attn, int block_tokens) {
  mamba_caches.reserve(static_cast<std::size_t>(n_mamba));
  for (int i = 0; i < n_mamba; ++i)
    mamba_caches.emplace_back(
        mx::zeros(mx::Shape{1, conv_kernel - 1, conv_dim}, mx::bfloat16),
        mx::zeros(mx::Shape{1, num_heads, head_dim, ssm_state_size}, mx::float32));
  attn_f32.assign(static_cast<std::size_t>(n_attn), NemoAttentionCache{});
  attn_int8.clear();
  attn_int8.reserve(static_cast<std::size_t>(n_attn));
  for (int i = 0; i < n_attn; ++i) {
    attn_int8.emplace_back();
    attn_int8.back().block_tokens = block_tokens;
  }
}

// --------------------------------------------------------------------------- //
// Per-layer cursor driver (structural parity with qwen_runtime).              //
// layer_pre computes one layer's mixer delta and stashes it; layer_post adds   //
// the residual and advances the per-kind counters. The op order is identical  //
// to run_step(); only the boundary between the two halves is new.              //
// --------------------------------------------------------------------------- //
struct StepCtx {
  Model* m = nullptr;
  ExpertReader* er = nullptr;
  mx::array h;                       // current hidden state [1,1,H] bf16
  std::optional<mx::array> delta;    // stashed mixer output for this layer
  std::size_t mamba_i = 0, attn_i = 0, moe_i = 0;
  int layer = -1;

  // Stage 1 batched MoE: when `batched`, layer_pre records the routed expert ids,
  // union-ready scores and xn, but DEFERS the ExpertReader read + moe_block_relu2
  // to the batched driver (which reads the union once for all sessions).
  bool batched = false;
  std::vector<int> moe_ids;         // deferred: this session's top-k expert ids
  std::optional<mx::array> moe_scores;  // [1,K] f32
  std::optional<mx::array> moe_xn;      // [1,1,H] bf16 normalized hidden

  // P5-overlap: async reads submitted at layer_pre_submit and consumed at
  // layer_pre_finish. State needed to finish the MoE after the ticket completes.
  veystrix::AsyncTicket pending = veystrix::kInvalidTicket;
  bool async_defer = false;          // request async submit in layer_pre
  bool deferred_async = false;       // true iff this layer submitted async reads
  std::vector<int> async_unique;     // unique expert ids (for fallback reads)
  std::vector<int> async_local;      // route -> unique position
  std::vector<int> async_route;      // per-session copy of pool route slots
  MoeLayer* async_moe = nullptr;     // borrowed layer weights
  PooledExpertTensors* async_pool = nullptr;  // borrowed pool
  std::optional<mx::array> async_scores;

  StepCtx(Model& mm, ExpertReader& e, mx::array h0)
      : m(&mm), er(&e), h(std::move(h0)) {}
};

void layer_pre(StepCtx& c, int l) {
  Model& m = *c.m;
  ExpertReader& er = *c.er;
  c.layer = l;

  mx::array xn = mx::fast::rms_norm(c.h, m.kinds[l] == LayerKind::MAMBA_SSM
                                             ? m.mamba_layers[c.mamba_i].norm_w
                                             : (m.kinds[l] == LayerKind::FULL_ATTN_QSA
                                                    ? m.attn_layers[c.attn_i].norm_w
                                                    : m.moe_layers[c.moe_i].norm_w),
                                     m.eps);

  mx::array delta = empty_f32();
  if (m.kinds[l] == LayerKind::MAMBA_SSM) {
    MambaLayer& ml = m.mamba_layers[c.mamba_i++];
    MambaConfig mc;
    mc.hidden_size = m.H;
    mc.num_heads = m.spec.mamba_num_heads;
    mc.head_dim = m.spec.mamba_head_dim;
    mc.n_groups = m.spec.mamba_n_groups;
    mc.ssm_state_size = m.spec.mamba_ssm_state_size;
    mc.conv_kernel = m.spec.mamba_conv_kernel;
    mc.intermediate_size = m.spec.mamba_intermediate_size;
    mc.conv_dim = m.spec.mamba_conv_dim;
    mc.eps = m.eps;
    mc.time_step_min = m.spec.mamba_time_step_min;
    mc.time_step_max = m.spec.mamba_time_step_max;
    MambaCacheState& cache = m.mamba_caches[ml.cache_idx];
    MambaLayerOutput out = mamba_layer_forward(xn, ml.weights, cache, mc);
    cache = MambaCacheState(std::move(out.conv_state), std::move(out.ssm_state));
    delta = std::move(out.output);
  } else if (m.kinds[l] == LayerKind::FULL_ATTN_QSA) {
    AttnLayer& al = m.attn_layers[c.attn_i++];
    if (m.use_int8_kv)
      delta = nemo_attention_int8_forward(xn, al.weights, al.cfg, al.int8_cache);
    else
      delta = nemo_attention_forward(xn, al.weights, al.cfg, al.cache);
  } else {
    MoeLayer& ml = m.moe_layers[c.moe_i++];
    NemotronRoute route = nemotron_route(xn, ml, m.top_k, m.routed_scale);

    if (c.batched) {
      // Stage 1: record this session's route + xn, defer the read+gather to the
      // batched driver. Router op is unchanged (M=1) so its bits are solo-exact.
      mx::array ids_i = mx::astype(route.indices, mx::int32);
      mx::eval(ids_i);
      const int32_t* ip = ids_i.data<int32_t>();
      c.moe_ids.assign(ip, ip + m.top_k);
      c.moe_scores = route.scores;
      c.moe_xn = xn;
      return;  // delta left unset; batched driver fills c.delta for this layer
    }

    mx::array ids_i = mx::astype(route.indices, mx::int32);
    mx::eval(ids_i);
    const int32_t* ip = ids_i.data<int32_t>();
    std::vector<int> ids(ip, ip + m.top_k);
    std::vector<int> unique = ids;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    std::vector<int> local(m.top_k);
    for (int i = 0; i < m.top_k; ++i)
      local[i] = static_cast<int>(
          std::lower_bound(unique.begin(), unique.end(), ids[i]) - unique.begin());
    mx::array local_idx(local.begin(), mx::Shape{1, m.top_k}, mx::int32);

    std::optional<Q4Projection> up, down;
    mx::array moe_route = local_idx;
    if (PooledExpertTensors* P = m.expert_pool.get()) {
      std::vector<veystrix::PagerRead> reads;
      P->begin_layer(l, unique, local, reads);
      if (c.async_defer) {
        // Submit now, finish in layer_pre_finish after the driver has issued
        // every other session's reads (overlap: their host build + GPU work
        // runs while these bytes are in flight).
        c.pending = P->submit_fills(reads);
        c.deferred_async = true;
        c.async_unique = unique;
        c.async_local = local;
        // Pool route_slots_ is shared mutable state: copy it NOW so a later
        // session's begin_layer cannot overwrite this session's route.
        const std::vector<int>& ri0 = P->route_indices();
        c.async_route.assign(ri0.begin(), ri0.end());
        c.async_moe = &ml;
        c.async_pool = P;
        c.async_scores = route.scores;
        c.moe_xn = xn;
        return;  // delta deferred to layer_pre_finish
      }
      P->run_fills(reads);
      up.emplace(P->make_array(kF1W), P->make_array(kF1S), P->make_array(kF1B));
      down.emplace(P->make_array(kF2W), P->make_array(kF2S), P->make_array(kF2B));
      const std::vector<int>& ri = P->route_indices();
      moe_route = mx::array(ri.begin(), mx::Shape{1, m.top_k}, mx::int32);
      cppprof::profiler().add_layer_cache(l, P->last_hits(), P->last_misses(),
                                          P->last_coalesced());
    } else {
      up.emplace(er.rows_u32(ml.fc1_base + ".weight", unique),
                 er.rows_bf16(ml.fc1_base + ".scales", unique),
                 er.rows_bf16(ml.fc1_base + ".biases", unique));
      down.emplace(er.rows_u32(ml.fc2_base + ".weight", unique),
                   er.rows_bf16(ml.fc2_base + ".scales", unique),
                   er.rows_bf16(ml.fc2_base + ".biases", unique));
    }

    mx::array x2 = mx::reshape(xn, mx::Shape{1, m.H});
    mx::array routed = moe_block_relu2(x2, moe_route, route.scores, *up, *down);
    routed = mx::astype(routed, mx::bfloat16);
    routed = mx::reshape(routed, mx::Shape{1, 1, m.H});

    mx::array su = linear(xn, ml.shared_up);
    mx::array sa = mx::square(
        mx::maximum(su, mx::astype(mx::array(0.0f), su.dtype())));
    mx::array sd = linear(sa, ml.shared_down);
    delta = routed + sd;
  }
  c.delta = std::move(delta);
}

// Complete an async MoE layer submitted by layer_pre with async_defer=true.
// All numeric ops are IDENTICAL to the synchronous path above (same gather
// order, same dtypes); only the *when* of the read wait moved.
void layer_pre_finish(StepCtx& c) {
  if (!c.deferred_async) return;
  c.deferred_async = false;
  PooledExpertTensors* P = c.async_pool;
  P->wait_fill(c.pending);  // acquire edge: slot bytes fully written
  Q4Projection up(P->make_array(kF1W), P->make_array(kF1S), P->make_array(kF1B));
  Q4Projection down(P->make_array(kF2W), P->make_array(kF2S), P->make_array(kF2B));
  const std::vector<int>& ri = c.async_route;
  mx::array moe_route(ri.begin(), mx::Shape{1, static_cast<int>(ri.size())}, mx::int32);
  MoeLayer& ml = *c.async_moe;
  mx::array x2 = mx::reshape(*c.moe_xn, mx::Shape{1, c.m->H});
  mx::array routed = moe_block_relu2(x2, moe_route, *c.async_scores, up, down);
  routed = mx::reshape(mx::astype(routed, mx::bfloat16), mx::Shape{1, 1, c.m->H});
  mx::array su = linear(*c.moe_xn, ml.shared_up);
  mx::array sa = mx::square(mx::maximum(su, mx::astype(mx::array(0.0f), su.dtype())));
  mx::array sd = linear(sa, ml.shared_down);
  c.delta = routed + sd;
}

void layer_post(StepCtx& c, int l);

// S>1 batched router: same op order as nemotron_route but with [1,S,E] shapes.
static NemotronRoute nemotron_route_batch(const mx::array& x, const MoeLayer& w,
                                          int top_k, float routed_scale) {
  mx::array gates = linear(x, w.gate);  // [B,S,E] bf16
  const int last = static_cast<int>(gates.shape().size()) - 1;
  const int S = static_cast<int>(gates.shape(1));
  mx::array slots = mx::reshape(mx::arange(0, top_k, 1, mx::int32),
                                mx::Shape{1, 1, top_k});  // broadcasts over S
  static const float cached_scale = routed_scale;
  static auto fn = mx::compile(
      [last, k = top_k](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array orig = mx::sigmoid(mx::astype(in[0], mx::float32));
        mx::array scores = orig + in[1];
        mx::array inds_all = mx::argpartition(mx::negative(scores), k - 1, last);
        mx::array inds = mx::take_along_axis(inds_all, in[2], last);
        mx::array top = mx::take_along_axis(orig, inds, last);
        mx::array denom = mx::sum(top, last, /*keepdims=*/true) + mx::array(1e-20f);
        top = top / denom;
        top = top * mx::array(cached_scale);
        return {inds, top};
      },
      /*shapeless=*/true);
  std::vector<mx::array> out = fn({gates, w.gate_bias, slots});
  return {out[0], out[1]};  // [1,S,K] each
}

mx::array run_prefill_chunk(void* model_ptr, void* reader_ptr,
                            const std::vector<int32_t>& tokens) {
  Model& m = *static_cast<Model*>(model_ptr);
  ExpertReader& er = *static_cast<ExpertReader*>(reader_ptr);
  const int S = static_cast<int>(tokens.size());
  if (S <= 0) throw std::runtime_error("run_prefill_chunk: empty chunk");
  const int B = 1, H = m.H;

  // Batched token embed [1,S,H], same affine dequant as embed_token.
  mx::array idx(tokens.begin(), mx::Shape{S}, mx::int32);
  mx::array ew = mx::take(m.embed_w, idx, 0);
  mx::array es = mx::take(m.embed_s, idx, 0);
  mx::array eb = mx::take(m.embed_b, idx, 0);
  mx::array row = mx::dequantize(ew, es, eb, m.group_size, m.bits, "affine");
  mx::eval(row);
  mx::array h = mx::reshape(row, mx::Shape{B, S, H});

  size_t mamba_i = 0, attn_i = 0, moe_i = 0;
  const bool use_sorted = (S * m.top_k >= 64);

  // Read/compute overlap for long chunks (VEYSTRIX_CPP_NEMO_PREFILL_AHEAD, default on;
  // =0 restores the synchronous per-layer reads). The next MoE layer's whole expert set is read
  // on the dispatch queue while the GPU works on the current layer and the layers between.
  static const bool ahead_enabled = [] {
    const char* e = std::getenv("VEYSTRIX_CPP_NEMO_PREFILL_AHEAD");
    return !(e && std::strcmp(e, "0") == 0);
  }();
  static const int ahead_min_tokens = [] {
    const char* e = std::getenv("VEYSTRIX_CPP_NEMO_PREFILL_AHEAD_MIN");
    const int v = e ? std::atoi(e) : 192;
    return v > 0 ? v : 192;
  }();
  const bool ahead_on = ahead_enabled && S >= ahead_min_tokens;
  std::unique_ptr<ExpertReader::FullRows> ahead;
  size_t ahead_idx = static_cast<size_t>(-1);
  auto submit_ahead = [&](size_t moe_index) {
    if (!ahead_on || moe_index >= m.moe_layers.size()) return;
    const MoeLayer& nl = m.moe_layers[moe_index];
    ahead = er.submit_full_layer({nl.fc1_base + ".weight", nl.fc1_base + ".scales",
                                  nl.fc1_base + ".biases", nl.fc2_base + ".weight",
                                  nl.fc2_base + ".scales", nl.fc2_base + ".biases"});
    ahead_idx = moe_index;
  };
  submit_ahead(0);  // overlaps the embedding and the layers before the first MoE
  for (int l = 0; l < m.n_layers; ++l) {
    mx::array xn = mx::fast::rms_norm(
        h, m.kinds[l] == LayerKind::MAMBA_SSM
               ? m.mamba_layers[mamba_i].norm_w
               : (m.kinds[l] == LayerKind::FULL_ATTN_QSA
                      ? m.attn_layers[attn_i].norm_w
                      : m.moe_layers[moe_i].norm_w),
        m.eps);
      mx::array delta = empty_f32();
    if (m.kinds[l] == LayerKind::MAMBA_SSM) {
      MambaLayer& ml = m.mamba_layers[mamba_i++];
      MambaConfig mc;
      mc.hidden_size = H;
      mc.num_heads = m.spec.mamba_num_heads;
      mc.head_dim = m.spec.mamba_head_dim;
      mc.n_groups = m.spec.mamba_n_groups;
      mc.ssm_state_size = m.spec.mamba_ssm_state_size;
      mc.conv_kernel = m.spec.mamba_conv_kernel;
      mc.intermediate_size = m.spec.mamba_intermediate_size;
      mc.conv_dim = m.spec.mamba_conv_dim;
      mc.eps = m.eps;
      mc.time_step_min = m.spec.mamba_time_step_min;
      mc.time_step_max = m.spec.mamba_time_step_max;
      MambaCacheState& cache = m.mamba_caches[ml.cache_idx];
      MambaLayerOutput out = mamba_layer_prefill(xn, ml.weights, cache, mc);
      cache = MambaCacheState(std::move(out.conv_state), std::move(out.ssm_state));
      // Materialize cache state: it is NOT an ancestor of the logits, so a
      // lazy final SSM step would retain its whole graph (Sonnet Q1). This
      // changes no arithmetic, only when the graph is evaluated.
      mx::eval(cache.conv_state, cache.ssm_state);
      delta = std::move(out.output);  // [1,S,H]
    } else if (m.kinds[l] == LayerKind::FULL_ATTN_QSA) {
      AttnLayer& al = m.attn_layers[attn_i++];
      if (m.use_int8_kv)
        delta = nemo_attention_int8_forward(xn, al.weights, al.cfg, al.int8_cache);
      else
        delta = nemo_attention_forward(xn, al.weights, al.cfg, al.cache);
    } else {
      const size_t this_moe = moe_i;
      MoeLayer& ml = m.moe_layers[moe_i++];
      NemotronRoute route = nemotron_route_batch(xn, ml, m.top_k, m.routed_scale);
      mx::array ids_i = mx::astype(route.indices, mx::int32);
      mx::eval(ids_i);
      const int32_t* ip = ids_i.data<int32_t>();
      std::vector<int> ids(ip, ip + S * m.top_k);
      std::vector<int> unique = ids;
      std::sort(unique.begin(), unique.end());
      unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
      std::vector<int> local(ids.size());
      for (size_t i = 0; i < ids.size(); ++i)
        local[i] = static_cast<int>(
            std::lower_bound(unique.begin(), unique.end(), ids[i]) - unique.begin());
      mx::array scores2 = mx::reshape(route.scores, mx::Shape{S, m.top_k});
      const bool prefetched = ahead && ahead_idx == this_moe;
      // Prefetched: the full expert stack is already in memory, so the gather indexes it with the
      // real expert ids. Otherwise the compact first-occurrence stack and its local indices.
      mx::array route_idx = prefetched
          ? mx::array(ids.begin(), mx::Shape{S, m.top_k}, mx::int32)
          : mx::array(local.begin(), mx::Shape{S, m.top_k}, mx::int32);
      std::optional<Q4Projection> up_o, down_o;
      if (prefetched) {
        ahead->wait();
        up_o.emplace(ahead->take(0), ahead->take(1), ahead->take(2));
        down_o.emplace(ahead->take(3), ahead->take(4), ahead->take(5));
        ahead.reset();
      } else {
        up_o.emplace(er.rows_u32(ml.fc1_base + ".weight", unique),
                     er.rows_bf16(ml.fc1_base + ".scales", unique),
                     er.rows_bf16(ml.fc1_base + ".biases", unique));
        down_o.emplace(er.rows_u32(ml.fc2_base + ".weight", unique),
                       er.rows_bf16(ml.fc2_base + ".scales", unique),
                       er.rows_bf16(ml.fc2_base + ".biases", unique));
      }
      Q4Projection& up = *up_o;
      Q4Projection& down = *down_o;
      mx::array x2 = mx::reshape(xn, mx::Shape{S, H});
      mx::array routed = use_sorted
          ? moe_block_relu2_sorted(x2, route_idx, scores2, up, down)
          : moe_block_relu2(x2, route_idx, scores2, up, down);
      routed = mx::reshape(mx::astype(routed, mx::bfloat16), mx::Shape{B, S, H});
      mx::array su = linear(xn, ml.shared_up);
      mx::array sa = mx::square(mx::maximum(su, mx::astype(mx::array(0.0f), su.dtype())));
      mx::array sd = linear(sa, ml.shared_down);
      delta = routed + sd;
      // The graph for this layer is built but not run yet: start reading the next MoE layer's
      // experts now, so the SSD works while the next eval keeps the GPU busy.
      submit_ahead(moe_i);
    }
    h = h + delta;
    }
  // Python applies norm_f + lm_head to the LAST hidden row only (M=1 GEMV),
  // not a GEMM over all S rows: the latter rounds differently and allocates
  // ~512 MiB of logits for a 2048-token chunk.
  mx::array last = mx::slice(h, {0, S - 1, 0}, {B, S, H}, {1, 1, 1});
  mx::array fused = mx::fast::rms_norm(last, m.norm_f_w, m.eps);
  mx::array lg = linear(fused, m.lm_head);
  return mx::reshape(lg, mx::Shape{B, static_cast<int>(lg.shape(lg.shape().size() - 1))});
}

// P5-overlap driver (see nemotron_runtime.hpp). Defined here because StepCtx
// and layer_pre/layer_pre_finish/layer_post live in this TU.
std::vector<mx::array> run_step_pipelined(
    void* model_ptr, void* reader_ptr, const std::vector<int32_t>& tokens,
    const CacheSwapFn& swap_in, const CacheSwapFn& swap_out) {
  Model& m = *static_cast<Model*>(model_ptr);
  ExpertReader& er = *static_cast<ExpertReader*>(reader_ptr);
  const int N = static_cast<int>(tokens.size());
  std::vector<StepCtx> ctxs;
  ctxs.reserve(N);
  for (int i = 0; i < N; ++i) {
    swap_in(i);
    ctxs.emplace_back(m, er, embed_token(m, tokens[i]));
    ctxs.back().mamba_i = ctxs.back().attn_i = ctxs.back().moe_i = 0;
    ctxs.back().async_defer = true;
    swap_out(i);
  }
  for (int l = 0; l < m.n_layers; ++l) {
    // Submit phase: every session's expert reads issued before any wait.
    for (int i = 0; i < N; ++i) {
      swap_in(i);
      layer_pre(ctxs[i], l);
      swap_out(i);
    }
    // Finish phase: wait only for THIS session's ticket, then build+eval, so
    // GPU work of one session overlaps SSD reads still in flight for another.
    for (int i = 0; i < N; ++i) {
      swap_in(i);
      layer_pre_finish(ctxs[i]);
      layer_post(ctxs[i], l);
      swap_out(i);
    }
  }
  std::vector<mx::array> logits;
  logits.reserve(N);
  for (int i = 0; i < N; ++i) {
    swap_in(i);
    mx::array fused = mx::fast::rms_norm(ctxs[i].h, m.norm_f_w, m.eps);
    mx::array lg = linear(fused, m.lm_head);
    logits.push_back(mx::reshape(
        lg, mx::Shape{1, static_cast<int>(lg.shape(lg.shape().size() - 1))}));
    swap_out(i);
  }
  return logits;
}

void layer_post(StepCtx& c, int l) {
  Model& m = *c.m;
  c.h = c.h + *c.delta;
  c.delta.reset();
  if (m.dump_layers)
    std::fprintf(stderr, "[NEMOLAYER step L=%02d] %s\n", l,
                 digest_array(c.h).c_str());
}

mx::array run_step_cursor(Model& m, ExpertReader& er, int token) {
  StepCtx c(m, er, embed_token(m, token));  // [1,1,H] bf16
  for (int l = 0; l < m.n_layers; ++l) {
    layer_pre(c, l);
    layer_post(c, l);
  }
  mx::array fused = mx::fast::rms_norm(c.h, m.norm_f_w, m.eps);
  mx::array logits = linear(fused, m.lm_head);
  return mx::reshape(logits, mx::Shape{1, logits.shape(logits.shape().size() - 1)});
}

// N=1 gate generation loop: copy of run_generation() with run_step -> cursor.
// Emits the same `step=... in=... out=... sha256=...` stream, so a paired run
// against the unmodified nemotron_generate binary gates token+digest identity.
std::vector<StepRecord> run_generation_cursor(const std::string& ckpt,
                                              const Config& cfg,
                                              int num_tokens) {
  Model m = load_model(ckpt, cfg);
  if (std::getenv("NEMOTRON_LOAD_ONLY")) return {};
  ExpertReader er(ckpt, cfg);
  const std::vector<int> prompt = split_ints(get_str(cfg, "prompt_ids"));
  if (prompt.empty()) throw std::runtime_error("empty prompt_ids in config");
  cppprof::profiler().begin_step();
  mem_telemetry("post-load");

  {
    const char* en = std::getenv("VEYSTRIX_CPP_NEMO_POOL");
    if (en && std::strcmp(en, "1") == 0) {
      const char* gm = std::getenv("VEYSTRIX_CPP_NEMO_POOL_GATHER");
      const bool compact = !(gm && std::strcmp(gm, "pooled") == 0);
      int slots = 64;
      const char* ns = std::getenv("VEYSTRIX_CPP_NEMO_POOL_SLOTS");
      if (ns) slots = std::atoi(ns);
      const bool nocache = nemo_pool_nocache_from_env();
      m.expert_pool = std::make_unique<PooledExpertTensors>(
          ckpt, cfg, m.n_layers, m.top_k, slots, compact, nocache);
    }
  }

  std::vector<StepRecord> records;
  records.reserve(num_tokens);
  std::optional<mx::array> logits;
  for (const int tok : prompt) logits = run_step_cursor(m, er, tok);
  if (!logits) throw std::runtime_error("no prefill logits");
  int token = prompt.back();

  for (int step = 0; step < num_tokens; ++step) {
    if (step > 0) {
      cppprof::profiler().begin_step();
      logits = run_step_cursor(m, er, token);
    }
    mx::array next = sample_greedy(*logits);
    mx::eval(next);
    int next_id = static_cast<int>(next.data<uint32_t>()[0]);
    const bool want_digest = !cppprof::profiler().no_digest;
    std::string digest = want_digest ? digest_array(*logits) : std::string("NO_DIGEST");
    records.push_back({step, token, next_id, digest});
    token = next_id;
  }
  return records;
}

// --------------------------------------------------------------------------- //
// Multi-session driver: ONE shared Model (weights), N isolated state objects.  //
// --------------------------------------------------------------------------- //
static Model* g_model = nullptr;
static ExpertReader* g_er = nullptr;
static uint64_t g_decode_bytes = 0;

static void swap_in(Model& m, SessionCacheState& sc) {
  m.mamba_caches = sc.mamba_caches;
  for (std::size_t k = 0; k < m.attn_layers.size() && k < sc.attn_int8.size(); ++k) {
    m.attn_layers[k].int8_cache = sc.attn_int8[k];
    m.attn_layers[k].cache = sc.attn_f32[k];
  }
}

static void swap_out(Model& m, SessionCacheState& sc) {
  sc.mamba_caches = m.mamba_caches;
  for (std::size_t k = 0; k < m.attn_layers.size() && k < sc.attn_int8.size(); ++k) {
    sc.attn_int8[k] = m.attn_layers[k].int8_cache;
    sc.attn_f32[k] = m.attn_layers[k].cache;
  }
}

struct NemotronSession {
  int id = 0;
  std::vector<int> prompt;
  int num_tokens = 0;
  int token = 0;
  int produced = 0;
  SessionCacheState cache;
  std::optional<mx::array> logits;
  bool logits_ready = false;
};

// One cooperative decode step for one session; MsScheduler invokes it on the
// single driver thread. It touches ONLY this session's mutable state.
static bool decode_step(void* state, int /*step*/, unsigned long long /*seed*/) {
  NemotronSession* s = static_cast<NemotronSession*>(state);
  Model& m = *g_model;
  ExpertReader& er = *g_er;

  if (!s->logits_ready) {
    cppprof::profiler().begin_step();
    swap_in(m, s->cache);
    s->logits = run_step(m, er, s->token);   // WHOLE token, verbatim primitive
    swap_out(m, s->cache);
    g_decode_bytes += cppprof::profiler().bytes_requested;
    s->logits_ready = true;
  }

  mx::array next = sample_greedy(*s->logits);
  mx::eval(next);
  int next_id = static_cast<int>(next.data<uint32_t>()[0]);
  std::string digest = digest_array(*s->logits);
  std::fprintf(stderr, "session=%d step=%d token=%d sha256=%s\n",
               s->id, s->produced, next_id, digest.c_str());
  s->token = next_id;
  s->logits_ready = false;
  ++s->produced;
  mem_telemetry("decode-step");
  return s->produced >= s->num_tokens;
}

static void check_same_config(const Config& a, const Config& b) {
  for (const auto& kv : a) {
    if (kv.first.rfind("prompt_", 0) == 0) continue;
    auto it = b.find(kv.first);
    if (it == b.end() || it->second != kv.second)
      throw std::runtime_error("config mismatch on " + kv.first);
  }
}

int multisession_main(int argc, char** argv) {
  // argv[0]=program, argv[1]="multi", argv[2]=ckpt, argv[3]=N, argv[4]=tokens
  if (argc < 6) {
    std::fprintf(stderr,
                 "usage: %s multi <checkpoint-dir> <num_sessions> <tokens> "
                 "<config0> <config1> ...\n", argv[0]);
    return 2;
  }
  const std::string ckpt = argv[2];
  const int num_sessions = std::atoi(argv[3]);
  const int tokens = std::atoi(argv[4]);
  if (num_sessions < 1 || tokens < 1 || argc != 5 + num_sessions) {
    std::fprintf(stderr, "invalid num_sessions/tokens/config count\n");
    return 2;
  }
  try {
    std::vector<Config> configs;
    for (int i = 0; i < num_sessions; ++i) {
      Config c = read_config(argv[5 + i]);
      if (c.empty())
        throw std::runtime_error("empty config: " + std::string(argv[5 + i]));
      if (get_int(c, "prompt_tokens", 0) == 0)
        throw std::runtime_error("config has no prompt_ids: " + std::string(argv[5 + i]));
      if (i > 0) check_same_config(configs[0], c);
      configs.push_back(std::move(c));
    }

    cppprof::profiler().init();
    Model model = load_model(ckpt, configs[0]);
    ExpertReader er(ckpt, configs[0]);
    g_model = &model;
    g_er = &er;

    const int n_mamba = static_cast<int>(model.mamba_layers.size());
    const int n_attn = static_cast<int>(model.attn_layers.size());
    const int block_tokens = 4096;  // production Nemotron attention BLOCK_TOKENS

    std::vector<NemotronSession> sessions;
    sessions.reserve(num_sessions);
    for (int i = 0; i < num_sessions; ++i) {
      NemotronSession s;
      s.id = i;
      s.num_tokens = tokens;
      s.prompt = split_ints(get_str(configs[i], "prompt_ids"));
      if (s.prompt.empty()) throw std::runtime_error("empty prompt");
      s.cache.init(n_mamba, model.spec.mamba_conv_kernel,
                   model.spec.mamba_conv_dim, model.spec.mamba_num_heads,
                   model.spec.mamba_head_dim, model.spec.mamba_ssm_state_size,
                   n_attn, block_tokens);
      // Exclusive sequential prefill (matches Python chunk_tokens=1 and the
      // solo run_generation: one run_step per prompt token).
      for (const int tok : s.prompt) {
        swap_in(model, s.cache);
        s.logits = run_step(model, er, tok);
        swap_out(model, s.cache);
      }
      s.token = s.prompt.back();
      s.logits_ready = true;
      std::fprintf(stderr, "[prefill] session=%d prompt_tokens=%zu\n",
                   s.id, s.prompt.size());
      sessions.push_back(std::move(s));
    }

    SchedulerConfig sched_cfg;
    sched_cfg.prefill_exclusive = true;
    sched_cfg.max_skew_steps = 1;
    if (const char* ms = std::getenv("VEYSTRIX_CPP_MS_MAX_SKEW")) {
      const int v = std::atoi(ms);
      if (v < 1) throw std::runtime_error("VEYSTRIX_CPP_MS_MAX_SKEW must be >= 1");
      sched_cfg.max_skew_steps = v;
    }
    MsScheduler scheduler(sched_cfg);
    for (int i = 0; i < num_sessions; ++i) {
      ScheduledSession ss;
      ss.id = i;
      ss.state = &sessions[i];
      ss.session = nullptr;  // Nemotron decode is greedy: no PRNG/position state
      ss.step_fn = &decode_step;
      ss.seed = 0;
      ss.prefill_steps = 0;  // prefill was already exclusive
      scheduler.add(ss);
    }

    // bytes_requested (profiler) is disabled by default, so report the always-
    // available physical disk-read delta across the decode window instead.
    const uint64_t phys_before = cppprof::physical_disk_bytes();
    const auto t0 = std::chrono::steady_clock::now();
    scheduler.drive(tokens + num_sessions + 4);
    const double decode_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    const uint64_t phys_after = cppprof::physical_disk_bytes();
    const uint64_t bytes_phys = phys_after > phys_before ? phys_after - phys_before : 0;

    int total = 0;
    for (const auto& s : sessions) {
      total += s.produced;
      if (s.produced != tokens)
        std::fprintf(stderr, "[warn] session=%d produced %d/%d steps\n",
                     s.id, s.produced, tokens);
    }
    const double agg = static_cast<double>(total);
    std::fprintf(stderr,
                 "[bench] sessions=%d tokens_each=%d agg_tokens=%d "
                 "decode_s=%.4f tok_s=%.4f bytes_physical=%llu "
                 "bytes_per_agg_token=%.2f rounds=%d max_skew=%d\n",
                 num_sessions, tokens, total, decode_s,
                 decode_s > 0 ? agg / decode_s : 0.0,
                 (unsigned long long)bytes_phys,
                 agg > 0 ? static_cast<double>(bytes_phys) / agg : 0.0,
                 scheduler.rounds(), sched_cfg.max_skew_steps);
    std::fprintf(stderr, "[done] sessions=%d tokens_each=%d steps=%d rounds=%d\n",
                 num_sessions, tokens, total, scheduler.rounds());
    return total == num_sessions * tokens ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "nemotron_runtime multi failed: %s\n", e.what());
    return 1;
  }
}

// --------------------------------------------------------------------------- //
// Stage 1 batched shared-read decode (opt-in VEYSTRIX_CPP_NEMO_BATCHED=1).
// All sessions advance one decode token in strict layer lockstep. Mamba and
// attention layers run per-session (they need that session's swapped-in caches);
// MoE layers are batched across all sessions: routes are unioned, expert rows
// read ONCE, and one moe_block_relu2 runs over the [N,K] batch. Stage 0b proved
// the MoE kernel is bit-invariant at T=N, so tokens/digests stay solo-exact.
// --------------------------------------------------------------------------- //
int batched_multisession_main(int argc, char** argv) {
  // argv: nemotron_runtime multi <ckpt> <N> <tokens> <config0> ...
  if (argc < 6) {
    std::fprintf(stderr, "usage: %s multi <ckpt> <N> <tokens> <config0> ...\n", argv[0]);
    return 2;
  }
  const std::string ckpt = argv[2];
  const int num_sessions = std::atoi(argv[3]);
  const int tokens = std::atoi(argv[4]);
  if (num_sessions < 1 || tokens < 1 || argc != 5 + num_sessions) return 2;
  try {
    std::vector<Config> configs;
    for (int i = 0; i < num_sessions; ++i) {
      Config c = read_config(argv[5 + i]);
      if (c.empty()) throw std::runtime_error("empty config");
      if (i > 0) check_same_config(configs[0], c);
      configs.push_back(std::move(c));
    }

    cppprof::profiler().init();
    Model model = load_model(ckpt, configs[0]);
    ExpertReader er(ckpt, configs[0]);
    const int n_mamba = static_cast<int>(model.mamba_layers.size());
    const int n_attn = static_cast<int>(model.attn_layers.size());
    const int n_layers = model.n_layers;
    const int K = model.top_k;
    const int H = model.H;
    const int block_tokens = 4096;

    // Per-session mutable state: cache + persistent StepCtx (h, counters).
    struct BSess {
      NemotronSession base;
      StepCtx* ctx = nullptr;
    };
    std::vector<BSess> sess(num_sessions);
    std::vector<std::unique_ptr<StepCtx>> ctxs;
    ctxs.reserve(num_sessions);
    for (int i = 0; i < num_sessions; ++i) {
      sess[i].base.id = i;
      sess[i].base.num_tokens = tokens;
      sess[i].base.prompt = split_ints(get_str(configs[i], "prompt_ids"));
      if (sess[i].base.prompt.empty()) throw std::runtime_error("empty prompt");
      sess[i].base.cache.init(n_mamba, model.spec.mamba_conv_kernel,
          model.spec.mamba_conv_dim, model.spec.mamba_num_heads,
          model.spec.mamba_head_dim, model.spec.mamba_ssm_state_size,
          n_attn, block_tokens);
      // Exclusive sequential prefill (identical to the solo path).
      for (const int tok : sess[i].base.prompt) {
        swap_in(model, sess[i].base.cache);
        sess[i].base.logits = run_step(model, er, tok);
        swap_out(model, sess[i].base.cache);
      }
      sess[i].base.token = sess[i].base.prompt.back();
      sess[i].base.logits_ready = true;
      auto c = std::make_unique<StepCtx>(model, er, embed_token(model, sess[i].base.token));
      c->batched = true;
      sess[i].ctx = c.get();
      ctxs.push_back(std::move(c));
      std::fprintf(stderr, "[prefill] session=%d prompt_tokens=%zu\n",
                   i, sess[i].base.prompt.size());
    }

    // One batched MoE layer for all sessions. All sessions have already run
    // layer_pre for this layer (routes/xn recorded). Returns each session's
    // MoE delta (routed + shared expert) via c.delta.
    auto batched_moe_layer = [&](int l) {
      std::vector<int> union_ids;
      for (int i = 0; i < num_sessions; ++i)
        union_ids.insert(union_ids.end(), sess[i].ctx->moe_ids.begin(),
                         sess[i].ctx->moe_ids.end());
      std::sort(union_ids.begin(), union_ids.end());
      union_ids.erase(std::unique(union_ids.begin(), union_ids.end()), union_ids.end());
      const int U = static_cast<int>(union_ids.size());
      MoeLayer& ml = model.moe_layers[sess[0].ctx->moe_i - 1];  // layer_pre incremented

      // Kernel guard (same rationale as Qwen Stage 1): keep the per-entry
      // gather_qmv kernel. Small union (B/U >= 4) would switch gather_qmm to a
      // different reduction path; fall back to per-session solo calls then.
      const bool split_for_kernel = (num_sessions * K >= 16) && (U > 0) &&
                                    (num_sessions * K / U >= 4);
      if (split_for_kernel) {
        for (int i = 0; i < num_sessions; ++i) {
          StepCtx& c = *sess[i].ctx;
          std::vector<int> u = c.moe_ids;
          std::sort(u.begin(), u.end());
          u.erase(std::unique(u.begin(), u.end()), u.end());
          std::vector<int> loc(K);
          for (int k = 0; k < K; ++k)
            loc[k] = (int)(std::lower_bound(u.begin(), u.end(), c.moe_ids[k]) - u.begin());
          mx::array lu(loc.data(), mx::Shape{1, K}, mx::int32);
          Q4Projection up(er.rows_u32(ml.fc1_base + ".weight", u),
                          er.rows_bf16(ml.fc1_base + ".scales", u),
                          er.rows_bf16(ml.fc1_base + ".biases", u));
          Q4Projection dn(er.rows_u32(ml.fc2_base + ".weight", u),
                          er.rows_bf16(ml.fc2_base + ".scales", u),
                          er.rows_bf16(ml.fc2_base + ".biases", u));
          mx::array x2 = mx::reshape(*c.moe_xn, mx::Shape{1, H});
          mx::array routed = moe_block_relu2(x2, lu, *c.moe_scores, up, dn);
          routed = mx::reshape(mx::astype(routed, mx::bfloat16), mx::Shape{1, 1, H});
          mx::array su = linear(*c.moe_xn, ml.shared_up);
          mx::array sa = mx::square(mx::maximum(su, mx::astype(mx::array(0.0f), su.dtype())));
          mx::array sd = linear(sa, ml.shared_down);
          c.delta = routed + sd;
        }
        return;
      }

      // Union read ONCE for all sessions, then one batched moe_block_relu2.
      Q4Projection up(er.rows_u32(ml.fc1_base + ".weight", union_ids),
                      er.rows_bf16(ml.fc1_base + ".scales", union_ids),
                      er.rows_bf16(ml.fc1_base + ".biases", union_ids));
      Q4Projection dn(er.rows_u32(ml.fc2_base + ".weight", union_ids),
                      er.rows_bf16(ml.fc2_base + ".scales", union_ids),
                      er.rows_bf16(ml.fc2_base + ".biases", union_ids));
      std::vector<int32_t> ridx((size_t)num_sessions * K);
      for (int i = 0; i < num_sessions; ++i) {
        StepCtx& c = *sess[i].ctx;
        for (int k = 0; k < K; ++k)
          ridx[i*K+k] = (int32_t)(std::lower_bound(union_ids.begin(), union_ids.end(),
                                                  c.moe_ids[k]) - union_ids.begin());
      }
      mx::array RIDX(ridx.data(), mx::Shape{num_sessions, K}, mx::int32);
      std::vector<mx::array> xs, ss;
      xs.reserve(num_sessions); ss.reserve(num_sessions);
      for (int i = 0; i < num_sessions; ++i) {
        xs.push_back(mx::reshape(*sess[i].ctx->moe_xn, mx::Shape{1, H}));
        ss.push_back(*sess[i].ctx->moe_scores);
      }
      mx::array X = mx::concatenate(xs, 0);   // [N,H] bf16
      mx::array SC = mx::concatenate(ss, 0);  // [N,K] f32
      mx::array routed = moe_block_relu2(X, RIDX, SC, up, dn);  // [N,H]
      mx::eval(routed);
      // Scatter: each session's routed row + its own shared expert.
      for (int i = 0; i < num_sessions; ++i) {
        StepCtx& c = *sess[i].ctx;
        mx::array row = mx::reshape(
            mx::astype(mx::slice(routed, mx::Shape{i, 0}, mx::Shape{i+1, H}),
                       mx::bfloat16),
            mx::Shape{1, 1, H});
        mx::array su = linear(*c.moe_xn, ml.shared_up);
        mx::array sa = mx::square(mx::maximum(su, mx::astype(mx::array(0.0f), su.dtype())));
        mx::array sd = linear(sa, ml.shared_down);
        c.delta = row + sd;
      }
    };

    // ---- lockstep decode rounds ------------------------------------------
    const uint64_t phys_before = cppprof::physical_disk_bytes();
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::optional<mx::array>> logits(num_sessions);
    for (int i = 0; i < num_sessions; ++i) logits[i] = sess[i].base.logits.value();
    int active = num_sessions;
    int rounds_run = 0;
    for (int rnd = 0; rnd < tokens + 2 && active > 0; ++rnd) {
      rounds_run = rnd + 1;
      // 1) sample + emit for every active session.
      std::vector<int> alive;
      for (int i = 0; i < num_sessions; ++i) {
        if (sess[i].base.produced >= sess[i].base.num_tokens) continue;
        mx::array next = sample_greedy(*logits[i]);
        mx::eval(next);
        int next_id = static_cast<int>(next.data<uint32_t>()[0]);
        std::string digest = digest_array(*logits[i]);
        std::fprintf(stderr, "session=%d step=%d token=%d sha256=%s\n",
                     i, sess[i].base.produced, next_id, digest.c_str());
        sess[i].base.token = next_id;
        ++sess[i].base.produced;
        if (sess[i].base.produced < sess[i].base.num_tokens) alive.push_back(i);
      }
      if (alive.empty()) break;
      active = (int)alive.size();

      // 2) seed each active session's StepCtx on its NEW token.
      for (int i : alive) {
        StepCtx& c = *sess[i].ctx;
        c.h = embed_token(model, sess[i].base.token);
        c.mamba_i = c.attn_i = c.moe_i = 0;
        c.layer = -1;
      }

      // 3) all layers in lockstep: mixers per-session, MoE batched.
      for (int l = 0; l < n_layers; ++l) {
        LayerKind kind = model.kinds[l];
        if (kind == LayerKind::MAMBA_SSM || kind == LayerKind::FULL_ATTN_QSA) {
          for (int i : alive) {
            swap_in(model, sess[i].base.cache);
            layer_pre(*sess[i].ctx, l);
            layer_post(*sess[i].ctx, l);
            swap_out(model, sess[i].base.cache);
          }
        } else {
          for (int i : alive) layer_pre(*sess[i].ctx, l);  // records routes
          batched_moe_layer(l);
          for (int i : alive) layer_post(*sess[i].ctx, l);
        }
      }
      // 4) final head per active session.
      for (int i : alive) {
        StepCtx& c = *sess[i].ctx;
        mx::array fused = mx::fast::rms_norm(c.h, model.norm_f_w, model.eps);
        mx::array lg = linear(fused, model.lm_head);
        logits[i] = mx::reshape(lg, mx::Shape{1, lg.shape(lg.shape().size()-1)});
      }
    }
    const double decode_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    const uint64_t phys_after = cppprof::physical_disk_bytes();
    const uint64_t bytes_phys = phys_after > phys_before ? phys_after - phys_before : 0;
    int total = 0;
    for (auto& s : sess) total += s.base.produced;
    const double agg = (double)total;
    std::fprintf(stderr,
                 "[bench] sessions=%d tokens_each=%d agg_tokens=%d decode_s=%.4f "
                 "tok_s=%.4f bytes_physical=%llu bytes_per_agg_token=%.2f rounds=%d max_skew=0\n",
                 num_sessions, tokens, total, decode_s,
                 decode_s > 0 ? agg/decode_s : 0.0, (unsigned long long)bytes_phys,
                 agg > 0 ? (double)bytes_phys/agg : 0.0, rounds_run);
    return total == num_sessions * tokens ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "batched Nemotron failed: %s\n", e.what());
    return 1;
  }
}

// N=1 cursor gate entry point (same CLI as nemotron_generate).
int nemotron_gate_main(int argc, char** argv) {
  const char* e = std::getenv("VEYSTRIX_CPP_PHASE_F_CURSOR");
  g_cursor_enabled = e && std::strcmp(e, "1") == 0;
  if (g_cursor_enabled)
    std::fprintf(stderr, "[phase-f] nemotron split-layer cursor decode ENABLED\n");

  // Reproduce nemotron_generate::main's argument handling, but route decode
  // through run_step_cursor when the env flag is set.
  cppprof::profiler().init();
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <checkpoint-dir> <reference.json> <num_tokens> "
                 "[flat-config]\n", argv[0]);
    return 2;
  }
  const std::string ckpt = argv[1];
  const std::string reference = argv[2];
  const int num_tokens = std::atoi(argv[3]);
  std::string config_path = argc >= 5 ? argv[4] : "";
  if (config_path.empty()) {
    size_t slash = reference.find_last_of('/');
    config_path = (slash == std::string::npos ? std::string(".")
                                              : reference.substr(0, slash)) +
                  "/nemotron_config.txt";
  }
  try {
    Config cfg = read_config(config_path);
    if (cfg.empty()) throw std::runtime_error("empty flat config: " + config_path);
    if (get_int(cfg, "prompt_tokens", 0) == 0)
      throw std::runtime_error("flat config has no prompt_ids");
    std::vector<StepRecord> records;
    if (g_cursor_enabled)
      records = run_generation_cursor(ckpt, cfg, num_tokens);
    else
      records = run_generation(ckpt, cfg, num_tokens);
    for (const auto& r : records)
      std::printf("step=%d in=%d out=%d sha256=%s\n", r.step, r.token_in,
                  r.token_out, r.digest.c_str());
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "nemotron_runtime gate failed: %s\n", ex.what());
    return 1;
  }
}

}  // namespace veystrix::engine::nemotron_runtime

int main(int argc, char** argv) {
  if (argc >= 2 && std::strcmp(argv[1], "multi") == 0) {
    const char* b = std::getenv("VEYSTRIX_CPP_NEMO_BATCHED");
    if (b && std::strcmp(b, "1") == 0)
      return veystrix::engine::nemotron_runtime::batched_multisession_main(argc, argv);
    return veystrix::engine::nemotron_runtime::multisession_main(argc, argv);
  }
  // Default: N=1 gate (same CLI as nemotron_generate).
  return veystrix::engine::nemotron_runtime::nemotron_gate_main(argc, argv);
}
