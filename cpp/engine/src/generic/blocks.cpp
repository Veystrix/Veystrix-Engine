// Block library for the generic runtime. Every block mirrors an mlx_lm module
// op-for-op; comments name the Python source line it reproduces.
#include "generic/generic_model.hpp"

#include <mlx/compile.h>
#include <mlx/fast.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <atomic>
#include <functional>
#include <mutex>
#include <set>
#include <stdexcept>

#include "cpp_profile.hpp"
#include "engine.hpp"
#include "generic/expert_stream.hpp"
#include "gpu_gate.hpp"
#include "row_fetch.hpp"

namespace veystrix::engine::generic {

namespace {

// mlx_lm weak-scalar convention: python float -> float32 -> array dtype.
mx::array weak(double v, mx::Dtype dt) { return mx::array(static_cast<float>(v), dt); }

// ---- compiled elementwise functions (identical graphs to the Python ones) ----
// mlx.nn.silu: @partial(mx.compile, shapeless=True) x * mx.sigmoid(x)
mx::array silu_c(const mx::array& x) {
  static auto fn = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        return {mx::multiply(in[0], mx::sigmoid(in[0]))};
      },
      true);
  return fn({x})[0];
}
// mlx_lm.models.activations.swiglu(gate, x): nn.silu(gate) * x  (compiled)
int env_int(const char* name, int fallback) {
  const char* e = std::getenv(name);
  return (e && *e) ? std::atoi(e) : fallback;
}
// VEYSTRIX_GENERIC_PIPE=1: one-token decode launches each routed expert's
// gate/up -> swiglu -> down as soon as its rows have landed (rowfetch), while the
// other experts are still being read. Same kernels per (token, expert) as one
// gather_qmm over all K, combined in router order: bit-identical.
bool pipe_enabled() {
  static const bool on = env_int("VEYSTRIX_GENERIC_PIPE", 0) == 1;
  return on;
}
// VEYSTRIX_GENERIC_SPEC_TOP=Q (decode): once this layer's rows have landed,
// warm the page cache with the next layer's Q highest-scored guessed experts
// (next router on this layer's MoE input). Real reads use the real route.
int spec_top() {
  static const int q = env_int("VEYSTRIX_GENERIC_SPEC_TOP", 0);
  return q;
}
// VEYSTRIX_GENERIC_PREFILL_PIPE=1: in prefill chunks that route to (nearly) every
// expert, the next layer's whole expert stack is read in the background while this
// layer computes, and used as the full stack (identity remap: same bytes, same result).
// VEYSTRIX_GENERIC_SPEC_FETCH=Q (one-token decode): once this layer's rows are in,
// the next layer's Q highest-scored guessed experts are copied/read into that
// layer's stack buffers in the background (while the GPU computes); at the next
// layer only the experts the real route adds are read. Same file bytes, unsorted
// gather: bit-identical.
int spec_fetch() {
  // Default 4 (Ornith pooled 13.29 vs 12.98 tok/s for page-cache prefetch only); 0 turns it off.
  static const int q = env_int("VEYSTRIX_GENERIC_SPEC_FETCH", 4);
  return q;
}
bool prefill_pipe() {
  // Default on (Ornith 8K-token prompt: TTFT ~47 -> ~30 s); VEYSTRIX_GENERIC_PREFILL_PIPE=0 turns it off.
  static const bool on = env_int("VEYSTRIX_GENERIC_PREFILL_PIPE", 1) == 1;
  return on;
}

mx::array swiglu_c(const mx::array& gate, const mx::array& x) {
  static auto fn = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array s = mx::multiply(in[0], mx::sigmoid(in[0]));
        return {mx::multiply(s, in[1])};
      },
      true);
  return fn({gate, x})[0];
}
// qwen3_next._precise_swiglu(h, gate, x) (compiled)
mx::array precise_swiglu_c(const mx::array& h, const mx::array& gate, const mx::array& x) {
  static auto fn = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array g = mx::astype(in[1], mx::float32);
        g = mx::multiply(g, mx::sigmoid(g));
        mx::array xf = mx::astype(in[2], mx::float32);
        return {mx::astype(mx::multiply(g, xf), in[0].dtype())};
      },
      true);
  return fn({h, gate, x})[0];
}
// gated_delta.compute_g(A_log, a, dt_bias) (compiled):
//   exp(-exp(A_log.f32) * softplus(a + dt_bias)); softplus = logaddexp(x, 0)
mx::array compute_g_c(const mx::array& A_log, const mx::array& a, const mx::array& dt_bias) {
  static auto fn = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array A = mx::exp(mx::astype(in[0], mx::float32));
        mx::array ad = mx::add(in[1], in[2]);
        mx::array sp = mx::logaddexp(ad, mx::array(0.0f, ad.dtype()));
        return {mx::exp(mx::multiply(mx::negative(A), sp))};
      },
      true);
  return fn({A_log, a, dt_bias})[0];
}

std::string tensors_of(const BlockCtx& c) {
  return Loader::expand(c.params.at("tensors").str("block tensors"), c.layer);
}

int64_t opt_bytes(const std::optional<mx::array>& a) { return a ? static_cast<int64_t>(a->nbytes()) : 0; }

mx::array split_part(const mx::array& x, int axis, int from, int to) {
  mx::Shape start(x.ndim(), 0), stop(x.shape());
  const int ax = axis < 0 ? axis + x.ndim() : axis;
  start[ax] = from;
  stop[ax] = to;
  return mx::slice(x, start, stop);
}

// =============================================================================
// linear.gdn — mlx_lm qwen3_5.GatedDeltaNet (separate in_proj_qkv/z/b/a)
// =============================================================================
class GdnMixer final : public Mixer {
 public:
  explicit GdnMixer(const BlockCtx& c) {
    const std::string p = tensors_of(c);
    hk_ = c.params.at("num_k_heads").i32("num_k_heads");
    hv_ = c.params.at("num_v_heads").i32("num_v_heads");
    dk_ = c.params.at("head_k_dim").i32("head_k_dim");
    dv_ = c.params.at("head_v_dim").i32("head_v_dim");
    kconv_ = c.params.get_i64("conv_kernel", 4);
    eps_ = static_cast<float>(c.params.get_num("eps", 1e-6));
    if (hv_ % hk_ != 0) throw std::runtime_error("linear.gdn: num_v_heads % num_k_heads != 0");
    if (dk_ % 32 != 0) throw std::runtime_error("linear.gdn: head_k_dim must be a multiple of 32 (kernel)");
    key_dim_ = hk_ * dk_;
    value_dim_ = hv_ * dv_;
    conv_dim_ = 2 * key_dim_ + value_dim_;
    in_qkv_ = c.loader.linear(p + ".in_proj_qkv");
    in_z_ = c.loader.linear(p + ".in_proj_z");
    in_b_ = c.loader.linear(p + ".in_proj_b");
    in_a_ = c.loader.linear(p + ".in_proj_a");
    out_ = c.loader.linear(p + ".out_proj");
    mx::array cw = c.loader.tensor(p + ".conv1d.weight");
    // mlx_lm sanitize: HF layout [C, 1, K] -> MLX [C, K, 1] (moveaxis(2, 1)).
    if (cw.shape(-1) != 1) cw = mx::moveaxis(cw, 2, 1);
    conv_w_ = cw;
    dt_bias_ = c.loader.tensor(p + ".dt_bias");
    A_log_ = c.loader.tensor(p + ".A_log");
    norm_w_ = c.loader.tensor(p + ".norm.weight");
    std::vector<mx::array> all{conv_w_, dt_bias_, A_log_, norm_w_};
    mx::eval(all);
  }
  std::string type() const override { return "linear.gdn"; }

  void reset(LayerState& st) const override {
    // ArraysCache(size=2): None, None -> zeros of the same dtype mlx_lm builds.
    st.arrays.clear();
    st.kv.reset();
    st.tokens = 0;
  }

  mx::array forward(const mx::array& x, LayerState& st) const override {
    const int B = x.shape(0), S = x.shape(1);
    mx::array qkv = in_qkv_(x);
    mx::array z = mx::reshape(in_z_(x), {B, S, hv_, dv_});
    mx::array b = in_b_(x);
    mx::array a = in_a_(x);
    mx::array conv_state = st.arrays.size() >= 1 ? st.arrays[0]
                                                 : mx::zeros({B, kconv_ - 1, conv_dim_}, x.dtype());
    mx::array conv_input = mx::concatenate({conv_state, qkv}, 1);
    const int n_keep = kconv_ - 1;
    mx::array new_conv = mx::contiguous(split_part(conv_input, 1, conv_input.shape(1) - n_keep,
                                                   conv_input.shape(1)));
    mx::array conv_out = silu_c(mx::conv1d(conv_input, conv_w_, 1, 0, 1, conv_dim_));
    mx::array q = mx::reshape(split_part(conv_out, -1, 0, key_dim_), {B, S, hk_, dk_});
    mx::array k = mx::reshape(split_part(conv_out, -1, key_dim_, 2 * key_dim_), {B, S, hk_, dk_});
    mx::array v = mx::reshape(split_part(conv_out, -1, 2 * key_dim_, conv_dim_), {B, S, hv_, dv_});
    const double inv_scale = std::pow(static_cast<double>(dk_), -0.5);
    q = mx::multiply(weak(inv_scale * inv_scale, q.dtype()), mx::fast::rms_norm(q, std::nullopt, 1e-6f));
    k = mx::multiply(weak(inv_scale, k.dtype()), mx::fast::rms_norm(k, std::nullopt, 1e-6f));
    // gated_delta_update: beta = sigmoid(b); g = compute_g(...); state zeros f32.
    mx::array beta = mx::sigmoid(b);
    mx::array g = compute_g_c(A_log_, a, dt_bias_);
    mx::array state = st.arrays.size() >= 2 ? st.arrays[1]
                                            : mx::zeros({B, hv_, dv_, dk_}, mx::float32);
    DeltaStepOutput d = gated_delta_step_kernel(q, k, v, g, beta, state);
    st.arrays = {new_conv, d.state};
    st.tokens += S;
    // Qwen3NextRMSNormGated: x = rms_norm(out, w, eps); _precise_swiglu(out, z, x)
    mx::array normed = mx::fast::rms_norm(d.y, norm_w_, eps_);
    mx::array out = precise_swiglu_c(d.y, z, normed);
    return out_(mx::reshape(out, {B, S, -1}));
  }

  NamedArrays save(const LayerState& st, const std::string& pfx) const override {
    NamedArrays out;
    if (st.arrays.size() == 2) {
      out.emplace_back(pfx + ".conv", st.arrays[0]);
      out.emplace_back(pfx + ".state", st.arrays[1]);
    }
    return out;
  }
  void load(LayerState& st, const std::string& pfx, const NamedArrays& a, int64_t tokens) const override {
    st.arrays.clear();
    const mx::array *conv = nullptr, *state = nullptr;
    for (const auto& kv : a) {
      if (kv.first == pfx + ".conv") conv = &kv.second;
      if (kv.first == pfx + ".state") state = &kv.second;
    }
    if (conv && state) st.arrays = {*conv, *state};
    else if (tokens > 0) throw std::runtime_error("linear.gdn: snapshot lacks state for " + pfx);
    st.tokens = tokens;
  }
  int64_t weight_bytes() const override {
    return in_qkv_.nbytes() + in_z_.nbytes() + in_b_.nbytes() + in_a_.nbytes() + out_.nbytes() +
           conv_w_.nbytes() + dt_bias_.nbytes() + A_log_.nbytes() + norm_w_.nbytes();
  }

 private:
  int hk_ = 0, hv_ = 0, dk_ = 0, dv_ = 0, kconv_ = 4, key_dim_ = 0, value_dim_ = 0, conv_dim_ = 0;
  float eps_ = 1e-6f;
  QLinear in_qkv_, in_z_, in_b_, in_a_, out_;
  mx::array conv_w_ = mx::array(0.0f), dt_bias_ = mx::array(0.0f), A_log_ = mx::array(0.0f),
            norm_w_ = mx::array(0.0f);
};

// =============================================================================
// attn.gated_gqa — mlx_lm qwen3_next.Qwen3NextAttention (and plain GQA when
// output_gate=false / qk_norm=false)
// =============================================================================
class GatedGqaMixer final : public Mixer {
 public:
  explicit GatedGqaMixer(const BlockCtx& c) : kv_(*c.kv) {
    const std::string p = tensors_of(c);
    heads_ = c.params.at("num_heads").i32("num_heads");
    kv_heads_ = c.params.at("num_kv_heads").i32("num_kv_heads");
    head_dim_ = c.params.at("head_dim").i32("head_dim");
    gate_ = c.params.get_bool("output_gate", false);
    qk_norm_ = c.params.get_bool("qk_norm", false);
    eps_ = static_cast<float>(c.params.get_num("eps", 1e-6));
    scale_ = static_cast<float>(std::pow(static_cast<double>(head_dim_), -0.5));
    if (const Json* r = c.params.find("rope")) {
      const std::string t = r->get_str("type", "default");
      if (t != "default" && t != "mrope" && t != "linear")
        throw std::runtime_error("attn.gated_gqa: rope type '" + t + "' is not ported (default/linear/mrope)");
      rope_dims_ = static_cast<int>(r->get_i64("dims", head_dim_));
      rope_base_ = static_cast<float>(r->get_num("base", 10000.0));
      rope_scale_ = static_cast<float>(r->get_num("scale", 1.0));
      rope_trad_ = r->get_bool("traditional", false);
    } else {
      rope_dims_ = 0;
    }
    q_ = c.loader.linear(p + ".q_proj");
    k_ = c.loader.linear(p + ".k_proj");
    v_ = c.loader.linear(p + ".v_proj");
    o_ = c.loader.linear(p + ".o_proj");
    const bool shift = c.params.get_bool("norm_shift", false);
    if (qk_norm_) {
      qn_ = c.loader.norm_weight(p + ".q_norm", shift);
      kn_ = c.loader.norm_weight(p + ".k_norm", shift);
      mx::eval(std::vector<mx::array>{*qn_, *kn_});
    }
  }
  std::string type() const override { return "attn.gated_gqa"; }

  void reset(LayerState& st) const override {
    st.arrays.clear();
    st.kv.emplace(kv_);
    st.tokens = 0;
  }

  mx::array forward(const mx::array& x, LayerState& st) const override {
    const int B = x.shape(0), L = x.shape(1);
    if (!st.kv) st.kv.emplace(kv_);
    mx::array queries = mx::array(0.0f), gate = mx::array(0.0f);
    mx::array qo = q_(x);
    if (gate_) {
      mx::array r = mx::reshape(qo, {B, L, heads_, -1});
      const int half = r.shape(-1) / 2;
      queries = split_part(r, -1, 0, half);
      gate = mx::reshape(split_part(r, -1, half, 2 * half), {B, L, -1});
    } else {
      queries = mx::reshape(qo, {B, L, heads_, -1});
    }
    mx::array keys = mx::reshape(k_(x), {B, L, kv_heads_, -1});
    mx::array values = mx::reshape(v_(x), {B, L, kv_heads_, -1});
    if (qk_norm_) {
      queries = mx::fast::rms_norm(queries, *qn_, eps_);
      keys = mx::fast::rms_norm(keys, *kn_, eps_);
    }
    queries = mx::transpose(queries, {0, 2, 1, 3});
    keys = mx::transpose(keys, {0, 2, 1, 3});
    values = mx::transpose(values, {0, 2, 1, 3});
    const int offset = static_cast<int>(st.kv->offset());
    if (rope_dims_ > 0) {
      queries = mx::fast::rope(queries, rope_dims_, rope_trad_, rope_base_, rope_scale_, offset);
      keys = mx::fast::rope(keys, rope_dims_, rope_trad_, rope_base_, rope_scale_, offset);
    }
    mx::array out = st.kv->update_and_attend(queries, keys, values, scale_);
    st.tokens += L;
    out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), {B, L, -1});
    if (gate_) out = mx::multiply(out, mx::sigmoid(gate));
    return o_(out);
  }

  NamedArrays save(const LayerState& st, const std::string& pfx) const override {
    if (!st.kv) return {};
    return st.kv->save(pfx + ".kv");
  }
  void load(LayerState& st, const std::string& pfx, const NamedArrays& a, int64_t tokens) const override {
    st.arrays.clear();
    st.kv.emplace(kv_);
    st.kv->load(pfx + ".kv", a, tokens);
    st.tokens = tokens;
  }
  int64_t kv_values_per_token() const override { return 2LL * kv_heads_ * head_dim_; }
  int64_t weight_bytes() const override {
    return q_.nbytes() + k_.nbytes() + v_.nbytes() + o_.nbytes() + opt_bytes(qn_) + opt_bytes(kn_);
  }

 private:
  KVPolicy kv_;
  int heads_ = 0, kv_heads_ = 0, head_dim_ = 0, rope_dims_ = 0;
  bool gate_ = false, qk_norm_ = false, rope_trad_ = false;
  float eps_ = 1e-6f, scale_ = 1.0f, rope_base_ = 10000.0f, rope_scale_ = 1.0f;
  QLinear q_, k_, v_, o_;
  std::optional<mx::array> qn_, kn_;
};

// =============================================================================
// mlp.swiglu — qwen3_next.Qwen3NextMLP: down(swiglu(gate(x), up(x)))
// =============================================================================
struct SwigluMlpWeights {
  QLinear gate, up, down;
  mx::array operator()(const mx::array& x) const { return down(swiglu_c(gate(x), up(x))); }
  int64_t nbytes() const { return gate.nbytes() + up.nbytes() + down.nbytes(); }
};

class SwigluMlp final : public Mlp {
 public:
  explicit SwigluMlp(const BlockCtx& c) {
    const std::string p = tensors_of(c);
    w_.gate = c.loader.linear(p + ".gate_proj");
    w_.up = c.loader.linear(p + ".up_proj");
    w_.down = c.loader.linear(p + ".down_proj");
  }
  std::string type() const override { return "mlp.swiglu"; }
  std::vector<mx::array> forward(const std::vector<mx::array>& xs, MlpStats&) const override {
    std::vector<mx::array> out;
    for (const auto& x : xs) out.push_back(w_(x));
    return out;
  }
  int64_t weight_bytes() const override { return w_.nbytes(); }

 private:
  SwigluMlpWeights w_;
};

// =============================================================================
// moe.softmax_topk — qwen3_next.Qwen3NextSparseMoeBlock with SwitchGLU
// =============================================================================
class SoftmaxTopkMoe final : public Mlp {
 public:
  explicit SoftmaxTopkMoe(const BlockCtx& c) : layer_(c.layer) {
    const std::string p = tensors_of(c);
    experts_ = c.params.at("num_experts").i32("num_experts");
    top_k_ = c.params.at("top_k").i32("top_k");
    norm_topk_ = c.params.get_bool("norm_topk_prob", true);
    const std::string act = c.params.get_str("activation", "swiglu");
    if (act != "swiglu") throw std::runtime_error("moe.softmax_topk: activation '" + act + "' not ported");
    router_ = c.loader.linear(p + "." + c.params.get_str("router", "gate"));
    std::vector<std::string> paths;
    std::vector<std::pair<int, int>> quant;
    for (const Json& e : c.params.at("experts").arr("experts")) {
      const std::string path = p + "." + e.str("expert proj");
      QuantSpec q = c.loader.quant_for(path);
      if (!q.quantized || q.mode != "affine")
        throw std::runtime_error("moe.softmax_topk: experts must be affine-quantized (" + path + ")");
      paths.push_back(path);
      quant.emplace_back(q.group_size, q.bits);
    }
    if (paths.size() != 3) throw std::runtime_error("moe.softmax_topk: experts = [gate, up, down]");
    const bool resident = c.resident_experts || !c.params.get_bool("streamed", true);
    table_ = std::make_unique<ExpertLayer>(c.loader.index(), paths, quant, resident);
    if (table_->experts() != experts_)
      throw std::runtime_error("moe.softmax_topk: num_experts does not match the checkpoint");
    const std::string se = c.params.get_str("shared_expert", "");
    if (!se.empty()) {
      SwigluMlpWeights w;
      w.gate = c.loader.linear(p + "." + se + ".gate_proj");
      w.up = c.loader.linear(p + "." + se + ".up_proj");
      w.down = c.loader.linear(p + "." + se + ".down_proj");
      shared_ = std::move(w);
      const std::string sg = c.params.get_str("shared_expert_gate", "");
      if (!sg.empty()) shared_gate_ = c.loader.linear(p + "." + sg);
    }
  }
  std::string type() const override { return "moe.softmax_topk"; }
  int64_t weight_bytes() const override {
    return router_.nbytes() + (shared_ ? shared_->nbytes() : 0) + (shared_gate_ ? shared_gate_->nbytes() : 0) +
           (table_->resident() ? table_->bytes_per_expert() * experts_ : 0);
  }
  int64_t streamed_bytes() const override {
    return table_->resident() ? 0 : table_->bytes_per_expert() * experts_;
  }
  void prefetch_hint() const override { table_->advise_all(); }
  void link_next(const Mlp* next) override { next_ = dynamic_cast<const SoftmaxTopkMoe*>(next); }
  bool prefill_prefetch_wanted(int tokens) const override {
    return prefill_pipe() && !table_->resident() &&
           static_cast<int64_t>(tokens) * top_k_ >= 4LL * static_cast<int64_t>(experts_);
  }
  void prefill_prefetch(int /*tokens*/) const override {
    if (pending_) return;
    std::vector<int> all(static_cast<size_t>(experts_));
    for (int e = 0; e < experts_; ++e) all[static_cast<size_t>(e)] = e;
    auto p = std::make_shared<PendingRead>();
    std::vector<rowfetch::Row> rows;
    p->set = table_->plan(all, rows);
    p->fetch = std::make_unique<rowfetch::Fetch>(std::move(rows), experts_, read_lanes());
    pending_ = std::move(p);
  }

  std::vector<mx::array> forward(const std::vector<mx::array>& xs, MlpStats& stats) const override {
    auto& pf = cppprof::profiler();
    const int n = static_cast<int>(xs.size());
    if (gpu_gate::failed()) throw std::runtime_error("generic: gated expert read failed");
    std::vector<mx::array> inds, scores, shared_out;
    std::vector<mx::array> to_eval;
    for (const auto& x : xs) {
      mx::array gates = router_(x);
      gates = mx::softmax(gates, -1, /*precise=*/true);
      mx::array ix = mx::argpartition(gates, experts_ - top_k_, -1);  // kth=-k
      ix = split_part(ix, -1, experts_ - top_k_, experts_);
      mx::array sc = mx::take_along_axis(gates, ix, -1);
      if (norm_topk_) sc = mx::divide(sc, mx::sum(sc, -1, /*keepdims=*/true));
      inds.push_back(ix);
      scores.push_back(sc);
      if (shared_) {
        mx::array sy = (*shared_)(x);
        if (shared_gate_) sy = mx::multiply(mx::sigmoid((*shared_gate_)(x)), sy);
        shared_out.push_back(sy);
      }
    }
    const bool decode1 = n == 1 && xs[0].shape(1) == 1;
    // Next layer's guess (decode): its router on this layer's input, evaluated with the route.
    std::optional<mx::array> guess_ix, guess_sc;
    const bool spec = decode1 && (spec_top() > 0 || spec_fetch() > 0) && next_ && !next_->table_->resident() &&
                      !table_->resident();
    if (spec) {
      mx::array g = mx::softmax(next_->router_(xs[0]), -1, /*precise=*/true);
      mx::array gi = mx::argpartition(g, next_->experts_ - next_->top_k_, -1);
      gi = split_part(gi, -1, next_->experts_ - next_->top_k_, next_->experts_);
      guess_sc = mx::astype(mx::take_along_axis(g, gi, -1), mx::float32);
      guess_ix = mx::astype(gi, mx::int32);
    }
    // Route sync. The shared expert is queued in the same submission so the GPU
    // runs it while the CPU reads the routed experts (early shared launch).
    if (!table_->resident()) {
      auto t0 = std::chrono::steady_clock::now();
      for (const auto& ix : inds) to_eval.push_back(mx::contiguous(ix));
      if (spec) {
        to_eval.push_back(*guess_ix);
        to_eval.push_back(*guess_sc);
      }
      if (!shared_out.empty()) {
        std::vector<mx::array> early(shared_out);
        early.insert(early.end(), scores.begin(), scores.end());
        mx::async_eval(early);
      }
      mx::eval(to_eval);
      auto t1 = std::chrono::steady_clock::now();
      const double sync_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      stats.route_sync_ms += sync_ms;   // the model adds it to the profiler per layer
    }
    std::shared_ptr<PendingRead> pending = std::move(pending_);
    pending_.reset();
    if (pending && decode1) pending.reset();  // only prefill uses it (destructor waits for the lanes)
    std::shared_ptr<SpecFetch> sf;
    {
      std::lock_guard<std::mutex> lk(spec_mu_);
      sf = std::move(specf_);
      specf_.reset();
    }
    if (sf && !(decode1 && !pipe_enabled())) sf.reset();
    if (decode1 && pipe_enabled() && !table_->resident()) {
      std::vector<mx::array> outs{pipelined(xs[0], to_eval[0], inds[0], scores[0], stats)};
      if (spec) prefetch_guess(*guess_ix, *guess_sc);
      if (shared_) outs[0] = mx::add(outs[0], shared_out[0]);
      return outs;
    }
    // VEYSTRIX_GENERIC_GATE=1 (one-token decode, no pipe): start this layer's reads,
    // build its expert compute at once and gate it on the GPU (gpu_gate) instead of
    // waiting for the rows on the CPU; the next layer's mixer and router are then
    // encoded while the rows land.
    const bool gate = decode1 && !table_->resident() && gpu_gate::enabled();
    std::shared_ptr<rowfetch::Fetch> gate_fetch;
    std::function<void()> gate_after;
    ExpertSet set;
    if (table_->resident()) {
      set = table_->fetch({});
    } else if (pending) {
      auto t0 = std::chrono::steady_clock::now();
      while (pending->fetch->next_ready() >= 0) {
      }
      if (!pending->fetch->ok()) throw std::runtime_error("generic: expert read failed");
      const rowfetch::Stats st = pending->fetch->stats();
      pending->fetch.reset();
      set = std::move(pending->set);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      stats.read_ms += ms;   // only the part not hidden behind compute
      stats.expert_bytes += set.bytes;
      stats.distinct_experts += experts_;
      pf.add_read(ms, static_cast<uint64_t>(set.bytes));
      pf.add_layer_cache(layer_, st.memcpy_rows, st.pread_rows, 0);
    } else if (sf) {
      auto t0 = std::chrono::steady_clock::now();
      while (sf->fetch->next_ready() >= 0) {
      }
      if (!sf->fetch->ok()) throw std::runtime_error("generic: expert read failed");
      const rowfetch::Stats st0 = sf->fetch->stats();
      sf->fetch.reset();
      const uint32_t* d = to_eval[0].data<uint32_t>();
      std::set<int> want(d, d + to_eval[0].size());
      std::vector<uint32_t> remap(static_cast<size_t>(experts_), 0u);
      std::vector<int> missing;
      const int G = static_cast<int>(sf->experts.size());
      for (int e : want) {
        auto it = std::find(sf->experts.begin(), sf->experts.end(), e);
        if (it != sf->experts.end()) {
          remap[static_cast<size_t>(e)] = static_cast<uint32_t>(it - sf->experts.begin());
        } else {
          remap[static_cast<size_t>(e)] = static_cast<uint32_t>(G + static_cast<int>(missing.size()));
          missing.push_back(e);
        }
      }
      rowfetch::Stats st1;
      if (!missing.empty() && gate) {
        std::vector<rowfetch::Row> rows;
        table_->add_rows(sf->set, missing, G, 0, rows);
        gate_fetch = std::make_shared<rowfetch::Fetch>(std::move(rows), static_cast<int>(missing.size()), read_lanes());
      } else if (!missing.empty()) {
        std::vector<rowfetch::Row> rows;
        table_->add_rows(sf->set, missing, G, 0, rows);
        rowfetch::Fetch f(std::move(rows), static_cast<int>(missing.size()), read_lanes());
        while (f.next_ready() >= 0) {
        }
        if (!f.ok()) throw std::runtime_error("generic: expert read failed");
        st1 = f.stats();
      }
      set = std::move(sf->set);
      set.remap = mx::array(remap.data(), mx::Shape{experts_}, mx::uint32);
      set.bytes = static_cast<int64_t>(missing.size()) * table_->bytes_per_expert();
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      stats.read_ms += ms;
      stats.expert_bytes += set.bytes;
      stats.distinct_experts += static_cast<int>(want.size());
      spec_hits_ += static_cast<long long>(want.size() - missing.size());
      spec_wanted_ += static_cast<long long>(want.size());
      if (pf.enabled && layer_ == 1 && spec_wanted_ % 800 < static_cast<long long>(want.size()))
        std::fprintf(stderr, "[SPECFETCH layer 1] guessed experts hit %lld / %lld (%.1f%%)\n", spec_hits_,
                     spec_wanted_, 100.0 * static_cast<double>(spec_hits_) / static_cast<double>(spec_wanted_));
      pf.add_read(ms, static_cast<uint64_t>(set.bytes));
      pf.add_layer_cache(layer_, st0.memcpy_rows + st1.memcpy_rows, st0.pread_rows + st1.pread_rows, 0);
      if (spec && gate_fetch) {
        gate_after = [this, gi = *guess_ix, gs = *guess_sc] { after_fetch_guess(gi, gs); };
      } else if (spec) {
        after_fetch_guess(*guess_ix, *guess_sc);
      }
    } else if (gate) {
      std::set<int> want;
      const uint32_t* d = to_eval[0].data<uint32_t>();
      for (size_t i = 0; i < to_eval[0].size(); ++i) want.insert(static_cast<int>(d[i]));
      std::vector<int> wanted(want.begin(), want.end());
      std::vector<rowfetch::Row> rows;
      set = table_->plan(wanted, rows, /*decode=*/true);
      gate_fetch = std::make_shared<rowfetch::Fetch>(std::move(rows), static_cast<int>(wanted.size()), read_lanes());
      stats.expert_bytes += set.bytes;
      stats.distinct_experts += static_cast<int>(wanted.size());
      pf.add_read(0.0, static_cast<uint64_t>(set.bytes));
      if (spec) gate_after = [this, gi = *guess_ix, gs = *guess_sc] { after_fetch_guess(gi, gs); };
    } else {
      std::set<int> want;
      for (size_t r = 0; r < inds.size(); ++r) {   // to_eval[0..n) are the routes (guesses follow)
        const mx::array& ix = to_eval[r];
        const uint32_t* d = ix.data<uint32_t>();
        for (size_t i = 0; i < ix.size(); ++i) want.insert(static_cast<int>(d[i]));
      }
      std::vector<int> wanted(want.begin(), want.end());
      auto t0 = std::chrono::steady_clock::now();
      set = table_->fetch(wanted, decode1);
      auto t1 = std::chrono::steady_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      stats.read_ms += ms;
      stats.expert_bytes += set.bytes;
      stats.distinct_experts += static_cast<int>(wanted.size());
      pf.add_read(ms, static_cast<uint64_t>(set.bytes));
      pf.add_layer_cache(layer_, set.rows_memcpy, set.rows_pread, 0);
      if (spec) after_fetch_guess(*guess_ix, *guess_sc);
    }
    const auto& P = table_->projs();
    std::vector<mx::array> outs;
    for (int l = 0; l < n; ++l) {
      mx::array idx = set.remap ? mx::take(*set.remap, inds[l], 0) : inds[l];
      // SwitchGLU.__call__
      const bool do_sort = idx.size() >= 64;
      mx::array x = mx::expand_dims(xs[l], {-2, -3});
      const mx::Shape ishape = idx.shape();
      std::optional<mx::array> inv_order;
      mx::array sidx = idx;
      if (do_sort) {
        // _gather_sort
        const int M = idx.shape(-1);
        mx::array flat = mx::flatten(idx);
        mx::array order = mx::argsort(flat);
        inv_order = mx::argsort(order);
        x = mx::take(mx::flatten(x, 0, -3), mx::floor_divide(order, mx::array(M, order.dtype())), 0);
        sidx = mx::take(flat, order, 0);
      }
      auto proj = [&](const mx::array& in, int p) {
        return mx::gather_qmm(in, set.w[p], set.s[p],
                              set.b.empty() ? std::nullopt : std::optional<mx::array>(set.b[p]),
                              std::nullopt, sidx, true, P[p].group_size, P[p].bits, "affine", do_sort);
      };
      mx::array x_up = proj(x, 1);
      mx::array x_gate = proj(x, 0);
      mx::array y = proj(swiglu_c(x_gate, x_up), 2);
      if (do_sort) {
        // _scatter_unsort
        y = mx::take(y, *inv_order, 0);
        mx::Shape un(ishape);
        for (int d = 1; d < y.ndim(); ++d) un.push_back(y.shape(d));
        y = mx::reshape(y, un);
      }
      y = mx::squeeze(y, -2);
      y = mx::sum(mx::multiply(y, mx::expand_dims(scores[l], -1)), -2);
      if (shared_) y = mx::add(y, shared_out[l]);
      outs.push_back(y);
    }
    if (gate_fetch) {
      // Everything above is still lazy; the wait goes in before any of it is encoded.
      const uint64_t v = gpu_gate::arm();
      if (v == 0) {
        while (gate_fetch->next_ready() >= 0) {
        }
        if (!gate_fetch->ok()) throw std::runtime_error("generic: expert read failed");
        if (gate_after) gate_after();
      } else {
        std::vector<mx::array> keep;  // the rows' destination buffers outlive the reads
        for (const auto* v2 : {&set.w, &set.s, &set.b}) keep.insert(keep.end(), v2->begin(), v2->end());
        gate_fetch->on_done([v, keep = std::move(keep), after = std::move(gate_after), f = gate_fetch](bool ok) {
          if (!ok) gpu_gate::mark_failed();
          gpu_gate::signal(v);
          if (after) after();
        });
      }
    }
    return outs;
  }


 private:
  // One token, one lane: per-expert launch as rows land. ix_eval: the evaluated
  // route [1,1,K] (uint32); ix: the same lazily; sc: scores [1,1,K].
  mx::array pipelined(const mx::array& x_in, const mx::array& ix_eval, const mx::array& ix,
                      const mx::array& sc, MlpStats& stats) const {
    auto& pf = cppprof::profiler();
    const uint32_t* d = ix_eval.data<uint32_t>();
    const size_t K = ix_eval.size();
    std::set<int> want;
    for (size_t i = 0; i < K; ++i) want.insert(static_cast<int>(d[i]));
    std::vector<int> wanted(want.begin(), want.end());
    std::vector<int32_t> local(K);
    for (size_t i = 0; i < K; ++i)
      local[i] = static_cast<int32_t>(std::lower_bound(wanted.begin(), wanted.end(), static_cast<int>(d[i])) -
                                      wanted.begin());
    auto t0 = std::chrono::steady_clock::now();
    std::vector<rowfetch::Row> rows;
    ExpertSet set = table_->plan(wanted, rows, /*decode=*/true);
    const auto& P = table_->projs();
    mx::array x = mx::expand_dims(x_in, {-2, -3});
    const uint32_t zero = 0;
    mx::array r0(&zero, mx::Shape{1, 1, 1}, mx::uint32);
    std::vector<std::optional<mx::array>> per(wanted.size());
    {
      const int N = static_cast<int>(wanted.size());
      auto f = std::make_shared<rowfetch::Fetch>(std::move(rows), N, read_lanes());
      auto build = [&](int g) {
        auto sl = [&](const mx::array& a) {
          mx::Shape lo(a.ndim(), 0), hi = a.shape();
          lo[0] = g;
          hi[0] = g + 1;
          return mx::slice(a, lo, hi);
        };
        auto proj = [&](const mx::array& in, int p) {
          return mx::gather_qmm(in, sl(set.w[p]), sl(set.s[p]),
                                set.b.empty() ? std::nullopt : std::optional<mx::array>(sl(set.b[p])),
                                std::nullopt, r0, true, P[p].group_size, P[p].bits, "affine", false);
        };
        mx::array x_up = proj(x, 1);
        mx::array x_gate = proj(x, 0);
        per[static_cast<size_t>(g)].emplace(proj(swiglu_c(x_gate, x_up), 2));  // [1,1,1,1,H]
      };
      const int tail = gpu_gate::tail();
      int launched = 0;
      bool gated = false;
      for (;;) {
        if (tail > 0 && N - launched <= tail) {
          // The last experts' compute goes behind a GPU gate (same ops, same bytes).
          for (int g = 0; g < N; ++g)
            if (!per[static_cast<size_t>(g)]) build(g);
          const uint64_t v = gpu_gate::arm();
          if (v != 0) {
            std::vector<mx::array> keep;
            for (const auto* a : {&set.w, &set.s, &set.b}) keep.insert(keep.end(), a->begin(), a->end());
            f->on_done([v, keep = std::move(keep), f](bool ok) {
              if (!ok) gpu_gate::mark_failed();
              gpu_gate::signal(v);
            });
            gated = true;
          } else {
            while (f->next_ready() >= 0) {
            }
          }
          break;
        }
        const int g = f->next_ready();
        if (g < 0) break;
        build(g);
        mx::async_eval(*per[static_cast<size_t>(g)]);
        ++launched;
      }
      if (!gated && !f->ok()) throw std::runtime_error("generic: expert read failed");
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    stats.read_ms += ms;
    stats.expert_bytes += set.bytes;
    stats.distinct_experts += static_cast<int>(wanted.size());
    pf.add_read(ms, static_cast<uint64_t>(set.bytes));
    std::vector<mx::array> ordered;
    ordered.reserve(K);
    for (int32_t j : local) ordered.push_back(*per[static_cast<size_t>(j)]);
    mx::array y = mx::concatenate(ordered, 2);  // [1,1,K,1,H] (router order, as gather_qmm returns)
    (void)ix;
    y = mx::squeeze(y, -2);
    return mx::sum(mx::multiply(y, mx::expand_dims(sc, -1)), -2);
  }

  void prefetch_guess(const mx::array& gi, const mx::array& gs) const {
    const int32_t* e = gi.data<int32_t>();
    const float* s = gs.data<float>();
    std::vector<std::pair<float, int>> ranked;
    for (size_t i = 0; i < gi.size(); ++i) ranked.emplace_back(-s[i], e[i]);
    std::sort(ranked.begin(), ranked.end());
    std::vector<int> top;
    for (size_t i = 0; i < ranked.size() && static_cast<int>(top.size()) < spec_top(); ++i)
      top.push_back(ranked[i].second);
    rowfetch::prefetch_cold(next_->table_->prefetch_rows(top), 8);
  }

  // After this layer's rows are in: speculative fetch of the next layer's top
  // SPEC_FETCH guesses into its stack buffers, then page-cache prefetch of the
  // next SPEC_TOP ranks.
  void after_fetch_guess(const mx::array& gi, const mx::array& gs) const {
    const int32_t* e = gi.data<int32_t>();
    const float* s = gs.data<float>();
    std::vector<std::pair<float, int>> ranked;
    for (size_t i = 0; i < gi.size(); ++i) ranked.emplace_back(-s[i], e[i]);
    std::sort(ranked.begin(), ranked.end());
    size_t at = 0;
    if (spec_fetch() > 0 && next_->top_k_ < 64) {   // unsorted gather only (row placement is free)
      std::vector<int> top;
      for (; at < ranked.size() && static_cast<int>(top.size()) < spec_fetch(); ++at) top.push_back(ranked[at].second);
      next_->speculate(top);
    }
    if (spec_top() > 0) {
      std::vector<int> more;
      for (; at < ranked.size() && static_cast<int>(more.size()) < spec_top(); ++at) more.push_back(ranked[at].second);
      if (!more.empty()) rowfetch::prefetch_cold(next_->table_->prefetch_rows(more), 8);
    }
  }

  void speculate(const std::vector<int>& experts) const {
    auto sp = std::make_shared<SpecFetch>();
    sp->experts = experts;
    sp->set = table_->alloc_stacks(static_cast<int>(experts.size()) + top_k_);
    std::vector<rowfetch::Row> rows;
    table_->add_rows(sp->set, experts, 0, 0, rows);
    sp->fetch = std::make_unique<rowfetch::Fetch>(std::move(rows), static_cast<int>(experts.size()), read_lanes());
    std::lock_guard<std::mutex> lk(spec_mu_);
    specf_ = std::move(sp);
  }

  struct PendingRead {
    ExpertSet set;
    std::unique_ptr<rowfetch::Fetch> fetch;
  };
  mutable std::shared_ptr<PendingRead> pending_;
  struct SpecFetch {
    std::vector<int> experts;   // stack rows 0..G-1
    ExpertSet set;              // capacity G + top_k rows
    std::unique_ptr<rowfetch::Fetch> fetch;
  };
  mutable std::shared_ptr<SpecFetch> specf_;
  mutable std::mutex spec_mu_;  // the GPU gate's completion sets the next layer's specf_ off-thread
  mutable long long spec_hits_ = 0, spec_wanted_ = 0;
  const SoftmaxTopkMoe* next_ = nullptr;
  int layer_ = 0;
  int experts_ = 0, top_k_ = 0;
  bool norm_topk_ = true;
  QLinear router_;
  std::unique_ptr<ExpertLayer> table_;
  std::optional<SwigluMlpWeights> shared_;
  std::optional<QLinear> shared_gate_;
};

}  // namespace

// ---- weights ---------------------------------------------------------------
mx::array QLinear::operator()(const mx::array& x) const {
  if (q.quantized) {
    mx::array y = mx::quantized_matmul(x, *w, *scales, biases, true, q.group_size, q.bits, q.mode);
    if (bias) y = mx::add(y, *bias);
    return y;
  }
  if (bias) return mx::addmm(*bias, x, mx::transpose(*w));
  return mx::matmul(x, mx::transpose(*w));
}
int64_t QLinear::nbytes() const { return opt_bytes(w) + opt_bytes(scales) + opt_bytes(biases) + opt_bytes(bias); }

mx::array QEmbedding::operator()(const mx::array& ids) const {
  if (q.quantized)
    return mx::dequantize(mx::take(*w, ids, 0), mx::take(*scales, ids, 0),
                          biases ? std::optional<mx::array>(mx::take(*biases, ids, 0)) : std::nullopt,
                          q.group_size, q.bits, q.mode);
  return mx::take(*w, ids, 0);
}
mx::array QEmbedding::as_linear(const mx::array& x) const {
  if (q.quantized) return mx::quantized_matmul(x, *w, *scales, biases, true, q.group_size, q.bits, q.mode);
  return mx::matmul(x, mx::transpose(*w));
}

Loader::Loader(const StIndex& idx, const Json& recipe) : idx_(idx), recipe_(recipe) {}

std::string Loader::expand(const std::string& tmpl, int layer) {
  std::string out = tmpl;
  const std::string key = "{L}";
  for (size_t at = out.find(key); at != std::string::npos; at = out.find(key))
    out.replace(at, key.size(), std::to_string(layer));
  return out;
}

QuantSpec Loader::quant_for(const std::string& path) const {
  QuantSpec q;
  q.quantized = idx_.has(path + ".scales");
  const Json* quant = recipe_.find("quant");
  if (!q.quantized) return q;
  if (!quant || quant->is_null())
    throw std::runtime_error("generic: " + path + " is quantized but the recipe has no quant block");
  auto apply = [&](const Json& j) {
    q.group_size = static_cast<int>(j.get_i64("group_size", q.group_size));
    q.bits = static_cast<int>(j.get_i64("bits", q.bits));
    q.mode = j.get_str("mode", q.mode);
  };
  if (const Json* d = quant->find("default"); d && d->is_object()) apply(*d);
  if (const Json* o = quant->find("overrides"); o && o->is_object())
    if (const Json* v = o->find(path); v && v->is_object()) apply(*v);
  return q;
}

mx::array Loader::tensor(const std::string& name) const { return idx_.load(name); }

QLinear Loader::linear(const std::string& path) const {
  QLinear L;
  L.q = quant_for(path);
  L.w = idx_.load(path + ".weight");
  if (L.q.quantized) {
    L.scales = idx_.load(path + ".scales");
    if (idx_.has(path + ".biases")) L.biases = idx_.load(path + ".biases");
    const int64_t in_w = static_cast<int64_t>(L.w->shape(-1)) * 32 / L.q.bits;
    const int64_t in_s = static_cast<int64_t>(L.scales->shape(-1)) * L.q.group_size;
    if (in_w != in_s)
      throw std::runtime_error("generic: quantization of " + path + " does not match the recipe");
  }
  if (idx_.has(path + ".bias")) L.bias = idx_.load(path + ".bias");
  return L;
}

QEmbedding Loader::embedding(const std::string& path) const {
  QEmbedding E;
  E.q = quant_for(path);
  E.w = idx_.load(path + ".weight");
  if (E.q.quantized) {
    E.scales = idx_.load(path + ".scales");
    if (idx_.has(path + ".biases")) E.biases = idx_.load(path + ".biases");
  }
  return E;
}

mx::array Loader::norm_weight(const std::string& path, bool shift) const {
  mx::array w = idx_.load(path + ".weight");
  // mlx_lm sanitize (qwen3_5): v + 1.0 for 1-D norm weights when shifting.
  if (shift && w.ndim() == 1) w = mx::add(w, weak(1.0, w.dtype()));
  return w;
}

// ---- registry ----------------------------------------------------------------
void register_qwen4_blocks(std::map<std::string, MixerFactory>& mixers, std::map<std::string, PreFactory>& pre);
void register_inkling_blocks(std::map<std::string, MixerFactory>& mixers, std::map<std::string, MlpFactory>& mlps);

namespace {
// Block families live one file each and register here (adding a family = one file + one line).
void register_all(std::map<std::string, MixerFactory>& mixers, std::map<std::string, MlpFactory>& mlps,
                  std::map<std::string, PreFactory>& pre) {
  register_qwen4_blocks(mixers, pre);
  register_inkling_blocks(mixers, mlps);
}
struct Registries {
  std::map<std::string, MixerFactory> mixers = {
      {"linear.gdn", [](const BlockCtx& c) { return std::unique_ptr<Mixer>(new GdnMixer(c)); }},
      {"attn.gated_gqa", [](const BlockCtx& c) { return std::unique_ptr<Mixer>(new GatedGqaMixer(c)); }},
  };
  std::map<std::string, MlpFactory> mlps = {
      {"moe.softmax_topk", [](const BlockCtx& c) { return std::unique_ptr<Mlp>(new SoftmaxTopkMoe(c)); }},
      {"mlp.swiglu", [](const BlockCtx& c) { return std::unique_ptr<Mlp>(new SwigluMlp(c)); }},
  };
  std::map<std::string, PreFactory> pre;
  Registries() { register_all(mixers, mlps, pre); }
};
Registries& registries() {
  static Registries r;
  return r;
}
}  // namespace

std::map<std::string, MixerFactory>& mixer_registry() { return registries().mixers; }
std::map<std::string, PreFactory>& pre_registry() { return registries().pre; }
std::map<std::string, MlpFactory>& mlp_registry() { return registries().mlps; }

std::vector<std::string> registered_blocks() {
  std::vector<std::string> out;
  for (const auto& kv : mixer_registry()) out.push_back(kv.first);
  for (const auto& kv : mlp_registry()) out.push_back(kv.first);
  for (const auto& kv : pre_registry()) out.push_back(kv.first);
  out.push_back("norm.rms");
  out.push_back("embed.quantized");
  return out;
}

}  // namespace veystrix::engine::generic
