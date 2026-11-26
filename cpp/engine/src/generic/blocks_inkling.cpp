// Inkling 2 Mini (inkling2_loop) generic block family:
//   mixers: "attn.inkling2_local"  (1,024-tok sliding window, fused in_proj, static K/V convs,
//                                   exact relative-position bias + zero-value sink)
//           "attn.inkling2_full"   (full causal attention, separate Q/K/V, log length scaling,
//                                   exact relative-position bias + zero-value sink + KVStore)
//   mlps:   "moe.inkling2_sigmoid" (sigmoid router + bias, top-8 of 128, fused w13/w2 NVFP4/affine,
//                                   soft-capped SwiGLU, per-expert global_scale, streamed/resident)
#include "generic/generic_model.hpp"

#include <mlx/compile.h>
#include <mlx/fast.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <set>
#include <stdexcept>

#include "cpp_profile.hpp"
#include "generic/expert_stream.hpp"
#include "gpu_gate.hpp"
#include "memory_budget.hpp"
#include "row_fetch.hpp"

namespace veystrix::engine::generic {

namespace {

mx::array split_last(const mx::array& x, int from, int to) {
  mx::Shape start(x.ndim(), 0), stop(x.shape());
  start.back() = from;
  stop.back() = to;
  return mx::slice(x, start, stop);
}

mx::array split_axis(const mx::array& x, int axis, int from, int to) {
  mx::Shape start(x.ndim(), 0), stop(x.shape());
  const int ax = axis < 0 ? axis + x.ndim() : axis;
  start[ax] = from;
  stop[ax] = to;
  return mx::slice(x, start, stop);
}

// Per-head L2 clipping (compiled): if ||x||_2 > thr, scale by thr / ||x||_2 (FP32 for stability).
mx::array l2_clip(const mx::array& x, float thr, float eps = 1e-6f) {
  if (thr <= 0.0f) return x;
  static auto fn = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array xf = mx::astype(in[0], mx::float32);
        mx::array norm = mx::sqrt(mx::sum(mx::square(xf), -1, /*keepdims=*/true) + in[2]);
        mx::array factor = mx::minimum(mx::array(1.0f, mx::float32), mx::divide(in[1], norm));
        return {mx::astype(mx::multiply(xf, factor), in[0].dtype())};
      },
      true);
  return fn({x, mx::array(thr, mx::float32), mx::array(eps, mx::float32)})[0];
}

// Fused scale + per-head L2 clipping (compiled) for Q and R projections.
mx::array scale_l2_clip(const mx::array& x, const mx::array& scale, float thr, float eps = 1e-6f) {
  if (thr <= 0.0f) return mx::multiply(x, scale);
  static auto fn = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array xf = mx::astype(mx::multiply(in[0], in[1]), mx::float32);
        mx::array norm = mx::sqrt(mx::sum(mx::square(xf), -1, /*keepdims=*/true) + in[3]);
        mx::array factor = mx::minimum(mx::array(1.0f, mx::float32), mx::divide(in[2], norm));
        return {mx::astype(mx::multiply(xf, factor), in[0].dtype())};
      },
      true);
  return fn({x, scale, mx::array(thr, mx::float32), mx::array(eps, mx::float32)})[0];
}

// Pre-bake Inkling 2 static residual short-conv kernels:
//   cw1: [C, W, 1] FP32 for L > 1 conv1d, 1wc: [1, W, C] FP32 for L == 1 fused dot reduction.
struct BakedSConv {
  mx::array cw1 = mx::array(0.0f);
  mx::array wc = mx::array(0.0f);
  int64_t nbytes() const { return cw1.nbytes(); }
};

BakedSConv bake_static_sconv(const mx::array& weight_ck) {
  const int C = weight_ck.shape(0), W = weight_ck.shape(1), hist = W - 1;
  mx::array wf = mx::astype(weight_ck, mx::float32);
  mx::array w_head = split_axis(wf, 1, 0, hist);
  mx::array w_last = mx::add(split_axis(wf, 1, hist, W), mx::array(1.0f, mx::float32));
  const float inv_norm = 1.0f / std::sqrt(static_cast<float>(W + 1));
  mx::array kern = mx::multiply(mx::concatenate({w_head, w_last}, 1), mx::array(inv_norm, mx::float32));
  BakedSConv out;
  out.cw1 = mx::contiguous(mx::reshape(kern, {C, W, 1}));
  out.wc = mx::contiguous(mx::reshape(mx::transpose(kern), {1, W, C}));
  return out;
}

// Inkling 2 static residual short convolution using pre-baked kernels.
// Uses a single fused compiled 4-tap elementwise kernel for all L >= 1 (eliminates grouped conv1d).
mx::array static_short_conv(const mx::array& x, const BakedSConv& kern,
                            std::optional<mx::array>& state_slot, bool save_full_history = false) {
  const int B = x.shape(0), L = x.shape(1), C = x.shape(2);
  const int W = kern.cw1.shape(1);
  const int hist = W - 1;
  mx::array st = state_slot ? *state_slot : mx::zeros({B, hist, C}, x.dtype());
  mx::array inp = mx::concatenate({st, x}, 1);  // [B, hist + L, C]
  state_slot = save_full_history ? mx::contiguous(inp) : mx::contiguous(split_axis(inp, 1, L, L + hist));
  if (L == 1) {
    static auto dot_c = mx::compile(
        [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
          mx::array y = mx::sum(mx::multiply(mx::astype(in[0], mx::float32), in[1]), 1, /*keepdims=*/true);
          return {mx::astype(y, in[0].dtype())};
        },
        true);
    return dot_c({inp, kern.wc})[0];
  }
  if (W == 4) {
    static auto tap4_c = mx::compile(
        [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
          mx::array y = mx::multiply(mx::astype(in[0], mx::float32), in[4]);
          y = mx::add(y, mx::multiply(mx::astype(in[1], mx::float32), in[5]));
          y = mx::add(y, mx::multiply(mx::astype(in[2], mx::float32), in[6]));
          y = mx::add(y, mx::multiply(mx::astype(in[3], mx::float32), in[7]));
          return {mx::astype(y, in[0].dtype())};
        },
        true);
    return tap4_c({split_axis(inp, 1, 0, L), split_axis(inp, 1, 1, L + 1),
                   split_axis(inp, 1, 2, L + 2), split_axis(inp, 1, 3, L + 3),
                   split_axis(kern.wc, 1, 0, 1), split_axis(kern.wc, 1, 1, 2),
                   split_axis(kern.wc, 1, 2, 3), split_axis(kern.wc, 1, 3, 4)})[0];
  }
  mx::array y = mx::conv1d(mx::astype(inp, mx::float32), kern.cw1, 1, 0, 1, C);
  return mx::astype(y, x.dtype());
}

// Inkling 2 full-attention K/V dynamic short convolution (ops.html):
//   y[t, h, d] = (x[t, h, d] + sum_{lag=0..W-1} (s[lag, h, d] + dyn[t, h, lag]) * x[t - lag, h, d]) / sqrt(2W + 1)
// state_slot holds the last W - 1 tokens [B, W - 1, dk].
mx::array dynamic_short_conv(const mx::array& x, const mx::array& static_w,
                             const std::optional<mx::array>& dyn_w,
                             std::optional<mx::array>& state_slot, int kv_heads, int head_dim,
                             bool save_full_history = false) {
  const int B = x.shape(0), L = x.shape(1), dk = x.shape(2);
  mx::array sw = static_w;
  if (sw.ndim() == 2 && sw.shape(0) == dk && sw.shape(1) != dk) sw = mx::transpose(sw);
  const int W = sw.shape(0);  // [W, dk]
  const int hist = W - 1;
  mx::array st = state_slot ? *state_slot : mx::zeros({B, hist, dk}, x.dtype());
  mx::array inp = mx::concatenate({st, x}, 1);  // [B, hist + L, dk]
  state_slot = save_full_history ? mx::contiguous(inp) : mx::contiguous(split_axis(inp, 1, L, L + hist));

  const float inv_norm = 1.0f / std::sqrt(static_cast<float>(2 * W + 1));
  if (W == 4 && dyn_w) {
    static auto dyn4_c = mx::compile(
        [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
          mx::array x0 = mx::astype(in[0], mx::float32);
          mx::array acc = mx::add(x0, mx::multiply(mx::add(in[4], mx::astype(in[8], mx::float32)), x0));
          acc = mx::add(acc, mx::multiply(mx::add(in[5], mx::astype(in[9], mx::float32)), mx::astype(in[1], mx::float32)));
          acc = mx::add(acc, mx::multiply(mx::add(in[6], mx::astype(in[10], mx::float32)), mx::astype(in[2], mx::float32)));
          acc = mx::add(acc, mx::multiply(mx::add(in[7], mx::astype(in[11], mx::float32)), mx::astype(in[3], mx::float32)));
          return {mx::astype(mx::multiply(acc, in[12]), in[0].dtype())};
        },
        true);
    mx::array sw_f = mx::astype(sw, mx::float32);
    auto xlag = [&](int lag) {
      return mx::reshape(split_axis(inp, 1, hist - lag, hist - lag + L), {B, L, kv_heads, head_dim});
    };
    auto slag = [&](int lag) {
      return mx::reshape(split_axis(sw_f, 0, lag, lag + 1), {1, 1, kv_heads, head_dim});
    };
    auto dlag = [&](int lag) { return split_last(*dyn_w, lag, lag + 1); };
    mx::array y = dyn4_c({xlag(0), xlag(1), xlag(2), xlag(3),
                          slag(0), slag(1), slag(2), slag(3),
                          dlag(0), dlag(1), dlag(2), dlag(3),
                          mx::array(inv_norm, mx::float32)})[0];
    return mx::reshape(y, {B, L, dk});
  }
  mx::array inp_f = mx::astype(inp, mx::float32);
  mx::array sw_f = mx::astype(sw, mx::float32);
  mx::array acc = mx::reshape(mx::astype(x, mx::float32), {B, L, kv_heads, head_dim});
  mx::array dyn_f = dyn_w ? mx::astype(*dyn_w, mx::float32) : mx::array(0.0f);  // [B, L, kv_heads, W]
  for (int lag = 0; lag < W; ++lag) {
    mx::array x_lag = mx::reshape(split_axis(inp_f, 1, hist - lag, hist - lag + L),
                                  {B, L, kv_heads, head_dim});
    mx::array s_lag = mx::reshape(split_axis(sw_f, 0, lag, lag + 1), {1, 1, kv_heads, head_dim});
    if (dyn_w) {
      mx::array d_lag = split_last(dyn_f, lag, lag + 1);  // [B, L, kv_heads, 1]
      s_lag = mx::add(s_lag, d_lag);
    }
    acc = mx::add(acc, mx::multiply(s_lag, x_lag));
  }
  acc = mx::multiply(acc, mx::array(inv_norm, mx::float32));
  return mx::astype(mx::reshape(acc, {B, L, dk}), x.dtype());
}

// Soft-capped SwiGLU (Inkling2SiTU, ops.html) directly on interleaved [g0, u0, g1, u1, ...] w13 output,
// with optional fused NVFP4 global_scale multiply in a single compiled Metal pass.
mx::array softcap_swiglu_interleaved(const mx::array& w13_out, int inter, float gcap, float ucap,
                                     const std::optional<mx::array>& gs_opt = std::nullopt) {
  mx::Shape gu_sh(w13_out.shape());
  gu_sh.back() = inter;
  gu_sh.push_back(2);
  mx::array gu = mx::reshape(w13_out, gu_sh);
  mx::array gate = mx::squeeze(split_last(gu, 0, 1), -1);
  mx::array up = mx::squeeze(split_last(gu, 1, 2), -1);
  static auto fn_capped = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array gf = mx::astype(in[0], mx::float32);
        mx::array uf = mx::astype(in[1], mx::float32);
        mx::array g_sig = mx::sigmoid(gf);
        mx::array g_cap = mx::multiply(in[2], mx::tanh(mx::divide(gf, in[2])));
        mx::array u_cap = mx::multiply(in[3], mx::tanh(mx::divide(uf, in[3])));
        return {mx::astype(mx::multiply(mx::multiply(g_cap, g_sig), u_cap), in[0].dtype())};
      },
      true);
  static auto fn_capped_gs = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array gf = mx::multiply(mx::astype(in[0], mx::float32), in[4]);
        mx::array uf = mx::multiply(mx::astype(in[1], mx::float32), in[4]);
        mx::array g_sig = mx::sigmoid(gf);
        mx::array g_cap = mx::multiply(in[2], mx::tanh(mx::divide(gf, in[2])));
        mx::array u_cap = mx::multiply(in[3], mx::tanh(mx::divide(uf, in[3])));
        return {mx::astype(mx::multiply(mx::multiply(g_cap, g_sig), u_cap), in[0].dtype())};
      },
      true);
  if (gcap > 0.0f && ucap > 0.0f) {
    if (gs_opt) {
      return fn_capped_gs({gate, up, mx::array(gcap, mx::float32), mx::array(ucap, mx::float32), *gs_opt})[0];
    }
    return fn_capped({gate, up, mx::array(gcap, mx::float32), mx::array(ucap, mx::float32)})[0];
  }
  mx::array gf = mx::astype(gate, mx::float32);
  mx::array uf = mx::astype(up, mx::float32);
  if (gs_opt) { gf = mx::multiply(gf, *gs_opt); uf = mx::multiply(uf, *gs_opt); }
  mx::array g_sig = mx::sigmoid(gf);
  mx::array g_cap = (gcap > 0.0f) ? mx::multiply(mx::array(gcap), mx::tanh(mx::divide(gf, mx::array(gcap)))) : gf;
  mx::array u_cap = (ucap > 0.0f) ? mx::multiply(mx::array(ucap), mx::tanh(mx::divide(uf, mx::array(ucap)))) : uf;
  return mx::astype(mx::multiply(mx::multiply(g_cap, g_sig), u_cap), gate.dtype());
}

std::string layer_prefix(const BlockCtx& c) {
  return Loader::expand(c.params.at("tensors").str("block tensors"), c.layer);
}

// =============================================================================
// attn.inkling2_local — Sliding-window (1,024-tok) attention with fused in_proj
// =============================================================================
class Inkling2LocalMixer final : public Mixer {
 public:
  explicit Inkling2LocalMixer(const BlockCtx& c) : kv_(*c.kv) {
    const std::string p = layer_prefix(c) + "attn";
    heads_ = c.params.at("num_heads").i32("num_heads");
    kv_heads_ = c.params.at("num_kv_heads").i32("num_kv_heads");
    head_dim_ = c.params.at("head_dim").i32("head_dim");
    d_rel_ = static_cast<int>(c.params.get_i64("d_rel", 16));
    rel_extent_ = static_cast<int>(c.params.get_i64("rel_extent", 1024));
    sconv_k_ = static_cast<int>(c.params.get_i64("sconv_kernel_size", 4));
    sw_ = c.params.get_i64("sliding_window", 1024);
    qk_clip_ = static_cast<float>(c.params.get_num("qk_l2_clip_threshold", 22.627416997969522));
    rel_clip_ = static_cast<float>(c.params.get_num("rel_l2_clip_threshold", 8.0));
    val_norm_ = c.params.get_bool("use_value_norm_no_gamma", true);
    eps_ = static_cast<float>(c.params.get_num("eps", 1e-6));
    scale_ = static_cast<float>(std::pow(static_cast<double>(head_dim_), -0.5));
    kv_.sliding_window = sw_;

    in_proj_ = c.loader.linear(p + ".in_proj");
    o_proj_ = c.loader.linear(p + ".o_proj");
    // Pre-slice K and V rows of in_proj so looped passes (pass > 0) avoid 2x redundant 8600-row matmuls.
    auto slice_qlinear = [](const QLinear& src, int r0, int r1) {
      QLinear out;
      out.q = src.q;
      auto sl0 = [&](const std::optional<mx::array>& a) -> std::optional<mx::array> {
        if (!a) return std::nullopt;
        mx::Shape lo(a->ndim(), 0), hi = a->shape();
        lo[0] = r0;
        hi[0] = r1;
        return mx::contiguous(mx::slice(*a, lo, hi));
      };
      out.w = sl0(src.w);
      out.scales = sl0(src.scales);
      out.biases = sl0(src.biases);
      out.bias = sl0(src.bias);
      return out;
    };
    auto concat_qlinear = [](const QLinear& a, const QLinear& b) {
      QLinear out;
      out.q = a.q;
      auto cat0 = [](const std::optional<mx::array>& x, const std::optional<mx::array>& y) -> std::optional<mx::array> {
        if (!x || !y) return std::nullopt;
        return mx::contiguous(mx::concatenate({*x, *y}, 0));
      };
      out.w = cat0(a.w, b.w);
      out.scales = cat0(a.scales, b.scales);
      out.biases = cat0(a.biases, b.biases);
      out.bias = cat0(a.bias, b.bias);
      return out;
    };
    bool is_core_layer = false;
    if (const Json* lp = c.loader.recipe().find("loop"); lp && lp->is_object()) {
      const int cs = static_cast<int>(lp->get_i64("core_start", 6));
      const int ce = static_cast<int>(lp->get_i64("core_end", 18));
      is_core_layer = (c.layer >= cs && c.layer < ce);
    }
    if (is_core_layer) {
      const int dq0 = heads_ * head_dim_, dk0 = kv_heads_ * head_dim_;
      const int total_rows = in_proj_.w->shape(0);
      in_k_ = slice_qlinear(in_proj_, dq0, dq0 + dk0);
      in_v_ = slice_qlinear(in_proj_, dq0 + dk0, dq0 + 2 * dk0);
      in_qrgs_ = concat_qlinear(slice_qlinear(in_proj_, 0, dq0), slice_qlinear(in_proj_, dq0 + 2 * dk0, total_rows));
      std::vector<mx::array> sliced;
      for (const QLinear* q : {&in_k_, &in_v_, &in_qrgs_}) {
        for (const auto* a : {&q->w, &q->scales, &q->biases, &q->bias}) {
          if (*a) sliced.push_back(**a);
        }
      }
      if (!sliced.empty()) mx::eval(sliced);
      in_proj_ = QLinear{};  // Materialize slices first so full in_proj_ is freed immediately (~180 MiB saved).
      mx::clear_cache();
    }
    k_conv_ = bake_static_sconv(c.loader.tensor(p + ".k_conv"));
    v_conv_ = bake_static_sconv(c.loader.tensor(p + ".v_conv"));
    q_scale_ = c.loader.tensor(p + ".q_scale");
    r_scale_ = c.loader.tensor(p + ".r_scale");
    rel_proj_ = c.loader.tensor(p + ".rel_proj");
    rel_proj_t_ = mx::contiguous(mx::transpose(rel_proj_));
    mx::eval(std::vector<mx::array>{k_conv_.cw1, k_conv_.wc, v_conv_.cw1, v_conv_.wc, q_scale_, r_scale_, rel_proj_, rel_proj_t_});
  }

  std::string type() const override { return "attn.inkling2_local"; }
  int64_t kv_values_per_token() const override { return 2LL * kv_heads_ * head_dim_; }
  int64_t sliding_window() const override { return sw_; }

  void reset(LayerState& st) const override {
    st.arrays.clear();
    st.kv.emplace(kv_);
    st.tokens = 0;
  }

  mx::array forward(const mx::array& x, LayerState& st) const override {
    return forward_qkv(x, x, x, st);
  }

  mx::array forward_qkv(const mx::array& xq, const mx::array& xk, const mx::array& xv,
                        LayerState& st) const override {
    const int B = xq.shape(0), L = xq.shape(1);
    if (!st.kv) st.kv.emplace(kv_);
    const int dq = heads_ * head_dim_;
    const int dk = kv_heads_ * head_dim_;
    const int dv = kv_heads_ * head_dim_;
    const int dg = heads_ * head_dim_;
    const int dr = heads_ * d_rel_;

    mx::array q_raw = mx::array(0.0f), k_raw = mx::array(0.0f), v_raw = mx::array(0.0f),
              r_raw = mx::array(0.0f), gate = mx::array(0.0f), sink = mx::array(0.0f);
    if (in_qrgs_.w) {
      // Core layers use sliced in_qrgs_ + in_k_ + in_v_ (no duplicate in_proj_ in RAM).
      mx::array pq = in_qrgs_(xq);
      q_raw = split_last(pq, 0, dq);
      r_raw = split_last(pq, dq, dq + dr);
      gate = split_last(pq, dq + dr, dq + dr + dg);
      sink = split_last(pq, dq + dr + dg, dq + dr + dg + heads_);
      k_raw = in_k_(xk);
      v_raw = in_v_(xv);
    } else if (&xk == &xq && &xv == &xq) {
      mx::array pq = in_proj_(xq);
      // Official Inkling 2 in_proj order: [Q(3072), K(1024), V(1024), R(384), Gate(3072), Sink(24)].
      q_raw = split_last(pq, 0, dq);
      k_raw = split_last(pq, dq, dq + dk);
      v_raw = split_last(pq, dq + dk, dq + dk + dv);
      r_raw = split_last(pq, dq + dk + dv, dq + dk + dv + dr);
      gate = split_last(pq, dq + dk + dv + dr, dq + dk + dv + dr + dg);
      sink = split_last(pq, dq + dk + dv + dr + dg, dq + dk + dv + dr + dg + heads_);
    } else {
      mx::array pq = in_proj_(xq);
      mx::array pk = in_proj_(xk);
      mx::array pv = in_proj_(xv);
      q_raw = split_last(pq, 0, dq);
      k_raw = split_last(pk, dq, dq + dk);
      v_raw = split_last(pv, dq + dk, dq + dk + dv);
      r_raw = split_last(pq, dq + dk + dv, dq + dk + dv + dr);
      gate = split_last(pq, dq + dk + dv + dr, dq + dk + dv + dr + dg);
      sink = split_last(pq, dq + dk + dv + dr + dg, dq + dk + dv + dr + dg + heads_);
    }

    std::optional<mx::array> k_st, v_st;
    if (st.arrays.size() >= 2) {
      k_st = st.arrays[0];
      v_st = st.arrays[1];
    }
    k_raw = static_short_conv(k_raw, k_conv_, k_st, st.save_conv_history);
    v_raw = static_short_conv(v_raw, v_conv_, v_st, st.save_conv_history);
    st.arrays = {*k_st, *v_st};

    mx::array q = scale_l2_clip(mx::reshape(q_raw, {B, L, heads_, head_dim_}), q_scale_, qk_clip_, eps_);
    mx::array k = l2_clip(mx::reshape(k_raw, {B, L, kv_heads_, head_dim_}), qk_clip_, eps_);
    mx::array v = mx::reshape(v_raw, {B, L, kv_heads_, head_dim_});
    if (val_norm_) v = mx::fast::rms_norm(v, std::nullopt, eps_);

    mx::array r = scale_l2_clip(mx::reshape(r_raw, {B, L, heads_, d_rel_}), r_scale_, rel_clip_, eps_);
    const int eff_e = static_cast<int>(std::min<int64_t>(rel_extent_, st.kv->offset() + L));
    mx::array r_t = mx::transpose(r, {0, 2, 1, 3});
    mx::array sink_t = mx::transpose(sink, {0, 2, 1});                              // [B, Hq, L]

    q = mx::transpose(q, {0, 2, 1, 3});
    k = mx::transpose(k, {0, 2, 1, 3});
    v = mx::transpose(v, {0, 2, 1, 3});

    mx::array out = (L > 1)
                        ? st.kv->update_and_attend_rel_sink(
                              q, k, v, r_t, sink_t, scale_,
                              (eff_e < rel_extent_) ? split_axis(rel_proj_t_, 0, 0, eff_e) : rel_proj_t_,
                              /*defer_compact=*/st.save_conv_history)
                        : st.kv->update_and_attend_rel_sink(
                              q, k, v,
                              mx::matmul(r_t, (eff_e < rel_extent_) ? split_last(rel_proj_, 0, eff_e) : rel_proj_),
                              sink_t, scale_, std::nullopt, /*defer_compact=*/st.save_conv_history);
    st.tokens += L;
    out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), {B, L, -1});
    static auto gate_mul_c = mx::compile(
        [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
          return {mx::multiply(in[0], mx::sigmoid(in[1]))};
        },
        true);
    out = gate_mul_c({out, gate})[0];
    return o_proj_(out);
  }

  NamedArrays save(const LayerState& st, const std::string& pfx) const override {
    NamedArrays out;
    if (st.kv) {
      auto kv = st.kv->save(pfx + ".kv");
      out.insert(out.end(), kv.begin(), kv.end());
    }
    if (st.arrays.size() >= 2) {
      out.emplace_back(pfx + ".kconv", st.arrays[0]);
      out.emplace_back(pfx + ".vconv", st.arrays[1]);
    }
    return out;
  }

  void load(LayerState& st, const std::string& pfx, const NamedArrays& a, int64_t tokens) const override {
    st.arrays.clear();
    st.kv.emplace(kv_);
    st.kv->load(pfx + ".kv", a, tokens);
    const mx::array *kc = nullptr, *vc = nullptr;
    for (const auto& kv : a) {
      if (kv.first == pfx + ".kconv") kc = &kv.second;
      if (kv.first == pfx + ".vconv") vc = &kv.second;
    }
    if (kc && vc) st.arrays = {*kc, *vc};
    st.tokens = tokens;
  }

  int64_t weight_bytes() const override {
    return in_proj_.nbytes() + o_proj_.nbytes() + k_conv_.nbytes() + v_conv_.nbytes() +
           q_scale_.nbytes() + r_scale_.nbytes() + rel_proj_.nbytes();
  }

 private:
  KVPolicy kv_;
  int heads_ = 24, kv_heads_ = 8, head_dim_ = 128, d_rel_ = 16, rel_extent_ = 1024, sconv_k_ = 4;
  int64_t sw_ = 1024;
  float qk_clip_ = 22.627417f, rel_clip_ = 8.0f, eps_ = 1e-6f, scale_ = 0.088388f;
  bool val_norm_ = true;
  QLinear in_proj_, in_qrgs_, in_k_, in_v_, o_proj_;
  BakedSConv k_conv_, v_conv_;
  mx::array q_scale_ = mx::array(0.0f), r_scale_ = mx::array(0.0f), rel_proj_ = mx::array(0.0f),
            rel_proj_t_ = mx::array(0.0f);
};

// =============================================================================
// attn.inkling2_full — Full-context attention (layers 5, 11, 17, 23; 8 visits)
// =============================================================================
class Inkling2FullMixer final : public Mixer {
 public:
  explicit Inkling2FullMixer(const BlockCtx& c) : kv_(*c.kv) {
    const std::string p = layer_prefix(c) + "attn";
    heads_ = c.params.at("num_heads").i32("num_heads");
    kv_heads_ = c.params.at("num_kv_heads").i32("num_kv_heads");
    head_dim_ = c.params.at("head_dim").i32("head_dim");
    d_rel_ = static_cast<int>(c.params.get_i64("d_rel", 16));
    rel_extent_ = static_cast<int>(c.params.get_i64("rel_extent", 1024));
    sconv_k_ = static_cast<int>(c.params.get_i64("sconv_kernel_size", 4));
    n_floor_ = c.params.get_i64("log_scaling_n_floor", 8192);
    log_alpha_ = static_cast<float>(c.params.get_num("log_scaling_alpha", 0.1));
    qk_clip_ = static_cast<float>(c.params.get_num("qk_l2_clip_threshold", 22.627416997969522));
    rel_clip_ = static_cast<float>(c.params.get_num("rel_l2_clip_threshold", 8.0));
    val_norm_ = c.params.get_bool("use_value_norm_no_gamma", true);
    eps_ = static_cast<float>(c.params.get_num("eps", 1e-6));
    scale_ = static_cast<float>(std::pow(static_cast<double>(head_dim_), -0.5));
    kv_.sliding_window = 0;

    q_proj_ = c.loader.linear(p + ".q_proj");
    k_proj_ = c.loader.linear(p + ".k_proj");
    v_proj_ = c.loader.linear(p + ".v_proj");
    o_proj_ = c.loader.linear(p + ".o_proj");
    k_static_ = c.loader.tensor(p + ".k_static");
    v_static_ = c.loader.tensor(p + ".v_static");
    q_scale_ = c.loader.tensor(p + ".q_scale");
    r_scale_ = c.loader.tensor(p + ".r_scale");
    rel_proj_ = c.loader.tensor(p + ".rel_proj");
    rel_proj_t_ = mx::contiguous(mx::transpose(rel_proj_));
    mx::eval(std::vector<mx::array>{k_static_, v_static_, q_scale_, r_scale_, rel_proj_, rel_proj_t_});
  }

  std::string type() const override { return "attn.inkling2_full"; }
  int64_t kv_values_per_token() const override { return 2LL * kv_heads_ * head_dim_; }

  void reset(LayerState& st) const override {
    st.arrays.clear();
    st.kv.emplace(kv_);
    st.tokens = 0;
  }

  mx::array forward(const mx::array& x, LayerState& st) const override {
    return forward_qkv(x, x, x, st);
  }

  mx::array forward_qkv(const mx::array& xq, const mx::array& xk, const mx::array& xv,
                        LayerState& st) const override {
    const int B = xq.shape(0), L = xq.shape(1);
    if (!st.kv) st.kv.emplace(kv_);
    const int dq = heads_ * head_dim_;
    const int dg = heads_ * head_dim_;
    const int dr = heads_ * d_rel_;

    mx::array qp = q_proj_(xq);
    // Official Inkling 2 q_proj order: [Q(3072), R(384), Gate(3072), Sink(24)].
    mx::array q_raw = split_last(qp, 0, dq);
    mx::array r_raw = split_last(qp, dq, dq + dr);
    mx::array gate = split_last(qp, dq + dr, dq + dr + dg);
    mx::array sink = split_last(qp, dq + dr + dg, dq + dr + dg + heads_);

    const int dk = kv_heads_ * head_dim_;
    mx::array kp = k_proj_(xk);
    mx::array vp = v_proj_(xv);
    mx::array k_raw = split_last(kp, 0, dk);
    mx::array v_raw = split_last(vp, 0, dk);

    // Dynamic K/V short convolution (ops.html): static taps k_static/v_static + per-token dynamic taps.
    std::optional<mx::array> k_st, v_st;
    if (st.arrays.size() >= 2) {
      k_st = st.arrays[0];
      v_st = st.arrays[1];
    }
    std::optional<mx::array> k_dyn, v_dyn;
    if (kp.shape(-1) > dk) {
      const int w_dyn = (kp.shape(-1) - dk) / kv_heads_;
      k_dyn = mx::reshape(split_last(kp, dk, kp.shape(-1)), {B, L, kv_heads_, w_dyn});
      v_dyn = mx::reshape(split_last(vp, dk, vp.shape(-1)), {B, L, kv_heads_, w_dyn});
    }
    k_raw = dynamic_short_conv(k_raw, k_static_, k_dyn, k_st, kv_heads_, head_dim_, st.save_conv_history);
    v_raw = dynamic_short_conv(v_raw, v_static_, v_dyn, v_st, kv_heads_, head_dim_, st.save_conv_history);
    st.arrays = {*k_st, *v_st};

    mx::array q = scale_l2_clip(mx::reshape(q_raw, {B, L, heads_, head_dim_}), q_scale_, qk_clip_, eps_);
    mx::array k = l2_clip(mx::reshape(k_raw, {B, L, kv_heads_, head_dim_}), qk_clip_, eps_);
    mx::array v = mx::reshape(v_raw, {B, L, kv_heads_, head_dim_});
    if (val_norm_) v = mx::fast::rms_norm(v, std::nullopt, eps_);

    // Log length scaling for contexts beyond log_scaling_n_floor (8,192 -> 1M tokens).
    const int64_t pos_end = st.kv->offset() + L;
    if (pos_end > n_floor_ && log_alpha_ > 0.0f) {
      const float ratio = static_cast<float>(pos_end) / static_cast<float>(n_floor_);
      const float mult = 1.0f + log_alpha_ * std::log(ratio);
      q = mx::multiply(q, mx::array(mult, q.dtype()));
    }

    mx::array r = scale_l2_clip(mx::reshape(r_raw, {B, L, heads_, d_rel_}), r_scale_, rel_clip_, eps_);
    const int eff_e = static_cast<int>(std::min<int64_t>(rel_extent_, st.kv->offset() + L));
    mx::array r_t = mx::transpose(r, {0, 2, 1, 3});
    mx::array sink_t = mx::transpose(sink, {0, 2, 1});

    q = mx::transpose(q, {0, 2, 1, 3});
    k = mx::transpose(k, {0, 2, 1, 3});
    v = mx::transpose(v, {0, 2, 1, 3});

    mx::array out = (L > 1)
                        ? st.kv->update_and_attend_rel_sink(
                              q, k, v, r_t, sink_t, scale_,
                              (eff_e < rel_extent_) ? split_axis(rel_proj_t_, 0, 0, eff_e) : rel_proj_t_,
                              /*defer_compact=*/st.save_conv_history)
                        : st.kv->update_and_attend_rel_sink(
                              q, k, v,
                              mx::matmul(r_t, (eff_e < rel_extent_) ? split_last(rel_proj_, 0, eff_e) : rel_proj_),
                              sink_t, scale_, std::nullopt, /*defer_compact=*/st.save_conv_history);
    st.tokens += L;
    out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), {B, L, -1});
    static auto gate_mul_c = mx::compile(
        [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
          return {mx::multiply(in[0], mx::sigmoid(in[1]))};
        },
        true);
    out = gate_mul_c({out, gate})[0];
    return o_proj_(out);
  }

  NamedArrays save(const LayerState& st, const std::string& pfx) const override {
    NamedArrays out;
    if (st.kv) {
      auto kv = st.kv->save(pfx + ".kv");
      out.insert(out.end(), kv.begin(), kv.end());
    }
    if (st.arrays.size() >= 2) {
      out.emplace_back(pfx + ".kconv", st.arrays[0]);
      out.emplace_back(pfx + ".vconv", st.arrays[1]);
    }
    return out;
  }

  void load(LayerState& st, const std::string& pfx, const NamedArrays& a, int64_t tokens) const override {
    st.arrays.clear();
    st.kv.emplace(kv_);
    st.kv->load(pfx + ".kv", a, tokens);
    const mx::array *kc = nullptr, *vc = nullptr;
    for (const auto& kv : a) {
      if (kv.first == pfx + ".kconv") kc = &kv.second;
      if (kv.first == pfx + ".vconv") vc = &kv.second;
    }
    if (kc && vc) st.arrays = {*kc, *vc};
    st.tokens = tokens;
  }

  int64_t weight_bytes() const override {
    return q_proj_.nbytes() + k_proj_.nbytes() + v_proj_.nbytes() + o_proj_.nbytes() +
           k_static_.nbytes() + v_static_.nbytes() + q_scale_.nbytes() + r_scale_.nbytes() +
           rel_proj_.nbytes();
  }

 private:
  KVPolicy kv_;
  int heads_ = 24, kv_heads_ = 8, head_dim_ = 128, d_rel_ = 16, rel_extent_ = 1024, sconv_k_ = 4;
  int64_t n_floor_ = 8192;
  float log_alpha_ = 0.1f, qk_clip_ = 22.627417f, rel_clip_ = 8.0f, eps_ = 1e-6f, scale_ = 0.088388f;
  bool val_norm_ = true;
  QLinear q_proj_, k_proj_, v_proj_, o_proj_;
  mx::array k_static_ = mx::array(0.0f), v_static_ = mx::array(0.0f), q_scale_ = mx::array(0.0f),
            r_scale_ = mx::array(0.0f), rel_proj_ = mx::array(0.0f), rel_proj_t_ = mx::array(0.0f);
};

// =============================================================================
// moe.inkling2_sigmoid — Sigmoid-routed top-8/128 MoE with fused w13/w2 NVFP4/affine
// =============================================================================
class Inkling2SigmoidMoe final : public Mlp {
 public:
  explicit Inkling2SigmoidMoe(const BlockCtx& c) : layer_(c.layer) {
    const std::string p = layer_prefix(c);
    experts_ = c.params.at("num_experts").i32("num_experts");
    top_k_ = c.params.at("top_k").i32("top_k");
    inter_ = static_cast<int>(c.params.get_i64("intermediate_size", 1152));
    route_scale_ = static_cast<float>(c.params.get_num("route_scale", 8.0));
    gate_softcap_ = static_cast<float>(c.params.get_num("gate_softcap", 4.0));
    up_softcap_ = static_cast<float>(c.params.get_num("up_softcap", 25.0));

    router_w_ = mx::contiguous(mx::transpose(c.loader.tensor(p + ".router_weight")));
    if (c.loader.has(p + ".router_bias")) router_b_ = c.loader.tensor(p + ".router_bias");
    mx::eval(router_b_ ? std::vector<mx::array>{router_w_, *router_b_}
                       : std::vector<mx::array>{router_w_});

    std::vector<std::string> paths;
    std::vector<std::pair<int, int>> quant;
    std::vector<std::string> modes;
    for (const Json& e : c.params.at("experts").arr("experts")) {
      const std::string path = p + "." + e.str("expert proj");
      QuantSpec q = c.loader.quant_for(path);
      if (!q.quantized)
        throw std::runtime_error("moe.inkling2_sigmoid: experts must be quantized (" + path + ")");
      paths.push_back(path);
      quant.emplace_back(q.group_size, q.bits);
      modes.push_back(q.mode);
    }
    if (paths.size() != 2)
      throw std::runtime_error("moe.inkling2_sigmoid: experts must be [experts.w13, experts.w2]");
    const bool resident = c.resident_experts || !c.params.get_bool("streamed", true);
    table_ = std::make_unique<ExpertLayer>(c.loader.index(), paths, quant, resident, modes);
    if (table_->experts() != experts_)
      throw std::runtime_error("moe.inkling2_sigmoid: num_experts does not match checkpoint");
  }

  std::string type() const override { return "moe.inkling2_sigmoid"; }
  bool is_resident() const override { return table_->resident(); }
  void load_resident() override { if (table_) table_->load_resident(); }
  void release_resident() override { if (table_) table_->release_resident(); }
  int64_t weight_bytes() const override {
    return router_w_.nbytes() + (router_b_ ? router_b_->nbytes() : 0) +
           (table_->resident() ? table_->bytes_per_expert() * experts_ : 0);
  }
  int64_t streamed_bytes() const override {
    return table_->resident() ? 0 : table_->bytes_per_expert() * experts_;
  }
  void prefetch_hint() const override { table_->advise_all(); }
  void link_next(const Mlp* next) override { next_ = dynamic_cast<const Inkling2SigmoidMoe*>(next); }
  bool prefill_prefetch_wanted(int tokens) const override {
    static const bool on = [] {
      const char* e = std::getenv("VEYSTRIX_GENERIC_PREFILL_PIPE");
      return !(e && std::strcmp(e, "0") == 0);
    }();
    return on && !table_->resident() &&
           static_cast<int64_t>(tokens) * top_k_ >= 2LL * static_cast<int64_t>(experts_);
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
  void spec_prefetch_next(const mx::array& x) const override {
    static const int spec_fetch_n = [] {
      const char* e = std::getenv("VEYSTRIX_GENERIC_SPEC_FETCH");
      return (e && *e) ? std::atoi(e) : 4;
    }();
    static const int spec_top_n = [] {
      const char* e = std::getenv("VEYSTRIX_GENERIC_SPEC_TOP");
      return (e && *e) ? std::atoi(e) : 0;
    }();
    if ((spec_fetch_n <= 0 && spec_top_n <= 0) || !next_ || next_->table_->resident()) return;
    const int batch_s = x.shape(1);
    if (batch_s < 1 || batch_s > 8) return;
    mx::array glog = mx::matmul(mx::astype(x, mx::float32), next_->router_w_);
    mx::array gp = mx::sigmoid(glog);
    mx::array gc = next_->router_b_ ? mx::add(gp, *next_->router_b_) : gp;
    mx::array gi = mx::argpartition(gc, next_->experts_ - next_->top_k_, -1);
    gi = split_last(gi, next_->experts_ - next_->top_k_, next_->experts_);
    mx::array gsc = mx::astype(mx::take_along_axis(gc, gi, -1), mx::float32);
    mx::array gix = mx::astype(gi, mx::int32);
    mx::eval(std::vector<mx::array>{gix, gsc});
    after_fetch_guess(gix, gsc, spec_fetch_n, spec_top_n, batch_s);
  }

  void prefetch_epilogue_from_loop(const mx::array& x,
                                   const std::vector<const Mlp*>& targets) const override {
    static const int loop_pf_k = [] {
      const char* e = std::getenv("VEYSTRIX_INKLING_LOOP_PREFETCH");
      return (e && *e) ? std::atoi(e) : 0;
    }();
    if (loop_pf_k <= 0 || targets.empty()) return;
    const int batch_s = x.shape(1);
    if (batch_s < 1 || batch_s > 8) return;
    std::vector<const Inkling2SigmoidMoe*> moes;
    std::vector<mx::array> gixs, gscs, to_ev;
    mx::array xf = mx::astype(x, mx::float32);
    for (const Mlp* m : targets) {
      auto* im = dynamic_cast<const Inkling2SigmoidMoe*>(m);
      if (!im || im->table_->resident()) continue;
      mx::array glog = mx::matmul(xf, im->router_w_);
      mx::array gp = mx::sigmoid(glog);
      mx::array gc = im->router_b_ ? mx::add(gp, *im->router_b_) : gp;
      mx::array gi = mx::argpartition(gc, im->experts_ - im->top_k_, -1);
      gi = split_last(gi, im->experts_ - im->top_k_, im->experts_);
      mx::array gsc = mx::astype(mx::take_along_axis(gc, gi, -1), mx::float32);
      mx::array gix = mx::astype(gi, mx::int32);
      moes.push_back(im);
      gixs.push_back(gix);
      gscs.push_back(gsc);
      to_ev.push_back(gix);
      to_ev.push_back(gsc);
    }
    if (moes.empty()) return;
    mx::eval(to_ev);
    std::vector<rowfetch::Row> all_rows;
    const int per_layer = std::min(12, loop_pf_k * std::min(batch_s, 2));
    for (size_t idx = 0; idx < moes.size(); ++idx) {
      const Inkling2SigmoidMoe* im = moes[idx];
      const int32_t* e = gixs[idx].data<int32_t>();
      const float* s = gscs[idx].data<float>();
      std::vector<float> acc(static_cast<size_t>(im->experts_), 0.0f);
      std::vector<int> seen;
      for (size_t i = 0; i < gixs[idx].size(); ++i) {
        const int id = e[i];
        if (id < 0 || id >= im->experts_) continue;
        if (acc[static_cast<size_t>(id)] == 0.0f) seen.push_back(id);
        acc[static_cast<size_t>(id)] += std::max(s[i], 1e-6f);
      }
      std::vector<std::pair<float, int>> ranked;
      ranked.reserve(seen.size());
      for (int id : seen) ranked.emplace_back(-acc[static_cast<size_t>(id)], id);
      std::sort(ranked.begin(), ranked.end());
      std::vector<int> top;
      for (size_t i = 0; i < ranked.size() && static_cast<int>(top.size()) < per_layer; ++i)
        top.push_back(ranked[i].second);
      if (!top.empty()) {
        auto rows = im->table_->prefetch_rows(top);
        all_rows.insert(all_rows.end(), rows.begin(), rows.end());
      }
    }
    if (!all_rows.empty()) rowfetch::prefetch_cold(std::move(all_rows), read_lanes());
  }

  std::vector<mx::array> forward(const std::vector<mx::array>& xs, MlpStats& stats) const override {
    auto& pf = cppprof::profiler();
    const int n = static_cast<int>(xs.size());
    if (gpu_gate::failed()) throw std::runtime_error("generic: gated expert read failed");
    static const int spec_fetch_n = [] {
      const char* e = std::getenv("VEYSTRIX_GENERIC_SPEC_FETCH");
      return (e && *e) ? std::atoi(e) : 4;
    }();
    static const int spec_top_n = [] {
      const char* e = std::getenv("VEYSTRIX_GENERIC_SPEC_TOP");
      return (e && *e) ? std::atoi(e) : 0;
    }();
    static const bool pipe_on = [] {
      const char* e = std::getenv("VEYSTRIX_GENERIC_PIPE");
      return e && std::strcmp(e, "1") == 0;
    }();

    std::vector<mx::array> inds, scores, to_eval;
    for (const auto& x : xs) {
      mx::array logits = mx::matmul(mx::astype(x, mx::float32), router_w_);
      mx::array probs = mx::sigmoid(logits);
      mx::array choice = router_b_ ? mx::add(probs, *router_b_) : probs;
      mx::array ix = mx::argpartition(choice, experts_ - top_k_, -1);
      ix = split_last(ix, experts_ - top_k_, experts_);
      mx::array sc = mx::take_along_axis(probs, ix, -1);
      sc = mx::multiply(mx::divide(sc, mx::add(mx::sum(sc, -1, /*keepdims=*/true), mx::array(1e-20f))),
                        mx::array(route_scale_));
      inds.push_back(ix);
      scores.push_back(mx::astype(sc, x.dtype()));
    }
    const int batch_s = (n == 1) ? xs[0].shape(1) : 0;
    const bool decode1 = n == 1 && batch_s == 1;
    const bool small_batch = n == 1 && batch_s <= 8;
    std::optional<mx::array> guess_ix, guess_sc;
    const bool spec = small_batch && (spec_top_n > 0 || spec_fetch_n > 0) && next_ &&
                      !next_->table_->resident() && !table_->resident();
    if (spec) {
      mx::array glog = mx::matmul(mx::astype(xs[0], mx::float32), next_->router_w_);
      mx::array gp = mx::sigmoid(glog);
      mx::array gc = next_->router_b_ ? mx::add(gp, *next_->router_b_) : gp;
      mx::array gi = mx::argpartition(gc, next_->experts_ - next_->top_k_, -1);
      gi = split_last(gi, next_->experts_ - next_->top_k_, next_->experts_);
      guess_sc = mx::astype(mx::take_along_axis(gc, gi, -1), mx::float32);
      guess_ix = mx::astype(gi, mx::int32);
    }
    if (!table_->resident()) {
      auto t0 = std::chrono::steady_clock::now();
      for (const auto& ix : inds) to_eval.push_back(mx::contiguous(ix));
      if (spec) {
        to_eval.push_back(*guess_ix);
        to_eval.push_back(*guess_sc);
      }
      mx::async_eval(scores);
      mx::eval(to_eval);
      stats.route_sync_ms += std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count();
    }
    std::shared_ptr<PendingRead> pending = std::move(pending_);
    pending_.reset();
    if (pending && small_batch) pending.reset();
    std::shared_ptr<SpecFetch> sf;
    {
      std::lock_guard<std::mutex> lk(spec_mu_);
      sf = std::move(specf_);
      specf_.reset();
    }
    if (sf && !(small_batch && !pipe_on)) sf.reset();

    if (decode1 && pipe_on && !table_->resident()) {
      std::vector<mx::array> out{pipelined(xs[0], to_eval[0], scores[0], stats)};
      if (spec && spec_top_n > 0) prefetch_guess(*guess_ix, *guess_sc, spec_top_n);
      return out;
    }

    const bool gate = small_batch && !table_->resident() && gpu_gate::enabled();
    std::shared_ptr<rowfetch::Fetch> gate_fetch, sf_fetch;
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
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      stats.read_ms += ms;
      stats.expert_bytes += set.bytes;
      stats.distinct_experts += experts_;
      pf.add_read(ms, static_cast<uint64_t>(set.bytes));
      pf.add_layer_cache(layer_, st.memcpy_rows, st.pread_rows, 0);
    } else if (sf) {
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
      if (gate) {
        sf_fetch = std::shared_ptr<rowfetch::Fetch>(std::move(sf->fetch));
        if (!missing.empty()) {
          std::vector<rowfetch::Row> rows;
          table_->add_rows(sf->set, missing, G, 0, rows);
          gate_fetch = std::make_shared<rowfetch::Fetch>(std::move(rows), static_cast<int>(missing.size()),
                                                         read_lanes());
        }
        set = std::move(sf->set);
        set.remap = mx::array(remap.data(), mx::Shape{experts_}, mx::uint32);
        set.bytes = static_cast<int64_t>(missing.size()) * table_->bytes_per_expert();
        stats.expert_bytes += set.bytes;
        stats.distinct_experts += static_cast<int>(want.size());
        pf.add_read(0.0, static_cast<uint64_t>(set.bytes));
        if (spec) {
          gate_after = [this, gi = *guess_ix, gs = *guess_sc, sf_n = spec_fetch_n, st_n = spec_top_n, batch_s] {
            after_fetch_guess(gi, gs, sf_n, st_n, batch_s);
          };
        }
      } else {
        auto t0 = std::chrono::steady_clock::now();
        while (sf->fetch->next_ready() >= 0) {
        }
        if (!sf->fetch->ok()) throw std::runtime_error("generic: expert read failed");
        const rowfetch::Stats st0 = sf->fetch->stats();
        sf->fetch.reset();
        rowfetch::Stats st1;
        if (!missing.empty()) {
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
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        stats.read_ms += ms;
        stats.expert_bytes += set.bytes;
        stats.distinct_experts += static_cast<int>(want.size());
        pf.add_read(ms, static_cast<uint64_t>(set.bytes));
        pf.add_layer_cache(layer_, st0.memcpy_rows + st1.memcpy_rows, st0.pread_rows + st1.pread_rows, 0);
        if (spec) after_fetch_guess(*guess_ix, *guess_sc, spec_fetch_n, spec_top_n, batch_s);
      }
    } else if (gate) {
      std::set<int> want;
      const uint32_t* d = to_eval[0].data<uint32_t>();
      for (size_t i = 0; i < to_eval[0].size(); ++i) want.insert(static_cast<int>(d[i]));
      std::vector<int> wanted(want.begin(), want.end());
      std::vector<rowfetch::Row> rows;
      set = table_->plan(wanted, rows, /*decode=*/true);
      gate_fetch = std::make_shared<rowfetch::Fetch>(std::move(rows), static_cast<int>(wanted.size()),
                                                     read_lanes());
      stats.expert_bytes += set.bytes;
      stats.distinct_experts += static_cast<int>(wanted.size());
      pf.add_read(0.0, static_cast<uint64_t>(set.bytes));
      if (spec) {
        gate_after = [this, gi = *guess_ix, gs = *guess_sc, sf_n = spec_fetch_n, st_n = spec_top_n, batch_s] {
          after_fetch_guess(gi, gs, sf_n, st_n, batch_s);
        };
      }
    } else {
      std::set<int> want;
      for (size_t r = 0; r < inds.size(); ++r) {
        const mx::array& ix = to_eval[r];
        const uint32_t* d = ix.data<uint32_t>();
        for (size_t i = 0; i < ix.size(); ++i) want.insert(static_cast<int>(d[i]));
      }
      std::vector<int> wanted(want.begin(), want.end());
      // On tight-RAM (<= 16 GB) Macs, split large streamed prefill MoE reads into
      // 32-expert slices (~137 MiB instead of 547 MiB) so 75% of SSD reads overlap GPU compute.
      static const int moe_slice_max = [] {
        const char* e = std::getenv("VEYSTRIX_INKLING_MOE_SLICE");
        if (e && *e) return std::atoi(e);
        return membudget::physical_ram_bytes() <= (size_t{20} << 30) ? 32 : 0;
      }();
      if (!small_batch && moe_slice_max > 0 && static_cast<int>(wanted.size()) > moe_slice_max) {
        const auto& P = table_->projs();
        std::vector<mx::array> accs;
        std::vector<mx::array> eff_scs;
        accs.reserve(static_cast<size_t>(n));
        eff_scs.reserve(static_cast<size_t>(n));
        for (int l = 0; l < n; ++l) {
          const int num_toks = xs[l].shape(0) * xs[l].shape(1);
          const int H = xs[l].shape(2);
          accs.push_back(mx::zeros({num_toks, H}, xs[l].dtype()));
          mx::array eff_sc = scores[l];
          if (P[1].has_gs) {
            mx::array gs1 = mx::take(P[1].global_scale, inds[l], 0);
            eff_sc = mx::astype(mx::multiply(mx::astype(eff_sc, mx::float32), gs1), eff_sc.dtype());
          }
          eff_scs.push_back(mx::flatten(eff_sc));
        }
        // Pre-plan expert slices so Slice k+1's SSD read overlaps with Slice k's GPU compute.
        std::vector<std::vector<int>> slices;
        for (size_t off = 0; off < wanted.size(); off += static_cast<size_t>(moe_slice_max)) {
          const size_t end = std::min(wanted.size(), off + static_cast<size_t>(moe_slice_max));
          slices.emplace_back(wanted.begin() + static_cast<std::ptrdiff_t>(off),
                              wanted.begin() + static_cast<std::ptrdiff_t>(end));
        }
        const bool overlap_slices =
            slices.size() > 1 && rows_enabled() &&
            membudget::memory_pressure_level() < 4 &&
            membudget::available_ram_bytes() >= (size_t{22} << 30) / 10;
        struct AsyncSlice {
          ExpertSet set;
          std::unique_ptr<rowfetch::Fetch> fetch;
          std::chrono::steady_clock::time_point started;
        };
        auto start_slice = [&](const std::vector<int>& sw) -> AsyncSlice {
          AsyncSlice a;
          a.started = std::chrono::steady_clock::now();
          std::vector<rowfetch::Row> rows;
          a.set = table_->plan(sw, rows, false);
          a.fetch = std::make_unique<rowfetch::Fetch>(std::move(rows), static_cast<int>(sw.size()),
                                                      prefill_read_lanes());
          return a;
        };
        auto finish_slice = [&](AsyncSlice& a, const std::vector<int>& sw) -> ExpertSet {
          while (a.fetch->next_ready() >= 0) {
          }
          if (!a.fetch->ok()) {
            throw std::runtime_error("generic: sliced prefill expert read failed: " + a.fetch->error_detail());
          }
          const rowfetch::Stats st = a.fetch->stats();
          a.fetch.reset();
          const double ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - a.started)
                                .count();
          stats.read_ms += ms;
          stats.expert_bytes += a.set.bytes;
          stats.distinct_experts += static_cast<int>(sw.size());
          pf.add_read(ms, static_cast<uint64_t>(a.set.bytes));
          pf.add_layer_cache(layer_, st.memcpy_rows, st.pread_rows, 0);
          return std::move(a.set);
        };

        std::optional<AsyncSlice> next_async;
        if (overlap_slices) next_async = start_slice(slices[0]);

        for (size_t si = 0; si < slices.size(); ++si) {
          const std::vector<int>& sw = slices[si];
          ExpertSet sset;
          if (overlap_slices) {
            sset = finish_slice(*next_async, sw);
            next_async.reset();
            if (si + 1 < slices.size()) next_async = start_slice(slices[si + 1]);
          } else {
            auto t0 = std::chrono::steady_clock::now();
            sset = table_->fetch(sw, false);
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
            stats.read_ms += ms;
            stats.expert_bytes += sset.bytes;
            stats.distinct_experts += static_cast<int>(sw.size());
            pf.add_read(ms, static_cast<uint64_t>(sset.bytes));
            pf.add_layer_cache(layer_, sset.rows_memcpy, sset.rows_pread, 0);
          }

          std::vector<int32_t> e_to_loc(static_cast<size_t>(experts_), -1);
          for (size_t j = 0; j < sw.size(); ++j)
            e_to_loc[static_cast<size_t>(sw[j])] = static_cast<int32_t>(j);
          const int n_loc = static_cast<int>(sw.size());

          for (int l = 0; l < n; ++l) {
            const int B = xs[l].shape(0), L = xs[l].shape(1), H = xs[l].shape(2);
            const int K = inds[l].shape(-1);
            const uint32_t* d = to_eval[l].data<uint32_t>();
            const size_t total_pairs = to_eval[l].size();
            std::vector<std::vector<std::pair<uint32_t, uint32_t>>> buckets(static_cast<size_t>(n_loc));
            size_t cnt = 0;
            for (size_t p_idx = 0; p_idx < total_pairs; ++p_idx) {
              const int32_t loc = e_to_loc[d[p_idx]];
              if (loc >= 0) {
                buckets[static_cast<size_t>(loc)].emplace_back(
                    static_cast<uint32_t>(p_idx / static_cast<size_t>(K)),
                    static_cast<uint32_t>(p_idx));
                ++cnt;
              }
            }
            if (cnt == 0) continue;
            std::vector<uint32_t> tok_v, pair_v, loc_v;
            tok_v.reserve(cnt);
            pair_v.reserve(cnt);
            loc_v.reserve(cnt);
            for (int loc = 0; loc < n_loc; ++loc) {
              for (const auto& [t_i, p_i] : buckets[static_cast<size_t>(loc)]) {
                tok_v.push_back(t_i);
                pair_v.push_back(p_i);
                loc_v.push_back(static_cast<uint32_t>(loc));
              }
            }
            const int cnt_i = static_cast<int>(cnt);
            mx::array tok_arr(tok_v.data(), mx::Shape{cnt_i}, mx::uint32);
            mx::array pair_arr(pair_v.data(), mx::Shape{cnt_i}, mx::uint32);
            mx::array sidx(loc_v.data(), mx::Shape{cnt_i}, mx::uint32);
            mx::array x_flat = mx::reshape(xs[l], {B * L, 1, H});
            mx::array x_sel = mx::take(x_flat, tok_arr, 0);
            auto proj_raw = [&](const mx::array& in, int p) -> mx::array {
              std::optional<mx::array> bias_opt =
                  (P[p].has_b && static_cast<size_t>(p) < sset.b.size())
                      ? std::optional<mx::array>(sset.b[p])
                      : std::nullopt;
              return mx::gather_qmm(in, sset.w[p], sset.s[p], bias_opt, std::nullopt, sidx, true,
                                    P[p].group_size, P[p].bits, P[p].mode, /*sorted_indices=*/true);
            };
            mx::array w13_out = proj_raw(x_sel, 0);
            std::optional<mx::array> gs0;
            if (P[0].has_gs) {
              mx::array gs = mx::take(P[0].global_scale, mx::take(mx::flatten(inds[l]), pair_arr, 0), 0);
              mx::Shape gsh(w13_out.ndim(), 1);
              gsh[0] = cnt_i;
              gs0 = mx::reshape(gs, gsh);
            }
            mx::array y_sel = proj_raw(softcap_swiglu_interleaved(w13_out, inter_, gate_softcap_, up_softcap_, gs0), 1);
            mx::array sc_sel = mx::reshape(mx::take(eff_scs[l], pair_arr, 0), {cnt_i, 1, 1});
            mx::array upd = mx::multiply(mx::reshape(y_sel, {cnt_i, 1, H}), sc_sel);
            accs[l] = mx::scatter_add(accs[l], tok_arr, upd, 0);
          }
          mx::eval(accs);
          sset = ExpertSet{};
          mx::clear_cache();
        }
        std::vector<mx::array> outs;
        outs.reserve(static_cast<size_t>(n));
        for (int l = 0; l < n; ++l) outs.push_back(mx::reshape(accs[l], xs[l].shape()));
        return outs;
      }
      auto t0 = std::chrono::steady_clock::now();
      set = table_->fetch(wanted, decode1);
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      stats.read_ms += ms;
      stats.expert_bytes += set.bytes;
      stats.distinct_experts += static_cast<int>(wanted.size());
      pf.add_read(ms, static_cast<uint64_t>(set.bytes));
      pf.add_layer_cache(layer_, set.rows_memcpy, set.rows_pread, 0);
      if (spec) after_fetch_guess(*guess_ix, *guess_sc, spec_fetch_n, spec_top_n, batch_s);
    }

    const auto& P = table_->projs();
    std::vector<mx::array> outs;
    for (int l = 0; l < n; ++l) {
      mx::array idx = set.remap ? mx::take(*set.remap, inds[l], 0) : inds[l];
      const bool do_sort = idx.size() >= 64;
      mx::array x = mx::expand_dims(xs[l], {-2, -3});
      const mx::Shape ishape = idx.shape();
      std::optional<mx::array> order_opt, inv_order;
      mx::array sidx = idx;
      if (do_sort) {
        const int M = idx.shape(-1);
        mx::array flat = mx::flatten(idx);
        mx::array order = mx::argsort(flat);
        order_opt = order;
        inv_order = mx::argsort(order);
        x = mx::take(mx::flatten(x, 0, -3), mx::floor_divide(order, mx::array(M, order.dtype())), 0);
        sidx = mx::take(flat, order, 0);
      }
      auto proj_raw = [&](const mx::array& in, int p) -> mx::array {
        std::optional<mx::array> bias_opt =
            (P[p].has_b && static_cast<size_t>(p) < set.b.size())
                ? std::optional<mx::array>(set.b[p])
                : std::nullopt;
        return mx::gather_qmm(in, set.w[p], set.s[p], bias_opt, std::nullopt, sidx, true,
                              P[p].group_size, P[p].bits, P[p].mode, do_sort);
      };
      mx::array w13_out = proj_raw(x, 0);
      std::optional<mx::array> gs0;
      if (P[0].has_gs) {
        mx::array gs = mx::take(P[0].global_scale, inds[l], 0);
        if (do_sort) {
          gs = mx::take(mx::flatten(gs), *order_opt, 0);
          mx::Shape gsh(w13_out.ndim(), 1);
          gsh[0] = gs.shape(0);
          gs = mx::reshape(gs, gsh);
        } else {
          gs = mx::expand_dims(gs, {-1, -2});
        }
        gs0 = gs;
      }
      mx::array y = proj_raw(softcap_swiglu_interleaved(w13_out, inter_, gate_softcap_, up_softcap_, gs0), 1);
      if (do_sort) {
        y = mx::take(y, *inv_order, 0);
        mx::Shape un(ishape);
        for (int d = 1; d < y.ndim(); ++d) un.push_back(y.shape(d));
        y = mx::reshape(y, un);
      }
      y = mx::squeeze(y, -2);
      // Fold w2 global_scale directly into router scores [B, L, K] instead of scaling the full [B, L, K, H] tensor.
      mx::array eff_sc = scores[l];
      if (P[1].has_gs) {
        mx::array gs1 = mx::take(P[1].global_scale, inds[l], 0);
        eff_sc = mx::astype(mx::multiply(mx::astype(eff_sc, mx::float32), gs1), eff_sc.dtype());
      }
      static auto moe_sum_c = mx::compile(
          [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
            return {mx::sum(mx::multiply(in[0], mx::expand_dims(in[1], -1)), -2)};
          },
          true);
      y = moe_sum_c({y, eff_sc})[0];
      outs.push_back(y);
    }
    if (gate_fetch || sf_fetch) {
      const uint64_t v = gpu_gate::arm();
      if (v == 0) {
        if (sf_fetch) {
          while (sf_fetch->next_ready() >= 0) {
          }
          if (!sf_fetch->ok()) throw std::runtime_error("generic: expert read failed");
        }
        if (gate_fetch) {
          while (gate_fetch->next_ready() >= 0) {
          }
          if (!gate_fetch->ok()) throw std::runtime_error("generic: expert read failed");
        }
        if (gate_after) gate_after();
      } else {
        struct GateJoin {
          uint64_t v;
          std::atomic<int> left;
          std::atomic<bool> ok{true};
          std::vector<mx::array> keep;
          std::function<void()> after;
          std::shared_ptr<rowfetch::Fetch> f1, f2;
        };
        const int cnt = (sf_fetch ? 1 : 0) + (gate_fetch ? 1 : 0);
        auto j = std::make_shared<GateJoin>();
        j->v = v;
        j->left.store(cnt, std::memory_order_relaxed);
        j->after = std::move(gate_after);
        j->f1 = sf_fetch;
        j->f2 = gate_fetch;
        for (const auto* v2 : {&set.w, &set.s, &set.b}) j->keep.insert(j->keep.end(), v2->begin(), v2->end());
        auto finish = [j](bool ok) {
          if (!ok) j->ok.store(false, std::memory_order_relaxed);
          if (j->left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            if (!j->ok.load(std::memory_order_relaxed)) gpu_gate::mark_failed();
            gpu_gate::signal(j->v);
            if (j->after) j->after();
          }
        };
        if (sf_fetch) sf_fetch->on_done(finish);
        if (gate_fetch) gate_fetch->on_done(finish);
        // Submit this layer's MoE graph to Metal immediately behind the GPU gate so the GPU
        // starts executing the microsecond I/O signals while the CPU encodes the next layer.
        mx::async_eval(outs);
      }
    }
    return outs;
  }

 private:
  // One-token decode pipelined execution: launch each routed expert on GPU as soon as its rows land,
  // with the last gate_tail experts behind a GPU gate so the next visit encodes while I/O completes.
  mx::array pipelined(const mx::array& x_in, const mx::array& ix_eval, const mx::array& sc,
                      MlpStats& stats) const {
    auto& pf = cppprof::profiler();
    const uint32_t* d = ix_eval.data<uint32_t>();
    const size_t K = ix_eval.size();
    std::set<int> want;
    for (size_t i = 0; i < K; ++i) want.insert(static_cast<int>(d[i]));
    std::vector<int> wanted(want.begin(), want.end());
    std::vector<int32_t> local(K);
    for (size_t i = 0; i < K; ++i)
      local[i] = static_cast<int32_t>(
          std::lower_bound(wanted.begin(), wanted.end(), static_cast<int>(d[i])) - wanted.begin());
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
        auto proj = [&](const mx::array& in, int p) -> mx::array {
          std::optional<mx::array> bias_opt =
              (P[p].has_b && static_cast<size_t>(p) < set.b.size())
                  ? std::optional<mx::array>(sl(set.b[p]))
                  : std::nullopt;
          mx::array y = mx::gather_qmm(in, sl(set.w[p]), sl(set.s[p]), bias_opt, std::nullopt, r0,
                                       true, P[p].group_size, P[p].bits, P[p].mode, false);
          if (P[p].has_gs) {
            const int eid = wanted[static_cast<size_t>(g)];
            mx::array gs = mx::slice(P[p].global_scale, {eid}, {eid + 1});
            y = mx::astype(mx::multiply(mx::astype(y, mx::float32), gs), y.dtype());
          }
          return y;
        };
        mx::array w13_out = proj(x, 0);
        per[static_cast<size_t>(g)].emplace(
            proj(softcap_swiglu_interleaved(w13_out, inter_, gate_softcap_, up_softcap_), 1));
      };
      const int tail = gpu_gate::tail();
      int launched = 0;
      bool gated = false;
      for (;;) {
        if (tail > 0 && N - launched <= tail) {
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
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    stats.read_ms += ms;
    stats.expert_bytes += set.bytes;
    stats.distinct_experts += static_cast<int>(wanted.size());
    pf.add_read(ms, static_cast<uint64_t>(set.bytes));
    std::vector<mx::array> ordered;
    ordered.reserve(K);
    for (int32_t j : local) ordered.push_back(*per[static_cast<size_t>(j)]);
    mx::array y = mx::concatenate(ordered, 2);
    y = mx::squeeze(y, -2);
    return mx::sum(mx::multiply(y, mx::expand_dims(sc, -1)), -2);
  }

  void prefetch_guess(const mx::array& gi, const mx::array& gs, int top_n) const {
    const int32_t* e = gi.data<int32_t>();
    const float* s = gs.data<float>();
    std::vector<std::pair<float, int>> ranked;
    for (size_t i = 0; i < gi.size(); ++i) ranked.emplace_back(-s[i], e[i]);
    std::sort(ranked.begin(), ranked.end());
    std::vector<int> top;
    for (size_t i = 0; i < ranked.size() && static_cast<int>(top.size()) < top_n; ++i)
      top.push_back(ranked[i].second);
    rowfetch::prefetch_cold(next_->table_->prefetch_rows(top), 8);
  }

  void after_fetch_guess(const mx::array& gi, const mx::array& gs, int sf_n, int st_n, int batch_s) const {
    const int32_t* e = gi.data<int32_t>();
    const float* s = gs.data<float>();
    // Deduplicate across batch_s tokens by summing router scores so multi-token shared experts rank first.
    std::vector<float> acc(static_cast<size_t>(next_->experts_), 0.0f);
    std::vector<int> seen;
    for (size_t i = 0; i < gi.size(); ++i) {
      const int id = e[i];
      if (id < 0 || id >= next_->experts_) continue;
      if (acc[static_cast<size_t>(id)] == 0.0f) seen.push_back(id);
      acc[static_cast<size_t>(id)] += std::max(s[i], 1e-6f);
    }
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(seen.size());
    for (int id : seen) ranked.emplace_back(-acc[static_cast<size_t>(id)], id);
    std::sort(ranked.begin(), ranked.end());
    const int eff_sf = sf_n;
    size_t at = 0;
    if (eff_sf > 0 && next_->top_k_ < 64) {
      std::vector<int> top;
      for (; at < ranked.size() && static_cast<int>(top.size()) < eff_sf; ++at) top.push_back(ranked[at].second);
      if (!top.empty()) next_->speculate(top, batch_s);
    }
    if (st_n > 0) {
      std::vector<int> more;
      for (; at < ranked.size() && static_cast<int>(more.size()) < st_n; ++at) more.push_back(ranked[at].second);
      if (!more.empty()) rowfetch::prefetch_cold(next_->table_->prefetch_rows(more), 8);
    }
  }

  void speculate(const std::vector<int>& experts, int batch_s) const {
    auto sp = std::make_shared<SpecFetch>();
    sp->experts = experts;
    const int cap = std::min<int>(experts_, static_cast<int>(experts.size()) + std::max(1, batch_s) * top_k_);
    sp->set = table_->alloc_stacks(cap);
    std::vector<rowfetch::Row> rows;
    table_->add_rows(sp->set, experts, 0, 0, rows);
    sp->fetch = std::make_unique<rowfetch::Fetch>(std::move(rows), static_cast<int>(experts.size()),
                                                  read_lanes());
    std::lock_guard<std::mutex> lk(spec_mu_);
    specf_ = std::move(sp);
  }

  struct PendingRead {
    ExpertSet set;
    std::unique_ptr<rowfetch::Fetch> fetch;
  };
  mutable std::shared_ptr<PendingRead> pending_;
  struct SpecFetch {
    std::vector<int> experts;
    ExpertSet set;
    std::unique_ptr<rowfetch::Fetch> fetch;
  };
  mutable std::shared_ptr<SpecFetch> specf_;
  mutable std::mutex spec_mu_;
  const Inkling2SigmoidMoe* next_ = nullptr;
  int layer_ = 0, experts_ = 128, top_k_ = 8, inter_ = 1152;
  float route_scale_ = 8.0f, gate_softcap_ = 4.0f, up_softcap_ = 25.0f;
  mx::array router_w_ = mx::array(0.0f);
  std::optional<mx::array> router_b_;
  std::unique_ptr<ExpertLayer> table_;
};

}  // namespace

void register_inkling_blocks(std::map<std::string, MixerFactory>& mixers,
                             std::map<std::string, MlpFactory>& mlps) {
  mixers["attn.inkling2_local"] = [](const BlockCtx& c) {
    return std::unique_ptr<Mixer>(new Inkling2LocalMixer(c));
  };
  mixers["attn.inkling2_full"] = [](const BlockCtx& c) {
    return std::unique_ptr<Mixer>(new Inkling2FullMixer(c));
  };
  mlps["moe.inkling2_sigmoid"] = [](const BlockCtx& c) {
    return std::unique_ptr<Mlp>(new Inkling2SigmoidMoe(c));
  };
}

}  // namespace veystrix::engine::generic
