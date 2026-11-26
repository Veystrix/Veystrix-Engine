// Synthetic BIT-EXACT test: batched (S>1) vs sequential (S=1) GDN prefill.
//
// Companion to tools/gen_prefill_modes_vectors.py. Loads the dumped Python
// oracles for a small seeded Qwen4ExpGatedDeltaNet and verifies, inside C++:
//
//   [A] gdn_prefill(x[S])                       == Python BATCHED oracle  (bit-exact)
//   [B] gdn_layer_forward(x[t:t+1]) x S          == Python SEQUENTIAL oracle (bit-exact)
//       (this is the tool's real_checkpoint_generate.cpp prefill path)
//   [C] gdn_prefill(x[t:t+1]) x S                vs Python SEQUENTIAL oracle (reported)
//       (isolates the gdn_prefill primitive itself at S=1)
//
// Then reports the NUMERIC GAP [max abs diff / Pearson r / bit-identical]
// between the C++ batched output and both C++ sequential outputs — i.e. what
// the tool's S=1 loop costs at this config.
//
// Usage: test_prefill_modes <vectordir>
#include "../src/prefill.hpp"
#include "../src/gdn_layer.hpp"

#include <mlx/mlx.h>
#include <mlx/fast.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

// Outputs held across the mode runs so the gaps can be reported at the end.
static std::optional<mx::array> g_batched_out, g_tool_seq_out, g_prim_seq_out;

// --------------------------------------------------------------------------- //
// Dense full-attention fallback (mirrors real_checkpoint_generate.cpp's
// full_attention_dense and the Python oracle's DenseAttention). Self-contained
// here so this test does not depend on the tool's generation driver.
// --------------------------------------------------------------------------- //
struct FAWeights {
  Linear q, k, v, o;
  mx::array q_norm, k_norm;
  FAWeights(Linear q_, Linear k_, Linear v_, Linear o_, mx::array qn, mx::array kn)
      : q(std::move(q_)), k(std::move(k_)), v(std::move(v_)), o(std::move(o_)),
        q_norm(std::move(qn)), k_norm(std::move(kn)) {}
};

static mx::array fa_rms(const mx::array& x, const mx::array& w, float eps = 1e-6f) {
  mx::array y = mx::astype(x, mx::float32);
  mx::array ms = mx::mean(mx::square(y), -1, /*keepdims=*/true);
  y = y * mx::rsqrt(ms + eps);
  mx::array gain = 1.0f + mx::astype(w, mx::float32);
  y = y * gain;
  return mx::astype(mx::reshape(y, x.shape()), x.dtype());
}

static mx::array fa_forward(const mx::array& x, const FAWeights& w,
                            int Hq, int Hkv, int D,
                            std::optional<mx::array>& ck,
                            std::optional<mx::array>& cv) {
  const int B = static_cast<int>(x.shape(0)), S = static_cast<int>(x.shape(1));
  mx::array qproj = linear(x, w.q), kproj = linear(x, w.k), vproj = linear(x, w.v);
  mx::array q4 = mx::reshape(qproj, mx::Shape{B, S, Hq, 2 * D});
  std::vector<mx::array> halves = mx::split(q4, 2, -1);
  mx::array q = halves[0];
  mx::array gate = mx::reshape(halves[1], mx::Shape{B, S, Hq * D});
  q = mx::transpose(fa_rms(q, w.q_norm), {0, 2, 1, 3});
  mx::array k = mx::transpose(
      fa_rms(mx::reshape(kproj, mx::Shape{B, S, Hkv, D}), w.k_norm), {0, 2, 1, 3});
  mx::array v = mx::transpose(
      mx::reshape(vproj, mx::Shape{B, S, Hkv, D}), {0, 2, 1, 3});
  if (!ck) {
    ck = k;
    cv = v;
  } else {
    ck = mx::concatenate({*ck, k}, 2);
    cv = mx::concatenate({*cv, v}, 2);
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  mx::array out = mx::fast::scaled_dot_product_attention(
      q, *ck, *cv, scale, /*mask_mode=*/"causal", /*mask_arr=*/std::nullopt,
      /*sinks=*/std::nullopt);
  out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), mx::Shape{B, S, Hq * D});
  return linear(out * mx::sigmoid(gate), w.o);
}

static bool bit_equal(const mx::array& a, const mx::array& b) {
  if (a.shape() != b.shape() || a.dtype() != b.dtype()) return false;
  mx::array av = mx::view(mx::reshape(a, {-1}), mx::uint32);
  mx::array bv = mx::view(mx::reshape(b, {-1}), mx::uint32);
  mx::eval(av, bv);
  if (av.size() != bv.size()) return false;
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  for (size_t i = 0; i < av.size(); ++i)
    if (pa[i] != pb[i]) return false;
  return true;
}

static void dump_first_diff(const char* label, const mx::array& a,
                            const mx::array& b) {
  mx::array av = mx::view(mx::reshape(a, {-1}), mx::uint32);
  mx::array bv = mx::view(mx::reshape(b, {-1}), mx::uint32);
  mx::eval(av, bv);
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  for (size_t i = 0; i < av.size(); ++i) {
    if (pa[i] != pb[i]) {
      std::fprintf(stderr, "  %s first diff at %zu: got 0x%08x expected 0x%08x\n",
                   label, i, pa[i], pb[i]);
      return;
    }
  }
}

static std::vector<float> to_host(const mx::array& a) {
  mx::array f = mx::reshape(mx::astype(a, mx::float32), {-1});
  mx::eval(f);
  const float* p = f.data<float>();
  return std::vector<float>(p, p + f.size());
}

// max abs diff / Pearson r / bit-identical for two equal-shaped arrays.
static void report_gap(const char* label, const mx::array& a, const mx::array& b) {
  std::vector<float> va = to_host(a), vb = to_host(b);
  const size_t n = std::min(va.size(), vb.size());
  if (n == 0 || va.size() != vb.size()) {
    std::printf("  %-34s size mismatch\n", label);
    return;
  }
  double maxd = 0.0, sa = 0.0, sb = 0.0, saa = 0.0, sbb = 0.0, sab = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double x = va[i], y = vb[i];
    maxd = std::fmax(maxd, std::fabs(x - y));
    sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
  }
  double dn = static_cast<double>(n);
  double num = dn * sab - sa * sb;
  double den = std::sqrt((dn * saa - sa * sa) * (dn * sbb - sb * sb));
  double corr = den > 0 ? num / den : std::nan("");
  std::printf("  %-34s max_abs_diff=%.9g  corr=%+.9f  bit_identical=%s\n",
              label, maxd, corr, bit_equal(a, b) ? "true" : "false");
}

struct Case {
  mx::array x, i_conv, i_rec, exp_out, exp_conv, exp_rec;
  Case(mx::array x_, mx::array ic, mx::array ir,
       mx::array eo, mx::array ec, mx::array er)
      : x(std::move(x_)), i_conv(std::move(ic)), i_rec(std::move(ir)),
        exp_out(std::move(eo)), exp_conv(std::move(ec)), exp_rec(std::move(er)) {}
};

static Case load_case(const std::string& dir) {
  auto load = [&](const char* n) { return mx::load(dir + "/" + n + ".npy"); };
  return Case(load("x"), load("i_conv_state"), load("i_rec_state"),
              load("expected_output"), load("expected_conv_state"),
              load("expected_rec_state"));
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <vectordir>\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  auto load = [&](const char* n) { return mx::load(dir + "/" + n + ".npy"); };

  // Shared weights (identical for both modes).
  GDNWeights weights(load("w_qkv"), load("w_z"), load("w_b"), load("w_a"),
                     load("conv_w"), load("A_log"), load("dt_bias"),
                     load("norm_w"), load("w_out"));

  Case bat = load_case(dir + "/batched");
  Case seq = load_case(dir + "/sequential");

  // Derive geometry from the saved weight shapes (tracks the generator).
  const mx::array& wqkv = *weights.in_proj_qkv.dense;
  const mx::array& wz = *weights.in_proj_z.dense;
  const int H = static_cast<int>(wqkv.shape(1));
  const int conv_dim = static_cast<int>(wqkv.shape(0));
  const int value_dim = static_cast<int>(wz.shape(0));
  const int key_dim = (conv_dim - value_dim) / 2;
  const int Hv = static_cast<int>(weights.A_log.shape(0));
  const int Dv = static_cast<int>(weights.norm_weight.shape(0));
  const int Dk = static_cast<int>(bat.i_rec.shape(3));  // config head_k_dim
  const int Hk = key_dim / Dk;
  const int K = static_cast<int>(bat.i_conv.shape(1)) + 1;  // config conv_kernel
  GDNConfig cfg{H, key_dim, value_dim, Hk, Hv, Dk, Dv, K,
                1e-6f, GDNActivation::Sigmoid};

  const int S = static_cast<int>(bat.x.shape(1));
  const int B = static_cast<int>(bat.x.shape(0));

  std::printf("prefill_modes: H=%d key=%d val=%d Hk=%d Hv=%d Dk=%d Dv=%d K=%d B=%d S=%d\n",
              H, key_dim, value_dim, Hk, Hv, Dk, Dv, K, B, S);

  bool all_ok = true;

  // ---- [A] C++ batched primitive == Python batched oracle -----------------
  {
    GDNCacheState cache(bat.i_conv, bat.i_rec);
    GDNLayerOutput got = gdn_prefill(bat.x, weights, cache, cfg, nullptr);
    mx::eval(got.output, got.conv_state, got.recurrent_state);
    bool ok = bit_equal(got.output, bat.exp_out) &&
              bit_equal(got.conv_state, bat.exp_conv) &&
              bit_equal(got.recurrent_state, bat.exp_rec);
    if (!ok) {
      std::fprintf(stderr, "FAIL [A] gdn_prefill(S=%d) != Python BATCHED oracle\n", S);
      if (!bit_equal(got.output, bat.exp_out)) dump_first_diff("output", got.output, bat.exp_out);
      if (!bit_equal(got.conv_state, bat.exp_conv)) dump_first_diff("conv", got.conv_state, bat.exp_conv);
      if (!bit_equal(got.recurrent_state, bat.exp_rec)) dump_first_diff("rec", got.recurrent_state, bat.exp_rec);
      all_ok = false;
    } else {
      std::printf("[A] gdn_prefill(S=%d) == Python BATCHED oracle: OK (bit-exact)\n", S);
    }
    // Keep the result for the gap report.
    g_batched_out = got.output;
  }

  // ---- [B] C++ tool path (gdn_layer_forward x S) == Python SEQUENTIAL -----
  {
    GDNCacheState cache(seq.i_conv, seq.i_rec);
    std::vector<mx::array> steps;
    steps.reserve(S);
    for (int t = 0; t < S; ++t) {
      mx::array xt = mx::slice(seq.x, mx::Shape{0, t, 0}, mx::Shape{B, t + 1, H});
      GDNLayerOutput g = gdn_layer_forward(xt, weights, cache, cfg);
      mx::eval(g.output, g.conv_state, g.recurrent_state);
      steps.push_back(g.output);
      cache = GDNCacheState(g.conv_state, g.recurrent_state);
    }
    mx::array out = mx::concatenate(steps, 1);
    mx::eval(out, cache.conv_state, cache.recurrent_state);
    bool ok = bit_equal(out, seq.exp_out) &&
              bit_equal(cache.conv_state, seq.exp_conv) &&
              bit_equal(cache.recurrent_state, seq.exp_rec);
    if (!ok) {
      std::fprintf(stderr, "FAIL [B] gdn_layer_forward x %d != Python SEQUENTIAL oracle\n", S);
      if (!bit_equal(out, seq.exp_out)) dump_first_diff("output", out, seq.exp_out);
      if (!bit_equal(cache.conv_state, seq.exp_conv)) dump_first_diff("conv", cache.conv_state, seq.exp_conv);
      if (!bit_equal(cache.recurrent_state, seq.exp_rec)) dump_first_diff("rec", cache.recurrent_state, seq.exp_rec);
      all_ok = false;
    } else {
      std::printf("[B] gdn_layer_forward x %d == Python SEQUENTIAL oracle: OK (bit-exact)\n", S);
    }
    g_tool_seq_out = out;
  }

  // ---- [C] C++ gdn_prefill(S=1) x S vs Python SEQUENTIAL (reported) -------
  {
    GDNCacheState cache(seq.i_conv, seq.i_rec);
    std::vector<mx::array> steps;
    steps.reserve(S);
    for (int t = 0; t < S; ++t) {
      mx::array xt = mx::slice(seq.x, mx::Shape{0, t, 0}, mx::Shape{B, t + 1, H});
      GDNLayerOutput g = gdn_prefill(xt, weights, cache, cfg, nullptr);
      mx::eval(g.output, g.conv_state, g.recurrent_state);
      steps.push_back(g.output);
      cache = GDNCacheState(g.conv_state, g.recurrent_state);
    }
    mx::array out = mx::concatenate(steps, 1);
    mx::eval(out, cache.conv_state, cache.recurrent_state);
    bool ok = bit_equal(out, seq.exp_out);
    std::printf("[C] gdn_prefill(S=1) x %d vs Python SEQUENTIAL: %s\n", S,
                ok ? "bit-identical" : "DIFFERS (reported, not a gate)");
    g_prim_seq_out = out;
  }

  // ---- Numeric gap: C++ batched vs C++ sequential -------------------------
  std::printf("\n=== C++ batched vs sequential gap (S=%d) ===\n", S);
  report_gap("batched vs tool-seq (gdn_layer_forward)", *g_batched_out, *g_tool_seq_out);
  report_gap("batched vs prim-seq (gdn_prefill S=1)", *g_batched_out, *g_prim_seq_out);
  report_gap("tool-seq vs prim-seq", *g_tool_seq_out, *g_prim_seq_out);
  report_gap("batched vs Python SEQUENTIAL oracle", *g_batched_out, seq.exp_out);

  // ---- [D/E] Full-attention DENSE fallback: batched vs sequential ----------
  {
    const std::string fdir = dir + "/full_attn";
    FAWeights faw(mx::load(fdir + "/fa_q_w.npy"), mx::load(fdir + "/fa_k_w.npy"),
                  mx::load(fdir + "/fa_v_w.npy"), mx::load(fdir + "/fa_o_w.npy"),
                  mx::load(fdir + "/fa_q_norm_w.npy"),
                  mx::load(fdir + "/fa_k_norm_w.npy"));
    const int D = static_cast<int>(faw.q_norm.shape(0));
    const int Hq = static_cast<int>(faw.q.dense->shape(0)) / (2 * D);
    const int Hkv = static_cast<int>(faw.k.dense->shape(0)) / D;

    mx::array fx_b = mx::load(fdir + "/batched/x.npy");
    mx::array fexp_b = mx::load(fdir + "/batched/expected_output.npy");
    std::optional<mx::array> ckb, cvb;
    mx::array fgot_b = fa_forward(fx_b, faw, Hq, Hkv, D, ckb, cvb);
    mx::eval(fgot_b);
    bool okb = bit_equal(fgot_b, fexp_b);
    if (!okb) {
      std::fprintf(stderr, "FAIL [D] full-attn batched != Python oracle\n");
      dump_first_diff("fa output", fgot_b, fexp_b);
      all_ok = false;
    } else {
      std::printf("[D] full-attn batched == Python oracle: OK (bit-exact)\n");
    }

    mx::array fx_s = mx::load(fdir + "/sequential/x.npy");
    mx::array fexp_s = mx::load(fdir + "/sequential/expected_output.npy");
    std::optional<mx::array> cks, cvs;
    std::vector<mx::array> fsteps;
    fsteps.reserve(S);
    for (int t = 0; t < S; ++t) {
      mx::array xt = mx::slice(fx_s, mx::Shape{0, t, 0}, mx::Shape{B, t + 1, H});
      fsteps.push_back(fa_forward(xt, faw, Hq, Hkv, D, cks, cvs));
    }
    mx::array fgot_s = mx::concatenate(fsteps, 1);
    mx::eval(fgot_s);
    bool oks = bit_equal(fgot_s, fexp_s);
    if (!oks) {
      std::fprintf(stderr, "FAIL [E] full-attn sequential != Python oracle\n");
      dump_first_diff("fa output", fgot_s, fexp_s);
      all_ok = false;
    } else {
      std::printf("[E] full-attn sequential x %d == Python oracle: OK (bit-exact)\n", S);
    }

    std::printf("\n=== Full-attention DENSE gap (S=%d) ===\n", S);
    report_gap("batched vs sequential (full-attn)", fgot_b, fgot_s);
  }

  if (!all_ok) return 1;
  std::printf("\ntest_prefill_modes OK (bit-exact oracles A+B; gap reported)\n");
  return 0;
}
