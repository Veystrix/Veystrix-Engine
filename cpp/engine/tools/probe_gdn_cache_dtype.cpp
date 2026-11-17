// probe_gdn_cache_dtype.cpp -- OWNER-RUN diagnostic (new file, no shared edits).
// Decisive test of the GDN conv-cache dtype hypothesis: run gdn_prefill at L0
// with the SAME inputs/weights, once with conv_state in bf16 (Python's dtype)
// and once in f32 (the end-to-end tool's cache init dtype), and compare both
// outputs to Python's real GDN output.
//
// Usage: /tmp/probe_gdn_dtype <gdn-dump-dir> <cpp-weight-dir>
//   gdn-dump-dir  : output of dump_python_intermediates.py (mixed_input,
//                   conv_state_in, rec_state_in, out, gdn_out)
//   cpp-weight-dir: output of probe_l0_weights.cpp (gdn_* triplets, w_* raw)
#include "prefill.hpp"
#include "gdn_layer.hpp"

#include <mlx/mlx.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

namespace {
std::string J(const std::string& d, const std::string& n) { return d + "/" + n + ".bin"; }
mx::Shape SH(const std::vector<int>& s) { return mx::Shape(s.begin(), s.end()); }
size_t PR(const std::vector<int>& s) { size_t n = 1; for (int x : s) n *= x; return n; }

std::vector<float> rf(const std::string& p, size_t n) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  std::vector<float> v(n);
  f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
  return v;
}
std::vector<uint32_t> ru(const std::string& p, size_t n) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  std::vector<uint32_t> v(n);
  f.read(reinterpret_cast<char*>(v.data()), n * sizeof(uint32_t));
  return v;
}
mx::array ld(const std::string& d, const std::string& n,
             const std::vector<int>& s, mx::Dtype dt) {
  auto v = rf(J(d, n), PR(s));
  mx::array a(v.data(), SH(s), mx::float32);
  if (dt != mx::float32) a = mx::astype(a, dt);
  return a;
}
mx::array ldu(const std::string& d, const std::string& n,
              const std::vector<int>& s) {
  auto v = ru(J(d, n), PR(s));
  return mx::array(v.data(), SH(s), mx::uint32);
}
Linear lql(const std::string& d, const std::string& n, int out, int in,
           int gs, int bits) {
  return Linear::quantized(ldu(d, n + "_weight", {out, in * bits / 32}),
                           ld(d, n + "_scales", {out, in / gs}, mx::bfloat16),
                           ld(d, n + "_biases", {out, in / gs}, mx::bfloat16),
                           gs, bits);
}
void cmp(const char* name, const mx::array& a, const mx::array& b) {
  mx::array x = mx::contiguous(mx::astype(a, mx::float32));
  mx::array y = mx::contiguous(mx::astype(b, mx::float32));
  mx::eval(x); mx::eval(y);
  const float* px = x.data<float>(); const float* py = y.data<float>();
  size_t n = x.size(); double md = 0, sa = 0, sb = 0, sab = 0; size_t ex = 0, nd = 0;
  for (size_t i = 0; i < n; ++i) {
    double u = px[i], v = py[i], df = std::fabs(u - v);
    if (df > md) md = df;
    if (u == v) ++ex; else ++nd;
    sa += u*u; sb += v*v; sab += u*v;
  }
  double corr = sab / std::sqrt(sa * sb);
  std::printf("  %-40s n=%-8zu maxdiff=%-12.6g corr=%.9f exact=%6.2f%% ndiff=%zu\n",
              name, n, md, corr, 100.0*ex/n, nd);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <gdn-dump> <cpp-weights>\n", argv[0]);
    return 2;
  }
  const std::string gd = argv[1], cw = argv[2];
  const int S = 16, H = 2560, Hk = 16, Hv = 48, Dk = 128, Dv = 128, K = 4;
  const int key_dim = Hk*Dk, value_dim = Hv*Dv, conv_dim = 2*key_dim + value_dim;
  const int gs = 32, bits = 4;

  auto mixed = ld(gd, "mixed_input", {1,S,H}, mx::bfloat16);
  auto conv_bf16 = ld(gd, "conv_state_in", {1,K-1,conv_dim}, mx::bfloat16);
  auto rec = ld(gd, "rec_state_in", {1,Hv,Dv,Dk}, mx::float32);
  auto py_out = ld(gd, "out", {1,S,H}, mx::bfloat16);

  GDNWeights w(
      lql(cw, "gdn_in_proj_qkv", conv_dim, H, gs, bits),
      lql(cw, "gdn_in_proj_z", value_dim, H, gs, bits),
      lql(cw, "gdn_in_proj_b", Hv, H, gs, bits),
      lql(cw, "gdn_in_proj_a", Hv, H, gs, bits),
      ld(cw, "w_conv1d_weight", {conv_dim, K, 1}, mx::bfloat16),
      ld(cw, "w_A_log", {Hv}, mx::bfloat16),
      ld(cw, "w_dt_bias", {Hv}, mx::bfloat16),
      ld(cw, "w_norm_weight", {Dv}, mx::bfloat16),
      lql(cw, "w_outproj", H, value_dim, gs, bits));
  GDNConfig cfg{H, key_dim, value_dim, Hk, Hv, Dk, Dv, K, 1e-6f,
                GDNActivation::Sigmoid};

  // Sanity: reconstitute the conv_input dtype in each case.
  {
    mx::array a = mx::concatenate({conv_bf16, mx::zeros(mx::Shape{1,S,conv_dim}, mx::bfloat16)}, 1);
    mx::array b = mx::concatenate({mx::astype(conv_bf16, mx::float32),
                                   mx::zeros(mx::Shape{1,S,conv_dim}, mx::bfloat16)}, 1);
    std::printf("[dtype] conv_input bf16-cache=%ld  f32-cache=%ld\n",
                (long)static_cast<mx::Dtype::Val>(a.dtype()),
                (long)static_cast<mx::Dtype::Val>(b.dtype()));
  }

  std::printf("\n=== gdn_prefill L0: conv_state bf16 (Python) vs f32 (tool cache) ===\n");
  GDNLayerOutput r_bf = gdn_prefill(mixed, w, GDNCacheState(conv_bf16, rec), cfg, nullptr);
  cmp("gdn out with bf16 conv_state vs Python", r_bf.output, py_out);
  GDNLayerOutput r_f32 = gdn_prefill(
      mixed, w, GDNCacheState(mx::astype(conv_bf16, mx::float32), rec), cfg, nullptr);
  cmp("gdn out with f32  conv_state vs Python", r_f32.output, py_out);
  cmp("gdn out f32-cache vs bf16-cache", r_f32.output, r_bf.output);
  // Dump the f32-cache branch so probe_moe_intermediates can replay the exact
  // end-to-end L0 hidden and compare to /tmp/bdbg5/L.L00.f32.
  {
    mx::array f = mx::contiguous(mx::astype(r_f32.output, mx::float32));
    mx::eval(f);
    std::ofstream o(J(gd, "gdn_out_f32cache"));
    o.write(reinterpret_cast<const char*>(f.data<float>()),
            static_cast<std::streamsize>(f.size() * sizeof(float)));
  }
  return 0;
}
