// probe_layer_intermediates.cpp -- OWNER-RUN diagnostic (new file, no shared edits).
//
// Replays the layer-20 GDN forward from raw intermediates captured by
// dump_python_intermediates.py and compares every C++ intermediate against the
// Python one, with IDENTICAL upstream inputs. This isolates the drift to a
// single op inside the GDN branch at REAL bf16 dims.
//
// Build (from repo root; MLX = /Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages/mlx):
//   xcrun clang++ -std=c++20 -O2 -I<MLX>/include -I<src> -I<decode/src> \
//     <all cpp/engine/src/*.cpp except decoder.cpp,model_adapter.cpp> \
//     cpp/decode/src/io_pager.cpp /tmp/nr_qd2.o cpp/engine/tools/probe_layer_intermediates.cpp \
//     -L<MLX>/lib -lmlx -Wl,-rpath,<MLX>/lib -o /tmp/probe_gdn
// Run:
//   /tmp/probe_gdn /tmp/<dumpdir>
//
// Every file in the dump dir is raw little-endian: f32 for activations/weights
// except w_outproj_weight which is uint32. dtype reconstruction is listed below.
#include "engine.hpp"
#include "gdn_kernel.h"

#include <mlx/mlx.h>
#include <mlx/fast.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

namespace {

std::string join(const std::string& dir, const std::string& name) {
  return dir + "/" + name;
}

std::vector<float> read_f32(const std::string& path, size_t n) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<float> v(n);
  f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
  if (f.gcount() != static_cast<std::streamsize>(n * sizeof(float)))
    throw std::runtime_error("short read " + path);
  return v;
}

std::vector<uint32_t> read_u32(const std::string& path, size_t n) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<uint32_t> v(n);
  f.read(reinterpret_cast<char*>(v.data()), n * sizeof(uint32_t));
  if (f.gcount() != static_cast<std::streamsize>(n * sizeof(uint32_t)))
    throw std::runtime_error("short read " + path);
  return v;
}

size_t prod(const std::vector<int>& s) {
  size_t n = 1;
  for (int x : s) n *= static_cast<size_t>(x);
  return n;
}

mx::Shape shape_of(const std::vector<int>& s) {
  return mx::Shape(s.begin(), s.end());
}

// Load an f32 file and reconstruct as the given MLX dtype.
mx::array load(const std::string& dir, const std::string& name,
               const std::vector<int>& shape, mx::Dtype dt) {
  auto v = read_f32(join(dir, name + ".bin"), prod(shape));
  mx::array a(v.data(), shape_of(shape), mx::float32);
  if (dt != mx::float32) a = mx::astype(a, dt);
  return a;
}

mx::array load_u32(const std::string& dir, const std::string& name,
                   const std::vector<int>& shape) {
  auto v = read_u32(join(dir, name + ".bin"), prod(shape));
  return mx::array(v.data(), shape_of(shape), mx::uint32);
}

std::map<std::string, std::string> read_config(const std::string& dir) {
  std::map<std::string, std::string> m;
  std::ifstream f(join(dir, "config.txt"));
  if (!f) throw std::runtime_error("cannot open config.txt in " + dir);
  std::string line;
  while (std::getline(f, line)) {
    auto p = line.find('=');
    if (p != std::string::npos) m[line.substr(0, p)] = line.substr(p + 1);
  }
  return m;
}

// Compare two arrays (cast to f32) with identical upstream input.
void cmp(const char* name, const mx::array& a, const mx::array& b) {
  mx::array fa = mx::contiguous(mx::astype(a, mx::float32));
  mx::array fb = mx::contiguous(mx::astype(b, mx::float32));
  mx::eval(fa);
  mx::eval(fb);
  const float* pa = fa.data<float>();
  const float* pb = fb.data<float>();
  const size_t n = fa.size();
  double maxd = 0.0, sa = 0.0, sb = 0.0, sab = 0.0;
  size_t exact = 0, nan_diff = 0;
  for (size_t i = 0; i < n; ++i) {
    double x = pa[i], y = pb[i], d = std::fabs(x - y);
    if (d > maxd) maxd = d;
    if (x == y) ++exact;
    if (std::isnan(x) || std::isnan(y)) ++nan_diff;
    sa += x * x; sb += y * y; sab += x * y;
  }
  double corr = (sa > 0 && sb > 0) ? sab / std::sqrt(sa * sb) : 0.0;
  std::printf("  %-26s n=%-9zu maxdiff=%-12.6g corr=%.9f exact=%6.2f%% nan=%zu\n",
              name, n, maxd, corr, 100.0 * exact / n, nan_diff);
}

// Dump a C++ array as raw f32 for external verification.
void dump_f32(const std::string& dir, const std::string& name,
              const mx::array& a) {
  mx::array f = mx::contiguous(mx::astype(a, mx::float32));
  mx::eval(f);
  std::ofstream o(join(dir, name + ".cpp.bin"), std::ios::binary);
  o.write(reinterpret_cast<const char*>(f.data<float>()),
          f.size() * sizeof(float));
}

// Verbatim copy of run_delta_kernel from cpp/engine/src/prefill.cpp (the
// unmasked production kernel; mask=None at a fresh prompt). The kernel source
// is kGdnDeltaKernelSource from gdn_kernel.h, identical to Python's.
mx::array delta_kernel(const mx::array& q, const mx::array& k,
                       const mx::array& v, const mx::array& g,
                       const mx::array& beta, const mx::array& state) {
  const int B = q.shape(0), T = q.shape(1);
  const int Hk = q.shape(2), Dk = q.shape(3);
  const int Hv = v.shape(2), Dv = v.shape(3);
  try {
    auto kernel = mx::fast::metal_kernel(
        "gated_delta_step", {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out"}, kGdnDeltaKernelSource);
    mx::array Tarr(static_cast<int32_t>(T));
    std::vector<mx::array> inputs{q, k, v, g, beta, state, Tarr};
    std::vector<mx::Shape> out_shapes{mx::Shape{B, T, Hv, Dv},
                                      mx::Shape(state.shape())};
    std::vector<mx::Dtype> out_dtypes{q.dtype(), state.dtype()};
    auto grid = std::make_tuple(32, Dv, B * Hv);
    auto threadgroup = std::make_tuple(32, 4, 1);
    std::vector<std::pair<std::string, mx::fast::TemplateArg>> tmpl = {
        {"InT", q.dtype()}, {"StT", state.dtype()},
        {"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk}, {"Hv", Hv},
    };
    auto out = kernel(inputs, out_shapes, out_dtypes, grid, threadgroup, tmpl,
                      std::nullopt, false, mx::StreamOrDevice{});
    return out[0];
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("delta kernel failed: ") + e.what());
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <dump-dir>\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  auto cfg = read_config(dir);

  const int B = 1;
  const int S = std::stoi(cfg.at("seq_len"));
  const int H = std::stoi(cfg.at("hidden_size"));
  const int Hk = std::stoi(cfg.at("num_k_heads"));
  const int Hv = std::stoi(cfg.at("num_v_heads"));
  const int Dk = std::stoi(cfg.at("head_k_dim"));
  const int Dv = std::stoi(cfg.at("head_v_dim"));
  const int K = std::stoi(cfg.at("conv_kernel"));
  const float eps = std::stof(cfg.at("eps"));
  const int key_dim = Hk * Dk, value_dim = Hv * Dv;
  const int conv_dim = 2 * key_dim + value_dim;
  const bool sigmoid = cfg.at("output_activation") == "sigmoid";
  std::printf("[cfg] S=%d H=%d Hk=%d Hv=%d Dk=%d Dv=%d K=%d conv_dim=%d eps=%g act=%s\n",
              S, H, Hk, Hv, Dk, Dv, K, conv_dim, eps,
              sigmoid ? "sigmoid" : "silu");

  // ---- Python reference arrays (identical upstream inputs) ----------------
  auto mixed_input = load(dir, "mixed_input", {B, S, H}, mx::bfloat16);
  auto conv_state_in = load(dir, "conv_state_in", {B, K - 1, conv_dim}, mx::bfloat16);
  auto rec_state_in = load(dir, "rec_state_in", {B, Hv, Dv, Dk}, mx::float32);
  auto py_mixed_qkv = load(dir, "mixed_qkv", {B, S, conv_dim}, mx::bfloat16);
  auto py_z = load(dir, "z", {B, S, Hv, Dv}, mx::bfloat16);
  auto py_a = load(dir, "a", {B, S, Hv}, mx::bfloat16);
  auto py_b = load(dir, "b", {B, S, Hv}, mx::bfloat16);
  auto py_conv_input = load(dir, "conv_input", {B, S + K - 1, conv_dim}, mx::bfloat16);
  auto py_conv_out = load(dir, "conv_out", {B, S, conv_dim}, mx::bfloat16);
  auto py_q_raw = load(dir, "q_raw", {B, S, Hk, Dk}, mx::bfloat16);
  auto py_k_raw = load(dir, "k_raw", {B, S, Hk, Dk}, mx::bfloat16);
  auto py_q_norm = load(dir, "q_norm", {B, S, Hk, Dk}, mx::bfloat16);
  auto py_k_norm = load(dir, "k_norm", {B, S, Hk, Dk}, mx::bfloat16);
  auto py_g = load(dir, "g", {B, S, Hv}, mx::float32);
  auto py_beta = load(dir, "beta", {B, S, Hv}, mx::bfloat16);
  auto py_delta_y = load(dir, "delta_y", {B, S, Hv, Dv}, mx::bfloat16);
  auto py_state_out = load(dir, "state_out", {B, Hv, Dv, Dk}, mx::float32);
  auto py_normed = load(dir, "normed", {B, S, Hv, Dv}, mx::bfloat16);
  auto py_out = load(dir, "out", {B, S, H}, mx::bfloat16);
  auto py_real_out = load(dir, "real_out", {B, S, H}, mx::bfloat16);

  auto w_conv1d = load(dir, "w_conv1d_weight", {conv_dim, K, 1}, mx::bfloat16);
  auto w_A_log = load(dir, "w_A_log", {Hv}, mx::bfloat16);
  auto w_dt_bias = load(dir, "w_dt_bias", {Hv}, mx::bfloat16);
  auto w_norm = load(dir, "w_norm_weight", {Dv}, mx::bfloat16);
  auto w_out_w = load_u32(dir, "w_outproj_weight", {H, value_dim / 8});  // 4-bit
  auto w_out_s = load(dir, "w_outproj_scales", {H, value_dim / 32}, mx::bfloat16);
  auto w_out_b = load(dir, "w_outproj_biases", {H, value_dim / 32}, mx::bfloat16);

  std::printf("\n=== A. Upstream ops (should be bit-exact if probe is faithful) ===\n");
  cmp("mixed_input (py) vs (py)", mixed_input, mixed_input);
  cmp("conv_input C++ vs Python",
      mx::concatenate({conv_state_in, py_mixed_qkv}, 1), py_conv_input);
  cmp("conv_state_out C++ vs Python",
      mx::contiguous(mx::slice(mx::concatenate({conv_state_in, py_mixed_qkv}, 1),
                               mx::Shape{0, S, 0}, mx::Shape{B, S + K - 1, conv_dim})),
      load(dir, "new_conv_state", {B, K - 1, conv_dim}, mx::bfloat16));

  mx::array conv_in = mx::concatenate({conv_state_in, py_mixed_qkv}, 1);
  mx::array conv_out = mx::conv1d(conv_in, w_conv1d, 1, 0, 1, conv_dim);
  conv_out = conv_out * mx::sigmoid(conv_out);  // silu
  cmp("conv_out C++ vs Python", conv_out, py_conv_out);

  mx::array q = mx::reshape(mx::slice(conv_out, mx::Shape{0, 0, 0},
                           mx::Shape{B, S, key_dim}), mx::Shape{B, S, Hk, Dk});
  mx::array k = mx::reshape(mx::slice(conv_out, mx::Shape{0, 0, key_dim},
                           mx::Shape{B, S, 2 * key_dim}), mx::Shape{B, S, Hk, Dk});
  mx::array v = mx::reshape(mx::slice(conv_out, mx::Shape{0, 0, 2 * key_dim},
                           mx::Shape{B, S, conv_dim}), mx::Shape{B, S, Hv, Dv});
  cmp("q_raw C++ vs Python", q, py_q_raw);
  cmp("k_raw C++ vs Python", k, py_k_raw);

  std::printf("\n=== B. Suspect primitives (C++ path, identical inputs) ===\n");
  auto qk = gdn_normalize_qk(q, k);
  cmp("q_norm C++ vs Python", qk.first, py_q_norm);
  cmp("k_norm C++ vs Python", qk.second, py_k_norm);

  auto gb = compute_g_beta(w_A_log, py_a, py_b, w_dt_bias);
  cmp("g C++ vs Python", gb.first, py_g);
  cmp("beta C++ vs Python", gb.second, py_beta);

  std::printf("\n=== C. Fused delta kernel: isolate the kernel itself ===\n");
  // (b) Feed the kernel IDENTICAL q/k/g/beta from Python -> does it match?
  auto dbg = delta_kernel(py_q_norm, py_k_norm, v, py_g, py_beta, rec_state_in);
  cmp("delta_y [py qk,g] vs Python", dbg, py_delta_y);
  // (a) Full C++ path -> total divergence at the kernel output.
  auto dcpp = delta_kernel(qk.first, qk.second, v, gb.first, gb.second, rec_state_in);
  cmp("delta_y [cpp qk,g] vs Python", dcpp, py_delta_y);

  // Compare recurrent state too (kernel state is f32, independent of y dtype).
  // Recompute with state output to compare state_out.
  auto kernel_with_state = [&](const mx::array& qq, const mx::array& kk,
                               const mx::array& vv, const mx::array& gg,
                               const mx::array& bb, const mx::array& st_in,
                               mx::array& state_out) -> mx::array {
    const int T = qq.shape(1);
    auto kernel = mx::fast::metal_kernel(
        "gated_delta_step", {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out"}, kGdnDeltaKernelSource);
    mx::array Tarr(static_cast<int32_t>(T));
    std::vector<mx::array> inputs{qq, kk, vv, gg, bb, st_in, Tarr};
    std::vector<mx::Shape> out_shapes{mx::Shape{B, T, Hv, Dv},
                                      mx::Shape{B, Hv, Dv, Dk}};
    std::vector<mx::Dtype> out_dtypes{qq.dtype(), mx::float32};
    auto grid = std::make_tuple(32, Dv, B * Hv);
    auto threadgroup = std::make_tuple(32, 4, 1);
    std::vector<std::pair<std::string, mx::fast::TemplateArg>> tmpl = {
        {"InT", qq.dtype()}, {"StT", mx::float32},
        {"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk}, {"Hv", Hv},
    };
    auto out = kernel(inputs, out_shapes, out_dtypes, grid, threadgroup, tmpl,
                      std::nullopt, false, mx::StreamOrDevice{});
    state_out = out[1];
    return out[0];
  };
  mx::array st_py = rec_state_in, st_cpp = rec_state_in;
  kernel_with_state(py_q_norm, py_k_norm, v, py_g, py_beta, rec_state_in, st_py);
  kernel_with_state(qk.first, qk.second, v, gb.first, gb.second, rec_state_in,
                    st_cpp);
  cmp("state [py qk,g] vs Python", st_py, py_state_out);
  cmp("state [cpp qk,g] vs Python", st_cpp, py_state_out);

  std::printf("\n=== D. Downstream (full C++ path) ===\n");
  mx::array step_y = mx::astype(dcpp, mixed_input.dtype());
  mx::array normed = rms_norm_gated(step_y, w_norm, py_z, eps, sigmoid);
  cmp("normed C++ vs Python", normed, py_normed);
  Linear out_proj = Linear::quantized(w_out_w, w_out_s, w_out_b,
                                      /*gs=*/32, /*bits=*/4);
  mx::array flat = mx::reshape(normed, mx::Shape{B, S, value_dim});
  mx::array out = linear(flat, out_proj);
  cmp("out (branch) C++ vs Python", out, py_out);
  cmp("python out vs real_out (sanity)", py_out, py_real_out);

  std::printf("\n=== E. FIXED primitives (replicate Python's bf16 dtype semantics) ===\n");
  // FIX 1: _normalize_qk must stay in bf16 end-to-end. C++ array/float promotes
  // to f32, so epsilon and scale must be explicit bf16 scalar arrays; scale is
  // computed in double like Python's `q.shape[-1] ** -0.5`.
  const double scale_d = std::pow(static_cast<double>(Dk), -0.5);
  auto norm_fix = [&](const mx::array& t, bool apply_scale) {
    // Fresh scalar arrays each call: MLX may donate an array on first use.
    mx::array eps = mx::array(1e-6f, t.dtype());
    mx::array y = mx::contiguous(t);
    y = y * mx::rsqrt(mx::sum(mx::square(y), -1, true) + eps);
    // Qwen4ExpGatedDeltaNet._normalize_qk applies scale ONLY to q, not k.
    if (apply_scale) {
      mx::array sc = mx::array(scale_d, t.dtype());
      y = y * sc;
    }
    return y;
  };
  // Use the freshly loaded Python q/k_raw so any donated intermediate from the
  // earlier kernel calls cannot corrupt the inputs.
  mx::array qn_fix = norm_fix(py_q_raw, true);
  mx::array kn_fix = norm_fix(py_k_raw, false);
  {
    mx::array e = mx::contiguous(mx::astype(mx::array(1e-6f, mx::bfloat16), mx::float32));
    mx::array s = mx::contiguous(mx::astype(mx::array(scale_d, mx::bfloat16), mx::float32));
    mx::eval(e); mx::eval(s);
    std::printf("[fix scalars] eps(1e-6)->bf16->f32 = %.9g ; scale(%.17g)->bf16->f32 = %.9g\n",
                e.data<float>()[0], scale_d, s.data<float>()[0]);
  }
  dump_f32(dir, "cpp_qn_fix", qn_fix);
  dump_f32(dir, "cpp_kn_fix", kn_fix);
  cmp("FIX q_norm C++ vs Python", qn_fix, py_q_norm);
  cmp("FIX k_norm C++ vs Python", kn_fix, py_k_norm);

  // FIX 2: softplus must be computed in bf16. C++ logaddexp(x, f32(0)) promotes
  // to f32; use a bf16 zero so it matches Python's nn.softplus(bf16) exactly.
  mx::array A_fix = mx::exp(mx::astype(w_A_log, mx::float32));
  mx::array sp_fix = mx::logaddexp(py_a + w_dt_bias,
                                   mx::array(0.0f, mx::bfloat16));
  mx::array g_fix = mx::exp(-A_fix * sp_fix);
  cmp("FIX g C++ vs Python", g_fix, py_g);

  // Feed the FIXED q/k/g into the kernel and downstream: should be bit-exact.
  mx::array v_fix = mx::reshape(
      mx::slice(py_conv_out, mx::Shape{0, 0, 2 * key_dim},
                mx::Shape{B, S, conv_dim}), mx::Shape{B, S, Hv, Dv});
  mx::array rec_fix = mx::array(rec_state_in);  // fresh state buffer
  mx::array st_fix = rec_fix;
  mx::array dy_fix = kernel_with_state(qn_fix, kn_fix, v_fix, g_fix, py_beta,
                                       rec_fix, st_fix);
  cmp("FIX delta_y C++ vs Python", dy_fix, py_delta_y);
  cmp("FIX state C++ vs Python", st_fix, py_state_out);
  mx::array dy_fix_bf = mx::astype(dy_fix, mixed_input.dtype());
  mx::array normed_fix = rms_norm_gated(dy_fix_bf, w_norm, py_z, eps, sigmoid);
  cmp("FIX normed C++ vs Python", normed_fix, py_normed);
  Linear out_proj_fix = Linear::quantized(w_out_w, w_out_s, w_out_b, 32, 4);
  mx::array out_fix = linear(mx::reshape(normed_fix, mx::Shape{B, S, value_dim}),
                             out_proj_fix);
  cmp("FIX out (branch) C++ vs Python", out_fix, py_out);

  // External-verification dumps.
  dump_f32(dir, "cpp_q_norm", qk.first);
  dump_f32(dir, "cpp_k_norm", qk.second);
  dump_f32(dir, "cpp_g", gb.first);
  dump_f32(dir, "cpp_delta_y_pyinputs", dbg);
  dump_f32(dir, "cpp_delta_y_cppinputs", dcpp);
  dump_f32(dir, "cpp_state_pyinputs", st_py);
  dump_f32(dir, "cpp_state_cppinputs", st_cpp);
  dump_f32(dir, "cpp_normed", normed);
  dump_f32(dir, "cpp_out", out);

  std::printf("\nDONE. Dumps written next to the Python .bin files as *.cpp.bin\n");
  return 0;
}
