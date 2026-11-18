// probe_moe_intermediates.cpp -- OWNER-RUN diagnostic (new file, no shared edits).
//
// Replays the layer-0 MoE forward from raw intermediates captured by
// dump_python_moe_intermediates.py and compares every C++ intermediate against
// Python, with IDENTICAL x / indices / scores / expert weights. Isolates the
// MoE COMPUTE from routing and from weight loading.
//
// Build (from repo root; MLX = <venv>/site-packages/mlx):
//   bash /tmp/build_probe_moe.sh   (see docs/engineering/moe_drift_localization.md)
// Run:  /tmp/probe_moe /tmp/moe_probe_l0
#include "engine.hpp"

#include <mlx/mlx.h>
#include <mlx/fast.h>

#include <algorithm>
#include <dirent.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <fcntl.h>
#include <map>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

namespace {

std::string join(const std::string& d, const std::string& n) { return d + "/" + n; }

// Python's nn.silu is `@partial(mx.compile, shapeless=True) x * mx.sigmoid(x)`
// (mlx/nn/layers/activations.py:138). Compiling the SAME graph here makes MLX
// emit the same fused JIT kernel instead of two precompiled metallib kernels.
mx::array silu_compiled(const mx::array& x) {
  static auto fn = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        return {in[0] * mx::sigmoid(in[0])};
      },
      /*shapeless=*/true);
  return fn({x})[0];
}

bool exists(const std::string& p) { return std::ifstream(p).good(); }

size_t file_size(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + p);
  return static_cast<size_t>(f.tellg());
}

std::vector<float> read_f32(const std::string& p, size_t n) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  std::vector<float> v(n);
  f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
  if (f.gcount() != static_cast<std::streamsize>(n * sizeof(float)))
    throw std::runtime_error("short read " + p);
  return v;
}

std::vector<uint32_t> read_u32(const std::string& p, size_t n) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  std::vector<uint32_t> v(n);
  f.read(reinterpret_cast<char*>(v.data()), n * sizeof(uint32_t));
  if (f.gcount() != static_cast<std::streamsize>(n * sizeof(uint32_t)))
    throw std::runtime_error("short read " + p);
  return v;
}

mx::Shape sh(const std::vector<int>& s) { return mx::Shape(s.begin(), s.end()); }
size_t prod(const std::vector<int>& s) {
  size_t n = 1;
  for (int x : s) n *= static_cast<size_t>(x);
  return n;
}

mx::array load(const std::string& d, const std::string& name,
               const std::vector<int>& shape, mx::Dtype dt) {
  auto v = read_f32(join(d, name + ".bin"), prod(shape));
  mx::array a(v.data(), sh(shape), mx::float32);
  if (dt != mx::float32) a = mx::astype(a, dt);
  return a;
}

mx::array load_u32(const std::string& d, const std::string& name,
                   const std::vector<int>& shape) {
  auto v = read_u32(join(d, name + ".bin"), prod(shape));
  return mx::array(v.data(), sh(shape), mx::uint32);
}

std::map<std::string, std::string> read_config(const std::string& d) {
  std::map<std::string, std::string> m;
  std::ifstream f(join(d, "config.txt"));
  if (!f) throw std::runtime_error("cannot open config.txt in " + d);
  std::string line;
  while (std::getline(f, line)) {
    auto p = line.find('=');
    if (p != std::string::npos) m[line.substr(0, p)] = line.substr(p + 1);
  }
  return m;
}

void dump_f32(const std::string& dir, const std::string& name,
              const mx::array& a) {
  mx::array f = mx::contiguous(mx::astype(a, mx::float32));
  mx::eval(f);
  std::ofstream o(join(dir, name + ".cpp.bin"), std::ios::binary);
  o.write(reinterpret_cast<const char*>(f.data<float>()),
          static_cast<std::streamsize>(f.size() * sizeof(float)));
}

void cmp(const char* name, const mx::array& a, const mx::array& b) {
  mx::array fa = mx::contiguous(mx::astype(a, mx::float32));
  mx::array fb = mx::contiguous(mx::astype(b, mx::float32));
  mx::eval(fa);
  mx::eval(fb);
  const float* pa = fa.data<float>();
  const float* pb = fb.data<float>();
  const size_t n = fa.size();
  double maxd = 0, sa = 0, sb = 0, sab = 0;
  size_t exact = 0, ndiff = 0;
  size_t firstdiff = 0;
  float fv1 = 0, fv2 = 0;
  for (size_t i = 0; i < n; ++i) {
    double x = pa[i], y = pb[i], dd = std::fabs(x - y);
    if (dd > maxd) maxd = dd;
    if (x == y) ++exact;
    else { if (ndiff == 0) { firstdiff = i; fv1 = pa[i]; fv2 = pb[i]; } ++ndiff; }
    sa += x * x; sb += y * y; sab += x * y;
  }
  double corr = (sa > 0 && sb > 0) ? sab / std::sqrt(sa * sb) : 0.0;
  std::printf("  %-34s n=%-9zu maxdiff=%-12.6g corr=%.9f exact=%6.2f%% ndiff=%zu",
              name, n, maxd, corr, 100.0 * exact / n, ndiff);
  if (ndiff) std::printf(" first[%zu]=%.7g/%.7g", firstdiff, fv1, fv2);
  std::printf("\n");
}

// Compare integer arrays exactly (router indices).
void cmp_int(const char* name, const mx::array& a, const mx::array& b) {
  mx::array ia = mx::contiguous(mx::astype(a, mx::int32));
  mx::array ib = mx::contiguous(mx::astype(b, mx::int32));
  mx::eval(ia); mx::eval(ib);
  const int32_t* pa = ia.data<int32_t>();
  const int32_t* pb = ib.data<int32_t>();
  size_t n = ia.size(), diff = 0;
  for (size_t i = 0; i < n; ++i) if (pa[i] != pb[i]) ++diff;
  std::printf("  %-34s n=%-9zu mismatches=%zu/%zu (%.2f%%)\n",
              name, n, diff, n, 100.0 * diff / n);
}

// Read one expert row of a BF16 safetensors tensor directly from disk (the
// same uint16 bytes the C++ pager reinterprets as bf16) and compare to the
// Python store.rows() -> view(bfloat16) values dumped by the Python tool.
void check_raw_bytes(const std::string& ckpt_dir, const std::string& key,
                     int expert, const mx::array& py_row, int out_dim,
                     int ng) {
  // Find the shard and read its header.
  std::string shard;
  {
    DIR* d = ::opendir(ckpt_dir.c_str());
    if (!d) throw std::runtime_error("cannot open checkpoint dir");
    std::vector<std::string> names;
    while (struct dirent* e = ::readdir(d)) {
      std::string n(e->d_name);
      if (n.size() > 12 && n.substr(n.size() - 12) == ".safetensors")
        names.push_back(n);
    }
    ::closedir(d);
    std::sort(names.begin(), names.end());
    for (const auto& n : names) {
      int fd = ::open((ckpt_dir + "/" + n).c_str(), O_RDONLY);
      if (fd < 0) continue;
      unsigned char lb[8];
      if (::pread(fd, lb, 8, 0) == 8) {
        uint64_t hn = 0;
        for (int i = 0; i < 8; ++i) hn |= uint64_t(lb[i]) << (8 * i);
        std::string hdr(hn, '\0');
        if (::pread(fd, hdr.data(), hn, 8) == (ssize_t)hn &&
            hdr.find("\"" + key + "\"") != std::string::npos) {
          shard = n;
          ::close(fd);
          break;
        }
      }
      ::close(fd);
    }
  }
  if (shard.empty()) throw std::runtime_error("tensor not found: " + key);
  int fd = ::open((ckpt_dir + "/" + shard).c_str(), O_RDONLY);
  unsigned char lb[8];
  ::pread(fd, lb, 8, 0);
  uint64_t hn = 0;
  for (int i = 0; i < 8; ++i) hn |= uint64_t(lb[i]) << (8 * i);
  std::string hdr(hn, '\0');
  ::pread(fd, hdr.data(), hn, 8);
  int64_t data_start = 8 + hn;
  size_t p = hdr.find("\"" + key + "\"");
  size_t obj = hdr.find('{', p);
  size_t close = hdr.find('}', obj);
  std::string body = hdr.substr(obj, close - obj + 1);
  // locate "data_offsets":[a,b]
  size_t dp = body.find("data_offsets");
  size_t lbk = body.find('[', dp);
  size_t comma = body.find(',', lbk);
  size_t rb = body.find(']', comma);
  int64_t off0 = std::strtoll(body.substr(lbk + 1, comma - lbk - 1).c_str(), nullptr, 10);
  (void)rb;
  size_t row_elems = static_cast<size_t>(out_dim) * ng;
  int64_t byte_off = data_start + off0 + static_cast<int64_t>(expert) *
                     static_cast<int64_t>(row_elems * 2);
  std::vector<uint16_t> bits(row_elems);
  if (::pread(fd, bits.data(), row_elems * 2, byte_off) !=
      (ssize_t)(row_elems * 2))
    throw std::runtime_error("short row read: " + key);
  ::close(fd);
  // void* constructor reinterprets the uint16 bytes as bf16 (the iterator
  // constructor would CONVERT the integers numerically instead).
  mx::array cpp_row(static_cast<void*>(bits.data()), mx::Shape{out_dim, ng},
                    mx::bfloat16, [](void*) {});
  char nm[160];
  std::snprintf(nm, sizeof(nm), "raw bf16 bytes exp=%d %s", expert, key.c_str());
  cmp(nm, cpp_row, py_row);
}

// gather_qmm with the sorted_indices flag, used to test the kernel difference.
mx::array gather_q(const mx::array& x, const Q4Projection& p,
                   const mx::array& route, int gs, int bits, bool sorted) {
  mx::array rhs = mx::astype(route, mx::int32);
  return mx::gather_qmm(x, p.weight, p.scales, p.biases, std::nullopt, rhs,
                        true, gs, bits, "affine", sorted);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) {
    std::fprintf(stderr, "usage: %s <dump-dir> [checkpoint-dir]\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  auto cfg = read_config(dir);

  const int H = 2560, E = 512, K = 10;
  // Derive T from the m2d dump size (dumped as little-endian f32).
  const int T = static_cast<int>(file_size(join(dir, "m2d.bin")) / (4ULL * H));
  const int inter = 640, gs = std::stoi(cfg.at("group_size"));
  const int bits = std::stoi(cfg.at("bits"));
  // Number of stacked experts = file bytes / (weight itemsize * out * packed_in).
  const int ne = static_cast<int>(file_size(join(dir, "w_gate_proj_weight.bin")) /
                                  (4ULL * inter * 320ULL));
  const int down_packed = 80;
  const int sgate_packed = 320;
  std::printf("[cfg] T=%d H=%d E=%d K=%d inter=%d gs=%d bits=%d n_experts_used=%d "
              "python_sorted=%s\n", T, H, E, K, inter, gs, bits, ne,
              cfg.at("sorted_gate_proj").c_str());

  // ---- Python reference ----------------------------------------------------
  auto py_m2d = load(dir, "m2d", {1, T, H}, mx::bfloat16);
  auto py_gate_logits = load(dir, "gate_logits", {1, T, E}, mx::bfloat16);
  auto py_gate_w = load(dir, "gate_w", {E, H}, mx::bfloat16);
  // MLX router indices are uint32; the dump writes their raw uint32 bytes.
  auto py_inds = mx::astype(load_u32(dir, "inds", {1, T, K}), mx::int32);
  // Python's softmax( bf16 logits, precise=True ) returns bf16 scores. The
  // dump stores exact bf16->f32 values; reconstruct as bf16 to preserve the
  // bf16*bf16 -> bf16 rounding in the weighted combine.
  auto py_scores = load(dir, "scores", {1, T, K}, mx::bfloat16);
  auto py_local = load(dir, "local", {T, K}, mx::int32);
  auto py_wanted = load(dir, "wanted", {ne}, mx::int32);
  auto py_routed_y = load(dir, "routed_y", {1, T, K, H}, mx::bfloat16);
  auto py_combined = load(dir, "combined", {1, T, H}, mx::bfloat16);
  auto py_sgl = load(dir, "sgl", {1, T, 1}, mx::bfloat16);
  auto py_sg = load(dir, "sg", {1, T, 1}, mx::bfloat16);
  auto py_se_g = load(dir, "se_g", {1, T, inter}, mx::bfloat16);
  auto py_se_u = load(dir, "se_u", {1, T, inter}, mx::bfloat16);
  auto py_se_swiglu = load(dir, "se_swiglu", {1, T, inter}, mx::bfloat16);
  auto py_se_s = load(dir, "se_s", {1, T, H}, mx::bfloat16);
  auto py_sg_s = load(dir, "sg_s", {1, T, H}, mx::bfloat16);
  auto py_moe2d = load(dir, "moe2d", {1, T, H}, mx::bfloat16);

  auto wgw = load_u32(dir, "w_gate_proj_weight", {ne, inter, 320});
  auto wgs = load(dir, "w_gate_proj_scales", {ne, inter, 80}, mx::bfloat16);
  auto wgb = load(dir, "w_gate_proj_biases", {ne, inter, 80}, mx::bfloat16);
  auto wuw = load_u32(dir, "w_up_proj_weight", {ne, inter, 320});
  auto wus = load(dir, "w_up_proj_scales", {ne, inter, 80}, mx::bfloat16);
  auto wub = load(dir, "w_up_proj_biases", {ne, inter, 80}, mx::bfloat16);
  auto wdw = load_u32(dir, "w_down_proj_weight", {ne, H, down_packed});
  auto wds = load(dir, "w_down_proj_scales", {ne, H, 20}, mx::bfloat16);
  auto wdb = load(dir, "w_down_proj_biases", {ne, H, 20}, mx::bfloat16);

  // ---- 1. Router -----------------------------------------------------------
  std::printf("\n=== 1. Router (gate_logits from identical m2d + gate_w) ===\n");
  {
    mx::array x2 = mx::reshape(py_m2d, mx::Shape{T, H});
    mx::array logits = linear(x2, Linear(py_gate_w));  // [T,E]
    cmp("gate_logits C++ vs Python", logits, py_gate_logits);
    RouterOutput r = router(logits, K, 0.0f);
    cmp_int("router inds C++ vs Python", r.indices, py_inds);
    cmp("router scores C++ vs Python", r.scores, py_scores);
    (void)r;
  }

  // ---- 2. Routed-expert gather_qmm compute -------------------------------
  std::printf("\n=== 2. Routed experts (identical x/indices/scores/weights) ===\n");
  ExpertWeights w(Q4Projection(wgw, wgs, wgb), Q4Projection(wuw, wus, wub),
                  Q4Projection(wdw, wds, wdb), gs, bits);
  {
    mx::array x2 = mx::reshape(py_m2d, mx::Shape{T, H});
    mx::array s2 = mx::reshape(py_scores, mx::Shape{T, K});
    mx::array out = moe_block(x2, py_local, s2, w);  // [T,H]
    cmp("moe_block (sorted=false) vs routed combine",
        mx::reshape(out, mx::Shape{1, T, H}), py_combined);
    // Also test with the Python sorted_indices flag on the same local indices
    // (unsorted) to see whether the flag itself changes MLX kernel numerics.
    mx::array x4 = mx::expand_dims(x2, std::vector<int>{-2, -3});
    mx::array g = gather_q(x4, w.gate, py_local, gs, bits, true);
    mx::array u = gather_q(x4, w.up, py_local, gs, bits, true);
    mx::array a = swiglu(g, u);
    mx::array d = mx::squeeze(gather_q(a, w.down, py_local, gs, bits, true), -2);
    mx::array comb = weighted_combine(d, s2);
    cmp("moe_block (sorted=true, invalid unsorted idx) vs routed combine",
        mx::reshape(comb, mx::Shape{1, T, H}), py_combined);
    // Proper sorted path: _gather_sort + gather_qmm(sorted_indices=True) +
    // _scatter_unsort, matching Python SwitchGLU.__call__ at S>=64.
    mx::array comb_sorted = moe_block_sorted(x2, py_local, s2, w);
    cmp("moe_block_sorted (proper) vs Python combined",
        mx::reshape(comb_sorted, mx::Shape{1, T, H}), py_combined);
    cmp("moe_block (unsorted) vs Python ref_comb",
        mx::reshape(out, mx::Shape{1, T, H}), load(dir, "ref_comb", {1, T, H}, mx::bfloat16));

    // Per-stage comparison against a Python reference replay with IDENTICAL
    // x / local / scores / weights and NO gather_sort, sorted_indices=False.
    std::printf("  -- primitive-by-primitive (Python unsorted reference) --\n");
    auto ref_g = load(dir, "ref_g", {1, T, K, 1, inter}, mx::bfloat16);
    auto ref_u = load(dir, "ref_u", {1, T, K, 1, inter}, mx::bfloat16);
    auto ref_a = load(dir, "ref_a", {1, T, K, 1, inter}, mx::bfloat16);
    auto ref_d = load(dir, "ref_d", {1, T, K, H}, mx::bfloat16);
    auto ref_comb = load(dir, "ref_comb", {1, T, H}, mx::bfloat16);
    mx::array cg = q4_projection(x4, w.gate, py_local, gs, bits, false);
    mx::array cu = q4_projection(x4, w.up, py_local, gs, bits, false);
    cmp("gate gather_qmm C++ vs Python", cg, ref_g);
    cmp("up   gather_qmm C++ vs Python", cu, ref_u);
    mx::array ca = swiglu(cg, cu);
    cmp("swiglu C++ vs Python", ca, ref_a);
    mx::array cd = mx::squeeze(q4_projection(ca, w.down, py_local, gs, bits, false), -2);
    cmp("down gather_qmm C++ vs Python", cd, ref_d);
    mx::array cweighted = cd * mx::expand_dims(s2, -1);
    auto ref_weighted = load(dir, "ref_weighted", {1, T, K, H}, mx::float32);
    cmp("weighted product C++ vs Python", cweighted, ref_weighted);
    mx::array ccomb = weighted_combine(cd, s2);
    cmp("weighted_combine C++ vs Python", mx::reshape(ccomb, mx::Shape{1, T, H}), ref_comb);
    // Try summing axis=-2 vs axis=1 explicitly, with precise on/off.
    for (int ax : {-2, 1}) {
      mx::array alt = mx::sum(cweighted, ax, false);
      char nm[128];
      std::snprintf(nm, sizeof(nm), "sum(axis=%d) C++ vs Python", ax);
      cmp(nm, mx::reshape(alt, mx::Shape{1, T, H}), ref_comb);
    }
  }

  // ---- 3. Shared expert ---------------------------------------------------
  std::printf("\n=== 3. Shared expert (identical weights) ===\n");
  {
    auto q = [&](const std::string& tag, int out, int in) -> Linear {
      int packed_in = in * bits / 32;
      int n_groups = in / gs;
      return Linear::quantized(
          load_u32(dir, "w_shared_" + tag + "_weight", {out, packed_in}),
          load(dir, "w_shared_" + tag + "_scales", {out, n_groups}, mx::bfloat16),
          load(dir, "w_shared_" + tag + "_biases", {out, n_groups}, mx::bfloat16),
          gs, bits);
    };
    Linear sgate = q("sgate", 1, H);     // shared_expert_gate: [1,H] -> [1]
    Linear se_gw = q("gate", inter, H);  // shared expert gate_proj
    Linear se_uw = q("up", inter, H);    // shared expert up_proj
    Linear se_dw = q("down", H, inter);  // shared expert down_proj

    mx::array x2 = mx::reshape(py_m2d, mx::Shape{T, H});
    mx::array sgl = mx::reshape(linear(x2, sgate), mx::Shape{1, T, 1});
    cmp("sgl C++ vs Python", sgl, py_sgl);
    mx::array sg = mx::sigmoid(sgl);
    cmp("sg C++ vs Python", sg, py_sg);

    mx::array gse = mx::reshape(linear(x2, se_gw), mx::Shape{1, T, inter});
    cmp("se_g C++ vs Python", gse, py_se_g);
    mx::array use = mx::reshape(linear(x2, se_uw), mx::Shape{1, T, inter});
    cmp("se_u C++ vs Python", use, py_se_u);
    mx::array sw = swiglu(mx::reshape(gse, mx::Shape{T, inter}),
                          mx::reshape(use, mx::Shape{T, inter}));
    cmp("se_swiglu C++ vs Python", mx::reshape(sw, mx::Shape{1, T, inter}),
        py_se_swiglu);
    mx::array sse = mx::reshape(linear(sw, se_dw), mx::Shape{1, T, H});
    cmp("se_s C++ vs Python", sse, py_se_s);
    mx::array sgs = sg * sse;
    cmp("sg*s C++ vs Python", sgs, py_sg_s);

    std::printf("\n=== 4. Combine ===\n");
    mx::array moe2d = py_combined + sgs;
    cmp("moe2d C++ vs Python", moe2d, py_moe2d);
  }

  // ---- 5. Hyper-connection input path (upstream of MoE) ------------------
  std::printf("\n=== 5. Hyper-connection / input path ===\n");
  {
    auto qh = [&](const std::string& tag, const std::string& part,
                  int out, int in) -> Linear {
      int packed = in * bits / 32;
      int ng = in / gs;
      std::string base = "hc_" + tag + "_" + part + "_";
      return Linear::quantized(
          load_u32(dir, base + "weight", {out, packed}),
          load(dir, base + "scales", {out, ng}, mx::bfloat16),
          load(dir, base + "biases", {out, ng}, mx::bfloat16), gs, bits);
    };
    auto py_gr0_mixed = load(dir, "gr0_mixed", {1, T, H}, mx::bfloat16);
    auto py_gr0_inj = load(dir, "gr0_inj", {1, T, 4}, mx::bfloat16);
    auto py_gr1_input = load(dir, "gr1_input", {1, T, 4 * H}, mx::bfloat16);
    auto py_gr1_mixed = load(dir, "gr1_mixed", {1, T, H}, mx::bfloat16);
    auto py_gr1_inj = load(dir, "gr1_inj", {1, T, 4}, mx::bfloat16);
    // Optionally use the f32-conv-cache GDN branch dumped by
    // probe_gdn_cache_dtype.cpp — this is the exact branch the end-to-end tool
    // produces, so the replay should reproduce its L0 hidden exactly.
    auto py_gdn_out = load(dir, "gdn_out", {1, T, H}, mx::bfloat16);
    {
      std::ifstream probe(join(dir, "gdn_out_f32cache.bin"), std::ios::binary);
      if (probe.good()) {
        py_gdn_out = load(dir, "gdn_out_f32cache", {1, T, H}, mx::bfloat16);
        std::printf("[replay] using f32-cache GDN branch override\n");
      }
    }
    auto hidden = load(dir, "layer_input", {1, T, 4 * H}, mx::bfloat16);

    Linear a_down = qh("attn", "input_mix_weight_down", 320, 4 * H);
    Linear a_up = qh("attn", "input_mix_weight_up", 4 * H, 320);
    Linear a_inj = qh("attn", "block_inject_weight", 4, 4 * H);
    auto a_norm = load(dir, "hc_attn_norm_w", {4 * H}, mx::bfloat16);
    GatedResidualOutput ar = gated_residual(hidden, a_norm, a_down, a_up, a_inj,
                                            4, H, 1e-6f);
    // Per-stage replica of gated_residual to find the ULP-level divergence.
    auto stages = [&](const mx::array& h, const mx::array& nw,
                      const Linear& dn, const Linear& up, const Linear& ij,
                      mx::array& normed, mx::array& low, mx::array& mix,
                      mx::array& prod) {
      mx::Shape gs{h.shape(0), h.shape(1), 4, H};
      mx::array w2d = mx::reshape(nw, mx::Shape{4, H});
      mx::array yg = mx::astype(mx::reshape(h, gs), mx::float32);
      yg = yg * mx::rsqrt(mx::mean(mx::square(yg), -1, true) + 1e-6f);
      yg = yg * mx::astype(w2d, mx::float32);
      normed = mx::astype(mx::reshape(yg, h.shape()), h.dtype());
      mx::array hc_arr = mx::array(4);
      low = linear(normed, dn) / hc_arr;
      mx::array sl = low * mx::sigmoid(low);
      mix = mx::sigmoid(linear(sl, up));
      prod = mx::reshape(mix, gs) * mx::reshape(normed, gs);
    };
    mx::array n0 = hidden, l0 = hidden, m0 = hidden, p0 = hidden;
    stages(hidden, a_norm, a_down, a_up, a_inj, n0, l0, m0, p0);
    cmp("attn normed C++ vs Python", n0,
        load(dir, "gr0_normed", {1, T, 4 * H}, mx::bfloat16));
    cmp("attn silu(low) C++ vs Python", l0 * mx::sigmoid(l0),
        load(dir, "gr0_low", {1, T, 320}, mx::bfloat16));
    // Split: is the 1-ULP gap in down/qmm, the /hc divide, or the SiLU fusion?
    auto silu_split = [&](const char* tag, const mx::array& normed_cpp,
                          const Linear& dn, const mx::array& low_cpp, int gi) {
      std::string g = "gr" + std::to_string(gi) + "_";
      if (!exists(join(dir, g + "lowpre.bin"))) {
        std::printf("  [%s] gr*_lowpre.bin missing: re-dump with the updated "
                    "dump_python_moe_intermediates.py\n", tag);
        return;
      }
      auto py_down = load(dir, g + "down", {1, T, 320}, mx::bfloat16);
      auto py_pre = load(dir, g + "lowpre", {1, T, 320}, mx::bfloat16);
      auto py_low = load(dir, g + "low", {1, T, 320}, mx::bfloat16);
      std::string t(tag);
      cmp((t + " down qmm  C++ vs Python").c_str(), linear(normed_cpp, dn), py_down);
      cmp((t + " low/hc    C++ vs Python").c_str(), low_cpp, py_pre);
      cmp((t + " silu UNFUSED(py input)").c_str(),
          py_pre * mx::sigmoid(py_pre), py_low);
      cmp((t + " silu COMPILED(py input)").c_str(), silu_compiled(py_pre), py_low);
      cmp((t + " silu COMPILED(C++ input)").c_str(), silu_compiled(low_cpp), py_low);
    };
    silu_split("attn", n0, a_down, l0, 0);
    cmp("attn mix C++ vs Python", m0,
        load(dir, "gr0_mix", {1, T, 4 * H}, mx::bfloat16));
    cmp("attn prod C++ vs Python", p0,
        load(dir, "gr0_prod", {1, T, 4, H}, mx::bfloat16));
    cmp("attn mixed C++ vs Python", ar.mixed_input, py_gr0_mixed);
    cmp("attn inj C++ vs Python", ar.injection_weights, py_gr0_inj);

    auto inject = [](const mx::array& h, const mx::array& b,
                     const mx::array& inj) {
      mx::array b4 = mx::expand_dims(b, std::vector<int>{-2});
      mx::array i4 = mx::expand_dims(inj, std::vector<int>{-1});
      return h + mx::reshape(b4 * i4, h.shape());
    };
    mx::array hid1 = inject(hidden, py_gdn_out, ar.injection_weights);
    cmp("hidden after attn inject", hid1, py_gr1_input);

    Linear m_down = qh("mlp", "input_mix_weight_down", 320, 4 * H);
    Linear m_up = qh("mlp", "input_mix_weight_up", 4 * H, 320);
    Linear m_inj = qh("mlp", "block_inject_weight", 4, 4 * H);
    auto m_norm = load(dir, "hc_mlp_norm_w", {4 * H}, mx::bfloat16);
    GatedResidualOutput mr = gated_residual(hid1, m_norm, m_down, m_up, m_inj,
                                            4, H, 1e-6f);
    mx::array n1 = hid1, l1 = hid1, m1 = hid1, p1 = hid1;
    stages(hid1, m_norm, m_down, m_up, m_inj, n1, l1, m1, p1);
    cmp("mlp normed C++ vs Python", n1,
        load(dir, "gr1_normed", {1, T, 4 * H}, mx::bfloat16));
    cmp("mlp silu(low) C++ vs Python", l1 * mx::sigmoid(l1),
        load(dir, "gr1_low", {1, T, 320}, mx::bfloat16));
    silu_split("mlp", n1, m_down, l1, 1);
    cmp("mlp mix C++ vs Python", m1,
        load(dir, "gr1_mix", {1, T, 4 * H}, mx::bfloat16));
    cmp("mlp prod C++ vs Python", p1,
        load(dir, "gr1_prod", {1, T, 4, H}, mx::bfloat16));
    cmp("mlp mixed C++ vs Python", mr.mixed_input, py_gr1_mixed);
    cmp("mlp inj C++ vs Python", mr.injection_weights, py_gr1_inj);
    cmp("mlp mixed vs MoE m2d (input)", mr.mixed_input, py_m2d);

    // Full layer-0 replay: C++ MoE on the C++ mlp mixed input, then inject.
    mx::array m2d_cpp = mx::reshape(mr.mixed_input, mx::Shape{T, H});
    mx::array s2b = mx::reshape(py_scores, mx::Shape{T, K});
    mx::array moe_cpp = mx::reshape(
        moe_block(m2d_cpp, py_local, s2b, w), mx::Shape{1, T, H});
    mx::array moe_total = moe_cpp + py_sg_s;  // routed + shared (sg*s, bit-exact)
    cmp("MoE branch from C++ m2d", moe_total, py_moe2d);
    mx::array final = inject(hid1, moe_total, mr.injection_weights);
    auto py_layer_out = load(dir, "layer_output", {1, T, 4 * H}, mx::bfloat16);
    cmp("FULL L0 hidden C++ vs Python", final, py_layer_out);
    dump_f32(dir, "cpp_l0_hidden_all", final);
  }

  // ---- 6. scale/bias byte identity: raw safetensors vs Python store -------
  if (argc == 3) {
    std::printf("\n=== 6. scales/biases byte identity (raw safetensors vs Python) ===\n");
    mx::eval(py_wanted);
    int e0 = static_cast<int>(py_wanted.data<int32_t>()[0]);
    auto gscales_py = load(dir, "w_gate_proj_scales", {ne, inter, 80}, mx::bfloat16);
    auto gbias_py = load(dir, "w_gate_proj_biases", {ne, inter, 80}, mx::bfloat16);
    mx::array gscales_row = mx::reshape(
        mx::slice(gscales_py, mx::Shape{0, 0, 0}, mx::Shape{1, inter, 80}),
        mx::Shape{inter, 80});
    mx::array gbias_row = mx::reshape(
        mx::slice(gbias_py, mx::Shape{0, 0, 0}, mx::Shape{1, inter, 80}),
        mx::Shape{inter, 80});
    try {
      check_raw_bytes(argv[2],
          "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales",
          e0, gscales_row, inter, 80);
      check_raw_bytes(argv[2],
          "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases",
          e0, gbias_row, inter, 80);
    } catch (const std::exception& ex) {
      std::printf("  raw-byte check skipped: %s\n", ex.what());
    }
  }

  std::printf("\nDONE.\n");
  return 0;
}
