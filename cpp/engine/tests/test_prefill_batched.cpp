// Bit-exact parity of run_prefill_batched against the synthetic Python oracle
// (tools/gen_prefill_batched_vectors.py).  The oracle feeds all S tokens through
// each layer in ONE call -- the production `language(ids, cache)` prefill path.
//
// Routed experts are STUBBED TO ZERO in both Python and C++; the router and the
// non-zero dense shared expert are real.  PLE is absent.  See the generator's
// docstring and the subagent report for exactly what is / is not proven.
#include "prefill_batched.hpp"

#include <mlx/mlx.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

namespace {

struct Diff {
  bool equal = true;
  size_t first = 0;
  uint32_t a = 0, b = 0;
  size_t n = 0;
};

Diff diff_bits(const mx::array& a, const mx::array& b) {
  Diff d;
  if (a.shape() != b.shape()) {
    d.equal = false;
    return d;
  }
  mx::array av = mx::astype(mx::reshape(a, {-1}), mx::uint32);
  mx::array bv = mx::astype(mx::reshape(b, {-1}), mx::uint32);
  mx::eval(av, bv);
  d.n = av.size();
  const uint32_t* pa = av.data<uint32_t>();
  const uint32_t* pb = bv.data<uint32_t>();
  for (size_t i = 0; i < d.n; ++i) {
    if (pa[i] != pb[i]) {
      d.equal = false;
      d.first = i;
      d.a = pa[i];
      d.b = pb[i];
      break;
    }
  }
  return d;
}

void report(const char* name, const Diff& d) {
  std::printf("  %-22s %s (n=%zu", name, d.equal ? "OK  " : "FAIL", d.n);
  if (!d.equal) std::printf(" first=%zu 0x%08x vs 0x%08x", d.first, d.a, d.b);
  std::printf(")\n");
}

mx::array as_bf16(mx::array a) {
  if (static_cast<mx::Dtype::Val>(a.dtype()) == mx::uint16)
    return mx::view(a, mx::bfloat16);
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "cpp/engine/tests/vectors_prefill_batched";
  auto load = [&](const std::string& n) { return mx::load(dir + "/" + n + ".npy"); };
  auto ls = [&](int l, const std::string& n) {
    return load("L" + std::to_string(l) + "_" + n);
  };

  // ---- geometry (must match meta.json) ------------------------------------
  const int H = 32, HC = 2, LOW = 16, NL = 2;
  const int Hk = 1, Hv = 2, Dk = 32, Dv = 32, K = 4;
  const int KEY = Hk * Dk, VALUE = Hv * Dv, CONV = 2 * KEY + VALUE;
  const float EPS = 1e-6f;

  DecoderConfig cfg;
  cfg.hidden_size = H;
  cfg.hc_count = HC;
  cfg.hc_lowrank = LOW;
  cfg.n_layers = NL;
  cfg.vocab = 64;
  cfg.eps = EPS;
  cfg.gdn = GDNConfig{H, KEY, VALUE, Hk, Hv, Dk, Dv, K, EPS,
                      GDNActivation::Sigmoid};

  FullAttentionConfig fa;
  fa.n_heads = 2;
  fa.kv_heads = 1;
  fa.head_dim = 16;
  fa.n_idx_heads = 2;
  fa.n_idx_kv = 1;
  fa.idx_head_dim = 16;
  fa.compress_ratio = 2;
  fa.token_budget = 64;
  fa.eps = EPS;
  fa.rope.dim = 16;
  fa.rope.base = 10000.0f;
  fa.rope.style = "interleaved";
  fa.rope.attention_scaling = 1.0f;
  fa.rope.mrope_section = {6, 6, 4};

  // ---- inputs / expected ---------------------------------------------------
  mx::array tokens = load("tokens");
  const int S = static_cast<int>(tokens.shape(1));
  mx::array expected_logits = load("expected_logits");
  std::vector<mx::array> expected_hidden;
  expected_hidden.reserve(NL);
  for (int l = 0; l < NL; ++l) expected_hidden.push_back(ls(l, "expected_hidden"));

  // ---- embedding: gather all S rows, dequantization is elementwise ---------
  mx::array embed_w = load("embed_w");
  auto embed = [&](const mx::array& ids) {
    mx::array flat = mx::reshape(ids, mx::Shape{-1});
    mx::array e = mx::take(embed_w, flat, 0);
    return mx::reshape(e, mx::Shape{1, S, H});
  };

  // ---- per-layer weights + caches ------------------------------------------
  std::vector<BatchedLayer> layers;
  layers.reserve(NL);
  std::vector<DecoderFullLayerCache> caches;
  caches.reserve(NL);
  for (int l = 0; l < NL; ++l) {
    layers.emplace_back(
        ls(l, "attn_norm_w"), Linear(ls(l, "attn_down_w")),
        Linear(ls(l, "attn_up_w")), Linear(ls(l, "attn_inject_w")),
        ls(l, "mlp_norm_w"), Linear(ls(l, "mlp_down_w")),
        Linear(ls(l, "mlp_up_w")), Linear(ls(l, "mlp_inject_w")),
        Linear(ls(l, "gate_w")), Linear(ls(l, "shared_gw")),
        Linear(ls(l, "shared_uw")), Linear(ls(l, "shared_dw")),
        Linear(ls(l, "shared_glw")), 2,
        (l == 0) ? AttnKind::Linear : AttnKind::FullAttention);
    BatchedLayer& L = layers.back();

    // Routed MoE stub: exactly the same zero output as Python's StubSwitchGLU.
    L.moe_forward = [](const mx::array& x, const mx::array&, const mx::array&) {
      return mx::zeros(mx::Shape{x.shape(0), x.shape(1)}, mx::float32);
    };

    if (L.kind == AttnKind::Linear) {
      L.gdn.emplace(Linear(ls(l, "gdn_qkv")), Linear(ls(l, "gdn_z")),
                    Linear(ls(l, "gdn_b")), Linear(ls(l, "gdn_a")),
                    as_bf16(ls(l, "gdn_conv")), ls(l, "gdn_A_log"),
                    ls(l, "gdn_dt_bias"), ls(l, "gdn_norm"),
                    Linear(ls(l, "gdn_out")));
    } else {
      L.full.emplace(Linear(ls(l, "fa_q_w")), Linear(ls(l, "fa_k_w")),
                     Linear(ls(l, "fa_v_w")), Linear(ls(l, "fa_o_w")),
                     ls(l, "fa_q_norm_w"), ls(l, "fa_k_norm_w"),
                     Linear(ls(l, "fa_index_qk_w")),
                     ls(l, "fa_index_q_norm_w"), ls(l, "fa_index_k_norm_w"));
    }

    mx::array conv = mx::zeros(mx::Shape{1, K - 1, CONV}, mx::float32);
    mx::array rec = mx::zeros(mx::Shape{1, Hv, Dv, Dk}, mx::float32);
    caches.emplace_back(DecoderLayerCache(GDNCacheState(conv, rec)));
  }

  // ---- head ----------------------------------------------------------------
  BatchedHead head(load("head_norm_w"), Linear(load("head_down_w")),
                   Linear(load("head_up_w")),
                   BatchedLMTriplet(load("head_lm_w"),
                                    as_bf16(load("head_lm_s")),
                                    as_bf16(load("head_lm_b"))));

  // ---- run -----------------------------------------------------------------
  BatchedPrefillDebug dbg;
  mx::array logits =
      run_prefill_batched(tokens, embed, layers, head, cfg, fa, caches,
                          32, 4, &dbg);
  mx::eval(logits);

  std::printf("test_prefill_batched (%d layers, S=%d, MoE routed=stub-zero):\n",
              NL, S);
  bool ok = true;
  for (int l = 0; l < NL; ++l) {
    char nm[64];
    std::snprintf(nm, sizeof(nm), "L%d hidden", l);
    Diff d = diff_bits(dbg.hidden_after_layer[l], expected_hidden[l]);
    report(nm, d);
    ok = ok && d.equal;
  }
  // The final head's mixed input is not dumped in C++; compare via logits only.
  Diff dl = diff_bits(mx::reshape(mx::astype(logits, mx::float32), mx::Shape{-1}),
                      expected_logits);
  report("logits", dl);
  ok = ok && dl.equal;

  if (!ok) {
    std::fprintf(stderr,
                 "FAIL: see per-layer diffs above (first divergent stage "
                 "localizes the mismatch).\n");
    return 1;
  }
  std::printf("test_prefill_batched OK\n");
  return 0;
}
