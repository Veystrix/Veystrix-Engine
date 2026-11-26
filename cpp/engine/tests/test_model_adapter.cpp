// verify_against_python test for the model-agnostic adapter/config layer.
//
// The flat config files in the directory given as argv[1] are produced from the
// REAL checkpoint config.json files by tools/gen_model_adapter_vectors.py.
// This test reads them, rebuilds the ModelSpec via from_qwen_config /
// from_nemotron_config, and asserts every field matches the Python side:
//   * integer/float fields are compared bit-exactly (bit_cast<uint32_t>)
//   * name-format strings are compared exactly
//   * derived per-layer fields (kind mapping + MoE mask) are checked against
//     the Python `expected_*` ground truth
//
// Standalone build (no CMake edits required; model_adapter has no MLX calls):
//   cd /Users/vincent/Documents/Veystrix && \
//   MLXROOT=.venv/lib/python3.12/site-packages/mlx && \
//   xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
//     cpp/engine/src/model_adapter.cpp cpp/engine/tests/test_model_adapter.cpp \
//     -o /tmp/test_model_adapter && \
//   python cpp/engine/tools/gen_model_adapter_vectors.py /tmp/model_adapter_flat && \
//   /tmp/test_model_adapter /tmp/model_adapter_flat
#include "model_adapter.hpp"

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace veystrix::engine;

namespace {

using KV = std::map<std::string, std::string>;

std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r')) ++a;
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) --b;
  return s.substr(a, b - a);
}

KV read_flat(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open flat config: " + path);
  KV kv;
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') continue;
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    kv[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
  }
  return kv;
}

const std::string& need(const KV& kv, const std::string& k) {
  auto it = kv.find(k);
  if (it == kv.end()) throw std::runtime_error("missing key in flat config: " + k);
  return it->second;
}

int get_int(const KV& kv, const std::string& k) {
  return std::stoi(need(kv, k));
}
float get_float(const KV& kv, const std::string& k) {
  return std::stof(need(kv, k));
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  if (s.empty()) return out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(item);
  return out;
}

LayerKind kind_from_string(const std::string& s) {
  if (s == "LINEAR_ATTN_GDN") return LayerKind::LINEAR_ATTN_GDN;
  if (s == "FULL_ATTN_QSA") return LayerKind::FULL_ATTN_QSA;
  if (s == "MAMBA_SSM") return LayerKind::MAMBA_SSM;
  if (s == "MOE") return LayerKind::MOE;
  throw std::runtime_error("unknown LayerKind string: " + s);
}

Activation activation_from_string(const std::string& s) {
  if (s == "SWIGLU") return Activation::SWIGLU;
  if (s == "RELU2") return Activation::RELU2;
  throw std::runtime_error("unknown Activation string: " + s);
}

// ---- check helpers --------------------------------------------------------

int g_checks = 0;

void fail(const std::string& what, const std::string& got, const std::string& want) {
  std::cerr << "FAIL: " << what << "  got='" << got << "' want='" << want << "'\n";
}

void expect_i(int got, int want, const std::string& what) {
  ++g_checks;
  if (got != want)
    fail(what, std::to_string(got), std::to_string(want));
}

void expect_f(float got, float want, const std::string& what) {
  ++g_checks;
  if (std::bit_cast<uint32_t>(got) != std::bit_cast<uint32_t>(want))
    fail(what, std::to_string(got), std::to_string(want));
}

void expect_s(const std::string& got, const std::string& want, const std::string& what) {
  ++g_checks;
  if (got != want) fail(what, got, want);
}

void expect_kinds(const std::vector<LayerKind>& got,
                  const std::vector<LayerKind>& want,
                  const std::string& what) {
  ++g_checks;
  if (got != want) fail(what, "vector mismatch", "vector match");
}

void expect_moe(const std::vector<bool>& got, const std::vector<bool>& want,
                const std::string& what) {
  ++g_checks;
  if (got != want) fail(what, "vector mismatch", "vector match");
}

// ---- common spec assertions ----------------------------------------------

void check_common(const ModelSpec& s, const KV& kv, ModelFamily family) {
  expect_s(to_string(s.family), need(kv, "model_family"), "model_family");
  expect_i(static_cast<int>(s.family), static_cast<int>(family), "family enum");
  expect_i(s.hidden_size, get_int(kv, "hidden_size"), "hidden_size");
  expect_i(s.n_layers, get_int(kv, "n_layers"), "n_layers");
  expect_f(s.norm_eps, get_float(kv, "norm_eps"), "norm_eps");

  auto want_kinds_s = split(need(kv, "expected_layer_kinds"));
  std::vector<LayerKind> want_kinds;
  for (auto& k : want_kinds_s) want_kinds.push_back(kind_from_string(k));
  expect_kinds(s.layer_kinds, want_kinds, "layer_kinds");
  expect_i(static_cast<int>(s.layer_kinds.size()), s.n_layers, "layer_kinds size");

  auto want_moe_s = split(need(kv, "expected_has_moe"));
  std::vector<bool> want_moe;
  for (auto& v : want_moe_s) want_moe.push_back(v == "1");
  expect_moe(s.has_moe, want_moe, "has_moe");
  expect_i(static_cast<int>(s.has_moe.size()), s.n_layers, "has_moe size");

  expect_i(s.expert.n_experts, get_int(kv, "n_experts"), "n_experts");
  expect_i(s.expert.top_k, get_int(kv, "top_k"), "top_k");
  expect_i(s.expert.group_size, get_int(kv, "group_size"), "group_size");
  expect_i(s.expert.moe_intermediate, get_int(kv, "moe_intermediate"), "moe_intermediate");
  expect_s(to_string(s.expert.activation), need(kv, "expert_activation"), "expert_activation");
  expect_s(s.expert.tensor_prefix_format, need(kv, "expert_prefix"), "expert_prefix (exact)");

  expect_i(s.hc.hc_count, get_int(kv, "hc_count"), "hc_count");
  expect_i(s.hc.hc_lowrank, get_int(kv, "hc_lowrank"), "hc_lowrank");
  expect_s(s.hc.output_gate_type, need(kv, "output_gate_type"), "output_gate_type");

  expect_i(s.use_qsa_indexer ? 1 : 0, get_int(kv, "use_qsa_indexer"), "use_qsa_indexer");
}

// ---- Qwen -----------------------------------------------------------------

void check_qsa(const ModelSpec& s, const KV& kv) {
  expect_i(s.qsa.n_heads, get_int(kv, "qsa_n_heads"), "qsa.n_heads");
  expect_i(s.qsa.kv_heads, get_int(kv, "qsa_kv_heads"), "qsa.kv_heads");
  expect_i(s.qsa.head_dim, get_int(kv, "qsa_head_dim"), "qsa.head_dim");
  expect_i(s.qsa.compress_ratio, get_int(kv, "qsa_compress_ratio"), "qsa.compress_ratio");
  expect_i(s.qsa.token_budget, get_int(kv, "qsa_token_budget"), "qsa.token_budget");
  expect_i(s.qsa.block_topk,
           get_int(kv, "qsa_token_budget") / get_int(kv, "qsa_compress_ratio"),
           "qsa.block_topk (derived)");
  expect_f(s.qsa.eps, get_float(kv, "qsa_eps"), "qsa.eps");
}

void check_gdn(const ModelSpec& s, const KV& kv) {
  expect_i(s.gdn.conv_kernel_dim, get_int(kv, "gdn_conv_kernel"), "gdn.conv_kernel_dim");
  expect_i(s.gdn.num_key_heads, get_int(kv, "gdn_n_key_heads"), "gdn.num_key_heads");
  expect_i(s.gdn.num_value_heads, get_int(kv, "gdn_n_value_heads"), "gdn.num_value_heads");
  expect_i(s.gdn.key_head_dim, get_int(kv, "gdn_key_head_dim"), "gdn.key_head_dim");
  expect_i(s.gdn.value_head_dim, get_int(kv, "gdn_value_head_dim"), "gdn.value_head_dim");
  expect_f(s.gdn.eps, get_float(kv, "norm_eps"), "gdn.eps");
}

ModelSpec build_qwen(const KV& kv) {
  return from_qwen_config(
      get_int(kv, "hidden_size"),
      get_int(kv, "n_layers"),
      split(need(kv, "layer_types")),
      get_int(kv, "n_experts"), get_int(kv, "top_k"),
      get_int(kv, "group_size"), get_int(kv, "moe_intermediate"),
      get_float(kv, "norm_eps"),
      get_int(kv, "hc_count"), get_int(kv, "hc_lowrank"),
      need(kv, "output_gate_type"),
      get_int(kv, "qsa_n_heads"), get_int(kv, "qsa_kv_heads"),
      get_int(kv, "qsa_head_dim"), get_int(kv, "qsa_compress_ratio"),
      get_int(kv, "qsa_token_budget"), get_float(kv, "qsa_eps"),
      get_int(kv, "gdn_conv_kernel"), get_int(kv, "gdn_n_key_heads"),
      get_int(kv, "gdn_n_value_heads"), get_int(kv, "gdn_key_head_dim"),
      get_int(kv, "gdn_value_head_dim"));
}

ModelSpec build_nemotron(const KV& kv) {
  return from_nemotron_config(
      get_int(kv, "hidden_size"),
      get_int(kv, "n_layers"),
      split(need(kv, "layer_types")),  // layers_block_type names
      get_int(kv, "n_experts"), get_int(kv, "top_k"),
      get_int(kv, "group_size"), get_int(kv, "moe_intermediate"),
      get_float(kv, "norm_eps"));
}

// Verify dispatch_layer for every layer against the spec / ground truth.
void check_dispatch(const ModelSpec& s, const KV& kv) {
  auto want_kinds_s = split(need(kv, "expected_layer_kinds"));
  for (int i = 0; i < s.n_layers; ++i) {
    LayerDispatch d = dispatch_layer(s, i);
    expect_i(d.layer, i, "dispatch.layer");
    expect_kinds({d.kind}, {kind_from_string(want_kinds_s[i])},
                 "dispatch.kind@" + std::to_string(i));
    expect_f(d.norm_eps, s.norm_eps, "dispatch.norm_eps");
    expect_i(d.use_gdn ? 1 : 0,
             d.kind == LayerKind::LINEAR_ATTN_GDN ? 1 : 0,
             "dispatch.use_gdn@" + std::to_string(i));
    expect_i(d.use_qsa ? 1 : 0,
             (d.kind == LayerKind::FULL_ATTN_QSA && s.use_qsa_indexer) ? 1 : 0,
             "dispatch.use_qsa@" + std::to_string(i));
    expect_i(d.use_mamba ? 1 : 0,
             d.kind == LayerKind::MAMBA_SSM ? 1 : 0,
             "dispatch.use_mamba@" + std::to_string(i));
    expect_i(d.use_moe ? 1 : 0, s.has_moe[i] ? 1 : 0,
             "dispatch.use_moe@" + std::to_string(i));
    if (d.use_qsa) {
      expect_i(d.qsa.n_heads, s.qsa.n_heads, "dispatch.qsa propagated");
    }
    if (d.use_gdn) {
      expect_i(d.gdn.conv_kernel_dim, s.gdn.conv_kernel_dim,
               "dispatch.gdn propagated");
    }
    if (d.use_moe) {
      expect_i(d.expert.n_experts, s.expert.n_experts,
               "dispatch.expert propagated");
      expect_s(d.expert.tensor_prefix_format, s.expert.tensor_prefix_format,
               "dispatch.expert prefix propagated");
    }
  }
}

void verify_qwen(const std::string& dir) {
  KV kv = read_flat(dir + "/qwen_model_adapter.txt");
  ModelSpec s = build_qwen(kv);
  check_common(s, kv, ModelFamily::QWEN4_EXP);
  check_qsa(s, kv);
  check_gdn(s, kv);
  expect_s(to_string(s.expert.activation), "SWIGLU", "Qwen activation is SwiGLU");
  expect_i(s.expert.top_k, 10, "Qwen top_k literal");
  expect_i(s.expert.n_experts, 512, "Qwen n_experts literal");
  check_dispatch(s, kv);
  // describe() must contain the key facts (debug helper sanity).
  std::string desc = describe(s);
  expect_i(desc.find("QWEN4_EXP") != std::string::npos ? 1 : 0, 1, "describe family");
  expect_i(desc.find("language_model.model.layers") != std::string::npos ? 1 : 0,
           1, "describe prefix");
}

void verify_nemotron(const std::string& dir) {
  KV kv = read_flat(dir + "/nemotron_model_adapter.txt");
  ModelSpec s = build_nemotron(kv);
  check_common(s, kv, ModelFamily::NEMOTRON_H);
  expect_s(to_string(s.expert.activation), "RELU2", "Nemotron activation is relu2");
  expect_i(s.expert.top_k, 6, "Nemotron top_k literal");
  expect_i(s.expert.n_experts, 128, "Nemotron n_experts literal");
  expect_i(s.use_qsa_indexer ? 1 : 0, 0, "Nemotron has no sparse QSA indexer");
  check_dispatch(s, kv);
  std::string desc = describe(s);
  expect_i(desc.find("NEMOTRON_H") != std::string::npos ? 1 : 0, 1, "describe family");
  expect_i(desc.find("backbone.layers") != std::string::npos ? 1 : 0,
           1, "describe prefix");
}

void verify_generic() {
  // Future-model extension point: a GENERIC spec with caller-supplied layers.
  ModelSpec s = from_generic_config(
      4096, 3,
      {LayerKind::MAMBA_SSM, LayerKind::FULL_ATTN_QSA, LayerKind::MOE},
      {false, false, true});
  expect_s(to_string(s.family), "GENERIC", "generic family");
  expect_i(s.n_layers, 3, "generic n_layers");
  LayerDispatch d0 = dispatch_layer(s, 0);
  expect_i(d0.use_mamba ? 1 : 0, 1, "generic mamba dispatch");
  LayerDispatch d1 = dispatch_layer(s, 1);
  expect_i(d1.use_qsa ? 1 : 0, 0, "generic dense attention: no QSA by default");
  LayerDispatch d2 = dispatch_layer(s, 2);
  expect_i(d2.use_moe ? 1 : 0, 1, "generic moe dispatch");

  bool threw = false;
  try {
    (void)dispatch_layer(s, 3);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  expect_i(threw ? 1 : 0, 1, "dispatch out_of_range on bad index");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <flat-config-dir>\n";
    return 2;
  }
  std::string dir = argv[1];
  try {
    verify_qwen(dir);
    verify_nemotron(dir);
    verify_generic();
  } catch (const std::exception& e) {
    std::cerr << "EXCEPTION: " << e.what() << "\n";
    return 1;
  }
  if (g_checks == 0) {
    std::cerr << "no checks ran\n";
    return 1;
  }
  std::cout << "test_model_adapter OK (" << g_checks << " checks)\n";
  return 0;
}
