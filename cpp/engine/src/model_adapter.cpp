// Implementation of the model-agnostic adapter/config layer.
// Pure C++ (no MLX calls) — see model_adapter.hpp for the design / format.
#include "model_adapter.hpp"

#include <sstream>
#include <stdexcept>

namespace veystrix::engine {

// ---------------------------------------------------------------------------
// Stringification
// ---------------------------------------------------------------------------

std::string to_string(ModelFamily f) {
  switch (f) {
    case ModelFamily::QWEN4_EXP: return "QWEN4_EXP";
    case ModelFamily::NEMOTRON_H: return "NEMOTRON_H";
    case ModelFamily::GENERIC: return "GENERIC";
  }
  return "UNKNOWN";
}

std::string to_string(LayerKind k) {
  switch (k) {
    case LayerKind::LINEAR_ATTN_GDN: return "LINEAR_ATTN_GDN";
    case LayerKind::FULL_ATTN_QSA: return "FULL_ATTN_QSA";
    case LayerKind::MAMBA_SSM: return "MAMBA_SSM";
    case LayerKind::MOE: return "MOE";
  }
  return "UNKNOWN";
}

std::string to_string(Activation a) {
  switch (a) {
    case Activation::SWIGLU: return "SWIGLU";
    case Activation::RELU2: return "RELU2";
  }
  return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

namespace {

// Qwen layer type -> primitive family.
LayerKind qwen_layer_kind(const std::string& t) {
  if (t == "linear_attention") return LayerKind::LINEAR_ATTN_GDN;
  if (t == "full_attention") return LayerKind::FULL_ATTN_QSA;
  throw std::invalid_argument("unknown Qwen layer type: " + t);
}

// Nemotron block type -> primitive family. Plain "attention" is mapped to the
// full-attention family but with use_qsa_indexer=false (dense attention).
LayerKind nemotron_layer_kind(const std::string& t) {
  if (t == "mamba") return LayerKind::MAMBA_SSM;
  if (t == "attention") return LayerKind::FULL_ATTN_QSA;
  if (t == "moe") return LayerKind::MOE;
  throw std::invalid_argument("unknown Nemotron block type: " + t);
}

}  // namespace

ModelSpec from_qwen_config(
    int hidden_size,
    int n_layers,
    const std::vector<std::string>& layer_types,
    int n_experts, int top_k, int group_size, int moe_intermediate,
    float norm_eps,
    int hc_count, int hc_lowrank, const std::string& output_gate_type,
    int qsa_n_heads, int qsa_kv_heads, int qsa_head_dim,
    int qsa_compress_ratio, int qsa_token_budget, float qsa_eps,
    int gdn_conv_kernel, int gdn_n_key_heads, int gdn_n_value_heads,
    int gdn_key_head_dim, int gdn_value_head_dim) {
  ModelSpec s;
  s.family = ModelFamily::QWEN4_EXP;
  s.hidden_size = hidden_size;
  s.n_layers = n_layers;

  if (static_cast<int>(layer_types.size()) != n_layers)
    throw std::invalid_argument("Qwen layer_types size != n_layers");
  s.layer_kinds.reserve(n_layers);
  for (const auto& t : layer_types) s.layer_kinds.push_back(qwen_layer_kind(t));
  // Every Qwen decoder layer carries a routed-expert MoE MLP.
  s.has_moe.assign(n_layers, true);

  s.expert.n_experts = n_experts;
  s.expert.top_k = top_k;
  s.expert.group_size = group_size;
  s.expert.moe_intermediate = moe_intermediate;
  s.expert.activation = Activation::SWIGLU;
  s.expert.tensor_prefix_format =
      "language_model.model.layers.{L}.mlp.switch_mlp.{P}.{T}";

  s.norm_eps = norm_eps;
  s.hc.hc_count = hc_count;
  s.hc.hc_lowrank = hc_lowrank;
  s.hc.output_gate_type = output_gate_type;

  s.use_qsa_indexer = true;
  s.qsa.n_heads = qsa_n_heads;
  s.qsa.kv_heads = qsa_kv_heads;
  s.qsa.head_dim = qsa_head_dim;
  s.qsa.compress_ratio = qsa_compress_ratio;
  s.qsa.token_budget = qsa_token_budget;
  s.qsa.block_topk = (qsa_compress_ratio > 0)
                         ? qsa_token_budget / qsa_compress_ratio
                         : 0;
  s.qsa.eps = qsa_eps;
  // mrope_section / rope_theta are RoPE-stage concerns (qsa.hpp reserves
  // QSARoPEConfig for them); they are intentionally not consumed here.

  s.gdn.conv_kernel_dim = gdn_conv_kernel;
  s.gdn.num_key_heads = gdn_n_key_heads;
  s.gdn.num_value_heads = gdn_n_value_heads;
  s.gdn.key_head_dim = gdn_key_head_dim;
  s.gdn.value_head_dim = gdn_value_head_dim;
  s.gdn.eps = norm_eps;

  return s;
}

ModelSpec from_nemotron_config(
    int hidden_size,
    int n_layers,
    const std::vector<std::string>& layer_block_types,
    int n_experts, int top_k, int group_size, int moe_intermediate,
    float norm_eps) {
  ModelSpec s;
  s.family = ModelFamily::NEMOTRON_H;
  s.hidden_size = hidden_size;
  s.n_layers = n_layers;

  if (static_cast<int>(layer_block_types.size()) != n_layers)
    throw std::invalid_argument("Nemotron layer_block_types size != n_layers");
  s.layer_kinds.reserve(n_layers);
  s.has_moe.reserve(n_layers);
  for (const auto& t : layer_block_types) {
    s.layer_kinds.push_back(nemotron_layer_kind(t));
    s.has_moe.push_back(t == "moe");
  }

  s.expert.n_experts = n_experts;
  s.expert.top_k = top_k;
  s.expert.group_size = group_size;
  s.expert.moe_intermediate = moe_intermediate;
  s.expert.activation = Activation::RELU2;
  s.expert.tensor_prefix_format =
      "backbone.layers.{L}.mixer.switch_mlp.{P}.{T}";

  s.norm_eps = norm_eps;
  // No hyper-connections on Nemotron; QSA indexer and GDN are unused.
  s.use_qsa_indexer = false;

  return s;
}

ModelSpec from_generic_config(int hidden_size, int n_layers,
                              const std::vector<LayerKind>& layer_kinds,
                              const std::vector<bool>& has_moe) {
  ModelSpec s;
  s.family = ModelFamily::GENERIC;
  s.hidden_size = hidden_size;
  s.n_layers = n_layers;
  s.layer_kinds = layer_kinds;
  s.has_moe = has_moe;
  return s;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

LayerDispatch dispatch_layer(const ModelSpec& spec, int layer_index) {
  if (layer_index < 0 || layer_index >= spec.n_layers)
    throw std::out_of_range("dispatch_layer: layer index out of range");

  LayerDispatch d;
  d.layer = layer_index;
  d.kind = spec.layer_kinds[layer_index];
  d.norm_eps = spec.norm_eps;
  d.hc = spec.hc;

  switch (d.kind) {
    case LayerKind::LINEAR_ATTN_GDN:
      d.use_gdn = true;
      d.gdn = spec.gdn;
      break;
    case LayerKind::FULL_ATTN_QSA:
      // Sparse QSA only when the model actually uses the indexer (Qwen).
      d.use_qsa = spec.use_qsa_indexer;
      d.qsa = spec.qsa;
      break;
    case LayerKind::MAMBA_SSM:
      d.use_mamba = true;  // primitive not ported yet (see integration plan)
      break;
    case LayerKind::MOE:
      break;
  }

  // A layer may also carry routed experts (always on Qwen, MoE-only layers on
  // Nemotron). The has_moe vector is authoritative when present.
  if (layer_index < static_cast<int>(spec.has_moe.size()))
    d.use_moe = spec.has_moe[layer_index];
  if (d.use_moe) d.expert = spec.expert;

  return d;
}

// ---------------------------------------------------------------------------
// Debug dump
// ---------------------------------------------------------------------------

std::string describe(const ModelSpec& s) {
  std::ostringstream o;
  o << "ModelSpec {\n";
  o << "  family            = " << to_string(s.family) << "\n";
  o << "  hidden_size       = " << s.hidden_size << "\n";
  o << "  n_layers          = " << s.n_layers << "\n";
  o << "  norm_eps          = " << s.norm_eps << "\n";
  o << "  use_qsa_indexer   = " << (s.use_qsa_indexer ? "true" : "false") << "\n";
  o << "  layer_kinds       = [";
  for (size_t i = 0; i < s.layer_kinds.size(); ++i) {
    if (i) o << ", ";
    o << to_string(s.layer_kinds[i]);
  }
  o << "]\n";
  o << "  has_moe           = [";
  for (size_t i = 0; i < s.has_moe.size(); ++i) {
    if (i) o << ", ";
    o << (s.has_moe[i] ? '1' : '0');
  }
  o << "]\n";
  o << "  expert.n_experts  = " << s.expert.n_experts << "\n";
  o << "  expert.top_k      = " << s.expert.top_k << "\n";
  o << "  expert.group_size = " << s.expert.group_size << "\n";
  o << "  expert.moe_intermediate = " << s.expert.moe_intermediate << "\n";
  o << "  expert.activation = " << to_string(s.expert.activation) << "\n";
  o << "  expert.prefix_fmt = " << s.expert.tensor_prefix_format << "\n";
  o << "  hc_count          = " << s.hc.hc_count << "\n";
  o << "  hc_lowrank        = " << s.hc.hc_lowrank << "\n";
  o << "  output_gate_type  = " << s.hc.output_gate_type << "\n";
  o << "  qsa{n_heads=" << s.qsa.n_heads
    << ", kv_heads=" << s.qsa.kv_heads
    << ", head_dim=" << s.qsa.head_dim
    << ", compress_ratio=" << s.qsa.compress_ratio
    << ", token_budget=" << s.qsa.token_budget
    << ", block_topk=" << s.qsa.block_topk
    << ", eps=" << s.qsa.eps << "}\n";
  o << "  gdn{conv_kernel=" << s.gdn.conv_kernel_dim
    << ", num_key_heads=" << s.gdn.num_key_heads
    << ", num_value_heads=" << s.gdn.num_value_heads
    << ", key_head_dim=" << s.gdn.key_head_dim
    << ", value_head_dim=" << s.gdn.value_head_dim
    << ", eps=" << s.gdn.eps << "}\n";
  o << "}";
  return o.str();
}

}  // namespace veystrix::engine
