// Model-agnostic adapter / config layer for the Veystrix C++ decode engine.
//
// Purpose
// -------
// The engine primitives in engine.hpp, qsa.hpp and gdn_conv.h are deliberately
// model-agnostic: they take every shape / epsilon / head count as an argument.
// What was missing is a single place that turns a *checkpoint config* into the
// exact set of primitives (and exact configs) a decode step must call. This
// header is that place. It knows about two concrete families today:
//
//   * Qwen3.8-Flash-Next  (QWEN4_EXP): 48 layers, GDN linear-attention on a
//     3:1 cadence with QSA full-attention, hyper-connections, top-10 of 512
//     SwiGLU Q4 experts on *every* layer (group_size 32).
//   * Nemotron 3.5 Lightning (NEMOTRON_H): 52 layers, a flat block-type list
//     of Mamba/SSM, dense-attention and MoE layers, top-6 of 128 relu2 Q4
//     experts on the MoE layers only (group_size 64).
//
// Both are expressed as the SAME ModelSpec, so the decode loop is written once
// and dispatches through dispatch_layer(). A GENERIC family plus an explicit
// from_generic_config() builder is the extension point for future models.
//
// Deliberate non-goals
// --------------------
// * No JSON parsing in C++. The caller (Python via
//   tools/gen_model_adapter_vectors.py, or C++ integration code) passes the
//   scalar values in. The flat key=value format is documented in that tool and
//   at the bottom of this file.
// * No new MLX code here. QSAConfig is reused from the sibling qsa.hpp adapter
//   (UNMODIFIED). GDNConfig is defined locally because no sibling currently
//   provides one; when a gdn.hpp adapter lands, replace this definition with an
//   include and keep the same field names (see the "Integration" note below).
// * No weight loading / pager logic (that stays in expert_store / io_pager).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "qsa.hpp"  // sibling adapter (do not edit): provides QSAConfig

namespace veystrix::engine {

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

// The checkpoint family. GENERIC is the catch-all for future models whose
// config is supplied explicitly through from_generic_config().
enum class ModelFamily { QWEN4_EXP, NEMOTRON_H, GENERIC };

// Which attention / recurrent / expert primitive family a layer needs.
// NOTE: Nemotron's plain dense-attention layers are represented as
// FULL_ATTN_QSA (the "full attention family") with spec.use_qsa_indexer=false
// and qsa.n_heads=0, so dispatch_layer() returns use_qsa=false for them. This
// keeps the enum to the four values the engine actually dispatches on while
// still distinguishing sparse QSA from dense attention.
enum class LayerKind { LINEAR_ATTN_GDN, FULL_ATTN_QSA, MAMBA_SSM, MOE };

// Expert MLP activation. SWIGLU = up * (gate*sigmoid(gate)) (Qwen),
// RELU2 = relu2(up) (Nemotron).
enum class Activation { SWIGLU, RELU2 };

// ---------------------------------------------------------------------------
// GDN configuration (locally defined — see note at top of file)
// ---------------------------------------------------------------------------

// Everything the GDN linear-attention path (engine.hpp gdn_* primitives +
// gdn_conv.h) needs to be driven for a layer. Sourced from the Qwen
// text_config linear_* fields.
// Integration: when a sibling gdn.hpp adapter defines GDNConfig, delete this
// struct and `#include "gdn.hpp"` above; keep the field names identical so no
// call site changes.
struct GDNConfig {
  int conv_kernel_dim = 0;   // linear_conv_kernel_dim (K)
  int num_key_heads = 0;     // linear_num_key_heads  (Hk)
  int num_value_heads = 0;   // linear_num_value_heads (Hv)
  int key_head_dim = 0;      // linear_key_head_dim   (Dk)
  int value_head_dim = 0;    // linear_value_head_dim (Dv)
  float eps = 1e-6f;         // RMSNorm epsilon for the gated output norm
};

// ---------------------------------------------------------------------------
// Config structs
// ---------------------------------------------------------------------------

// Expert / MoE configuration shared by both families.
struct ExpertConfig {
  int n_experts = 0;          // routed experts (512 Qwen / 128 Nemotron)
  int top_k = 0;              // experts per token (10 / 6)
  int group_size = 32;        // affine Q4 group size (32 / 64)
  int moe_intermediate = 0;   // per-expert intermediate width (640 / 1856)
  Activation activation = Activation::SWIGLU;

  // Tensor-name convention for the streamed expert stacks. Placeholders:
  //   {L} = decoder layer index
  //   {P} = projection part  (Qwen: gate_proj|up_proj|down_proj;
  //                           Nemotron: fc1|fc2)
  //   {T} = tensor part      (weight|scales|biases)
  // Qwen:    language_model.model.layers.{L}.mlp.switch_mlp.{P}.{T}
  // Nemotron: backbone.layers.{L}.mixer.switch_mlp.{P}.{T}
  std::string tensor_prefix_format;
};

// Qwen hyper-connection parameters (zeroed for families without them).
struct HyperConnectionConfig {
  int hc_count = 0;                 // hc_count (Qwen: 4)
  int hc_lowrank = 0;               // hc_lowrank (Qwen: 320)
  std::string output_gate_type;     // "sigmoid" (Qwen) / "" (none)
};

// The single model-agnostic spec consumed by the decode loop.
struct ModelSpec {
  ModelFamily family = ModelFamily::GENERIC;

  int hidden_size = 0;              // transformer hidden width H
  int n_layers = 0;                 // number of decoder layers

  // Per-layer attention/recurrent kind. Index == decoder layer index.
  std::vector<LayerKind> layer_kinds;
  // Per-layer flag: does this layer contain a routed-expert MoE block?
  // Qwen: true for all layers. Nemotron: true only on "moe" block types.
  std::vector<bool> has_moe;

  ExpertConfig expert;
  float norm_eps = 1e-6f;           // RMSNorm / LayerNorm epsilon
  HyperConnectionConfig hc;

  QSAConfig qsa;                    // valid for the QSA full-attention family
  bool use_qsa_indexer = false;     // true only for sparse-QSA (Qwen) attention
  GDNConfig gdn;                    // valid for LINEAR_ATTN_GDN layers
  // Nemotron Mamba-2 geometry (declared here to avoid a header dependency on
  // mamba_layer.hpp; fields mirror MambaConfig one-for-one). Valid for
  // MAMBA_SSM layers. Populated by the extended from_nemotron_config overload.
  int mamba_num_heads = 0, mamba_head_dim = 0, mamba_n_groups = 0;
  int mamba_ssm_state_size = 0, mamba_conv_kernel = 0;
  int mamba_intermediate_size = 0, mamba_conv_dim = 0;
  float mamba_time_step_min = 0.001f, mamba_time_step_max = 100.0f;
};

// ---------------------------------------------------------------------------
// Builders (scalars in, no JSON)
// ---------------------------------------------------------------------------

// Qwen3.8-Flash-Next from text_config scalars. `layer_types` is the ordered
// list of "linear_attention" / "full_attention" strings. Every Qwen layer has
// an MoE block, so has_moe is set true for all n_layers.
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
    int gdn_key_head_dim, int gdn_value_head_dim);

// Nemotron 3.5 Lightning from config.json scalars. `layer_block_types` is the
// ordered "layers_block_type" list of "mamba" / "attention" / "moe". Only
// "moe" layers carry routed experts. No hyper-connections.
ModelSpec from_nemotron_config(
    int hidden_size,
    int n_layers,
    const std::vector<std::string>& layer_block_types,
    int n_experts, int top_k, int group_size, int moe_intermediate,
    float norm_eps);

// Full overload: additionally carries Mamba-2 geometry and time-step limits
// read from the checkpoint config.json (owner inputs). Calls the base builder
// then fills the mamba_* fields.
ModelSpec from_nemotron_config(
    int hidden_size,
    int n_layers,
    const std::vector<std::string>& layer_block_types,
    int n_experts, int top_k, int group_size, int moe_intermediate,
    float norm_eps,
    int mamba_num_heads, int mamba_head_dim, int mamba_n_groups,
    int mamba_ssm_state_size, int mamba_conv_kernel,
    int mamba_intermediate_size, int mamba_conv_dim,
    float mamba_time_step_min, float mamba_time_step_max);

// Extension point for future models: the caller has already decided the layer
// kinds, the MoE mask and the exact expert/qsa/gdn configs; fill them on the
// returned spec before use.
ModelSpec from_generic_config(int hidden_size, int n_layers,
                              const std::vector<LayerKind>& layer_kinds,
                              const std::vector<bool>& has_moe);

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

// The exact set of primitives (and the exact configs to hand them) that apply
// to one decoder layer. A layer may run an attention/recurrent branch AND an
// MoE branch (Qwen); Nemotron layers run exactly one of the two.
struct LayerDispatch {
  int layer = -1;
  LayerKind kind = LayerKind::MOE;
  bool use_gdn = false;      // gdn_* / gated_delta_step_kernel / rms_norm_gated
  bool use_qsa = false;      // qsa_indexer_pre_rope + qsa_rms_norm + attention
  bool use_mamba = false;    // Mamba/SSM scan (mamba_layer.hpp primitive)
  bool use_moe = false;      // router + q4_projection + activation + combine
  QSAConfig qsa;             // valid when use_qsa
  GDNConfig gdn;             // valid when use_gdn
  // Mamba geometry (mirrors ModelSpec mamba_* fields) for MAMBA_SSM layers.
  int mamba_num_heads = 0, mamba_head_dim = 0, mamba_n_groups = 0;
  int mamba_ssm_state_size = 0, mamba_conv_kernel = 0;
  int mamba_intermediate_size = 0, mamba_conv_dim = 0;
  float mamba_time_step_min = 0.001f, mamba_time_step_max = 100.0f;
  ExpertConfig expert;       // valid when use_moe
  float norm_eps = 0.f;
  HyperConnectionConfig hc;
};

// Pure function: ModelSpec + layer index -> the primitives to run.
// Throws std::out_of_range for an invalid layer index.
LayerDispatch dispatch_layer(const ModelSpec& spec, int layer_index);

// ---------------------------------------------------------------------------
// Debug helpers
// ---------------------------------------------------------------------------

std::string to_string(ModelFamily f);
std::string to_string(LayerKind k);
std::string to_string(Activation a);

// Multi-line human-readable dump of every field (for debugging / testFailure).
std::string describe(const ModelSpec& spec);

// ---------------------------------------------------------------------------
// Flat key=value config format (produced by gen_model_adapter_vectors.py)
// ---------------------------------------------------------------------------
//
// One file per model, UTF-8, one `key=value` per line. '#' comments and blank
// lines are ignored. Vector values are comma-separated with no spaces. Float
// values are decimal/scientific (e.g. 1e-06) parsed with std::stof. The
// `expected_*` keys are the Python-side ground truth used by the test.
//
//   model_family=QWEN4_EXP|NEMOTRON_H|GENERIC
//   hidden_size=2560
//   n_layers=48
//   layer_types=linear_attention,linear_attention,full_attention,...
//   expected_layer_kinds=LINEAR_ATTN_GDN,FULL_ATTN_QSA,...
//   expected_has_moe=1,1,...
//   n_experts=512
//   top_k=10
//   group_size=32
//   moe_intermediate=640
//   expert_activation=SWIGLU|RELU2
//   expert_prefix=language_model.model.layers.{L}.mlp.switch_mlp.{P}.{T}
//   norm_eps=1e-06
//   hc_count=4
//   hc_lowrank=320
//   output_gate_type=sigmoid
//   use_qsa_indexer=1
//   qsa_n_heads=4
//   qsa_kv_heads=1
//   qsa_head_dim=128
//   qsa_compress_ratio=4
//   qsa_token_budget=2048
//   qsa_eps=1e-06
//   gdn_conv_kernel=4
//   gdn_n_key_heads=16
//   gdn_n_value_heads=48
//   gdn_key_head_dim=128
//   gdn_value_head_dim=128
//
// (The qsa_* / gdn_* / hc_* keys are emitted as 0 / empty for families that do
// not use them; the builders ignore them for those families.)

}  // namespace veystrix::engine
