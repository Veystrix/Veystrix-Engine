// Inkling 2 DSpark semi-autoregressive parallel drafter (mtp.safetensors).
//
// Architecture (vLLM vllm/models/inkling2/nvidia/dspark.py + DeepSpec RNNHead):
//   - Target captures features at `target_layer_ids` (e.g. [5, 23, 41, 48]):
//     AttnRes partial block inputs at visits 5, 23, 41 and final-norm output at 48.
//   - Each of the 3 dense draft layers mixes the 4 target features via softmax(fuse_weight)
//     + dflare_norm into context K/V (k_proj_ctx, v_proj_ctx, k_norm_ctx, RoPE theta=1e6).
//     Local layers (0, 2) keep a 1,024-token sliding window; global layer (1) keeps full K/V.
//   - Drafting runs an anchor + K mask_embed tokens (K <= block_size - 1 = 7) with bidirectional
//     block attention over [context K/V, block K/V], then a recurrent rank-256 Markov head
//     (markov_w1, rnn_proj, hidden_out) sharing the target's embed_tokens, embed_norm, and lm_head.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "generic/generic_model.hpp"

namespace veystrix::engine::generic {

namespace mx = mlx::core;

struct DSparkConfig {
  int hidden_size = 2304;
  int num_hidden_layers = 3;
  int num_attention_heads = 24;
  int num_key_value_heads = 8;
  int head_dim = 128;
  int intermediate_size = 9216;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 1000000.0f;
  int sliding_window_size = 1024;
  int block_size = 8;
  int markov_rank = 256;
  std::set<int> local_layer_ids;
  std::vector<int> target_layer_ids;
};

struct DSparkLayerKV {
  std::optional<mx::array> k;  // [1, Hkv, cap, D] BF16
  std::optional<mx::array> v;  // [1, Hkv, cap, D] BF16
  int64_t len = 0;
  int64_t cap = 0;
};

struct DSparkState {
  std::vector<DSparkLayerKV> layers;
  int64_t ctx_len = 0;
  std::optional<mx::array> last_features;  // [1, num_targets, H] BF16 for f_rnn
};

class Inkling2DSpark {
 public:
  static std::unique_ptr<Inkling2DSpark> try_load(const Loader& loader, const Json& recipe);

  // Fold hidden_out into the target's LM head once at load time (8-bit affine, ~53 MB)
  // so draft_block runs 1 batched lm_head call + K tiny 256-col Markov bias matmuls.
  void attach_lm_head(const std::function<mx::array(const mx::array&)>& lm_head_fn, int unpadded_vocab);

  const DSparkConfig& config() const { return cfg_; }
  const std::vector<int>& target_layer_ids() const { return cfg_.target_layer_ids; }
  int max_draft_tokens() const { return std::max(0, cfg_.block_size - 1); }
  int64_t weight_bytes() const { return weight_bytes_; }

  void reset(DSparkState& st) const;

  // Rewind draft context KV caches to new_n tokens.
  void truncate(DSparkState& st, int64_t new_n) const;

  // Snapshot / restore DSparkState arrays.
  NamedArrays save(const DSparkState& st, const std::string& pfx = "dspark") const;
  void load(DSparkState& st, const NamedArrays& a, const std::string& pfx = "dspark") const;

  // Commit S newly accepted/prefilled positions' target features [S, T, H] BF16
  // into the draft layers' context KV caches and update last_features.
  void commit_context(DSparkState& st, const mx::array& target_features_sth) const;

  // Propose up to `num_draft` tokens (1 <= num_draft <= block_size - 1) given the anchor
  // token (at position st.ctx_len) and the target's shared embed/head projections.
  // If `min_cum_prob > 0`, stops drafting early once cumulative top-1 probability drops
  // below `min_cum_prob` (avoids streaming wasted experts for low-confidence tail tokens).
  std::vector<int32_t> draft_block(
      const DSparkState& st, int32_t anchor_token, int num_draft,
      const std::function<mx::array(const mx::array&)>& embed_normed_fn,
      const std::function<mx::array(const mx::array&)>& lm_head_fn,
      float logit_softcap, int unpadded_vocab, float min_cum_prob = 0.0f) const;

 private:
  struct Layer {
    bool is_local = false;
    mx::array input_ln = mx::array(0.0f);
    mx::array post_ln = mx::array(0.0f);
    mx::array dflare_norm = mx::array(0.0f);
    QLinear qkv_proj, k_proj_ctx, v_proj_ctx, o_proj;
    mx::array q_norm = mx::array(0.0f);
    mx::array k_norm = mx::array(0.0f);
    mx::array k_norm_ctx = mx::array(0.0f);
    QLinear gate_up_proj, down_proj;
  };

  struct MarkovHead {
    QEmbedding markov_w1;
    mx::array rnn_w_sp = mx::array(0.0f);    // [2r, 3r] BF16 (transposed for matmul)
    mx::array rnn_w_hf = mx::array(0.0f);    // [2H, 3r] BF16 (transposed for matmul)
    mx::array rnn_bias = mx::array(0.0f);    // [3r] F32
    mx::array hidden_out_w = mx::array(0.0f);  // [r, H] BF16 (transposed for matmul)
    mx::array hidden_out_b = mx::array(0.0f);  // [H] BF16
    mx::array dflare_norm = mx::array(0.0f);
    bool folded = false;
    std::optional<mx::array> fold_w, fold_s, fold_b;  // 8-bit affine [V_trim, r]
    std::optional<mx::array> fold_bias;               // [1, V_trim] BF16
  };

  mx::array mix_features(const mx::array& features_sth, int row, const mx::array& norm_w) const;
  void append_layer_kv(DSparkLayerKV& kv, const mx::array& k, const mx::array& v, bool is_local) const;

  DSparkConfig cfg_;
  float scale_ = 0.088388f;
  mx::array fuse_alpha_ = mx::array(0.0f);  // [num_layers + 1, num_targets] F32
  mx::array mask_embed_ = mx::array(0.0f);  // [1, 1, H] BF16
  mx::array final_norm_ = mx::array(0.0f);  // [H] BF16
  std::vector<Layer> layers_;
  MarkovHead markov_;
  int64_t weight_bytes_ = 0;
};

}  // namespace veystrix::engine::generic
