// Bit-exact C++ port of the production Qwen4ExpPLELayer (layer index 1).
//
// Mirrors, op for op, the Python code in
//   mlx_vlm/models/qwen4_exp/language.py:
//     Qwen4ExpNGramEmbedding / ShardedEmbedding / Qwen4ExpPLELayer /
//     Qwen4ExpRMSNorm (group_size variant).
//
// The n-gram embedding shards are stored PACKED (affine Q4), exactly as in the
// real checkpoint.  Following mlx.nn.QuantizedEmbedding.__call__, we gather the
// packed rows FIRST and then call mx::dequantize on the gathered rows -- this
// is required for bit-exactness (dequantizing before gather gives a different
// reduction order).
//
// The key/value projections are veystrix::engine::Linear, which can be either
// dense or packed Q4 (Linear::quantized).  Both paths are bit-exact via the
// existing test_linear_quantized coverage.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "engine.hpp"

namespace veystrix::engine {

// One packed affine-Q4 n-gram embedding shard.  `weight` is uint32 packed
// [rows, dims*bits/32]; `scales`/`biases` are [rows, dims/group_size].
struct PLEEmbShard {
  mx::array weight, scales, biases;
  PLEEmbShard() = delete;
  PLEEmbShard(mx::array w, mx::array s, mx::array b)
      : weight(std::move(w)), scales(std::move(s)), biases(std::move(b)) {}
};

// All weights/config needed by one PLE layer.  layer_multipliers / head_offsets
// / head_vocab_sizes are the arrays STORED in the real checkpoint (the C++
// engine should pass them straight through).  Compute-path helpers below are
// provided for building the module from config (the synthetic oracle).
struct PLEWeights {
  Linear key_proj;    // ple_embed_dim -> hc_hidden_size (dense or packed Q4)
  Linear value_proj;  // ple_embed_dim -> hidden_size   (dense or packed Q4)
  mx::array norm_key, norm_query, norm_conv;  // RMS gains, shape [hc_hidden]
  mx::array conv_w;   // depthwise conv1d weight [hc_hidden, kernel, 1]
  mx::array layer_multipliers;   // int64 [ngram_size]
  mx::array head_offsets;        // int64 [ngram_heads]
  mx::array head_vocab_sizes;    // int64 [ngram_heads]
  std::vector<PLEEmbShard> emb_shards;
  std::vector<int64_t> shard_offsets;  // length n_shards + 1, cumulative rows

  int emb_group_size = 32;
  int emb_bits = 4;
  std::string emb_mode = "affine";

  int ngram_size = 3;
  int heads_per_ngram = 8;
  int hc_count = 4;
  int hidden_size = 2560;
  // Optional: ple_embed_dim from config.  When 0, the head width is derived
  // from the stored scales shape (scales[rows, dims/group_size]).
  int ple_embed_dim = 0;
  int eos_token_id = 248044;
  float eps = 1e-6f;
  int conv_kernel = 4;
  int conv_dilation = 3;  // == ngram_size
  int short_conv_state_len = 9;  // (kernel-1)*dilation

  PLEWeights() = delete;
  PLEWeights(Linear kp, Linear vp, mx::array nk, mx::array nq, mx::array nc,
             mx::array cw, mx::array lm, mx::array ho, mx::array hvs,
             std::vector<PLEEmbShard> shards, std::vector<int64_t> offs,
             int egs, int eb, int ns, int hpn, int hc, int hs, int ped,
             int eos, float eps_, int ck, int cd, int sl,
             std::string mode = "affine")
      : key_proj(std::move(kp)), value_proj(std::move(vp)),
        norm_key(std::move(nk)), norm_query(std::move(nq)),
        norm_conv(std::move(nc)), conv_w(std::move(cw)),
        layer_multipliers(std::move(lm)), head_offsets(std::move(ho)),
        head_vocab_sizes(std::move(hvs)), emb_shards(std::move(shards)),
        shard_offsets(std::move(offs)), emb_group_size(egs), emb_bits(eb),
        emb_mode(std::move(mode)), ngram_size(ns), heads_per_ngram(hpn),
        hc_count(hc), hidden_size(hs), ple_embed_dim(ped),
        eos_token_id(eos), eps(eps_),
        conv_kernel(ck), conv_dilation(cd), short_conv_state_len(sl) {}
};

// Mutable per-layer cache.  `context` is [B, ngram_size-1] int64 (the trailing
// token history); `conv_state` is [B, short_conv_state_len, hc_hidden] float.
// An unset optional means "None" (the Python path initializes zeros/eos).
struct PLECache {
  std::optional<mx::array> context;
  std::optional<mx::array> conv_state;
  PLECache() = default;
  PLECache(mx::array ctx, mx::array st)
      : context(std::move(ctx)), conv_state(std::move(st)) {}
};

// Compute one PLE layer.  hidden_states: [B,S,hc*H] float.  input_ids: [B,S]
// int64 (cast internally).  mask: optional 2-D bool [B,S] (causal / padding).
// Returns [B,S,hc*H] float (the value added to hidden_states by the decoder).
mx::array ple_forward(const mx::array& hidden_states,
                      const mx::array& input_ids,
                      const PLEWeights& w,
                      PLECache& cache,
                      const mx::array* mask = nullptr);

// ---- Compute-path fallbacks (mirror the Python helpers exactly) ----------
// These let a caller build the module from config instead of checkpoint arrays.

// _splitmix64 with full 64-bit wrap-around arithmetic.
uint64_t ple_splitmix64(uint64_t value);

// _build_layer_multipliers: returns int64 multipliers [ngram_size].
std::vector<int64_t> ple_build_layer_multipliers(
    int unigram_vocab_size, int ngram_size, int ple_layer_index, int64_t seed);

// _is_prime / _find_nth_prime_after.
bool ple_is_prime(int64_t value);
int64_t ple_find_nth_prime_after(int64_t start, int count);

// ngram_heads_vocab_sizes for all heads (head index = ple_layer_index *
// ngram_heads + local head idx, 0-based).
std::vector<int64_t> ple_head_vocab_sizes(int ngram_vocab_size_base,
                                          int ngram_heads,
                                          int ple_layer_index);

}  // namespace veystrix::engine
