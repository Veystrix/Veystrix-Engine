// Paged, precision-tiered KV store for generic attention blocks.
//
// Token positions [0, n) of one attention layer are split into three regions:
//
//   prefix  [0, P)      protected first tokens (system prompt, tool schemas,
//                       attention sinks); precision `prefix_bits`
//   middle  [P, H)      aged history; K precision `hist_k_bits`, V `hist_v_bits`
//   recent  [H, n)      sliding window of >= W tokens; precision `recent_bits`
//
// H = max(P, floor((n - W) / page) * page) moves in whole pages (256 tokens),
// so the recent window holds between W and W + page - 1 tokens once full.
// Every token is quantized ONCE from its original bf16 activation into each
// format it will ever be read in (middle-format copies are written at write
// time; when a page ages out of the window only its recent copy is dropped),
// so there is never an INT8 -> INT4 cascade.
//
// Precision 16 means bf16 (exact). Special cases keep the reference paths:
//   * all-bf16 single region  -> mx::fast::scaled_dot_product_attention, the
//     same call as mlx_lm KVCache + base.scaled_dot_product_attention (exact);
//   * single quantized region with equal K/V bits -> the op sequence of
//     mlx_lm base.quantized_scaled_dot_product_attention (exact vs mlx_lm
//     QuantizedKVCache).
// Otherwise attention is computed over the concatenated per-region scores with
// ONE softmax (no dequantization of the history to bf16: q@K^T and p@V use
// mx::quantized_matmul on the packed rows).
//
// Storage grows in 256-token pages (memory follows the tokens present). All
// members are MLX arrays (immutable handles), so copying a KVStore is a cheap
// copy-on-write fork.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace veystrix::engine::generic {

namespace mx = mlx::core;

struct KVPolicy {
  std::string name = "bf16";  // label used in capabilities / fingerprints
  int prefix_tokens = 0;      // P (multiple of page)
  int prefix_bits = 16;
  int64_t recent_tokens = -1; // W; -1 = unbounded (everything recent)
  int recent_bits = 16;
  int hist_k_bits = 16;
  int hist_v_bits = 16;
  int group_size = 64;
  int page = 256;
  // Sliding-window cap (>0 evicts tokens older than sliding_window; keeps local attention O(1) RAM).
  int64_t sliding_window = 0;
  // 1M-context sparse-paged mode: keeps exact prefix + recent in RAM plus per-page min/max key
  // summaries and selects top-K cold history pages per query.
  bool sparse_paged = false;
  int sparse_top_pages = 32;

  bool exact() const {
    return recent_tokens < 0 && recent_bits == 16 && !sparse_paged;
  }
  // Parse a mode name: bf16 | int8 | k8v4 | int4 | tiered | tiered4 | tiered4+ |
  // tiered-k8v4 | sparse-paged, then apply explicit overrides (window/prefix/bits; <0 = keep).
  static KVPolicy from_mode(const std::string& mode, int64_t window = -2, int prefix = -1,
                            int hist_k_bits = -1, int hist_v_bits = -1);
  std::string describe() const;  // stable string (snapshot identity)
  // Bytes per token per (kv_heads*head_dim) element pair at steady state for
  // a context of `tokens` (for the memory plan / reports).
  double bytes_per_value(int64_t tokens) const;
};

// One stored region segment: either bf16 rows or affine-packed rows.
// Quantised / multi-region attention materialises its scores; a prefill chunk's query rows are processed in
// blocks so one score tile holds at most kScoreTile values (64 Mi, 256 MiB in float32).
constexpr int64_t kScoreTile = int64_t{1} << 26;
int attention_query_block(int L, int kL, int Hq);
// [a, b) query-row spans of at most qb rows, except that a remainder under 16 rows joins the span before it:
// MLX picks another SDPA kernel for <= 8 query rows, so a short tail would change the numerics.
std::vector<std::pair<int, int>> query_spans(int L, int qb);

struct KVSeg {
  int kbits = 16;  // 16 = bf16 rows in k[0]; 4/8 = packed {data, scales, biases}
  int vbits = 16;
  std::vector<mx::array> k;  // bf16: {data}; packed: {data, scales, biases}
  std::vector<mx::array> v;
  int64_t start = 0;         // token position of row 0 of the buffers
  int64_t cap = 0;           // rows allocated
  int64_t len = 0;           // rows written (from row 0)
};

class KVStore {
 public:
  KVStore() = default;
  explicit KVStore(const KVPolicy& p) : policy_(p) {}

  const KVPolicy& policy() const { return policy_; }
  int64_t offset() const { return n_; }

  // Write S new tokens (keys/values [B, Hkv, S, D], original bf16 activations,
  // already rotated) and return the attention output for queries
  // [B, Hq, S, D] with the mlx_lm "causal" convention for S > 1.
  mx::array update_and_attend(const mx::array& queries, const mx::array& keys,
                              const mx::array& values, float scale, bool causal_single = false,
                              const std::optional<mx::array>& mask = std::nullopt);

  // Exact relative-position bias + zero-value sink attention (Inkling 2 Mini):
  //   score(i, j, h) = q_i . k_j * scale + rel_logits[h, i, i - j] for 0 <= i - j < E
  //   sink contributes exp(sink[h, i]) to the softmax denominator with zero value.
  // If `rp_t_ed` ([E, d_rel]) is provided, `rel_logits` is `r` ([B, Hq, L, d_rel]) and
  // the dot product r . rp_t[d] is fused on-the-fly inside the Metal mask kernel.
  mx::array update_and_attend_rel_sink(const mx::array& queries, const mx::array& keys,
                                       const mx::array& values, const mx::array& rel_logits,
                                       const mx::array& sink, float scale,
                                       const std::optional<mx::array>& rp_t_ed = std::nullopt,
                                       bool defer_compact = false);

  // Snapshot as named arrays (contiguous, exact) + scalar meta.
  std::vector<std::pair<std::string, mx::array>> save(const std::string& prefix) const;
  void load(const std::string& prefix, const std::vector<std::pair<std::string, mx::array>>& arrays,
            int64_t n_tokens);
  int64_t nbytes() const;
  void append_state_arrays(std::vector<mx::array>& out) const;

  // Roll back the tail of the KV cache to `new_n <= n_` (used by speculative verification).
  // Returns false if a compaction boundary dropped rows needed for rollback.
  bool truncate(int64_t new_n);
  // Run deferred recent-window compaction (and lazy INT4 aging in sparse-paged mode).
  void compact_if_needed();

  // Region boundaries for the current length (for tests/telemetry).
  int64_t history_end() const;
  int64_t indexed_pages() const { return static_cast<int64_t>(page_k_min_.size()); }

 private:
  KVPolicy policy_;
  int64_t n_ = 0;
  KVSeg prefix_, middle_, recent_;
  bool init_ = false;
  bool dense_recent_ = false;
  // Quest-style per-page min/max key summaries for sparse-paged 1M context (1 entry per 256-tok page).
  std::vector<mx::array> page_k_min_, page_k_max_;
  int64_t indexed_to_ = 0;

  void append_seg(KVSeg& seg, const mx::array& k, const mx::array& v, int64_t first_pos);
  void compact_recent(int64_t keep_from);
  void update_page_index(const mx::array& k, int64_t first_pos);
  // Sparse-paged BF16 prefix + BF16 recent form one adjacent region. Keep them
  // in a single contiguous buffer until history ages past P + W, then split the
  // protected prefix back out and hand aged pages to the existing INT4 middle.
  bool use_dense_recent() const;
  void split_dense_prefix();
};

}  // namespace veystrix::engine::generic
