// Bit-exact skip of all-zero fc2 K-groups for Nemotron relu2 experts.
//
// Nemotron expert = fc1 -> relu2 -> fc2 (no SwiGLU gate). relu2(h)=max(h,0)^2
// produces EXACT zeros. A fc2 K-group (group_size=64) whose 64 input
// activations are all zero contributes exactly nothing, so its compressed
// weight chunk need not be read from SSD.
//
// STAGES (see research_notes/Second miracle no new hardware/fc2_skip.md):
//   1. probe_zero_fraction         - zero-activation / zero-group fractions
//   2. compact_fc2_kgroups          - EXPERIMENT: K-compaction (NOT bit-exact)
//      zero_fill_skipped_groups     - THE bit-exact mechanism (dense layout +
//                                     unchanged stock kernel, skipped groups'
//                                     weight/scale/bias bytes zero-filled)
//      fc2_*_skip                   - shipped bit-exact projections
//   3. fc2_chunk_layout / repack_fc2_kgroup_major - K-group-major layout
//   4. Fc2SkipPolicy / should_skip_fc2 - two-stage read + break-even
//
// NOTE (stage 4): enabling the two-stage read requires a HOST SYNC PER LAYER
// (relu2 activations must be read back from the GPU to find zero groups).
// Marked NEEDS OWNER APPROVAL in code and in the research note. Off by default.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace mx = mlx::core;
namespace veystrix::engine {

// ------------------------------- Stage 1 ------------------------------------

struct ZeroFraction {
  int64_t total_activations = 0;   // rows * intermediate
  int64_t zero_activations = 0;    // exactly-zero activations (any row)
  int64_t total_group_slots = 0;   // rows * n_groups
  int64_t zero_group_slots = 0;    // (row, group) pairs that are all zero
  int64_t total_groups = 0;        // n_groups (intermediate / group_size)
  int64_t zero_groups = 0;         // groups all-zero in EVERY row (skippable)
  double activation_zero_fraction = 0.0;
  double group_slot_zero_fraction = 0.0;
  double group_zero_fraction = 0.0;   // fraction skippable across all rows
  std::vector<int> zero_group_ids;    // ids of fully-zero (all rows) groups
};

// relu2_act: [rows, intermediate] (bf16/f32/f16). group_size divides last dim.
// A group is "fully zero" iff all group_size activations are exactly zero in
// every row (the only case safe to skip in a batched kernel).
ZeroFraction probe_zero_fraction(const mx::array& relu2_act, int group_size);

// Convenience: ids of fully-zero groups.
std::vector<int> fully_zero_groups(const mx::array& relu2_act, int group_size);

// ------------------------------- Stage 2 ------------------------------------

struct Fc2Compacted {
  mx::array x, w, scales, biases;
  std::vector<int> kept_groups;
  Fc2Compacted(mx::array x_, mx::array w_, mx::array s_, mx::array b_)
      : x(std::move(x_)), w(std::move(w_)), scales(std::move(s_)),
        biases(std::move(b_)) {}
};

// EXPERIMENT ONLY. Remove the zero groups from the K dimension (compaction).
// DEMONSTRABLY NOT bit-identical: compacting K changes the SIMD lane
// assignment and the summation tree. Kept so the owner can reproduce the
// negative result; not used by the shipped projection.
Fc2Compacted compact_fc2_kgroups(const mx::array& x, const mx::array& w,
                                 const mx::array& scales,
                                 const mx::array& biases,
                                 const std::vector<int>& zero_groups,
                                 int group_size, int bits);

struct Fc2ZeroFilled {
  mx::array w, scales, biases;
  Fc2ZeroFilled(mx::array w_, mx::array s_, mx::array b_)
      : w(std::move(w_)), scales(std::move(s_)), biases(std::move(b_)) {}
};

// THE BIT-EXACT MECHANISM. Keep the DENSE K layout and the UNCHANGED stock
// kernel; return copies of w/scales/biases with the skipped groups' bytes
// zero-filled. Because x is exactly zero in those groups, they contribute
// `scale*0 + 0*bias = 0` and adding zero cannot change the reduction tree.
// This is bit-identical to the dense computation (verified byte-for-byte).
Fc2ZeroFilled zero_fill_skipped_groups(const mx::array& w,
                                       const mx::array& scales,
                                       const mx::array& biases,
                                       const std::vector<int>& zero_groups,
                                       int hidden, int intermediate,
                                       int group_size, int bits);

struct Fc2SkipStats {
  int64_t kept_groups = 0;
  int64_t skipped_groups = 0;
  int64_t total_groups = 0;
};

// y = quantized_matmul(relu2_act, W', S', B', transpose=true, G, bits)
// with W'/S'/B' zero-filled at the fully-zero groups. Bit-exact vs dense.
mx::array fc2_quantized_matmul_skip(const mx::array& relu2_act,
                                    const mx::array& w, const mx::array& scales,
                                    const mx::array& biases, int hidden,
                                    int intermediate, int group_size, int bits,
                                    Fc2SkipStats* stats = nullptr);

// Same mechanism for the gather path (single stacked expert replicated to E
// entries if rhs_indices selects it). Bit-exact vs dense gather_qmm.
mx::array fc2_gather_qmm_skip(const mx::array& relu2_act,
                              const mx::array& w, const mx::array& scales,
                              const mx::array& biases,
                              const mx::array& rhs_indices,
                              int hidden, int intermediate,
                              int group_size, int bits,
                              Fc2SkipStats* stats = nullptr);

// ------------------------------- Stage 3 ------------------------------------

struct Fc2ChunkLayout {
  int hidden = 0, intermediate = 0, group_size = 0, bits = 0;
  int64_t n_groups = 0;
  int64_t words_per_group_weight = 0;   // group_size * bits / 32
  int64_t bytes_weight_per_group = 0;   // hidden * words_per_group * 4
  int64_t bytes_scale_per_group = 0;    // hidden * scale_itemsize
  int64_t bytes_bias_per_group = 0;
  int64_t chunk_bytes = 0;              // per (expert, group)
  int64_t total_bytes = 0;              // per expert, all groups
};

Fc2ChunkLayout fc2_chunk_layout(int hidden, int intermediate, int group_size,
                                int bits, int scale_itemsize = 4);

// Dense w [hidden, intermediate/8] uint32, scales/biases [hidden, n_groups]
// -> bytes: for g in 0..n_groups-1: for n in 0..hidden-1:
//             8 x uint32 LE packed weights, then scale, then bias.
// (scale/bias raw bytes in the source dtype, e.g. 4-byte float32 or 2-byte
//  bfloat16.) The Python helper writes the matching format.
std::vector<uint8_t> repack_fc2_kgroup_major(const mx::array& w,
                                             const mx::array& scales,
                                             const mx::array& biases,
                                             int hidden, int intermediate,
                                             int group_size, int bits);

// Original dense format kept as fallback behind a flag.
enum class Fc2Layout { Dense, KGroupMajor };
Fc2Layout fc2_layout_from_env();  // VEYSTRIX_NEMOTRON_FC2_LAYOUT=kgroup|dense

// ------------------------------- Stage 4 ------------------------------------

struct Fc2SkipPolicy {
  bool enabled = false;             // VEYSTRIX_NEMOTRON_FC2_SKIP == 1 (default 0)
  double min_skip_fraction = 0.0;   // threshold param: only enable when the
                                    // observed fully-zero group fraction >= this
  int64_t round_trip_bytes = 0;     // break-even: extra round trip cost, in
                                    // bytes (only enable when saved > this)
};

Fc2SkipPolicy fc2_skip_policy_from_env();

// Break-even rule: enable per layer/expert only when the skip pays for the
// extra GPU->host sync round trip (NEEDS OWNER APPROVAL).
bool should_skip_fc2(const Fc2SkipPolicy& policy, const ZeroFraction& zf,
                     int64_t bytes_saved);

}  // namespace veystrix::engine
