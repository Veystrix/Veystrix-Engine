// Generic routed-expert source for MoE blocks (recipe `moe.*` with stacked
// expert tensors, e.g. mlx_lm `switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`).
//
// Layout is derived from the safetensors headers: for every stacked tensor the
// absolute file offset of expert 0 and the bytes of one expert row. Two modes:
//
//   streamed (default): per call, only the wanted experts' rows are read from
//     the shard into compact MLX buffers [n_wanted, ...]; the OS page cache is
//     the expert cache (no private pool: measured loser on 16 GB). Adjacent
//     wanted experts are coalesced into one read; reads run on parallel lanes.
//   resident: the full stacked tensors are loaded at open (memory plan says
//     they fit); no reads at run time.
//
// The bytes are the file's bytes, so a compact gather_qmm over remapped
// indices computes exactly what gather_qmm over the full stack computes.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "row_fetch.hpp"

namespace veystrix::engine::generic {

namespace mx = mlx::core;
class StIndex;

struct ExpertTensorLoc {
  std::string name;
  int fd = -1;
  int nocache_fd = -1;
  int64_t base = 0;       // absolute offset of expert 0's row
  int64_t row_bytes = 0;  // bytes of one expert's slice
  mx::Shape row_shape;    // shape of one expert's slice
  mx::Dtype dtype = mx::uint32;
  int64_t experts = 0;
};

struct ExpertProjLoc {
  ExpertTensorLoc w, s, b;
  bool has_b = false;
  bool has_gs = false;
  mx::array global_scale = mx::array(0.0f);  // tiny [E] F32 per-expert scale (NVFP4), always resident
  int group_size = 64;
  int bits = 4;
  std::string mode = "affine";
};

// The projection arrays of one fetch (3 projs for gate/up/down or 2 for fused w13/w2).
struct ExpertSet {
  std::vector<mx::array> w, s, b;  // b has entries only for projections with biases
  std::vector<std::optional<mx::array>> gs;  // per-proj [n_wanted] global_scale when has_gs
  std::optional<mx::array> remap;  // uint32 [E] -> compact row; nullopt = identity
  std::vector<int> wanted;         // sorted distinct expert ids (streamed)
  int64_t bytes = 0;               // bytes read for this set
  int rows_memcpy = 0, rows_pread = 0;  // rowfetch: page-cache hits copied vs rows read
  std::vector<uint8_t*> bases;     // destination of row 0 per stacked array (w,s,b per proj), alloc_stacks only
};

class ExpertLayer {
 public:
  ExpertLayer(const StIndex& idx, const std::vector<std::string>& proj_paths,
              const std::vector<std::pair<int, int>>& quant /* (gs,bits) per proj */,
              bool resident, const std::vector<std::string>& modes = {});

  int experts() const { return experts_; }
  bool resident() const { return resident_; }
  const std::vector<ExpertProjLoc>& projs() const { return projs_; }
  int64_t bytes_per_expert() const;

  // Compact arrays for the wanted experts (sorted, distinct). Resident: the
  // full stacks with no remap.
  ExpertSet fetch(const std::vector<int>& wanted, bool decode = false) const;

  // Page-cache read-ahead of every expert row of this layer (fire and forget,
  // F_RDADVISE). Used during prefill for the next layer.
  void advise_all() const;

  // Streamed only: allocate the compact stacks for `wanted` (sorted, distinct)
  // without reading, and append one rowfetch::Row per (expert, tensor) whose
  // group is the compact row. Rows are expert-major, so experts complete one
  // after another. Rows already in the page cache are memcpy'd from a shared
  // mapping (VEYSTRIX_CPP_ROWS_MMAP), the rest pread: same bytes either way.
  // decode: one-token decode read (VEYSTRIX_GENERIC_ROW_ORDER=tensor reorders only these).
  ExpertSet plan(const std::vector<int>& wanted, std::vector<rowfetch::Row>& rows, bool decode = false) const;
  // Uninitialised stacks of `capacity` rows each (no remap). add_rows appends the
  // rows that place experts[i] at stack row first_row + i (group = group_base + i).
  // Any row placement gives the same gather_qmm result per (token, expert) when the
  // indices are not sorted (decode), because each output row reads only its own
  // expert row.
  ExpertSet alloc_stacks(int capacity) const;
  void add_rows(const ExpertSet& set, const std::vector<int>& experts, int first_row, int group_base,
                std::vector<rowfetch::Row>& rows) const;
  // Rows (dst unused) of the given experts, for rowfetch::prefetch_cold.
  std::vector<rowfetch::Row> prefetch_rows(const std::vector<int>& experts) const;

  // Temporarily load all experts resident in RAM (e.g. for stage-major prefill),
  // then release back to streamed state when finished.
  void load_resident();
  void release_resident();

 private:
  std::vector<ExpertProjLoc> projs_;
  int experts_ = 0;
  bool resident_ = false;
  bool initially_resident_ = false;
  std::vector<mx::array> res_w_, res_s_, res_b_;
};

// Reads in flight per fetch (VEYSTRIX_GENERIC_READ_LANES, default 8).
int read_lanes();
// Parallel read lanes for large sliced prefill fetches
// (VEYSTRIX_GENERIC_PREFILL_READ_LANES or VEYSTRIX_GENERIC_READ_LANES, default 16).
int prefill_read_lanes();
// fetch() goes through rowfetch (memcpy page-cache hits from a shared mapping,
// pread the rest); VEYSTRIX_GENERIC_ROWS=0 preads coalesced runs instead.
bool rows_enabled();

}  // namespace veystrix::engine::generic
