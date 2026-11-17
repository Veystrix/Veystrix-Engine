# moe_stream port — native expert loader + pager into the MoE step

Owner note for `cpp/engine/src/moe_stream.{hpp,cpp}`,
`cpp/engine/tools/gen_moe_stream_vectors.py`, `cpp/engine/tests/test_moe_stream.cpp`.

## Goal

Close the I/O gap: stop feeding the C++ engine expert weights via `.npy` vectors.
Instead, read the **real Qwen checkpoint** shards into a persistent slot pool via
the native pager (`cpp/decode/src/io_pager.cpp` +
`models/flashnext/native_read.c`), remap the router's expert ids to slots, and
run the exact `moe_block` (gather_qmm) path on those slot-resident weights.

## Bit-exact slot rules (from OPUS_E0_E1_SPEC.md, E0/E1)

- **Pin slots with a refcount until command buffers complete.** In this port the
  only synchronization primitive we have for the singlestream MLX path is
  `mx::eval(out)`: it blocks until every enqueued GPU kernel that reads the
  slot buffers has finished. Each slot used by a forward is refcounted up before
  the `moe_block` call and down only after `mx::eval` returns. A slot is never
  evicted/reused while its refcount is non-zero.
- **Combine in router-rank order.** `plan_remap` emits slots in the router's
  route order (first-appearance dedup, no sorting by expert id). We pass those
  slot ids directly as `rhs_indices` to `moe_block` and leave `scores` in the
  same order, so `weighted_combine` reduces in router-rank order. Sorting or
  re-ranking would change the floating-point reduction order and break parity.
- Fail closed: if `plan_remap` returns slot `-1` (pool exhausted and no safe
  eviction), forward throws instead of producing a partial mapping.

## What is bit-exact here

- Generator (`gen_moe_stream_vectors.py`) uses the real `SafeTensorStore.rows_np`
  on the real Qwen3.8-Flash-Next checkpoint (layer 0, experts 0..7, K=4) and the
  exact `mx.gather_qmm` SwiGLU path, saving only `x`, `route`, `scores`,
  `expected` (+ `meta.json`) via `mx.save`.
- C++ test (`test_moe_stream.cpp`) loads those four vectors via `mx::load`, then
  reads the SAME byte ranges directly out of the real shard with
  `plan_remap` + `run_batched_reads` (batched positioned pread), wraps the bytes
  as `mx::array` (weight=uint32, scales/biases=bf16 reinterpret — identical bits
  to `store.to_mx`/`expert_store.view_bf16_if_needed`), and calls `moe_block` on
  the slot-remapped route. Output is compared byte-for-byte via
  `mx::view(..., mx::uint32)`.

Physically the two paths read the same bytes:
`file_offset = 8 + header_len + data_offsets[0] + expert * row_bytes`, and
`row_bytes = prod(shape[1:]) * itemsize`. This matches `TensorRef.start` /
`TensorRef.row_bytes` in `models/flashnext/store.py:add_shard`.

## Integration instructions

- This is a standalone port; CMakeLists.txt is owned elsewhere. Link the sources
  directly (see the verification command in the handoff report). `moe_stream.cpp`
  includes `io_pager.hpp` from `cpp/decode/src`; that is a separate CMake project,
  so compile it into the engine target by adding its source + `native_read.c`
  (and `-framework Dispatch` via libdispatch, already linked by MLX on macOS).
- `StreamingMoE` is persistent: construct once per MoE layer, call `forward`
  per decode step. Resident experts keep their slot across steps; misses are
  deduped and batched into one `run_batched_reads` call.
- The constructor needs `num_slots`; choose it from the decode step's top-k plus
  headroom (e.g. 16 for K=10). `read_width` is the number of concurrent pread
  lanes (8 matches the E0/E1 harness).

## Status

Full pager+MLX integration: implemented and verified bit-exact (see report).
Remaining for production: hook `StreamingMoE::forward` into `Decoder` in place
of the `.npy`-fed `ExpertWeights`, and pin the cross-stream event if the real
pipeline moves reads to a separate CPU stream (here `mx::eval` is the barrier).
