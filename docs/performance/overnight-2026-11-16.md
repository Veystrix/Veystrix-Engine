# Overnight speed work, 16 November 2026

Branch `overnight-speed-2026-11-16` (from `unlimited-generation` at f9a9713). The owner was away for ten hours and asked for continuous experiments on prefill and decode speed with two fixed rules: the weights stay 4-bit, and quality comes before speed. Nothing here lowers precision, prunes experts, drops tokens or changes the routed experts. A change that is not bit-identical is opt-in and carries a measured quality check.

Source of ideas: `Veystrix-Core Prefill and Decode Optimization.txt` (a generic Apple Silicon survey written without access to this repository).

## How the plan maps onto this engine

The plan is written for llama.cpp / MLX with weights resident in memory. This engine streams routed experts from SSD on a 16 GB M4 Air (no M5 neural accelerators), so many items do not apply or were already measured. Sorted by what the plan says:

| Plan item | Status here |
|---|---|
| Raise `iogpu.wired_limit_mb` | Not done: needs `sudo`, and a wired shortfall is a kernel fault. Weights are streamed, so the working set is not the limit. Left to the owner |
| 16 KiB-aligned zero-copy buffers | Tried for experts (`VEYSTRIX_NEMOTRON_EXPERT_MAP`, `..._ZERO_COPY`): slower, see handoff 15 Nov |
| mmap + `MADV_WILLNEED`, residency sets | Measured: `madvise(WILLNEED)` blocks, 0.2 ms per call |
| simdgroup_matrix GEMM tuning, M5 `matmul2d` | MLX already ships tiled kernels; the M4 has no neural accelerators |
| FlashAttention Nq crossover, block skip | MLX's `scaled_dot_product_attention` already does it; see attention below |
| ANE / AMX prefill | Not attempted: fp16 ANE numerics would change outputs and the routed experts are streamed |
| Zero-allocation decode, fused ops | Decode is 75-90% SSD wait; the GPU part is 41 ms of 137 ms |
| K=1 MTP, EAGLE, lookup drafting | Measured, no gain (verification rows add expert reads) |
| KV quantization, SnapKV, DuoAttention | INT8 KV is already the app default; token eviction changes outputs, so not done |
| TEAL / CATS activation sparsity | Changes outputs; not done |
| Lower-bit weights (SpinQuant, QTIP, IQ3) | Forbidden by the owner's rule |

## Baseline, real checkpoint (Nemotron 3.5 Lightning, INT8 KV, 8,192-token prefill, cool machine, AC)

Profile driver: scratchpad `prefill_profile.py` (per layer-type wall time, `pool.fetch` waits, prefetch log). 8,192 tokens, chunk 2,048, layer-major segment 8,192: 252 tok/s. Time split of 32.3 s: Mamba 9.7 s (23 layers), routed + shared experts 18.2 s of which 4.6 s waiting for SSD reads, attention 4.3 s (6 layers).

## Result 1: read-ahead defect in layer-major prefill (adopted)

`pool.prefetch_layer(next MoE layer)` was called when a MoE layer began, while the one-layer staging area still held that layer's own unread experts, so it issued 0-1 reads. Every second MoE layer was then read on demand (92-110 experts, 0.27 s each). After the fix, stale experts of earlier layers also piled up (30, 43, ... 78 staged but never used) and shrank each later prefetch.

Changes (rollback `VEYSTRIX_NEMOTRON_PREFETCH_ORDER=legacy`):
- `models/nemotron35/backend_core.py`: ask for the next MoE layer right after the layer's first chunk has taken its experts out of the staging area; ask for the first MoE layer before layer 0 runs.
- `models/nemotron35/streamed_experts.py`: `prefetch_layer` drops staged experts of other layers first.

Same experts are read, same arithmetic: token digest identical (`3abc72800267a328`, 16 tokens after the 8,192-token prompt) in every run, MLX peak 3.47 GB and available-memory floor 5.8-5.9 GiB unchanged, swap flat, thermal level 0.

| Arm | tok/s | MoE s | fetch wait s |
|---|---|---|---|
| legacy (3 runs) | 252.0 / 252.0 / 251.6 | 19.2-19.3 | 5.1-5.2 |
| order fixed only | 262.9 / 261.1 | 18.0 | 3.9 |
| first version of the purge (dropped the running layer's staged experts too; re-read some, failed `test_each_layers_experts_are_read_once_per_segment`) | 277.0 / 280.7 | 15.6-15.8 | 1.2-1.3 |
| final: purge other layers only, free the finished layer's leftovers at `end_layer` and top the next layer up | 287.4 / 287.0 (legacy 249.6 in between) | 15.4 | 0.95-0.98 |

+15% prefill, exact. SSD bytes read in the 8,192-token prefill rise from 21.2 to 22.7 GB (whole layers are read ahead, the router uses about 100 of 128 experts); the SSD is not the limit there. Evidence: `results/nemotron35/*night-prefetch-*`, `*night-prefill-profile-base*`.
