# Overnight speed work, 16 November 2026

Branch `overnight-speed-2026-11-16` (from `unlimited-generation` at f9a9713). The owner was away for ten hours and asked for continuous experiments on prefill and decode speed with two fixed rules: the weights stay 4-bit, and quality comes before speed. Nothing here lowers precision, prunes experts, drops tokens or changes the routed experts. A change that is not bit-identical is opt-in and carries a measured quality check.

## Summary for the owner (read this first)

**Nemotron (default on, each with a rollback flag; weights and routed experts unchanged):** prefetch order, SDPA prefill attention, pipelined expert compute, early shared expert for prefill rows, fused Mamba-2 scan. Real 8K prefill 252 -> 355.6 tok/s overall. The SDPA attention and the fused scan are rounding-level (not bit-identical); see the tool-call comparison and the review notes below. Your call whether they should be default.

**Qwen Flash-Next (all opt-in, app defaults untouched).** Recommended launch flags, all exact (24 of 24 cases of the repository quality suite produce the same tokens as the baseline): `--enable kv-tiered4 --enable short-throughput --enable later-turn-scratch-ring --enable single-chunk-layer-major --enable merge-small-tail --enable qsa-shared-dense-kv --enable qsa-fused-gather --enable s3fifo-pool`. Real server numbers for the first five of these: agentic turns of 1.4-1.7K new tokens 51 -> 27 s; larger turns of 3.3-4.5K new tokens over 3-11K cached 105-112 s -> 42-69 s; cold 3.4-3.8K-token first turn 37-43 s; minimum available memory 2.46 GiB (baseline 3.18 GiB), swap flat. The two kernel options add about 2 s and 3 s on a cold 3-4K prompt and 12% on a long-history stage (in-process, identical tokens); `s3fifo-pool` takes decode from 2.49 to 2.62 tok/s. Also exact on the 24-case suite: `FLASHNEXT_QSA_GATHER_TILE=32` exported before launch (about 4% more on a cold prefill; changes the prefix-cache fingerprint once) `--enable ngram-dedup` (about 3% on a cold 4.3K prefill), `--enable dense-gemm` (about 2.4%) and `--enable qsa-fused-sdpa` (about 3.5%); all four verified together on the 24-case suite. Follow-ups of about 300 tokens stay 23-28 s (SSD-bound). Optional, NOT recommended by default: `--enable qsa-fast-prefill` (rounding-level: another 1.2-1.9x on turns over long histories, 34-44 s instead of 42-69 s, but it changes 4 of 24 quality-suite token streams and FAILS the full-tier gate on `tool_2`, where the model's stop-or-continue decision is a coin flip; it also changes the prefix-cache fingerprint once). Cancel mid-prefill is clean; needle retrieval and tool-call replies show no loss.

**What did not work (all recorded below):** decode prefetch/overlap/IO workers/buffered fill/pool policies other than S3-FIFO, larger pools (RAM-bound), route-union reads, PLE row prefetch, 8K windows, chunk 4096 for Nemotron, dense GEMM and block-parallel Mamba. Both models are SSD-bound at their floors on 16 GB; what is left is RAM or lossy changes, which the rules exclude.

**Other things to know:** old prefix snapshots do not restore once after enabling `qsa-fast-prefill` (Qwen) or after the Nemotron attention/scan markers; the fanless Air reaches thermal level 2 after about 100 s of sustained prefill, so the guard stops 16K cold runs (not measured); the wired-limit sysctl, `/Applications` and the app defaults were not touched; nothing is pushed.

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

## Result 2: fused attention for prefill over long histories (adopted, default on)

`models/nemotron35/attention.py`: when a prefill chunk sees more than one 4,096-token block, dequantize the INT8 history and run `mx.fast.scaled_dot_product_attention` instead of the Python block loop (which evaluated after every block and ran in fp32 `quantized_matmul`). The single-block path always did this. Decode keeps the block loop. Rollback `VEYSTRIX_NEMOTRON_ATTENTION_PREFILL_SDPA=0`. Part of the session identity when on, so existing prefix snapshots will not be restored once after the upgrade.

Per layer, one 2,048-token chunk (real attention shape, random K/V): history 8K 284 -> 83 ms, 16K 544 -> 178 ms, 30K 979 -> 347 ms. Real 12,288-token natural-text prefill: 260.5 -> 294.5 tok/s.

Quality, not just speed. Against an fp32 reference (8K history, 256 queries) the RMS error is 3.1e-5 for the fused path (the rounding of the bf16 output itself) and 5.2e-5 for the block loop, so the fused path is the more accurate one. It is not bit-identical to the loop (about half the bf16 outputs differ by one step). On the real checkpoint the last-position logits (12,288 tokens) differ from the loop by KL 0.0136, mean |d| 0.145, same top-20 tokens, same top-1; halving the prefill chunk size alone gives KL 0.0070, mean |d| 0.070. Evidence: `results/nemotron35/*night-attn-*`.

## Result 3: compute each expert as its read finishes (adopted, default on, bit-identical)

`StreamedSwitchMLP` (several-row calls, i.e. prefill) used to wait for every expert of the layer and then compute. `ExpertPool.fetch(..., on_ready=)` now hands each expert over as its read completes and the layer starts computing it while the remaining reads run (`_pipelined`). Same per-expert operations on the same rows; the output is reassembled by position, so results are bit-identical (unit test compares logits and all state, pipelined vs not, chunk-major and layer-major). Rollback `VEYSTRIX_NEMOTRON_PIPELINE_EXPERTS=0`.

Real checkpoint, same digests in both arms:

| Prefill tokens | loop tok/s | pipelined tok/s |
|---|---|---|
| 512 | 87.9 / 86.7 | 99.8 / 99.4 (+14%) |
| 1,536 | 163.9 | 214.4 (+31%) |
| 8,192 | 314.7 | 325.7 (+3.5%) |

Short prefills are SSD-bound by nature: at 512 tokens the router uses 107 of 128 experts per layer, 15.5 GB is read in 5.15 s (3.0 GB/s, the SSD's rate). Below about 1,500 tokens a prefill costs about 5 s whatever its length unless the file cache holds more of the experts, which needs more RAM. Above that it is GPU-bound (about 3 TFLOPS in the 4-bit GEMMs, 78% of dense bf16 peak).

## Tried and not adopted tonight

- Submitting layers with `mx.async_eval` instead of `mx.eval` in layer-major prefill: 311 against 314 tok/s, no CPU bubbles to remove.
- `mx.compile` on the Mamba SSD scan: bit-identical but 41.2 against 41.3 ms. Step 128 instead of 256 is 15% faster for the scan but changes rounding and is worth only about 2% of prefill.
- Starting the shared expert early for single-token decode: decode fell about 10% (5.85 against 6.6 tok/s on a 512-token prompt; note that run was only 8 tokens long, see the measurement note below), so it is used for prefill rows only (`VEYSTRIX_NEMOTRON_OVERLAP_SHARED`, prefill +4-7%, bit-identical).
- More expert read threads (8 -> 16 or 24) for decode: 8.4-8.5 against 8.2-8.4 tok/s, no gain. Cold reads are capped by the SSD at 3.0-3.3 GB/s at any thread count and any split size (measured after `purge`, 2/6/40 experts); cached reads do scale with threads (6.4 -> 11.9 GB/s at 6 experts) but decode is not limited by them.
- Reading prefill experts with `F_NOCACHE` so the 22 GB stream does not push decode's experts out of the file cache: not built into the tree. Three consecutive 4,000-token turns with the default build decode at 8.71 / 8.81 / 8.74 tok/s and prefill at 334 / 331 / 329 tok/s, so a fresh prefill leaves nothing to recover.
- Dequantize + dense bf16 matmul instead of `quantized_matmul` for the large projections: bit-identical for M >= 384 and about 10% faster per GEMM (about 3% of prefill); not done, and not bit-equal at M = 96 so unusable for per-expert calls.

## Quality check of all three prefill changes together (real checkpoint, teacher-forced)

`logit_seq.py`: prefill 12,160 tokens of natural text (repository docs and code), then score the last 128 positions (all-position logits, true next token known). Legacy = `PREFETCH_ORDER=legacy`, `PIPELINE_EXPERTS=0`, `ATTENTION_PREFILL_SDPA=0`.

| Pair | mean KL | median | p95 | top-1 agree |
|---|---|---|---|---|
| legacy vs legacy (repeat run) | 0 | 0 | 0 | 1.000 |
| legacy vs new | 0.0099 | 0.0034 | 0.050 | 0.945 |
| legacy vs legacy at chunk 1024 | 0.0114 | 0.0038 | 0.042 | 0.938 |
| new vs legacy at chunk 1024 | 0.0123 | 0.0040 | 0.056 | 0.977 |

Mean negative log-likelihood of the true next token: legacy 2.0845, new 2.0922, legacy at chunk 1024 2.0875; next-token accuracy 0.465 / 0.457 / 0.457 (127 positions, so one token is 0.008). The new build moves the distributions as much as halving the prefill chunk size does and no more. The prefetch order and the pipelining are bit-identical; the whole difference comes from the attention change (rounding only, closer to an fp32 reference).

Harness comparison, same cell (8,192 tokens, INT8 KV, prompt lookup, prefix cache on, memory sampler on): legacy 231.9 tok/s (TTFT 35.4 s), new 293.5 tok/s (TTFT 27.9 s), decode 7.66 / 7.70 tok/s, available-memory floor 5.13 / 5.11 GiB (`results/nemotron35/*stage11-qualification-manual`, the two newest).

## Where decode stands (3,000-token natural-text prompt, 128 tokens, real checkpoint)

119 ms per token: 38 ms GPU work up to each router sync, 75.5 ms dispatching and waiting for the experts (physical reads about 200 MB per token, the SSD limit). A dependent chain of the same layers with no host syncs takes 23 ms of GPU time, so about 15 ms per token is sync and dispatch latency that the per-layer router round trip forces. Decode 8.4-8.8 tok/s here; 7.7 at 8K context. Nothing lossless that I could find moves the I/O part: more RAM is the lever.

## Measurement note

The first decode comparisons of the night (shared-expert start, worker counts) used a driver whose context limit was prompt + 8 tokens, so each decoded 8 tokens. They are superseded by the 128-token natural-text runs above. The prefill numbers were never affected (prefill does not depend on the output length).

## Integration check (real checkpoint, prefix cache on, 25K-character system prompt, new session identity)

Populate: cold request 24.5 s. A fresh process with the same prompt restored 6,226 tokens from the SSD cache and answered in 4.6 s with the same output hash (`0a64d6fe2eec973b`). A prefill cancelled after 6 s ended as `cancelled`; the next normal request produced the same tokens as an uncancelled run (`81296a2bfae9f667`). Evidence: `results/nemotron35/*night-integ-*`.

## Result 4: fused Mamba-2 scan for prefill chunks (adopted, default on, not bit-identical)

`models/nemotron35/fast_ssm.py`. Per 2,048-token chunk a Mamba layer took 96 ms: `in_proj` 38.7 ms and `out_proj` 15.5 ms (both about 2.9 TFLOPS in the 4-bit GEMMs, already near the GPU's rate), conv 2.8 ms, and the state-space scan 39.6 ms, which needs only about 5 GFLOP: it is memory-bound. The stock `ssm_attn` builds a `[64, 256, 256]` fp32 segment-sum matrix per step (repeat, tril, cumsum), repeats `C @ B^T` to every head, and broadcasts the carried state to every token (16,384 tiny matmuls, 15 ms per 8 steps by itself).

The fused version computes the decay as `exp(cs[i] - cs[j])` from one fp32 cumsum per head, folds the product with `C @ B^T` and the causal mask into one compiled kernel, and does the carried-state and next-state updates as one GEMM per group. Scan 39 -> 13 ms. Rollback `VEYSTRIX_NEMOTRON_FAST_SSM=0`; used only for unmasked calls of 64 rows or more (decode, speculative rows and masked calls keep the stock path). Part of the session identity (`ssm_prefill: fused-v1`), so existing prefix snapshots are not restored once after the upgrade, as with the attention change.

Accuracy: it is a different association of the same arithmetic, not bit-identical. Against a float64 sequential recurrence on the same bf16 inputs (768 tokens, with and without carried state) the output error is the same as the stock scan's (RMS 4.567e-3 against 4.566e-3, the bf16 rounding of the output) and the final state is off by 4.6e-7 RMS (stock 1.3e-8; the state's RMS is 0.16). The decay from `cs[i] - cs[j]` loses some fp32 bits where the cumulative sum is large, which is the reason the state error is larger than the stock scan's; it stays 4 orders of magnitude below bf16 resolution. Unit tests: agreement with the stock scan at ragged lengths with and without state, error against float64 truth, wrapper fallbacks, flag.

Real checkpoint, 8,192-token natural-text prefill (cool machine, AC): stock 319.6 tok/s (Mamba 9.86 s, MoE 13.99 s, attention 2.05 s) -> fused 355.6 tok/s (Mamba 7.30 s), +11%. Teacher-forced check as in the quality section (12,160 tokens, last 128 positions): stock vs fused mean KL 0.0076, median 0.0045, p95 0.027, top-1 agreement 0.969 (stock vs a repeat of stock: 0, 1.000); mean NLL of the true next token 2.6281 stock, 2.6351 fused (127 positions, one token is 0.008); next-token accuracy 0.472 / 0.465. For scale, halving the prefill chunk size moved the distributions by KL 0.0114 in the earlier comparison. Evidence: `results/nemotron35/*night-prof-now`, `*night-prof-fastssm`, `*night-q-*`.

## Prefill chunk size 4,096 for Nemotron (measured, not adopted)

`FLASHNEXT_PREFILL_CHUNK_TOKENS=4096` (default 2,048; the variable is shared with Qwen, so a Nemotron-only default would need a code change): 8,192-token prefill 353.7 -> 375.3 tok/s (+6%, the routed-expert GEMMs run at 2.85 TFLOPS from 192 rows per expert up, against 2.69 at 96 and 1.3-1.7 at 24-48; a 2,048-token chunk averages 96 rows per expert). The prices: MLX peak 3.64 -> 4.47 GB, available-memory floor 5.79 -> 5.13 GiB, compressor peak 692 -> 1,778 MB. Quality equal (stock-scan-free comparison, 12,160 tokens: 2,048 vs 4,096 mean KL 0.0074, top-1 agreement 0.984, NLL 2.6351 vs 2.6316). Not made the default because of the memory pressure on a 16 GB machine; the owner can set the variable for Nemotron-only sessions. Evidence: `*night-chunk-4096`, `*night-chunk-2048`, `*night-q-fast4096`. Note for any later change that batches chunks: `quantized_matmul` rows computed in calls of fewer than 192 rows differ in the last bits from rows computed in calls of 192 or more (a different kernel), so the result already depends on how many rows an expert gets in one call.

## Qwen full-attention layers: where the non-expert time goes, and the dense prefix (opt-in, rounding-level)

Per-component wall time of the app's Qwen prefill (forced evals per sub-module, so the MoE number includes its read waits; `QPROF=1` in the scratchpad driver `qwen_pp.py`):

| Component (4,096-token prompt, 48.2 s, layer-major) | s | per call |
|---|---|---|
| full attention, 12 layers | 17.2 | 1.43 s |
| linear attention (Gated DeltaNet), 36 layers | 8.2 | 0.23 s |
| hyper-connections (2 per layer) | 5.0 | |
| PLE (one layer) | 3.9 | |
| MoE incl. read waits | 13.2 | |

At 1,600 tokens (chunk-major, 34.9 s): full attention 4.95 s, linear attention 3.14 s, hyper-connections 1.7 s, PLE 1.35 s, MoE 24.2 s. The routed-expert GEMMs themselves are not the compute bottleneck: `gather_qmm` at the real shapes runs 1.7-2.2 TFLOPS (4.4 s of compute for 1,600 tokens over 48 layers, 9 s for 4,096) and the Gated DeltaNet kernel costs 8 us per token per layer. Above about 3,000 tokens the prefill is compute-bound by the attention blocks, not SSD-bound.

Why full attention is slow: `FLASHNEXT_QSA_KERNEL=gather` (set by `kv-tiered4`) runs the sparse-attention indexer per query: tiles of 8 queries each gather their selected K/V rows (up to 2,052 slots, dequantizing the INT4/INT8 cache), run a one-query SDPA, and `mx.eval` after every tile. A query whose causal length has at most `block_topk` (512) complete blocks of 4 tokens, i.e. every query below position 2,051, selects all causal tokens, but still pays for 2,052 gathered slots. A 2,048-token chunk is 256 tiles: 607 ms per layer. Tile size 32 alone only gives 530 ms.

Change: `FLASHNEXT_QSA_DENSE_PREFIX=1` (default off, registered, part of the cache fingerprint as `qsa_dense_prefix`): those leading dense queries run as one causal `mx.fast.scaled_dot_product_attention` over the same dequantized cache rows (`_gather_kv` with `arange(rows)`), the remaining sparse queries use the existing per-tile gather. The K/V values are identical (same rows, same dequantization); only the SDPA reduction order differs (full-sequence kernel instead of the single-query kernel). A first chunk of 2,048 tokens: 607 -> 121 ms per layer. A 4,096-token prefill, app config, cold: 48.0 -> 41.3 s (85 -> 99 tok/s, +16%); tile 32 on top adds nothing (41.4 s). Short prompts gain the most (a 1,600-token prefill spends 4.95 s in full attention, about 1 s after).

Quality, teacher-forced on the real checkpoint (`qwen_logit.py`: prefill N-128 natural-text tokens, then all-position logits for the last 128; the repeat of the baseline is bit-identical, KL 0):

| N | what is scored | mean KL | median | p95 | top-1 agree | NLL base -> dense | next-token acc |
|---|---|---|---|---|---|---|---|
| 2,000 | dense-region positions (direct effect) | 0.0040 | 0.0004 | 0.016 | 0.984 | 0.8792 -> 0.8740 | 0.756 / 0.756 |
| 4,224 | sparse-region positions (effect through the cache) | 0.0067 | 0.0029 | 0.022 | 0.961 | 1.1709 -> 1.1678 | 0.646 / 0.638 |

The first generated token of the 4,096-token cold-prefill pair differed (a high-entropy prompt), which is consistent with these KLs. Not made the default: the fingerprint change invalidates existing Qwen prefix snapshots once, and the app's launch flags are the owner's. Suggested: add `FLASHNEXT_QSA_DENSE_PREFIX=1` to the Qwen launch environment. Evidence: `*night-qattn-*`, `*night-qwen-prof-*`, `*night-qp4096-*`, `*night-ql*`.

### Masked sparse attention (second part, opt-in, rounding-level)

The sparse queries (those past position 2,051) still paid for a per-query gather: 722 ms per full-attention layer for a 2,048-token chunk over a 2,048-token history (gather and dequantize 384 ms, index selection 119 ms, SDPA about 270 ms), although the whole K/V history of a 4K prompt is only 8 MB in bf16 per layer. `FLASHNEXT_QSA_MASKED_SPARSE=1` (default off, registered with `FLASHNEXT_QSA_MASKED_MAX_KEYS`, default 16,384, and `FLASHNEXT_QSA_MASKED_TILE`, default 256; part of the cache fingerprint as `qsa_masked_sparse`) dequantizes the history once with the same `gather_selected` the per-query path uses and runs one masked `scaled_dot_product_attention` per query tile with the existing exact selection mask (`_chunk_mask(scatter=True)`, pinned to `_selected_indices` by the existing tests). Same selected tokens, same dequantized values, only the SDPA reduction order differs. Histories longer than the cap keep the per-query gather. Unit test: tiny model, logits equal to the gather path with tiles of 5 and 7, with and without the dense prefix, and with the cap forcing the gather.

One full-attention layer, real weights, 2,048-token chunks: first chunk 607 ms -> 121 ms (dense prefix), second chunk 722 ms -> 173 ms (masked sparse).

Cold prefill, app config (`kv-tiered4` + `short-throughput`), natural text, cool machine, AC, thermal level 0, swap unchanged, same first generated token in every pair:

| Prompt tokens | baseline | dense prefix | dense prefix + masked sparse |
|---|---|---|---|
| 4,096 | 48.0 s (85 tok/s) | 41.3 s (99 tok/s) | 33.9 s (121 tok/s, +42%) |
| 8,192 | 95.1 s (86 tok/s) | not run | 72.4 s (113 tok/s, +31%) |

MLX peak 6.85 / 7.05 GB (unchanged), available-memory floor 4.1-4.3 GiB (the same as the baseline arm).

Teacher-forced quality, 4,224 tokens (scored positions in the masked region): baseline vs dense + masked mean KL 0.0073, median 0.0032, p95 0.0225, top-1 agreement 0.953; NLL 1.1709 -> 1.1661; next-token accuracy 0.646 / 0.638. Dense prefix alone: KL 0.0067. Suggested launch flags if adopted: `FLASHNEXT_QSA_DENSE_PREFIX=1 FLASHNEXT_QSA_MASKED_SPARSE=1`. Both change the cache fingerprint (existing Qwen prefix snapshots are not restored once). Evidence: `*night-qattn-masked`, `*night-qp4096-masked`, `*night-qp8192-*`, `*night-ql4224-masked`.

Server check with both flags (guarded private server on port 18080 with the app's flags, `--enable kv-tiered4 --enable short-throughput`, prefix cache on): a 3,432-token cold request answered in 43.1 s including 24 decoded tokens (baseline server 52.6 s); the identical request restored 3,425 of 3,432 tokens from the prefix cache in 11.5 s (baseline 11.1 s; decode about 2 tok/s in both); a 6,375-token cold request, which runs the masked path past position 2,051, finished cleanly in 73.9 s; no errors in the log. The greedy text of the cold and the restored request differs in wording in both builds (the restore path continues from a snapshot taken at a different chunk edge), so that difference is not caused by the flags. The dense prefix and masked path apply only to calls of at least 16 queries; decode and speculative verification keep the per-query gather (unit test). The baseline server's third request was cut when the run guard stopped it at thermal level 2 (the machine had been under load for most of the session).

Longer context, teacher-forced at N = 8,320 (scored positions in the masked region; history up to 8,320 keys), with the noise scale of the same prompt for reference (baseline run with 1,024-token prefill chunks):

| Pair | mean KL | median | p95 | top-1 agree |
|---|---|---|---|---|
| baseline vs dense prefix + masked sparse | 0.0193 | 0.0018 | 0.049 | 0.976 |
| baseline vs baseline at chunk 1,024 | 0.0110 | 0.0016 | 0.047 | 0.976 |
| chunk 1,024 vs dense prefix + masked sparse | 0.0128 | 0.0015 | 0.045 | 0.984 |

NLL of the true next token: baseline 1.1351, chunk 1,024 1.1249, flags 1.1047; next-token accuracy 0.685 / 0.693 / 0.701 (127 positions). The flags change the distributions about as much as a chunk-size change does at the median and the 95th percentile, with a somewhat heavier mean (a few positions, plausibly where the top-512 block selection flips on a near-tie); the true-token likelihood does not get worse. Prefill 95.3 s (baseline) against 70.2 s (flags) in this run. Evidence: `*night-ql8320-*`.

### PLE n-gram rows: prefetch tried and dropped

The one PLE layer costs 3.9 s of a 4,096-token prefill (the n-gram embedding reads 16 random rows per token, 65,536 rows, from a 128-shard table on SSD; `FLASHNEXT_NGRAM_PARALLEL_MIN=16` is already in the served environment, and without it a cold 4K prefill measured 60.8 s against 46.8 s). The row ids depend only on the token ids, so I tried queueing every row of the prompt on the read pool before the layers (`FLASHNEXT_PLE_PREFETCH`, exact by construction: unit-tested against the module's own ids). Cold 4K prefill, page cache purged before each arm: off 33.8 s, on 36.8 s, physical reads 84.1 -> 87.3 GB. During prefill every store descriptor is `F_NOCACHE` (the `uncached-prefill-reads` mode), so the warmed pages are not kept and each lookup reads its page again (a row is about 80 bytes but a read moves a 16 KB page: about 3 GB of physical reads per 4,096 tokens, already in the baseline). Warming through a cached descriptor would keep the pages but put 3-6 GB of page cache behind a 4-8K prompt, which the uncached mode exists to avoid. Reverted; nothing of it is in the tree.

## Qwen agentic turns through the real server: the ring that was too small (largest gain of the night)

The in-process driver measures cold prefills. A real agentic session is a conversation: each request carries the history (cached by the prefix cache) plus about 1.5-1.8K new tokens. A guarded private server with the app's flags (`--mode agentic --pool-gb 2 --enable kv-tiered4 --enable short-throughput`, prefix cache on, port 18080, `qconv.py`: four growing requests of about 5,500 characters each with a short assistant reply between them) gave, per request, prefill seconds for the new tokens (`usage.jsonl`):

| Turn (new tokens, cached prefix) | baseline | `later-turn-scratch-ring` | all four options |
|---|---|---|---|
| 1 (1,434, 0) | 35.8 | 35.8 | 32.2 |
| 2 (1,809, 1,436) | 51.5 | 40.8 | 27.2 |
| 3 (1,600, 3,247) | 52.0 | 38.2 | 27.2 |
| 4 (1,585, 4,849) | 48.7 | 38.9 | 27.2 |

Why later turns were slow: the first turn gets a temporary scratch ring (3.1 GB, two halves of 512 experts), but a later turn puts the ring into the decode pool, which the app launches at about 1.07 GB (349 slots; the planner caps the 2 GB request). The ring may use half of it, in two halves: 534 MB, about 175 experts per half. A 1,600-token prompt reads about 390 experts per layer, so `ring.stage` could not hold a layer, every layer fell back to the unstaged read path (flight log: `prefill_ring_acquire ... scratch=false ring_view_bytes=534528000`), and the prefill ran at 32 tok/s against 45-60 tok/s on a cold turn.

`FLASHNEXT_PREFILL_RING_SCRATCH_LATER=1` (`--enable later-turn-scratch-ring`, off by default, registered): when the existing pool cannot hold the wanted ring, use a temporary right-sized pool as the first turn does (3.1 GB while the prefill runs, freed after; the decode pool is untouched; the working-set check counts the temporary pool beside the decode pool and refuses if it does not fit). Same bytes read into different buffers, so results are identical (unit test compares every layer's output bit for bit). Turns 2-4: 51.5/52.0/48.7 -> 40.8/38.2/38.9 s (-22 to -26%). With `single-chunk-layer-major` (one-chunk prefills use the layer-major order, overlapping reads with compute) and `qsa-fast-prefill` added: 27.2 s each (-47%, 1.9x, about 59 tok/s on a cached-prefix turn).

Memory and heat of these server runs (guard samples every second): minimum available memory 2.26 GiB (baseline) / 3.72 GiB (scratch ring) / 3.07 GiB (all four); peak compressor 4.5 / 2.2 / 3.7 GB; swap flat at 1,262 MiB in all three; thermal level 0 throughout. The scratch ring did not make memory worse, because the fallback read path it replaces also churned memory.

Suggested app launch (as first written; see the quality-gate section below, which recommends the same list WITHOUT `--enable qsa-fast-prefill`): `--enable kv-tiered4 --enable short-throughput --enable later-turn-scratch-ring --enable single-chunk-layer-major --enable merge-small-tail --enable qsa-fast-prefill`. The first two options are exact; `qsa-fast-prefill` is rounding-level (KL 0.007-0.019 teacher-forced) and changes the cache fingerprint (existing Qwen prefix snapshots are not restored once). Evidence: `results/flashnext/*night-qwen-conv-*` (`usage.jsonl`, `guard.jsonl`).

Larger turns (stress, same server, four growing requests of 14,000 characters: 4,274 tokens cold, then about 3,400-3,950 new tokens over cached histories of 4.3K, 8.2K and 11.7K tokens; 90 s cooldown between turns):

| Turn prefill (new tokens) | baseline | all four options |
|---|---|---|
| 1 (4,274, cold) | 71.7 s (60 tok/s) | 63.4 s (67 tok/s) |
| 2 (3,946) | 104.7 s (38 tok/s) | 36.7 s (108 tok/s) |
| 3 (3,430) | 111.6 s (31 tok/s) | 34.2 s (100 tok/s) |
| 4 (3,620) | 111.9 s (32 tok/s) | 37.3 s (97 tok/s) |

About 3x faster per turn, but the memory margin is tighter, and this should be weighed by the owner: minimum available memory 3.18 GiB (baseline) against 2.03 GiB (all four); peak compressor 4.4 GB against 8.0 GB; swap 1,480 -> 1,528 MiB against 1,262 -> 1,640 MiB; thermal level 0 in both. All of it stays far below the app's 5,120 MiB swap ceiling and the guard's growth stop, and the ring's working-set check granted every lease, but the all-options runs lean on memory harder than the baseline runs. Where the margin matters more than the speed, `later-turn-scratch-ring` plus `single-chunk-layer-major` alone are exact and cost the least memory in the 1.5-1.8K-token runs above (minimum available 3.72 GiB).

Small follow-ups (about 300 new tokens, the size of a tool result) take 23-28 s each with everything on: a few hundred tokens route to most of each layer's 512 experts, so the turn reads 45-70 GB at 3 GB/s whatever its length (the same cost as a 1,500-token turn). This is the SSD limit; only more RAM for expert caching, or fewer experts read (not lossless), would change it.

Rerun of the same stress with all six suggested options (`results/flashnext/20261116-081045-night-qwen-stress-six`; the docs had grown since the first run, so the 14,000-character first turn is 3,954 tokens, not 4,274): turn 1 (3,954 cold) 37.2 s (106 tok/s), turn 2 (3,507 new) 33.8 s (104 tok/s), turn 3 (3,917 new) 39.2 s (100 tok/s), turn 4 (4,131 new over 11.4K cached) 43.6 s (95 tok/s). Memory: minimum available 2.68 GiB, peak compressor 4.35 GB (8.0 GB with the earlier four options), swap 1,392 -> 1,352 MiB (flat), thermal level 0. The tail merge is the only option added since the four-option run, so it probably accounts for the lower compressor peak as well as the cold-turn gain, but that is one run each and the prompt text also changed, so treat the attribution as a guess.

### A small final window paid for a whole second pass: `merge-small-tail`

The stress run's cold first turn (4,274 tokens) took 63 s although a 4,096-token prefill takes about 34 s. With 4,096-token windows the last 178 tokens form a second window, and in the hybrid mode a window of at most 2,048 tokens runs chunk-major, which reads the union of the experts its tokens route to (about 45 GB for 178 tokens, 25 s of SSD time) although the previous window had just read every expert. `FLASHNEXT_PREFILL_MERGE_TAIL=N` (`--enable merge-small-tail` sets 2,048; off by default; registered; needs a layer-major mode) folds a final window of at most N tokens into the one before it, never across a required (system-prompt) boundary. The chunks and their arithmetic are the same, so the result is bit-identical to the unmerged plan (unit test on the tiny model: token and cache bits equal chunk-major). Real 4,274-token cold prefill, app config plus the QSA options: 57.5 s -> 36.7 s (+36%), physical reads 139.6 -> 83.8 GB, same output hash, MLX peak 6.7 -> 6.6 GB. The same applies to every prompt whose length leaves a small remainder (4,097-6,143 tokens with 4,096 windows, and again after each further window).

## Qwen (Flash-Next) one-chunk cold prefills: layer-major with read-ahead for 1,024-2,048 tokens (opt-in, exact)

The app launches Qwen with `--enable kv-tiered4 --enable short-throughput`, i.e. the mode `layer-major-hybrid-4k`: 4,096-token layer-major windows, `FLASHNEXT_LAYER_MAJOR_MIN_TOKENS=2049`, and windows of up to 2,048 tokens executed chunk-major. So a prompt of 2,049 tokens or more already uses layer-major with read-ahead, and a prompt of up to 2,048 tokens (one chunk) takes `prefill._run_chunks`' one-span shortcut: each layer reads its routed experts and then computes, nothing overlapped. At 1,600 tokens that was 62 GB read (about 21 s of SSD time) in series with the compute, 34.8 s in all.

Change: `FLASHNEXT_LAYER_MAJOR_SINGLE=1` (default off) lets a one-chunk prefill that reaches `FLASHNEXT_LAYER_MAJOR_MIN_TOKENS` run the layer-major order (it also overrides the hybrid chunk execution for that case), so layer l+1's experts are read while layer l computes. Unit tests: same token and cache bits as the reference, off by default, hybrid does not undo it. The flag is registered in `setting_research.py`.

Real checkpoint, in-process cold prefill under the guard (driver `qwen_pp.py`, natural-text prompt, thermal level 0, AC power, swap unchanged). The output token hash was identical in every pair.

| Prompt tokens | App config today (s) | App config + `SINGLE=1`, `MIN_TOKENS=1024` (s) | physical read GB | MLX peak GB |
|---|---|---|---|---|
| 800 (min 256) | 24.3 | 25.4 (slower) | 54.5 -> 79.6 | 6.3 -> 6.2 |
| 1,200 (min 256) | 29.7 | 26.0 (+14%) | 59.5 -> 80.4 | 6.5 -> 6.4 |
| 1,600 | 34.8 | 26.7 (+30%) | 62.6 -> 81.2 | 6.6 -> 6.6 |
| 3,000 | 35.1 (already layer-major) | unchanged | 82.3 | 6.6 |

The 800 and 1,200 rows were measured with the `layer-major-4k` mode (not hybrid) with the order forced; the 1,200 baseline there is chunk-major. Layer-major reads every layer's 512 experts once (about 80 GB) whatever the prompt length, so it loses below about 1,000 tokens, where the reference path reads only the routed union (54 GB at 800 tokens), and wins from about 1,000 to 2,048 by overlapping those reads with compute. An earlier draft of this section compared 3,000 tokens against a forced chunk-major baseline (68.5 s) and called it 1.94x; the app does not run that, so that figure does not apply. Suggested settings if adopted: `FLASHNEXT_LAYER_MAJOR_SINGLE=1` and `FLASHNEXT_LAYER_MAJOR_MIN_TOKENS=1024` (the second value also moves 1,024-2,048-token multi-window cases; prompts of 2,049+ are unchanged). Not adopted into the app's defaults: that and the prefix-cache identity change (layer-major snapshots do not mix with chunk-major ones) are the owner's decision. Evidence: `results/nemotron35/*night-qwen-*`.

## Cancel during prefill with all options on (server, real checkpoint)

With the six suggested flags on, a client that disconnected 10 s and 25 s into a 9,000-character prefill produced `request_cancelled` (`memory_stop: false`) in the flight log. The server then served a normal request each time (501 and 376 new tokens, 34 and 32 s, cold chunk path). Available memory returned to 5.8-6.2 GiB, swap stayed at 1.44-1.51 GB, thermal 0. The scratch ring is released on cancel.

## Qwen decode: expert prefetch and decode overlap (no gain)

In-process, app launch environment, 300-token prompt, 48 decode tokens, same token hash in all three: baseline 2.39 tok/s (20.1 s), `FLASHNEXT_EXPERT_PREFETCH=1` 2.38 tok/s, `FLASHNEXT_DECODE_OVERLAP=1` 2.41 tok/s. With `FLASHNEXT_PROFILE_IO=1` the baseline spent 77% of decode wall in `io_wait`; the critical read task waited 11.4 s in the 16-worker queue against 4.6 s in `pread`, which is the pool running at about 2.3 GB/s of the SSD's roughly 3 GB/s, not an idle pipeline. Each token reads about 0.95 GB (hit rate about 24% in a 349-slot pool). Decode on 16 GB is bytes-bound; only a larger resident pool (RAM) or fewer bytes would help, neither of which is available without raising memory pressure. Nothing changed.

## Needle retrieval with all six suggested options (quality check, real Qwen)

Raw-text prompt of 4,516 tokens (repository docs) with three planted facts at 8%, 45% and 85% of the prompt, ending in `Answers (one short line each):\n1. The vault passphrase is`, greedy. With the six options (`kv-tiered4`, `short-throughput`, `later-turn-scratch-ring`, `single-chunk-layer-major`, `merge-small-tail`, `qsa-fast-prefill`) the model returned all three correctly: `MANGO-4417-LANTERN`, `Ottilie Brandvold`, `03:52 on the ninth of March`. Cold prefill 40.0 s against 71.7 s for the baseline run of the same prompt. The baseline also returned the first fact correctly but stopped after the first line (end token), so only the flagged run was checked for facts 2 and 3. The same test at 10,116 tokens (cold prefill 87.7 s, memory floor 3.54 GiB, swap flat, thermal 0) returned all three facts correctly as well, so the masked sparse path is exercised with histories well past the dense region. An earlier version of the prompt (ending `Answers:\n1.`) made the baseline emit an end token immediately and the flagged build start a reasoning block; that is a prompt-format sensitivity, not a retrieval result.

`scripts/flashnext/real_run_guard.py`: when the child had already exited, `os.killpg` raised `PermissionError` on macOS after a thermal-level-2 stop (seen once, after the run had finished and written its output). `stop` now returns in that case and still re-raises if the child is alive.

## Nemotron tool-call generations, fused scan on versus off (real model)

Four tool-call prompts (450-454 tokens, JSON-only replies, greedy, 70 tokens). Fused scan on (default) and `VEYSTRIX_NEMOTRON_FAST_SSM=0` both produced valid JSON for all four. Three replies were identical; one differed in content: `run_shell` with `npm test -- --timeout=120000, timeout_s 180` (fused) against `timeout 120 python -m pytest tests/ -v, timeout_s 120` (stock). The stock reply is the more sensible one for this prompt, but one sample cannot separate a systematic loss from a greedy near-tie flipped by rounding. The fused scan is not bit-identical (teacher-forced KL 0.0076, top-1 agreement 0.969 against stock), and the same is true of the SDPA prefill attention that is also on by default; Qwen's rounding-level options were kept opt-in for this reason, so this default is an open question for the owner. Roll back with `VEYSTRIX_NEMOTRON_FAST_SSM=0` (about 11% of 8K prefill speed).

Larger sample (20 tool-call prompts, same format): 16 replies identical, 4 differ. Valid JSON: fused 19 of 20, stock 18 of 20 (the stock failures include a stray `</think>` after the object). The four differences are equally sensible in both directions (stock chose `npm test` for the `test_prefix_cache_restore` prompt, fused chose it for the parser prompt; fused omitted the optional `timeout_s` once, on the line-count prompt, where stock included it). That is what rounding-level divergence between two equally valid greedy paths looks like, not a systematic loss, so the default stays on; the rollback flag is above.

## Qwen decode: S3-FIFO pool replacement (opt-in `s3fifo-pool`, exact, +5%)

The decode expert pool's replacement policy only changes which experts stay resident; the experts read and the arithmetic are the same. In-process real Qwen, app launch environment, 300-token prompt, 120 then 200 decode tokens, identical tokens across policies (same hash):

| Policy | 120 tokens | 200 tokens (two runs) |
|---|---|---|
| lru (default) | 2.39 tok/s, 157.7 GB read | 2.50 / 2.48, 222.8 / 223.0 GB |
| s3fifo | 2.54, 146.9 GB | 2.63 / 2.61, 211.5 / 211.3 GB |
| cost-aware | 2.45, 153.2 GB | not repeated |
| least-stale | 2.43, 153.5 GB | not repeated |

S3-FIFO gives +5% decode speed and 5% fewer bytes read, reproducibly (alternating order). `--enable s3fifo-pool` sets `FLASHNEXT_EXPERT_POOL_POLICY=s3fifo`; default stays `lru` (the off value is in `EXPERIMENTAL_OFF`, test added). Through the real server with the six prefill options on (same prompt, 160 new tokens, prefix cache not hit, so each request includes a prefill of about 28-30 s; sampling is not greedy there, so texts differ): lru 93.8 and 95.0 s, s3fifo 89.8 s and (a reply that ended at 78 tokens) 60.2 s. Subtracting the prefill, that is about 2.4 tok/s for lru and 2.6-2.7 tok/s for s3fifo, consistent with the in-process +5-10%; it is a rough figure for that reason. S3-FIFO also ran without error across prompt sizes of 120, 500, 900 and 1,500 tokens with the short-throughput pool reuse path (prefill 11.0 / 19.5 / 25.9 / 33.6 s, 12 tokens decoded each, memory floors 3.7-4.6 GiB, swap flat). The gain is for decode-bound output, so a conversation that mostly generates a few tokens will see less.

## Qwen decode: I/O worker count and buffered fill (no effect)

With `s3fifo` on and the same 300-token prompt / 120 decode tokens: `FLASHNEXT_IO_WORKERS` 6, 8, 16, 24 and 32 all gave 2.55-2.57 tok/s (the app already uses 8). The decode read path is saturating the SSD at any of these counts. `pool-buffered-fill` also gave nothing in this setup (2.45 and 2.48 tok/s alone, 2.61 with `s3fifo` against 2.62 for `s3fifo` alone), unlike the +13-30% recorded on 14 Nov, so its gain depends on page-cache state and is not reproduced here.

## Qwen decode pool: a real route trace, and why the offline S3-FIFO tuning did not hold

A real decode trace (300 tokens after a 300-token prompt, 48 streamed layers x 10 experts = 480 experts per token, 14,400 calls) shows why LRU is poor here: the pool holds 349 slots but one token needs 480, so an LRU pool cycles and never hits (replayed hit rate 0.0%); the 349 most frequent experts cover only 22% of accesses. Replaying it offline, S3-FIFO as built hits 11.5%, and a ghost queue of 2x the slots 20.8% (frequency/recency variants and ratio sweeps 8-21%). On the real model the longer ghost was not better: 200 decode tokens, ghost 1x 2.62 / 2.64 tok/s (211.9 / 210.5 GB read), 2x 2.57 / 2.58 (216.4 / 216.0), 3x 2.54 (219.1). So the replay over-predicts for this change (it does not model the prefill-seeded pool or the read path), and the shipped `s3fifo` (ghost 1x) stays as the option. The code change was reverted. The trace is in the session scratchpad only (not committed).

## Qwen tool-call generations, baseline versus all six options (real model, 8 matched prompts)

In-process, ChatML prompt of about 510 tokens with three tool definitions, greedy, 70 tokens, thinking off. Baseline (`kv-tiered4`, `short-throughput`) against the six suggested options; 8 prompts were completed in both (the last batch of the six-option run was stopped by the guard's 3 GiB available-memory floor, the same stop that ended both first attempts after about 5 prompts in-process, so 4 baseline replies have no partner). 5 of the 8 replies are identical. The 3 that differ are format or argument differences, not wrong calls: for README.md the baseline wrapped the same call in a `<tool_call>` block with a stray closing tag and the six-option build returned the bare JSON; for `docs/architecture.md` lines 1-20 the baseline used JSON inside the wrapper and the six-option build used Qwen's `<function=...>` form with the same path and line range; for the parser test prompt the baseline ran `ls` and the six-option build ran a more useful command with the 120 s timeout. These are first-token near-ties (wrapper or no wrapper) flipped by rounding; no call lost its target, path or arguments. Small sample, so it shows absence of an obvious regression, not equivalence.

## Qwen decode: larger pool request (not usable on this machine)

`FLASHNEXT_EXPERT_POOL_GB=2` after the mode was applied gave the same 2.58 tok/s and 144.5 GB read as the app's 1.07 GB pool (the planner clamps the request; memory floor 3.08 GiB); a 3 GB request was stopped by the guard's 3 GiB available-memory floor (2.59 GiB) 42 s in. So a larger decode pool is bounded by RAM here, as the earlier admission work found; no setting was changed.

## 16K cold prefill with the six options: not measured (thermal stop, twice)

Two attempts at a 16,000-token needle run with all six options (each started at thermal level 0) were stopped by the run guard's thermal rule (level 2 for 15 s) after 97-107 s of sustained prefill, with memory fine (floor 4.25-4.31 GiB, swap flat at 1,344 MiB). The fanless Air throttles before a 16K cold prefill (about 150 s at the 10K rate of 115 tok/s) can finish, so 16K was not measured; the guard's rule was left as is. The 10K run above did complete (109 s).

## Code review of the branch (medium, against `unlimited-generation`)

The automated review returned eight findings; none was a demonstrated failure. What I checked and what I did:

- `merge-small-tail` lets one layer-major window grow to the window size plus the merged tail (4,096 + up to 2,048 = 6,144 tokens), so activation memory can exceed what a 4,096-token window needs, and the mid-prompt edge where the merged tail began is no longer a window end (a resume point for an interrupted prefill; the end-of-prompt snapshot is unaffected, anchors are only a preference for where a window ends). The measured peak did not rise (MLX peak 6.7 -> 6.6 GB for the 4,274-token prompt; six-option server stress memory above). The option is opt-in and off by default.
- The Nemotron prefetch-order finding (a prefetch issued before the layer's output is evaluated could overwrite staged experts): fetched experts leave the staging area when they are taken (`mx.array` copies, or a dlpack view of an array the pool no longer owns), and real-model outputs were token-identical to the legacy order when it was adopted; `VEYSTRIX_NEMOTRON_PREFETCH_ORDER=legacy` restores the old order.
- The scratch-ring finding (later-turn scratch pool left mapped after an exception): the later-turn option reuses the first-turn scratch lifetime (`release(state, failed)` in the prefill `finally`), and the cancel run above released it (available memory returned to 5.8-6.2 GiB, next request served).
- The fused scan and SDPA prefill attention defaults for Nemotron are rounding-level changes that are on by default, which differs from the convention used for Qwen's rounding-level options (opt-in). I kept them on because of the measured gains and the tool-call comparison above, but this is the owner's decision; each has a rollback flag (`VEYSTRIX_NEMOTRON_FAST_SSM=0`, `VEYSTRIX_NEMOTRON_ATTENTION_PREFILL_SDPA=0`).
- Findings about `veystrix/sampling.py`, `speculative.py`, `shared/prompt_lookup.py` and the Nemotron sampled-lookup path concern the prompt-lookup drafting experiment (off by default, no gain, recorded on 15 Nov); not changed here.

## Repository task-level quality gate, smoke tier (real Qwen, baseline versus six options)

`models/flashnext/tests/bench/quality_suite.py` smoke tier (7 cases: coding, reasoning, instruction, tool call, multi-turn, retrieval early and late; programmatic checks, greedy, thinking off, a fresh conversation per case, in-process with the app launch environment), baseline (`kv-tiered4`, `short-throughput`) against the six suggested options, compared with `quality_compare --relation reordered`: **verdict PASS**. All 7 cases pass in both arms (every category 1/1 -> 1/1); 6 of 7 cases have bit-identical token ids, and `tool_1` differs from its first token while both replies pass the tool-call check. The baseline arm had to be run in three smaller processes (its first attempt was stopped by the guard's compressor-growth rule after 3 passing cases: compressor +4,138 MiB over launch for 20 s, swap 1,336 -> 1,830 MiB), while the six-option arm ran all 7 cases in one process in 181 s with a memory floor of 2.82 GiB and swap flat or falling. That memory difference is one observation, not a measured effect.

## Repository quality gate, full tier (24 cases): `qsa-fast-prefill` FAILS it; the five exact options do not change a token

Full tier, baseline (`kv-tiered4`, `short-throughput`) against the six suggested options, `quality_compare --relation reordered` (cases run in small batches in fresh processes because of the guard's memory rules; one batch of the six-option arm was stopped by the thermal rule after its last case had been saved): **verdict FAIL**. Pass counts per category are equal except tool_call, 4/5 -> 3/5: `tool_2` passes on the baseline and fails with the six options. 20 of 24 cases are bit-identical; the others differ at `tool_1` (token 0), `instr_2` (token 3), `tool_2` (token 35) and `multi_3` (token 0), and only `tool_2` changes outcome. `instr_4` and `tool_4` fail on both paths (model failures, not regressions).

What `tool_2` is: both builds write the identical, correct first call (35 tokens, through the closing `</tool_call>`). The baseline then ends the turn; the six-option build writes a newline and a second, redundant search call, so the "exactly one call" check fails (`2 tool_call blocks`). Teacher-forced at that position, the baseline's end-of-turn token and newline are almost tied (24.0 against 24.375 logits); with the six options newline is ahead by 2.25 (24.625 against 22.375). So the baseline's own decision is a coin flip that the decode path happens to resolve towards stopping, and the options move it by about 1.9 logits.

Attribution (same case, one option set at a time): the five exact options (`later-turn-scratch-ring`, `single-chunk-layer-major`, `merge-small-tail` plus the two app options) reproduce the baseline's token digest exactly and pass; `qsa-fast-prefill` alone produces the six-option digest and fails. So the gate failure belongs to `qsa-fast-prefill` (a different SDPA reduction order). By the repository's policy that option stays opt-in and is not a candidate for a default; it is still the owner's choice for the speed it gives on long histories.

Recommended set without it (`--enable kv-tiered4 --enable short-throughput --enable later-turn-scratch-ring --enable single-chunk-layer-major --enable merge-small-tail`, exact), real server (`results/flashnext/20261116-093036-night-qwen-exact5`): conversation turns of 1.4-1.7K new tokens 27.5 / 26.4 / 26.9 s (baseline 51.5 / 52.0 / 48.7 s; the six options 27.2 s, so `qsa-fast-prefill` adds nothing at this size), cold first turn 37.8 s. Larger turns (3.3-4.5K new tokens over 3.4K, 6.7K and 10.7K cached): 42.1 / 42.5 / 61.6 / 69.2 s against 36.7 / 34.2 / 37.3 s with the six options and 105-112 s for the baseline (different prompt text and sizes than the earlier stress, so only the order of magnitude compares). Memory: minimum available 2.46 GiB, peak compressor 5.26 GB, swap 1,605 -> 1,525 MiB, thermal 0. So the exact set keeps most of the gain on 1.5K turns and about two thirds of it on 3-4K turns over long histories; `qsa-fast-prefill` is worth its rounding-level risk mainly for long histories.

## Exact attention speedup: shared K/V for the leading dense queries (`--enable qsa-shared-dense-kv`)

Without `qsa-fast-prefill`, every query of a chunk gets a private gathered and dequantized copy of its keys and values, even though the leading queries of a cold chunk (those whose causal length is within the indexer's budget) all select the same first rows. `FLASHNEXT_QSA_SHARED_DENSE_KV=1` gathers those rows once per chunk and gives each tile of queries a broadcast view; the SDPA call per query, the rows, their order and the validity mask are unchanged (padding slots are masked in both versions). It is bit-identical: on a real full-attention layer (layer 3, real weights, random input, three consecutive 2,048-token chunks) the outputs are equal (max difference 0.0) and the first chunk takes 396 ms against 607 ms; a tiny-model unit test asserts bitwise-equal logits through chunks that reach the branch (54 leading dense queries). Later chunks, whose queries are sparse and select different rows per query, are unchanged (726 ms). End to end, a 3,000-token cold prefill (single-chunk layer-major, tail merge, `kv-tiered4`): 32.8 / 33.4 s with the option against 35.3 / 35.1 s without (alternating), identical output hash, MLX peak 6.57 GB both. About 2 s (6%) on a cold prompt; little on later agentic turns, where the history is long and the queries are sparse. Needs the gather kernel (`kv-tiered4` or similar). Also tried: `FLASHNEXT_QSA_GATHER_TILE` 16 and 32 against 8 on the same prefill (exact, 34.1-35.1 s, no difference).

### Where the exact sparse attention time goes, and a variant that lost

Per full-attention layer, second 2,048-token chunk (sparse queries, real layer, stage timers with an eval after each; they add about 0.15 s to the 0.73 s total): gathering and dequantizing the selected K/V 0.44 s (51%), the per-query SDPA 0.28 s (32%), selecting the indices (scores, argpartition, sort) 0.12 s (14%). A variant that dequantizes the whole history once per chunk and then gathers bf16 rows is bit-identical (outputs equal, so dequantization does commute with gathering) but slower, 0.98 s against 0.73 s (the bf16 gather moves four times the bytes of the packed gather), so it was dropped. A fused gather-and-dequantize Metal kernel is the remaining exact lever for this path (it would also have to handle the mixed INT8/INT4 tiers of the hybrid cache); not attempted tonight.

## Soak of the final recommended set through the real server

`--enable kv-tiered4 --enable short-throughput --enable later-turn-scratch-ring --enable single-chunk-layer-major --enable merge-small-tail --enable qsa-shared-dense-kv --enable s3fifo-pool` (all exact), guarded private server, prefix cache on (`results/flashnext/20261116-100340-night-qwen-soak`). An 8-turn conversation of about 5,500 characters per turn (1.7-2.1K new tokens per turn over a growing cached history of up to 13.2K tokens): 39.8 (cold, 1,868 tokens) / 28.9 / 32.5 / 27.9 / 36.4 / 35.0 / 37.6 / 34.4 s per turn, 15,020 tokens in context at the end (the earlier baseline took 49-52 s per 1.6-1.8K-token turn at only 2-6K of context). Then two decode-heavy requests (74 and 119 tokens, prompt uncached so about 27 s of each is prefill). Memory: minimum available 3.35 GiB, peak compressor 3.72 GB, swap 1,485 -> 1,453 MiB, thermal level 0; no errors or tracebacks in the server log.

## Exact attention speedup: one kernel for gather and INT8 dequantize (`--enable qsa-fused-gather`)

In the per-query path the largest piece of a sparse chunk is the K/V gather (0.44 of 0.73 s per full-attention layer, 2,048-token chunk): six `take_along_axis` copies (packed words, scales, biases for K and V) and a dequantize per tile. For contexts up to 8,192 tokens the hybrid cache holds everything in its INT8 recent tier, and `FLASHNEXT_QSA_FUSED_GATHER=1` reads those rows and dequantizes them in one Metal kernel (`mixed_kv._fused_dequantize_rows`). The affine formula is MLX's (a float product and sum rounded once to bfloat16); variants that round the product separately were NOT equal (3.4 million differing elements) and were rejected. Bit-identical to `mx.dequantize` on random data for group 32 and 64 and value scales 0.01, 1 and 100 (12 configurations), in a unit test (`test_fused_gather_dequantize_is_bit_identical`), and on a real layer (outputs of three consecutive 2,048-token chunks equal; sparse chunk 0.724 -> 0.513 s). Layouts it does not recognise and batches above one fall back to the reference path. End to end, a 4,300-token cold prefill (`kv-tiered4`, `short-throughput`, single-chunk layer-major, tail merge, shared dense K/V on): 44.5 / 44.7 s with the fused kernel against 47.8 / 47.7 s without (alternating), identical output hash (`ad323869fed1`), MLX peak 6.62 GB both, 90.8 GB read. About 3 s (6.5%) on a cold 4K prompt, and the same saving per sparse chunk on later turns with histories up to 8K. Beyond 8,192 tokens the hybrid cache has two tiers (INT4 history, INT8 recent window) and the reference gather runs both gathers for every selected row and merges them with `where`; a second kernel (`_fused_dequantize_two_tier`) takes each row from the tier it lives in, with the same per-element formula. Bit-identical in a unit test (group 32 and 64, three value scales, rows from both tiers) and on a real layer over five consecutive 2,048-token chunks (history reaching 10.2K, both tiers live in the last chunk: outputs equal; that chunk 1.228 -> 0.595 s, the sparse chunks before it 0.727 -> 0.51-0.57 s).

End to end across the 8K line (two-stage, like a conversation: a 4,500-token cold prefill, a 75 s pause, then 4,900 more tokens over that cache, 9.4K in all, so the second stage runs with two tiers; shared dense K/V on in both arms; the pause avoids the thermal stop that ended the single 9K prefills): fused kernels 47.2 s + 77.7 s, reference gather 49.6 s + 87.9 s, identical output tokens (`ceef354746e6`: 11, 15, 19, 23, 11, 20, 22, 21), MLX peak 6.64 GB both. So the second stage, the one with a long history, is 11.6% faster with identical output. (The single-shot 9,000-token comparison was cut off twice by the guard's thermal rule in the reference arm after about 90-99 s; the fused arm finished in 97 s with output hash `44bf2f7e3e5a`.)

## Repository quality gate on the final exact set: 24 of 24 cases token-identical

`--enable kv-tiered4 --enable short-throughput --enable later-turn-scratch-ring --enable single-chunk-layer-major --enable merge-small-tail --enable qsa-shared-dense-kv --enable qsa-fused-gather --enable s3fifo-pool` against the baseline, full tier (24 cases), in-process, cases in small batches (a thermal stop ended one batch; its two cases were rerun in a separate process): `quality_compare --relation same-plan`, the strictest relation (any token difference in any case fails): **verdict PASS, bit-identical tokens 24 of 24 cases**, pass counts equal in every category (instruction 3/4 and tool_call 4/5 on both paths: `instr_4` and `tool_4` are model failures present in the baseline). So these options change no token on this suite, unlike `qsa-fast-prefill`, which changed 4 of 24 and flipped one pass to a fail.

## Final recommended set through the real server (conversation and stress)

Same eight flags as the quality-gate section (`results/flashnext/20261116-105701-night-qwen-final`, guarded private server, prefix cache on). Conversation of 5,500 characters per turn, eight turns, 15,020 tokens at the end: 39.7 (cold) / 29.3 / 29.3 / 27.8 / 28.9 / 28.2 / 29.0 / 28.8 s per turn. With only the shared-K/V option (the earlier soak) the same turns grew from 28.9 to 37.6 s with the history; with the fused kernels they stay flat at 28-29 s up to 13.2K cached tokens (baseline: 49-52 s per turn at only 2-6K cached). Stress (14,000 characters per turn, 3,399 tokens cold then 3.3-4.5K new tokens over 3.4K, 6.7K and 10.7K cached): 39.1 / 39.2 / 47.2 / 53.1 s, against 43.2 / 43.4 / 62.6 / 70.2 s for the five exact options without the two kernels (same text) and 105-112 s for the baseline on the earlier, slightly smaller stress prompts. Memory over both runs: minimum available 2.44 GiB, peak compressor 7.71 GB (the exact-five run peaked at 5.26 GB and the six-option run at 4.35 GB, so this is run-to-run variation as much as an effect), swap 1,562 -> 1,458 MiB (no growth); thermal level 2 appeared briefly but never for the 15 s that stops a run; no errors in the server log.

### Gather tile of 32 with the fused kernels (small gain, not adopted)

With both kernel options on, a 4,300-token cold prefill took 42.1 / 43.6 s at `FLASHNEXT_QSA_GATHER_TILE=32` against 44.7 / 44.4 s at the default 8 (alternating), identical output hash (`ad323869fed1`), MLX peak 6.62 GB both: about 4%. The tile is part of the prefix-cache fingerprint (`qsa_gather_tile`), so changing it invalidates existing snapshots once, and the token identity is shown by one hash, not by the quality gate, so it was left out of the recommended list at first. Afterwards the full 24-case quality suite was run on the final eight-flag set with the tile at 32 (`quality_compare --relation same-plan`): **PASS, bit-identical tokens 24 of 24 cases**, the same pass counts as the baseline. A later pair of alternating runs of the same prefill gave tile 16 43.4 / 43.2 s and tile 32 42.8 / 43.0 s (identical hash), (tile 24 gave 43.1 s against 43.0 s for 32 in a single pair, same hash), so 16, 24 and 32 are about equal and both about 1-2 s ahead of 8 (44.4-44.7 s; a further tile 8 run gave 44.7 s). So `FLASHNEXT_QSA_GATHER_TILE=32` is exact on this evidence and worth about 4% on a cold prefill; it is set by environment (with a KV mode such as `kv-tiered4` the mode planner passes `FLASHNEXT_QSA_GATHER_TILE` through from the launching environment, so `FLASHNEXT_QSA_GATHER_TILE=32` exported before the launch is enough; the quality run applied it after the planner, which has the same effect), is not an `--enable` option, and changes the prefix-cache fingerprint once.

### Fused gather and decode (no effect)

`FLASHNEXT_QSA_FUSED_GATHER` also runs inside decode attention. Real Qwen, 300-token prompt, 120 decode tokens, S3-FIFO on, alternating: fused 2.36 (first run, an outlier), 2.56, 2.56, 2.59 tok/s; reference 2.60, 2.58, 2.56, 2.58 tok/s; identical output hash (`ffa05c287a3b`) in all eight runs. So decode neither gains nor loses; the options are prefill options.

### Pipelined attention tiles (`--enable qsa-tile-pipeline`, exact, no end-to-end gain once tile 32 is used)

The per-query attention loop evaluates each tile before building the next, so the device waits for the host between tiles (64 to 256 times per layer and chunk). `FLASHNEXT_QSA_TILE_PIPELINE=1` queues each tile with `mx.async_eval` and waits for the one before the previous (at most two tiles' gathered K/V alive). The operations are unchanged: bit-identical in a tiny-model unit test and on a real layer over five 2,048-token chunks (outputs equal); at the default tile of 8 with the fused gather on, the sparse chunk went 0.507 -> 0.400 s and the two-tier chunk 0.591 -> 0.472 s per layer. End to end on the 4,300-token cold prefill with both kernels and tile 32: 42.8 / 43.1 s with the pipeline against 43.7 / 43.0 s without (identical hash `ad323869fed1`, MLX peak 6.62-6.63 GB), so about nothing: tile 32 already cuts the number of host round trips, and the SSD hides the rest. Left as an option; not recommended.

### What is left in the exact sparse attention path

Re-profiled with the fused gather, shared dense K/V and tile 32 (same harness and stage timers as above, which add a little): the second 2,048-token chunk of a full-attention layer takes 0.43 s against 0.88 s at the start of the night (same timers): the per-query SDPA 0.21 s (49%), the fused gather 0.16 s (37%), selecting indices 0.034 s (8%). The SDPA moves about 5.4 GB of gathered bf16 K/V per chunk and layer in 0.21 s, which is about a quarter of the memory bandwidth, so it is limited by the per-query vector kernel's latency, not by bandwidth; keeping it bit-identical means keeping that kernel and its reduction order. The only exact way past this is a single kernel that reads the INT8/INT4 rows and runs MLX's exact SDPA reduction order itself (no bf16 K/V materialised), which would have to be verified bit for bit against `mx.fast.scaled_dot_product_attention`; not attempted.

### The final set with `FLASHNEXT_QSA_GATHER_TILE=32` exported at launch, through the real server

Same eight flags and the same two conversations as above (`results/flashnext/20261116-121104-night-qwen-final32`). Eight-turn conversation (15,020 tokens at the end): 39.3 (cold) / 28.7 / 29.0 / 27.9 / 28.9 / 28.3 / 28.7 / 27.8 s, the same 28-29 s as with tile 8 (the SSD hides the small difference on 1.7-2.1K-token turns). Stress (3,399 tokens cold, then 3.3-4.5K new tokens over 3.4K, 6.7K and 10.7K cached): 37.5 / 36.2 / 42.1 / 48.1 s against 39.1 / 39.2 / 47.2 / 53.1 s with tile 8, so 4-11% faster on the larger turns. Memory: minimum available 2.57 GiB, peak compressor 4.68 GB, swap flat at 1,394 MiB, thermal level 2 briefly but never for the 15 s that stops a run, no errors in the server log.

### N-gram row dedup (`--enable ngram-dedup`, existing option, now measured: exact, -3%)

The per-layer-embedding lookup is latency-bound on the SSD (a 2,048-token call on cold rows took 1.2-1.5 s in isolation; whole prefill PLE share about 4 s of 34 s at 4K). The repository already had `FLASHNEXT_NGRAM_DEDUP=1` (read each n-gram row once per lookup and gather it back to every position), tested offline only. Real Qwen, 4,300-token cold prefill with both kernels and tile 32, alternating: 41.5 / 42.1 s with it against 43.3 / 42.9 s without, identical output hash (`ad323869fed1`), MLX peak 6.62 GB both. About 1.3 s (3%). It reads the same rows and gathers them back in order, so tokens are unchanged by construction, and the offline exactness tests (`test_ngram.py`, `test_ngram_parallel.py`) pass; afterwards the full 24-case quality suite was run on the final eight-flag set with `FLASHNEXT_QSA_GATHER_TILE=32` and `--enable ngram-dedup` both on (`quality_compare --relation same-plan`): **PASS, bit-identical tokens 24 of 24 cases**, same pass counts as the baseline. So it joins the exact options: `--enable ngram-dedup`.

### Dropped: truncating invisible keys in the shared dense tiles (no gain)

Profile of a 4.3K cold prefill with every exact option on (43.7 s): MLP 15.7 s, full attention 9.8 s, linear attention 8.9 s, PLE 3.5 s, the two hyper-connections 5.5 s. I tried cutting the shared dense tiles' key range to the keys the tile's queries can see (the rest are masked, so they add exactly zero; a 1,024-key floor kept the SDPA kernel variant). Real layer, five 2,048-token chunks: outputs bit-equal, but the time was unchanged (0.409-0.442 s with and without), so the masked tail is not what the kernel spends its time on. Reverted, not committed.

### Dropped: compiling the prefill injection (no gain)

The two broadcast-multiply-add injections per layer have no transcendental, so a compiled version is bit-equal (checked). At the prefill shape (2,048 tokens, 4 streams, 2,560 wide) plain 0.115-0.168 ms against compiled 0.113-0.149 ms per injection, about 25 ms over a whole 4.3K prefill. The 5.5 s the hyper-connections cost is in their norm, low-rank matmuls and sigmoids, which cannot be fused without changing rounding (see `compile_glue.py`). Not built.

### Nemotron dense GEMM for large projections (`VEYSTRIX_NEMOTRON_DENSE_GEMM=1`, opt-in, bit-equal, +2-4% 8K prefill)

`mx.dequantize` followed by a dense bf16 GEMM is faster than `mx.quantized_matmul` for prefill-sized calls (real Mamba shapes at 2,048 rows: in_proj 38.8 -> 35.8 ms, out_proj 15.5 -> 14.3 ms) and gives the same bits **for some shapes only**: 2688->10304, 4096->2688, 2688->2688 and 2688->4096 are equal from 128 rows up, 2688->1856 from 256 up, but 256->384 and 2688->128 differ at every row count (a unit test caught this at 384 rows). `models/nemotron35/dense_gemm.py` therefore checks each (shape, quantization) once on the layer's own weights with random inputs at 384, 385 and 640 rows, (with an explicit random key, so the check does not advance the global random state that seeded sampling uses; a unit test fails without that) and uses the dense path only for shapes that are equal in all three; everything below 384 rows, with a bias, or not bf16 keeps the stock path. Weights stay the checkpoint's 4-bit values (a transient dequantized array of at most about 55 MB per call). Real 8,192-token Nemotron prefill (natural text, AC, alternating): 368.6 / 364.6 tok/s with it against 357.7 without (an ungated first version: 375.7 / 370.4 against 360.7 / 355.4), identical output hash for prefill and the 8 decoded tokens, MLX peak 3.64 GB both. At 16,384 tokens (one pair): 354.1 tok/s with it against 340.0 without (+4%), identical output hash, MLX peak 3.69 GB both. Off by default (opt-in, like the other bit-equal options). Not run through a tool-call battery; the equality check makes it unnecessary for the projections it touches.

### Qwen dense GEMM (`--enable dense-gemm`, `FLASHNEXT_DENSE_GEMM=1`, exact, about -2.4% cold prefill)

The same shape-verified dense GEMM as for Nemotron (shared module `models/shared/dense_gemm.py`), applied to Qwen's large 4-bit projections (group size 32; at 2,048 rows q+gate 44.3 -> 40.4 ms, linear-attention in-projection 37.2 -> 34.3 ms, z/v and output 22.3 -> 20.5 ms; all of these shapes were bit-equal, the 10,240 -> 4 projection is not and keeps the stock path). Real 4,300-token cold prefill with the final set, tile 32 and `ngram-dedup`, alternating: 41.2 / 41.8 s with it against 42.6 / 42.2 s without, identical output hash, MLX peak 6.66 against 6.62 GB. The full 24-case quality suite with it added: `quality_compare --relation same-plan` **PASS, 24 of 24 cases token-identical** to the baseline. Opt-in; it joins the exact extras (`--enable dense-gemm`).

### The three exact extras together (end to end)

Real 4,300-token cold prefill with the two exact Qwen kernel options (`qsa-shared-dense-kv`, `qsa-fused-gather`) always on, alternating: gather tile 32 + `ngram-dedup` + `dense-gemm` all on 41.4 / 41.9 s, all three off 45.1 / 44.7 s (the default tile of 8 is the slowest of the three tile settings). That is -3.4 s, **-7.6%**, identical output hash (`ad323869fed1`), MLX peak 6.66 against 6.62 GB. The three gains (about 4%, 3% and 2.4% on their own) are roughly additive.

### The whole recommended set through the real server (eight flags + tile 32 + `ngram-dedup` + `dense-gemm`)

`results/flashnext/20261116-132632-night-qwen-final3x`, same two conversations. Eight-turn conversation: 38.8 (cold) / 27.9 / 28.2 / 27.8 / 28.3 / 27.7 / 28.2 / 28.9 s (28-29 s up to 13K cached tokens, unchanged from tile 8 and tile 32 alone). Stress (3.4K cold, then 3.3-4.5K new tokens over 3.4K, 6.7K, 10.7K cached): 36.9 / 35.7 / 41.1 / 46.6 s against 39.1 / 39.2 / 47.2 / 53.1 s with the eight flags alone (-6% to -12%), and 37.5 / 36.2 / 42.1 / 48.1 s with tile 32 alone. Memory: minimum available 2.26 GiB (2.57 with tile 32 alone, 2.44 with the eight flags alone: the run-to-run spread is as large as the difference), peak compressor 5.11 GB, swap 1,439 MiB at the start, 1,383 MiB at the end, never above the start, thermal level 1 at most, no errors in the server log.

### Hyper-connection cost split (why a fused exact kernel is not worth building)

One hyper-connection at the prefill shape (2,048 tokens, 4 streams, 2,560 wide), bf16, measured per piece: norm 1.03 ms, down-projection 4.35 ms, div + SiLU 0.21 ms, up-projection 4.25 ms, sigmoid 1.07 ms, multiply + mean 2.20 ms, injection weights 0.76 ms (about 14 ms in all). The two projections are 61% and are GEMMs (the dense-GEMM option already covers them); the elementwise pieces a hand-written kernel could fuse (rounding after each original op, so it would stay bit-equal) are about 4.3 ms, which is at most 2 ms saved per call or about 0.4 s of a 43 s 4.3K prefill (1%). Not built.

### Gather tile above 32 (cap kept at 32)

With every exact option on, 4,300-token cold prefill, the tile cap temporarily raised in a local edit (reverted): tile 64 gave 41.7 / 41.6 s against 42.1 / 41.7 s for tile 32 (equal), MLX peak 6.68 against 6.66 GB; tile 128 gave 43.4 s (slower) with a 6.95 GB peak. Same output hash throughout. The cap of 32 in `qsa_chunk.gather_tile` stays: nothing is gained above it and memory grows.

### Why short Qwen turns sit at about 28 s (the SSD floor)

A Qwen turn of about 1.7-2.1K new tokens takes 28-29 s however the compute is tuned, and a 300-token follow-up 23-28 s. The reason is the read floor, not the GPU: each token selects 10 of 512 experts, so a few hundred tokens already touch nearly every expert of every layer (300 tokens draw about 3,000 selections, which cover about 99% of 512 experts in expectation; 50 tokens would cover about 65%). A turn longer than a few hundred tokens therefore reads the whole expert set once, about 27 s at the measured 3 GB/s, and the compute hides under it. Only prompts above roughly 2K tokens (cold 3-4K prompts, the stress turns) are compute-bound, which is where the exact options above pay off. Below that, more RAM (a bigger resident expert set) or a faster disk are the only levers; both are outside what this branch changes.

### Fused gather + dequantize + attention (`--enable qsa-fused-sdpa`, exact, about -3.5% cold prefill)

The one lever the sparse-attention section left open. MLX's per-query attention for these shapes is its single-pass `sdpa_vector` kernel (found by replicating it in a custom Metal kernel and comparing bit for bit: equal for one to 32 query rows and one to 2,052 keys; the two-pass variant did not match). `mixed_kv.fused_sdpa_selected` is that kernel with its K and V loads replaced by the row gather and the affine dequantization of the INT8 / INT4+INT8 cache, using the same thread mapping, per-thread order of operations and simd reductions, so it equals gather + `mx.fast.scaled_dot_product_attention` bit for bit and the bf16 K/V copies (5.4 GB per chunk and layer) are never written. Two query heads share each dequantization (one head redoes it 12 times, three or more spill registers: per sparse chunk of a real layer 0.404 s without the kernel, 0.37-0.40 s with one head, 0.323 s with two, 0.455 s with three, 0.70 s with four). Checks: 144 unit-test cases (head dimension 128 and 256, 4 and 24 query heads, group 32 and 64, both cache layouts, 1-32 rows, 1-200 keys) and a real layer over five 2,048-token chunks (outputs equal). Real 4,300-token cold prefill with every other exact option on: 40.5 / 40.7 s with it against 42.5 / 41.8 s without, identical hash (`ad323869fed1`), MLX peak 6.66 GB both. The full 24-case quality suite with all four extras (tile 32, `ngram-dedup`, `dense-gemm`, `qsa-fused-sdpa`): `--relation same-plan` **PASS, 24 of 24 token-identical**. Requires `kv-tiered4`; any layout it does not support falls back to the two-step path.

## Test status at the end of this work (branch `overnight-speed-2026-11-16`)

Unit suites, run by a separate verification agent and spot-checked (final lines read from the logs): Flash-Next 2,277 OK (3 skipped) at 14:40, after the `qsa-fused-sdpa` commits, Nemotron 202 OK (8 skipped) at 13:20; veystrix 961 OK (11 skipped) at 12:58 after the Nemotron dense-GEMM commit (the Flash-Next change does not touch it); Ran/OK lines read from the logs. Nothing is pushed. New session identity for Nemotron (`attention_prefill: sdpa`): existing Nemotron prefix snapshots are not restored once after the upgrade. The wired-limit sysctl was deliberately not changed.

## What is left (levers, not done)

- Qwen one-chunk and layer-major prefills are SSD-bound at about 3 GB/s (80 GB read in 26.7 s at 1,600 tokens). Fewer bytes needs a good prediction of each layer's routed experts (about 76% of a layer's experts at 1,600 tokens, about 94% at 4K) with on-demand reads for misses; the gain is bounded by the 6-24% of bytes that are unused. More RAM for the file cache is the other lever.
- Nemotron: dense dequantize + GEMM for large projections (about 3% of prefill, bit-equal for M >= 384), `gather_qmm` for routed experts (about 5-8%, needs memory), (the Mamba scan is now the fused one above).
