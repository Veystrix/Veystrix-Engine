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

## Test status at the end of this work (branch `overnight-speed-2026-11-16`)

Unit suites, run by a separate verification agent and spot-checked (final lines read from the logs): Nemotron 193 tests OK (8 skipped), veystrix 961 OK (11 skipped), Flash-Next 2,257 OK (3 skipped). Nothing is pushed. New session identity for Nemotron (`attention_prefill: sdpa`): existing Nemotron prefix snapshots are not restored once after the upgrade. The wired-limit sysctl was deliberately not changed.

## What is left (levers, not done)

- Qwen one-chunk and layer-major prefills are SSD-bound at about 3 GB/s (80 GB read in 26.7 s at 1,600 tokens). Fewer bytes needs a good prediction of each layer's routed experts (about 76% of a layer's experts at 1,600 tokens, about 94% at 4K) with on-demand reads for misses; the gain is bounded by the 6-24% of bytes that are unused. More RAM for the file cache is the other lever.
- Nemotron: dense dequantize + GEMM for large projections (about 3% of prefill, bit-equal for M >= 384), `gather_qmm` for routed experts (about 5-8%, needs memory), (the Mamba scan is now the fused one above).
