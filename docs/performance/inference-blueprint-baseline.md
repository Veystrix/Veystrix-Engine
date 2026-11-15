# Inference blueprint baseline

The local checkpoint headers establish the formats that future packing, KV,
and long-context work must support.

| Installed checkpoint | Quantization | Layer order | Declared context |
| --- | --- | --- | ---: |
| TensorFold Flash-Next MLX 4-bit MTP | Affine Q4, group 32 | 36 linear attention, 12 full attention | 262,144 |
| mlx-community Nemotron 3.5 Lightning MLX 4-bit | Affine Q4, group 64 | 23 Mamba, 23 MoE, 6 attention, verified in order | 262,144 |

Flash-Next's checkpoint already contains 128 quantized n-gram table shards
totalling 32,000,153,600 bytes. Its text configuration declares a 20 million
base n-gram vocabulary. Nemotron's local conversion declares `rope_theta` of
10,000 and no RoPE scaling metadata. A million-token run still requires
validation against the installed checkpoint and attention implementation.

```sh
.venv/bin/python -m models.flashnext.preflight "$QWEN_CHECKPOINT" --headers --identity --json
.venv/bin/python -m models.nemotron35.check "$NEMOTRON_CHECKPOINT"
```

## Full expert packs

Both checkpoints have now been packed for every routed layer with
`models.shared.pack_experts`. The measured layout matches the blueprint's
model-derived accounting.

| Model | Layers | Experts/layer | Pack stride | Full pack | Build wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nemotron 3.5 | 23 | 128 | 5,636,096 B | 16,592,732,160 B (15.45 GiB) | 20.46 s |
| Flash-Next | 48 | 512 | 3,080,192 B | 75,698,864,128 B (70.50 GiB) | about 2.5 min |

The pack size is 64 KiB-aligned and includes the versioned header and padding.
Peak RSS while packing was about 14 MiB (Nemotron) / 16 MiB (Flash-Next), so
the build is a streaming copy, not a RAM-resident conversion. Disk free fell
from 136 GiB to 50 GiB; UCH-05's dynamic cache watermark is now wired to the
commit path for that reason. The packs were deleted after the comparison
(50 GiB -> 155 GiB free); rebuild with the commands above.

Guarded one-round pairs on these full packs (fresh server and private prefix
directory per arm, 2 GB pool, greedy, first-prefill-logit digest gate):

| Model | Original shards | Full pack | Physical reads | Token / logit parity |
| --- | ---: | ---: | --- | --- |
| Nemotron 3.5 | 3.30 tok/s | 3.79 tok/s | 28.09 → 27.88 GB | match |
| Flash-Next | 2.21 tok/s | 2.59 tok/s | 72.29 → 70.01 GB | match |

Both pairs were repeated with three alternating rounds: Nemotron
3.55/3.54/3.50 vs 4.00/3.95/3.97 tok/s (paired +0.45 tok/s, band 0.025,
resolved; reads unresolved; footprint +47 MB), and Flash-Next
2.26/2.32/2.34 vs 2.58/2.57/2.58 tok/s (paired +0.25 tok/s, band 0.04,
resolved; physical reads −1.53 GB, band 0.25 GB, resolved). Every arm's
greedy token digest and first-prefill logit digest matched. The scope is one
~570-token coding prompt and 24 output tokens.

The one-round pairs are also kept. The source records are
[here](../../results/nemotron35/20261113-141728-performance-v2-expert-pack-manual/performance-arms.jsonl)
and [here](../../results/flashnext/20261113-141837-performance-v2-expert-pack-manual/performance-arms.jsonl).
Three-round records: [Nemotron](../../results/nemotron35/20261113-143218-performance-v2-expert-pack-manual/performance-arms.jsonl),
[Flash-Next](../../results/flashnext/20261113-143509-performance-v2-expert-pack-manual/performance-arms.jsonl).
They do not establish a sustained speedup or long-prompt behavior.

## Decode diagnostics

`FLASHNEXT_DECODE_PROFILE_DIR` and `FLASHNEXT_PROFILE_IO=1` enable the
Flash-Next route trace and I/O timers. `VEYSTRIX_NEMOTRON_DECODE_PROFILE_DIR`
records Nemotron's routes, router-index synchronization and copy, expert
dispatch and wait, and pool counter deltas. These host timings include
dependency stalls and can overlap background reads. Use a Metal System Trace
to isolate GPU time. Nemotron counts layer-major speculative verifier calls
separately from the token-major trace.

`VEYSTRIX_NEMOTRON_PREFILL_CACHE_LIMIT_MB` sets an optional MLX allocator
cache limit during Nemotron prefill. It restores the prior limit on exit
and respects tighter process-wide limits. Flash-Next has the equivalent
`FLASHNEXT_PREFILL_CACHE_LIMIT_MB` control.

`VEYSTRIX_BENCH_LOGIT_DIGEST=1` records the first complete greedy prefill logit
row as a SHA-256 digest over little-endian FP32 values in the flight log.
The paired Least-Stale benchmark uses this digest and the emitted token IDs
as independent exactness gates. It runs Flash-Next with native MTP off so the
ordinary prompt sampler supplies the reference row.

## Expert eviction

`FLASHNEXT_EXPERT_POOL_POLICY=least-stale` and
`VEYSTRIX_NEMOTRON_EXPERT_POOL_POLICY=least-stale` select the experimental
policy. Least-Stale prefers entries from older forward passes, then layers
already visited in the current pass. It protects every resident expert
requested by the active batch. Speculative reads stay in the runtime's
existing staging path until the router requests them. Both runtimes default
to LRU. [SpecMD §4.2](https://arxiv.org/html/2602.03921v1#S4.SS2)

| Historical Flash-Next route replay | LRU | S3-FIFO | Least-Stale |
| --- | ---: | ---: | ---: |
| 1 GB | 0.0% | 32.5% | 24.6% |
| 2 GB | 46.6% | 47.3% | 42.9% |
| 4 GB | 63.6% | 63.0% | 46.3% |

The historical trace varies from two to nine routed experts per layer. The
rates describe simulated decode residency for that trace. At 2 GB,
Least-Stale cut same-pass collision misses from 14,984 to 805 while lowering
the overall hit rate. The replay accepts either runtime's decode-profile
JSON with an explicit `--record-bytes` for the checkpoint.

```sh
.venv/bin/python -m models.flashnext.tests.bench.bench_pool_policy_trace \
  results/flashnext/20260922-221213-route-trace-manual/route-trace.json.gz \
  --gb 1 2 4 --policies lru s3fifo least-stale --collisions

.venv/bin/python -m models.flashnext.tests.bench.bench_server_short_paired \
  --runtime nemotron35 --checkpoint "$NEMOTRON_CHECKPOINT" \
  --performance-v2 --least-stale
```

| Short diagnostic | Prompt / output tokens | LRU decode | Least-Stale decode |
| --- | ---: | ---: | ---: |
| Flash-Next | 633 / 24 | 2.35 tok/s | 2.43 tok/s |
| Nemotron | 576 / 24 | 3.82 tok/s | 3.96 tok/s |

Each pair used a fresh server and private prefix directory per arm, with the
OS file cache retained. The first prefill logit digest and every emitted
greedy token matched across policies. System swap stayed flat within each of
these four arms. The earlier Nemotron diagnostic recorded swap growth in
both arms. These short runs establish parity for the selected prompts.
Sustained speed, long-context memory use, and concurrent admission still
require qualification. [Flash-Next record](../../results/flashnext/20261113-131916-performance-v2-least-stale-manual/performance-arms.jsonl)
[Nemotron record](../../results/nemotron35/20261113-132543-performance-v2-least-stale-manual/performance-arms.jsonl)
[Earlier Nemotron record](../../results/nemotron35/20261113-131444-performance-v2-least-stale-manual/performance-arms.jsonl)

## Expert-pool residency sweep

Three alternating rounds with a 5 GiB swap guard and all research flags off:

| Model | 2 GB pool | 6 GB pool | Reads | Footprint |
| --- | ---: | ---: | --- | --- |
| Nemotron 3.5 | 3.89 tok/s median | 3.65 tok/s median | 25.62 → 22.22 GB | 4.38 → 8.69 GiB |

Token and first-logit digests matched. The 13% read reduction did not pay for
the extra cache overhead. Qwen's 6 GB arm was refused before readiness by the
validated `POOL_CAP_GB = 2` rule (the recorded 6 GB arms neared the 3 GiB
compressor guard), and the failure is preserved in
`results/flashnext/20261113-145202-performance-v2-pool-compare-manual/`. The measured result agrees with the memory
arithmetic: inside the blueprint's 11 GiB working set Nemotron can hold at
most ~56% of its 15.39 GiB expert pool (85% would need 15.4 GiB before core
and KV), and Qwen at 262K/Q8 can hold only ~6.8% of 63.28 GiB. The modeled
decode targets that assume 85–95% / 75% residency are unreachable on 16 GB;
the practical ceiling is discussed in the handoff.

## Speculation on the live server

Prompt-lookup speculation is the only large decode win found on this machine.
On a 256-token code-copy task, Nemotron with
`VEYSTRIX_NEMOTRON_PROMPT_LOOKUP=1` decoded 5.22 tok/s against 3.80 without it
(three rounds, paired +1.45 tok/s, band 0.035, resolved) and read 42.8 GB
instead of 129.3 GB, with identical greedy tokens. On the open-ended
repository review it is slower: 3.66 versus 3.87 tok/s (paired −0.23, band
0.04, resolved). It stays opt-in and is recommended for repetitive edit/test
loops, not review. The same flag on Qwen is a regression (1.64 → 1.30 tok/s,
302 → 329 GB read), because its 2 GB pool makes the verification union
stream new experts, so it stays off there.
[Copy record](../../results/nemotron35/20261113-151651-performance-v2-flag-veystrix-nemotron-prompt-lookup-manual/performance-arms.jsonl)
[Review record](../../results/nemotron35/20261113-152802-performance-v2-code-review-flag-veystrix-nemotron-prompt-lookup-manual/performance-arms.jsonl)

## Read scheduling, lookahead and recurrent-scan gates

`models/shared/edf_io.py` adds an opt-in earliest-deadline-first read pool
(`VEYSTRIX_NEMOTRON_EDF_IO=1`). In a one-round Nemotron pair the candidate
submitted 10,390 reads with 0 failures and matched the baseline token and
first-logit digests (3.35 vs 3.37 tok/s; 27.58 → 27.11 GB physical reads).
`models/shared/route_history.py` adds a two-layer-out advisory prefetch
(`VEYSTRIX_NEMOTRON_LOOKAHEAD_LAYERS=1`); the candidate issued 36 history
reads and consumed 25, matching both digests (3.48 vs 3.40 tok/s). Neither
short pair resolves a throughput effect. Offline replay of the recorded
8-token coding decode (184 layer observations, 23 MoE layers) gave a
12-deep history recall of 35.96% against 28.17% for last-token persistence
and 4.69% for random selection; at depth 6 history and last-token tie at
28.17%. That is why the runtime default lookahead depth is 12. The sample is
eight tokens, so it establishes signal, not a stable rate.
[Recall record](../../results/nemotron35/20261113-143036-lookahead-recall-manual/lookahead-recall.jsonl)

The block-parallel Mamba prefill (`VEYSTRIX_NEMOTRON_PARALLEL_MAMBA=1`) was
compared on the 6.5K-token repository-review prompt. It failed the exactness
gate: the stock arm emitted `e42ebe8781e9…`, the parallel arm emitted
`ce5d9503a238…` (3.66 vs 3.91 tok/s). The record is marked `failed`. The path
stays opt-in and the blueprint's PRF-01 bit-match requirement is not met.
[Failed record](../../results/nemotron35/20261113-142300-performance-v2-parallel-mamba-manual/performance-arms.jsonl)

Qwen's installed `mlx_lm.models.gated_delta` already runs the GDN recurrence
as one Metal kernel per call (`gated_delta_kernel`), so the Python per-token
loop the blueprint targets does not exist in the current runtime. Only the
chunk-parallel kernel (KER-08) remains open.

## Remaining blueprint work

The full packs exist but their comparison is one-round diagnostic evidence.
The parallel Mamba path needs a correctness fix. Paged KV, artifact caching
and request-state ownership are implemented as tested modules; the first two
still need a live runtime that owns more than one conversation state, and the
third is the prerequisite for continuous batching.

The next baseline measurements are the 8K per-layer Metal trace, a full-width
coding route capture, and a multi-client concurrency run on the 16 GiB target.
The 262K/1M context ladder and sustained thermal tests still require live
qualification; the server plans and advertises at most the checkpoint's
declared 262,144 positions and only a research override can go past it.
