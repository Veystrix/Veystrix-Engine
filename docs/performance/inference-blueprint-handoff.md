# Veystrix inference blueprint handoff

> **Historical record.** Work moved to `main` on 2026-11-16; the branch names and paths below describe the state at the time. See the [documentation index](../README.md) for current status.

The work is in the current checkout on `unlimited-generation`. The original
research blueprint is the user attachment; [baseline notes](inference-blueprint-baseline.md)
record the local checkpoint geometry and earlier exactness runs. This file is
the live status of the blueprint: every row names the changed files, the
reproduction command, the record, and what is still unqualified.

A Codex quota window is not exposed by this session's tools, so the
"stop at 10% remaining" condition could not be read. The work below stopped
when the remaining items needed new architecture (batched backend decode) or
long live runs rather than at a measured quota boundary.

## Implemented in this checkout

| Blueprint area | Implementation and evidence |
| --- | --- |
| Stage 0 checkpoint and correctness baseline | Both local shard inventories passed header validation. The short server benchmark checks a hash of the full first prefill logit row, a greedy output-token digest, and the actual runtime path. Flash-Next already has a decode attribution harness. Nemotron records per-layer router synchronization, expert dispatch and wait, routes, and pool deltas. |
| DEC-01 / MIC-01 | Removed redundant Flash-Next router evaluation outside profiling. Nemotron batches speculative verifier argmax results into one host transfer. SSD routing still requires the chosen expert IDs on the host. |
| MIC-12 | Both backends can defer Python's automatic cyclic GC during generation with `VEYSTRIX_DEFER_DECODE_GC=1`. The setting is restored on exit. |
| MOE-01 / MIC-02 | Both runtimes can select Least-Stale. Flash-Next has an optional flat INT32 slot lookup that tracks evictions, failed fills and ring leases. |
| DEC-11 / MIC-18 | Flash-Next can reserve QSA indexer key and position buffers for slice writes. Nemotron caps the MLX allocator cache during prefill. Nemotron has block-backed INT8 attention. |
| MOE-02 / MOE-10 | `models.shared.pack_experts` writes versioned 64 KiB-aligned expert-major records, validates source identity, publishes atomically, and supports both local checkpoint layouts. **Full packs for all layers now exist** (48 Flash-Next layers, 23 Nemotron layers); see below. |
| MOE-03 / MOE-08 / MIC-13 | Flash-Next issues one native GCD batch per layer into the existing Metal-backed pool. Nemotron uses parallel worker reads and one `preadv` per packed expert, with optional zero-copy DLPack ownership. |
| MOE-04 / MOE-05 | Nemotron orders one-layer-ahead candidates by score and now also prefetches the **layer after next** from the previous pass's route history (`models/shared/route_history.py`, opt-in `VEYSTRIX_NEMOTRON_LOOKAHEAD_LAYERS=1`, depth 12). Advisory only; the real router decides. Offline trace recall 35.96% vs 28.17% last-token and 4.69% random. |
| EXP-I deadline I/O | `models/shared/edf_io.py` is an earliest-deadline-first read executor. Demand reads beat speculative reads at the same deadline; `cancel_stale(layer)` drops prefetches whose layer has passed. Nemotron opts in with `VEYSTRIX_NEMOTRON_EDF_IO=1`; the live run submitted 10,390 reads with 0 failed and 0 pending. |
| PRF-01 / PRF-02 | Nemotron's opt-in Mamba prototype runs local scans in parallel groups of four 64-token blocks and propagates FP32 boundary state. **It is not promoted**: a real 6.5K coding prompt produced a different greedy token digest from stock (record below). PRF-01 requires a bit-match; the prototype is tolerance-level only. Qwen's installed MLX `gated_delta_kernel` already executes the GDN recurrence as one Metal kernel per call, so the Python per-token loop the blueprint targets does not exist there; only the chunk-parallel kernel (KER-08) remains. |
| DEC-04 / DEC-06 | Prompt lookup in both runtimes adapts draft depth from 1 to 4 and cools down after low acceptance or lower measured throughput. The existing exact verifier is retained. |
| UCH-01 / UCH-02 / UCH-08 | `veystrix.universal_cache.ArtifactTier` stores model-neutral, content-addressed artifacts (`blake2b-256`) with an atomic hard-link commit, size and CRC32C validation. `veystrix.fragment_cache.TokenizedFragmentCache` caches exact rendered-prompt token IDs in SQLite WAL keyed by tokenizer identity and `blake2b(text)`; opt-in `VEYSTRIX_FRAGMENT_CACHE=1`. Verified on the live server: first request miss/store, second identical request hit, identical text digest. |
| UCH-03 | Nemotron indexes bounded, previously validated token tapes in a RAM radix trie and retains rolling-hash lookup for disk-only entries. It still verifies the tape and snapshot on restore. Flash-Next's radix index remains model-scoped and is not yet extended (see remaining work). |
| UCH-04 | Every Nemotron snapshot already carries a SHA-256 over tensor bytes (`models/nemotron35/state.py`) and is validated on restore; the artifact tier adds a CRC32C file check. The blueprint's atomic tmp→fsync→rename commit is present. SHA-256 is stronger than CRC32C but slower; CRC32C exists for the neutral tier, not as a replacement for the tensor digest. |
| UCH-05 / UCH-07 | `dynamic_quota_bytes` clamps the configured cache ceiling to `min(quota, 25% free)` and is used by `Transaction.make_room` and the Nemotron snapshot commit check. `Transaction.compact` drops stale reuse/pin records and empty namespaces. |
| LCT-01 / LCT-07 | `models/shared/paged_kv.py` implements a 64-token block table, copy-on-write forks, sink/recent pinning, victim ordering and spill/restore. Exact storage is tested; runtime integration waits on batched request state. |
| LCT-05 | `models/nemotron35/config.position_qualification` reports the declared 262,144 positions and the absent RoPE scaling metadata. By owner decision the backend plans and advertises the 1M window by default; `VEYSTRIX_NEMOTRON_EXTENDED_CONTEXT=0` restores the strict gate, and `/health` reports `million_token_validation: false` plus `extended_context_enabled`. Positions past 262,144 remain numerically unvalidated. See the verification matrix below. |

## Full expert-pack comparisons (session 2)

Both local checkpoints were packed for every routed layer and compared against
the original shards with the same prompt, greedy decoding, a fresh private
prefix directory per arm, a fixed 2 GB expert pool, and the first-prefill-logit
digest gate. Each result is a one-round **diagnostic pair**, not a promotion
result.

| Model | Pack | Baseline | Full pack | Physical reads (baseline → pack) | Swap |
| --- | --- | ---: | ---: | --- | --- |
| Nemotron 3.5 | 15.45 GiB, 23 layers | 3.30 tok/s | 3.79 tok/s | 28.09 → 27.88 GB | 1,069.7 MiB flat |
| Flash-Next | 70.50 GiB, 48 layers | 2.21 tok/s | 2.59 tok/s | 72.29 → 70.01 GB | 1,069.7 → 1,220.4 MiB baseline arm; +7 MiB pack arm |

Both pairs were repeated with the standard three alternating rounds
(forward, reverse, forward), which resolves the effect instead of leaving it
as a diagnostic:

| Model | Baseline rounds | Pack rounds | Paired decode effect | Physical reads |
| --- | --- | --- | --- | --- |
| Nemotron 3.5 | 3.55 / 3.54 / 3.50 tok/s | 4.00 / 3.95 / 3.97 tok/s | **+0.45 tok/s** (band 0.025, resolved) | unresolved (+0.07 GB median, band 0.20 GB) |
| Flash-Next | 2.26 / 2.32 / 2.34 tok/s | 2.58 / 2.57 / 2.58 tok/s | **+0.25 tok/s** (band 0.04, resolved) | **−1.53 GB** median (band 0.25 GB, resolved) |

All twelve arms emitted the same greedy token digest per model and the same
first-prefill logit digest. Nemotron's process footprint grew about 47 MB
(resolved); Flash-Next's footprint change was inside the band. Swap stayed
flat within each arm except one Flash-Next baseline arm, which grew 36.6 MiB
between the request-start and request-end samples. These are the first
resolved full-pack effects in this checkout; they still cover one coding
prompt of about 570 tokens with 24 output tokens, not long prompts or
concurrency.

Both models produced identical greedy token IDs and identical first-logit
digests across arms; Nemotron recorded 5,153 packed reads in the pack arm and
0 in the baseline arm, and Flash-Next's health path confirmed its packed reader
ran. Pack build cost measured by `/usr/bin/time -l`: Nemotron 16.59 GB in
20.46 s, Flash-Next 75.70 GB (the pack is disk-allocated; the benchmark reads
it through the pool). Disk free fell from 136 GiB to 50 GiB, which is why
UCH-05's dynamic watermark is now wired into the commit path.

[Flash-Next one-round record](../../results/flashnext/20261113-141837-performance-v2-expert-pack-manual/performance-arms.jsonl)
[Nemotron one-round record](../../results/nemotron35/20261113-141728-performance-v2-expert-pack-manual/performance-arms.jsonl)
[Nemotron three-round record](../../results/nemotron35/20261113-143218-performance-v2-expert-pack-manual/performance-arms.jsonl)
[Flash-Next three-round record](../../results/flashnext/20261113-143509-performance-v2-expert-pack-manual/performance-arms.jsonl)

## Expert-pool size sweep and the physical ceiling (session 3)

The blueprint's decode targets depend on 85–95% expert residency (Nemotron) and
75%+ (Qwen). I tested whether buying residency with a larger pool helps on the
installed 16 GB machine, with all other flags off, three alternating rounds,
and the swap guard on:

| Model | 2 GB pool | 6 GB pool | Physical reads | Footprint |
| --- | ---: | ---: | --- | --- |
| Nemotron 3.5 | 3.32 / 4.07 / 3.89 tok/s (median 3.89) | 3.51 / 3.70 / 3.65 tok/s (median 3.65) | 25.62 → 22.22 GB | 4.38 → 8.69 GiB |

All Nemotron arms matched token and first-logit digests. The larger pool cut
reads by 13% but did **not** improve decode; the extra cache bookkeeping and
allocator pressure offset the read reduction. A 24-token run cannot resolve a
small gain, but it rules out a large residency win at 6 GB.

Qwen's 6 GB arm never started: the runtime refuses a pool above 2 GB
(`models/flashnext/settings/modes.py:POOL_CAP_GB`) because the recorded 6 GB
arms ran near the 3 GiB compressor guard line. The failed arm is preserved in
`results/flashnext/20261113-145202-performance-v2-pool-compare-manual/`. Qwen's
validated ceiling on this machine is therefore a 2 GB pool.

The reason is arithmetic: the blueprint's own working set is 11 GiB, while
85% / 95% Nemotron residency needs 15.4 / 17.0 GiB of experts plus 1.15 GiB
core and KV/workspace. Even 95% residency exceeds the whole 16 GB machine.
The maximum Nemotron residency inside 11 GiB is ~56%. For Qwen, 55% residency
needs 41.5 GiB and 85% needs 60.5 GiB; at 262K with Q8 KV the machine can hold
about 6.8% of its experts. The modeled 15–30 tok/s (Nemotron) and 6–14+ tok/s
(Qwen) rows therefore cannot be reached on this hardware by any
implementation; they are, as the blueprint states, analytical sensitivity
bounds, not measured targets. The reachable envelope is roughly 5–9 tok/s on
Nemotron and 3–5 tok/s on Qwen after the remaining launch/overhead and
speculation work; reaching the paper's numbers needs 32–64 GB of unified
memory or a much smaller checkpoint.

### Single-flag screens (session 3)

One alternating round per flag with the logit digest and token gate on. These
are screens, not promotion runs, but they rule out the cheap switches:

| Runtime | Flag | Baseline | Candidate |
| --- | --- | ---: | ---: |
| Nemotron | `VEYSTRIX_NEMOTRON_DECODE_OVERLAP=1` | 3.85 tok/s | 3.83 |
| Nemotron | `VEYSTRIX_NEMOTRON_GATHER=1` | 3.80 | 3.52 |
| Nemotron | `VEYSTRIX_NEMOTRON_METAL_FMA_DEQUANT=1` | 3.84 | 3.59 |
| Nemotron | `VEYSTRIX_DEFER_DECODE_GC=1` | 3.71 | 3.75 |
| Qwen | `FLASHNEXT_PROMPT_LOOKUP=1` (24 tokens) | 2.29 | 2.26 |
| Qwen | `FLASHNEXT_DECODE_OVERLAP=1` | 2.30 | 2.07 |
| Qwen | `FLASHNEXT_COMPILE_GDN=1` | 2.41 | 1.88 |

Every arm matched greedy tokens and the first-logit digest. No flag is a
keeper. The remaining gap is structural: per-layer router sync, expert-launch
serialization, and SSD latency, not a disabled switch.

The pool sweep also carries a qualification: the 6 GB Nemotron arms pushed
system swap from ~1.1 GiB to ~3.4–3.7 GiB, and later arms inherited that swap.
The read reduction (25.6 → 21.7 GB) is real; the decode comparison is
confounded by accumulated swap. Even the optimistic reading is that 6 GB buys
~13 points of hit rate and no throughput.

## Speculation: the one large decode win (session 3)

The 24-token screen hid the only large lever. On a 256-token code-copy task
(the blueprint's repetitive edit/refactor case), Nemotron prompt-lookup
speculation is resolved:

| Nemotron, 256 output tokens | Baseline | Prompt lookup | Paired effect |
| --- | ---: | ---: | --- |
| Decode | 3.75 / 3.82 / 3.80 tok/s | 5.22 / 5.20 / 5.25 tok/s | **+1.45 tok/s** (band 0.035, resolved) |
| Physical reads | 129.3 GB | 42.8 GB | **−86.1 GB** (resolved) |
| Greedy tokens | digest `5a81b8…` | same digest | exact |

Three alternating rounds, 2 GB pool, logit digest gate on. Speculation cuts
expert-union traffic by 67% because one verified batch amortizes the SSD
misses across several accepted tokens.

The same flag on Qwen is a regression: 1.64 → 1.30 tok/s and 302 → 329 GB of
reads at 256 tokens, because Qwen's 2 GB pool forces the verification union
to stream new experts. It stays off for Qwen.

On the open-ended repository-review workload the flag resolves **negative**:
3.87 / 3.86 / 3.94 tok/s without it versus 3.62 / 3.66 / 3.71 with it
(paired −0.23 tok/s, band 0.04, resolved; reads +1.3%). The gate does not
catch it because review prompts still produce some matching n-grams.

Decision: keep `VEYSTRIX_NEMOTRON_PROMPT_LOOKUP=1` **opt-in** and use it for
repetitive edit/test/copy loops, where it is +37%; leave it off for open-ended
review. Qwen keeps it off in all cases.

[Copy-task record](../../results/nemotron35/20261113-151651-performance-v2-flag-veystrix-nemotron-prompt-lookup-manual/performance-arms.jsonl)
[Review record](../../results/nemotron35/20261113-152802-performance-v2-code-review-flag-veystrix-nemotron-prompt-lookup-manual/performance-arms.jsonl)
[Qwen 256-token record](../../results/flashnext/20261113-150636-performance-v2-flag-flashnext-prompt-lookup-manual/performance-arms.jsonl)

## Read scheduling and route lookahead (session 2)

| Run | Baseline | Candidate | Gate |
| --- | ---: | ---: | --- |
| Nemotron EDF I/O (EXP-I) | 3.35 tok/s, 27.58 GB reads, TTFT 7.74 s | 3.37 tok/s, 27.11 GB reads, TTFT 6.57 s | token digest and first-logit digest match; candidate health reports `edf_io` with 10,390 completed / 0 failed |
| Nemotron route lookahead (MOE-04/05) | 3.48 tok/s, TTFT 7.76 s | 3.40 tok/s, TTFT 7.96 s | token and logit digest match; candidate health reports 36 lookahead reads issued, 25 consumed |

A one-round pair cannot resolve a 1–2% throughput change; both runs establish
that the path executes and stays exact, not a speedup. Swap was flat in both
candidate arms. Reproduce with `--edf-io` or `--route-lookahead`.

Offline replay of the recorded 8-token coding decode (184 layer observations)
gives the lookahead real signal: at depth 12 the previous-pass history covers
35.96% of the next token's experts, versus 28.17% for last-token persistence
and 4.69% for random; at depth 6 history and last-token tie at 28.17%. The
runtime default depth is therefore 12. Eight tokens is a signal check, not a
stable rate. [Recall record](../../results/nemotron35/20261113-143036-lookahead-recall-manual/lookahead-recall.jsonl)

## Parallel Mamba at real context (PRF-01) — failed exactness

The 6.5K-token repository-review prompt with stock Mamba-2 versus
`VEYSTRIX_NEMOTRON_PARALLEL_MAMBA=1` produced:

| Arm | Decode | TTFT | Token digest |
| --- | ---: | ---: | --- |
| Stock | 3.66 tok/s | 29.49 s | `e42ebe8781e9…` |
| Parallel scan | 3.91 tok/s | 30.01 s | `ce5d9503a238…` |

The pair is recorded as `status: failed`, `token_parity: false`. The prototype
must stay opt-in and must not be promoted until it reproduces a bit-exact FP32
reference at the checkpoint's real context. The fixture-level tolerance
evidence from the earlier pass is not sufficient.

[Failed record](../../results/nemotron35/20261113-142300-performance-v2-parallel-mamba-manual/performance-arms.jsonl)

## Tokenized fragment cache on the live server (UCH-08)

`bench_fragment_cache` starts the normal `veystrix.cli --server`, sends the
same 2K-token rendered prompt twice, and reads `/v1/cache/diagnostics`:

| Request | TTFT | Fragment hits / misses | Text digest |
| --- | ---: | --- | --- |
| 1 (cold) | 12.95 s | 0 / 1, 1 store | parity |
| 2 (warm) | 12.37 s | 1 / 1 | same |

The gain here is small because the prefix cache already served most of the
work; the result establishes that the cache is on the live encode path and
returns the same token IDs. [Record](../../results/nemotron35/20261113-142546-fragment-cache-manual/fragment-cache.jsonl)

## Verification

- Flash-Next unit suite: **1,881 passed, 3 skipped**, including the new paged
  KV tests.
- Nemotron unit suite: **137 passed, 8 skipped**, including the EDF/lookahead
  tests and the position-qualification test.
- `veystrix` suite: **874 passed, 8 skipped** after the `ModelService.__init__`
  fragment-cache attribute was made tolerant of test doubles.
- New focused tests: `models/nemotron35/tests/unit/test_edf_lookahead.py`,
  `models/flashnext/tests/unit/test_paged_kv.py`,
  `veystrix/tests/test_universal_cache.py`,
  `veystrix/tests/test_fragment_cache.py`,
  `veystrix/tests/test_request_state.py`.
- `models/shared/paged_kv.py` and `veystrix/request_state.py` are unit-tested
  but not on a live runtime path yet; they are marked pending below.

## Environment state at this cutoff

- Checkout: `/Users/vincent/Documents/Veystrix`, branch `unlimited-generation`.
- Qwen checkpoint: `/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP`
  (`$QWEN_CHECKPOINT`). Nemotron checkpoint:
  `/Users/vincent/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit-mlx-community`
  (`$NEMOTRON_CHECKPOINT`).
- The full packs were deleted after the comparison to reclaim 86 GB (`df`
  went from 50 GiB to 155 GiB free); the measurement records above are
  retained. Rebuild them with the two `pack_experts` commands under
  Reproduction. The observed build cost was 20.46 s for Nemotron and about
  2.5 min for Flash-Next.
- Python: `/Users/vincent/Documents/Veystrix/.venv` (MLX 0.32.2, mlx-lm 0.31.3).
- Tests use `unittest`, not pytest. Counts: Flash-Next 1,881 passed / 3
  skipped; Nemotron 137 passed / 8 skipped; `veystrix` 874 passed / 8 skipped;
  the focused set above passes 70 tests.

## Reproduction

```sh
# Full packs (destination must not exist)
.venv/bin/python -m models.shared.pack_experts nemotron35 "$NEMOTRON_CHECKPOINT" /path/nemotron35-full.pack
.venv/bin/python -m models.shared.pack_experts flashnext "$QWEN_CHECKPOINT" /path/flashnext-full.pack

# Full-pack diagnostic pairs
.venv/bin/python -m models.flashnext.tests.bench.bench_server_short_paired \
  --runtime nemotron35 --checkpoint "$NEMOTRON_CHECKPOINT" --performance-v2 \
  --expert-pack /path/nemotron35-full.pack --rounds 1
.venv/bin/python -m models.flashnext.tests.bench.bench_server_short_paired \
  --runtime flashnext --checkpoint "$QWEN_CHECKPOINT" --performance-v2 \
  --expert-pack /path/flashnext-full.pack --native-expert-pack --rounds 1

# Deadline I/O, route lookahead, and the parallel-scan exactness gate
.venv/bin/python -m models.flashnext.tests.bench.bench_server_short_paired \
  --runtime nemotron35 --checkpoint "$NEMOTRON_CHECKPOINT" --performance-v2 --edf-io --rounds 1
.venv/bin/python -m models.flashnext.tests.bench.bench_server_short_paired \
  --runtime nemotron35 --checkpoint "$NEMOTRON_CHECKPOINT" --performance-v2 --route-lookahead --rounds 1
.venv/bin/python -m models.flashnext.tests.bench.bench_server_short_paired \
  --runtime nemotron35 --checkpoint "$NEMOTRON_CHECKPOINT" --performance-v2 --parallel-mamba --rounds 1

# Fragment cache on the normal server
.venv/bin/python -m models.nemotron35.tests.bench.bench_fragment_cache \
  --checkpoint "$NEMOTRON_CHECKPOINT" --tokens 2048
```

Runtime controls added in this pass: `VEYSTRIX_NEMOTRON_EDF_IO`,
`VEYSTRIX_NEMOTRON_LOOKAHEAD_LAYERS` (+ `VEYSTRIX_NEMOTRON_LOOKAHEAD_DEPTH`),
`VEYSTRIX_FRAGMENT_CACHE`, and `VEYSTRIX_NEMOTRON_EXTENDED_CONTEXT`.
`VEYSTRIX_NEMOTRON_PARALLEL_MAMBA` stays experimental and currently fails the
real-context token-parity gate.

## Remaining work in dependency order

1. **Full-pack promotion.** Three alternating arms per condition are now
   recorded for both models; the remaining gap is the large coding workload,
   long prompts, peak-memory/swap-growth deltas, and sustained runs. Rebuild
   the packs when needed and delete them afterwards (they are 86 GB).
2. **Parallel Mamba correctness (PRF-01).** Find the chunk-boundary divergence
   and compare against an FP32 reference at 2K–8K tokens; keep the path off
   until greedy token IDs match. Then evaluate the chunk-parallel Qwen GDN
   kernel (KER-08), since the installed `gated_delta_kernel` already removes
   the Python token loop.
3. **Live integration of paged KV (LCT-01/07).** Attach `models/shared/paged_kv.py`
   to a batched backend: one block table per request, shared pages for the
   common prefix, sink/recent pins, then LCT-03 SSD cold pages and LCT-04 QSA
   sparse fetch.
4. **Universal cache completion.** Extend the radix index to Flash-Next,
   wire `ArtifactTier` into repo/AST/tool-log producers with a compaction
   trigger that observes active decode (UCH-07), and measure warm TTFT with
   the fragment cache at 8K–64K prompts.
5. **Independent request state and continuous batching.** `veystrix/request_state.py`
   now provides admission caps, decode-first ordering, cancellation and
   session-keyed state ownership, but the backends still own one live
   conversation. The next step is a backend API that accepts a `RequestState`
   and a batched decode step; then CON-03 expert-union batching, CON-04
   context affinity and CON-05 warm handover can be measured on 2–4 clients.
6. **Experimental kernels and quantization.** KER-06 (fused norm+router+top-k),
   KER-05 (4-branch residual), KER-03 (fused quantized expert), QNT-06
   (hot-Q4/cold-Q3), QNT-05 tiered KV beyond the current opt-in, and the
   split-precision prototype remain. They need quality gates; nothing in this
   pass promotes an approximate path.
7. **Long context and harness.** LCT-02/LCT-08/LCT-09 checkpoint forks and
   parallel restore, LCT-04 QSA sparse SSD paging, the 1M Nemotron numerical
   qualification, and Pi/Hermes/Qwen Code task-per-hour measurements remain.

Use the current local formats when validating the plan. Qwen uses affine
Q4 group 32. Nemotron uses group 64 and its checkpoint metadata declares
262,144 positions. A 1M Nemotron run remains a separate qualification gate.


## LCT-05 regression fix and harness verification (session 4)

A client request with a 1M `max_completion_tokens` hit the LCT-05 gate even
though its prompt was small: the gate counted the requested output cap as
positions. The gate now checks only real prompt positions. By owner decision
the default window is 1M again (the app launches with
`VEYSTRIX_NEMOTRON_EXTENDED_CONTEXT=1` when the configured context exceeds
262,144); `VEYSTRIX_NEMOTRON_EXTENDED_CONTEXT=0` restores the strict gate.

Changed files: `veystrix/backends/nemotron35.py` (prompt-only gate; 1M default
with strict opt-out), `apps/macos/Veystrix/Sources/VeystrixKit/Settings.swift`
(1M default), `apps/macos/Veystrix/Sources/VeystrixKit/LaunchPlan.swift`
(extended-context env), `apps/macos/Veystrix/Sources/Veystrix/SettingsView.swift`
(unvalidated-range note), the Swift/backend tests, and `README.md`. The engine
bundle and `Veystrix.app` were rebuilt and reinstalled. Settings backups:
`settings.json.bak-20261113-160533` (strict-gate pass) and a later 1M backup.

Live verification against the **packaged** engine (`dist/engine`), both models:

| Endpoint | Nemotron | Flash-Next |
| --- | --- | --- |
| `/v1/models` context window | 1,048,576 | 262,144 |
| `/health` admitted context | 1,048,576 | 262,144 |
| `/v1/chat/completions` streaming + `[DONE]` | pass | pass |
| 1M `max_completion_tokens` accepted and streams | pass | pass |
| `/v1/messages` delta + `message_stop` | pass | pass |
| `/v1/responses` `response.completed` | pass | pass |
| `/v1/cache/diagnostics` | 200 | 200 |

Harness suites: `test_server_harness_protocol`, `test_hermes_integration`,
`test_qwen_code_client`, `test_harness_models`, `test_dual_harness_cache` —
**87 passed, 8 skipped**. `veystrix.harness_configure` dry-run generated valid
configurations for Qwen Code, DeepSeek desktop/headless, OpenCode, Kimi Code,
pi and Hermes against the installed checkpoints. Full suites after the change:
Flash-Next 1,881, Nemotron 137, `veystrix` 874, Swift 70 — all passing.

Not qualified: a real 1M-token Nemotron run (LCT-05), prompts above 262,144
with the override, and long-context retrieval quality.
