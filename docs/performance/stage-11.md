# Stage 11 release qualification

**Decision: the 16 GiB release gate is open.** Neither checkpoint has a qualifying result at 20 stream tok/s. Keep the experimental flags opt-in and leave the release untagged.

## Guarded runs on Apple M4, 16 GiB

These are backend token-callback measurements at concurrency 1. The figures exclude the initial untimed request used to seed the prefix-hit arm. TTFT includes backend prefill and excludes HTTP serialization. Guard traces cover process startup as well as inference.

| Model and workload | Stream tok/s | TTFT | Bytes read per output token | Available floor | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen 512 × 128, cold process | — | — | — | 2.49 GiB | [Stopped run](../../results/flashnext/20261114-121342-stage11-512-128-cold-manual/qualification.json), [guard](../../results/flashnext/20261114-121342-stage11-512-128-cold-manual/guard.jsonl) |
| Qwen 512 × 128, cold process, 128-token prefill chunks | — | — | — | 2.99 GiB | [Stopped run](../../results/flashnext/20261114-121603-stage11-512-128-chunk128-cold-manual/qualification.json), [guard](../../results/flashnext/20261114-121603-stage11-512-128-chunk128-cold-manual/guard.jsonl) |
| Nemotron 512 × 128, cold process | 6.50 | 9.36 s | 268 MB | 4.67 GiB | [Run](../../results/nemotron35/20261114-121453-stage11-512-128-cold-manual/qualification.json) |
| Nemotron 512 × 128, warm process and cold prefix | 5.62 | 9.47 s | 286 MB | 4.43 GiB | [Run](../../results/nemotron35/20261114-122429-stage11-512-128-warm-cold-anchor-manual/qualification.json) |
| Nemotron 512 × 128, prefix hit | 6.45 | 2.59 s | 244 MB | 4.95 GiB | [Run](../../results/nemotron35/20261114-121632-stage11-512-128-prefix-hit-manual/qualification.json) |
| Nemotron 512 × 512, cold process | 6.13 | 9.30 s | 261 MB | 4.84 GiB | [Run](../../results/nemotron35/20261114-121754-stage11-512-512-cold-manual/qualification.json), [guard](../../results/nemotron35/20261114-121754-stage11-512-512-cold-manual/guard.jsonl) |

The Nemotron runs used BF16 KV, plain greedy decode and no expert pool. Swap stayed flat and thermal pressure stayed at level 0. The three 128-output arms share the same token digest. The prefix arm restored 487 IDs. The 512-output run computes about 55 prompt tok/s. The [first warm-process attempt](../../results/nemotron35/20261114-122318-stage11-512-128-warm-cold-manual/qualification.json) changed the prompt's prefill split when it disabled the prefix store. Its different digest exposed a harness error. The corrected warm measurement preserves the system-anchor split and disables only restoration. Independent full-length reference evidence and run-to-run percentiles remain open. The artifacts identify checkpoints via metadata and, in the later runs, shard stat information.

Qwen's first run added 280 MiB of swapouts. Narrowing the chunks eliminated swapouts on the retry, which still crossed the floor before producing output and recorded 15.5 GiB of cumulative compression. The rest of the Qwen matrix and the longer Nemotron matrix remain unqualified. The host measured 1.46 GiB swap already in use by the later runs, with no growth in the completed Nemotron arms.

## What must change

- **Qwen.** The [five historical sustained runs](../../results/flashnext/20261114-102624-bench-production-manual/roofline.json) used a 103-token prompt and generated 512 tokens each. Stream p10/median/p90 were 2.65/2.66/2.69 tok/s. The median arm read 742 MB/token at about 2.79 GB/s, equivalent to 266 ms/token in measured I/O, plus about 110 ms/token outside it. Perfect expert residency would still cap that execution path near 9 tok/s. The 20 tok/s budget needs both compute and I/O reductions, for example a compute budget of 30 ms and a read budget of 56 MB/token at that bandwidth.
- **Nemotron.** The new 512-output BF16 arm spent 163 ms per emitted token. Its 261 MB/token would consume about 94 ms/token at the Qwen arm's 2.79 GB/s. The remaining 69 ms is a conditional estimate pending direct wait and overlap instrumentation. The historical 576-input INT8-KV server run measured 5.03 engine tok/s plain and 4.71 with adaptive MTP. See [the paired experiments and independent stock comparison](real-runs-2026-11-14.md). Priority targets are expert bytes and verify overhead, alongside independent BF16 logit parity.

## Repeat one matrix cell

Run `.venv/bin/python -m veystrix.bench.qualification --list-matrix` for all 60 cells. Run each condition on a quiet 16 GiB host, with a separately generated reference digest for the same input IDs and sampling semantics. Produce the checkpoint's shard SHA-256 manifest outside the timed runs to avoid warming the SSD cache ahead of an arm.

```sh
run_dir=$(.venv/bin/python -c 'from veystrix.results import new_run_directory; print(new_run_directory("nemotron35", "stage11"))')
VEYSTRIX_RESULTS_DIR="$run_dir" VEYSTRIX_NEMOTRON_NATIVE_MTP=0 \
  VEYSTRIX_NEMOTRON_KV=bf16 FLASHNEXT_PREFILL_CHUNK_TOKENS=128 \
  .venv/bin/python scripts/flashnext/real_run_guard.py \
  --log "$run_dir/guard.jsonl" --floor-samples 1 --swap-growth-mib 0 -- \
  .venv/bin/python -m veystrix.bench.qualification --model nemotron35 \
  --checkpoint "$NEMOTRON_CHECKPOINT" --prompt-tokens 512 --output-tokens 512 \
  --checkpoint-hashes "$CHECKPOINT_HASHES" --mode cold-process
```

Create five independent result directories with the same configuration. Then call `python -m veystrix.bench.qualification --assess <qualification.json files> --expect-token-sha256 <reference>`. The gate digest encodes token IDs as signed 32-bit little-endian integers. A second digest records the Stage 0 string-list encoding. A passing cell needs five complete 512-output records, median at least 20 and p10 at least 18 stream tok/s, reference digest parity, and valid memory and thermal records. HTTP streaming overhead and multi-turn memory behavior need separate server experiments.

## Follow-up qualification

Flash-Next now has a `prefill-16gb` launch feature for 128-token chunks, a 256-expert scratch ring, uncached expert reads during prefill, and eager linear state. Use `--enable prefill-16gb --pool-gb 0` for the server. The guarded [512-input, 128-output probe](../../results/flashnext/20261114-123149-stage11-bounded-prefill-probe-manual/qualification.json) completed at 2.63 stream tok/s with a 4.04 GiB floor and flat swap. A [second cold arm](../../results/flashnext/20261114-123528-stage11-512-128-cold-process-r1/qualification.json) and [warm process with a cold prefix](../../results/flashnext/20261114-123704-stage11-512-128-warm-process-cold-prefix-r1/qualification.json) matched its digest. The [first prefix-hit attempt](../../results/flashnext/20261114-123923-stage11-512-128-prefix-hit-r1/guard.jsonl) crossed the floor at 2.98 GiB. Ring eligibility now accounts for the original prompt length after a restore. A later [cold startup](../../results/flashnext/20261114-124959-stage11-512-128-cold-process-r1/guard.jsonl) crossed the floor before the first measured request. The capped allocator now also applies during model load.

The [512-output Flash-Next arm](../../results/flashnext/20261114-124412-stage11-512-512-cold-process-r1/guard.jsonl) and [2K-input arm](../../results/flashnext/20261114-124642-stage11-2048-128-cold-process-r1/guard.jsonl) stopped at thermal level 2. Neither grew swap. The [Nemotron 2K-input arm](../../results/nemotron35/20261114-124952-stage11-2048-128-cold-process-r1/qualification.json) cancelled before yielding a valid result. The successful 512-token probes establish the benefit and limits of the low-transient path on this host. Coverage of longer contexts and sustained decode remains open.

The [corrected Flash-Next prefix arm](../../results/flashnext/20261114-125403-stage11-512-128-prefix-hit-r1/qualification.json) restored 489 of 512 tokens and matched the cold and warm-process digest. TTFT was 4.94 s. The guard recorded a 3.74 GiB floor, flat swap and thermal level 0 for the full process.

The [serial matrix index](../../results/shared/20261114-124959-stage11-matrix-manual/matrix.json) is updated atomically as each guarded attempt finishes. A single pass covers the matrix diagnostically. For the release gate, add `--rounds 5`, both independently produced reference digest maps, and both offline shard hash manifests. Resume the same configuration with `--resume PATH/TO/matrix.json`. The runner waits for the host to cool before each cell and records guard refusals, stopped runs, short outputs and measurements.

```sh
.venv/bin/python -m veystrix.bench.qualification_matrix \
  --flashnext "$QWEN_CHECKPOINT" --nemotron35 "$NEMOTRON_CHECKPOINT" \
  --seconds-per-cell 600 --cooldown-seconds 120
```
