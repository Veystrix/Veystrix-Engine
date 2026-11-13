# Flash-Next Stage 0 benchmark

Run from the repository root on a quiet Apple Silicon host. Start with a
checkpoint and a short prompt whose greedy digest you can identify in
`production.json`. The example selects the pool-off 16 GiB configuration.

```sh
VEYSTRIX_MEMORY_POLICY=legacy FLASHNEXT_EXPERT_POOL_GB=0 \
  FLASHNEXT_SLAB_GLOBAL=0 FLASHNEXT_SLAB_PACK=0 \
  FLASHNEXT_NATIVE_MTP=off FLASHNEXT_PREFIX_CACHE=0 \
  FLASHNEXT_PROMPT_LOOKUP=0 \
  .venv/bin/python -m models.flashnext.tests.bench.bench_production \
  --telemetry paired --tokens 32 --arms 5 --min-arms 5 --drop 0 \
  --expect-token-sha256 "$REFERENCE_DIGEST"
```

The paired protocol reverses arm order on alternating rounds. Both arms must
match the reference digest when supplied and each other regardless. The digest
is SHA-256 of the UTF-8 Python tuple representation of generated token IDs,
including its punctuation and spacing. Use a reference with the same checkpoint,
prompt, token limit and inference settings. Set `--drop 2 --min-arms 7 --arms 7`
to discard two initial pairs and still retain five. The default telemetry mode is
`off`. Use `timing` for an instrumented single condition and `shapes` for a
diagnostic run. Shape mode adds Python and worker bookkeeping and should be
excluded from throughput comparisons.

Use `--fresh-arms` to reload the backend in the same process per arm. Start a
new invocation for a cold process and use separate controlled prefix state
when comparing prefix hits with cold prefixes. The harness leaves the OS page
cache as found. The benchmark records available memory and swap throughout
instrumented arms. The roofline reports whether retained arms stayed above
3 GiB as a diagnostic. The sampled floor does not stop the benchmark.

## Reading the artifacts

Both files live under the same `results/flashnext/` run directory.

- `production.json` keeps the original production evidence. The Stage 0
  envelope adds provenance and protocol. Instrumented arms add `stage0`
  with the request timeline, counters, memory samples, and optional shape
  histogram. Off arms retain the legacy measurements. The harness preserves
  completed arm rows after a handled interruption or failed digest check.
- `roofline.json` summarizes retained instrumented arms after a completed
  run. Its schema is [version 1](../../veystrix/bench/roofline.schema.json).
  The checkpoint identity covers config/index content and shard metadata.
  Additional metadata hashes bind the tokenizer and template when present.
  Provenance records the Git commit, tracked diff hash, runtime/harness
  fingerprints, host and dependency versions.

Event timestamps use a per-arm monotonic origin and integer nanoseconds.
TTFT starts before tokenization and ends at first token delivery to the
in-process sink. The timeline includes the existing prefill callback and
reports TTFT with its cost removed as a diagnostic. HTTP queueing, SSE
serialization and client receipt require a server benchmark. Stream decode
uses the first and last token callbacks, including callbacks whose detokenizer
piece is empty. Engine decode runs through the final cache commit, which can
occur after the last emitted token. Decode physical bytes use the prefill
callback and the final commit as endpoints.

Computed prefill counts the evaluated suffix when the prefix cache restores
state. Where MLX jointly evaluates logits and the first sample,
`first_logits_ready` identifies that joint boundary. Speculative prefill
records the append-return boundary. Target rounds include singleton cache
commits and speculative verifications. Verification acceptance counts matching
draft proposals before length or stop truncation. Rollback and replay can
contribute to engine wall time beyond the measured verification span.

Physical reads come from the process counter. Positioned read sizes in shape
mode are logical Python worker reads and can include OS cache hits. Native
batch reads and mmap faults can contribute to physical bytes without a
corresponding positioned-read count. Expert hits and misses count unique
requested rows per observed decode layer call. MLX allocator and RSS snapshots
appear at arm boundaries. The available-memory floor and compressor/swap
deltas come from sampled host readings, so transients between samples can be
missed.

Effective storage GB/s divides physical bytes by observed expert wait and
n-gram host time. The remainder is an estimate of non-I/O time. These host
windows can include scheduling, cached reads and concurrent GPU work, so
the reported ceilings are model estimates. The 3 GB/s scenario holds the
estimated non-I/O time and per-token bytes fixed. Missing denominators and
unsupported counters are null.

The resolution band reports empirical p10, median and p90 after five retained
observations. Half the p10–p90 span gives the reported resolution. Paired
overhead uses the generation time ratio for equal token counts. The gate
requires p90 overhead below 1%. The band describes observed dispersion.
Shape counts cover Python MLX quantized dispatch and the custom Flash-Next MoE
dispatch. Python eval counts are API calls. Internal Metal kernel launches,
actual device synchronization counts, physical IOPS and GPU active time
require platform tracing.

The [checkpoint-free fixture](../../models/flashnext/tests/inputs/stage0-telemetry.json)
contains synthetic timestamps and token IDs for deterministic arithmetic and
digest tests.
