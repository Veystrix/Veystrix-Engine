# Flash-Next prefill and prefix restoration

Set `FLASHNEXT_PREFILL_AUTOTUNE=1` to choose a balanced chunk plan once
per target-model prefill. The planner estimates the widest chunk at the
request's longest context, using current host headroom and the existing
prefill memory model. It chooses among the configured cap and standard
candidate widths. The smallest candidate remains available when no
estimate fits. This is shape selection and does not reject or stop
inference. Native MTP and fused-draft prefill keep their existing plans.

An optional `FLASHNEXT_PREFILL_SHAPE_PROFILE` points to a version 1 JSON
file with a `seconds_by_chunk` map. If all eligible widths have measured
costs, the tuner minimizes estimated total chunk time. Otherwise it
chooses the largest eligible width. Use measurements from the same
checkpoint and execution order. The benchmark writes shape samples from
non-final chunks of cold reference arms. Combine per-width observations
from separate `--chunk-tokens` runs to cover all eligible widths.

The selected width and plan version are part of the prefix-cache identity.
Within that namespace, the cache still requires exact prefix tokens and
the same chunk history at the restore boundary. Switching widths opens a
different namespace. Returning to an earlier width can reuse its compatible
SSD snapshots. The autotuned prefill uses one selected plan from lookup
through snapshot creation, even if available memory changes during the
request.

`FLASHNEXT_QSA_PREFILL_TILE_EVAL=1` evaluates each reference QSA query tile
before constructing the next tile's live mask and score graph. The query
tile shape, argpartition and attention reductions stay the same. The
existing GDN scan is already fused. Its convolution-state lifetime can be
tested with `FLASHNEXT_EVAL_LINEAR_STATE=1`, which releases retained
prefill inputs while keeping the same operations. Both knobs apply to
prefill-sized work. The QSA tile mode is fingerprinted separately.

## Run a comparison

```sh
.venv/bin/python -m models.flashnext.tests.bench.bench_stage6_prefill \
  --checkpoint "$QWEN_CHECKPOINT" --sizes 2048 8192 16384 32768 \
  --candidate autotune --rounds 5 --tokens 8
```

Select `--candidate layer` to compare the existing layer-major expert
reuse path with identical chunk shapes, or `--candidate kernels` to
compare the prefill-only QSA and GDN lifetime changes. Layer-major results
include actual execution windows and expert bytes, so a fallback can be
distinguished from a layer-major run.

Each arm starts with a private empty prefix cache. Its second request uses
a fresh conversation with the same prompt and verifies a prefix hit and an
identical greedy continuation. `--state-check` also hashes the continuation
KV, recurrent, indexer and position state after the timed request. The
benchmark records TTFT, computed prefill tokens and time, reused tokens,
stream decode, physical bytes, MLX peak and sampled OS memory. Paired bands
appear after five rounds. The benchmark runs in-process with the external
guards disabled and the OS page cache as found.
