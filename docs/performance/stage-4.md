# Flash-Next Stage 4: decode and verification kernel lab

Stage 4 asks for a kernel laboratory built from real decode shapes, a
comparison of the pinned MLX against one newer release, and fused kernels for
single-row decode and for two-to-eight-row verification, with a target of 25%
less non-I/O decode time "or the stage reports which term prevents that".

The target is not met, and the lab says why. Single-row decode kernels run at
the memory-bandwidth limit of the machine, so no exact kernel can make them
faster. A row-batched MoE kernel for verification is built, bit-identical and
off by default; it removes about 10 to 12% of a modelled four- or eight-row
verification round.

Everything here was measured with synthetic weights on the GPU. No checkpoint
was loaded and no full-model digest has been checked (see "Not yet measured").

## What it adds

| Piece | Where | Default |
|---|---|---|
| Shape corpus from the Stage 0 trace | `models/flashnext/tests/bench/kernel_corpus.py`, `models/flashnext/tests/inputs/stage4-shape-corpus.json` | n/a |
| Route-overlap windows from the recorded route trace | `models/flashnext/tests/inputs/stage4-route-windows.json` | n/a |
| Kernel lab | `models/flashnext/tests/bench/bench_kernel_lab.py` | n/a |
| Row-batched MoE kernels | `models/flashnext/metal_runtime.py` (`block_plan`, `_BLOCK_BODY`, `execute(..., block=)`), wired in `expert_cache._pool_pass` | off (`FLASHNEXT_METAL_BLOCK_MOE=0`) |
| Header slicing that also accepts MLX 0.32.4 | `metal_runtime._slice_qmv_header` | always; byte-identical output on the pinned wheel |

## Run the lab

```sh
.venv/bin/python -m models.flashnext.tests.bench.bench_kernel_lab --label mlx-0.32.2
```

A full run takes about 140 s and writes
`results/flashnext/<stamp>-kernel-lab-<label>-manual/kernel-lab.json`.
`--only block` runs just the verification section (about 46 s). Run it on AC
power with nothing else using the GPU. The lab records thermal pressure
throughout and times one canary shape at the start and the end; a fanless
machine reaches thermal level 1 near the end of a full run, and level 2 if
two runs are started back to back.

Every figure is the median of five chains, each chain being many calls on
distinct inputs evaluated under one `mx.eval`. "Independent" calls do not
feed each other; "dependent" calls are chained so each waits for the last.
Timing one call per `mx.eval` adds a 105 to 158 µs round trip and must not be
summed; the lab reports it separately.

To time a different MLX without touching `.venv`, install it into a scratch
directory and put that directory on `PYTHONPATH`:

```sh
uv pip install --target "$SCRATCH/mlx-0.32.4" mlx==0.32.4 mlx-metal==0.32.4
PYTHONPATH="$SCRATCH/mlx-0.32.4" .venv/bin/python -m models.flashnext.tests.bench.bench_kernel_lab --label mlx-0.32.4
```

## Findings

Runs: `results/flashnext/20261113-233249-kernel-lab-manual`,
`20261113-233400-kernel-lab-manual` (pinned, no block section),
`20261114-001335-kernel-lab-mlx-0.32.2-block-manual` and
`20261114-002110-kernel-lab-mlx-0.32.4-block-manual`. All on AC power, Low
Power Mode off, thermal level 0 at start and end; the last two reached level 1
during the final one or two block rows. Run-to-run spread is 1 to 3% for the
large dense shapes. The one-row MoE figure ranged from 251 to 315 µs across
the four runs (the eight-row figures agree within 8%), so MoE comparisons are
only made between rows timed side by side in one run.

### Launch overhead is not the cost

A trivial kernel costs 1.4 to 1.5 µs when chained. One decode step issues
about 900 traced launches, so launch overhead is about 1.4 ms of a step.

### Single-row decode is bandwidth-bound

The large shapes move weight bytes at about 100 GB/s regardless of size:
2560→10240 (15.6 MB) takes 142 to 154 µs, the output head 2560→248320
(379 MB) takes 3.8 to 4.0 ms. The same shape at other bit widths (an ad hoc
run on the pinned MLX, no saved artifact) shows time following bytes, not
element count:

| 2560→10240, one row | µs | MB | GB/s |
|---|---|---|---|
| 2-bit | 73.3 | 8.19 | 111.7 |
| 4-bit | 140.7 | 14.75 | 104.8 |
| 8-bit | 264.3 | 27.85 | 105.4 |
| unquantized bf16 | 488.4 | 52.43 | 107.4 |

The custom MoE executor is at the same limit: 251 to 315 µs per layer-call for
one row, about 100 GB/s over the routed expert bytes. A kernel that already
reads each weight byte once at the bandwidth limit cannot be sped up without
reading fewer bytes, and reading fewer bytes changes the model.

Float32 activations with bf16 scales and biases are about twice as slow as
bf16 activations for stock `quantized_matmul` at one row. The lab takes bf16
as the production dtype from reading the code; a comment in
`metal_runtime.py` says float32. That is not reconciled and needs one traced
real run.

### Modelled decode step, one row

Kernels in the corpus only (quantized matmuls, the layer-0 gather, the MoE
executor, the GDN step). Router matmul, norms and glue are not traced.

| ms per step | independent | dependent |
|---|---|---|
| other dense | 19.8 to 21.0 | 22.8 to 24.6 |
| MoE executor (47 layers) | 11.8 to 14.8 | 15.5 to 16.7 |
| GDN | 2.4 to 6.7 | 3.1 to 8.5 |
| tiny dense | 0.4 | 1.3 to 3.6 |
| layer-0 gather | 0.3 | 0.3 |
| total | 38.7 to 39.3 | 48.3 to 50.8 |

The GDN step measured 2.4 to 2.5 ms in the first two runs and 6.4 to 6.7 ms
in the two runs that include the block section, where it was timed after
thermal pressure had reached level 1 and a 30 to 36 s settle wait. The higher
figure is treated as heat-affected. A real run on 2026-11-09 measured about 72 ms of GPU drain
per token at a 2 GB pool, so 21 to 33 ms per token is outside what the corpus
models.

### The MoE executor scales linearly with rows

Each (row, slot) pair is an independent grid slice that reads its expert
again. Eight rows routed to the same experts cost 6.0 to 7.5 times one row
(1.83 to 1.98 ms); eight rows with disjoint experts cost 7.3 to 9.3 times
(2.28 to 2.44 ms). Stock dense
`quantized_matmul` at eight rows costs 2.3 to 2.4 times one row, because it
dequantizes a tile once and applies it to all rows. That stock path sums in a
different order from the one-row path, so it cannot be used for verification
that must reproduce one-row decode bit for bit.

### Consecutive tokens share experts

On the recorded route trace (768 decode tokens, top-8 rows only), the number
of distinct experts per (row, expert) pair at one layer is 0.78 for windows of
two consecutive tokens, 0.60 for four and 0.46 for eight. Consecutive greedy
tokens stand in for the rows of a speculative block; a real draft block may
overlap more or less.

### Row-batched MoE kernels

With `FLASHNEXT_METAL_BLOCK_MOE=1` and two or more rows, `_pool_pass` groups
pairs by pool slot on the host (the routes are already on the host there) and
the gate and up+SwiGLU kernels run one threadgroup per (expert, up to two
rows): each block of weight bytes is read once and applied to both rows. Per
row the arithmetic and its order are those of the per-pair kernels, so the
output is bit-identical; the tests compare raw bytes for bf16 and float32,
group sizes 32 and 64, two to eight rows, and shapes that reach every branch
of the general path. At group size 64 the down projection is batched too; at
group size 32 the fused down+combine kernel is unchanged.

µs per layer-call, pool path, bf16, top-8, group size 32, pinned MLX
(`20261114-001335`). "Trace" averages the 24 recorded windows.

| rows | routing | distinct/pair | per-pair | block, 2 rows | block, 4 | block, 8 | host plan |
|---|---|---|---|---|---|---|---|
| 2 | same | 0.50 | 366.9 | 253.7 | 268.4 | 365.3 | 5.6 |
| 2 | disjoint | 1.00 | 459.2 | 462.3 | 460.7 | 491.7 | 7.8 |
| 2 | trace | 0.79 | 412.8 | 377.0 | 382.8 | 433.4 | 7.0 |
| 4 | same | 0.25 | 724.0 | 481.1 | 537.7 | 712.9 | 8.6 |
| 4 | disjoint | 1.00 | 919.4 | 909.3 | 922.4 | 985.7 | 12.9 |
| 4 | trace | 0.60 | 810.3 | 630.0 | 645.9 | 813.4 | 11.0 |
| 8 | same | 0.125 | 1407.2 | 1041.4 | 1106.0 | 1409.6 | 14.5 |
| 8 | disjoint | 1.00 | 1839.6 | 1834.7 | 1874.2 | 2048.3 | 23.9 |
| 8 | trace | 0.48 | 1624.3 | 1228.9 | 1267.2 | 1635.4 | 25.4 |

Two rows per group is the default. Wider groups read fewer bytes but each
thread carries more state, and at eight rows per group the kernel is no
faster than per-pair even when every row shares every expert. On disjoint
routes the two-row kernel matches per-pair within 1%. The host plan column is
the cost of `block_plan` and two array constructions, paid per layer-call and
included below.

Modelled verification round on the trace pattern (ms, independent):

| rows | per-pair | block | change |
|---|---|---|---|
| 2 | 48.7 | 47.4 | -2.8% (inside run-to-run spread) |
| 4 | 79.7 | 71.7 | -10.0% |
| 8 | 146.7 | 129.3 | -11.9% |

The dependent-mode figures are -4.6%, -9.0% and -10.4%.

### MLX 0.32.4

| | Result |
|---|---|
| Dense shapes, one to eight rows | No difference beyond run-to-run spread (2560→10240: 142.1 vs 143.1 µs; output head 3761 vs 3755 µs). |
| Modelled step | 39.0 vs 39.3 ms independent. |
| Custom executor | Did not build: `qmv_fast_impl` gained a `partial_rows` template parameter. Builds after the header-slicing change. |
| Numerics | One-row `quantized_matmul` with bf16 activations returns different bits from 0.32.2 (the helpers now widen each activation before summing). Multi-row results are identical. |

Because one-row bf16 results change, an upgrade cannot be assumed to keep
greedy digests, and it buys no speed on these shapes. The pin stays at
0.32.2. The new gated-delta kernels in 0.32.4 were not exercised separately:
the lab's GDN step calls the same API under both versions.

## Acceptance status

| Rule | Status |
|---|---|
| Each kernel numerically matched against the MLX reference | Met: worst relative RMS 0.0016; block kernels bit-identical to per-pair. |
| Full greedy digest identical | Not measured. Needs a real run. |
| Non-I/O decode time down 25%, or the limiting term reported | Not met. One row: weight traffic at the bandwidth limit (about 105 GB/s) for the 32 to 36 ms of dense and MoE work per step. Two to eight rows: per-row arithmetic that bit-identical verification cannot share; the batched kernel recovers 10 to 12% of a modelled round. |
| No per-token allocation growth | Not measured on the real model. The block path adds two small route arrays per layer-call. |
| Rollback | `FLASHNEXT_METAL_BLOCK_MOE=0` (the default) leaves kernels, names and cache keys unchanged. |

## Not yet measured

These need the real checkpoint and the owner's approval, on AC power with
Low Power Mode off:

- Greedy digest with `FLASHNEXT_METAL_BLOCK_MOE=1` against the per-pair path,
  on a multi-row verification workload.
- Real verification-round time with and without the flag, and the overlap of
  real draft blocks.
- Where the 21 to 33 ms per token outside the corpus goes (router, norms,
  glue, host work).
- The activation dtype the quantized matmuls actually receive.

## Flags

| Flag | Default | Effect |
|---|---|---|
| `FLASHNEXT_METAL_BLOCK_MOE` | `0` | `1` uses the row-batched MoE kernels in the pool path when a call has two or more rows. |
| `FLASHNEXT_METAL_BLOCK_MOE_ROWS` | `2` | Rows per group: `2`, `4` or `8`. Anything else means `2`. |

## Known hazards

- Editing `metal_runtime.py` and `expert_cache.py` changes the engine
  fingerprint, so saved sessions from before this change are not reused.
- The per-pair fused up+SwiGLU kernel applies SwiGLU at the wrong columns
  when the output width is not a multiple of 4, and the per-pair fast kernel
  reads and writes past the last row when the input width is a multiple of
  512 and the output width is not a multiple of 8. Production widths (640 and
  2560) are unaffected; neither was changed here.
- When more distinct experts are routed than the pool can hold, `_pool_pass`
  falls back to the streamed path, which is not row-batched.
