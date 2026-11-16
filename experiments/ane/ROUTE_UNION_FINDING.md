# Route-union test: first positive signal for ANE-assisted decode (2026-11-17)

## What was tested

Fable's gating test for "ANE drafts + GPU verifies exactly": compute the union
of experts touched across K consecutive decode tokens, divided by the sum of
per-token expert sets. A ratio well below 1.0 means a K-token verification
batch can share SSD expert reads — the condition under which drafting could win
where native MTP lost (each extra verification row costing 150–200 ms).

I previously said this test was **blocked** because the saved traces lacked
per-token expert IDs. **That was wrong.** `bench_route_trace.py` records an
`events` list of `(turn, phase, token, layer, sorted(expert_ids))`; the trace at
`results/flashnext/20260922-221213-route-trace-manual/route-trace.json.gz`
contains **36,864 decode sets** (256 tokens × 48 layers × 3 turns). No hardware
run was needed.

## Measured result (`experiments/ane/route_union_analysis.py`)

| K | union / sum | median union experts | bytes read per token vs K=1 |
|---:|---:|---:|---:|
| 2 | 0.812 | 13 | 0.81× |
| 3 | 0.696 | 16 | 0.70× |
| 4 | 0.625 | 20 | 0.63× |
| 6 | **0.542** | 26 | **0.54×** |

Fable's gate was `< 0.6`. **K=6 clears it (0.54); K=4 is close (0.625).**

Interpretation: for a 6-token batch, the model touches 26 distinct experts per
layer instead of 8 × 6 = 48 — a **46% reduction in expert reads per token**.
Applied to Qwen's 289 ms expert-read wait, an ideal shared-read verification
would cut it to ~157 ms, i.e. roughly 2.6 → 4+ tok/s before acceptance losses.

## Critical caveats (must be checked before believing the speedup)

1. **The trace is from a different checkpoint and config.** It records
   `/Users/gioma/models/...` with `resident_experts: 8` and `record_bytes`
   3,211,264 (3.06 MiB), and only **8 routed experts per token/layer** — the
   current config has `num_experts_per_tok: 10`. The ratio at 10 experts/token
   may differ. Re-run the recorder on the current checkpoint (needs permission).
2. **This is the union of all routed experts, not cache misses.** For a warm
   pool the *miss* union is likely smaller (more sharing), which would help the
   idea — but that needs the pool state per token, which the trace does not
   store beyond pinned/slab sets.
3. **Acceptance matters.** At 75% per-token acceptance, K=6 yields ~3.4 accepted
   tokens per verification pass, not 6. Draft quality must be measured.
4. **Bit-identical verification is the hard gate.** Batch-K GPU verification
   must reproduce batch-1 argmax exactly; our own records show MLX matmul
   accumulation order changes between <192 and ≥192 rows. The existing
   `dense_gemm._bit_equal` shape-verified path is the model for how to enforce
   this, but it has only been validated at ≥384 rows, not K=4–6.

## Miss-adjusted result (more realistic, weaker)

Using the recorded `pinned` and `slab` sets as an approximation of always-resident
experts (17–24 per layer), I recomputed the union over routed experts **outside**
that resident set — i.e. approximate SSD misses:

| K | miss-union ratio (median) | mean |
|---:|---:|---:|
| 2 | 0.846 | 0.836 |
| 3 | 0.750 | 0.750 |
| 4 | 0.700 | 0.694 |
| 6 | **0.621** | 0.620 |

With residency accounted for, **K=6 is 0.62 — just above Fable's 0.6 gate**, and
K=4 is 0.70. So the raw all-experts ratio (0.54) overstates the benefit: much of
the overlap Fable's gate hoped for is already captured by the current pin/slab
policy. This weakens but does not eliminate the case — the approximation is
rough (pinned/slab are not the exact live pool state), and a 0.62 ratio still
implies ~38% fewer miss reads in a K=6 batch.

## Why this matters

This is the **first measured evidence from our own data** bearing directly on
whether a K-token batch shares enough expert reads to make ANE drafting + exact
GPU verification worth building. The raw signal clears the gate; the realistic
miss-adjusted signal is borderline. It justifies a careful design and a
cheap offline follow-up, **not** an immediate hardware benchmark.

## Next steps (no hardware run needed yet)

- Extend `route_union_analysis.py` to compute the union of **missed** experts
  using the recorded pinned/slab sets as an approximation.
- Check whether the current 10-expert config changes the ratio by re-recording
  one short trace (needs explicit permission).
- Design the exact-verify kernel that reproduces batch-1 reduction order,
  modeled on `models/shared/dense_gemm.py::_bit_equal`.
