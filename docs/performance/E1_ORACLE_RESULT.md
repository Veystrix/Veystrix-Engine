# E1 — oracle pipeline (Opus plan), 2026-11-17

Decisive test: pass 1 records the REAL greedy route of every MoE call in
execution order; pass 2 replays those exact routes as prefetch hints 1–3 calls
ahead (buffered pread) while the real router still decides. This is the upper
bound of ANY predictor (exact future knowledge, zero training). If it cannot
speed decode, no trained predictor can. Tool: `bench_overlap_oracle.py --experiment e1`.
Guarded, thermal stop disabled. Gate PASS: oracle token/logit/layer digests
identical to reference (`e020e09b…` both sides).

## Result (1,200 tokens, Qwen chat, ahead-depth 3, width 10, 8 threads)

| metric | reference | oracle |
|---|---|---|
| decode rate | 2.14 tok/s | **1.85 tok/s** (SLOWER) |
| ms/token | 466.5 | 539.3 |
| physical MB/token | 915.0 | 823.5 (−10%) |
| critical-path main-thread I/O ms/token | 5.48 | 8.39 |

Prefetch issued: 15.55 M rows = **5.3 GB of prefetch reads during the sweep**
(≈4.4 GB/token of hint traffic) to save ~91 MB/token of misses. Wrong/duplicate
hints dominate. VM: swap 0 both arms, but oracle compress/decompress and
reactivated pages are far higher (prefetch churning the page cache).

## Verdict: KILL — even an oracle cannot save this path

Opus's kill criterion was "oracle below 4.5 tok/s means no predictor can save
this path." We got 1.85 tok/s — the oracle made decode ~13% SLOWER, and
critical-path I/O (5–8 ms) was never the bottleneck anyway. The prefetches add
gigabytes of drive traffic and page-cache churn that cost more than the misses
they remove. This is the same failure mode as our earlier read-ahead experiments
(`handoff-2026-11-15.md:72`: "decode is bound by the bytes read, not by a lack
of overlap"), now confirmed with perfect route knowledge.

## How this reconciles E0 with E1

- E0 proved GPU compute and SSD reads CAN overlap (10% step inflation at 3.36 GB/s).
- E1 proves perfect prefetch hints still cannot help: the win is not "hide I/O
  behind compute" — it is that **every prefetched byte must be a byte that would
  have been read anyway**. Our route trace has only ~0.36 persistence recall
  (E3), so 3-deep × 10-wide hints are mostly wasted bandwidth. A trained head
  would need near-1.0 precision to pay off, which E3 shows is implausible.
- Therefore the serial ceilings in `STEP_B_LONG_TRACE_FINAL.md` stand as the
  practical answer on 16 GB: no predictor/prefetch route reaches 5/15 tok/s.

## Decision (follows Opus's own KEEP/DROP)

- DROP trained predictor (Opus §4) and all prefetch-hint variants (Edge0 as hint,
  SpecFit-style prediction) — E1 is their ceiling and it fails.
- The remaining bit-exact levers are byte reduction: exact LRU slot cache
  (lead E, +31% on this model), pool-as-sole-cache + F_NOCACHE to stop double
  caching, and the single-sync slot remap (lead F). These do not rely on prediction.
- Hardware fallbacks (Opus §5): second SSD, second Mac, or 24–32 GB remain the
  only proven route to 5+/15+ with no quality loss.

Evidence: `results/flashnext/20261117-183612-overlap-e1/e1.json`, guard.jsonl.
