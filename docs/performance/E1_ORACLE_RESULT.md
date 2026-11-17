# E1 — oracle pipeline (Opus plan), 2026-11-17 (FINAL, corrected)

Decisive test: pass 1 records the REAL greedy route of every MoE call; pass 2
replays those exact routes as prefetch hints while the real router still decides.
This is the upper bound of ANY predictor (perfect future knowledge, zero
training). Tool: `bench_overlap_oracle.py --experiment e1 --prefetch-mode next_token`.
Guarded, thermal stop disabled (owner: Mac throttles itself). Gate PASS: oracle
token/logit/layer digests identical to reference (`e020e09b…` both sides).

## Result (1,200 tokens, Qwen chat, one-token lead, width 10, 8 threads)

| metric | reference | oracle | delta |
|---|---|---|---|
| decode rate | 2.19 tok/s | **2.72 tok/s** | **+24.6%** |
| ms/token | 457.5 | 367.1 | −90 ms |
| physical MB/token | 873.1 | 857.5 | −1.8% |
| critical-path main I/O ms/token | 5.16 | 12.61 | +7.5 ms |
| prefetch issued | — | 5.18 M rows / 1,768 MB (≈1.47 GB/token) | — |
| VM swapouts | 0 | 0 | clean |

Guard: floor 2.56 GiB, swap flat, no swapouts, no compression-stop.

## Validity note
An earlier E1 run was invalidated: it issued each future route 3× (delta=1..3)
and wasted 5.3 TB of prefetch traffic, making the oracle 13% SLOWER. The harness
was fixed to issue each route exactly once (`next_token` mode: at token t, layer
L, prefetch the recorded route for token t+1, layer L — one token of lead).
This rerun is the valid one.

## Verdict: KILL (below threshold) — but overlap gives a real +25% with a PERFECT oracle

Opus's kill line: "oracle below 4.5 tok/s means no predictor can save this path."
We got 2.72 tok/s → **KILL**. The targets (5/15) are unreachable by prediction
on 16 GB. Reconciliation with E0:

- E0 proved GPU compute and SSD reads can overlap (3.36 GB/s at 10% step inflation).
- Corrected E1 shows a PERFECT oracle turns that into ~+25% decode (2.19→2.72)
  by moving reads off the critical path — physical bytes barely changed (−1.8%),
  so the win is overlap, not fewer bytes. This confirms the serial ceiling was
  too pessimistic, but only marginally.
- Even a perfect oracle lands at 2.72, far below 5. The byte-volume floor still
  dominates: ~873 MB/token ÷ 3.36 GB/s ≈ 260 ms drive time vs ~200 ms compute →
  ~3.8 tok/s is the absolute ceiling at this byte volume, and that needs a
  PERFECT predictor.
- E3 (offline) shows real zero-training predictors reach only ~0.36 recall, far
  from the 1.0 the oracle assumes. A trained head (E4) cannot close that gap
  enough to matter. So prediction is a dead end for the targets.

## Path forward (per Opus's own KEEP/DROP)

- DROP trained predictor (E4) and all prefetch-hint variants.
- KEEP the structural byte-reduction levers (they don't need prediction):
  exact LRU slot cache (lead E, +31% on this model), pool-as-sole-cache +
  F_NOCACHE (E2), single-sync slot remap (lead F).
- KEEP hardware fallback (Opus §5): **second SSD** first (~5.5–6.5 GB/s
  aggregate → Nemotron 15+, Qwen compute-bound ~7+), then second Mac, then
  24–32 GB RAM. Any remaining gap to 5/15 is a memory/bandwidth limit, not a
  software-prediction limit.

Evidence: `results/flashnext/20261117-185834-overlap-e1/e1.json`, guard.jsonl.
