# K=2 read-sharing for Qwen — design and honest status (2026-11-17)

## What the owner asked
Document a reusable design, build the safe method for Qwen, report the speedup.

## Critical finding from source inspection — do not build a redundant system

The safe K=2 mechanism (share expert reads across 2 tokens, keep per-row compute
bit-identical) is **already substantially implemented** in Qwen's exact
speculative verifier:

- `models/flashnext/qwen4_verifier.py:315-345` calls `switch(hidden, indices, ...)`
  with **all K rows at once**. `StreamingSwitchGLU.__call__` therefore unions the
  routed experts across the K rows and issues one read per distinct expert.
- The parent verifier `_switch_glu` (mlx_vlm `speculative_verifier.py:1305-1320`)
  flattens the K rows into one call — the same union-read shape.
- `models/flashnext/speculative.py` has the full framework: cache snapshot/
  rollback (`snapshot_cache`, `restore_cache`, `rollback_verified`), MTPGreedy,
  FastDraftGreedy, prompt-lookup drafters, and `ThroughputGate`.
- The verify path uses the **exact verifier specifically because batched cached
  forwards change numerics** (`_target_replay` comment: replay token-by-token
  because batched forwards leave singleton-different recurrent states, proven
  up to 1.77 abs). That is the same bit-exactness discipline our quality test
  confirmed.

So the "build K=2" work is NOT a new verifier — it is **turning on and tuning an
existing speculative path** so that K=2 (depth 2) with a cheap drafter actually
wins.

## Why MTP was rejected before (and what would be different)

Prior measurements (real runs): native MTP depth 1/2/3 was slower; at 8K MTP
on 1.17/1.13 vs off 2.32/2.19 tok/s. Causes recorded in the repo:
- low draft acceptance, verification round cost ~0.73 of a plain step;
- memory pressure (MTP on 1.63 GB pool + guards);
- each extra verification row costs its own reads **when rows route to different
  experts** — our K=2 union ratio 0.790 means only ~21% sharing, so 2 rows still
  touch ~1.79x the experts of 1 row.

What K=2 adds that changes the calculus: the read sharing is now **measured**
(0.790 union on the current checkpoint, guarded), and the design is constrained
to the bit-identical verifier path. It does not eliminate the 1.79x read growth;
it only makes the second row ~21% cheaper than it was.

## Honest expected outcome

The repo already tried this general mechanism and measured it slower. The new
evidence (0.790 union) quantifies a ceiling but does not reverse the sign:
a 21% saving on the second row, minus draft cost and verification overhead, is
unlikely to overcome a measured regression. **There is a real chance the answer
is "no speedup" even after a correct implementation.** That must be reported,
not tuned away.

## Build plan (smallest honest change)

1. Use the existing `FastDraftGreedy` / prompt-lookup drafter with `n_draft=2`
   (or MTPGreedy depth 2) behind an opt-in flag — no new verifier.
2. Confirm via `set_route_observer` that the 2-row verifier call unions reads
   (physical bytes/token should drop ~10% if sharing is active).
3. Paired A/B under the guard: baseline vs K=2, alternating, 3 pairs, greedy,
   same token digest required, physical bytes and TPOT collected.
4. Report the measured speedup — including "none" — and delete the flag if it
   regresses.

## Reusable design for future models (owner's general question)

- `models/shared/` holds model-independent pieces (`edf_io`, `route_history`,
  `dense_gemm`, `prompt_lookup`). A future model needs:
  1. a `speculative_verify=True` path that unions expert reads for K rows and
     reproduces singleton numerics (the bit-identical requirement);
  2. a route observer to measure the union ratio (our `live_route_union*.py`);
  3. a bit-pattern test on real weights (our `quality_test_k2*.py`) before the
     path is trusted.
- So: one shared framework, one small per-model adapter, one mandatory quality
  test. Not zero work per model, but not a rewrite.

## Status
The quality prerequisite is proven on both models; the Qwen speculative
framework already exists. The remaining step is the guarded paired A/B — which
needs the owner's authorization for that specific benchmark. No code has been
changed to the engine.
