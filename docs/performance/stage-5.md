# Flash-Next Stage 5: native MTP speculation

Stage 5 asks for native MTP drafting with exact target verification, full
state rollback, an expert union per verification block, an adaptive draft
length, and 8, 12 and then 20 tok/s on the real model.

Most of the mechanism was already in the checkout. This pass adds the
adaptive draft length and the forced-rejection tests, and records what is
still missing. The real-checkpoint runs of 14 November 2026 are in
[the real-run record](real-runs-2026-11-14.md) and summarised under
"Measured on the real checkpoint": native MTP is not faster than plain
decoding at any depth, with or without adaptive depth or rollback.

## What exists

| Requirement | State |
|---|---|
| MTP head loaded without duplicating base weights | `models/flashnext/mtp.py`, `FLASHNEXT_NATIVE_MTP=on` (off by default) |
| Exact greedy verification | `MTPGreedy` in `models/flashnext/speculative.py` with `Qwen4ExactSpeculativeVerifier`; a partly rejected block restores the snapshot and replays the accepted tokens one at a time |
| Expert union per block | `expert_cache._pool_pass` reads each distinct routed expert once per layer-call |
| Block kernels | Stage 4's `FLASHNEXT_METAL_BLOCK_MOE=1`, off by default |
| Forced rejection restores state | New tests: every position of depths 1 to 4, tokens and cache state equal to a one-token-at-a-time run (maximum absolute difference 0.0 on the tiny model) |
| Adaptive draft length | New: `FLASHNEXT_MTP_ADAPTIVE=1`, off by default |
| Sampled decoding with exact acceptance | Missing. Native MTP is used for greedy turns only; other turns use the ordinary decoder |
| MTP state in the prefix cache | Missing. `prefix_session.py` does not cover the MTP decoder's caches |

## Adaptive draft length

With a fixed depth, a block that is not fully accepted pays for the batched
verification and then for a one-token replay of every accepted token, so too
deep a draft is slower than plain decoding.

With `FLASHNEXT_MTP_ADAPTIVE=1` the decoder asks a controller
(`models/shared/spec_depth.py`) for the depth of each cycle, between 0 and
`FLASHNEXT_MTP_DEPTH`. The configured depth is the ceiling because the memory
planner reserves verifier workspace for that width. Depth 0 drafts nothing and
verifies one row, which is a plain decode step inside the same decoder.

The controller picks the depth with the highest expected tokens per second:

```
tokens(d)  = 1 + sum over i = 1..d of p1 * ... * pi
seconds(d) = draft * d + single + row * d
             + sum over a = 0..d-1 of P(exactly a accepted) * (a + 1) * repair
```

- `pi` is the chance that draft position `i` is accepted given the earlier
  ones were. It is estimated from decayed success and failure counts, pulled
  toward the estimate for position `i - 1`, so a rarely tried deep position
  starts near its neighbour instead of a fixed guess.
- `single` is the measured time of a one-row cycle (verification plus the MTP
  fold), `row` the extra verification time per drafted row, `draft` the
  drafting time per token and `repair` the replay time per token after a
  rejection. All four are running averages of wall time, so expert misses,
  route overlap and I/O delay are priced in as they happen.
- Every eighth cycle or so it tries one position deeper, more often when the
  deeper depth is nearly as good and less often when it is clearly worse, and
  it climbs immediately after a fully accepted block. Every 64th cycle it
  takes a depth-0 sample to keep `single` current.

The controller is deterministic and never changes which tokens are emitted;
any depth schedule yields the same greedy output, and the tests check that
against a one-token-at-a-time run, including a forced schedule through every
depth.

In simulation (seeded acceptance draws, fixed costs, 400 cycles, 8 seeds, both
ceilings 4 and 8) the realised rate was at least 0.92 of the best fixed depth
as a median over the whole run and at least 0.96 over the second half, across
five cases: high acceptance, acceptance falling with depth, low acceptance,
rows as costly as plain steps, and costly repair. Over 32 seeds the costly
repair case dips to 0.84 on its worst seed, because its best depth is only
slightly better than depth 0. These are properties of the controller, not
measurements of the model.

`MTPStats` now also carries `draft_seconds`, `verify_seconds`,
`repair_seconds`, `fold_seconds`, `depth_counts` and, when adaptive, the
controller's state.

## Rollback instead of replay (opt-in)

After a partly rejected block, `MTPGreedy` restores the cache snapshot and
replays the accepted tokens one at a time. `FLASHNEXT_MTP_ROLLBACK=1` instead
keeps the verified prefix with the rollback `FastDraftGreedy` already uses:
the verifier's captured recurrent states for the last kept token, the
attention cache trimmed to the kept length, and the per-layer embedding
history restored from the verifier output. The body of that rollback is now a
module-level function, `rollback_verified`, shared by both decoders;
`FastDraftGreedy` calls it unchanged. The flag is read at construction and in
`reset()`. With it off, `MTPGreedy` makes the same calls in the same order as
before.

With the flag on, `_target_verify` keeps the verifier output for one cycle.
A partial rejection calls the rollback with `keep_count = accepted + 1`.
When generation stops inside a block, a fully used block needs no repair and a
shorter prefix uses the rollback with `keep_count = keep`; a stop before the
first token still restores the snapshot. The MTP-cache trimming and folding
are unchanged. The rollback evaluates the recurrent states it leaves in the
cache, so the verifier output is not kept alive by lazy slices after the
cycle. `MTPStats` gains `rollback_seconds` and `rolled_back` (tokens kept by
rollback); when the adaptive controller is on, the rollback time is its
repair cost.

What the tiny-model tests show (random weights, two layers, depths 1 to 4):

- Emitted tokens equal plain greedy decoding in every case run: a forced
  rejection at every position of every depth, multi-cycle runs with the real
  draft head and with an oracle head that mixes accepts and rejections (once
  with the adaptive controller), stops and token budgets that end inside a
  block, and a second turn after rollback cycles. The tests assert exact
  token equality.
- With the flag on, the target cache is not bit-identical to a one-token-at-a-time
  run. The tests require integer arrays to be exactly equal and floating arrays
  to be within an absolute bound of 1e-5 (`MTP_ROLLBACK_STATE_BOUND`); the
  largest difference any of them observed is 6e-7. That is float32 rounding
  of attention keys, values and indexer keys, and of recurrent and
  embedding-history rows, that the batched verifier computed and the rollback
  keeps instead of recomputing. The prompt rows stay exactly equal.
- The existing accepted-block path already differs at that level. A test with
  the flag off, every draft accepted, no replay of any kind and the generator
  abandoned after whole cycles leaves floating arrays that differ from
  one-token decoding by a nonzero amount, also up to 6e-7, in every float
  array including the embedding history. So rounding-level state differences
  are not introduced by the rollback; the existing forced-rejection test with
  the flag off reaches exactly 0.0 only because a replay recomputes every kept
  token one at a time. (An earlier run that appeared to show the embedding
  history differing only with the rollback was masked by an end-of-run replay.)
- With the flag off, replay still runs, `rolled_back` stays 0 and the
  rollback function is never called.

These tests bound the state difference on a tiny random model. On the real
checkpoint the rollback arms produced the same token digest as plain decoding
at depths 1, 2 and 3 over 128 tokens (see below); longer outputs and other
prompts have not been checked.

## Measured on the real checkpoint

14 November 2026, 128 tokens, 103-token prompt, no pool and no pinned
experts (the configuration that kept 3 GiB available); details and limits in
[the real-run record](real-runs-2026-11-14.md). Every arm has the same token
digest.

| Decoder | Decode tok/s, replay | Decode tok/s, rollback | Accepted drafts per round |
|---|---|---|---|
| Plain | 2.58 | | |
| MTP depth 1 | 2.36 | 2.55 | 0.78 of 1 |
| MTP depth 2 | 2.17 (2 arms) | 1.66 | 1.33 of 2 |
| MTP depth 3 | 1.35 | 1.72 (2 arms) | 1.78 of 3 |
| MTP adaptive, ceiling 3 | 1.45 (1 arm) | | 1.42 |

An extra verification row costs 0.73 of a plain step in wall time, because
consecutive tokens route to mostly different experts and decode is
read-bound, and it is accepted 78% of the time. That caps the gain near 1.03
before draft and fold costs. Several MTP arms were stopped by the run guard
for thermal pressure or available memory; plain decoding was not.

Still unmeasured: the greedy digest and round time with
`FLASHNEXT_METAL_BLOCK_MOE=1`, which only acts with the expert pool on, and
no pool configuration fitted the 3 GiB floor on the day; 512-token MTP arms,
which were not run because no MTP arm beat plain decoding at 128.

## Open design questions

- The one-token replay after a partial rejection is what makes a miss
  expensive. The external-draft decoder has an exact rollback that avoids a
  replay (`_rollback_verified`); a comment in `MTPGreedy` records that a
  batched pass left recurrent state different from one-token decoding on the
  real model. Whether the rollback can replace the replay has to be settled
  on the real checkpoint, not on the tiny model.
- Exact sampled decoding needs the target's full distribution for every
  verified row, which the verifier currently avoids computing.

## Flags

| Flag | Default | Effect |
|---|---|---|
| `FLASHNEXT_NATIVE_MTP` | `off` | Existing. Uses the MTP decoder for greedy turns. |
| `FLASHNEXT_MTP_DEPTH` | `3` | Existing. Draft depth; the ceiling when adaptive. |
| `FLASHNEXT_MTP_ADAPTIVE` | `0` | `1` lets the controller choose each cycle's depth from 0 to the ceiling. |
| `FLASHNEXT_MTP_ROLLBACK` | `0` | `1` keeps the verified prefix with the exact rollback instead of replaying accepted tokens after a partial rejection or an early stop. |
| `FLASHNEXT_MTP_CACHE_LIMIT_MB` | `512` | Allocator cache cap during MTP decode, in MB; `0` means no cap. Default behaviour is unchanged. |
