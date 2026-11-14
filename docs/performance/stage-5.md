# Flash-Next Stage 5: native MTP speculation

Stage 5 asks for native MTP drafting with exact target verification, full
state rollback, an expert union per verification block, an adaptive draft
length, and 8, 12 and then 20 tok/s on the real model.

Most of the mechanism was already in the checkout. This pass adds the
adaptive draft length and the forced-rejection tests, and records what is
still missing. Nothing here was run on the real checkpoint, so there is no
speed figure (see "Not yet measured").

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

## Not yet measured

These need the real checkpoint and the owner's approval, on AC power with
Low Power Mode off:

- Greedy digest with `FLASHNEXT_NATIVE_MTP=on` against the plain decoder, with
  `FLASHNEXT_MTP_ADAPTIVE` and `FLASHNEXT_METAL_BLOCK_MOE` each off and on.
- Decode tok/s, accepted tokens per cycle, verification time per cycle and
  physical MB per accepted token, fixed depth against adaptive, at the 128-
  and 512-token gates.
- Whether adaptive depth lifts native MTP above the plain decoder at all. The
  settings text currently says native MTP is slower on the streamed runtime.

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
