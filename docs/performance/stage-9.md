# Nemotron Stage 9: adaptive native MTP depth

Stage 9 asks for speculative Nemotron decoding at 20 tok/s or more through the
shared draft and verification framework. Native MTP with exact verification
already exists (`VEYSTRIX_NEMOTRON_NATIVE_MTP`, depth from
`VEYSTRIX_NEMOTRON_MTP_DEPTH`, 1 to 4). This pass adds the adaptive draft
depth built for Flash-Next Stage 5. Real-checkpoint figures from
14 November 2026 are under "Measured on the real checkpoint".

## What it adds

With `VEYSTRIX_NEMOTRON_MTP_ADAPTIVE=1` (default `0`),
`EngineCore.native_mtp_tokens` asks the shared controller
(`models/shared/spec_depth.py`, described in [Stage 5](stage-5.md)) for each
round's draft depth, from 0 up to the configured depth. A round with no drafts
is a plain decode step and is the controller's sample of plain speed.

The controller is fed measured wall time per round: drafting (the MTP forward
loop), verification (the layer-major pass, including the snapshot copy), and
repair (restoring the snapshot and re-decoding one token at a time after a
mismatch). The depth it records is the number of tokens actually drafted, which
can be shorter than requested when a stop token or the output limit cuts the
draft.

One controller lives per conversation: it survives turns, is rebuilt when the
configured depth changes, and is dropped by `EngineCore.reset()`. Emitted
tokens are the same at any depth; the tests compare against plain greedy
decoding on the tiny checkpoint under the adaptive schedule, a forced schedule
through every depth, and a forced mismatch at every draft position.

`core.mtp_adaptive_stats()` returns the controller's state. The flight
recorder's `nemotron_decode_mode` record carries `mtp_adaptive`.

## Measured on the real checkpoint

14 November 2026, paired server harness, greedy, no expert pool, 576-token
prompt that asks for repeated code lines to be copied (favourable to draft
acceptance); details in [the real-run record](real-runs-2026-11-14.md). The
six arms of each pair share one token digest.

| Pair | Tokens | Plain tok/s (median of 3) | MTP tok/s (median of 3) |
|---|---|---|---|
| Fixed depth 2 | 128 | 4.94 | 4.94 |
| Adaptive, ceiling 4 | 512 | 5.03 | 4.71 |

Native MTP does not speed up Nemotron decoding in this configuration. The 8K
code-review prompt and any pool of 1 GiB or more did not fit the 3 GiB
available-memory floor on the day, so there is no figure for them. Accepted
tokens per round and physical MB per accepted token were not recorded by the
harness.

## Still missing for Stage 9

- An external drafter (DFlash, DSpark) has not been evaluated against the
  16 GB budget.
- The 20 tok/s gate is not met: the best measured decode is 5.0 tok/s.
