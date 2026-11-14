# Nemotron Stage 9: adaptive native MTP depth

Stage 9 asks for speculative Nemotron decoding at 20 tok/s or more through the
shared draft and verification framework. Native MTP with exact verification
already exists (`VEYSTRIX_NEMOTRON_NATIVE_MTP`, depth from
`VEYSTRIX_NEMOTRON_MTP_DEPTH`, 1 to 4). This pass adds the adaptive draft
depth built for Flash-Next Stage 5. Nothing here was run on the real
checkpoint; there is no speed figure.

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

## Not yet measured

These need the real checkpoint and the owner's approval, on AC power with Low
Power Mode off:

- Greedy digest with adaptive depth on and off.
- Decode tok/s, accepted tokens per round and physical MB per accepted token,
  fixed depth against adaptive, at the 128- and 512-token gates.

The 8K comparison recorded on 2026-11-13
([native MTP at 8K](native-mtp-8k-1024-2026-11-13.md)) used a fixed depth of
two.

## Still missing for Stage 9

- An external drafter (DFlash, DSpark) has not been evaluated against the
  16 GB budget.
- The 20 tok/s gate has not been attempted.
