## What changed

<!-- One short paragraph. Link the active handoff/ADR when it changes a decision. -->

## Why

<!-- The problem and the decision, not a list of diffs. -->

## Verification

- [ ] `./scripts/verify.sh` passed (or `./scripts/verify.sh --python-only` when the Swift toolchain is unavailable locally)
- [ ] Live-model evidence added under `results/<model>/<run>/` when model behaviour can change
- [ ] Quality gate run when output, routing, quantisation or speculation can change
- [ ] Docs/handoff updated; no stale branch or status headers introduced

## Safety

- [ ] No preflight profile, expert bound, MTP check or memory-admission check was weakened
- [ ] No unrelated user changes or result folders were modified or deleted
