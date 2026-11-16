# Reopening ANE development (owner decision, 2026-11-17)

The owner has decided to resume Apple Neural Engine development now that the models run correctly on the Mac. This document supersedes the *recommendation* of `REPORT.md` (2026-11-07) but keeps its measurements as the baseline. `REPORT.md` remains the historical record and is not edited.

## Why the earlier rejection still matters

`REPORT.md` measured, on this exact M4 Air / 16 GB / macOS 27.2, that:

- batch-1 (decode-sized) predictors **never reach the ANE** — they silently run on CPU (~35 µs, 0 MB neural footprint);
- routing numerics matched the FP32 reference only **98.8%** of 2,048 rows, so ANE routing is not bit-exact;
- ANE traffic added **+3.5% to +17.6%** to bandwidth-bound GPU work;
- dense offload needs a duplicate FP16 copy of the 3.4 GB dense core, which does not fit.

The deep research in `reports/Apple Neural Engine integration.md` corroborates each mechanism with independent 2026 measurements. Any reopened work must **explain why it is not subject to these same effects**, not merely restate optimism.

## The thesis: the bottleneck changes with RAM

The rejection was earned against an **SSD-bound** token: 289 of 385 ms is expert-read wait, ~70 ms GPU, 51 ms router sync. Amdahl caps the whole ANE prize at ~1.33× there.

But `higher-mac-support` already makes **Nemotron's 15.38 GiB experts fully resident at 32 GB+** (`models/nemotron35/admission.py`, `veystrix/memory_tiers.py`). When expert reads stop dominating, decode becomes **compute-bound**: the recorded arithmetic estimate is ~41 ms of GPU work → roughly 24 tok/s. That is the regime in which ANE results are reported (NPUMoE, MacPaw, Bryngelson) and in which a second silicon block can help rather than contend.

Therefore:

- **Track B (32 GB and larger Macs) is the credible path.** Evaluate ANE offload of *static, repeated, non-routing* work there: attention projections and dense FFN of the resident model. This is where the memory fits and where the GPU has work to offload.
- **Track A (16 GB) stays conservative.** Do not put ANE in the output-affecting hot path. The only defensible 16 GB role is **output-neutral work during SSD stalls** (the GPU is idle while waiting on experts), gated on a contention measurement. Expected gain is small.

## Hard gates (unchanged)

1. **Bit-identical token digest** and BF16 rounding boundaries (`docs/flashnext/AGENT_INVARIANTS.md:42-44`). Any output-affecting ANE path must pass the 24-case quality gate with 24/24 token-identical results. A "close" numerical match is a failure.
2. **Every path opt-in with a rollback** (`docs/flashnext/quality-validation-policy.md`). ANE must be a toggle defaulting off until it clears the gates.
3. **No benchmark, sweep, or hardware test without the owner's explicit per-workload permission** (`AGENT_INVARIANTS.md:155`). Nothing in this plan authorises a run.
4. **No model weights ship** in the app or DMG. Any `.mlmodelc` used for production would be a legal/notice and ADR decision, not just a build change.
5. **One large model at a time**, AC power, Low Power Mode off, ≥3 GiB available, stop on swap/compressor growth, thermal level 2, or <25% battery. All runs under `scripts/flashnext/real_run_guard.py`.

## Phase 0 — Restore the experiment harness (no benchmarks)

The original scripts (`e1_placement.py` … `e5_memory.py`) were untracked and are gone; only `REPORT.md` and six sanitised result JSONs remain, with the scripts' `out/` artifacts git-ignored. Before any new measurement:

- Recreate placement/verification/latency/overlap/numerics scripts from the `REPORT.md` §8 recipe, tracked this time.
- Add the **native Swift/ObjC Core ML caller** the report flags as unmeasured, so Python `coremltools` marshalling is no longer confounded with ANE dispatch cost.
- Add **hardware power confirmation** (`sudo powermetrics` / IOReport with the right channel) so energy claims are measured rather than inferred from `DevicePowerState` polling.
- Keep all generated models and weights in git-ignored `out/`.

## Phase 1 — Reproduce the baseline, plugged in (needs permission)

Repeat only the measurements needed to confirm the prior numbers still hold on the current tree and on AC power: batch-1 vs batch-16+ placement, latency, and GPU contention. Success criterion: reproduce `REPORT.md`'s direction of effect. If the native caller or power data materially changes the picture, record that before proceeding.

## Phase 2 — Track B feasibility on a 32 GB+ host (needs permission and hardware)

This branch cannot be validated on the owner's 16 GB Mac. Required first gate:

- Confirm the full-residency path actually removes expert-read wait on a 32 GB+ machine (measure ms/token and MB/token before adding ANE).
- Only if decode is genuinely GPU-bound there, build a Core ML model for one static dense sub-layer (attention out-projection is the smallest safe candidate), verify 24/24 bit-identical on a fixed prompt, then measure prefill/decode with ANE on vs off under the guard.

## Phase 3 — Track A, only if a measurable idle-silicon window exists (needs permission)

- Instrument GPU idle time during expert-read stalls. If there is no idle window, stop — ANE would only add contention.
- If a window exists, test an **output-neutral** auxiliary model (candidate: a cache-admission scorer or a speculative drafter with exact verification) with contention as the primary metric, not tok/s.

## Decision rule

ANE is adopted only if it clears the bit-identical gate **and** shows a paired, guard-clean speedup that clears the measurement resolution band. Otherwise it stays a closed experiment, as `REPORT.md` records. Reopening development is not the same as shipping it.

## References

- `REPORT.md` — the 2026-11-07 rejection, raw results in `results/`.
- `reports/Apple Neural Engine integration.md` — the 2026-11-17 deep research synthesis.
- `research_notes/Apple Neural Engine integration/*.md` — the seven underlying note files.
