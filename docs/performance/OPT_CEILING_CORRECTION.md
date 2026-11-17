# IMPORTANT CORRECTION — the "OPT ceilings" were cold-trace artifacts

**Date:** 2026-11-17. **Source:** Opus's v0.2 toolkit (Downloaded, copied into
`veystrix-io-residency/`), specifically `CEILINGS_REVISED.md`. Verified here with real traces.

## The error in my earlier conclusion
I concluded "no residency method reaches 5+/15+ on 16 GB; even a perfect cache caps at
~4 tok/s (Qwen) / ~11 tok/s (Nemotron)." Those OPT numbers came from **68-token (Qwen) and
96-token (Nemotron) traces starting from an empty cache** — dominated by **first-touch
(compulsory) misses** that no cache, however perfect, can avoid. I mistook compulsory
misses for a hard capacity ceiling.

## The signature that proves it (verified on our real traces)
Nemotron OPT hit was 85.4% at 8 GB *and* 10 GB — flat. A perfect cache gaining nothing
from +2 GB means zero capacity misses remain at 8 GB; the entire 14.6% residual is first-touch.
The v0.2 `--prefix-sweep` confirms the mechanism on the real 96-token trace:

| trace length N | OPT hit @6GB | compulsory share | steady-state hit |
|---:|---:|---:|---:|
| 24 | 0.673 | 0.327 | 1.000 |
| 48 | 0.766 | 0.234 | 1.000 |
| 64 | 0.804 | 0.194 | 0.998 |
| 80 | 0.813 | 0.167 | 0.976 |
| 96 | 0.827 | 0.146 | 0.969 |

Hit rises and compulsory share falls with N — exactly as predicted. Qwen shows the same
pattern (compulsory 0.320 → 0.164 from N=24 to 68).

## Revised ceilings (steady-state / long-decode, from the SAME real traces)
| Model | budget | cold tok/s | steady tok/s (S3-FIFO) | steady tok/s (OPT) |
|---|---|---:|---:|---:|
| Nemotron | 6 GB | 10.41 | 10.25 | **15.34** |
| Nemotron | 8 GB | 11.09 | 12.43 | 17.15 |
| Qwen | 8 GB | 3.97 | 3.94 | **6.16** |

**Nemotron 15 tok/s is NOT ruled out** — steady OPT crosses it at ~5.8–6 GB. **Qwen 5 tok/s
is NOT ruled out** — steady OPT reaches 6.16 at 8 GB. Both now hinge on (a) whether a long
decode actually shows the warm-cache reuse, and (b) getting close enough to OPT online.

## What this means for the plan
The door is reopened. Opus's decision sequence (in CEILINGS_REVISED.md §4):
- **A (done here, laptop-only):** prefix sweeps above confirm the window artifact. ✓
- **B (needs permission):** one guarded ≥1,000-token trace per model, plus a separate
  calibration trace, to get the REAL long-decode ceiling and recalibrate G0. Continue
  Nemotron if long-trace OPT predicts ≥12 tok/s and `nextuse` beats S3-FIFO by ≥5 pts;
  continue Qwen if long-trace OPT ≥4.5 tok/s.
- **C:** two A/Bs — single-tier pool (capacity), then `nextuse` policy (policy), digest-gated.
- New `nextuse` policy (learned next-use from gap statistics, bit-exact, lookup-table) beat
  S3-FIFO on synthetic sticky-routing traces; needs long calibration.
- Caveat: our real traces are still short (68/96 tokens). Step B's long trace is what settles
  whether the steady-state reuse actually materializes. That is the critical next run.

## Copied v0.2 toolkit
- Source: `~/Downloads/veystrix-io-residency-v0.2.zip` → `veystrix-io-residency/` (v0.1 backed
  up to `veystrix-io-residency-v0.1-backup/`).
- Tests: **73 passed, 1 skipped** (74 total). My `bridge_flashnext.py` hook preserved.
- New: first-touch/capacity split (`cold`/`steady`), `--prefix-sweep`, `nextuse` and
  `prefill-pinned-s3fifo` policies, `VEYSTRIX_ROUTE_TRACE_PREFILL=1`.
- Analysis outputs: `results/io_residency/20261117/sim_{nemo,qwen}_v02.json`.

## Bottom line
I was wrong to close the residency path. The honest current state: targets are **possible but
unproven**, and one guarded long-trace capture per model will decide it. Recommend proceeding
to Step B (needs owner permission) rather than stopping.
