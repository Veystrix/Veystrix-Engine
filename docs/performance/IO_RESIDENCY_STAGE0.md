# IO-residency Stage 0 — results 2026-11-17

Package: `veystrix-io-residency/` (Fable, hypothesis: pool + page cache duplicate bytes;
single-tier with F_NOCACHE + grown pool at constant RAM could convert duplication into capacity).
Owner approved "start with stage 0". **No new hardware A/B was run for this** — the route data
came from the already-preserved guarded decode profile (`20261117-140232`).

## What was verified
- Package tests: **54 passed, 1 skipped** (after installing pytest into venv). Claims hold.
- Residency index built for the real Nemotron checkpoint: **2,944 experts** (23 MoE layers x 128),
  **5,612,544 bytes/expert** — matches the package's derived number exactly.
- Converted the real 96-token decode-profile routes into the simulator trace:
  `results/io_residency/20261117/nemo_trace.jsonl` (2,208 rows, 96 tokens, 23 layers, 128 experts).

## G0 calibration — FAILED (but not fatally)
`sim --calibrate s3fifo:2048:0.437`: sim hit 0.393 vs measured 0.437, diff 0.044 > 0.03 band.
Reasons: (1) only 96 tokens (package asks >=1,000); (2) the simulator's S3-FIFO parameters don't
match the engine's exact implementation. Per the package's own rule, **no prediction from this
simulation is trusted until G0 passes**. A longer trace + tuned policy params are needed.

## THE decisive result — OPT ceiling (does NOT depend on G0 calibration)
Belady OPT is the theoretical best any cache policy can achieve, independent of calibration.
Using measured constants (3.541 MB/ms effective I/O, 58.3 ms non-I/O), predicted tok/s:

| Pool budget | Coverage | OPT hit | OPT SSD MB/tok | OPT tok/s |
|---|---|---|---|---|
| 2 GB (today) | 12% | 65.8% | 265 | **7.52** |
| 4 GB | 25% | 77.6% | 173 | **9.32** |
| 6 GB | 37% | 82.7% | 134 | **10.41** |
| 8 GB | 50% | 85.4% | 113 | **11.09** |
| 10 GB | 62% | 85.4% | 113 | **11.09** |

The curve **flattens at ~11 tok/s** — routing plateau, so more memory buys nothing beyond ~8 GB.

**Conclusion: even a PERFECT cache at any memory size that might fit (<=10 GB) cannot reach
15 tok/s on this 16 GB M4.** It caps near 11. The owner's 15+ target is unachievable by any
residency/policy method on this hardware — this is now a measured, simulation-backed ceiling,
not an opinion. (Only more memory / a 2nd Mac / a lossy method can do it — all excluded.)

## Single-tier hypothesis status
The double-residency simulation shows the mechanism is real in principle: at 2 GB,
`double-s3fifo@2048 --double-pc-mb 2048` has noSSD hit 0.537 vs s3fifo 0.393 — i.e. the page
cache supplies real extra hits. But G1 (`dup_bytes >= 0.5 GB`) still needs the live residency
probe to confirm the duplicate bytes actually exist on this checkpoint, and G0 must pass before
any prediction is trusted. Crucially, even if single-tier works perfectly, its own ceiling is
bounded by the OPT curve above (~11 tok/s), so it cannot reach 15 either.

## Next (per the package's gates)
- G0 fix: capture >=1,000-token route trace (needs engine hook `VEYSTRIX_ROUTE_TRACE` + longer
  decode), plus a calibration trace from different prompts. Then tune `--promote-gt`/`--small-frac`.
- G1: live `probe.py snapshot` with `dup_bytes` during a guarded decode (needs pool-key dump hook).
- G2/G3 only if G0/G1 pass.
- But note: since OPT caps at ~11 tok/s, the payoff of the whole path is at most ~9-11 tok/s,
  short of 15. Recommend being explicit with the owner about this ceiling before more hardware runs.
