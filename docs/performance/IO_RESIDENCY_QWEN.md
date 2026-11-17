# IO-residency on Qwen — Stage 0 + simulation, 2026-11-17

Owner: "implement it on qwen first because it has more I/O wait" (Qwen: 156.9 ms/token
absolute I/O wait vs Nemotron's 123.1 ms). Guarded run authorized.

## What was done
- Built Qwen residency index on the real checkpoint: **24,576 experts** (48 layers x 512),
  **3,072,000 bytes/expert** (Q4 G32) — `results/io_residency/20261117/qwen_index.json`.
- Fixed a stale harness model id: `bench_layer_gate.make_prompt` hardcoded `"flashnext"`,
  but this branch serves `installation_id("flashnext", path)`. Patched to compute the id
  (diagnostic harness only; no engine behavior change). First captured run 404'd, second worked.
- **Guarded 1,100-token decode-profile capture** (`bench_decode_profile --arms io`): guard clean,
  no stop, swap growth 0, floor 3.69 GiB. It decoded 68 tokens before EOS. Saved at
  `results/flashnext/20261117-163948-decode-profile-manual/`.
- Converted the real routes to a trace: `results/io_residency/20261117/qwen_trace.jsonl`
  (3,264 rows, 68 tokens, 48 layers, 512 experts).
- Constants from the Qwen host-window profile: effective I/O 408.3 MB / 156.86 ms =
  **2.603 MB/ms**; non-I/O = **143.24 ms/token**.

## G0 calibration
Not passed. The trace is only 68 tokens (package wants >=1,000), and `pagecache-only@2048`
predicts 968 MB/token vs the measured 408 — the simulator's ideal-cache model does not reproduce
the real OS page cache. **No policy prediction from this simulation is trusted.** (A longer trace
is needed; the harness hit EOS at 68 tokens.)

## THE decisive result — OPT ceiling (calibration-independent)
Belady OPT, best any cache can do, using measured constants (2.603 MB/ms, 143.24 ms non-I/O):

| Page-cache/pool budget | Coverage | OPT hit | OPT SSD MB/tok | OPT tok/s |
|---|---|---|---|---|
| 0.5 GB | 0.7% | 29% | 1045 | 1.84 |
| 1 GB | 1.4% | 44% | 823 | 2.18 |
| 2 GB | 2.7% | 57% | 634 | 2.59 |
| 4 GB | 5.4% | 70% | 447 | 3.18 |
| 8 GB | 10.9% | 81% | 283 | **3.97** |

**Even a PERFECT cache at 8 GB yields only ~4 tok/s on Qwen** — below the 5+ target. To hit
5 tok/s, I/O must be <=56.8 ms/token => <=148 MB/token physical; OPT needs far more than 8 GB
coverage to get there, impossible in 16 GB alongside the model. Caveat: this trace is short (68
tokens) and the OPT hit rates here are lower than Nemotron's at similar coverage because Qwen's
512-expert/layer working set is much larger — so the ceiling is if anything optimistic.

## Conclusion
- The I/O-residency / single-tier hypothesis is **implemented and tested on Qwen as far as it can
  go without quality risk**: tools built, trace captured under guard, simulation run.
- Qwen's absolute I/O wait is real (156.9 ms), but it is NOT reachable to 5 tok/s by any cache
  policy or residency trick on 16 GB: even perfect caching caps near 4 tok/s.
- Combined with the Nemotron result (OPT caps ~11 tok/s), **no residency method reaches either
  target on this 16 GB M4.** This is now simulation-backed, not opinion.
- The only paths to the targets remain the owner's deferred levers (more memory / 2nd Mac) or a
  lossy method (forbidden). Recommend reporting this as the final measured answer.

## Files
- `results/io_residency/20261117/`: qwen_index.json, qwen_trace.jsonl, nemo_trace.jsonl, sim_*.json
- `results/flashnext/20261117-163948-decode-profile-manual/`: guarded Qwen trace capture
- `veystrix-io-residency/`: tools + 54 passing tests
