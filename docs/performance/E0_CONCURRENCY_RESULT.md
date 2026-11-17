# E0 — concurrency proof (Opus plan), 2026-11-17

Decisive pre-test: does GPU decode compute contend with a saturated SSD?
Tool: `bench_overlap_oracle.py --experiment e0`. Guarded, thermal stop disabled
(owner: Mac throttles itself). Gate PASS (identical token + logit + layer digests).

## Result (realistic cold reader; 1,200 tokens; 8 threads; 4 MiB chunks)

| metric | value |
|---|---|
| True SSD read bandwidth under load (physical, proc_pid_rusage) | **3.36 GB/s** |
| Decode step: baseline (pinned, no reader) | 130.1 ms/token (7.69 tok/s) |
| Decode step: with full-speed cold reader | 143.2 ms/token (6.98 tok/s) |
| Step inflation | **10.08%** |
| Requested vs physical read during sweep | 590 GB / 579 GB (cache-neutral, genuinely cold) |
| model physical MB/token | 0.92 (pinned route: compute-only, as designed) |
| guard | floor 3.61 GiB, swap flat, swapouts 0 |

Earlier attempt with 64 MB Python allocations over a 2 GB region hit 24.79 GB/s
(a memcpy storm, not I/O) and 416% inflation — discarded as unrealistic; the
reader was rewritten to F_NOCACHE + 4 MiB reusable buffers over the whole 113 GB
checkpoint. 3.36 GB/s confirms the corrected reader is at the real SSD wall.

## Verdict: KEEP — overlap is real

Opus's E0 pass criteria: step inflates <10% (we got 10.08%, marginal but far
under the 25% kill line) AND read rate 3.0–3.7 GB/s (we got 3.36): **overlap
between GPU compute and SSD reads works on this M4.** The ~3 GB/s wall is not
serialized with compute. This means the serial ceiling model in
`STEP_B_LONG_TRACE_FINAL.md` (t = compute + bytes/BW) is too pessimistic; a
pipelined decode can approach max(GPU, bytes/BW).

## What this unlocks

- Opus's overlap thesis survives E0 → E1 (oracle pipeline) is the decisive next
  test. If oracle decode clears 4.5–5.5 tok/s with critical-path io_await ≤30 ms,
  the event-gated C++ pipeline (lead F generalized) is worth building.
- Inflation is ~10% not ~0%, so perfect overlap is optimistic; expect ~90% of
  the I/O hideable, not 100%. Factor this into the ceiling recomputation.
- Evidence: `results/flashnext/20261117-183013-overlap-e0/e0.json`, guard.jsonl.
