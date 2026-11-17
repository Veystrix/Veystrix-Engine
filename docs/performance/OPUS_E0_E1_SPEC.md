# E0/E1 — the decisive overlap experiments (Opus plan, built 2026-11-17)

Opus's report (`OPUS_MIRACLE_RESPONSE.md`) claims the "impossible on 16 GB"
ceilings come from a *serial* model (`t = compute + bytes/BW`). If decode I/O
and GPU work truly overlap, the floor is `max(compute, bytes/BW)` instead. Our
own "perfect predictor + overlap tied" result is ambiguous: byte-bound (bytes
unchanged, so time unchanged) or no real overlap (a serialization bug). E0 and
E1 separate those. **These must run before any C++ pipeline port or hardware
purchase.**

Harness: `models/flashnext/tests/bench/bench_overlap_oracle.py`
Bit-exact gate: 100% identical token IDs, identical SHA-256 of raw logits at
every step, per-layer residual checksums for the first 64 steps, plus a chaos
mode (random 0–5 ms delays in I/O threads). Reference = production sync path.

All runs need fresh owner permission and the guard:
`scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 2048
--compressor-growth-mib 4096 --log <dir>/guard.jsonl -- <bench>`.
Preserve all `results/**`.

---

## E0 — concurrency proof (run FIRST; ½ day, cheapest)

Does GPU compute contend with a saturated SSD? Pin a fixed expert route so a
decode step is compute-only (zero physical reads), then read 408 MB/token of
cold F_NOCACHE ranges at full speed in the background.

```sh
DIR=$(.venv/bin/python -c 'from veystrix.results import new_run_directory; print(new_run_directory("flashnext","overlap-e0"))')
scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 2048 \
  --compressor-growth-mib 4096 --log "$DIR/guard.jsonl" -- \
  .venv/bin/python -m models.flashnext.tests.bench.bench_overlap_oracle \
    --experiment e0 --tokens 1200 --prompt chat --io-threads 8 --experts 4
# repeat with --chaos
```

- **KEEP** if step inflation <10% AND read bandwidth 3.0–3.7 GB/s: overlap is
  real and the ceilings in `STEP_B_LONG_TRACE_FINAL.md` are too pessimistic.
- **KILL** if inflation >25% OR bandwidth <2.4 GB/s (>20% drop): GPU and SSD
  contend; overlap cannot deliver the targets on this machine — go to the
  hardware fallbacks (second SSD, second Mac, 24 GB) in Opus §5.
- Gate must pass in both arms (identical token IDs / logits / layer checksums)
  including under chaos. A divergence is a slot-reuse race, not a result.

## E1 — oracle pipeline

Pass 1 records the REAL greedy route of every MoE call in execution order.
Pass 2 replays those exact routes as prefetch hints 1–3 calls ahead (buffered
pread fills the page cache) while the **real router still decides** — so
routing is untouched and the output is bit-exact by construction. This is the
upper bound of any trained predictor, with zero training.

```sh
DIR=$(.venv/bin/python -c 'from veystrix.results import new_run_directory; print(new_run_directory("flashnext","overlap-e1"))')
scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 2048 \
  --compressor-growth-mib 4096 --log "$DIR/guard.jsonl" -- \
  .venv/bin/python -m models.flashnext.tests.bench.bench_overlap_oracle \
    --experiment e1 --tokens 1200 --warmup 100 --prompt chat \
    --ahead-depth 3 --ahead-width 10 --io-threads 8
# repeat --prompt code, --prompt agentic, and with --chaos
```

- **KEEP** if gate passes AND oracle ≥5.5 tok/s AND critical-path io_await
  ≤30 ms: a real predictor (Opus §4, hint-only) is worth building.
- **KILL** if oracle <4.5 tok/s: no predictor can save this path; the bottleneck
  is bytes, so only more RAM / faster disk / fewer bytes (lossless only) help.
- Report per run: tok/s, ms/token, critical-path io_await, physical MB/token,
  prefetch MB issued, swapouts (must be 0), sustained rate after 5 min (fanless).

---

## What each outcome unlocks (Opus KEEP/DROP)

- **E0 KEEP + E1 KEEP** → build the event-gated C++ pipeline (Opus lead F
  generalized): one `ExpertFetch` primitive per layer on the CPU stream, GPU
  waits on cross-stream events, single-sync slot remap. Hint-only predictor
  (never committed routing) keeps the bit-exact rule. Then revisit MTP as an
  I/O batching window once decode is compute-bound.
- **E0 KILL** → no software overlap; skip the C++ pipeline. Go to hardware
  fallbacks (second SSD ~5.5–6.5 GB/s aggregate; second Mac over TB4; 24 GB).
- **E0 KEEP + E1 KILL** → overlap possible but prediction not the lever; try
  the exact LRU slot cache (lead E, llama.cpp #27861, +31% on this model) and
  the pool-as-sole-cache + F_NOCACHE idea (lead E/Opus §2.3) for byte reduction.

## Reasons Opus gave for deferring/dropping (carry into the decision)

- DEFER SpecFit / MTP / DraftExpert while I/O-bound: they raise unique expert
  bytes per step, which our M4 bandwidth analysis says loses.
- DROP committed-routing Edge0 (breaks the no-quality-loss rule); use it only
  as a hint. DROP base-expert residuals (expected >4 bits/weight); DROP PagedMoE
  kernel (MLX already reads Q4 codes directly); DROP ZipMoE as a headline win
  (~10% cap).
- Reconcile the server-vs-profile gap (2.3 tok/s production vs 3.33 short
  harness) separately — it may cost more than any lead, but it is a workload
  difference, not yet proven to be a hidden bug.

## Harness internals (for reviewers / Codex)

- `wrap_rows_np`: times `store.rows_np` on the main thread (critical path) vs
  pool threads (background prefetch); injects chaos delays only off-thread.
- `install_switch_patch`: installed once, behaviour switched by state flags;
  clears `_routed_host` so forced/recorded routes are not replaced by a stale
  one-sync host list (same defect guard as `bench_read_ceiling.py:71`).
- `install_layer_patch`: per-layer residual checksums for the first 64 steps.
- `run_decode` / `gate`: token IDs + canonical signed-int32 token digest + raw
  logit SHA-256 (float32 little-endian, as `logit_reference.py`) + layer digests.
- Output JSON via `veystrix.results.output_path("flashnext","overlap-oracle",...)`.
