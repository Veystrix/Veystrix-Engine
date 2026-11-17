# Veystrix I/O residency: one hypothesis, three diagnostics, one A/B

> **v0.2:** see `CEILINGS_REVISED.md`. The handoff's OPT ceilings mix first-touch misses from 68/96-token
> traces into the bound. v0.2 adds the first-touch/capacity split, a trace-length sweep, the `nextuse`
> policy, and prefill-count capture. 74 tests.

Evidence labels: **[measured]** = from the brief's guarded runs, **[derived]** = arithmetic on
measured numbers only, **[cited]** = external evidence quoted in the brief, **[estimate]**,
**[idea]**. Nothing in this package has been run on the M4 yet. There are no before/after
numbers here, only tools, tests, and pass/fail checks set in advance.

## 1. What the existing numbers already say

**Nemotron bytes add up exactly.** 41,847 MB / 7,456 misses = 5.6125 MB per miss **[measured]**.
One up+down expert at Q4 G64 is 2 × 2688 × 1856 weights × 4.5 bits = 5,612,544 B **[derived]**.
So every pool miss reads one whole expert. The logical demand is 23 layers × 6 × 5.6125 MB =
**774.5 MB/token**, and 43.7% hits leave 436 MB **[derived, matches measured]**.

**The drive is the wall.** 436 MB in 123.1 ms is 3.54 GB/s **[derived]**. That is *above* the
3.0–3.3 GB/s sequential figure, so some "misses" are probably already being served by the
OS page cache **[estimate]**. Prefetch, overlap, 6–32 workers and a perfect predictor all tied
**[measured]**. All of that only moves *when* bytes are read. With the drive saturated, only
fewer bytes per token help.

**Coverage is under the break-even.** If Lightning keeps Nemotron-3-Nano's 128 routed
experts/layer (check `n_routed_experts`), routed experts total 16.5 GB, a 2 GB pool covers 12%,
and the ~1/3 break-even **[cited]** sits at 5.5 GB **[derived]**.

**One cause fits two "dead ends" [idea, this is the hypothesis].** pread fills the in-process
pool *and* the OS page cache. The page cache is a hidden second expert cache, and it mostly
holds what the pool just admitted.
- *F_NOCACHE 8% slower:* that was measured where the pool is auto-sized to 0. F_NOCACHE then
  removes the **only** cache, so it should lose.
- *Bigger pool regresses:* every pool MB copies bytes the page cache already holds, so there is
  little new unique capacity, plus extra memory pressure.

The configuration nobody measured is **one tier**: F_NOCACHE *plus* a pool grown by exactly the
page-cache share, at the same total RAM. This is not a re-run of either rejected A/B; it is
the combination neither tested.

**The main way this fails [estimate].** macOS can drop page-cache pages for free. Pool pages
are anonymous memory and get *compressed* instead. So turning cache into pool memory can raise
pressure even at equal bytes. That may be the real reason bigger pools lost. The guard's
compressor/swap metrics decide this, which is why the A/B is guarded and sized to measured
duplication, not guessed.

### Ceilings (be honest about 15 tok/s)
Nemotron, using measured non-I/O time of 58.3 ms/token and 3.541 MB/ms **[derived]**:

| hit rate | 43.7% (today) | 60% | 70% | 80% | 90% | 95% | 100% |
|---|---|---|---|---|---|---|---|
| io ms/token | 123.1 | 87.5 | 65.6 | 43.7 | 21.9 | 10.9 | 0 |
| tok/s | 5.51 | 6.86 | 8.07 | 9.80 | 12.47 | 14.44 | 17.15 |

15 tok/s needs ~96% hits *and* today's 58.3 ms non-I/O, or a smaller hit rate plus a cut to
`route_sync_and_copy` (50.5 ms, which includes GPU wait). Better residency alone gets
roughly 8–12 tok/s.

Qwen: non-I/O = 143.2 ms. 5 tok/s leaves 56.8 ms for I/O, i.e. ≤ 148 MB/token physical
(−64%) **[derived]**. Qwen's pool is already 0, so it has no duplication to reclaim. Its lever is
**policy**: the kernel's recency-style eviction versus S3-FIFO/LFU with frequency pinning at the
same RAM. Routing is unchanged, so this is bit-exact by construction.

## 2. Plan

### Stage 0: diagnostics (no output change; one guarded run per model, with your OK)
1. **Routing trace.** Set `VEYSTRIX_ROUTE_TRACE=<path>` and add 3 lines next to the existing
   `route_tolist` (see `trace.py` docstring). Decode-only. Capture ≥ 1,000 tokens, plus a separate
   **calibration** trace from different prompts (used for pins).
2. **Physical I/O.** `diskstats.py` wraps the benchmark. It gives physical MB/token, read ops/token
   and mean read size, which separate IOPS-bound from bandwidth-bound.
3. **Residency snapshots.** `probe.py snapshot` runs from a second shell every ~5 s
   (mmap+mincore, no I/O). With the pool key list dumped by the engine (`resident_keys()`),
   it reports `dup_bytes`. Without it, page-cache expert bytes is an upper bound.

### Stage 1: simulate (no hardware)
`sim.py` replays the trace through LRU / S3-FIFO / LFU / pinned-S3-FIFO / OPT (the hit-rate
ceiling) / double-residency (pool + duplicating page cache). It converts SSD MB/token to
predicted ms and tok/s with your measured constants.

### Stage 2: one A/B (opt-in flag, off by default)
`VEYSTRIX_SINGLE_TIER_POOL=1`, `VEYSTRIX_POOL_BUDGET_MB=<old pool + measured dup>`,
`VEYSTRIX_POOL_NOCACHE=1`, optional `VEYSTRIX_POOL_PIN_FILE`. Reference implementation:
`pool.py`. It uses the same `S3FIFO` class as the simulator (a test proves identical hit/miss
sequences), so the A/B checks the simulator's prediction directly.

## 3. Pass/fail checks, set before any run

| check | pass condition | on fail |
|---|---|---|
| G0 simulator calibration | `--calibrate s3fifo:2048:0.437` within ±3 pts on the Nemotron trace (Qwen: `pagecache-only` at measured page-cache MB reproduces 408 MB/token ±10%) | tune `--promote-gt`/`--small-frac` to match the engine's s3fifo; trust no prediction until it passes |
| G1 capacity exists | Nemotron @2 GB: `dup_bytes` ≥ 0.5 GB (Qwen: page-cache expert bytes ≥ 1 GB) | **DROP** single-tier: there's no hidden RAM to reclaim. Record the number |
| G2 payoff | single-tier at (2048 + dup) MB predicts ≥ 15% fewer SSD MB/token than `double-s3fifo@2048 --double-pc-mb <measured>` | **DROP** before hardware |
| G3 hardware A/B | token digests identical (and logit digests where available) on all prompts; guard clean; physical MB/token down (diskstats); tok/s up beyond the resolution band | **DROP**, keep result dirs |

Also record OPT at the largest budget the guard allows. That is the best any cache policy can do
on 16 GB, and it gives the 15 tok/s question a hard answer either way.

## 4. Commands (each needs your permission; outputs under `results/io_residency/<date>/`)

```bash
R=results/io_residency/$(date +%Y%m%d)
G="python scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 2048 --compressor-growth-mib 4096"

# Stage 0, Nemotron (replace <BENCH> with the existing 96-token guarded decode command)
$G --log $R/guard_nemo_diag.jsonl -- env VEYSTRIX_ROUTE_TRACE=$R/nemo_trace.jsonl \
   python -m veystrix_residency.diskstats --tokens 96 --out $R/disk_nemo_base.json -- <BENCH>
# second shell, during decode:
python -m veystrix_residency.probe build-index --model-dir <nemotron dir> \
   --pattern '<regex with (?P<layer>\d+) for routed-expert tensors>' --out $R/nemo_index.json
while sleep 5; do python -m veystrix_residency.probe snapshot --index $R/nemo_index.json \
   --pool-keys $R/pool_keys.json --out $R/resid_$(date +%s).json; done

# Stage 1 (laptop-only, no model loaded)
python -m veystrix_residency.sim --trace $R/nemo_trace.jsonl --expert-bytes 5612544 \
  --budgets-mb 1024,2048,3072,4096,5120 --policies s3fifo,lru,lfu,pinned-s3fifo,opt,double-s3fifo \
  --pin-trace $R/nemo_calib_trace.jsonl --double-pc-mb <measured page-cache MB> \
  --io-mb-per-ms 3.541 --non-io-ms 58.3 --calibrate s3fifo:2048:0.437 --out $R/sim_nemo.json
python -m veystrix_residency.pins --trace $R/nemo_calib_trace.jsonl --expert-bytes 5612544 \
  --budget-mb <B> --pin-frac 0.5 --out $R/pins_nemo.json

# Stage 2: arm B (arm A = Stage 0 baseline, same prompts, same build)
$G --log $R/guard_nemo_single.jsonl -- env VEYSTRIX_SINGLE_TIER_POOL=1 VEYSTRIX_POOL_BUDGET_MB=<B> \
   VEYSTRIX_POOL_NOCACHE=1 VEYSTRIX_POOL_PIN_FILE=$R/pins_nemo.json \
   python -m veystrix_residency.diskstats --tokens 96 --out $R/disk_nemo_single.json -- <BENCH>
```

## 5. Hooking it into the engine (for Codex)

I had no repo access, so these are integration points, not a diff:
- **Trace:** next to the `route_tolist` call (48/token Qwen, 23/token Nemotron): `_TRACE.record(layer, ids)`;
  after each sampled token: `_TRACE.end_token()`.
- **Pool:** where the current s3fifo pool resolves (layer, expert) → buffer. If `from_env(ranges)`
  returns a pool, call `get_batch(keys)` per MoE layer. `read_into` should write into the existing
  Metal-backed slot memory (pass it as `backing`). The native C path can reuse `native_read`. Wire
  `wait_slot_free` to the existing fence that guarantees the previous layer's command buffer is
  done with a slot. **This is the one real correctness hazard**, so test it under the digest gate.
- **Pool key dump** for the probe: write `json.dump(pool.resident_keys())` every 32 tokens.

## 6. Other leads the probe answers for free
- **Why expert packing showed no server gain [idea]:** +70.5 GiB disk means a second copy. If any
  path (prefill, warmup, the shared expert) still reads the original shards, the page cache splits
  across two copies. A probe index over *both* file sets shows this directly.
- **IOPS vs bandwidth:** if `mean_read_kb` is small and ops/token are high, the gap-merged pread
  (already built, opt-in) and page-aligned slots are the next decode lever. If reads are large and
  near 3 GB/s, they're not.

## 7. Tests (v0.1 core: 55; v0.2 total 74, all passing on Linux/Python 3.13; macOS paths are guarded by platform checks)
`python -m pytest -q tests`. The tests cover: OPT ≥ every policy on 24 trace × capacity combinations (some with drifting popularity);
LRU monotone in capacity; a routed expert is never evicted during its own layer; pinned experts
are read at most once; pool bytes equal on-disk bytes after every batch under eviction
pressure; pool hit/miss sequence equals the simulator's; the flag is inert when unset; mincore
sees a real cache drop and re-read; the ioreg parser and per-token math.
Not testable here: F_NOCACHE (macOS only), the ioreg call itself, and Metal-slot fences.
