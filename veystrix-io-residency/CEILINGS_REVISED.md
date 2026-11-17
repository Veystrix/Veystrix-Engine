# The OPT "ceilings" in the handoff are not ceilings yet

Labels: **[measured]** guarded runs in the handoff · **[derived]** arithmetic on measured numbers ·
**[cited]** papers · **[synthetic]** this toolkit on generated traces (mechanism only) · **[idea]**.
No hardware run backs this document. It says which run would settle the question and how.

## 1. The gap in the constraint

The handoff concludes "even a perfect cache caps at ~11 tok/s (Nemotron) / ~4 tok/s (Qwen)". Those
OPT figures come from **96-token (Nemotron) and 68-token (Qwen) decode traces that start from an empty
cache**. A trace that short contains many **first-touch (compulsory) misses**: experts that
are seen for the first time inside the window. Every cache misses those, whatever its size or
policy. In a long decode, or a decode whose pool was warmed by the prefill that just streamed those
same experts, they mostly disappear.

**The signature is in the handoff's own table.** Nemotron OPT is 85.4% at 8 GB *and* 85.4% at 10 GB
**[measured]**. If adding 2 GB to a perfect cache recovers nothing, OPT at 8 GB already has **zero
capacity misses** in this window. The entire 14.6% residual is first-touch **[derived]**. So the
96-token window splits cleanly:

| pool | cold OPT (handoff) | tok/s | capacity-miss share | OPT over non-first-touch accesses | tok/s |
|---|---|---|---|---|---|
| 2 GB | 65.8% | 7.51 | 19.6% | 77.1% | 9.22 |
| 4 GB | 77.6% | 9.32 | 7.8% | 90.9% | 12.78 |
| 6 GB | 82.7% | 10.40 | 2.7% | 96.8% | **15.33** |
| 8 GB | 85.4% | 11.08 | 0% | 100% | 17.15 |

The tok/s columns use the handoff's own constants: 774.5 MB logical/token, 3.541 MB/ms, 58.3 ms
non-I/O **[derived]**. The 15 tok/s line needs ≥ 96.2% hits, which this bracket crosses at
about **5.8 GB** of unique expert RAM **[derived, linear interpolation]**.

**Honest reading.** The truth lies between the two tok/s columns. The right column is optimistic in one way:
a 96-token window cannot show reuse distances longer than 96 tokens, so long decodes add capacity
misses that this window hides. The left column is pessimistic: it charges every first touch to the cache.
**15 tok/s on Nemotron is not ruled out. It is also not shown.** One longer trace settles it.

**Qwen.** The handoff numbers can't be split the same way (no flat segment). Qwen 5 tok/s needs ≥ 90.0%
of lookups served without the SSD **[derived]**. That is reachable at 8 GB if first-touch misses are ≥ 10% of
the 68-token window, or at 4 GB if they are ≥ 22% **[derived]**. `sim.py` now prints that share
directly, from the trace you already have, in seconds. For context, today's production page cache
already serves 72.3% of Qwen's 1,474.6 MB/token (408 MB physical) **[derived]**.

## 2. Getting close to OPT online

OPT needs the future. The engine's S3-FIFO is 22.1 points below OPT at 2 GB on Nemotron (43.7% vs
65.8%) **[measured/derived]**. That gap is where policy work pays. FlashMoE (arXiv 2601.17063) approximates
Belady with a learned model on SSD-backed MoE and reports up to 51% higher hit rate than LRU/LFU
**[cited]**. SpecMD (arXiv 2602.03921) shows why recency fails here: layers are visited in a fixed
sequential order, not by recency-based reuse **[cited]**.

**New policy `nextuse` (this toolkit).** Time is counted in *tokens*, because every layer is visited
once per token. For each resident expert it predicts the next-use token from E[gap | gap ≥ age],
conditioned on a fast decayed count (~10-token memory) and a slow one (~100-token memory). It learns
those gap statistics from a **separate calibration trace** and evicts the latest predicted next use. A lookup
table, no ML runtime. It changes only residency, never routing, so outputs are bit-identical. It is
not the rejected least-stale or cache-aware routing: least-stale is a fixed heuristic, and cache-aware
routing changed the routes.

On synthetic sticky-routing traces **[synthetic]** it beats S3-FIFO at every tested size:

| synthetic shape | S3-FIFO | nextuse | OPT | share of the S3-FIFO→OPT gap recovered |
|---|---|---|---|---|
| Nemotron shape, 2 GB | 58.3% | 67.9% | 79.6% | 45% |
| Nemotron shape, 4 GB | 76.7% | 79.0% | 87.5% | 21% |
| Qwen shape, 8 GB | 71.2% | 74.4% | 82.1% | 29% |

Three caveats. It needs a long calibration trace: fitted on 32 tokens it lost to S3-FIFO at 4–8 GB
**[synthetic]**. With fewer slots than one token's lookups, LRU collapses to 0 hits and nextuse doesn't.
Near one token's working set, LRU can edge it. These are mechanism checks, not predictions. Your traces decide.

## 3. The memory question, which matters as much as the policy

Every row above assumes that RAM is available for *unique* expert bytes. The measured "bigger pool
regresses" and "3 GB hits a 2.59 GiB floor" results were taken with pool *and* page cache both holding
experts **[measured]**. The single-tier pool (F_NOCACHE, pool grown by the measured duplicated bytes)
is the way to find out how much unique RAM 16 GB really gives. **Please paste the Stage 0 probe
numbers (page-cache expert GB, `dup_bytes`) and the diskstats output.** They were not in the
handoff, and they set the budget column for every decision below.

## 4. Decision sequence, with pass/fail thresholds set in advance

**A. Laptop only, existing traces, minutes, no permission needed.**
```bash
python -m veystrix_residency.sim --trace <nemo_96tok.jsonl> --expert-bytes 5612544 \
  --budgets-mb 2048,4096,6144,8192 --policies s3fifo,opt \
  --prefix-sweep 24,48,64,80,96 --sweep-budget-mb 6144 --io-mb-per-ms 3.541 --non-io-ms 58.3
python -m veystrix_residency.sim --trace <qwen_68tok.jsonl> --expert-bytes 3072000 \
  --budgets-mb 2048,4096,8192 --policies s3fifo,opt \
  --prefix-sweep 24,36,48,60,68 --sweep-budget-mb 8192 --io-mb-per-ms 2.603 --non-io-ms 143.24
```
Leave `nextuse` out of step A. In a dry run with only 32 calibration tokens it *lost* to S3-FIFO at
4–8 GB **[synthetic]**; with a 1,000-token calibration trace it won at every size. Its gap tables
need long calibration, so it is judged in step B only.
- **Pass:** the `cold` column reproduces about 14.6% for Nemotron, and the prefix sweep shows OPT's hit rate
  *rising* with N while the first-touch share falls. That confirms the window explanation.
- **Kill:** OPT is flat across N **and** the first-touch share is < 5%. Then the handoff ceilings stand and
  residency cannot reach the targets; record that and stop this path.

**B. One guarded trace run per model.** No output change; needs your permission.
`VEYSTRIX_ROUTE_TRACE=<path> VEYSTRIX_ROUTE_TRACE_PREFILL=1`, ≥ 1,000 decode tokens, plus a separate
calibration trace from different prompts. This one run gives the real ceiling and recalibrates G0
(the earlier G0 failure was partly trace length).
- **Continue (Nemotron)** if long-trace OPT at the probe-measured unique budget predicts ≥ 12 tok/s
  and nextuse beats S3-FIFO by ≥ 5 points. Otherwise DROP the residency path for the 15 target.
- **Continue (Qwen)** if long-trace OPT at that budget predicts ≥ 4.5 tok/s (5.0 needs policy *plus*
  either overlap or bandwidth gains, see §5).
- Also check `prefill-pinned-s3fifo`: pinning what the prompt's own prefill used most
  (KTransformers' "PrefillHot" per SeqMoE, arXiv 2609.12978 **[cited]**). It is the cheapest way to cut
  first-touch misses in a real request.

**C. Two A/Bs, one change each**, token+logit digest gate, guarded:
1. `VEYSTRIX_SINGLE_TIER_POOL=1 VEYSTRIX_POOL_BUDGET_MB=<B>` (S3-FIFO) vs baseline. This tests capacity.
2. Add `VEYSTRIX_POOL_POLICY=nextuse VEYSTRIX_POOL_NEXTUSE_CALIB=<calib trace>` vs C1. This tests policy.

KEEP each only if physical MB/token falls (diskstats), tok/s rises beyond the resolution band, and the guard
is clean.

## 5. The other directions, briefly
- **Overlap becomes viable at high hit rates [idea].** Prior overlap/prefetch ties were measured while I/O
  (123–157 ms) was about 2× non-I/O. At 90%+ hits, Nemotron's remaining I/O (≤ 22 ms) is far below its 58 ms of
  compute. Next-layer router prediction (recall@10 83.6% on Qwen **[cited]**) could then hide most of it, so
  the serial constant model would be pessimistic there. Re-test overlap *after* C, not before.
- **Layout [derived].** A Qwen expert is 9 slices (3 × 819,200 + 6 × 102,400 B), 187.5 pages of 16 KB.
  The upside is bounded by the effective 2.6 → ~3.0–3.3 GB/s, so ≤ ~25% of I/O time, and only if
  diskstats shows small mean reads. That's not a 2× lever, but it composes with residency. The earlier packing
  win vanished in the server, plausibly because a duplicate file copy split the page cache. A
  single-tier F_NOCACHE pool removes that mechanism.
- **Speculative/MTP.** Union reads made verification slower (0.79–0.83 union ratio). That only
  changes once hit rates are high. Revisit after C.
- **Compression / precision.** Closed (lossy, or 0.9–2.1% lossless) **[cited in handoff]**.

## 6. What changed in the toolkit (v0.2, 74 tests)
- `sim.py`: `cold`/`steady` columns (first-touch vs capacity split), `--prefix-sweep`,
  `--calib-prefix-tokens`, policies `nextuse` and `prefill-pinned-s3fifo`.
- `policy.py`: `NextUseModel` / `NextUse`, `compulsory_flags`.
- `trace.py`: `VEYSTRIX_ROUTE_TRACE_PREFILL=1` writes one prefill-count line per request.
- `pool.py`: `VEYSTRIX_POOL_POLICY=nextuse` + `VEYSTRIX_POOL_NEXTUSE_CALIB`. Tests show it is
  byte-exact under eviction and matches the simulator's hit/miss sequence exactly.
