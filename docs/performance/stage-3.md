# Flash-Next Stage 3: cost-aware pool policy and decode prefetch

Stage 3 adds a fourth expert-pool eviction policy, `cost-aware`, and an
opt-in decode prefetch. Both are off by default. No real-model run has been
made: the only numbers in this document come from an offline replay of one
recorded route trace (see "Not yet measured").

## What it adds

**Cost-aware score.** `models/shared/cost_aware.py` scores each resident
record; the lowest score is evicted (ties go to the older last use, then the
lower slot):

```
score = recent * [used within the last 48 layer accesses]
      + frequency * p
      + reuse * (1 - (1 - p)^visits)
      - size                          (+ refill * relative refill cost, when refill > 0)
```

`p` is a decayed route probability for the (layer, expert) pair: at each
visit to a layer its row is multiplied by `1 - alpha` and `alpha` is added to
the routed experts. `visits` is how many times the record's layer comes round
inside the pool's survival horizon, `slots / max(miss_rate, 0.5)` accesses,
where `miss_rate` is a running average of misses per access (initially 8).
The table is keyed by (layer, expert), not by slot, so it persists across
evictions. That matters for pools smaller than one token's working set: the
recorded trace routes about 380 records per token (1221.4 MB at 3.21 MB each),
so a pool of 155 or 311 slots is evicted through before a layer comes round
again, and a recency policy never sees a hit (LRU decodes at 0.0% at both
sizes below). The persistent table lets the pool keep the records most likely
to be routed again instead of the most recent ones. `refill` and `size` are
inert by default (both 0.0). `size` is a constant here because all records
have the same size, so it cannot change an ordering. `refill` needs per-layer
read timings; the runtime only collects them when `refill` is non-zero, and
the trace has no timing to calibrate it against.

**Decode prefetch.** With `FLASHNEXT_EXPERT_PREFETCH=1`, N staging slots are
carved out of the pool's own mapping. The mapped bytes do not change and decode
capacity drops by N. The count is capped at one eighth of the pool's slots
(`ExpertPool.__init__`). After each layer's routing, records predicted for the
next `DEPTH` layers (probability at least `MIN_P`, at most `WIDTH` per target
layer, not already resident) are queued for reads into free staging slots. A
read queued without a free slot may displace the oldest completed entry whose
deadline has passed. Reads run on `WORKERS` threads in deadline order (current step plus
layers ahead). With `GATE=1`, queued prefetch reads are held while a
demand read is in flight. When the router later demands a staged record, the
pool copies it from its staging slot into the slot it assigned instead of
reading the SSD; a read still running is waited for and counted `late`, and a
failed or cancelled one falls back to a direct read. Memory is bounded by N
records inside the pool the run already has.

## Why neither can change output

Both change only which bytes are resident and when they were read. A record in
a pool slot is the same bytes from the same source whether it was read on
demand, read by the prefetch, or kept from an earlier token, and the prefetch
copy path falls back to a direct read when a staged record is not ready.
This is from reading the code. The paired bench's token and first-logit digest
gates check it on the real model; that run has not been made.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `FLASHNEXT_EXPERT_POOL_POLICY` | `lru` | `lru`, `s3fifo`, `least-stale` or `cost-aware` |
| `FLASHNEXT_EXPERT_POOL_COST_WEIGHTS` | unset | JSON file, format of `CostWeights.to_mapping()`; read only for `cost-aware` |
| `FLASHNEXT_EXPERT_PREFETCH` | `0` | `1` enables the decode prefetch |
| `FLASHNEXT_EXPERT_PREFETCH_SLOTS` | `16` | staging slots taken from the pool (capped at one eighth of it) |
| `FLASHNEXT_EXPERT_PREFETCH_DEPTH` | `2` | layers ahead to predict for |
| `FLASHNEXT_EXPERT_PREFETCH_WIDTH` | `2` | records per target layer per call |
| `FLASHNEXT_EXPERT_PREFETCH_MIN_P` | `0.5` | minimum predicted probability |
| `FLASHNEXT_EXPERT_PREFETCH_WORKERS` | `2` | prefetch read threads |
| `FLASHNEXT_EXPERT_PREFETCH_GATE` | `1` | hold prefetch reads while a demand read is in flight |

Pool statistics appear in `/health` under
`cache.flashnext_kv.expert_pool_stats` (hits, misses, evictions, and for the
prefetch `issued`, `used`, `late`, `cancelled`, `expired`, `wasted`, `failed`,
`staging_slots`, `staging_bytes`); `expert_pool_policy` and
`expert_pool_mapped_bytes` sit beside it.

## Offline results

Trace: `results/flashnext/20260922-221213-route-trace-manual/route-trace.json.gz`
(record 3,211,264 bytes, 3 turns, 768 decode tokens, 36,864 decode rows). The
replay starts with an empty pool, includes the cold fill, and counts only
decode rows. `cost-aware` drives the runtime's `CostAwarePolicy` in the
runtime's access order (advance, hits, victims, inserts, observe). MB/token is
modelled demand: misses x record bytes / decode tokens (10^6 bytes). It is
bytes, not time.

```sh
.venv/bin/python -m models.flashnext.tests.bench.bench_pool_policy_trace \
  results/flashnext/20260922-221213-route-trace-manual/route-trace.json.gz \
  --gb 0.5 1 2 --predict --json
```

Result: `results/flashnext/20261113-190612-pool-policy-trace-manual/pool-policy-trace.json`.
Decode hit rate and modelled demand MB/token:

| Pool (slots) | lru | s3fifo | least-stale | cost-aware | belady (bound) |
|---|---|---|---|---|---|
| 0.5 GB (155) | 0.0% / 1221.4 | 0.0% / 1221.4 | 0.0% / 1221.4 | 23.3% / 936.8 | 34.9% / 795.4 |
| 1 GB (311) | 0.0% / 1221.4 | 32.5% / 824.1 | 24.6% / 920.7 | 37.5% / 762.9 | 53.9% / 563.6 |
| 2 GB (622) | 46.6% / 651.8 | 47.3% / 643.8 | 42.9% / 697.8 | 53.1% / 573.1 | 67.9% / 392.6 |

The same command with `arc` and `bip8` is in the JSON. The sizes above are
decimal GB. The planner on a 16 GiB machine works in GiB steps (see the real-model
section), so the replay at 0.5, 1 and 2 GiB (167, 334 and 668 slots):

```sh
.venv/bin/python -m models.flashnext.tests.bench.bench_pool_policy_trace \
  results/flashnext/20260922-221213-route-trace-manual/route-trace.json.gz \
  --gb 0.536870912 1.073741824 2.147483648 \
  --policies lru s3fifo least-stale cost-aware --json
```

Result: `results/flashnext/20261113-190936-pool-policy-trace-manual/pool-policy-trace.json`.

| Pool (slots) | lru | s3fifo | least-stale | cost-aware |
|---|---|---|---|---|
| 0.5 GiB (167) | 0.0% / 1221.4 | 0.0% / 1221.4 | 0.0% / 1221.4 | 24.7% / 920.2 |
| 1 GiB (334) | 0.0% / 1221.3 | 34.2% / 804.0 | 30.1% / 853.8 | 39.4% / 740.5 |
| 2 GiB (668) | 49.7% / 614.6 | 48.7% / 626.3 | 43.1% / 694.7 | 54.4% / 557.4 |

At 2 GB `lru` and `s3fifo` are within 1 point of each other and cost-aware is
5.8 to 6.5 points above them; at 1 GB cost-aware is 5.0 points above
`s3fifo` and at 0.5 GB it is the only policy in the table that hits. At 2 GB
cost-aware is 12.1% below LRU in modelled demand MB/token and 11.0% below
`s3fifo`; belady, an offline bound no online policy reaches, is 31.5% below
cost-aware.

**Route predictor** (same run, `--predict`, alpha 0.3, 292,103 routed experts
over 36,864 rows). The table is scored before each row is observed:

| Predictor | precision | recall |
|---|---|---|
| frequency top-4 | 0.538 | 0.271 |
| frequency top-8 | 0.420 | 0.423 |
| frequency top-12 | 0.342 | 0.517 |
| frequency top-16 | 0.284 | 0.572 |
| last-visit persistence | 0.407 | 0.407 |

**Prefetch model.** `--prefetch-slots N` models the mechanism for any replay
policy: decode capacity is `slots - N`, a `RouteFrequency` is observed after
each row (the policy's own table for `cost-aware`, an alpha-0.3 table
otherwise, as in the runtime), the layer order is learned from the rows, and
after each decode row up to `--prefetch-per-layer` records predicted for the
next `--prefetch-depth` layers are issued by the rule above. A staged record
that a later row misses costs no blocking read. **The model assumes every issued
read completes before its target layer and never delays a demand read, so it is
an upper bound on the benefit. `--prefetch-per-layer` (R) is an assumption about
how many records the SSD can read between two layers' demand reads, not a
measurement.** Blocking MB/token is `(misses - used)`, physical MB/token is
`(misses + issued - used)`, both x record bytes / tokens. The model does not
cap the staging count at one eighth of the pool as the runtime does (19 slots
at 155).

Setting A is the runtime defaults with R = 1:

```sh
.venv/bin/python -m models.flashnext.tests.bench.bench_pool_policy_trace \
  results/flashnext/20260922-221213-route-trace-manual/route-trace.json.gz \
  --gb 0.5 1 2 --policies lru s3fifo cost-aware --prefetch-slots 16 --json
```

Result: `results/flashnext/20261113-190637-pool-policy-trace-manual/pool-policy-trace.json`
(`--prefetch-depth 2 --prefetch-width 2 --prefetch-min-p 0.5 --prefetch-per-layer 1`).
No-prefetch demand MB/token is both its blocking and its physical figure.

| Pool | Policy | No prefetch | Decode hit | Issued | Used | Wasted | Precision | Misses covered | Blocking | Physical |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 GB | lru | 1221.4 | 0.0% | 36,403 | 27,284 | 9,119 | 74.9% | 9.3% | 1107.3 | 1259.5 |
| 0.5 GB | cost-aware | 936.8 | 21.3% | 33,134 | 14,963 | 18,171 | 45.2% | 6.5% | 898.2 | 1036.8 |
| 1 GB | lru | 1221.4 | 0.0% | 36,403 | 27,284 | 9,119 | 74.9% | 9.3% | 1107.3 | 1259.5 |
| 1 GB | cost-aware | 762.9 | 36.1% | 3,451 | 2,408 | 1,043 | 69.8% | 1.3% | 770.4 | 784.8 |
| 2 GB | lru | 651.8 | 45.4% | 7,925 | 4,907 | 3,018 | 61.9% | 3.1% | 646.0 | 679.2 |
| 2 GB | cost-aware | 573.1 | 52.6% | 0 | 0 | 0 | n/a | 0.0% | 579.0 | 579.0 |

Setting B is a generous one, with R = 4:

```sh
.venv/bin/python -m models.flashnext.tests.bench.bench_pool_policy_trace \
  results/flashnext/20260922-221213-route-trace-manual/route-trace.json.gz \
  --gb 0.5 1 2 --policies lru s3fifo cost-aware --prefetch-slots 32 \
  --prefetch-width 4 --prefetch-min-p 0.3 --prefetch-per-layer 4 --json
```

Result: `results/flashnext/20261113-190706-pool-policy-trace-manual/pool-policy-trace.json`.
At 0.5 GB the runtime would cap 32 staging slots at 19, so that row is not
reachable there.

| Pool | Policy | No prefetch | Decode hit | Issued | Used | Wasted | Precision | Misses covered | Blocking | Physical |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 GB | lru | 1221.4 | 0.0% | 147,264 | 79,218 | 68,046 | 53.8% | 27.1% | 890.1 | 1505.9 |
| 0.5 GB | cost-aware | 936.8 | 19.2% | 147,001 | 52,002 | 94,999 | 35.4% | 22.0% | 769.1 | 1383.7 |
| 1 GB | lru | 1221.4 | 0.0% | 147,264 | 79,218 | 68,046 | 53.8% | 27.1% | 890.1 | 1505.9 |
| 1 GB | cost-aware | 762.9 | 34.6% | 137,960 | 33,405 | 104,555 | 24.2% | 17.5% | 659.7 | 1236.6 |
| 2 GB | lru | 651.8 | 44.2% | 69,389 | 19,015 | 50,374 | 27.4% | 11.7% | 602.1 | 892.2 |
| 2 GB | cost-aware | 573.1 | 52.1% | 2,285 | 1,057 | 1,228 | 46.3% | 0.8% | 581.2 | 590.8 |

`s3fifo` rows are in both JSON files. In all 18 (size, policy) rows across the
two settings, modelled physical MB/token is above the no-prefetch figure, from
+1.0% to +70.6%. Modelled blocking MB/token falls in 13 of the 18 rows (5 of 9
in setting A, 8 of 9 in setting B), by at most 27.1% (setting B, lru at 0.5 GB, which pays
+23.3% physical bytes). With cost-aware at 2 GB setting A issues nothing (no
non-resident record reaches p = 0.5) and the 16 staging slots only cost 0.5
points of hit rate. The model is the best case.

**Calibration.** Coordinate descent over `recent`, `frequency`, `reuse` and
`alpha` (grid in the tool, two passes, objective the mean decode hit rate over
0.5, 1 and 2 GB on turns 0 and 1), then scored on turn 2 against the other
policies. `refill` and `size` are not searched. The holdout turn starts with an
empty pool; in a real session the pool is warm from earlier turns, so the
holdout numbers describe a cold pool.

```sh
.venv/bin/python -m models.flashnext.tests.bench.calibrate_cost_weights \
  results/flashnext/20260922-221213-route-trace-manual/route-trace.json.gz \
  --out cost-weights.json
```

Result: `results/flashnext/20261113-190911-cost-weights-calibration-manual/calibration.json`
(the defaults plus 30 candidate settings, about two minutes). Train mean: defaults 37.6%,
calibrated 38.2% (`recent` 0.1, `frequency` 0.5, `reuse` 0.5, `alpha` 0.2).
Decode hit rate on holdout turn 2:

| | 0.5 GB | 1 GB | 2 GB | mean |
|---|---|---|---|---|
| defaults | 23.4% | 38.0% | 54.3% | 38.6% |
| calibrated | 24.0% | 38.5% | 54.5% | 39.0% |
| lru | 0.0% | 0.0% | 48.0% | 16.0% |
| s3fifo | 0.0% | 32.5% | 48.3% | 26.9% |
| least-stale | 0.0% | 24.6% | 45.1% | 23.2% |
| belady | 34.9% | 54.3% | 68.7% | 52.6% |

The holdout gain is +0.43 points, under the 0.5-point threshold, so the
defaults stand and no weights file was written. Setting `frequency` to 0 cost
about 8 points on the train mean (29.7%); the other 28 candidate settings lie
between 36.5% and 38.2%, so the score is insensitive to the weights as long as
the frequency term is on. One trace and three turns cannot say more.

## Not yet measured

- No real-model run has been made for Stage 3. Tokens per second, physical
  MB/token, blocking-read ms/token and token digests on the real checkpoint
  are unverified. Every number above is modelled bytes from a replay.
- The prefetch model is an upper bound (reads always complete in time and never
  delay demand reads) and its R is assumed. The runtime also expires queued
  reads whose deadline passed and counts `late`, `cancelled` and `expired`
  records, which the model cannot show.
- The trace is one recorded session from 2026-09-22 (3 turns of 256 tokens),
  recorded against a checkpoint path from a different machine. Its routing may
  differ from current workloads, and the routing of the paired bench's
  code-copy prompt has not been recorded.
- The calibration holdout is one cold-start turn.

## Real-model comparisons

The commands below start real servers with the checkpoint. Run them one at a
time on a quiet machine, plugged in, with Low Power Mode off, under the
guards you normally use. The options need `--runtime flashnext --performance-v2`
and each arm starts a fresh server.

The pool size on this machine needs care. With the default memory policy on a
16 GiB host, the elastic planner (`models/flashnext/memory_budget.py`) picks
0, 0.5, 1 or 2 GiB, the largest step not above
`min(FLASHNEXT_EXPERT_POOL_GB, unreserved memory)`. The cap is in decimal bytes
and the steps are in GiB, so from the code (not run): a request of 2 GB can give
at most 1 GiB, 1 GB at most 0.5 GiB, and 0.5 GB gives no pool.
The bench reads `expert_pool_mapped_bytes` from each arm and fails the run when
it is 0 or differs between arms. For exact decimal-GB pools set
`VEYSTRIX_MEMORY_POLICY=legacy` (inherited by both arms). A fixed pool then
allocates without a headroom check, and the first real chat at a fixed 2 GB once
stopped the compressor guard, so that choice is yours. The new options add the
`absolute-swap-only` 5120 MiB swap ceiling and watchdog settings that
`--pool-compare` uses.

```sh
# Policy A/B at each pool size (repeat with 0.5, 1 and 2).
VEYSTRIX_MEMORY_POLICY=legacy .venv/bin/python -m models.flashnext.tests.bench.bench_server_short_paired \
  --runtime flashnext --checkpoint "$QWEN_CHECKPOINT" --performance-v2 \
  --pool-policy lru cost-aware --isolated-pool-gb 1 --max-tokens 128

# Prefetch A/B at a fixed policy (flag 0 versus 1, same mapped pool bytes).
VEYSTRIX_MEMORY_POLICY=legacy .venv/bin/python -m models.flashnext.tests.bench.bench_server_short_paired \
  --runtime flashnext --checkpoint "$QWEN_CHECKPOINT" --performance-v2 \
  --expert-prefetch --fixed-pool-policy cost-aware --isolated-pool-gb 1 --max-tokens 128
```

`--pool-policy BASE CAND` accepts `lru`, `s3fifo`, `least-stale` and
`cost-aware`; `--cost-weights PATH` goes to the candidate arm when it is
`cost-aware`. `--max-tokens 128` lengthens the default 24-token request so the
cold pool fill is a smaller part of decode; it is a choice, not a measured
setting. The same A/B is the test-terminal card `cost-aware` (default 24
tokens, 2 GB). Each arm's final `/health` is stored in its record under
`final_health`, including `expert_pool_stats`. The existing token-digest and
first-logit gates apply unchanged. `physical_read_bytes` covers the whole
request, prefill included.

Promotion rule: promote only if the token digests match, the mapped pool
bytes are equal, and the paired decode or physical-read effect clears the
resolution band. Otherwise keep it opt-in or reject. For the prefetch, high
prediction accuracy without a latency drop is a rejection.
