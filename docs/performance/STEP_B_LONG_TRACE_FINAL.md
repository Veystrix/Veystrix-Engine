# Step B — long-trace ceilings (DEFINITIVE), 2026-11-17

Owner said "continue". Ran the guarded ≥1,000-token long-trace captures for both models
(Opus's Step B). Both guard-clean, no output change (diagnostic route capture only).

## Captures
- Qwen: `experiments/io_residency/capture_trace.py` direct harness, EOS suppressed,
  1,200 tokens in 547 s, finish=length, floor 2.26 GiB, swap growth 0. Trace 1,201 tokens,
  48 layers, deduped (only 0.1% prefill-step dupes). `results/io_residency/20261117/qwen_long_trace_dedup.jsonl`
- Nemotron: `bench_decode_profile --tokens 1200`, 1,200 tokens in 243 s, floor 3.98 GiB,
  swap growth 0. Trace 1,200 tokens, 23 layers. `results/io_residency/20261117/nemo_long_trace.jsonl`

## The first-touch mechanism is REAL but SMALL at long length
Prefix sweep (OPT@6-8GB) on the long traces: compulsory (first-touch) share falls to
~1.4% by 1,200 tokens for both models. So my 68/96-token short traces DID overstate
first-touch (Opus was right that this was an artifact). BUT the long traces show a NEW
effect the short traces hid: **capacity misses appear and grow with length** (Nemotron
capacity-miss 0.060@200 -> 0.110@1200). So the steady-state is NOT "100% hit" like
Opus's short-trace projection; it settles at a real, lower ceiling.

## DEFINITIVE steady-state ceilings (1,200-token traces, measured constants)
NEMOTRON (3.541 MB/ms, 58.3 ms non-I/O):
| pool | coverage | OPT hit | OPT tok/s | S3-FIFO tok/s |
|---|---|---|---|---|
| 2 GB | 12% | 0.671 | 7.68 | 5.27 |
| 4 GB | 25% | 0.811 | 10.03 | 6.92 |
| 6 GB | 37% | 0.888 | 12.08 | 8.55 |
| 8 GB | 50% | 0.936 | 13.82 | 10.36 |
| 10 GB | 62% | 0.966 | **15.22** | 12.38 |
| 12 GB | 74% | 0.986 | 16.27 | 14.40 |

QWEN (2.603 MB/ms, 143.24 ms non-I/O):
| pool | coverage | OPT hit | OPT tok/s | S3-FIFO tok/s |
|---|---|---|---|---|
| 8 GB | 11% | 0.854 | 4.43 | 3.22 |
| 12 GB | 16% | 0.915 | **5.22** | 3.92 |
| 16 GB | 22% | 0.949 | 5.80 | 4.60 |
| 20 GB | 27% | 0.971 | 6.25 | 5.21 |

## Verdict — honest, between the two earlier estimates
- **Neither earlier estimate was right.** My short-trace "11/4 tok/s ceilings" were too
  pessimistic (first-touch artifact). Opus's "15.34 tok/s @6GB / 6.16 @8GB" were too
  optimistic (short traces hid long-range capacity misses; real OPT@6GB=12.08, @8GB=13.82).
- **Nemotron 15 tok/s:** reachable ONLY with >=~10 GB of unique expert RAM AND a near-OPT
  policy. Production on 16 GB auto-sizes the pool to 0; a 10 GB pool is far beyond what fit
  (8 GB cache already caused regressions/memory pressure; 10 GB may not even admit). Even with
  10 GB, achieving 0.966 hit needs a near-Belady policy — S3-FIFO gets 0.897 there (12.38 tok/s).
- **Qwen 5 tok/s:** reachable ONLY with >=~12 GB of unique expert RAM AND near-OPT policy
  (S3-FIFO at 12 GB is only 3.92). A 12 GB pool on 16 GB alongside the model is implausible.
- **Practical conclusion:** on this 16 GB M4 with a realizable pool (<=~2-4 GB) and a real
  online policy (S3-FIFO/nextuse, not OPT), the ceiling remains ~5-7 tok/s Nemotron and
  ~2-3 tok/s Qwen. The targets (15/5) require both >10 GB unique expert RAM and a near-perfect
  cache policy — not achievable in this 16 GB configuration. This is now backed by REAL
  long traces, not short-trace artifacts.

## Next (per Opus's gates)
- Step C A/Bs (single-tier pool, nextuse policy) remain valid experiments to squeeze toward
  the achievable ceiling (e.g. Nemotron ~7->10, Qwen ~2->3), but they will NOT reach 15/5 on
  16 GB. The single-tier hypothesis needs the G1 `dup_bytes` probe to confirm any unique RAM
  is actually reclaimable; that probe still has not been run on hardware.
- Recommend: report these definitive ceilings honestly; if the owner still wants the A/Bs
  for partial gains, run them, but set expectations that 15/5 is a hardware (memory) limit.

## Files
- `results/io_residency/20261117/`: sim_{nemo,qwen}_long.json, long traces, v02 sims
- `results/nemotron35/20261117-172308-decode-profile-manual/`: guarded Nemotron long capture
- Qwen long capture guard: `experiments/ane/results/qwen-long-trace-guard.jsonl`
