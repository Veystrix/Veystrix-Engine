# E3 — predictor recall from real traces (offline, 2026-11-17)

Opus's zero-training pre-gate, run from the real 1,200-token traces in
`results/io_residency/20261117/` (no hardware, no permission needed).
Tool: `models/flashnext/tests/bench/bench_predictor_recall.py`.

Result JSON: `results/flashnext/20261117-180329-predictor-recall-manual/`.

## Numbers (mean routed/layer: Qwen 10.16, Nemotron 6.0)

| predictor | K | Qwen precision/recall | Nemotron precision/recall |
|---|---|---|---|
| persistence (prev token, same layer) | 10 | **0.363** | **0.286** |
| cross-layer (prev layer, same token) | 10 | 0.020 | 0.000 |
| frequency (popular experts) | 14 | 0.050 | 0.054 |
| frequency | 50 | 0.047 | 0.053 |
| frequency | 100 | 0.049 | 0.052 |
| frequency | 170 | 0.051 | 0.052 |

Hit rate (>=1 correct): persistence 0.87 Qwen / 0.82 Nemotron; frequency@170
0.997 / 0.999; cross-layer 0.18 / 0.00.

## Verdict against Opus's rule

Opus said "below 0.6 recall@14 means train the head." No zero-training predictor
comes close: persistence tops out at 0.36/0.29, and naive cross-layer reuse is
0.02/0.00 — adjacent layers route almost disjoint expert sets. Frequency needs
~170 prefetched experts (33% of Qwen's 512) to cover 86% of routing, which is
far too many bytes to prefetch usefully.

So:
- The persistence baseline is the best free predictor but weak (~0.3 recall).
- A TRAINED cross-layer head (Opus §4: layer L+1's router on layer L's residual)
  is the only path to high recall, and its ceiling cannot be measured from routes
  alone. The naive cross-layer 0.02 is the floor a trained head must beat.
- This is consistent with E1 being the decisive test: only an oracle (exact
  future routes) tells us whether high-recall prefetch actually helps throughput.

Keep E1. Do not build a trained predictor until E1 shows the oracle clears 4.5 tok/s.
