# Research survey for SSD-bound MoE decode on the 16 GB Air, 15 November 2026

**Result: no technique found that is lossless, needs no new weights, and helps on this machine. No code was changed from this survey.**

Scope set by the owner's goal: search GitHub and the literature, apply what helps. Earlier owner choices still hold: lossless only, no new weights, no quality loss for speed. The owner also said to drop memory guards. That does not speed anything up, because the guards only stop runs; they do not limit the engine. They stay lifted for measurement runs as recorded in `lookup-drafting-2026-11-15.md`. No default was changed.

## Where the limit is

Qwen reads about 740 MB of experts per token at about 2.7 tok/s, roughly 2 GB/s, which is most of the Air's SSD bandwidth. Nemotron spends about 41 ms on the GPU and 80–90 ms waiting for SSD per token. Only fewer bytes per token or more RAM can change that.

## What was surveyed

Sources are the repositories and papers returned by search on 15 November 2026. Results were read through search summaries and two fetched pages, not the full papers; numbers are the authors' own.

| Source | Idea | Verdict for this machine |
| --- | --- | --- |
| `danveloper/flash-moe` (397B on a 48 GB M3 Max, 4.4 tok/s) | Parallel `pread` per expert, trust the OS page cache (+38% after removing its own LRU), FMA dequant (+12%) | Veystrix measured the same page-cache result (pool is slower than the macOS cache). The dequant rewrite is already how MLX's kernels work. |
| same, "rejected" list | MTP/speculation break-even because MoE I/O scales with tokens; read-ahead, temporal prediction (-18%), mmap (-5x), `dispatch_io` (-70%), expert clustering (0%) all rejected | Matches this repo's own measurements (read-ahead, mmap, MTP, lookup). Independent confirmation, not new gain. |
| same | 2-bit experts reach 5.7–7 tok/s but break JSON tool calling | Changes outputs and breaks the Rakazo bots' tool calls. Not done. |
| `ssd-moe/deepseek-v4-flash-mlx`, `hatim-mlf/moe-stream-mac`, `Jakevin/moe-offload` | SSD expert streaming with an LRU cache in MLX; decode limited by SSD | Same design Veystrix already has. |
| HOBBIT, MoE-APEX, DyMoE, SliceMoE | Load low-precision copies of cache-miss experts | Needs extra lower-bit weights and changes outputs. Excluded by the owner's rules. |
| ProMoE, ExpertFlow, FineMoE | Predict next experts and prefetch | Veystrix's read-ahead reached 58–72% recall and made decode slower; flash-moe saw -18% to -38%. SSD DMA competes with the GPU for memory bandwidth. |
| MoE-Spec | Cap experts loaded per layer during verification | Lossy (changes accepted tokens). Excluded. |
| DraftExpert | Self-speculation with a trained draft expert per layer, 1.3–1.6x on a 4090 and a phone | Needs training and extra weights per model. Excluded. |
| EcoSpec | Pick draft tokens that reuse experts already in the verify set; 1.36–1.62x | Needs a trained 0.6–1.5B predictor. Excluded. |
| EVICT (arXiv 2605.00342) | Truncate the draft tree to the cost-effective prefix; training-free, lossless; up to 2.35x on a GPU serving stack | Considered in detail below. |

## EVICT-style truncation, checked against the measured numbers

EVICT only helps when an extra verified row costs less than the token it can win. Measured on Nemotron (`lookup-drafting-2026-11-15.md`): a plain token costs about 137 ms; a lookup round of about 4.6 rows costs about 700–850 ms and yields about 4.3 tokens. That is roughly 150–200 ms for each extra drafted row, more than a whole plain token, because each row routes to different experts and reads them from the SSD. Even at 92–95% acceptance a drafted row then costs more than it saves, so truncating the draft cannot make speculation win; the best truncation is zero rows. This is derived from existing measurements, not a new run. A paper's gain on a GPU where extra rows share resident weights does not transfer.

## What is left

1. **More RAM (32 GB or more)**, which stops the experts leaving memory. The only measured lever.
2. **A GPU with the weights resident**, where HyperQwen, EVICT and DraftExpert-style methods apply.
3. **Lossy options**, if the owner ever accepts them, with the cost stated first: 2-bit or mixed-precision experts (breaks tool calling per flash-moe), expert budgeting. Not started.

Not read in full: the arXiv text of EVICT, MoE-APEX, DyMoE, SliceMoE, MoE-CORE and flash-moe's `paper/flash_moe.pdf` and `results.tsv` (58 experiments, not fetched).
