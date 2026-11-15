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

## Second pass (same day, after "keep searching, no new weights")

| Source | Idea | Result here |
| --- | --- | --- |
| MINE (arXiv 2611.09424), EntroLLM, `drxddy/ecq`, DFloat11 | Lossless entropy coding of weights to read fewer bytes | **Measured on the real Nemotron checkpoint** (`scratchpad/entropy.py`, read-only, 8 expert tensors from layers 1, 8, 20, 40): 4-bit values carry 3.65–3.73 bits of entropy, so packed bytes compress to about 7.43/8; the bf16 scales and biases compress well (high byte about 1.1 bits of 8). Ideal order-0 coding gives **84.4% of the original size, a 15.6% cut at best**. Real coding loses some of that, needs a decoder kernel and a repacked copy of 14 GB of experts. flash-moe's LZ4 test lost 13%. Not worth building. |
| "Who Should Own the Expert Cache?" (arXiv 2608.12103, read in full) | Kernel page cache as the expert tier | Recency serves essentially the same demand as an oracle frequency policy; an oracle-pinned arena is only 1.09–1.11x faster; kernel readahead advice with 64.7% recall changes time by 0.3%, perfect one-layer advice 5%. Confirms the macOS file cache is the right tier and prefetch cannot add much. |
| Mixture of Cache-Conditional Experts (arXiv 2412.00099) | Bias router toward resident experts, training-free | Lossy but needs no new weights. Already built for Qwen here (`routing.py` `cache-aware`, +8.3%, 16.8% fewer reads, 2026-08-31). Ported to Nemotron as an opt-in, off-by-default switch: `models/nemotron35/cache_routing.py` (below). |
| Layered Prefill (MLSys 2026) | Layer groups instead of token chunks so experts load once per prefill | Already in the repo as layer-major prefill. |
| 125B Qwen3.8 at 18 tok/s on 64 GB (lilting.ch) | "N-gram table on SSD" | An embedding table inside a 1.75-bit build (KLD 0.23) on a 64 GB machine. Not a speed-up technique and needs new weights. |

### Real Rakazo traffic on Qwen (usage log, 11 requests of 11–15K tokens)

The previous reply is cached with the prompt (cached = previous prompt + previous completion exactly), so the uncached tail is only the new group messages. A tail still costs: 27 tokens 7.8 s, 94 tokens 16.7 s, 392 tokens 38.5 s, 849 tokens 44.6 s; that is decode-like cost for small tails because a few tokens share almost no expert reads. Decode is 1.6–2.9 tok/s, so a 237-token reply took 125 s and a 27-token one 12 s. **Reply length and the number of requests per group message set the bots' latency; the engine is near its limit.** Shorter bot replies are the largest lever left for the Rakazo workload, and it is outside the engine.

### Cache-aware routing for Nemotron (built, offline-tested, not yet measured on the real model)

`VEYSTRIX_NEMOTRON_CACHE_ROUTING=off|observe|swap` (default off), `…_EPSILON` (0.02), `…_WINDOW` (3 tokens), `…_STATS` (JSON path). Decode rows only; prefill and verification route exactly. `observe` counts how many swaps each tolerance would allow and changes no output; `swap` takes a recently used expert in place of a non-resident one scoring within epsilon. Tiny-model tests: `models/nemotron35/tests/unit/test_cache_routing.py` (observe and epsilon 0 reproduce exact routing, prefill is exact, swap prefers recent experts and keeps weights normalised). A real run of `observe` then `swap` is needed before any claim; it needs the live engine stopped, and `swap` needs a quality check on tool-call prompts before it is offered anywhere.

### Nemotron cache-aware routing, observe run on the real model (evidence: `results/nemotron35/20261115-224958-cache-ab-nemotron-observe`)

One guarded run (memory floor off at the owner's instruction; swap, thermal and battery stops on; AC power, thermal 0, swap flat, available memory floor 6.4 GiB), stopped by the owner after 5 of 6 requests (the sampled code request did not run). Exact routing, output unchanged. Decode 6.5–7.9 tok/s. Over 11,968 routed-layer calls (71,808 chosen experts): **45% of chosen experts had been used in the last 3 tokens**; of the other 55%, a recently used expert scoring within epsilon existed for 1.9% (0.005), 4.0% (0.01), **8.9% (0.02)**, 25% (0.05), 48% (0.1) and 71% (0.2). The recency window overstates misses (the macOS cache holds more), so real read savings are smaller than these fractions. Rough expectation, not measured: epsilon 0.02 about +3%, 0.05 about +8%, larger epsilon more with more quality risk. No swap run has been done; `swap` stays off and needs a paired speed run and a tool-call quality check first.

### Nemotron cache-aware routing: swap measured (evidence: `results/nemotron35/20261115-225848-cache-ab-nemotron-{off,swap}`, `results/nemotron35/20261115-230825-quality-ab-nemotron-{off-0,swap-0.05,swap-0.1}`)

Real model, one run per arm, engine otherwise stopped, guard memory floor off at the owner's instruction (swap flat, thermal 0, available memory floor 5.4–6.2 GiB). Greedy decode, speed from the usage log; the exact arm ran first (cooler) in each pair.

| Arm | Speed | Known-answer questions | Tool-call JSON (8) |
| --- | --- | --- | --- |
| exact | 6.44 tok/s median (24-request set), 7.0–7.8 on the 6-prompt set | 15/16 (counts "strawberry" wrong) | 8/8 valid and correct |
| swap, epsilon 0.05 | 6.81 (+5.7%); 6-prompt set +3% to +11%, mean about +6% | 15/16 | **7/8**: one answer lost its closing brace (invalid JSON) |
| swap, epsilon 0.1 | 7.07 (+9.8%) | 15/16 | **6/8**: after a valid call the model wrote `</think>` and a second, invented call (twice) |

About 0.6 experts of 6 were swapped per layer call at 0.05, and 1.3 at 0.1. Speed samples are small (8–9 requests long enough to time), single runs, so the percentages are rough. **Verdict: swap buys 5–10% and breaks tool calls at both tolerances tried. Do not offer it in the app.** The Rakazo bots depend on tool calls, so it stays an off-by-default developer switch. Epsilon 0.02 was not run (about +3% expected, below what one run can resolve). Observe mode costs nothing in output and can be left available for diagnostics.
