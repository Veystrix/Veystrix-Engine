# Direct search notes — Cerebras, 2026-11-17

## How Cerebras inference works (the essential mechanism)
- WSE-3: 44 GB on-chip SRAM, ~900,000 cores, aggregate 21 PB/s (WSE-3 Turbo
  in CS-4: 43.2 PB/s). ~50% of silicon is SRAM. Weights + KV live in SRAM
  directly beside compute. **No off-chip weight read per token.**
- Multi-trillion models: partition at LAYER boundaries across wafers. Each wafer
  holds entire layers (one dense layer + KV, or two linear-attention layers)
  in its SRAM; weights NEVER move. Only activations cross wafer-to-wafer via
  RDMA over Ethernet ~2 microseconds/hop. Tokens pipeline through wafer stages;
  high throughput comes from many requests staggered in the pipeline.
- Speed: gpt-oss-120b ~3000 tok/s (4-bit experts), Qwen3-235B 1500, GPT-5.6 Sol
  750. All since Oct 2024 use a draft model (speculative) — even Cerebras uses
  speculative decoding. CS-4 disaggregates prefill (cheap GPU) from decode
  (Cerebras).
- Source: cerebras.ai/blog/how-we-deploy-trillion-parameter..., GPT-5.6 Sol blog,
  CS-4 page, Semianalysis "Faster Tokens Please", WaferLLM OSDI'25.

## The three exits from the memory wall (Semianalysis / Praveen Kumar)
1. Read fewer bytes/token: quantization (4-bit), MoE (active slice), speculative
   (one read emits several tokens). Everyone does this. Cerebras numbers all
   involve a draft model; Kimi on Cerebras is 4-bit.
2. Add more parts: tensor parallelism sums bandwidth across chips, but each layer
   ends in a cross-chip sync; past a few parts sync costs more than bytes saved.
3. Change the memory: move weights DRAM/HBM -> SRAM (10-100x faster per byte,
   10-100x smaller per dollar). Cerebras (44 GB/wafer), Groq (230 MB/chip).
   Capacity is the limit; whole model + all KV must fit in SRAM.

## Why this matters for Veystrix (16 GB M4, no new hardware)
Cerebras does NOT split I/O time — it ELIMINATES weight I/O by keeping all
weights in the fastest tier next to compute. Our M4 already has a fast tier:
unified memory (~100+ GB/s) and the SSD is the slow tier. Our problem is that
the 125B model's experts (70+ GB) don't fit in the fast tier, so every token
must read from SSD. Cerebras's lesson: the ONLY real fix is more fast memory
(next to compute). Software can only: (a) fit more experts in RAM (cache tiering
— bounded by 16 GB, already at ceiling), (b) read fewer bytes (quant/spec —
mostly closed for us), (c) overlap (E0/E1 — +25% max). No software trick
'replicates' wafer-scale SRAM on an M4. This directly supports the conclusion
that targets need hardware we cannot buy.

## Transferable ideas worth the research streams
- Disaggregated prefill/decode (Cerebras CS-4): use a fast prefill path, decode
  specialized. On one machine this is just prefetch/scheduling, already explored.
- Speculative decoding to amortize one weight read over several tokens: this is
  the 'read fewer bytes' exit; our E1 shows the offload version only +25% because
  bytes are unchanged unless routes overlap. K=2 slower on Metal (2610.34440 r/u).
- Pipeline parallelism across wafers with activation-only transfer: on one M4
  this maps to layer pipelining (compute layer L while reading L+1) — tried.
