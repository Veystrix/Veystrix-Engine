# Direct web evidence (coordinator-collected, 2026-11-17)

Primary measured results found by direct web search. Every number has a source URL.
These are the strongest signals because several are on the SAME model (Qwen3.8-Flash-Next)
or the same hardware class (16 GB M4).

## A. SAME MODEL + M4 — Jakevin/moe-offload (https://github.com/Jakevin/moe-offload)
Qwen3.8-Flash-Next (125B MoE, 512 experts/layer, top-10), MLX. 8 prompts x 128 tok.
Greedy output identical to reference in all cached runs. Expert cache = 20% of routed
experts (~7.4 GB), layer 0 pinned.
- Mac mini M4 Pro 64 GB internal SSD:
  - Reference tracer (28 layers resident): 2.52 decode / 5.3 prefill tok/s, 24.9 GiB RSS
  - v1 Python pread, no prefetch: 8.51 / 14.8, 84.3% hit, 11.6 GiB
  - v2 native pread, fewer syncs: 9.17 / 18.9, 84.3% hit, 10.3 GiB
  - v2 + router prefetch top-12 max 3/layer: 9.73 / 19.4, 94.7% hit, 8.9 GiB
- Profile of v1 per token (118 ms): GPU 27%, Python + mx.eval syncs 41%, SSD reads 29%,
  bookkeeping 3%. "Remaining gap is mostly Python per-layer synchronization."
- Next-layer router prediction recall@10/@16/@24 = 83.6% / 94.1% / 96.7%.
- Fully resident llama.cpp GGUF of this model (different setup): 24.6 tok/s.
- Nemotron 3.5 Bolt 74B-A9B (IQ2_XXS GGUF, 240 experts/layer, top-9), llama.cpp Metal:
  full baseline 13.9 tok/s at 20.8 GiB; caches 4/8/12 GB gave 11.07/11.05/11.28 tok/s
  at 9.9/13.9/17.9 GiB peak (caching LOST on 64 GB for this model).
- **CORRECTION (verified from raw README 2026-11-17): the 4.39 tok/s / 16 GB result is
  NEMOTRON 3.5 Bolt (llama.cpp Metal), NOT Qwen.** 4 GB cache = 4.39 tok/s (86.1% hit,
  8.5 GiB peak); 8 GB cache thrashed and was aborted. Qwen's 16 GB bit-exact number is
  not published; its 64 GB numbers are the 8.51–9.73 tok/s rows above.
- **C++ decode path (mlx-qwen38/cpp/): bit-exact vs Python loader (raw logits every
  step + 122/122 final cache tensors identical), 9.0–9.6 tok/s at ~11.5 GB RSS on
  M4 Pro 64 GB.** Dense attention only, total length <= 2048 tokens. This is the concrete
  proof that removing Python/mx.eval sync overhead is the biggest lever.
- "A2 hot-expert seeding" was bit-identical but 1.008x (CI 0.967–1.051) -> not adopted
  (another measured dead end).
- Long-prompt prefill (M4 Pro 64 GB, 20 GB cap, 14,937-token agent prompt):
  one-shot no cap ~105 s; chunked 2048 no-cache 315.7 s TTFT; cache miss 305.9 s;
  **cache hit 63.9 s (4.94x faster)**. Cached vs uncached bit-exact (last-position logits,
  220 state arrays, 32 greedy tokens). Changing chunk boundaries can change logits slightly
  (max |diff| 1.375, argmax unchanged).

## B. Direct overlap / pipelining — bit-exact, huge gains
- Jetson Orin Nano 8 GB, Gemma 4 26B-A4B Q4, llama-bench tg128
  (https://www.anyesh.me/writing/moe-stream-jetson-orin):
  `--cpu-moe` mmap baseline 1.94 tok/s; streaming 32 slots x 8 IO threads overlap ON
  2.95 tok/s; overlap OFF 2.23. Overlap alone worth 32%. Logits bit-for-bit identical,
  32 greedy steps, 10.5 GiB streamed. NVMe random-read ceiling ~1.27 GB/s puts IO-bound
  limit near 5 tok/s. Shared-FFN GPU compute reordered to hide NVMe reads (1 sync/layer).
- TinyGiant prototype, M1 MacBook Pro 16 GB (https://github.com/ggml-org/llama.cpp/discussions/27149):
  sequential layer read 2.97 GB/s; double-buffered pipeline 6.06 GB/s (2.09x).
  Qwen3-30B-A3B per-layer I/O 263 MB -> 30 MB (11.4x) by reading only 8 active experts.
  Expert-contiguous re-layout reduces page faults 36x (192 vs 6912 pages/expert).
  Hit-rate sweep (CPU-only, f16 experts): 50% 2.20->2.40; 75% 2.92->3.81 (1.30x);
  88% 3.47->4.10. With Q4 experts (~2.2 MB), pipelining hides ALL I/O -> compute-bound
  even at 0% hit. 4.7 tok/s on ~2 GB RAM.

## C. Prefill techniques (exact)
- Wave-Partitional Prefill, llama.cpp PR #25294 (MERGED,
  https://github.com/ggml-org/llama.cpp/pull/25294): when a ubatch touches more experts
  than cache holds, run GEMMs in W waves, mask and sum. Each touched expert loaded once
  per ubatch. GLM-5.2 on GB10: prefill 2.28 -> 5.65 tok/s (waved), decode ~1.83-2.20,
  73-79% hit. Bit-exact under matched kernels + ubatch. O_DIRECT + async I/O pool.
- Full-layer staging + second SSD mirror + RAM victim cache:
  https://github.com/xMytze/glm53-ssd-streaming — 7,943-token prompt prefill 29 -> 240 t/s,
  decode 2.4-2.9 -> 5.8-6.0 t/s. 78% cache hit; ~18 GB/s combined SSD reads.
- io_uring pack pager + KICK/WAIT doorbell (llama.cpp PR #31469,
  https://github.com/ggml-org/llama.cpp/pull/31469): 16.80/21.85 tok/s, 100% byte-exact,
  576/576 identical greedy tokens.
- SYCL routed-row matmul + counting-sort routing (PR #23142, MERGED): +70% MoE prefill
  on Qwen3.6-35B; counting sort contributed up to 10%.
- Prompt/prefix caching: 4,312-token system prompt 19.42 s -> 0.35 s prefill (57x),
  ~30x end-to-end, exact mode (https://www.drmhse.com/posts/running-qwen3-8-9b-on-a-16gb-mac).

## D. Expert cache break-even rule (critical for a 16 GB Mac)
- https://github.com/AbhayXplor/moe-expert-cache-study: cache helps only when pool >=
  ~1/3 of total expert bytes. Qwen3.6-35B-A3B (19.569 GB experts) 2048 MiB pool (11%):
  22.02 -> 8.03 tok/s (-64%); 4096 MiB (22%): 22.29 -> 11.91 (-46%).
  REAP-48 (7.552 GB experts) 3072 MiB (43%): 12.68 -> 25.67 (+104%);
  5120 MiB (71%): 12.56 -> 39.65 (+216%). Adaptive pool beats static placement by 17%
  with 688 MiB less VRAM. LFU sim: 88/256 experts (34%) -> 98.8% hit.

## E. Exact speculative decoding (no changed tokens)
- Qwen3-9B on 16 GB M4 (https://www.drmhse.com/posts/running-qwen3-8-9b-on-a-16gb-mac):
  9B native drafter carries decode 20.8 -> 36.4 tok/s "without changing a token."
- Qwen3.8-Flash-Next 95.5 GB on 64 GB M5 Pro (note.com/hacklog_stealth/n/nd5833cb2bfe2):
  27.6 tok/s with draft head, 4k prefill ~367 tok/s. Requires iogpu.wired_limit_mb=59392
  and --moe-stream-cache 36; setting 38 drops to 3.7 tok/s. Does NOT work on 16 GB
  because the resident portion alone fills memory.

## F. Multi-Mac / other (exact)
- 2x M4 16GB via llama.cpp RPC, Qwen3.6-35B-A3B 4-bit: 25.0 tok/s
  (https://llamaperf.com/gpu/m4-16gb). 50/50 split works, 40/60 OOMs.
- KV cache quantization on 16 GB M4 Qwen3-9B: 20.80 (BF16) / 20.01 (8-bit) / 21.13
  (4-bit) — noise, no effect. Flash/sparse attention + sinks: no effect (only 8/32
  layers use attention). Quantized GEMM 3.29 vs dense BF16 3.73 TFLOP/s (4-bit = 88%
  of dense) => dequantizing bounded at ~1.13x.

## BOTTOM LINE FROM DIRECT EVIDENCE
On the SAME model (Qwen3.8-Flash-Next) and M4 class:
- 2.3 -> 4.39 tok/s on 16 GB is demonstrated with a 4 GB expert cache + native pread.
- 2.3 -> 9.73 tok/s on 64 GB is demonstrated with native pread + fewer syncs + router
  prefetch (removing the 41% Python/mx.eval sync overhead is the biggest single lever).
- Nemotron 6.7 -> 13.9 tok/s on 64 GB full residency; caching regressed for Nemotron.
- Prefill: 4.94x-57x via prompt caching; 2.48x via wave-partitional prefill.
Both targets (5+ and 15+) appear REACHABLE on 16 GB only with careful cache sizing,
sync removal, and overlap; 15+ Nemotron likely needs more memory or a second Mac.
