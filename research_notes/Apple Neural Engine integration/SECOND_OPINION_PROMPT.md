# Second-opinion prompt: Veystrix ANE + prefill/decode speed

> **ERRATUM (2026-11-17):** the line "~345 KB each" in the copyable prompt is
> wrong. 345 KB is one of the **nine regions** of an expert; one whole expert is
> **3,072,000 bytes (~3 MB)**. 480 experts × ~3 MB = 1,253 MB/token with zero
> cache hits, exactly matching the measured figure. There is **no read
> amplification**. The answer received from this prompt built a false
> "UBC thrashing / F_NOCACHE" theory on that error — discard it.
> Use `CREATIVE_SOLUTION_PROMPT.md` for a corrected, less leading version.

Copy everything below this line into the other AI.

---

You are an expert in Apple Silicon performance engineering, MLX/Metal, Core ML, and the Apple Neural Engine (ANE). I need an honest technical second opinion — challenge my assumptions, tell me where my reasoning is wrong, and do NOT just agree with me. Distinguish clearly between (a) measured facts, (b) arithmetic estimates, and (c) speculation. If you use web sources, cite URLs and dates, and flag single-author/unreplicated claims as such.

## Project context

Veystrix is a free macOS app: a SwiftUI shell plus a Python engine running on MLX. It runs two local 4-bit Mixture-of-Experts models on a **fanless 16 GB M4 MacBook Air**:

- **Flash-Next / Qwen 3.8 Flash-Next** (~105 GB on disk): 48 layers, 12 full-attention + 36 linear-attention (GDN), hidden 2560, 512 routed experts per layer with exact top-10, ~75.5 GB of experts total. QSA sparse attention, PLE n-gram table, 3.4 GB dense core.
- **Nemotron 3.5 Lightning 30B-A3B** (~24 GB): 52 layers (13 attention + 23 Mamba + 16 MoE), hidden 2688, 128 experts, 15.38 GiB routed experts, 1.25 GiB dense core.

The app is distributed free/ad-hoc (not App Store, not notarized). Source code is private. Hard rules from the repo: preserve bit-identical token digests and BF16 rounding boundaries; keep Q4 affine G32 weights, exact top-10-of-512 routing and exact softmax/renorm; every optimization must be opt-in with a rollback; **never run benchmarks/sweeps/hardware tests without my explicit permission**; all real runs under a safety guard (≥3 GiB available, AC power, stop on swap/compressor growth, thermal level 2, battery <25%).

## Measured bottlenecks (all on this 16 GB M4, plugged in, under the guard)

- **Qwen decode:** ~385 ms/token = **289 ms expert-read wait (75%)**, ~51 ms router sync, ~70 ms GPU. Throughput 2.4–2.7 tok/s. ~480 expert reads/token (48×top-10), ~345 KB each; 742 MB–0.95 GB/token. SSD reaches 3.46 GB/s at ≥1 MB but the 345 KB expert layout only reaches 2.3 GB/s at 4 threads / 2.9 at 16.
- **Qwen prefill:** 4,096-token cold prefill ~48.2 s split: full attention 17.2 s, GDN 8.2 s, hyper-connections 5.0 s, PLE 3.9 s, MoE incl. read waits 13.2 s. Reads 83.8–139.6 GB for a 4.3K prompt. With all exact-speed flags: ~41.2 s.
- **Nemotron decode:** ~120 ms/token, ~75 ms expert wait, ~8 tok/s, 179 MB/token physical reads even at a 77% page-cache hit rate. Arithmetic estimate: with experts fully resident (32 GB Mac) ~41 ms GPU work → ~24 tok/s. Not measured.
- **Nemotron prefill:** 8K tokens 252 tok/s (32.3 s: Mamba 9.7 s, experts 18.2 s incl. 4.6 s SSD wait, attention 4.3 s); after accepted speed defaults ~355 tok/s.

Both models are SSD/RAM-bound at their floor on 16 GB, not GPU-bound.

## Already accepted (default-on) speed work

A curated bit-identical 24/24 quality-gated flag set: tiered4 KV, 4K hybrid layer-major prefill, single-chunk layer-major, merge-small-tail, shared dense KV, fused gather, fused SDPA, ngram-dedup, dense GEMM, S3-FIFO pool, gather tile 32, plus Nemotron's bit-equal dense GEMM. Measured gains: +14–36% prefill, +5% decode, no token change.

## Thoroughly tried and REJECTED (do not suggest these without new evidence)

decode prefetch/overlap; more I/O threads (6/8/16/24/32 all ~2.55–2.57 tok/s); pool policies other than S3-FIFO (cost-aware, least-stale); larger pools on 16 GB (guard stops at ~2.6 GiB floor); route-union prefill (30.5 s vs 29.9 s — no gain); PLE row prefetch (slower); Nemotron decode read-ahead and dropping stale predictions (slower); mmap/zero-copy of unaligned shards; native MTP at every depth on both models (slower); lookup drafting v2; Nemotron parallel Mamba (failed exactness); Nemotron 4K chunks (memory pressure); qsa-fast-prefill (failed 24-case quality gate); QSA gather tile >32; wired memory limit; expert packing (no server-side gain: 66.6 s vs 66.1 s); 4-bit palettized Core ML weights (silently run on GPU, accuracy ruined).

Critically: a **perfect** expert predictor with overlapped reads tied the baseline (+0.1%, −2.9%, −0.7%) and early submission lost 8.2% — the cause is memory-controller contention. Decode is a strict per-layer chain (router needs previous layer, experts need the router), so reads cannot overlap compute without speculation, and wrong guesses cost more than right ones save.

## Higher-memory work already implemented (not validated on hardware)

Tiers 16/24/32/64/96/128 GB. Flash-Next's pool is passed as a single 32-bit `uint32` MLX array → **8 GiB hard kernel-addressable cap**, so its 75.5 GB experts can never be fully resident without a kernel change. Nemotron's 15.38 GiB experts **are** fully resident at 32 GB+ (fail-closed arithmetic), which should remove most of its 75 ms expert wait.

## ANE investigation — what we found

1. **In-repo experiment (2026-11-07, this exact machine) rejected ANE:** batch-1 predictors never reach ANE (stay on CPU, 35 µs, 0 neural footprint); routing numerics only 98.8% exact (fails bit-identical); ANE side-load adds +3.5% (200 Hz) to +17.6% (flat out) to bandwidth-bound GPU work; duplicate FP16 weights don't fit. Status: "evaluated and rejected. ANE development is stopped."

2. **We reopened it and measured (2026-11-17, guard, plugged in):** baseline reproduced. A real layer-5 Nemotron `q_proj` dequantized to FP16 runs on ANE in **6.3 ms vs 17.6 ms MLX GPU** (mlprogram, 2.8×); INT8 5.24 ms with half the weight bytes. But **end-to-end** offloading all 13 attention projections was **3.3% slower** (279 vs 289 tok/s warm, floor 3.6 GiB), 4 layers 274 tok/s, Mamba 8 layers 276 tok/s, and Mamba 39 layers tripped the memory guard (compressor 7.3 GB). Cause: ~21 ms per call of Python/Core ML marshalling (GPU sync, format conversion, Core ML predict, copy back) swamps the 12 ms saved per projection. Palettized 4-bit fell back to GPU and ruined accuracy. Native PyObjC zero-copy buffers segfaulted — MLX buffers aren't safe Core ML backings. Apple *does* provide single-allocation paths (`MLMultiArray` over IOSurface via `init(pixelBuffer:shape:)`, page-aligned `dataPointer` + deallocator, newer `MLTensor.init(bytesNoCopy:)`, `MLPredictionOptions.outputBackings`), but these need a native Swift bridge, not Python.

3. **maderix/ANE (MIT, 2026)** — the key discovery. He bypasses Core ML entirely via reverse-engineered private APIs (`_ANEClient`, `_ANECompiler`, `_ANEInMemoryModel`) and runs arbitrary MIL compute graphs on ANE, including **full backpropagation** (Stories110M 91 ms/step; Qwen3-0.6B 412 ms/step on M4). Relevant details:
   - **Direct ANE dispatch is ~60–80 µs**, vs our ~21 ms through Core ML — ~250× cheaper.
   - Working **GPU↔ANE zero-copy via shared IOSurface** (`gpu_ane_share.m`, `gpu_prefill_ane_decode.m`).
   - Fork `m0at/ANEtransformers`: **ANE baked-weight prefill 76.6 tok/s** on Mistral 7B Q4_0 (M5, 24 GB), single layer 23.9 ms vs BLAS 118.1 ms (4.9×). BUT MLX on the same chip does 200+ tok/s — full-model ANE prefill still loses to a tuned GPU because CPU round-trips for RMSNorm/RoPE/attention/transposes dominate. Decode stays on CPU (19.7 tok/s) because dispatch dominates at S=1.
   - Hard constraints: fp16 only via private API (INT8/INT4 fail to compile); weights **baked into the compiled program** (128 programs for 7B, 80 s cold compile, 220 ms warm reload via forging `hexStringIdentifier`, cache lost on reboot, ~119 compile-per-process leak); no `reduce_sum`/`rsqrt` so RMSNorm cannot run on ANE (2 CPU round-trips/layer); channel-first `[1,C,1,S]` layout with silent wrong results if transposed incorrectly; max 2 outputs/program; silent all-zeros below S=16 (fp32) / S=32 (fp16); kernel chaining fails (Error 15, likely entitlement); numerics not bit-identical (cosine 0.998, max err 0.61).
   - Private APIs may break with any macOS update and complicate notarization.

4. **hollance/neural-engine** (community reference) confirms: you cannot force Core ML onto ANE (`.all` is a hint); shared memory still has format-conversion cost when switching processors; gather/custom/RNN/broadcastable-ND layers are unsupported; ANE computes in FP16; Core ML's "smart compute system" may pick GPU when idle and ANE when GPU is loaded.

## The user's current hypothesis (please evaluate this specifically)

> "The model needs to stay completely on RAM, but I'm storing it on SSD and reading layer by layer. What if we dip all pages at a time and let it read (stage all experts up front), and give ANE the jobs it's good at while the GPU works at the same time? Nothing is stopped."

My analysis so far: "stage all pages" is a RAM purchase — Qwen needs 75.5 GB (impossible on 16 GB, and its 8 GiB single-buffer kernel cap blocks full residency even on big Macs), Nemotron needs 15.38 GiB (achievable at 32 GB+). "ANE + GPU concurrently" is real but both share one memory controller (measured +3.5–17.6% contention in decode-shape work), and decode's dependency chain means there's no independent work to hand ANE without speculation that already failed. Prefill is the only plausible window (compute-bound, all tokens known up front); the one interesting variant is splitting *token chunks* data-parallel across ANE and GPU — but that needs duplicate weights and careful KV handling, and no room on 16 GB. Overall: right idea for 32 GB+ prefill, not for 16 GB decode.

## Questions I want answered

1. Is my analysis of the ANE+GPU concurrency idea correct, or am I missing a scheduling trick that actually works on 16 GB? Be specific about which layer(s) of which model could run on ANE without breaking the dependency chain or quality gate.
2. Given maderix's private-API route removes the 21 ms marshalling, is there a concrete, realistic path to **faster prefill on the 16 GB M4** for either model? What would you build first, and what is the expected gain range? I want a design, not encouragement.
3. The bit-identical rule disqualifies FP16 ANE numerics. Is there any way to use ANE for a **non-output-affecting** task that still shortens prefill or decode (e.g. expert prefetch prediction, cache admission, draft/verification, KV work) without changing tokens? Explain why it would or would not beat the measured baselines.
4. What is the single biggest thing I have NOT considered in this whole picture — a technique, a measurement, or a hardware fact that could change the conclusion? Search your knowledge and the web for 2025–2026 work on MoE expert streaming, NPU+GPU co-execution, or ANE inference that is NOT in my rejection list.
5. If the honest answer is "no ANE win on 16 GB," say so bluntly and tell me what the highest-value engineering work would be instead, given the measured bottleneck.

Please structure your answer: (1) verdict up front, (2) reasoning with numbers, (3) concrete next experiments ranked by expected value, (4) explicit list of what you are uncertain about. Do not tell me to "just benchmark" — I need permission for that and the point is to decide what is worth asking permission for.
