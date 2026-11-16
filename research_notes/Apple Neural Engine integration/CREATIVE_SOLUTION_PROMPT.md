# Creative solution prompt: Veystrix ANE + prefill/decode speed

Copy everything below this line into the other AI (e.g. Fable 5.2).

---

You are a brilliant, pragmatic Apple Silicon systems engineer. I want you to be **encouraging and constructive**. My previous advisor kept concluding "no, it can't work" — I don't want that. Your job is to **find a way to make it work**, or to tell me the smallest change in constraints that would make it work. Be creative and generous with ideas. Assume there is a clever solution until you prove otherwise.

BUT: I also need to trust you. So for every idea, give it an honest evidence label: **[measured]** (I have a number), **[cited]** (a real source with URL and date), **[estimate]** (arithmetic from measured numbers), or **[idea]** (your proposal, unproven). Do not present an [idea] as a result. Do not just tell me to "benchmark it" — I need permission for every hardware run, so the point is to decide what is worth asking permission for. If the honest answer to part of the question is "this needs more RAM or a relaxed quality rule," say so plainly and then design the best version under that new condition. I can handle the truth; I just want you on my side, looking for the yes.

## The machine and the rules

Veystrix is a free macOS app (SwiftUI shell + Python engine on MLX) running two local 4-bit Mixture-of-Experts models on a **fanless 16 GB M4 MacBook Air**, AC plugged in for all tests:

- **Flash-Next / Qwen 3.8 Flash-Next** (~105 GB on disk): 48 layers (12 full-attention, 36 linear-attention/GDN), hidden 2560, 512 experts/layer, exact top-10 routing. **One whole expert = 3,072,000 bytes (~3 MB)**, made of nine separate regions (three ~820 KB weights + six ~100 KB scale/bias blocks). Total experts **75.5 GB**. QSA sparse attention, PLE n-gram table, 3.4 GB dense core. Its decode pool is one MLX `uint32` array → **8 GiB hard kernel cap**, so full residency is impossible even on big Macs without a kernel redesign.
- **Nemotron 3.5 Lightning 30B-A3B** (~24 GB): 52 layers (13 attention, 23 Mamba, 16 MoE), hidden 2688, 128 experts, **15.38 GiB experts**, 1.25 GiB dense core. Fully resident at 32 GB+ (implemented, not hardware-validated).

Hard rules: outputs must stay **bit-identical** (exact token IDs and BF16 rounding boundaries); Q4 affine G32 weights, exact top-10-of-512 routing and exact softmax/renorm must be preserved; every change opt-in with rollback; safety guard (≥3 GiB available, stop on swap/compressor growth, thermal level 2, battery <25%); no benchmark without my explicit permission.

## Measured bottlenecks (our own runs, guarded, plugged in)

- **Qwen decode:** 385 ms/token = **289 ms SSD expert-read wait (75%)**, 51 ms router sync, ~70 ms GPU → 2.4–2.7 tok/s. 480 experts/token × ~3 MB = 1,253 MB with no cache hits; with the app's 1 GiB/349-slot pool (~24% hit rate) ~0.95 GB physical. SSD does 3.46 GB/s at ≥1 MB but only 2.3–2.9 GB/s for the 345 KB per-region reads. Each expert is read as nine separate regions.
- **Qwen prefill:** 4,096-token cold prefill 48.2 s: full attention 17.2 s (12 layers), GDN 8.2 s (36 layers), hyper-connections 5.0 s, PLE 3.9 s, MoE incl. read waits 13.2 s. Reads 83.8–139.6 GB for 4.3K tokens. All-exact-flags version ~41.2 s.
- **Nemotron decode:** ~120 ms/token, ~75 ms expert wait, ~8 tok/s, 179 MB/token physical at 77% page-cache hit.
- **Nemotron prefill:** 8K 252 tok/s (32.3 s: Mamba 9.7 s, experts 18.2 s incl. 4.6 s SSD wait, attention 4.3 s); ~355 tok/s after accepted speed defaults.

Both are SSD/RAM-bound, not GPU-bound. Note the byte math is already self-consistent — there is no unexplained read amplification.

## What is already done (so don't re-propose it)

Default-on, bit-identical 24/24 quality-gated speed set: tiered4 KV, 4K hybrid layer-major prefill, single-chunk layer-major, merge-small-tail, shared dense KV, fused gather, fused SDPA, ngram-dedup, dense GEMM, S3-FIFO pool, gather tile 32, Nemotron bit-equal dense GEMM. Gains +14–36% prefill, +5% decode.

Measured and **rejected** (don't suggest without new evidence): decode prefetch/overlap; more I/O threads (6–32 all ~2.55 tok/s); pool policies other than S3-FIFO; larger pools on 16 GB (guard floor); route-union prefill (30.5 vs 29.9 s); PLE row prefetch; Nemotron read-ahead/stale-drop; unaligned mmap/zero-copy; native MTP at all depths; lookup drafting v2; parallel Mamba (fails exactness); 4K chunks (memory); qsa-fast-prefill (quality gate); gather tile >32; wired limit; contiguous expert packing (66.6 vs 66.1 s server-side); `F_NOCACHE` for experts (8% slower); `F_RDAHEAD=0` (1.3%, below resolution). A **perfect** expert predictor with overlapped reads tied baseline (+0.1/−2.9/−0.7%) and early submission lost 8.2% — memory-controller contention. `AGENT_INVARIANTS.md` says no benchmarks without permission.

## The ANE findings so far

1. **Our own 2026-11-07 experiment on this exact machine** rejected ANE: batch-1 predictors never reach ANE (CPU, 35 µs); routing 98.8% exact (fails bit-identical); ANE side-load adds +3.5% to +17.6% to bandwidth-bound GPU work; duplicate FP16 weights don't fit.
2. **Our 2026-11-17 follow-up:** a real Nemotron `q_proj` runs on ANE in 6.3 ms vs 17.6 ms MLX GPU (FP16), INT8 5.24 ms. But end-to-end via Core ML/Python was 3.3% slower (279 vs 289 tok/s) due to ~21 ms/call marshalling; memory floor 3.6 GiB. Palettized 4-bit silently ran on GPU and lost accuracy. Native zero-copy buffers segfaulted; Apple's single-allocation APIs (IOSurface-backed `MLMultiArray`, page-aligned `dataPointer`+deallocator, `MLTensor.init(bytesNoCopy:)`, `outputBackings`) need a native Swift bridge.
3. **maderix/ANE (MIT, 2026):** private APIs (`_ANEClient`, `_ANECompiler`, `_ANEInMemoryModel`) let him run arbitrary MIL graphs on ANE, including full backprop. Dispatch ~60–80 µs (250× cheaper than Core ML), working GPU↔ANE zero-copy via shared IOSurface. Fork `m0at/ANEtransformers`: ANE prefill 76.6 tok/s on Mistral 7B (M5) — but MLX does 200+ tok/s on the same chip because CPU round-trips (RMSNorm/RoPE/attention/transposes), not ANE compute, dominate. ANE via private API: FP16 only, weights baked into compiled programs (~128 programs for 7B), no reduce ops, max 2 outputs, min sequence 16/32, 119-compile leak, cosine 0.998 (not bit-identical).
4. **hollance/neural-engine:** you can't force Core ML onto ANE; shared memory still needs format conversion; gather/custom/RNN/ND layers unsupported; ANE is FP16.

## The user's idea to evaluate creatively

"Model stays on SSD and is read layer by layer. What if we dip all pages at a time (stage experts in RAM up front), give ANE the jobs it's good at, run the GPU at the same time, and never let anything stop?"

The obvious objections: 75.5 GB of Qwen experts can't fit in 16 GB; decode is a serial chain so there's no independent work for ANE; both processors share one memory controller. **Do not stop at those objections.** Instead: find the version of this idea that works. Be specific about which layers, which model, which phase (prefill vs decode), and what has to be true for it to win. If it only works on a 32 GB Mac, design that version and say exactly what hardware is needed.

## What I want from you

1. **Lead with the most promising path you can find** — a concrete design, not a list of objections. If it's a version of the user's idea, develop it fully. If you conclude one phase/model can't work, immediately offer the next-best thing that can.
2. Find the **non-obvious opportunity** in the data — something I have not considered. Look for: asymmetric use of ANE and GPU on different parts of the *same* layer; batch-K verification tricks that preserve bit-identical digests; better expert-read scheduling; anything from 2025–2026 research on MoE expert streaming or NPU+GPU co-execution not in my rejected list. Verify any paper you cite (URL + date) and flag unreplicated results.
3. For your top 3 ideas, give: the mechanism, the exact expected gain range with arithmetic, the main risk, and the first cheap test (no hardware run) that would tell us whether to ask permission for a real benchmark.
4. A short "what would make the impossible possible" section: the minimal relaxation of a constraint (RAM, bit-identical rule, precision) that unlocks a big win — so I can make an informed decision about whether to accept it.

Be warm, specific, and imaginative. I want a plan I can get excited about, with enough honesty that I can act on it.
