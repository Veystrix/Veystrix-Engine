# Can the Apple Neural Engine (ANE) Deliver the "Second Miracle" for Veystrix on a 16 GB M4 Air under Bit-Exact / No-New-Hardware Constraints?

> Scope: exhaustive, evidence-first hunt to OVERTURN the prior rejection of ANE. Prior rejection to beat: GPU 74.9 vs ANE 23.2 tok/s on a test layer; ~12× lower ANE bandwidth; no INT4; no bit-exact ANE layer path; decode is SSD-read bound (873 MB/token, 3.36 GB/s wall = 260 ms drive time). Hard constraints: no second SSD, no second Mac, no more RAM, NO quality loss (bit-identical tokens/logits, exact routing). Targets: Qwen 5+ tok/s, Nemotron 15+ tok/s.

## KEY QUESTION 0 — Bottom-line verdict and plausibility × payoff ranking

### Takeaway
The ANE cannot deliver the second miracle under these constraints. Three independent, well-sourced facts kill it: (1) ANE cannot touch the SSD — it only reads unified memory, so it cannot reduce or bypass the 873 MB/token byte wall; (2) where the byte wall could be attacked by compression, the only ANE-supported compression that helps (int4 lookup-table palettization) is provably lossy (2.55 points ImageNet top-1, greedy-token rank correlation 0.82 vs 0.998 for int8) and CoreML/ANE does not support the grouped-affine (per_block, i.e. G32/G64) format Veystrix uses — it falls back to CPU; (3) ANE has *lower* effective memory bandwidth than the GPU (44–70% of GPU on MacPaw's M4 Pro measurements; ~24 GB/s activation path, ≤143 GB/s weight stream on M5 Pro, ~2.4–6.6× above a ~10 GB/s M4 base activation path) and is *not bit-exact* with GPU/CPU (ANE fp16 relative error ~2.5e-2 vs GPU 3.2e-3; CoreML ANE export flipped up to 85 decisions in one validated runtime). The ANE's genuine win is energy (~2× tok/s/W), not latency. Best measured end-to-end decode: GPU 200 tok/s vs ANE 115 tok/s at B=1, widening to 2322 vs 847 at B=32 — the ANE is the *slower* decode device.

### Cited Findings
- Decode on the tested TinyLlama-class loop is weight-stream-bound on both engines; GPU is 1.74× faster at B=1 (200 vs 115 tok/s) and 2.7× faster at B=32 (2322 vs 847); ANE's weight stream delivers 1.20 GB at an effective 154 GB/s vs GPU ~240 GB/s — [arXiv 2611.19101, §9](https://arxiv.org/html/2611.19101v1).
- ANE attains only 44–70% of the GPU's effective memory bandwidth while its compute throughput is 2.0–2.7× higher; efficient ANE decode therefore *requires* amortizing each weight read over several tokens — [MacPaw Research](https://research.macpaw.com/projects/neural-engine).
- int4 LUT stream on ANE: 203 tok/s but greedy-token rank correlation 0.82 (vs 0.998 int8); costs 2.55 ImageNet top-1 points; int8 holds the greedy token only 92% of the time vs fp64 — [arXiv 2611.19101, §9](https://arxiv.org/html/2611.19101v1).
- CoreML issue #2510 (Apple triaged/closed): "per_block" (grouped) quantization is not supported on ANE — ANE only supports per-cout/per-tensor; affected tensors fall back to CPU; palettization works best on ANE — [apple/coremltools#2510](https://github.com/apple/coremltools/issues/2510).
- On a validated heterogeneous runtime, an ordinary CoreML ANE export changed decisions vs upstream PyTorch (up to 85 hard mismatches); a parity gate was required to use ANE artifacts — [mlx/discussions/4551](https://github.com/ml-explore/mlx/discussions/4551).
- ANE is bit-deterministic for a *fixed graph + fixed input* (raw fp16 output bytes identical on M1) — but this does not imply bit-equality to a GPU/CPU reference — [ANE architecture/performance survey](https://www.researchgate.net/publication/407509615_Apple_Neural_Engine_Architecture_Programming_and_Performance).
- ANE 16-core; M4 Air spec 120 GB/s memory bandwidth, no dedicated matrix unit on M4 GPU (arrives M5) — [Apple M4 Tech Specs](https://support.apple.com/en-us/122209).

### Plausibility × payoff ranking (1 = best bet, 5 = worst; brute-force honest)

| Rank | Angle | Mechanism | Bit-exact? | Realistic number | Plausibility | Payoff | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | Q3: bit-exact Q4 G32/G64 GEMV on ANE | Direct/fused dequant+GEMV | No (CoreML per_block→CPU; fp16 err 2.5e-2) | 115 vs GPU 200 tok/s; lossy | Very low | Medium | Falsifiable in one afternoon; expect fail |
| 2 | Q1: ANE reduces SSD bytes | Lossless decompressor / compressed fused GEMV | No for ANY lossy encoding; lossless codec not implementable on matmul-only ANE | 0 bytes saved | Essentially zero | Very high (bytes = the wall) | Mechanistically dead |
| 3 | Q2: ANE hides I/O latency | Concurrent ANE + SSD/GPU, fences | N/A (timing) | Zero-copy works but 0.95× copy path; fence SIGSEGV; SSD serial | Low | Low–medium | Probe, but SSD serializes anyway |
| 4 | Q4: ANE does router top-k | Exact top-k on ANE | Risky (logit error flips near-ties) | Host is only 6%; ≤1 ms saved | High technically | Near-zero | Not worth it |
| 5 | Q5: published exact MoE-on-ANE path | Literature | No published bit-exact claim | NPUMoE targets prefill, dynamic ops stay CPU/GPU | No | N/A | No such work exists |

### Inferences
- The bottleneck invariant is unchanged by ANE: 873 MB/token at 3.36 GB/s = ~260 ms drive time. To hit Qwen 5+ tok/s (≤200 ms/token) or Nemotron 15+ tok/s (≤66.7 ms/token) requires 4.4–13 GB/s of *effective SSD reads*, i.e. 1.3–4× more bytes/sec than measured. No compute engine — ANE included — changes the number of bytes that must cross the SSD bus unless the on-disk representation itself shrinks losslessly.
- ANE is a poor tool for the byte wall specifically because its fast path (weight stream) tops out at ~143 GB/s on M5 Pro and ~24–66 GB/s on base M4 — both far *above* the 3.36 GB/s SSD, so ANE is never the bottleneck once bytes are resident; but it cannot read those bytes from disk any faster than the GPU/CPU. Moving compute to ANE cannot make the SSD faster.
- The ANE's only proven benefit at decode is energy (roughly 2× tok/s/W, 21 W vs 48 W system power), which matters for battery/thermals, not for tok/s on an AC-powered Air chasing 5/15 tok/s.

### Gaps
- No published measurement of ANE on the exact M4 *Air* (fanless, base M4, 16 GB) with SSD-resident MoE weights; closest is base M4 Mac mini in the roofline paper and M4 Pro/Max in MacPaw/Orion. Base-M4 Air thermal envelope during sustained ANE load is unmeasured.
- No source gives the exact SSD→unified-memory DMA rate when the consumer is the ANE rather than the GPU; assumption is that SSD read is engine-agnostic (both read the same unified-memory buffers).
- Exact per-token bytes for Veystrix's Qwen and Nemotron configurations are internal (873 MB/token given in context); external sources cannot confirm whether Qwen/Nemotron differ.

## KEY QUESTION 1 — Can ANE reduce SSD BYTES (bit-exact decompressor, fused dequant+GEMV from compressed bytes, or exact expert-subset prefilter)?

### Takeaway
No. ANE is a fixed-function matmul/conv engine reachable only as a compiled graph; it has no general integer/streaming decompressor, so it cannot be a bit-exact decompressor for a lossless encoding. Its one compressed-weight feature (int4 LUT palettization streamed during DMA) is lossy and fails bit-exactness. Grouped-affine Q4 (G32/G64 = CoreML per_block) is unsupported on ANE and falls back to CPU. A "prefilter that identifies an exact expert subset" cannot exist without changing routing, which is forbidden. Verdict: 0 bytes/token saved; mechanism dead.

### Cited Findings
- The ANE weight-stream path can decompress weights *during the DMA* via a compile-time constant operator; aneforge emits int4 lookup-table palettization and unstructured sparse forms, and the compression is explicitly lossy: "an int4 lookup-table stream raises throughput and efficiency at a cost of 2.55 points of ImageNet top-1" — [arXiv 2611.19101, Abstract & §3.3](https://arxiv.org/html/2611.19101v1).
- The three constant operators have distinct rates: fp16 `dense const` 140 GB/s / 70 Gelem/s; int8 `constexpr_affine_dequantize` 74 GB/s / 74 Gelem/s; int4 `constexpr_lut_to_dense` 61 GB/s / 122 Gelem/s; int4 reaches 203 tok/s but is numerically wrong on the decode model (rank corr 0.82 vs int8 0.998) — [arXiv 2611.19101, §9 & Table S26](https://arxiv.org/html/2611.19101v1).
- Grouped/blockwise quantization (G32/G64 analogue) is not an ANE operator: "ANE only support per-cout/per-tensor quantization"; per_block falls back to CPU; a blockwise_shift_scale constant operator measured at 112.5 tok/s — slower than fp16's 116.3 — [apple/coremltools#2510](https://github.com/apple/coremltools/issues/2510); [arXiv 2611.19101, Table S26](https://arxiv.org/html/2611.19101v1).
- ANE supports linear INT4/INT8 quantization and palettization in Core ML Tools, but palettization (LUT) is what "works best" on ANE, not affine group quantization — [coremltools opt-overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html); [apple/coremltools#2510](https://github.com/apple/coremltools/issues/2510).
- ANE weights are baked at *compile time*; changing them requires unload/patch/reload (494 ms per step) — there is no runtime "decompress arbitrary blob" path — [Orion, arXiv 2603.06728](https://arxiv.org/abs/2603.06728).
- NPUMoE explicitly keeps dynamic operations (top-k, scatter/gather, routing) on CPU/GPU because they are "not NPU-friendly"; it offloads only dense, static computation, and targets prefill, not decode — [NPUMoE, arXiv 2604.18788](https://arxiv.org/abs/2604.18788).

### Inferences
- "Lossless + bit-exact + fewer bytes" would require a general decompressor (zstd/LZ) implemented as an ANE graph, but ANE's ISA, as exposed by CoreML/aneforge/Orion, offers only matmul/conv/elementwise/reduction/softmax primitives and compile-time constant tensors — no byte-stream decoder. Any compression that *does* help (int4 LUT, sparse) is lossy and violates the no-quality-loss constraint.
- Even if bytes were compressed losslessly, the dequantized values must match the current Q4 path bit-for-bit. ANE's fp16 math has ~8× the relative error of the GPU (2.5e-2 vs 3.2e-3), so a fused ANE dequant+GEMV cannot be bit-identical to the GPU path even with identical input bytes — [arXiv 2611.19101, §9](https://arxiv.org/html/2611.19101v1).
- A bit-exact "expert subset prefilter" is self-contradictory: selecting a smaller exact subset is exactly changing routing, which the hard constraints forbid. The only allowed prefilter would be one that provably returns the same top-k as the original router for all inputs — which requires computing the original router logits exactly anyway.

### Gaps
- No source demonstrates a lossless codec running on ANE, or a bit-exact grouped-affine dequant on ANE. Searches returned none; I treat this as strong negative evidence, not proof of impossibility for undocumented private APIs.
- No published bit-exactness comparison between ANE `constexpr_affine_dequantize` (int8) and a GPU Q4 dequant path at the logit level; the closest is greedy-token agreement (92%), which is not bit-exactness.

### ONE falsifiable experiment (Angle 1)
Take one expert's exact on-disk Q4 bytes and build an ANE graph that ingests the compressed blob and emits the same dequantized values + GEMV as the current GPU path. Diff output logits bit-for-bit against the GPU Q4 path over 1,000 random routing decisions, and count SSD bytes/token. **Falsified if any logit bit differs or bytes/token do not drop.** Predicted outcome: fails immediately (CoreML per_block→CPU; any ANE LUT/sparse form is lossy; fp16 rounding breaks bit-identity).

## KEY QUESTION 2 — Can ANE hide I/O latency (run on ANE while SSD reads happen on GPU/CPU stream with cross-stream events)? What is ANE's real bandwidth to unified memory vs SSD?

### Takeaway
Partially mechanism-possible, payoff-negative. Zero-copy GPU→ANE over a shared IOSurface has been demonstrated on M4 (same base address, linear output, CPU-diff verified), but a proper cross-stream fence (MTLSharedEvent→IOSurfaceSharedEvent→_ANESharedWaitEvent) crashes with SIGSEGV, and the zero-copy timing was 0.95× the copy path (i.e. slightly slower) in a later run. More fundamentally, SSD reads are serial and engine-agnostic; overlapping ANE compute with SSD prefetch cannot increase SSD throughput, and ANE can already consume weights far faster than 3.36 GB/s. Verdict: no tok/s gain; at best hides the ~6% host time.

### Cited Findings
- Zero-copy GPU→ANE on M4 via one IOSurface is verified end-to-end (bricks 1–4): GPU writes surface, ANE reads same base address, output linear in GPU data, checked against a CPU reference — [kai-ane README, M4](https://github.com/crafter-agents/kai-ane).
- The fence chain needed for true pipelining builds but evaluation with a wait event attached crashes with SIGSEGV; an immediate rerun also saw newSharedEvent return nil; ANE `perfStats` init returns nil and `hwExecutionTime` reads 0 — [kai-ane README](https://github.com/crafter-agents/kai-ane).
- Zero-copy timing is noisy: recorded 1.13–1.40× faster in one run, but a later verification measured 0.95× (slightly slower) than the copy path — [kai-ane README & brick4-evidence.txt](https://github.com/crafter-agents/kai-ane).
- ANE effective bandwidths: M5 Pro ~24 GB/s standalone-activation, 112–143 GB/s weight-stream; base M4 activation ceiling ~10 GB/s with a weight path 2.4–6.6× higher — both orders of magnitude above the 3.36 GB/s SSD — [arXiv 2611.19101, §§3.2–3.3](https://arxiv.org/html/2611.19101v1).
- ANE achieves 44–70% of GPU effective memory bandwidth on MacPaw's M4 Pro — [MacPaw Research](https://research.macpaw.com/projects/neural-engine).
- Concurrent GPU+ANE serving works for *independent requests* (M4 Max: 9.9 ms ANE vs 12.2 ms GPU per short request; gain from queue isolation, not a faster engine), but "requests are never split across the two engines" — [mlx/discussions/4551](https://github.com/ml-explore/mlx/discussions/4551).
- Passing KV cache as a graph input every step (host-feeding) costs the ANE 0.51 ms at KV=256 rising to 4.38 ms at KV=2048 (128 MB/token) and erases its efficiency advantage by KV≈1300 — [arXiv 2611.19101, §9](https://arxiv.org/html/2611.19101v1).

### Inferences
- Hiding I/O only works if there is idle time to fill. Here the ANE compute for a token (~a few ms once weights are resident) is much shorter than the 260 ms SSD read; the CPU/GPU host time is only 6% of the budget. Overlapping ANE with SSD saves at most that ~6% (≈15 ms), yielding maybe 3.9→4.2 tok/s — still short of Qwen 5+ and far short of Nemotron 15+.
- The broken fence is the immediate engineering blocker: without a working wait event the ANE must be CPU-synchronized between stages, collapsing the "pipeline" into sequential-on-shared-memory, which yields no latency overlap.
- The roofline paper's re-fed KV experiment is a cautionary analogue: host marshalling of data into the ANE creates exactly the "engine idles while host marshals" signature that would negate any I/O-hiding scheme.

### Gaps
- No direct measurement of ANE↔SSD concurrent DMA on M4 Air; no source details whether `_ANESharedWaitEvent` can bind to an SSD-completion event (only to Metal events, and that path crashes).
- No measurement of whether macOS 27 "Core AI" exposes a supported cross-stream event API for ANE (MacPaw used it for split execution but did not describe the sync primitive) — [MacPaw Research](https://research.macpaw.com/projects/neural-engine).

### ONE falsifiable experiment (Angle 2)
On the M4 Air, reproduce kai-ane brick 3/4 with a real SSD streaming pattern: while the GPU/CPU issues an async read of layer L+1 weights into an IOSurface, run layer L on the ANE, gated by a working MTLSharedEvent→_ANESharedWaitEvent fence. Measure wall ms/token vs the serial baseline. **Falsified if the wait-event fence still SIGSEGVs, or if tok/s does not improve beyond the ~6% host-time fraction.** Predicted: fence fails or gain <5%.

## KEY QUESTION 3 — CoreML/ANE 2024–2026: MLX ANE backend, mlx-ane, coremltools, int4/int8/bf16 GEMV, bit-exactness, dispatch latency, M4. Can we run a Q4 affine G32/G64 GEMV bit-exactly on ANE?

### Takeaway
No. MLX (the framework Veystrix's stack most likely uses) has no ANE backend; ANE access is via CoreML, the new macOS 27 "Core AI", or private `_ANEClient`/e5rt paths. CoreML/ANE does not support grouped (per_block, G32/G64) affine quantization — it falls back to CPU and only supports per-cout/per-tensor or palettization. ANE's int8 is dequantized to fp16 before compute (bandwidth-only saving), its fp16 GEMV error accumulates as √K, and dispatch floors are ~136 µs/call. A bit-exact Q4 G32/G64 GEMV on ANE is not achievable with any documented path.

### Cited Findings
- "Even Apple's MLX framework has no ANE support" / "MLX doesn't support the neural engine"; Apple itself routes ANE work through CoreML/Core AI, not MLX — [Hacker News discussion](https://news.ycombinator.com/item?id=47780473); [Reddit r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1cz5xuv/apple_has_not_released_any_capable_opensource_llm/); [MacPaw Research](https://research.macpaw.com/projects/neural-engine).
- ANE reachable via Core AI in macOS 27 (operator-placement control), or by bypassing CoreML through `_ANEClient` → `ANECompiler` (XPC) → `AppleH16ANEInterface.kext` → hardware; Orion uses private `_ANEClient`/`_ANECompiler` with a catalog of 20 MIL/layout/numerical restrictions — [MacPaw Research](https://research.macpaw.com/projects/neural-engine); [kai-ane README](https://github.com/crafter-agents/kai-ane); [Orion, arXiv 2603.06728](https://arxiv.org/abs/2603.06728).
- "INT8 == FP16 throughput: the ANE dequantizes INT8 to FP16 before compute; INT8 only saves DRAM bandwidth, not cycles." True peak ~19 TFLOPS FP16 (marketing 38 TOPS = 19×2 INT8); conv > matmul 3×; SRAM cliff at ~32 MB working set — [maderix/ANE via kai-ane README](https://github.com/crafter-agents/kai-ane).
- ANE GEMM error grows with K (relative error 2.6e-3 at N=512 → 6.2e-2 at N=8192), consistent with per-product fp16 rounding over a long contraction; ANE is consistently less accurate than GPU in fp16 — [arXiv 2611.19101, §3.1](https://arxiv.org/html/2611.19101v1).
- Dispatch cost: ANE Edisp=681 µJ and Tdisp=136 µs per call (M5 Pro); at a fused single-token decode the dispatch floor (~70 µs) is <1% of the 8.7 ms token — [arXiv 2611.19101, §5 & §9](https://arxiv.org/html/2611.19101v1).
- CoreML issue #2510: "ANE only support per-cout/per-tensor quantization"; per_block (grouped) falls back to CPU; Apple's guidance says per-block is "mainly for GPU" and palettization works best on ANE — [apple/coremltools#2510](https://github.com/apple/coremltools/issues/2510).
- Quality is "within the 8-bit quality class," not bit-exact: MacPaw reports 98.4% teacher-forced top-1 agreement with bf16 (vs 99.0% for MLX 8-bit) and required a parity gate elsewhere — [MacPaw Research](https://research.macpaw.com/projects/neural-engine); [mlx/discussions/4551](https://github.com/ml-explore/mlx/discussions/4551).

### Inferences
- A Q4 affine G32/G64 GEMV is, in CoreML terms, per_block/grouped int4 quantization — exactly the mode that is unsupported on ANE. The supported alternative (int4 LUT palettization) is lossy. Therefore no documented ANE path can consume Veystrix's existing Q4 G32/G64 weights at all, let alone bit-exactly.
- Even if the format were supported, bit-exactness to the GPU reference is impossible: ANE int8→fp16 dequant and long-contraction fp16 rounding produce different bits; ANE is bit-deterministic only in the weak sense (same graph+input → same bytes), not reference-equivalent.
- Feasibility on the target machine is further reduced: Core AI requires macOS 27, aneforge/Orion/kai-ane require private APIs (App Store / production risk, and may break at any OS update), and none of those tools are integrated into MLX.

### Gaps
- No source describes a public, stable, bit-exact ANE GEMV API for int4 affine group quantization; none found.
- It is unclear whether Veystrix's "Q4 affine G32/G64" is bit-identical to CoreML per_block or uses a distinct scale/zero-point convention; this affects whether it would even map, but either way ANE lacks the operator.
- No logit-level bit-diff study between ANE fp16 and GPU bf16 on a MoE model (closest: 98.4% top-1 agreement).

### ONE falsifiable experiment (Angle 3)
Convert one Q4 G32/G64 layer to a CoreML `.mlmodelc` with `compute_units="ALL"`, profile which device executes the dequant+GEMV, and diff ANE vs current GPU logits bit-for-bit for 100 tokens. **Falsified if the profiler shows CPU fallback for the grouped dequant (predicted per issue #2510) or if any logit bit differs.** This is the fastest, cleanest kill-shot: expect failure on the very first check.

## KEY QUESTION 4 — Could ANE do the ROUTER (top-k selection) exactly, freeing GPU/CPU, and would that matter (host time only 6%)?

### Takeaway
Technically easy (ANE has a topk operator) but strategically irrelevant and risky. The MoE router is tiny; host time is already only 6% of the token budget, so even a perfect ANE router saves ≲1 ms/token. Worse, exact routing is mandatory, and ANE's fp16 logit error (~2.5e-2 relative) can flip near-tie top-k decisions — a single flipped expert changes which weights are read and breaks bit-identical output. NPUMoE and the Laya runtime both keep dynamic routing off the ANE for exactly these reasons. Verdict: not worth it.

### Cited Findings
- ANE exposes topk/argmax operators; they are among the few bridge operators with a persistent worker and are timed on silicon — [arXiv 2611.19101, §2 Exclusions](https://arxiv.org/html/2611.19101v1).
- NPUMoE keeps routing, top-k, scatter/gather on CPU/GPU because they are "not NPU-friendly" and introduce dynamic shapes conflicting with NPU shape constraints; it offloads only static dense compute — [NPUMoE, arXiv 2604.18788](https://arxiv.org/abs/2604.18788).
- A MacPaw "prompt router on the ANE" is a 350M *auxiliary encoder* (5.5 ms at 128 tokens), not an MoE token router; it left chat generation rate <1% affected — [MacPaw Research](https://research.macpaw.com/projects/neural-engine).
- ANE numerics are not reference-exact: fp16 decode relative error 2.5e-2 vs GPU 3.2e-3; CoreML ANE exports have produced up to 85 hard decision flips vs PyTorch — [arXiv 2611.19101, §9](https://arxiv.org/html/2611.19101v1); [mlx/discussions/4551](https://github.com/ml-explore/mlx/discussions/4551).
- Host time is only 6% of Veystrix's budget (given in context); dispatch floor per ANE call is ~136 µs — [arXiv 2611.19101, §5](https://arxiv.org/html/2611.19101v1).

### Inferences
- The maximum theoretical gain is bounded by Amdahl's law: removing all host work (6%) caps speedup at 1/0.94 ≈ 1.064×, i.e. ~3.9→4.1 tok/s for Qwen — still below 5, and irrelevant to the 15 tok/s target.
- The risk asymmetry is bad: a single near-tie routing flip (well within ANE's fp16 error) breaks the bit-identical-routing constraint and corrupts the entire token's expert selection, so any latency win is unacceptable under the hard constraints.

### Gaps
- No published measurement of ANE MoE router top-k agreement with a GPU/CPU router at the exact-index level; the closest are auxiliary-encoder timings and decision-mismatch counts for classification models.
- No data on how often Veystrix's Qwen/Nemotron routers have near-ties (which would quantify flip risk); this is internal.

### ONE falsifiable experiment (Angle 4)
Implement the exact router on ANE (or Core AI) and compare selected top-k expert indices bit-for-bit against the current GPU router over 10,000 tokens, while measuring ms/token saved. **Falsified if any expert index differs OR if savings are <1 ms/token.** Predicted: either a routing mismatch appears, or the gain is immaterial.

## KEY QUESTION 5 — Any published work running MoE expert GEMV or dequantization on ANE with exact output?

### Takeaway
No. The closest published work either targets prefill and keeps dynamic routing/dynamic ops off the ANE (NPUMoE), uses direct ANE access for dense models without bit-exact claims (Orion, aneforge), or measures energy/throughput with acknowledged numeric error (MacPaw, roofline paper). No source reports bit-identical tokens/logits or exact routing for MoE expert GEMV/dequant on ANE. The literature instead documents the obstacles: dynamic shapes, top-k/scatter being NPU-unfriendly, lossy compression, and decision flips.

### Cited Findings
- NPUMoE (Benazir & Lin, Apr 2026): MoE on Apple NPUs is hard because routing is unpredictable, top-k/scatter/gather are not NPU-friendly, and many small-kernel launches incur dispatch/sync overhead; solution offloads only dense static compute, keeps CPU/GPU fallback for dynamic ops, targets prefill; reports 1.32–5.55× latency and 1.81–7.37× energy improvements — no bit-exactness claim — [NPUMoE, arXiv 2604.18788](https://arxiv.org/abs/2604.18788).
- Orion (Kumaresan, Mar 2026): direct ANE execution of GPT-2 124M at 170+ tok/s on M4 Max; 20 MIL/layout/numerical restrictions; weights baked at compile time; no bit-exact claim; independent report finds CPU decode outpacing ANE on M4 Max (283 vs 170 tok/s) — [Orion, arXiv 2603.06728](https://arxiv.org/abs/2603.06728); [arXiv 2611.19101, §11](https://arxiv.org/html/2611.19101v1).
- aneforge/e5rt fused-graph dispatch (Bryngelson, Nov 2026): measures ANE vs GPU/CPU with energy accounting; explicitly reports that compression encodings are lossy and that the ANE is less accurate than the GPU — [arXiv 2611.19101](https://arxiv.org/html/2611.19101v1).
- MacPaw Core AI engine (2026): ANE decode for LFM2.5-2.6B with DSpark speculation; 98.4% top-1 agreement with bf16 (not bit-identical) — [MacPaw Research](https://research.macpaw.com/projects/neural-engine).
- Laya/ANE heterogeneous runtime uses a parity gate because CoreML ANE export changed up to 85 decisions — [mlx/discussions/4551](https://github.com/ml-explore/mlx/discussions/4551).
- A known open question in the community is whether ANE decode can even beat the GPU/CPU; evidence says no on M4 Max/M5 Pro — [arXiv 2611.19101, §9 & §11](https://arxiv.org/html/2611.19101v1).

### Inferences
- The research frontier is explicitly about *getting acceptable accuracy and energy* from ANE, not bit-exact equivalence. Under Veystrix's no-quality-loss constraint, none of the published ANE techniques are admissible.
- The one setting where ANE decode approaches GPU throughput is the lossy int4 LUT arm (203 vs 200 tok/s on the synthetic model), and that arm is numerically wrong (rank corr 0.82) — so the only "competitive" ANE decode is the one that violates the constraint.

### Gaps
- No exhaustive survey of private/undocumented ANE work (maderix, Asahi Linux driver, Reddit reverse-engineering threads) reporting bit-exact MoE expert GEMV; based on abstracts and readmes, none claims it, but private-API projects are incompletely documented — [Asahi-Linux ANE driver cited in arXiv 2611.19101 §11](https://arxiv.org/html/2611.19101v1).
- NPUMoE's full text was not read (only abstract); if any appendix reports exact-output MoE expert GEMV, this would change the verdict. Flagged as the one literature item worth a deeper read.

### ONE falsifiable experiment (Angle 5)
Replicate the NPUMoE expert-GEMV offload on the M4 Air's Qwen/Nemotron experts and diff output logits bit-for-bit against the GPU path for 1,000 tokens; require exact top-k indices too. **Falsified if any logit or expert index differs** — expected, given NPUMoE's loss-tolerant framing and ANE's numeric error.

## Cross-cutting note: the constraint is the SSD byte wall, not compute
Every mechanism above fails for the same reason: Veystrix's 873 MB/token at 3.36 GB/s (~260 ms) is a DRAM/SSD I/O problem. ANE is a compute accelerator with *lower* bandwidth than the GPU and no direct SSD path. Compression that would cut bytes is lossy on every ANE-supported encoding; compression that is lossless cannot run on ANE. Concurrency cannot make a serial SSD faster, and the host is already only 6%. The honest conclusion: **the ANE is not a second miracle. It is, at best, an energy optimization (≈2× tok/s/W, 19–20 °C cooler, ~13 h projected battery vs ~7 h) — valuable for a different objective, but it cannot move Qwen to 5+ or Nemotron to 15+ tok/s under bit-exact constraints on a 16 GB M4 Air.** [MacPaw Research](https://research.macpaw.com/projects/neural-engine); [arXiv 2611.19101](https://arxiv.org/html/2611.19101v1); [apple/coremltools#2510](https://github.com/apple/coremltools/issues/2510); [kai-ane](https://github.com/crafter-agents/kai-ane); [NPUMoE](https://arxiv.org/abs/2604.18788); [Orion](https://arxiv.org/abs/2603.06728); [mlx/discussions/4551](https://github.com/ml-explore/mlx/discussions/4551).
