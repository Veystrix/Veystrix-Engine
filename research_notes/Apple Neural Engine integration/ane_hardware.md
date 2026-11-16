# Apple Neural Engine (ANE) hardware architecture, capabilities, and hard limits

**Scope / decision context:** Could the ANE accelerate a 4-bit Mixture-of-Experts (MoE) LLM on a 16 GB M4 MacBook Air? Report current as of **17 Nov 2026**. Claims are labeled **Apple claim**, **independent measurement**, or **speculation**. The single most important new primary source is a June 2026 reverse-engineering guide/paper (Bryngelson, Georgia Tech) that measured real silicon on M1 and M5 and decompiled the private stack; see [guide](https://ane-guide.readthedocs.io) and [arXiv:2606.22283](https://arxiv.org/abs/2606.22283). A second independent source (maderix, Feb 2026) directly clocked the M4 ANE below Core ML ([Part 1](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine), [Part 2](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)).

---

## Key Question 1: Architecture — cores, MAC/systolic structure, SRAM, memory interface, unified memory; how it differs from CPU/GPU

### Takeaway
The ANE is a fixed-function **graph-execution engine**, not a general-purpose programmable core: it accepts a compiled whole-graph program and runs it end-to-end, with convolution/matrix-multiply as its dominant primitives. Apple has never published its ISA or microarchitecture; the best public description comes from 2026 reverse engineering (measured + decompiled + patent analysis), which puts on-chip SRAM at roughly 32 MB and reports a cache-like hierarchy. It shares unified memory with CPU/GPU but has its own independent power/clock domain.

### Cited Findings
- Apple describes the ANE as "an IP block in the chip dedicated to the acceleration of AI workloads," working alongside "next-generation ML accelerators in the CPU" and the GPU over higher-bandwidth unified memory — [Apple Newsroom, M4, May 2024](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip).
- A 16-core ANE has shipped in M1, M2, M3, and M4 (base parts); it is a specialized tensor accelerator operating **independently** of CPU/GPU (not a co-processor like AMX), optimized for matrix multiplication, supporting INT8 and FP16 — [Hübner et al., "Apple vs. Oranges," arXiv:2502.05317, Feb 2025](https://arxiv.org/html/2502.05317v1).
- The ANE is "a graph execution engine — a fixed-function accelerator that takes a compiled neural network graph and executes the entire thing as one atomic operation. You don't issue individual multiply-accumulate instructions"; M4 ANE codename **H16G**, 16 cores, queue depth **127** evaluation requests, independent DVFS, and hard power gating to **0 mW idle** — [maderix Part 1, Feb 2026](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine) (independent reverse engineering; queue depth and DVFS read from IOKit `IOReportLegend`, not Apple-confirmed).
- The ANE has **no public instruction set, no driver interface, and no documented way to confirm a computation ran on it**; applications reach it only indirectly through Core ML as a placement hint — [Bryngelson 2026 guide, Introduction](https://ane-guide.readthedocs.io).
- Programs are described in **MIL** (Model Intermediate Language), compiled to a small (~2,688 bytes for 1024×1024 matmul) parameterized **E5 binary** whose behavior is controlled by tensor-shape descriptors at runtime — strong evidence of a small set of fixed compute primitives parameterized by descriptors, not traditional machine code — [maderix Part 1, Feb 2026](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine).
- Native tensor layout is channels-first **4D**, with the last axis unpacked and required to be contiguous and 64-byte aligned; reshapes/transposes likely trigger memory copies — [Apple ML Research, "Deploying Transformers on the ANE," June 2022](https://machinelearning.apple.com/research/neural-engine-transformers). The reverse-engineered format is NCDHW + interleave `[Batch, Channels, Depth, Height, Width]` — [maderix Part 1](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine).
- All ANE I/O uses **IOSurfaces** (same mechanism as GPU textures), enabling (in principle) zero-copy GPU↔ANE pipelines — [maderix Part 1, Feb 2026](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine).
- **On-chip SRAM ≈ 32 MB (independent measurement/inference):** a single 2048×2048 FP16 matmul working set of ~24 MB runs at peak single-op throughput (5.7 TFLOPS), while 4096×4096 (~96 MB) spills to DRAM and drops ~30% to 4.0 TFLOPS; the gradual (not hard-wall) degradation suggests a cache-like hierarchy — [maderix Part 2, Feb 2026](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615).
- Roofline characterization (measured on M1 Max): ~12 FP16 TFLOP/s, ridge point near **141 FLOP/byte**, **2 MB working-set threshold**, **0.23 ms floor per dispatch**, ~0.37 pJ/FLOP at peak compute; on a 256-channel 3×3 conv it runs ~3.8× faster than the same chip's GPU and ~9× more energy-efficient — [Bryngelson 2026 guide, Introduction](https://ane-guide.readthedocs.io).
- CPU vs ANE: M4's CPU cores carry Apple's SME (Scalable Matrix Extension) matrix unit; the ANE remains a separate fixed-function engine. For LLM inference the measured recommendation is **hybrid: prefill on ANE, single-token decode on SME/CPU (or GPU)** — [maderix Part 2, Feb 2026](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615). GPU vs ANE differs fundamentally: the GPU is programmable (Metal/Metal 4/MPS) and good at bandwidth-bound, irregular work, whereas the ANE is graph-fixed and excels at dense convolutions — [hollance/neural-engine "ANE vs GPU" doc](https://github.com/hollance/neural-engine/blob/master/docs/ane-vs-gpu.md); [Apple ML Research 2022](https://machinelearning.apple.com/research/neural-engine-transformers).

### Inferences
- For an MoE LLM, the ANE's fixed-function, conv-centric datapath is well suited to the **dense, repeated matmuls of expert FFNs** when batched, but MoE's dynamic expert routing/gather and bandwidth-heavy decode run counter to the design.
- The ~32 MB SRAM and ~2 MB "working-set threshold" mean only small tensors/tiles stay close to compute; expert weights and especially the KV cache will live in DRAM.
- The internal MAC/systolic-array geometry and exact SRAM bank topology are explicitly listed as **unknown** even by the reverse engineers — treat any public "systolic array size" claim as unverified.

### Gaps
- No authoritative public description of the MAC array geometry, clock frequency (DVFS makes it dynamic), core-to-op assignment, or whether SRAM is banked/unified/per-core ([maderix Part 1](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine) lists these as open questions). Maynard Handley's ~122-page patent-based hardware reconstruction exists (AArch64-Explore Vol. 7) but was not read in depth here — [referenced in maderix comments/README](https://github.com/name99-org/AArch64-Explore).
- I found no reliable public number for M4 ANE L2 cache size (Apple's 2022 note mentions "L2 cache residency" qualitatively but gives no capacity) — [Apple ML Research 2022](https://machinelearning.apple.com/research/neural-engine-transformers).

---

## Key Question 2: Published TOPS / throughput by chip and precision; is INT4 / INT8 native?

### Takeaway
Apple's headline number for M4 is **38 TOPS** (16-core ANE). Independent measurement shows the *compute* rate is ≈19 FP16 TFLOP/s, and that "38 TOPS" is 19 TFLOPS × 2 by the industry convention of counting INT8 as double FP16 — the hardware does **not** actually run INT8 twice as fast. **INT4 is not a native compute precision**: Core ML supports 4-bit *weights* (palettization/quantization), but the ANE dequantizes low-precision weights (INT8 → FP16) before computing; measured INT4 lookup-table weights still run ~2.37× faster than FP16 in one 2026 study, by saving bandwidth.

### Cited Findings
- **Apple claims (peak TOPS):** A11 (2017) 0.6 TFLOPS FP16; A15 (5th-gen 16-core) 15.8 TFLOPS FP16 — [Apple ML Research, June 2022](https://machinelearning.apple.com/research/neural-engine-transformers). M4 16-core ANE "up to 38 trillion operations per second," "60× faster than the first Neural Engine in A11 Bionic" — [Apple Newsroom, M4, May 2024](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip).
- **Per-chip TOPS table (community-maintained, largely echoing Apple):** M1 ≈11 TOPS; M2 / M2 Pro / M2 Max ≈15.8 TOPS (A15-derived 16-core); M3 / M3 Pro / M3 Max ≈15.8 TOPS; **M3 Ultra 32-core ≈36 TOPS**; **M4 / M4 Pro / M4 Max 16-core = 38 TOPS** — [hollance/neural-engine supported-devices](https://github.com/hollance/neural-engine/blob/master/docs/supported-devices.md#m4-pro) and [Kunal Ganglani blog, 2024](https://www.kunalganglani.com/blog/apple-m4-chip-what-it-means-for-developers). These are nominal/marketing-derived, not independently clocked.
- **Memory bandwidth by generation (context for the roofline):** M1 67 GB/s, M2 100 GB/s, M3 100 GB/s, **M4 120 GB/s** (base M4), max unified memory on base M4 16–32 GB — [Hübner et al., arXiv:2502.05317, Feb 2025](https://arxiv.org/html/2502.05317v1); M4 LPDDR5X-7500 — [Wikipedia, Apple M4](https://en.wikipedia.org/wiki/Apple_M4).
- **M4 independent measurement (direct `_ANEClient`, M4 Mac Mini, macOS 15.x):** true single-precision ceiling ≈5.7 FP16 TFLOPS for one matmul; with deep conv graphs (~32+ layers) peak ≈**18.6 FP16 TFLOPS**, ≈94% of the theoretical ~19 TFLOPS (16 cores × ~1.2 TFLOPS/core), at ~2.8 W; **INT8 W8A8 ≈35.1 TOPS** in the project README table but "INT8 and FP16 deliver nearly identical throughput" in the write-up — the ANE **dequantizes INT8 weights to FP16 before compute**, so INT8 saves bandwidth, not cycles — [maderix Part 2](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615) and [maderix/ANE README](https://github.com/maderix/ANE).
- The same source states Apple's "38 TOPS INT8" = 19 TFLOPS FP16 × 2, "following the industry convention," and calls the 2× INT8 claim "misleading" because the hardware does not execute INT8 twice as fast — [maderix Part 2, Feb 2026](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615).
- **INT4:** Core ML tools officially support **4-bit and 8-bit weights** and **8-bit activations**, with per-tensor/per-channel/per-block granularity — [Apple Core ML Tools quantization docs](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html). On A17 Pro/M4, quantizing both activations and weights to int8 "can leverage optimized compute on the Neural Engine" (Apple claim) — [same Apple doc](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html).
- On the **direct/unentitled** ANE path, int4 lookup-table weights run ~**2.37× faster than fp16**, and structured sparsity 1.55–1.64× faster at 0.43× the bytes — measured/decompile-derived, primarily M1 and M5, not M4 — [Bryngelson 2026 guide, Introduction](https://ane-guide.readthedocs.io).
- **BF16:** I found no evidence that the ANE supports bfloat16 as a native compute precision; public Apple docs and the 2026 characterization center on FP16, INT8, and INT4/compressed weights — [Apple Core ML quant docs](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html); [Bryngelson 2026](https://ane-guide.readthedocs.io).

### Inferences
- For a 4-bit MoE model, ANE "4-bit support" is a **weight-storage/compression feature, not a 4-bit MAC**: compute is effectively FP16-class. Expect throughput governed by FP16 MAC rate (~19 TFLOPS-class on M4) and by DRAM bandwidth when weights/activations spill.
- The 16 GB M4 MacBook Air is the **120 GB/s base M4**, the slowest M4 bandwidth tier; this is the binding constraint for token-by-token decode and for fetching active experts, not the ANE's TOPS.
- Reported TOPS figures conflict across sources (11/15.8/18.6/19/35.1/38) because they mix FP16 vs INT8 conventions, Apple claims vs measured, and single-op vs deep-graph workloads — the report writer should preserve this distinction rather than quote a single number.

### Gaps
- No reliable independent ANE TOPS measurement specifically on the **M4 MacBook Air** (fanless; DVFS/thermal throttling may lower sustained throughput vs the M4 Mac Mini used by maderix) — [maderix Part 2](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615).
- M4 Ultra (~76 TOPS, 32-core) appears only in a low-quality wiki and is **speculation**; no Apple confirmation found — [clawbot wiki](https://clawbot.ai/wiki/devops/apple-m4-ultra-neural-engine.html).
- No M5 ANE TOPS figure was published in the sources reviewed (Apple only says "faster 16-core Neural Engine"); Bryngelson measured M5 silicon but the specific ANE TOPS was not captured here — [Apple M5 Newsroom, Oct 2025](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon); [Bryngelson 2026](https://arxiv.org/abs/2606.22283).
- An "int8/4-bit MoE inference on the Apple NPU" paper (Benazir 2026) is cited as related work but was not read; could contain directly relevant MoE measurements — [Bryngelson 2026 references](https://ane-guide.readthedocs.io/en/latest/references.html).

---

## Key Question 3: Which operations/kernels the ANE actually accelerates; what forces fallback

### Takeaway
The ANE's native strengths are **convolution (including 1×1 conv used as matmul), dense matrix multiply, and elementwise ops**; transformers can run on it with a specific 4D channels-first layout and 1×1-conv substitution for linear layers. Ops that are unsupported — gather, RNN/LSTM/GRU, custom layers, broadcastable/"ND" layers, dilated conv, certain pooling/upsampling — force fallback to GPU or CPU. A 2026 study found advertised capability bits can overstate reality: several "supported" ops (e.g., 3D convolution) never actually lower to the engine.

### Cited Findings
- Apple's ANE-optimized transformer reference swaps every `nn.Linear` for `nn.Conv2d` (1×1 conv), uses a single `bchq,bkhc->bkhq` einsum for batched scaled-dot-product attention to avoid intermediate transpose/reshape, and avoids reshapes because ANE buffers have an unpacked axis — [Apple ML Research, June 2022](https://machinelearning.apple.com/research/neural-engine-transformers); [apple/ml-ane-transformers](https://github.com/apple/ml-ane-transformers).
- Expressing matmul as a **1×1 convolution** is ~3× faster than the matmul op on the ANE: "the ANE is fundamentally a convolution engine" — [maderix Part 2, Feb 2026](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615). Conv is the ANE's "primary compute primitive" — [maderix Part 1](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine).
- Known **unsupported** Core ML layer types that force CPU/GPU fallback: custom layers, RNN layers (LSTM/GRU), **gather**, dilated convolutions, broadcastable and "ND" layers, certain broadcasting (e.g. `CxHxW × Cx1x1`), pooling with kernel >13 or stride >2, upsampling factor >2 — [hollance/neural-engine, "Which Core ML layers are not supported"](https://github.com/hollance/neural-engine/blob/master/docs/unsupported-layers.md). Community-sourced and self-described as possibly incomplete/wrong.
- Device-switch behavior: Core ML usually collapses S→U→S→U patterns into one ANE→GPU switch; ANE→CPU→ANE switching does occur and is cheaper than ANE→GPU — [hollance/neural-engine](https://github.com/hollance/neural-engine/blob/master/docs/unsupported-layers.md). Every switch adds overhead.
- "Several advertised operations never lower to the engine at all, three-dimensional convolution among them"; the authors count an op as supported only after a compile-and-run confirms it, not from feature bits or compiler acceptance — [Bryngelson 2026 guide, Introduction](https://ane-guide.readthedocs.io).
- The project reports an **operation-by-device matrix** and per-chip "operation floors" decoded from 28 compiler targets (A11–A18, M1–M5) — [Bryngelson 2026 guide](https://ane-guide.readthedocs.io).
- Reported usable ops in the direct-API M4 work include matmul/conv and fused transformer kernels (SDPA/QKV projection, SwiGLU FFN, RMSNorm and their backward passes) — [maderix/ANE README](https://github.com/maderix/ANE).
- FP16 is the ANE's native compute precision; FP32 I/O is used externally — [Apple ML Research 2022](https://machinelearning.apple.com/research/neural-engine-transformers); [maderix Part 2](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615).

### Inferences
- MoE-specific risk: dynamic expert selection and token routing typically require **gather / scatter / dynamic indexing**, all in the unsupported category → those portions will fall back to CPU/GPU, likely negating a full-ANE MoE implementation.
- A static, densely-computed MoE (where all/fixed experts are evaluated per token) maps better to the ANE; a top-k routed MoE does not.
- Attention/softmax and layernorm need to be expressed in ANE-friendly primitives and fused; Apple's guidance shows it is possible for BERT-style encoders but requires deliberate graph restructuring.

### Gaps
- The full operation-by-device support matrix is in the Bryngelson guide's appendices but was not fetched in detail here — [guide appendices](https://ane-guide.readthedocs.io).
- No published measurement of an MoE routing/gather kernel running on the ANE was found beyond the cited Benazir 2026 related work — [Bryngelson references](https://ane-guide.readthedocs.io/en/latest/references.html).

---

## Key Question 4: Hard constraints — fixed shapes, batch size, minimum tensor sizes, weight layout/palettization, on-chip memory, no dynamic control flow, no sparse/unstructured gather

### Takeaway
The ANE compiles a **static graph with explicit shapes** into a parameterized program; there is no dynamic control flow or unstructured gather. Performance is dominated by dispatch overhead (~0.095 ms) for small ops, a ~32 MB on-chip SRAM limit, and 64-byte alignment of the last tensor axis. Weight compression (palettization/INT4/INT8/sparsity) is supported as a *storage/bandwidth* optimization with compute still in FP16; some linear blockwise-INT4 configs fail to compile on the ANE while palettized ones succeed.

### Cited Findings
- ANE buffers are 4D/channels-first with the last axis unpacked, contiguous, and **64-byte aligned**; using a singleton last axis can inflate buffers by 32× (16-bit) or 64× (8-bit), reducing L2 residency and hitting DRAM — [Apple ML Research, June 2022](https://machinelearning.apple.com/research/neural-engine-transformers).
- The ANE is **bandwidth-bound for short sequences at batch=1** (latency stays flat from seq len 32→128 despite 4× compute); mitigations are larger batch or quantization/pruning — [Apple ML Research 2022](https://machinelearning.apple.com/research/neural-engine-transformers).
- Dispatch/XPC+IOKit overhead ≈0.095 ms; any operation under ~1 ms is dispatch-limited (256×256 matmul: 0.101 ms total, ~0.006 ms compute); a 0.23 ms dispatch floor is independently reported — [maderix Part 2](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615); [Bryngelson 2026](https://ane-guide.readthedocs.io).
- On-chip SRAM ≈32 MB; per-tensor footprint must stay under it to avoid DRAM spills; single ops use only ~30% of peak capacity, so chain 16–64 ops per program to approach peak — [maderix Part 2](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615).
- **Static graph / no dynamic control flow:** the program is a fixed compiled graph (E5), and the ANE "executes it end-to-end as one atomic operation" — [maderix Part 1](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine). Model shapes are fixed at conversion/compile time (Core ML static shapes).
- **No unstructured gather / sparse scatter:** `gather` is on the unsupported layer list; the only measured sparsity acceleration is *structured* (1.55–1.64× faster at 0.43× bytes) — [hollance/neural-engine](https://github.com/hollance/neural-engine/blob/master/docs/unsupported-layers.md); [Bryngelson 2026](https://ane-guide.readthedocs.io).
- **Palettization / INT4:** Core ML supports 4-bit weights (palettization is the primary 4-bit path) and 8-bit activations; on A17 Pro/M4, int8 W8A8 is an ANE-optimized configuration — [Apple Core ML Tools quantization docs](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html); [Apple palettization docs](https://apple.github.io/coremltools/docs-guides/source/opt-palettization.html).
- A real toolchain failure mode: compiling a static-shape LLM with **linear blockwise-INT4** weights to the ANE crashes `coreai-build` with SIGSEGV in `MPSGraph anePreCompileBinary`, while **palettized** weights compile fine — [apple/coreai-models issue #55](https://github.com/apple/coreai-models/issues/55). Treat as a bug report, not a formal spec.
- Compile cache matters: first compile ~20–40 ms; E5 binaries cached on disk under `com.apple.e5rt.e5bundlecache/<build>/H16G.bundle/`; the private path also needs a writable temp dir — [maderix Part 1](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine).

### Inferences
- Full-ANE execution of a token-by-token generative MoE is unlikely: single-token decode is below the dispatch floor, KV cache and routed-expert weights exceed SRAM, and routing/gather is unsupported. Batched prefill of a *static* expert set is the realistic accelerator target.
- Weight palettization/INT4 still helps a lot, but mainly by reducing DRAM traffic on a bandwidth-limited 16 GB/120 GB/s M4 Air — not by enabling 4-bit MACs.

### Gaps
- Exact minimum tensor sizes / channel-count constraints for each op were not enumerated in the sources read.
- Behavior under thermal throttling on the fanless M4 MacBook Air is unmeasured.

---

## Key Question 5: Academic and reverse-engineering work with measured numbers

### Takeaway
ANE characterization is a young field that exploded in 2026. The most rigorous source (Bryngelson, arXiv:2606.22283 / ane-guide) measures M1 and M5 silicon, decompiles the private runtime/compiler/driver/firmware, and marks every claim measured/decompile-derived/predicted. Independent M4 benchmarks (maderix) and community catalogs (hollance) corroborate the broad picture but differ on exact TOPS.

### Cited Findings
- **Bryngelson 2026** — "Apple Neural Engine: Architecture, Programming, and Performance," Georgia Tech, v1 Jun 21 2026, v3 Oct 14 2026: reaches the ANE from an ordinary user process below Core ML; 28 compiler targets decoded; 93-command firmware protocol; roofline ~12 FP16 TFLOP/s (M1 Max), 141 FLOP/byte ridge, 2 MB working-set threshold, 0.23 ms dispatch floor, 0.37 pJ/FLOP; 3.8× faster than GPU and 9× more energy-efficient on 256-channel 3×3 conv; INT4 LUT 2.37× faster than FP16; structured sparsity 1.55–1.64× at 0.43× bytes — [arXiv abstract](https://arxiv.org/abs/2606.22283); [full guide](https://ane-guide.readthedocs.io).
- The same guide stresses the direct path is **callable but unsupported, version-fragile, not App-Store-safe**, and "Core ML remains the path for shipped software" — [guide](https://ane-guide.readthedocs.io); [sbryngelson/ane-guide README](https://github.com/sbryngelson/ane-guide).
- **maderix/ANE (Feb–Mar 2026):** from-scratch transformer *training* on ANE via reverse-engineered `_ANEClient`/`_ANECompiler` private APIs; M4 training throughput Stories110M 91 ms/step, Qwen3-0.6B 412 ms/step; peak M4 FP16 ~18.6 TOPS, INT8 W8A8 ~35.1; AGENTS note ~32 MB SRAM, 3× conv-vs-matmul, deep graphs, zero idle power — [maderix/ANE GitHub](https://github.com/maderix/ANE); [Part 1](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine); [Part 2](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615); [independent summary, Menon Lab blog](https://themenonlab.blog/blog/apple-neural-engine-reverse-engineered-training).
- **hollance/neural-engine** — long-running community catalog ("Everything we actually know about the ANE"), 2.6k GitHub stars; explicitly experimental, not Apple-insider — [repo](https://github.com/hollance/neural-engine).
- Other named projects: tinygrad's HWX format + `AppleH11ANEInterface` IOKit path; eiln/ane reverse-engineered Linux driver + `anecc` compiler; libane native runtime; whisper.cpp Core-ML-routed access; Maynard Handley's patent-based hardware reconstruction (AArch64-Explore Vol. 7) — [Bryngelson related work](https://ane-guide.readthedocs.io); [maderix Part 1 comments](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine). Community LLM-on-ANE efforts exist (e.g. ane.cpp / Qwen3-4B int8 on M3 Max at ~11.66 tok/s generate, 30.08 tok/s prompt) but are hobbyist, private-API dependent — [apple-neural-engine-field-guide](https://github.com/skyfallsin/apple-neural-engine-field-guide).

### Inferences
- For a production decision, only Apple-documented Core ML behavior is safe; private-API benchmarks demonstrate *hardware capability ceilings* but cannot be relied on in shipped software.
- Numbers from different groups must be compared carefully: FP16 TFLOP/s vs INT8 TOPS, single-op vs deep-graph, M1/M5 vs M4.

### Gaps
- Did not read the full arXiv PDF or appendices; several potentially relevant studies (Benazir2026 on MoE-on-NPU, Hubner2025 HPC, Orion2026 concurrent LLM training work) are cited but not summarized here — [Bryngelson references](https://ane-guide.readthedocs.io/en/latest/references.html).
- Conflicting 15.8 vs 18.6 TFLOPS FP16 figures within maderix's own materials were not reconciled.

---

## Key Question 6: What changed in M4 (2024) and M5 (2025); later M5 Pro/M6; "Neural Accelerators" in GPUs; LLM relevance

### Takeaway
M4 (2024) roughly doubled ANE throughput (marketing 38 TOPS vs ~15.8 on M3) and moved to LPDDR5X/120 GB/s on base parts, with next-gen CPU ML accelerators (SME). **M5 (Oct 2025)** is the bigger LLM-relevant shift: Apple moved AI matrix compute into the **GPU**, adding a "Neural Accelerator" to every GPU core (>4× peak GPU AI compute vs M4) with direct programming via Metal 4 Tensor APIs, plus a faster 16-core ANE and ~30% higher unified memory bandwidth (153 GB/s). No reliable M5 Pro/M6 ANE details were found by Nov 2026.

### Cited Findings
- **M4 (May 2024):** 16-core ANE, up to 38 TOPS, 60× vs A11; "next-generation ML accelerators in the CPU"; higher-bandwidth unified memory (base M4 120 GB/s, LPDDR5X-7500) — [Apple Newsroom M4](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip); [Wikipedia M4](https://en.wikipedia.org/wiki/Apple_M4). M4 Pro/Max (Nov 2024) use the same 16-core, 38-TOPS ANE — [hollance supported-devices](https://github.com/hollance/neural-engine/blob/master/docs/supported-devices.md#m4-pro).
- **M5 (announced Oct 15, 2025):** built on 3rd-gen 3 nm; next-gen 10-core GPU with **a Neural Accelerator in each core**, >4× peak GPU AI compute vs M4 and >6× vs M1; 16-core CPU with SME; "improved 16-core Neural Engine"; unified memory bandwidth **153 GB/s** (+~30% vs M4, >2× M1), up to 32 GB; developers can program the GPU Neural Accelerators via **Metal 4 Tensor APIs**; Core ML/MPS/Metal 4 apps see automatic gains — [Apple Newsroom, M5, Oct 2025](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon). Apple cites local LLM apps (LM Studio) and diffusion (Draw Things) as beneficiaries — [same source](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon).
- Apple explicitly frames M5 AI gains as coming primarily from **GPU Neural Accelerators + faster ANE + more bandwidth**, and speeds Apple Intelligence / Foundation Models on the ANE — [Apple Newsroom M5](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon).
- Bryngelson 2026 measured M5 silicon directly and decoded M5 among 28 compiler targets, finding cross-generation predictions held — [arXiv:2606.22283](https://arxiv.org/abs/2606.22283).
- M5 also ships in 14-inch MacBook Pro, iPad Pro, Vision Pro (not MacBook Air, as of the announcement) — [Apple Newsroom M5](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon).

### Inferences
- The strategic direction is **heterogeneous AI compute**: for LLMs, Apple's own messaging points to GPU Neural Accelerators (Metal 4) plus ANE for Apple Intelligence, not ANE alone. A 16 GB M4 Air predates the GPU Neural Accelerators, so its best AI matrix paths are CPU SME, GPU, and ANE via Core ML.
- Higher bandwidth (M5 153 GB/s vs M4 Air 120 GB/s) directly benefits decode and expert-weight fetching, often more impactful than ANE TOPS for MoE.

### Gaps
- No reliable information found on **M5 Pro / M5 Max / M6** ANE core counts, TOPS, or memory bandwidth by Nov 2026 — mark any such claims as speculation.
- No independent M5 ANE TOPS or MoE benchmark numbers were captured in the sources reviewed.
