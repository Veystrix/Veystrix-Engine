# Hybrid CPU/GPU/NPU (ANE) Execution on Apple Silicon — Feasibility for SSD-Bound MoE LLM Decode

**Research date: 2026-11-17. Target workload: fanless 16 GB M4 MacBook Air, autoregressive MoE decode ~385 ms/token = ~289 ms SSD expert-read wait + ~70 ms GPU + ~51 ms "router sync".**

> Framing note for the report writer: the single most important finding is that **the dominant bottleneck (289/385 ms = 75%) is SSD I/O**, and the ANE has no data path that can accelerate SSD reads. It shares the same unified memory and SSD subsystem as the GPU. Every measured ANE advantage (compute efficiency, low power, op-specific speedups) is irrelevant to an I/O-bound token. Amdahl's-law ceiling: even *freeing* all GPU+router work entirely yields at most 385/289 = **1.33×**; realistically ANE cannot touch the bottleneck and adds dispatch overhead, so it is likely **neutral-to-negative for this specific workload**. The value of ANE is in auxiliary fixed models (router/classifier/drafter) that run alongside generation — which is a different proposition from accelerating the SSD-bound token itself.

---

## Key Question 1: Published work on hybrid CPU/GPU/NPU scheduling — what partitioned worked, what didn't, measured numbers

### Takeaway
Heterogeneous scheduling *can* win on unified-memory SoCs, but only when (a) work is genuinely compute-bound, (b) the runtime jointly optimizes overlap and device allocation, and (c) both accelerators expose comparable software stacks. All strong positive results are on non-Apple SoCs (AMD Ryzen AI, Qualcomm) or use speculative decoding to make ANE weight reads amortizable — none address an SSD-I/O-bound MoE decode, and none are on a fanless M4 MBA.

### Cited Findings
- **HeteroMosaic (MICRO 2026, arXiv:2607.12839v4, 25 Jul 2026)**, on three AMD Ryzen AI unified-memory SoCs: up to **1.73× vs iGPU baseline, 1.78× vs NPU baseline, 2.05× vs llama.cpp, 45.3% less energy**; average 1.25× over prior SOTA. Key mechanism: dependency-preserving *causal micro-batching* to expose cross-accelerator overlap + trace-guided critical-interval co-optimization accounting for unified-memory contention, DVFS, NPU runtime behavior. — [HeteroMosaic](https://arxiv.org/html/2607.12839v4)
- Same paper: a **heterogeneous roofline model** shows heterogeneity helps compute-bound ops but "red point" ops near the bandwidth-bound regime **cannot benefit from additional heterogeneous compute**. It explicitly models a shared-memory bandwidth cap: when iGPU and NPU contend, effective bandwidth degrades and the heterogeneous bound shifts downward. Non-GEMM ops reach **up to 35% of runtime at 16K prompt** in Llama3-8B, limiting GEMM-only partitioning. — [HeteroMosaic](https://arxiv.org/html/2607.12839v4)
- **HeteroInfer (Chen et al. 2025)** on Snapdragon 8 Gen 3: NPU-primary/GPU-secondary partitioning, but the platform is ~**1:10 GPU:NPU throughput skewed**, so gains are confounded with simply assigning work to the stronger accelerator; no NPU-only baseline for the same stack. — [HeteroMosaic §3.2.2](https://arxiv.org/html/2607.12839v4)
- **Xu et al., ASPLOS 2025, "Fast On-device LLM Inference with NPUs"**: maximize *prefill* on mobile NPU (integer compute), keep precision-sensitive "shadow-outlier" path on CPU/GPU. Couples heterogeneity with model restructuring + changed quantization. — [ASPLOS25 PDF](https://xumengwei.github.io/files/ASPLOS25-NPU.pdf); [HeteroMosaic §3.1.2](https://arxiv.org/html/2607.12839v4)
- **PowerInfer-2 (Xue et al. 2024)**: heterogeneous NPU/CPU + activation sparsity + **storage-aware/flash-I/O pipelining** to run models exceeding DRAM. Notable because it is the one work that actually coordinates NPU compute with SSD I/O — but via neuron sparsity, not plain ANE. — [HeteroMosaic §3.1.1](https://arxiv.org/html/2607.12839v4)
- **Apple "Deploying Transformers on the Apple Neural Engine" (Apple ML Research)**: Apple's own guidance frames ANE as reducing memory footprint, app-responsiveness impact, and battery use for fixed transformer models. — [Apple ML Research](https://machinelearning.apple.com/research/neural-engine-transformers)
- **maderix (Feb 2026) recommendation**: "ideal LLM inference strategy on M4 is hybrid: prefill (large batch) on ANE, decode (single token) on SME [CPU]." A commenter immediately objects that the **KV cache cannot fit in ANE SRAM for any decent LLM**, so decode should stay on CPU/GPU; author agrees. — [maderix ANE benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)

### Inferences
- The literature's "hybrid wins" all require compute-bound work + a runtime that can schedule across accelerators with low-overhead sync. An SSD-I/O-bound decode is the opposite regime: the "slowest participating accelerator / shared-memory cap" term in the roofline is the SSD, which no accelerator scheduling changes.
- No published system demonstrates ANE accelerating an MoE expert-read/SSD path. The closest (PowerInfer-2) uses a different NPU and relies on activation sparsity + I/O pipelining, not ANE.
- On Apple hardware specifically, the pragmatic published recommendation is ANE for prefill/auxiliary fixed models, GPU/CPU for decode — which does not target this bottleneck.

### Gaps
- No published heterogeneous-scheduling study on an Apple Silicon Mac with SSD-streamed MoE experts.
- No measured data for a 16 GB fanless M4 MBA (all data are M2/M3/M4 Pro/Max/Mini or non-Apple SoCs).

---

## Key Question 2: ANE dispatch overhead and graph compilation — can small ops (router projection, 1-token GEMM) beat GPU?

### Takeaway
**No.** For tiny ops, ANE is dominated by fixed dispatch overhead (~0.095 ms XPC+IOKit; ~2.3 ms IOSurface round-trip for a full forward path) and loses to GPU/CPU. ANE needs deep graphs (16–64 ops), batching, or speculative multi-token verification to amortize weight reads. A single-token router GEMM is close to the ANE's worst case.

### Cited Findings
- **Dispatch overhead**: ~**0.095 ms** XPC+IOKit per dispatch (maderix, confirmed by Orion). A 256×256 matmul takes 0.101 ms of which compute is only ~0.006 ms — dispatch-limited. "Anything under ~1 ms is dominated by the 0.095 ms dispatch overhead." — [maderix](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615); [Orion](https://arxiv.org/html/2603.06728v1)
- **Full-path round-trip**: Orion reports ANE decode at ~**5.76 ms/tok (170 tok/s)** vs **CPU 3.48 ms/tok (283 tok/s)** for GPT-2 124M — CPU is *faster* because of ANE's **~2.3 ms IOSurface round-trip per dispatch** across 12 layers. Single-token dispatch floor ~0.03 ms. — [Orion §8.1, 8.3](https://arxiv.org/html/2603.06728v1)
- **Minimum viable batch/graph**: deep graphs of **16–64 ops** reach **94% ANE utilization** vs **~30% for single ops**; 1×1 conv is **3× faster than matmul**; working sets must stay under **~32 MB SRAM** or throughput drops ~30%. Small models (<1 MB, <1 ms inference) "lose to dispatch overhead" and CoreML adds **2–4× overhead** for small ops. — [maderix](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615); [Orion §2.1](https://arxiv.org/html/2603.06728v1); [ONNX/CoreML analysis](https://www.xybrid.ai/blog/onnx-runtime-6x-faster-apple-silicon-coreml)
- **Compilation/weight-baking constraints (critical for MoE)**: ANE **bakes weights at compile time**; naive weight update costs **~4.2 s full recompile** for 60 kernels; **~119 compiles per process** limit; **~49 KB minimum IOSurface** (single-token [1,768,1,1] must be padded to [1,768,1,16]); **32K-channel conv rejected** (forces CPU fallback for vocab projection); first-call compile ~**1015 ms for 24 programs**. Orion's delta-reload reduces recompile 4,200→494 ms (8.5×), but still ~494 ms/step of disk I/O. — [Orion §2–5](https://arxiv.org/html/2603.06728v1)
- **When ANE does win per-op (large, batched)**: classifier forward 10.77→1.06 ms (10.2×), softmax(vocab 32k) 81.11→2.40 ms (33.8×) — but these are large ops, not tiny router GEMMs. — [Orion §8.4](https://arxiv.org/html/2603.06728v1)
- **MacPaw's ANE router win is a 350M *encoder* with 128 tokens**, not a 1-token projection: ANE INT8 5.5 ms vs MLX bf16 15.8 ms (2.9× faster), 126 vs 797 mJ (6.3× less energy). A one-token router projection would be dominated by the ~2.3 ms round-trip. — [MacPaw Research](https://research.macpaw.com/projects/neural-engine)
- **ANE's own efficient-decode recipe is the opposite of MoE**: MacPaw "efficient ANE decoding depends on weight reuse" — one ANE call verifies a draft tree of up to 64 tokens, giving 62× lower per-token latency vs single-token calls (0.038 ms/tok vs 2.42 ms/call for FFN blocks). MoE expert switching breaks weight reuse. — [MacPaw Research](https://research.macpaw.com/projects/neural-engine)

### Inferences
- A 1-token router GEMM on ANE would incur ~0.095 ms dispatch + ~2.3 ms IOSurface round-trip ≈ ≥2.4 ms, versus a GPU kernel of tens of microseconds; ANE loses badly. The ~51 ms "router sync" is almost certainly a stall/wait, not 51 ms of compute that ANE could replace — moving it to ANE does not remove the stall.
- Compile-time weight baking + 32 MB SRAM + 119-compile limit make dynamically swapping 4-bit MoE experts from SSD fundamentally incompatible with ANE: experts could not be streamed in without recompilation.
- The only way ANE beats GPU here is if the "router" is a large, fixed, batched auxiliary model (MacPaw's 350M encoder case) — not a single-token projection.

### Gaps
- No direct measurement of a 1-token MoE router projection on ANE vs M4 GPU. Numbers above are inferred from dispatch/round-trip floors.
- No measurement of ANE Core AI (macOS 27) dispatch overhead specifically — MacPaw's numbers use Core AI and may differ from private-API/CoreML figures.

---

## Key Question 3: Thermal/power behavior — does ANE offload reduce heat/power when SSD is the bottleneck?

### Takeaway
ANE is dramatically more power-efficient than GPU in absolute terms (6.6 vs ~1.0 TFLOPS/W; ~17.6 W vs 37–51 W system power; 19–20 °C cooler), which matters on a fanless Mac. **But when SSD wait dominates (75% of token), total system power is already low during the stall, and GPU is active only ~70/385 ms (18% duty). Reducing heat further does not shorten an I/O-bound token.**

### Cited Findings
- **Peak efficiency**: M4 ANE **6.6 TFLOPS/W** at peak (19 TFLOPS FP16 at ~2.8 W) vs M4 GPU ~1.0, A100 ~0.08, H100 ~0.13; **zero idle power (hard power gating)**. — [maderix](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615); [Orion Table 1](https://arxiv.org/html/2603.06728v1)
- **System-level (MacPaw, M3 Pro MacBook Pro, 20 min, AC)**: ANE-only drew **17.6 W vs 37–51 W** for GPU configs; **19–20 °C lower median sensor temperature**; 9–37% lower fan speed; no throttling in any config; projected battery **13.0 h vs 7.1 h** GPU 8-bit. — [MacPaw Research](https://research.macpaw.com/projects/neural-engine)
- **Energy per request (MacPaw, M4 Pro)**: ANE-only **3.4× less energy than GPU 8-bit + DSpark, 4.1× less than GPU 8-bit**; system drew **21 W vs 48 W**, with −20%/−43% latency. ANE decode **158 mJ/token vs 409–575 mJ** for 8-bit GPU. — [MacPaw Research](https://research.macpaw.com/projects/neural-engine)
- **Independent HPC power data (arXiv:2502.05317, 2025)**: Apple M-series CPU/GPU power ranges a few to 10–20 W, >200 GFLOPS/W for GPU and accelerator reached by all four chips — confirms general M-series efficiency, not ANE-specific. — [Apple vs Oranges](https://arxiv.org/html/2502.05317v1)
- **NPU vs GPU power literature (non-Apple)**: NPUs 20–60% less power than GPUs at similar work, but for *small* models the difference is limited and can marginally favor GPUs on FPS/W. — [MDPI systems 13 797](https://www.mdpi.com/2079-8954/13/9/797); [KTH YOLOv5 thesis](https://kth.diva-portal.org/smash/get/diva2:1886212/FULLTEXT01.pdf)

### Inferences
- The thermal argument for ANE is real in aggregate but **does not translate to this workload**: an SSD-bound token spends 75% of its time in low-power I/O wait. Shaving the 70 ms GPU slice's heat is a second-order effect on token latency, and there is no throttling evidence on the M4 MBA at this duty cycle.
- ANE offload reduces *system power/heat*; it does not reduce *SSD wait*. Confusing these is the core risk in the feasibility argument.
- Caveat: MacPaw's power figures come from full-model ANE inference under continuous load; a fanless MBA in an SSD-bound regime is a different thermal envelope.

### Gaps
- No published sustained-load thermal/power measurement of an SSD-bound MoE decode on any Apple Silicon machine.
- No measurement of whether ANE routing to reduce GPU duty cycle measurably lowers M4 MBA skin temperature in an I/O-bound workload.

---

## Key Question 4: ANE memory bandwidth and unified-memory behavior — can it hold/stream 4-bit MoE experts?

### Takeaway
**ANE cannot hold or stream a useful set of 4-bit MoE experts.** It has only ~32 MB on-chip SRAM, attains only **44–70% of GPU effective memory bandwidth**, dequantizes INT8→FP16 (no INT4 compute benefit), bakes weights at compile time, and has a ~119-compile/process limit. Its one advantage is zero-copy IOSurface access to unified memory — but that memory is shared with the SSD-bound path.

### Cited Findings
- **Bandwidth**: "The ANE attains only **44–70 percent of the GPU's effective memory bandwidth**, whereas its compute throughput is **2.0–2.7× higher**." M4 Pro schematic gives measured weight-read bandwidth ~122.4 GB/s at fabric level. — [MacPaw Research](https://research.macpaw.com/projects/neural-engine)
- **SRAM cliff**: ~32 MB on-chip SRAM; 24 MB matmul working set fits (5.7 TFLOPS), 96 MB spills to DRAM (−30%, 4.0 TFLOPS); gradual cache-like degradation. — [maderix](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615); [Orion §2.1](https://arxiv.org/html/2603.06728v1)
- **INT4/INT8 reality**: Apple's "38 TOPS" is 19 TFLOPS FP16 × 2 by convention; ANE **dequantizes INT8 to FP16 before compute** — "no 2× INT8 speedup," INT8 saves only memory bandwidth. Orion lists quantization (INT8/INT4) as **not yet supported**. — [maderix](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615); [Orion §11 Limitations](https://arxiv.org/html/2603.06728v1)
- **Compile-time weights / stream incompatibility**: weights baked at compile time; ~119 compiles/process; ~4.2 s recompile for 60 kernels; delta-reload still ~494 ms of disk I/O for 60 kernels. — [Orion §2–5](https://arxiv.org/html/2603.06728v1)
- **Unified memory access**: ANE uses IOSurface-backed zero-copy shared memory in fixed [1, C, 1, S] fp16 layout; shares one unified memory pool with CPU/GPU. — [Orion §2.1, 4.2](https://arxiv.org/html/2603.06728v1)
- **MacPaw's working ANE LLM is a static INT8 2.6B model with weight reuse**, explicitly the opposite of MoE expert streaming. — [MacPaw Research](https://research.macpaw.com/projects/neural-engine)
- **Commentator consensus**: full KV cache for any decently sized model cannot stay in ANE SRAM; decode belongs on CPU/GPU. — [maderix comments](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)

### Inferences
- Because ANE is **bandwidth-limited relative to GPU** and the workload is **I/O-limited relative to both**, ANE offers no bandwidth advantage for streaming experts. Lower ANE bandwidth would if anything worsen an expert-stream path.
- Baked weights + 32 MB SRAM + no INT4 compute make ANE structurally unable to host a rotating expert set. This is a hard architectural mismatch, not a tuning problem.
- Realistic ANE resident-set size for a single compiled program is ≲32 MB of weights for full speed (graceful degradation beyond); that is far below any useful MoE expert bank.

### Gaps
- No published measurement of ANE streaming 4-bit weights from SSD (SSD-I/O + ANE co-design).
- No direct ANE-vs-GPU unified-memory bandwidth number isolated from the SSD path on an M4 MBA.

---

## Key Question 5: Known failed attempts — projects that tried LLM layers on ANE and abandoned it, with reasons

### Takeaway
The mainstream runtime ecosystem (llama.cpp, MLX, MLC-LLM, Ollama, LM Studio) **uniformly bypasses ANE**, and published attempts cite dispatch overhead, dynamic-shape incompatibility, compile-time weight baking, and lack of low-level control as the reasons.

### Cited Findings
- **Contra Collective (11 Jul 2026, M5 Max)**: "not an oversight" — every local LLM runtime targets Metal because ANE is a fixed-shape matrix machine and 1-token decode with growing KV cache is "close to the worst case for it." Hybrid prefill-ANE/decode-GPU "in theory" works but "nobody ships this in the mainstream local stack because the engineering cost of straddling two accelerators... outweighs the win." — [Contra Collective](https://contracollective.com/blog/gpu-vs-apple-neural-engine-local-llm-inference-m5-max-2026)
- **HN commentary (2026)**: "The LLM ecosystem standardized around CPU/GPU execution, and... seems unwilling to devote resources to ANE. Even Apple's MLX framework has no ANE [backend]." — [HN](https://news.ycombinator.com/item?id=47780473)
- **Orion framework comparison (Table 12, 2026)**: MLX, llama.cpp, MLC-LLM all "bypass the ANE entirely"; CoreML can schedule to ANE but gives no control; Orion is the only system targeting ANE directly — and it is a research prototype using **private APIs** that "may change without notice." — [Orion §9, §10](https://arxiv.org/html/2603.06728v1)
- **ANEgpt (prior work)**: implemented transformer training on ANE but "suffers from NaN divergence after the first training step on resume," uses Python, generates MIL via string concatenation, has no compiler — illustrating the fragility of ANE LLM attempts. — [Orion §2.3](https://arxiv.org/html/2603.06728v1)
- **CoreML overhead tax**: small ops run 2–4× slower via CoreML than direct API; "for latency-sensitive workloads (LLM token decoding)... the CoreML tax is severe." — [maderix](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- **Unsloth issue #6007**: community request for ANE/CoreML backend support for on-device LLMs. — [Unsloth issue](https://github.com/unslothai/unsloth/issues/6007)

### Inferences
- The abandonment is principled, not accidental: the ANE's fixed-shape, compile-time-weight design conflicts with autoregressive decode. A new Veystrix attempt would be fighting the same structural headwinds with private APIs.
- Orion's 2026 results prove ANE *can* run small transformers end-to-end, but its own limitations (M4 Max only, private APIs, no INT4, single-token decode slower than CPU) stop well short of a production SSD-bound MoE path.

### Gaps
- No documented attempt specifically to stream MoE experts through ANE (searched; none found) — the failure mode there is inferred from ANE's baked-weight architecture.
- No long-term maintenance/status data on Orion/ANEgpt beyond the papers.

---

## Key Question 6: Cases where ANE wins decisively, and whether such models could assist a larger LLM

### Takeaway
ANE wins decisively for **small, fixed-shape, always-on models** — embeddings, re-rankers, classifiers, speech/keyword spotting, prefill, speculative drafters, prompt routers, safety filters. These can assist a larger LLM (routing prediction, speculative decoding, prefix scoring, safety filtering) and are the only credible ANE role here — but they are auxiliary, and do not shorten an SSD-bound expert-read token.

### Cited Findings
- **Prompt router (MacPaw, M4 Pro, 350M bidirectional encoder, 128 tokens)**: ANE INT8 **5.5 ms / 126 mJ** vs MLX bf16 **15.8 ms / 797 mJ** vs Core AI fp16 **18.9 ms / 948 mJ** — 2.9× faster, 6.3× less energy. Running the router on ANE lowered chat generation rate by **<1%**, vs **20%** when on GPU. — [MacPaw Research](https://research.macpaw.com/projects/neural-engine)
- **Speculative decoding / tree verification on ANE (MacPaw)**: ANE tree decoding on M4 Pro reaches **136.9 tok/s** vs GPU 8-bit + DSpark **144.9 tok/s** (only ~5% lower), using a **328M DSpark drafter**; ANE-only used 2.6–3.4× less energy than MLX 8-bit + DSpark. — [MacPaw Research](https://research.macpaw.com/projects/neural-engine)
- **Split execution (MacPaw)**: assigning weight-free attention to GPU and keeping weight reads on ANE improves long-context latency — at 16K tokens, split tree decoding −64% energy / −17% latency vs GPU 8-bit+DSpark; split prefill −59% energy / −22% latency (lowest latency). — [MacPaw Research](https://research.macpaw.com/projects/neural-engine)
- **ANE as home turf**: "Embedding models, rerankers, classifiers, speech models, anything with a fixed shape and a modest parameter count runs beautifully through CoreML on the Neural Engine at very low power"; Apple's own on-device features "route through CoreML and the Neural Engine rather than MLX on the GPU." — [Contra Collective](https://contracollective.com/blog/gpu-vs-apple-neural-engine-local-llm-inference-m5-max-2026)
- **Per-op ANE speedups (Orion)**: classifier forward 10.2×, vocab softmax 33.8× — candidate for safety classifiers / token sampling acceleration. — [Orion §8.4](https://arxiv.org/html/2603.06728v1)
- **Apple's stated ANE wins**: minimize ML inference impact on app memory, app responsiveness, and device battery life. — [Apple ML Research](https://machinelearning.apple.com/research/neural-engine-transformers)

### Inferences
- The only measured ANE "routing helps generation" result (MacPaw, <1% vs 20% impact) used a **large batched 350M encoder**, not a 1-token router GEMM, and ran alongside GPU generation. If Veystrix's "router" is a small projection, this result does not transfer.
- A speculative drafter or safety classifier on ANE could coexist with the SSD-bound decode and reduce GPU contention — but with 75% of the token spent in SSD wait, there is little GPU contention to relieve.
- Therefore: ANE auxiliary models can improve energy/therms and *overall* system responsiveness, but the latency win on THIS token is bounded by Amdahl to ~1.15× (if the 51 ms router sync were fully removed) and realistically less.

### Gaps
- No measurement of an ANE router/drafter specifically in an SSD-bound MoE setting.
- No data on ANE safety classifier / prefix-scoring / cache-admission models for LLM serving (searched; conceptual only).

---

## Key Question 7: As of 2026, new Apple frameworks/announcements that materially change feasibility

### Takeaway
Two real 2026 changes: **Core AI (WWDC26, macOS 27+)** gives explicit CPU/GPU/ANE control and enabled MacPaw's ANE LLM engine; **M5 GPU Neural Accelerators (Metal 4 tensor API)** boost *GPU* matmul 4× TTFT / 25% token gen — but these are GPU tensor cores, not ANE, and **neither exists on the user's M4 MBA**. Net: nothing in 2026 makes ANE a viable accelerator for an SSD-bound MoE decode.

### Cited Findings
- **Core AI (WWDC26, macOS 27+)**: "designed with Apple silicon in mind... use the latest model architectures and inference techniques across the CPU, GPU, and Neural Engine." Enables running LLMs/generative AI entirely on-device with explicit ANE placement; requires OS 27+, so CoreML remains relevant. — [Apple Developer — Core AI](https://developer.apple.com/documentation/coreai); [WWDC26 session 324](https://developer.apple.com/videos/play/wwdc2026/324/); [InfoQ](https://www.infoq.com/news/2026/06/apple-core-ai-wwdc/); [HN](https://news.ycombinator.com/item?id=48449665)
- MacPaw's ANE engine is built on Core AI — the concrete proof that 2026 tooling unlocked ANE LLM inference (prefill+decode entirely on ANE). — [MacPaw Research](https://research.macpaw.com/projects/neural-engine)
- **M5 GPU Neural Accelerators (Oct 2025 announcement, ML research 2026)**: dedicated matmul units in every GPU core via Metal 4 tensor API; Apple reports **up to 4× TTFT vs M4 baseline, up to 25% faster token generation**, >4× peak GPU AI compute vs M4, +~30% unified-memory bandwidth to 153 GB/s; BaseRT paper reports up to **3.65× prefill**. These are **GPU**, not ANE, and require M5. — [Apple Newsroom M5](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/); [Apple ML — MLX M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5); [BaseRT arXiv:2607.19438v1](https://arxiv.org/html/2607.19438v1); [WWDC25 tech talk 111432](https://developer.apple.com/videos/play/tech-talks/111432/)
- **No evidence found of a distinct "MLTensor unified compute engine" API**; the relevant new unified API is Core AI. MLX remains GPU-targeted. — [Orion Table 12](https://arxiv.org/html/2603.06728v1); [Core AI docs](https://developer.apple.com/documentation/coreai)

### Inferences
- Core AI lowers the *engineering* cost of placing auxiliary fixed models on ANE (no private APIs), which strengthens the "ANE for router/classifier/drafter" proposition — but it does not change ANE bandwidth, SRAM, or compile-time-weight constraints, so it does not help SSD-bound expert reads.
- The M5 Neural Accelerators improve the *GPU* path, which is the accelerator that actually matters for MoE decode — but the user's M4 MBA cannot use them. On M4, GPU remains the right decode engine.
- Net for Veystrix's M4 MBA: framework changes do not overcome the physics of the SSD bottleneck.

### Gaps
- Core AI requires macOS 27; no data on whether it is available/compatible with the user's current OS on the M4 MBA — verify before relying on it.
- No Core AI documentation yet quantified for tiny single-token ops or MoE expert streaming.

---

## Bottom-line answer to the objective (for synthesis)

- **Could ANE improve an SSD-bound MoE decode in theory?** Only marginally and only on the non-bottleneck slices. Amdahl ceiling: freeing all GPU+router = 1.33×; freeing only router sync = 1.15×. ANE cannot accelerate the 289 ms SSD wait at all.
- **Is there measured evidence ANE helps THIS bottleneck?** **No.** All positive ANE results are compute-bound, batched/speculative, fixed-model, or non-Apple-SoC scenarios. No paper or benchmark covers ANE + SSD-streamed MoE experts.
- **What ANE would actually do if added**: add ~0.095 ms dispatch + ~2.3 ms round-trip per op; fail to stream 4-bit experts (baked weights, 32 MB SRAM, no INT4 compute, ~119-compile limit); possibly reduce system power/heat (real, but irrelevant to I/O-bound latency).
- **Where ANE genuinely wins and could assist**: auxiliary fixed models — prompt router (MacPaw: 2.9× faster / 6.3× less energy than GPU, <1% impact on generation), speculative drafter (136.9 tok/s), embeddings/re-rankers/safety classifiers. On a fanless MBA these reduce heat and free the GPU, but they do not shorten the SSD expert-read token.
- **Recommendation implied by evidence**: keep expert-read/decode on GPU (or CPU/SME for 1-token latency); invest in reducing SSD expert-read wait (prefetch/expert caching/quantization/IO pipelining) rather than ANE offload. Consider Core AI ANE only for an auxiliary router/classifier/drafter if energy/therms matter; do not expect a token-latency win on the SSD-bound path.
