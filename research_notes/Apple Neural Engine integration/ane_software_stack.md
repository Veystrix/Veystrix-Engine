# Apple Neural Engine (ANE) Software Stack & Tooling for Third-Party macOS Apps (2026)

Scope: public, third-party-accessible APIs only (Core ML / coremltools / MLX / runtimes). Private/reverse-engineered ANE paths are flagged explicitly. Current as of Nov 2026; macOS version requirements noted. All Apple platforms that can run Core ML require Apple silicon for ANE.

---

## Q1. How does a third-party macOS app request ANE execution? (computeUnits, MLComputePlan, newer Swift APIs)

### Takeaway
The only supported public route to the ANE is **Core ML**. An app sets `MLModelConfiguration.computeUnits`; Core ML's (private) Espresso compiler partitions the graph across CPU/GPU/ANE. There is **no API to pin a single operation to ANE**, and — correcting a premise in the assignment — the public `MLComputeUnits` enum has only **four** cases (`.all`, `.cpuOnly`, `.cpuAndGPU`, `.cpuAndNeuralEngine`). There is **no `.neuralEngine` and no `.gpuAndNeuralEngine` case**. Newer public APIs (WWDC24) that help build LLM pipelines are `MLTensor`, `MLState`, multi-function models, and `MLComputePlan`.

### Cited Findings
- Apple's `MLComputeUnits` enum topic section lists exactly four cases: `.all`, `.cpuOnly`, `.cpuAndGPU`, `.cpuAndNeuralEngine`. Overview text: "Use `all` to allow the OS to select the best processing unit to use (including the neural engine, if available)." — [Apple Developer docs JSON](https://developer.apple.com/tutorials/data/documentation/coreml/mlcomputeunits.json)
- `.all` = "allow the model to use all compute units available, including the neural engine"; `.cpuAndNeuralEngine` = "both the CPU and neural engine, but not the GPU"; `.cpuAndGPU` = "both the CPU and GPU, but not the neural engine". — [Apple MLComputeUnits reference](https://developer.apple.com/documentation/coreml/mlcomputeunits); [cpuAndNeuralEngine](https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine)
- `MLComputeUnits` introduced macOS 10.14/iOS 12, but `.cpuAndNeuralEngine` requires **macOS 13 / iOS 16+** ("unavailable on earlier systems"). — [Apple docs JSON](https://developer.apple.com/tutorials/data/documentation/coreml/mlcomputeunits.json); [ONNX Runtime Core ML provider header](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/providers/coreml/coreml_provider_factory.h)
- Practical meaning of policies: `.all` = "ANE preferred, GPU fallback, CPU last resort"; `.cpuAndNeuralEngine` = ANE + CPU only, no GPU; `.cpuAndGPU` = GPU + CPU, ANE explicitly excluded; `.cpuOnly` = no delegation. — [MVP Factory, "CoreML in iOS 18", Oct 2026](https://mvpfactory.io/blog/coreml-compute-units-scheduling-in-ios-18-neural-engine-vs-gpu-vs-cpu-fallback)
- **Silent fallback:** "a single unsupported op forces an entire subgraph onto the CPU, not just that op. No exception. No warning in the logs. It happens silently at model load time." ANE-incompatible patterns: dynamic shapes w/o fixed bound, custom `MLCustomLayer`s, certain gather/scatter with non-static indices, bfloat16 on older ANE. — [MVP Factory, Oct 2026](https://mvpfactory.io/blog/coreml-compute-units-scheduling-in-ios-18-neural-engine-vs-gpu-vs-cpu-fallback)
- `MLComputePlan` (iOS 17 / macOS 14+): `MLComputePlan.load(contentsOf:configuration:)`, then walk `plan.modelStructure.program.functions[...].block.operations` and call `plan.computeDeviceUsage(for: op)?.preferredComputeDevice` → `.cpu` / `.gpu` / `.neuralEngine`. Surfaces model structure, supported & preferred compute devices, operation support status, and estimated relative cost per op. — [WWDC24 session 10161 transcript](https://developer.apple.com/videos/play/wwdc2024/10161/); [MVP Factory](https://mvpfactory.io/blog/coreml-compute-units-scheduling-in-ios-18-neural-engine-vs-gpu-vs-cpu-fallback); [ANE Book Ch.1, Law 6](https://alvaro-videla.com/ane-book/01-ane-laws.html)
- Apple's own framing: "the MLComputePlan API offers model debugging and profiling information... The API surfaces the model structure and runtime information for each operation, including the supported and preferred compute devices, operation support status, and estimated relative cost." — [WWDC24 10161](https://developer.apple.com/videos/play/wwdc2024/10161/)
- **Caveat — the plan is not ground truth:** "MLComputePlan.deviceUsage(for:) reports almost every attention op as ANE-preferred... The reality at runtime on iPhone is very different. The static plan is a 'what's eligible' report, not 'what actually runs'. Real bench numbers are the only source of truth." — [imperatormk/coreml-birefnet ANE_DISPATCH_FINDINGS.md](https://github.com/imperatormk/coreml-birefnet/blob/main/ANE_DISPATCH_FINDINGS.md)
- Newer Core ML types from WWDC24 (macOS 15/iOS 18+): **MLTensor** (multi-dimensional array with async math/transform ops dispatched to Apple silicon, materialized to `MLShapedArray` to read), **MLState** (Core ML-managed state tensors, in-place KV-cache updates), **multi-function models** (multiple adapters/functions sharing one base model), plus the enhanced Xcode performance report and `MLComputePlan`. — [WWDC24 10161](https://developer.apple.com/videos/play/wwdc2024/10161/)
- Apple's demo in that session: modified HuggingFace Swift Transformers with **Mistral 7B** on an **M3 Max**; KV-cache via Core ML state finished in ~5s vs ~8s without (1.6x). "Support for states must be explicitly added to the model during the preparation phase." — [WWDC24 10161](https://developer.apple.com/videos/play/wwdc2024/10161/)
- Private/lower-level layers: Core ML dispatches with the help of **MPS Graph** and **BNNS Graph** (public but aimed at tight Metal integration / real-time CPU); the actual ANE runtime is **Espresso / ANECompilerService** (private, visible only via `log stream`). — [WWDC24 10161](https://developer.apple.com/videos/play/wwdc2024/10161/); [imperatormk findings, Round 7](https://github.com/imperatormk/coreml-birefnet/blob/main/ANE_DISPATCH_FINDINGS.md)
- A developer forum thread "Using the Apple Neural Engine for MLTensor operations" indicates ambiguity about whether MLTensor math itself can be ANE-dispatched; no public API was established there. — [Apple Developer Forums thread 775589](https://developer.apple.com/forums/thread/775589)

### Inferences
- To "force" ANE in a third-party app: set `computeUnits = .cpuAndNeuralEngine` (excludes GPU fallback), set an appropriate `minimum_deployment_target` (macOS 15/iOS 18+ unlocks ANE-native MIL ops such as `ios18.conv`), then **verify actual placement with `MLComputePlan` plus on-device benchmarks** — compilation success alone does not prove ANE execution.
- `MLTensor` and `MLState` are "glue" APIs for LLM pipelines; they do not give a per-op ANE escape hatch, and MLTensor math is documented as running on "Apple silicon's compute capabilities" generally, not specifically ANE.
- Hybrid CPU/GPU/ANE dispatch in one model is automatic under `.all`, but coarse-grained (per-subgraph), and a single hostile op can push the whole graph off ANE.

### Gaps
- No public per-operation ANE pinning API exists; confirmed absence, not just lack of docs.
- Could not confirm detailed WWDC 2025 / WWDC 2026 Core ML API additions (e.g. full semantics of `MLComputeEngine`) from primary sources; the WWDC24 session covered MLTensor/MLState/multi-function/MLComputePlan, not an ANE-specific "MLComputeEngine" custom-layer story. The assignment's premise about `MLComputeEngine` could not be verified from Apple docs.
- macOS version floor for `MLComputePlan` is stated as iOS 17 by practitioners; we did not open the exact Apple availability line for the macOS equivalent (expected macOS 14).

---

## Q2. coremltools: conversion, palettization/quantization, compute_precision, deployment targets, and realistic ANE model sizes on a 16 GB Mac

### Takeaway
Convert PyTorch → `mlprogram` with `minimum_deployment_target` = macOS 15 / iOS 18+ to get ANE-native ops; compress with palettization (best for ANE) or W8A8 / INT4 (GPU-friendly); ANE has an empirically hard per-package weight ceiling around **~250 MB**, so any multi-billion-parameter LLM must be **sharded** into 1–3 layers per `.mlpackage` and orchestrated from Swift.

### Cited Findings
- Compression types in Core ML: **palettization** (LUT, `nbits` ∈ {1,2,3,4,6,8}), **linear quantization** (INT4/INT8 weights, INT8 activations), **pruning** (sparse weights); they can be combined. — [coremltools opt-overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)
- OS availability table: iOS16/macOS13 = per-tensor palettization {1,2,4,6,8}, 8-bit weight-only quant, pruning (via ahead-of-time decompress to fp16); iOS17/macOS14 = W8A8 activation quant "accelerated on NE for A17 Pro/M4 chips", on-the-fly decompression; iOS18/macOS15 = grouped-channel palettization, 3-bit, INT8 LUTs, per-channel scale+LUT, INT4 per-block, quantization of non-zero sparse values. Optimizations apply to `mlprogram` only for iOS16/macOS13+. — [coremltools opt-whats-new](https://apple.github.io/coremltools/docs-guides/source/opt-whats-new.html)
- Direct Apple guidance: *"Weight palettization (all bits from 1 to 8) typically works the best on the Neural Engine for runtime memory and latency gains. On the GPU you may see runtime memory benefits."* Also: *"8-bit activation plus weight quantization, also referred to as the W8A8 mode, can lead to considerable latency benefits on the Neural Engine by leveraging the faster int8-int8 compute path supported in newer hardware (A17 pro, M4)."* And: *"INT4 per-block quantization of weights can work really well for models using the GPU on a Mac."* — [coremltools opt-overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)
- `compute_precision=ct.precision.FLOAT16` is the Core ML default; FLOAT32 used when full precision needed on CPU/GPU. ANE natively accelerates float16 (conversion to fp16 is done by the runtime). — [coremltools typed-execution](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html); [MLX issue #18, vade comment](https://github.com/ml-explore/mlx/issues/18)
- `skip_model_load=True` in `ct.convert(...)` is routinely used to avoid a massive ANE compile *during conversion*. — [ANE Book Ch.1 Law 2](https://alvaro-videla.com/ane-book/01-ane-laws.html); [Apple "On Device Llama 3.1 with Core ML"](https://machinelearning.apple.com/research/core-ml-on-device-llama)
- **ANE empirical laws (practitioner, M4 Max h16g, 16 ANE cores; 35+ conversion experiments):** all tensors must be 4-D shaped `[1, C, T, 1]`; ANE-native matmul is MIL `ios18.conv` (requires `minimum_deployment_target` macOS 15/iOS 18+, `compute_units=CPU_AND_NE` preferred over `ALL` to avoid silent GPU fallback); INT8 per-tensor is the production baseline; per-block INT4 falls off ANE in small shards. — [ANE Book Ch.1](https://alvaro-videla.com/ane-book/01-ane-laws.html)
- **Shard-size ceiling:** "A single `.mlpackage` that compiles and stays fully ANE-resident has an empirical upper limit of approximately **250 MB** of model weights... The validated ceiling is 223 MB (a 3-layer Phi-4-mini shard)... > ~1 GB → hard ANEF error -14 (no ANE at any quantization)... models larger than ~2B parameters require sharding." For 7B+ models: 1–3 transformer layers/shard; embeddings and LM head split separately. — [ANE Book Ch.1, Laws 4 & 5](https://alvaro-videla.com/ane-book/01-ane-laws.html)
- Apple's Llama-3.1-8B reference (GPU path, M1 Max): fp16 model = ~16 GB; INT4 block-wise (block 32) = 4.2 GB; ~33 tokens/s with fused SDPA + stateful KV cache. "low-bit palettization greatly reduces the model's memory footprint and improves latency on the neural engine, [while] block-wise quantization... is optimized for the GPU." — [Apple, "On Device Llama 3.1 with Core ML"](https://machinelearning.apple.com/research/core-ml-on-device-llama)
- Custom operators: unsupported PyTorch ops need a Core ML **custom layer** implemented in Swift; Apple explicitly advises "Use Custom Operators as a Last Resort." Custom layers run on CPU/Metal, not ANE. — [coremltools custom-operators guide](https://apple.github.io/coremltools/docs-guides/source/custom-operators.html); [Apple custom layers doc](https://developer.apple.com/documentation/coreml/creating-and-integrating-a-model-with-custom-layers)

### Inferences
- On a 16 GB Mac, a ~7–8B model at INT4 (~4–5 GB) fits comfortably in memory **on the GPU via MLX/Core ML GPU**, but does **not** fit the ANE's ~250 MB single-package ceiling; routing a full LLM to ANE requires ~10–30 shards and a Swift orchestrator, which adds host round-trip overhead.
- Realistic ANE-resident workloads on 16 GB Macs: vision encoders, small classifiers, and individual 1–3 layer transformer shards of a larger model — not monolithic LLMs.
- Choose quantization by target: palettization / W8A8 → ANE; INT4 per-block → GPU. Always build for macOS 15/iOS 18+ targets.

### Gaps
- The ~250 MB ANE ceiling and error -14 are from an independent black-box book, **not** Apple documentation; no official Apple statement of an ANE model-size limit was found. Treat as strong empirical signal, not spec.
- No authoritative coremltools doc explicitly naming a `neural_engine` *target* argument — deployment target + compute units + op format (`mlprogram`/`ios18.conv`) is the actual mechanism.
- Cross-generation ANE support matrix for INT4/INT8 ops was not obtained.

---

## Q3. Does MLX / mlx-lm, llama.cpp, PyTorch MPS, or another common runtime use the ANE? (M5 "Neural Accelerators" distinction)

### Takeaway
No mainstream runtime uses the ANE. **MLX explicitly will not** (closed API); llama.cpp/ggml, PyTorch MPS, and MLX all run on **Metal GPU + CPU**. The M5 "Neural Accelerators" are **GPU matrix-multiply units**, not the ANE. Third-party ANE runtimes exist but depend on **private / reverse-engineered APIs** and are unsupported.

### Cited Findings
- MLX issue #18 "ANE support" was closed as `wontfix`; maintainer awni: *"at the moment we don't have plans to support ANE in MLX given it is a closed source API. If / when that changes we will be first in line to add it as a supported device."* On a May 2025 follow-up: *"No plan for that at the moment."* He also noted the GPU is often faster for large models on high-GPU-core machines. — [ml-explore/mlx#18](https://github.com/ml-explore/mlx/issues/18)
- mlx-examples issue #925: "MLX doesn't support the ANE unfortunately." — [ml-explore/mlx-examples#925](https://github.com/ml-explore/mlx-examples/issues/925)
- Technical reasons given in the MLX issue: ANE is inference-only (forward prop), fixed hardware op set, fp16-native, requires OS scheduling with no residency guarantee, API exposed only via Core ML/Swift, internally Espresso (private C++); custom-layer + IOSurface-backed zero-copy hybrid was proposed as a theoretical workaround but requires per-model engineering. — [ml-explore/mlx#18, vade comments](https://github.com/ml-explore/mlx/issues/18)
- MLX design: "Operations in MLX can run on either the CPU or the GPU without needing to move memory around" — no ANE device. — [Apple ML Research, "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU"](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- **M5 Neural Accelerators ≠ ANE:** "The GPU Neural Accelerators introduced with the M5 chip provides dedicated matrix-multiplication operations... MLX leverages the Tensor Operations (TensorOps) and Metal Performance Primitives framework introduced with Metal 4 to support the Neural Accelerators' features." Requires **macOS 26.2+**; up to 4× TTFT speedup vs M4. A separate "16-core Neural Engine" block (used by Core ML) is explicitly distinguished in 2026 coverage. — [Apple ML Research M5 post](https://machinelearning.apple.com/research/exploring-llms-mlx-m5); [Sitepoint 2026 guide](https://www.sitepoint.com/local-llms-apple-silicon-mac-2026/)
- llama.cpp: official ggml backend is Metal/CPU; a long-standing "Neural Engine Support #336" discussion shows no mainline ANE backend; 2026 ANE forks/PRs (e.g. ane-infer, llama-cpp-ane) are community reverse-engineering efforts using **private APIs** and Metal shader hybrids, explicitly outside supported Apple APIs. — [llama.cpp discussion #336](https://github.com/ggml-org/llama.cpp/discussions/336); [thebasedcapital/ane-infer](https://github.com/thebasedcapital/ane-infer); [richginsberg/llama-cpp-ane](https://github.com/richginsberg/llama-cpp-ane)
- PyTorch MPS = Metal Performance Shaders backend (GPU). ANE use from the PyTorch ecosystem goes through **ExecuTorch's Core ML backend** (which converts via coremltools and runs via Core ML), not MPS. — [Apple, On Device Llama 3.1 with Core ML — ExecuTorch section](https://machinelearning.apple.com/research/core-ml-on-device-llama)

### Inferences
- For the Veystrix MLX stack, ANE is **not reachable from MLX directly**. The only supported design is a hybrid: convert the ANE-suitable subgraphs to Core ML (`.mlpackage`/`.mlmodelc`) and call Core ML from Swift, leaving the rest on MLX/Metal. Data handoff between MLX arrays and Core ML tensors is the main engineering cost.
- New M5 Macs narrow ANE's advantage for LLM prefill because GPU neural accelerators deliver large TTFT gains with none of ANE's model-size/op constraints.

### Gaps
- No measured MLX↔Core ML copy/interop benchmark was found; feasibility of true zero-copy hybrid (IOSurface/Metal buffers shared between MLX and Core ML) is only discussed theoretically in the MLX issue.
- No official statement from Apple on whether it will ever open ANE to non-Core-ML runtimes.

---

## Q4. Apple's own on-device LLM strategy: how Apple Intelligence uses ANE vs GPU

### Takeaway
Apple's on-device foundation model is a ~3B-parameter model compressed to **2 bits/weight** via QAT, with KV-cache sharing and dynamically swapped **LoRA adapters**. Apple has publicly said the on-device model runs on the ANE, but has **never published the exact ANE-vs-GPU execution split**. Third parties can invoke it only through the Foundation Models framework, not with arbitrary weights.

### Cited Findings
- "~3B-parameter on-device model optimized for Apple silicon through architectural innovations such as **KV-cache sharing** and **2-bit quantization-aware training**" — [Apple Intelligence Foundation Language Models Tech Report 2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025); [arXiv:2507.13575](https://arxiv.org/html/2507.13575v2)
- KV-cache sharing: model split into Block 1 = 62.5% of layers and Block 2 = 37.5% with K/V projections removed; KV cache memory −37.5%, TTFT −~37.5%. Embeddings quantized to 4-bit, KV cache to 8-bit, plus low-rank (LoRA) quality-recovery adapters. — [arXiv:2507.13575](https://arxiv.org/html/2507.13575v2)
- 2024 Apple AFM paper states "our AFM-on-device model running on **Apple Neural Engine**..." in the context of adapters — [arXiv:2407.21075](https://arxiv.org/html/2407.21075v1) (found via search snippet; treat as indicative).
- Apple Intelligence comprises multiple specialized generative models that "adapt on the fly"; LoRA adapters are dynamically loaded/swapped over a shared base model. — [Apple, "Introducing Apple's On-Device and Server Foundation Models"](https://machinelearning.apple.com/research/introducing-apple-foundation-models); [arXiv:2507.13575 §5](https://arxiv.org/html/2507.13575v2)
- Vision support uses a 300M-param RW-ViTDet on-device encoder with window attention (only three global layers), illustrating ANE-friendly architectural choices — [arXiv:2507.13575 §2.3](https://arxiv.org/html/2507.13575v2)
- Foundation Models framework gives third-party developers "guided generation, constrained tool calling, and LoRA adapter fine-tuning"; Apple supplies an **adapter training toolkit** for specializing the on-device system LLM. — [Apple Foundation Models adapter doc](https://developer.apple.com/apple-intelligence/foundation-models-adapter); [Tech Report 2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)
- Apple Intelligence support is gated by chip ANE capability (A17 Pro / M1 class or newer); older, less capable ANEs are cited as the reason older iPhones are excluded. — [Wikipedia, "Apple Intelligence"](https://en.wikipedia.org/wiki/Apple_Intelligence) (secondary source)

### Inferences
- Apple's published details emphasize **model architecture and quantization**, not engine placement. The presence of a ~1 GB-class 2-bit 3B model is consistent with ANE use via sharding (per the ~250 MB ANE package ceiling) and/or ANE+GPU split, but Apple has not confirmed the mechanics publicly.
- Core ML multi-function models (WWDC24) are the public analogue of Apple's adapter-switching strategy: one base model, multiple functions/adapters sharing weights.
- Third-party apps cannot load arbitrary weights into Apple's on-device FM; they can only use the Foundation Models framework (and train LoRA adapters via Apple's toolkit), so this is not a route to general ANE compute.

### Gaps
- No published data on which fraction of the foundation model runs on ANE vs GPU, nor ANE dispatch/wake latency inside Apple Intelligence.
- No statement on whether smaller supporting models (vision encoder, safety/quality filters, etc.) are ANE-only while the foundation model is GPU-heavy.

---

## Q5. Known projects running transformers / LLMs on ANE — achievements and blockers

### Takeaway
Apple's `ml-ane-transformers` shows large gains for encoder-style transformers (up to 10×), but recent hands-on work shows its "ANE-friendly" rewrite can be **slower on modern hardware/iOS**, and full-resolution attention models can defeat ANE compilation. Large LLM-on-ANE demos rely on sharding; several 2026 ANE backends use private APIs.

### Cited Findings
- **apple/ml-ane-transformers**: "up to **10 times faster** and **14 times** lower peak memory consumption compared to baseline implementations" on A14+/M1+; DistilBERT tutorial sees 2.84× at seq len 128; at seq len 512 / batch 8 up to 10× lower latency. 4 of 606 ops (embedding lookups) run on CPU. Load/compilation times increase due to op count — a one-time cost, load asynchronously. — [apple/ml-ane-transformers README](https://github.com/apple/ml-ane-transformers); [Apple research article](https://machinelearning.apple.com/research/neural-engine-transformers)
- **apple/ml-vision-transformers-ane**: companion project for deploying attention-based vision transformers to ANE. — [GitHub](https://github.com/apple/ml-vision-transformers-ane)
- **Contradicting the 2022 advice on modern hardware (iPhone 14 Pro / A16 / iOS 26.2, coremltools 9.0):** the `ml-ane-transformers`-style Conv2d-1×1 attention rewrite was *slower* on ANE (374 ms vs 227 ms for the standard rewrite); pure conv was ~8× faster than GPU (3.4 ms vs 27 ms); windowed Swin attention (N=49/window) is ANE-native (15 ms vs 226 ms unwindowed at N=4096); `grid_sample`/`mb.resample` ran 8× slower on ANE than GPU. — [imperatormk/coreml-birefnet ANE_DISPATCH_FINDINGS.md](https://github.com/imperatormk/coreml-birefnet/blob/main/ANE_DISPATCH_FINDINGS.md)
- **Full-model ANE compile failures:** full BiRefNet Swin at 1024² hangs ANE compile (>200 s), 512² >30 min, while 256² compiles in 78 s and runs at 199 ms; Swin window-partition reshapes at high resolution, not deformable conv, were the blocker. — [imperatormk findings, Rounds 6–7](https://github.com/imperatormk/coreml-birefnet/blob/main/ANE_DISPATCH_FINDINGS.md)
- **Anemll** (ANE LLM project) with ANE profiler: reports `CPU_AND_NE` fastest/consistent (6.25 ms) vs `ALL` picking ANE with higher variance — [Anemll ANE_PROFILER.md](https://github.com/Anemll/Anemll/blob/main/anemll/utils/ANE_PROFILER.md); **anemll-profile** cost-model profiler — [GitHub](https://github.com/Anemll/anemll-profile)
- **CoreML-LLM** / Swift LLM projects targeting ANE + GPU on iOS 18+ / macOS 15+ — [john-rocky/CoreML-LLM](https://github.com/john-rocky/CoreML-LLM)
- **Private/reverse-engineered paths**: tinygrad's ANE support and newer `ane-infer`, `llama-cpp-ane`, `whisper_ane` projects use private Espresso APIs and are not Apple-supported. — [MLX issue #18 comments](https://github.com/ml-explore/mlx/issues/18); [thebasedcapital/ane-infer](https://github.com/thebasedcapital/ane-infer); [Synopsis/whisper_ane](https://github.com/Synopsis/whisper_ane)

### Inferences
- Apple's own ANE-transformer guidance is generation-specific; benchmark every architectural trick on the target chip/OS rather than assuming the 2022 recipes still hold.
- The recurring blocker for ANE on transformers is **row-reduction ops (softmax/LayerNorm) at large sequence length** and reshape patterns that blow up the ANE compiler — not raw FLOPs.

### Gaps
- No single canonical "LLM fully on ANE" reference implementation from Apple; all large-LLM ANE work is shard-based community experimentation.

---

## Q6. Runtime practicalities: dispatch/wake latency, graph compile, load time, memory, GPU interop, hybrid pipelines

### Takeaway
ANE compile time can range from hundreds of ms to tens of minutes and scales badly with graph size/resolution; per-process memory limits kill large graphs; ANE↔GPU transfers are expensive; Core ML supports hybrid GPU+ANE within a model but falls back silently. Direct MLX↔ANE data sharing is not supported.

### Cited Findings
- ANE first-call compile cost is "~340–4500 ms per model"; benchmarks used 3 warmup runs to factor it out. — [imperatormk findings, methodology](https://github.com/imperatormk/coreml-birefnet/blob/main/ANE_DISPATCH_FINDINGS.md)
- Large-graph compile: 512² full BiRefNet reached **48 minutes of ANECompilerService CPU time** and still didn't finish; 256² compiles in 78 s. ANE is "not always faster than GPU" — per-pixel GPU was faster; ANE only won because a smaller model was run. — [imperatormk findings, Rounds 6–7](https://github.com/imperatormk/coreml-birefnet/blob/main/ANE_DISPATCH_FINDINGS.md)
- Memory: "FP16 NN-format BiRefNet hit 3.2 GB peak espresso working set and got JetsamEvent-killed." — [imperatormk findings, methodology](https://github.com/imperatormk/coreml-birefnet/blob/main/ANE_DISPATCH_FINDINGS.md)
- Compiler internals from `log stream --predicate 'process == "ANECompilerService"'`: ~9,600 Espresso layers at 512²; 11,677 ANE-eligible checks vs 286 non-eligible in a 60 s window; 526 context transfers + 262 single-op fallback segments; ~30+ **single-threaded** serial passes over the full graph; hundreds of "Kernel validation warning" fallback rejections. — [imperatormk findings, Round 7](https://github.com/imperatormk/coreml-birefnet/blob/main/ANE_DISPATCH_FINDINGS.md)
- Latency comparison (A17 Pro / iPhone 15 Pro, MobileNetV3-style): ANE 8–12 ms, GPU 15–25 ms, CPU 60–120 ms. — [MVP Factory, Oct 2026](https://mvpfactory.io/blog/coreml-compute-units-scheduling-in-ios-18-neural-engine-vs-gpu-vs-cpu-fallback)
- Graph-split cost: manually decomposing softmax (`reduce_max → sub → exp → reduce_sum → div`) forces `reduce_max`/`sub` to GPU, creating ANE↔GPU↔ANE ping-pong and worse latency than the single fused ANE softmax op. Row-reductions along the last axis break ANE tile streaming; penalty scales ~N². — [imperatormk findings, Rounds 2–4](https://github.com/imperatormk/coreml-birefnet/blob/main/ANE_DISPATCH_FINDINGS.md)
- Core ML "states": passing the KV cache as `MLState` (vs model I/O) improved Llama-3.1-8B decode on M1 Max from 1.25 → 16.26 tokens/s (~13×) by avoiding cache copies; compiler may update state in place. — [Apple, On Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama)
- A pathological case: adding QK-Norm caused Core ML to fuse `qk_norm→matmul→softmax→matmul` into an MPS kernel (`sdpa_tile_fwd_8x8x8_noEdgeCheck`) with a stack-space bug on A16/iOS 26.2, crashing during `MLModel(contentsOf:)`. — [imperatormk findings, Round 2](https://github.com/imperatormk/coreml-birefnet/blob/main/ANE_DISPATCH_FINDINGS.md)
- Hybrid pipelines: Core ML can dispatch one model across CPU+GPU+ANE under `.all` (ANE preferred → GPU fallback → CPU last); a common workaround is to isolate known-incompatible ops into a separate `.cpuOnly`/GPU model so they don't contaminate ANE subgraph scheduling. — [MVP Factory](https://mvpfactory.io/blog/coreml-compute-units-scheduling-in-ios-18-neural-engine-vs-gpu-vs-cpu-fallback)
- MLX↔Core ML interop: zero-copy via IOSurface-backed fp16 buffers + Core ML custom layers implemented in MLX was proposed as a theoretical "best of both worlds" pattern but requires ad-hoc per-architecture work; no measured copy-cost data. — [MLX issue #18, vade comments](https://github.com/ml-explore/mlx/issues/18)

### Inferences
- ANE is best for small, stable, conv/MLP-heavy subgraphs with minimal reductions; it is a poor fit for long-context attention and for graphs that trigger many CPU/GPU fallback segments.
- Precompile `.mlmodelc` and load asynchronously; expect multi-minute first-launch compiles for large ANE graphs — a first-launch UX blocker.
- For an MLX-based app, a hybrid design is possible in principle (Core ML for ANE-suitable shards, MLX/Metal for the rest) but there is no off-the-shelf zero-copy bridge; budget for per-model integration work.

### Gaps
- No quantitative public benchmark of ANE wake-up/dispatch latency (as opposed to compile time) or of MLX-array→Core ML copy costs.

---

## Q7. Documentation / blogs for compute plans, Instruments ANE debugging, and measuring ANE vs GPU vs CPU

### Takeaway
The main public tools are `MLComputePlan` (code-level per-op device inspection), Xcode's Core ML performance report (no code), and the Instruments Core ML instrument (per-subgraph compute-unit timelines). Community tools dump per-layer device/cost from Core ML models.

### Cited Findings
- WWDC24 session 10161 documents the enhanced Xcode performance report: estimated time per op, compute-device support, hover hints for unsupported ops, exportable/comparable runs; and `MLComputePlan` for the same info in code. — [WWDC24 10161 transcript](https://developer.apple.com/videos/play/wwdc2024/10161/)
- `MLComputePlan.load(contentsOf:configuration:)` + `computeDeviceUsage(for:)` code snippet, recommended as a CI gate to "assert zero CPU fallback" — [MVP Factory](https://mvpfactory.io/blog/coreml-compute-units-scheduling-in-ios-18-neural-engine-vs-gpu-vs-cpu-fallback); [ANE Book Law 6](https://alvaro-videla.com/ane-book/01-ane-laws.html)
- Instruments → **Core ML** template: the "Compute lane shows ANE/GPU/CPU usage per layer; filled checkmark = executed on that unit; empty checkmark = supported but not chosen"; pair with `os_signpost` points of interest to isolate dispatch overhead. — [CoreML optimizer skill reference](https://github.com/ckorhonen/claude-skills/blob/main/skills/coreml-optimizer/references/neural-engine-guide.md); [MVP Factory](https://mvpfactory.io/blog/coreml-compute-units-scheduling-in-ios-18-neural-engine-vs-gpu-vs-cpu-fallback)
- Community tools: **CoreMLInspect** — "See the device (CPU/GPU/ANE) and estimated cost for every layer in your CoreML model" — [smpanaro/CoreMLInspect](https://github.com/smpanaro/CoreMLInspect); **Anemll/anemll-profile** — ANE cost-model profiler — [GitHub](https://github.com/Anemll/anemll-profile); `ane_profile.swift` in FluidAudio — [FluidAudio](https://github.com/FluidInference/FluidAudio/blob/main/Scripts/ane_profile.swift)
- For compiler-level debugging (private, diagnostic only): `log stream --predicate 'process == "ANECompilerService"' --style compact --info --debug` reveals Espresso passes/fallbacks; "Apple does not redact internal Espresso pass names." — [imperatormk findings, Round 7](https://github.com/imperatormk/coreml-birefnet/blob/main/ANE_DISPATCH_FINDINGS.md)
- coremltools `MLModel._get_compute_plan()` can inspect placement from Python — [ANE Book Law 6](https://alvaro-videla.com/ane-book/01-ane-laws.html); Xcode model preview / performance tab uses `.mlmodelc` on connected devices — [WWDC24 10161](https://developer.apple.com/videos/play/wwdc2024/10161/)

### Inferences
- A robust workflow: convert for macOS 15+ → assert ANE residency with `MLComputePlan` in CI → confirm with Xcode performance report on target hardware → validate real latency with signposts/Instruments → use `ANECompilerService` logs only for diagnosing pathological compile times.

### Gaps
- Apple's Core ML Instruments documentation page was not opened directly; details here come from a community skill reference and practitioner blogs.
- No dedicated Apple doc quantifying ANE-vs-GPU-vs-CPU counters beyond the Xcode/Instruments UI.
