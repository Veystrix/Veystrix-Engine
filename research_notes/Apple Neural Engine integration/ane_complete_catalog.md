# Apple Neural Engine — Complete Verified Constraint & Capability Catalog

**Living reference for the team building ANE features. Research notes only — no code changes.**

Compiled 2026-11-17. Every rule below carries (a) the exact statement, (b) the exact source file/section and URL, and (c) an evidence class. Where sources conflict, the conflict is preserved in §8 rather than silently resolved.

---

## 0. Source registry, evidence classes, and how to read this document

### 0.1 Evidence classes (used in every rule tag)

| Tag | Meaning |
| --- | --- |
| **[A-DOC]** | Apple-documented: Apple Machine Learning Research, Apple reference repo README, or coremltools official guide. Authoritative for *public API / supported* behavior; Apple documents almost no ANE internals. |
| **[C-MEAS]** | Community-measured: direct measurement by an independent researcher/project (physical Apple silicon, clocks, power, breakpoints, or private-API probes). Reproducible in principle; not Apple-guaranteed. |
| **[S-SRC]** | Single-source: asserted by exactly one research body (one person/repo/guide), decompile-derived or otherwise not corroborated here. Treat as a hypothesis until re-measured on our hardware. |
| **[C-ANEC]** | Community catalog (hollance): explicitly self-labeled "incomplete and possibly wrong"; mostly iOS 13/14-era. Use as a starting point, not gospel. |

### 0.2 Primary sources read in full for this catalog

| Key | Source | Vintage / nature | URL |
| --- | --- | --- | --- |
| **H** | hollance/neural-engine `docs/` — all files: `supported-devices.md`, `why-care.md`, `running-on-ane.md`, `prevent-running-on-ane.md`, `is-model-using-ane.md`, `programming-ane.md`, `ane-vs-gpu.md`, `16-bit.md`, `unsupported-layers.md`, `model-surgery.md`, `internals.md`, `other.md`, `reverse-engineering.md`, plus `os-log.md` | ~2019–2022, iOS 13/14 era, ML consultant, self-labeled experimental | https://github.com/hollance/neural-engine/tree/master/docs |
| **G** | *Apple Neural Engine: Architecture, Programming, and Performance*, Spencer H. Bryngelson, ed. v3 (2026-10-14), read in full: Introduction; ch. 3 Numerics, 4 Capability surface; 5 Software stack, 6 Dispatch without Core ML, 7 Weights and compression; 9 Roofline, 11 ANE/GPU/CPU, 12 Cross-chip; 14 LLM case study, 17 Model-design rules, 19 Pitfalls; 21 Memory hierarchy; 33 Telemetry; 36 Predicted upper tier; Appendix A op-by-device matrix, Appendix D glossary; back matter | 2026, **single research body**, measured on M1/M2/M5 + decompiled firmware/driver/compiler; claims carry per-claim provenance marks | https://ane-guide.readthedocs.io/en/latest/ · repo https://github.com/sbryngelson/ane-guide |
| **M1/M2/M3** | maderix (Manjeet Singh), "Inside the M4 Apple Neural Engine" Parts 1–3 (Substack), Feb–Mar 2026; human + Claude collaboration; direct `_ANEClient` private-API work on M4 Mac mini, macOS 15.x | 2026, single author, community-measured | P1 https://maderix.substack.com/p/inside-the-m4-apple-neural-engine · P2 https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615 · P3 https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-c8b |
| **MR** | `maderix/ANE` README (MIT), updated through 2026 | 2026, single-source, includes a **newer INT8 W8A8 claim** that conflicts with M1–M3 (see §8) | https://github.com/maderix/ANE |
| **MO** | `m0at/ANEtransformers` README + `docs/ANE_2PHASE.md`, `docs/ANE_FUSED_KV.md`, `docs/BUILDING_MISTRAL.md`, `docs/specs/07-ane-vs-gpu-neural-accelerators.md`, `docs/specs/01-dispatch-overhead.md` | 2026, M5 24 GB, single-source; fork of maderix | https://github.com/m0at/ANEtransformers |
| **AML** | Apple Machine Learning Research, "Deploying Transformers on the Apple Neural Engine" (2022-06-06) | Apple-authored design guidance | https://machinelearning.apple.com/research/apple-neural-engine |
| **AREF** | `apple/ml-ane-transformers` README | Apple-authored reference implementation notes | https://github.com/apple/ml-ane-transformers |
| **CML** | coremltools Guide: `opt-overview.md`, `opt-palettization-overview.md`, `opt-palettization-perf.md`, `opt-quantization-overview.md`, `opt-quantization-perf.md`, `multifunction-models.md`, `mlmodel-debugging-perf-utilities.md`, `model-intermediate-language.md` | Apple-authored tooling docs (copyright 2024, current main branch) | https://apple.github.io/coremltools/docs-guides/source/ · raw: https://github.com/apple/coremltools/tree/main/docs-guides/source |
| **APL** | Apple Developer docs, `MLComputePlan` | Apple API reference (page title verified; JS-rendered detail not fully retrievable here) | https://developer.apple.com/documentation/coreml/mlcomputeplan |

### 0.3 Critical framing caveat

The sources span **three generations of knowledge**: (1) hollance's black-box iOS-era observations, (2) Apple's 2022 public transformer guidance and current coremltools docs, and (3) 2026 private-API/reverse-engineering work (ane-guide, maderix, m0at). Many apparent "contradictions" are **different access paths** (public Core ML vs private `_ANEInMemoryModel`/`e5rt`) or **different hardware generations** (M1 vs M4/M5). §8 states both sides verbatim with citations and gives the reconciliation where one is evident.

---

## 1. Which Core ML layer/op types force CPU/GPU fallback

### 1.1 Every layer type hollance names as non-ANE (the canonical community list)

Source: **H**, `docs/unsupported-layers.md` → "Problematic layers", "Broadcastable layers", "Custom layers". File-level warning: *"incomplete and possibly wrong… may even be different between versions of the ANE."* **[C-ANEC]**

| # | Layer/op named | Exact rule as stated |
| --- | --- | --- |
| 1 | Custom layers | "you can only provide a CPU and GPU implementation. Because there is no public API to program the ANE, custom layers cannot run on the ANE." |
| 2 | RNN layers — LSTM, GRU | "RNN layers such as LSTM or GRU" do not work on ANE. |
| 3 | `gather` | Named in "Problematic layers" (H `unsupported-layers.md`); this is why gather-heavy routing cannot run end-to-end on ANE. |
| 4 | Dilated convolutions | Named as not working (note: G Appendix A says dilated conv *is* native on M1+ — see §8). |
| 5 | Broadcastable layers | `AddBroadcastableLayer`, `DivideBroadcastableLayer`, `MultiplyBroadcastableLayer`, `SubtractBroadcastableLayer`, "and a number of others". |
| 6 | "ND" layers | `ConcatNDLayer`, `SplitNDLayer`, `LoadConstantNDLayer`, "and so on" — any rank-agnostic ND layer. |
| 7 | Certain broadcasting ops | e.g. "multiply of `CxHxW` and `Cx1x1` tensors". |
| 8 | Pooling with kernel size > 13 | "pooling layers with kernel size greater than 13". |
| 9 | Pooling with stride > 2 | "or stride greater than 2". |
| 10 | Upsampling factor > 2 | "upsampling with scaling factors greater than 2". |

H also lists layers it observed **working**, despite the above: "upsampling: Recently I worked with a model where upsampling layers appeared to run just fine on the ANE (Core ML also has a `Espresso::ANERuntimeEngine::upsample_kernel`)" and "deconvolution" — H `unsupported-layers.md` → "Layers that do work". **[C-ANEC]**

Model-surgery replacements H recommends (H `docs/model-surgery.md`): `AddBroadcastableLayer`→`AddLayer`; `MultiplyBroadcastableLayer`→`MultiplyLayer`/`ScaleLayer`/linear `Activation`; `ConcatND`→`Concat`; `LoadConstantND`→`LoadConstant` (rank-3 only, may need flattening); no `SubtractLayer` exists — emulate with linear activation `alpha=-1` then `AddLayer`. **[C-ANEC]**

### 1.2 Complete op-status census (2026, direct ANE path) — every op with no ANE path

Source: **G**, Appendix A "Operation-by-device matrix" (187 MIL ops; M1/M2/M5 columns measured on silicon, M3/M4 decompile-predicted). Status legend in Appendix A: *No path* = "rejected on every family from the M1 through the M5, computed off-engine"; *Bridge* = "reachable only through a decompose, software fallback, or compiler-internal route, never as a standalone code-generated operation"; *Family-gated* = native only from a named later family. **[C-MEAS / S-SRC]**

**No path on any family M1→M5 (must be computed off-engine):**
`reduce_prod`; `mod`; `logical_and`, `logical_or`, `logical_xor`; `scatter`, `scatter_along_axis`, `scatter_nd`; `reverse_sequence`; `band_part`; `non_zero`; `one_hot`; `shape`; `sliding_windows`; `range_1d` (M1 codegen rejects); `list_gather`, `list_length`, `list_read`, `list_scatter`, `list_write`, `make_list`; recurrences `gru`, `lstm`, `rnn`; `tan`, `asin`, `acos`, `asinh`, `acosh`, `atanh`, `sinh`, `cosh`; `random_bernoulli`, `random_categorical`, `random_normal`; `constexpr_cast`. (G App. A tables `apa-elementwise`, `apa-compare-logical`, `apa-data-movement`, `apa-recurrent`, `apa-trig-math`, `apa-detection-sampling`, `apa-quant-dtype`.) G ch. 4 "Hard limits" restates the installed base: `"reduce_prod, the scatter family (scatter, scatter along axis, scatter ND), mod, one_hot, non_zero, band_part, reverse_sequence, shape, sliding_windows, the logical and, or, and xor operations, the recurrent gru, lstm, and rnn cells, the inverse and hyperbolic trigonometric functions, and the random-sampling operations other than the uniform generator."` **[C-MEAS]**

**Family-gated (off-engine on older chips):**
- `sin`, `cos`: native only from A15/M3 (G App. A `apa-trig-math`; G ch. 4 — "family-gated… M1 and A14 reject them and decompose on the host").
- Texture-engine ops `resize`, `resize_bilinear`, `resize_nearest_neighbor`, `upsample_bilinear`, `upsample_nearest_neighbor`: Bridge on M1 (software fallback, different rounding), native from M2/A14. `crop_resize`, `resample`, `affine`: Family-gated, M2/A14 onward, "no host substitution wired" (G App. A `apa-image-texture`).
- Rank/sort bridge: `topk`, `argsort` — Family-gated M1, native from M2; "the validator is callable on the M1 but code generation rejects it" (G App. A `apa-detection-sampling`, note on `topk`). `random_uniform` hardware generator from M3, host random below.
- `reduce_argmin`, `global_argmax`, `global_argmin`: Bridge on M1/M2, native from M3 (G App. A `apa-reduction`).
- `constexpr_blockwise_shift_scale`, `constexpr_sparse_blockwise_shift_scale`: Bridge on M1/M2, native from M3 (G App. A `apa-quant-dtype`).

**Bridge / codegen-rejected on M1 even though validator accepts:** `sort`, `dynamic_slice` validators accept but code generator rejects (G ch. 4 "Attested is not reachable" and ch. 17 end); `range_1d`; `pixel_buffer_to_tensor` (true four-character-code input) — "the program does not compile on the direct route" (G ch. 19 "Direct four-character-code image input"). **[S-SRC]**

**Gather envelope (the most important one for transformer routing):** M1 software path is valid *only* for data batch=1, data depth=1, index channel=3, index width/depth=1, gather-axes count=3; "Outside it the compiler aborts rather than falling back." `gather_along_axis` same; `gather_nd` is Bridge on M1, native texture path from M2 (G App. A `apa-data-movement`; G ch. 4; G ch. 17 validator table "Gather"). **[C-MEAS]**

### 1.3 Apple-documented fallback example

Apple's own reference model: *"Note that 4 of the 606 operations in the optimized model are executed on the CPU. These are the embedding lookup related operations and they are more efficient to do on the CPU for this particular model configuration."* — **AREF**, README "Tutorial: Optimized Deployment of Hugging Face distilbert" (final note). Apple's research article adds: Core ML "blends CPU, GPU, and ANE… to create the most effective hybrid execution plan… even if the entire execution cannot take place there due to idiosyncrasies of different implementations." — **AML**, "The Apple Neural Engine" section. **[A-DOC]**

### 1.4 Private-API-only fallbacks (different path, not Core ML)

`_ANEInMemoryModel` MIL path: **no reduction ops** — `reduce_mean`, `reduce_sum`, `rsqrt` "Fails" → RMSNorm cannot run on ANE, forcing 2 CPU↔ANE round-trips per layer (MO README §"4. No Reduction Ops"; README "What Works on ANE" table rows `reduce_mean/reduce_sum`, `rsqrt`). SDPA hardware ignores `attn_mask`, so causal attention must decompose into Q@Kᵀ (ANE) → mask+softmax (CPU) → scores@V (ANE) (MO README §"What Works" notes; MR "Limitations"; M3 "What runs on ANE, what doesn't"). Many element-wise ops still fall back to CPU in maderix training (MR "On The Hype"; M3). **[C-MEAS / S-SRC]**

**Rule to carry forward:** "unsupported layer" is not a single list — it is (a) path-dependent (public Core ML vs private `_ANEInMemoryModel`), (b) generation-dependent (M1 vs M2/M3+ gates), and (c) sometimes a *validator accepts / codegen rejects* split. Always confirm by compile-and-run on the target before relying on an op.

---

## 2. Precision

### 2.1 FP16 datapath and FP32-class accumulator

| Rule | Source / section | Class |
| --- | --- | --- |
| "The multiply array is fp16 end to end… Inputs are fp16, weights are fp16, outputs are fp16." Frontend accepts fp32/int32/bf16 **as type annotations only**; backend never implements them as wider arithmetic. Bias may be fp32 but is added in the fp16 datapath. | **G** ch. 3 "fp16 datapath" | [C-MEAS] |
| "The reduction in between is held in a wide register of fp32 class on every ANE device"; fixed hardware property. Test evidence: reduction of 16,000 ones is bit-exact; `[4096]+[1]×1024 → 5116` (naive fp16 → 4096, exact → 5120); four-lane first-stage tile proven by cancellation sweep. | **G** ch. 3 "Wide accumulator", Table `tbl:c3-cancel-sweep` | [C-MEAS] |
| fp16 with "wide accumulator" was already community-inferred from hollance: "The ANE appears to use float16 for everything." | **H** `16-bit.md` | [C-ANEC] |
| CPU uses float32 for all calculations and intermediates (from iOS 14/macOS 11 also fp16 on CPU). GPU uses fp16 weights/intermediates but fp32 calculations; `allowLowPrecisionAccumulationOnGPU` in `MLModelConfiguration` makes GPU calculate in fp16 ("a bit faster but you may lose precision"). | **H** `16-bit.md` | [C-ANEC] |
| "It is not sub-splittable from user space… because the firmware per-task-descriptor latency profiler is gated" — relevant to measuring accumulation internally. | **G** ch. 2 "What one dispatch costs" | [S-SRC] |
| Peak marketed throughput is stated in FP16 by Apple: A11 = 0.6 TFLOPS FP16, A15 16-core = 15.8 TFLOPS ("26 times the processing power"). | **AML** "The Apple Neural Engine", Figure 1 | [A-DOC] |

### 2.2 INT8 / INT4 / palettization — Core ML vs private API

**Apple-documented (public Core ML / coremltools):**
- "Core ML supports INT4 and INT8 quantization options for weights and INT8 for activations." — **CML** `opt-overview.md` line 69. **[A-DOC]**
- Quantization math is affine: `w_unquantized = scale*(w_quantized - zero_point)`; `w_quantized = clip(round(w_unquantized/scale)+zero_point)`. "For quantizating weights, 8-bit as well as 4-bit precision is supported. For activation quantizations, 8-bit is supported." Symmetric (`zero_point=0`) is coremltools default. Granularities: `per_tensor`, `per_channel`, `per_block`. — **CML** `opt-quantization-overview.md` "How Quantization Works", "Quantization precision", "Symmetric Quantization", "Quantization Granularity". **[A-DOC]**
- "On newer hardware with A17 Pro or M4 chips… quantizing both activations and weight to `int8` can leverage optimized compute on the Neural Engine." — **CML** `opt-quantization-overview.md` line 59; `opt-quantization-perf.md` lines 13–16 (W8A8 ResNet50 faster on iPhone 15 Pro/A17 Pro than W16A16). **[A-DOC]**
- Recommendation: "Weight palettization (all bits from 1 to 8) typically works the best on the Neural Engine for runtime memory and latency gains"; "W8A8 mode can lead to considerable latency benefits on the Neural Engine by leveraging the faster int8-int8 compute path supported in newer hardware (A17 pro, M4)"; "INT4 `per-block` quantization… can work really well for models using the GPU on a Mac"; "In most cases, you do not lose much accuracy with 6 or 8 bits of palettization or 8 bits of weight-only quantization." — **CML** `opt-overview.md` lines 100–114 (as of iOS18/macOS15). **[A-DOC]**
- Palettization bits `N={1,2,3,4,6,8}`; `per_grouped_channel` from iOS18/macOS15; per-channel scale; vector palettization via `cluster_dim`; 8-bit LUT from iOS18/macOS15; palettized `mlprogram` weights require iOS16/macOS13/watchOS9/tvOS16+. — **CML** `opt-palettization-overview.md` "Palettization Overview", "Granularity", "Per-channel scale", "Vector Palettization", "Quantizing the LUT", "Feature Availability". **[A-DOC]**
- "this latency advantage is available only when palettized weights are loaded and are decompressed 'just in time' for computation. Starting with `iOS17/macOS14`, this is more likely to happen for models running primarily on the Neural Engine backend." — **CML** `opt-palettization-perf.md` lines 4–7. **[A-DOC]**
- Caution: activation quantization "may observe a considerable slowdown… for the compute units (CPU and sometimes GPU) that employ load-time weight decompression… Therefore, it is recommended to use activation quantization only when your model is fully or mostly running on the Neural Engine (NE)." — **CML** `opt-quantization-perf.md` lines 7–11. **[A-DOC]**

**Community-measured / decompiled (direct private path):**
- "The datapath reconstructs compressed weights to fp16 before they reach the multiplier; it dequantizes an int8, int4, or palettized weight to fp16 on the way in, and the multiply that follows is the same fp16 multiply as for an uncompressed weight." Affine `w=s(q-z)`, M1 symmetric `z=0`. — **G** ch. 3 "fp16 datapath", ch. 7 "Compression forms". **[C-MEAS]**
- Whether a compressed form *streams* (bandwidth gain) or *folds* to dense fp16 in DRAM (no bandwidth gain) is a per-chip hardware decision, not a frontend choice: M1 streams int4 LUT (2.37×) and structured sparsity (1.55–1.64× at 0.43× bytes), folds int8 and blockwise; from A14/M2 int8 streams (0.52× fp16 latency at 8192-wide weight); A15 streams blockwise; M5 streams all four (1.6–1.8×). — **G** ch. 7 "What streams on each family", Tables `tbl:c7-streams`, `tbl:c7-hal-bytes`, `tbl:c7-int8-matmul`. **[C-MEAS/S-SRC]**
- "int8 affine and blockwise forms fold to dense fp16 before the data-movement step. The fold expands the int8 weight to a dense fp16 constant in DRAM before the multiply, so the bytes that cross the DRAM boundary are full-width fp16 and the layer gets no bandwidth gain on the M1." — **G** ch. 3 "Compressed weights and the streaming gate". **[C-MEAS]**
- hollance's older reading: "Core ML has support for quantizing weights using 8-bit or smaller integers. However, it appears this is just for storing the weights in the mlmodel file… there is no evidence to suggest the ANE currently does this. As of iOS 14 and macOS 11, Core ML does support 8-bit operations but only for the matrix multiplication layers, not for convolution layers or other operations. It's unknown whether such 8-bit operations are actually performed on the ANE or only on the CPU." — **H** `16-bit.md` → "Quantized operations". **[C-ANEC]** Superseded for current chips by Apple's W8A8 statement above — but note the disagreement in §8.
- m0at (M5, private API): "INT8, INT4, palettized/LUT — all fail to compile through `_ANEInMemoryModel`. Every combination tested: `constexpr_affine_dequantize` → COMPILE FAIL; `constexpr_lut_to_dense` → COMPILE FAIL; `constexpr_blockwise_shift_scale` → COMPILE FAIL. The model must be fully dequantized to fp16… CoreML quantized models somehow run on ANE — the internal compiler flag or op that enables this is unknown." — **MO** README §"2. No Integer Quantization (via Private API)"; spec 07 §1.4 "Quantization Support — Zero". **[S-SRC]**
- **Updated contrary claim:** maderix README now reports "INT8 W8A8 quantization — 1.88x throughput (M4, H16G)" using MIL `quantize`/`dequantize` between layers and `constexpr_affine_dequantize` weights, with int8 activations halving L2 SRAM bandwidth (18.6→35.1 TOPS for 128× conv 512ch). — **MR** README "Current results" / "How It Works" item 6. **[S-SRC]** This directly contradicts MO and M2 — see §8.

### 2.3 Accuracy consequences and activation range limits

| Limit / effect | Exact rule and number | Source / section | Class |
| --- | --- | --- | --- |
| fp16 ceiling | Max finite magnitude **65504**; value above saturates to infinity silently; "compute datapath has no fp32 margin". | **G** ch. 3 "Saturation hazard"; Table `tbl:c3-constants` | [C-MEAS] |
| MAC output ceiling | Multiply-accumulate output stage saturates at **2¹⁵ = 32768** (largest passing value 32752; 32768 → infinity), on matmul, linear, and any multi-tap convolution, regardless of K; interior partial >32768 overflows even if later cancellation would recover. | **G** ch. 3 "Saturation hazard" | [S-SRC] |
| Width-slice crop gain | Nonzero *last/width-axis* slice begin offset applies fixed ×16 crop-DMA gain: value ≤4094 finite (4094×16=65504), 4095/4096 → inf. Control with zero offset stays finite at 60000. Hazard on width axis only. Source: G ch. 3; G ch. 19 "Slice saturation hazard on the M1"; G ch. 12 "fp16 divergence axes"; G Appendix D glossary `slice ×16 saturation`. | **G** | [C-MEAS] |
| Generation of slice hazard — internal disagreement | G ch. 3 says M1 + A14 affected, "non-saturating route arrives on A15 generation and later"; G ch. 19 says "A14 generation saturates on the same slice; the A15 generation and every part above it take a plain fp16 route"; G ch. 12 says M2 saturates bit-identically to M1 and "**corrects the earlier reading that the non-saturating route arrives at the A14**" — i.e. clean only at M5. See §8. | **G** ch. 3 / 12 / 19 | [S-SRC] |
| Activation-range heuristic (older) | "if your model has activations that are relatively large (`> 1e2`) or relatively small (`< 1e-4`), you will lose precision. Or worse, the model may not actually work correctly at all — i.e. very small numbers become 0." | **H** `16-bit.md` | [C-ANEC] |
| Where precision is lost | "Precision is lost on cancellation-heavy steps. The transformer decoder down-projection… large positive and a large negative contribution nearly cancel… The loss does not come from the accumulator… It comes from the per-product fp16 rounding of the inputs and weights… There is no fp16-safe form for such a step on the engine; the anchor has to be computed wider, on the CPU or the GPU." | **G** ch. 3 "Where precision holds and where it is lost" | [C-MEAS] |
| Down-projection error magnitude | "~3 percent error in fp16 on the engine" over contraction 5632 (enough to flip greedy argmax); large-K matmul relative error grows 1.2e-2 at N=2048 → 6.2e-2 at N=8192 (M5). | **G** ch. 14 "Hybrid decoder placement"; ch. 11 "Where the GPU leads" | [C-MEAS] |
| Quantized-weight accuracy | int8 weight tracks fp16 result at cosine near 1.0, relative error near 0.6–1% vs fp32 (fp16 itself ~0.02–0.02% vs fp32). | **G** ch. 3 / ch. 7 | [C-MEAS] |
| NaN / denormal / rounding behavior | Engine coerces NaN→+Inf at input boundary, never produces NaN; flushes denormals inside MAC (M1; M5 preserves); drops sign of zero before reciprocal; round-half-to-even on fp16 output (M5 returned 2050 where M1 returns 2048 for one tie case); gelu origin bias −0.000543, swish −0.001259. | **G** ch. 3 "Edge behavior a correctness oracle must model", "Determinism" | [S-SRC] |
| Activation LUT accuracy | 33-knot piecewise-linear; worst abs error sigmoid 0.0034, tanh 0.0017, gelu 0.0059; sin/cos/atan seam error 0.04–0.12; softmax subtracts hardware max so `[1000,1,2,3]→[1,0,0,0]`. | **G** ch. 3 "Activation functions", "Edge behavior" | [C-MEAS] |
| FP16 underflow in training | "backward matmuls underflow in fp16; fixed with global loss scaling (256×NLAYERS)". | **MR** README "Limitations"; **M3** "Bug 1: FP16 gradient underflow" | [C-MEAS] |
| Determinism | M1 engine is bit-deterministic across reruns, recompile, fresh process, and batch position; changing graph association order changes ~31% of elements by ~1 ULP (max abs diff ~7.8e-3). | **G** ch. 3 "Determinism" | [C-MEAS] |

---

## 3. Shape / layout constraints

### 3.1 Rank, axes, layout

| Rule | Source / section | Class |
| --- | --- | --- |
| Native layout is 5-D `N,D,C,H,W`; **max tensor rank = 5**; rank-6 rejected at compile (`"tensor rank 6 exceeds the ANE maximum of 5"`). Transforms act only on the last four dimensions. | **G** ch. 4 "What fits"/"compile-legal envelope"; ch. 17 "Shape and rank limits" | [C-MEAS] |
| Apple's transformer format: represent `(B,S,C)` as **`(B,C,1,S)`** — 4D and channels-first; map sequence S to the last axis. Swap every `nn.Linear` for `nn.Conv2d` (1×1 conv) to migrate. | **AML** "Principle 1: Picking the Right Data Format" | [A-DOC] |
| Private-API IOSurface tensor format `[1, C, 1, S]` (4D), channel-first, `data[c*S + s]`; transformer token-first `data[t*dim+d]` requires a full transpose on input **and** output. Wrong layout does **not** crash — cosine similarity to BLAS reference 0.02 (random) vs 0.998 with correct transpose. | **MO** README §"Channel-First Transpose" and §"5. Channel-First Memory Layout"; **MR** "How It Works" item 3 | [C-MEAS] |
| Last axis is **unpacked/contiguous and aligned to 64 bytes** (Apple). If last axis is singleton it is padded to 64 bytes → 32× memory cost in fp16, 64× in 8-bit, reducing L2 residency. | **AML** "Principle 1" | [A-DOC] |
| Three alignment scales on M1: **16-byte DMA-width granule** (`dram_alignment` at HAL `0x1c0`), **256-byte segment alignment** (`0x218`, 16 B at A14+), **16 KB DART page**; buffer strides must be multiples of 16 B. Width axis aligns to 16-byte DMA granule. | **G** ch. 21 "Memory hierarchy", ch. 24 "HAL gates" Table, ch. 28 "DART/IOMMU" Table `tbl:c28-alignment`, ch. 17 "Mode and divisibility rules" | [S-SRC] |
| Matrix-multiply depth must be **1 on both operands** (`"depth > 1 is not supported for MatMult"`); fully-connected/linear input rank must be < 5; conv pinned to 4-D NCHW input. | **G** ch. 4 "Hard limits"/"compile-legal envelope"; ch. 17 validator table (`MatrixMult`, `Linear`) | [C-MEAS] |
| Grouped conv: group count must divide both Cin and Cout (else reject); for speed should divide core count (4 on M1) so groups ∈ {1,2,4}. Channel count should be multiple of interleave factor; non-multiple pads out lanes. | **G** ch. 17 "Mode and divisibility rules", Tables `tbl:c17-design-rules`, `tbl:c17-divisibility` | [C-MEAS] |
| Max-pool, transpose, pad constraints: transpose angles ≤16384 (M1) / 65536 (A16+); pad only height/width axes (no channel/depth); reflect/symmetric pad requires texture engine (M1 unavailable). | **G** ch. 17 validator table (`Transpose`, `Pad`) | [C-MEAS] |

### 3.2 Exact numeric shape limits

Source for this subsection: **G** ch. 4 Tables `tbl:c4-shape-limits`, `tbl:c4-compile-envelope`; ch. 17 Tables `tbl:c17-design-rules`, `tbl:c17-validators`; Appendix A Table `tbl:apa-numeric-limits`. All **[C-MEAS]** except where noted.

| Limit | older | M1/A13 | A14/M2 | A15/M3 | A16+/M5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Max kernel W, default/large | 29 | 29 | 32 | 32 | 32 |
| Max kernel W, fp16 | 13 | 13 | 16 | 16 | 16 |
| Min kernel W, large mode | 16 | 16 | 1 | 1 | 1 |
| Max kernel H | 29 | 29 | 32 | 32 | 32 |
| Max kernel D | 1 | 16 | 16 | 16 | 16 |
| Max tensor W/H | 16384 | 16384 | 16384 | 16384 | **65536** |
| Max tensor D | 1 | 16384 | 16384 | 16384 | **65536** |
| Max tensor C | 65536 | 65536 | 65536 | 65536 | 65536 |
| Max tensor N (batch) | 4096 | 65536 | 65536 | 65536 | 65536 |
| Reduction→transpose threshold | none | 192 | 192 | 384 | 384 |
| Arg-min/arg-max reduction axis | — | 2048 (2049 rejects) | 2048 | 2048 | 2048 |
| Kernel-memory budget | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB |
| Activation-LUT budget | 150 B | 86 B | 86 B | 86 B | 86 B |
| Context-switch live-tensor limit | 2 | 2 | ∞ | ∞ | ∞ |
| Instruction/segment alignment | 256 B | 256 B | 16 B | 16 B | 16 B |
| Has texture engine | no | no | yes | yes | yes |
| DMA width granule | 16 B | 16 B | 16 B | 16 B | 16 B |

Compile-legal envelope quirks on M1: any axis 16384 accepts, 16385 rejects; conv `Cin`/`Cout`=16385 **accept**; zero-size dim `(0,8)` rejected on-device as type mismatch; input dtypes only `fp16` or `uint8` (fp32/int32/bf16/int8 rejected by frontend); conv kernel width >15 rejected by frontend, fp16 datapath caps at 13; kernel **height** unconstrained (16×3 compiles, 3×16 does not); divisible group count required. **[C-MEAS]**

### 3.3 Minimum sequence length, outputs per program, IOSurface minimum

These come from the M5 private-API fork (single-source; not Apple-documented, not in ane-guide). All **[S-SRC]**:

| Constraint | Exact rule | Citation |
| --- | --- | --- |
| Min sequence length | "ANE has a minimum sequence length of **S = 16 (fp32 I/O) or S = 32 (fp16 I/O)**. Below this, outputs are silently all zeros." Compile-time tile default 64; input zero-padded, CPU ops process only real tokens. | **MO** README §"ANE Tile Size" line 287; §"10. Minimum Sequence Length" line 388; table line 492; **MO** `docs/APPLE_ANE_OPEN_LETTER.md` line 121 |
| Max outputs per program | "ANE supports exactly **2 outputs per program**. Programs with 3+ outputs silently return correct data only for the last output" / "3+ outputs silently corrupt all outputs except the last one." Therefore K+V fused, Q separate. | **MO** README line 211 and §"6. Maximum 2 Outputs Per Program" line 368; `docs/ANE_FUSED_KV.md` lines 5, 116 |
| IOSurface minimum | "The smallest IOSurface ANE will accept is **49,152 bytes** — even for a single scalar output." Format always `[1,C,1,W]` channel-first fp16 (or fp32 with cast); width padded to 32-element minimum. | **MO** README §"11. IOSurface Minimum Size" line 392 and table line 493; spec 07 §1.5 "IOSurface Constraints" lines 151–156 |
| IO count (working config) | "2 inputs per program — Works (conv(baked,input0)+add(result,input1))"; multi-input ANE requests otherwise cause 0x1d error, so inputs are packed into the spatial dimension. | **MO** README "What Works" table line 531; **MR** README "Limitations" "Single-input constraint" |
| Static shapes only | Direct path compiles one program per concrete shape; symbolic/dynamic dimensions parse but fail to compile. "For a variable-length workload the options are to pad to a fixed maximum length and compile once, to bucket a small set of lengths… or to recompile per length." | **G** ch. 19 "Symbolic shapes are not reachable on the direct path" **[S-SRC]** |

### 3.4 SRAM size and the "cliff" — two incompatible numbers

| Claim | Exact statement | Source | Class |
| --- | --- | --- | --- |
| **~2 MB on-chip working set (M1)** | "The primary design limit is not the 141 FLOP-per-byte crossing but the on-chip working set, 2 MB on the M1. While a layer's activation fits in the 2 MB on-chip memory… Once the activation exceeds 2 MB, every layer streams its full activation in and out of DRAM." Confirmed from byte counters: at activation exactly 2 MB, a 96-layer matmul chain moved 426 MB DRAM/dispatch, intensity fell to 60 FLOP/byte, throughput 12→4.8 TFLOP/s. M5 = 4.72 MB. HAL field `max_operand_bytes` `0x1b8` = 2 MB (1 MB on M9). Also "2 MB stores about 1.05×10⁶ fp16 elements, so a square activation of side near 1024 is at the edge." | **G** ch. 9 "On-chip working-set threshold"; ch. 21; ch. 24 HAL table; ch. 17 "Working-set rule"; Appendix A `matmul SRAM working set` | [C-MEAS] |
| **~32 MB SRAM (M4/M5, private path)** | "The cliff between 24 MB (fast) and 96 MB (slow) places the ANE's on-chip SRAM at roughly ~32 MB. The transition isn't a hard wall — performance degrades gradually, suggesting a cache-like hierarchy rather than a hard scratchpad." 2048² matmul (24 MB working set) 5.7 TFLOPS; 4096² (96 MB) 4.0 TFLOPS, ~30% drop. M5 measured spec "On-chip SRAM ~32 MB — `sram_probe.m` — perf cliff at 32 MB working set". "Stay under 32 MB. Keep per-tensor footprint in SRAM. Spilling to DRAM kills throughput." | **M2** "The SRAM Cliff", "Rules for maximizing ANE throughput"; **MR** README; **MO** README §"ANE Hardware Reference" | [C-MEAS/S-SRC] |
| Reconciliation hypothesis | G's 2 MB is the *compiler HAL operand tile / residency working set* (field `0x1b8`) on M1; maderix's 32 MB is the *aggregate on-chip buffer pool* inferred from a 2048→4096 matmul cliff on M4/M5. They describe different layers and different chips; both can be true. This is not proven; see §8. | Analysis in this catalog | — |
| SRAM banking layout | On-chip pool interleaved across **64 banks at 16-byte granule**, `bank = floor(addr/16) mod 64`; throughput period 64 bytes (32 fp16 elements); stride optimizer keeps strides ≤2 MB ceiling (`0x1f8`). | **G** ch. 21 "Memory hierarchy", ch. 28 | [S-SRC] |

---

## 4. Dispatch / compile behavior

### 4.1 Dispatch latency — public Core ML vs private path

| Path | Number | Source | Class |
| --- | --- | --- | --- |
| Private `_ANEClient` round-trip (M4/M5) | **~60–80 µs** per dispatch (CPU→ANE→CPU); 7 matmuls × 32 layers × 70 µs = 15.7 ms overhead for S=1 decode. | **MO** README §"8. Dispatch Overhead" line 376; spec 07 §1.3 | [C-MEAS] |
| Mega-kernel fusion note | "~160 µs XPC overhead per dispatch" when fusing full transformer layers; 12-layer mega-kernel 5081 µs vs 15227 µs for 24 separate evals (3–4× forward speedup). | **M3** "What's next" | [S-SRC] |
| Core ML overhead tax | "For small operations, CoreML adds **2–4× overhead**. The gap narrows at high-throughput configurations… for latency-sensitive workloads (LLM token decoding, real-time inference), the CoreML tax is severe." | **M2** "CoreML vs _ANEClient: The Overhead Tax" | [C-MEAS] |
| M1 firmware-level floor (private/direct path, read-only trace) | ~**190 µs** per tiny dispatch, ~98% overhead. Budget: user-space binding+copies ~25 µs; build firmware request ~16 µs; doorbell ~2–3 µs; **firmware round trip ~130 µs**; kernel completion ~10 µs. `ANE_ProgramSendRequest` external method ~163 µs. Two interrupts per inference. | **G** ch. 2 "What one dispatch costs", Table `tbl:exec-budget` | [C-MEAS] |
| M1 roofline dispatch floor | **0.23 ms** per eval (relu/sigmoid/avgpool/small conv all 0.23–0.26 ms; 64-element linear 0.23 ms). Fused chain floor ~0.19 ms (of which ~0.13 ms firmware round trip). Kernel submission ~0.163 ms. | **G** ch. 9 "Dispatch floor"; ch. 11 "Dispatch floor and conv throughput"; Table `tbl:roofline` | [C-MEAS] |
| M4 private path (XPC+IOKit) | 256×256 matmul at 0.101 ms, "most of the time is XPC + IOKit overhead (~0.095 ms). The actual compute is only ~0.006 ms." | **M2** "Matmul Scaling: The Basics" | [C-MEAS] |
| Concurrent submissions | M1 driver keeps at most one firmware command in flight; two concurrent submission threads serialize (measured 1.04×). Overlapping ≥2 streams in one process is "the unfinished path on the M1": second stream's completion event never fires and its waiter blocks. | **G** ch. 2 "Submissions serialize at one in flight", "Host drives an autonomous coprocessor" | [S-SRC] |
| Core ML cannot be forced | `computeUnits = .all` is a *preference*, not a guarantee; ANE used "whenever possible" but fallback on any unsupported layer or system load. | **H** `running-on-ane.md`; **H** `other.md` | [C-ANEC] |

### 4.2 Compile time, cache, limits

| Rule | Exact statement | Source | Class |
| --- | --- | --- | --- |
| First compile (private path) | "First compile takes ~20–40ms. Cache hits are effectively free." / "Compile time per program 30–50 ms (measured during 128-program compile)"; 10 dynamic kernels at startup ≈0.4 s. | **M1** "Compilation Cache"; **MO** README table line 489; **M3** "Pipeline 3" | [C-MEAS] |
| 128-program cold compile | ~80 s for Mistral 7B (includes dequant Q4→fp16, SHA256 hash, tmpDir symlink, compile); warm start ~220 ms by forging `hexStringIdentifier` from daemon cache (~400× faster init). | **MO** README §"2-Phase Loading" lines 252–283 | [S-SRC] |
| Compile cost scales with batch | Backward convolution graph on M1: ~1.9 s at batch 4, ~35 s at batch 64, ~79 s at batch 128, never finishes at batch 1000. Wide reconvergent fan-in plateaus near 9.2 s/cluster (memoized cut search exponential in branch width; internal time budget bails). | **G** ch. 19 "A single slow compile is a separate failure mode" | [S-SRC] |
| Failed-compile backoff | Failed compile restarts the shared compile service; wait ≥1 restart interval (~15 s) before next compile, else unrelated compiles stall. No hard failure-count cap needed. | **G** ch. 19 "Pacing compiles after a failure" | [S-SRC] |
| On-disk program cache (ANE compiler service) | Cache tree `cache/com.apple.e5rt.e5bundlecache/<os-build>/<hash>/`, keyed by `model.anehash` (double SHA-256 over the program); content keyed so identical graphs+weights+options hit cache; `force_recompilation` defeats warm start. Compiler service writes under `rootless.storage.ane_model_cache`. | **G** ch. 5 "Compiler is its own gated service", "Runtime surface and its tunable dictionary" | [S-SRC] |
| On-disk program cache (maderix path) | `~/Library/Caches/<app>/com.apple.e5rt.e5bundlecache/<build>/<hash>/<hash>.bundle/H16G.bundle/H16G.e5` (+ `main/main_ane/model.anehash`); E5 binary 2–3 KB regardless of matrix size (1024² matmul = 2688 B; 128² = 2680 B). | **M1** "The E5 Binary", "Compilation Cache" | [C-MEAS] |
| Daemon cache volatility | "The `aned` daemon caches compiled programs in memory, keyed by `SHA256(MIL + weights + options)`. This cache survives process restarts but is **lost on reboot**." Cold start "survives restarts, lost on reboot." | **MO** README §"12. Daemon Cache Volatility" line 396; §"2-Phase Loading" line 278–283 | [S-SRC] |
| Compile limit per process | "ANE compilation leaks resources. After **~119 compilations in a single process**, subsequent compiles fail. Workaround: `exec()` restart… or the 2-phase forge strategy." Consistent M4/M5; upper bound ~119 kernels/process. | **MO** README §"7. Compile Limit (~119 Programs)" line 372; spec 07 §1.1 line 39; **MR** "exec() restart — bypasses ~119 ANE compile limit"; **M3** "Pipeline 1" | [C-MEAS/S-SRC] |
| Loaded-program cap (different cap) | "The loaded-program cap is near **128 programs per process**: the next load fails with `GetANEFModel: must re-compile` and forces a recompile." In-flight cap **127** outstanding requests per program (dispatch semaphore inside the in-memory model); driver in-flight count capped at 127 per request. | **G** ch. 14 "Two serving controls and two hard caps", Table `tbl:c14-reference`; ch. 28 (in-flight 127) | [S-SRC] |
| Queue depth | M4 ANE "queue depth of 127 evaluation requests"; `loadModel` → "programHandle assigned, queueDepth = 127". | **M1** "What is the Neural Engine?", "_ANEClient" | [C-MEAS] |
| Weight patching without recompile | A host can patch a compiled program's weights in place: conv weight in `0xC0`-stride layout, matmul in `0x40`-stride; driver path `ANEScheduler::pendingRequestsPerPatchMutableBuffer`. (Contrast maderix "runtime weight override is dead" — see §8.) | **G** ch. 7 "Patching weights in a compiled program" | [S-SRC] |
| Multifunction model weight duplication | coremltools `save_multifunction`: merging models that share weights "deduplicates shared weights by calculating the hash of the weight values" — one `.mlpackage` asset, multiple independently invokable functions, common base weights shared. On the private ANE path each baked-weight program instead carries its own fp16 copy (Mistral 7B 128 programs). | **CML** `multifunction-models.md` lines 8–31; **MO** README §"Fitting a 7B Model in 24 GB" | [A-DOC] + [S-SRC] |
| Compile is out-of-process and gated | Compiler is separate XPC service `com.apple.ANECompilerService`; the daemon `aned` holds `com.apple.ane.iokit-user-access`; developers reach the engine through the daemon. Programs are compiled for the oldest target generation needed (`TargetArchitecture=h13`) and run on all later generations. | **G** ch. 5 "Access-control model", "Compiler is its own gated service"; ch. 12 "Compiling for a target generation" | [S-SRC] |

---

## 5. Verification — every documented way to check whether ops actually ran on ANE (and caveats)

### 5.1 hollance methods (black-box, Core ML) — H `is-model-using-ane.md` + `os-log.md`

All **[C-ANEC]**.

| Method | Exact procedure / signal | Caveat |
| --- | --- | --- |
| Thread name | Pause debugger; a thread named **`H11ANEServicesThread`** means Core ML is using the Neural Engine "for at least some portion of your model". | Does **not** prove the whole model ran on ANE; Core ML can split across devices in one inference. |
| A/B compute units | Change `MLModelConfiguration.computeUnits` from `.all` to `.cpuAndGPU` / `.cpuOnly`. If `.all` isn't much faster, ANE may be partial or unused. | Indirect; CPU can be faster than GPU for some ops (H `why-care.md`), so a null result is ambiguous; system load affects placement. |
| Breakpoint `-[_ANEModel program]` | Symbolic breakpoint (exact spelling with leading dash). If hit, Core ML is using ANE. | Only confirms *some* portion; set other breakpoints to be sure. |
| Engine-specific breakpoints | ANE = `Espresso::ANERuntimeEngine`; GPU = `Espresso::MPSEngine`, `Espresso::MetalLowmemEngine`; CPU = `Espresso::BNNSEngine`. GPU breakpoints: `Espresso::MPSEngine::context::__launch_kernel`; CPU: `Espresso::BNNSEngine::convolution_kernel::__launch`, `Espresso::elementwise_kernel_cpu::__launch`. Fallbacks: `Espresso::layer::__launch`, `Espresso::net::__forward`. `-[_ANEClient evaluateWithModel...]` measures ANE time. | "The symbols mentioned above may change between iOS versions." Find current ones via `image list Espresso` then `image dump symtab <path>` at the lldb prompt. |
| Instruments Time Profiler | New empty app, call model ~100× in loop; Time Profiler → Call Tree → find `-[MLNeuralNetworkEngine predictionFromFeatures:...]` → Focus on subtree; inspect function names / timings. Option-click expands all. | Heuristic — names are clues, not authoritative placement proof. |
| `sudo powermetrics` (macOS) | ANE Power line e.g. `ANE Power: 0 mW`; combined CPU+GPU+ANE power. | Whole-package sample; nonzero ANE power could come from other processes/system tasks; 0 mW can mean idle or unsampled short burst. |
| `os_log` / unified log | Console messages like `[coreml] Error computing NN outputs -1`, `Error plan build: -1`. Use Instruments **os_log** instrument for more detail; `sudo log collect --device --last 1d`; `log show --archive system_logs.logarchive --predicate '(subsystem IN {"com.apple.espresso","com.apple.coreml"}) && (category IN {"espresso","coreml"})' --info --debug --last 1d`; on Mac `log stream` with same predicate. Model/layer names appear as `<private>` unless a device profile is installed. | Error messages "aren't very helpful"; `<private>` redaction; presence of unsupported-layer warnings is suggestive, not a runtime placement trace. |

### 5.2 Apple-documented verification

| Method | Exact statement | Source | Class |
| --- | --- | --- | --- |
| Xcode Core ML Performance Report (Xcode 14+) | "Developers can review runtime statistics, **layer dispatch to each engine**, and accomplish additional tasks." Generate via Performance tab after adding the `.mlpackage` as a resource. | **AML** "Putting It All Together"; **AREF** tutorial "To verify performance…" | [A-DOC] |
| `MLComputePlan` (public API) | `MLComputePlan.load` parses a compiled model and reports the segmenter's choice per operation; `deviceUsage(for:)` returns the set of devices that *could* run an op and the one device the planner *preferred*, with a `Reason`; `estimatedCost(of:)` returns the per-operation cost weight. | **G** ch. 5 "Placement segmenter above the runtime" (describes the Apple API); **APL** `MLComputePlan` reference page | [A-DOC]/[S-SRC] |
| `MLModelBenchmarker` | "`MLModelBenchmarker` utilizes the model's compute plan to estimate the execution time of individual operations within the model" (coremltools debugging utility). | **CML** `mlmodel-debugging-perf-utilities.md` line 315 | [A-DOC] |

**Key caveat (all compute-plan methods):** a compute plan / performance report is a **static/estimated placement** produced by the segmenter, not proof of runtime execution. hollance's team notes directly: "our compute plan showed ANE for INT8 mlprogram ops, but static plans are not proof of runtime placement — keep power-counter verification in any future claim" (existing team note `hollance_repo.md` lines 49–53). **[team analysis]**

### 5.3 2026 private-path verification and its hard gate

Source: **G** ch. 33 "Telemetry and hardware counters". **[S-SRC]** unless noted.

- Readable without entitlement beyond root: free-running firmware timestamp at MMIO `0x2_6b17_8000`; whole-engine IOReport channels `AMC Stats|Perf Counters|ANE0 RD/WR`, `ANE0 DCS RD/WR`, `Energy Model|-|ANE0` (mJ), `PMP|AF BW|ANE0 RD+WR`, `SoC Stats|Cluster Power States|ANE0`, throttle-trigger family `ANE_THROTTLE_{SW,HW,PPT,DITHER,EXT}_TRIG`, interrupt statistics. These are what the guide's roofline/power figures actually use.
- Signpost stream (capturable on engine subsystem, no entitlement beyond root): `_ANEF_MODEL_EVALUATE` → `_ANEF_MODEL_EVAL_DRIVER_REQUEST` ×3 (Cast, AneInference, Cast), `_ANEF_MODEL_EVAL_PERFCOUNTER_SAMPLE`, `_ANEF_MODEL_COMPILE/LOAD/UNLOAD`, `_ANEF_IOSURFACES_MAP`, `_ANEF_INPUT_BUFFERS_READY`, `_ANEF_ENQUEUE_OUTPUT_SET`. Confirms dispatch *structure* only.
- **Blocked:** the 24 per-task-descriptor counters (`kANE_NE_COMPUTE_CYCLES`, `kANE_FP16_CYCLES`, `kANE_INT8_CYCLES`, `kANE_L2_*_STALL_CYCLES`, `kANE_DMA_READ_BYTES`, `kANE_DPE_ENERGY`, etc.). Master enable at `ANEProgramCreateArgs+0x6c` (`statsMask`). Forcing it non-zero makes `create` succeed but `load` return 0 because the compiled program has no stats-descriptor section (`initStatsBufferSection` bails on size 0); `aned` zeroes the mask for `coreAnalyticsClientType=ThirdPartyAppUsingANE`; host check string `perfStatsIOSurface is NULL!`.
- maderix used `powermetrics` ANE/CPU power in the training dashboard (M3). **[C-MEAS]**

**Verification recommendation for our team:** require two independent signals — (1) Apple compute plan *or* Xcode performance report for segment-level placement, plus (2) `powermetrics`/IOReport ANE power or energy delta actually measured during the target workload. Treat a compute plan alone as insufficient.

---

## 6. Interaction with the GPU

### 6.1 Memory-controller / bandwidth numbers (contention context)

Source: **G** ch. 9 "Roofline", ch. 11 "ANE, GPU, and CPU", ch. 33; all decompile/measurement-derived **[C-MEAS/S-SRC]**.

| Channel / number | Value |
| --- | --- |
| ANE DRAM bandwidth ceiling, M1 | ~85 GB/s at memory controller (roofline B); saturating weight-stream wall clock ~51 GB/s (matches compiler's internal 50 GB/s constant); standalone elementwise/activation path only ~24 GB/s; single ReLU ~10 GB/s effective. |
| ANE weight stream, M5 | ~145 GB/s over two DRAM read channels (near 3× M1's 51 GB/s). |
| GPU bandwidth roof, M5 | ~229.7 GB/s; compute roof ~30.9 fp16 TFLOP/s; ridge 134 FLOP/B. |
| ANE bandwidth roof, M5 standalone | ~24.1 GB/s; compute roof 10191 (matmul) / 18771 (conv) GFLOP/s; ridge **424 FLOP/B** (~3× GPU ridge, ~29× CPU ridge 15). |
| Measured DRAM traffic in a 96-layer matmul chain | 17.2 GB reads + 16.9 GB writes over 432 ms → 79–90 GB/s (engine's share); energy ~0.48 pJ/FLOP, ~2.3 W sustained. |
| Cross-process contention side-channel | Competing ANE clients cause a **2.3× latency jump** and a 20–50 bit/s occupancy channel (M1 measured); a concurrent **1 GB/loop CPU memory-copy hog moves victim latency only 4 µs (1.03×)** — i.e. engine-queue serialization, not DRAM controller contention, dominates. Data isolation intact across 9000 concurrent results. Source: **G** ch. 32 "Security", Appendix F provenance line 262. |
| Broker arbitration | One physical engine, one time-shared request queue; daemon down-adjusts a client's requested queue depth under contention and sets QoS per request. Source: **G** ch. 5. |

hollance's older contention anecdote: *"if NPU supports all the layers of the model and the GPU is resting, with a high probability CoreML will run it on GPU. But if GPU is heavy loaded with work — you can expect the model to be run on NPU"* — H `other.md` (citing a Twitter post by Eugene Bokhan), plus "smart compute system" on A12+ that decides CPU/GPU/ANE. **[C-ANEC / single anecdote]**

### 6.2 Zero-copy IOSurface sharing, overlap, device choice

- **Zero-copy GPU↔ANE:** I/O uses IOSurfaces ("the same shared memory mechanism used for GPU textures"). "This means zero-copy transfers between GPU and ANE are theoretically possible if you share the same IOSurfaceRef." Reproduced in maderix repo files `gpu_ane_share.m` and `gpu_prefill_ane_decode.m`; M4 seq=256 pipeline: GPU prefill 6.7 ms (Stories110M) / 9.7 ms (Qwen3-0.6B) + ANE decode 1.9 / 2.3 ms = 8.8 / 12.0 ms total. — **M1** "The IOSurface Protocol"; **MR** "How It Works", "Performance"; team note `maderix_ANE.md`. **[C-MEAS]**
- **Unified memory vs switching cost:** "iPhones, iPads, and Apple Silicon Macs have shared memory… you don't need to upload data from one processor to the other. But that doesn't mean switching between processors has no cost: the data still needs to be converted into a suitable format. For example, on the GPU the data needs to be put into a texture object first." — **H** `ane-vs-gpu.md`. **[C-ANEC]**
- **Can ANE and GPU overlap?** Within one engine, no two on-engine ops overlap (G ch. 22: "two operations on one engine cannot overlap"; only MAC + planar-engine pairs can run concurrently). Across processors, the recommended hybrid is **prefill/encoder/vision on ANE, autoregressive decode on GPU** (G ch. 14 verdict; M2 "ANE vs SME"; MO spec 07 recommends GPU Neural Accelerators/MLX for the whole LLM). Speculative decoding is the one proposed genuinely concurrent ANE+GPU pattern (small draft model on ANE, main model on GPU, 2–3× if agreement >90%) — **MO** spec 07 §4.1. **[S-SRC]**
- **Device selection costs:** Core ML's segmenter solves a shortest path over per-op/per-backend nodes using learned regression decision trees; cost = `max(flops/gflops, bytes/bw) + launch + transfer`. "The transfer cost is why the minimum-cost solution favors long single-backend runs: each boundary charges a tensor repack between the engine channel-interleaved fp16 layout and the host layout." — **G** ch. 5 "Placement segmenter". **[S-SRC]**
- **Throughput/energy crossover (single stream, M5/M1 measurements):** ANE leads conv stack (~4.2× faster, ~13× energy efficient on M5; ~2×/~14.5× on M1), ResNet-18 (~6.1×/~11×), short-sequence attention S=197; GPU leads large square matmul (~2.8–3.0×; 30.9 vs 10.2 fp16 TFLOP/s), long attention S=512 (~2.1× faster but ANE keeps 1.8× energy edge), all autoregressive decode (batch 16: GPU 2.7× faster, 4.6× more energy efficient). Serving crossovers: encoder block near batch 23, self-attention near batch 6. CPU wins below the dispatch floor. — **G** ch. 11 Tables `tbl:c11-m1class`, `tbl:c11-serving`, `tbl:c11-verdict`. **[C-MEAS]**
- **Core ML cannot be forced** to ANE; `.all` is a hint; system load can move work to GPU (see §5.1, §6.1). **[C-ANEC]**

---

## 7. Multi-ANE / Ultra, and M5 "Neural Accelerators" vs ANE

### 7.1 Multi-die / Ultra

| Rule | Source | Class |
| --- | --- | --- |
| "The Neural Engine is identical across base/Pro/Max variants of the same generation — Pro and Max add GPU cores and memory bandwidth but do not multiply the ANE. The Ultra is a notable exception: it fuses two dies and gets **two ANEs**." | **MO** README §"ANE Across Apple Silicon" line 434 | [S-SRC] |
| M3 Ultra: 2×16-core ANEs = 32 cores, 36 TOPS, 819 GB/s. "In theory, 2× the ANE compute. In practice, there is **no public API to dispatch across both ANEs simultaneously**. CoreML may handle this internally, but `_ANEInMemoryModel` likely targets a single ANE." | **MO** README table line 443 and commentary line 451 | [S-SRC] |
| hollance device table: M1 Ultra 32-core ANE / 22 TOPS (2022); M2 Ultra 32-core / 31.6 TOPS (2023); M3 Ultra 32-core / 36 TOPS (2025); A20 Pro "dual 16-core Neural Engine, for a total of 32 cores" (2026). | **H** `supported-devices.md` | [C-ANEC] |
| Compiler decode: kernel load-balancer steers whole independent submissions to the least-busy engine die and **never exchanges tensor data** between dies; driver supports up to **four dies**. "The M1 and the M1 Max each register a single engine die, so this steering engages only on a multi-die part such as the Ultra." | **G** ch. 36 "Multi-die collective layer" | [S-SRC] |
| A true cross-die collective dialect `mlir::silc` exists (`silc.all_reduce`, `all_gather`, `all_slice`, `mesh`, `call`; HAL collective-enable byte `0x48b` reads **zero on all 28 targets**; all CCDMA register setters are stubs asserting "CCDMA is not supported for this arch"). No part in the guide can run it — "decoded but inert". | **G** ch. 36, Tables `tbl:c36-collops`, `tbl:c36-capbytes` | [S-SRC] |
| 64-core ceiling: compiler emits a 64-core cost model for `H17d` from `ne_core_count` alone (thresholds 7/10/20/40 bucket die sizes 8/16/32/64 via suffixes g/s/c/d); no measured 64-core silicon. | **G** ch. 36 "64-core ceiling" | [S-SRC] |

**Implication for Veystrix:** only the load-balancer style die steering is decoded; there is no usable public or private single-program two-die ANE path in these sources. Treat "2× ANE on Ultra" as unavailable to our code.

### 7.2 M5 GPU "Neural Accelerators" ≠ ANE

| Rule | Source | Class |
| --- | --- | --- |
| "The M5 uses a faster 16-core Neural Engine. **This is the first M-series chip to include Neural Accelerators in the GPU cores.**" A19 is "the first Apple Silicon SoC to include Neural Accelerators in the GPU cores." A20 Pro: "dual 16-core Neural Engine… The CPU and GPU also introduces Neural Accelerators." | **H** `supported-devices.md` (M5, A19, A20 Pro entries) | [C-ANEC] |
| "M5 has **10 GPU cores**, each with embedded **Neural Accelerator blocks**. These are NOT the standalone ANE (which is a single shared resource). Neural Accelerators are directly accessible via **Metal 4 TensorOps API**." Unlike ANE: no baked weights (MTLBuffer), runtime weight loading, scales with GPU core count. "This is where Apple invested the 4× compute gain in M5 over M4." | **MO** `docs/specs/07-ane-vs-gpu-neural-accelerators.md` §2.1–2.2 | [S-SRC] |
| "The M5 GPU has new 'Neural Accelerators' inside each GPU core (Apple's '133 TOPS' figure includes both ANE + GPU NAs), but these are accessed through Metal, not `_ANEInMemoryModel`." | **MO** README line 449 | [S-SRC] |
| M5 ANE itself: 16 cores, ~19 fp16 TFLOPS / ~38 INT8 TOPS, 153 GB/s bandwidth, 24 GB max — **identical ANE to M4 Max**; Max's advantage is memory bandwidth/RAM, not ANE. | **MO** README tables lines 438–447 | [S-SRC] |
| GPU NAs via Metal 4 TensorOps / MLX: runtime weights, INT8/INT4 supported, ~2–5 µs dispatch vs ANE 60–80 µs, no per-kernel compile limit. Decode 25–30 tok/s (MLX) vs CPU NEON 17.2 tok/s vs ANE infeasible. | **MO** spec 07 §2.3–2.4 | [S-SRC] |
| "ANE is the wrong accelerator for Mistral 7B transformer inference… use Metal 4 GPU Neural Accelerators (TensorOps) or MLX framework." | **MO** spec 07 "Executive Summary" | [S-SRC] |

**Clarification for the team:** "M5 Neural Accelerators" is a **GPU** feature reached through Metal/MLX; it is not the ANE and not reachable through `_ANEInMemoryModel`. Any capacity/TOPS comparison that merges the two (e.g. Apple's 133 TOPS figure) must be decomposed before use.

### 7.3 First-generation caveat

Hollance: A11's 2-core Neural Engine (iPhone 8/8+/X) "cannot be used by Core ML. It's only used for tasks such as Face ID and Animoji." A12 is first Core-ML-usable ANE (iOS 12+). — **H** `supported-devices.md`. Independent comment from Maynard Handley (Hacker News on M1) argues the A11 unit was "something like a Lattice FPGA… vastly different from the next model which is the start of the 'real' ANE lineage," so beginning capability tables at A11 is misleading. **[C-ANEC/community comment]** See §8.

---

## 8. CONTRADICTIONS BETWEEN SOURCES

Each item states **both** claims verbatim with citations, then a reconciliation where one is evident. Do not treat a single side as settled without testing on our hardware.

1. **INT8/INT4/LUT through the private path: works vs impossible.**
   - (a) "INT8, INT4, palettized/LUT — all fail to compile through `_ANEInMemoryModel`… The model **must** be fully dequantized to fp16… CoreML quantized models somehow run on ANE — the internal compiler flag or op that enables this is unknown." — **MO** README §2; spec 07 §1.4 ("All quantization formats FAIL to compile on ANE", error "unsupported type"). Also M1 Part 1 hardware profile lists precision fp16 only; M2 states "There is no 2× INT8 speedup. INT8 and FP16 deliver nearly identical throughput. The ANE dequantizes INT8 weights to FP16 before compute."
   - (b) "INT8 W8A8 quantization — **1.88× throughput (M4, H16G)**… 18.6 TOPS→35.1 TOPS… INT8 activations halve L2 SRAM bandwidth between tiles via MIL `quantize`/`dequantize` ops. Weights use `constexpr_affine_dequantize` (int8 stored, fp16 at compile time)." — **MR** README (updated) "Current results" and "How It Works" item 6.
   - (c) Apple/CML: W8A8 is real and uses "faster int8-int8 compute path supported in newer hardware (A17 pro, M4)" — **CML** `opt-overview.md` line 102, `opt-quantization-overview.md` line 59. G decompiles a "double-int8 mode that packs two int8 products into one MAC" (1.4–2× fp16 rate, ch. 9) while simultaneously saying compressed int8 weights *fold to fp16* and add no bandwidth gain on M1 (ch. 3, 7).
   - **Reconciliation:** (a) and (b) are likely different MIL op sequences/compiler versions (maderix updated code; MO may predate the `quantize`/`dequantize` trick). (c) distinguishes *compute* int8 (A17/M4+, real) from *stored/streamed* int8 weights. Our team's earlier finding — "compute plan showed ANE for INT8 mlprogram ops, but static plans are not proof of runtime placement" — remains the right caution.

2. **"ANE is 16-bit / fp16 only" vs fp32 accumulation and 8-bit MAC.**
   - (a) "The ANE appears to use float16 for everything" — **H** `16-bit.md`; "Precision fp16 only" — **MO** table.
   - (b) "fp16 end to end with a wide accumulator of fp32 class… representable sums come back near exact" — **G** ch. 3; Apple markets FP16 peak throughput — **AML**.
   - **Reconciliation:** operands/output fp16, accumulator wider; "16-bit" is a storage/operand statement, not accumulator. Both are correct at different levels.

3. **On-chip SRAM: 2 MB vs ~32 MB.**
   - (a) 2 MB operand working set, HAL `0x1b8`, 2.28–2.34 MB measured threshold, M5 4.72 MB — **G** ch. 9/17/21/24/Appendix A.
   - (b) ~32 MB, cliff between 24 MB and 96 MB working sets on M4/M5 — **M2** "The SRAM Cliff"; **MO** README table line 487.
   - **Reconciliation:** different chips and different abstractions (compiler tile/residency vs aggregate on-chip pool). Unresolved; must measure on our target.

4. **Dispatch floor / overhead: 60–80 µs vs 0.095 ms vs 0.19–0.23 ms.**
   - **MO/MR** 60–80 µs; **M2** ~0.095 ms XPC+IOKit for 256² matmul; **G** ~0.19 ms full-call with 0.13 ms firmware round trip (M1), 0.23 ms roofline floor (M1); **M3** ~160 µs XPC overhead for mega-kernel fusion.
   - **Reconciliation:** chip generation (M1 vs M4/M5), path (private `_ANEClient` vs full e5rt stack), and what is counted (XPC+IOKit only vs full host+firmware round trip). Not a single constant.

5. **Does INT8 double throughput?**
   - (a) "There is no 2× INT8 speedup. INT8 and FP16 deliver nearly identical throughput… Apple's '38 TOPS INT8' is computed as 19 TFLOPS FP16 × 2, following the industry convention" — **M2** "INT8 = FP16: The '38 TOPS' Reality"; **MO** spec 07 §1.4.
   - (b) "The multiply array has a double-int8 mode that packs two int8 products into one MAC, so int8 arithmetic runs at about twice the fp16 rate… measured int8 convolution ~1.4× faster, trends toward 2× at scale" — **G** ch. 9; Apple coremltools W8A8 gains — **CML**.
   - (c) hollance: no evidence ANE does 8-bit ops directly; unknown whether 8-bit matmul runs on ANE at all — **H** `16-bit.md`.
   - **Reconciliation:** (a) concerns dequantized-to-fp16 stored int8 weights on the M4 private path; (b) concerns genuine int8 MAC mode available in Core ML on A17/M4+; (c) is pre-2020 and superseded for current chips.

6. **Compile cache location/lifetime.**
   - (a) On-disk content-addressed compiler cache, `cache/com.apple.e5rt.e5bundlecache/<os-build>/<hash>/`, `force_recompilation` bypass — **G** ch. 5; `~/Library/Caches/<app>/…/H16G.bundle/H16G.e5` — **M1**.
   - (b) "The `aned` daemon caches compiled programs **in memory**… lost on reboot" — **MO** README §12.
   - **Reconciliation:** two different caches — persistent `.e5` bundle cache vs the daemon's in-memory *loaded program instance* cache. Both can coexist.

7. **Compile limit: ~119 vs ~128.**
   - (a) ~119 compiles then failure/resource leak — **MO**, **MR**, **M3**.
   - (b) "loaded-program cap is near 128 programs per process: the next load fails with `GetANEFModel: must re-compile`" — **G** ch. 14.
   - **Reconciliation:** compile-count leak vs simultaneously loaded program cap; numerically close but distinct mechanisms. Plan against ~119 as the conservative bound.

8. **Baked weights only vs weight-as-input / runtime patching.**
   - (a) "Weights are compiled into the program binary. The `weightsBuffer` IOSurface (meant for runtime weight override) **does not work**… Every unique weight matrix needs its own compiled program" — **MO** README §3; spec 07 §1.5 "Baked weights (Fatal for Multi-Layer)".
   - (b) spec 07 §1.2 in the *same document* calls "ANE MIL Weight-As-Input Technique… Weights NOT baked — passed at execution time" the only viable reuse approach; and **G** ch. 7 says a host can patch compiled weights in place.
   - **Reconciliation:** (b) is a later workaround in the same ongoing investigation (the dynamic pipeline in maderix Part 3 packs weights into the spatial input dimension). Runtime IOSurface *weight override* is dead, but *weight-as-activation-input* works. Treat (a)'s "fatal" as superseded by the dynamic pipeline.

9. **Core ML quantized models on ANE — mechanism unknown.**
   - Community says "CoreML quantized models somehow run on ANE — the internal compiler flag or op that enables this is unknown" (**MO**) and "it's unknown whether such 8-bit operations are actually performed on the ANE or only on the CPU" (**H**), while Apple documents W8A8 on ANE (**CML**). Our compute plan shows ANE for INT8 mlprogram ops but that is not runtime proof.

10. **Last-axis alignment: 64 bytes vs 16 bytes.**
    - (a) "the last axis of an ANE buffer is not packed; it must be contiguous and aligned to **64 bytes**" — **AML** "Principle 1".
    - (b) "The width axis aligns to a **16-byte** direct-memory-access granule"; three scales 16 B / 256 B / 16 KB — **G** ch. 17/21/24/28.
    - **Reconciliation:** Apple's 64 B is a *transformer buffer layout* recommendation; G's 16 B is the *DMA hardware granule*. Satisfy both by using 64-byte-aligned last-axis widths (a multiple of 64 is also a multiple of 16).

11. **Slice-saturation generation (internal to G).** ch. 3 says A15+ clean; ch. 19 says A15+ clean and A14 saturates; ch. 12 says M2/A14 saturates and only M5 is clean, explicitly "corrects the earlier reading". Prefer ch. 12 (latest correction): saturated on M1/M2, clean from M5; verify per target.

12. **"First ANE" framing.** H and M1 call A11 (2017) the first Neural Engine; community hardware analysis (Maynard Handley, Hacker News thread on M1) says A11 was an FPGA-like precursor and the "real" ANE lineage starts with A12. Affects capability tables that begin at A11. **[community comment vs community docs]**

13. **Core ML smart-placement direction.** H `other.md` anecdote: GPU idle → Core ML likely uses GPU; GPU heavily loaded → NPU. G ch. 5 says placement is a Dijkstra cost minimization with fixed launch + transfer penalties favoring fewer/larger segments, not a load-reactive heuristic. **Reconciliation:** G describes the ahead-of-time segmenter; the anecdote may describe runtime QoS/queue arbitration. Unresolved.

---

## 9. Cross-cutting checklist for the team before building ANE features

1. Decide **path** first: public Core ML (`computeUnits=.all`, MIL/mlprogram, compute plan) vs private `_ANEInMemoryModel`/`e5rt` (no public support, fragile, private APIs, App Store risk noted in all community sources). This determines most of the "contradictions".
2. Target the **oldest generation** you must support and compile for it (`TargetArchitecture=h13` style); later generations are operation supersets (G ch. 12).
3. Enforce shape rules before compiling: rank ≤5, W/H ≤16384 (65536 on A16+), conv kernel W ≤13 fp16 (M1), matmul depth=1, `S≥16` fp32 / `S≥32` fp16, ≤2 outputs/program, 64-byte-aligned last axis, IOSurface ≥49152 B.
4. Keep live operands below the working-set threshold and fuse chains (16–64 ops) to amortize the dispatch floor; avoid cancellation-heavy reductions on-engine.
5. Treat compute plans as *placement hints*, not runtime proof; pair with power/energy measurement.
6. Gate every "verified" claim on measurement on the actual Veystrix hardware (owner rule: 16 GB M4 first).
7. Revisit §8 periodically — at least the INT8, SRAM, and compile-limit items are actively changing as the community code evolves.

---

*End of catalog. Source files were read in full on 2026-11-17; external copies cached under the session temp dir `ane_research/` for re-verification. No files under `/Users/vincent/Documents/Veystrix` other than this note were modified.*
