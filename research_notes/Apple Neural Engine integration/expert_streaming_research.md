# Expert-read amortization & NPU/ANE+GPU co-execution: 2025–2026 literature audit

**Prepared:** 2026-11-17
**Question from Veystrix:** given a 16 GB M4 Mac, two 4-bit MoE models on NVMe SSD, Qwen decode 385 ms/token of which 289 ms is expert-read wait (480 expert reads/token × ~3 MB, 24% hit in a 349-slot S3-FIFO pool, SSD ~3.3 GB/s and nearly saturated), and a rejection list that already includes perfect-expert-predictor+overlap (TIED), MTP at all depths (LOSES), contiguous expert packing (+6.3% Nemotron / 0% Qwen server-side), pool policies other than S3-FIFO, larger pools, route-union prefill, 4K chunks, F_NOCACHE/F_RDAHEAD=0, etc. — is there a 2026 technique that cuts the read wait while keeping bit-identical greedy output?

## Executive answer (blunt)

**No, not as published.** Every 2026 paper that actually reduces *SSD expert bytes* on single-user decode does one of four things Veystrix has already rejected or cannot use:

1. **Changes routing** so more experts are cache hits (S2-MoE's gating bias; Skliar et al.'s cache-prior reranking; BigMoMo's expert pruning). These provably change token IDs — disqualified by the bit-identical requirement.
2. **Changes weights / adds a trained module** (DraftExpert's self-distilled per-layer draft expert; ZipMoE's lossless compression, which targets BF16 exponent bits, not 4-bit weights).
3. **Relies on idle bandwidth** (PCIe host DRAM, or an unsaturated UFS/SSD) so prefetch can hide latency. Veystrix's SSD is *already saturated* — the user's own "perfect predictor + overlap TIED" measurement is the decisive experiment, and it is confirmed by the literature: no paper shows a gain from better prediction when the bus is the bottleneck.
4. **Relies on an NPU runtime with runtime weight/address binding** (Qualcomm QNN/Hexagon HTP) that Apple's ANE does not expose. ANE bakes weights at compile time.

There is **exactly one mechanism in the literature that is bit-exact in principle, targets read *sharing* rather than read *overlap*, and has not been tested by Veystrix**: **expert-wise (not token-wise) multi-token verification**, i.e. load each needed expert once, run *all* draft rows routed to it, accumulate in the baseline order, scatter back. Specter §3.4.2, FlashMoE's prefill loop, EStream §3.2 and BigMoMo §3.4 all describe this. It is distinct from the rejected "overlap/prefetch" experiment and from the rejected "batched verification rows" MTP experiment. Expected upside is modest (roughly 5–15% of read wait at K=2, depending on Qwen's consecutive-token expert-set union), and it may still lose once draft cost is included. A cheap offline test below settles it in under an hour without touching the runtime.

For **ANE weight binding specifically: flat no.** Confirmed by three independent 2026 sources (ANEForge, Orion, Anemll Forge).

---

## 1. Source verification table

| Source | URL / identifier | Date | Authors / affiliation | Venue & replication status | Code |
|---|---|---|---|---|---|
| Specter | arXiv:2511.14102v3, DOI 10.1145/3845814.3850026 | v1 18 Nov 2025, v3 22 Oct 2026 | Wenfeng Wang, Xiaofeng Hou, Xinfeng Xia, Xinkai Wang, Mingxuan Zhang, Peng Tang, Jiacheng Liu, Chao Li — Shanghai Jiao Tong Univ. | **ASPLOS '27** (accepted, with artifact appendix). No independent replication found. | github.com/wwwwwf/specter_asplos |
| S2-MoE | arXiv:2608.15018v2 | v1 15 Aug 2026, v2 19 Aug 2026 | Haochen Huang, Shengxuan Qiu, Meng Li — Peking Univ. | Preprint (13 pp, 10 figs), no venue stated. No independent replication found. | github.com/angerybob/S2-MoE |
| DraftExpert | arXiv:2607.24434v1 | 27 Jul 2026 | Dengke Han (single author) | Preprint (9 pp, 6 figs), no venue. No code link in paper. No replication. | — |
| ZipMoE | arXiv:2601.21198v2 | v1 29 Jan 2026, v2 22 May 2026 | Yuchen Yang, Yaru Zhao, Pu Yang, Shaowei Wang, Zhi-Hua Zhou — Nanjing Univ. | **ICML 2026**. No independent replication found. | github.com/npnothard/ZipMoE-ICML26 |
| FlashMoE | arXiv:2601.17063v2 | v1 22 Jan 2026, v2 5 Oct 2026 | Byeongju Kim, Jungwan Lee (KAIST), Donghyeon Han (Chung-Ang), Hoi-Jun Yoo (KAIST), Sangyeob Kim (Yonsei) | Preprint, no venue. No code link. No replication. | — |
| EStream | arXiv:2609.06551v1 | 6 Sep 2026 | Junming Zhang, Zhenzhe Zheng, Fan Wu (SJTU), Xiaoyao Huang (China Telecom), Jie Wu (Temple) | Preprint; arXiv HTML still contains an unfilled "2018 / Woodstock NY" template block. No code link. No replication. | — |
| BigMoMo | arXiv:2609.14643v2 | v1 ~Sep 2026, v2 3 Oct 2026 | Maoliang Li, Hailong Zou, Taohong Han, Haoze Chi, Jiayu Chen, Zihao Zheng, Zimo Guo, Jie Zhang, Guojie Luo, Xiang Chen — PKU / TJU | αXiv indexes it under NeurIPS 2026; acceptance not confirmed in the paper. No code link. No replication. | — |
| ANEForge | arXiv:2606.17090v1; Wiley *Software: Practice and Experience*, DOI 10.1002/spe.70124 | 12 Jun 2026 | Spencer H. Bryngelson — Georgia Tech | Peer-reviewed journal article + reproducible capsule (Zenodo 20672609). Verified on M5 Pro/macOS 26.5/ANECompiler 3520.4.1 and M1 Max. | github.com/sbryngelson/ANEForge (MIT) |
| Orion | arXiv:2603.06728v1 | 6 Mar 2026 | Ramchand Kumaresan | Preprint, open source. Single-author, self-reported numbers. | github.com/mechramc/Orion (MIT) |
| Anemll Forge | github.com/Anemll/anemll-forge; HF blog "Three-bit MLP vector quantization on the ANE" | Release 0.2 Oct 2026; 0.2.1 19 Oct 2026 | ANEMLL community project | Research software, explicitly "not full-model validation"; 56 stars. | github.com/Anemll/anemll-forge |
| Mixture of Cache-Conditional Experts | arXiv:2412.00099v2; OpenReview forum `ul4W26KEKz` | v2 24 Jun 2025 | Andrii Skliar (Contextual AI), Ties van Rozendaal, Romain Lepert, Todor Boinovski, Mart van Baalen, Markus Nagel, Paul Whatmough, Babak Ehteshami Bejnordi — Qualcomm AI Research | **TMLR 2025**; 26 citations. No on-device code release found. | — |
| Batch Speculative Decoding Done Right (EQSPEC/EXSPEC) | arXiv:2510.22876v3; OpenReview submission 21406 | Oct 2025 (v3 2026) | Ranran Haoran Zhang, Soumik Dey, Ashirbad Mishra, Hansi Wu, Binbin Li, Rui Zhang | Under review at OpenReview; no acceptance. | — |
| FluxMoE | arXiv:2604.02715v1 | 3 Apr 2026 | Qingxiu Liu, Cyril Y. He, Hanser Jiang, Zion Wang, Alan Zhao, Patrick P. C. Lee — CUHK / SCITIX | Preprint, vLLM-based, server-only. | — |
| Expert Streaming (FSE-DP) | arXiv:2603.27624 | 2026 | S. Ma et al. | Multi-GPU data-parallel; not single-device. | — |

All five arXiv IDs in the original task verified as real and matching the described topics. Numbers below are transcribed from the full texts (HTML versions), not from abstracts alone.

---

## 2. The five named papers: exact mechanism, numbers, hardware, and the Veystrix assumption check

### 2.1 Specter — arXiv:2511.14102 (ASPLOS '27)

**What it does, exactly.** A 4-bit quantized copy of the *target* MoE is used as the draft model. Because it shares routing architecture and parameter lineage, its router predicts the target's expert IDs with 90.8% mean top-4 agreement (measured on Qwen1.5-MoE, 1,000 requests × 24 layers = 24,000 request-layer pairs; Table 1 across models/quantizers: mean 90.8–99.4%). The draft produces K tokens and an *Expert Lookahead Buffer* of per-layer activation counters; a "Layer-Aware Prefetch Planner" issues H2D transfers in three phases (Cold Start / Prefetch Overlap / Final Cleanup) at chunk boundaries. Speculation depth K chosen offline; K=8 for Qwen1.5-MoE, K=16 for DeepSeek-V2-Lite and Phi-3.5-MoE (their τ̄=96.63% gives E[accepted]=28.7, hence K_max=32).

The part that matters for Veystrix is **§3.4.2 Cache-Aware Expert Verification Reordering**: *"the target router first computes the authoritative routing decisions for all K tokens. Rather than processing FFN of tokens in chronological order, Specter gathers tokens dispatched to identical experts and packs their activation states into contiguous sub-batches. This spatial aggregation converts fragmented, memory-bound GEMVs into dense, compute-bound GEMMs."* Cache-resident experts are computed first; missing experts are fetched asynchronously while those GEMMs run; outputs are scattered back with routing weights. Paper states this *"guarantees strict mathematical equivalence"* — target routing unchanged, causal attention unchanged.

**Measured numbers / hardware.** 1.50× mean TPOT speedup, up to 1.88×, vs SpecMoEOff, single-request. Hardware: single GPU with experts in **host DRAM over PCIe** (not SSD); models DeepSeek-V2-Lite, Qwen1.5-MoE, Phi-3.5-MoE. They explicitly report *"expert explosion"*: K=1→16 grows per-layer distinct experts 6.0→31.7 on DeepSeek-V2-Lite (5.3×, ~half of 64 experts), growth factors 7.18 (Qwen1.5-MoE) and 4.65 (Phi-3.5-MoE).

**Does the assumption hold for Veystrix?**
- **No** for the headline result. Specter's gains come from hiding *millisecond PCIe* transfers behind draft compute plus GEMM-speeding verification, on a bus with spare capacity. Veystrix's SSD is saturated, and its own perfect-predictor+overlap experiment TIED baseline — exactly the condition under which Specter's lookahead adds nothing.
- **No** for the INT4-as-draft trick: Veystrix already has two 4-bit MoEs loaded; a third resident copy (draft) does not fit in 16 GB. Their measurements also showed expert-set growth with depth, i.e. the "expert explosion" they warn about.
- **Partially yes** for §3.4.2's expert-wise reordering: that is a *read-sharing* technique (one expert load serves K rows), not a read-overlap technique. It is bit-exact by construction. This is the one piece worth testing (§7).

### 2.2 S2-MoE — arXiv:2608.15018

**What it does, exactly.** Self-speculative decoding in llama.cpp with three components:
- **Routing-aware adaptive speculative expansion**: utility U_i = p_i·T_AR / (|E_i \ ⋃_{j∈S} E_j|·T_exp + C_i^draft). Admit candidate if U_i ≥ 1. Cost model calibrated R²=0.943–0.999, MAPE 0.8–4.5%.
- **Reuse-aware expert gating**: aggregates gated logits across draft tokens with confidence weights, picks a capped set of "globally preferred experts", then **injects an additive bias** into gating logits: ℓ'_{b,e} = ℓ_{b,e} + λ_b·1[e∈E*], λ_b ∝ (top-1 score − (k+1)-th score).
- **Context-aligned self-speculative**: draft and target share one KV cache; draft KV writes are rolled back, only accepted tokens committed.

**Numbers / hardware.** Up to 5.3×, ~2.0× mean on Jetson Orin (16/32/64 GB); 1.2–2.9× on RTX 4090. Draft length 8, width 2. Models: OLMoE-1B-7B, DeepSeek-V2-Lite, Qwen3-30B-A3B, GPT-OSS-120B. Qwen3 config: 128 experts, top-8, cap 14, 3 draft experts.

**Fatal for Veystrix.** The paper is explicit: *"Reuse-aware expert gating intentionally introduces a bounded routing bias ... and is therefore a controlled target-routing approximation rather than a strictly lossless transformation."* Measured PPL ratio 1.012–1.013, top-1 agreement >89%, task accuracy "small and non-systematic" differences. **Token IDs change.** Also its best speedups come from the adaptive expansion and gating, not from an SSD-specific mechanism; on Jetson, offload is to SSD, but UFS/NVMe behavior differs, and Veystrix's bus is saturated. Rejected.

### 2.3 DraftExpert — arXiv:2607.24434

**What it does, exactly.** Trains **one lightweight accelerator-resident draft expert per MoE layer** by self-distilling four signals from the frozen target: residual (missing routed-expert residual after shared+top-1), logit KL, top-1 token CE, and router-agreement KL. Inference drafts with a *fixed* shared + top-1 + draft-expert path. A confidence–expansion controller stops before token i when q_i < τ **and** marginal predicted verifier-expert expansion Δ_i > B_i. Draft-router predictions trigger non-blocking prefetches of likely verifier experts. All emitted tokens are exactly verified by the frozen target.

**Numbers / hardware.** 1.45× mean decode TPS; acceptance 84–87%; prefetch hit 86–88%. Verify cost normalized to one AR token: fixed-K 2.35× (CPU→GPU) / 3.10× (Flash→NPU); with confidence+expansion 1.48×/1.70×; **with prefetch 1.25×/1.35×** (hit/waste 88/12%, 86/14%). Hardware: RTX 4090 24 GB with CPU-resident BF16 experts; Xiaomi MIX Flip 2, Snapdragon 8 Elite, Hexagon HTP v81 (~73/146 TOPS INT8/INT4, 8 MB SRAM), llama.cpp Q4_0, Flash-resident experts. Models: DeepSeek-V2-Lite, Moonlight-16B-A3B (27 layers, 64 routed + 2 shared, top-6). Flash-staged AR baseline 10.18 TPS; resident-memory reference 14.36 TPS.

**Fatal for Veystrix.**
- Requires **training a new expert per layer** (self-distillation on unlabeled text) and keeping it resident — a weight change, not a drop-in runtime trick. Bit-exact *outputs* are preserved, but the model artifact is different.
- Even after all tricks, verification costs **1.25–1.35× one plain token** — i.e. each extra verification row is still *more expensive* than the 137 ms plain token Veystrix measured. On a saturated SSD the prefetch contribution (which supplied ~0.23–0.35× of that saving) disappears.
- Their gain is measured against a *Flash*-staged baseline (10.18 TPS vs 14.36 resident); Veystrix's baseline is already near NVMe peak bandwidth, so the headroom is far smaller.
- No code released, single-author preprint, no replication.

### 2.4 ZipMoE — arXiv:2601.21198 (ICML 2026)

**What it does, exactly.** Lossless compression of **BF16 exponent bits** only. Exponent-bit symbol support is 11–15% of the space (Shannon entropy 2.551–2.651 bits vs 8); ZSTD reaches 68% of original size, LZ4HC 74%, entropy floor ~66%. Offline: split sign–mantissa (SM) bits from exponent (E) bits, compress E into E-chunks, pack SM into SM-chunks. Runtime: E/SM/full tensors cached in four pools (F, C, S, E) planned by a rank-based Poisson-binomial + DP model; a DAG scheduler with L CPU decompression threads overlaps SM loading with E decompression; a coalesced CUDA kernel reassembles BF16. Constant-factor bound ALG ≤ (3 − 1/L)·OPT.

**Numbers / hardware.** Up to 72.77% latency reduction and 6.76× throughput vs MoE-Infinity / DeepSpeed / Accelerate. TPOT reduction 62.65–97.97% (decoder-only), TTFT 53.25–87.90%. Platform: Jetson AGX Orin 64 GB and 32 GB, **Samsung 970 EVO, 3.5 GB/s**, JetPack 6.2.1. Models: DeepSeek-V2-Lite, Qwen1.5-MoE, SwitchTransformers-Large-128. Decompression overlaps SSD I/O with ≥3 threads; 7.47% peak-read throughput loss from memory-controller contention. OS page cache used elastically; with 32 GB background memory pressure the gains persist.

**Fatal for Veystrix.**
- The entropy argument is **specific to BF16**. Veystrix's experts are 4-bit; the sign/mantissa/exponent bit-field decomposition does not exist, and 4-bit codebooks are already near 4 bits/weight. ZipMoE's own ~2× "tensor coverage" trick (cache SM bits, recompute E bits) has no analogue.
- Zero-copy path reads SM-chunks into host-pinned UMA memory registered for direct GPU access — adjacent to Veystrix's rejected "native zero-copy of unaligned shards".
- Their SSD is 3.5 GB/s and *not* described as saturated; their gain comes from replacing I/O with idle CPU, not from reducing saturated-bus bytes.
- Not applicable. (Cheap sanity check in §7.3 anyway.)

### 2.5 FlashMoE — arXiv:2601.17063

**What it does, exactly.** Separates experts from non-experts at file level (one `.pt` per layer-expert), loads non-experts at init and each needed expert once per iteration. ML cache replacement: a 3-layer, 128-hidden SiLU FFN takes [1/recency, frequency/max_frequency], trained with MSE against Belady-eviction labels (512 TriviaQA samples × up to 512 tokens, 200 epochs, ~2 h total pipeline, ~113 KB per-layer model). Evict the expert with the highest predicted "distance". Cache replacement runs asynchronously, overlapped with ~3 ms SSD loads (FFN compute ~158 µs).

**Numbers / hardware.** Cache hit rate +21% vs LRU, +51% vs LFU, +28% vs ARC, +21% vs LeCaR; I/O −22%/−35% vs LRU/LFU. End-to-end only **+22% (OLMoE) / +7% (Qwen3-30B)** over LRU; 2.6× vs other systems; initial load 4× faster than llama.cpp. Hardware: Ryzen 5 9600X, RTX 5070 Ti 16 GB, **SK hynix P51 PCIe 5.0 7.4 GB/s**, 64 GB DDR5, Ubuntu 24.04. LRU here had ~73% hit vs Belady 86%; LRU evicted experts were reused within 5 steps 34.2% of the time.

**Fatal for Veystrix.** This is a **cache replacement policy**, squarely in the rejected "pool policies other than S3-FIFO" bucket. Their measured end-to-end gain on the Qwen model is 7%, on an SSD with 2.2× Veystrix's bandwidth and 4× its RAM. Nothing new here.

---

## 3. EStream / BigMoMo / EdgeMoE: "one compiled expert graph with runtime weight binding"

### 3.1 EStream (arXiv:2609.06551) — the exact idea exists, on Qualcomm, for prefill

EStream is the paper that literally implements the phrase in the task: **"topology-invariant expert graph sharing compiles one expert graph for all experts of the same shape, and runtime route and parameter binding passes each expert's tokens and weight address as inputs on every call."** Concretely: one shared graph with fixed tensor bases for X, Y and three parameter planes (gate/up/down); a host-residency map Φ_l(e) → slot s; at invocation, scalar workers resolve ψ_i=Φ_l(e_i), prefix-sum route counts by slot, build a CSR-style (token, route-index) record, and compose it with parameter addresses p_s^j = base_j + s·S_j. The NPU then gathers rows, DMAs weight tiles, runs gate/up → SwiGLU → down, and scatter-adds. **"Only the route descriptors, valid extent, and placement map change across invocations. The graph topology, tensor bases, and strides remain fixed."** Expert virtualization keeps the pool in UFS and pages groups through a bounded NPU-addressable arena with an Available → Filling → Ready → InUse protocol; weights are offline-transformed into the NPU's 32×32 tile-major layout so no runtime reformat is needed.

**Numbers / hardware.** 2.25–27.57× pure-prefill TTFT speedup and 1.19–12.29× lower peak physical memory vs fastest baseline across 18 settings (OLMoE, LFM2.5, DeepSeek-V2, 7B–16B, 256–4,096-token prompts) on a **OnePlus 15 / Snapdragon 8 Elite Gen 5**, weight-only Q4, no approximation. Pipeline bubble rates: on-demand 34.41%, prefetch-after-routing 21.36%, prefetch-before-routing 9.05%. UFS queue depth Q0=2 (95% of peak grouped-read bandwidth).

**Transferability to ANE: none.** EStream's runtime binding works because the HTP exposes scalar control threads, DMA engines, software-managed VTCM, and low-level QNN primitives that let addresses be parameters. Apple's ANE has no equivalent dataflow: MIL programs are compiled to E5 microcode with weights baked in (Orion constraint #7), and the compute unit cannot branch on runtime routing. Also EStream targets **prefill-only**, where one request touches 80–100% of layer-expert pairs — the opposite of Veystrix's batch-1 decode. Its "one graph" insight is still the conceptual ancestor of the §7 test, but the implementation mechanism is Qualcomm-specific.

### 3.2 BigMoMo (arXiv:2609.14643) — same window, but approximate

BigMoMo uses the multi-token verification window so a single expert movement serves multiple draft tokens: *"verification stage generates multiple draft tokens that activate the same expert to share one expert loading."* Three techniques: (a) **Acceptance-Aware Node Pruning** (drop low-value draft nodes; exact) and **Hierarchy-Aware Expert Pruning** (drop low-impact experts, reassign routing weight to nearest-score retained expert — **explicitly an approximation to target computation**); (b) runtime-adaptive on-flash organization via a hypergraph of co-loaded experts and incremental swaps into "anchor regions"; (c) cross-hierarchy movement scheduling with VTCM double buffering and readiness-aware group selection.

**Numbers / hardware.** Mean 4.83× over on-demand AR, 1.82× (alphaXiv: 1.76×) over best speculative baseline. Incremental normalized TPOT (Qwen3/LFM2/Gemma4/ERNIE4.5): SD-OnDemand 1.00 → +Admission 0.82/0.81/0.75/0.71 → +Organization 0.57/0.60/0.45/0.44 → full 0.36/0.43/0.31/0.31. Hardware: OnePlus 15 (8 Elite Gen 5, 16 GB, 11.52 GB available, UFS 4.1), OnePlus 13 (8 Elite, 24 GB), OnePlus Ace 5 (8 Gen 3, 12 GB); llama.cpp + Hexagon NPU, Q4_0, draft length 2, branch width 2. Models up to Qwen3-Next-80B-A3B.

**Verdict.** Its read-sharing idea is the same one flagged in §7, but the paper's wins are dominated by **expert pruning (approximate)** and **adaptive flash layout** — which is contiguous packing, already rejected by Veystrix's own measurement. UFS 4.1 (~4 GB/s) with small-random-read penalties is a different regime from a near-saturated NVMe at 3.3 GB/s. Not transferable as-is; the exact subset (node pruning + read sharing) is the §7 experiment.

### 3.3 EdgeMoE / Expert Streaming / FluxMoE

- **EdgeMoE** (Yi et al., arXiv 2310.12963) uses expert-wise bitwidth adaptation and a meta-expert predictor with address-table lookups; CPU/SSD expert streaming, no runtime weight binding on an NPU. Superseded by the 2026 papers above.
- **Expert Streaming / FSE-DP** (arXiv:2603.27624) is multi-GPU fully-sharded expert data parallelism for low-batch serving — irrelevant to one Mac.
- **FluxMoE** (arXiv:2604.02715) formalizes "model = compute graph + streamed parameters" with **PagedTensor** (stable virtual addresses, physical blocks rebound before kernel launch) — conceptually the closest GPU analogue to EStream. But it is vLLM server-side, batch 32–256, H100-class GPUs, and its storage hierarchy is compressed GPU memory + pinned host DRAM (not SSD). Up to 3.0× at batch 256 / 4K context. The virtual-address trick relies on CUDA streams/events and unified virtual addressing, not available for ANE.

### 3.4 Can "one compiled expert graph with runtime weight binding" be done on ANE? — **No**

Three independent, dated sources:

- **ANEForge** (arXiv:2606.17090, peer-reviewed SPE, Bryngelson, Georgia Tech): *"The compiler lowers the whole graph to MIL ... **packs the weights into one binary blob**, and produces a callable program handle. ... The program handle is reused across calls, so a model compiles once and runs many times."* Its `compress="int4"` option streams **int4 lookup-table encoded static weights** through the engine's dequantization path — a *compile-time encoding*, not runtime binding. A small program calls in ~90 µs against a ~70 µs dispatch floor. Critically: **"the engine has no data-dependent control flow"** and only one hardware lane on tested machines. Verified only on M5 Pro (macOS 26.5) and M1 Max; no M4 verification.
- **Orion** (arXiv:2603.06728, Kumaresan; M4 Max, macOS 15, open source): constraint #7 — **"Weights baked at compile time: stale weights → recompile after update."** Full recompile ~4.2 s for 60 kernels; their *delta compilation* (unload → patch BLOBFILE on disk → reload, preserving hexStringIdentifier) reduces this to **494 ms per step for 60 kernels (~8.3 ms/kernel)**. That is ~2 orders of magnitude slower than Veystrix's 137 ms *entire token*, so per-expert/per-token swapping is impossible. The one true runtime-binding path is **LoRA adapter-as-input**: low-rank A/B matrices passed as IOSurface inputs and hot-swapped with zero recompilation. But base weights remain baked; this cannot bind a full expert matrix. (Additional ANE constraints discovered: multi-input surfaces must have uniform allocation size; inputs bound alphabetically; padded surfaces read as packed [1,C,1,S] from byte 0 — directly relevant to the rejected "native zero-copy of unaligned shards".)
- **Anemll Forge** (GitHub, release 0.2/0.2.1, Oct 2026): **pre-M5 chips are unsupported and fail before loading**; requires macOS 27 / Xcode 27 / Core AI; target is Qwen3.8-27B with **3-bit vector-lookup MLP weights** fitted by GPTQ+QAT (lossy; KL-512 0.052 to BF16, top-1 agreement 91.9%), first-start ANE compile 20–25 min, ≥32 GB RAM recommended. It uses Core AI / Core ML compiled packages plus DFlash2 speculative drafter — not runtime expert binding, and incompatible with the user's M4/16 GB.

**Bottom line:** there is no public or private ANE API that compiles one expert graph and binds a new full expert weight tensor per call. The closest public thing (Core ML `MLState`, macOS 15+) binds KV state, not expert weights. Private `_ANEClient`/`_ANECompiler`/`e5rt` expose compile → load → eval with baked weights; Orion's reload path costs ~8 ms/kernel and touches the filesystem. Any "ANE expert streaming" design must therefore keep a *compiled program per expert* (or per expert group), which is what EStream explicitly avoided on QNN and what ANE cannot avoid.

---

## 4. Cache-aware routing: exact gains, and why it fails the bit-identical test

**Skliar et al., "Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference"**, arXiv:2412.00099v2 (24 Jun 2025), TMLR 2025. Authors: Andrii Skliar (Contextual AI; work done at Qualcomm), Ties van Rozendaal, Romain Lepert, Todor Boinovski, Mart van Baalen, Markus Nagel, Paul Whatmough, Babak Ehteshami Bejnordi (Qualcomm AI Research).

**Mechanism.** Three training-free rerankings of the router's expert ordering, all of which **change which experts are selected**: (1) Max-Rank: promote cached experts among router ranks < M, while always keeping top-J; (2) Cumulative-Probability Threshold: choose M dynamically so ∑_{j≤M} w_j ≥ p; (3) **Cache-Prior Reranking**: z' = z + λ·Δ_avg·m̃_t, where m̃_t is the cache bitmask (plus forced top-J), Δ_avg is a running estimate of max(z)−min(z) for the layer, λ∈[0,1]. Crucially, z' is used **only for ranking**; the original z computes the expert weights — but the selected expert *set* changes, so the model output changes.

**Measured numbers.** Cache miss rate reductions >50%: Qwen1.5-MoE (30/60 cached) 35%→16% with lifetime 26→58 tokens; DeepSeek-V2 (32/64) 28%→7%, lifetime 19→76; Phi-3.5-MoE (8/16) 22%→9%; Mixtral-8x7B (4/8) 40%→21%. Perplexity impact 0.1–3%; downstream accuracy <0.1%; in some cases accuracy *improves* (they call it implicit regularization). On-device: **2× speedup** on 4-bit Qwen1.5-MoE with a 12 GB Snapdragon device (10 GB available, 30 experts cached/layer) and on 8-bit Qwen1.5-MoE with a 16 GB device (45 experts cached/layer), Android 14, llama.cpp with mlock, 10 runs. Throughput vs cache hit rate is near-linear (Figure 8).

**Verdict.** This is the single most on-point empirical result for Veystrix's 24% cache hit — and it is **disqualified**: it explicitly trades routing fidelity for cache locality, and the paper says so ("lossy routing strategies", "surpasses the oracle cache bound by allowing slight flexibility in expert selection"). Token IDs are not preserved. If the user ever relaxes bit-exactness, this is the first thing to revisit (5-line router change, training-free, measured on a 4-bit Qwen on a 16 GB phone). Under the stated constraint: no.

S2-MoE's reuse-aware gating (§2.2) and BigMoMo's expert pruning (§3.2) are the same category — both change routing, both therefore change token IDs.

---

## 5. "Exact speculative decoding" that preserves greedy IDs bit-for-bit

- **Classical speculative decoding** (Leviathan et al. 2023; Chen et al. 2023) is distribution-exact, and with greedy sampling it accepts a draft token iff it equals the target argmax — so accepted tokens are bit-identical *provided the target's numerical path is unchanged*. The problem is never the accept rule; it is **batched verification changing the reduction order / routing**.
- **Specter §3.4.2** is the cleanest statement of the fix that matters here: gather tokens by expert, keep the target's authoritative routing, keep causal attention, scatter outputs back, layer-wise sync — "guarantees strict mathematical equivalence". FlashMoE's prefill loop ("load each expert once per iteration, run MLP on all routed tokens, redistribute and accumulate back to token indices via indexed addition") and EStream §3.2 are the same pattern on different hardware. **All three preserve the per-token expert computation; they only reorder independent FFN work.** If the scatter-add uses the same accumulation order as baseline (expert index order, or the exact baseline loop order), greedy IDs are bit-identical.
- **EQSPEC / EXSPEC** (arXiv:2510.22876, Zhang et al., Oct 2025; OpenReview 21406) is the paper that names the "losslessness" requirement for *batched* speculative decoding: the ragged-tensor problem (sequences accept different numbers of draft tokens) corrupts position IDs, attention masks and KV state. EQSPEC's "Unpad–Append–Repad" makes batched output identical to non-spec decoding (the rebuttal claims first-and-only lossless batch SD), but realignment costs up to 40% of overhead; EXSPEC groups same-length sequences from a sliding `SequencePool` to cut that. **This addresses batch of independent requests, not batch-1 verification rows**, and notes 95% bit-equality in practice due to floating-point non-determinism. Not useful for Veystrix.
- **Orion** provides the relevant negative result on ANE numerics: fp16 overflow, silent stale weights, and manual SDPA decomposition can break token-for-token parity; once fixed, CPU and ANE greedy paths gave 100% top-1 agreement for GPT-2 124M. So bit-exactness is achievable on ANE, but only with a fully controlled, fixed program — another reason runtime weight binding is out.
- **No 2026 paper found** that verifies K tokens *sequentially through the target* specifically to share expert reads while preserving batch-1 reduction order. The closest is FlashMoE's prefill (which does exactly the load-once/compute-all-rows part) combined with Specter's correctness argument. The combination is the untested idea below.

---

## 6. Why the headline numbers do not transfer (one-line summary)

| Paper | Headline | Destroys transfer |
|---|---|---|
| Specter | 1.50× TPOT | PCIe host DRAM, spare bus; requires resident INT4 target copy; perfect-predictor+overlap already TIED for Veystrix |
| S2-MoE | 5.3× max | gating bias changes routing → token IDs change |
| DraftExpert | 1.45× | trained draft expert; verify still 1.25–1.35× one plain token; prefetch useless on saturated SSD |
| ZipMoE | 72.77% latency cut | lossless compression of **BF16 exponent bits**; Veystrix is 4-bit |
| FlashMoE | +51% hit rate, 2.6× | rejected cache-policy class; only +7% end-to-end on Qwen; 7.4 GB/s SSD |
| EStream | 2.25–27.57× TTFT | QNN/HTP-only runtime binding; prefill-only |
| BigMoMo | 4.83× / 1.82× | expert pruning is approximate; adaptive flash layout ≈ rejected contiguous packing; UFS regime |
| Skliar et al. | >50% miss cut, 2× | changes expert selection → token IDs change |

---

## 7. The one untested idea, and the first cheap offline test

### 7.1 Mechanism

**Expert-wise K=2 verification with shared reads and baseline-order accumulation.**

Currently Veystrix's MTP/drafting loses because verification is token-row-wise: each extra row triggers its own set of expert reads (~150–200 ms extra), and the SSD is already saturated, so no overlap helps. Instead:

1. With a cheap drafter (or, for the measurement, an *oracle* next token from a recorded trace — this is an offline upper bound, not a deployed change), construct a 2-token verification window.
2. At each MoE layer, compute the target router for both tokens (both hidden states exist inside the same causal forward pass), then **group tokens by expert**. Distinct layer-experts = |E_1 ∪ E_2|, not |E_1| + |E_2|.
3. Read each distinct expert **once**. Run its FFN on both routed rows (a 2-row GEMM, not two GEMVs). Accumulate weighted outputs into each token's residual in the *same expert order* the baseline uses, so the per-token arithmetic is unchanged.
4. Greedy accept: token 1 emitted only if target argmax matches; token 2 emitted only if all prefix matches. No rerouting, no gating bias, no weight change, no pool change → bit-identical by construction.

Why it can beat the rejected experiments: the rejected "perfect predictor + overlap" reduced *latency exposure*, not *bytes*. This reduces bytes by (1 − |E_1∪E_2| / (|E_1|+|E_2|)) × 289 ms. Specter measured ~7.18× expert-set growth from K=1 to K=16 on Qwen1.5-MoE; if K=2 growth is ~1.7–1.8×, the saving is ~10–15% of read wait (~30–45 ms/token) **before** draft cost. If growth is ~2.0×, there is zero saving and the idea dies.

### 7.2 First cheap offline test (no runtime change, <1 hour)

Use an existing recorded greedy trace of, say, 200 Qwen tokens with per-token, per-layer top-k expert IDs (or re-run 200 tokens with routing logging — the router logits are already computed).

```
for each consecutive pair (t, t+1):
  for each MoE layer l:
    u = |experts_l(t) ∪ experts_l(t+1)|
    a = |experts_l(t)|, b = |experts_l(t+1)|
  union_ratio = sum_l u / sum_l (a + b)
  save_ms = 289 * (1 - union_ratio)
report mean / p10 / p90 union_ratio, separately for Qwen and Nemotron
```

Decision rule:
- **mean union_ratio > 0.92** → kill it; the SSD is saturated and there are no bytes to save, full stop.
- **0.80–0.92** → implement the expert-wise K=2 path on a single layer first; measure bytes read via `/proc`-equivalent I/O counters or `fs_usage`, and TPOT. This is the only case with a plausible 25–50 ms/token win.
- **< 0.80** → promising, but check draft acceptance: if the cheap drafter's greedy match rate is < ~70%, expected accepted tokens/verification ≈ p_accept will eat most of the saving.

Also test the same stat with gap 2, 3, 4 tokens: if the union saturates slowly, a longer window could give more sharing — but Specter's "expert explosion" data suggests it does not, and MTP at depth was already rejected.

### 7.3 Two even cheaper sanity checks (do in parallel)

- **Lossless compressibility of 4-bit shards**: `zstd -19` and `lz4hc` on 10 representative ~3 MB expert shards. If ratio > 0.9, ZipMoE's direction is dead for Veystrix (expected: 4-bit codebooks are near-entropy). If ratio < 0.85, revisit "compress on SSD, decompress on the 8 CPU cores while other shards stream" as a bytes-reduction play — note this is *not* palettized Core ML (rejected) and not zero-copy of unaligned shards (rejected).
- **ANE delta-reload floor**: on the M4, time Orion's unload → write-BLOBFILE → reload for one 3 MB expert program. If ~8 ms/kernel as published, one expert swap costs more than the 137 ms token only after ~17 experts — confirms per-token ANE expert binding is impossible (480 swaps/token ≈ 3.8 s).

---

## 8. Final verdict

Under the bit-identical constraint and with a saturated 3.3 GB/s SSD on a 16 GB M4:

- **Nothing published in 2025–2026 reduces Qwen's 289 ms expert-read wait or Nemotron's 75 ms expert wait as a drop-in.** The papers that report big numbers either (a) change routing, (b) change weights/train a drafter, (c) assume spare I/O bandwidth, or (d) run on a QNN-class NPU with runtime weight binding that ANE does not have.
- **The only remaining candidate is the read-*sharing* mechanism** (expert-wise verification across 2 draft tokens), which is bit-exact in principle and was not part of the rejected overlap/MTP experiments. Its entire fate is decided by one offline statistic: the consecutive-token expert-set union ratio. Run §7.2 before writing any code.
- If that ratio is high, the correct engineering conclusion is that **the workload is at the storage bandwidth floor**: 480 experts/token × 3 MB × (1−0.24) ≈ 1.1 GB/token, and 1.1 GB / 3.3 GB/s ≈ 333 ms, which is essentially the measured 289 ms. No scheduler can beat that; only fewer bytes (different model/quantization/precision, or relaxed output equivalence) can.

## ACTIONABLE FOR VEYSTRIX 16 GB

1. **Run §7.2 first, offline, today.** Compute the consecutive-token expert-set union ratio over ~200 recorded greedy tokens for both models. This single number decides whether any bit-exact read-sharing scheme is possible. Decision thresholds are in §7.2; do not implement anything until this returns.
2. **If union_ratio > 0.92: stop.** The SSD is at its bandwidth floor (≈1.1 GB of misses per token / 3.3 GB/s ≈ 330 ms ≈ the measured 289 ms). No 2026 scheduler, predictor, cache policy or ANE trick can improve it while keeping bit-identical outputs. Record this as a hardware limit, not a software bug.
3. **If union_ratio < 0.92:** implement expert-wise (not token-wise) K=2 verification in software on the CPU/Metal path only — group rows by expert, load each expert once, accumulate in baseline expert order. Keep greedy accept/reject. Compare TPOT and bytes-read against baseline; this is the only untested mechanism with a plausible bit-exact win.
4. **Do not pursue:** ANE runtime weight binding (Orion's delta reload is ~8.3 ms/kernel; ANE bakes weights at compile time; Anemll Forge is M5+/32 GB only), ZipMoE lossless compression on 4-bit weights (entropy argument is BF16-specific), S2-MoE / cache-aware routing / BigMoMo expert pruning (all change token IDs), DraftExpert (needs trained per-layer experts, verify still >1× one token), FlashMoE (rejected policy class).
5. **Only if bit-exactness is ever relaxed:** Skliar et al.'s cache-prior reranking is the highest-value revisit — training-free, ~5 lines at the router, >50% miss reduction and 2× measured on a 4-bit Qwen on a 16 GB phone. That is the one result that directly attacks Veystrix's 24% hit rate.

ACTIONABLE FOR VEYSTRIX 16 GB
