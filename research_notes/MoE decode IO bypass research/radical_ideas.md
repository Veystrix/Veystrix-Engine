# Radical / unconventional ideas for bypassing MoE SSD I/O wait in bit-exact decode

Scope note: This document assumes the assignment brief's figures — M4 with 16 GB unified RAM, one ~3 GB/s SSD, 408–436 MB of expert reads per token, ~130–157 ms/token of I/O wait, steady-state expert cache needing 10–12 GB, targets 5+ and 15+ tokens/s, and a hard requirement of **bit-identical output** (no quality loss). All arithmetic below that uses these numbers is labelled as derived from the brief, not as an external citation. Conventional caching, prediction, overlap, and compression are treated as exhausted per the brief.

## Ranking of ideas by plausibility × payoff (executive summary)

### Takeaway
The single most promising *genuinely novel* direction is **lossless residual (delta) coding of experts around a shared per-layer base**, combined with **computation on already-compressed Q4 codes** — it is bit-exact, needs no custom SSD firmware, and is the only family found here that plausibly cuts *bytes read* by more than ~2× without hardware changes. The second family is **lossless entropy/grammar-compressed expert codes with compressed-domain GEMV**, but its ceiling is bounded by the empirical entropy of 4-bit codes (~<4 bits/weight). Everything that reduces *arithmetic* (Decode-Late/LUT/T-MAC) is excellent engineering but reduces ~0 I/O bytes; everything that reduces *bytes* by an order of magnitude (in-flash GEMV, computational storage) requires custom hardware and is effectively science fiction for an off-the-shelf M4 + SSD.

### Cited Findings
- SSD offloading is dominated by Flash read energy: ~110.8 pJ/b for SSD path vs 4.2 pJ/b for HBM-only (~26× weight-access energy), 80% of per-token energy for DeepSeek-R1, and prefetching hides latency but **not** energy — [Source](https://arxiv.org/html/2508.06978)
- The same paper shows Flash read energy must fall to ~1/10 current (~10 pJ/b) for MoE-on-SSD to beat a smaller dense model on HBM at low batch — [Source](https://arxiv.org/html/2508.06978)
- PuzzleMoE reports that experts within an MoE layer exhibit weight similarity and exploits it by merging magnitudes with masks and sign patterns — [Source](https://arxiv.org/html/2511.04805v1)
- Lossless grammar-compressed matrices support matrix–vector multiplication in time proportional to the *compressed* size, but the theory is deliberately limited to matrices with sparse / few-distinct-value structure — [Source](https://www.vldb.org/pvldb/vol15/p2175-tosoni.pdf)
- "Impossibility Results for Grammar-Compressed Linear Algebra" proves inner product on richly (LZ-style) compressed vectors needs Ω(n²)-ish time, formally ruling out a generic "zip-inner-product" — [Source](https://proceedings.neurips.cc/paper_files/paper/2020/file/645e6bfdd05d1a69c5e47b20f0a91d46-Paper.pdf)
- "Decode Late, Multiply Once" refactoring is exact for integer arithmetic, cuts multiplications from n to 2^k, but the paper explicitly states that in memory-bandwidth-bound inference "reduced arithmetic does not by itself raise throughput" — [Source](https://arxiv.org/html/2610.31913v1)
- AiF integrates GEMV into flash chips and reports large gains by using internal flash bandwidth instead of external bandwidth — [Source](https://dl.acm.org/doi/full/10.1145/3695053.3731073)
- FaaSMoE shows multi-tenant stateless expert sharing cuts per-tenant memory from 36.25 GB to 12.04 GB with 6 tenants — [Source](https://arxiv.org/html/2604.26881v1)
- M4-generation Apple GPU has **no dedicated matrix datapath** (matmul2d runs on shader cores; fp8 is emulated at 0.94× fp16) — [Source](https://arxiv.org/html/2606.12765)
- M4 ANE: ~19 TFLOPS fp16, ~90 µs dispatch overhead, ~32 MB on-chip SRAM cliff, only 64 KiB of weights resident (larger streamed in 16 DMA chunks) — [Source](https://maderix.github.io/articles/inside-the-m4-ane-part-4); [Source](https://maderix.github.io/articles/inside-the-m4-ane-part-2)

### Inferences
- **Byte-budget arithmetic (from brief):** 408–436 MB at 3 GB/s = 136–145 ms, consistent with the reported 130–157 ms wait. For 15 tok/s (66.7 ms/token total) the I/O component must fall to roughly 30–50 ms, i.e. a **~3–5× byte reduction** (to ~90–140 MB/token) assuming 15–35 ms of non-I/O compute; for 5 tok/s (200 ms/token) only a modest ~1.2–1.5× improvement or better overlap is needed. (Derived from brief figures.)
- **Ranking (plausibility × payoff):** (1) lossless expert-base + residual/delta coding — high plausibility, 2–3× byte payoff; (2) lossless entropy/grammar-compressed Q4 codes + compressed-domain GEMV — high plausibility, ~1.1–1.5×; (3) exact context-hash memoization of MoE FFN calls — high plausibility, unbounded payoff on repetitive text, ~0 on novel text; (4) data-layout / read-amplification tuning — high plausibility, up to ~1.5–2× if amplification exists; (5) persistent/cross-request expert server — high plausibility, ~0 single-user but large multi-tenant; (6) Decode-Late/LUT/T-MAC restructuring — high plausibility, ~0 bytes (compute only); (7) M4 ANE/AMX/SME islands — high plausibility, ~0 bytes; (8) in-flash/computational storage — ~8–10× bytes but requires custom hardware (science fiction for this product); (9) tiny-model full-output prediction — not bit-exact; (10) exact reconstruction of an expert from other experts — mathematically implausible for full-rank weights.
- **Key falsification pattern for every lossless compression idea:** measure the *empirical entropy / zero-density of the actual Q4 expert codes* (or of residuals after subtracting a shared base). If codes are near-maxentropy 4 bits/weight and residuals are dense, the idea dies immediately with a one-hour offline analysis script.

### Gaps
- No source found that reports the empirical bit entropy or cross-expert residual sparsity of Q4/affine-quantized experts in Mixtral/DeepSeek/GLM-class models. This is the decisive unknown for ideas #1 and #2 and must be measured locally.
- No published result demonstrates bit-exact (lossless) expert merging that reduces bytes; all merging papers found (PuzzleMoE, HC-SMoE, D², "Dropping Experts, Recombining Neurons") are lossy.

## Idea 1 — Bit-exact algebraic reformulation: can the matmul read fewer weight bits per token?

### Takeaway
No — the two rigorous bit-exact reformulations found (add-before-multiply/"decode late" and LUT/T-MAC kernels) eliminate *multiplications* and *on-chip decoded-weight buffers*, but they do not reduce the number of weight bits fetched from SSD. A truly lower-bound result exists: computing an inner product on richly compressed vectors is provably hard, and lossless compression only helps if the weights have structure (few distinct values / low entropy). So "compute the dot product from an exact compressed projection of the weights" is possible only in the narrow sense of entropy coding the existing Q4 codes, not in the sense of projecting a full-rank weight matrix.

### Cited Findings
- "Decode Late, Multiply Once" refactoring: for k-bit codes, multiplications and decodes drop from n to ≤2^k via per-code bin sums; exact for integer arithmetic, and with FP32 bins its numerical error is ~10× *below* a conventional FP32 MAC pipeline — [Source](https://arxiv.org/html/2610.31913v1)
- The same paper: "one bin is redundant" — q₀ = T − Σ_{v≥1} q_v, so for k=1 only ai=1 elements need accumulating; declares the arithmetic gain saturates at 1+ρ (ρ = multiplier/adder cost ratio) and explicitly says arithmetic reduction does not raise throughput in memory-bound inference — [Source](https://arxiv.org/html/2610.31913v1)
- Its local-memory reduction is w_d/k (e.g. 64 KiB → 8.8 KiB for k=2, ~19 KiB for k=4), but that is *on-chip decoded-weight* memory; if baseline already stores k-bit codes, scatter-accumulate needs 6–12% *more* memory — [Source](https://arxiv.org/html/2610.31913v1)
- T-MAC (EuroSys 2025) removes dequantization and multiplication via LUTs; on M2-Ultra it reaches 71 tok/s (8 cores) for BitNet-3B and up to 4× throughput / 70% energy reduction vs llama.cpp — [Source](https://arxiv.org/html/2407.00088); [Source](https://dl.acm.org/doi/10.1145/3689031.3696099)
- T-MAC's tables are precomputed from activations and indexed by weight bits — so the weight bits themselves are still fully read; T-MAC is a compute optimization, not an I/O reduction — [Source](https://arxiv.org/html/2407.00088)
- LUT Tensor Core: hardware LUT matmul uses 57.2% of an FP16 Tensor Core's area and is up to 72.2× faster than software LUT-GEMM in GEMM, again a compute/hardware play — [Source](https://arxiv.org/html/2408.06003v3)
- Affine quantization is exactly D(z)=s(z−b); group-wise scales give per-block decode vectors — this structure is what enables the bin-sum identity but does not create redundancy in the stored codes — [Source](https://karlstratos.com/notes/quantization.pdf)
- Information-theoretic analysis of quantized MatMul gives distortion bounds for *lossy* schemes; exact MatMul from fewer bits is not achievable for arbitrary matrices — [Source](https://arxiv.org/html/2410.13780v3); [Source](https://arxiv.org/pdf/2601.17187)
- "Impossibility Results for Grammar-Compressed Linear Algebra": for LZ-family lossless compression, inner product cannot be done in O(compressed size) — [Source](https://proceedings.neurips.cc/paper_files/paper/2020/file/645e6bfdd05d1a69c5e47b20f0a91d46-Paper.pdf)

### Inferences
- Feasibility: high (pure software/kernel change, already open-sourced in T-MAC). Bit-exactness: yes. Byte reduction: ~0 for an already-Q4 model; the "one bin redundant" trick saves 1 of 16 bins (k=4) of *accumulator state*, not SSD bytes. Payoff for I/O wait: negligible alone.
- The only byte-relevant loophole is if the stored weights are *not* already maximally compressed — i.e. entropy coding of the Q4 codes (Idea 2). With group-wise affine scales, the per-group scale/zero metadata also costs bytes; folding that into an exact bin-sum decoder is possible but small.
- **Falsification experiment:** instrument bytes read from SSD for one layer with and without a T-MAC/Decode-Late kernel. Prediction: identical byte count; only time spent after the read changes. If bytes drop, the storage format (not the math) was the cause.
- Value for Veystrix: adopt T-MAC/Decode-Late to shrink the *compute* portion of the token budget, which raises the I/O budget share available for prefetch — but this is an incremental, not a "miracle."

### Gaps
- No measured result on whether these kernels change SSD read *patterns* (sequential vs random), which could indirectly affect read amplification.
- No source quantified the per-group affine scale/zero metadata fraction of total Q4 expert bytes.

## Idea 2 — Lossless compressed-domain expert storage (entropy / grammar coding of Q4 codes)

### Takeaway
This is the most physically plausible route to a genuine byte reduction that is still bit-exact: store the 4-bit expert codes with a lossless entropy/grammar coder whose representation supports GEMV directly. The theory exists (compressed-matrix GEMV linear in compressed size; up to x42 lossless compression / x5 speedup / x90 energy on weight matrices with low entropy) and works best precisely when weights have few distinct values — which is exactly the Q4 case.

### Cited Findings
- Grammar-based lossless compression of real-valued matrices achieves size proportional to k-th order empirical entropy and supports left/right matrix–vector multiplication in time and space proportional to the compressed size — [Source](https://www.vldb.org/pvldb/vol15/p2175-tosoni.pdf)
- Column reordering by a column-similarity score yields up to a further 16% peak-memory reduction during MvM — [Source](https://www.vldb.org/pvldb/vol15/p2175-tosoni.pdf)
- "Compact and Computationally Efficient Representation of Deep Neural Networks" reports up to 42× compression, 5× speedup and 90% energy savings by *losslessly* converting weight matrices to entropy-bounded formats, and notes dot-product complexity then tracks matrix entropy — [Source](https://arxiv.org/html/1805.10692v2)
- Impossibility results show the approach cannot be generalized to rich compressors: LZ-family grammar compression forces Ω(n²)-style inner-product time, so the practical scheme must use restricted (RLE/entropy) coding — [Source](https://proceedings.neurips.cc/paper_files/paper/2020/file/645e6bfdd05d1a69c5e47b20f0a91d46-Paper.pdf)
- DiBA (diagonal + binary factorization) reduces a dense Wx to m+k+n floating-point multiplications with 0/1 mixing, but is explicitly an *approximation* with no claimed speedups — [Source](https://arxiv.org/html/2605.05994v1)
- "Multiplying Matrices Without Multiplying" uses learned projections for approximate MatMul — acknowledged approximate, not bit-exact — [Source](http://proceedings.mlr.press/v139/blalock21a/blalock21a.pdf)

### Inferences
- Feasibility: medium-high — pure host-side format + kernel work, no custom hardware. Bit-exactness: guaranteed by the lossless coder. Byte reduction: bounded above by H(Q4 codes)/4; if the Q4 distribution is non-uniform this could be ~10–35%, likely not the ~3–5× needed for 15 tok/s on its own, but it stacks with Idea 3.
- The discriminator is empirical: quantized weights are 4-bit but their entropy can be substantially below 4 bits if the distribution is peaked (typical for NN weights near zero). A bit-plane + RLE/arithmetic coder that still permits direct accumulation is the concrete design.
- **Falsification experiment:** offline, compute per-expert Shannon entropy of the 4-bit code stream (and of a bit-plane/RLE encoding) for all experts of one MoE layer. If mean entropy ≥ ~3.8 bits/weight or RLE yields <10% savings, abandon. A second check: confirm a round-trip compressed GEMV is bit-identical to dequant-then-MAC over 10^6 random dot products.

### Gaps
- No measured entropy figures for Q4 expert weights in the cited literature; all compression ratios reported are for full-precision or mixed matrices.
- No source confirms that a compressed-domain GEMV kernel exists for *group-wise affine* (per-block scale/zero) Q4 codes specifically.

## Idea 3 — Cross-expert structure: shared base + lossless residual coding (the strongest novel bet)

### Takeaway
If experts in a layer are similar (a premise supported by the expert-merging literature), store one exact base expert per layer and each other expert as a *lossless integer residual* in Q4 code space. Reading an activated expert then costs base + sparse residual rather than a full expert. This is bit-exact by construction (integer codes reconstructed exactly), requires no hardware changes, and can plausibly reduce bytes per token 2–3× — the only idea found that reaches the needed scale while staying physical.

### Cited Findings
- PuzzleMoE establishes that different experts have weight similarity and merges magnitudes under similarity/saliency masks, storing sign patterns and masks to reconstruct an approximation — [Source](https://arxiv.org/html/2511.04805v1)
- HC-SMoE, D², STUN, NAEE, Sub-MoE and "Dropping Experts, Recombining Neurons" all reduce expert count or parameters but are lossy / retraining-based — [Source](https://arxiv.org/html/2511.04805v1); [Source](https://aclanthology.org/2025.findings-emnlp.820.pdf)
- CoMoE analyses "expert parameter merging, knowledge distillation, and parameter sharing decomposition" as a family and reports ~70% memory reduction (15.6 GB → 4.7 GB) for Switch-Base-128, but with model changes — [Source](https://arxiv.org/html/2508.09208)
- The lossless-grammar MvM result requires "a relatively small number of distinct values" to be effective — the same structural condition residual sparsity would provide — [Source](https://www.vldb.org/pvldb/vol15/p2175-tosoni.pdf)

### Inferences
- Feasibility: high for the storage/reconstruction layer (it is just integer delta coding + a base selection algorithm); the hard part is finding a base that makes residuals compressible. Bit-exactness: yes if Δ is coded losslessly (e.g. sparse-index + value, or entropy-coded), unlike all cited merging papers.
- Plausible byte reduction: if residuals are ~20–30% of an expert's size, an expert costs ~0.2–0.3 expert-equivalents to read once the base is resident; for 10 experts/token read → ~2–3× fewer bytes. Storage shrinks too (~2–4×), which may push the whole model under the 16 GB ceiling.
- **Falsification experiment (one script, decisive):** for one MoE layer, pick the expert minimizing total Hamming/absolute deviation, compute Δ_i = code_i − code_base per expert, and measure (a) fraction of exactly-zero codes and (b) entropy of nonzero residual values. If zero fraction < ~50% and residual entropy is near 4 bits, residual coding cannot pay for itself — kill the idea in an afternoon.
- A stricter variant worth testing: residual coding *across layers* or *across adjacent expert IDs* if routing/initialization correlates them.

### Gaps
- No source reports residual/zero-density statistics between experts of any MoE model. The single decisive measurement is missing from the literature.
- No bit-exact expert-merging scheme found; all evidence is from lossy merging, so extrapolation must be validated empirically.

## Idea 4 — Compute where the data lives: in-flash GEMV / computational storage

### Takeaway
The only approach that attacks the physical bottleneck rather than its symptoms: perform the expert GEMV inside the SSD/flash array so that bytes never cross the host interface. Proven feasible in research (AiF, DockerSSD, InstAttention/InstInfer) with large reported gains, but every implementation requires programmable NAND or a computational storage drive — not available on a commodity M4 SSD. This is the highest-payoff, lowest-feasibility "miracle."

### Cited Findings
- AiF (ISCA 2025) integrates GEMV into flash chips, using high internal flash bandwidth rather than limited external bandwidth — [Source](https://dl.acm.org/doi/full/10.1145/3695053.3731073)
- DockerSSD reports up to 2× for I/O-intensive workloads and 7.9× for distributed LLM inference using containerized in-storage processing on an FPGA SSD prototype — [Source](https://arxiv.org/html/2506.06769)
- InstAttention offloads attention + KV cache to Computational Storage Drives and reports up to 11.1× throughput vs FlexGen on long-sequence inference with A6000 — [Source](https://ieeexplore.ieee.org/document/10946721)
- InstInfer-class CSDs exploit internal bandwidth but are "tightly coupled to specialized hardware"; commodity NVMe + GPU Direct Storage is the low-cost alternative — [Source](https://papers.neurips.cc/paper_files/paper/2025/file/4431224d3762aa655f0aee4eaf04ff16-Paper-Conference.pdf)
- Commodity SSD read bandwidth is now very high (e.g. Samsung 9100 Pro: 14 GB/s at ~$0.1/GB), which is why software-only approaches remain competitive — [Source](https://arxiv.org/html/2512.22195v1)
- SSD offloading energy analysis concludes Flash read energy (~102 pJ/b) dominates and that latency hiding cannot fix energy — [Source](https://arxiv.org/html/2508.06978)

### Inferences
- Feasibility for Veystrix as-is: essentially zero — you cannot reflash a commodity SSD controller, and Apple's M4 does not expose a programmable CSD path. Feasibility as a *product bet*: requires shipping custom hardware or a CSD, which a first-mover commercial product could in principle do.
- Bit-exactness: yes (the GEMV arithmetic is unchanged; AiF explicitly targets in-flash GEMV). Byte reduction: up to ~perfect — only activations and small outputs cross PCIe (orders of magnitude less than full experts).
- **Falsification experiment:** this cannot be tested on current hardware; the falsification gate is a procurement question — can Veystrix obtain a CSD/programmable-SSD with ≥2 GB/s internal GEMV throughput and documented integer/bit-exact accumulation? If no, the idea is science fiction for this product.
- A software-spirit cousin that *is* testable: check whether the M4's ANE can act as a "prefilter island" (Idea 7) to realize part of the near-data benefit without new storage hardware.

### Gaps
- No source found for in-storage *MoE expert GEMV* specifically (ISP work targets attention/KV or generic analytics).
- No pricing/availability data for CSDs suitable for a commercial M4-class product (Samsung SmartSSD pricing cited as ~$2,526 used in one appendix — [Source](https://papers.neurips.cc/paper_files/paper/2025/file/4431224d3762aa655f0aee4eaf04ff16-Paper-Conference.pdf)).

## Idea 5 — Lossless memoization of MoE FFN outputs; repeated prompts, tool defs, and n-grams

### Takeaway
The bit-exact version of "predict the whole output on a tiny model" is **exact memoization keyed by the full context state**: if a (layer, token hidden state / KV-state hash) recurs, the expert output is mathematically identical and need not be recomputed or re-read. This cannot help novel text but can be decisive for repeated system prompts, tool definitions, and self-repetition loops. Predicting the output with a small model and *not verifying* is not bit-exact and is excluded here.

### Cited Findings
- IMPRESS reduces TTFT up to 2.8× by loading only important prefix KVs, reporting <0.2% accuracy change — i.e. *not* bit-exact, and it targets prefill, not decode — [Source](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf)
- MatKV materializes KV caches on SSD for RAG; it reports slight accuracy *deviations* from concatenating independently computed KVs and leaves exactness as future work — [Source](https://arxiv.org/html/2512.22195v1)
- KVShare reuses KV cache across requests but conditionally recomputes a portion, hence approximate — [Source](https://arxiv.org/html/2503.16525)
- IBM reports large TTFT speedups from persistent shared KV cache (23.6× for 128k context with DRAM caching) — [Source](http://research.ibm.com/blog/accelerating-ai-inference-with-ibm-storage-scale)
- Speculative-decoding approaches hide offloading latency but still verify with the big model, which still reads experts — [Source](https://arxiv.org/html/2508.21706); [Source](https://arxiv.org/html/2510.10302v2)

### Inferences
- Feasibility: high (a hash table over hidden states / KV fingerprints plus a persisted value store). Bit-exactness: yes *if and only if* the key includes every input to the expert function (hidden state, router scores, layer id); attention state must be part of the key, which is why only exact-prefix or exact-suffix repeats hit.
- Byte reduction: unbounded on workloads with repeated long prefixes/templates or generation loops (entire layers skipped), ~0 on fresh prose. For a product with fixed system prompts/tool definitions, the prefill phase (not decode) is the main winner; decode wins only on loops.
- Tiny-model full-output prediction is rigorously excluded: without verification, any approximate draft can differ by one bit; with verification, the big model's weights must still be read. No source found claiming bit-identical output from a predictive shortcut.
- **Falsification experiment:** log the fraction of decode steps whose (layer, hash of hidden state + KV fingerprint) is an exact repeat over 1,000 real requests. If <1%, the memoization idea is dead for decode; keep it only for prefill prefix reuse.

### Gaps
- No public measurement of exact hidden-state/KV-state repeat rates in real assistant workloads (tool-use templates, repeated system prompts).
- No source on whether expert outputs are stable enough to key by a *compressed* fingerprint without collisions (would need a cryptographic-strength fingerprint to preserve bit-exactness).

## Idea 6 — Persistent expert server / cross-request expert sharing

### Takeaway
Because expert layers are stateless, they can be disaggregated into a long-lived service that keeps a shared expert pool in RAM across requests and users. This is bit-exact (no model change) and is published (EaaS, FaaSMoE), but for a single-user 16 GB M4 product it changes little: the working set that exceeds RAM is unchanged. It becomes powerful only if Veystrix is a multi-user/server product or keeps a daemon alive across sessions.

### Cited Findings
- EaaS disaggregates stateful attention from stateless MoE experts, uses async peer-to-peer expert servers and replication; experts are "pure, idempotent" functions — [Source](https://arxiv.org/pdf/2509.17863)
- FaaSMoE: multi-tenant stateless expert blocks shared across tenants; per-tenant CPU 187.81% → 54.4%, memory 36.25 GB → 12.04 GB with 6 tenants; block size ~20 experts is the sweet spot — [Source](https://arxiv.org/html/2604.26881v1)
- Diff-MoE exploits global vs temporal expert locality with a differential cache and cross-layer prediction, reporting 2.74×/2.22×/1.55× throughput over DeepSpeed/Pre-gated/MoE-Infinity — [Source](https://dl.acm.org/doi/10.1145/3712285.3759903)
- MoE-CORE assigns separate cache regions to prefill and decode and uses workload-domain hints, reporting up to 1.75× TPOT on DeepSeek-V4-Flash-W4A8 — [Source](https://arxiv.org/pdf/2610.01950)

### Inferences
- Feasibility: high (architecture/daemon work). Bit-exactness: yes. Byte reduction: near-zero for one user on one machine; for N concurrent users hitting overlapping expert sets, SSD reads can be amortized N-way — exactly the FaaSMoE effect.
- This does not solve the *within-token* 408 MB read; it only avoids re-reading the same expert across requests. If the brief's steady-state 10–12 GB working set already fits the service's larger RAM, the SSD can be removed from the hot path entirely — a real product win.
- **Falsification experiment:** measure the inter-request overlap of activated expert sets for Veystrix's actual traffic. If Jaccard overlap between consecutive requests is < ~20%, a shared pool provides no amortization.

### Gaps
- No data on expert-activation overlap for the specific model/workload Veystrix uses.
- No source on latency/overhead of an on-machine expert daemon over local IPC on macOS (FaaSMoE uses HTTP/FaaS with much larger overheads).

## Idea 7 — M4 hardware islands (ANE / AMX / SME / GPU texture) as bit-exact prefilters

### Takeaway
The M4 has substantial matrix hardware, but none of it reduces SSD bytes directly; its value is shrinking the non-I/O part of the token budget and acting as a bit-exact compute island. Crucially, M4-generation GPUs lack dedicated matrix units (those arrive with M5), and the ANE is a fixed-function, high-dispatch-overhead, closed-ISA engine best suited to prefill/deep graphs, not single-token decode.

### Cited Findings
- M4 ANE: ~19 TFLOPS fp16, ~90 µs dispatch, ~32 MB SRAM cliff, 64 KiB resident weight budget, single ops use ~30% of capacity; conv expressed as 1×1 is much faster than matmul — [Source](https://maderix.github.io/articles/inside-the-m4-ane-part-2); [Source](https://maderix.github.io/articles/inside-the-m4-ane-part-4)
- Recommended split: ANE for prefill (large batch, deep graphs), CPU SME for single-token decode (zero dispatch overhead) — [Source](https://maderix.github.io/articles/inside-the-m4-ane-part-2)
- AMX is most efficient for FP32/FP64 (M4 Pro: up to 68% of GPU FP32 at 42% power); ANE exceeds 700 GFLOP/W at low precision under batched workloads — [Source](https://www.sciencedirect.com/science/article/pii/S0167739X26000270)
- Draw Things shipped ANE matmul invocations from its own runtime, reporting up to 1.8× on M4 and ~22 TFLOPS with dynamic int8 matmul via public CoreML APIs on macOS 26 — [Source](https://engineering.drawthings.ai/p/making-apple-neural-engine-work-in)
- M4 GPU: matmul2d executes on shader cores with no dedicated matrix datapath and no ANE routing; fp8 is emulated (0.94× fp16); accumulator ≥ fp32 — [Source](https://arxiv.org/html/2606.12765)
- M4 GPU GEMV reaches only ~187 GFLOPS FP16 (~1.3% of peak) vs ~14.2 TFLOPS GEMM — [Source](https://repository.cern/records/54saz-aa871/files/fastML_2025_dimitrios_danopoulos.pdf)
- Neural/texture accelerator matrix paths (4.4× vs ALU) are an M5 feature; "whether it is usable on something like an M4 Max is untested" — [Source](https://syllogi-graphikon.vercel.app/posts/metal-neural-texture-compression)
- ANE op support is a fixed published-ish matrix: convolution/matmul native, texture-engine ops need M2+, top-k/sort need M2+, 37 ops rejected on all families — [Source](https://ane-guide.readthedocs.io/en/latest/appendices/a-op-device-matrix.html)

### Inferences
- Feasibility: high for SME/AMX (inline, deterministic), medium for ANE (dispatch cost, closed ISA, CoreML partitioner opacity — one report notes "MLComputeUnitsAll is a wish" and that ANE tests can pass while running elsewhere). Bit-exactness: can be made exact by keeping accumulation in ≥FP32 / integer and verified against the CPU reference.
- Byte reduction: zero. The payoff is a smaller compute term in the token budget, which matters if the post-read matmul is a meaningful fraction of the ~130–157 ms; given GEMV is ~1.3% of GPU peak, compute is likely small, so this cannot be the miracle.
- A genuine "bit-exact island" use: route the *gate/router* and *shared expert* (always-read, small matrices) to ANE/SME so the SSD read path can be scheduled more aggressively — engineering, not a new paradigm.
- **Falsification experiment:** microbenchmark a single expert GEMV on CPU SME vs ANE vs GPU for the actual expert shape; if ANE dispatch (~90 µs) exceeds the GEMV compute time, ANE is useless for decode.

### Gaps
- No source quantifies M4 GEMV time for the actual Veystrix expert dimensions; the ~187 GFLOPS figure is for a different shape.
- No public documentation of the ANE ISA (explicitly closed), so any ANE route carries long-term product risk — [Source](https://hiraditya.github.io/posts/apple-has-the-silicon-not-the-language)

## Idea 8 — Data layout / "following the data": 4D ordering, Hilbert curves, read-amplification elimination

### Takeaway
If the logical 408–436 MB/token maps poorly to NAND pages (random 4 KiB reads, bad locality), the *physical* bytes read can exceed the logical bytes. Ordering expert tensors along the actual read trajectory (layer → activated expert → gate/up/down projection) and packing with space-filling curves is bit-exact and purely software, but its payoff is bounded by whatever amplification currently exists.

### Cited Findings
- HiFC measures sustained GDS I/O: sequential reads ~4.99 GiB/s but random writes ~1.6 GiB/s; pSLC region matters and TLC migration degrades throughput — evidence that SSD I/O character strongly depends on access pattern — [Source](https://papers.neurips.cc/paper_files/paper/2025/file/4431224d3762aa655f0aee4eaf04ff16-Paper-Conference.pdf)
- SolidAttention finds finer-grained partitioning increases random I/O overhead and reduces bandwidth utilization, while larger blocks plateau once SSD bandwidth is saturated — [Source](https://www.usenix.org/system/files/fast26-zheng.pdf)
- IMPRESS shows disk I/O can be 51–98% of TTFT and motivates I/O-efficient identification of needed data — [Source](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf)
- Column reordering for compressed matrices yields up to 16% memory reduction during MvM — a directly analogous layout lever — [Source](https://www.vldb.org/pvldb/vol15/p2175-tosoni.pdf)

### Inferences
- Feasibility: high (offline repacking + sequential read path). Bit-exactness: trivially yes. Byte reduction: equal to the current amplification factor; if reads are already aligned/paged, payoff ≈ 0. Worth doing as a cheap prerequisite for Ideas 2/3.
- **Falsification experiment:** measure NAND-level read amplification (or simply achieved throughput vs 3 GB/s) for current expert reads. If achieved throughput already ≈ rated sequential bandwidth, layout changes are pointless.

### Gaps
- No source reports read amplification for MoE expert reads on a commodity M4 SSD.
- No source evaluates Hilbert/Z-order layouts for MoE weight tensors specifically.

## Idea 9 — Expert merging / reconstruction from other experts / X-LoRA-style mixing (bit-exactness analysis)

### Takeaway
There is a large 2024–2026 literature on merging, dropping, and LoRA-mixing experts, and it robustly reduces memory — but every method found is lossy or requires retraining/model changes. No source reports exact reconstruction of one expert's weights from others, which would require linear dependence among full-rank weight matrices — mathematically implausible. These ideas are useful as *evidence* for Idea 3 (experts are similar) but cannot themselves satisfy the bit-identical requirement.

### Cited Findings
- PuzzleMoE merges experts fine-grained (25%/50% expert reduction) with saliency masks, sign patterns and bit-packing, accepting approximation — [Source](https://arxiv.org/html/2511.04805v1)
- "Dropping Experts, Recombining Neurons" is retraining-free pruning that recombines neurons, not exact — [Source](https://aclanthology.org/2025.findings-emnlp.820.pdf)
- X-LoRA dynamically mixes LoRA experts with token/layer scalings and requires a dual forward pass and training/fine-tuning of adapters — [Source](https://pubs.aip.org/aip/aml/article/2/2/026119/3294581/X-LoRA-Mixture-of-low-rank-adapter-experts-a); [Source](https://huggingface.co/docs/peft/en/package_reference/xlora)
- HMoRA and EWoRA likewise require training and change the model — [Source](https://proceedings.iclr.cc/paper_files/paper/2025/file/e43a33994a28f746dcfd53eb51ed3c2d-Paper-Conference.pdf); [Source](https://aclanthology.org/2025.findings-ijcnlp.108.pdf)
- "Importance-driven expert scheduling" substitutes low-importance active experts with functionally similar cached ones, preserving "nearly lossless" — explicitly not bit-exact — [Source](https://arxiv.org/html/2508.18983v2)
- MoBiLE reduces the number of experts for "unimportant" tokens with "negligible degradation" — not bit-exact — [Source](https://arxiv.org/html/2510.12357)

### Inferences
- Feasibility of bit-exact variant: very low. Bit-exactness: no for all cited methods. Byte reduction: large but irrelevant under the constraint.
- The one legitimate transfer: reuse their *finding* that experts are similar to motivate lossless residual coding (Idea 3), and reuse their saliency metrics to choose a good base expert.
- **Falsification experiment (for the exact-reconstruction sub-idea):** compute the rank of the stacked per-expert weight matrices for one layer; if rank equals the number of experts × matrix rank (i.e. no dependence), exact reconstruction from peers is impossible.

### Gaps
- No paper found attempting bit-exact expert merging; this appears to be a genuine open niche.
- No quantitative cross-expert similarity statistics in a form usable for lossless coding (magnitudes/signs reported, not Q4-code residuals).
