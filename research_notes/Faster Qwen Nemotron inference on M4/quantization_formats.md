# Quantization, Weight Formats, Compression & Model-Level Techniques for Faster DECODE/PREFILL of 4-bit MoE on a 16 GB M4 — Rigorous No-Quality-Loss Analysis

> Scope note: Veystrix is SSD-read bound in decode (~809 MB physical reads/token) on a 16 GB M4 with MLX, weights = Q4 affine G32 (top-10/512 MoE, MTP). The hard constraint is bit-identical output (preserve BF16 rounding boundaries and exact dequantized values). "Effectively lossless on benchmarks" is NOT the same as bit-exact and is flagged as quality-risky throughout.

---

## Q1 — Lossless weight formats: which reduce bytes read/token with ZERO quality loss?

### Takeaway
For weights that are **already 4-bit**, general-purpose lossless compression (ZSTD/LZ4, bit-plane disaggregation) yields only ~0.9–2.1% additional savings — essentially useless against an 809 MB/token SSD-read budget. Lossless compression is valuable only on BF16/FP8 payloads (17–25% and ~8–10%). No source supports a bit-exact scheme that materially shrinks already-Q4 expert weights; most MoE "compression" work (delta/SVD/pruning) is lossy.

### Cited Findings
- **RPI paper (arXiv:2503.18869, 2025)** measured lossless compression of *quantized* weights: for INT4 models, ZSTD gives only 1.01× ratio / **0.9%** savings (LLaMA-3.1 8B), 1.02× / **2.1%** (70B), 1.01× / **1.2%** (Mixtral 8×7B), 1.02× / **1.6%** (LLaMA-MoE-3.5B). Reason given: "FP8 and INT4 models, already subjected to lossy quantization, show limited compressibility as the reduced bit precision minimizes representational redundancy." — [RPI 2025](https://arxiv.org/html/2503.18869)
- Same paper, Table I (straightforward lossless, no reformatting): ZSTD on native BF16 weights saves **17.3–23.0%** (Mistral 7B 17.3%, OPT 13B 19.4%, LLaMA-3.1 8B 20.6%, Mixtral 8×7B 21.3%, Gemma-2 2B 23.0%); LZ4 saves 0–18%. — [RPI 2025](https://arxiv.org/html/2503.18869)
- With their **bit-plane disaggregation** transform (group same-position bits into planes), BF16 weights reach 1.34× / **25.2%** lossless savings; FP8 1.09× / 8.3%; INT4 1.01× / 0.9%. The transform is implemented in a custom on-chip memory controller (7 nm, 32 lanes, ~2 TB/s, <5.69 mm²), not software on an M4. — [RPI 2025](https://arxiv.org/html/2503.18869)
- Same work: KV cache is far more compressible with bit-plane + cross-token de-correlation (44.8–46.9% reduction; up to 2.69× ZSTD per layer) — but this targets the KV cache, not weights, and needs custom hardware. — [RPI 2025](https://arxiv.org/html/2503.18869)
- Decompression throughput is NOT the bottleneck in principle: ZSTD decompresses at roughly 1–3 GB/s per core and LZ4 ~2.5–3.7 GB/s on commodity CPUs, far above the ~0.8 GB/s SSD read rate implied by 809 MB/token at 2.3 tok/s — but this only helps if the compression ratio is large, which it is not for Q4. — [Zstd/LZ4 speed table](https://mamonas.dev/posts/which-compression-algorithm-saves-most); [OpenZL benchmarks](https://github.com/boris-chu/go-openzl/blob/main/documentation/BENCHMARKS.md); [HN LZ4-vs-ZSTD](https://news.ycombinator.com/item?id=36826207)
- M4 Pro memory bandwidth is ~273 GB/s (Apple figure); a Mac's CPU/GPU share one memory pool. For Llama-3-8B, tokens/sec ceiling ≈ bandwidth/weight-size: FP16 ~<6 tok/s, Q4_K_M ~18 tok/s — i.e. Q4 file size is the dominant decode lever. — [Apple 2024](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/); [EdgeLayer GGUF](https://edgelayer.substack.com/p/no-gpu-required-gguf-llamacpp-and)
- MoE **expert redundancy/dedup** directions are mostly LOSSY: D2-MoE (ICML 2025) merges each expert into a shared base via Fisher info + SVD-compressed deltas; MoE-SVD (ICML 2025) claims 60% compression / 1.5× speedup via SVD (no retraining); CAMERA / GS-MoE (2025) exploit cross-matrix / subspace redundancy with SVD + pruning. None claim bit-exact dequantized values. — [D2-MoE](https://icml.cc/virtual/2025/poster/43461); [MoE-SVD](https://icml.cc/virtual/2025/poster/44786); [CAMERA](https://arxiv.org/html/2508.02322); [GS-MoE](https://arxiv.org/html/2610.32009)
- A secondary source claims "codebook" lossless packing gives 10–30% RAM reduction via bitwise generic packing of indexed values; this is a blog, not a peer-reviewed result, and is unconfirmed for LLM Q4 weights. — [Medium blog](https://bigattichouse.medium.com/codebook-lossless-llm-compression-10-25-ram-reduction-with-bitwise-generic-packing-of-indexed-c35ba49fc2b8)

### Inferences
- The decisive fact for Veystrix: ~1–2% lossless savings on Q4 cannot move an SSD-read-bound 809 MB/token workload meaningfully. Bit-exact on-disk compression of the stored Q4 weights is a dead end.
- The bit-plane work is useful mainly as evidence that losslessness ≠ bandwidth savings when data is already low-precision, and that real gains need custom hardware not available on an M4.
- Because decode speed ≈ bandwidth / active-weight-bytes, the only bit-exact decode levers are (a) eliminate SSD paging so reads come from unified memory at ~100+ GB/s, and (b) reduce *active* weight bytes read per token (see Q4 on active experts). Prefix caching (Q4) helps prefill bit-exactly.

### Gaps
- No primary source quantifies zstd/lz4 decompression throughput specifically on an M4/M4 Max (OpenBenchmarking has M4 zstd entries but raw decompression GB/s was not retrieved).
- No source found for lossless **nibble reordering**, **delta coding**, or **zero-weight sparse encoding** that preserves bit-exact dequantized Q4 values with measured ratios on real MoE weights.
- No rigorous study of **deduplicating repeated expert weights across layers** in modern MoEs (Qwen/Nemotron); whether such exact duplicates exist at all is unverified.

---

## Q2 — Quality impacts of GPTQ / AWQ / SmoothQuant / FP8 / INT4 / GGUF K-quants / EXL2 / SpQR ("effectively lossless"?)

### Takeaway
Every one of these changes the stored or effective weight/activation values and is therefore **lossy vs bit-exact**. Several are "near-lossless" (≤1% perplexity) on easy metrics, but Apple's own LLM-KICK work shows perplexity is a poor proxy and masking of knowledge/coding regressions is common. None are acceptable under a strict bit-identical-output constraint.

### Cited Findings
- **SpQR (Dettmers et al., ICLR 2024, arXiv:2306.03078):** isolates outlier weights at higher precision, rest at 3–4 bits; <1% relative perplexity loss, >4× memory compression, **15% speedup**, runs 33B on 24 GB "without performance degradation." Reported deltas: 7B ≈3.5 bpw/0.1% ppl, 13B 4.63 bpw/0.2%, 30B 4.69 bpw/0.3%, 65B 4.66 bpw/0.8%. — [SpQR paper](https://arxiv.org/abs/2306.03078); [alphaxiv](https://alphaxiv.org/abs/2306.03078)
- **GPTQ vs AWQ vs GGUF (third-party benchmark):** AWQ retains ~95.2% of FP16 MMLU, GPTQ (g64) ~94.8%, GGUF Q4_K_M ~91.5% (4× RTX 3090 setup). — [llm-inference-benchmarks](https://github.com/cap10bad/llm-inference-benchmarks)
- **GPTQ failure mode:** "GPTQ collapses on code, not on perplexity. GPTQ's perplexity is fine — basically tied with the others." — [Towards AI test](https://pub.towardsai.net/i-tested-gguf-vs-awq-vs-gptq-the-fastest-4-bit-collapses-on-code-at-46-d65c271d7cdf)
- **AWQ:** ~0.4 pt accuracy loss on an internal regression gate; generally better than naive GPTQ at 4-bit; protects salient channels by activation magnitude. — [Lamplit Labs](https://blogs.lamplitlabs.com/posts/quantization-gguf-vs-awq-vs-gptq-quality-latency); [Sitepoint](https://www.sitepoint.com/quantized-local-llms-4bit-vs-8bit-analysis/)
- **SmoothQuant (MIT Han Lab, ICML 2023):** W8A8 quantization for Llama-1/2/3, Falcon, Mistral, Mixtral "with negligible loss"; naive W8A8 fails on OPT-175B. Still changes both weights and activations to 8-bit — lossy vs bit-exact. — [MIT Han Lab](https://hanlab.mit.edu/projects/smoothquant); [arXiv:2211.10438](https://arxiv.org/html/2211.10438v6)
- **FP8:** in RPI 2025, FP8 quantization gives 50% lossy savings plus only ~8.3% extra lossless; FP8 changes values and typically needs per-tensor/per-block scaling — lossy. — [RPI 2025](https://arxiv.org/html/2503.18869)
- **GGUF (Llama-3-8B, llama.cpp numbers):** Q4_0 = 4.66 GB, +0.47 perplexity; Q4_K_M = 4.92 GB, +0.175 PPL (with a WikiText imatrix, +0.151). Q4_K keeps nested 6-bit page scales/minima; total 4.5 bits/weight. Q4_K_M promotes attention-value and FFN-down tensors to Q6_K. — [EdgeLayer / llama.cpp](https://edgelayer.substack.com/p/no-gpu-required-gguf-llamacpp-and); [llama.cpp perplexity README](https://github.com/ggml-org/llama.cpp/blob/b2786/examples/perplexity/README.md)
- **imatrix effect (llama.cpp):** Q4_K_M PPL delta 0.175→0.151, Q3_K_M 0.66→0.50, Q2_K 3.52→2.42 — better scale selection reduces but does not eliminate loss. — [EdgeLayer](https://edgelayer.substack.com/p/no-gpu-required-gguf-llamacpp-and)
- **Newer PTQ methods (all lossy):** FlatQuant (ICML 2025) <1% accuracy drop at W4A4 on Llama-3; SpinQuant (ICLR 2025) W4A4KV4 only 2.9 pt gap to FP16; LeanQuant / NestQuant / GuidedQuant improve 4-bit quality but still quantize. — [Rapid-MLX survey](https://github.com/raullenchai/Rapid-MLX/blob/main/docs/inference-efficiency-papers-2025.md)
- **Apple's own caution (LLM-KICK, arXiv:2310.01382):** "perplexity fails to capture subtle change in true capabilities"; all pruning methods suffer significant degradation on knowledge-intensive tasks even at 25–30% sparsity; quantization more successful but still not equivalent. — [Apple ML Research](https://machinelearning.apple.com/research/compressing-llms)
- **Numerical nondeterminism (NeurIPS 2025, arXiv:2506.09501):** bf16 can cause up to 9% accuracy variation; LayerCast (store 16-bit, compute FP32) restores determinism — highlights how sensitive outputs are to rounding boundaries. — [Rapid-MLX survey](https://github.com/raullenchai/Rapid-MLX/blob/main/docs/inference-efficiency-papers-2025.md)
- **EXL2:** named in multiple format explainers (e.g. Q4/GPTQ family) but no concrete perplexity/accuracy numbers were found in this research. — [local-llm.net](https://www.local-llm.net/learn/quantization-explained/)

### Inferences
- "Effectively lossless on benchmarks" methods (SpQR, AWQ, GGUF Q4_K_M, FlatQuant) all accept some measurable error; they cannot guarantee bit-identical greedy tokens.
- Even BF16 compute-order changes (LayerCast finding) perturb outputs — consistent with the project's observation that recompiling GDN changed rounding.
- Verdict for the hard constraint: **reject all of GPTQ, AWQ, SmoothQuant, FP8, new INT4 schemes, SpQR, GGUF K-quants, EXL2 as not bit-exact.** They remain options only if the quality bar is relaxed to benchmark-equivalence.

### Gaps
- No peer-reviewed EXL2 perplexity/accuracy table located.
- No head-to-head study on Qwen3/Nemotron-style MoEs specifically (most 4-bit quality data is Llama/Mistral dense).

---

## Q3 — Bit-exact lower-read-cost techniques (compact scales/zeros, merged affine params, dequant-preserving K-quants)

### Takeaway
The MLX affine Q4 format is already near-minimal; its dequantization math must be reproduced exactly, and group size 32 is the quality-preserving choice (g64 roughly doubles perplexity damage). No source demonstrates a scheme that keeps the exact dequantized values while shrinking the read. The only confirmed bit-exact wins are engine-level (async_eval, prefix caching, bit-exact prefill GEMM), not format changes.

### Cited Findings
- **MLX affine dequant definition (official docs, v0.32.4):** per group g, α=max(w), β=min(w), s=(α−β)/(2^b−1), ŵᵢ=round((wᵢ−β)/s); 4-bit values packed LSB-first into uint32 (8 elements per word). "To dequantize the elements of w, we also save s and β (the returned scales and biases)." Bit-exactness therefore requires preserving α/β/s and the same round-to-nearest. — [MLX quantize docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
- MLX supported affine group sizes: 32, 64, 128; bits 1–8, scale type = same as input, with bias. MXFP4/MXFP8/NVFP4 modes exist but are different value representations (E2M1/E4M3, E8M0/E4M3 scales, no bias) — not bit-exact vs affine Q4. — [MLX quantize docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
- **Group-size quality data (silica, Apple Silicon, M3 Max):** 8-bit ~lossless; **4-bit/g64 +17.8% PPL, 4-bit/g32 +8.0% PPL** vs fp16; llama.cpp Q4_K_M +6.2% PPL. Smaller groups preserve quality because scale/min adapt locally. — [silica README](https://github.com/mlelarge/silica)
- **MoE roofline (silica, M3 Max 40-core, 400 GB/s):** Qwen3-30B-A3B 4-bit decodes **110 tok/s** at batch=1, reading ~3B active params/token (≈7× a dense 30B), reaching 51% of usable BW; loads pre-quantized 4/8-bit checkpoint in ~17 GB vs ~60 GB fp16. — [silica README](https://github.com/mlelarge/silica)
- **Engine-level bit-exact wins (silica):** decode == mlx-lm within ~1.5% (~70% of usable BW); `async_eval` +50% at 4-bit; `mx.compile` perf-neutral; ~12% slower than hand-tuned llama.cpp; custom Metal kernels declined because the ~30% gap to ceiling is scale-independent and bounded by Apple's `mx.quantized_matmul`. Ships a `PrefixCache` for single-stream prefix reuse. — [silica README](https://github.com/mlelarge/silica)
- Quantized KV and rotating KV do not compose in mainline MLX (constraint on KV-based prefill/decode tricks). — [silica README](https://github.com/mlelarge/silica)
- **Bit-exact prefill GEMM on Apple Silicon (arXiv:2606.25426, 2026):** a pre-packed direct-AMX fp32 kernel beats every Accelerate fp32 path at all 12 LLM prefill GEMMs (geomean 1.58× over BNNSMatMul, 1.17× over BNNS Graph) with **max-abs-diff = 0** at every shape; BNNS Graph is bit-exact at only 3/12 shapes (error up to 1.4e-3 at the rest). Integrated into llama.cpp fp32 prefill: 291→420 tok/s (**1.44×**, bit-identical) at 128-token prefill; gain narrows to 1.10× at 256 and reverses to 0.82× at 512. On M4+, AMX is replaced by SME; MpGEMM beats Accelerate by 1.23× on M4 Pro (prefill). — [arXiv:2606.25426](https://arxiv.org/html/2606.25426)
- No source found for "storing scales/zeros more compactly while preserving exact dequantized values," "merging affine params into the kernel to cut reads," or a K-quant variant that preserves affine-Q4 dequantized values.

### Inferences
- Veystrix's Q4 affine **G32** is already the quality-correct choice; moving to g64/g128 or to MX/NV formats changes values and violates the constraint.
- A bit-exact "read fewer bytes" path would need the scales/biases (stored as BF16/FP32, part of the read) to be compacted losslessly — but RPI's INT4 result (~0.9–2.1%) suggests negligible total gain even if the affine params are compressible.
- Realistic bit-exact levers are engine/kernel-level, not format-level: (1) ensure the working set fits in unified memory so reads are RAM-speed not SSD-speed; (2) `async_eval`-style latency hiding; (3) prefix caching to skip recomputation/reads on repeated prompts; (4) bit-exact prefill GEMM tuning (M4 SME analogue of the AMX work), noting it only applies to fp32 weights, not the quantized matmul path.
- The project's "F_NOCACHE rejected" and "compile GDN changes rounding" findings are consistent with the literature: bypassing cache/numerical shortcuts risks bit-exactness or hurts reads.

### Gaps
- No published measurement of whether MLX's per-group scales/biases are compressible losslessly, nor of their exact share of the 809 MB/token.
- No bit-exact scale-compaction scheme with concrete ratios found.
- Prefill speedup specifically on 4-bit MoE in MLX (as opposed to fp32 GEMM) is unquantified here.

---

## Q4 — Model-level techniques: early exit / layer skipping / dynamic depth / fewer experts / smaller variants

### Takeaway
These can give ~1.2–1.5× (occasionally more) but all alter computation vs the original model; most need training/distillation and report small-but-nonzero accuracy drops. They do NOT preserve bit-identical outputs. Switching to smaller MoE variants (e.g. Qwen3-30B-A3B) gives the largest speedups (64–110 tok/s on Apple Silicon) but is a different model, not the same quality.

### Cited Findings
- **ZEDA ("Post-Trained MoE Can Skip Half Experts via Self-Distillation," 2026):** adds zero-output experts to the router of a post-trained MoE, then self-distills with the frozen original as teacher (SFT + on-policy) plus a group auxiliary loss. On Qwen3-30B-A3B (128 experts, top-8) and GLM-4.7-Flash (64 experts, top-4) it replaces **51–53% of expert activations with zero experts**, gives **~1.20× end-to-end speedup**, "marginal average accuracy drop" over 11 benchmarks. Explicitly NOT pruning/naive top-k reduction; original experts are kept. — [To Data & Beyond / ZEDA](https://todatabeyond.substack.com/p/why-half-the-experts-in-an-moe-model)
- **MoE++ (ICLR 2025 Oral):** zero-computation experts (zero/copy/constant) skip FFN entirely; **1.1–2.1× expert forward throughput**; requires training/architecture change. — [Rapid-MLX survey](https://github.com/raullenchai/Rapid-MLX/blob/main/docs/inference-efficiency-papers-2025.md)
- **TrimMoE (arXiv:2608.00573):** removes **23% of layer executions** entirely (11% skipped, 12% terminated by early exit); trains lightweight per-layer exit heads + calibrates thresholds — training-assisted. — [TrimMoE](https://arxiv.org/pdf/2608.00573)
- **LayerSkip (Meta, ACL 2024):** training recipe for early exit + self-speculative decoding; improves early-exit accuracy (e.g. 77% at layer 12) but "regression in the final layer reducing accuracy by 3%." — [LayerSkip](https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding); [arXiv:2404.16710](https://arxiv.org/pdf/2404.16710)
- **Routing-aware early exit (MDPI Electronics 2026):** measured end-to-end latency reductions of **53.84% / 10.77% / 34.24%** vs full depth; head invocation depends on router confidence/entropy; needs added LM-head evaluations/training. — [MDPI](https://mdpi.com/2079-9292/15/21/5159)
- **Caveat paper:** "The Diminishing Returns of Early-Exit Decoding in Modern LLMs" (arXiv:2603.23701) — early-exit value is shrinking for modern models. — [arXiv](https://arxiv.org/pdf/2603.23701)
- **Self-speculative decoding (SWIFT, ICLR 2025):** skips intermediate layers as draft, 1.3–1.6×, no draft model/training; Falcon (AAAI 2025) 2.91–3.51× "lossless" with a 2-layer drafter. Speculative decoding is output-preserving in distribution but changes the exact sampling trajectory (greedy token can differ under verification). — [Rapid-MLX survey](https://github.com/raullenchai/Rapid-MLX/blob/main/docs/inference-efficiency-papers-2025.md)
- **Prefix / KV reuse (bit-exact for repeated context):** KVLink (NeurIPS 2025) precomputes per-document KV and reports **96% TTFT reduction**; content-based prefix caching in a Native Apple Silicon inference paper reports up to **28× speedup** on repeated image queries. silica ships a single-stream `PrefixCache`. — [Rapid-MLX survey](https://github.com/raullenchai/Rapid-MLX/blob/main/docs/inference-efficiency-papers-2025.md); [Native LLM/MLLM on Apple Silicon](https://arxiv.org/html/2601.19139v2); [silica](https://github.com/mlelarge/silica)
- **Smaller/faster MoE variants:** Qwen3-30B-A3B 4-bit reaches ~64 tok/s (MLX, quality 79.51% on one HF eval) and 110 tok/s on M3 Max (silica); Qwen3 "speed benchmark" lists 30B-A3B BF16 137.18, FP8 155.55, GPTQ-INT4 31.29 (different GPU setup). vLLM-MLX exposes `--moe-top-k 4` to lower active experts at runtime (quality impact unquantified here). — [HF wolfram](https://huggingface.co/posts/wolfram/819510719695955); [Qwen speed benchmark](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html); [vllm-mlx moe-top-k](https://github.com/waybarrios/vllm-mlx/blob/main/docs/guides/moe-top-k.md)
- **Apple Silicon throughput context:** a Native LLM/MLLM inference paper reports 21–87% higher throughput than llama.cpp across Qwen3-0.6B→Nemotron-30B and 4.3× aggregate at 16 concurrent streams. — [arXiv:2601.19139](https://arxiv.org/html/2601.19139v2)
- MoE routing "tail redundancy" (dropping ~28% of the low-weight routing tail) is discussed in community research but no peer-reviewed quality table was retrieved; naive global top-k reduction is known to hurt. — [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1v678cw/research_moe_routing_tail_redundancy_dropping_28/)

### Inferences
- Under the strict bit-identical constraint, early-exit/layer-skip/fewer-expert methods are inadmissible as-is: they change the function computed, and reported small accuracy drops are unacceptable if any greedy token may change.
- ZEDA (~1.2×, "close" accuracy) and top-k reduction are the closest to Veystrix's top-10/512 setting but still require retraining/distillation and are not bit-exact; treat as quality-risky.
- Bit-exact, model-level-adjacent options that DO fit: prefix caching for repeated prompts (prefill), and choosing a smaller MoE checkpoint only if the product can accept a different model.
- For the SSD-bound decode case, dropping active experts (ZEDA-style) is the only model-level change that reduces bytes/token; but it trades away bit-exactness.

### Gaps
- No Apple-Silicon measured tok/s for early-exit or ZEDA on Qwen3.8-Flash-Next / Nemotron specifically (only Qwen3-30B-A3B, GLM-4.7-Flash).
- No quantified quality/speed table for `--moe-top-k 4` on Qwen3-30B-A3B.
- Prefill-specific numbers for a 4-bit MoE on an M4 (not M3 Max fp32 GEMM) were not found.

---

## Cross-cutting verdict for the report writer

- **Bit-exact / safe:** keep Q4 affine G32 weights unchanged; optimize engine (async_eval-style latency hiding), use prefix caching for prefill, fit working set in unified memory to avoid SSD paging, consider a bit-exact prefill GEMM path for fp32 portions. Expected decode gain from these: engine-level up to ~1.5× in favorable setups, but ~0 from on-disk lossless compression of Q4 (only 1–2% available).
- **Lossy / NOT recommended under hard constraint:** GPTQ, AWQ, SmoothQuant, FP8, SpQR, GGUF Q4_K_M/Q4_0, EXL2, FlatQuant/SpinQuant/SLiM, INT4 requantization of any group size, MXFP4/NVFP4 substitution, expert dedup via SVD/delta, early exit, layer skip, ZEDA/zero-expert, top-k reduction, quantized KV.
- **Key quantitative anchors:** Q4 lossless ZSTD savings ≈0.9–2.1% [RPI 2025]; g32 vs g64 PPL +8.0% vs +17.8% [silica]; Qwen3-30B-A3B 4-bit 110 tok/s at ~3B active/token, 51% BW [silica]; ZEDA 51–53% expert skips → ~1.20× [ZEDA]; bit-exact AMX prefill kernel 1.44× end-to-end [arXiv:2606.25426]; SpQR 15% speedup at <1% PPL [SpQR].
