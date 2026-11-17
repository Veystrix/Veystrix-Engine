# Second miracle hunt — genuinely novel lossless bytes/token reductions for MoE decode on a 16 GB M4

Scope: hard constraints — no second SSD/Mac/RAM, **bit-identical dequantized weights and exact routing**. Baseline (coordinator's measured facts): Qwen3.8-Flash-Next 512 exp/layer, top-10, 48 layers, Q4 affine G32, ~873 MB physical/token, expert = 3,072,000 B; Nemotron 3.5 Lightning 128 exp/layer, top-6, 23 MoE layers, Q4 affine G64, ~857 MB/token, expert = 5,612,544 B (2× 2688×1856 at ~4.5 b/w; BF16 scale+bias ≈ 20% Qwen / ≈ 11% Nemotron of bytes). E0 overlap works (3.36 GB/s, 10% inflation); E1 perfect oracle prefetch only +25% — bytes are the only lever. Targets: Qwen ≥5 tok/s (≤~672 MB/token at 3.36 GB/s), Nemotron ≥15 tok/s (≤~225 MB/token, ~4× cut). Prior notes already closed: plain code-stream compression (nibble entropy 3.86/4, whole-artifact ceiling ~1.04–1.05×, dictionary zstd expands) — see `../MoE decode IO bypass research/lossless_representation.md`. This note reports only *new* angles not already in that file; every external fact is sourced, every projection is labelled inference.

---

## Key Question 1: Can the BF16 scale/bias stream be stored more compactly bit-exactly — and does this reconcile ZipMoE/DFloat11 with our Q4 codes?

### Takeaway
**Yes, partially, and the apparent ZipMoE conflict dissolves.** The Q4 *code* nibbles have no exponent field, so ZipMoE/DFloat11 cannot compress them — but the per-group `scale` and `bias` in MLX affine Q4 are stored in the input dtype (BF16/FP16), and *those are exactly the exponent-bearing fields ZipMoE/DFloat11/NeuZip/ZipServ target*. Huffman/triple-bitmap on the exponent bits of the scale stream is bit-exact on the stored parameters, hence `s*q+b` is unchanged. Realistic total saving ≈ 6% of Qwen bytes / ≈ 3% of Nemotron bytes — real but far short of a miracle. A 2026 fused-decode design (tile-level ANS) is the first to make such coding viable at low batch.

### Cited Findings
- MLX affine dequant: per group g, α=max(w), β=min(w), s=(α−β)/(2^b−1), ŵ=round((w−β)/s); "we also save s and β (the returned scales and biases)"; **scale type = input dtype**, supported group sizes 32/64/128. Preserving s,β bit patterns and the same rounding is the exactness condition. — [MLX quantize docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
- DFloat11: Huffman-code the **8 exponent bits of BF16 values**, leave sign+mantissa raw, decode on GPU at transformer-block granularity; ~30% BF16 memory cut, bit-identical output; k (Huffman tree depth) = 4–8; explicitly "lossless". — [DFloat11, NeurIPS'25 poster](https://neurips.cc/virtual/2025/poster/115225); [openreview PDF](https://openreview.net/pdf/6b1607ff4d79f3a474030fbbf74f738e0e5350c9.pdf); [LeanModels/DFloat11](https://github.com/LeanModels/DFloat11)
- ZipMoE targets BF16 exponent fields precisely: exponent-symbol support only 14.06%/11.33%/14.84% of possible values (entropy 2.65/2.56/2.55 bits); ZSTD → 68% of original, LZ4HC → 74%, near Shannon floor 66%; 2× tensor coverage by caching the incompressible SM half. — [ZipMoE](https://arxiv.org/html/2601.21198v1)
- **The conflict is one of scope, not mechanism:** DFloat11 maintainers decline *quantized models* because "an INT8 model might already use more than 7 bits of entropy" — that statement is about the **weight codes**, not about the BF16 scale fields still present inside a quantized expert. — [DFloat11 lossless compression write-up](https://ai-infrastructure.net/dfloat11-lossless-compression/)
- ZipServ (ASPLOS'26) gives a *fixed-length, constant-time* alternative: Tensor-Core-Aware Triple Bitmap Encoding (TCA-TBE) = 3 one-bit planes of a 3-bit exponent codeword per 8×8 tile + top-7 exponent lookup + 8-bit packed sign/mantissa; average 11.3 bits/element vs 16, near the 10.6-bit Shannon floor; top-7 exponent share r₃≈0.96; fused ZipGEMM "load-compressed, compute-decompressed" into tensor-core registers; up to 30% size cut, 2.21× kernel speedup over cuBLAS, 1.22× end-to-end over vLLM; bit-exact. — [ZipServ HTML](https://arxiv.org/html/2603.17435v1); [ASPLOS PDF](https://www.cse.ust.hk/~weiwa/papers/zipserv-asplos26.pdf)
- "Approaching Shannon Bound with Lossless LLM Weight Compression" (Tan et al., 2026) — tile-addressable ANS (rANS), per-layer codebook, each GEMM tile an independently decodable substream, decoded into shared memory and consumed in the same tile order; within 0.01–0.1 bits of Shannon; covers INT4 and group-quantized (AWQ/SQ8); bit-identical inference results; **but** its own entropy study says group-quantized formats retain only **1.1–1.3×** redundancy above their entropy bound (vs "6–10×" for raw INT4/FP4 symbol streams — a scope discrepancy that must be measured locally). — [arXiv:2606.15789](https://arxiv.org/html/2606.15789)
- Independent ceiling cross-check: per-group MLX encoding entropy ~3.7 bits → only ~1.08×; per-tensor ~1.7 bits → 2.4× (the latter changes values). — [drxddy/ecq](https://github.com/drxddy/ecq)

### Inferences
- **Mechanism / losslessness:** offline, split each expert's code bytes from its scale/bias bytes; Huffman (DFloat11-style) or TCA-TBE the exponent bits of the latter only; decompress to the original BF16 bit pattern before dequant. Because the code stream is untouched and s,β are reproduced bit-for-bit, `s*q+b` and routing are unchanged. Bytes/token (inference from coordinator's percentages): scale stream 20%×30% ≈ **6% Qwen** (873→~821 MB), 11%×30% ≈ **3.3% Nemotron** (857→~829 MB). ANS on the scale stream (1.1–1.3× on that substream) gives ~2–6% total — consistent.
- **M4 decompression cost:** negligible in bytes (scale stream is ~1/5 or ~1/9 of the read); decode can be folded into the affine dequant path. The ANS paper's caveat is decisive for Veystrix: at **batch 1 its fused kernel is decoder-bound** and degrades to unfused decode-then-GEMM — so on M4 you should fuse the scale decode with the **SSD→staging read** (which takes ~100 ms), not with GEMM; then decode is fully hidden.
- **Falsifiable 1-script test:** for one expert tensor, split code bytes and scale/bias bytes; histogram the BF16 exponent byte of the scale stream; build a DFloat11-style Huffman tree (k=4–8) and a TCA-TBE top-7 codebook; report (a) bits/scale, (b) total expert bytes saved %, (c) SHA-256 of dequantized expert before/after. Pass if saved% ≥ ~5% (Qwen) with hash equality and exponent support ≤ ~20/256; kill if the scale exponents are ~uniform (support > ~100/256).
- This reconciles the user's flagged conflict: ZipMoE *is* applicable, just to the ~20%/~11% BF16 metadata, not to the ~80%/~89% nibble stream. It cannot by itself reach 5 or 15 tok/s.

### Gaps
- No published measurement of the entropy of MLX affine **scale/bias fields** specifically in Qwen3.8-Flash-Next / Nemotron-Lightning; the 6%/3.3% projection uses DFloat11's generic 30% BF16 saving, not a target-specific measurement.
- Unknown whether Veystrix stores scales as BF16 or FP16 (FP16 exponent Huffman is similar but not identical to DFloat11's BF16 code).
- The Tan et al. 2606.15789 "6–10× for INT4" claim conflicts with the byte-exact 3.86/4-bit result in prior notes; likely explained by counting per-group scale metadata separately, but unresolved. Local measurement required.

---

## Key Question 2: Exact algebraic reformulation / compressed-domain GEMV — can the affine Q4 GEMV be computed from a smaller exact projection of the weights?

### Takeaway
**No.** Two independent theory results close this: (a) exact reformulations (Decode-Late/bin-sum, T-MAC LUT-GEMM) remove *multiplications* and on-chip decoded-weight buffers but read the **same weight bits**; (b) computing an inner product on richly compressed vectors is provably hard (Ω(n²)-ish for LZ-family grammar compression), and "return of allocation-flexible coding to the inference path" is explicitly listed as an **open problem** in the 2026 transform survey. The only thing that survives is entropy decoding *inside* the GEMM tile (tile-level ANS), which reduces bytes only to the code/scale entropy bound (~1.1–1.3× for group-quantized), not to a smaller exact projection.

### Cited Findings
- "Decode Late, Multiply Once": for k-bit codes, per-code bin sums cut multiplies and decodes from n to ≤2^k; exact for integer arithmetic; one bin redundant (q₀ = T − Σ_{v≥1} q_v); **explicitly states that in memory-bound inference reduced arithmetic does not by itself raise throughput**, and scatter-accumulate needs 6–12% *more* local memory when codes are already k-bit. — [arXiv:2610.31913](https://arxiv.org/html/2610.31913v1)
- T-MAC removes dequantization and multiplication via activation-derived LUTs indexed by weight bits; on M2-Ultra 71 tok/s on BitNet-3B, up to 4× throughput / 70% energy vs llama.cpp; **tables are indexed by the weight bits, which are all still read** — compute optimisation, not I/O reduction. — [T-MAC, EuroSys'25](https://arxiv.org/html/2407.00088); [ACM](https://dl.acm.org/doi/10.1145/3689031.3696099)
- "Impossibility Results for Grammar-Compressed Linear Algebra": inner product on LZ-style losslessly compressed vectors requires Ω(n²)-ish time; no generic "zip-inner-product". — [NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/645e6bfdd05d1a69c5e47b20f0a91d46-Paper.pdf)
- Grammar-compressed matrices support GEMV in time proportional to compressed size **only for matrices with few distinct values / sparse structure** — the exact structural condition our near-uniform Q4 codes do not satisfy. — [PVLDB 15, Tosoni et al.](https://www.vldb.org/pvldb/vol15/p2175-tosoni.pdf)
- 2026 survey (200 works, 43 transform methods): "Every deployed LLM GEMM kernel is shared-scale, fixed-allocation"; "Neither per-coefficient bit allocation nor entropy-coded indices are expressible in the operand tile that kernel's matrix instruction consumes; a decoder may run inside the kernel, but what it hands the instruction is dense, equal-width"; contribution (vii) lists **"the return of allocation-flexible coding to the inference path" as an open problem**. — [arXiv:2608.25188](https://arxiv.org/html/2608.25188v1)
- Tile-level ANS is the first concrete realization of that open problem: rANS decode warps write reconstructed tiles into shared memory, pipelined with tensor-core GEMM, double-buffered; each weight decoded once, never reloaded from HBM; bit-exact. — [arXiv:2606.15789](https://arxiv.org/html/2606.15789)
- The coordinator's "matters" reference could not be resolved to a methods paper. A search surfaced a 2025 title "Data Matters: A Statistical Analysis of LLM Tensor Distributions and Quantization Effects" (Kim & Kim), but it appears to be a tensor-distribution/quantization-effects analysis rather than an exact-GEMV algorithm, and no primary source was retrieved. Treat as unverified; it is listed in Gaps, not used as evidence.

### Inferences
- **Exact-losslessness argument:** all surviving schemes are bijections on the stored representation (entropy coder) or exact algebraic reassociations (bin-sum identity); none approximates the weight values. A genuinely *smaller exact projection* of a full-rank weight matrix is impossible without information loss — the inner product depends on every code, so any exact method must read (entropy of) every code.
- **Bytes/token:** entropy-coded GEMV is bounded by H(codes)+H(scales). With H(code)≈3.86/4 and group-quantized redundancy 1.1–1.3×, expect ~1.05–1.3× total (873 → ~670–830 MB). Not a 4× cut.
- **M4 decompression cost:** fused decode at batch 1 is decoder-bound per 2606.15789; M4 has no tensor cores (matmul2d runs on shader cores; fp8 emulated) — [M4 GPU characterization](https://arxiv.org/html/2606.12765) — so the ZipServ "decode into tensor-core registers" trick has no direct M4 analogue; decode into SIMD/SME registers or into the staging buffer instead.
- **Falsifiable 1-script test:** implement decode-late bin sums on one expert; count SSD `read()` bytes with and without the kernel. Prediction: identical bytes, shorter compute only; if bytes drop, the storage format changed, not the math. Separately, round-trip a tile-level ANS-coded expert through fused decode and compare dequantized output bitwise over 10^6 random dot products.

### Gaps
- No published exact/compressed-domain GEMV kernel for **group-wise affine Q4** specifically (the survey's open problem); tile-ANS is demonstrated on CUTLASS/GPU, not Metal/MLX.
- No M4/Metal benchmark of fuse-decode-into-GEMM at batch 1; the only Apple-Silicon proxy is an M2 Pro fused rANS prototype at 29% bandwidth overhead. — [MLX discussion](https://github.com/ml-explore/mlx/discussions/3040)
- "matters" not confirmed — the coordinator may have meant a different paper; no exact-reduction method found under that name.

---

## Key Question 3: Lossless delta/XOR/dedup across experts or layers — can the +0.0004 Pearson result be overturned?

### Takeaway
**No new evidence overturns it; the prior null stands, and the one remaining niche is explicitly unstudied.** Cross-expert/cross-layer delta coding of *already-Q4* codes has no published positive result; all expert-merging/dedup literature is lossy or retraining-based. Bit-exact "expert B = base + residual" remains a genuine open niche but its decisive statistic (Q4-code residual zero density) has never been published.

### Cited Findings
- No exploitable linear redundancy: median Pearson +0.0004 over 250 adjacent same-role bf16 layer pairs; byte-level XOR/delta not found profitable. — [IJCA byte-exact study](https://www.ijcaonline.org/archives/volume187/number120/rotem-ijca-2026-a71cdd2650e6.pdf)
- Intel (2025): 2-bit slices from 4 consecutive FP4 values are "close to uniformly random," "little to no redundancy across tensor blocks"; FP4 values "not suitable for lossless compression"; only FP8 scales compress. — [arXiv:2508.19263](https://arxiv.org/html/2508.19263)
- All merging papers are lossy: PuzzleMoE (magnitudes/masks/sign), HC-SMoE, D², CoMoE (~70% memory) — none claim bit-exact dequantized values. — [PuzzleMoE](https://arxiv.org/html/2511.04805v1); [CoMoE](https://arxiv.org/html/2508.09208)
- Cross-layer tying that *does* create bit-identical experts requires pretraining ("single Python pointer assignment", up to 75% FFN parameter cut); with tied weights and independent routers, top-1 cross-layer agreement is only 0.087–0.126, so one resident expert does not serve two layers. — [Expert tying](https://arxiv.org/abs/2606.16825); [agreement ablation](https://arxiv.org/html/2606.16825v1)
- "No paper found attempting bit-exact expert merging; this appears to be a genuine open niche"; "no source reports residual/zero-density statistics between experts of any MoE model." — [radical_ideas.md (prior note)](../MoE%20decode%20IO%20bypass%20research/radical_ideas.md)

### Inferences
- Exact-losslessness argument (if attempted): store expert_i codes as base XOR Δ_i with Δ_i losslessly coded; reconstruct bitwise; dequant unchanged. But the prior +0.0004 correlation and near-maxentropy Q4 codes predict Δ_i is dense (~4 bits), so Δ_i costs as much as the original — a net loss after indexing.
- Bytes/token: expected **no reduction**; worst case slight expansion. Not worth kernel work.
- Falsifiable 1-script test: for one layer, pick the per-position modal base expert, compute Δ_i = code_i XOR code_base for all experts, report exact-zero fraction and entropy(nonzero Δ). Kill if zero fraction < ~50% or residual entropy > ~3.5 bits (consistent with prior null). Only if it passes should a fused XOR-reconstruct kernel be built.
- This is the correct place to spend 30 minutes offline; it cannot be inferred from literature because the needed statistic does not exist in published form.

### Gaps
- No published Q4-code residual zero-density / entropy for Qwen3.8-Flash-Next or Nemotron-Lightning.
- No evidence of byte-identical duplicate experts in 512-expert layers; a SHA-256 audit of all (layer, expert) tensors is cheap and decisive but not in the notes.

---

## Key Question 4: Bit-plane disaggregation where only needed planes are read, bit-exact — any new result?

### Takeaway
**Closed for bit-exact code reads, with a new twist:** bit-plane splitting is a bijection, so *all* planes must be fetched to reconstruct exact codes — reading only "needed" planes is the lossy dynamic-precision trick (PagedMoE) and fails. The genuinely novel observation is that ZipServ **does** use bit planes — but on the BF16 *exponent codeword* (3 planes per 8×8 tile), where skew makes it pay; on near-uniform Q4 codes it cannot.

### Cited Findings
- Bit-plane decomposition + zstd-19 = 1.043× on Q4_K — no gain over plain zstd. — [IJCA study](https://www.ijcaonline.org/archives/volume187/number120/rotem-ijca-2026-a71cdd2650e6.pdf)
- RPI bit-plane transform: BF16 1.34× / 25.2%, FP8 1.09× / 8.3%, **INT4 1.01× / 0.9%** — implemented in a custom memory controller, not M4 software. — [arXiv:2503.18869](https://arxiv.org/html/2503.18869)
- ZipServ TCA-TBE: each 8×8 tile's 3-bit exponent codeword stored as **three independent 64-bit bitmaps (one per bit-plane)**; branch-free warp-local decoding; 11.3 bits/element vs 10.6 Shannon floor. — [ZipServ](https://arxiv.org/html/2603.17435v1)
- PagedMoE's prefix/bitplane codebook reaches avg 2.5 bits and 8.89 tok/s, but lower-precision prefixes are a lossy approximation of the 4-bit expert — FAILS bit-exactness. — [PagedMoE](https://arxiv.org/pdf/2611.03173)

### Inferences
- Exactness: reassembling all planes is bit-identical; "read only needed planes" cannot be bit-exact for a full-precision result. Therefore the only legal use of bit planes is as a *storage layout* for a skewed substream (ZipServ exponents), not as a partial-fetch scheme.
- Bytes/token: Qwen code planes → ~0% saving (consistent with 0.9–2.1% ZSTD measurement); BF16 scale exponent planes → fold into the ~6%/3.3% of Key Question 1.
- M4 decompression cost: bit-plane reassembly is a few shifts/ORs per 64-bit word — negligible relative to ~100 ms SSD read.
- Falsifiable 1-script test: measure per-bit-plane entropy of the Q4 code stream; verify Σ H(plane) ≈ global nibble entropy (~3.86 bits) and that zstd on the planes ≈ zstd on packed codes. Prediction: no byte saving; if a plane is nearly constant, check whether that is just a sign-bit skew already captured by the nibble entropy.

### Gaps
- No M4 measurement of per-plane entropy for the target models; the cited numbers are from Q4_K and RPI testbeds.

---

## Key Question 5: Cross-token EXACT union sharing — why was K=2 read-sharing 16% slower on Metal, and can a different kernel/layout fix it?

### Takeaway
**Yes — and this is the strongest new byte lever in the whole hunt, because it overturns the prior pessimistic reading of the 16% result.** arXiv:2610.34440 proves deduplication is faster exactly when its byte share ū(n) < its bandwidth ratio r(n); stock Metal Qwen3 has r(2)=0.50 (per-slot 131 GB/s vs dedup 65 GB/s; re-read cost 0.92–0.98 = *no on-chip cache reuse*), so depth 2 loses. But r(n) was measured **with weights in unified memory**; in Veystrix's SSD regime the per-slot kernel's ≥131 GB/s is ~40× faster than the ~3.36 GB/s SSD, so the bandwidth penalty is invisible. **A read-once-from-SSD staging kernel has r_ssd ≈ 1 and should win by the routing-union factor (~1.8×) at n=2.** The paper itself names r(n) as "the lever for kernel work."

### Cited Findings
- Model: dedup faster iff ū(n) < r(n); saving = per-slot expert time × (1 − ū(n)/r(n)); with equal fixed work, **a deduplicating kernel at r(n)=1 is faster at every pass size with reuse**, and higher bandwidth at small n moves the crossover to fewer tokens. — [arXiv:2610.34440 §2–3, §6](https://arxiv.org/html/2610.34440v1)
- Measured (Table 1): Qwen3 on llama.cpp Metal r(2)=0.50, ū(2)=0.77; r(8)=0.35; re-read cost 0.92–0.98 (no cache benefit). OLMoE Metal r(2)=0.59, ū=0.82; gpt-oss Metal r(2)=0.37. Per-slot bandwidth exceeds DRAM on CUDA from n=4 — "the gap lies in the deduplicating kernel." — [arXiv:2610.34440 Table 1](https://arxiv.org/html/2610.34440v1)
- Reuse is real but front-loaded: layer-mean ρ(2)=0.480 (Qwen3), ρ(9)=0.780; adjacent-token overlap ~25% of a layer's expert set; dedup crossover is 5–8 tokens, or never, on these in-memory engines. — [arXiv:2610.34440 §2, §4](https://arxiv.org/html/2610.34440v1)
- MonoMoE: token-major kernels underuse bandwidth when few tokens reach each expert; a *weight-major fused kernel* loops over expert weights and applies each to all its tokens (up to 1.54× over vLLM Triton Grouped GEMM on H200) — the canonical "read each distinct expert once" layout. — [MonoMoE](https://arxiv.org/abs/2609.04244)
- Grouped GEMM precedent: sorted tokens-per-expert + cumulative offsets + one grouped_mm launch; weight layout [E,K,N] avoids runtime transpose. — [grouped GEMM guide](https://github.com/whitecircle/halo/blob/6b72e05cf7f77f7b60bae301a08a68763a3967e4/agent-docs/optimization/grouped-gemm.md)
- SpecFit (exact, target untouched) selects verification subtrees by expert-union marginal cost, verifying ~2–5 nodes instead of 7, saving 12–22% experts/token and gaining more as the cache shrinks — evidence that union-aware selection is the right companion to a dedup kernel. — [SpecFit](https://arxiv.org/html/2610.15417)
- Project's own K=2 result: read-sharing was ~16% slower on Metal; the coordinator's union ratio over 2 tokens ≈ 1.8× one token. — coordinator's measured context; see the prior note `../MoE decode IO bypass research/speculative_sharing.md`

### Inferences
- **Why K=2 lost:** on stock Metal the dedup kernel moves the distinct experts at ~0.5× the per-slot kernel's bandwidth (no re-read cache benefit, unfavourable tile/occupancy), and n=2 has the *highest* ū (~0.77–0.83) — both terms of 1 − ū/r are negative. That is a kernel-quality result, not a routing fact.
- **Why a different layout fixes it here:** in SSD offload the per-slot kernel's extra bandwidth has nowhere to apply (SSD is the limiter), while a weight-major staging kernel still reads each distinct expert **once** from SSD into a staging buffer and serves both tokens from fast unified memory. The relevant ratio is SSD bytes, for which ū_ssd = 1/r_ssd by construction → r_ssd ≈ 1 > ū(2)=0.8 ⇒ win. Expected bytes/token (inference from coordinator's union ratio): Qwen 873 → ~873/1.8 ≈ **485 MB**, Nemotron 857 → ~**476 MB**; at 3.36 GB/s that is ~144/~142 ms — Qwen's 200 ms budget is met; Nemotron's 67 ms is not.
- **Exactness:** trivial — the same weight bytes are read (once) and applied to the exact routed tokens; routing and arithmetic unchanged. No approximation, unlike S2-MoE/BigMoMo expert pruning. — [S2-MoE §6.2.2](https://arxiv.org/html/2608.15018v1); [BigMoMo §4.2](https://arxiv.org/html/2609.14643v1)
- **M4 decompression cost:** none; cost is kernel occupancy/grouping. The 2610.34440 one-layer benchmark fits r(n) in <1 h/model — do that first.
- **Falsifiable 1-script/1-experiment test:** (1) reproduce the paper's one-layer expert-op benchmark on Veystrix's Metal kernel to get r(2), ū(2); (2) A/B baseline per-slot vs weight-major "read distinct expert once into staging MTLBuffer, then grouped-GEMM over its tokens" at n=2, identical routing trace; assert identical token IDs and measure SSD bytes/token and ms/token. Falsifier: if measured in-memory r(2) > 0.8 on Veystrix's kernel, the old conclusion was wrong and depth-2 dedup may already win; if staged dedup does not beat 1.8× byte reduction, the staging kernel is bandwidth-bound in RAM and the layout must change again.

### Gaps
- No measurement of r(n) for Veystrix's exact MLX/Metal MoE kernel; 2610.34440's Qwen3-Metal numbers are llama.cpp, not Veystrix.
- No published "SSD staging + weight-major dedup" end-to-end tok/s number — the paper's dedup benchmarks are in-memory only, so the r_ssd≈1 synthesis is an inference, not a measured fact.
- Whether the earlier 16% slowdown came from the dedup kernel's bandwidth or from extra bookkeeping/launch overhead is unknown; the benchmark in (1) separates them.

---

## Key Question 6: Expert-contiguous APFS layout and read amplification — can the SSD deliver more useful bytes per syscall?

### Takeaway
**No new lever found; this remains a gap, not a result.** APFS has no user-facing read-path tuning knob, the documented global-lock issue affects metadata-heavy parallelism (many small files), not a few-large-file streaming pattern, and the project already measured achieved throughput near the saturated-worker wall. The one untried combination is kernel advisory prefetch (`F_RDADVISE`) into page cache followed by Metal fast-resource loading (`loadBytes`) from that warm cache — which is a *latency* trick, not a byte-count trick.

### Cited Findings
- APFS "obtains a global kernel lock during read-only operations such as readdir()"; 12 parallel walks 172 s vs single 523 s (Mojave), Linux/ext4 25 s — metadata, not bulk pread. — [Greg Szorc, APFS global locks](https://gregoryszorc.com/blog/2018/10/29/global-kernel-locks-in-apfs/)
- APFS vs HFS+ bulk throughput "within a few percentage points" — little to gain from filesystem choice for sequential reads. — [OWC](https://eshop.macsales.com/blog/42901-the-speed-of-apfs-just-how-fast-is-it/)
- Project log: F_RDAHEAD=0 / F_NOCACHE neutral; F_RDADVISE ("untimed ahead advice") reached 3.61–3.78 GB/s; MTLIO warm 13.1–13.3 GB/s but only 5.4–7.5% of misses warm; mlock gained ~0 ms; skipping streaming recovered 12–14 ms (contention, not eviction). — [Mference experiment log](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md)
- `F_RDADVISE` is a true non-blocking async prefetch (24 MB in ~0.5 ms, resident ~5 ms later), `F_RDADVISEV` batches up to 64 ranges; `MADV_WILLNEED` is synchronous on Darwin (~11 ms/24 MB). — [Bun commit](https://github.com/oven-sh/bun/commit/adc354d9983fe1866ceeb28e6a6ecd30ce443cca); [fcntl man page](https://keith.github.io/xcode-man-pages/fcntl.2.html)
- SolidAttention: finer partitioning increases random I/O overhead and reduces bandwidth utilization; larger blocks plateau once SSD bandwidth is saturated — relevant only if current reads are sub-block. — [FAST'26](https://www.usenix.org/system/files/fast26-zheng.pdf)

### Inferences
- Bit-exactness: all layout/advice changes only move bytes; routing and values unchanged, subject to the project's IO-08 warning that *speculative use* of advised experts changed token IDs — advice must be prefetch-only.
- Bytes/token: no reduction predicted; benefit is latency hiding, already bounded by E1 (+25% ceiling). Cannot break the byte floor.
- Falsifiable test (if pursued): instrument achieved GB/s vs 3.36 GB/s rating and count `read()` syscalls/bytes per token before/after expert-contiguous repack; if achieved throughput already ≈ rating, abandon. Separately test RDADVISE→MTLIO `loadBytes` combination (the prior note's stated open gap).
- Rank lowest among this note's leads: it cannot change the 873/857 MB byte count.

### Gaps
- No published read-amplification measurement for MoE expert reads on a commodity M4 SSD; no source on internal+TB4 APFS RAID0 large-read throughput at 15+ tok/s.

---

## Key Question 7: Ranked synthesis — which findings are real byte levers, their M4 cost, falsifiable tests, and the arithmetic ceiling

### Takeaway
Rank 1 is the **read-once SSD staging dedup kernel** (up to ~1.8× bytes, bit-exact, no numerical risk); rank 2 is **scale/bias exponent coding** (~6%/3.3%); rank 3 is **tile-level ANS fused decode** (~1.1–1.3× for group-quantized, with the batch-1 caveat); ranks 4–6 are closed/open-niche/negative. Even optimistically combined, **Qwen's 5 tok/s becomes reachable, Nemotron's 15 tok/s remains arithmetically impossible** without a ~4× cut that no bit-exact lever provides.

### Cited Findings
- Rank 1 evidence: [2610.34440 §6 r(n)=1 lever](https://arxiv.org/html/2610.34440v1); [MonoMoE weight-major kernel](https://arxiv.org/abs/2609.04244); [coordinator union ratio ~1.8×].
- Rank 2 evidence: [MLX affine docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html); [DFloat11](https://openreview.net/pdf/6b1607ff4d79f3a474030fbbf74f738e0e5350c9.pdf); [ZipServ](https://arxiv.org/html/2603.17435v1).
- Rank 3 evidence: [tile-level ANS](https://arxiv.org/html/2606.15789); [Great Inversion survey open problem](https://arxiv.org/html/2608.25188v1).
- Negative evidence: [grammar-compressed linear algebra impossibility](https://proceedings.neurips.cc/paper_files/paper/2020/file/645e6bfdd05d1a69c5e47b20f0a91d46-Paper.pdf); [Decode-Late](https://arxiv.org/html/2610.31913v1); [IJCA entropy ceiling](https://www.ijcaonline.org/archives/volume187/number120/rotem-ijca-2026-a71cdd2650e6.pdf).

### Inferences
Ranked table (bytes/token are inferences from coordinator figures unless sourced; all are bit-exact):

| Rank | Mechanism | Bytes/token (from 873 Qwen / 857 Nemotron) | M4 cost | Falsifiable test |
|---|---|---|---|---|
| 1 | Weight-major read-once-from-SSD dedup staging (r_ssd≈1) | ~485 / ~476 (~1.8×) | kernel/grouping only; no decode | one-layer r(n) benchmark + n=2 A/B, assert identical token IDs |
| 2 | DFloat11/ZipServ exponent coding on scale+bias BF16 stream | ~821 / ~829 (~6% / 3.3%) | negligible; fuse decode with SSD read | scale-exponent histogram + Huffman/TCA-TBE + SHA-256 dequant |
| 3 | Tile-level ANS on codes+scales fused into GEMM | ~670–830 / ~660–820 (~1.05–1.3×) | decoder-bound at batch 1; overlap with SSD read | ANS round-trip bit-exact dot products; entropy per tensor |
| 4 | Exact algebraic reformulation (Decode-Late/T-MAC) | unchanged | lower compute only | count read() bytes with/without kernel |
| 5 | Cross-expert delta/XOR | unchanged (expected) | — | residual zero-density + entropy; kill if <50% zeros |
| 6 | Bit-plane partial fetch | unchanged for bit-exact codes | few shifts/word | per-plane entropy sum ≈ nibble entropy |
| 7 | APFS layout / RDADVISE+MTLIO | unchanged (latency only) | — | achieved GB/s vs rating; token-ID parity |

Arithmetic ceiling (inference): best plausible stack = 1.8× (dedup) × 1.1–1.3× (ANS) × ~1.06 (scale coding) ≈ **2.1–2.5×**, i.e. Qwen 873 → ~350–415 MB (104–123 ms at 3.36 GB/s) and Nemotron 857 → ~340–410 MB (101–122 ms). Qwen's 200 ms target is met with compute hidden; Nemotron's 67 ms target needs ≤225 MB — a further ~1.5–1.8× beyond the stack, which no bit-exact lever in this note or prior notes supplies. **The "second miracle" that reaches Nemotron 15 tok/s does not exist in software under the stated constraints.** The realistic second miracle is the ~1.8× dedup byte cut for Qwen, contingent on measuring r(n) on Veystrix's own Metal kernel.

### Gaps
- All byte projections for Qwen3.8-Flash-Next / Nemotron-Lightning are inferred from coordinator measurements plus literature ratios; none has been measured on the actual models.
- The joint gain of dedup + ANS + scale coding may be sub-additive (both compete for the same byte budget and both add decode/grouping overhead); no source combines them.
- No M4-specific fused-decode throughput for rANS/TCA-TBE; the GPU numbers (0.01–0.1 bits of Shannon; 2.21× kernel speedup) do not transfer directly to a tensor-core-less M4 GPU.
- The "matters" reference and the Tan et al. "6–10× INT4 entropy" claim both require local reconciliation before any projection is trusted.
