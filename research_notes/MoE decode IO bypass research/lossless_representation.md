# Lossless Weight Representation Techniques for Reducing Bytes-Read-per-MoE-Expert-Token (Bit-Identical Q4 Output)

**Scope:** Veystrix "miracle hunt." Target hardware: 16 GB M4 Mac, single SSD ~3 GB/s. Workloads: Qwen3.8-Flash-Next (Q4 affine G32, expert = 3,072,000 B, 408 MB physical read/token, 2.3 tok/s) and Nemotron Lightning (Q4 affine G64, expert = 5,612,544 B, 436 MB/token). Targets 5+ / 15+ tok/s. **Hard constraint: dequantized Q4 values must be bit-identical.** Generic ZSTD/LZ4 measured at only 0.9–2.1% (given, confirmed independently below).

---

## Key Question 1: Is there ANY lossless representation that ~halves the bytes of a G32/G64 Q4 weight stream?

### Takeaway
**No.** Peer-reviewed, byte-exact measurements put the information-theoretic ceiling for GGUF Q4_K tensors at ~1.076× (4-bit nibble entropy 3.86/4.0 bits), and ~1.04× for whole artifacts. A 2× lossless representation of the code stream is information-theoretically impossible; the "miracle" as stated does not exist.

### Cited Findings
- A 2026 study that verifies byte-exact roundtrips with SHA-256 on every file/report and bills the entropy-profile to each file measured the GGUF Q4_K nibble alphabet at **median 3.86 bits of entropy per 4-bit symbol** (p10 3.71, p90 3.94, CI [3.78, 3.91]) — only 0.14 bits below the 4.0-bit uniform maximum. Packed-byte entropy 7.73/8 bits; sign bit 0.96/1; magnitude 2.91/3. [Source](https://www.ijcaonline.org/archives/volume187/number120/rotem-ijca-2026-a71cdd2650e6.pdf)
- The same study's **marginal-byte ceiling for Q4_K is ≈1.076× at the tensor-stream level**, dropping to **1.041–1.045× when whole GGUF Q4_K_M artifacts are scored** (because Q6_K tensors compress at only ~1.01×). Its best coder (mixture-CDF K=4) reaches 1.052×; dictionary-trained zstd-19 actually *expands* the stream (0.998×) once the 112,640-byte dictionary is billed per file. [Source](https://www.ijcaonline.org/archives/volume187/number120/rotem-ijca-2026-a71cdd2650e6.pdf)
- Concrete zstd numbers on Q4_K: zstd-3 = 1.034×, zstd-19 = 1.040×, xz-9 = 1.044×, bit-plane decomposition + zstd-19 = 1.043× — matching Veystrix's measured 0.9–2.1% and showing **bit-plane disaggregation adds nothing**. [Source](https://www.ijcaonline.org/archives/volume187/number120/rotem-ijca-2026-a71cdd2650e6.pdf)
- Intel (2025) tested FP4 (MXFP4/NVFP4) and found concatenating 2-bit slices from 4 consecutive FP4 values into bytes produced streams "close to uniformly random," with "little to no redundancy across tensor blocks"; they concluded **"the quantized values in FP4 do not exhibit consistent statistical redundancy and are not suitable for lossless compression."** Only the FP8 scale factors compressed. On DeepSeek-R1 NVFP4, scale factors were 39 GB of 395 GB (10%) → overall ~5% savings. [Source](https://arxiv.org/html/2508.19263)
- DFloat11's maintainers explicitly decline to support quantized models: an INT8 model "might already use more than 7 bits of entropy," so there is no exponent redundancy left to harvest. DFloat11's ~30% lossless gain is for BF16 only. [Source](https://ai-infrastructure.net/dfloat11-lossless-compression/)
- Independent Apple-Silicon prototype (drxddy/ecq) reports **per-tensor 4-bit entropy ~1.7 bits → 2.4×**, but **per-group (mlx-lm default, the Veystrix-relevant case) entropy ~3.7 bits → only 1.08×**; the authors state "the compression-accuracy tradeoff is fundamental" and that per-group quantization makes entropy coding ineffective. [Source](https://github.com/drxddy/mlx-entropy-ext)
- TurboQuant's entropy table for Lloyd–Max 4-bit quantization gives H = 3.764 bits/symbol (saving 0.236 BPW) — again ~1.06×. [Source](https://cksac.github.io/turboquant-model/techniques/entropy-codec/)

### Inferences
- The code nibble stream is nearly a maximally-entropic source. Entropy coding can recover at most ~4–8% of the packed code bytes; the rANS/mixture-CDF gap to ceiling is small. A 2× reduction would require ~2 bits/symbol entropy, which does not occur in group-quantized trained weights.
- **Arithmetic consequence for Veystrix:** even a *perfect* 2× byte reduction gives 2.3 × 2 ≈ 4.6 tok/s (linear IO-bound model) — short of 5 tok/s and far short of 15 tok/s. Reaching 15 tok/s from 2.3 needs a ~6.5× byte reduction, which is not available from any lossless representation. This is a second, independent reason the "miracle" fails.
- The user's measured 0.9–2.1% ZSTD/LZ4 result is corroborated and explained; the only larger lossless wins lie *outside* the code stream (layout, caching, scale factors — see Q2/Q3).

### Gaps
- The IJCA measurements cover Qwen2.5 up to 7B, GGUF Q4_K_M, not Qwen3.8-Flash-Next or Nemotron Lightning specifically. The ceiling may differ slightly for different affine schemes (G32 vs the Q4_K superblock structure), but the 3.86-bit entropy measurement is unlikely to move by 2 bits.
- Exact scale/bias precision in the Veystrix GGUF files (FP16 vs 6-bit min/max) is unknown; this determines whether scale coding contributes 2% or 10%.

---

## Key Question 2: Which LOSSLESS mechanisms actually reduce read bytes, and by how much? (Ranked)

### Takeaway
Five mechanisms are genuinely bit-identical. Only layout/caching reaches ~2× physical-read savings; pure representation coding peaks at ~1.05× (or an unverified ~1.33× in one recent system).

### Ranked findings (mechanism / losslessness / ratio / M4 cost / falsifiable test)

**Rank 1 — Lossless expert-pack layout + frequency/recency caching (biggest real win, zero numeric risk).**
- *Mechanism:* offline repack gate/up/down tensors of each (layer, expert) into one aligned block; manifest gives O(1) offset; one `O_DIRECT` read; VRAM/RAM caches retain hot experts. [Source](https://arxiv.org/html/2609.18110v1)
- *Exact-losslessness argument:* paper states explicitly: "reorganization changes only the physical layout on SSD; the tensor contents, quantization formats, routing decisions, and model computation remain unchanged"; "the expert-pack transformation and optional compression are lossless, so the tensor contents used for computation remain unchanged." [Source](https://arxiv.org/html/2609.18110v1)
- *Ratio / bytes-per-token:* up to **1.8× less per-token SSD traffic** vs llama.cpp and 1.5× vs KTransformer; 77.7% of peak SSD sequential bandwidth (9 GiB/s of 9 GiB/s testbed). FlashesMoE-style ML cache replacement gives hit-rate +51% and **2.6× speedup** over existing systems. [Source](https://arxiv.org/abs/2609.18110) [Source](https://arxiv.org/html/2601.17063)
- *M4 cost:* negligible CPU (one manifest lookup + one aligned read per miss). cache hit rate under realistic budgets ~57–80% (PagedMoE Fixed4: 57% with 24 GiB cache; SSD-LLaMA: 80.1% VRAM hit with 22 GiB). [Source](https://arxiv.org/pdf/2611.03173) [Source](https://arxiv.org/html/2609.18110v1)
- *Falsifiable test:* instrument `read(2)`/`fio` on the Veystrix model; compare total physical bytes read per token before/after expert-pack + hot-expert caching; assert byte-for-byte SHA-256 equality of every decompressed expert against the GGUF original.

**Rank 2 — rANS/ANS entropy coding of the packed 4-bit codes with offline per-tensor tables.**
- *Mechanism:* rANS over the 16-symbol nibble alphabet with probability tables trained offline per tensor; decode fused into dequant/GEMV. [Source](https://arxiv.org/html/2609.18110v1) [Source](https://github.com/ml-explore/mlx/discussions/3040)
- *Exact-losslessness argument:* entropy coding is a bijection on symbol streams; SSD-LLaMA verifies "after decompression, every byte of the quantized weights and their associated scale data is identical to the original expert pack." [Source](https://arxiv.org/html/2609.18110v1)
- *Ratio:* **33.2% SSD traffic reduction (8.89 → 5.93 GB/token)** in SSD-LLaMA on DeepSeek-class Q4 models. **Contradiction to note:** this is far above the ~7.6% Q4_K ceiling in the IJCA study; likely explained by different quantization/scale accounting or structured models not disclosed. Treat 1.05–1.33× as the credible range pending a Veystrix-specific measurement. Apple-Silicon prototype: 1.84× over 4-bit and 2.32× measured speedup on M2 Pro *for per-tensor quantization only*; per-group only 1.08×. [Source](https://arxiv.org/abs/2609.18110) [Source](https://github.com/drxddy/ecq) [Source](https://github.com/ml-explore/mlx/discussions/3040) [Source](https://www.ijcaonline.org/archives/volume187/number120/rotem-ijca-2026-a71cdd2650e6.pdf)
- *M4 cost:* decode overhead ~29% of memory bandwidth on M2 Pro with a fused Metal kernel; SSD-LLaMA's CUDA rANS decode + H2D = 0.54 s/token vs 0.93 s/token for SSD read (i.e., decoder must be faster than SSD to be a net win). Scalar zstd decompresses at ~211 MB/s on Q4_K, xz-9 at 10 MB/s — too slow for a 3 GB/s SSD. [Source](https://github.com/ml-explore/mlx/discussions/3040) [Source](https://arxiv.org/html/2609.18110v1) [Source](https://www.ijcaonline.org/archives/volume187/number120/rotem-ijca-2026-a71cdd2650e6.pdf)
- *Falsifiable test:* train a static 16-symbol rANS table per Veystrix expert tensor (train on a disjoint model/layer to avoid profile-billing bias), compress + roundtrip-verify SHA-256, and measure actual `bits/symbol`; abort if >3.95 bits.

**Rank 3 — Scale/bias (quantization parameter) entropy coding.**
- *Mechanism:* compress scale/min streams separately from codes; ZipNN/DFloat11 exponent Huffman for FP16/BF16 scale fields; Huffman for FP8 scales. [Source](https://arxiv.org/html/2508.19263) [Source](https://arxiv.org/html/1510.00149v3)
- *Exact-losslessness argument:* preserves the exact scale/bias bits → `s*q+b` bit-identical; Huffman is a prefix-code bijection.
- *Ratio:* FP8 scale factors compressed to 0.20–0.30 of original (70–80% savings on the *scale stream*), but scales are only ~10% of an NVFP4 dataset → ~5–10% total; Q4_K marginal-byte study finds within-superblock conditioning on (scale,min) adds only ~0.032× over byte-marginal. [Source](https://arxiv.org/html/2508.19263) [Source](https://www.ijcaonline.org/archives/volume187/number120/rotem-ijca-2026-a71cdd2650e6.pdf)
- *M4 cost:* small (scales are a small stream; static Huffman table). [Source](https://arxiv.org/html/2508.19263)
- *Falsifiable test:* split a Veystrix expert into code bytes and scale bytes, measure Shannon entropy of each stream, and compute whole-expert ceiling `1 / (0.8*H_code/4 + 0.2*H_scale/16)`.

**Rank 4 — Bit-plane disaggregation / "fetch only needed bitplanes."**
- *Mechanism:* split 4-bit codes into sign/magnitude or 4 bitplanes; entropy-code separately; PagedMoE's prefix-compatible tree codebook lets a cached low-precision prefix be extended by fetching only missing bitplanes. [Source](https://arxiv.org/pdf/2611.03173) [Source](https://www.ijcaonline.org/archives/volume187/number120/rotem-ijca-2026-a71cdd2650e6.pdf)
- *Exact-losslessness argument:* bitplane splitting is a bijection; if all 4 planes are fetched and reassembled, values are identical. **However** PagedMoE's actual benefit comes from *dynamic precision* (avg 2.5 bits), which **fails bit-identical** — so only the plain bijection form is valid here.
- *Ratio:* bit-plane decomposition + zstd-19 = 1.043× on Q4_K — essentially no gain over plain zstd. [Source](https://www.ijcaonline.org/archives/volume187/number120/rotem-ijca-2026-a71cdd2650e6.pdf)
- *M4 cost:* extra bookkeeping; marginal.
- *Falsifiable test:* measure entropy of each bitplane separately; confirm sum ≈ global nibble entropy (it must, up to rounding); confirm no bytes saved.

**Rank 5 — Cross-expert / cross-layer delta, XOR, weight-sharing, dedup.**
- *Mechanism:* XOR adjacent same-role weight matrices; dedup repeated blocks.
- *Exact-losslessness argument:* delta/XOR coding is lossless if the reference block is stored or already present.
- *Ratio:* **No exploitable linear redundancy found.** Median Pearson correlation +0.0004 over 250 adjacent same-role bf16 layer pairs from two Qwen2.5 models. [Source](https://www.ijcaonline.org/archives/volume187/number120/rotem-ijca-2026-a71cdd2650e6.pdf)
- *M4 cost:* N/A — not worth implementing.
- *Falsifiable test:* compute byte-level XOR zero-density and Pearson r for adjacent expert/layer tensors in Veystrix; abort if zero-density < ~50%.

### Inferences
- The bytes/token accounting for Veystrix: 408 MB baseline. Rank 1 alone → ~227–272 MB; Rank 2 (if 33% holds) → ~274 MB; Rank 3 → ~390 MB. Best plausible *combined* lossless ≈ 200–260 MB/token → ~3.6–4.7 tok/s. This reaches the 5 tok/s target only marginally and **cannot reach 15 tok/s**.
- Decompression throughput is a first-order constraint: to be a net win, decoder throughput must exceed the 3 GB/s SSD. Scalar Huffman (~1.35 GB/s) or zstd (~2.3 GB/s, from the quantized-LLM compressibility study) would *slow* a 3 GB/s SSD path; a fused/vectorized rANS decoder or decode-once-into-cache is required. [Source](https://doi.org/10.48550/arxiv.2403.01384)

### Gaps
- No published measurement of entropy of the *scale/min* fields specifically in Qwen3.8-Flash-Next.
- No published A17/M4 benchmark of rANS decode throughput for 4-bit weights; M2 Pro prototype numbers are the closest proxy (29% of bandwidth overhead).

---

## Key Question 3: Which "lossless-looking" techniques FAIL the bit-identical requirement?

### Takeaway
Several headline 1.39-bit / 3.27× / 2.5-bit results achieve their gains by **changing the quantization scheme or routing**, not by lossless representation. They are disqualified.

### Cited Findings
- **EntroLLM** reports "effective bit-widths" of 1.39 bits for uint4 and 11.3× compressibility improvement — but it achieves this by replacing block/group quantization with **tensor-level unsigned/asymmetric quantization** ("block-based quantization flattens the integer set and increases entropy; tensor-level yields a spikier distribution"). This changes the dequantized values → FAILS. [Source](https://arxiv.org/html/2505.02380v4)
- **drxddy/ecq**'s 3.27× / 3.58× claims are for per-tensor quantization; its own table shows per-group at only 1.08×. Adopting per-tensor would change values → FAILS. [Source](https://github.com/drxddy/ecq)
- **PagedMoE** reaches 8.89 tok/s with 24 GiB cache at average routed-expert precision **2.5 bits** using a prefix tree codebook; lower precision prefixes "directly executed" are a lossy approximation of the 4-bit expert → FAILS. (Its Fixed4 baseline is the relevant lossless comparator: 57% hit, 1.53 GiB/token, 5.65 tok/s.) [Source](https://arxiv.org/pdf/2611.03173)
- **ReMoE** fine-tunes the router to increase expert reuse (EOR 27.3% → 34.5%): changing routing changes which experts process each token → output changes → FAILS. [Source](https://arxiv.org/html/2605.27081v1)
- **DFloat11 / Unweight / ZipNN** compress BF16 exponents (~2.6 bits entropy of 8) — powerful and bit-exact on BF16, but inapplicable to Q4 *codes* (no exponent field). Unweight reaches ~30% on MLP weights / ~20% total on Llama-3.1-8B with bit-identical outputs; its program explicitly includes "Mixture-of-Experts serving" work, so it is worth watching for future scale-factor compression. [Source](https://ai-infrastructure.net/dfloat11-lossless-compression/) [Source](https://research.cloudflare.com/papers/unweight-2026.pdf)
- **Requantizing to INT4/FP8, different group size, pruning, zero-point tricks** — all explicitly excluded by the assignment.

### Inferences
- Every published >2× result on Q4 weights changes the numeric content. The bit-identical constraint reduces the achievable set to ~1.05× representation + layout/caching.
- Prefix/bitplane codebooks are the closest thing to a "bit-slicing" win, but their demonstrated advantage comes from reading *fewer bitplanes* (lossy), not from storing 4 bits in fewer lossless bits.

### Gaps
- Need to confirm whether any Veystrix scale fields are FP16 (DFloat11-style exponent Huffman applicable) versus 6-bit GGUF-style min/max (little headroom).

---

## Key Question 4: Decompression cost on M4 and net effect on tok/s

### Takeaway
Entropy decoding at 1–2 GB/s is slower than the SSD; without a fast fused decoder or decode-once-into-cache, lossless compression is a net throughput loss.

### Cited Findings
- Entropy coder throughputs (single-core x86 Sapphire Rapids): zstd-3 211 MB/s, zstd-19 121 MB/s, xz-9 10 MB/s, mixture-CDF rANS 38 MB/s on Q4_K. Huffman ~1.35 GB/s and zstd ~2.3 GB/s in the quantized-LLM compressibility study. [Source](https://www.ijcaonline.org/archives/volume187/number120/rotem-ijca-2026-a71cdd2650e6.pdf) [Source](https://doi.org/10.48550/arxiv.2403.01384)
- Apple Silicon prototype: M2 Pro fused Metal decode+GEMV gave decode overhead 29% of memory bandwidth and 2.32× measured speedup for per-tensor; M3 Pro standalone matmul 0.18 ms (entropy-coded) vs 0.12 ms (native 4-bit) for a 1024×1024 matrix. [Source](https://github.com/ml-explore/mlx/discussions/3040) [Source](https://github.com/drxddy/mlx-entropy-ext)
- DFloat11 at batch 1 is ~2× slower than BF16 due to decode; diffusion workloads hide it (+4–5% latency). Relevant caution: decompression is not free even on strong GPUs. [Source](https://ai-infrastructure.net/dfloat11-lossless-compression/)
- SSD-LLaMA: compressed H2D + CUDA decode 0.54 s/token vs 0.93 s/token SSD read — decoder faster than storage, so net positive. [Source](https://arxiv.org/html/2609.18110v1)

### Inferences
- On Veystrix's 3 GB/s SSD, a decoder must sustain >3 GB/s to avoid becoming the bottleneck. Using SIMD bitplane/Huffman tables or decoding once on cache miss (amortized over many hits) are the viable options. Scalar rANS at 38 MB/s is disqualifying.
- Caching is doubly valuable: it removes SSD reads entirely on a hit and removes decode cost (weights already materialized).

### Gaps
- No M4-specific rANS/ANS benchmark found; M2/M3 Pro numbers are the only Apple-Silicon proxy.

---

## Bottom line for the report writer

1. A bit-identical, ~2× smaller *representation* of G32/G64 Q4 weights does not exist: the measured code entropy is 3.86/4 bits and the whole-artifact lossless ceiling is ~1.04–1.05× (best case ~1.076× tensor-stream, ~1.33× only in one uncorroborated system). [Source](https://www.ijcaonline.org/archives/volume187/number120/rotem-ijca-2026-a71cdd2650e6.pdf) [Source](https://arxiv.org/abs/2609.18110)
2. The realistic lossless byte reductions are **layout + expert caching (1.5–1.8× physical reads)**, plus **~5–8% entropy coding** and **a few % scale coding**. Combined best case ≈ 2× physical reads → ~4–5 tok/s, not 15 tok/s. [Source](https://arxiv.org/html/2609.18110v1) [Source](https://arxiv.org/html/2601.17063)
3. All >2× "Q4 compression" results (EntroLLM 1.39-bit, ecq 3.27×, PagedMoE 2.5-bit, ReMoE router tuning) change values or routing and are disqualified. [Source](https://arxiv.org/html/2505.02380v4) [Source](https://github.com/drxddy/ecq) [Source](https://arxiv.org/pdf/2611.03173) [Source](https://arxiv.org/html/2605.27081v1)
4. Highest-value next experiment: replicate the IJCA byte-exact ceiling protocol on Veystrix's exact GGUF files (measure nibble entropy, scale entropy, XOR zero-density) and instrument physical `read()` bytes before/after expert-pack + caching. These two cheap tests determine whether any further lossless work is worthwhile.
