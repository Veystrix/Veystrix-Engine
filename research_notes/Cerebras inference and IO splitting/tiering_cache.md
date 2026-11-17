# Splitting MoE expert weight I/O across memory tiers and caches on a 16 GB Apple M4 — re-examining the "cache cannot reach targets" conclusion

Scope: novel tiering/splitting ideas for Qwen3.8-Flash-Next (512 experts/layer, top-10, 48 layers) and Nemotron (128 experts/layer, top-6, 23 layers), Q4 affine (U32-packed codes + BF16 scales/biases), on one 16 GB M4, **no new hardware, no quality loss (bit-identical tokens/logits, exact routing)**. Ground truth from the parent brief (do not re-litigate): resident weights + KV + OS leave little for an expert pool; production auto-sizes the pool to 0; the OS page cache is the real cache (~74% Nemotron hit); pool 2→4 GB regressed (double caching with page cache + memory pressure); steady-state OPT cache ceilings are **Nemotron 15 tok/s needs ~10 GB unique expert RAM, Qwen 5 tok/s needs ~12 GB — impossible in 16 GB**. Rejected as ground truth: F_NOCACHE used alone, F_RDAHEAD=0, wired-limit sysctls, more I/O workers, non-S3FIFO pool policies, perfect expert predictor, cache-aware routing, expert packing, mlock, decode overlap (~+25% cap), >2 GB pool, K=2 sharing. Adjacent subagent notes cover Cerebras/wafer-scale and "split I/O time" systems (CoPilotIO/Weave/COMET) — those confirm the same wall; this note focuses on **memory tiers and caches**.

---

## Key Question 0. Is the "cache cannot reach targets" conclusion robust, or did we miss a tiering trick?

### Takeaway
The conclusion is **robust**. Tiering/partitioning only changes *where the same bytes live*; it cannot change (a) the number of distinct experts the frozen router selects per token, (b) the bytes per expert (Q4 U32), or (c) SSD bandwidth. To hit the targets the miss bytes must essentially vanish from the critical path, which requires a working set of ~10–12 GB unique expert RAM that the 16 GB budget cannot hold. **No idea below beats the ceiling.** The only mechanisms that could in principle change the arithmetic are lossless byte reduction (ZipMoE — does not transfer to Q4 codes) and byte-identical expert dedup (none expected in a converged checkpoint); both are addressed below and both are predicted kills.

### Cited Findings
- OPT/Belady cache analysis and the 10 GB / 12 GB ceilings are parent-supplied local measurements/analysis (not externally sourced) — Veystrix parent brief.
- Page cache is the real cache at ~74% Nemotron hit while the application pool auto-sizes to 0; a 2→4 GB pool *regressed* from double caching + memory pressure — Veystrix parent brief.
- Merged llama.cpp PR #27861 (exact model family, top-10/512): top-32 "hot" experts cover only ~10% vs 6.2% uniform (no static skew), but per-layer LRU-64 gets ~67% and LRU-128 ~81% hit (temporal locality); decode 18.4→24.2 tok/s with ~4.1 GiB of slots, and a sharp cliff at ~24 GiB where 300 slots collapsed performance — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861).
- Cache-size sweeps on the exact models show graceful but bounded gains: Nemotron 4 GB cache 4.39 tok/s (86.1% hit) on 16 GB M4, 8 GB thrashed; on a 48 GB Mac, 10 GiB cache → 16.19 tok/s (83.5%, 14.64 GiB RSS) and 16 GiB → 16.53 tok/s (88.6%, 20.60 GiB RSS); fully resident Nemotron baseline is ~13.9 tok/s at ~20.8 GiB — [Jakevin/moe-offload](https://github.com/Jakevin/moe-offload); [Palm-Infra](https://github.com/TencentYoutuResearch/Palm-Infra/blob/main/docs/ssd-offload.md).
- Cerebras analysis: the only true exits are (1) fewer bytes/token, (2) more parallel parts, (3) faster/closer memory; on one 16 GB M4, software can only fit more experts (bounded by 16 GB), read fewer bytes (mostly closed), or overlap (~+25% max) — [direct_cerebras_notes.md](file:///Users/vincent/Documents/Veystrix/research_notes/Cerebras%20inference%20and%20IO%20splitting/direct_cerebras_notes.md); [direct_io_split_notes.md](file:///Users/vincent/Documents/Veystrix/research_notes/Cerebras%20inference%20and%20IO%20splitting/direct_io_split_notes.md).

### Inferences
- **Arithmetic check for the sole-cache ceiling:** at the measured ~74% hit, Nemotron miss bytes ≈ 0.26 × 436 MB ≈ 113 MB/token; at ~3 GB/s that is ~38 ms/token of unavoidable SSD time before compute/sync. A 15 tok/s budget is 67 ms, so even a perfect app pool that merely *matched* page cache would leave almost nothing for compute. Matching page cache is the best a relocation trick can do; it is not enough.
- The literature's big gains (10×/9.75×) come from weak baselines or large caches (10–24 GB) and from mixing in precision reduction — none is an additive multiplier on an already-tuned 16 GB setup — [moe_expert_io.md](file:///Users/vincent/Documents/Veystrix/research_notes/Faster%20Qwen%20Nemotron%20inference%20on%20M4/moe_expert_io.md).
- Verdict: the parent conclusion stands. Research value now is (i) two cheap *decisive* tests (SHA-256 cross-layer dedup; F_NOCACHE sole-pool A/B), (ii) one small free win (BF16 scale/bias compression), (iii) a fresh lossless mechanism (ZipMoE) explicitly killed for our format.

### Gaps
- No public bit-exact tok/s measurement for Qwen3.8-Flash-Next on a 16 GB M4; the closest 16 GB point is Nemotron at 4.39 tok/s — [Jakevin/moe-offload](https://github.com/Jakevin/moe-offload).
- The OPT ceiling derivation itself was not independently re-derived in this note; it is taken as parent ground truth.

---

## Key Question 1. Idea 1 — single-tier application pool as *sole* cache with F_NOCACHE (avoid double caching). Theoretical best with a 4–8 GB sole pool, and can smarter partitioning help?

### Takeaway
This is a **genuine untested A/B worth running**, but its ceiling is "match the page cache, minus an ~8% read penalty" — it **cannot beat the 16 GB ceiling**. The plausible win is elimination of double buffering (one hot expert currently occupies both a page-cache page and a pool buffer), which could double the effective hit capacity for the same RAM; the plausible loss is that anonymous pool buffers are VM-compressed under pressure and lose the page cache's instant clean-page reclaim, plus F_NOCACHE's measured ~8% slowdown and loss of readahead. Predicted outcome: neutral-to-regression; run it to close the question definitively.

### Cited Findings
- Apple guidance: "Caching increases the memory footprint of your application and if used inappropriately can be more expensive than simply reloading data from the disk"; for streaming data "tell the file system not to add that data to the file-system caches" via `fcntl(fd, F_NOCACHE, 1)`; and "Mapped files bypass the system disk caches, so only one copy of the file is stored in memory" — [Apple File System Performance Tips](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/FileSystem/Articles/FilePerformance.html).
- Self-caching applications (RocksDB) are the canonical case for opting out of the OS page cache: "in some situations, we want some data to opt-out of system cache," and macOS uses `F_NOCACHE` because `O_DIRECT` is unavailable — [RocksDB Direct-IO wiki](https://github.com/EighteenZi/rocksdb_wiki/blob/master/Direct-IO.md); [libuv issue #1600](https://github.com/libuv/libuv/issues/1600).
- Semantics caveat: `F_NOCACHE` "does not add data to the cache, it will read data from the cache if it is already there," so it is not a clean-room direct I/O — [Apple StackExchange](https://apple.stackexchange.com/questions/423259/how-to-achieve-oflag-direct-in-dd-on-macos). It is now per-file-descriptor, though historically per-vnode — [Apple filesystem-dev list](https://lists.apple.com/archives/filesystem-dev/2007/Sep/msg00012.html).
- Local measurements: F_NOCACHE was ~8% slower on this M4; pool 2→4 GB regressed via double caching + memory pressure; mlock recovered ~0 ms while skipping expert streaming recovered 12–14 ms (the mechanism was memory-system *contention*, not resident-page eviction) — [os_io_tricks.md](file:///Users/vincent/Documents/Veystrix/research_notes/MoE%20decode%20IO%20bypass%20research/os_io_tricks.md); Veystrix parent brief.
- Pinned-cache evidence on Apple Silicon: Palm-Infra pins the expert cache by default because "macOS VM compressor otherwise compresses cold anonymous buffers and decompresses them inside matmul" — [Palm-Infra](https://github.com/TencentYoutuResearch/Palm-Infra/blob/main/docs/ssd-offload.md).
- Explicit-read/prefetch variants that read exactly the router-selected experts as sorted, gap-merged large ranges gained +13–52% tok/s and are bit-exact — these are a *separate* intervention from the pool-vs-page-cache choice — [llama.cpp #25779](https://github.com/ggml-org/llama.cpp/discussions/25779).

### Inferences
- Mechanism for a win: if the current hot set is cached twice (page cache + pool), a sole pool with `F_NOCACHE` frees roughly one copy, so a 4 GB sole pool could behave like an ~8 GB cached set in hit-rate terms. Mechanism for a loss: (1) F_NOCACHE's ~8% penalty and lost readahead; (2) an anonymous pool is subject to the VM compressor that file-backed page-cache pages are not — under pressure, clean file pages can simply be dropped and re-read, whereas anonymous buffers must be compressed/swapped, and Palm-Infra shows decompression lands inside matmul. The OS page cache is therefore *structurally* the better cache under 16 GB pressure, which likely explains why the pool auto-sizes to 0.
- Theoretical best with a 4–8 GB sole pool: bounded by the measured residency curves (~47.9% hit at 1 GiB, 83.5% at 10 GiB on a 48 GB Mac; 86.1% at 4 GB for Nemotron on 16 GB), i.e. at most "page-cache parity." Smarter partitioning (LRU/S3FIFO per layer, pinned dense/attention) can only approach this; the no-static-skew result means a smarter partition cannot create hit rate that the working set does not contain — [Palm-Infra](https://github.com/TencentYoutuResearch/Palm-Infra/blob/main/docs/ssd-offload.md); [Jakevin/moe-offload](https://github.com/Jakevin/moe-offload); [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861).
- Bit-exactness: automatic — only residency/caching flags change; the same bytes reach the same kernels. Caveat: validate that `F_NOCACHE` per-fd state does not leak to other descriptors of the same file on this macOS version.
- **Cannot beat the ceiling.** It is a memory-efficiency experiment, not a byte-reduction mechanism.

### Gaps
- No A/B has combined `F_NOCACHE` on the expert file with a *sole* 4–8 GB application pool and page-cache disabled as the cache. This is the single cleanest untested experiment in the whole area.
- No measurement of whether the freed page-cache memory can be repurposed into a larger wired pool without triggering the VM compressor on the M4.
- No published model relating sole-pool hit rate to the 74% page-cache baseline on these exact experts.

---

## Key Question 2. Idea 2 — importance-based expert splitting: pin a small exact "hot" set in RAM, rest on SSD; layer-specific, token-dependent, or compressed-hot residency

### Takeaway
**Cannot beat the ceiling, and the premise is empirically weak.** A static hot set fails because there is no static skew (top-32 covers ~10% vs 6.2% uniform). Layer-specific/token-dependent pinning exploits real *temporal* locality, but the achievable hit rates (LRU-64 ~67%, LRU-128 ~81%) are below what the targets need, and 2–4 GB pools already regress. Compressed-hot residency is addressed in Q3.

### Cited Findings
- Top-32 hot experts cover only ~10% vs 6.2% uniform; per-layer LRU-64 ~67% / LRU-128 ~81% hit — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861) (parent brief measured ~10% for Flash-Next top-32, consistent).
- Rank-based, expert-ID-agnostic cache planning (ZipMoE) and rank-based marginal inclusion probabilities are the state of the art for "importance" without static identity — [ZipMoE, ICML 2026](https://arxiv.org/html/2601.21198).
- ExpertFlow RPP predictor reaches up to 95% prediction accuracy and 91.96% cache hit (+61.15% over LRU), but on GPU/CPU server setups, and the parent already rejected the "perfect expert predictor" — [ExpertFlow](https://arxiv.org/html/2410.17954v2); Veystrix parent brief.
- SeqMoE: 45% expert residency → 96.97% hit / 80.22% of full-load performance, again on server hardware — [SeqMoE](https://arxiv.org/html/2609.12978).
- Diff-MoE (global-hot/temporal-hot/cold tiers + next-layer predictor) reports 2.74×/2.22×/1.55× over DeepSpeed/Pre-gated MoE/MoE-Infinity without architecture changes — [Diff-MoE](https://dl.acm.org/doi/full/10.1145/3712285.3759903).
- ADEPT's 81% hit and 3.45× "simulated TPOT" gain is a simulation, not an end-to-end measurement — [IEEE ADEPT](https://ieeexplore.ieee.org/document/11397596).

### Inferences
- Layer-specific budgets and token-dependent pinning are bit-exact (they change admission/eviction, not weights or routing). They are a *modest* lever: the gains in the literature mostly come from larger caches or weak baselines, and the parent's own pool experiments already hit the memory cliff before reaching the needed hit rate.
- Even an oracle predictor cannot help if the miss bytes dominate and working set > RAM — consistent with the parent's "perfect expert predictor" rejection. **Cannot beat the ceiling.**
- Falsifiable test (cheap, offline): replay recorded routing traces; compare hit-rate curves for (a) fixed static top-32, (b) per-layer LRU-64/128, (c) ZipMoE-style rank-based admission at 4 GB and 8 GB budgets. Pass only if any curve reaches the ~10 GB-required hit rate at ≤6 GB RAM — predicted fail from PR #27861.

### Gaps
- No per-layer activation-frequency histogram for Qwen3.8-Flash-Next / Nemotron on 16 GB; needed to set per-layer budgets.
- No measured benefit of "compressed-hot" residency on a Q4 U32 expert set (see Q3).

---

## Key Question 3. Idea 3 — compressed caching: ZipMoE-style split-state cache, BF16 scale/bias compression, bit-exact partial representations

### Takeaway
This is the **freshest genuine mechanism** (ZipMoE, ICML 2026, lossless, UMA/Apple-Silicon-motivated), but it **does not transfer to our Q4 U32 codes** (measured zstd 95.2%). The only surviving byte-level win is BF16 scale/bias exponent coding (~6% Qwen / ~3% Nemotron, bit-exact) — too small to change the ceiling. **Cannot beat the ceiling.**

### Cited Findings
- ZipMoE (ICML 2026) targets UMA edge hardware explicitly including Apple Silicon. Lossless. BF16 exponent symbols are heavily skewed — support set only 14.06% / 11.33% / 14.84% of symbols on three models; zstd compresses BF16 parameters to ~68% of original, near the ~66% Shannon bound (LZ4HC 74%). Decompression is off the critical path with ≥3 CPU worker threads (7.47% peak SSD throughput reduction); caching sign–mantissa chunks gives "2× tensor coverage under the same memory budget," with four pools F(full)/C(compressed)/S(SM-chunks)/E(E-chunks) and a provable (3−1/L) makespan approximation. Decoder-only TPOT reductions 62.65–97.97%, throughput 1.79–42.49× vs baselines. Experiments were on Jetson AGX Orin 64 GB + Samsung 970 EVO, **not** on Apple Silicon — [ZipMoE](https://arxiv.org/html/2601.21198); [proceedings](https://proceedings.mlr.press/v306/yang26ds.html); [code](https://github.com/npnothard/ZipMoE-ICML26).
- ZipMoE limitations acknowledged by the authors/secondary notes: relies on low-entropy BF16 exponents (fails for other numeric formats), static offline compression, rank statistics need historical data and adapt poorly to non-stationary workloads — [ZipMoE paper](https://arxiv.org/html/2601.21198); [paper note](https://en.papernotes.org/ICML2026/model_compression/zipmoe_efficient_on-device_moe_serving_via_lossless_compression_and_cache-affini/).
- Our format kills the main mechanism: the real MLX expert tensor `layers.0.mlp.switch_mlp.gate_proj.weight` is `U32` (8 four-bit codes packed per 32-bit word); zstd levels 1/9/19/22 all give **95.2%** (4.8% saving), zlib 95.5%, with SHA-256 bit-exact roundtrip — [compression_contradiction.md](file:///Users/vincent/Documents/Veystrix/research_notes/Second%20miracle%20no%20new%20hardware/compression_contradiction.md).
- Surviving micro-wins from the same audit: BF16 scale/bias exponent coding ≈ 6% (Qwen) / 3% (Nemotron) of bytes; read-once SSD staging dedup via route union ≈ 1.8× fewer reads, bit-exact — [compression_contradiction.md](file:///Users/vincent/Documents/Veystrix/research_notes/Second%20miracle%20no%20new%20hardware/compression_contradiction.md).
- Lossless rANS expert-block decompression reconstructs exact weights; full (gate+up+down) cache gives 2.78× over no-cache, while caching only gate/up yields higher hit but lower throughput — [SSD-LLaMA](https://arxiv.org/html/2609.18110).
- ECF8 (ICLR 2026): lossless entropy-aware FP8, −26.9% memory, bit-exact; but applies to FP8 weights, not Q4 — [compression_contradiction.md](file:///Users/vincent/Documents/Veystrix/research_notes/Second%20miracle%20no%20new%20hardware/compression_contradiction.md).

### Inferences
- ZipMoE's "cache the incompressible half, fetch the compressible half on miss" is a real new tiering primitive, but its 2× coverage relies on BF16's incompressible sign–mantissa vs compressible exponent split. Q4 U32 codes are uniformly high-entropy, so there is no low-entropy half to split off; a "partial code" cache cannot avoid a full page-granularity SSD read. Applying ZipMoE would require storing weights as BF16 on SSD, which is 4–8× larger before compression (~2.7× after 68% compression) — worse for a 16 GB M4.
- Bit-exactness: ZipMoE is lossless by construction; scale/bias exponent coding is bit-exact if it is pure entropy coding. Both are compatible with the gate. Payoff is the issue, not exactness.
- **Cannot beat the ceiling.** At best ~6% byte reduction (scale/bias) moves the SSD floor by ~6%; the 10–12 GB working-set gap remains.
- Falsifiable test (offline, 1 script): re-run the per-tensor compression audit separately on scale and bias bytes (not codes) with zstd-19/xz-9e and a 2-D context coder; verify SHA-256 roundtrip; if combined saving >25% of expert bytes, revisit. Prior measurement says ≤6%, so predicted fail.

### Gaps
- No test of ZipMoE's split-state cache on MLX/Apple Silicon, or on a Q4-quantized MoE — the paper's experiments are FP/BF16 on Jetson.
- No measurement of whether MLX's U32 layout could be *repacked on disk* into a representation with a compressible sub-field while reconstructing bit-exact codes (likely impossible given measured entropy, but not formally proven).

---

## Key Question 4. Idea 4 — KV-cache / activation compression to FREE RAM for a bigger expert pool (we already use INT8 KV): any bit-exact or near-exact tricks that free 2–4 GB?

### Takeaway
**No bit-exact KV compression exists.** All KV quantization (FP8/INT8/INT4), MQA/GQA/MLA applied post-hoc, and KV eviction change logits or tokens and fail the gate. The only exact RAM savings are *management* (paged/block allocation, capping `n_ctx`, prefix sharing, freeing caches promptly) and those free *waste*, not 2–4 GB of live KV. **Cannot beat the ceiling.**

### Cited Findings
- Rigorous review: "the only cleanly PASSING 'exact' prefill wins are memory-management and scheduling (PagedAttention, prefix caching, chunked prefill, wired-memory pinning)"; "Any quantization, approximation, or token/token-range dropping is mathematically lossy and fails the bit-identical gate" — [kv_prefill.md](file:///Users/vincent/Documents/Veystrix/research_notes/Faster%20Qwen%20Nemotron%20inference%20on%20M4/kv_prefill.md).
- FP8 KV: 2× KV capacity, "negligible accuracy loss" (not zero), can degrade at 128K+ — [vLLM FP8 KV](https://vllm.ai/blog/2026-04-22-fp8-kvcache); [Spheron guide](https://www.spheron.network/blog/flashattention-2-vs-flashattention-3-h100-h200-guide). INT8 KV: max per-element error 0.004, attention score error <0.1 — bounded-error, not bit-exact — [arXiv:2601.04719](https://arxiv.org/html/2601.04719v1).
- On Apple Silicon, Open-TQ-Metal int4 KV: cosine 0.992–0.998, KL <1e-10, identical top-1 under greedy — but still "NOT bit-identical logits" — [Open-TQ-Metal](https://arxiv.org/html/2604.16957v1).
- PagedAttention cuts KV memory waste from 60–80% to <4%; prefix caching gives exact reuse of matching prefixes — [vLLM blog](https://vllm.ai/blog/2023-06-20-vllm); [SGLang](https://arxiv.org/pdf/2312.07104).
- Wired-memory pinning recovered 10× throughput in an Apple Silicon study (0.6→6.0 tok/s) by avoiding SSD paging — [Open-TQ-Metal §3.4](https://arxiv.org/html/2604.16957v1).
- KV size math: ~5.4 GB at 16K for a 70B model; exceeding 16 GB unified memory triggers macOS compression/swap with sharp throughput collapse; advice is to pre-allocate context to actual size, not model max — [zachrattner calculator](https://zachrattner.com/projects/ai-mac-cluster/llm-speed-calculator); [Contra Collective](https://contracollective.com/blog/kv-cache-context-length-apple-silicon-local-inference-2026).

### Inferences
- "Near-exact" is not "exact": even the best int4 KV result explicitly reports identical top-1 but not identical logits; under a strict token-digest gate this fails. There is no published bound of the form "BF16-rounded logits identical" for any KV quantization — [kv_prefill.md](file:///Users/vincent/Documents/Veystrix/research_notes/Faster%20Qwen%20Nemotron%20inference%20on%20M4/kv_prefill.md).
- Paged allocation/freeing can reclaim the 60–80% fragmentation waste, but it does not shrink the live KV of the active context; if Veystrix already pre-allocates sensibly, the freed bytes are unlikely to be 2–4 GB.
- **Cannot beat the ceiling** under the gate. If the gate were relaxed to argmax identity, FP8 KV would be the safest candidate (2× KV capacity), but that is explicitly out of scope.

### Gaps
- No measurement of Veystrix's actual KV bytes/token and current fragmentation/waste on the 16 GB M4; without it the exact management upside is unknown.
- No MLX-native paged-KV or prefix-cache implementation identified (SGLang/vLLM are CUDA-oriented) — engineering feasibility open — [kv_prefill.md](file:///Users/vincent/Documents/Veystrix/research_notes/Faster%20Qwen%20Nemotron%20inference%20on%20M4/kv_prefill.md).

---

## Key Question 5. Idea 5 — splitting the model: dense/shared experts + attention resident, stream only routed experts; early layers resident, late layers streamed; per-layer budgets by routing frequency

### Takeaway
Keeping dense/shared weights and attention resident is **already standard and bit-exact** (it is part of the "resident weights" baseline). Making *early* layers resident and streaming *late* layers, or optimizing per-layer budgets, is unproven and **cannot beat the ceiling** unless per-layer reuse distributions differ sharply — which the no-static-skew evidence argues against.

### Cited Findings
- Palm-Infra locks "always-used dense weights" in RAM and uses async pread only for routed experts; vLLM-metal keeps experts on disk and caches in unified memory with outputs "identical to fully resident"; omlx reports residency sweeps (100%/50%/25%/12.5% → 122.5/59.4/40.1/29.3 tok/s) — [Palm-Infra](https://github.com/TencentYoutuResearch/Palm-Infra/blob/main/docs/ssd-offload.md); [vLLM-metal](https://docs.vllm.ai/projects/vllm-metal/en/latest/moe_offload); [omlx](https://github.com/jundot/omlx/blob/main/docs/MoE_Expert_Offload.md).
- MoE-CORE uses nonuniform layer-wise cache capacity, domain-informed initialization, and routing-history-aware replacement; best config 21.5 ms TPOT on an appliance — [MoE-CORE](https://arxiv.org/html/2610.01950).
- Fiddler moves small activations to CPU rather than big weights to GPU: >3 tok/s for unquantized Mixtral-8x7B on a 24 GB GPU; bit-exact (execution placement only) — [Fiddler](https://arxiv.org/abs/2402.07033).
- SSD-LLaMA reorganizes each expert into one aligned block and uses a 3-tier cache; caching only a subset of projections (gate/up) reduces throughput despite higher hit rate — [SSD-LLaMA](https://arxiv.org/html/2609.18110).
- No static expert skew (top-32 ~10%) and identical per-layer top-k structure — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861).

### Inferences
- Per-layer budgets can only help if miss *penalty* or *reuse probability* varies by layer. Since every layer's top-k is on the critical path and routing skew is absent, an even allocation is close to optimal; concentrating cache in early layers just exposes late layers fully.
- Bit-exactness: yes — residency/placement only. **Cannot beat the ceiling**: it reallocates the same insufficient RAM.
- Falsifiable test: instrument per-layer expert-access frequencies and per-layer miss stall time over 20 prompts; if any layer's reuse probability or stall contribution is >2× the mean, try a nonuniform budget; otherwise kill.

### Gaps
- No per-layer routing-frequency or per-layer miss-stall data for the two target models on the M4.
- No source quantifies how much of the resident set is genuinely "dense/shared + attention" vs routed experts for these exact checkpoints.

---

## Key Question 6. Idea 6 — cross-layer expert dedup: are any experts byte-identical across layers (SHA-256 audit)? Can tied/duplicated experts share one resident copy?

### Takeaway
Almost certainly **no byte-identical experts exist across layers in a converged checkpoint**, so the SHA-256 audit is a cheap, decisive kill with a huge (2× working set) but implausible upside. Weight tying is a *training-time* architectural change; released checkpoints have independently trained experts. **Cannot beat the ceiling** (expected zero duplicates).

### Cited Findings
- MoRE (ICML 2026 workshop) introduces expert weight tying across adjacent layers specifically as a *new architecture* with a learnable depth embedding; standard MoEs "maintain a separate pool of experts at every layer, with no weight-space tying," and layers otherwise learn "redundant experts that serve overlapping roles" — [MoRE](https://ericqiu6.github.io/assets/pdf/_icml_workshop__MoRE__Mixture_of_Reused_Experts.pdf).
- "Tying the Loop" (2026): expert tying shares FFN weights across consecutive layers, "reduce memory footprint by almost 2× at virtually no degradation," but it is a pretraining architectural modification (single pointer assignment in HF) evaluated on pretrained-from-scratch OLMoE/Qwen3/DeepSeek-style models, including a Qwen3-MoE loss penalty of +0.02 (g=2) to +0.067 (g=4) — [Tying the Loop](https://arxiv.org/html/2606.16825).
- Expert Upcycling (2026) creates duplicate experts only at the *start* of continued pretraining, then "breaks the symmetry among duplicates to drive specialization" — [Expert Upcycling](https://arxiv.org/abs/2604.19835).
- IBM's shared-MoE patent dedups experts that are "common across multiple models," i.e. across separately served models, not within one converged checkpoint — [IBM patent summary](https://patentlyze.com/patent/shared-mixture-experts-ai-inference/).
- Model-Optimizer handles `_tied_weights_keys` by deduping byte-identical per-expert tensors by name when a checkpoint *does* declare tying — [NVIDIA Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer/blob/73d77842/modelopt/torch/export/moe_utils.py).

### Inferences
- Bit-exactness: if byte-identical duplicates existed, sharing one resident copy is trivially bit-exact. The question is empirical, not algorithmic.
- Predicted result: SHA-256 over each expert's packed bytes across all layers will find ~0 cross-layer duplicates; any near-duplicates (from upcycling-like initialization) would not be bit-identical and therefore fail the gate.
- **Cannot beat the ceiling** in expectation. This is still the single cheapest decisive test in the note: a one-off offline hash pass over the expert files (no hardware, no engine changes).
- Falsifiable test: SHA-256 every per-expert byte range (all 48 Qwen layers × 512 experts, 23 Nemotron × 128), group by digest, report duplicate count and reclaimable bytes. Pass only if ≥1 GiB of unique experts collapse — predicted fail.

### Gaps
- No SHA-256 audit of these two checkpoints has been run; the ~0-duplicate verdict is an inference from how MoE checkpoints are trained, not a measurement.
- No source on whether Qwen3.8-Flash-Next or Nemotron were produced via upcycling (which could leave *near*-duplicates, still non-bit-exact).

---

## Key Question 7. Idea 7 — macOS memory compressor, purgeable memory, page-cache tiering: can we structure memory so SSD-resident experts are always served from a fast page-cache tier and only cold ones hit NAND?

### Takeaway
The page cache is **already the best available "fast tier"** on this machine, and it is already doing ~74% of the work. macOS pressure APIs do not give per-process control of the file cache, and the VM compressor *hurts* anonymous expert pools. Structuring memory cannot turn cold NAND reads into warm ones; it **cannot beat the ceiling**.

### Cited Findings
- `mlock` recovered ~0 ms; skipping expert streaming recovered 12–14 ms — the bottleneck was memory-system contention, not resident-page eviction — [os_io_tricks.md IO-06](file:///Users/vincent/Documents/Veystrix/research_notes/MoE%20decode%20IO%20bypass%20research/os_io_tricks.md).
- Apple's `memorystatus` pressure levels and `DISPATCH_MEMORYPRESSURE_*` let cooperating apps drop `NSCache`/purgeable VM, but "purgeable-memory APIs do not give direct control over the file cache"; the `purge` tool is obsolete/churning — [XNU memorystatus doc](https://github.com/apple-open-source/macos/blob/master/xnu/doc/vm/memorystatus_notify.md); [mac-ram-cleaner](https://github.com/chumafox/mac-ram-cleaner); [macOS swap notes](https://github.com/daviddwlee84/dotfiles/blob/main/docs/tools/macos-swap.md).
- macOS compresses memory and swaps under pressure with no reliable per-process knob for the compressor — [Apple StackExchange](https://apple.stackexchange.com/questions/340333/tell-macos-to-only-use-swap-and-compressed-memory-when-really-needed-macos-10-1?rq=1).
- Palm-Infra pins its expert cache precisely because the macOS VM compressor otherwise compresses cold anonymous buffers and decompresses them inside matmul — [Palm-Infra](https://github.com/TencentYoutuResearch/Palm-Infra/blob/main/docs/ssd-offload.md).
- `F_RDADVISE` is a true non-blocking async prefetch on Darwin (24 MB issued in ~0.5 ms, pages resident ~5 ms later; 3.61–3.78 GB/s cold-ish probe), but the parent's speculative-read experiment caused token divergence and slowdown; advice-only prefetch is the one safe form — [os_io_tricks.md](file:///Users/vincent/Documents/Veystrix/research_notes/MoE%20decode%20IO%20bypass%20research/os_io_tricks.md).
- File-cache guidance: data cached in memory speeds access "until that memory gets swapped out to disk, at which point you pay the price for accessing the disk once again"; cache files you will access multiple times — [Apple File System Performance Tips](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/FileSystem/Articles/FilePerformance.html).

### Inferences
- The clean page cache is the natural tier for a read-only expert file: under pressure, clean file pages can be reclaimed and re-read from SSD without the compress/decompress cost that anonymous buffers incur. That is a structural reason it outperforms an application pool on 16 GB, and it supports the Q1 prediction that a sole anonymous pool will not win.
- Pressure-notification/purgeable tricks can at best marginally reduce contention; they cannot create capacity or change which bytes must cross the SSD bus. **Cannot beat the ceiling.**
- Bit-exactness: these controls do not alter byte values (timing only). The one caveat is that any prefetch must be advice-only — never substitute a predicted expert — because the parent measured token divergence with speculative reads.

### Gaps
- No published experiment controlling `vm_compressor` for ML inference on macOS; no per-process knob found.
- No measurement of how long RDADVISE-advised pages stay resident under a 16 GB guard/streaming pressure — [os_io_tricks.md](file:///Users/vincent/Documents/Veystrix/research_notes/MoE%20decode%20IO%20bypass%20research/os_io_tricks.md).
- No test of `NSPurgeableData`/`vm_purgeable` applied to expert blobs.

---

## Key Question 8. Idea 8 — 2024–2026 literature on expert cache tiering / MoE offload cache / memory hierarchy, and final ranking by plausibility × payoff

### Takeaway
The literature is rich but consistently confirms the same structure: (1) all large gains come from large caches, weak baselines, or precision changes; (2) bit-exact systems wins are I/O-path/layout/replacement/scheduling, not new memory hierarchy; (3) the only genuinely new 2026 mechanism, ZipMoE, is defeated by our Q4 format. **None of the nine ideas beats the 16 GB ceiling.**

### Cited Findings
- Tiering/predictive caches: ExpertFlow (up to 10×, 93.72% GPU-memory reduction, 91.96% hit) — [ExpertFlow](https://arxiv.org/html/2410.17954v2); SeqMoE (45% residency → 96.97% hit) — [SeqMoE](https://arxiv.org/html/2609.12978); Diff-MoE (global-hot/temporal-hot/cold) — [Diff-MoE](https://dl.acm.org/doi/full/10.1145/3712285.3759903); MoE-CORE (phase-split memory, nonuniform layers) — [MoE-CORE](https://arxiv.org/html/2610.01950); FlashMoE (ML replacement + SSD subdivision, 21%/51% over LRU/LFU in copy counts) — [FlashMoE](https://arxiv.org/html/2601.17063).
- Fine-grained/sliced caching: PagedMoE/SliceMoE reduce SSD bytes (2.08→1.41 GiB/tok) but execute reduced-bit experts → quality-risky, fail the gate — [PagedMoE](https://arxiv.org/html/2611.03173); [SliceMoE](https://arxiv.org/html/2512.12990v4). MoE-APEX (1.34–9.75×) substitutes low-precision miss experts → lossy — [MoE-APEX](https://dl.acm.org/doi/10.1145/3779212.3790187); FloE (9.3× compression) has 4.4–7.6% degradation — [FloE](https://arxiv.org/html/2505.05950v2).
- Exact offload systems on Apple Silicon / edge: Jakevin (explicit pread + sync reduction) — [Jakevin/moe-offload](https://github.com/Jakevin/moe-offload); streamlx (multi-MB span-merged preads, ~70% next-layer prediction, 3.4→9.2 tok/s from 8→18 GiB) — [streamlx](https://github.com/srcterm/streamlx); Palm-Infra (pinned cache + cross-layer prediction, 10.89→12.70 tok/s) — [Palm-Infra](https://github.com/TencentYoutuResearch/Palm-Infra/blob/main/docs/ssd-offload.md); OD-MoE (cacheless, 3.69 tok/s full precision) — [OD-MoE](https://arxiv.org/html/2512.03927).
- "Split I/O time" sibling systems (CoPilotIO OSDI'26, Weave, COMET, CoX-MoE) all overlap or partition compute resources, not the SSD byte wall — [direct_io_split_notes.md](file:///Users/vincent/Documents/Veystrix/research_notes/Cerebras%20inference%20and%20IO%20splitting/direct_io_split_notes.md).

### Inferences — final ranking (plausibility × payoff, with bit-exactness and ceiling verdict)

| Rank | Idea | Bit-exact? | Realistic gain / source | M4 feasibility | Falsifiable test | Beats ceiling? |
|---|---|---|---|---|---|---|
| 1 (test first: cheapest decisive, huge upside if true) | Cross-layer SHA-256 dedup (Q6) | Yes if duplicates | Expected ~0 duplicates; ~[MoRE](https://ericqiu6.github.io/assets/pdf/_icml_workshop__MoRE__Mixture_of_Reused_Experts.pdf), [Tying the Loop](https://arxiv.org/html/2606.16825) | Offline hash pass, no hardware | Hash all per-expert byte ranges; pass only if ≥1 GiB collapses | **No (predicted kill)** |
| 2 (must close: only untested A/B) | Sole pool + F_NOCACHE (Q1) | Yes | ≤page-cache parity minus ~8%; [Apple](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/FileSystem/Articles/FilePerformance.html), [RockDB](https://github.com/EighteenZi/rocksdb_wiki/blob/master/Direct-IO.md), [os_io_tricks.md](file:///Users/vincent/Documents/Veystrix/research_notes/MoE%20decode%20IO%20bypass%20research/os_io_tricks.md) | 2-day engine A/B | 200-token decode, sole 4/8 GB pool + F_NOCACHE vs page cache; gate on tok/s, hit rate, token IDs, RSS | **No** |
| 3 (fresh mechanism, killed for our format; small free win) | ZipMoE split-state / compressed BF16 scales (Q3) | Yes | ~0 on Q4 codes (95.2% zstd); ≤6% on scale/bias; [ZipMoE](https://arxiv.org/html/2601.21198), [compression_contradiction.md](file:///Users/vincent/Documents/Veystrix/research_notes/Second%20miracle%20no%20new%20hardware/compression_contradiction.md) | Offline re-measure; 3-thread decode on M4 feasible | Per-tensor scale/bias compression audit + SHA-256 roundtrip; pass only if >25% of expert bytes | **No** |
| 4 | Importance pinning / layer budgets (Q2, Q5) | Yes | top-32 ~10%; LRU-64 67%, LRU-128 81%; [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861) | Easy, but 2–4 GB pool already regressed | Trace replay hit-rate curves at 4/8 GB; pass only if target hit reached at ≤6 GB | **No** |
| 5 | macOS memory tiering / purgeable / compressor (Q7) | Yes | mlock ~0 ms; no file-cache control; [os_io_tricks.md](file:///Users/vincent/Documents/Veystrix/research_notes/MoE%20decode%20IO%20bypass%20research/os_io_tricks.md), [Palm-Infra](https://github.com/TencentYoutuResearch/Palm-Infra/blob/main/docs/ssd-offload.md) | Low | Measure whether any pressure API changes miss bytes/token | **No** |
| 6 | KV compression to free 2–4 GB (Q4) | **No** (all quant/eviction fail; management only) | FP8 2× KV but not bit-exact; [kv_prefill.md](file:///Users/vincent/Documents/Veystrix/research_notes/Faster%20Qwen%20Nemotron%20inference%20on%20M4/kv_prefill.md) | N/A under gate | None needed — disqualified on exactness | **No** |
| — (adjacent streams) | Overlap / prefetch / split-I/O (CoPilotIO, RDADVISE, E0/E1) | Yes if advice-only | E0 +25% cap; RDADVISE 3.6–3.8 GB/s probe; [os_io_tricks.md](file:///Users/vincent/Documents/Veystrix/research_notes/MoE%20decode%20IO%20bypass%20research/os_io_tricks.md), [direct_io_split_notes.md](file:///Users/vincent/Documents/Veystrix/research_notes/Cerebras%20inference%20and%20IO%20splitting/direct_io_split_notes.md) | Medium | End-to-end RDADVISE decode with token-ID gate | **No** |

### Inferences — honest bottom line
- **Every idea in scope is either non-beating or non-exact.** Tiering cannot manufacture fast memory, cannot reduce the Q4 byte count, and cannot make a 10–12 GB working set fit in 16 GB. The parent's "cache cannot reach targets" conclusion is confirmed with fresh mechanisms considered.
- **What is still worth doing** because it is cheap and falsifiable, in order: (1) SHA-256 cross-layer dedup audit (kill/2× upside); (2) sole-pool + F_NOCACHE A/B (closes the one untested knob); (3) scale/bias-only compression re-measure (≤6% free win); (4) an advice-only `F_RDADVISE` end-to-end decode test (adjacent stream, the only measured >wall cold-ish path). None of these is predicted to reach 15/5 tok/s.
- The only mechanisms that would change the conclusion are ones explicitly out of scope: more RAM, a faster SSD, or accepting quality loss (low-precision/substituted experts, reduced top-k) — [moe_expert_io.md](file:///Users/vincent/Documents/Veystrix/research_notes/Faster%20Qwen%20Nemotron%20inference%20on%20M4/moe_expert_io.md).

### Gaps
- No end-to-end bit-exact measurement on a 16 GB M4 for Qwen3.8-Flash-Next at any cache size; all high tok/s numbers are on ≥24–64 GB machines.
- ZipMoE has not been evaluated on Apple Silicon or on Q4-quantized experts; its 2× coverage trick is theoretically inapplicable to high-entropy U32 codes but not formally proven impossible to adapt.
- No public data on byte-identical experts within a converged Qwen/Nemotron checkpoint; the dedup verdict is an inference.
- Whether the freed page-cache memory from a sole-pool design could be successfully re-wired (without VM compression) remains unmeasured.
