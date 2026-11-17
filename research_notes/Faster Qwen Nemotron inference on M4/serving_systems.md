# Serving systems, batching/scheduling, and alternative inference runtimes for faster MoE 4-bit decode (and prefill) on a 16 GB Apple M4, with zero quality loss

Scope note: this is a single-user local macOS app (MLX on 16 GB M4). Targets: Qwen3.8-Flash-Next MoE 4-bit >=5 tok/s (currently ~2.3, SSD-read bound), Nemotron >=15 tok/s (currently ~6.7), plus prefill/TTFT. Hard constraint: bit-identical token digests and BF16 rounding — any batching/scheduling change must not alter per-request output. All findings below are 2024–2026 and flagged for Apple Silicon relevance and bit-exactness where the source supports it.

## Key Question 1: What is continuous/in-flight batching, what gains are reported, and could batching K>=2 token steps help an SSD-bound Mac?

### Takeaway
Continuous (in-flight/iteration-level) batching admits new requests into freed batch slots every decode step, dramatically improving *aggregate* throughput and TTFT under concurrency (1.4x–2.9x reported), but it does not speed up a single request's decode and for MoE it *reduces* per-stream tok/s because weight bytes read per step grow with the number of distinct experts a batch touches. On a single-user app, and specifically on an SSD-bound Mac where K=2 expert read-sharing was measured 16% slower, batching K>=2 steps is very unlikely to help decode — the evidence points the opposite way.

### Cited Findings
- Continuous batching replaces request-boundary static batches with per-step (iteration-level) scheduling; on vLLM, switching to per-step scheduling took throughput 1820 → 3120 tok/s and p95 TTFT 1.4 s → 0.9 s (ShareGPT-4k, 4-bit AWQ, 8x GPU) — [Alaya NeW Cloud docs](https://docs.alayanew.com/en/docs/tech-blog/2026-04-vllm-on-h100).
- Controlled benchmark (Qwen2.5-7B bf16, DGX Spark GB10): continuous vs static batching throughput ratio rose with output-length variance: 1.00x (constant 128), 1.15x (CV .28), 1.40x (CV .58), 2.60x (CV .94), 2.91x (CV .99). When all requests finish together, continuous batching gives *no* gain — [Towards Data Science benchmark](https://towardsdatascience.com/continuous-batching-beats-static-and-dynamic-by-up-to-291-in-vllm/).
- Original Anyscale result: continuous batching gave up to 23x throughput vs naive static batching on OPT-13B/A100 as sequence-length variance grew; vLLM more than doubled naive continuous batching by dynamically reserving KV space — [Anyscale blog](https://www.anyscale.com/blog/continuous-batching-llm-inference).
- **MoE-specific mechanism against batching**: "A decode step on a dense model reads every weight once whether one token or twelve are in the batch... A decode step on this MoE reads the router and the eight experts each token selected. One stream touches 8 of 256 experts per layer; twelve streams can touch up to 96, so the weight bytes read per step grow with the batch, and each stream's share of the step shrinks." MoE per-stream decode fell 208.3 → 52.5 tok/s from 1 to 12 concurrent requests (aggregate still favored MoE 1.4x) — [Scalably.io vLLM MoE vs dense benchmark, 2x RTX 5090](https://scalably.io/blog/moe-vs-dense-llm-inference-vllm-benchmark-rtx-5090).
- The same source explains the batch-size/MoE compute-starvation problem: "With DeepSeek-V3 (256 experts, k=8) and batch=32 in decode, each expert processes only 1 token on average: total compute starvation. To match the per-GPU throughput of a dense model, MoE needs batches >>10x larger... Wide-EP distributes the experts across many GPUs and allows larger effective batches per expert, at the cost of all-to-all comms that add milliseconds per step" — [lo0 tech blog, continuous batching + MoE](https://blog.lo0.es/en/posts/continuous-batching-orca-vllm-chunked-prefill-goodput/).
- vllm-metal brings this exact machinery (V1 scheduler, paged KV, chunked prefill, continuous batching) to Apple Silicon; it reuses mlx_lm's token-wise MoE layers "unchanged" on a packed token axis, replacing only attention with a paged varlen Metal kernel — [vLLM blog, vllm-metal v0.28.0](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0).
- Semantic caution: vllm-metal's MoE layers "process each token independently... on a packed token axis without knowing request boundaries," which is the design property that makes batching *numerically* safe per token (same token math regardless of batch membership) — [vLLM blog](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0).

### Inferences
- For a single-user app, continuous batching only yields the 2-3x gains when multiple requests overlap; with one request there is nothing to admit, so it cannot move the single-stream Qwen 2.3 tok/s / Nemotron 6.7 tok/s numbers.
- Batching K>=2 decode steps to share SSD expert reads is theoretically the inverse of the MoE batching penalty: each extra token step expands the union of experts that must be read per fused step. This corroborates the user's measured 16% slowdown for K=2 expert read-sharing. Expect batching to make the SSD-bound MoE *slower*, not faster.
- Continuous batching is still worth adopting as a *serving architecture* if the app fans out multiple requests (e.g., agent tool calls, multi-session): it cuts TTFT and improves aggregate throughput, and vllm-metal/vllm-mlx are the Apple Silicon options.

### Gaps
- No source directly measured batching K>=2 decode steps to share SSD expert reads on Apple Silicon; conclusion is a strong inference from the MoE weight-traffic argument plus the user's own negative K=2 measurement, not a published benchmark.
- No source quantified whether fused multi-step batching preserves bit-identical token digests vs single-step (token-wise MoE layers suggest yes for the math, but attention/reduction order could differ — untested).

## Key Question 2: Prefix caching, chunked prefill, and fused kernels — reported MoE tok/s and TTFT improvements

### Takeaway
Prefix/KV caching is the single biggest TTFT lever (78–92% TTFT reduction, up to ~2.3x aggregate throughput at high concurrency) and is bit-exact in the sense that cached KV is reused, not recomputed. Chunked prefill + mixed prefill/decode packing lowers TTFT under load. Fused MoE/attention kernels give ~13% decode throughput and ~12–17% TPOT gains. These help prefill/TTFT targets far more than single-request decode.

### Cited Findings
- SGLang HiCache tiered caching: adding an L3 tier raised DeepSeek-V4-Flash effective input throughput 9.4K → 145.5K tok/s (cache hit ~98%) and Inkling-Small 15.5K → 67.1K tok/s (96.8% hit) — [SGLang blog, HiCache](https://www.sglang.io/blog/hicache).
- SGLang prefix-cache hit-rate benchmark (GLM-5.2-FP8, MI355X): at concurrency 1, raising cache hit 30% → 90% cut mean TTFT ~78% (1425 ms → 299 ms) while TPOT stayed ~3.6–4.0 ms — [SGLang PR #37359](https://github.com/sgl-project/sglang/pull/37359).
- Qwen3.8 on GB200, warm shared prefix: at concurrency 32, cache-on gave 2.29x output throughput (966.6 → 2214.5 tok/s) and 92.5% lower p95 TTFT; at concurrency 1 throughput slightly *decreased* (−3.5%) despite better TTFT. Caveat: attention backend and page size also changed between cache-on/off configs — [Satyam Srivastava blog](https://satyam-53.github.io/blog/qwen-gb200-prefix-cache/).
- RunInfra SGLang audit: one shared 2048-token prefix at 90% hit, concurrency 32: cache on 2488 vs cache off 875 output tok/s (~2.8x); many distinct prefixes still favored caching — [RunInfra SGLang](https://runinfra.ai/engines/sglang).
- Chunked prefill + fused kernels on DeepSeek MoE (vLLM GB200): kernel fusion (RoPE+Quant+Q write, Concat K), NVFP4 MoE GEMM/dispatch, minimized chunking overhead → 26.2K prefill / 10.1K decode TPGS; chunk size default 256 tokens (`VLLM_MOE_DP_CHUNK_SIZE`), activation chunking 16K default — [vLLM blog, GB200 Part I](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1).
- vLLM QWEN3 Thinking fused MoE kernel config PR: +13.7% output tok/s, −12% median TPOT, −17% P99 TPOT (benchmark 1895 tok/s output after) — [vLLM PR #24330](https://github.com/vllm-project/vllm/pull/24330).
- SGLang on GLM-4.7 FP8 MoE (TP8): 65% faster TTFT from serving optimizations — [Hugging Face / Novita blog](https://huggingface.co/blog/novita/sglang-glm4-moe).
- AMD GLM-5.2 (256 experts, top-8) SGLang end-to-end tuning: at 1M-token input concurrency 1, throughput 21,877 → 156,598 tok/s (7.16x), avg TTFT 45.7 s → 6.34 s; GSM8K accuracy preserved — [AMD ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/glm5.2-mi355x-kvcache/README.html).
- Apple Silicon relevance: vllm-metal's packed varlen prefill+decode step, on Qwen3.6-35B-A3B 4-bit (8 requests, ~6K prompt tokens), cut batch wall time vs competitors; mlx_lm was +143% (10.99 s) when one long prompt was added, vllm-metal only −6% (3.64 s) — [vLLM blog](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0).
- Bit-exactness: prefix caching reuses existing KV (no recomputation), so cached tokens are numerically identical; the cited Qwen3.8 experiment used temperature 0 and still found the *configuration* (backend/page size) was a confounder, not caching itself — [Satyam Srivastava blog](https://satyam-53.github.io/blog/qwen-gb200-prefix-cache/).

### Inferences
- For Veystrix's prefill/TTFT target, a persistent prefix cache of the system prompt / repeated context is the highest-value, lowest-risk change: it is bit-exact by construction and gives the largest measured TTFT wins.
- Kernel fusion and packed varlen steps help decode modestly (~13%) and are numerically benign when they fuse the same math, but each fused kernel must be validated for bit-identical digest because reduction order can change.

### Gaps
- No published TTFT/throughput numbers for prefix caching or chunked prefill specifically on a 16 GB M4 with Qwen3.8-Flash-Next / Nemotron.
- No bit-exactness guarantee from the server projects: they report task-level F1 or throughput, not token-digest equality.

## Key Question 3: Alternative Apple Silicon runtimes — which is fastest for 4-bit MoE, by how much, and do they preserve bit-identical output?

### Takeaway
On Apple Silicon, MLX-based runtimes are generally fastest for single-stream 4-bit MoE decode (which Veystrix already uses); llama.cpp Metal is competitive/sometimes faster depending on architecture, Ollama is consistently slowest. For concurrency/TTFT, vllm-metal (and vllm-mlx) lead. No runtime guarantees bit-identical output — quantization formats and kernels differ, so switching runtimes is the *biggest* bit-exactness risk and should be rejected unless a digest test passes.

### Cited Findings
- Qwen3.5-35B-A3B 4-bit on M4 Max 128 GB: MLX ~130 tok/s native (~90–108 via HTTP), llama.cpp ~71 tok/s, Ollama ~43 tok/s; MLX HTTP >2x Ollama; HTTP path costs ~20% vs MLX Python. Different quants per engine (Ollama Q4_K_M, llama.cpp Unsloth Q4_K_XL, MLX native 4-bit) — [Ante Kapetanovic benchmark](https://antekapetanovic.com/blog/qwen3.5-apple-silicon-benchmark/).
- Mac Mini M4 16 GB, small MoE sweep: no universal backend winner — MLX-LM wins Granite-4.0-H-Tiny (59.4 vs 45.5 tok/s at long context), llama.cpp narrowly wins LFM2.5-8B-A1B (63.7 vs 55.0) and Trinity-Nano (61.1 vs 55.1); MLX-LM draws less power — [yuvrajsingh.io M4 benchmark](https://yuvrajsingh.io/posts/mac-mini-m4-moe-benchmark/).
- M4 Max hybrid Mamba-2 MoE greedy batch-1 decode (mlx-lm 0.31.3 vs llama.cpp b8680, Q4_K_M vs MLX 4-bit): Nemotron-3 Nano 30B-A3B 159.7 vs 86.2 tok/s (MLX ~1.85x); Granite-4.0-H Tiny 202.2 vs 117.3; Apple Core AI int8/fp16 exports comparable or slower — [mlx-lm Discussion #1847](https://github.com/ml-explore/mlx-lm/discussions/1847).
- vllm-mlx on M4 Max 4-bit: 21–87% higher throughput than llama.cpp; Qwen3-30B-A3B 109.7 vs 89.9 tok/s (1.17x), Nemotron-30B-A3B 121.8 vs 85.1 (1.43x); Qwen3-0.6B throughput scales 441 → 1642 tok/s at 16 concurrent (3.7x) — [arXiv "Native LLM and MLLM Inference at Scale on Apple Silicon"](https://arxiv.org/html/2601.19139).
- SiliconBench (9 Apple Silicon engines, M5 Pro): on Qwen3-0.6B BF16, vllm-metal alone >2x throughput from concurrency 1→16 on both chat and agent, and the only stack above 2x on both; at chat c=16 vllm-metal ≈ ollama; at agent c=16 vllm-metal +32%; only llama.cpp, vllm-metal, omlx pass completion+fidelity+coverage gates; ollama, mistral.rs, Apple sglang fail coverage of newer models — [SiliconBench arXiv](https://arxiv.org/html/2609.19169).
- SiliconBench larger MoE (Qwen3.6-35B-A3B 4-bit): at c=4 vllm-metal's throughput advantage over omlx is only 4–17% and cache-configuration-sensitive; omlx leads at c=1, vllm-metal keeps lower TTFT under load — [SiliconBench arXiv](https://arxiv.org/html/2609.19169).
- Bit-exactness: SiliconBench measured F1 task fidelity against NVIDIA A100, and "token-level comparison was dominated by chat-template differences"; all newer-model implementations matched fidelity reference within ~1.4pp, but this is *not* token-digest equality — [SiliconBench arXiv §6.3](https://arxiv.org/html/2609.19169).
- Ollama failure case: bare-GGUF import omitted Qwen3 ChatML delimiters causing ~92% parse failures; also streams ~254 content-free tokens on tool calls — [SiliconBench arXiv §5/§6.3](https://arxiv.org/html/2609.19169).
- msty was not covered by any source found in this research (gap).

### Inferences
- Veystrix already uses MLX, the fastest single-stream family on M4 for these MoE models; switching runtime for raw speed offers little upside and large bit-exactness risk (different quant formats, kernels, KV math). Prefer optimizing within MLX.
- vllm-metal/vllm-mlx are worth evaluating only as a serving layer for multi-request fan-out (TTFT/aggregate), with a digest-equivalence gate first.
- Different quantizations per engine (Q4_K_M ~4.5 bpw, MLX 4-bit gs64 ~4.5, Unsloth Q4_K_XL ~6.2 for 30B) mean cross-engine tok/s comparisons are not apples-to-apples and any engine switch violates the fix quant/bit-exactness requirement.

### Gaps
- No direct head-to-head of llama.cpp Metal vs MLX specifically on Qwen3.8-Flash-Next or the exact Nemotron model at the target quant on a 16 GB M4.
- No source reports token-digest equivalence between any two runtimes; msty excluded entirely.

## Key Question 4: Distributed / multi-device (exo, TP/PP across Macs) — can 2–4 Macs give 2x+ for a single request, and is quality preserved?

### Takeaway
For a model that already fits on one Mac, clustering almost always makes single-request decode slower or only modestly faster: tensor parallelism (TP) over Thunderbolt RDMA gives ~1.3x decode and ~1.7–2x prefill on two matched M-series Macs; pipeline parallelism (PP) over TCP regresses. exo's published 1.8x/3.2x (2/4 devices) TP claims are unverified by independent tests. For a model that doesn't fit, clustering is a capacity play, not a speed play. No source verified bit-identical output across distributed configs.

### Cited Findings
- Apple MLX distributed (M5 Ultra): Qwen 3.8 decode 19.69 tok/s single → 33.27 (2x TP) → 50.89 (4x TP); Mistral Medium 3.5 7.70 → 13.69 → 23.37; Muse Spark Open MoE 15.74 (2x PP) → 24.99 (2x TP) → 36.73 (4x TP). Two-machine TP can trail PP at very large contexts (65,536 tokens) for MoE; decode unaffected — [Apple ML research, MLX distributed inference](https://machinelearning.apple.com/research/mlx-distributed-inference).
- SiliconBench (Qwen3.5-35B-A3B 8-bit, 2x M4 Pro Mac mini TB5 RDMA, single request): EXO single 66.8 → two-node TP 87.0 tok/s at 512 ctx (1.30x; 1.30–1.37x across contexts); mlx_lm 49.3 → 63.3 (1.28–1.32x); llama.cpp PP-over-TCP 40.0 → 32.4 (0.80–0.84x, regression). Prefill sped ~2x at short ctx, 1.7–1.8x at 4K — [SiliconBench arXiv §6.4, Table 4](https://arxiv.org/html/2609.19169).
- exo README claims: pipeline parallelism 49.3 TPS single → 44.4 (2 nodes) → 39.7 (3 nodes) for Llama-3.2-3B on M4 Pro (regression); tensor parallelism 1.8x on 2 devices, 3.2x on 4 (project claim, model unnamed) — [SnackOnAI exo analysis](https://www.snackonai.com/p/exo-a-distributed-inference-engine-runs-a-trillion-parameter-model-on-a-thunderbolt-cable); [llmformac.com](https://llmformac.com/mac-llm-cluster).
- Heterogeneous exo cluster (M5 Ultra 256 GB + M3 Ultra 96 GB, Llama-3.3-70B 4-bit): JACCL/RDMA 2-node decode 17.5–18.2 → 18.7–21.6 tok/s (~15% gain), Ring regressed ~16%; prefill more than 2x slower sharded (4.6 s → 10.6–12.5 s). A 164 GB MoE shard ran at 0.86 tok/s — [echalupa.com measured exo test](https://echalupa.com/blog/exo-heterogeneous-mac-studio-cluster).
- Interface bandwidth: intra-Mac M2 Ultra unified memory ~800 GB/s vs Thunderbolt 4 usable ~2.5–3 GB/s (~250x slower), 10GbE ~1.1 GB/s; Llama-3.3-70B 4-bit: single M2 Ultra ~11.5 tok/s vs 2-Mac exo TB ~8.2, 10GbE ~6.9 — [Think Different blog](https://www.thinkdifferent.blog/blog/the-multi-mac-ai-cluster-insane-overkill-or-the-future/).
- exo has no PagedAttention, no speculative decoding, no continuous batching; JACCL RDMA needs identical macOS builds (26.2+) — [SnackOnAI](https://www.snackonai.com/p/exo-a-distributed-inference-engine-runs-a-trillion-parameter-model-on-a-thunderbolt-cable); [echalupa.com](https://echalupa.com/blog/exo-heterogeneous-mac-studio-cluster).
- Ring attention / long-context: not specifically benchmarked for single-request decode speedup in sources found; Apple's MoE TP at 65K context showed diminishing/negative returns for 2 nodes — [Apple ML research](https://machinelearning.apple.com/research/mlx-distributed-inference).

### Inferences
- The user's target of 2x+ for a single request from 2–4 Macs is only realistic for *prefill* (compute-bound), not decode, at moderate context. Decode gains are ~1.3x (2 nodes) / ~2.5x (4 nodes) on large matched Macs; on a heterogeneous or SSD-bound setup, expect regressions.
- This is especially true for an SSD-bound MoE: adding machines does not fix per-node SSD reads of routed experts, and TP all-to-all/broadcast adds millisecond-scale per-step overhead.
- Quality: TP/PP split math (all-reduce/reduction order) is a plausible source of bit-level drift; no source tested token digests, so treat as unverified for the hard constraint.

### Gaps
- No bit-exactness test of distributed inference.
- No ring-attention-specific single-request decode numbers; no 16 GB M4 multi-Mac test.

## Key Question 5: Systems tricks — CUDA-graph equivalent (Metal command buffer capture), fused/persistent kernels, NVMe striping/RAID, RAM disk, mmap weights

### Takeaway
The highest-value zero-quality-loss systems trick found is raising MLX's Metal command-buffer limits (`MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER`), which gave +5–14% decode on launch-heavy MoE with **bit-identical outputs and unchanged prefill/memory**. Expert streaming/prefetching from SSD (hot-expert pinning, mmap + MADV_WILLNEED, O_DIRECT) is directly relevant to the SSD-bound Qwen model. CUDA-graph-style replay is hard in MLX due to changing array addresses; Metal's capture is a profiling tool, and graph replay is not a production MLX feature.

### Cited Findings
- **Bit-identical +5–8% (MoE, M3 Ultra, GLM-5.3-Flash ~1600 kernels/token)**: defaults 35.68 → 33.94 ms/token at 20k ctx (+5%) and 30.91 → 28.71 ms short ctx (+7.7%) with `MLX_MAX_OPS_PER_BUFFER=1000`, `MLX_MAX_MB_PER_BUFFER=400`; output hashes bit-identical, prefill and peak memory unchanged. Dense Qwen3.8-27B 8-bit saw no gain (52.8 → 53.0 ms) — [frederic.ai profiling post](https://frederic.ai/posts/where-the-other-half-goes); [MLX issue #4521](https://github.com/ml-explore/mlx/issues/4521).
- **Bit-identical +11.9–13.7% (M5 Max, MoE)**: changing only `MLX_MAX_MB_PER_BUFFER=2048`, Qwen3.6-35B-A3B 8-bit MoE 100.5 → 112.5 tok/s (+11.9%) and 99.6 → 113.3 (+13.7%); generated tokens and sampled logits matched across settings. Dense models +0–2.6%. Custom fused expert kernels gave ~13% at defaults but only ~1–2% on top (same overhead). Cost: ~0.57 GB extra prefill peak for MoE, 1.3–2.1 GB for dense — [MLX issue #4521 comment, M5 Max](https://github.com/ml-explore/mlx/issues/4521).
- Mechanism: MLX/Device commits a Metal command buffer after 50 ops or when referenced distinct input buffers exceed 50 "MB" of elements; MoE weight matrices (27–113 MB) trip the byte rule ~once per layer → 45–90 command buffers per step, each ~20 µs GPU + CPU completion handler. Raising limits cuts boundaries to ~7/step — [MLX issue #4521](https://github.com/ml-explore/mlx/issues/4521); [frederic.ai](https://frederic.ai/posts/where-the-other-half-goes).
- Decode-step profile on Apple Silicon (19.9k ctx, 35.8 ms/token): 56% large matvecs (bandwidth-bound, 500–650 GB/s), 12% launch-bound small ops, 9% lightning indexer, 23% gaps/sampling/Python. Routed experts 9.6 ms/step at 520 GB/s ("as good as gather_qmm gets"). Compiling FFN block saved 2.5 ms/step; hyper-connections ~2 ms pure launch cost — [frederic.ai](https://frederic.ai/posts/where-the-other-half-goes).
- CUDA-graph-equivalent in MLX: lazy evaluation records a whole subgraph and batches it into few command buffers (similar payoff to CUDA Graphs but no capture/replay lifecycle); true capture+replay is hard because MLX array addresses almost always change, unlike llama.cpp's preallocated arena — [MLX issue #2358](https://github.com/ml-explore/mlx/issues/2358); [metalworking book](https://metalworking.vercel.app/book/16-arrays-that-record/). MetalGraph replay measured 3.09 ms with matching checksums in one experiment — [Hashnode](https://selfenrichment.hashnode.dev/revisiting-metalgraph-what-i-learned-about-mlx-replay-and-sidecar-performance). `mlx.metal.start_capture`/`stop_capture` and `MTL_CAPTURE_ENABLED` are for Xcode frame capture/profiling (50–100x slower under capture), not replay — [vllm-metal profiling docs](https://docs.vllm.ai/projects/vllm-metal/en/latest/profiling/); [MLX PR #707](https://github.com/ml-explore/mlx/pull/707).
- **Directly relevant to SSD-bound Qwen3.8-Flash-Next (Strix Halo/NVMe measurements)**: model reads 1.47 GB of routed experts per token (48 layers x 10 experts x 3.06 MiB); NVMe sequential 16 MiB 4.91 GB/s, random 3 MiB one-at-a-time 3.62 GB/s, 8 in flight 3.87 GB/s → uncapped ~2.5 tok/s, "so the cache decides the speed." Expert cache with 4608/6144 slots (75%) streamed 28–38 tok/s (37–39 after moving prefetch off decode path) on Qwen3-Coder-30B, vs 79–91 resident; correctness = exactly the all-resident perplexity (21.1029 / 20.9475) — [1bit.gg MoE expert streaming](https://1bit.gg/moe-streaming.html).
- Guanaco / expert streaming: mmap whole GGUF + `MADV_RANDOM`, `MADV_WILLNEED` prefetch of router-selected experts, `MADV_DONTNEED` eviction to a `max_experts` budget, EWMA hot-expert pinning, PILOT cross-layer transition-matrix prefetch, imatrix seeding — runs a 120 GB MoE in <4 GB RAM; disk streaming "in-place on the mmap" so bytes are identical — [neutron42-jp/qwen3.8flashnext-ssdstreaming](https://github.com/neutron42-jp/qwen3.8flashnext-ssdstreaming).
- Qianwen3.8-Flash-Next specific (Ryzen/RTX): spec decoding with draft model measured *slower* (2.0 vs 4.3 tok/s) because "the verify batch needs the union of the drafted tokens' experts, so a streaming target reads 3-5x the experts per step" — corroborates that multi-token batching hurts SSD-bound MoE — [neutron42-jp repo](https://github.com/neutron42-jp/qwen3.8flashnext-ssdstreaming).
- llama.cpp streaming expert PRs/discussions exist (e.g. `-ot exps=CPU --no-host --no-repack --load-mode mmap`, `O_DIRECT` reader pool, per-expert LRU/prefetch): "with a proper x4 adapter or native M.2 slot, throughput would roughly double" for a PCIe-3.0-x2-limited SSD — [llama.cpp PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294); [llama.cpp discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149); [moe-ssd-streaming-windows](https://github.com/tonbistudio/moe-ssd-streaming-windows); [deniz.in llama.cpp offloading guide](https://deniz.in/running-moe-llms-larger-than-ram-a-llama-cpp-guide-to-expert-offloading-and-ssd).
- Apple/M4 relevance of SSD read speed: NVMe RAID/striping across multiple drives and RAM disks were not benchmarked in sources found; on a single 16 GB Mac there is limited room for a RAM disk of large experts, and unified memory means OS page cache already acts as the expert LRU.

### Inferences
- The single best "free" win for Veystrix is setting MLX command-buffer env vars (verified bit-identical, +5–14% on MoE); this is a config change, not a code/quant change, so it satisfies the hard constraint.
- Because the Qwen model is SSD-bound at ~1.47 GB expert reads/token, the remaining lever is expert-locality: hot-expert pinning/prefetching and keeping the working set in unified memory (page cache) so SSD reads shrink — this is a *memory-locality* scheduling problem, not a batching problem.
- Multi-token batching (K>=2), speculative verification, and expert read-sharing all increase the union of experts read per step and are therefore counterproductive on this SSD-bound path — consistent across the 1bit.gg, neutron42-jp, and user's own measurements.
- NVMe striping/RAID-0 is plausible but unverified for Macs; a RAM disk is unlikely to help because the OS page cache already caches hot experts in the same unified memory.

### Gaps
- No Apple Silicon benchmark of NVMe RAID/striping or RAM disk for expert weights.
- No production Metal graph replay with bit-identical digests for MLX (only profiling capture and one toy MetalGraph checksum test).
- Fused persistent kernels for MoE on Metal: no published Apple Silicon numbers found beyond the ~13% fused-expert-kernel comment and vLLM CUDA fusion numbers.

## Key Question 6: Bit-exactness reality check against the hard constraint

### Takeaway
No serving paper or runtime project in this research guarantees bit-identical token digests across batch sizes or engines. They measure throughput, TTFT, task F1, or perplexity — not digest equality. The only sources that explicitly verified bit-identical outputs were (a) the MLX command-buffer-limit experiments (hashes matched) and (b) expert-streaming correctness (perplexity exactly equal to resident). The hard constraint should be enforced by an in-app digest test, not assumed from any runtime's claims.

### Cited Findings
- SiliconBench: task-level F1 within ~1.4–1.5pp of NVIDIA reference; "token-level comparison was dominated by chat-template differences" — i.e., no token-digest equivalence across engines — [SiliconBench arXiv §6.3](https://arxiv.org/html/2609.19169).
- MLX command-buffer-limit studies explicitly compared output hashes / sampled logits across settings and found them identical — [MLX issue #4521](https://github.com/ml-explore/mlx/issues/4521); [frederic.ai](https://frederic.ai/posts/where-the-other-half-goes).
- Expert streaming reproduced exactly the all-resident perplexity (21.1029 / 20.9475) — [1bit.gg](https://1bit.gg/moe-streaming.html).
- Prefix caching reuses KV rather than recomputing (numerically identical cached tokens), and temperature-0 tests still showed confounders from backend/page-size, not caching — [Satyam Srivastava blog](https://satyam-53.github.io/blog/qwen-gb200-prefix-cache/).
- literature notes output drift from kernel nondeterminism, mixed-precision, and cross-provider divergence as a known problem — [SiliconBench arXiv §2](https://arxiv.org/html/2609.19169).

### Inferences
- Safe candidates under the hard constraint: prefix caching (KV reuse), command-buffer limit tuning, expert pinning/prefetch (mmap bytes identical), and math-preserving kernel fusion — all have source-level evidence of exact/identical outputs and can be gated with a digest test.
- Unsafe/needs gating: runtime switches (different quant formats/kernels), batching multiple token steps (different reduction order in attention/mixed steps), distributed TP/PP (all-reduce order), chunked prefill changes (different prefill chunk math), and speculative decoding (already shown slower for SSD MoE).
- Given the app is single-user, the realistic speed path that preserves quality is: tune MLX command buffers → add prefix cache for TTFT → improve expert memory locality (pin/prefetch) → consider vllm-metal only for multi-request fan-out with a digest gate. Do not chase continuous batching for single-stream decode.

### Gaps
- No published bit-exactness study comparing batch size 1 vs K>=2 on MLX/Metal for these MoE models.
- No quantified BF16 rounding differences for candidate fused kernels on Apple Silicon.
