# Second miracle — radical bit-exact approaches to beat the byte wall on a 16 GB M4 with no new hardware

Scope and honesty note. Every claim below is tagged with its epistemic status. The strongest new lead (Q1) comes from a single independent engineer's technical write-up, not a peer-reviewed paper; the two 2026 arXiv systems that would matter most (Q3/Q5) are single-author preprints; the LUT and EuroSys results are peer-reviewed. Where I could not find a source, I say so in Gaps. The framing constraint is inherited from prior work: at 873 MB/token and ~3.36 GB/s the SSD floor is ~260 ms; targets of 5 / 15 tok/s need ~200 / ~67 ms per token. Only mechanisms that *certifiably* read fewer bytes, or that eliminate the read entirely, can matter. Mechanisms that only hide latency or reduce arithmetic are labeled as such and ranked low.

Ordering: Q1–Q5 are **PHYSICALLY PLAUSIBLE** (ranked by plausibility × payoff for the decode byte wall). Q6–Q8 are **SCIENCE FICTION** for this product as constrained (either forbidden hardware, a circular dependency, or no evidence of existence), and are kept because the owner is willing to be first to try something.

---

## Q1 — (PHYSICALLY PLAUSIBLE, RANK 1) "Certify, don't predict": read fewer bytes and *prove* the greedy token is unchanged

### Takeaway
This is the one lead that genuinely transcends the proven-dead predictor line (E1 perfect oracle +25%, E3 free predictor recall 0.36, E4 gated off). Instead of guessing which weights matter, compute a cheap upper bound on each candidate contribution's effect and skip only what is *proven* unable to change the answer. On the LM output head this gives a lossless ~5× byte reduction and ~1.9× batch-1 decode speedup. The output head is small, so the direct win is only ~+6% end-to-end; the transferable prize is the *framework*: apply the same certificate one layer earlier, to the expert weights, so that most tokens certify without reading the full Q4 expert set.

### Cited Findings — feasibility / bit-exactness / bytes
- Feasibility on M4: HIGH for the head variant (a low-rank int8 shadow + a fused gather-GEMV, both plain Metal/MLX kernels). Bit-exactness: PASS at the *token* level — the returned argmax is mathematically proven equal to the dense argmax, with no accuracy knob. It is not bit-exact at the *weight* level (unread rows are never computed), which is the correct under-reading of "no quality loss" for greedy decoding. Byte reduction: ~5× on the head in the reported 7B setting; unknown on experts.
- Mechanism (from the source): build an orthonormal basis B (rank r ≈ 0.36 d) from PCA on real decode hidden states; the shadow score is the logit component inside B (exact), and Cauchy–Schwarz bounds the out-of-basis part by ρ·δ[v], giving a per-row ceiling U[v] the true logit cannot cross. Collapse to per-block ceilings U_b, open blocks in descending U_b, and stop the instant the best found logit m ≥ max ceiling of all unopened blocks. On Qwen2.5-7B the exact fetch closes after reading ~2.2% of the vocabulary on top of an 18% shadow read; measured on W7900 and H100 with argmax match 1.0000 and worst bound violation −2.73 (bound holds) — [Certified LM-head decode](http://knlp.io/lm_head_decode.html) (independent engineer's write-up, not peer-reviewed; reproduces kernel and CSVs on the page).
- Measured speedups: 1.97× batch-1 decode on W7900, 1.85× on H100 (graph); the byte reduction is model-driven and hardware-independent. Honesty caveat from the same source: the head is only ~7% of per-token decode traffic at 7B, so the full-step gain is roughly +6% — [Certified LM-head decode](http://knlp.io/lm_head_decode.html).
- The same source reports the negative result that motivates abandoning prediction: a trained small adapter cannot relocate the predictor-to-oracle gap ("a representational property of the base model"), so predict-then-fetch is both lossy and capped; certify-then-skip is neither — [Certified LM-head decode](http://knlp.io/lm_head_decode.html). This is independent corroboration of the in-repo E1/E3/E4 dead ends, with a different way out.
- Formal backing for the certificate exists: for strict greedy acceptance, if KL(p‖q) < M*(p) = p(x0)log₂[p(x0)/(p(x0)+p(x1))] + p(x1)log₂[p(x1)/(p(x0)+p(x1))], the draft argmax is guaranteed accepted; a simpler sufficient condition is margin γ_p > √(2ε) when KL ≤ ε (Pinsker) — [When Is a Draft Accepted? A Theory of Acceptance in Speculative Decoding](https://arxiv.org/pdf/2606.30265) (arXiv preprint, 2026). This turns "approximate logits" into a *provable* token-identical shortcut.
- The closest published MoE "read fewer expert bytes" result is explicitly NOT bit-exact: a ternary Hadamard-rotated sketch of gate/up weights (~30 bytes per 256-byte BF16 block, +7% resident memory) scores channels and reads only the top-k at full precision, giving 98.1% accuracy at 17.8% of routed-expert reads and 1.60–1.90× faster MoE blocks at batch 1 (up to 2.69× at batch 4); the remainder is approximated by a rank-32 map — [Skip MoE channels via top-k channel selection](https://learnaivisually.com/ai-explained/moe-channel-selection-topk-vs-calibrated-threshold) summarizing arXiv:2610.37386. This is the calibration point: reading 17.8% of expert bytes buys a lot but loses exactness. Adding a certificate on top of this sketch is, to my knowledge, unstudied.
- Anthropic's FlashProve is adjacent but orthogonal: it proves that an *integer* committed-weight MoE under greedy decoding produced given tokens, with a verifier that needs neither weights nor much compute; it explicitly does not cover floating-point implementations — [Provable inference for large language models](https://www.anthropic.com/research/provable-inference) (research prototype, 2026). It establishes that exactness-as-a-certificate is a live 2026 research direction.

### Inferences
- The arithmetic that killed prediction does not kill certification. E1 says a *perfect* prefetcher only buys +25% because bytes are still read; E3 says cheap predictors recall 0.36. Certification changes the question from "which bytes will be needed?" to "which bytes can be proven unnecessary?" — an entirely different, bit-exact operation.
- The expert case is harder than the head case in a specific, diagnosable way. The head needs only an argmax over independent rows, so rows with a provably lower ceiling can be dropped. An MoE expert produces a dense contribution vector that is *summed* into the residual stream; skipping it changes every coordinate. Therefore a direct head-style block certificate does not transfer to experts.
- A plausible transfer is a two-stage certificate at the *model output*, not per expert: (i) read a bounded subset of weight bits / sketched channels for all routed experts; (ii) bound the resulting error in the final logits by composing per-layer error bounds (router gating weights, down-projection norms, residual gain, RMSNorm Lipschitz constants); (iii) if the top-1/top-2 margin exceeds the bound, emit the identical greedy token and skip the remaining full-precision reads; otherwise fall back to the full read. This reads fewer bytes with token-exactness. No source found that does this for MoE experts — it is a genuine open niche, and the single highest-value experiment in this report.
- Error bounds must be computed as residual norms ‖h − Bq‖ directly, never as √(‖h‖² − ‖q‖²) (catastrophic cancellation makes the bound useless) — [Certified LM-head decode](http://knlp.io/lm_head_decode.html). The same numerical rule would apply to any expert-level certificate.

### Gaps
- No published attempt to certify greedy-token identity while reading only part of MoE expert weights, and no measurement of how tight a composed per-layer bound can be. This is the whole bet.
- No measurement of the top-1/top-2 logit margin distribution for Qwen3.8-Flash-Next or Nemotron-Lightning. If margins are typically small (near-ties), the certificate would rarely close and this collapses to a ~6% head win.
- The certified-head write-up is not peer-reviewed and the exact numbers are single-run/hardware-specific; the 1.97× must be reproduced before planning around it.
- **ONE falsification experiment:** instrument one decode layer; compute approximate expert outputs from high-nibble-only Q4 codes plus a rigorous layer-wise error bound propagated to the final logits; over 1,000 greedy tokens, record the fraction where (margin > bound). Kill if < ~30% certify. Gate: emitted token IDs must be identical on 100% of certified steps (byte-for-byte greedy match), and bytes/token must fall measurably. Do this offline before touching engine code.

---

## Q2 — (PHYSICALLY PLAUSIBLE, RANK 2) Exact context-hash / MoE FFN memoization across turns and users

### Takeaway
Repeat rates in real traffic are high (40–99% prefix sharing depending on workload), so exact memoization is a real lever — but almost all of it is *prefill*, not the decode byte wall. For decode, an exact memoization hit requires the full hidden state (and, for QSA layers, the selected KV blocks) to recur bit-for-bit, which in practice means generation loops, not ordinary prose. Bit-exact; unbounded payoff on repeats, ~0 on fresh text.

### Cited Findings — feasibility / bit-exactness / bytes
- Feasibility: HIGH (hash table over hidden-state/KV fingerprints plus a persisted value store). Bit-exactness: PASS if and only if the key includes every function input (hidden state, router scores, layer id, KV state); a cryptographic-strength fingerprint is needed to avoid collisions. Byte reduction: entire layers skipped on a hit, zero on a miss.
- Mooncake-derived Conversation trace: ~40% prefix caching ratio; Tool&Agent: 59%; 48% of Conversation and 76% of Tool&Agent requests share ≥50% of their prompt prefix, largely due to repeated system/tool prompts — [DualMap, arXiv:2602.06502](https://arxiv.org/html/2602.06502v1).
- Coding agents show the extreme end: 95.7% global token-weighted prefix hit rate, with misses concentrated at user-initiated steps where human thinking time exceeds eviction time; misses cause 3.8× more prefilled tokens than truly unique input — [TraceLab, arXiv:2606.30560](https://arxiv.org/html/2606.30560v2). A 7-day local-agent measurement reports 99.2% input-token cache hit rate (26,732 calls, 3.27B input tokens), explicitly token-weighted rather than per-request — [Doforu, Nov 2026](https://doforu.ai/blog/prompt-cache-hit-rate-agent-workloads) (vendor blog, self-reported).
- The ceiling is bounded even with unlimited storage: Kimi traces show at most ~55% of the cache could be reused — [Prefix caching and KV cache offloading](https://vidyal.ai/blog/prefix-caching/). Only exact prefixes reuse: one changed token near the top turns the whole prompt into a miss — same source.
- KV-cache precomputation is confirmed token-exact: loading a publisher-precomputed KV and continuing matches from-scratch prefilling (24/24 greedy tokens and logits-level match on Qwen3-4B) — [Can I buy your KV cache?, arXiv:2606.13361](https://arxiv.org/pdf/2606.13361).
- Crucial scope limit: prefix caching reduces TTFT (prefill), not decode throughput. Decode still reads all routed expert weights every token. Every source above measures prefill tokens or input-token share.

### Inferences
- For a single-user interactive assistant, session-prefix reuse will help TTFT substantially but cannot move tok/s; the 873 MB/token byte wall is untouched. The memoization idea only reaches decode on genuine exact state repetition — greedy repetition loops, which are a known failure mode of greedy decoding — [Decoding Strategies: Greedy, Beam, Temperature](https://www.compilenrun.com/docs/ai/llm-inference/llm-decoding-strategies).
- With the Qwen3.8-Flash-Next architecture, 3 of every 4 layers are GatedDeltaNet with a compact fixed-size recurrent state, so a full hidden-state hash is cheaper than for pure-attention models — but the 1-in-4 QSA layers need their selected KV blocks in the key, which is why only exact prefix/suffix repeats hit — [Qwen3.8-Flash-Next technical blog](https://qwen.ai/blog?id=qwen3.8-flash-next); [Qwen Sparse Attention gallery](https://sebastianraschka.com/llm-architecture-gallery/qwen-sparse-attention).
- The unbounded payoff claim is real but conditional on workload shape; the industry numbers are a reason to *measure Veystrix traffic*, not to assume the win.

### Gaps
- No measurement of the fraction of Veystrix requests that share prefixes, and none of exact hidden-state repeats during decode. This is the owner's question and only local logs can answer it.
- No source on cryptographic-fingerprint overhead versus the value of an MoE FFN output at token granularity.
- **ONE falsification experiment:** log 1,000 real Veystrix requests; (a) measure token-weighted shared-prefix fraction (expect 40–95%); (b) hash (layer, hidden state, router logits, KV fingerprint) at every decode step and count exact repeats. Kill the decode variant if exact decode-step repeats < 1%. Keep the prefill variant regardless if (a) is high.

---

## Q3 — (PHYSICALLY PLAUSIBLE, RANK 3) Cross-request persistent expert server and the 6.3% overlap mystery

### Takeaway
The 6.3% overlap number could not be located in the notes I reviewed; the closest local measurement is a +17 percentage-point LRU→optimal headroom for decode prefetch at low residency. The literature says a persistent, stateless expert pool is bit-exact and gives large amortization only with multiple concurrent tenants or overlapping expert sets. On a single-user 16 GB machine there is no second request to overlap with, so payoff is near zero; engineering routing locality would require changing routing, which is forbidden.

### Cited Findings — feasibility / bit-exactness / bytes
- Feasibility: HIGH (a daemon keeping a shared expert pool in RAM across requests). Bit-exactness: PASS (experts are pure/idempotent functions; a cache miss changes when weights are read, never which expert runs). Byte reduction: near-zero for one user on one machine; N-way amortization only with N concurrent users whose activated sets overlap.
- A real Mac implementation of exactly this design reports bit-identical greedy outputs with expert offload at 25–50% residency, and gives the crucial local measurement: at 12.5% residency the OS page cache served repeated misses at ~6.5 GB/s effective versus ~3.4 GB/s from SSD, so *lower* residency decoded faster because the RAM not used by resident slots went to page cache — [jundot/omlx MoE Expert Offload docs](https://github.com/LtheKid/jundot-omlx/blob/main/docs/experimental/MoE_Expert_Offload.md) (project documentation, self-reported, gemma-4-26b and DeepSeek-V4.1 on Mac). This is a free, bit-exact, on-commodity tier that the 3.36 GB/s byte-wall model may be undercounting.
- The same source quantifies headroom: LRU→optimal decode-prefetch headroom is +17pp hit rate at low residency; lower residency decodes faster (2.7→4.1 tok/s after slots warm) — [jundot/omlx](https://github.com/LtheKid/jundot-omlx/blob/main/docs/experimental/MoE_Expert_Offload.md).
- Published multi-tenant evidence: FaaSMoE shares stateless expert blocks across tenants, cutting per-tenant CPU 187.81%→54.4% and memory 36.25→12.04 GB with 6 tenants; block size ~20 experts is optimal — [FaaSMoE](https://arxiv.org/html/2604.26881v1). EaaS likewise disaggregates stateless experts from stateful attention — [EaaS, arXiv:2509.17863](https://arxiv.org/pdf/2509.17863).
- Why locality cannot be engineered without changing the model: adjacent tokens agree on only ~25% of a layer's expert set, so most prefetched experts are never read again — [Edge0, arXiv:2609.18063](https://arxiv.org/html/2609.18063v2). Even when experts are made literally bit-identical by tying at training time, top-1 cross-layer agreement is only 0.087–0.126 — [Expert tying, arXiv:2606.16825](https://arxiv.org/html/2606.16825v1).

### Inferences
- With one user and one SSD, "co-locating co-selected experts on the same device" is already true; the only transferable benefit is physical byte adjacency in the pack, which Veystrix already tried without gain (expert packing +70 GiB, no server gain, per internal report).
- The most useful transfer from Q3 is not a server but a *measurement*: if Veystrix's measured effective read bandwidth is really 3.36 GB/s while an offload engine sees 6.5 GB/s from page cache on a Mac, the byte wall may be partly an I/O-path artifact. That is a cheap, high-value check and is bit-exact.
- "6.3% overlap" is best interpreted as: with a single request stream there is nothing to overlap; the overlap that matters is layer L+1 reads against layer L compute (already tried as E0 overlap) or page-cache hits against SSD misses.

### Gaps
- The specific 6.3% figure is not present in the notes I reviewed; I cannot confirm its measurement basis.
- No source on IPC/daemon overhead for an on-machine expert server on macOS (FaaSMoE uses HTTP/FaaS with far larger overheads).
- **ONE falsification experiment:** measure Jaccard overlap of activated (layer, expert) sets between consecutive Veystrix requests. Kill the shared-pool idea if < ~20%. In parallel, measure achieved read bandwidth with and without page-cache bypass on a repeated-expert trace; if page cache beats the SSD by >1.5×, re-derive the byte wall before rejecting overlap ideas.

---

## Q4 — (PHYSICALLY PLAUSIBLE, RANK 4) GatedDeltaNet + QSA: smaller exact state, skipped layers, deferred materialization

### Takeaway
The architecture confirms that 3 of every 4 layers already compress history into a *fixed-size* recurrent state, so there is no large KV stream to attack in those layers; QSA spends its budget on up to 2,048 selected tokens in 1 of 4 layers. The one genuinely new 2026 result, DeltaLog, is bit-exact and cuts the recurrent-state *compute* tax by 1.19–1.69× — but reads no fewer bytes and therefore barely dents a byte-wall-bound token.

### Cited Findings — feasibility / bit-exactness / bytes
- Architecture: Qwen3.8-Flash-Next is 125B-total / 6B-active, 12 QSA layers + 36 GDN layers in a 3:1 schedule; GDN carries state S_t ∈ R^{d_k×d_v} with the gated delta recurrence S_t = α_t(I − β_t k_t k_tᵀ)S_{t−1} + β_t k_t v_tᵀ, y_t = S_tᵀ q_t — [Qwen blog](https://qwen.ai/blog?id=qwen3.8-flash-next); [Gradients & Tides reading notes](https://jianyuh.github.io/llm/architecture/training/2026/08/27/Qwen3.8-Next-GDN-QSA-Gated-Residual.html); [Raschka gallery](https://sebastianraschka.com/llm-architecture-gallery/qwen-sparse-attention).
- QSA: keys pooled into 4-token micro-blocks, average-pooled, RMSNorm'd, partial RoPE on 64/128 dims, top-512 blocks = 2,048 tokens + current tail; up to 7.6× prefill and 4.9× decode attention-kernel speedup at 1M tokens — [Qwen blog](https://qwen.ai/blog?id=qwen3.8-flash-next).
- DeltaLog: represents the recurrent state as a dense base plus a bounded log of compact updates; "in exact arithmetic, it is algebraically equivalent to eager dense-state updates" (no weight or recurrence change); 1.19–1.69× lower one-token decoding latency on H200 and 1.30–1.86× on RTX 4090, but only 1.08–1.20× end-to-end on Qwen3.6-35B-A3B and 1.05–1.12× on Kimi-Linear-48B — [DeltaLog, arXiv:2608.15533](https://arxiv.org/html/2608.15533v1).
- Gated DeltaNet-2 (decoupling erase and write) is a *new architecture*, not a transformation of an existing checkpoint — [arXiv:2605.22791](https://arxiv.org/html/2605.22791) (NVIDIA preprint). It cannot be applied to a frozen checkpoint without changing weights.

### Verdict (feasibility / bit-exactness / bytes)
- Feasibility: MEDIUM (DeltaLog is a kernel/scheduler change with an exactness proof). Bit-exactness: PASS in exact arithmetic; floating-point ordering of deferred updates must still be verified against the dense kernel and the greedy token stream. Byte reduction: ZERO — the gain is compute/latency.

### Inferences
- The "skip a layer from a smaller exact state" idea has no headroom in GDN layers: the state is already a compact fixed-size matrix that must be updated every token (the delta rule is data-dependent), and the dominant bytes are the expert FFN weights in every layer, which are independent of the recurrent state.
- DeltaLog's small end-to-end numbers are a useful calibration: even a 1.3–1.9× kernel-level win shrinks to ~1.05–1.2× in serving, consistent with Veystrix's other compute-only levers being irrelevant to a 260 ms I/O token.
- QSA's indexer is frozen and its routes are recorded during checkpoint forward and replayed on recompute — [NVIDIA NeMo AutoModel docs](https://docs.nvidia.com/nemo/automodel/latest/nemo-automodel/nemo_automodel/components/models/qwen3_8_flash_next/layers). That makes exact re-computation of a QSA layer from recorded routes possible, but again affects prefill/recompute, not decode expert bytes.

### Gaps
- No source quantifies GDN state-update time versus expert-read time on M4; the DeltaLog gains are on GPUs with very different bandwidth/compute ratios.
- No source reports whether any layer can be *omitted* with bit-identical output in a frozen model; the delta recurrence is data-dependent and has no known exact shortcut.
- **ONE falsification experiment:** time a single GDN state update on M4 and express it as a fraction of the ~260 ms/token budget. Kill this angle if compute is <5% of the token (expected), leaving DeltaLog as an optional cheap win only.

---

## Q5 — (PHYSICALLY PLAUSIBLE, RANK 5) Reforming the Q4 affine GEMV to read fewer bytes: LUT / T-MAC / FLUTE / "in-VRAM rate"

### Takeaway
Every 2025–2026 LUT-based Q4 kernel (T-MAC, FLUTE, LUT Tensor Core, SAIL) is bit-exact with respect to the quantized weights, but none reduces the bytes read from storage; they replace multiplies with table lookups and sometimes *increase* memory traffic when the LUT is large. The concept that genuinely reduces bytes is a compact in-memory code (QTIP trellis), which reads 2.4× fewer bytes at 2.27× the speed — but it represents an already-2-bit-quantized model, which fails the "no quality loss vs the shipped Q4 checkpoint" constraint. Net: this angle is a compute lever, not a byte lever.

### Cited Findings — feasibility / bit-exactness / bytes
- T-MAC (EuroSys 2025, peer-reviewed): LUT-based mpGEMM without dequantization or multiplication; mirror consolidation halves the table and is described as lossless; up to 4× throughput and 70% energy reduction vs llama.cpp; BitNet-b1.58-3B at 30 tok/s single core and 71 tok/s on 8 cores on M2-Ultra — [T-MAC paper](https://dl.acm.org/doi/10.1145/3689031.3696099); [Microsoft Research overview](https://www.microsoft.com/en-us/research/blog/advances-to-low-bit-quantization-enable-llms-on-edge-devices); [arXiv](https://arxiv.org/html/2407.00088).
- FLUTE (EMNLP 2024 findings): offline restructuring of the LUT-quantized weight matrix, vectorized/duplicated LUT; 2–4× faster than existing GEMM kernels at batch <32, group size 128, and 1.5–2× end-to-end — [FLUTE](https://aclanthology.org/2024.findings-emnlp.724.pdf). Again: arithmetic, not bytes.
- LUT Tensor Core (ISCA 2025): 1.44× compute density/energy efficiency over prior LUT accelerators; LUT-GEMM improves only GEMV cases — [ISCA paper](https://dl.acm.org/doi/10.1145/3695053.3731057); [arXiv](https://arxiv.org/html/2408.06003v3).
- SAIL (PACT 2026): near-cache C-SRAM LUT-GEMV, "bit-exact in-memory type conversion", up to 10.4× speedup and 19.9× tokens per dollar in gem5; explicitly requires adding 256×512-bit C-SRAM per LLC slice and one ISA instruction — [SAIL poster](https://pact2026.github.io/program/posters/pact26-final3.pdf); [arXiv](https://arxiv.org/html/2509.25853v1). Science fiction for a stock M4.
- The one byte result: on the same 252 projections of Qwen3-4B, the QTIP 2-bit trellis kernel reads 0.91 GB versus 2.18 GB for an unfolded Leech-lattice layout (2.40× fewer bytes) and runs 2.27× faster at near-equal fractions of the byte bound (61% vs 65%). The paper's "bit-exact" refers to decoded layouts verified against f64 and an f64 reference, *not* to preservation of the original 4-bit checkpoint — [Unfolding the Leech Lattice, arXiv:2609.02652](https://arxiv.org/html/2609.02652) (single-author preprint, Sep 2026). It also establishes that saving a byte only saves time while the kernel is bandwidth-bound; on an A100 the same formats lose to FP16.
- A useful performance warning for M4: "fetching a large LUT quickly saturates memory bandwidth, providing fewer operations per byte fetched and producing little net speedup" — [SAIL](https://arxiv.org/html/2509.25853v1). And the Leech paper attributes 39% of its gap to the byte floor to launch geometry, recovering 11.7% by fusing q/k/v and gate/up at identical bytes and bit-identical output — [Leech](https://arxiv.org/html/2609.02652).

### Verdict (feasibility / bit-exactness / bytes)
- Feasibility: HIGH for software LUT kernels, but payoff is compute-only on a byte-bound token. Bit-exactness: PASS (with ≥f32 accumulation and verification). Byte reduction: ZERO for T-MAC/FLUTE/LUT Tensor Core; positive only for genuinely lower-rate codes, which change the model.

### Inferences
- This closes angle 3 cleanly: there is no algebraic identity that lets a Q4 affine GEMV compute the exact same function while reading fewer weight bytes, because the shipped Q4 codes already sit near their entropy bound (entropy 3.86/4.0 bits, established in prior notes). A lossless re-encoding cannot beat that; a lower-rate code changes the weights.
- The residual value is the Leech paper's launch-geometry finding: if Veystrix's 3.36 GB/s includes kernel-launch or synchronization inflation, fusing q/k/v and gate/up is bit-exact and costs no bytes. This overlaps with the already-tried sync-removal work, so it is a calibration item, not a miracle.
- "Decode-Late" and "matters" were not locatable as distinct 2025–2026 techniques in these searches; no claims are made about them.

### Gaps
- No source on a bit-exact *lossless* representation of already-quantized Q4 codes below ~4 bits/weight; entropy argues none exists.
- No M4-specific GEMV time for Veystrix's actual expert shape; the Leech and SAIL numbers are GPU/gem5.
- **ONE falsification experiment:** instrument bytes fetched by the current Q4 expert kernel and compare (a) baseline, (b) a T-MAC-style LUT kernel. Pass only if bytes/token fall while greedy tokens stay identical. Expect bytes unchanged → kill this angle for byte reduction and keep fusion only as a cheap latency tweak.

---

## Q6 — (SCIENCE FICTION, RANK SF-1) Prefill-time precomputation of exact expert contributions for likely continuations

### Takeaway
Precomputing KV caches for known continuations is proven token-exact and useful for TTFT, but precomputing *expert outputs* for future decode tokens is blocked by a circular dependency: a token's hidden state at step t+1 depends on the expert outputs at step t, so you cannot know the inputs to the experts you wish to precompute until you have already run them. Only the prefill tokens' expert contributions are precomputable — and those are computed during prefill already.

### Cited Findings — feasibility / bit-exactness / bytes
- Feasibility: LOW-to-impossible for exact expert outputs. Bit-exactness: would PASS if inputs were known (expert functions are deterministic), but the inputs are not knowable ahead of time. Byte reduction: potentially large *if* the continuation were known, which is the blocker.
- The positive adjacent result is KV-only: precomputed KV caches can be serialized, loaded, and continued with token-exact greedy output (24/24 tokens and logits-level match on Qwen3-4B) — [Can I buy your KV cache?, arXiv:2606.13361](https://arxiv.org/pdf/2606.13361).
- The autoregressive blocker is basic: for temperature 0, t_i = argmax_t P(t | t_<i, x), so future tokens depend on the model's own previous outputs — [Decoding Strategies](https://www.compilenrun.com/docs/ai/llm-inference/llm-decoding-strategies). The hidden state fed to the router at step t+1 is a function of the expert output at step t.
- Prefill already reads every expert the prompt routes to, once per layer, at high bandwidth: ~226 of 384 experts per layer and 130 GiB total for a 433-token DeepSeek prompt at 8–9 GB/s — [jundot/omlx](https://github.com/LtheKid/jundot-omlx/blob/main/docs/experimental/MoE_Expert_Offload.md). Any "precompute exact expert contributions during prefill" scheme can only cover these prompt tokens, which are already computed.

### Inferences
- Storing "likely continuation" expert contributions in spare RAM is exactly the failed predictor idea (E1/E3) upgraded to a cache: it needs the future token sequence to be known, and even if guessed, the hidden states along that guessed path require running the full model — at which point no bytes were saved. Prefill-time computation cannot precompute decode-time expert inputs.
- A bounded, bit-exact variant does exist but adds nothing new: persist the *prefill* KV state and any prefetched expert weights that prefill touched, then serve decode from that state. This is Q2 + Q3, not a new mechanism.

### Gaps
- No source demonstrating exact precomputation of expert outputs for future decode tokens; the search found only KV precomputation.
- **ONE falsification experiment (logic, not code):** write down the dependency chain for token t+1's router input as a function of token t's expert outputs. If it includes the expert outputs themselves (it does), the scheme is circular and cannot be run. No hardware test needed — this is ruled out analytically.

---

## Q7 — (SCIENCE FICTION, RANK SF-2) Exact weight-dependence / linearity across experts

### Takeaway
No 2025–2026 source reports exact linear dependence among experts of a frozen checkpoint. All merging, tying, substitution, and low-rank sharing techniques either retrain the model or alter the computed output; some explicitly flip 16–25% of benchmark items. The one legitimate action is a cheap audit for byte-identical duplicate tensors, which prior work already recommends.

### Cited Findings — feasibility / bit-exactness / bytes
- Feasibility: essentially zero for a bit-exact transformation; the required linear dependence among full-rank weight matrices is mathematically implausible. Bit-exactness: FAIL for all cited techniques. Byte reduction: large but disqualifyingly lossy.
- Expert tying can cut unique FFN parameters up to 75% but is a *pretraining* modification — [Expert tying, arXiv:2606.16825](https://arxiv.org/abs/2606.16825). Meta-expert sharing likewise trains from scratch with 70–80% as many experts — [arXiv:2610.27598](https://arxiv.org/html/2610.27598v1).
- Cache-aware routing is not bit-exact: adding a bonus to resident experts' logits flips 16–25% of BBH items — [PSR, arXiv:2610.34721](https://arxiv.org/html/2610.34721).
- "Importance-driven expert scheduling" substitutes similar cached experts and is only "nearly lossless" — [arXiv:2508.18983](https://arxiv.org/html/2508.18983v2). Dynamic precision sharing (PagedMoE) is close to GPTQ-INT4/MXFP4, i.e. lossy — [PagedMoE, arXiv:2611.03173](https://arxiv.org/abs/2611.03173).
- The determinism caveat is relevant to any future exactness claim: conditioning routing on physical residency plus async prefetch moved perplexity 8.755→8.674→8.627 as "prefetch copies in flight" varied 1→2→4 — [PSR](https://arxiv.org/html/2610.34721).

### Inferences
- There is no algebraic identity that reconstructs one expert from peers exactly, so angle 7's premise fails. The only useful transfer is the *finding* that experts are similar, which motivates lossless residual coding — but that is already covered by the closed entropy analysis (3.86/4.0 bits).
- A hash audit of all (layer, expert) tensors is nearly free and could, in the unlikely event of byte-identical duplicates, give an exact dedup win; no source reports such duplicates for a 512-expert model.

### Gaps
- No published duplicate-tensor audit for a 512-expert-per-layer checkpoint; no rank study of stacked expert weight matrices.
- **ONE falsification experiment:** SHA-256 every (layer, expert) gate/up/down tensor and count exact duplicates; additionally compute the rank of the stacked matrices for one layer. Kill if 0 duplicates and rank = N × matrix rank (no dependence). Expected: kill.

---

## Q8 — (SCIENCE FICTION, RANK SF-3) Compute-in-memory / computational storage that is bit-exact on commodity hardware

### Takeaway
The 2025–2026 literature is unambiguous: in-flash GEMV, computational storage drives, and near-cache C-SRAM give the largest byte reductions and are explicitly bit-exact, but every one requires programmable NAND, a CSD, or a modified cache/ISA. The owner forbids new hardware, so this remains the highest-payoff, lowest-feasibility "miracle" and should be recorded as a product bet, not an experiment.

### Cited Findings — feasibility / bit-exactness / bytes
- Feasibility: ZERO on a stock M4 SSD. Bit-exactness: PASS in the cited systems (AiF targets in-flash GEMV; SAIL explicitly does bit-exact in-memory type conversion). Byte reduction: near-perfect — only activations and small outputs cross the host interface.
- AiF (ISCA 2025) integrates GEMV into flash chips to exploit internal flash bandwidth instead of the limited external interface — [ACM](https://dl.acm.org/doi/full/10.1145/3695053.3731073).
- DockerSSD reports up to 2× for I/O-intensive workloads and 7.9× for distributed LLM inference using containerized in-storage processing on an FPGA SSD prototype — [arXiv:2506.06769](https://arxiv.org/html/2506.06769).
- SAIL (PACT 2026) is the most explicitly bit-exact near-data design, with 10.4× speedup, ~2% hardware overhead and one ISA instruction — [SAIL poster](https://pact2026.github.io/program/posters/pact26-final3.pdf).
- InstInfer-class CSDs are "tightly coupled to specialized hardware"; commodity NVMe + GPU Direct Storage is offered as the low-cost alternative — [NeurIPS 2025 paper](https://papers.neurips.cc/paper_files/paper/2025/file/4431224d3762aa655f0aee4eaf04ff16-Paper-Conference.pdf). Samsung SmartSSD pricing cited in the same paper is ~$2,526 used — a product-bet datum, not a Veystrix option.

### Inferences
- No 2025–2026 source reports a bit-exact computational-storage path exposed on Apple Silicon; the ANE has no independent route to the SSD, and Core ML cannot consume the Q4 gather_qmm op — [local ANE integration report](reports/Apple%20Neural%20Engine%20integration.md). This confirms the "no bit-exact ANE path" dead end.
- The only software-spirit cousin that *is* available on commodity hardware is the OS page cache acting as a free near-data tier (~6.5 GB/s effective for repeated experts versus ~3.4 GB/s SSD in the Mac measurements of Q3) — [jundot/omlx](https://github.com/LtheKid/jundot-omlx/blob/main/docs/experimental/MoE_Expert_Offload.md). That belongs in Q3's falsification experiment, not here.

### Gaps
- No source on CSD availability or pricing for an M4-class product; no in-storage *MoE expert GEMV* implementation (ISP work targets attention/KV or generic analytics).
- **ONE falsification experiment (a procurement gate, not a bench):** can Veystrix obtain a CSD/programmable SSD with ≥2 GB/s internal bit-exact GEMV and a documented integer accumulation mode? If no, this stays science fiction for this product.

---

## Cross-cutting conclusion for the report writer

- The only idea in this report that both (a) reads fewer bytes, (b) is provably output-identical, and (c) runs on existing hardware is **Q1's certify-don't-predict framework**. Its direct head-level payoff is ~+6% end-to-end; its expert-level payoff is unknown and is the single experiment worth funding. It is the legitimate "second miracle" candidate precisely because it sidesteps, rather than repeats, the predictor ceiling that killed E1/E3/E4.
- Everything else either helps prefill (Q2), helps compute/latency only (Q3–Q5), is circular (Q6), has no evidence (Q7), or needs forbidden hardware (Q8).
- Two cheap, bit-exact checks should precede any new engine work: (1) log actual Veystrix prefix and exact-decode-state repeat rates (Q2); (2) verify the effective SSD bandwidth assumption against page-cache-served repeated-expert reads (Q3). Both can invalidate the 3.36 GB/s byte wall that makes all targets look impossible.
