# Sharing ONE physical expert read across MULTIPLE tokens/queries — bit-exact "second miracle" hunt on 16 GB M4

Scope: all six assigned angles. Hard constraints: no new hardware, no quality loss (bit-identical tokens/logits, exact routing). All three user-supplied arXiv IDs were **verified as real**: 2610.34440 (Lan, ReClan AI Studio), 2607.24434 (DraftExpert, Han), 2610.15417 (SpecFit, Cao et al.). Local measurements from the parent brief are labeled as such and are not externally sourced.

---

## Q1: Continuous batching / multi-request serving — can sending 2–8 requests together cut SSD bytes/token on a 16 GB M4?

### Takeaway
Batching is bit-exact but on local MoE it **multiplies** active experts and can worsen TPOT, so naive continuous batching loses. It can only win when paired with (a) attention–MoE overlap or (b) runahead scheduling that hides expert-fetch stalls. No Apple-silicon number exists; the winning published design (OSDI '26 dual-batch) gains ~1.56× system throughput at batch 1→2, not fewer SSD bytes.

### Cited Findings
- OSDI '26 (arXiv 2606.10493) states the local-batching penalty directly: "With small batch sizes and sparse expert activation, batching multiplies the number of active experts per step, amplifying memory-bandwidth pressure; consequently, each request experiences worse TPOT compared to non-batched local execution." — [OSDI '26 paper](https://arxiv.org/html/2606.10493)
- The same paper's fix — dual-batch attention–MoE interleaving plus intra-node prefill/decode disaggregation with zero-copy shared weights — sustaining <15% latency increase and ~50% throughput gains; per-layer attention ≈350 µs and MoE ≈450 µs, and batch 1→2 raised system decode 21.5→33.6 tok/s (1.56×) with ~16 tok/s still per request, batch 6 ≈34.7 tok/s. Hardware was dual-socket CPU + RTX 5090s, FP8/INT4 DeepSeek-V3, **not** an M4. — [OSDI '26 paper](https://arxiv.org/html/2606.10493); [Chinese technical breakdown with batch table](https://marsggbo.github.io/blog/2026/20260723-osdi-wang)
- Cohere MoESD blog: expected active experts vs batch size (BS1 8.0, BS2 14.7, BS4 25.4, BS8 40.6, uniform 8/15.5/29.1/51.6), i.e. batch 2 already touches ~1.84× the experts of batch 1; the SD gain curve peaks around **BS=4**. — [Cohere](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding)
- Cohere measured target-side verification T(4)/T(1) = **1.25× empirical** (Amdahl prediction 1.46×) at BS=1 with K=3, giving observed 1.95× SD speedup; the routed-expert fraction f is only 0.30–0.55 because the model has shared experts. — [Cohere](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding)
- Bevy (arXiv 2611.04794) uses **runahead inference of concurrent requests** instead of predicting routes: a request whose experts are resident runs ahead to reveal its next experts, which are then fetched while other ready requests compute — "without predicting expert selections (therefore lossless)"; up to 3.46× decode throughput of llama.cpp and 1.55× of the fastest prefetching baseline. It also reports that prefetching can be *harmful* when an expert cache is in device memory. — [Bevy](https://arxiv.org/abs/2611.04794); [Bevy §4.2–4.3](https://arxiv.org/html/2611.04794)
- XShare reports up to 30% fewer expert activations under standard batching, up to 3× lower peak GPU load in expert-parallel deployments and up to 14% throughput gains in speculative decoding — but it "maximizes aggregate gating mass over a per-batch subset," i.e. it changes which experts run, so it is **not bit-exact**. — [XShare](https://arxiv.org/html/2602.07265v1); [local expert_reuse notes](file:///Users/vincent/Documents/Veystrix/research_notes/MoE%20decode%20IO%20bypass%20research/expert_reuse.md)
- Cross-request expert overlap measured locally at only 6.3% (parent brief; no external source). Within-request consecutive-token union is 0.79–0.83 of two tokens' experts (parent brief). — Veystrix local measurements

### Inferences
- **Exactness:** batching, request coalescing, dual-batch interleaving and Bevy-style runahead are scheduling/data-movement only — same weights, same kernels, same routing ⇒ bit-exact (modulo reduction order). XShare and cache-aware routing are not.
- **Why it could win where K=2 lost:** K=2 lost because a *deduplicating kernel* runs at lower effective bandwidth. Batching/runahead keep the fast per-slot kernel and instead (i) keep more SMs/I/O busy, (ii) overlap attention and MoE phases, or (iii) overlap a fetch with another ready request's compute. None pays the dedup bandwidth penalty.
- **16 GB feasibility:** memory is the binding constraint — batch 2 needs ~1.8× distinct experts live per step (Cohere), so a 2–4 GB expert pool will thrash more and *increase* SSD bytes/token unless overlap more than compensates. On a single unified-memory M4 there is no separate CPU-MoE/GPU-attention bus to interleave as in OSDI '26, so the dual-batch trick must be emulated with MLX streams/Metal command queues; unproven. No new RAM required.
- **Falsifiable test:** serve 2, 4 and 8 concurrent fixed prompts through the existing engine; log physical SSD bytes/token, aggregate tok/s and per-request TPOT, and diff greedy token IDs vs batch-1. **Pass** only if aggregate tok/s ≥1.4× at b=2, per-request TPOT degrades <15%, and tokens are byte-identical. If bytes/token rises faster than throughput, kill (this is the OSDI prediction).

### Gaps
- No tok/s or bytes/token figure for continuous batching on Apple silicon / MLX / llama.cpp Metal.
- No source independently confirms the local 6.3% cross-request overlap figure.
- No published dual-batch attention–MoE implementation for unified memory (M4).

---

## Q2: A deduplicating expert-GEMM kernel for MLX/Metal — can it overcome the r(n) bandwidth penalty, and what n is needed?

### Takeaway
**No, not for a Qwen3-like model at realistic n.** The decisive result is arXiv 2610.34440: dedup is faster exactly where the union-byte share u(n) is below the bandwidth ratio r(n). On llama.cpp Metal and MLX-LM the per-slot kernel has the *higher* effective bandwidth, so dedup loses at n=2 everywhere and, for Qwen3/gpt-oss/DeepSeek-V2-Lite, still loses at n=8. Crossover is ≈n=8 only for OLMoE on llama.cpp Metal, "or never." A real-drafter run on llama.cpp Metal Qwen3-30B-A3B found plain decoding fastest on 33/37 held-out MT-Bench prompts.

### Cited Findings
- Lan (2610.34440) criterion: step time = requested bytes / effective bandwidth + fixed time; dedup wins iff u(n) < r(n). At n=2, u(2)=0.72–0.83 on four models while r(2)=0.32–0.75 on all engines ⇒ "at two tokens the deduplicating kernel is slower everywhere." — [arXiv 2610.34440](https://arxiv.org/html/2610.34440)
- Table 1 exact values (β in GB/s): **llama.cpp Metal Qwen3**: re-read cost 0.92–0.98; n=2 β_slot=131, β_dedup=65, r=0.50, u=0.77; n=8 β_slot=280, β_dedup=98, r=0.35, u=0.43 (**u>r at n=8 — no crossover**). **Metal gpt-oss**: re-read 0.98–1.04; r(2)=0.37, u(2)=0.72; r(8)=0.34, u(8)=0.36 (still loses). **Metal DeepSeek-V2-Lite**: r(8)=0.42, u(8)=0.62. **Metal OLMoE**: r(8)=0.51, u(8)=0.50 (crossover only at n≈8). **MLX-LM OLMoE**: r(8)=0.32, u(8)=0.50 (no crossover). By contrast **CUDA** OLMoE re-read cost 0.18–0.22 and gpt-oss 0.34, with crossovers by n≤8. — [arXiv 2610.34440 Table 1](https://arxiv.org/html/2610.34440)
- Kernel dispatch cause: on llama.cpp Metal an expert product goes to a tiled matmul only at ≥32 rows; below that a matrix-vector kernel multiplies each token's experts separately. MLX-LM switches to a tiled kernel only at ≥4·N/m tokens. CUDA's tiled kernel loads each selected expert once per tile for 9–128 tokens. This is why the per-slot path has higher effective bandwidth on Metal/MLX. — [arXiv 2610.34440 §D.1](https://arxiv.org/html/2610.34440)
- Nsight Compute profiles (70 profiles, OLMoE and gpt-oss, n=2..8): dedup kernel reads the union bytes Ub within 0.2–2.5%; per-slot kernel requests nmb but reads between Ub and 2.35·Ub. — [arXiv 2610.34440 §4.1](https://arxiv.org/html/2610.34440)
- Real draft experiment: Qwen3-30B-A3B (Q4_K_M) on llama.cpp Metal drafted by Qwen3-0.6B (Q8_0), greedy, 80 first-turn MT-Bench prompts, 37 with all draft lengths; **plain decoding fastest on 33/37**, median speedup 0.877 at k=1 and 0.517 at k=8. — [arXiv 2610.34440 §D.3](https://arxiv.org/html/2610.34440)
- Fit error ε is 0.031 on llama.cpp Metal and 0.148 on MLX-LM; predictions of crossover were computed *after* the measurements on three models (only DeepSeek-V2-Lite had pre-registered predictions). Bandwidth fit on other models misses a held-out model by 9–206%. — [arXiv 2610.34440 §5](https://arxiv.org/html/2610.34440)
- An MLX issue reports the dedup product at one row runs near half bandwidth at four rows (Haddad 2026), cited there as an unreviewed engineering report. — [arXiv 2610.34440 §1](https://arxiv.org/html/2610.34440)

### Inferences
- **Exactness:** a dedup kernel reads the same expert tensors and performs the same per-token products; bit-exact if accumulation order per expert is preserved. (The parent's earlier K=2 experiment already confirmed identical digests while being 16% slower — consistent with this paper.)
- **Why K=2 lost:** precisely because u(2)≈0.72–0.83 > r(2)≈0.37–0.59 on Metal; the saved bytes do not pay for the lower bandwidth.
- **n needed:** for Qwen3 on Metal, extrapolating the published r(n), u(n) trend, crossover would require n > 8 (roughly n≈10–12), at which point geometric acceptance and draft cost dominate — the paper's theory assumes zero draft cost and acceptance high enough to fill the pass. Therefore **dedup cannot deliver the second miracle on this engine at realistic n**.
- **16 GB feasibility:** memory is not the blocker (union bytes are *fewer* than per-slot bytes), but the engine must stage a larger contiguous union buffer; the blocker is kernel bandwidth, which no amount of RAM fixes. No new hardware helps.
- **Falsifiable test (microbenchmark, no model-quality risk):** pin one MoE layer; feed chosen-expert-id passes at n=2,4,6,8,12; measure bytes and time for both kernels; compute measured r(n) and u(n) from routing. **Pass only if u(n) < r(n) at some n≤12 with lower wall time and identical logits.** Given the table above, predicted outcome for Qwen is **kill**.

### Gaps
- r(n), u(n) for n>8 are not published; the n≈10–12 estimate is an inference, not a sourced number.
- No measurement on Veystrix's exact engine build or on an M4 (the paper uses llama.cpp Metal, MLX-LM, llama.cpp CUDA on other GPUs).
- No source tests a hand-written Metal dedup kernel that closes the bandwidth gap (e.g. persistent resident-expert tiles); leaving this as a physical limit is an inference, not a proof.

---

## Q3: Speculative decode whose draft and target SHARE loaded experts — DraftExpert, SpecFit, Cohere MoESD, EAGLE-3/Lookahead/Medusa variants

### Takeaway
This is the strongest "second miracle" candidate. It does **not** use a dedup kernel: it keeps the fast per-slot path but changes *which/how many* draft tokens are verified so that each loaded expert serves more accepted tokens, and prefetches the rest. DraftExpert and SpecFit are exact (frozen target verifies / unmodified target) and report 1.45× and 7–22% gains in CPU→GPU and Flash→NPU offload — the closest published analogue to SSD→M4.

### Cited Findings
- **DraftExpert (2607.24434)** mechanism: one lightweight accelerator-resident draft expert per layer; drafter = shared + top-1 + draft-expert (fixed footprint); self-distillation with residual, logit/token and **router-agreement** losses; confidence–expansion truncation (stop when q_i<τ **and** marginal verifier-set expansion Δ_i>B_i); draft-router expert prefetch. Exact mode: every emitted token verified by the frozen full MoE. Models DeepSeek-V2-Lite and Moonlight-16B-A3B, top-6/64 experts, CPU→GPU (RTX 4090 BF16) and Flash→NPU (Snapdragon 8 Elite Hexagon HTP v81, llama.cpp Q4_0). — [DraftExpert](https://arxiv.org/html/2607.24434)
- DraftExpert results: average decode TPS **1.45×** over AR offload; draft acceptance 84–87%; prefetch hit 86–88%. Exact-table: DS CG 2.19→2.99 TPS (1.36×), DS MN 10.18→15.47 (1.52×), ML CG 1.94→2.50 (1.29×), ML MN 8.50→13.69 (1.61×). Training-free shared+top-r is *worse than AR* (1.19, 4.80, 1.01, 3.82 TPS). Verify-side cost vs one-token AR: fixed-K 2.35×/3.10× → conf.+exp. 1.48×/1.70× → +prefetch 1.25×/1.35×. — [DraftExpert Tables 2–5](https://arxiv.org/html/2607.24434)
- **SpecFit (2610.15417)** mechanism: choose the ancestor-closed draft subtree maximizing predicted throughput = expected committed tokens / predicted cycle time, where cycle cost = per-node term + per-layer **union of activated experts** (monotone submodular coverage, each node pays only for new experts); exact search over subtrees; also decides when to stop drafting. Unmodified target verifies; does not prune or substitute routed experts. — [SpecFit](https://arxiv.org/html/2610.15417)
- SpecFit results (Qwen3-30B-A3B-Instruct-2507, A100 PCIe 40GB, pinned host experts, 48 LRU slots/layer; Red Hat EAGLE-3 drafter; batch 1): CNN/DailyMail **8.27 vs 6.80 tok/s (+21.6% vs cached EAGLE-3)**, GSM8K **10.79 vs 10.08 (+7.0%)**; 1.97×/1.72× vs vanilla EAGLE-3. It verifies fewer nodes (2.88 vs 7) and commits fewer tokens (2.21 vs 2.54) but shorter cycles (265.8 vs 371.7 ms). RTX 4090 24GB, 32 slots: 6.70 vs 6.05 tok/s with cache (+10.8%), 2.51× AR; uncached 3.89 vs 3.29 (+18.1%). GSM8K cached transfers 82.0 vs 92.7 expert weights/output token. — [SpecFit §4](https://arxiv.org/html/2610.15417)
- SpecFit caveats: route atlas prepared in separate reference runs and excluded from request timing; no ablation separating expert-union awareness from adaptive stopping; per-cycle ratio maximization not proven long-run optimal (predicted gap <1% under i.i.d. assumption); route predictors reach 84.7% / 94.7% accuracy on Qwen3-30B-A3B. — [SpecFit §4.6 / Limitations](https://arxiv.org/html/2610.15417)
- **Cohere MoESD**: adjacent-token expert overlap at step 1 = 0.381 (3.0 of 8 experts shared) vs independence 0.118 and uniform 0.0625; decays to 0.299 by step 4 but stays 2–4× above independence. Unique experts in verification only 1.25–1.42× one token at BS=1 (naïve worst case K+1). Observed 1.95× speedup at BS=1, K=3. — [Cohere](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding); [MoESD paper](https://arxiv.org/html/2505.19645)
- NOT bit-exact members of this family: MoE-Spec expert budgeting (drops long-tail experts — "introduces an explicit quality-efficiency trade-off"), Cache-Prior (flips 16–25% of BBH items), PSR, LightMoE similarity redirection. — [MoE-Spec](https://arxiv.org/html/2602.16052); [local expert_reuse notes](file:///Users/vincent/Documents/Veystrix/research_notes/MoE%20decode%20IO%20bypass%20research/expert_reuse.md)
- "Efficient MoE with SD via expert coactivation" (2609.22471) increases coactivation via global load-balancing loss, shared experts, autoregressive expert selection and consistency loss — all **training-time model changes**, not applicable to a frozen checkpoint. — [arXiv 2609.22471](https://arxiv.org/html/2609.22471)
- Cascade (2506.20675): speculation degrades 18/35 model–task pairs, slowdowns up to 1.5×; utility-aware gating limits worst-case slowdown to 5% and gives 7–14% over static-K. — [Cascade](https://arxiv.org/html/2506.20675v1)

### Inferences
- **Exactness:** DraftExpert, SpecFit and Cohere MoESD preserve bits because the target model and router are untouched; draft/truncation only change how many candidates are proposed and when weights move. Requires Veystrix's standard 500-token/20-prompt digest gate to confirm.
- **Why this wins where K=2 lost:** K=2 tried to serve 2 tokens from one dedup kernel and paid r(2)<u(2). These methods instead (i) avoid proposing tokens whose marginal union is large (truncation/subtree choice) so the *per-slot* kernel runs on fewer distinct experts, and (ii) prefetch the remaining loads so the per-slot kernel's high bandwidth is not idle. They exploit the same 20–38% temporal overlap but convert it into fewer loads / hidden latency rather than into a lower-bandwidth dedup kernel.
- **16 GB feasibility:** DraftExpert's per-layer draft experts are tiny (one small FFN per layer) and stay resident; SpecFit needs a route atlas or draft-router prediction (cheap, offline) and a subtree search on the critical path (paper says it must be cheap, exact search used); no large memory. Both fit on 16 GB. DraftExpert's Flash→NPU arm (10.18→15.47 TPS) is the closest hardware analogue to M4 SSD offload.
- **Falsifiable test:** replay a recorded 500-token routing trace from the existing drafter and implement SpecFit's oracle subtree selection *offline*; compare union expert-load count/cycle and projected cycle time to fixed-K verification. **Pass** if ≥15% fewer union expert loads per committed token and ≥10% projected tok/s, then confirm with one end-to-end run with identical token IDs. Cheaper ablation: apply DraftExpert's confidence–expansion truncation only (no training) and measure verify cost vs fixed K.

### Gaps
- H2D/Flash transfer cost (SpecFit/DraftExpert) is not identical to SSD→unified-memory paging on M4; the transfer-cost coefficients must be refitted.
- No Apple-silicon/MLX implementation or tok/s number for DraftExpert or SpecFit.
- SpecFit's route atlas is excluded from timing; an online route predictor's accuracy/latency on Veystrix's models is unmeasured.
- The parent's earlier K=2 experiment used MTP-style verify; whether that engine's drafter already overlaps loaded experts is not stated in the brief.

---

## Q4: Prefill→decode reuse — can decode reuse the experts prefill already read, exactly, within one request?

### Takeaway
Bit-exact reuse is real but only at the level of **expert residency/importance statistics**, not FFN outputs. Prefill routing predicts which experts matter in early decode; using it for cache warmup is bit-exact and partly exploited already. It is a modest lever, not a second miracle.

### Cited Findings
- SliceMoE (2512.12990) states: "experts that are frequently activated during prefill tend to remain important in the early decode phase, providing a clear opportunity for prefill-guided eviction during the phase transition"; its Predictive Cache Warmup progressively reshapes the cache during prefill. Cache state differs substantially between prefill (layer-wise, token-parallel) and decode (token-by-token). — [SliceMoE](https://arxiv.org/html/2512.12990v5)
- Context-aware CXL design (2512.04476): "For the same input sequence, the expert activation distributions from prefill closely match those in decoding" ⇒ expert placement is decided once after prefill using activation frequency and routing-score sums, with no further migration during decode. — [arXiv 2512.04476](https://arxiv.org/html/2512.04476v1)
- MoE-CORE (2610.01950): stages complete expert layers in alternating buffers during prefill, then uses nonuniform layer-wise cache capacity, domain-informed initialization, routing-history-aware replacement and cross-layer prefetch in decode; best measured config 21.5 ms TPOT with MTP depth 2 on an AI appliance. — [MoE-CORE](https://arxiv.org/html/2610.01950)
- OSDI '26: intra-node prefill–decode disaggregation with zero-copy shared weights and a dual-batch attention–MoE overlap holds concurrency interference <15% with ~50% throughput gain. — [OSDI '26](https://arxiv.org/html/2606.10493)
- MoE FFN outputs are per-position functions of the layer's hidden state (which attends over all prior tokens); the retrieved reuse mechanisms all cache/placement **weights or activations statistics**, not FFN results. — synthesis of [Cohere](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding) and [SliceMoE](https://arxiv.org/html/2512.12990v5)

### Inferences
- **Exactness:** using prefill-derived importance solely for cache admission/eviction is bit-exact (same weights, same routing). Dropping low-slice bits (as in some bit-sliced schemes) is not.
- **Why it can win where K=2 lost:** no dedup kernel involved; it removes future SSD reads by keeping the right experts resident, so it is a byte-reduction (fewer distinct reads) rather than a read-sharing (one read for many tokens) trick.
- **16 GB feasibility:** prefill statistics are tiny (L×E counters); no new memory. Works best with a 2–4 GB expert pool, which Veystrix has.
- **Falsifiable test:** record cache state immediately after prefill for 20 prompts; compare first 32 decode tokens' miss rate and SSD bytes under (a) prefill-guided warmup, (b) empty cache, (c) random resident set. **Pass** if (a) gives ≥20% fewer SSD bytes over the first 32 tokens with identical token IDs, and the advantage persists beyond token 32.

### Gaps
- No number for how long prefill-derived residency remains predictive on Qwen 512-expert/top-10 or Nemotron 128-expert/top-6.
- No M4/MLX measurement of prefill→decode cache-warmup benefit; parent describes it as "already partly exploited," so incremental headroom is unknown.

---

## Q5: Cross-token n-gram / context-hash memoization of MoE FFN outputs — bit-exact reuse on real prompts?

### Takeaway
**No bit-exact mechanism found; treat as a dead end.** The FFN input at layer l, position t is the residual hidden state produced by attention over the whole preceding context, so an identical surface n-gram at a different position has a different input and a different FFN output. Caching FFN outputs by token n-gram/context hash is therefore not bit-exact. The only exact memoization is KV/prefix caching, already standard.

### Cited Findings
- LongCat-Flash-Lite / N-gram Embedding (2601.21204) is the only work found combining "n-gram" with caching in MoE inference: an N-gram Embedding table trained as part of the model, with an "N-gram Cache ... inspired by the design principles of the KV cache" that caches **n-gram embedding lookups** during drafting to avoid redundant work in verification. It is a training-time architecture change, not memoization of FFN outputs for an existing checkpoint. — [arXiv 2601.21204](https://arxiv.org/html/2601.21204v1)
- MoE-Infinity's EAM traces which **experts** are activated (L×E counters), not FFN results; its savings come from caching/prefetching weights. — [MoE-Infinity](https://arxiv.org/html/2401.14361v1)
- Cache-conditioned shortcuts are demonstrably not output-exact: Cache-Prior (a cache bonus added to router logits) flips 16–25% of BBH items vs exact routing; PSR's logical cache is deterministic but still a biased router. — [local expert_reuse notes](file:///Users/vincent/Documents/Veystrix/research_notes/MoE%20decode%20IO%20bypass%20research/expert_reuse.md) citing [arXiv 2610.34721](https://arxiv.org/html/2610.34721)
- Semantic caches (e.g. Redis LangCache) explicitly match *similar meaning*, not bit-identical state — the opposite of this requirement. — [Avi Chawla blog](https://blog.dailydoseofds.com/p/moe-inference-engineering-clearly)
- Adjacent-token expert-set agreement is only ~25% (parent brief / [2609.18063](https://arxiv.org/html/2609.18063v2) as cached locally), so even an "n-gram of expert sets" admission signal cannot assume near-identical reuse. — Veystrix local notes

### Inferences
- **Exactness:** fails. Bit-identical token text does not imply bit-identical hidden state at a different position; no source supports FFN-output reuse across positions.
- **16 GB feasibility:** irrelevant — disqualified on exactness before memory matters.
- **Falsifiable test (cheap offline kill):** for 10 prompts, record the full input tensor to one MoE FFN at every position where the same surface token n-gram occurs; hash the tensors. **Pass only if 100% of repeated n-grams have byte-identical FFN inputs and outputs.** Expected: immediate kill at the input-hash step. This requires instrumentation but no kernel changes.

### Gaps
- No published study explicitly attempting bit-exact FFN-output memoization across token positions was found after targeted searches; the nearest designs cache expert *weights*, *activations statistics*, or *embeddings*.
- No measured reuse fraction on real prompts for such a scheme (because the premise is unsound).

---

## Q6: Persistent expert server / shared cache across concurrent users — can it work given only 6.3% overlap?

### Takeaway
Bit-exact in principle (MRU/LRU caches keep the same weights), but the measured 6.3% cross-request overlap makes persistent cross-user sharing low value. The only worthwhile refinement is Bevy-style runahead, which shares *work*, not weights, and needs no prediction.

### Cited Findings
- MoE-Infinity: "Accesses concentrate on a subset of experts within each prompt, while the identity of that subset changes across prompts" — request-level skew, not a global popularity order. — [MoE-Infinity](https://arxiv.org/html/2401.14361)
- vLLM-metal MoE offload uses an MRU cache in unified memory and states "outputs are identical to the fully resident model"; it is a clean bit-exact baseline but every MoE layer waits for routing before reading experts. — [vLLM-metal docs](https://docs.vllm.ai/projects/vllm-metal/en/latest/moe_offload)
- FlashMoE's ML replacement policy beats LRU by 21% and LFU by 51%, translating to 22% and 35% "I/O reduction" — copy counts, not physical bytes. — [FlashMoE](https://arxiv.org/html/2601.17063v2)
- Bevy: runahead of concurrent requests, no route prediction, up to 3.46× vs llama.cpp and 1.55× vs the fastest prefetching baseline; prefetching can be harmful when an expert cache is resident. — [Bevy](https://arxiv.org/abs/2611.04794)
- Cross-request overlap 6.3% (parent brief, local measurement). — Veystrix local measurement

### Inferences
- **Exactness:** shared cache and runahead are bit-exact; cache-aware routing is not.
- **Why it could win where K=2 lost:** a persistent cache avoids the SSD read entirely (no kernel bandwidth question), and runahead overlaps the remaining read with another request's compute. But with 6.3% overlap the cache mostly holds wrong experts, so the byte saving is small.
- **16 GB feasibility:** a shared server process is feasible but consumes RAM that is already tight; serving concurrent users increases active experts per step (Q1), potentially raising total SSD bytes.
- **Falsifiable test:** replay two interleaved request traces through the existing cache; measure cache hit rate and bytes/token vs one request at a time. **Pass** if hit rate >20% and bytes/token drop ≥15% with identical outputs; given 6.3% overlap, predicted **kill**.

### Gaps
- No external corroboration of the 6.3% overlap figure.
- No bit-exact persistent-expert-server benchmark on Apple silicon.

---

## Overall ranking (plausibility × payoff)

| Rank | Lever (angle) | Bit-exact? | Published gain (source) | Why / why not on M4 |
|---|---|---|---|---|
| 1 | **SpecFit-style union-aware subtree selection + DraftExpert truncation/prefetch (Q3)** | Yes (target verifies) | 1.45× avg (DraftExpert); +21.6%/+7.0% vs cached EAGLE-3 (SpecFit); 1.52–1.61× on Flash→NPU | Highest plausibility: no dedup kernel, directly exploits 20–38% temporal overlap, proven on the closest offload analogues. Needs offline route atlas or draft router and a refit cost model. |
| 2 | **Bevy-style runahead / multi-request scheduling (Q1, Q6)** | Yes (no prediction) | 3.46× vs llama.cpp, 1.55× vs fastest prefetch | Targets the exact expert-fetch stall; needs ≥2 concurrent requests. No new hardware; untested on Mac. |
| 3 | **Dual-batch attention–MoE overlap at batch 2 (Q1)** | Yes | 21.5→33.6 tok/s (1.56×), ~50% throughput gain, <15% latency increase | Hardware mismatch: needs separable CPU-MoE/GPU-attention resources; M4 unified memory must emulate with streams. Batching alone multiplies active experts and can hurt TPOT. |
| 4 | **Prefill-guided expert-cache warmup (Q4)** | Yes (statistics only) | Prefill activation distribution matches early decode; concrete gains not quantified for our models | Modest, cheap, already partly exploited; incremental headroom unknown. |
| 5 | **Persistent shared cache across users (Q6)** | Yes | FlashMoE 22–35% fewer copies | 6.3% cross-request overlap ⇒ small byte saving; low priority. |
| 6 | **Deduplicating expert-GEMM kernel (Q2)** | Yes | Loses at n=2 everywhere; on Metal Qwen3 u(8)=0.43>r(8)=0.35; plain decode fastest on 33/37 real-draft prompts | **Killed on Metal/MLX at realistic n.** The per-slot kernel's higher bandwidth is a kernel property, not fixable by more RAM. |
| 7 | **N-gram/context-hash FFN-output memoization (Q5)** | **No** | None found | Premise unsound: FFN inputs are position-dependent; not bit-exact. |

### Honest bottom line
On Metal/MLX, "sharing" literally defined as *one physical read serving n tokens through a deduplicating kernel* **cannot win** at realistic n (≤8): the published crossover is n≈8 only for OLMoE on llama.cpp Metal and never within n≤8 for Qwen3, gpt-oss or DeepSeek-V2-Lite on Metal/MLX, and a real-drafter Qwen3 test was slower on 33/37 prompts. This is exactly consistent with the earlier K=2 result (16% slower, identical digest). The only plausible bit-exact "second miracles" are **not read-sharing at all**: they are verification-shape and scheduling techniques (SpecFit/DraftExpert truncation + Prefetch, Bevy runahead, dual-batch overlap) that reduce the *number of distinct experts loaded* or hide the remaining loads while leaving the fast per-slot kernel and exact target routing untouched. Rank 1 is the single best bet; test it offline against a recorded routing trace before touching engine code.
