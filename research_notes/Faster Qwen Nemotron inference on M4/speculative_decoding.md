# Exact (bit-identical) speculative / parallel decoding for SSD-bound MoE on Apple M4

Scope: methods whose **output is provably identical to greedy target decoding** (or where the relaxed variant is explicitly flagged), with reported speedups, hardware, and memory overhead. Cross-checked against Veystrix ground truth: Qwen3.8-Flash-Next (MoE, 4-bit, ~2.3 tok/s, ~809 MB physical reads/token, SSD-read bound), Nemotron (~6.7 tok/s); native MTP depths 1–8, prompt-lookup v2, n-gram drafting, adaptive draft lengths, and K=2 read-sharing already measured and **rejected** here.

## Q1. Catalog of exact speculative / multi-token methods, with numbers and exactness verdict

### Takeaway
Greedy speculative sampling with target-side accept/reject is provably lossless, and the literature reports large speedups on **dense, GPU-resident, bandwidth-bound** models (EAGLE-3 5.5–6.5×; ReDrafter 2.8× H100 / 2.3× Apple Silicon; SpecInfer 1.5–2.8×). But every method that makes MoE verification cheap (expert gating, expert budgeting, early-exit routing changes) **breaks bit-identical output**, and every method that is cheap on a GPU needs either training, a separate accelerator, or reads more experts per accepted token — the exact failure mode already measured in this repo.

### Cited Findings

**Consistent apples-to-apples greedy comparison (temperature = 0, Vicuna 13B, mean over MT-bench/HumanEval/GSM8K/Alpaca/CNN-DM), from EAGLE-3 paper Table 1** — [EAGLE-3, arXiv:2503.01840](https://arxiv.org/html/2503.01840v1):
- Standard speculative sampling (Vicuna-68M draft): **1.92×** (average acceptance length τ=2.24) — [EAGLE-3](https://arxiv.org/html/2503.01840v1)
- Prompt lookup decoding (PLD): **1.74×** (τ=1.80) — [EAGLE-3](https://arxiv.org/html/2503.01840v1)
- Lookahead: **1.62×** (τ=1.67) — [EAGLE-3](https://arxiv.org/html/2503.01840v1)
- Medusa: **2.12×** (τ=2.51) — [EAGLE-3](https://arxiv.org/html/2503.01840v1)
- Hydra: **2.80×** (τ=3.50) — [EAGLE-3](https://arxiv.org/html/2503.01840v1)
- EAGLE: **3.07×** (τ=3.96) — [EAGLE-3](https://arxiv.org/html/2503.01840v1)
- EAGLE-2: **4.26×** (τ=4.83) — [EAGLE-3](https://arxiv.org/html/2503.01840v1)
- EAGLE-3: **5.51×** mean, up to **6.47×** HumanEval (τ=7.54) — [EAGLE-3](https://arxiv.org/html/2503.01840v1)

**Per-method detail and exactness verdicts:**

- **Vanilla speculative sampling / assisted generation** — EXACT. Accept/reject against target probabilities; Leviathan et al. Appendix A.1 proves consistency with vanilla autoregressive decoding — [EAGLE-3, §2.1](https://arxiv.org/html/2503.01840v1). Speedups above; memory overhead = separate draft model (e.g. 68M for 13B target).

- **EAGLE / EAGLE-2 / EAGLE-3** — EXACT (greedy and sampling). EAGLE-3 "does not modify the target model's weights and uses strict speculative sampling acceptance conditions, ensuring no loss in performance"; the paper explicitly says it does not evaluate generation quality for that reason — [EAGLE-3, §4 Metrics](https://arxiv.org/html/2503.01840v1). Draft = one Transformer decoder layer + FC layers fusing low/mid/high target features; trained on ~68K ShareGPT + 464K UltraChat (~8× EAGLE's data) — [EAGLE-3](https://arxiv.org/html/2503.01840v1). EAGLE-3 speedup 3.0–6.5× on Llama 3.1 8B / 3.3 70B / DeepSeek-R1-Distill 8B, GPU (PyTorch/vLLM); throughput gain collapses with batch: **1.75× at batch 1 → 1.01× at batch 56** — [EAGLE-3, §4.3 Table 3](https://arxiv.org/html/2503.01840v1). Caveat: EAGLE-2/3 depend on a trained, model-family-specific drafter; S2-MoE reports EAGLE-3 gains "vary significantly with draft prediction quality and training characteristics" and are unstable across MoE families — [S2-MoE, §6.2.1](https://arxiv.org/html/2608.15018v1). P-EAGLE adds up to 1.69× over vanilla EAGLE-3 by parallelizing draft steps in vLLM — [AWS blog](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/).

- **Medusa / Hydra (multi-token heads)** — **QUALITY-RISKY at T>0; conditionally exact at greedy with strict verification.** Medusa-1 (frozen backbone) is described as "lossless inference acceleration" (>2.2×), while Medusa-2 jointly fine-tunes the backbone (2.3–3.6×) and needs "a special training recipe that preserves the backbone model's capabilities" — [Medusa, arXiv:2401.10774](https://arxiv.org/abs/2401.10774). The EAGLE-3 authors explicitly state: "Methods like Medusa relax acceptance conditions under non-greedy settings, which do not guarantee lossless acceleration. Therefore, we do not compare EAGLE-3 with these methods when temperature=1" — [EAGLE-3, Table 1 note](https://arxiv.org/html/2503.01840v1). Medusa's "typical acceptance scheme" boosts acceptance "while maintaining generation quality" but is not bit-identical — [Medusa](https://arxiv.org/abs/2401.10774). Hydra (sequentially-dependent Medusa heads) 2.80× mean but same family of acceptance relaxations — [EAGLE-3 Table 1](https://arxiv.org/html/2503.01840v1); Hydra paper arXiv:2402.05109. Extra heads add LM-head/weights memory. **Verdict: do not rely on Medusa/Hydra for a zero-loss guarantee.**

- **Lookahead Decoding** — EXACT (lossless). "A new lossless, parallel decoding algorithm to accelerate LLM inference without needing any auxiliary component" — [Lookahead, arXiv:2402.02057](https://arxiv.org/html/2402.02057v1). Uses Jacobi iteration to build and verify n-grams from the model's own outputs (zero-cost draft). Up to 1.8× MT-bench, ~1.5× across several settings, 4× with strong scaling on multiple GPUs in code completion; no finetuning/draft model — [LMSYS blog](https://www.lmsys.org/blog/2023-11-21-lookahead-decoding/); [arXiv:2402.02057](https://arxiv.org/abs/2402.02057). Note: this is the n-gram/Jacobi family already rejected empirically here.

- **Prompt lookup / suffix automata (PLD)** — EXACT (candidate tokens verified by target). Draft tokens are copied from the prompt/context via n-gram match; 1.74× mean in EAGLE-3's table — [EAGLE-3 Table 1](https://arxiv.org/html/2503.01840v1); [prompt-lookup-decoding repo](https://github.com/apoorvumang/prompt-lookup-decoding/). Already measured and rejected here (lookup v2: 7.30→6.15 Nemotron).

- **REST (retrieval-based speculative decoding)** — EXACT (standard speculative verification; drafts retrieved from a datastore). 1.62×–2.36× on 7B/13B, single-batch, code and text — [REST, arXiv:2311.08252](https://arxiv.org/abs/2311.08252); [NAACL 2024](https://aclanthology.org/2024.naacl-long.88/). Requires an external retrieval index; broadcast verification of retrieved n-grams. No Apple/MoE numbers found.

- **TriForce** — EXACT ("provably preserving model quality"): hierarchical speculative decoding = retrieval-based draft + hierarchical (multi-level) verification with KV-cache offloading for long sequences — [TriForce, arXiv:2404.11912v3](https://arxiv.org/html/2404.11912v3); [project page](https://infini-ai-lab.github.io/TriForce/). Reported **2.2× on a single A100** — [TriForce GitHub](https://github.com/Infini-AI-Lab/TriForce); up to **13.1×** in long-sequence scaling setups — [TriForce arXiv PDF](https://arxiv.org/pdf/2404.11912). Caveat: benefits are long-context/server-scale; no edge/MoE/4-bit evidence found.

- **SpecInfer (tree-based verification)** — EXACT ("maintaining identical output"): token-tree draft verified with tree attention; 1.5–2.8× vs SOTA serving systems, multi-GPU/distributed — [SpecInfer, arXiv:2305.09781](https://arxiv.org/abs/2305.09781); [alphaXiv](https://www.alphaxiv.org/abs/2305.09781). Tree attention verifies many candidates in one pass but, for MoE, the union of experts across all tree nodes is loaded (see Q3).

- **Staged speculative decoding** — EXACT ("preserving output fidelity"): cascade of draft models/stages verified progressively; targeted at small-batch on-device inference; up to **3.16× with greedy decoding** — [Staged SD, arXiv:2308.04623](https://arxiv.org/abs/2308.04623); [summary](https://www.emergentmind.com/topics/staged-speculative-decoding). Memory overhead = multiple draft models; not demonstrated for MoE here.

- **Early-exit / self-speculative (LayerSkip, Kangaroo)** — EXACT when late layers re-verify. Kangaroo is explicitly "Lossless self-speculative decoding via double early exiting" — [Kangaroo, arXiv:2404.18911](https://arxiv.org/abs/2404.18911); LayerSkip enables "early exit inference and self-speculative decoding" — [LayerSkip, arXiv:2404.16710](https://arxiv.org/abs/2404.16710). Draft = first N layers of the same model (near-zero extra weights). I found **no paper with the exact title "Speculative Decoding via Early-Exit"**; these two are the closest matches (gap noted below).

- **ReDrafter (Recurrent Drafter)** — EXACT under standard speculative verification (RNN drafter + beam search + dynamic tree attention; verification by target model at temperature 0). **Most relevant Apple data point:** up to **2.8× on Nvidia H100 (PyTorch)** and up to **2.3× on Apple Silicon Metal GPUs with an MLX implementation** — [ReDrafter, arXiv:2403.09919](https://arxiv.org/abs/2403.09919); [Apple ML research](https://machinelearning.apple.com/research/recurrent-drafter). Draft is a small RNN conditioned on LLM hidden states, trained by knowledge distillation — [ReDrafter](https://arxiv.org/abs/2403.09919). Caveats: (a) abstract does not use the word "lossless" — exactness is inferred from its use of standard target verification, not stated; (b) benchmarks are dense Vicuna, models resident in GPU/unified memory, not SSD-bound 4-bit MoE; (c) requires training a drafter per model.

- **Mirror Speculative Decoding (Apple, Dec 2025)** — likely EXACT at verification level but **infeasible here**. Uses early-exit signals for parallel rollouts and "explicitly maps computation across heterogeneous accelerators (GPU and NPU)" plus multi-token speculative streaming; 2.8–5.8× wall-time on SpecBench, 14B–66B, +30% over EAGLE3 — [Apple ML research](https://machinelearning.apple.com/research/mirror); [arXiv:2510.13161](https://arxiv.org/abs/2510.13161). Requires a separate NPU and server-scale models; MLX on M4 cannot use the ANE this way, and the multi-token streaming relaxation needs exactness review. Not a fit for a single 16 GB M4.

- **Quality-risky MoE-specific methods (DO NOT recommend under the zero-loss constraint):**
  - **S2-MoE** (self-speculative MoE on edge; llama.cpp; Jetson Orin 16/32/64 GB, RTX 4090): 1.3–5.3× Jetson, 1.2–2.9× RTX 4090; expert-level SSD offloading — [S2-MoE, arXiv:2608.15018](https://arxiv.org/html/2608.15018v1). **BUT** its own §6.2.2 states: "Reuse-aware expert gating intentionally introduces a bounded routing bias to improve expert reuse, and is therefore a controlled target-routing approximation rather than a strictly lossless transformation." Measured Top-1 agreement only 89–93%, PPL ratio 1.012–1.013, Mean KL 0.026–0.065 — [S2-MoE §6.2.2 Table 2](https://arxiv.org/html/2608.15018v1). **Fails bit-identical.**
  - **MoE-Spec (expert budgeting):** training-free, loads only top-scoring experts during verification, +15–45% throughput over EAGLE — [MoE-Spec OpenReview](https://openreview.net/forum?id=BnqJcRBl9D); [ADS](https://ui.adsabs.harvard.edu/abs/2026arXiv260216052M/abstract). Dropping experts the plain decoder would use changes logits → **quality-risky**; full exactness proof not accessible (OpenReview blocked by bot check).
  - **Adaptive Verification for MoE Speculative Decoding** (tree verification aggregates the union of experts across verified tokens and trims it) — [EMNLP 2026 Findings](https://aclanthology.org/2026.findings-emnlp.380.pdf). Trimming the expert union is an approximation → **quality-risky**.
  - **JetSpec** (claimed up to 9.64× "lossless" tree drafting) appears only in a Reddit research post in these searches — [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1ufntl5/research_jetspec_speculative_decoding_with/) — no primary source found; treat as unverified.

### Inferences
- All *provably* exact speedups in the literature are measured on dense, GPU-resident, bandwidth-bound models at batch 1. The single largest lever is **average acceptance length τ**; speedup scales roughly with τ divided by the verifier-cost ratio (see Q4).
- Medusa/Hydra "typical acceptance" and MoE routing/gating/budgeting relaxations are precisely the class the user must exclude; EAGLE-3 and S2-MoE both admit this in their own text.
- ReDrafter is the only method here with a published MLX/Apple-Silicon measurement, but it is a dense-model, memory-resident result; 2.3× should not be extrapolated to an SSD-bound 4-bit MoE.

### Gaps
- No published speculative-decoding result (exact or otherwise) on **MLX + 4-bit MoE + SSD-offloaded experts on an M4 16 GB** was found.
- No paper titled exactly "Speculative Decoding via Early-Exit"; closest are Kangaroo and LayerSkip.
- "S的性能" in the assignment most plausibly maps to **S2-MoE** (MoE self-speculative on edge); if a different paper was intended, that needs clarification.
- ReDrafter's exactness is inferred from standard verification; the abstract does not explicitly claim "lossless."

## Q2. Feasibility for MoE + 4-bit + MLX on 16 GB M4

### Takeaway
No surveyed exact method is a drop-in win here. The feasible-on-MLX options require training a drafter (EAGLE-3, ReDrafter) and still amplify expert reads; the zero-cost draft options (PLD, n-gram, lookahead) are already measured and rejected; the MoE-aware options that cut verification cost are explicitly lossy.

### Cited Findings
- S2-MoE is the only work that combines MoE + edge + SSD offloading + expert-level streaming, but it is implemented in **llama.cpp**, not MLX, on Jetson Orin/RTX 4090; 16 GB Jetson with SSD offload is the closest hardware analogue, where EAGLE-3 achieved only ~1.0–1.9× vs AR on Qwen3-30B-A3B and ~1.0–1.5× on DeepSeek-V2-Lite — [S2-MoE §6, Table 5](https://arxiv.org/html/2608.15018v1). (Raw table shows, e.g., Qwen3 HE: AR 0.47 T/s vs EAGLE3 0.89 T/s = 1.87×, S2-MoE 1.22 T/s = 2.60×; Qwen3 MT: EAGLE3 1.28×, S2-MoE 2.10×.)
- EAGLE-3 requires a trained, model-specific drafter (ShareGPT + UltraChat, ~8× EAGLE training data) and fuses low/mid/high target features — a non-trivial training and integration cost, and published EAGLE-3 checkpoints exist for Qwen3-30B-A3B (AngelSlim/Qwen3-a3B_eagle3) but not, from these sources, for the exact Qwen3.8-Flash-Next 4-bit build — [EAGLE-3](https://arxiv.org/html/2503.01840v1); [S2-MoE §5](https://arxiv.org/html/2608.15018v1).
- ReDrafter's MLX implementation and Apple-Silicon benchmark make it the most portable exact candidate, but it benchmarks dense Vicuna on H100/Metal GPUs — [Apple ML research](https://machinelearning.apple.com/research/recurrent-drafter); [arXiv:2403.09919](https://arxiv.org/abs/2403.09919).
- Mirror-SD requires GPU+NPU parallelism and 14B–66B server models — [Apple ML research](https://machinelearning.apple.com/research/mirror).
- Staged SD and SpecInfer need multiple draft models / multi-GPU serving — [Staged SD](https://arxiv.org/abs/2308.04623); [SpecInfer](https://arxiv.org/abs/2305.09781).
- 4-bit weight quantization is orthogonal to speculative decoding (EAGLE-3 lists quantization as a separate acceleration family); quantization of the *draft* is used only "when memory permits" in S2-MoE — [EAGLE-3 §5](https://arxiv.org/html/2503.01840v1); [S2-MoE §5](https://arxiv.org/html/2608.15018v1).

### Inferences
- A training-free, zero-cost exact drafter is the only easy MLX integration path, and that path is already empirically exhausted here (PLD/n-gram/lookahead/adaptive lengths all regressed).
- Any trained drafter added to a 16 GB M4 consumes unified memory that competes with the already-offloaded expert set; with ~809 MB physical reads/token, even a small draft that triggers extra expert reads is net negative — consistent with the measured 1.8× expert touches/token on K=2 verification.

### Gaps
- No MLX-specific memory/throughput profile of a speculative drafter on an M4 16 GB was found; no measurement of how MLX batches MoE expert gathers during tree verification.

## Q3. Exactness-preserving ways to reduce MoE verifier cost (shared expert reads, batched K>1)

### Takeaway
The only published mechanisms that substantially lower MoE verifier cost are (a) exploiting *natural* temporal correlation in expert routing, and (b) deliberately biasing/ trimming routing — (b) is not bit-identical. On an SSD-bound MoE, shared reads across draft tokens are exactly the lever that fails here (K=2 read-sharing measured slower).

### Cited Findings
- Natural expert overlap: for Cohere's MoE (k=8 of N=128 experts), adjacent-token overlap at step 1 is **0.381** (3.0 of 8 experts shared) vs 0.118 independence / 0.0625 uniform baselines; it holds across 13 Spec-Bench categories (0.377–0.385) and 7 languages — [Cohere blog, Apr 2026](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding). Verifying K=3 tokens (4 tokens total) activates ~**2.55×** top-k unique experts empirically vs 3.2–3.6× predicted by independence/uniform baselines — [Cohere blog](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding).
- Verification cost ratio: with shared experts, routed-expert fraction f≈0.30–0.55; for K=3, T(4)/T(1) ≈ 1.25 empirically (Amdahl estimate 1.46) at batch 1; much of the gain is "fixed-overhead amortization" of non-expert kernels, not expert sharing — [Cohere blog](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding).
- Formal throughput decomposition for edge MoE: Throughput = (tokens/iteration ÷ GB read/iteration) × bandwidth; increasing output density helps only if parameter density does not rise proportionally — [S2-MoE §1 Eq. 1](https://arxiv.org/html/2608.15018v1).
- S2-MoE's reuse ratio (unique experts in an interval ÷ autoregressive activations) stays high across layers/intervals, i.e. **little natural reuse**; it reduces the ratio from 0.60→0.45 (HumanEval) and 0.59→0.39 (GSM8K) only by the lossy routing bias — [S2-MoE §3, §6.3.3 Table 4](https://arxiv.org/html/2608.15018v1).
- Context alignment (draft and target share the same KV cache, draft KV updates rolled back, only accepted tokens committed) raises acceptance rates +24–79% without changing the target — [S2-MoE §4.3, §6.3.2 Table 3](https://arxiv.org/html/2608.15018v1). This is an *exactness-safe* component in isolation.
- MoE-Spec / Adaptive Verification reduce the expert union but change which experts run — **not exact** (see Q1).
- The repo's own K=2 read-sharing measurement (slower, ~1.8× expert touches/token) is consistent with S2-MoE's "limited expert parameter reuse" finding for MoE — [S2-MoE §3 Challenge 2](https://arxiv.org/html/2608.15018v1).

### Inferences
- There is no published *exact* trick that reduces the number of expert weights read during verification below the union required by the target's own routing. You can only (i) exploit overlap the model already has, (ii) amortize fixed overheads, or (iii) change routing (lossy).
- On an SSD-bound M4, fixed-kernel overhead amortization is a small fraction of wall time, so the Cohere-style 1.25× verifier ratio is optimistic; expert-read amplification dominates.

### Gaps
- No study quantifying expert overlap specifically for Qwen3.8-Flash-Next or Nemotron 4-bit; no MLX measurement of tree-verification expert union size.

## Q4. Theoretical max speedup on SSD-bound decode; can any exact method reach 2×?

### Takeaway
No — for this workload 2× is unlikely with any exact method. The ceiling is `acceptance_length / verifier_cost_ratio`, and both terms are unfavorable: draft acceptance length is short, while verifying K+1 tokens reads close to (K+1)× the experts on an SSD-bound MoE with low natural reuse.

### Cited Findings
- SD speedup is dominated by Tt(1)/Tt(K+1); as verification becomes free the speedup approaches the **acceptance length (AL)**, the theoretical maximum — [Cohere blog](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding); [MoESD, arXiv:2505.19645](https://arxiv.org/pdf/2505.19645). With AL=2.73 and T(4)/T(1)=1.25, implied speedup 2.18×, observed 1.95× — [Cohere blog](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding).
- For SSD-bound decode the analogous ratio is (unique experts read for verification ÷ experts read for one plain token). The repo measures ~1.8× expert touches per token at K=2; with AL≈2–3 the implied speedup is ≈1.1–1.7×, below 2× and matching the observed regressions.
- MoE-SD predicts a **non-monotonic** speedup curve: at low batch size verification adds expert weight loading on top of decode, limiting gains; the sweet spot is moderate batch size where both decode and verification load nearly all experts anyway and verification is nearly free — [Cohere blog](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding); [MoESD](https://arxiv.org/pdf/2505.19645). Veystrix runs batch 1, i.e. the unfavorable regime.
- Even EAGLE-3's own throughput study shows the gain at batch 1 (1.75×) decaying to ~1.0× by batch 56 on GPUs — [EAGLE-3 §4.3](https://arxiv.org/html/2503.01840v1).
- S2-MoE's best MoE results (up to 5.3×) come with the lossy gating and SSD-offload setup; its exact-components-only baselines (ES: expert-sparsity self-speculation) are mostly **below 1×** (e.g., Jetson 16 GB Qwen3 HE 0.94×, DeepSeek 0.74×; RTX 4090 Qwen3 1.09×) — [S2-MoE Table 5 and Table 8](https://arxiv.org/html/2608.15018v1).

### Inferences
- To get 2× exactly you need roughly `AL ≥ 2 × (U/k)` where U/k is the unique-expert amplification. With U/k≈1.8 (measured here), AL≥3.6 is needed; native MTP depths 1–8 produced low acceptance here, so the required AL is not achievable with the available drafters. Reaching AL≈6 (EAGLE-3 τ≈6.6 on dense GPU models) would require a drafter quality not shown for this 4-bit MoE.
- Practical implication for the report writer: exact speculative decoding is not a credible path to the 5 tok/s Qwen / 15 tok/s Nemotron targets on this hardware; effort is better spent reducing bytes per token (expert packing/caching, resident-expert working set) — the parameter-density term in S2-MoE's equation — or on lossy methods only if the quality constraint is relaxed.

### Gaps
- No direct measurement of acceptance length achievable by EAGLE-3/ReDrafter-style drafters on Qwen3.8-Flash-Next 4-bit; no closed-form model of SSD page-cache effects on repeated expert reads.
