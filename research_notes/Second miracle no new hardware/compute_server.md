# Compute-side optimizations and the server-vs-profile gap (16 GB M4, Qwen 5+ / Nemotron 15+ tok/s, bit-exact)

Scope: the six assigned angles — (1) the ~135 ms/token server-vs-profile gap, (2) GPU kernel fusion / persistent kernels for the ~81 ms decode compute slice, (3) single-sync slot remap (llama.cpp PR #31469), (4) top-k router exactness, (5) QSA / GatedDeltaNet decode, (6) 2025–2026 single-stream MoE systems papers. Hard constraints: no new hardware, no quality loss (bit-identical token digests). I inspected the Veystrix source read-only (no code changed); all local code facts are cited by file path. Web search was unavailable this session, so citations are to primary sources fetched directly and to local files; I did not independently re-run any experiment.

Framing arithmetic used throughout (from the owner's measured profile): Qwen short-prompt harness 300 ms/token = io_await 156.9 ms (52%) + host 18.2 ms (6%) + GPU ~81 ms (27%) + ~44 ms (15%) unaccounted; if ALL host vanished, 3.33 → 3.55 tok/s. Byte wall: 873 MB/token ÷ 3.36 GB/s ≈ 260 ms drive time. Ideal overlap ceiling is max(io, compute+host) ≈ 157–160 ms → ~6.3 tok/s for Qwen; Nemotron's I/O exceeds compute+host by ~2×, so compute work alone cannot reach 15 tok/s.

---

## Key Question 1: Where can 135 ms/token hide in the server path, and is there a bit-exact fix?

### Takeaway
I found **no code path in Veystrix that can plausibly cost 135 ms/token of per-token HTTP/tokenization/sampler/SSE/Swift-bridge overhead** — every per-token server cost I could identify is sub-millisecond. The honest conclusion is that the gap is most likely a *measurement-regime difference* (context length 14–31 K in production vs short harness, cache warmth, or whether the profile's `io_await` bucket already credits overlap), not a removable per-token tax. This contradicts the owner's hypothesis that it "may cost more than any lead," and the falsifiable test below is designed to prove me wrong quickly and cheaply.

### Cited Findings
- Architecture: the Swift UI talks HTTP to a router `ThreadingHTTPServer` (`VeystrixHTTPServer`, daemon threads, `request_queue_size=16`); the router owns a supervised **child engine process** launched as `python -m veystrix.engine_guard ... -m veystrix.cli`, with its own HTTP port and API key. There are therefore two HTTP hops (router→engine and UI→router), not one — [veystrix/server.py](/Users/vincent/Documents/Veystrix/veystrix/server.py) L1943-1959; [veystrix/model_process.py](/Users/vincent/Documents/Veystrix/veystrix/model_process.py) L91-125.
- The SSE relay is efficient per token: `_ResponseBuffer.write()` appends one frame and notifies a condition; `drain()` pops and immediately does `destination.write(frame); destination.flush()`. The 25 ms wait in `drain()` fires only when the frame queue is **empty**, so it adds no per-token latency; the keepalive thread emits one invisible `: generating` comment every 15 s under a separate lock — [veystrix/server.py](/Users/vincent/Documents/Veystrix/veystrix/server.py) L1891-1940, L2124-2157.
- `_sse()` serializes each frame with `json.dumps` and writes+flushes directly for all tokens after the first model delta; no batching delay or Nagle-style hold is introduced in app code — [veystrix/server.py](/Users/vincent/Documents/Veystrix/veystrix/server.py) L2094-2118.
- Sampler cost difference harness vs chat: greedy is a single `mx.argmax`; the production sampler at temperature 1.0 / top-p 0.95 / top-k 20 adds `argpartition`, `min`, `where`, `sort`, `take`, `softmax`, `argsort`, `cumsum`, and a scatter — roughly 8–10 tiny GPU launches on a ~150 K vocab. These are launch-bound kernels (~20–50 µs each), i.e. well under 1 ms/token combined — [veystrix/sampling.py](/Users/vincent/Documents/Veystrix/veystrix/sampling.py) L117-185.
- On an MLX decode step the *entire* generator Python between two steps (stop checks, detokenizer, logging, queue) measured **0.29 ms/token**; graph construction measured 4.65 ms and `mx.async_eval` 30.7 ms of a 35.8 ms step. This is the closest published measurement of the "server path" cost and it is ~0.3 ms, not ~135 ms — [frederic.ai profiling post](https://frederic.ai/posts/where-the-other-half-goes).
- The same profile shows context length is a real compute lever: "lightning indexer, 11 layers" costs 3.1 ms at 20 k and is **60% of the difference between a short context (30.9 ms) and 20 k (35.8 ms)**; it is bypassed below 2,048 tokens. Production 14–31 K context therefore *does* raise the GPU slice, but by single-digit milliseconds, not 135 — [frederic.ai profiling post](https://frederic.ai/posts/where-the-other-half-goes).
- Server-side prefix caching is already implemented (live tape + SSD snapshots, `VEYSTRIX_FRAGMENT_CACHE` opt-in, prefix session restore), so "prompt cache missing in production" is not an obvious 135 ms source; the project's own standard requires recording cache conditions per run — [veystrix/server.py](/Users/vincent/Documents/Veystrix/veystrix/server.py) L1091-1097, L1781; [docs/measurement-standard.md](/Users/vincent/Documents/Veystrix/docs/measurement-standard.md) L42-49.
- Known project measurement discipline: greedy comparisons require exact token digests; three arms, fresh child processes, alternating order, and a paired resolution band are mandatory — [docs/measurement-standard.md](/Users/vincent/Documents/Veystrix/docs/measurement-standard.md) L44-49. A 135 ms claim should be held to this bar.

### Inferences
- **Mechanism candidates and their honest magnitude:** HTTP/SSE relay ≈ sub-ms; tokenization (HF fast tokenizer, Rust) ≈ µs; sampler ≈ <1 ms; Swift bridge ≈ one JSON/SSE parse per token ≈ sub-ms; tool-call regex filters ≈ µs. None is a bit-exact fix worth pursuing for 135 ms.
- **Most likely explanation:** the "profile" and "production" numbers are not like-for-like. The short-prompt harness avoids the lightning indexer (<2,048 tokens) and may run with a warm page cache; the 512-token/14–31 K server path pays indexer + longer attention + colder experts. A further possibility is that the profile's 156.9 ms `io_await` already credits overlap that the production path does not achieve. Resolving this is a measurement task, not an optimization task.
- **Falsifiable test (cheap):** add hop-level monotonic timestamps to the existing flight recorder at five points — engine decode-loop enter/exit, engine SSE frame write, router `_ResponseBuffer.drain()` write, Swift receive — and compute per-token inter-arrival deltas. Predictions: (a) if all deltas are sub-ms, the gap is context/cache regime and this angle is closed; (b) if one hop shows ~135 ms, that hop is the target. Second test: run the "short-prompt harness" at 14 K, 21 K, 31 K context with identical cache state and see if 300 ms rises to ~435 ms. Bit-exactness is unaffected by instrumentation, so this is safe to run in production behind the existing opt-in `VEYSTRIX_REQUEST_LOG`/flight recorder.
- **Bit-exact fix projection:** if the gap does turn out to be sampler (the only substantive GPU-side per-token difference), replacing greedy-equivalent or sampling kernels with a fused top-k+mask+categorical kernel must reproduce the same reduction order; this is gated by digest equality. Expected value remains <1 ms.

### Gaps
- No published source measured HTTP/SSE/bridge per-token overhead on this exact codebase; the 0.29 ms generator figure is from a different model/server (GLM-5.3-Flash on M3 Ultra), so it is an anchor, not proof for Veystrix.
- I did not run the hop-level instrumentation (read-only task), so the location of the 135 ms remains unresolved by direct measurement.
- Whether the 2.3 tok/s production figure includes prefill/tool-call/reasoning overhead in its denominator is unknown; that would change the interpretation entirely.

---

## Key Question 2: GPU kernel fusion / persistent kernels / megakernels for the ~81 ms compute slice

### Takeaway
Fusion is real but bounded: the best bit-exact published MLX/Metal work reports **+5–14% from command-buffer limits (already rejected here), +1–3% from launch-end fusion, +2.5% from fp32 router weights, ~+9% total** — and PR #31469 found that after the sync wall is removed the residual is a ~5,566–6,930-node, launch/occupancy-bound graph at 7.6% SM. Applied to Veystrix's 81 ms compute slice, that is roughly 1–7 ms/token → ~2.3 → ~2.4 tok/s. Megakernels are the only direction with a plausible order-of-magnitude node-count collapse, but their headline tok/s comes from a dense model with NVFP4 and speculative decoding and is not transferable to this SSD-bound MoE.

### Cited Findings
- Bit-exact decode-step profile (M3 Ultra, GLM-5.3-Flash, 19.9 k ctx, 35.8 ms): 56% large matvecs (20.2 ms, bandwidth-bound at 500–650 GB/s), 12% small launch-bound ops (4.2 ms), 9% lightning indexer (3.1 ms), 23% remainder/gaps/sampling/Python (8.2 ms). Routed experts 9.6 ms/step = 520 GB/s, "as good as gather_qmm gets." Compiling the FFN block saved 2.5 ms/step — [frederic.ai profiling post](https://frederic.ai/posts/where-the-other-half-goes).
- Bit-exact fusion result: fusing the two launch-bound ends of the HyperConnection (5 → 3 launches: head kernel + MLX matvec + tail kernel) gave 32.69 → 32.41 ms (−1%) at 20 k and 28.25 → 27.33 ms (−3%) short, output hashes identical. A **fully fused single kernel was bit-exact but 8–11% slower** (one threadgroup streams the 1.5 MB mixing matrix from one GPU core); reproducing `rms_norm` exactly required MLX's 1024-thread layout and plain `a += b*c` ("fp contract(off)" did not match) — [frederic.ai profiling post](https://frederic.ai/posts/where-the-other-half-goes); [mlx-vlm issue #2281](https://github.com/Neywa-Labs/mlx-vlm/issues/2281).
- The exact fused HC path exists in mlx-vlm but is **skipped for single-token decode (`L == 1`)**, which is every non-speculative decode step; ~450 unfused launches/token on a 45-layer model — [mlx-vlm issue #2281](https://github.com/Neywa-Labs/mlx-vlm/issues/2281). (Veystrix uses mlx-vlm kernels for Qwen, so this is a directly relevant code-level lead.)
- Storing router and mixing weights as fp32 once instead of re-casting on every call: another ~2.5% at long context, no kernel at all, identical outputs — [frederic.ai profiling post](https://frederic.ai/posts/where-the-other-half-goes).
- Command-buffer limits (`MLX_MAX_OPS_PER_BUFFER=1000`, `MLX_MAX_MB_PER_BUFFER=400`): +5% at 20 k / +7.7% short (35.68 → 33.94 / 30.91 → 28.71 ms), bit-identical hashes; M5 Max MoE alone with `MLX_MAX_MB_PER_BUFFER=2048`: 100.5 → 112.5 (+11.9%) and 99.6 → 113.3 (+13.7%), logits/tokens matched; custom fused expert kernels gave +13% at defaults but only +1–2% on top — [MLX issue #4521](https://github.com/ml-explore/mlx/issues/4521). **Already rejected in Veystrix as a regression, so this headroom is unavailable.**
- Node census after sync fix (RTX 3060, ubatch=1): MUL_MAT 1487, SCALE 780, UNARY 603, ADD 286, GET_ROWS 232, hyper-connection ops ~365, SET_ROWS 112, GLU 90, ROPE 90, FLASH_ATTN_EXT 22, LIGHTNING_INDEXER 22, TOP_K 22 — total ~5,566 nodes; GPU 7.6% SM / 32 W. Raising batch M=1→8 only trims ~25% and plateaus; "the wall is the node count, not the batch width" — [llama.cpp PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469).
- lithos-metal (Oct 2026) generates Metal megakernels: one **layer-wise mixer** (GDN or full-attention) becomes a single dispatch with persistent worker threadgroups claiming tasks from a shared queue; MLP stays separate; pre-encoded indirect command buffers for CPU-light replay. Qwen3.8-27B-NVFP4 on M5 Max: >200 tok/s/user claimed, ~1.8–2× MLX/Ollama at short context and ~127 projected tok/s at 32 k; in-kernel global barriers were "no cheaper than dispatch boundaries" — [lithos-metal blog](https://lithosai.com/blog/lithos-metal).
- Caveat on transfer: lithos-metal's numbers use DSpark speculative decoding and NVFP4, not bit-exact single-stream Qwen3.8-Flash-Next decode; no source provides a bit-exact node-count→tok/s curve for MLX MoE on M4 — [lithos-metal blog](https://lithosai.com/blog/lithos-metal).

### Inferences
- **Mechanism:** (a) enable/replicate the existing exact fused HC/norm path at `L==1` in mlx-vlm (1–3%); (b) fuse router-tail + read-range emission (decode only); (c) hoist per-call fp32 re-casts of router/mixing weights (~2.5%); (d) convert the indexer to one kernel (see Q4); (e) longer-term, a layer-wise mixer megakernel to collapse node count.
- **Bit-exactness:** (a)–(d) have source-level evidence of identical hashes/logits; (e) is unverified and lower confidence — Megakernel reduction order must be matched element-for-element and digest-gated. Prefill fusion is the risky part (near-tie flips); keep all fusion decode-only.
- **Magnitude:** at the owner's 81 ms compute slice, ~9% ≈ 7 ms → 2.3 → ~2.4 tok/s Qwen; 7 ms hidden behind 156.9 ms I/O does not change the byte-wall ceiling. Even *eliminating* the entire GPU slice only reaches max(156.9, 18.2) ≈ 157 ms (~6.3 tok/s) and would still miss Nemotron 15+. Compute is therefore a component of the answer, not the second miracle.
- **Falsifiable test:** A/B one fused HC path at `L==1` (or the fp32-weight hoist) with the project's digest harness over ≥200 greedy tokens at fixed 14–31 k context; require identical digests and measure ms/token. Gate: reject on any digest drift or any regression (the fully-fused-kernel result shows that is a live risk).

### Gaps
- No measurement of fusion gains on Veystrix's exact MLX 0.32.2 + mlx-vlm Qwen kernels on a 16 GB M4.
- No bit-exact megakernel implementation for MoE decode on Apple Silicon; lithos-metal is dense + speculative.

---

## Key Question 3: Does single-sync slot remap (PR #31469 KICK/WAIT) apply when host is only 6%?

### Takeaway
**No — not as a headroom lever.** PR #31469's pathology is ~8 ms/layer × 48 layers ≈ 340 ms/token of host sync. Veystrix's profile has only 18.2 ms of host work in total, so that pathology cannot be present; if it were, host would be ~95%, not 6%. The transferable insight is diagnostic (count host syncs/step) and the PR's own conclusion that after the sync fix the wall is node count (Q2), not a path to 15+ tok/s.

### Cited Findings
- PR #31469's `MOE_NOSPLIT` diagnostic: per-layer scheduler split forces "a device sync + top-k readback per MoE layer per decode step"; locked-clock cost **2.32 tok/s with splits vs 10.89 without = ~340 ms/token of host sync at 48 layers (~8 ms/layer)**; the stated correct fix is "refresh all 48 layer tables in ONE sync per decode step" — [llama.cpp PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469).
- The in-stream `KICK`/`WAIT` doorbell resolves `topk_indices → slot_ids` on-device, runs shared-expert FFN between KICK and WAIT, and cuts 48 splits to 1 CUDA-graph split; slot-remap parity tests measured **0.000e+00 max abs diff**, 28,080/28,080 byte-identical router records, 576/576 identical greedy tokens; headline out-of-core decode 1.47/1.63 → 16.29/22.03 tok/s, 100% byte-exact — [llama.cpp PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469).
- Same source's honesty: the 20 tok/s figure was recorded with a boosted GPU and is not reproducible cold; `MOE_DIAG` found the ~900 ms device step unchanged with expert I/O on vs off, GPU at 7.6% SM — i.e. after the sync was fixed the wall became node count/launch — [llama.cpp PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469).
- Owner's measured Veystrix profile: all host work 18.2 ms (6% of 300 ms); removing ALL host gives only 3.33 → 3.55 tok/s — (owner-supplied profile in task context).
- MLX's lazy evaluator already batches a whole subgraph into few command buffers, a CUDA-Graph-like payoff without a capture lifecycle; true capture/replay is hard because MLX array addresses change — [MLX issue #2358](https://github.com/ml-explore/mlx/issues/2358) (as summarized in prior local note [overlap_fusion.md](/Users/vincent/Documents/Veystrix/research_notes/MoE%20decode%20IO%20bypass%20research/overlap_fusion.md)).

### Inferences
- **Arithmetic disqualifies it:** a 340 ms sync pathology inside an 18.2 ms host budget is impossible. Either Veystrix already does ~1 sync/step, or the 156.9 ms `io_await` bucket contains sync stalls misattributed to I/O. Both cases reduce this to a measurement question, not a new optimization.
- **Bit-exactness:** slot remap is a permutation/gather of identical weights (0.000e+00 measured), so a genuine fix would satisfy the gate — but there is nothing to fix if syncs are already minimal.
- **Feasibility:** low value; MLX has no Python-exposed on-device doorbell; a native Metal kernel + `mx.depends` boundary control would be needed for emulation.
- **Falsifiable test:** instrument sync count per decode step and per-layer top-k readbacks. Prediction: ~1/step already. If ~48/step is found, reclassify ~340 ms of `io_await` as host and re-run the profile; only then does KICK/WAIT become the top lead.

### Gaps
- No source states how many host syncs Veystrix's Qwen decode actually performs; that single number decides this angle.
- No Apple-Silicon/MLX port of the KICK/WAIT doorbell exists in the sources found.

---

## Key Question 4: Can the top-10/512 router be made cheaper bit-exactly, and is it on the critical path?

### Takeaway
The router's top-k is **not on the critical path**: the decode graph has only 22 `TOP_K` and 22 `LIGHTNING_INDEXER` nodes versus 1,487 `MUL_MAT` and ~1,500+ elementals, and the "router tail" is estimated at ~1–3% of a step. The one measurable, bit-exact router-adjacent win is fusing the **lightning indexer** (the sparse-attention selector, not the MoE router) into one Metal kernel: 2.5–2.7 ms/token at 20 k, growing with context — relevant precisely because production is 14–31 K.

### Cited Findings
- Decode node census: `LIGHTNING_INDEXER 22`, `TOP_K 22`, `FLASH_ATTN_EXT 22` vs `MUL_MAT 1487`, `SCALE 780`, `UNARY 603`, `ADD 286` — [llama.cpp PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469).
- Lightning indexer measured 3.1 ms/step at 20 k (0.28 ms × 11 layers, ~40 launches + a sort), bypassed below 2,048 tokens, and accounts for 60% of the short-vs-20 k step difference (30.9 → 35.8 ms) — [frederic.ai profiling post](https://frederic.ai/posts/where-the-other-half-goes).
- "The indexer as one Metal kernel, scoring pools and selecting the top 512 in one pass instead of forty launches, is worth 2.5 to 2.7 ms per step at 20k and grows with context. … The router tail is the same size or smaller" — [frederic.ai profiling post](https://frederic.ai/posts/where-the-other-half-goes).
- "Storing the router and mixing weights as fp32 once instead of re-casting them on every call (another 2.5% at the long context, no kernel at all)" — [frederic.ai profiling post](https://frederic.ai/posts/where-the-other-half-goes).
- Bit-exact caution from the same source: fusing launch-bound ends required reproducing MLX's exact reduction layout; different threadgroup sizes gave different rounding — [mlx-vlm issue #2281](https://github.com/Neywa-Labs/mlx-vlm/issues/2281).

### Inferences
- **Mechanism:** single-pass indexer (score pools + top-512 select in one kernel) and fp32 router-weight hoist; both are decode-only and preserve the selected expert set and arithmetic order when implemented as the same reduction.
- **Bit-exactness:** plausible but must be gated — a different top-k/selection reduction order can flip near-tie logits (the same failure mode flagged for prefill GEMM fusion). Validator: token digest + router-record equality (the PR #31469 parity test is the template).
- **Magnitude:** ~2.5–2.7 ms at 20 k, likely 3–5 ms at 31 K — small but one of the few compute levers that *grows with the production context*; ~2–4% of a 300 ms step, so ~2.3 → ~2.35 tok/s. It does not change the ranking of the byte wall.
- **Feasibility:** medium-high for the fp32 hoist (no kernel at all); medium for the indexer kernel (needs native Metal).
- **Falsifiable test:** digest-gated A/B at fixed 21 K and 31 K contexts; accept only if digests are identical and ms/token drops by ≥2 ms.

### Gaps
- No source quantified the MoE top-10/512 router kernel cost alone on Apple Silicon; the 1–3% "router tail" is an author's estimate, not a profiler row.
- Whether Veystrix's Qwen checkpoint uses the same lightning indexer layout (T/4 pools, top-512) is unverified locally.

---

## Key Question 5: QSA attention / GatedDeltaNet decode at 14–31 K — bit-exact wins?

### Takeaway
Gated DeltaNet's recurrent state is **context-length independent**, so 14–31 K does not inflate its decode compute; the long-context GPU cost in this architecture is the sparse-attention indexer (Q4), not the recurrence. The transferable bit-exact idea is keeping recurrent-state column slices in registers across tokens and fusing the GDN mixer's launch-bound ends, but published gains are single-digit ms/token and the headline megakernel numbers come from a dense model with speculative decoding.

### Cited Findings
- Gated DeltaNet is a linear-attention architecture whose RNN-style state is independent of context length; hybrids with sliding-window attention improve long-context behavior — [Gated Delta Networks, ICLR 2025](https://arxiv.org/abs/2412.06464).
- The GDN recurrence: `S̄_t = α_t S_{t−1}; δ_t = β_t(v_t − S̄_tᵀk_t); S_t = S̄_t + k_t δ_tᵀ; o_t = S_tᵀq_t`, with FP32 key-by-value state. Different value columns are independent, so a persistent worker can retain a slice of state columns in registers across a block of tokens, "avoid[ing] reading and writing the full recurrent state at every token step" — [lithos-metal blog](https://lithosai.com/blog/lithos-metal).
- lithos-metal fuses one GDN mixer into a single megakernel (QKV, A/B, Z projections; causal conv; norm; recurrence; gated norm; output projection), 80 logical workers × 8 SIMD groups × 8-column state slices; MLP remains a separate kernel. Layer-wise fusion was chosen because in-kernel global barriers were "no cheaper than dispatch boundaries" — [lithos-metal blog](https://lithosai.com/blog/lithos-metal).
- On MLX, the analogous "fused linear-attention kernel and its two side projections" is only 4.2 ms (12%) of a 35.8 ms step; hyper-connections add ~2 ms of pure launch cost — [frederic.ai profiling post](https://frederic.ai/posts/where-the-other-half-goes).
- Qwen3.8-Flash-Next's decode graph contains both `DSV4_HC_PRE/POST` (GDN/hyper-connection) ops and QSA attention layers; PR #31469 keeps dense, recurrent, attention, and LM-head layers resident while streaming only experts — [llama.cpp PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469).

### Inferences
- **Mechanism:** (a) register-resident state slices to avoid full-state read/write each token; (b) fuse GDN mixer launch-bound ends around MLX's own matvec (the same head+matvec+tail pattern that was bit-exact and 1–3% faster); (c) indexer-as-one-kernel (Q4) for the long-context portion.
- **Bit-exactness:** the GDN state math is fixed FP32; register residency is a placement change (bit-exact by construction). Fusion of norm/gating ends requires reproducing MLX's reduction layout and must be digest-gated. (a)+(b) are safer than full megakernelization.
- **Magnitude:** published fusion numbers are 1–3% and 2.5%; a 4.2 ms small-op bucket is the absolute ceiling for "fusing everything small" in that profile. On Veystrix that is ≤~4 ms/token — useful but not a second miracle.
- **Feasibility:** low-medium on MLX 0.32.2 Python with mlx-vlm kernels; native Metal required for register-resident state.
- **Falsifiable test:** implement (b) on one GDN layer behind a flag; digest-gate; measure at 14 K vs 31 K. Predict ≤3% gain and zero digest drift; reject otherwise.

### Gaps
- No bit-exact measurement of GDN/QSA decode fusion on the exact Qwen3.8-Flash-Next checkpoint or on a 16 GB M4.
- No source separates QSA attention cost from MoE cost in Veystrix's 81 ms GPU slice; indexer numbers are from GLM-5.3-Flash.

---

## Key Question 6: 2025–2026 systems work on single-stream MoE decode that cuts non-I/O time without changing output

### Takeaway
The literature's large numbers (1.3–10×) almost always come from I/O path rewrites, batch throughput, or precision changes; for **single-stream, bit-exact, non-I/O time** the credible gains are ~5–14% (command-buffer limits, unavailable here) and ~1–3% (launch-end fusion), with megakernels as an unverified frontier. No source found demonstrates a bit-exact 2× cut of the non-I/O slice for this model class on a 16 GB M4.

### Cited Findings
- PR #31469: 1 CUDA-graph split + on-device KICK/WAIT + O_DIRECT pager → 1.47 → 16.29/22.03 tok/s with 100% byte-exact output on Qwen3.8-Flash-Next (RTX 3060 12 GB / 16 GB RAM); but cold-reproducible decode is ~2.2 tok/s and the post-fix wall is a 6,930-node graph at 7.6% SM — [llama.cpp PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469).
- MoE-Lightning (ASPLOS 2025, arXiv 2411.11217): CGOPipe fine-grained per-layer GPU–CPU–I/O pipeline with paged weights, T(decode) = max(comm, T_cpu, T_gpu), up to 10.3× *batch* throughput vs FlexGen/DeepSpeed on Mixtral 8×7B/T4 16 GB; explicitly "We do not consider disk offloading in this work" — [arXiv 2411.11217](https://arxiv.org/html/2411.11217) (as cited in prior local note [overlap_fusion.md](/Users/vincent/Documents/Veystrix/research_notes/MoE%20decode%20IO%20bypass%20research/overlap_fusion.md)).
- Fiddler (ICLR 2025): moves activations to CPU instead of weights to GPU, unquantized Mixtral-8×7B on 24 GB GPU at >3 tok/s, 1.26× single-batch vs SOTA; bit-exact because it only changes execution placement — [Fiddler](https://arxiv.org/abs/2402.07033) (as cited in prior local note [moe_expert_io.md](/Users/vincent/Documents/Veystrix/research_notes/Faster%20Qwen%20Nemotron%20inference%20on%20M4/moe_expert_io.md)).
- lithos-metal: layer-wise Metal megakernels + pre-encoded command replay, ~1.8–2× MLX/Ollama at short context on M5 Max; headline >200 tok/s uses NVFP4 + DSpark speculative decoding, so it is not a bit-exact single-stream result — [lithos-metal blog](https://lithosai.com/blog/lithos-metal).
- MLX command-buffer issue #4521: the only "free" bit-exact gain (+5–14%, hashes matched), already rejected in Veystrix — [MLX issue #4521](https://github.com/ml-explore/mlx/issues/4521).
- Prior Veystrix synthesis reached the same ceiling: "Nemotron 15+ is not reachable by overlap alone because its 123 ms I/O exceeds its ~58 ms compute+host by ~2×, so I/O *volume* must fall"; "Qwen 5+ tok/s is plausible if near-perfect overlap is achieved" — [overlap_fusion.md](/Users/vincent/Documents/Veystrix/research_notes/MoE%20decode%20IO%20bypass%20research/overlap_fusion.md).

### Inferences
- The honest ranking for this angle: (1) bit-exact launch-end fusion + fp32 router hoist (~1–3% + 2.5%, M3 Ultra evidence); (2) indexer-as-one-kernel at long context (2.5–2.7 ms at 20 k, grows); (3) layer-wise mixer megakernel (frontier, high effort, transferability unverified); (4) single-sync slot remap (N/A here, Q3). None of these is a "second miracle" that delivers 2× on the compute slice; the byte wall (873 MB/token ÷ 3.36 GB/s ≈ 260 ms) remains the binding constraint.
- Every credible candidate is gated by the same test: bit-identical greedy digests at fixed 14–31 K context, measured with the project's paired multi-arm standard.
- Rejected/unsafe: precision changes, expert substitution, reduced top-k, cross-engine switches, and speculative decoding (different reduction order; already slower for SSD-bound MoE in local evidence).

### Gaps
- No 2025–2026 paper found that isolates and cuts *single-stream* MoE decode compute by ≥2× while preserving exact output on Apple Silicon.
- Full texts of MoE-Lightning and several cited papers were not re-fetched this session; their numbers are quoted from prior local notes that cited them.

---

## Overall ranking and reachability verdict

### Takeaway
Ranked by plausible bit-exact value under the hard constraints: **#1 close the 135 ms measurement question (cheap, may invalidate the whole premise); #2 indexer-as-one-kernel (2.5–2.7 ms at 20 k, grows with production context); #3 bit-exact launch-end fusion + fp32 router-weight hoist (~1–3% + 2.5%); #4 single-sync slot remap (disqualified by the 6% host budget); #5 megakernels (frontier, multi-month, non-transferable headline numbers).** Compute cannot be the second miracle: even the ideal case of zero compute and zero host leaves 156.9 ms of I/O → ~6.3 tok/s Qwen, and Nemotron 15+ requires I/O *volume* reduction or more memory, not compute.

### Cited Findings
- Owner's profile and byte-wall arithmetic (task context); Qwen byte wall 873 MB/token ÷ 3.36 GB/s ≈ 260 ms — (user-supplied measured facts).
- Qwen ideal-overlap ceiling ~6.3 tok/s; Nemotron overlap alone caps ~6–7 tok/s — [overlap_fusion.md](/Users/vincent/Documents/Veystrix/research_notes/MoE%20decode%20IO%20bypass%20research/overlap_fusion.md).
- Best bit-exact compute gains published: +5–14% command buffers (unavailable), +1–3% fusion, +2.5% fp32 weights — [MLX issue #4521](https://github.com/ml-explore/mlx/issues/4521); [frederic.ai](https://frederic.ai/posts/where-the-other-half-goes).

### Inferences
- Recommended next action is a measurement, not an optimization: hop-level timestamp instrumentation + equal-context harness runs. It is bit-exact, cheap, and decides whether a real 135 ms target exists.
- If a target is found, it is most likely in the context-dependent indexer/attention path (Q4/Q5), not in HTTP/sampler/bridge (Q1).
- Do not spend effort on PR #31469's sync fix (host is 6%) or on megakernels as a near-term path; keep them as a post-mortem option if profiling ever shows a >5,000-node launch-bound graph inside the 81 ms slice.

### Gaps
- All proposed gains except the measurement are extrapolated from other models/hardware; none has been measured on Veystrix's exact 16 GB M4 + MLX 0.32.2 + mlx-vlm path.
- Nemotron 15+ remains unreachable under the two hard constraints on the evidence found; this should be reported as a capacity/quality tradeoff, not an engineering miss.
