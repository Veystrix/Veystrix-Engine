# Bit-Exact Compute/I-O Overlap, Kernel Fusion, and Latency-Hiding for MoE Decode on Veystrix (M4 / MLX)

Scope: find overlap/fusion levers the already-rejected experiments missed. Rejected ground truth (do NOT re-propose): decode overlap + early-submit (−8.2%), early shared expert (−10%), perfect predictor + overlap (tied), layer-prefetch, K>1 verify (slower), async_eval (no gain). Production already overlaps shared-expert compute while reads are in flight in some paths. Target: hide most of the 130–157 ms SSD `io_await` in Qwen (300 ms/token, I/O 156.9 ms = 52%, host 18 ms = 6%, GPU ~27%) and Nemotron (181 ms, I/O 123 ms = 68%, host 27.8 ms) with ZERO quality loss (bit-identical digests).

Important framing distinction used throughout: techniques that genuinely *hide* SSD wait must overlap compute with in-flight reads or start expert compute per-expert as bytes land. Techniques that only shrink the host/GPU-launch slice (e.g. MLX command-buffer tuning) reduce tokens/sec but hide ~0 ms of `io_await` directly. Both are reported, but ranked accordingly.

---

## Key Question 1: Can compute start on each expert the instant its bytes land — and does removing per-layer host synchronization beat the rejected "early-submit"?

### Takeaway
The strongest *novel* mechanism in this whole hunt is not prefetching at all — it is restructuring the dependency/sync graph so that (a) slot resolution and top-k-to-slot remapping happen **on-device** instead of via a host sync per MoE layer, and (b) expert GEMMs are dispatched on **fine-grained per-expert completion** rather than waiting for all reads in a layer. llama.cpp PR #31469's `KICK`/`WAIT` doorbell and its `MOE_NOSPLIT` diagnostic quantify the cost of the naive per-layer sync at **~8 ms/layer × 48 layers ≈ 340 ms/token** (2.32 vs 10.89 tok/s with the split removed), and the PR's own conclusion is that the fix is to "refresh all 48 layer tables in ONE sync per decode step." That single-sync, on-device-resolution design is distinct from the rejected early-submit and is the #1 candidate.

### Cited Findings
- PR #31469 (Qwen3.8-Flash-Next, 48 layers, 512 experts, top-10, RTX 3060 12 GB / 16 GB RAM, Linux NVMe): an *in-stream* GPU doorbell resolves `topk_indices -> slot_ids` on-device via `GGML_OP_MOE_SLOT_RESOLVE_KICK` and `..._WAIT`; `KICK` is scheduled immediately after router top-k and the shared-expert FFN runs between KICK and WAIT, overlapping shared-expert GPU compute with host miss handling + async H2D DMA. This replaced 48 per-layer scheduler splits with **1 CUDA-graph split**; headline out-of-core decode 1.47/1.63 → **16.29/22.03 tok/s, 100% byte-exact** (0.000e+00 max abs diff in slot-remap tests, 28,080/28,080 byte-identical router records, 576/576 identical greedy tokens). — [llama.cpp PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469)
- Same PR, `MOE_NOSPLIT` diagnostic: the per-layer split "forces a device sync + top-k readback per MoE layer per decode step"; locked-clock cost **2.32 tok/s with splits vs 10.89 tok/s without = ~340 ms/token of host sync at 48 layers (~8 ms/layer)**. The PR explicitly states "the correct fix is to refresh all 48 layer tables in ONE sync per decode step rather than one split per layer." — [llama.cpp PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469)
- Same PR, "arrival-slack slot budgeting": more VRAM slots allocated to early layers 0–4 **where compute-overlap slack is zero** — i.e. the design explicitly accounts for the fact that early layers have no prior compute to hide their reads behind, a fine-grained per-layer overlap analysis. — [llama.cpp PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469)
- Same PR, a `MOE_STATIC_HOT` zero-sentinel slot makes cold misses contribute zero weights so routing stays coherent with **zero paging callbacks (all 48 splits retired)**; and a one-contiguous-blob-per-expert pack (`gate|up|down`) was "verified byte-identical to the GGUF ... IO change only." — [llama.cpp PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469)
- Caution from the SAME source (important for honesty): later commits report the 20 tok/s figure was recorded with a boosted GPU and is **not reproducible cold** (decode ~2.2 tok/s), and a `MOE_DIAG` breakdown found the ~900 ms device step unchanged with expert I/O on vs off, with the GPU at **7.6% SM / ~6930-node graph** — i.e. after the sync was fixed the wall became **kernel/node count and launch**, not I/O. — [llama.cpp PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469)
- Related: MLX's lazy evaluator already batches a whole subgraph into few command buffers (a CUDA-Graph-like payoff), but true capture/replay is hard because MLX array addresses almost always change; llama.cpp's preallocated arena is the contrast. — [MLX issue #2358](https://github.com/ml-explore/mlx/issues/2358)

### Inferences
- **Mechanism (candidate #1):** keep top-k and slot-table in device memory; at decode step start, push all 48 layers' slot tables in ONE host write; dispatch each expert's GEMM against its slot as soon as that slot's bytes are committed. This is "compute-on-arrival" at expert granularity. It attacks the serialized "wait for all 10 experts, then GEMM" barrier without re-introducing the rejected early-submit of *reads*.
- **Bit-exactness:** fully compatible with the gate. Slot remap is a permutation/gather of identical weights (PR measured 0.000e+00 abs diff); no math, precision, top-k, or routing change. Validation gate = token digest + router-record equality.
- **I/O hidden (budget arithmetic, my inference — NOT a source number):** if the current 156.9 ms I/O is serialized before ~99 ms of compute+host, perfect overlap of compute behind reads caps total at max(156.9, 99)+other ≈ 200 ms → ~5 tok/s for Qwen. So this lever alone is plausibly enough for Qwen 5+ but **not** for Nemotron 15+ (123 ms I/O vs only ~58 ms compute+host; overlap alone caps ~6–7 tok/s). Source-anchored overlap efficiency for a fine-grained pipeline: MoE-Lightning's CGOPipe reaches the throughput upper bound with 2–3× less CPU memory (see KQ2), and llama.cpp double-buffering reports ~2.09× pipeline throughput — i.e. roughly 30–50% of exposed I/O can realistically be hidden, not 100%.
- **M4/MLX feasibility:** medium-high. MLX has no on-device "doorbell" primitive exposed to Python, but `mx.array` slot tables + a custom Metal kernel reading slot ids, and `mx.eval`/`mx.depends` boundary control, can emulate it. Production already overlaps shared-expert compute, so the *incremental* win is the per-layer-sync removal and per-expert dispatch, not the shared-expert overlap itself.
- **Falsifiable test:** profile per-layer host sync count/token (should drop from ~48 to 1); then A/B bit-exact greedy digest over ≥200 tokens while measuring `io_await` and total ms/token. Prediction: large ms/token drop with unchanged digest; if digest changes, slot remap is wrong. Distinguish from rejected early-submit by holding read-issue time constant and varying only sync/dispatch granularity.

### Gaps
- No source measured this exact on-device-slot-resolution trick on Apple Silicon / MLX; PR #31469 is Linux/CUDA and its headline numbers are pushed back by its own later commits.
- Whether Veystrix currently issues one host sync per MoE layer (the ~8 ms/layer pathology) is unknown; if it already does one sync/step, the ~340 ms headroom does not exist here.
- Whether Metal command buffers with shared events can implement a KICK/WAIT-style doorbell without a host round-trip was not confirmed by a source.

---

## Key Question 2: Multi-stream / double-buffered overlap — read layer l+1 while computing layer l

### Takeaway
Classic fine-grained GPU–CPU–I/O pipelining is well proven (MoE-Lightning up to 10.3× batch throughput; ~2.09× pipeline throughput for double-buffered next-layer reads) and colibri on M4 Max explicitly submits resident-expert GPU work *before* missed-expert disk reads so I/O and compute overlap (~1.4× on M4 Max). BUT this is the closest match to the already-rejected "layer-prefetch" and "decode overlap + early-submit" — so the only defensible novel variant is a genuine **multi-command-buffer / multi-stream** implementation where the read of layer l+1 runs on a separate Metal queue/command buffer and does not perturb l's critical path, with per-expert completion signaling. Rank it behind the single-sync redesign.

### Cited Findings
- MoE-Lightning (ASPLOS 2025, arxiv 2411.11217) introduces **CGOPipe**, a fine-grained per-layer GPU–CPU–I/O pipeline with **paged weights**: chunk each layer's weight transfer into `n` pages (n = number of micro-batches) and interleave intermediate-result transfers with paged weight transfers to avoid I/O bubbles; T(decode) = max(comm, T_cpu, T_gpu). Up to **10.3× throughput** vs FlexGen/DeepSpeed for Mixtral 8x7B on a single T4 16 GB; reaches the GPU-memory throughput bound with 2–3× less CPU memory. NOTE: explicitly "We do not consider disk offloading in this work" and it is *batch* throughput, not single-stream M4 decode. — [arxiv 2411.11217](https://arxiv.org/html/2411.11217)
- Double-buffered overlap of next-layer I/O under current-layer compute reported at **~2.09× pipeline throughput** for explicit-read MoE streaming (as summarized in a prior Veystrix research note citing llama.cpp discussion #27149; not independently re-fetched this session). — [llama.cpp discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149)
- colibri Metal backend (M4 Max 128 GB, warm cache): "Metal's ~5 ms submit latency makes per-matmul dispatch a loss — everything is batched into few command buffers per layer, and the **resident experts' GPU work is submitted before the missed experts' disk reads so I/O and compute overlap**"; routed-expert SwiGLU is batched and zero-copy from RAM slabs. CPU 0.30 → Metal **0.42 tok/s (~1.4×)**. Decode token-exact vs CPU; prefill GEMM can flip near-tie logits (different accumulation order). — [colibri docs/metal.md](https://github.com/JustVugg/colibri/blob/v2.1.0/docs/metal.md)
- OD-MoE: cacheless-SSD, CPU–GPU **parallel shadow model**; Mixtral-8x7B **3.69 tok/s (75.5% of full-GPU), full-precision answer quality**. — [arxiv 2512.03927](https://arxiv.org/html/2512.03927)
- streamlx (M4 Air, MLX): span-merged multi-MB reads; resident experts start on GPU via `async_eval` while misses fetched; next-layer routing predicted from residual (~70% accurate). — [srcterm/streamlx](https://github.com/srcterm/streamlx). NOTE: `async_eval` is on Veystrix's REJECTED list ("no gain"), so its specific mechanism should not be re-proposed.
- 1bit.gg expert streaming (Qwen3.8-Flash-Next-class): expert cache 4608/6144 slots (75%) reached 28–38 tok/s, **37–39 after "moving prefetch off decode path"** — i.e. getting the prefetch/issue work off the critical decode path mattered more than the overlap itself. — [1bit.gg](https://1bit.gg/moe-streaming.html)

### Inferences
- **Mechanism (candidate #2):** two command buffers — CB_A computes layer l on GPU; CB_B (a separate Metal queue or background I/O worker) issues reads of layer l+1's exact routed experts; a shared MTLSharedEvent gates l+1's GEMM on per-expert read completion, not on a full-layer join. Some early layers have zero slack (per PR #31469), so overlap benefit is back-loaded.
- **Bit-exactness:** yes if reads fetch identical bytes and GEMM math is unchanged; colibri shows token-exact decode with this pattern; prefill-GEMM accumulation-order flips are a warning to keep fusion decode-only.
- **I/O hidden:** source anchors ~2.09× pipeline throughput (≈50% of exposed I/O hidden) and colibri ~1.4× end-to-end. Applied to the 156.9 ms budget, a plausible 40–55 ms hidden → ~4 tok/s Qwen; not sufficient alone for 15+ Nemotron.
- **M4/MLX feasibility:** medium. MLX exposes an async eval path and command-buffer control, but there is no documented public per-buffer stream-affinity/event API at the Python layer; likely needs a native Metal/C++ path. The `async_eval` rejection is a warning that naive overlap did not help here.
- **Falsifiable test:** instrument a 2-deep pipeline and measure GPU idle gaps and `io_await` overlap fraction; A/B digest. Prediction: reduced exposed I/O and unchanged digest; if no change, the rejected layer-prefetch/async_eval path is being re-created and this should be dropped.

### Gaps
- This is partially redundant with rejected "layer-prefetch" / "decode overlap + early-submit"; no source proves a multi-stream Metal implementation beats those specific rejected attempts on M4.
- No Apple-Silicon source reports a true multi-queue read/compute overlap with per-expert events.

---

## Key Question 3: MLX command-buffer scheduling and CUDA-graph-equivalent capture on Metal — what is measurable and is it bit-exact?

### Takeaway
The cheapest verified win in this entire domain is raising MLX's Metal command-buffer limits: `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER` give +5–14% decode on launch-heavy MoE with **bit-identical output hashes and unchanged prefill/memory**. It does NOT hide SSD wait (it removes GPU/CPU launch overhead), so it is a "free points" config change rather than the core I/O solution. True CUDA-graph-equivalent capture/replay in MLX is hard (array addresses change; `start_capture` is a profiling tool), so treat it as a lower-value, higher-risk direction.

### Cited Findings
- MLX issue #4521: `Device` commits a Metal command buffer after 50 ops or when input buffers exceed 50 "MB"; for tens-of-MB weight matrices the byte rule trips ~once/layer → 45–90 command buffers per decode step, each ~20 µs GPU + a CPU completion handler. M3 Ultra, GLM-5.3-Flash (~1600 kernels/token): defaults 35.68 → **33.94 ms/token at 20k ctx (+5%)**, 30.91 → **28.71 ms short (+7.7%)** with `MLX_MAX_OPS_PER_BUFFER=1000, MLX_MAX_MB_PER_BUFFER=400`; **outputs bit-identical, prefill/memory unchanged**; dense Qwen3.8-27B unchanged. — [MLX issue #4521](https://github.com/ml-explore/mlx/issues/4521)
- M5 Max data point in same issue: only `MLX_MAX_MB_PER_BUFFER=2048` raised Qwen3.6-35B-A3B (MoE) 100.5 → **112.5 (+11.9%) / 99.6 → 113.3 (+13.7%)**; dense +0–2.6%; generated tokens and sampled logits matched. Custom fused expert kernels gave +13% at defaults but only +1–2% on top of the buffer setting (same overhead removed). Cost: +0.57 GB MoE prefill peak. — [MLX issue #4521](https://github.com/ml-explore/mlx/issues/4521)
- vllm-metal now sets these variables by default for its decode path; model cards recommend `MLX_MAX_MB_PER_BUFFER=2048 MLX_MAX_OPS_PER_BUFFER=100000` for ≈+12% decode on M3 Ultra. — [vllm-metal configuration](https://docs.vllm.ai/projects/vllm-metal/en/latest/configuration/)
- CUDA-graph-equivalent in MLX: lazy evaluation batches a subgraph into few command buffers (similar payoff, no capture lifecycle); true capture/replay is hard because MLX array addresses almost always change, unlike llama.cpp's preallocated arena. `mlx.metal.start_capture`/`stop_capture` and `MTL_CAPTURE_ENABLED` are Xcode frame-capture/profiling (50–100× slower under capture), not replay. One toy MetalGraph replay measured 3.09 ms with matching checksums. — [MLX issue #2358](https://github.com/ml-explore/mlx/issues/2358); [Hashnode MetalGraph note](https://selfenrichment.hashnode.dev/revisiting-metalgraph-what-i-learned-about-mlx-replay-and-sidecar-performance); [vllm-metal profiling docs](https://docs.vllm.ai/projects/vllm-metal/en/latest/profiling/)
- Decode profile behind the #4521 numbers (frederic.ai, per prior note): 56% bandwidth-bound matvecs, 12% launch-bound small ops, 9% indexer, 23% gaps/sampling/Python; routed experts ~520 GB/s; compiling the FFN block saved 2.5 ms/step. — [frederic.ai](https://frederic.ai/posts/where-the-other-half-goes)

### Inferences
- **Mechanism (candidate #3):** set the two env vars; optionally gate them to decode only (prefill peak memory cost). This is a config lever, not a code/quant change.
- **Bit-exactness:** explicitly verified (output hashes / sampled logits matched across settings). Safe under the gate.
- **I/O hidden:** ~0 ms of the 156.9 ms `io_await` — it compresses the launch/host slice. Value is +5–14% on the ~99 ms compute+host component, i.e. a few ms/token; worth taking because it is free and composable.
- **M4/MLX feasibility:** high for env vars (immediate); low for true graph replay.
- **Falsifiable test:** run the standard bit-exact digest harness with and without the env vars at fixed prompt/decode length; measure ms/token and peak memory; keep only if digest identical and prefill peak acceptable.

### Gaps
- No measurement of these env vars specifically on a 16 GB M4 with Qwen3.8-Flash-Next / Nemotron; the +5–14% is from M3/M5 Ultra/Max, which have different command-buffer defaults and much more memory.
- Decode-only command-buffer limits are suggested by maintainers but not benchmarked.

---

## Key Question 4: Kernel fusion — fusing router + read-issue + expert GEMM, persistent kernels, and megakernels

### Takeaway
Fusion is the right structural answer to PR #31469's discovery that after sync is fixed the wall is a ~6900-node, launch/occupancy-bound graph. On MLX, custom fused expert kernels and compiling the FFN block gave ~13% and 2.5 ms/step respectively; native-Metal megakernels (lithos-metal) and chip-specific fused kernels (BaseRT) are the frontier. Caveat: fusion can change reduction order and flip near-tie logits in *prefill* (colibri #622) — keep fusion decode-only and digest-gate it.

### Cited Findings
- vLLM Qwen3 fused MoE kernel PR: **+13.7% output tok/s, −12% median TPOT, −17% P99 TPOT** (GB200). — [vLLM PR #24330](https://github.com/vllm-project/vllm/pull/24330)
- MLX fused expert kernels + compiled FFN block: +13% (but largely the same overhead as the buffer-limit fix) and 2.5 ms/step from compiling the FFN; routed experts at 520 GB/s "as good as gather_qmm gets." — [MLX issue #4521](https://github.com/ml-explore/mlx/issues/4521); [frederic.ai](https://frederic.ai/posts/where-the-other-half-goes)
- lithos-metal: open-source engine that **generates Metal megakernels** for Apple Silicon — one layer's mixer becomes a single megakernel; MLP remains a separate region; a "worker" model. — [lithos-metal blog](https://lithosai.com/blog/lithos-metal); [design.md](https://github.com/lithos-ai/lithos-metal/blob/main/docs/design/design.md)
- BaseRT: native Metal, chip-specific kernel fusion, dedicated GEMV decode kernels (M=1), one specialized variant per quant 2–16 bit; dequantization fused; recovers large performance vs generic stacks. — [BaseRT paper](https://arxiv.org/html/2607.00501v1); [Hugging Face blog](https://huggingface.co/blog/basecompute/basert-release)
- metalSPARSE: simdgroup-per-row GEMV for decode, register-tiled/simdgroup variants, decompress-on-tile-load — shows the shape of decode-specialized Metal kernels. — [arxiv 2611.14545](https://arxiv.org/pdf/2611.14545v1)
- Bit-exactness warning: colibri Metal decode is token-exact, but prefill GEMM runs in a different accumulation order and "can occasionally pick a different top token on near-tie logits"; a `COLI_METAL_GEMM_MIN` switch keeps GEMMs on CPU for bit-exact prefill. — [colibri docs/metal.md](https://github.com/JustVugg/colibri/blob/v2.1.0/docs/metal.md)

### Inferences
- **Mechanism (candidate #4):** fuse the router's top-k + slot-table write + read-range emission into one dispatch, and fuse gate/up/down GEMM + SwiGLU per expert into one Metal kernel; persist a megakernel for the decode layer to cut node count. This directly attacks the launch-bound wall rather than I/O.
- **Bit-exactness:** decode-safe if the fused GEMV reproduces the same per-element accumulation as `gather_qmm`/`mul_mat_id`; must be digest-gated. Prefill fusion is the risky part (order flips) — exclude it.
- **I/O hidden:** indirect — fusion does not hide SSD wait, but reduces the compute side of max(I/O, compute), improving overlap ratio; source gains +13% throughput / 2.5 ms/step are on the compute slice.
- **M4/MLX feasibility:** medium; MLX lazy eval already fuses some, so incremental headroom is the +1–2% beyond buffer tuning plus custom Metal kernels. Requires native code.
- **Falsifiable test:** implement one fused expert block; compare token digest and per-expert logits against baseline over near-tie probes; measure ms/token and node count. Predict small but real gain; reject on any digest drift.

### Gaps
- No published bit-exact node-count-to-tok/s curve for MLX MoE on M4.
- No measured "router+read-issue+GEMM" single-dispatch implementation on MLX; it is an inference from the PR #31469 KICK placement + fusion literature.

---

## Key Question 5: Speculative issuing of exact top-k and interleaving draft/verify with reads — can it work where K>1 verify failed?

### Takeaway
BigMoMo shows the speculative *window* can be exploited for MoE offload (4.83× on mobile NPU/flash via pruning + flash co-location + batching ready experts to overlap NPU compute with pending transfers), but Veystrix's K>1 verify was already slower and local evidence shows a verify batch reads the **union** of drafted tokens' experts (3–5× more SSD reads). Treat this as the highest-risk angle and only pursue the specific "prune branches/experts by acceptance rate + issue only exact top-k" refinement, with a strict digest gate.

### Cited Findings
- BigMoMo (arxiv 2609.14643): exploits the multi-token verification window to decouple expert movement from single-token execution → weight reuse, contiguous flash reads, **load–compute overlap**; prunes speculative branches and expert activations using acceptance rates, routing impact, movement cost; reorganizes on-flash experts by co-loading patterns; batches ready experts to overlap NPU computation with pending transfers. **Mean 4.83× vs on-demand autoregressive offloading, 1.82× vs best speculative MoE baseline**, up to 80B models, two mobile platforms. — [arxiv 2609.14643](https://arxiv.org/abs/2609.14643)
- Counter-evidence for SSD-bound MoE: Qwen3.8-Flash-Next spec decoding measured **slower (2.0 vs 4.3 tok/s)** because "the verify batch needs the union of the drafted tokens' experts, so a streaming target reads 3–5× the experts per step." — [neutron42-jp/qwen3.8flashnext-ssdstreaming](https://github.com/neutron42-jp/qwen3.8flashnext-ssdstreaming)
- User's own ground truth: K>1 verify = slower; perfect predictor + overlap = tied; speculative MTP depths 1–8 and lookup drafting v2 rejected (per local research notes).

### Inferences
- **Mechanism (candidate #5):** if drafting is attempted at all, prune the draft to high-acceptance branches and issue reads for ONLY the exact top-k of the tokens that will actually be verified, interleaving their reads with shared-expert/dense compute. This is "speculative issue of only exact top-k," not generic multi-token verify.
- **Bit-exactness:** verification preserves target-token distribution; but any change to expert set/precision violates the gate unless it is a pure read-scheduling change. Very high risk.
- **I/O hidden:** BigMoMo's 4.83× is mobile-NPU + flash-reorganization specific and is NOT transferable as a number; the SSD-union penalty suggests it could be *negative* here.
- **M4/MLX feasibility:** low for a win, given repeated local rejections.
- **Falsifiable test:** only run after #1–#4: issue reads for a single exact top-k draft at measured acceptance >95%, compare digest and ms/token; abandon if bytes/token rise (union effect).

### Gaps
- No Apple-Silicon/MLX measurement of high-hit-rate draft/verify interleaved with SSD reads.
- BigMoMo's benefit decomposition (how much from pruning vs flash reorg vs overlap) is not available from the abstract; full paper not fetched (HTML not retrieved this session).

---

## Key Question 6: Ranked shortlist and falsifiable test plan for Veystrix

### Takeaway
Rank by plausible value under the zero-loss gate and the "hide SSD wait" objective: (1) single-sync on-device slot resolution + per-expert compute-on-arrival; (2) MLX command-buffer limit tuning (free, bit-exact, but hides ~0 ms I/O); (3) fine-grained fused/persistent decode kernels to cut the launch-bound node count; (4) genuine multi-stream read/compute overlap — flagged as close to rejected layer-prefetch; (5) speculative exact-top-k issuing — highest risk. Nemotron 15+ is not reachable by overlap alone; Qwen 5+ is marginally reachable if overlap is near-perfect.

### Cited Findings
- Highest-value mechanism and its sync-cost quantification: [llama.cpp PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469).
- Verified free win + bit-exactness: [MLX issue #4521](https://github.com/ml-explore/mlx/issues/4521); [vllm-metal configuration](https://docs.vllm.ai/projects/vllm-metal/en/latest/configuration/).
- Fusion/persistent-kernel frontier and prefill bit-exactness caveat: [lithos-metal](https://lithosai.com/blog/lithos-metal); [colibri docs/metal.md](https://github.com/JustVugg/colibri/blob/v2.1.0/docs/metal.md).
- Overlap pipeline evidence and its collision with rejected methods: [arxiv 2411.11217](https://arxiv.org/html/2411.11217); [llama.cpp discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149).
- Reachability budget: Qwen I/O 156.9 ms vs compute+host ~99 ms; Nemotron I/O 123 ms vs compute+host ~58 ms (Veystrix profile in Context) — overlap alone cannot satisfy Nemotron 15+.

### Inferences
- **Ranked cards (mechanism / bit-exact / I-O-hidden / M4-MLX feasibility / falsifiable test):**
  1. **Single-sync on-device slot resolution + per-expert compute-on-arrival.** Bit-exact (permutation/gather; 0.000e+00 diff measured). Hides up to ~99 ms of Qwen I/O in the ideal, realistically ~40–55 ms. Feasibility medium-high. Test: sync count 48→1; digest equality; ms/token drop.
  2. **MLX command-buffer limits** (`MLX_MAX_OPS_PER_BUFFER`, `MLX_MAX_MB_PER_BUFFER`). Bit-exact (hashes matched). Hides ~0 ms I/O; +5–14% on compute/launch slice. Feasibility high. Test: env-var A/B + digest.
  3. **Fused/persistent decode kernels (router+issue+GEMV, megakernel).** Bit-exact if decode-only and digest-gated. Hides 0 ms I/O directly; cuts compute side of the overlap ratio; +13%/2.5 ms source gains. Feasibility medium. Test: near-tie logit probe + digest + node count.
  4. **Multi-stream double-buffered read/compute.** Bit-exact if bytes/math unchanged. ~2.09× pipeline / ~1.4× end-to-end source anchors. Feasibility medium; collision with rejected layer-prefetch/async_eval is the main risk. Test: GPU idle-gap instrumentation + digest; drop if no gain.
  5. **Speculative exact-top-k issue.** Bit-exactness high-risk; I/O impact possibly negative (union reads); ~4.83× is non-transferable mobile number. Feasibility low. Test: bytes/token + digest; abandon on rise.
- **Reachability verdict (my inference from the profile):** Qwen 5+ tok/s is plausible if #1 achieves near-perfect overlap; Nemotron 15+ is not reachable by overlap alone because its 123 ms I/O exceeds its ~58 ms compute+host by ~2×, so I/O *volume* must fall (a different researcher's scope).
- **Composition:** #2 is free and composes with everything; #1 and #3 target different walls (sync vs launch) and may stack; #4 overlaps #1's domain and should be tested second.

### Gaps
- No single source provides a bit-exact, Apple-M4, 16 GB measurement of any of these overlap/fusion techniques on the exact Qwen/Nemotron models; all numbers are from T4/CUDA, M3/M5 Ultra/Max, or mobile NPU.
- Whether Veystrix's current decode already does one host sync/step and already fuses expert GEMMs is unknown; that determines whether #1/#3 have headroom or are no-ops.
- No falsifiable experiment in this note has been run; all are proposals with predicted outcomes.
