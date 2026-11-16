# ANE for MoE DECODE — final gap analysis

Compiled 2026-11-17 for the Veystrix 16 GB M4 build decision.
Scope: **decode / token generation only**, **bit-identical output required**, target is a
INT4 MoE (Qwen-class ~75 GB / Nemotron-class ~15 GB) on a **base M4, 16 GB unified memory**.
No code changed. Sources are primary (papers, project READMEs/docs, vendor pages) and dated.

---

## BOTTOM LINE UP FRONT

**There is no published or demonstrated ANE role that speeds up the decode of a
bit-identical-constrained MoE on a 16 GB base M4.** Every candidate that actually moves
tokens on the ANE during decode fails on at least one of: wrong hardware (M5 Pro / M1 Max /
M4 Pro / **eight** M2 Ultra machines), wrong model class (dense 0.6B–66B, not MoE),
no bit-identity (98.4% top-1 agreement, random-weight drafters, projected-not-measured
speedups), or wrong bottleneck (LM head / router instead of expert weight reads).

The one genuine hardware superpower the ANE has — its **own DRAM-controller (DCS)
bandwidth path, independent of the GPU** — does not help here, because Veystrix's MoE
decode is bound by **SSD expert-weight reads + dispatch/sync**, and ANE cannot stage INT4
expert weights at runtime. Decode must stay on CPU/Metal. Evidence below.

---

## Q1 — Draft model / speculation helper on ANE, target on GPU or CPU

### 1.1 Mirror Speculative Decoding — Apple Research, Dec 2025 (the strongest "yes, NPU drafter" result)
Source: arXiv:2510.13161v2, <https://arxiv.org/html/2510.13161v2>;
Apple ML page <https://machinelearning.apple.com/research/mirror>.

- **Exact hardware (quote):** *"Target models are distributed across eight M2 Ultra GPUs using
  Megatron-style tensor parallelism ..., while the draft runs on eight NPUs (Apple Inc. (2023a))."*
  Appendix F.1: *"both target and draft models were distributed across eight Apple M2 Ultra
  systems, each integrating a high-throughput GPU and a dedicated Neural Engine (NPU)."*
  So this is **8 separate M2 Ultra machines** (8 GPUs + 8 ANEs), UltraFusion 2.5 TB/s between
  dies — not one Mac, not 16 GB.
- **Models:** targets Qwen3-14B, Qwen3-32B, Mistral-24B, OPT-66B — **all dense decoder-only**.
  Drafts 0.6B / 0.5B / 200M, trained from scratch with Speculative Streaming (SPD architecture),
  8 tracks sharded across the 8 NPUs. **No MoE anywhere in the paper.**
- **Measured:** 2.8×–5.8× wall-time speedups (Table 1); 3.8× average on Qwen3-14B greedy,
  up to 5.84× on math reasoning. **No tok/s reported — speedup factors only.** β=1, γ=7,
  early-exit at 1/2 depth, κ=8, bf16 token channel. "For determinism, interconnects are pinned
  and frequency scaling is disabled."
- **Bit identity:** the paper claims *"lossless and guarantees identical outputs to the target
  model under the same decoding process (see Appendix B)"* — i.e. **distributional equivalence /
  greedy argmax identity**, **not bit-identical floating-point logits**. It explicitly focuses
  on efficiency and *"without reporting accuracy metrics."* No public code or single-Mac
  reproduction.
- **Verdict for us:** concept proven at server scale; **not transferable to one 16 GB M4, no MoE,
  no bit-exact evidence, no tok/s.**

### 1.2 ANEForge exact speculative decoding — best *measured, exact* result, but wrong chip/model
Source: <https://github.com/sbryngelson/ANEForge> (README + `docs/llm.md`);
arXiv:2606.17090 (v2, 2026-10-15).

- **Exact quote (docs):** *"Measured 2.28x on Qwen3-8B with a Qwen3-0.6B draft (7.4 -> 16.8
  tok/s), using a tiled on-ANE lm_head."* `verify(K) ~ verify(1)` measured 1.05× for K=5.
  Profile: ~64% target verify (one 8B forward — irreducible floor), ~36% draft, ~0 orchestration.
- **Hardware:** repo *"verified on M5 Pro and M1 Max"*; lm_head table measured on M2 Pro
  (macOS 26.5.2). **No decode benchmark on a base M4** (only a roofline datapoint:
  5.4 TF/s fp16 GEMM, Mac16,12 16 GB). Target is **dense Qwen3-8B**, not MoE.
- **Bit identity:** *"The output is exact: the same tokens plain greedy decode would produce"*
  — via the fp16-head tie-margin host-fp32 fallback (see Q4). GPT-2 medium: 16/16 greedy match
  vs HF fp32. **This is the only ANE decode result with a published, measured bit-exactness
  mechanism.** But it is whole-model-on-ANE on a different chip class, not a drafter assisting
  a GPU/CPU MoE.
- **ANEForge's own MoE note (critical):** *"MoE decode at 30B scale is weight-bandwidth-bound:
  ~13.5 ms per MoE layer (~89 GB/s), because each token reads all of the experts' weights.
  Here int8 does help (~1.3x) ... exploiting [sparsity] needs a host-side FFN split (measured
  ~4.6x in a prototype), not the dense on-ANE path."* Qwen1.5-MoE-A2.7B runs coherent at only
  ~2 tok/s (int8), cosine >0.99 vs HF — **not bit-identical**, and the 30B-A3B needs ≫16 GB.
- **Verdict:** proof that exact ANE speculation exists; **no demonstrated path to a GPU/CPU MoE
  target on 16 GB M4.**

### 1.3 caiovicentino/ane-speculative-decoding — M4 hardware, but the 3.3× is projected
Source: <https://github.com/caiovicentino/ane-speculative-decoding> (11 stars, 1 commit, 2026).

- **Proven with measurements (on M4, macOS 15+, CoreML + llama.cpp):** ANE and GPU have
  independent DCS channels — *"GPU (main model, 4B Q4): 27.5→25.4 tok/s, 7.5% degradation"*
  when ANE active; ANE draft (46M params) *"1,199 tok/s"* alone, 743 tok/s concurrent;
  IOReport shows ANE DCS at 0 GB/s during GPU inference.
- **The headline is NOT measured:** *"Baseline (GPU only): 27.5 tok/s → Speculative
  (ANE+GPU, 70%): 84.9 tok/s ← 3.3x"* is explicitly listed under **"Projected (requires
  engineering work)"**: *"3.3x speedup assumes 70% draft acceptance rate ... ours has random
  weights ... Full integration requires connecting CoreML output to llama.cpp verification ...
  KV cache synchronization ... is non-trivial."* No verification kernel, no acceptance
  measurement, no end-to-end bit-identity claim. Target is dense 4B Q4.
- **Verdict:** the independent-DCS finding is a real hardware fact worth recording; the
  "3.3× ANE drafter" is a simulation, not a demonstration, and bits are unverified.

### 1.4 TheStage AI — vendor claim, no reproducible detail
Source: <https://app.thestage.ai/blog/4×-Faster-On-Device-LLMs:-Speculative-Decoding-on-the-ANE>
(search snippet: *"In the TheStage Apple SDK, this produces 2.4–4.1× higher decoding
throughput ..."*). The page body was not retrievable. Treat as marketing-level evidence:
no model, no chip, no tok/s, no bit-exactness statement. Do not base a build decision on it.

### 1.5 DFlash / DSpark and Apple ReDrafter — relevant alternative, but the drafter is on GPU/MLX, NOT ANE
Sources: DFlash arXiv:2602.06036 (ICML 2026) *"over 6× lossless acceleration ... up to 2.5×
higher speedup than EAGLE-3"*; DSpark arXiv:2607.05147; Liquid AI LFM2.5-DSpark
(<https://www.liquid.ai/blog/lfm2.5-dspark>) *"up to 3.18× throughput improvement on a GPU and
up to 2.87× on-device"*; MLX DFlash on Apple Silicon reported at ~85 tok/s / 3.3×
(r/LocalLLaMA), and a DFlash companion exists for **Nemotron-3.5-Lightning-30B-A3B (MoE)** via
llama.cpp (`-hfd ...-DFlash-GGUF`). These are **exact** (speculative verification) and MoE-capable
targets exist — but every one runs the drafter on Metal/MLX/GPU, not the ANE. They are the
strongest *non-ANE* alternative and should not be conflated with an ANE win.

### 1.6 MacPaw Research — whole model on ANE, faster/lower-energy, but not bit-identical and M4 **Pro**
Source: <https://research.macpaw.com/projects/neural-engine> (Oct 2026, Core AI on macOS 27).

- M4 Pro Mac mini: LFM2.5-2.6B **prefill and decode entirely on ANE** with a 328M DSpark
  drafter. ANE decode: plain 32.2 / chain 98.1 / **tree 136.9 tok/s**; GPU 8-bit MLX 81.5,
  GPU 8-bit + DSpark 144.9 tok/s (ANE ~5% lower). ANE prefill 1.9× GPU at ~1/5 energy;
  2.6–3.4× less energy per request.
- **Quality is explicitly not bit-exact:** *"98.4% Teacher-forced top-1 agreement with bf16,
  against 99.0% for MLX 8-bit, within the 8-bit quality class."*
- Critical hardware fact: *"The ANE attains only 44 to 70 percent of the GPU's effective memory
  bandwidth, whereas its compute throughput is 2.0 to 2.7 times higher."* Decode needs
  weight reuse (tree verify) to win. No public code; M4 **Pro** only.
- **Prompt router on ANE** (350M bidirectional encoder, 5.5 ms/call @128 tok): <1% chat
  generation rate impact vs 20% if run on GPU. **Important distinction:** this is a *prompt*
  router (route/tool selection before generation), **not** the MoE per-token top-k expert router.

**Q1 conclusion:** A drafter on the ANE has been demonstrated three ways — Apple's 8-machine
server paper, ANEForge's exact dense-8B result, and MacPaw's whole-model-on-ANE engine — plus a
projected M4 simulation. **None** is a bit-identical ANE drafter for a MoE target on a single
16 GB base M4.

---

## Q2 — Can the ANE help the MoE ROUTER (top-k expert selection) during decode?

**No plausible win; and it would not be bit-identical.**

- NPUMoE (arXiv:2604.18788, UVA, M2 Max/Ultra) is explicit: *"layerNorm, masking,
  expert routing (top-k/argmax), and token dispatch/merge operations (scatter, gather, weighted
  combine) execute on the CPU due to their dynamic indexing and irregular memory access
  patterns."* It calls top-k / dynamic indexing *"not NPU-friendly"* and keeps routing on CPU.
  Its ANE role is **prefill** dense FFN/attention, not per-token routing. Its capacity-tier +
  pruning policy **drops tokens** (10–20% in worse tiers) → breaks bit-identity anyway.
- ANEForge does expose a native `topk` bridge op, but its MoE path builds the top-k mask from
  `amax + select` because that graph cut *"won't compose across layers"* — and that is with the
  *whole model* on ANE.
- Bit-exactness: fp16 router logits near a top-k tie can flip the selected expert order; a
  different expert set changes the output. Restoring identity needs an fp32 host recompute of
  the router logits every token — a round trip that costs more than the tiny router matmul.
- Performance: the ~51 ms (Qwen) / ~37 ms (Nemotron) router sync is, per our own trace,
  **waiting/memory copy**, not router compute. Moving the matmul to ANE leaves the copy
  untouched and adds a cross-engine hop. MacPaw's "router on ANE" is a *prompt* router, not
  MoE top-k, so it does not transfer.

---

## Q3 — Can ANE prefetch/decompress expert weights faster than the SSD path?

**No published work, and the mechanism is structurally unavailable.**

- ANEForge streams int8 / int4-LUT / sparse weights through the engine's dequant path
  (~4× smaller for int4), but those weights are **baked into a compiled program** (~2 GB ceiling)
  and delivered via IOSurface/`share_buffer`; there is **no runtime weight binding**
  (our earlier finding: runtime override is dead or ~5× slower). Compressed formats are for
  *storage footprint*, not for accelerating SSD→SRAM streaming.
- NPUMoE explicitly assumes **Scenario 2 — all experts already resident in unified memory**
  (*"we focus on scenario 2 and assume all model weights reside in device memory"*); it has no
  SSD path at all.
- No 2025–2026 work was found on ANE-assisted decompression/dequantization of MoE expert
  weights fetched from SSD. The ANE cannot read storage directly — bytes must pass through the
  CPU/IOSurface anyway — so it cannot sit "in front of" the SSD path without adding a hop.
  For 15–75 GB of INT4 experts on a 16 GB machine, the bottleneck is the SSD/PCIe read and the
  dispatch, neither of which an NPU dequant block removes.

---

## Q4 — ANE for the LM head / sampling / logits processing at decode?

**A bit-exact mechanism exists (ANEForge), but it is a correctness guard, not a MoE decode win.**

- ANEForge `ane_lm_head` (M2 Pro measured, fp16): gpt2-medium 10.8→3.4 ms/tok
  (39.6→59.4 tok/s); Qwen3-0.6B 30.1→8.2 ms/tok (17.8→29.3 tok/s); also drops the 206–622 MB
  host `_lmT` allocation.
- **Bit-exactness bridge (quote):** *"A tie-margin fallback recomputes a step on the host fp32
  head when the top-2 gap is under `lmhead_tie_margin` (default 3e-3) times the max logit
  magnitude — about 6× the measured fp16-head error, firing on ~2–3% of tokens. Greedy then
  matches the host (and Hugging Face) exactly while the rest keep ANE speed; sampling stays on
  the ANE."* This is the **only** published, measured bit-exactness bridge in this body of work.
- Why it doesn't help Veystrix's MoE: it requires the hidden state to be *on the ANE* (the
  ANEForge runtime runs the whole target there). In a GPU/CPU MoE, using ANE just as an LM head
  means copying the hidden state to ANE and logits back **every token**, on top of the 51/37 ms
  router sync and expert reads. The LM head is not the bottleneck; this would add latency.
  Our internal notes already recommend the tie-margin rule as *general bit-exactness
  infrastructure* for any low-precision path — correctly labeled as **~0 decode speedup** on
  the MoE path.

---

## Q5 — Final verdict and the hardware reason

**Verdict: NO. For a bit-identical-constrained MoE on a 16 GB base M4, there is no ANE decode
role with a plausible measured win. Decode must stay on GPU/CPU.** The strongest exact ANE
decode numbers (ANEForge 2.28×) are for a dense 8B on M5 Pro/M1 Max; the strongest MoE-relevant
ANE system (NPUMoE) targets prefill and drops tokens; the M4-Hardware ANE spec result
(caiovicentino) is a projection with random draft weights; MacPaw's 136.9 tok/s ANE tree decode
is M4 Pro, whole-model-on-ANE, and only 98.4% top-1 agreement. None satisfies
"bit-identical + 16 GB base M4 + MoE + decode" simultaneously.

**Hardware reason (one paragraph).** The ANE's advantage is low-energy, single-program fused
matmul on its own memory path; it is not a general accelerator. Its weights are baked into a
~2 GB compile-time program with no runtime re-binding, its dataplane is fp16-only (no INT4
experts, and fp16 rounding can flip argmax/top-k at ties, requiring an fp32 host fallback), and
its dispatch floor is ~90 µs for a single token — larger than the ~23 µs of useful work. A 16 GB
MoE decode is dominated by reading 15–75 GB of INT4 experts from a saturated SSD and by
dispatch/sync (~51 ms Qwen / ~37 ms Nemotron), not by the dense arithmetic the ANE is good at.
Its independent DCS bandwidth (real, ~7.5% GPU degradation when both run) cannot help when the
limiting resource is the SSD and when expert weights cannot be staged into the engine. Adding
any ANE hop (drafter, router, LM head) therefore inserts a cross-engine transfer into a path
that is already memory/copy-bound, without removing the dominant cost — so the honest answer
is no.

---

## ACTIONABLE FOR VEYSTRIX 16 GB

**NONE for ANE decode.** Keep decode on CPU/Metal and proceed with the already-in-flight
non-ANE candidate: **expert-wise K=2 read sharing** (group two tokens by expert, union load once,
baseline-order accumulation) — it is bit-identical by construction and targets the actual
SSD-read bottleneck. The only ANE-adjacent item worth stealing is ANEForge's **tie-margin fp32
fallback** as a general bit-exactness guard for any future low-precision head; expect **zero**
decode speedup from it on the MoE path.
