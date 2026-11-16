# Direct / low-level ANE runtimes for LLM inference (2025–2026)

Research date: **2026-11-17**. Target hardware: **16 GB Apple M4 Mac (base M4 / M4 Air)**, bit-identical-token constraint, Python-driven MLX engine.

Every URL below was fetched and confirmed live on 2026-11-17 unless explicitly flagged. "Unreplicated preprint" means a single-author arXiv paper with no independent reproduction of its headline numbers; this flag is applied aggressively because several of these results are extraordinary.

---

## 0. Verification ledger

| Source | Verified URL | Date / version | Authors | Status / caution |
|---|---|---|---|---|
| ANEForge paper | https://arxiv.org/abs/2606.17090 | v1 2026-06-12, v2 2026-10-15; 8 pp | **1** — Spencer H. Bryngelson (UT Austin) | arXiv preprint, **unreplicated**. But backed by a real repo (368 commits), PyPI package, CI, Zenodo archive, docs site, cross-chip leaderboard, and an SPE journal DOI badge (10.1002/spe.70124). |
| ANE guide (companion) | https://arxiv.org/abs/2606.22283 | v1 2026-06-21, v2 2026-10-15; **303 pp, 218 tables** | **1** — same author | Preprint; decompilation-derived claims marked by author. |
| ANEForge docs | https://aneforge.readthedocs.io | live | — | Detailed, CI-gated operator/roofline data. |
| ANEForge repo | https://github.com/sbryngelson/ANEForge | 76 stars, 21 forks, MIT | — | Note: repo is **sbryngelson**, not "e5rt" as a standalone project. |
| ANEForge rooflines | https://github.com/sbryngelson/ANEForge/blob/main/bench/results/ROOFLINES.md | 11 machines / 15 submissions | — | **Includes a base M4, 16 GB machine** (Mac16,12, macOS 26.2). |
| Orion paper | https://arxiv.org/abs/2603.06728 | v1 2026-03-06 only | **1** — Ramchand Kumaresan (affiliation given only as a GitHub URL) | **Unreplicated preprint**; "Murai Labs" in README; GitHub-only identity. Treat numbers as claims. |
| Orion repo | https://github.com/mechramc/Orion | 134 stars, 16 forks, 49 commits, MIT | — | Native ObjC, no Python at runtime; Python wrapper "planned". |
| maderix/ANE | https://github.com/maderix/ANE | 7.3k stars, 957 forks, MIT | 1 (M. Singh) | Foundational reverse-engineering; explicitly "research project, not production". |
| m0at/ANEtransformers | https://github.com/m0at/ANEtransformers | 5 stars, 12 commits, fork of maderix | — | Mistral-7B C/ObjC engine; all numbers on **M5 24 GB**, not M4. |
| ANEMLL main | https://github.com/Anemll/Anemll ; https://www.anemll.com/ | v0.3.5 Beta (2026-02-14) | community | CoreML-based, not direct. |
| anemll-forge | https://github.com/Anemll/anemll-forge | 56 stars, 107 commits, release 0.2.1 (2026-10-19) | community | **M5/M6 only, macOS 27 + Core AI, 32 GB recommended** — not usable on M4 16 GB. |
| CoreML-LLM | https://github.com/john-rocky/CoreML-LLM (docs: PHASE_B_DECISION.md, HANDOFF.md, PHASE_D_...) | 2026-04 | — | Essential negative results on bit-exactness and KV handoff. |
| NPUMoE | https://arxiv.org/abs/2604.18788 | v1 2026-04-20 | **2** — Afsara Benazir, Felix Xiaozhu Lin (UVA) | Preprint; authors acknowledge generative-AI editing; "will make codebase open-source" (not yet). **M2 Max/M2 Ultra only.** |
| FusionML | https://arxiv.org/abs/2607.22785 ; https://github.com/ommo007/FusionML | v1 2026-07-24 | **1** — Om Mohite | Preprint, community-replicated on 5 chips incl. **M4**. Deliberately retires its own earlier FP32-confounded claims. |
| MacPaw Research | https://research.macpaw.com/publications/llm-inference-apple-neural-engine | 2026-10-10 | institutional (MacPaw) | Corporate lab, detailed methods; **M2/M3/M4 Pro only**, macOS 27 Core AI. |
| SqueezeBits Yetter | https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176 | 2025-08-26 | corporate | CoreML ANE prefill + MLX GPU decode; iPhone 15 Pro, iOS 18.6.1; handoff cost **never measured**. |
| AtomGradient | https://github.com/AtomGradient/hybrid-ane-mlx-bench ; paper PDF | 2026-03-07 | 1 org, 5 stars | **M2 Ultra 192 GB**; reports a measured CoreML-vs-ANE routing trap. |
| CrossInfer | https://arxiv.org/html/2610.35309v2 | 2026-10-27 | multi-author | VLM state handoff across engines; **M4 Max** only; not a Python/MLX runtime. |
| FusionML/CrossInfer "ANE exclusion" | see above | — | — | Critically, their "ANE dispatch is too expensive" conclusion is about **CoreML dispatch (20–24 ms)**, not private-API dispatch (~0.07–0.13 ms). See §5. |
| ANEForge ane-leaderboard | https://huggingface.co/spaces/aneforge/ane-leaderboard | live | — | Interactive cross-chip table. |
| "ANEForge / e5rt" naming | — | — | — | There is **no separate e5rt project**. e5rt is Apple's private Espresso C API; ANEForge wraps ~30 of its 235 symbols. |
| "CrossInfer", "MacPaw Yetter", "HeteroLLM" as one thing | — | — | — | CrossInfer = arXiv 2610.35309; Yetter = SqueezeBits; HeteroLLM = arXiv 2501.14794 (mobile SoCs, not Apple-specific). They are distinct. |

---

## 1. Headline verdict

1. **ANEForge is the only project here that is simultaneously (a) Python-callable, (b) unentitled, (c) CoreML-free, (d) installable today, and (e) has a base-M4 16 GB datapoint.** It is also the only one whose API is structurally capable of wrapping a single baked-weight projection with IOSurface I/O for an MLX engine to call (`af.input → af.compile → net(x)`, `compile_multi`, `share_buffer`, and a raw `ane_e5rt_*` C ABI).
2. **Orion's interesting contribution (delta compilation, LoRA-as-input) is real engineering but is ObjC-only, macOS 15+, GPT-2/110M-scale, and single-author unreplicated.** Its headline 170 tok/s is *slower* than its own CPU baseline (283 tok/s).
3. **No published source gives a measured ANE-vs-MLX-GPU speedup for prefill-shaped matmuls (S≥256) on the base M4 / M4 Air.** The M4 datapoint that exists is ANE-only (5.4 TF/s peak fp16 GEMM). Everything claiming a speedup over MLX is on M5, M4 Pro/Max/Ultra, M2 Ultra, or a different OS/Core AI stack.
4. **For a bit-identical constraint, the ANE is a liability at the cross-engine boundary.** CoreML-LLM measured a bit-exact failure at token 50 from fp16 ANE/GPU rounding; FusionML saw one near-tie divergence after 289 identical characters; NPUMoE *drops* 10–20% of routed tokens. The only demonstrated bit-exact trick is ANEForge's **lm_head tie-margin host-fp32 fallback** (~2–3% of tokens recomputed), which preserves greedy equality with HF fp32 exactly.
5. **The single most promising technique for a 16 GB bit-identical MoE pipeline is not "put the MoE on ANE"; it is FusionML's eager-materialization CPU+GPU row-split for prefill, with ANE used (if at all) only as a power-optimized, acceptance-gated prefill island.** Details in the final section.

---

## 2. ANEForge / e5rt (arXiv 2606.17090 + arXiv 2606.22283 + docs)

### 2.1 What it actually is

A Python package (`pip install aneforge`) that builds a lazy tensor graph, lowers it to Apple MIL, packs weights into one blob, compiles through Apple's `aned` daemon, and dispatches **only** to the ANE via `e5rt_*` symbols in `Espresso.framework` resolved with `dlopen`/`dlsym`. No entitlement, no `.tbd` linkage, no CoreML, SIP left on, code signing untouched. A small Objective-C++ shim (`aneforge/_lib/ane_e5rt_dispatch.mm` → `libane_e5rt_dispatch.dylib`) is compiled locally on first dispatch. Device mask is `0x4` = ANE (`0x1` BNNS/CPU, `0x2` MPSGraph/GPU).

- Compiler targets: 58 fused ops (paper v2 says 100 fused ops / 4 attention composites; README says 58 + 19 bridge — the counts grew between versions) + 19 native bridge ops. MIL vocabulary 166, of which 116 compile and run; 45 hardware layer descriptors, 50 per-layer validators, 26 authored layers, 19 promoted.
- Native fused attention (`af.sdpa`) reaches the hardware SDPA layer that coremltools' `scaled_dot_product_attention` never emits (Apple decomposes it to matmul→softmax→matmul).
- Autograd: full forward, backward **and Adam optimizer update** on the ANE (F+B+O; Orion only does F+B with O on CPU).
- Numerical/linalg: fixed-iteration Krylov CG, factored FFT, Cholesky, eigenvalues as static-dataflow ANE programs.

### 2.2 Install (exact)

```
# macOS 14+ (verified 26.5 / 26.6.2), Apple Silicon, Xcode CLT, Python 3.10+
pip install aneforge
python -m aneforge.build          # optional: pre-build the e5rt dispatch dylib

# or from source:
git clone https://github.com/sbryngelson/ANEForge.git
cd ANEForge
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[models]"   # adds torch/torchvision/transformers
sh aneforge/_lib/build.sh
PYTHONPATH=. python3 tests/op_smoketest.py
```

No entitlement, no SIP disable, no jailbreak. Verified on M5 Pro and M1 Max; FAQ says "any Apple Silicon Mac M1–M5, all variants, macOS 14+"; M4 = family H16 (one of 28 compiler targets) and `af.compile(y, target="h16s")` gates a graph for M4. **Caveat:** the operator corpus (90 variants) was actually run on hardware only for M1 Max and M5 Pro; macOS 14/15 verification is import-only CI. M4 is a ground-truth capability family but not a silicon-measured anchor in the paper. Roofline JSON for a real base M4 exists (see §5).

### 2.3 Python API (the part relevant to wrapping a projection)

```python
import aneforge as af
x   = af.input((1, C, 1, S))            # lazy Tensor node, fp16 I/O
y   = x @ Wfc                           # W baked into the program's weight blob
net = af.compile(y, compress="int8")    # ONE fused ANE program; ~750 ms cold compile
out = net(x_fp16)                       # ~80 us eval, reused handle
net.release()
```

Key callable surfaces:

- `af.input(shape)`, `af.image_input(shape, scale, bias)` (uint8→fp16 dequant on engine).
- `af.compile(out, compress=None|"int8"|"int4"|"sparse"|"blockwise"|"auto", target="h16s", validate=True)` → callable `Program`.
- `af.compile_multi(outs)` → `MultiModel` with N **named** output ports.
- `Program.share_buffer(src_op, src_port, dst_op, dst_port)` → output→input buffer aliasing; this is how KV and optimizer state stay resident on silicon.
- `Program.set_input(...)`, `Program.execute()`, `Program.read_output(...)`.
- `af.load_llm(name, compress=...)`, `model.prefill(ids)`, `model.generate(ids, max_new_tokens, eos_id, ane_lm_head=...)`, `af.LlamaPrefill(cfg, weights)`, `af.LlamaConfig`, `af.rope`, `af.prefill_block`.
- `from aneforge.speculative import spec_generate`; `from aneforge.moe import load_gguf`; `from aneforge.qwen35 import load_gguf`.
- `af.tune(out, atol=...)` — cost-model + on-device measurement of equivalent lowerings, returns the lossless pick.
- `af.Trainer(..., device_optimizer=True, resident_state=True)`, `af.UnrolledTrainer`.
- `af.estimate(out, target='hXX')`, `af.project_peak('h17s')`, `af.op_info`, `af.device_status`.

Raw C ABI (for a non-Python MLX engine to call directly): `ane_e5rt_program_compile(mil_path, cache_dir, device_mask=0x4, input_names/sizes, output_names/sizes)`, `ane_e5rt_program_set_input_fp16`, `ane_e5rt_program_execute`, `ane_e5rt_program_get_output_fp16`, `ane_e5rt_program_release`; multi-op `add_op`, `chain_ops`, `share_buffer`, `execute_multi`; async `execute_async`/`wait_for_completion`; cross-process IOSurface-by-Mach-port tensor handoff (~70 µs). Full call sequence is in the e5rt dispatch reference doc.

**Answer to "can it wrap a single q_proj with baked weights + IOSurface I/O for MLX to call?" — Yes, architecturally.** A one-op graph (`af.input → matmul(af.const(Wq)) → af.compile`) is exactly the supported unit; `compile_multi` + `share_buffer` gives named ports and resident state; and the C ABI lets any native code drive it. What does **not** exist today is an MLX integration: no `mx.array ↔ IOSurface` zero-copy bridge is documented by either project (llms-explorer explicitly lists "can an IOSurface or MTLBuffer holding Core ML KV outputs be wrapped as an MLX array without a copy?" as **open**). Also note: e5rt's buffer type 0 is a plain CPU pointer (`get_data_ptr`), not an IOSurface — the IOSurface path is `share_buffer` aliasing and the separate cross-process Mach-port handoff.

### 2.4 Weights, streaming, chaining

- `compress="int8"` = per-channel affine dequant during tile DMA; `"int4"` = 4-bit LUT; `"sparse"`; `"blockwise"`; `"auto"` is family-aware. On A15+ (M3/M4/M5) all four stream natively. A 4096² matmul weight blob drops 33.6 MB → 8.4 MB with int4. Compression is **off by default**; default program is byte-identical to fp16.
- Accuracy gating: int4 falls back int4→int8→fp16 within `compress_atol`.
- Chaining: `af.tune` fuses Q/K/V/attention/O into one program; **for S≤256 this is 3.8–5.3× faster** because the unrewritten block pays the dispatch floor five times (M5 Pro measurement). At S=512/1024 the tuner keeps the native SDPA route (measured 1.00×). Attention rewrite is **bit-identical**: "relative max-abs difference 0.0000 in every row."
- `execute_multi` runs K aliased steps under one `execute_sync`; measured **performance-neutral** (64 ops: 66 ms vs 60 ms for 64 separate calls) — per-step host dispatch is not the bottleneck once buffers are bound. Ceiling ~110–120 programs per PID.

### 2.5 Measured throughput / latency

All headline timings: **M5 Pro, macOS 26.6.2, ANE compiler 9.509.0, Espresso 3525.1.1, v0.4.0**, min over 30 calls.

| Workload | ANE | GPU fp16 | Notes |
|---|---|---|---|
| ResNet-18 forward | 0.334 ms, 2.34 mJ | 1.92 ms MPS / 1.83 ms MLX | 5.5–5.8× faster, ~9× less energy than MPS (paired 5.5–12.7, median 8.5); vs MLX energy 14.7× |
| Dispatch floor | 74 µs intercept (128 fused 1×1 convs cost nothing measurable) | GPU floor 130–150 µs | Sustained loop ≈2.6× the probe minimum (195 µs vs 74 µs) |
| Single relu, cross-over | CPU wins <64 KiB; GPU wins >512 KiB | — | ANE advantage comes from fusion, not single ops |
| Attention 8h S=256 | 844 → 220 µs (3.83×) after rewrite | — | bit-identical |
| ViT-B/16 attention shape S=197 | 917 → 226 µs (4.05×) | — | bit-identical |
| GEMM floor M=64,K=256,N=256 | M1 Max 164 µs / M2 Pro 110 µs / M5 Pro 92 µs | — | no M4 row in the paper |

LLM numbers from the docs (chip not always stated; M2 Pro for lm_head table):

- Qwen3-0.6B: **~8,600 prompt tok/s prefill**, ~75 tok/s decode.
- Qwen3-8B (36 layers, 16 GB fp16, 9 chunks under the ~2 GB/program ceiling): **~7.5 tok/s decode**; with 0.6B draft + tiled on-ANE lm_head, exact speculative decoding **2.28× (7.4 → 16.8 tok/s)**.
- On-ANE lm_head: gpt2-medium 10.8 → 3.4 ms/tok (39.6 → 59.4 tok/s); Qwen3-0.6B 30.1 → 8.2 ms/tok (17.8 → 29.3 tok/s).
- Qwen1.5-MoE-A2.7B coherent at ~2 tok/s int8; 30B-A3B MoE layer ~13.5 ms (~89 GB/s), int8 ~1.3×; host-side FFN-split prototype measured ~4.6×.
- Qwen3.5-27B GGUF int8 decodes coherently end-to-end on pure ANE (needs ≫16 GB).

### 2.6 Numerical accuracy

| Model | Agreement |
|---|---|
| ResNet-18 | cosine 1.0000 vs torchvision fp32, top-1 match, relerr 2.3e-3 |
| ViT-B/16 | cosine 1.0000, top-5 5/5 |
| MiniLM L6 | cosine 1.0000 |
| SD U-Net / VAE decoder (one fwd) | relerr 0.041 / 0.021 |
| Attention rewrite (all S) | **max-abs diff 0.0000** |
| GPT-2 medium, 16-layer decoder | **16/16 greedy match vs HF fp32** |
| Qwen3-0.6B | prefill logits cosine ~1.0, identical argmax; "The capital of France is" → " Paris" |
| Qwen3-0.6B int8 | byte-identical to fp16 |
| Qwen3-8B int8 | **collapses into repetition** — run large models in fp16 |
| Qwen3.5-27B | fp32 reference matches llama.cpp exactly (corr 1.0); on-device int8/fp16 coherent but lower fidelity; per-channel int8 alone moves logit corr ~0.0001, fp16 activation outliers are the real loss |
| Cross-chip numeric cliffs | matmul inf cliff ~32759 on all chips; pre-A16 slice clamps \|x\|>4094 (M4 base is "exact, no clamp"); reduce exact-sum ≤2048 |

**Bit-identical mechanism worth stealing:** `ane_lm_head` runs fp16, so a greedy argmax could flip on a close tie. ANEForge adds a tie-margin fallback: if top-2 gap < `lmhead_tie_margin` (default `3e-3`) × max logit magnitude, recompute that step with the host fp32 head (~2–3% of tokens). Greedy then matches host/HF **exactly**; sampling stays on ANE. This is the only bit-exactness bridge with a published, measured mechanism in this body of work.

### 2.7 Anchoring caveats

- Single-author preprint; the paper is a "research project ... not for production" (FAQ), private symbols can change any macOS release, e5rt Program is single-threaded, no App Store distribution.
- Compiler-validated ≠ numerically correct on another chip: "whether that curve ranks as well on another chip was not measured."
- fp16-only dataplane; no bf16, no fp32 dataplane; ANE's NaN handling is non-IEEE (`gelu(0.0) = -0.000754`, `mul(+inf,x) → -52416`).
- Single hardware lane: gains come from fusing work into one program, not concurrent dispatch.

---

## 3. Orion (arXiv 2603.06728, github.com/mechramc/Orion)

### 3.1 Install / requirements

`macOS 15+`, Apple Silicon (M1+), Xcode CLT, Python 3.10+ **only for weight conversion**. Runtime is pure ObjC — no Python wrapper exists (roadmap Phase 3/4 "planned"). Private APIs: `_ANEClient`, `_ANECompiler`, `_ANEInMemoryModel`, `_ANEInMemoryModelDescriptor`, `_ANEModel`, `_ANERequest`, `_ANEIOSurfaceObject`, loaded via `dlopen`/`objc_getClass`. Entitlement status **not stated** by the paper; ANEForge's Table 1 marks it "✓a" (inferred unentitled, because these in-memory-model classes are callable without entitlement). Would build/run on an M4 16 GB Mac, but models are GPT-2 124M and Stories110M only — no 7B+.

```
git clone https://github.com/mechramc/Orion.git && cd Orion
make
python model/convert/hf_to_blobs_gpt2.py     # GPT-2 124M inference
./orion infer --prompt "The meaning of life is" --max_tokens 128 --ane
./orion infer --prompt "..." --ane-prefill    # ANE prefill → CPU decode
./orion bench kernels
```

### 3.2 Delta compilation / adapter-as-input / LLM numbers

- **Delta compilation:** weights are baked at compile time; naive training recompiles 60 weight-bearing kernels per step (~4,200 ms). Orion v2 does `unloadWithQoS → patch BLOBFILE on disk → loadWithQoS`, bypassing `ANECCompile()`; same `hexStringIdentifier` (3× SHA-256 over MIL+weights+options) preserves program identity. **4,200 → 494 ms (8.5×) recompile; 5,108 → 1,345 ms total step (3.8×); 1,000 steps 22.4 min vs ~85 min; 0 compiles during training; no `exec()` restarts.** Note this is a *training* optimization, not a decode/prefill one.
- **LoRA adapter-as-input:** `Y = X·W_base + α·(X·A)·B` with A,B as IOSurface inputs, base W baked; 8 adapter matrices for attention Q/K/V/O; hot-swap with zero recompiles, 17/17 tests. Discovered constraints #18–20: all IOSurface inputs must have **uniform byte allocation**; inputs are bound **alphabetically by MIL parameter name**; over-allocated surfaces are read as packed `[1,C,1,S]` from **byte 0**.
- **Inference (M4 Max, GPT-2 124M):** CPU decode 283 tok/s (3.48 ms); **ANE full-forward decode 170 tok/s (5.76 ms)**; **ANE prefill first call 12 tok/s (~1015 ms compile for 24 programs); cached prefill 165 tok/s**. So *no* ANE-over-CPU win on decode; the paper attributes it to ~2.3 ms IOSurface round-trip per dispatch. Hybrid `--ane-prefill` mode exists precisely because of this.
- **Accuracy:** 100% top-1 argmax vs CPU fp32; max logit error 0.073 across 12 layers; exact 64-token greedy continuation for "The meaning of life is"; training 0 NaN / 1,000 steps with loss 11.83→8.9 (v2) / 12.29→6.55 (v1).
- 20-constraint catalog (14 newly documented): notably `concat` rejected; `gelu` invalid → tanh approx; `rsqrt` invalid → `pow(x,-0.5)`; 32K-channel conv rejected (forces lm_head to CPU); ~49 KB IOSurface min so S=1 fails at eval (min decode bucket S=16); multi-output surfaces ordered alphabetically; BLOBFILE offset is `uint64(64)`; 1×1 conv ≈3× matmul throughput; 32 MB SRAM cliff (~30% drop); ~119 compiles/process.
- 27-op graph IR, 5 passes (DCE, identity elimination, cast fusion, SRAM annotation, constraint validation), 13 frontends.

**Replication flag:** single author, single v1, GitHub-only affiliation, all results self-reported on one M4 Max under macOS 15, no external reproduction found. The "3.8× training speedup" has no GPU/MLX comparison — it is against the authors' own v1.

---

## 4. maderix/ANE, m0at/ANEtransformers, ANEMLL, CoreML-LLM

### 4.1 maderix/ANE (the foundation)

7.3k stars, MIT, C/ObjC via `_ANEClient`/`_ANECompiler`. Honest self-description: research hack, ~5–9% of peak utilization, many elementwise ops fall back to CPU. Key numbers (M4):

- Peak fp16 ~18.6 TOPS, INT8 W8A8 ~35.1 TOPS (**1.88×** — INT8 saves L2 bandwidth, not compute; dequantized to fp16 before compute).
- Dynamic-weight kernels avoid recompilation: Stories110M **91 ms/step**, Qwen3-0.6B **412 ms/step** (all fwd+bwd dx on ANE, dW on CPU cblas).
- **GPU prefill → ANE decode over shared IOSurface at S=256:** Stories110M 6.7 ms + 1.9 ms = 8.8 ms; Qwen3-0.6B 9.7 ms + 2.3 ms = 12.0 ms. Explores `_ANESharedEvents`/`_ANESharedSignalEvent`/`_ANESharedWaitEvent` as Metal-style fences.
- Channel-first CPU layout eliminates transpose; vDSP RMSNorm 6.7→0.7 ms; GCD-async cblas overlap.

### 4.2 m0at/ANEtransformers (7B, C/ObjC, M5 only)

Fork with a full Mistral-7B Q4_0 engine (~9,000 lines C/ObjC, GGUF mmap, no deps beyond system frameworks). **Measured on M5 24 GB, not M4:**

| Backend | Prefill tok/s | Notes |
|---|---|---|
| ANE baked conv1x1 (128 programs, 4/layer) | **76.6** at S=361, ~2.6 TFLOPS | fp16 weights baked; 4 ANE dispatches + 6 CPU phases/layer |
| Metal GEMM (Q4_0, 64-tok tiles) | 58.0 | 1.3× slower than ANE |
| BLAS tiled (AMX) | 37.0 | |
| **MLX reference** | **~200+** | **2.6× faster than ANE** |
| CPU NEON W4A8 SDOT decode | 19.7 tok/s | ANE dispatch ~70 µs makes S=1 pointless |

Single-layer S=64: BLAS 118.1 ms → ANE 23.9 ms (**4.9×**). Accuracy: transpose round-trip bit-exact; ANE conv vs CPU matmax abs err <0.04; single layer max_err 0.026 / mean 0.0007; **full 32-layer prefill cosine 0.998, max_err 0.61**; text matches BLAS path.

**7B-in-16 GB memory budget** (their 24 GB accounting, which fits 16 GB with less headroom): Q4_0 GGUF mmap 3.8 GB; ANE compiled programs ~0 GB in user process (live in `aned`); IOSurfaces ~5 MB (shared across all 128 programs, sized to the largest = FFN); KV cache 256 MB @4K; dequant scratch 352 MB peak during init; steady state ~8.3 GB, peak ~8.6 GB. Techniques: on-demand embedding dequant (saves 496 MB), stream compilation one weight at a time, symlink weight blobs into tmpDir (saves 130 ms per FFN program), 2-phase cold-compile (~80 s, daemon cache lost on reboot)/warm-forge via forged `hexStringIdentifier` (~220 ms load, ~400× faster init).

**Hard constraints relevant to a projection wrapper:** ANE has max **2 outputs** per program (3+ silently corrupt all but last — hence Q separate, K+V fused); 16 BLOBFILE weight tensors max per program (Orion); runtime weight override IOSurface is **dead**; INT8/INT4/LUT all **fail to compile** via `_ANEInMemoryModel` (must dequantize to fp16) — note this directly contradicts ANEForge's e5rt-path compressed-weight streaming, so the two access routes genuinely differ; `reduce_mean/reduce_sum/rsqrt` fail (RMSNorm on CPU); silent all-zeros for S<16 (fp32) or S<32 (fp16); min IOSurface 49,152 bytes; `_ANEChainingRequest` returns Error 15 (entitlement-gated); ~119 compiles/process; wrong layout gives plausible-but-wrong results (cosine 0.02 vs 0.998 correct).

### 4.3 ANEMLL (CoreML, 1.5k stars)

HF→CoreML conversion pipeline, FP16 + LUT quantization, monolithic/chunked, Swift + Python inference, IOSurface-backed buffers, serial prediction, ping-pong buffers for FFN chaining, 16-deep ring buffer for monolithic outputs. v0.3.5 Beta (2026-02-14). Requirements: **macOS Sequoia (15+), 16 GB min / 32 GB for 8B**, native arm64 Python 3.9–3.11 (x86_64 under Rosetta silently loses ANE). Prefill default 64 tokens, fixed Q=64 so every call costs the same; 512-wide sliding K cache (448 old + 64 new). Community repro on M4 Max: R1-8B ANEMLL 9.3 tok/s vs MLX 8-bit ~50 and llama.cpp 8-bit ~41. ANE weight-read bandwidth: base M4/M5 60–70 GB/s, M4 Pro/Max 120–152 GB/s — base M4 ANE is ~53% of system memory bandwidth. CoreML cannot steer placement directly; Core ML silently moves all ops to CPU at ≥16 layers (Outlier.host measurement on M1 Ultra: 8 layers 22.9 ms on ANE, 16 layers 90.7 ms all-CPU, 32 layers 139–242 ms; GPU→ANE handoff measured only 0.84 ms — residency, not handoff, is the wall).

### 4.4 anemll-forge (NOT usable on M4 16 GB)

Qwen3.8-27B on **M6** (M5/M5 Pro/M5 Max ~2× slower), macOS 27 + Xcode 27/Core AI, Python 3.11, **32 GB recommended**. First compile 20–25 min. Release 0.2: 3-bit MLP VQ + online Hadamard rotations, GPTQ+QAT, KL-512 0.184→0.052, top-1 86.0→91.9% vs BF16. C2T 8-bit attention on M6 vs Splash GPU (UD-IQ3_XXS, same benchmark): prefill 306/275/237 tok/s at 8K/32K/64K vs GPU 311/272/231, decode 60.8/54.5/47.9 vs 54.2/52.0/48.7 — i.e. ANE roughly at GPU parity with lower energy. M5 cannot compile FP8 attention (Core AI falls back to GPU ~2 s/call); M5 uses FP16-instead-of-FP8 C2T.

**KV handoff mechanics (best documented anywhere):** target KV stays in **host-owned FP16 IOSurface buffers**; on context growth, allocate new K/V buffers, copy cached prefix rows up to `min(hi, new capacity)`, preserve position/pending commits/DeltaNet state, clear cached binding plans. Payload = **64 KiB per history position FP16** (48.125 KiB with V8 INT8-V) for Qwen3.8-27B. An ownership cycle `Buffer → ndarray → Buffer` plus un-drained autorelease pools caused **~120 GiB leak over 160 transitions between 8K and 16K**; `.np` now returns a zero-copy view and pools are drained around `cai_buffer_create`/`cai_buffer_address`. These are the concrete buffer ownership rules to copy if you implement ANE KV in Python.

### 4.5 john-rocky/CoreML-LLM (negative results that matter for bit-identical)

- ANE+GPU chunk pipelining regressed **23–25%** (Mac Studio 32.8–33.2 → 24.9–25.5 tok/s) and **failed bit-exactness on `summary` at token 50** from fp16 rounding between ANE and GPU backends. Structural blocker: c4 consumes c3's `hidden_states_out`, leaving ~1 µs overlap against a ~16 ms GPU c3.
- IOSurface-backed `MLMultiArray` used as model input becomes **locked and cannot be reused as output backing**; CoreML silently allocates new buffers → **16× slowdown**.
- GPU prefill via CoreML `.cpuAndGPU` on Mac Studio: 2,697 ms vs 278 ms ANE (**9.7× slower**) with decode unchanged 33.0→32.9 tok/s. ANE/GPU kernel overlap 0.87–0.99; pure-ANE submissions from one process overlap only 0.02–0.06 (ANE driver serializes).
- fp32 upcast, accumulation-order control, canonical zeroing, and serial `decode_q1` replacements were all tried and **refuted** as fixes for verify-chunk fp16 drift; the mechanism was *semantic* (verify writes drafter proposals into KV before acceptance). Output-space tolerance and delayed KV write-through are the remaining options.

---

## 5. ANE + GPU co-execution (2026)

| Project | Hardware | Technique | Measured result | Python/MLX usable? |
|---|---|---|---|---|
| **FusionML** (arXiv 2607.22785) | M1, M2, M3 Pro, **M4 (24 GB)**, M4 Pro — 5 chips | Per-layer **CPU+GPU row split** of prefill linears behind an **eager `mx.eval` boundary**; contention-calibrated ρ≈0.25–0.40; no-regression gate with probed baseline arm | row-split matmul: 1.38× ready / **0.66× lazy** / 1.34× eager-boundary; block prefill **1.15–1.38×** on all actively-cooled chips; full 32-block unchanged; **Qwen2.5-7B TTFT 1.18–1.25×** with token-identical outputs in 5/6 configs (one near-tie divergence after 289 chars); decode exactly neutral (7.1→7.1 tok/s on M4). Max rel err ≤3e-3 FP16, ≤3.9e-3 through 32 blocks. ANE excluded: CoreML dispatch 20–24 ms fixed + 7 µs/row. | **Yes** — patches `mlx-lm` `nn.Linear`, code released at https://github.com/ommo007/FusionML |
| **MacPaw Research** (2026-10-10) | M2 Pro, M3 Pro, **M4 Pro**, macOS 27 Core AI | ANE-only; ANE prefill + GPU decode; co-prefill; split prefill (weights on ANE, attention on GPU) | M4 Pro: ANE decode 32.2 tok/s (plain), 98.1 chain, 136.9 tree vs GPU 8-bit 81.5 / 144.9 DSpark; prefill FFN INT8 at 256 tok = 10.3 TFLOP/s vs GPU fp16 5.3; **ANE prefill 1.9× faster than GPU, ~1/5 energy**; 2.77 s/request at 21 W vs 3.45 s at 48 W for GPU8+DSpark; 98.4% top-1 agreement vs bf16 (vs 99.0% MLX8); 16K split-decode 82 ms vs 108 ANE-only | No public code; M4 **Pro** only, Core AI macOS 27 |
| **SqueezeBits Yetter** (2025-08-26) | iPhone 15 Pro, iOS 18.6.1 | Stateless CoreML ANE prefill (KV as outputs) + MLX GPU decode; cache concatenated to 2 states (`[layers, kv_heads, max_len, head_dim]`) | Claims lower energy; **no handoff cost published**, figures as images | No (iPhone) |
| **AtomGradient hybrid-ane-mlx-bench** | M2 Ultra 192 GB, macOS 26.3 | 5 strategies; private-API ANE-LM sequential/batch + MLX decode | **CoreML `compute_units=ALL` routes to GPU, not ANE (ANE power ≈0 W)** — the "hybrid ANE" result is GPU-vs-GPU. Genuine ANE batch dispatch: **268 tok/s** prefill at 74 tokens (11.3× sequential), but sequential ANE-LM is 42 ms/token and TTFT 17.8 s. ANE prefill cuts GPU power 62.05 W → 0.22 W. | No (research bench, M2 Ultra) |
| **NPUMoE** (arXiv 2604.18788) | M2 Max 64 GB, M2 Ultra 192 GB | Static expert-capacity tiers + grouped expert execution + load-aware graph residency, CoreML ANE offload | Prefill latency **1.32–5.55×**, energy **1.81–7.37×**, CPU cycles 1.78–5.54× vs CoreML CPU/naive/ANEMLL; Qwen3-30B-A3B wall 0.572 s vs 1.139/1.144. **But** <1.1% accuracy loss via **10–20% token dropping** and 31.7–37.5% padding; GPU still 1.26–5.4× faster. | No code yet; M2 only; **not bit-identical** |
| **CrossInfer** (arXiv 2610.35309) | M4 Max, RK3588, QCS6490 | Versioned context separated from engine-local KV; import/reconstruct state so any engine can continue a stream | Qwen2.5-VL-3B: TTFT 4.5+121.3+100.5+1.0 ms; all answers match serial reference | No |
| **cadamcat/llms-on-apple-neural-engine** | M5 Pro, macOS 27, Core AI | Qwen3-4B FP16, graph sized per input | GPU faster at every input 500–16K (2.0–3.9× decode, 2.9–15.9× prefill); ANE saves component energy only on short prefill; at 1024 pos ANE ≈6,760 pos/s vs MLX ≈27,300 | No |
| **Outlier.host** (2026-07-23) | M1 Ultra | Negative result | CoreML puts all ops on ANE only to 8 layers (2.9 ms/layer); at 16 layers all 160 ops silently on CPU; GPU→ANE handoff only 0.84 ms | — |
| **BaseRT / uzu** (arXiv 2607.00501) | Apple Silicon | Native Metal; uzu routes GPU kernels + MPSGraph to reach ANE | Baseline context only; no base-M4 ANE numbers | GPU-first |

**Reconciling the dispatch-overhead contradiction (important):** FusionML's "ANE dispatch is 20–24 ms, useless at layer granularity" is a **CoreML** measurement; maderix/Orion/ANEForge/m0at all measure **60–130 µs** for private-API dispatch. These are different access paths. The user-facing conclusion is: *CoreML is disqualified for layer-granular hybrid execution; private-API ANE (e5rt / `_ANEInMemoryModel`) is not, by ~200×.*

---

## 6. Direct answers to the five key questions

### Q1 — What can be installed and run on an M4 16 GB Mac *today*?

| Project | GitHub | macOS | Entitlement / API | Fits 16 GB? | Verdict |
|---|---|---|---|---|---|
| ANEForge | https://github.com/sbryngelson/ANEForge | 14+ (verified 26.x) | **No entitlement**, private Espresso `e5rt` + `_ANEInMemoryModel`; SIP on | Yes for ≤4B fp16 or ≈8B int8/int4 (8B fp16 = 16 GB → 9 chunks, won't fit with OS); 27B MoE no | **Best fit. `pip install aneforge` + local dylib build.** |
| Orion | https://github.com/mechramc/Orion | 15+ | Private `_ANEClient`/`_ANECompiler`; entitlement not stated (inferred unentitled) | Yes (124M/110M only) | Installable but tiny models, no Python API |
| maderix/ANE | https://github.com/maderix/ANE | 15+ | Private APIs, unentitled | Yes (0.6B) | Research probes + GPU↔ANE IOSurface demo |
| m0at/ANEtransformers | https://github.com/m0at/ANEtransformers | 15+ | Private APIs, unentitled | **Yes** — 7B Q4_0: 8.3 GB steady / 8.6 GB peak | Only 7B+ direct-ANE stack that plausibly fits 16 GB; C/ObjC, M5-tested |
| ANEMLL | https://github.com/Anemll/Anemll | 15+ | **CoreML** (public), native arm64 Python 3.9–3.11 | 16 GB min; 32 GB for 8B | CoreML cannot force ANE; silent CPU fallback at depth |
| anemll-forge | https://github.com/Anemll/anemll-forge | **27 + Core AI** | Core AI Swift bridge | 32 GB recommended | **No** |
| CoreML-LLM | https://github.com/john-rocky/CoreML-LLM | 15+ | CoreML | iPhone/Mac Studio | No code runtime for us; read the negative results |
| FusionML | https://github.com/ommo007/FusionML | stock MLX | Public APIs only | Yes (M4 24 GB tested; should run 16 GB) | **No entitlement, no private API, token-identical** |
| AtomGradient | https://github.com/AtomGradient/hybrid-ane-mlx-bench | 26.3 | CoreML + private ANE-LM | M2 Ultra | No |

All of the above URLs were confirmed live. Install steps are in the papers' "Building"/"Install" sections quoted above.

### Q2 — Python-callable API wrapping one projection with baked weights + IOSurface I/O?

**Only ANEForge.** Concrete path:

```python
import aneforge as af
Wq = ...                                   # numpy fp16 [out, in]
x  = af.input((1, C, 1, S))                 # fixed S (pad/bucket)
y  = x @ Wq                                 # single projection; W baked into blob
qproj = af.compile(y, compress=None, target="h16s")   # M4 = H16
q = qproj(x_fp16)                           # ~80 us eval after one-time compile
```

For multi-output/resident variants use `af.compile_multi([q,k,v])` and `Program.share_buffer(...)`. For a native caller, drive `ane_e5rt_program_compile/set_input_fp16/execute/get_output_fp16` from the C ABI. The open gap is the **MLX side**: no documented zero-copy `IOSurface/MTLBuffer → mx.array` import. mlx-lm expects its own `KVCache` objects; ANE KV is a flat FP16 `[1,C,1,S]` IOSurface. Copying a 7B KV at 4K is on the order of hundreds of MB (64 KiB/position for 27B-class), so a per-step handoff is expensive; a *prefill-only* handoff (once per prompt) is the tractable shape.

### Q3 — Measured speedups over MLX GPU for S≥256 prefill matmuls on base M4 / M4 Air?

**None found. Explicitly.** The closest evidence, with its exact scope:

- ANEForge ROOFLINES base M4 (Mac16,12, 16 GB, macOS 26.2, commit 23ef10a6c370): **peak fp16 GEMM 5.4 TF/s**, streaming bandwidth 7.7 GB/s, ridge 708 FLOP/byte, peak perf/W 1,112 GF/s/W. No MLX column in that dataset. Its "decode @b1" cell is intentionally blank for M4 because that run used a ~4× heavier model.
- ANEForge paper's GEMM/attention speedups are M5 Pro only; the device-map table has M1 Max / M2 Pro / M5 Pro, no M4.
- m0at (base M5): ANE prefill 76.6 tok/s vs their Metal 58.0 (1.3×) but **MLX ~200+ tok/s**.
- MacPaw: **M4 Pro** ANE prefill 1.9× faster than GPU — but it's Core AI on macOS 27, not base M4 and not private-API e5rt.
- ANEMLL author's community measurement: base M4 ANE weight-read ≈64 GB/s vs system ≈120 GB/s (53%).
- maderix M4 S=256: GPU prefill 6.7 ms + ANE decode 1.9 ms (Stories110M).
- FusionML on M4: 1.15–1.38× prefill from **CPU+GPU**, with ANE explicitly excluded.

Bottom line: on base M4 you should assume **no proven latency win over MLX GPU for prefill**; the credible wins are energy (ANE) and token-identical TTFT (FusionML). Validate on-device before designing around an ANE speedup.

### Q4 — Numerical accuracy vs FP32/BF16

See tables in §2.6, §3.2, §4.2, §5. Condensed:

| Source | Max abs / rel error | Cosine | Token agreement |
|---|---|---|---|
| ANEForge ResNet-18 | relerr 2.3e-3 | 1.0000 | top-1 match |
| ANEForge attention rewrite | **0.0000** | 1.0000 | — |
| ANEForge GPT-2 medium | — | — | **16/16 greedy vs HF fp32** (with tie-margin fallback) |
| ANEForge Qwen3-0.6B int8 | byte-identical to fp16 | ~1.0 | identical argmax |
| Orion GPT-2 124M | max logit err 0.073 | — | **100% top-1, exact 64-token greedy** |
| m0at single layer | max 0.026, mean 0.0007 | — | text matches BLAS |
| m0at full 32-layer prefill | max 0.61 | 0.998 | text matches BLAS |
| FusionML split | ≤3e-3 (≤3.9e-3 @32 blocks) | — | token-identical 5/6; 1 divergence after 289 chars |
| CoreML-LLM ANE+GPU | fp16 rounding mismatch | — | **fails bit-exact at token 50** |
| MacPaw ANE | — | — | 98.4% top-1 vs bf16 |
| NPUMoE | — | — | <1.1% accuracy loss with 10–20% token drop — **not bit-identical** |
| ANEMLL Forge 0.2 | KL-512 0.052 vs BF16 | — | top-1 91.9% — not bit-identical |

### Q5 — IOSurface buffer ownership rules (consolidated)

1. **Layout:** fp16 `[1, C, 1, S]`, channel-first, data[c*S+s]; last axis contiguous and 64-byte aligned (Apple's own transformer guide). Wrong layout gives plausible-but-wrong output (cosine 0.02).
2. **Minimum allocation:** ~49,152 bytes; S=1 compiles but fails at eval (`0x1d`), min decode bucket S=16 (fp32) / 32 (fp16).
3. **Uniform sizing + alphabetical ordering:** all multi-input and multi-output surfaces must have the same byte allocation, and are bound in **alphabetical order of MIL variable/parameter name**; over-allocated surfaces are read packed from byte 0, ignoring nominal dims.
4. **Weights:** baked at compile time via BLOBFILE (offset `uint64(64)`); runtime `weightsBuffer` override is dead on `_ANEInMemoryModel`; max 16 weight tensors per program (bias consumes a slot); ~119 compiles/process; daemon cache is per-PID, lost on reboot.
5. **Residency:** `e5rt_io_port_bind_buffer_object` / `ane_e5rt_program_share_buffer` aliases output→input port; seed once from host, loop `execute_sync`, read back at checkpoints. Works unentitled; KV and optimizer state stay on silicon.
6. **Chaining:** `_ANEChainingRequest`/Path B and direct IOSurface ownership are gated by `com.apple.aned.private.allow`. The e5rt `execute_multi` reproduction is performance-neutral. CoreML's sanctioned alternative is `MLState` (macOS 15+).
7. **Processes:** compile and dispatch must be in the same process; cross-process reuse needs same-codesign-identity `posix_spawn` (fork() SIGSEGVs); IOSurface-by-Mach-port handoff ~70 µs.
8. **CoreML-specific traps:** an IOSurface-backed `MLMultiArray` consumed as input becomes locked and cannot back a later output (→16× slowdown); only 2 outputs per program; planet of depth → full CPU fallback past ~8–16 layers.
9. **Python-specific traps (Forge):** drain autorelease pools around `cai_buffer_create`/`cai_buffer_address`; break `Buffer → ndarray → Buffer` cycles or a 160-step KV resize leaks ~120 GiB; `.np` should return a zero-copy view.
10. **Copy cost:** 64 KiB/history position FP16 (27B-class); 8K ≈512 MiB, 64K ≈4 GiB — a handoff is milliseconds to >100 ms even at raw memcpy speed.

---

## ACTIONABLE FOR VEYSTRIX 16 GB

**1. Adopt FusionML's eager-boundary CPU+GPU row split as the default prefill path — first step: pip-install stock `mlx-lm`, clone https://github.com/ommo007/FusionML, and reproduce the single row-split matmul probe (2048×1600×1600×6400 fp16) on the M4 with `mx.eval` inserted before the split; require adjacent precision-matched FP16 runs and ≥1.10× with identical tokens before touching the engine.** Evidence: 1.38× ready / 0.66× lazy / 1.34× eager-boundary mechanism; 1.15–1.38× block prefill and 1.18–1.25× TTFT on real Qwen2.5-7B through stock MLX-LM, token-identical in 5/6 configs, decode neutral; replicated on an M4. This is the only technique here that is both Python-native and measured *on an M4* with a bit-identical objective. It needs no private API or entitlement and therefore carries none of the ANE's numerical/placement risk.

**2. Add ANEForge's fp16-head tie-margin fallback as a general bit-exactness guard, and keep any ANE work acceptance-gated — first step: `pip install aneforge`, build the dylib, run `examples/llm_prefill.py` with Qwen3-0.6B (fits 16 GB), then implement the same "if top-1/top-2 gap < 3e-3 × max logit, recompute logits on the CPU/GPU fp32 head" rule in Veystrix's sampler and measure its trigger rate and exact HF agreement on 1,000 decode tokens.** Evidence: ANEForge reports the fp16 head cuts gpt2-medium decode 10.8→3.4 ms/tok but a tie margin of 3e-3 (≈6× measured fp16 head error, firing on ~2–3% of tokens) restores exact greedy equality with host fp32 and HF while keeping ANE speed for the rest; token-level exactness otherwise fails at close ties (FusionML's one divergence; CoreML-LLM's token-50 bit-exact failure). This gives a principled way to use fast low-precision kernels *and* satisfy a bit-identical contract.

**3. Prototype exactly one ANE-island matmul for prefill (not MoE routing, not KV handoff) and treat it as optional — first step: on the 16 GB M4, compile a single fixed-S `af.input((1,C,1,S)) → matmul(Wq)` program with `target="h16s"` and `compress=None`, time it against `mx.matmul` at S=256/512/1024 with warm handles, and abort if it is not ≥1.2× faster or if outputs differ from fp32 beyond your token-equality test; only then consider `compile_multi` + `share_buffer`.** Evidence: ANEForge's primitive API matches this shape exactly, e5rt dispatch is ~80 µs, fusion of QKV+attn+O is 3.8–5.3× at S≤256 and bit-identical, and the base M4 roofline shows 5.4 TF/s peak fp16 GEMM with all four compressed formats streaming natively on A15+. But no source measures an ANE-over-MLX prefill win on base M4, m0at's M5 result shows MLX ~200+ vs ANE 76.6 tok/s, and ANEForge itself notes the native fused attention is avoided for prefill because its per-layer graph cut dominates (matmul path ~170× faster). Do **not** pursue NPUMoE-style capacity tiers/token dropping (10–20% token drop breaks bit-identity), cross-engine KV handoff (no measured cost; 64 KiB/position; bit-exact failures documented), or anemll-forge (M5/M6, macOS 27, 32 GB).
