# Can the ANE speed up transformer **prefill** with bit-identical output?

Research date: 2026-11-17. Target: Veystrix, 4-bit MoE LLM on MLX, base 16 GB M4 Mac.
Question: is there a concrete ANE prefill path that preserves **bit-identical output tokens** (same token IDs and BF16 rounding as the MLX reference)?

## Bottom line (read this first)

1. **ANEForge's "fp16 head with tie-margin fallback" is real, but it is (a) in the docs and released code, not in the paper; (b) a *decode* feature, not a *prefill* feature; (c) a *greedy-argmax* exactness guard, not a bitwise-logit guard; and (d) corrects only the final `lm_head`, not any intermediate layer.** `LlamaPrefill.prefill()` in the shipped source calls `self._logits(...)` directly and **never calls** the guarded `_decode_logits()`. Porting it to prefill is trivial but buys almost nothing, because prefill calls the head **once per prompt**, not once per token.
2. **There is no known principled scheme that makes arbitrary ANE fp16 intermediate prefill layers bit-identical to an MLX BF16 reference with a sparse "recompute when uncertain" fallback.** Per-step error bounds exist (runtime-certified attention, tritify, ExBLAS) but they certify (i) a norm bound, (ii) a discrete decision, or (iii) reproducibility to *their own* exact/correctly-rounded reference — not bitwise equality to a specific MLX BF16 kernel's reduction tree. A nonzero fp16 error in layer 1 is present on essentially every input, so "recompute whenever the bound is nonzero" degenerates to "recompute everything."
3. **ANEForge's own use of the words "bit-identical"/"byte-identical" means "two lowerings of the same fp16 graph produce the same bytes," not "matches a BF16/fp32 framework reference bitwise."** This is an easy and consequential misreading of the paper.
4. Honest answer to Q5: **no ANE *intermediate-layer* prefill feature is bit-identical.** The only defensible feature is **head-only ANE prefill with a ported tie-margin + a per-prompt exactness gate**, and even that delivers only *token-ID* identity, with expected prefill speedup ~1.0–1.2× on short prompts/large vocab and ~1.0× on long prompts. For literal BF16 bit identity, the only route is an offline pinned-configuration conformance harness (empirical gate, not a theorem).

---

## Q1. What exactly is the ANEForge tie-margin fallback?

### Where the claim lives

- **Paper (arXiv:2606.17090v2, "ANEForge: Python for direct computation on the Apple Neural Engine", Spencer H. Bryngelson, UT Austin):** the tie-margin fallback is **not mentioned**. A full-text scan of the HTML for `tie|fallback|margin|bit-exact|bitwise|identical|fp32 head` yields only 10 hits, all about CoreML's CPU fallback, the byte-identical default compression path, and the "attention rewrite ... verified bit-identical to the unoptimized graph" (§5.5). §4 explicitly says of the LLM loaders: **"We do not evaluate these loaders here."** So the headline exactness claim is **not a peer-reviewed/paper claim**.
- **Repo docs (`docs/llm.md`, "lm_head on the ANE" section):** the source of the brief's claim. Exact quote:

  > "The head runs fp16, so a greedy argmax could flip against the host fp32 on a close tie. A tie-margin fallback recomputes a step on the host fp32 head when the top-2 gap is under `lmhead_tie_margin` (default `3e-3`) times the max logit magnitude -- about 6x the measured fp16-head error, firing on ~2-3% of tokens. Greedy then matches the host (and Hugging Face) exactly while the rest keep ANE speed; the recompute builds `_lmT` only if a tie fires, and sampling stays on the ANE (softmax shifts by ~2e-3, unaffected)."

- **Released code (`aneforge/llm.py`):** the mechanism **is implemented**, not just described. Exact lines:

  - `llm.py:279-281`:
    ```python
    self._lmhead_compress: str | None = None  # head always fp16; quantizing it degrades logits more than
                                              # an intermediate layer (no residual to absorb the error)
    self.lmhead_tie_margin = 3e-3  # greedy near-tie (top-2 gap < margin*max|logit|) recomputes on host fp32
    ```
  - `llm.py:395-404` — the whole fallback:
    ```python
    def _decode_logits(self, hidden_row, use_ane, greedy):
        """Step logits with the greedy tie-margin fallback: an ANE-head near-tie recomputes on the host fp32
        head, so greedy matches the host exactly (see `lmhead_tie_margin`). Sampling stays on the ANE."""
        use_ane = self.ane_lm_head if use_ane is None else bool(use_ane)
        logits = self._logits(hidden_row, use_ane)
        if greedy and use_ane and self._lmh is not None:      # only when the ANE head, not a host fallback, ran
            p = np.partition(logits, -2)
            if p[-1] - p[-2] < self.lmhead_tie_margin * (float(np.abs(logits).max()) + 1e-9):
                logits = self._logits(hidden_row, ane=False)  # host fp32 recompute for this near-tie step
        return logits
    ```
  - `llm.py:349-355`: the fallback target is `hidden_row @ lm` where `lm = np.ascontiguousarray(np.asarray(self.w["lm_head"]).T, np.float32)` — a **host fp32 matmul over the ANE-produced hidden row**.
  - `llm.py:387-393`: `_ane_logits` casts the hidden row to fp16 and concatenates fp32-converted tile outputs.

### Exact mechanism, threshold, trigger rate, result

| Item | Exact value / behavior |
|---|---|
| Fast path | `lm_head` as tiled fp16 ANE `compile_multi` (`llm.py:357-385`); vocab tiled to family max tensor dim; 1.2 GB head-byte budget, else host. |
| Trigger predicate | `greedy and use_ane and ANE head program exists`, then `top1(logits) - top2(logits) < 3e-3 * (max|logits| + 1e-9)`. Uses **logit gaps**, not softmax probabilities (`np.partition(logits, -2)`). |
| Threshold | `lmhead_tie_margin = 3e-3`, i.e. **0.3% of max |logit|**; docs say ≈ **6× the measured fp16-head error** (~5e-4 relative). |
| Fallback action | Recompute **only the head** on host fp32 (`_logits(..., ane=False)`), lazily materializing the fp32 `_lmT` transpose (≈206 MB GPT-2, ≈2.5 GB for a 151936×4096 head) on the first tie. |
| Trigger rate | **~2–3% of tokens** → ~97–98% keep ANE speed (matches the brief's "97–98%"). |
| Reported exactness | "Greedy then matches the host (and Hugging Face) exactly" — **greedy token-ID equality**, empirically validated (README: GPT-2 medium "16/16 greedy match vs HF fp32"; `docs/llm.md` "identical argmax"). |
| Not covered | Sampling (`temperature > 0`) explicitly stays on ANE; docs say softmax shifts ~2e-3 and is "unaffected". |
| Speed | ANE head vs host head, M2 Pro, median of 3: gpt2-medium 10.8 → 3.4 ms/tok (3.2×); Qwen3-0.6B 30.1 → 8.2 ms/tok (3.7×). End-to-end decode 39.6 → 59.4 tok/s and 17.8 → 29.3 tok/s (~1.5–1.65×). |

### Three load-bearing caveats that the brief's framing misses

1. **It is decode-only as shipped.** `llm.py:421-423`:
   ```python
   def prefill(self, token_ids):
       """Prefill `token_ids` and return the next-token logits [1, vocab] (the transformer runs on the ANE)."""
       return self._logits(self._hidden(token_ids)[-1])[None]
   ```
   `prefill()` calls `_logits`, **not** `_decode_logits`; there is no tie-margin guard on `model.prefill(ids)`. The guarded path is used by `generate()` (`llm.py:625`) and `_prefill_seed()` (`llm.py:544`) — the latter seeds the decode KV cache after a token-by-token prefill loop, so the *first generated token* is guarded, but the public prefill API is not. **The brief's "fp16 head with tie-margin fallback" does not currently apply to prefill at all.**
2. **"Exact" is argmax equality, not bitwise logits.** On the ~97–98% of steps with no tie, the returned logits are ANE fp16 values, *not* the fp32 values — docs measure a ~5e-4 relative head error and a ~2e-3 softmax shift. If Veystrix's requirement is "the BF16-rounded logit bits match the reference," the tie-margin does **not** deliver it even on the head, let alone intermediate layers. `docs/llm.md` itself says "matches the host", where the host head is applied to the **already-fp16 ANE hidden state** (`_hidden` at `llm.py:326-336` returns `np.asarray(net(emb.astype(np.float16))).astype(np.float32)`). So body-error-induced flips cannot be repaired by this fallback; the observed 16/16 GPT-2 match is an empirical result on those tests, not a theorem.
3. **"Bit-identical" in the paper means something else.** §5.5 Table 5: the attention rewrite's tuned output "matched the unoptimized output to a relative max-abs difference of 0.0000" — i.e. **two ANE fp16 lowerings agreed exactly**. §4: "the default path's program is byte-identical to plain half precision." `docs/aneforge-api.md:225,232,244`: `opt='routes'` is "the **lossless** default route pass (cost-model-driven, never changes numerics)", `opt=0` is "the byte-identical historical path", and `compress=` "always takes the byte-identical (`opt=0`) lowering". None of these statements compares to a BF16/fp32 framework reference. This is the single most likely source of a false "ANE is bit-exact" conclusion.

### Arithmetic: why the guard is cheap in decode and nearly irrelevant in prefill

- Head-only cost model (M2 Pro numbers, Qwen3-0.6B, vocab 151936): host 30.1 ms, ANE 8.2 ms; tie rate 2–3% ⇒ expected per-token head cost `0.975×8.2 + 0.025×30.1 ≈ 8.75 ms` → `30.1/8.75 ≈ 3.4×` on the head portion, consistent with the reported 1.65× end-to-end decode gain because the body/decode overhead is not removed.
- Prefill calls the head **once** per prompt. At ANEForge's reported ~8,600 prompt-tok/s for Qwen3-0.6B on M5 Pro, a 1000-token prefill is ≈116 ms; saving ≈21 ms on the single head call would be ≈1.22× — but this mixes an M5-Pro prefill rate with M2-Pro head timings and is **illustrative only**. At 10k tokens (≈1.16 s) the same saving is ≈1.018×. For a 4-bit MoE with routing/scatter overhead the head is an even smaller fraction, so **expected prefill speedup from a head-only tie-margin port is ~1.0× in the general case**.

---

## Q2. Can "recompute in fp32 when uncertain" work for **intermediate** prefill layers?

### Why the final head is special (and why that does not generalize)

The final head's certificate is a **decision margin**: `argmax` is a discrete choice among |V| logits, and if the top-2 gap exceeds the worst-case numerical perturbation, the decision is provably invariant under that perturbation. The guard needs only one scalar per step, costs one `partition`, and the fallback is one matmul.

An intermediate layer has **no such discrete decision**. Its output is a real-valued tensor consumed by RMSNorm → QKV → RoPE → GQA softmax → SwiGLU → residual → next layer. To guarantee that a perturbation `δx_l` at layer `l` cannot flip the final token you must bound

```
δlogits ≤ J_L · J_{L-1} · ... · J_{l+1} · δx_l
```

where each `J` is the local Lipschitz operator of a nonlinear layer (softmax, RMSNorm with its `1/rms` factor, SwiGLU, MoE router top-k which is **discontinuous**). In the worst case the product grows as ∏‖J_i‖, and at a top-k router the map is not Lipschitz at all (arbitrarily small input changes swap experts). Any bound loose enough to be cheap is loose enough to be useless; any bound tight enough to certify the final argmax requires propagating intervals/symbolic error through all remaining layers — effectively exact arithmetic. Moreover `δx_l ≠ 0` for essentially **every** fp16 matmul, so "recompute whenever the bound is nonzero" fires on ~100% of layers. This is why no production system ships an intermediate-layer bit-exact fallback.

### What the closest research actually guarantees (none of it is bitwise-to-MLX-BF16)

- **Runtime-Certified Bounded-Error Quantized Attention (arXiv:2605.20868, Dean Calver, May 2026)** — the strongest per-layer error-bound + fallback work found. INT8 keys / INT4 values in VRAM, FP16 originals in pinned system RAM, two-term decomposition with runtime-computable bounds:
  - `E_val = ‖Σ a_t(V̂_t − V_t)‖₂ ≤ η` and blockwise `E_val ≤ Σ_b ρ_b η_b` (Thms 4.1/4.2; η ≈ 0.05 mean, 0.18 max for INT4 g=16).
  - `E_key ≤ 2·V_max·TV(a,a′)` with `TV(a,a′) = ½Σ|a_t − a′_t|` bounded via softmax total-variation (Thm 4.3); key-tail mass `≤ e^{2Δ}(1−τ_cov)`, implemented conservatively as `e^{3Δ}` with `τ_cov = 0.995`.
  - Four-rung fallback ladder ending in `torch.scaled_dot_product_attention` over all-FP16 KV (the production Dense baseline `O_dense`).
  - **Explicit limitation (abstract and §1):** "The certification is local (per-head, per-step) and does not guarantee end-to-end model correctness." All formal bounds certify `‖O_quant − O_ref‖₂` where `O_ref` is **the same certified kernel with FP16 KV**; the arithmetic-path gap `‖O_ref − O_dense‖₂` is merely "characterised in §9.10". So even at the layer where it applies, the certificate is not to the reference kernel used by MLX, and it is a norm bound, not bitwise equality. Also the goal is explicitly "not raw speedups"; tier-2 retention costs ≈16 GB at 128K — incompatible with a 16 GB M4.
- **tritify (arXiv:2611.13730, 10 Nov 2026, Pelloran Inc.)** — exactly-integer forward path for decoder LMs, MoE decoders, SAM 2.1; reproduces SHA-256 of outputs across CPU ISAs and CPU/GPU, including distributed 106B MoE split across three machines. Crucially its own abstract says: **"We do not claim a per-device speed-up, bit-equality of floating-valued intermediates, or confidentiality."** Claim (2) is "decision fidelity ... reproduces the float reference's decision on the models and inputs tested, with the numeric exceptions measured" — i.e. token decisions with **measured exceptions**, per-layer fidelity through 16 layers shown empirically, "no fidelity cliff". This is the closest thing to a general bit-exact scheme, and it is (a) integer-only, not ANE fp16, (b) decision-exact with exceptions, not bitwise-to-MLX-BF16, and (c) no speedup.
- **ExBLAS / ReproBLAS (Iakymchuk et al.)** — correctly rounded, bitwise reproducible dot products via 2098-bit fixed-point **superaccumulators** and multi-level FPE reduction. Gives "bit-wise identical results from run-to-run on the same input data on the same or different architectures" — but to the **correctly rounded** result, not to whatever reduction tree MLX happened to use. Not available on ANE; large superaccumulator overhead.
- **Compensated summation (Kahan / 2Sum / FMA residuals)** — improves accuracy toward correctly rounded sums; error bounds are effectively independent of `n` when accumulating in double the precision (`|E_n| ≤ [2ε + O(nε²)]Σ|x_i|`), but it still does **not** reproduce a specific kernel's rounding order bit-for-bit. Useful as a building block, not a certificate.

**Conclusion for Q2:** No. A sparse "recompute in fp32 when uncertain" scheme exists for the final *discrete* decision; for intermediate *continuous* layers it either fires always or certifies nothing useful. The published per-layer bounds certify a norm vs a same-kernel reference or a decision on tested inputs, never bitwise equality to an MLX BF16 reference.

---

## Q3. Production engines: fast low-precision path + exactness guard?

- **ONNX Runtime `onnxruntime-ep-mlx` (`docs/OP_ARCHITECTURE.md`, "MatMulInteger", "GroupedMatMul and direct QDQ products", "FusedMatMul")** — the most concrete production-style technique found, on MLX/Metal rather than ANE:
  - For `K ≤ 256`, power-of-two static scales in `[2^-40, 2^40]`, scale/zero-point layouts constant along K: `MatMulInteger`/QDQ uses an **exact int32 dot**; each centered integer is in `[-255,255]`, every partial absolute sum `≤ 16,646,400 < 2^24`, so "DQ values, products and all partial sums are exact normal float32 values under the exponent guard" and both final binary scalings are exact. This is a genuine exact fast path — with a **tight, provable precondition**.
  - General path: float32 DQ in registers + SIMD float32 accumulation, with **FMA residuals and TwoSum error compensation** (including the 32-lane reduction) before rounding the dot to fp32; bias added after rounding. Exceptional lanes fall back to the original FMA/SIMD sum.
  - But the doc is explicit: this is "**not integer-only ALU or broadly bit-exact**"; "Original NumPy float32 reductions remain named disagreement diagnostics: neither their reduction order nor a fitted output is the oracle"; "MLX has no native integer GEMM primitive, so this is a correctness fallback, not an accelerated integer matrix product." `FusedMatMul` likewise uses "Narrow scaling [with] dtype-rounded alpha and FP32 product/scaling before the final output cast, avoiding an intermediate fp16/bf16 result" — a precision-preservation technique, not a bit-identity-to-MLX-ref guarantee.
- **Core ML / ONNX Runtime generally:** no production exactness guard that returns bit-identical outputs relative to a different framework. ORT's `auto_convert_mixed_precision(model, feed_dict, rtol=0.01, atol=0.001, keep_io_types=True)` is explicitly tolerance-based. Core ML's `numericCompute`/`computePrecision` and device selection do not report which unit ran a call (ANEForge §1), and the public 4-bit palettized paths on ANE are documented as producing **wrong answers** on real models (cadamcat, M5 Pro, cited in `ane_llm_evidence.md:60`). Compensated summation on GPU (Kahan/two-sum kernels) is a numerical-quality technique, not a cross-engine bit-identity contract.
- **ANEForge's `af.tune`** is the closest in spirit: `docs/aneforge-api.md:225` "lossless default route pass (cost-model-driven, never changes numerics)"; paper §5.5 "validates each against the unoptimized baseline, returning the lossless selection by default"; `af.tune(out, atol=0.1)` admits lossy int8 only with margin. Again: lossless **relative to the unoptimized ANE graph**, not relative to MLX.

No engine was found that offers "fast fp16 path + guard ⇒ bitwise identical to a different runtime's BF16 kernel" for arbitrary layers. The invariant always has to be defined against a *specific, fixed* reference path.

---

## Q4. Does the bit-identical requirement make sense for a speed toggle?

- The literature separates **decision fidelity** (same token IDs) from **bitwise identity**. Speculative decoding is "exact" in the sense of same tokens as greedy (ANEForge: `verify(K) ~ verify(1)`, 2.28× on Qwen3-8B, "output is exact: the same tokens plain greedy decode would produce"), not same logit bits. tritify explicitly claims "decision fidelity ... with the numeric exceptions measured" and denies bit-equality of float intermediates. ANEForge's tie-margin is a **token-level gate**, not a bitwise gate.
- A bitwise gate relative to MLX BF16 would require reproducing MLX's exact reduction order, cast points, and rounding on the ANE fp16 datapath — which is a different ISA with different accumulation and no BF16 compute. There is no mechanism on ANE to request "round exactly like MLX's Metal BF16 kernel." So a *general* bitwise toggle is not merely unimplemented; it is ill-defined without pinning a reference kernel and a fixed input shape.
- For a **fixed** prompt/model/shape, however, an empirical bitwise conformance check is well-defined and cheap: run both paths, compare final-logit bit patterns, and ship the fast path only under a gate keyed by (model hash, prompt length bucket, OS/ANE compiler version). This is a *pinned configuration*, not a general theorem — see Idea 2 below.
- Practical consequence: Veystrix should decide which contract it actually needs. "Same token IDs under greedy" is attainable for the head-only path with the tie-margin (empirically). "Same BF16 logit bits" is attainable only by running the reference path or by pinning a configuration verified offline on representative prompts; there is no ANE technique that delivers it generally.

---

## Q5. One ANE prefill feature, expected speedup, first cheap test — and the honest verdict

**Verdict: there is no bit-identical ANE intermediate-layer prefill feature.** The tie-margin certificate is fundamentally a final-decision margin; it does not transfer to attention projections or FFN/MLP projections, and it does not produce bitwise logits even at the head. Anyone claiming otherwise is conflating "two ANE lowerings are byte-identical" with "ANE output equals the MLX BF16 reference bit-for-bit."

The single feature that is technically defensible is **head-only ANE prefill with a ported tie-margin and an explicit token-equality gate**:
- Change `prefill()` to call `_decode_logits(..., greedy=True)` so the existing `3e-3` guard applies to the one head matmul. Keep everything else on MLX; no intermediate layers on ANE.
- Expected speedup: **~1.0× for normal-length prompts**, up to ~1.2× only for very short prompts with very large vocab (single head call amortized over few prompt tokens); the 2–3% host recomputes cost a few ms each. Not a meaningful TTFT win for Veystrix's MoE prefill.
- Even then the guarantee is **greedy token-ID equality vs the host head on the same hidden state**, not bitwise BF16 logits, and it must be revalidated per model on M4 (ANEForge is "verified on M5 Pro and M1 Max"; the repo has an M4 roofline JSON but no M4 validation claim; the private `e5rt`/`aned` surface is unversioned).
- First cheap offline test: for 100 fixed prompts at 2–3 length buckets, dump (a) MLX reference final logits in BF16, (b) ANE-body + host-fp32-head logits, (c) ANE-body + ANE-fp16-head logits + tie-margin; record token-ID mismatch rate and the top-2 gap distribution near the `3e-3 × max|logit|` threshold. Confirm the fallback firing rate is 2–3% and that (c) matches (b) in argmax on all 100; separately count bitwise BF16-logit mismatches (expected nonzero on ~97–98%).

---

## Sources (with locators)

- ANEForge paper: https://arxiv.org/abs/2606.17090 ; HTML v2 https://arxiv.org/html/2606.17090v2 (§4 loaders not evaluated; §5.5 Table 5 bit-identical rewrite; §1 CoreML fallback).
- ANEForge repo: https://github.com/sbryngelson/ANEForge — `aneforge/llm.py` lines 263-266, 279-281, 326-336, 338-355, 357-404, 421-423, 544, 625; `docs/llm.md` "lm_head on the ANE"; `docs/aneforge-api.md:225,232,244`; README "Language models"/"Measured".
- Runtime-Certified Bounded-Error Quantized Attention: https://arxiv.org/abs/2605.20868 ; HTML https://arxiv.org/html/2605.20868v1 (§3.4 fallback ladder, §4.1–4.5 bounds, §1 "does not guarantee end-to-end model correctness").
- tritify: https://arxiv.org/abs/2611.13730 (abstract: exact integer path, SHA-256 reproducibility, "no per-device speed-up, bit-equality of floating-valued intermediates").
- ExBLAS: https://hal.science/hal-01202396v3/document ; https://github.com/riakymch/exblas (superaccumulator, correctly rounded, bitwise reproducible).
- ONNX Runtime MLX EP: https://github.com/onnxruntime/onnxruntime-ep-mlx/blob/main/docs/OP_ARCHITECTURE.md (MatMulInteger exact-int32 precondition; QDQ FMA+TwoSum; "not broadly bit-exact"; FusedMatMul FP32 scaling).
- Compensated summation background: https://en.wikipedia.org/wiki/Kahan_summation_algorithm (error bound `|E_n| ≤ [2ε + O(nε²)]Σ|x_i|`).
- Veystrix prior work (context, not external): `research_notes/Apple Neural Engine integration/ane_llm_evidence.md`, `hybrid_feasibility.md` (16 GB M4 MBA, SSD-bound MoE decode ~385 ms/token; ANE cannot touch that bottleneck; private-API and 4-bit toolchain fragility).

---

## ACTIONABLE FOR VEYSTRIX 16 GB

1. **Port the ANEForge tie-margin to `prefill()` as a token-exact head-only experiment — do NOT put any transformer layer on the ANE.** Evidence: `aneforge/llm.py:395-404` implements the `3e-3 × max|logit|` guard firing on 2–3% of tokens; `docs/llm.md` reports ~3.2–3.7× head speedup and end-to-end decode 1.5–1.65×, and explicitly says quantizing the head is worse than an intermediate layer because "no residual to absorb the error". But `prefill()` (line 421-423) bypasses the guard, and the head is called once per prompt, so expected prefill speedup is ~1.0× (best case ~1.2× for short prompts + large vocab). First offline step: on the M4, for 100 fixed prompts × 2 length buckets, compare MLX-reference logits, host-fp32-head logits, and ANE-fp16-head + tie-margin logits; measure token-ID mismatch rate, tie-fire rate, and bitwise BF16-logit mismatch rate. If token IDs are 100% identical but BF16 bits are not, label this feature "token-exact only" in Veystrix's docs and stop.
2. **If and only if "bit-identical" is literal, replace the general ANE idea with an offline bitwise-conformance gate over pinned configurations.** Evidence: no engine provides cross-runtime bitwise identity (ORT-MLX EP says "not broadly bit-exact"; tritify denies bit-equality of float intermediates and claims no speedup; ExBLAS certifies to correctly-rounded, not to MLX's reduction tree). A pinned (model hash, prompt-length bucket, OS/ANE-compiler version) configuration can be *verified* bitwise offline and then gated; this is empirical, not a proof, and any untested shape must fall back to MLX. First offline step: dump raw BF16 logit bit patterns from the MLX reference and from ANEForge prefill for 100 fixed prompts of each target length; compute per-position bitwise mismatch counts and the minimum top-2 logit gap. If a single configuration shows zero bitwise mismatches on the pin set, ship that configuration behind the gate; otherwise the honest conclusion is "none is bit-identical — the tie-margin only works at the final head, and only for argmax."
