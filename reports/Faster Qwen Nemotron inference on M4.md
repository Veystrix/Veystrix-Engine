# Faster Qwen Nemotron inference on M4

**Research question:** Find every credible way to raise Qwen decode from ~2.3 to **5+ tok/s** and Nemotron decode from ~6.7 to **15+ tok/s** on the 16 GB M4, plus faster prefill, with **no quality loss** (bit-identical token digests, BF16 rounding boundaries, Q4 affine G32 weights, exact top-10/512 routing). Stop after 10+ methods.

**Scope note:** This is research only. **No hardware benchmark was run for this report** (owner decision 2026-11-17). Every number below is a published measurement from a cited source, with the exact-model / exact-hardware evidence separated from analogous evidence. The owner's rule "validate on 16 GB first" still governs implementation.

---

## 1. Executive summary

- **10+ concrete methods were found.** Ranked in §4. The single most important diagnostic: on the *same model* under an MLX M4 profile, per-token time is **41% Python / `mx.eval` synchronization, 29% SSD reads, 27% GPU** — so Veystrix's decode is **at least as sync-bound as it is SSD-bound**. Removing the sync overhead is the highest-value bit-exact change and is **not** in `DO_NOT_IMPLEMENT.md`.
- **Qwen 5+ tok/s on 16 GB: reachable, but only as a stack** — native/sync-removed decode + correctly sized expert cache + explicit merged reads + overlap + MLX buffer config + High Power mode, gated on bit-identical digests. No single technique reaches 2×.
- **Nemotron 15+ tok/s on 16 GB: not reachable bit-exact on 16 GB.** The published *fully resident* baseline is ~13.9 tok/s at ~20.8 GiB; the best published 16 GB result is **4.39 tok/s** with a 4 GB cache. 15+ requires ≥32 GB memory or a second Mac, or a quality-risky method (§7). This is reported honestly rather than forced.
- **Prefill: large bit-exact wins exist** — prefix caching 4.94×–57× TTFT, wave-partitional prefill 2.48×, full-layer staging 8×, pre-packed fp32 GEMM 1.44×. These are the safest wins available.
- **No exact speculative method reaches 2× here** (required acceptance length ≥3.6 vs achievable ~1.5–2) — consistent with the already-measured K=2 regression. See §8.

---

## 2. Baseline and targets

| | Model | Current | Target | Ratio | Binding constraint |
|---|---|---|---|---|---|
| Decode | Qwen3.8-Flash-Next, 125B MoE, 512 experts/layer, top-10, Q4 affine G32 | **2.3 tok/s** | **5+** | 2.2× | ~809 MB–1.47 GB experts read/token + Python sync |
| Decode | Nemotron 3.5 Bolt / Lightning, 74B-A9B, 240 experts/layer, top-9 | **6.7 tok/s** | **15+** | 2.2× | resident set ~20.8 GiB > 16 GB |
| Prefill | both | tens of tok/s | as high as possible | — | compute + repeated expert re-reads per chunk |

Quality gate: output must remain bit-identical (same token digests, BF16 rounding, exact routing). Methods marked **LOSSY** are excluded regardless of speed.

---

## 3. The decisive diagnostic (why 2.3 tok/s)

From `Jakevin/moe-offload` (same model, MLX, Mac mini M4 Pro; raw README verified 2026-11-17):

- Fully resident reference: **2.52 tok/s** at 24.9 GiB RSS — essentially Veystrix's 2.3.
- Per-token profile (118 ms): **GPU 27%, Python + `mx.eval` syncs 41%, SSD reads 29%, bookkeeping 3%.**
- With the I/O and sync path rewritten (details below): **8.51 → 9.17 → 9.73 tok/s**, greedy-identical output.
- C++ port of the decode path on the MLX C++ API: **bit-exact** (token ids, raw logits each step, 122/122 final cache tensors) at **9.0–9.6 tok/s**, ~11.5 GB RSS.

Independent corroboration — `1bit.gg` measured Qwen3.8-Flash-Next reading **1.47 GB of routed experts per token**; NVMe ~3.7 GB/s caps decode at ~2.5 tok/s, matching 2.3. Their hot-expert pinning + prefetch/streaming recovered **28–39 tok/s with perplexity exactly equal to resident**.

**Conclusion:** the SSD read volume is real, but so is a ~41% sync tax and a large IOPS/latency inefficiency. Apple internal NVMe measures 2.4–6.06 GB/s (6.06 GB/s with double buffering) vs Veystrix's effective ~1.86 GB/s (809 MB × 2.3). There is ~3× headroom before raw bandwidth becomes the wall.

---

## 4. Ranked methods (numbered, evidence-backed)

Legend — **EXACT** = bit-identical verified by source; **DIGEST TEST** = expected exact but must be verified in-app; **LOSSY** = fails the hard constraint (listed only in §8). "Reach?" = whether the method alone plausibly delivers the target ratio on 16 GB.

| # | Method | Measured evidence | Exactness | Prefill/Decode | Reach 2× alone? |
|---|---|---|---|---|---|
| 1 | **Native / sync-removed decode path** (C++ MLX decode, fused ops, no per-layer Python↔GPU sync) | Same model, M4 Pro: 2.52 → **9.0–9.6 tok/s**, bit-exact; profile shows 41% sync | **EXACT** | Decode | **Yes (Qwen)** on ≥64 GB; on 16 GB removes the largest single tax |
| 2 | **Explicit sorted, gap-merged `pread` of exactly the routed experts** (instead of mmap faults / per-slice reads) | Same model: 8.51 → **9.17 tok/s**; other Apple SSD MoE: +13–52% tok/s, read count 28.6k→8.5k, avg read 450 KB→1.5 MB | **EXACT** | Decode | No, but 2–4× class |
| 3 | **Per-expert coalescing / sidecar layout** (9 scattered sub-slices → 1 contiguous block) | ~**+20% decode**, +9–10% prefill; cold expert read 1396→463 µs (3.02×); 9× fewer read-ops; SIDECAR run: 3.97→14.14 tok/s across cache sizes, bitwise-identical logits 6/6 | **EXACT** (deterministic; no expert permutation) | Decode + prefill | No, but novel and composes with everything |
| 4 | **Next-layer router prefetch** (predict l+1 experts from current residual; read ahead) | Same model: 84.3%→**94.7% hit**, 9.17→**9.73 tok/s**; recall@10/16/24 = 83.6/94.1/96.7%; cross-layer prediction 10.89→12.70 tok/s | **DIGEST TEST** (prediction only changes *when* weights arrive) | Decode | No |
| 5 | **Correctly sized pinned expert cache**, sized by the ≥⅓-of-expert-bytes break-even rule | Nemotron 16 GB: 4 GB cache **4.39 tok/s**; 8 GB thrashes. Palm: 1 GiB 12.38 → 10 GiB **16.19 tok/s**; llama.cpp #27861: 18.4→**24.2 tok/s** at 48 slots (~4.1 GiB), collapse past ~24 GB | **EXACT** | Decode | No; essential enabler |
| 6 | **Overlap / double-buffered async I/O** (fetch l+1 while computing l; shared-expert compute hides misses) | Jetson: 1.94→**2.95 tok/s**, overlap alone **+32%**, bit-for-bit logits; TinyGiant M1 16 GB: 2.97→**6.06 GB/s** (2.09×); 75% hit 2.92→3.81 tok/s | **EXACT** | Decode | No, but large and bit-exact |
| 7 | **MLX Metal command-buffer limits** (`MLX_MAX_MB_PER_BUFFER=2048`, `MLX_MAX_OPS_PER_BUFFER=1000`) | MoE on M5 Max: **+11.9–13.7%**; launch-heavy MoE M3 Ultra +5–8%; output hashes bit-identical, prefill/memory unchanged | **EXACT** | Decode | No; pure config, zero risk |
| 8 | **High Power / AC mode** (Low Power off) | Controlled M4/M5 tests: **1.4–2.17×**, no quality cost | **EXACT** | Decode + prefill | Sometimes, free |
| 9 | **MLX memory-residency API** (`set_wired_limit`, `set_cache_limit`, `set_memory_limit`, `clear_cache`) | Compressed memory **19.69→1.7 GB (−91%)**; distinct from the already-rejected wired-limit *sysctl* | **EXACT** | Decode + prefill | No; removes pressure |
| 10 | **`async_eval` / one-step-ahead evaluation** with buffer-limit env vars | silica: **+50% at 4-bit**, decode == mlx-lm; streamlx uses async_eval on M4 Air | **DIGEST TEST** (scheduling only) | Decode | No |
| 11 | **io_uring + O_DIRECT out-of-core pager + KICK/WAIT doorbell** | Same model (PR #31469): 1.47→**16.29/22.03 tok/s**, 208→12–35 MiB/tok, **100% byte-exact**, 576/576 greedy tokens | **EXACT** | Decode | Yes in principle (Linux/CUDA proof) |
| 12 | **Lossless rANS expert reconstruction + 3-tier cache** (SSD-LLaMA) | **2.78×** over no-cache; prefill 1.52–4.19×, decode 2.10–15.58× across models | **DIGEST TEST** (lossless by construction) | Decode + prefill | No |
| 13 | **Prefix / prompt caching** (snapshot model + KV state at reusable prefixes) | Same model: TTFT **315.7 s → 63.9 s (4.94×)**, bit-exact when chunk boundaries match; Qwen3-9B: 19.42 s → **0.35 s (57×)**; KVLink 96% TTFT cut; SGLang 78–92% TTFT cut | **EXACT** | **Prefill** | N/A — prefill |
| 14 | **Wave-Partitional Prefill** (split expert GEMMs into waves when a batch touches more experts than the cache holds; mask + sum) | Merged llama.cpp PR #25294: prefill **2.28→5.65 tok/s (2.48×)**, bit-exact with matched kernels/ubatch | **EXACT** | **Prefill** | N/A — prefill |
| 15 | **Full-layer staging for prefill + second SSD mirror** | 7,943-token prompt: prefill **29→240 t/s**, decode 2.4–2.9→**5.8–6.0 t/s**, 78% hit, ~18 GB/s combined SSD | **DIGEST TEST** | Prefill + decode | Prefill yes |
| 16 | **Pre-packed fp32 prefill GEMM** | **1.44× end-to-end** (291→420 tok/s), max-abs-diff = 0 | **EXACT** | **Prefill** | N/A — prefill |
| 17 | **More unified memory / full residency (deferred by 16 GB-first rule)** | Same model: **2.3 tok/s (16 GB) → 12–16 tok/s (64 GB SSD offload) → 58–83 tok/s (128 GB)**; 64 GB M5 Pro 27.6 tok/s with draft head | **EXACT** | Decode + prefill | **Yes** (but not on 16 GB) |
| 18 | **Second Mac via llama.cpp RPC / tensor parallel** | 2× M4 16 GB, Qwen3.6-35B-A3B 4-bit: **25 tok/s**; TP over RDMA ~1.3× decode, ~1.7–2× prefill | **DIGEST TEST** | Decode + prefill | Yes (not 16 GB alone) |
| 19 | **Last-row-only prefill below 2,048 tokens** (untried repo lead) | Open lead in `DO_NOT_IMPLEMENT.md:78`; TTFT/memory candidate | **DIGEST TEST** (currently changes logits — gate required) | **Prefill** | N/A — prefill |
| 20 | **Multi-buffer / 64-bit offsets past the 8 GiB single-`uint32` pool cap** (untried repo lead) | Deferred by 16 GB-first rule; unlocks larger expert pools | **DIGEST TEST** | Decode | Enabler |
| 21 | **Expert-wise K>1 verification read-sharing at larger K** (untried repo lead) | Union ratio 0.542–0.62 at K=6 vs 0.796 at K=2; ~59 ms/token read wait potentially saved | **EXACT by construction** | Decode | No — K=2 already measured 16% slower here; larger K untested |
| 22 | **SwiftLM-style GPU-vs-SSD streaming rewrite** | Same top-8 quality: 0.58→**4.95 tok/s (8.5×)** M1 Ultra; 32 GB M6: 14.1 streaming vs 48.4 resident | **EXACT** (full top-k preserved) | Decode | No, but large |

**Count: 22 distinct methods**, exceeding the requested 10.

---

## 5. Top methods — implementation notes

### 5.1 Native / sync-removed decode (#1) — highest priority
The 41% `mx.eval`/Python sync figure is the key. The proven route is a C++/Metal fused decode path on the MLX C++ API, bit-exact against the Python loader (raw logits and all cache tensors). Start by profiling Veystrix's own per-token split (sync / SSD / GPU) before committing; if the split resembles the reference, this is the largest bit-exact win. Dense-attention-only caveat in the reference C++ port (≤2,048 tokens) must be addressed.

### 5.2 Explicit merged reads + coalescing + prefetch (#2, #3, #4)
These are distinct from already-rejected "expert prefetch" and "expert packing":
- **Explicit sorted/gap-merged pread** reads exactly the router-selected experts, but as large contiguous ranges (avg 450 KB → 1.5 MB) instead of thousands of small faults. +13–52% measured.
- **Coalescing** makes each expert's 9 scattered sub-slices contiguous. It is deterministic and bit-exact, needs no routing trace, costs negligible disk, and gives ~3× faster cold-expert I/O and ~20% decode — and it composes with the I/O path.
- **Next-layer prefetch** raises hit rate 84.3→94.7%. Prediction changes only timing, not values; verify with the digest gate.

### 5.3 Cache sizing (#5) — the 16 GB constraint
The break-even rule from `moe-expert-cache-study`: a cache helps only when its pool is **≥⅓ of total expert bytes**. At 11% coverage a Qwen 35B regressed −64%; at 71% coverage REAP-48 gained +216%. This explains why the 8 GB Nemotron cache thrashed on 16 GB while 4 GB worked (4.39 tok/s). On 16 GB, a few GB is the realistic ceiling; pin the cache to prevent the macOS VM compressor from compressing cold buffers (Palm: 12.38→16.19 tok/s with pinned cache + prefetch).

### 5.4 Zero-cost config (#7, #8, #9, #10)
Apply first: High Power mode, `MLX_MAX_MB_PER_BUFFER=2048`, `MLX_MAX_OPS_PER_BUFFER=1000`, the MLX residency API, `async_eval`. All are scheduling/placement, none change math; #7 is verified bit-identical hashes.

---

## 6. Prefill focus

Prefill is compute-bound and far more tractable than decode under the quality gate:

1. **Prefix caching (#13)** is the largest win when prompts repeat (agent turns, system prompts): 4.94× on the exact model, 57× on Qwen3-9B, bit-exact when chunk boundaries match. Caveat from the source: changing chunk boundaries can shift logits slightly (max |diff| 1.375, argmax unchanged) — keep chunking fixed for exactness.
2. **Wave-partitional prefill (#14)**: 2.28→5.65 tok/s, bit-exact with matched kernels/ubatch; available in merged llama.cpp PR #25294.
3. **Full-layer staging + second SSD mirror (#15)**: 29→240 t/s prefill; each 2,048-token batch reads ~99 GB from two SSDs.
4. **Pre-packed fp32 GEMM (#16)**: 1.44× end-to-end, max-abs-diff 0. Note BNNS Graph is *not* bit-exact at 9/12 shapes — avoid.
5. **Memory pressure**: cap context, free caches promptly, use paged KV (60–80%→<4% waste). KV quantization is lossy and excluded.
6. **Rejected for prefill**: FlashAttention is mathematically exact but **not bit-identical at BF16** (~10× more numeric deviation; HF confirms FA2 vs eager is almost always not bit-identical). qsa-fast-prefill already fails the quality gate here.

---

## 7. Feasibility verdict per target

**Qwen 5+ tok/s on 16 GB — reachable only as a stack.** Evidence: 4.39 tok/s was reached on 16 GB for *Nemotron* with a 4 GB cache; the same model reaches 9.0–9.73 tok/s on 64 GB with the I/O/sync rewrite. On 16 GB the credible path is #7+#8+#9 (free, bit-exact) → #5 (correctly sized cache) → #2+#3+#4 (merged reads, coalescing, prefetch) → #6 (overlap) → #1 (sync removal). Each step must be gated on a bit-identical digest. Expect the cache to be the limiter: a ~7.4 GB expert cache may not coexist with trunk + KV + OS on 16 GB.

**Nemotron 15+ tok/s on 16 GB — not reachable bit-exact.** The fully resident baseline is ~13.9 tok/s at ~20.8 GiB, i.e. the target exceeds the resident ceiling; the best published 16 GB figure is 4.39 tok/s. Paths that reach 15+: more memory (#17; 64 GB → 12–16 tok/s, ≥128 GB → 58–83), a second Mac (#18; 25 tok/s on 2× M4), or quality-risky methods (§8) — all deferred or excluded under the owner's rules. State this plainly rather than forcing a positive result.

**Prefill — strongly improvable and bit-exact** via §6, especially prefix caching and wave-partitional prefill.

---

## 8. DO NOT implement (quality gate / already rejected)

Cross-reference `docs/performance/DO_NOT_IMPLEMENT.md`.

- **Already measured and rejected on this hardware:** ANE execution; native MTP depths 1–8; K=2 read-sharing (16% slower, commit c68ec90); lookup drafting v2; expert prefetch as previously implemented; decode overlap as previously implemented; >I/O workers; non-S3FIFO pools; pools >2 GB; perfect expert predictor; cache-aware routing (quality gate); expert packing (+70 GiB, no server gain); F_NOCACHE; F_RDAHEAD=0; wired-limit sysctl; PLE row prefetch; qsa-fast-prefill; A2 hot-expert seeding (1.008×, CI includes 1.0 — not adopted).
- **Lossy / fails bit-identical:** Medusa / Hydra (relaxed acceptance at T>0); S2-MoE (own §6.2.2 admits approximate routing, top-1 agreement 89–93%); MoE-Spec (drops experts at verify); FP8 / INT8 / INT4 KV cache; eviction (H2O / SnapKV / StreamingLLM, 1–3% loss); expert substitution / reduced top-k / router fine-tuning; layer skipping / early exit (needs training); FlashAttention at BF16; any requantization (GPTQ/AWQ/SmoothQuant/SpQR/GGUF variants) that changes dequantized values; lower-precision experts. Note Veystrix's existing **G32** is already the quality-correct choice (G32 +8.0% PPL vs G64 +17.8%).
- **Dead ends for storage:** lossless ZSTD/LZ4 on already-4-bit weights saves only 0.9–2.1% — compression is not a lever. Expert packing grows disk without speed. BNNS Graph prefill GEMM is not bit-exact at most shapes.
- **Do not recommend FusionML:** an older repo note recommends it, but later source-verified research downgraded/rejected it (on the M4 raw data it loses to MLX; no LLM artifacts).

---

## 9. Recommended validation order (research only — nothing has been run)

1. **Profile first.** Reproduce the per-token sync / SSD / GPU split on Veystrix's 16 GB M4. This decides whether #1 or #2 is the top lever. No code change; no hardware run performed here.
2. **Zero-cost config** (#7, #8, #9): High Power, MLX buffer env, residency API. Verify digest unchanged.
3. **Cache sizing** (#5) using the ≥⅓ break-even rule; start at the smallest size that avoids thrash. Verify digest.
4. **I/O path** (#2, #3): explicit merged pread + expert coalescing. Both bit-exact by construction; verify digest.
5. **Overlap + prefetch** (#4, #6): prefetch direction and double buffering; verify digest.
6. **Sync removal** (#1): C++ / fused decode path; bit-exact gate against raw logits and cache tensors.
7. **Prefill** (#13, #14, #15): prefix caching and wave-partitional prefill, with fixed chunk boundaries for exactness.
8. **Defer** #17/#18/#20 until the 16 GB path is proven, per the owner's rule.

Every step: guarded run + bit-identical token digest vs the current engine. If any step drops quality, drop it entirely (owner rule).

---

## 10. Sources (consolidated)

**Exact model / hardware (primary):**
- Jakevin/moe-offload — https://github.com/Jakevin/moe-offload (Qwen3.8-Flash-Next on M4 Pro: 2.52→9.73 tok/s, 41% sync profile, C++ bit-exact 9.0–9.6, Nemotron 16 GB 4.39, prefix cache 4.94×)
- 1bit.gg — Qwen3.8-Flash-Next 1.47 GB experts/token, 28–39 tok/s with equal perplexity (reported in research notes)
- llama.cpp PR #31469 — https://github.com/ggml-org/llama.cpp/pull/31469 (same model, io_uring/O_DIRECT, 100% byte-exact)
- alexschiltmans sidecar doc — https://github.com/alexschiltmans/llama.cpp-per-layer-experts/blob/per-layer-experts/docs/moe-ssd-streaming.md (Qwen3.8 sidecar, bitwise-identical)
- Tencent Palm-Infra ssd-offload — https://github.com/TencentYoutuResearch/Palm-Infra/blob/main/docs/ssd-offload.md
- srcterm/streamlx — https://github.com/srcterm/streamlx ; RasoulNik/ssdmoe — https://github.com/RasoulNik/ssdmoe ; SharpAI/SwiftLM — https://github.com/SharpAI/SwiftLM
- mbolt coalescing gist — https://gist.github.com/doramirdor/0aeb975a99eb5a4644a3d105b57e3909
- AbhayXplor expert-cache study — https://github.com/AbhayXplor/moe-expert-cache-study (break-even rule)
- llamaperf M4 16 GB — https://llamaperf.com/gpu/m4-16gb (2× M4 RPC 25 tok/s; same-model baselines)

**Same class / general (secondary):**
- llama.cpp PR #25294 (Wave-Partitional Prefill, merged) — https://github.com/ggml-org/llama.cpp/pull/25294
- TinyGiant / llama.cpp discussion #27149 — https://github.com/ggml-org/llama.cpp/discussions/27149
- Jetson overlapping streaming — https://www.anyesh.me/writing/moe-stream-jetson-orin
- SSD-LLaMA — https://arxiv.org/html/2609.18110
- ExpertFlow — https://arxiv.org/html/2410.17954v2 ; SeqMoE — https://arxiv.org/html/2609.12978 ; Fiddler — https://arxiv.org/abs/2402.07033
- llama.cpp PR #27861 (GPU-resident LRU) — https://github.com/ggml-org/llama.cpp/pull/27861
- Qwen3-9B 16 GB M4 (57× prefix cache, bandwidth ceiling) — https://www.drmhse.com/posts/running-qwen3-8-9b-on-a-16gb-mac
- vLLM-metal MoE offload — https://docs.vllm.ai/projects/vllm-metal/en/latest/moe_offload
- M4 Mac MoE benchmark (llama.cpp vs MLX-LM) — https://yuvrajsingh.io/posts/mac-mini-m4-moe-benchmark/
- M4 SSD-streaming 64 GB analysis — https://note.com/hacklog_stealth/n/nd5833cb2bfe2

**Speculative decoding (exactness verdicts):** EAGLE-3 greedy table (EAGLE-3 5.51× mean but dense GPU); ReDrafter up to 2.3× on dense Apple Silicon; S2-MoE (lossy); SGLang prefix caching. Detail in `research_notes/Faster Qwen Nemotron inference on M4/speculative_decoding.md`.

Full per-subtopic notes (with inline URLs): `research_notes/Faster Qwen Nemotron inference on M4/` — `direct_web_evidence.md`, `moe_expert_io.md`, `speculative_decoding.md`, `kv_prefill.md`, `apple_silicon_mlx.md`, `quantization_formats.md`, `serving_systems.md`, `local_repo_leads.md`.
