# Sequential speedup roadmap — 22 methods

**Branch:** `speedup-roadmap`
**Owner targets:** Qwen decode **2.3 → 5+ tok/s**, Nemotron decode **6.7 → 15+ tok/s**, plus faster prefill.
**Date opened:** 2026-11-17
**Research basis:** `reports/Faster Qwen Nemotron inference on M4.md` and `research_notes/Faster Qwen Nemotron inference on M4/` (8 note files, primary sources inline).

This is the execution ledger. We go **one method at a time**, in the order below, and never stack a new method until the current one has passed its gate and been recorded. Method IDs 1–22 are the canonical IDs from the report; the **Step** column is the order we will actually attempt them (cheap/bit-exact first, risky/heavy last).

## Rules of engagement (non-negotiable)

1. **16 GB M4 first.** No larger Mac until the 16 GB path is proven (owner rule).
2. **Bit-identical digest gate.** Every method must produce the same token digest, BF16 rounding boundaries, Q4 affine G32 weights, and exact top-10/512 routing as the current baseline. If a method drops quality, **drop it entirely** (owner rule).
3. **Guarded runs only, with explicit permission.** No hardware benchmark runs are authorized by this document. Each A/B must use `scripts/flashnext/real_run_guard.py` with the established envelope; permission is asked immediately before each run.
4. **One method at a time.** Each A/B is candidate vs the *current accepted baseline*, not against a stale number. Accepted methods become the new baseline.
5. **Record everything.** Results into `results/...` (preserved; never delete), guard JSONL into `experiments/ane/results/` or the appropriate folder, and update the status table below.
6. **Do not implement anything in `DO_NOT_IMPLEMENT.md`** without a new premise. That registry stays authoritative for dead ends.

## Honest feasibility verdict (from research, before we start)

- **Qwen 5+ on 16 GB: reachable only as a stack** of Steps 1–8. No single method delivers 2×.
- **Nemotron 15+ on 16 GB: not reachable bit-exact.** Fully-resident ceiling is ~13.9 tok/s at ~20.8 GiB; best published 16 GB figure is 4.39 tok/s. Steps 19–21 (more memory / second Mac) are the only credible routes and are deferred by rule 1. We will still attempt the 16 GB-safe steps; we do not expect 15+ until memory or device count changes.
- **Prefill: strongly improvable and bit-exact** (Steps 15–18).

---

## Status tracker

| Step | Method | Name | Phase | Target | Expectation | Gate | Status |
|---|---|---|---|---|---|---|---|
| 0 | — | Profile per-token sync / SSD / GPU split | Prereq | both | **DONE 2026-11-17:** host 6.1%, I/O wait 52.3% | guarded, clean | DONE |
| 1 | #8 | High Power / AC mode | Free config | both | **not applicable:** already on AC, `lowpowermode 0`, and MacBook Air M4 has no High Power Mode | none needed | N/A — already optimal |
| 2 | #7 | MLX command-buffer limits | Free config | decode | **A/B measured:** Qwen decode unresolved + TTFT +0.86 s regression; Nemotron decode **−0.5 tok/s resolved regression** | digest identical | REJECTED on our 16 GB M4 |
| 3 | #9 | MLX memory-residency API | Free config | both | compressed mem −91% | digest | PENDING |
| 4 | #10 | `async_eval` one-step-ahead | Free config | decode | **already rejected:** `DO_NOT_IMPLEMENT.md:56` 311 vs 314 tok/s; `FLASHNEXT_OVERLAP` already on | digest | REJECTED (prior) |
| 5 | #5 | Correctly sized pinned expert cache | Cache | decode | **A/B measured:** Nemotron 2→4 GB is a resolved REGRESSION (−1.61 tok/s, +0.86 s TTFT) | digest identical | 2 GB retained; 4 GB rejected |
| 6 | #2 | Explicit sorted, gap-merged `pread` | Read path | decode+prefill | **A/B measured:** TTFT −0.69 s (resolved), decode unresolved | digest identical | PARTIAL — flag off |
| 7 | #3 | Per-expert coalescing / sidecar layout | Read path | decode+prefill | **already rejected:** expert packing +9% in-process, no server gain, +70 GiB disk | digest | REJECTED (prior) |
| 8 | #4 | Next-layer router prefetch | Prefetch | decode | **already implemented & rejected:** `decode_prefetch.py` + Nemotron `LOOKAHEAD_LAYERS`; `DO_NOT_IMPLEMENT.md:20` 2.39→2.38 tok/s | digest | REJECTED (prior) |
| 9 | #6 | Overlap / double-buffered async I/O | Overlap | decode | **already rejected here:** `DO_NOT_IMPLEMENT.md:21,25` decode overlap 2.39→2.41 (within band), perfect predictor tied | digest | REJECTED (prior) |
| 10 | #1 | Native / sync-removed decode path | Sync removal | decode | 1.065× Qwen / 1.386× Nemotron ceilings measured; owner wants full port later for future-model reuse | raw logits + cache tensors | PLANNED LATER (owner decision 2026-11-17) |
| 11 | #14 | Wave-Partitional Prefill | Prefill | prefill | 2.28→5.65 tok/s | digest, fixed chunk boundaries | PENDING |
| 12 | #13 | Prefix / prompt caching | Prefill | prefill | 4.94×–57× TTFT | digest, fixed chunking | PENDING |
| 13 | #16 | Pre-packed fp32 prefill GEMM | Prefill | prefill | 1.44×, max-abs-diff 0 | digest (avoid BNNS Graph) | PENDING |
| 14 | #15 | Full-layer staging + second SSD mirror | Prefill | prefill+decode | 29→240 t/s prefill | digest | PENDING |
| 15 | #19 | Last-row-only prefill < 2,048 tokens | Prefill | prefill | TTFT/memory lead | digest required (currently changes logits) | PENDING |
| 16 | #12 | Lossless rANS reconstruction + 3-tier cache | I/O stack | decode+prefill | 2.78× | digest | PENDING |
| 17 | #11 | io_uring + O_DIRECT pager + KICK/WAIT | I/O stack | decode | 1.47→16–22 tok/s (source) | byte-exact | PENDING |
| 18 | #22 | SwiftLM-style GPU-vs-SSD rewrite | I/O stack | decode | 8.5× (source, full top-k) | digest | PENDING |
| 19 | #17 | Full residency on more memory | Deferred | decode+prefill | 2.3→58–83 tok/s at 128 GB | digest | DEFERRED (16 GB rule) |
| 20 | #18 | Second Mac via RPC / tensor parallel | Deferred | decode+prefill | 25 tok/s on 2× M4 | digest | DEFERRED (16 GB rule) |
| 21 | #20 | Multi-buffer / 64-bit offsets past 8 GiB cap | Deferred | decode | unlock larger pool | digest | DEFERRED (16 GB rule) |
| 22 | #21 | K>1 verification read-sharing at larger K | Deferred | decode | union 0.54–0.62 at K=6 | digest (K=2 already slower) | DEFERRED (needs new premise) |

---

## Phase 0 — Prerequisite: profile — DONE 2026-11-17

**Step 0 measured (guarded, clean, token IDs identical across 3 passes).** Saved at
`results/flashnext/20261117-host-window-profile/`. Per token (300.1 ms, 3.33 tok/s on
this short-prompt harness): **I/O wait 156.86 ms = 52.3%**, all exclusive host time
**18.22 ms = 6.1%** (largest: `route_tolist` 12.50 ms). Removing 100% of host work
would yield only 3.55 tok/s (1.065×). Therefore:

- **Step 10 (full C++ engine port) is PLANNED LATER by owner decision (2026-11-17)** — real
  but bounded gain (1.065× Qwen / 1.386× Nemotron), wanted for future-model reuse; do not
  start until the owner says so. Scaffold at `cpp/decode/`, plan at `CPP_DECODE_CHECKLIST.md`.
- **Steps 5–9 (I/O: cache sizing, merged pread, coalescing, prefetch, overlap) are ACTIVE NOW** —
  they target the 52.3% / 67.9% I/O wait.

Local evidence supersedes the reference's 41%-sync profile here:
`models/flashnext/hostwindow.py:7-8` (added host work is free),
`models/flashnext/adaptive_topk.py:89-103` (one sync per layer already enabled).

**Nemotron profiled separately 2026-11-17** (guarded, digests captured, saved at
`results/nemotron35/20261117-140232-decode-profile-manual/`): 5.51 tok/s, 181.4 ms/token;
I/O wait 67.9%, host route sync 27.8%. C++ ceiling is 1.386× (7.64 tok/s) — real but
far short of 15. Cache hit rate only 43.7% at a 2 GB pool, 436 MB read/token, so
**cache sizing (Method #5) and I/O reduction (#2/#3) are the top Nemotron levers too.**
See `docs/performance/CPP_DECODE_CHECKLIST.md`.

---

## Phase 1 — Free, bit-exact config (Steps 1–4)

**Step 1 — Method #8 NOT APPLICABLE (checked 2026-11-17).** Machine is a MacBook Air M4
(`hw.model Mac16,12`): already on AC, `pmset lowpowermode 0`, and this model has no High Power
Mode (that is MacBook Pro M1 Pro/Max+/Mac Studio only). The source's 1.4–2.17× compares
Low-Power-on vs off; our state is already the "off/fast" side. No A/B possible or needed.

**Step 2 A/B — Method #7 MEASURED & REJECTED 2026-11-17.** Both guarded, 3 rounds, digest identical.
- Qwen (`results/flashnext/20261117-145953-performance-v2-flag-flashnext-mlx-auto-buffer-caps-manual/`):
  decode 2.08 → 2.14 (unresolved, inside band); **TTFT 23.34 → 24.52 s resolved regression (+0.86 s)**.
- Nemotron (`results/nemotron35/20261117-150713-performance-v2-flag-veystrix-mlx-auto-buffer-caps-manual/`):
  **decode 6.4 → 5.88 tok/s resolved regression (−0.5 vs band 0.32)**; physical reads +3.7 GB resolved.
- The source's +11.9–13.7% was on an M5 Max (different chip/driver); on our 16 GB M4 it does not
  reproduce and hurts prefill. Flag stays **off**. Do not retry. (Rust research independently
  confirmed command-buffer count matters on Apple Silicon, but the specific MLX caps hurt here.)

### Step 1 — Method #8: High Power / AC mode
- **Premise:** macOS Low Power mode caps sustained GPU/SSD clocks; disabling it is free and changes no math.
- **Change:** verify harness runs with AC / High Power; Low Power off.
- **Evidence:** 1.4–2.17× in controlled M4/M5 tests.
- **Gate:** same digest; record thermal/clock state so it is not confounded with later steps.
- **Status:** PENDING.

### Step 2 — Method #7: MLX Metal command-buffer limits
- **Premise:** larger command buffers reduce launch overhead on launch-heavy MoE decode.
- **Change:** `MLX_MAX_MB_PER_BUFFER=2048`, `MLX_MAX_OPS_PER_BUFFER=1000` (source also tested 400 MB + 1000 ops).
- **Evidence:** MoE +11.9–13.7% on M5 Max; +5–8% launch-heavy MoE; output hashes bit-identical; prefill/memory unchanged. Dense models gain nothing.
- **Gate:** digest; confirm no prefill regression.
- **Status:** PENDING.

### Step 3 — Method #9: MLX memory-residency API
- **Premise:** macOS VM compressor compresses cold anonymous buffers and decompresses them inside matmul.
- **Change:** use `set_wired_limit` + `set_cache_limit` + `set_memory_limit` + `clear_cache` (MLX API). **Not** the already-rejected `iogpu.wired_limit_mb` sysctl.
- **Evidence:** compressed memory 19.69 → 1.7 GB (−91%).
- **Gate:** digest; watch for swap growth under the guard.
- **Status:** PENDING.

### Step 4 — Method #10: `async_eval` one-step-ahead
- **Premise:** overlap evaluation of the next step with host work.
- **Change:** enable `async_eval` / one-step-ahead evaluation with buffer-limit env vars.
- **Evidence:** +50% at 4-bit (silica); decode == mlx-lm.
- **Gate:** digest (scheduling only, expected exact).
- **Status:** PENDING.

---

## Phase 2 — Cache sizing (Step 5)

**Step 5 A/B — MEASURED 2026-11-17 (guarded, clean, digest identical).** Saved at
`results/nemotron35/20261117-144030-performance-v2-pool-compare-manual/`. Nemotron 2 GB → 4 GB:
decode **6.8 → 5.19 tok/s (resolved, median −1.61 vs band 0.15)**; TTFT **5.70 → 6.60 s
(resolved, +0.86 vs band 0.12)**; footprint +2.14 GB (resolved). Physical read bytes unresolved.
**Bigger cache is a clear regression on this 16 GB M4** — memory pressure/compressor dominates
(compressor peak 1991 → 5276 MB), matching the reference's 8 GB-cache thrash.

**Follow-up 1 vs 2 GB — MEASURED 2026-11-17**, saved at
`results/nemotron35/20261117-144538-performance-v2-pool-compare-manual/`. The trend continues:
**1 GB is faster than 2 GB (6.94 vs 6.56 tok/s, resolved median −0.58 vs band 0.16)**, with
**6.17 GB fewer physical reads** and 1.05 GB less footprint (both resolved); TTFT unchanged.
1 GB won in all 3 rounds. So on this 16 GB M4 the pool sweet spot is at or below 1 GB —
smaller pool = less memory pressure = faster. Do not increase the pool on 16 GB.

**Production already does the right thing here:** `models/nemotron35/admission.py:141-145`
records the same trend (no pool 7.1–7.4 tok/s / 195–202 MB per token; 2 GiB 6.4–7.2 / 245–260;
3 GiB 5.3–5.5 / 290), and its auto-sizing returns **0** when routed experts cannot fit beside
the dense core and OS reserve. The A/B above forced pools that production would not use, so
**no code change is needed** — this is confirmation, not a new optimization. Method #3
(expert coalescing / packing) is already measured and rejected in `DO_NOT_IMPLEMENT.md`
(+9% in-process, no server gain, +70 GiB disk), so it is not retried.

### Step 5 — Method #5: Correctly sized pinned expert cache
- **Premise:** a cache helps only when its pool is ≥⅓ of total expert bytes. At 11% coverage a model regressed −64%; at 71% coverage another gained +216%. On 16 GB, a few GB is the realistic ceiling; 8 GB thrashed (Nemotron, 4 GB = 4.39 tok/s).
- **Change:** sweep small cache sizes (start ≤4 GB), pin the cache to avoid VM compression, allocate per-layer proportional to observed routing (trace first if needed).
- **Evidence:** Nemotron 16 GB 4 GB cache = 4.39 tok/s, 86.1% hit; Palm 1→10 GiB 12.38→16.19 tok/s; llama.cpp #27861 18.4→24.2 tok/s at 48 slots.
- **Gate:** digest + no thrash + guard floor. **This step determines the ceiling for all later I/O steps on 16 GB.**
- **Status:** PENDING.

---

## Phase 3 — Read path (Steps 6–7)

**Step 6 A/B — MEASURED 2026-11-17 (guarded, clean, digest identical across all 6 arms).**
Saved at `results/flashnext/20261117-143117-performance-v2-flag-flashnext-sorted-merged-pread-manual/`.
Medians: decode 2.14 → 2.17 tok/s (verdict **unresolved**, sign differs between rounds);
TTFT 23.13 → 22.56 s (verdict **resolved**, median −0.69 s vs band 0.30); physical read
96.7 → 95.8 GB (unresolved). Implemented in `store.py:_pread_into_merged` behind
`FLASHNEXT_SORTED_MERGED_PREAD=1` (off by default). Interpretation: prefill reads many rows
per call so merging engages; decode's reads are chunked to ~2 rows and top-10/512 experts
are too sparse for adjacent runs, so no decode gain. Keep flag off by default; no quality
issue. **Do not retry gap-span merging for decode unless row-per-call count increases.**

### Step 6 — Method #2: Explicit sorted, gap-merged `pread`
- **Premise:** mmap faults and per-slice reads cause 16 KiB random read storms; reading exactly the router-selected experts as large merged ranges is bit-exact.
- **Change:** replace fault/per-slice reads with sorted, gap-merged parallel `pread` of the selected experts' byte ranges.
- **Evidence:** same model 8.51→9.17 tok/s; elsewhere +13–52%, read count 28.6k→8.5k, avg read 450 KB→1.5 MB.
- **Gate:** digest.
- **Status:** PENDING.

### Step 7 — Method #3: Per-expert coalescing / sidecar layout
- **Premise:** an expert's 9 sub-slices (gate/up/down × weight/scales/biases) are scattered over a large span; making them contiguous is deterministic and bit-exact.
- **Change:** one-time offline sidecar/relayout so each expert is one contiguous block. **No expert permutation** (permutation risks changing token identity and is excluded).
- **Evidence:** ~+20% decode, cold read 1396→463 µs (3.02×), 9× fewer read-ops (486→54/token), negligible extra disk; sidecar run bitwise-identical.
- **Gate:** digest.
- **Status:** PENDING.

---

## Phase 4 — Prefetch and overlap (Steps 8–9)

### Step 8 — Method #4: Next-layer router prefetch
- **Premise:** the router completes before expert compute, creating a prefetch window; predicting l+1 experts from the current residual is bit-exact (changes timing only).
- **Change:** add next-layer expert prediction; read ahead into the cache.
- **Evidence:** same model hit 84.3→94.7%, 9.17→9.73 tok/s; recall@10/16/24 = 83.6/94.1/96.7%; Palm cross-layer 10.89→12.70 tok/s.
- **Gate:** digest.
- **Status:** PENDING.

### Step 9 — Method #6: Overlap / double-buffered async I/O
- **Premise:** fetch l+1 while computing l; GPU shared-expert compute hides SSD misses with one sync per layer.
- **Change:** reorder the layer so router top-k issues reads first, compute shared FFN while NVMe works, join before expert matmul.
- **Evidence:** Jetson 1.94→2.95 tok/s, overlap alone +32%, bit-for-bit logits; TinyGiant M1 16 GB 2.97→6.06 GB/s (2.09×); 75% hit 2.92→3.81 tok/s.
- **Gate:** digest.
- **Status:** PENDING.

---

## Phase 5 — Sync removal (Step 10)

### Step 10 — Method #1: Native / sync-removed decode path
- **Premise:** 41% of per-token time is Python/`mx.eval` sync on the same model. This is the single largest bit-exact lever.
- **Change:** C++/Metal fused decode path on the MLX C++ API (no per-layer Python↔GPU sync). Address the reference port's dense-attention-only ≤2,048-token limitation.
- **Evidence:** same model 2.52→9.0–9.6 tok/s; bit-exact on token ids, raw logits every step, 122/122 cache tensors; C++ path 9.0–9.6 at ~11.5 GB.
- **Gate:** raw logits + full cache tensor equality, not just final tokens.
- **Status:** PENDING (do not start until Step 0 profile confirms the sync share).

---

## Phase 6 — Prefill (Steps 11–15)

### Step 11 — Method #14: Wave-Partitional Prefill
- **Premise:** when a ubatch touches more experts than the cache holds, run expert GEMMs in waves, mask and sum; each expert loaded once per ubatch.
- **Evidence:** merged llama.cpp PR #25294: 2.28→5.65 tok/s (2.48×); bit-exact with matched kernels/ubatch; O_DIRECT + async I/O pool.
- **Gate:** digest; keep kernels and ubatch matched.
- **Status:** PENDING.

### Step 12 — Method #13: Prefix / prompt caching
- **Premise:** agent turns reuse long system/tools prefixes; snapshot model + KV state at prefix points.
- **Evidence:** same model TTFT 315.7→63.9 s (4.94×); Qwen3-9B 19.42→0.35 s (57×); KVLink 96% TTFT cut. Bit-exact when chunk boundaries match.
- **Gate:** digest; **keep chunk boundaries fixed** (changing them shifted logits slightly in the source, max |diff| 1.375, though argmax was unchanged).
- **Status:** PENDING.

### Step 13 — Method #16: Pre-packed fp32 prefill GEMM
- **Premise:** pre-pack weights into an fp32 prefill GEMM layout.
- **Evidence:** 1.44× end-to-end, max-abs-diff = 0. **BNNS Graph is NOT bit-exact at 9/12 shapes — do not use.**
- **Gate:** digest.
- **Status:** PENDING.

### Step 14 — Method #15: Full-layer staging + second SSD mirror
- **Premise:** stage the next layer's whole expert block into pinned host buffers while the GPU computes the current layer; mirror the GGUF across two SSDs.
- **Evidence:** 7,943-token prompt prefill 29→240 t/s, decode 2.4–2.9→5.8–6.0 t/s, 78% hit, ~18 GB/s combined SSD.
- **Gate:** digest; depends on a second physical SSD (hardware availability).
- **Status:** PENDING.

### Step 15 — Method #19: Last-row-only prefill below 2,048 tokens
- **Premise:** untried repo lead (`DO_NOT_IMPLEMENT.md` "still open"); TTFT/memory candidate for short prompts.
- **Evidence:** repo lead, premise only — no published number.
- **Gate:** currently changes logits → **bit-exactness gate must be cleared before any A/B**. If it cannot be made exact, drop it.
- **Status:** PENDING (gate not yet cleared).

---

## Phase 7 — I/O stack rewrites (Steps 16–18)

### Step 16 — Method #12: Lossless rANS reconstruction + 3-tier cache
- **Premise:** pack each expert into one aligned block, reconstruct exact weights in VRAM with lossless rANS, cache all three of gate/up/down.
- **Evidence:** SSD-LLaMA 2.78× over no-cache; prefill 1.52–4.19×, decode 2.10–15.58×. Caching only gate/up gives higher hit but lower throughput.
- **Gate:** digest (lossless by construction; audit that decompressed weights equal the original quantized values).
- **Status:** PENDING.

### Step 17 — Method #11: io_uring + O_DIRECT pager + KICK/WAIT doorbell
- **Premise:** compact VRAM slot pool + host RAM victim cache + batched io_uring/O_DIRECT pager + on-device doorbell overlapping shared-expert compute with miss servicing.
- **Evidence:** same model (PR #31469) 1.47→16.29/22.03 tok/s, 208→12–35 MiB/tok, **100% byte-exact**, 576/576 greedy tokens. Linux/CUDA proof; needs an MLX/Metal port.
- **Gate:** byte-exact.
- **Status:** PENDING (port effort high).

### Step 18 — Method #22: SwiftLM-style GPU-vs-SSD streaming rewrite
- **Premise:** rewrite SSD expert streaming with a tuned GPU/SSD split at full top-k.
- **Evidence:** 0.58→4.95 tok/s (8.5×) at full top-8 quality; 32 GB M6 14.1 streaming vs 48.4 resident.
- **Gate:** digest (full top-k preserved).
- **Status:** PENDING.

---

## Phase 8 — Deferred by the 16 GB-first rule (Steps 19–22)

### Step 19 — Method #17: Full residency on more memory
- **Premise:** SSD reads disappear when experts fit in unified memory.
- **Evidence:** same model 2.3 tok/s (16 GB) → 12–16 (64 GB) → 58–83 (128 GB); 64 GB M5 Pro 27.6 tok/s with draft head.
- **Gate:** digest. **Status:** DEFERRED until 16 GB path proven.

### Step 20 — Method #18: Second Mac via RPC / tensor parallel
- **Premise:** pool two 16 GB Macs over Thunderbolt.
- **Evidence:** 2× M4 16 GB, Qwen3.6-35B-A3B 25 tok/s; TP over RDMA ~1.3× decode, ~1.7–2× prefill (50/50 split; 40/60 OOMs).
- **Gate:** digest. **Status:** DEFERRED until 16 GB path proven.

### Step 21 — Method #20: Multi-buffer / 64-bit offsets past the 8 GiB pool cap
- **Premise:** current single `uint32` pool caps at 8 GiB; multi-buffer/64-bit offsets unlock a larger expert pool.
- **Evidence:** repo lead, premise only.
- **Gate:** digest. **Status:** DEFERRED until 16 GB path proven.

### Step 22 — Method #21: K>1 verification read-sharing at larger K
- **Premise:** route union across K tokens shares reads; union ratio 0.542–0.62 at K=6 vs 0.796 at K=2.
- **Evidence:** repo lead. **K=2 was already measured 16% slower here (commit c68ec90)** — the genuinely open part is larger K only.
- **Gate:** digest; requires a new premise after K=2 result. **Status:** DEFERRED.

---

## How we will record each step

For every step, before touching code:
1. Confirm the guard envelope and ask for run permission.
2. Run candidate vs current accepted baseline with identical prompt/seed/token count.
3. Compare token digest (and raw logits for Step 10), tok/s, physical MB/token, guard floor, swap/compressor growth.
4. If digest differs → **drop the method**, mark REJECTED with the number, add to `DO_NOT_IMPLEMENT.md`.
5. If digest matches but slower → mark NO GAIN, keep the flag OFF (do not delete code; owner decision 2026-11-17).
6. If digest matches and faster → mark ACCEPTED, record the gain, promote baseline, update the tracker.
7. Commit the step on this branch with the measurement in the commit message.

## Links
- Research report: `reports/Faster Qwen Nemotron inference on M4.md`
- Research notes: `research_notes/Faster Qwen Nemotron inference on M4/` (8 files)
- Dead-end registry: `docs/performance/DO_NOT_IMPLEMENT.md`
- Active speed profile: `docs/performance/EXACT_SPEED.md`
