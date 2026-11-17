================================================================================
OPUS — SECOND MIRACLE HUNT (hard constraint: 16 GB M4 ONLY, no new hardware)
================================================================================
SELF-CONTAINED. Paste this whole message as the opener. Work only from what is
here plus your OWN fresh research — I want you to research deeply (papers,
2025-2026 arXiv, Apple internals, other engines) AND reason from first principles.

HARD CONSTRAINTS (non-negotiable):
- Must run on THIS machine: MacBook Air M4, 16 GB unified memory, ONE internal
  SSD (~3.36 GB/s measured cold), fanless, Thunderbolt 4. NO second SSD, NO
  second Mac, NO more RAM. I cannot buy hardware. If your answer requires any
  of those, it is not an answer — say "no software solution exists" instead.
- NO QUALITY LOSS: bit-identical tokens, BF16 rounding, Q4 affine G32/G64,
  exact top-10/512 and top-6/128 routing. A method that changes logits/tokens
  is OUT regardless of speed. Speculative verification is fine IF the final
  output is the guaranteed exact greedy/searched result.
- ANE is ALLOWED and I want it seriously evaluated — but it must be bit-exact
  and must actually help (see the ANE findings below; current evidence is negative).

================================================================================
1. WHAT WE ALREADY PROVED (do NOT re-litigate; build on it)
================================================================================
MODEL A: Qwen3.8-Flash-Next, qwen4_exp, 125B MoE, 512 experts/layer, top-10,
  48 layers, Q4 affine G32 (U32-packed codes), GDN+QSA attention. ~873 MB
  physical/token. Production 2.3 tok/s; our harness ~2.19. TARGET 5+.
MODEL B: Nemotron 3.5 Lightning, 128 experts/layer, top-6, 23 MoE layers,
  Q4 affine G64, ~857 MB/token. ~7 tok/s. TARGET 15+.

E0 (concurrency proof, PASSED): GPU decode compute and saturated SSD reads
  coexist. Cold F_NOCACHE reads over the full 113 GB checkpoint ran at
  **3.36 GB/s** while a fully-cached decode step inflated only **10.1%**
  (130→143 ms/token). Gate bit-identical. => overlap is physically possible.
E1 (oracle pipeline, KILLED for targets): pass 1 records the EXACT greedy
  route of every MoE call; pass 2 prefetches those exact future routes 1 token
  ahead while the real router still decides (bit-exact by construction).
  Result: 2.19 → **2.72 tok/s (+25%)**, physical bytes barely changed (−1.8%).
  Even a PERFECT predictor/prefetcher gives only +25% and cannot reach 5.
E3 (offline predictor recall): best free predictor ~0.36 recall; cross-layer
  ~0.02 (adjacent layers route nearly disjoint experts). A trained predictor
  cannot approach the 1.0 oracle E1 assumed.
Byte wall: 873 MB ÷ 3.36 GB/s ≈ 260 ms drive time; ~200 ms compute. Even
  perfect overlap caps Qwen ~3.8 tok/s. Cache ceilings (real 1,200-token
  traces): Qwen OPT 5.22 tok/s needs 12 GB unique expert RAM; Nemotron 15.22
  needs 10 GB. Neither fits with the model in 16 GB. Production auto-sizes pool to 0.

WHAT IS CLOSED (measured, do not propose):
- Prediction/prefetch (E1), any trained head (E3), Edge0 committed routing
  (breaks quality). SpecFit/MTP read-sharing K=2 was 16% slower on Metal.
- Lossless code-stream compression: I measured the REAL MLX expert tensor
  (dtype U32-packed, [512,640,320]): zstd-1/9/19/22 = 95.2%, zlib = 95.5%,
  bit-exact roundtrip. Entropy 3.86/4 bits. CLOSED. (The vramkit −33% result
  was on clean int4 in a different format — does NOT transfer to MLX U32.)
- Algebraic reformulation (Decode-Late, T-MAC, LUT-GEMV): reduce arithmetic,
  read the SAME weight bits. Grammar-compressed inner product is provably hard.
- Cross-expert delta/XOR, bit-plane partial fetch: no redundancy / all planes
  needed for exactness. Bigger pool (2→4 GB) regressed. F_NOCACHE alone 8% slower.
- ANE: cannot touch the byte wall (shares memory bus; its fast path is still
  way above SSD speed), grouped-affine Q4 (= CoreML per_block) is NOT supported
  and falls back to CPU, palettized int4 is lossy, fp16 error ~2.5e-2 vs GPU
  3.2e-3 and CoreML exports flipped up to 85 greedy decisions. Only energy win
  (~2x tok/s/W). So ANE is OUT unless you find a genuinely bit-exact role.
- OS I/O tricks: F_RDADVISE can only reach ~3.6 GB/s (hides latency, not more
  bandwidth) and is disabled in production on memory-constrained Macs; MTLIO
  13 GB/s is warm-cache only; 3.36 GB/s is the M4 Air hardware ceiling.

Server-vs-profile gap (2.3 production vs 3.33 short harness): likely a
measurement-regime artifact (14-31K context; lightning indexer bypassed below
2048 tokens), NOT a hidden 135 ms/token tax. Compute fusion is real but bounded
(~1-7 ms); even zero compute leaves ~156 ms I/O → ~6.3 tok/s Qwen.

================================================================================
2. THE THREE LEADS THAT SURVIVED 8+6 RESEARCH STREAMS — attack these
================================================================================
The byte wall is the only thing left. These are the ONLY bit-exact byte levers
that survived. Combine them, find a better one, or prove none reaches the target.

LEAD 1 (strongest, RANK 1) — read-once SSD staging dedup kernel, ~1.8x bytes.
  arXiv:2610.34440 proves dedup beats per-slot iff byte-share u(n) < bandwidth-
  ratio r(n), and explicitly names r(n) "the lever for kernel work". On stock
  Metal Qwen3, r(2)=0.50 (per-slot 131 GB/s vs dedup 65 GB/s, no on-chip cache
  reuse) and u(2)=0.77 → depth-2 loses (explains our 16% slowdown). BUT that
  r(n) was measured with weights in unified memory. In Veystrix's SSD regime,
  the per-slot kernel's 131 GB/s is IRRELEVANT (SSD is 3.36 GB/s). A weight-major
  "read each distinct expert ONCE from SSD into a staging MTLBuffer, then
  grouped-GEMM over its tokens" kernel has r_ssd ≈ 1 > u(2)≈0.8 → should win by
  the route-union factor ~1.8x. Bit-exact trivially (same bytes, same routing,
  same arithmetic). MonoMoE (arXiv:2609.04244) is the weight-major kernel
  precedent. Projected Qwen 873→~485 MB (~144 ms → ~5 tok/s budget met);
  Nemotron 857→~476 MB (still >67 ms budget). THIS IS THE ONE I WANT YOU TO
 stress-test and design. Is the r_ssd≈1 synthesis sound? Does it really win in
  SSD-offload? What is the exact kernel design and the falsifiable A/B?

LEAD 2 — certify-don't-predict (radical stream Q1, a genuine open niche).
  A low-rank int8 "shadow" gives a PROVABLE upper bound on logits; blocks are
  opened highest-bound-first and the search stops when the best logit exceeds
  the ceiling of every unopened block — returning the mathematically GUARANTEED
  exact argmax while reading only ~20-25% of bytes. Published on the LM head:
  ~5x lossless head-byte reduction, 1.97x batch-1 decode, argmax match 1.0000.
  Head-only is ~6% end-to-end, but the PRIZE is applying the same certificate
  ONE LAYER EARLIER to skip full-precision expert reads when the top-1/top-2
  margin exceeds a composed per-layer error bound. No source does this for MoE
  experts — a real open niche. Can you design an exact per-layer / per-expert
  certificate that guarantees the final greedy token while skipping expert
  reads? Kill if <30% of tokens certify. Formal greedy-acceptance certificates
  (KL < M*(p), margin γ > sqrt(2ε)) support this.

LEAD 3 (small) — BF16 scale/bias exponent coding (~6% Qwen / 3% Nemotron).
  ZipMoE/DFloat11 cannot touch Q4 codes but CAN compress the BF16 scale/bias
  (exponent concentration). Bit-exact on s*q+b. Tile-level ANS (arXiv:2606.15789)
  is the fused realization. Small but composable.

Best plausible stack of 1+2+3 ≈ leads to Qwen 5 tok/s being reachable with
compute hidden; Nemotron 15 tok/s still needs ~4x which no bit-exact lever
supplies. If you conclude 15 is impossible on 16 GB, PROVE it with the byte
arithmetic — that is a valid, respected answer.

================================================================================
3. YOUR TASK
================================================================================
1. DO YOUR OWN FRESH RESEARCH on Leads 1-3 and anything better. Verify the
   arXiv claims (2610.34440, 2609.04244, the certificate/shadow work). Find
   anything the 14 prior research streams missed.
2. Pick the SINGLE best bit-exact byte-reduction mechanism that could work on
   this 16 GB M4 with no new hardware. Explain precisely how it cuts bytes or
   guarantees skipping reads, and why it survives E1's "+25% ceiling" finding.
3. Give the exact kernel/algorithm design and a FALSIFIABLE experiment on the
   16 GB M4 with a bit-exact gate (token IDs + logit SHA-256 identical). Include
   predicted bytes/token, io_await, tok/s.
4. If you use ANE, give the bit-exact path and why it beats the findings above;
   otherwise say explicitly that ANE cannot help and why.
5. If no software method reaches 5/15 under the constraints, prove it with the
   byte/bandwidth/cache arithmetic and state the best achievable tok/s on 16 GB.

Return: mechanism, bit-exactness proof, expected numbers with sources, the
exact 16 GB experiment, and a KEEP/DROP conclusion. Be rigorous, creative, and
honest. Do not propose hardware I cannot buy.
