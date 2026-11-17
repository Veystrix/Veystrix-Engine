================================================================================
OPUS — MIRACLE HUNT: reach Qwen 5+ tok/s and Nemotron 15+ tok/s on a 16 GB M4
================================================================================
SELF-CONTAINED. Every number, failed attempt, and new lead is inlined. No file
access or prior conversation is needed. Paste this whole message as the opener.

You are asked to find a genuine BREAKTHROUGH that others missed — the "miracle"
that reaches the targets, or to prove with numbers that none exists. Be creative
AND rigorous. The owner is commercializing this and is willing to be the FIRST
to try something. Any no-quality-loss method that reduces SSD I/O wait is approved.

================================================================================
1. THE SYSTEM AND TARGETS
================================================================================
HARDWARE: MacBook Air M4, 16 GB unified memory, ONE internal NVMe SSD (~3.0-3.3
GB/s sequential, ~6 GB/s double-buffered), fanless, AC power. Validation on THIS
machine first. Larger memory / 2nd Mac are in scope for research now (owner wants
any method) but the 16 GB result is what matters first.

ENGINE: Veystrix, private macOS app. Swift UI -> HTTP -> Python MLX engine. MLX
0.32.2, full C++ API available, app already ships a native .dylib. Engine is ~94%
Python. Qwen kernels come from mlx-vlm. No in-repo Metal.

MODEL A (primary): Qwen3.8-Flash-Next, arch qwen4_exp, 125B MoE, 512 experts/layer,
top-10, 48 layers, GatedDeltaNet + QSA attention, Q4 affine G32. Decode 2.3 tok/s
(408 MB physical/token, 300 ms/token: io_await 156.9ms=52%, host 18ms=6%, GPU 27%).
Expert = 3,072,000 B; 24,576 total. TARGET 5+ tok/s.

MODEL B (secondary): Nemotron 3.5 Lightning, arch nemotron_h, 30B-A3B, 23 MoE
layers, 128 routed experts/layer, top-6, Q4 affine G64. Decode ~7 tok/s (436 MB/
token, 181 ms/token: I/O 123ms=68%, host 27.8%). Expert = 5,612,544 B. TARGET 15+.

HARD RULE: NO QUALITY LOSS. Bit-identical tokens, BF16 rounding, Q4 G32/G64, exact
top-10/512 and top-9/240 routing. Any logit/token change = OUT.

================================================================================
2. THE HARD MEASUREMENTS (do not ignore these)
================================================================================
Physical-wall: 408-436 MB/token at ~3 GB/s = 130-145 ms/token I/O floor. To reach
15 tok/s (Nemotron) needs ~96% cache hit; 5 tok/s (Qwen) needs <=148 MB/token.

DEFINITIVE steady-state cache ceilings from REAL 1,200-token guarded traces
(short 68/96-token traces were a first-touch artifact — do not repeat that error):
  Nemotron OPT: 7.68@2GB, 10.03@4GB, 12.08@6GB, 13.82@8GB, 15.22@10GB, 16.27@12GB
  Nemotron S3-FIFO: 5.27@2GB, 6.92@4GB, 8.55@6GB, 10.36@8GB, 12.38@10GB, 14.40@12GB
  Qwen OPT: 4.43@8GB, 5.22@12GB, 5.80@16GB, 6.25@20GB
  Qwen S3-FIFO: 3.22@8GB, 3.92@12GB, 4.60@16GB, 5.21@20GB
So with a REALIZABLE 16 GB pool (<=~4 GB), ceilings are ~5-7 Nemotron / ~2-3 Qwen.
Targets need >10 GB unique expert RAM AND a near-Belady policy. A cache ALONE cannot
do it on 16 GB. You must find a mechanism OUTSIDE pure caching, or combine several.

================================================================================
3. EVERYTHING ALREADY FAILED (with numbers — do NOT propose as novel)
================================================================================
DECODE/IO: expert prefetch 2.39->2.38; decode overlap 2.39->2.41 (band), early submit
-8.2%, early shared expert -10%; perfect predictor + overlap tied; I/O workers 6-32
all 2.55-2.57 (SSD saturated); cache policies LRU/cost-aware/least-stale tied
(S3-FIFO best); pool 2->4 GB = -1.61 tok/s regression, 1 GB beat 2 GB, production
auto-sizes pool to 0; native MTP all depths slower (depth2 exact verifier 2.298->
1.924, 16%); lookup drafting v2 7.30->6.15; expert packing +9% in-process but no
server gain +70 GiB; F_NOCACHE 8% slower; F_RDAHEAD=0 1.3% in band; wired-limit
noise/guard; MLX command-buffer caps regress here; mx.async_eval 311 vs 314; ANE all
rejected (GPU 74.9 vs ANE 23.2 tok/s, INT4 unsupported, no bit-exact ANE layer).
K=2/K>1 read-sharing 16% slower. Sorted gap-merged pread: TTFT -0.69s only.

PREFILL: route-union no gain; PLE row prefetch slower; QSA tile tuning no gain;
qsa-fast-prefill fails quality; pipelined tiles tied; dequantize-all-history slower.

COMPRESSION: lossless ZSTD/LZ4 on Q4 codes only 0.9-2.1% — CLOSED by byte-exact
study (nibble entropy 3.86/4.0 bits, whole-artifact ceiling 1.04-1.05x, dictionary
zstd expands). All requantization lossy.

ALREADY BUILT: exact-speed profile on by default (dense GEMM, fused gather/SDPA,
single-chunk layer-major, merge-small-tail, ngram-dedup, S3-FIFO, prefix cache in
agentic); C++ pager scaffold (cpp/decode/, bit-exact tests); I/O residency v0.2
toolkit (simulator, 74 tests).

================================================================================
4. NEW LEADS FROM THE 8-STREAM RESEARCH (this is where the miracle may be)
================================================================================
Ranked by plausibility x payoff. Full detail with URLs is in the 8 note files.

A. TRAINED PREROUTER (Edge0, arXiv 2609.18063) — STRONGEST. A small trained head
predicts the NEXT layer's routing one token ahead; the prediction is COMMITTED as
the routing (staged set == routed set, nothing dropped), so expert reads overlap
compute. Published +80-84% decode on unfitted machine; Prerouter swap +59% decode
on Mac mini M4 Pro; 35B MoE ~20 tok/s in 3 GiB on 24 GB. Quality concern: replacing
routing costs benchmark points — Edge0 uses an unmerged recovery LoRA. THIS IS a
different, far stronger mechanism than our rejected heuristic next-layer prefetch.

B. SpecFit (arXiv 2610.15417) — training-free, target model untouched, selects the
verification subtree by EXPERT-UNION MARGINAL COST. 1.97-2.51x over EAGLE-3 on
offloaded Qwen3-30B-A3B; gains GROW as cache shrinks (21.6%->31.6%) and on slower
links (+18% on RTX 4090); 12-22% fewer H2D experts/token; tolerates 60%-accurate
route predictors. Best fit because it needs no training and attacks the exact
bottleneck. Combine with A or use alone.

C. ZipMoE (arXiv 2601.21198) — bit-field decomposition into low-entropy exponent
bits (only 11-15% of symbol values used); ZSTD to 68% of original near the 66%
Shannon floor; split-state cache gives 2x tensor coverage at same memory by caching
only the incompressible half and reconstructing the rest on spare CPU with
decompression hidden under I/O. Up to 72.8% latency reduction. Explicitly targets
Apple Silicon UMA ~3.5 GB/s SSD — essentially Veystrix's hardware. NOTE tension:
another byte-exact Q4_K study measured entropy 3.86/4.0 bits and only 1.04-1.05x
whole-artifact ceiling; ZipMoE likely works on BF16-style exponent fields, NOT our
Q4 codes. MUST reconcile this conflict with a Veystrix-specific measurement.

D. F_RDADVISE / F_RDADVISEV revisited (OS stream, rank 1 there) — TRUE async Darwin
prefetch (24 MB issued in 0.5 ms, resident ~5 ms later); MADV_WILLNEED is synchronous
on Darwin. Earlier EPERM was likely a wrong struct/fd type (production uses
struct radvisory{off_t ra_offset; int ra_count;} on macOS arm64); F_RDADVISEV batches
64 ranges. A project probe measured 3.61-3.78 GB/s (above the ~3 GB/s wall) and was
deferred to a real-decode test — that test is the single highest-value next
experiment. Advice-only = bit-exact. WARNING: our speculative reads once caused
token divergence — gate any prefetch on bit-identical output.

E. llama.cpp #27861 exact LRU slot cache — zero dummy slot + summed partials, no new
kernels: 18.4->24.2 tok/s (+31%) on Flash-Next; reproduced 17.6->25-29 even on PCIe
3.0 + slow DDR4. 48-96 slots/layer. Our production pool is auto-sized 0 — this is
the concrete cache to rebuild. Note: 1bit.gg's "28-39 tok/s hot-expert pinning" was
a MISATTRIBUTION (that was Coder-30B); Flash-Next exact streamed is 3.5-6.2 tok/s,
24 resident, 40-49 resident WITH native MTP n=3. Static hot pinning FAILS for
Flash-Next (top-32 covers ~10%); recency/LRU works.

F. Single-sync on-device slot resolution (llama.cpp PR #31469) — the genuine gap:
per-layer host sync+topk readback costs ~8 ms/layer x48 = ~340 ms/token; fix is
refresh all 48 layer tables in ONE sync; slot remap bit-exact (0.000e+00 diff).
This is distinct from rejected early-submit. Combine with A/B for I/O batching.

G. MTP as I/O BATCHING window — native MTP n=3 takes resident Flash-Next from 24 to
40-49 tok/s; leaves drive bytes/token unchanged but raises resident decode 1.6-1.9x.
Rejected MTP depths here were on the SSD-bound path; with high hit rate (A/B/E/F)
verification batching could finally pay.

H. DraftExpert (arXiv 2607.24434) — one resident draft expert per layer, fixed
footprint, expansion-aware truncation + prefetch, EXACT verification, 1.45x avg
CPU->GPU and Flash->NPU; verify cost 2.35x->1.25x of one AR step.

I. Within-layer co-selection concentration (arXiv 2610.09372, bit-exact by
construction): at top-2, 0.8% of expert pairs are co-selected by 42% of tokens;
correlated placement + dedup dispatcher removes up to 58% of dispatched rows WITHOUT
changing router assignments or expert computation. Strong byte-reduction lead when
combined with co-access-aware on-disk layout.

J. Farthest-future eviction (NeurIPS 2026, bit-exact): changing only WHICH expert is
evicted (routing untouched) cuts transfers 2.1x, within 3.5% of optimal; 16 tokens
of future demand recover 96%, a frequency rule using past recovers only 27% — this
explains why our cache policies tied (they lacked future knowledge). SpecFit/prefetch
provide that future knowledge.

K. Lossless expert-base + residual/delta coding (radical #1): store one exact base
expert per layer, others as lossless integer residuals in Q4 code space; bit-exact by
construction, plausibly 2-3x fewer bytes, no custom hardware. Motivated by PuzzleMoE
(experts are similar) but EXACT where that is lossy. Falsify with one offline script
measuring residual zero-density/entropy. NOTE tension with the 3.86/4.0-bit entropy
result — measure before believing.

L. Distributed: MLX/JACCL tensor parallel across 2-4 Macs over Thunderbolt 5 RDMA
(macOS 26.2): MoE decode 24.99 tok/s on 2x M5, 36.73 on 4x; ~1.7x/2 nodes, ~2.6x/4.
EXO two-Mac test on a model that FITS was 18-23% SLOWER — pooling pays only when it
keeps experts off SSD. Bevy-style runahead scheduling up to 3.46x vs llama.cpp, no
expert prediction. 2x M4 16 GB -> 25 tok/s on Qwen3.6-35B (unverified by our research).

M. Lossless expert-pack + frequency/recency caching (SSD-LLaMA arXiv 2609.18110):
8.89->5.93 GB/token (33.2% physical traffic cut), bit-identical reconstruction.
Fused direct-consume (PagedMoE) 3.94-8.79x over materializing BF16 — but PagedMoE's
dynamic precision is LOSSY; salvage only the direct-consume kernel idea.

N. Context-hash memoization of MoE FFN calls — unbounded payoff on repeated prompts,
zero on novel. Cross-request persistent expert server (EaaS/FaaSMoE) pays only multi-user.

IMPORTANT ANALYTIC CORRECTIONS:
- M4 bandwidth diagnostic (arXiv 2610.34440): sharing across n tokens wins only when
  byte-share u(n) < bandwidth-ratio r(n). On Metal r(2)~0.37-0.59 while u(2)~0.72-0.83
  => depth-2 sharing is PROVABLY slower (explains our -16%); crossover n=5-8 or never.
- Lossless representation cannot reach 15: even a perfect 2x byte halving gives ~4.6
  tok/s from 2.3; 15 tok/s needs ~6.5x. Compression is necessary but NOT sufficient.

================================================================================
5. YOUR TASK
================================================================================
Find the breakthrough, or prove none exists. Concretely:
1. Pick the SINGLE most promising combination from the leads (A+B+E+F+G is the
   current front-runner: trained/learned next-layer routing + union-cost-aware exact
   verification + exact LRU slot cache + single-sync slot remap + MTP batching — all
   bit-exact except A's routing replacement which needs a quality fix).
2. Explain precisely why it beats the ~3 GB/s wall or the steady-state cache ceiling.
3. Design a falsifiable experiment we can run on the 16 GB M4, with a bit-exact gate:
   predicted bytes/token, io_await, tok/s, and the digest comparison.
4. If it needs training (A), propose the smallest training recipe and how to keep
   output bit-identical (or quantify the quality cost honestly).
5. If you conclude the targets are impossible on 16 GB, prove it with the OPT bounds
   and the entropy/byte math, and state the exact hardware that WOULD suffice.

Return: mechanism, bit-exactness verdict, expected numbers with sources, the exact
16 GB experiment, and a KEEP/DROP recommendation. Be the first to try something if
that is what it takes. Rigorous and positive.
