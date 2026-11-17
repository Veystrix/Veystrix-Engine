================================================================================
VEYSTRIX — MASTER HANDOFF: make Qwen decode 5+ tok/s and Nemotron 15+ tok/s
================================================================================
SELF-CONTAINED: every measured number, every failed attempt, all context is
inlined below. No file access or prior conversation is required. This is the
complete state of a multi-session effort. Read it all before proposing anything.

The owner wants: any method that reaches the targets with NO quality loss. You
(Opus) are asked to "make it work somehow." Bring any idea from anywhere. The
constraints and the hard measurements below are real; do not ignore them, but
do look for the gap in them. Be rigorous and honest: a falsifiable negative
result with numbers is valuable, but a genuine speedup is the goal.

================================================================================
1. THE SYSTEM
================================================================================
HARDWARE: MacBook Air M4, 16 GB unified memory, ONE internal SSD, fanless, on
AC power, Low Power Mode off. Internal NVMe ~3.0-3.3 GB/s sequential, ~6 GB/s
double-buffered. This 16 GB machine is the ONLY validation target first.

APP: Veystrix, private macOS app. Swift UI (5.5k lines) talks HTTP to a Python
MLX engine on 127.0.0.1:port/v1/chat/completions. All GPU kernels are MLX /
mlx-vlm (C++/Metal underneath); no in-repo Metal shaders. App ships a native C
dylib already. Engine is >90% Python (~76k first-party lines; ~8.9k-line hot path).

MODEL A — Qwen3.8-Flash-Next (PRIMARY):
  arch qwen4_exp, 125B MoE, 512 experts/layer, top-10, 48 layers,
  GatedDeltaNet + full QSA sparse attention, Q4 affine G32, MTP head.
  Decode ~2.3 tok/s production / 3.33 tok/s short-prompt harness (300.1 ms/token).
  Reads ~408 MB physical/token. Expert = 3,072,000 bytes. 24,576 experts total.

MODEL B — Nemotron 3.5 Lightning (SECONDARY):
  arch nemotron_h, 30B-A3B, 52 layers (23 MoE layers), 128 routed experts/layer,
  6 experts/token, Mamba/SSM + attention, Q4 affine G64.
  Decode ~6.7-7 tok/s. Reads ~436 MB/token. Expert = 5,612,544 bytes.
  2 GB pool -> only 43.7% cache hit (5,792 hits / 7,456 misses).

TARGETS: Qwen decode >=5 tok/s, Nemotron decode >=15 tok/s, better prefill/TTFT,
NO quality loss (bit-identical tokens, BF16 rounding, Q4 G32/G64, exact routing).

================================================================================
2. THE MEASUREMENTS — where the time goes (all guarded, digests verified)
================================================================================
QWEN host-window profile (300.1 ms/token, 3 passes identical token IDs):
  io_await (blocked on SSD)  156.86 ms = 52.3%   <-- THE WALL
  all host work               18.22 ms =  6.1%   (route_tolist 12.5, to_mx 2.9,
                                                  keep_loop 1.9, moe_issue 0.54, plan 0.36)
  GPU ~27%. Removing 100% of host work => 3.33->3.55 tok/s (1.065x). NOT Python-bound.

NEMOTRON decode profile (181.4 ms/token, 5.51 tok/s, 96 tokens, 23 MoE calls/token):
  expert_dispatch+wait       123.1 ms = 67.9%   <-- THE WALL
  route_sync+copy             50.5 ms = 27.8%
  other                        7.8 ms =  4.3%
  Removing 100% host sync => 5.51->7.64 tok/s (1.386x). I/O dominates.

THE PHYSICAL WALL: ~408-436 MB/token at ~3 GB/s = ~130-145 ms/token minimum,
decode ceiling near 7 tok/s on this machine. To reach 15, must roughly halve
bytes/token (cache hit ~44% -> ~95%) without more memory, or hide reads.

================================================================================
3. THE SIMULATION-BACKED CEILINGS (Belady OPT, calibration-independent)
================================================================================
A trace-driven cache simulator (veystrix-io-residency) was built, verified
(54 tests pass), and fed REAL route traces from guarded decode runs. Belady OPT
is the best any cache policy can theoretically achieve — it does NOT depend on
simulator calibration. Using the measured constants:

NEMOTRON (3.541 MB/ms effective I/O, 58.3 ms non-I/O):
  pool 2 GB (12% cover): OPT 65.8% hit -> 7.52 tok/s
  pool 4 GB (25%):       OPT 77.6%    -> 9.32
  pool 6 GB (37%):       OPT 82.7%    -> 10.41
  pool 8 GB (50%):       OPT 85.4%    -> 11.09
  pool 10 GB (62%):      OPT 85.4%    -> 11.09  (flat)
  => Even a PERFECT cache caps at ~11 tok/s. Curve flattens at ~8 GB.

QWEN (2.603 MB/ms effective I/O, 143.24 ms non-I/O; 68-token trace):
  2 GB (2.7%): OPT 57% -> 2.59 tok/s
  4 GB (5.4%): OPT 70% -> 3.18
  8 GB (10.9%): OPT 81% -> 3.97
  => Even a PERFECT cache at 8 GB caps at ~4 tok/s, below the 5 target.
  Qwen's 512-expert/layer working set is too large for 16 GB residency.

IMPLICATION: no cache policy or residency trick reaches either target on this
16 GB M4. These ceilings are simulation-backed, not opinions. BUT: G0 calibration
failed for both (traces short: 68 Qwen / 96 Nemotron tokens vs >=1000 wanted;
simulator S3-FIFO params don't match engine). OPT bounds themselves are sound.

================================================================================
4. EVERYTHING ALREADY TRIED AND FAILED (with measured numbers — DO NOT REPEAT)
================================================================================
DECODE / I-O:
- Expert prefetch: 2.39->2.38. Decode overlap: 2.39->2.41 (within band); early
  submit -8.2%; early shared expert -10%. Next-layer router prefetch/lookahead: tied.
- I/O workers 6/8/16/24/32: all 2.55-2.57 (SSD saturated).
- Cache policies other than S3-FIFO: cost-aware 2.45/lru 2.39/s3fifo 2.54; least-stale tied.
- Pool size: 2->4 GB = -1.61 tok/s resolved regression; 1 GB beat 2 GB (6.94 vs 6.56).
  Production auto-sizes pool to 0 on 16 GB. BIGGER CACHE IS NOT THE ANSWER.
- Perfect expert predictor + overlapped reads: tied.
- Native MTP/speculative all depths: depth1 2.36 vs 2.58; depth2 2.17; depth3 1.35;
  exact-verifier depth2 2.298->1.924 (16% slower, digest identical). Lookup v2 7.30->6.15.
- Nemotron read-ahead/stale-drop: 5.14/5.04/4.96 vs 6.67 kept. Parallel Mamba failed exactness.
- FLASHNEXT_COMPILE_GDN 2.41->1.88. mx.async_eval 311 vs 314. Cache-aware routing +8.3%
  but failed quality gate. Expert packing +9% in-process, no server gain, +70 GiB disk.
- F_NOCACHE (O_DIRECT) 8% SLOWER. F_RDAHEAD=0 1.3% below band. Wired-limit sysctl
  +13.5% standalone/-0.4% harness; MLX wired limit hit memory guard.
- MLX command-buffer caps: Nemotron -0.5 tok/s, Qwen TTFT +0.86s (both resolved regressions).
- ANE: GPU 74.9 vs ANE 23.2 tok/s; ANE bw 7.7 vs 94 GB/s; INT4 unsupported; 2 GB cap;
  Core ML 3.3% slower; palettized 4-bit ruined accuracy; no bit-exact ANE layer exists;
  FusionML loses to MLX on M4. ANE is closed.

PREFILL:
- Route-union through router 30.5 vs 29.9s (no gain). PLE row prefetch 33.8->36.8s (+87 GB).
- QSA gather tile tuning no gain. qsa-fast-prefill fails quality gate. Pipelined tiles tied.
- Dequantize-all-history 0.98 vs 0.73s (4x bytes). Fused-gather rounding change not bit-exact.
- Block-parallel Mamba digest mismatch. Nemotron 4K chunk +6% but memory pressure.

COMPRESSION: lossless ZSTD/LZ4 on 4-bit weights saves only 0.9-2.1% (RPI 2025
arXiv:2503.18869) — no redundancy. All requantization (GPTQ/AWQ/FP8) lossy => OUT.

BIG PORTS RULED OUT: io_uring pager (O_DIRECT rejected; overlap rejected), full-layer
staging+2nd SSD (only ONE physical SSD; portable staging already on), rANS 3-tier
(4-bit 0.9-2.1% compressible), SwiftLM rewrite (8.5x from a 0.58 tok/s baseline; we're
already 2.3-7), last-row prefill (changes logits — fails bit-exact gate).

================================================================================
5. WHAT IS ALREADY BUILT / DEPLOYED (build on it, don't redo)
================================================================================
- SORTED GAP-MERGED PREAD: models/flashnext/store.py _pread_into_merged, flag
  FLASHNEXT_SORTED_MERGED_PREAD=1 (off by default). A/B: TTFT -0.69s resolved win,
  decode unresolved. Helps prefill only. 22 unit tests pass.
- C++ PAGER SCAFFOLD: cpp/decode/ io_pager.{hpp,cpp} + native_read.c reuse;
  deterministic id-remap + dedup + batched read + zero-copy mx::array. 2 bit-exact
  ctests pass. Builds against venv MLX 0.32.2. Live hookup needs engine changes.
- EXACT-SPEED PROFILE ON BY DEFAULT: dense GEMM, fused gather/SDPA, single-chunk
  layer-major, merge-small-tail, ngram-dedup, S3-FIFO, qsa-shared-dense-kv. Prefill
  gains +14-30%, +36%, -6.5%, -3.5%, -2.4%; 24/24 bit-identical quality gate.
  Agentic mode has prefix caching ON.
- I/O RESIDENCY TOOLKIT: veystrix-io-residency/ (simulator, residency probe via
  mincore, physical diskstats, single-tier pool). 54 tests pass. Real Qwen trace +
  Nemotron trace captured under guard. Documents: IO_RESIDENCY_STAGE0.md,
  IO_RESIDENCY_QWEN.md. The single-tier hypothesis (F_NOCACHE + pool grown by
  measured duplicated page-cache bytes at constant RAM) is NOT yet A/B tested on
  hardware, but its ceiling is bounded by the OPT curves above.

================================================================================
6. EVIDENCE FROM THE SAME MODEL / HARDWARE (leads worth checking)
================================================================================
Jakevin/moe-offload (same Qwen model, M4 Pro 64 GB): reference 2.52; v1 Python
pread 8.51; v2 native pread+fewer syncs 9.17; v2+router prefetch 9.73 tok/s (84->95%
hit). Bit-exact C++ runner: 9.0-9.6 tok/s — i.e. optimized Python already captured
nearly all the C++ gain. Fully resident llama.cpp GGUF = 24.6 tok/s. Nemotron Bolt
74B-A9B on 64 GB full baseline 13.9 tok/s (caching LOST there).
Overlap Jetson: 1.94->2.95 tok/s (+32%, bit-identical). TinyGiant M1 16 GB:
double-buffered 2.09x, expert-contiguous re-layout 36x fewer page faults; with Q4
experts pipelining hides ALL I/O. Expert cache break-even rule: pool >= ~1/3 of
expert bytes to help; LFU sim 34% residency -> 98.8% hit.

================================================================================
7. HARD RULES
================================================================================
1. NO QUALITY LOSS: bit-identical token digests, BF16 rounding, Q4 G32/G64,
   exact top-10/512 and top-9/240 routing. Any logit/token change = OUT.
2. 16 GB M4 first. "Add RAM / 2nd Mac" are the owner's DEFERRED levers, not your task.
3. Guarded runs with explicit permission: scripts/flashnext/real_run_guard.py
   --floor-gib 1.5 --swap-growth-mib 2048 --compressor-growth-mib 4096 --log <jsonl> -- <bench>
   Preserve all results/**. Ask before hardware runs.
4. One change at a time, A/B vs current baseline, token+logit digest gate.
5. Document results, guard JSONL, model/executable identity, resolution band,
   and a clear KEEP/DROP. Negative results with numbers are valid.

================================================================================
8. YOUR TASK — "make it work somehow"
================================================================================
The straightforward methods are exhausted and the OPT ceilings are real. To
reach 5+/15+ on 16 GB with no quality loss you must find something outside the
above — explicitly open directions (think hard, question assumptions):
- A bit-exact way to CUT bytes/token by >2x on this hardware. Is there expert
  duplication/reuse across adjacent tokens or layers the traces miss? Cross-layer
  expert co-activation, n-gram/prompt-lookup of expert sets, skill-based routing
  locality that the short traces don't reveal?
- A genuinely different on-disk representation that is lossless w.r.t. the Q4
  values but read-efficient (per-expert records, page-aligned slabs, reordered
  shards) — explain why prior "expert packing" failed and why yours won't.
- Hiding latency via the router's prefetch window WITHOUT the rejected overlap
  mechanisms — prefetch only the exact experts, timed earlier.
- A bit-exact H2D/DMA trick on Apple Silicon (managed memory, MTLResidencySet,
  zero-copy) that reduces the effective read cost below the ~3 GB/s wall.
- Reconsider whether the OPT simulation's constant model is pessimistic (it uses
  measured effective bandwidth and non-I/O time; could pipelining hide more).
- Any research paper on MoE offloading / SSD streaming / expert tiering that is
  bit-exact and could plausibly halve bytes/token. Read it, extract mechanism +
  numbers, design a falsifiable 16 GB test.

APPROACH: inspect the measurements first, pick ONE best-evidenced idea, implement
opt-in (off by default), bit-exact gate, guarded A/B, honest keep/drop.

RETURN: changed files + flag, exact benchmark command + guard, result dirs + JSONL,
model/executable identity, digest evidence, MB/token and io_await before/after
with resolution band, and a clear verdict. If you conclude the targets are
physically impossible on 16 GB, prove it with numbers (OPT bounds) and say so.

Be rigorous, positive, and creative. The owner wants a real shot and has approved
any no-quality-loss method that reduces I/O wait.
