================================================================================
FABLE — FULLY SELF-CONTAINED BRIEF: Attack the SSD I/O wait in Veystrix
================================================================================
(Everything you need is in this message. No file access is required or assumed.
 Every number below is from a real guarded measurement on Veystrix's own machine.
 Copy/paste this entire message as the opening prompt.)

--------------------------------------------------------------------------------
0. YOUR MISSION AND TONE
--------------------------------------------------------------------------------
You are solving the ONE remaining real bottleneck in Veystrix: SSD expert-read
wait during LLM decode on a 16 GB Apple M4. Be POSITIVE and constructive. Every
failed attempt listed below is a MEASURED narrowing of the search space, not a
reason for pessimism — each one closed a specific door with a number, and that is
valuable: it means we can aim precisely. There is genuine, high-value work left.

The owner has explicitly approved: "any experimental result or research-paper
method is approved if it reduces I/O wait, as long as there is no quality loss."
So bring ideas from anywhere — arXiv papers, other inference engines (llama.cpp,
MLX, vLLM), OS-level I/O, file layouts, cache policies, scheduling, prefetch
timing. Find the mechanism that cuts bytes read per token or hides the read
latency, prove it bit-exact, and it will be implemented.

--------------------------------------------------------------------------------
1. HARDWARE, MODELS, TARGETS
--------------------------------------------------------------------------------
HARDWARE: MacBook Air M4, 16 GB unified memory, ONE internal SSD, fanless, on AC
power with Low Power Mode off. Apple internal NVMe: ~3.0-3.3 GB/s sequential,
~6 GB/s double-buffered. This is the ONLY machine we validate on first.

ENGINE: Veystrix, a private macOS app. A Swift UI talks HTTP to a Python MLX
engine on 127.0.0.1:port/v1/chat/completions. All GPU kernels are MLX / mlx-vlm
(C++/Metal underneath); there are no in-repo Metal shaders. The app already ships
a native C dylib (native_read.bundled.dylib), so native code is deployable.

MODEL A - Qwen3.8-Flash-Next (the primary target):
  - arch qwen4_exp, 125B MoE, 512 experts/layer, top-10 routing, 48 layers
  - hybrid: GatedDeltaNet (linear attention) + full QSA sparse attention
  - weights: Q4 affine, group_size=32 (Q4 G32), MTP head present
  - decode: ~2.3 tok/s in production; short-prompt harness 3.33 tok/s, 300.1 ms/token
  - reads ~408 MB physical per token (harness: 408.3 / 405.6 / 408.1 MB in 3 passes)

MODEL B - Nemotron 3.5 Lightning (secondary target):
  - arch nemotron_h, NemotronHForCausalLM, 30B-A3B, 52 layers, hidden 2688
  - 6 experts/token, Mamba/SSM + attention, weights Q4 affine group_size=64 (Q4 G64)
  - decode: ~6.7-7 tok/s; ~436 MB read per token
  - 2 GB expert pool gives only 43.7% cache hit rate (5,792 hits / 7,456 misses)

TARGETS (owner): Qwen decode >=5 tok/s, Nemotron decode >=15 tok/s, no quality
loss. Also improve prefill/TTFT. Validate on 16 GB first; larger Macs only after.

--------------------------------------------------------------------------------
2. THE DECISIVE MEASUREMENTS — what is actually slow
--------------------------------------------------------------------------------
These are the most important numbers in this brief. They tell us WHERE to work.

QWEN host-window profile (guarded, 3 passes, identical token IDs; per token):
  pass 1: 300.1 ms/token, 3.33 tok/s, 408.3 MB/token
  pass 2: 305.1 ms/token, 3.28 tok/s, 405.6 MB/token
  pass 3: 300.3 ms/token, 3.33 tok/s, 408.1 MB/token
  Host windows per token (last pass):
    route_tolist     12.50 ms exclusive (48 calls/token)
    to_mx_host        2.89 ms exclusive
    keep_loop         1.93 ms exclusive
    moe_issue_host    0.54 ms exclusive
    plan_host         0.36 ms exclusive
    io_await        156.86 ms SHARED (blocked on the drive), 48 calls/token
    TOTAL            18.22 ms exclusive host + 156.86 ms shared I/O
  => I/O wait = 156.86/300.1 = 52.3%; all host work = 18.22/300.1 = 6.1%.
  Even if 100% of host work vanished: 300.1 -> 281.9 ms = 3.33 -> 3.55 tok/s
  (1.065x). CONCLUSION: Qwen is I/O-bound, not Python-bound.

NEMOTRON decode profile (guarded, 96 tokens, digests captured; per token):
  decode 17.41 s / 96 tokens = 5.51 tok/s = 181.4 ms/token; 2,208 MoE layer calls (23/token)
  expert_dispatch_and_wait = 123.1 ms/token = 67.9%  <-- THE I/O WAIT
  route_sync_and_copy      =  50.5 ms/token = 27.8%  (host sync + route copy)
  other                    =   7.8 ms/token =  4.3%
  cache: 5,792 hits / 7,456 misses = 43.7% hit; 41,847 MB read / 96 = 436 MB/token
  If ALL host route sync vanished: 181.4 -> 130.9 ms = 5.51 -> 7.64 tok/s (1.386x).
  CONCLUSION: Nemotron is dominated by I/O wait (67.9%) and a badly undersized cache.

THE PHYSICAL WALL: ~408-436 MB/token at ~3 GB/s effective is ~130-145 ms/token
minimum, i.e. a decode ceiling near 7 tok/s on this machine. To reach 15 tok/s
you must roughly HALVE the bytes read per token (raise cache hit from ~44% toward
~95%) without increasing memory pressure — OR hide the reads almost completely.
That is the problem to attack. Be honest if 15 is physically impossible on 16 GB,
but pursue it hard and let the measurements decide.

--------------------------------------------------------------------------------
3. HARD RULES (non-negotiable)
--------------------------------------------------------------------------------
1. NO QUALITY LOSS. Output must be bit-identical: same token digests, BF16 rounding
   boundaries, Q4 affine G32/G64 weights, exact top-10/512 and top-9/240 routing.
   A method that changes logits or tokens is OUT regardless of speed.
2. 16 GB M4 FIRST. Do not propose "just add RAM / another Mac" as the solution —
   those are the owner's explicitly deferred levers, not your task.
3. GUARDED RUNS ONLY, WITH EXPLICIT PERMISSION. The guard wrapper is:
   scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 2048
     --compressor-growth-mib 4096 --log <jsonl> -- <your benchmark command>
   Ask the owner before every hardware run. Preserve all results/** directories.
4. ONE CHANGE AT A TIME, A/B vs the current accepted baseline, token (and where
   possible logit) digest gate ON.
5. Document everything: result dirs, guard JSONL, exact model/executable identity,
   resolution band, and a clear keep/drop recommendation.
6. The measured baseline for Qwen A/B is ~2.1-2.3 tok/s; for Nemotron ~6.4-7 tok/s.

--------------------------------------------------------------------------------
4. WHAT HAS ALREADY FAILED — DO NOT REPEAT (every entry has a measured number)
--------------------------------------------------------------------------------
(This is the authoritative dead-end registry, embedded verbatim in substance.)

DECODE / I-O:
- Expert prefetch FLASHNEXT_EXPERT_PREFETCH: 2.39 -> 2.38 tok/s.
- Decode overlap FLASHNEXT_DECODE_OVERLAP: 2.39 -> 2.41 (within band); early
  submit -8.2%; early shared expert -10%.
- I/O workers 6/8/16/24/32: all 2.55-2.57 tok/s — SSD saturated at worker level.
- Cache policies other than S3-FIFO: cost-aware 2.45, lru 2.39, s3fifo 2.54;
  least-stale within band.
- Larger pool: 2->4 GB caused a resolved -1.61 tok/s regression (+0.86 s TTFT,
  +2.14 GB footprint); 1 GB beat 2 GB (6.94 vs 6.56 tok/s). Production already
  auto-sizes the pool to 0 on 16 GB. A BIGGER CACHE IS NOT THE ANSWER on 16 GB —
  memory pressure dominates. The 3 GB stop hits a 2.59 GiB memory floor; 6 GB refused.
- Perfect expert predictor + overlapped reads: +0.1 / -2.9 / -0.7% (tied).
- Native MTP / speculative, all depths (Qwen): depth 1 2.36 vs 2.58 plain; depth 2
  2.17; depth 3 1.35; adaptive 1.45. Depth-2 with exact verifier: 2.298 -> 1.924
  tok/s (0.837x, 16% SLOWER), digest identical. Lookup drafting v2: 7.30 -> 6.15.
- Nemotron read-ahead / stale-drop: 5.14/5.04/4.96 vs 6.67 kept. Parallel Mamba
  failed exactness gate (3.66 vs 3.91). 4K chunks raised MLX peak 3.64 -> 4.47 GB
  and compressor 692 -> 1778 MB.
- FLASHNEXT_COMPILE_GDN: 2.41 -> 1.88 tok/s. mx.async_eval: 311 vs 314 tok/s.
- Cache-aware routing: +8.3% but failed trajectory quality gate.
- Expert packing: +9% in-process but 66.6 vs 66.1 s through server, +70.5 GiB disk;
  Nemotron pack +6% but +15.5 GiB disk. Zero-copy shards +2.3% (marginal).
- F_NOCACHE (macOS O_DIRECT) for experts: 8% SLOWER. F_RDAHEAD=0: 1.3%, below band.
- Wired-limit sysctl: +13.5% standalone, -0.4% in harness. MLX wired limit (2 GB)
  hit the memory guard (swap +1.66 GB, compressor peak 6.8 GB).
- MLX command-buffer caps (MLX_MAX_MB_PER_BUFFER): Nemotron decode -0.5 tok/s
  (resolved), Qwen TTFT +0.86 s (resolved regression). Rejected on our M4.
- ANE everything: GPU beats ANE 74.9 vs 23.2 tok/s at batch 1; ANE bandwidth 7.7 vs
  94 GB/s; INT4 unsupported; ~2 GB program cap; Core ML offload 3.3% slower;
  palettized 4-bit ruined accuracy; PyObjC zero-copy segfault; no published bit-exact
  ANE layer (tie-margin is final-head token-ID only); FusionML loses to MLX on M4.

PREFILL:
- Route-union prefill: through router 30.5 vs 29.9 s (no gain).
- PLE row prefetch: 33.8 -> 36.8 s, +87.3 GB reads.
- QSA gather tile 16/32 vs 8: no difference; tile 128 slower/more memory.
- qsa-fast-prefill: changes 4/24 token streams, fails quality gate.
- Pipelined attention tiles: 42.8/43.1 vs 43.7/43.0 s end-to-end (tied).
- Truncating invisible keys: bit-equal but time unchanged. Compiling prefill
  injection: bit-equal, ~25 ms over 4.3K.
- Dequantize-all-history + bf16 gather: 0.98 s vs 0.73 s (4x bytes).
- Fused-gather rounding change: not bit-identical (3.4M elements differ).
- Nemotron 4096 chunk: +6% but memory pressure. Block-parallel Mamba: digest mismatch, slower.

COMPRESSION:
- Lossless ZSTD/LZ4 on already-4-bit weights saves only 0.9-2.1% (RPI 2025,
  arXiv:2503.18869) — 4-bit data has almost no redundancy. Bit-plane
  disaggregation needs custom silicon. All requantization (GPTQ/AWQ/FP8/INT4
  variants) is lossy => OUT.

RULED-OUT BIG PORTS (with reasons):
- io_uring/O_DIRECT pager: its O_DIRECT is already rejected here (F_NOCACHE 8%
  slower); its overlap is rejected; only the id-remap part is portable and it does
  not address the 156 ms/token of drive wait.
- Full-layer staging + 2nd SSD mirror: this machine has only ONE physical SSD.
  The portable part (two-half ring staging, read layer l+1 while l computes) is
  already implemented and on.
- Lossless rANS 3-tier cache: 4-bit weights 0.9-2.1% compressible — cannot cut the bus.
- SwiftLM rewrite: its 8.5x was 0.58 -> 4.95 tok/s from a terrible single-queue
  baseline; our engine already runs 2.3-7 tok/s, so no headroom.
- Last-row-only prefill: changes logits — fails the bit-exact gate.

STILL OPEN (not rejected; do not add to the rejected list):
- Full residency on 32 GB+ for Nemotron (~24 tok/s arithmetic estimate, unvalidated)
- Multi-buffer / 64-bit offsets to exceed Flash-Next's 8 GiB single-uint32 pool cap
- Batch-K (K>1) expert-wise verification with shared reads (union ratio 0.790 Qwen /
  0.826 Nemotron; K=2 measured slower, larger K untested)
- Last-row-only prefill below 2,048 tokens (open TTFT/memory lead)

--------------------------------------------------------------------------------
5. WHAT IS ALREADY DONE / IN PLACE — build on it, do not redo it
--------------------------------------------------------------------------------
- SORTED GAP-MERGED PREAD (opt-in, off by default): implemented in
  models/flashnext/store.py as _pread_into_merged with flag
  FLASHNEXT_SORTED_MERGED_PREAD=1 and FLASHNEXT_PREAD_GAP_BYTES. A/B result:
  TTFT 23.34 -> 22.56 s (resolved -0.69 s win); decode 2.14 -> 2.17 (unresolved).
  22 read-mode unit tests pass. Note: helps PREFILL, not decode.
- C++ PAGER SCAFFOLD in cpp/decode/: io_pager.{hpp,cpp} + native_read.c reuse;
  deterministic expert id-remap + dedup + batched read + zero-copy mx::array handoff.
  Two bit-exact ctests pass (io_pager, pager_read). Toolchain verified: builds
  against venv MLX 0.32.2 (mlx_probe OK). Live hookup needs a ctypes hook into
  build_plan or the deferred full C++ engine port.
- EXACT-SPEED PROFILE is ON by default: curated set for Qwen = kv-tiered4,
  layer-major-hybrid-4k, single-chunk-layer-major, merge-small-tail,
  qsa-shared-dense-kv, qsa-fused-gather, qsa-fused-sdpa, ngram-dedup, dense-gemm,
  s3fifo-pool, FLASHNEXT_QSA_GATHER_TILE=32. Nemotron = VEYSTRIX_NEMOTRON_DENSE_GEMM=1
  (plus prefetch order, expert pipelining, fused Mamba scan, SDPA prefill already on).
  Measured gains: single-chunk layer-major +14-30%, merge-small-tail +36% on 4.3K
  cold prompt, fused gather -6.5%, fused SDPA -3.5%, ngram-dedup -3%, dense GEMM -2.4%;
  combined quality gate 24/24 token-identical PASS.
- Agentic/chat mode has prefix caching ON (FLASHNEXT_PREFIX_CACHE=1 in agentic mode).
- All A/Bs listed in section 4 were actually run under the guard and are preserved.

--------------------------------------------------------------------------------
6. EXTERNAL EVIDENCE FROM THE SAME MODEL / SAME HARDWARE (use as leads)
--------------------------------------------------------------------------------
Jakevin/moe-offload (https://github.com/Jakevin/moe-offload) — SAME MODEL
(Qwen3.8-Flash-Next, 125B, 512 experts/layer, top-10), MLX, M4 Pro 64 GB:
  reference 28 layers resident: 2.52 decode / 5.3 prefill tok/s, 24.9 GiB
  v1 Python pread, no prefetch: 8.51 / 14.8, 84.3% hit, 11.6 GiB
  v2 native pread, fewer syncs: 9.17 / 18.9, 84.3% hit, 10.3 GiB
  v2 + router prefetch top-12: 9.73 / 19.4, 94.7% hit, 8.9 GiB
  per-token profile of v1 (118 ms): GPU 27%, Python+mx.eval sync 41%, SSD 29%,
  bookkeeping 3% — but note v2's native pread removed most of that sync before
  any C++ port: a bit-exact C++ runner reached only 9.0-9.6 tok/s, i.e. optimized
  Python already captured nearly all the gain. Next-layer router prediction
  recall@10/16/24 = 83.6%/94.1%/96.7%. Fully resident llama.cpp GGUF = 24.6 tok/s.
  Nemotron 3.5 Bolt 74B-A9B on 16 GB M4: 4 GB cache = 4.39 tok/s (86.1% hit,
  8.5 GiB peak); 8 GB cache thrashed/aborted. On 64 GB, full baseline 13.9 tok/s
  at 20.8 GiB; caches 4/8/12 GB gave 11.07/11.05/11.28 (caching LOST there).
  C++ decode path bit-exact (raw logits every step + 122/122 cache tensors), 9.0-9.6
  tok/s, dense attention only <=2048 tokens. Prompt cache: 14,937-token prompt
  TTFT 315.7 s no-cache -> 63.9 s cache hit (4.94x), bit-exact.

Overlap/pipelining (bit-exact) elsewhere:
  Jetson Orin 8 GB Gemma 26B Q4: mmap baseline 1.94 -> 2.95 tok/s with overlap
  (overlap alone +32%), logits bit-for-bit identical; its NVMe random ceiling
  ~1.27 GB/s put the I/O-bound limit near 5 tok/s.
  TinyGiant M1 16 GB: sequential 2.97 -> 6.06 GB/s double-buffered (2.09x);
  Qwen3-30B per-layer I/O 263 -> 30 MB (11.4x) reading only active experts;
  expert-contiguous re-layout cut page faults 36x; hit-rate sweep 50% 2.20->2.40,
  75% 2.92->3.81, 88% 3.47->4.10; with Q4 experts pipelining hides ALL I/O
  (compute-bound even at 0% hit) => 4.7 tok/s on ~2 GB RAM.

Cache break-even rule (AbhayXplor/moe-expert-cache-study): a pool helps only when
it is >= ~1/3 of total expert bytes; below that it strongly regresses (-64% at
11% coverage), above it strongly helps (+216% at 71%). LFU sim: 88/256 experts
(34%) -> 98.8% hit. This explains why bigger pools failed on our 16 GB machine.

Wave-Partitional Prefill (llama.cpp PR #25294, MERGED): prefill 2.28 -> 5.65
tok/s, bit-exact under matched kernels+ubatch; each touched expert loaded once
per ubatch. Full-layer staging + 2nd SSD mirror (xMytze/glm53-ssd-streaming):
prefill 29 -> 240 t/s, decode 2.4-2.9 -> 5.8-6.0, 78% hit, ~18 GB/s combined SSD.
io_uring pager (PR #31469): 16.80/21.85 tok/s, 100% byte-exact, 576/576 tokens.

--------------------------------------------------------------------------------
7. YOUR TASK — what to look for
--------------------------------------------------------------------------------
PRIMARY: reduce physical bytes read per token OR hide the read latency, bit-exactly.
Explore (not exhaustive — use your judgment; the owner approved any such method):
  - Raise expert cache hit rate toward ~90% on 16 GB WITHOUT increasing footprint.
    The pool is auto-sized to 0 because bigger pools regress — is there a better
    residency policy, expert duplication across layers, a working-set/locality
    exploit, or a reuse pattern we missed? Note the ~1/3 break-even rule.
  - Expert-contiguous on-disk layout so one expert's 9 sub-slices are one read.
    Prior "expert packing" had no server gain — find out WHY and whether a
    different layout (per-expert records with lossless framing; reordering only
    within a shard; page-aligned expert slabs) changes that.
  - Route-union / co-access-aware layout so one token's 10 experts share cache
    lines/pages, or a layout aligned to SSD read granularity.
  - Prefetch timed to the router's natural window, WITHOUT the specific early
    shared-expert compute / early-submit mechanisms that were rejected.
  - Staged reads: O_DIRECT alternatives — pinned buffers, large-iovec preadv,
    madvise/page-cache-hint variants — that reduce per-read overhead rather than
    bypassing the page cache (F_NOCACHE itself is rejected here).
  - Any research-paper MoE caching/offloading/SSD-streaming/expert-tiering method
    that is bit-exact and could plausibly halve bytes/token. Read the paper,
    extract the mechanism and its numbers, design a falsifiable 16 GB test.

APPROACH:
  1. Inspect the code and the raw measurement files FIRST (paths given above).
  2. Pick the single experiment with the strongest measured or well-evidenced chance
     of lowering io_await or physical MB/token.
  3. Implement behind an opt-in flag, OFF by default, with a bit-exact gate.
  4. Produce a patch + a falsifiable guarded A/B, OR a clean artifact that rules
     the idea out with numbers. A negative result with numbers is a valid outcome.

RETURN (deliverable):
  - Changed files and the exact opt-in flag name.
  - The exact benchmark command and guard wrapper used.
  - Result-directory paths and guard JSONL.
  - Exact model and executable identities.
  - Digest-equality evidence (token and, where possible, logit).
  - Physical MB/token and io_await BEFORE and AFTER, with the resolution band.
  - A clear, honest KEEP or DROP recommendation.

Be positive and rigorous. The owner wants a real shot at reducing I/O wait and has
pre-approved experimental methods under the no-quality-loss rule. Find one that works.
