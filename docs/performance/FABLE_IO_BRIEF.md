# Fable brief — attack the SSD I/O wait on the 16 GB M4

> Copy this whole document to Fable as the opening prompt. It is self-contained:
> ground truth, targets, hard rules, what has already been tried (with numbers),
> what is already done, and what is explicitly approved.

---

## 0. Who you are and what to be

You are tackling the **one remaining real bottleneck** in Veystrix: **SSD expert-read wait
during decode** on a **16 GB Apple M4 MacBook Air**. Be **positive and constructive**: treat every
failed attempt below as a *measured* narrowing of the search space, not as a reason to give up.
There is real, valuable work here — find the mechanism that cuts bytes read per token or hides the
read latency, prove it bit-exact, and we will ship it. The owner has explicitly said: **any
experimental result or research-paper method that genuinely reduces I/O wait is approved for
implementation**, as long as the no-quality-loss rule below holds. Bring ideas from anywhere —
papers, other engines, kernel/layout tricks, OS-level I/O, scheduler changes. We want your best
positive shot.

## 1. Ground truth — the machine, the models, the problem

- **Hardware:** MacBook Air M4, **16 GB unified memory**, single internal SSD, fanless, on AC with
  Low Power off. Apple internal NVMe measures ~3.0–3.3 GB/s sequential; double-buffered ~6 GB/s.
- **Engine:** Veystrix, private macOS app. Swift UI talks HTTP to a Python MLX engine. All GPU
  kernels are MLX / mlx-vlm (no in-repo Metal). The app already ships a native C dylib.
- **Model A — Qwen3.8-Flash-Next:** `qwen4_exp`, 125B MoE, **512 experts/layer, top-10**, 48 layers
  (GDN linear attention + full QSA attention), **Q4 affine G32**, MTP head. Decode **~2.3 tok/s**,
  reads **~408 MB physical per token** (short-prompt harness 3.33 tok/s, 300 ms/token).
- **Model B — Nemotron 3.5 Lightning:** `nemotron_h`, **30B-A3B**, 52 layers, **6 experts/token**,
  **Q4 affine G64**, Mamba/SSM + attention. Decode **~6.7–7 tok/s**, reads **~436 MB/token**,
  with a 2 GB expert pool the cache hit rate is only **43.7%** (5,792 hits / 7,456 misses).

**Targets (owner):** Qwen decode **5+ tok/s**, Nemotron decode **15+ tok/s**, with no quality loss.
Also improve prefill/TTFT. **Validate on the 16 GB M4 first.** Larger Macs only after 16 GB success.

## 2. The decisive measurements (what is actually slow)

Guarded profiles, saved on disk:
- `results/flashnext/20261117-host-window-profile/` — Qwen, 300.1 ms/token:
  **`io_await` = 156.86 ms = 52.3%**; all exclusive host time = 18.22 ms = **6.1%**; GPU ~27%.
  Removing 100% of host work yields at most **1.065× (3.55 tok/s)** — so the problem is I/O, not Python.
- `results/nemotron35/20261117-140232-decode-profile-manual/` — Nemotron, 181.4 ms/token (5.51 tok/s):
  **`expert_dispatch_and_wait` = 123.1 ms = 67.9%**; host route sync = 50.5 ms = 27.8%;
  **436 MB read/token**, 43.7% cache hit. Removing all host sync caps at 7.64 tok/s (1.386×).

**The wall:** ~408–436 MB/token at ~3 GB/s ≈ **130–145 ms/token minimum** → decode ceiling near
**7 tok/s** on this machine. To reach 15 you must roughly **halve bytes/token or more** (i.e. raise
expert reuse from ~44% toward ~95%) without blowing memory — or hide the reads almost completely.
That is the problem. Be honest if it is physically impossible on 16 GB, but pursue it hard.

## 3. Hard rules (non-negotiable)

1. **No quality loss.** Output must be bit-identical: same token digests, BF16 rounding boundaries,
   Q4 affine G32/G64 weights, exact top-10/512 and top-9/240 routing. A method that changes logits
   or tokens is **out**, no matter the speed.
2. **16 GB M4 first.** Do not propose "just add RAM / another Mac" as the solution (those are the
   owner's deferred levers #17/#18, not your task).
3. **Guarded runs only, with explicit permission.** Use `scripts/flashnext/real_run_guard.py` with
   the established envelope (`--floor-gib 1.5 --swap-growth-mib 2048 --compressor-growth-mib 4096`).
   Ask before every hardware run. Preserve all `results/**`.
4. **One change at a time**, A/B vs the *current accepted baseline*, token+logit digest gate on.
5. Document everything: result dirs, guard JSONL, exact model/executable identity, resolution band.
6. Check `docs/performance/DO_NOT_IMPLEMENT.md` before any idea — it is authoritative on dead ends.

## 4. What has ALREADY FAILED (do not repeat — numbers included)

Full registry: `docs/performance/DO_NOT_IMPLEMENT.md`; live tracker: `docs/performance/SPEEDUP_ROADMAP.md`.

**Decode / I/O:**
- **Native MTP / K=2 read-sharing**: 16% SLOWER (2.298→1.924 tok/s), digest identical. All depths 1–8 slower.
- **Lookup / prompt-lookup drafting v2**: 7.30→6.15 tok/s.
- **Expert prefetch** (`FLASHNEXT_EXPERT_PREFETCH`): 2.39→2.38. **Next-layer router prefetch** / lookahead: tied.
- **Decode overlap** (`FLASHNEXT_DECODE_OVERLAP`): 2.39→2.41 (within band); **early-submit −8.2%**; **early shared expert −10%**.
- **More I/O workers (6/8/16/24/32)**: all tied — SSD saturated at the worker level.
- **Cache policies** other than S3-FIFO: cost-aware 2.45, lru 2.39, s3fifo 2.54; least-stale tied.
- **Pool size on 16 GB**: 2→4 GB caused a **resolved −1.61 tok/s regression** (+0.86 s TTFT, +2.14 GB footprint); **1 GB beat 2 GB** (6.94 vs 6.56). Production auto-sizes the pool to 0 on 16 GB already. A bigger cache is NOT the answer on 16 GB — memory pressure wins.
- **Expert packing / coalescing**: +9% in-process but no server gain, +70 GiB disk. Rejected.
- **`F_NOCACHE`** (macOS O_DIRECT): **8% SLOWER**. `F_RDAHEAD=0`: 1.3%, below band. Wired-limit sysctl and MLX wired limit: gain is noise / hits the memory guard.
- **MLX command-buffer caps** (`MLX_MAX_MB_PER_BUFFER`): on our 16 GB M4, Nemotron decode −0.5 tok/s (resolved), Qwen TTFT +0.86 s. Rejected.
- **`mx.async_eval`**: 311 vs 314 tok/s. `FLASHNEXT_OVERLAP` already on by default.
- **Perfect expert predictor + overlapped reads**: +0.1 / −2.9 / −0.7% (tied).

**Prefill:**
- Route-union prefill (through router 30.5 vs 29.9 s); PLE row prefetch 33.8→36.8 s (+87 GB reads); QSA gather tile tuning (no gain); pipelined attention tiles (end-to-end tied); dequantize-all-history (0.98 vs 0.73 s); `qsa-fast-prefill` fails quality gate.

**Compression:**
- Lossless ZSTD/LZ4 on already-4-bit weights saves only **0.9–2.1%** (RPI 2025, arXiv:2503.18869). All requantization (GPTQ/AWQ/FP8/etc.) is lossy → out.

## 5. What has ALREADY BEEN DONE / is in place (build on it)

- **Sorted, gap-merged pread** (Method #2) was implemented in `models/flashnext/store.py`
  (`_pread_into_merged`, opt-in `FLASHNEXT_SORTED_MERGED_PREAD=1`, off by default). A/B: **TTFT −0.69 s (resolved win)**, decode unchanged (unresolved). 22 read-mode unit tests pass. This is the one real win — note it helps prefill, not decode.
- **C++ pager scaffold** in `cpp/decode/` (`io_pager.{hpp,cpp}`, `native_read.c` reused): deterministic expert id-remap + dedup + batched read + zero-copy `mx::array` handoff. **2 bit-exact ctests pass** (`ctest`: `io_pager`, `pager_read`). Toolchain verified: builds against venv MLX 0.32.2, `mlx_probe` OK. Live hookup needs the (deferred) full C++ engine port; a small ctypes hook into `build_plan` is possible.
- **`exact-speed` profile is on by default**: dense GEMM, fused gather/SDPA, single-chunk layer-major, merge-small-tail, ngram-dedup, S3-FIFO — prefill wins already deployed. Prefix/agentic caching is on.
- **4 GB / 1 GB pool A/B, buffer-caps A/B, wired-limit A/B, both decode profiles** — all measured, documented above.

## 6. The five big ports we already ruled out (with reasons)

1. **#11 io_uring/O_DIRECT pager**: O_DIRECT rejected here; overlap rejected; only the remap portion is portable and it does not address the 156 ms of drive wait.
2. **#15 full-layer staging + 2nd SSD mirror**: only one physical SSD on this machine (`diskutil list`); portable staging (two-half ring) is already in `prefill_ring.py` and on.
3. **#12 lossless rANS 3-tier cache**: 4-bit weights are 0.9–2.1% compressible — cannot cut the bus.
4. **#22 SwiftLM rewrite**: 8.5× came from a 0.58 tok/s baseline; our engine already exceeds it.
5. **#19 last-row prefill**: changes logits — fails the bit-exact gate.

## 7. Your mission — what to look for

**Primary: reduce physical bytes read per token, or hide the read latency, bit-exactly.** Concretely, explore (not exhaustive — your judgment):
- **Raise expert cache hit rate toward ~90% on 16 GB without increasing footprint** (the pool is already auto-sized to 0 because bigger pools regress — is there a fundamentally better residency policy, expert duplication across layers, or a working-set/locality exploit we missed?).
- **Expert-contiguous on-disk layout** that makes the 9 sub-slices of one expert one read — but note prior "expert packing" had no server gain; find out why and whether a different layout (e.g. per-expert records with lossless framing, or re-ordering only within a shard) changes that.
- **Route-union / co-access-aware layout** so one token's 10 experts share cache lines / pages.
- **Prefetch timed to the router's prefetch window** without the overlap mechanisms that were rejected (the rejection was about *early shared-expert compute* and *early submit*, not necessarily all prefetch timing).
- **Two-phase / staged reads** (O_DIRECT alternative: pinned buffers, `preadv` with large iovecs, madvise variants) that reduce per-read overhead rather than bypassing the page cache.
- **Research-paper methods** on MoE expert caching/offloading, SSD streaming, or KV/expert tiering that are **bit-exact** and could plausibly halve bytes/token. Read the paper, extract the mechanism and its measured numbers, and design a falsifiable 16 GB test.

**Approach:** inspect the code and raw measurement files first. Pick the single experiment with the strongest *measured or well-evidenced* chance of lowering `io_await` or physical MB/token. Implement it behind an opt-in flag, **off by default**, with a bit-exact gate. Produce a patch + a falsifiable guarded A/B, or a clean artifact that rules the idea out with numbers.

**Return:** changed files, exact benchmark command, result-directory links, exact model/executable identity, digest equality evidence, physical MB/token and `io_await` before/after with resolution band, and a clear keep/drop recommendation.

Be positive: the owner has approved any experimental method or research-paper idea that reduces I/O wait under the no-quality-loss rule. Find one that works.
