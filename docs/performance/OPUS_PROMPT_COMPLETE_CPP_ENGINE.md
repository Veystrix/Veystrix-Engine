# Prompt for Opus — make the C++ engine complete, faster-or-equal to Python, and droppable

Paste everything below into Opus.

---

You are helping finish a port of the Veystrix inference engine from Python (MLX 0.32.2, Apple Silicon) to C++. I need your help with three things: (1) close the remaining gaps so C++ can fully replace Python for serving, (2) make C++ at least as fast as Python, and (3) design real concurrency so I can hit aggregate output targets of ~5 tok/s (Qwen Flash-Next, 113 GB, 125B/6B-active) and ~15 tok/s (Nemotron-3.5-Lightning, 24 GB, 30B/3B-active). Please read the code before answering and give me a concrete, prioritized plan with code-grounded reasoning. Label every claim MEASURED / ESTIMATE / CITED.

## Current verified state (all evidence is real-checkpoint, guarded with real_run_guard.py, 5 GB swap cap)

Repo: `/Users/vincent/Documents/Veystrix`. Branches: `cpp-engine-qdense` (core engine, PR #3), `ms-phase1-integration` (multi-session + features, PR #5). All pushed. Read `docs/performance/CPP_FULL_PORT_STATUS.md`, `docs/performance/CPP_PORT_PROBLEMS_FOR_OPUS.md`, and `docs/engineering/*_parity*.md` first.

Bit-exactness (tokens AND raw logit SHA-256, 3/3 clean runs each):
- Qwen gate (b): 16 tokens, 8/8 tokens + 8/8 digests. Fresh prompts at 16/64/256/512/1024 tokens all bit-exact vs Python. 1024-token prefill+4 decode ~31 s (was 174 s before I/O fixes).
- Nemotron bf16 AND INT8 KV (production default): 4/4 tokens + 4/4 digests on real 24 GB checkpoint.
- Non-greedy THINKING/INSTRUCT: 8/8 sampled tokens + 8/8 digests e2e; 11/11 isolated sampler cases. Keyless default-key draw required.
- Prefix cache: save/load skips prefill, 8/8 tokens + digests bit-exact.
- MTP: real shard-22 head weights; head-level bit-exact at positions 0/7/11 and nonzero-position with prepopulated QSA cache; fused Metal RoPE path now bit-exact (0/248320 diff); sparse indexer mask at past_len=2052 bit-exact (0 diffs on mask/SDPA/logits). MTP is OFF by default.
- Multi-session: N=2 and N=3 sessions with distinct prompts, every session's tokens+digests identical SOLO vs CONCURRENT on real 113 GB checkpoint.
- 37/38 ctests pass (the one failure is a pre-existing worktree .venv path artifact, not an engine bug).

Architecture already in C++: `cpp/engine/src/` has engine primitives (MoE/GDN/QSA/RoPE/Mamba/INT8-attention/KV-quant/MTP), `moe_stream.cpp`/`cross_shard_moe.cpp` (streamed experts, first-occurrence compact stack), `prefix_cache`, `nemo_pool.cpp` (bounded expert pool with in-flight coalescing + LRU), `shared_expert_pool` (pin counts, completion fences, per-layer union fetch), `session_state`, `admission`, `ms_scheduler` (single-thread cooperative, work-conserving), `cooperative_decode`. Multi-session tool: `cpp/engine/tools/multisession_real.cpp`; gate tools `check_real_parity.py`, `check_nemotron_parity.py`.

## What is NOT done (my honest gaps)

1. **Speed/profit unproven.** I have not measured steady-state DECODE tok/s of C++ vs Python head-to-end under identical conditions. Prefill at 1024 tok is ~31 s C++ vs ~30 s Python — i.e. NOT faster. The whole reason for C++ was profit. I fixed the obvious I/O (PLE coalesced preads, sorted gather_qmm for prefill, batched IO pager) but I don't know where the remaining time goes.
2. **True concurrency is not implemented.** Current multi-session is ONE GPU-driving thread with cooperative round-robin. `nemo_pool`/`shared_expert_pool` have in-flight coalescing and pin/fence logic but are only tested with synthetic fixtures — they are NOT wired into `real_checkpoint_generate.cpp`/`multisession_real.cpp`. No fence/completion thread, no I/O worker, no semaphore. The architecture doc marks new sync/thread/worker counts as NEEDS OWNER APPROVAL — the owner has now approved pursuing concurrency for the tok/s target, but I want your design review before adding threads.
3. **Serving integration missing.** C++ is a CLI tool (`real_checkpoint_generate`, `nemotron_generate`), not a library behind the 3 HTTP protocols (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`) that Python serves via `veystrix/server.py`. No FFI bindings, no ChatGuard equivalent (thermal/battery/swap supervisor), no universal prefix cache manager, no admission service.
4. **Feature long tail not ported**: expert-pool policies (s3fifo/cost-aware/least-stale), adaptive context admission/window budgeting, cache-aware routing, prompt lookup/drafting, continuous batching inside the server (scheduler exists standalone), warm cache, MTP native in decode default, KV tiers inside the full attention path (only selftests validate the primitives).
5. **Device caveat**: GPU↔CPU `mx.quantize/dequantize` differ ~1 ULP (irreducible MLX backend rounding); same-device parity is bit-exact, which is what serving uses.

## My concrete questions for you

### A. Full port / dropping Python
A1. What is the MINIMAL complete feature set that must exist in C++ before Python can be dropped without user-visible regression? Give me an explicit ordered checklist (protocol handlers, guards, admission, cache policies, sampling defaults, KV modes), separating "must have day 1" from "can follow".
A2. What is the right software boundary to serve the 3 HTTP APIs from C++? (shared lib + ctypes/CFFI vs. embedding a C++ HTTP server vs. keeping Python `server.py` as a thin shell that calls a C++ dylib.) Concrete file/module plan please.
A3. How should I keep rollback/shadow validation after cutover — e.g. run both engines on a sample of live requests and compare token+digest streams? Where would you put that hook?
A4. What is your recommended definition of "C++ can replace Python" (gate list with exact thresholds) that I can use as the drop criterion?

### B. Speed parity / profit
B1. How would you profile the C++ decode path to find where tok/s is lost? Give exact instruments/commands on macOS/MLX C++ (Instruments, `mx::benchmark`, `native_read` counters, swap/compressor stats) and what to look for.
B2. Which of these is the most likely dominant cost for C++ decode: per-token MoE shard preads, per-token `mx::eval` barriers, lazy Metal kernel compilation/dispatch, PLE n-gram reads, or head/logit projection? How would you A/B each without breaking bit-exactness?
B3. Is it realistic for C++ to MATCH Python tok/s given both use the same MLX 0.32.2 kernels and the same GPU? If most time is in identical Metal kernels, where can C++ actually win (I/O batching, fewer Python/GIL overheads, no Python object churn, fuse more ops)? Be blunt if the ceiling is low.
B4. What changes would make prefill faster than the ~30 s Python baseline (e.g. dense GEMM for dequantized experts, batched multi-token decode, persistent expert pool across layers)? Which preserve bit-exactness?

### C. Concurrency design for ~5/15 tok/s aggregate (the main goal)
Target: aggregate output across N concurrent requests; share expert READS and cached state between requests. Hardware: one Mac (Apple Silicon, ~16 GB unified, 4 TB SSD), thermal/swap guard caps. I will add threads if you approve the design.
C1. Given the earlier nondeterminism I saw from lazy IO nodes inside the GPU graph (fixed by eager eval), what is the SAFEST concurrency model that keeps every session bit-identical to solo? Options to compare: (a) one GPU-driving thread + a dedicated completion/fence thread + N CPU I/O prefetch threads; (b) multiple GPU streams with per-stream MLX arrays and explicit cross-stream fences; (c) keep one GPU thread but make all I/O async via a single poller. Recommend one with reasoning.
C2. How should `shared_expert_pool` pin/fence semantics interact with the scheduler's "waiting for expert reads at layer L" state so that a read shared by N sessions is issued once and every waiter is released exactly once? Where does the fence live — per session, per layer, per step?
C3. For EXPERT READ SHARING: my `nemo_pool` coalesces concurrent (layer,expert) misses via shared futures, and `shared_expert_pool::fetch_union(layer, keys)` dedups per call. What is the correct grouping point in a round-robin scheduler so sessions at the same layer batch their routes into one union fetch? How do I avoid holding pins too long (memory) vs. re-reading?
C4. For TOKEN/KV SHARING: sessions often share a system prompt. How should I share prefix-cache snapshots (copy-on-write KV) across sessions without an API lock bottleneck? What's the right granularity (whole prefix vs per-layer deltas)?
C5. MEMORY BUDGET: expert pool RAM, per-session KV, page-cache floor, and the 5 GB swap-growth guard all compete. How do you suggest sizing admission for N=2..8 on a 16 GB machine to maximize aggregate tok/s without tripping the guard? Any measurement-driven formula?
C6. What is a realistic aggregate tok/s ceiling for each model under this hardware, and which lever (expert sharing, read coalescing, KV sharing, larger pool, denser prefill) moves it most? My offline simulator (docs/performance/multi-session/) estimated Qwen ~3.2 tok/s SSD-bound and Nemotron ~26 tok/s GPU-bound — do you agree, and what would change those numbers?

## What I want back

1. A prioritized, concrete PLAN (files/functions, phase order, estimates) for: (a) speed profiling + first profit wins, (b) true concurrency design with exact threading/sync primitives and the bit-exactness invariant spelled out, (c) serving integration + drop-Python gate.
2. Direct answers to A1–A4, B1–B4, C1–C6, flagged MEASURED/ESTIMATE/CITED.
3. Any RISKS or places where my current design is wrong (especially: is the single-thread cooperative scheduler the right base, or should I replace it?).
4. A short list of commands/tests you want me to run next and the exact evidence format you want back.

Do NOT modify code; this is a design review + plan. Be specific and code-grounded — cite file:line from the repo where relevant.
