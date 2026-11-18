# opencode brief: finish the Veystrix C++ engine (speed, concurrency, serving, drop-Python gate)

Source: `docs/performance/OPUS_PROMPT_COMPLETE_CPP_ENGINE.md` (branch `ms-phase1-integration` @ `a033fa8`).
This file answers that prompt's questions (A1–A4, B1–B4, C1–C6) from the code. It also turns the answers into
phased work for opencode. Paste section 1 as the opencode task. Sections 2–7 are the guidance it must follow.

Evidence labels used throughout: **MEASURED** (a recorded run), **CITED** (a named doc in this repo),
**ESTIMATE** (arithmetic, not run), **IDEA** (design decision, no evidence yet).

---

## 1. The task (paste this into opencode)

> You are working in `/Users/vincent/Documents/Veystrix` on the Veystrix C++ inference engine (`cpp/engine/`).
> Base your work on branch `ms-phase1-integration` (it contains `cpp-engine-qdense`). Create one new branch per
> phase (`cpp-p0-profile`, `cpp-p1-solo-speed`, …). Never merge to `main`.
>
> Goal, in order: (1) make the C++ engine **at least as fast as Python solo**, (2) add **real decode concurrency**
> (I/O overlapped with other sessions' GPU work, shared expert reads) aiming at aggregate ~5 tok/s Qwen Flash-Next and
> ~15 tok/s Nemotron-3.5-Lightning, (3) put the engine **behind the existing Python HTTP server as a dylib** with
> shadow validation, and (4) pass the **drop-Python gate** in section 6.
>
> Read first, in this order: `docs/performance/OPENCODE_PROMPT_COMPLETE_CPP_ENGINE.md` (this file — sections 2–7 are
> binding), `docs/flashnext/AGENT_INVARIANTS.md`, `docs/performance/DO_NOT_IMPLEMENT.md`,
> `docs/performance/CPP_FULL_PORT_STATUS.md`, `docs/engineering/multi_session_phase1_runtime.md`,
> `docs/engineering/shared_expert_pool_phase1.md`, `docs/engineering/multisession_real_parity.md`,
> `docs/performance/E0_CONCURRENCY_RESULT.md`, and on branch `multi-session-phase0`
> `docs/performance/multi-session/architecture.md`.
>
> Work phase by phase (section 5). Inside a phase you may write code and run offline/synthetic tests freely.
> **Any real-checkpoint run, benchmark or hardware measurement needs the owner's explicit "go" first**: list the
> exact commands, expected duration and guard settings, then wait. Report every result in the evidence format
> (section 7). Stop and ask when a stop condition in section 4 is hit.

---

## 2. What the code shows today (these facts drive the plan)

All file:line references are on `ms-phase1-integration` @ `a033fa8`.

**F1 — Nemotron C++ decode has no expert pool.** `ExpertReader` in `cpp/engine/tools/nemotron_generate.cpp:279-320`
issues one serial `pread` per expert per tensor. It then copies each buffer into a new `mx::array` (iterator
constructor), with a route sync at `:663`. Python Nemotron decode uses a slot pool with in-flight coalescing and
8-lane copies. 77% of its touched expert bytes come from page cache and it averages ≈1.4 misses per layer (CITED:
`architecture.md` §7, citing `overnight-2026-11-16.md`). `nemo_pool.cpp` is a verbatim port of that pool, but it is
only fixture-tested and not wired in. **ESTIMATE:** C++ Nemotron decode is currently slower than Python's
7–8.8 tok/s. This is unmeasured; Phase 0 measures it.

**F2 — The Qwen decode pool is a fixed per-layer LRU.** `num_slots` defaults to 16 per layer
(`real_checkpoint_generate.cpp:1585`), so 48 × 16 = 768 resident experts (≈2 GiB, ESTIMATE from the "~0.8 GiB
saved at top_k slots" comment). Python uses one budgeted pool with S3-FIFO, which gave +5% tok/s with 5% fewer bytes
(CITED: `EXACT_SPEED.md`). Nobody has compared the C++ hit rate or physical MB/token against Python yet.

**F3 — Decode reads block the GPU-driving thread.** `StreamingMoE::forward` does `mx::eval(r)` (route to CPU,
`moe_stream.cpp:259`), then synchronous `run_batched_reads` (`:320`), and only then builds the MoE graph. A
single session cannot overlap its own reads, because router → experts → next layer is a strict chain. As long as a
whole `run_step` is atomic, other sessions cannot use that wait either.

**F4 — Multi-session today interleaves whole tokens.** `multisession_real.cpp:469` runs one full `run_step` per
session per turn. So the N=3 run (32.3 s concurrent vs 37.4 s for the three solo runs, MEASURED) mostly saved
repeated weight loading. It did not overlap I/O. Bit-identity to solo is proven (MEASURED, N=2 and N=3). Overlap
is not.

**F5 — Long prefill is a bytes floor for both engines.** Qwen routes top-10 of 512 experts. At S=1024 the expected
number of distinct experts per layer is 512·(1−(1−10/512)^1024) ≈ 512, and ≈508 already at S=256 (ESTIMATE). So
layer-major prefill reads almost the whole routed-expert set once. At the measured 3.36 GB/s (E0) that explains
~30 s for both Python and C++. Also, prefill builds a new `StreamingMoE`/`CrossShardMoE` per layer per call
(`real_checkpoint_generate.cpp:1918-1931`). Each constructor calls `find_shard` 10 times, and each call scans shard
headers (`moe_stream.cpp:110`) and allocates slot buffers. That costs something, but probably little next to
~100 GB of reads (ESTIMATE).

**F6 — Diagnostics on the hot path.** Every decode step SHA-256s the full 248,320-float logit row
(`emit_record`, `real_checkpoint_generate.cpp:2177`). `run_step` also runs `snprintf` for debug-capture names and
`std::getenv` per layer even when capture is off. The parity gates need the digests, but serving and timing
runs must not pay for them. AGENT_INVARIANTS also requires a disabled diagnostic to cost only a branch.

**F7 — Two separate pool implementations.** `StreamingMoE` (Qwen) has its own slot bookkeeping (`slot_plan`).
`SharedExpertPool` wraps `NemoPool` (Nemotron layout) and is the only one with pins and fences. The note in
`shared_expert_pool_phase1.md` documents a window between residency and pin that is safe only with one driver.

**F8 — Default-profile risk.** The C++ parity references use bf16 KV, MTP off and fused RoPE. The app's
"Qwen exact speed" profile turns on `kv-tiered4` and layer-major prefill features (`EXACT_SPEED.md`;
`veystrix/model_process.py:46-51`). If that is the shipped default, C++ must match that profile or the cutover
changes outputs. The KV-tier primitives are bit-exact, but the integrated attention path is not validated yet
(`CPP_FULL_PORT_STATUS.md`).

## 3. Design decisions (answers to A1–A4, B1–B4, C1–C6)

### B — speed (answer first, because it decides the order)

**B3 — Can C++ match or beat Python? Plainly: the solo ceiling is about the same as Python, and the profit is in
concurrency.** Both engines run the same MLX 0.32.2 kernels on the same GPU. Both are bound by SSD reads or I/O
latency: Qwen spends ≈289 of ≈385 ms per token waiting on I/O, and Nemotron ≈75 of ≈115 ms on expert dispatch and
wait (CITED, `architecture.md` §7). Solo C++ can gain:
(a) parity, by giving Nemotron a pool (F1);
(b) fewer bytes, by turning C++'s smaller RAM footprint (no Python heap or object churn) into a larger effective
expert cache;
(c) a few ms/token of graph-building and launch overhead.
Expect solo decode within about ±10% of Python once F1/F2 are fixed (ESTIMATE). Real profit comes from overlapping
one session's reads with another session's GPU work. E0 showed this overlap is physically real: a saturated cold
reader inflated the decode step by only 10.08% (MEASURED).

**B2 — Most likely dominant decode cost.** Qwen: MoE miss reads, which are bytes-bound (CITED). Nemotron C++: serial
`pread` plus copy with no pool (F1). Per-layer `mx::eval` route syncs are inherent: routes must reach the CPU to
plan reads, and Python also does one sync per layer. Lazy Metal kernel compilation costs only on the first step, so
report step 0 separately (do not add warm-up work to the decode path without approval). PLE n-gram reads and the
head projection are minor (ESTIMATE). Phase 0 measures each one. Every A/B allowed here changes only *when* bytes
arrive or *which buffer* holds them. It never changes the graph, gather order or batch composition, so the digest
gate must stay identical. Any digest change means the change is wrong.

**B1 — How to profile.** Add opt-in `VEYSTRIX_CPP_PROFILE=1`. When it is off, the cost must be a single branch.
Per step it writes one JSONL row:
- `build_ms`: CPU graph-building time;
- `route_sync_ms`: time inside each `mx::eval(r)`, summed and per layer;
- `read_wall_ms`, `bytes_requested`, and `bytes_physical` (`proc_pid_rusage` `ri_diskio_bytesread` delta);
- `hits`, `misses`, `coalesced` per layer;
- `final_eval_ms`, `sample_ms`, `digest_ms`;
- MLX active, cache and peak memory.

On the owner's Mac (owner-run only):
- `xcrun xctrace record --template 'Metal System Trace' --time-limit 20s --launch -- <bin> <args>` for GPU busy/idle
  gaps between kernels;
- `--template 'Time Profiler'` for CPU hot spots;
- `sudo fs_usage -w -f diskio <pid>` to check read sizes (only if the owner agrees to sudo).

Never use `memory_pressure`, purge, or swap manipulation as preparation (AGENT_INVARIANTS). Capture the same
counters from the Python engine (its existing flashnext telemetry), so the comparison uses the same prompt, the same
**32-token arm**, interleaved and order-reversed arms, and a reported resolution band.

**B4 — Faster prefill without breaking bit-exactness.** The floor is bytes divided by bandwidth (F5). Levers that
keep the graph identical:
(1) prefix-cache hits, which skip prefill entirely — the biggest TTFT lever, and it already exists;
(2) reading layer L+1's experts while layer L computes. At S ≥ ~256 almost every expert is needed anyway, so this
adds no wasted bytes (ESTIMATE). It is worth doing only if Phase 0 shows prefill GPU time is a meaningful share.
`DO_NOT_IMPLEMENT.md` rejected route-union prefill and pipelined attention tiles, so measure the compute share
before building anything;
(3) build each layer's MoE reader once at load and reuse it (F5); cache the shard index from
`model.safetensors.index.json`.

Never change gather_qmm batch composition, expert stack order or sorting. P5 proved that rounding depends on both.

### C — concurrency (the main goal)

**C1 — Concurrency model: keep ONE GPU-driving thread and make the I/O asynchronous.** This is option (a) without a
separate fence thread (IDEA):
- Exactly one engine thread builds and evaluates MLX graphs on the default stream, as today. Reject multiple GPU
  streams (b): cross-stream lazy nodes caused the P1 nondeterminism, and per-session isolation currently depends on
  one stream's ordering.
- Expert reads go to the **existing** libdispatch user-interactive queue with the **same width (8)**. Add an
  asynchronous variant of `run_batched_reads`: `dispatch_group_async` per lane, with `dispatch_group_notify` setting
  an atomic ticket. Worker count and QoS stay unchanged.
- Completion: one `std::mutex` plus one `std::condition_variable` on a ready-queue. The driver waits on it only when
  no session is runnable. No dedicated fence thread: only the driver knows when its own `mx::eval` returned, so the
  driver releases fences itself.
- New sync objects to enumerate for owner approval: 1 engine driver thread (it replaces the caller thread once the
  engine is behind the dylib), 1 mutex, 1 condition variable, and atomic tickets. Nothing else.

**Bit-exactness invariants for concurrency (all must hold; write a test per line):**
1. Each session's op sequence, shapes, dtypes and gather order are identical to solo. Only the interleaving of
   independent evaluations changes.
2. No MLX array ever wraps bytes that are still being written. A layer's MoE graph is built only after every ticket
   it depends on has completed, and the completion is read with acquire ordering. No lazy Load nodes in any graph
   (the P1 rule).
3. A slot read by session S at layer L is **pinned from plan time** (hits and misses alike, in the same critical
   section as the eviction choice) **until S's next `mx::eval` returns**. That eval materializes layer L's MoE output.
   Close the residency-then-pin window from F7 by holding the pool mutex across both steps.
4. Eviction never picks a pinned or in-flight slot. If the pool is exhausted, the session **parks** until a fence
   releases a slot (backpressure). It does not throw. Admission must make this rare.
5. Per-session PRNG, KV, GDN, PLE and position state. No shared mutable numeric state. The PLE swap-in/swap-out
   trick in `multisession_real.cpp` must become true per-session PLE state before per-layer interleaving starts.
6. One failed read wakes every waiter with an error; retry once, then fail **that session only** (§4 of the
   contract).

**C2 — Fences.** One in-flight ticket per `(layer, expert)`. Each session has a wait-set per `(session, layer)` and
becomes Ready when the set is empty. Pins are counted per slot and owned per `(session, layer)`. `release_fence(session,
layer)` runs exactly once, after that session's next eval. Use an idempotence guard and assert in debug builds.
The fence lives per session per layer, not per step.

**C3 — Read sharing and the grouping point.** Do not add a barrier. The union comes from the **in-flight table**:
when session A reaches layer L it submits its misses immediately. When B reaches L, keys still in flight attach to A's
tickets, and resident keys are hits. The scheduler's job is to keep sessions close in *layers* so that this works.
- Run soft lockstep: always run the runnable session that is furthest behind, bounded by
  `max_skew_layers` (IDEA: start at 4 and sweep {2, 4, 8} in Phase 3).
- Pins last at most one layer-step of the owning session.
- Retention beyond pins is the pool policy's job.
- This needs **one** pool abstraction with pins for both models. Prefer adding pin/fence and in-flight coalescing to
  the Qwen `StreamingMoE` slot model, which is the validated bit-exact layout. Wire `NemoPool`/`SharedExpertPool`
  into Nemotron. Both must expose the same `plan → pin → submit → wait → build → eval → release` interface.

**C4 — Shared prefixes.** Make snapshots immutable: `shared_ptr<const Snapshot>` in a map keyed by the token-chain
SHA, with a lock held only for map lookup and insert. Use whole-prefix granularity: GDN recurrent state cannot be
split per token, and attention KV is the large part. A restore copies the small GDN, conv and PLE state and
references the KV arrays.

Verify that MLX copies rather than donates on the first `slice_update` into a shared array: the snapshot's reference
keeps it from being uniquely owned. Test: hash the snapshot before and after two sessions decode different
continuations from it, and check that both match a solo restore.

**C5 — Memory and admission: measure, then admit.**
For each model, measure:
- `avail_post_load`;
- the per-session marginal `m(ctx)` at ctx ∈ {2K, 8K};
- the per-step transient peak `T`;
- the pool size `P`.

Admit session N+1 only if `avail_now − m(ctx_req) − T ≥ max(3 GiB, page_cache_floor)` and swap grew by 0 over the
last 30 s. Starting caps (IDEA): Qwen N≤2 and Nemotron N≤4. Raise a cap only after a guarded sweep at N+1 shows no
swap growth. Seed page-cache floors from the Phase 0 anchors (CITED: Nemotron ≈195 MB per aggregate token at N=4,
Qwen ≈0.9 GB at N=3), then replace them with measured values.

**C6 — Realistic ceilings and which lever matters.**
- **Nemotron:** GPU ≈38 ms/token gives a single-stream aggregate ceiling of ≈26 tok/s, or ≈24 tok/s with E0's 10%
  inflation (ESTIMATE). **15 tok/s at N=3–4 is plausible** if the ≈75 ms of dispatch and wait hides behind other
  sessions' GPU work. Levers, in order: the pool (F1), overlap (C1), then a larger cache.
- **Qwen:** bytes-bound. At the measured 3.36 GB/s, the solo 1.17 GB/token gives at most ≈2.9 tok/s with perfect
  overlap. The N=3 anchor of 0.9 GB per aggregate token gives ≈3.7 tok/s (ESTIMATE; the old 3.2 used 2.9 GB/s).
  **5 tok/s needs ≤0.67 GB per aggregate token.** Levers, in order:
  1. a larger effective expert cache funded by C++ RAM savings;
  2. higher N with tight soft lockstep;
  3. S3-FIFO;
  4. batch-K shared reads (still open in `DO_NOT_IMPLEMENT.md`).
- **Cross-session row batching** (one qmv with M=N rows): IDEA only. It would cut GPU time per aggregate token, but it
  changes kernel shapes. It can be a separate opt-in experiment after Phase 3, and only with real-weight proof that
  every row is bit-identical to its solo M=1 result. It must never become a default without that proof.

### A — full port and dropping Python

**A2 — Boundary: keep `veystrix/server.py` as the HTTP and protocol shell, and load the C++ engine as a dylib through
a narrow C ABI.** Tokenizer, chat templates, tool-call parsing and the three endpoints are ~2.7k lines of Python
(`server.py`) plus helpers. Rewriting them in C++ is a large regression surface that adds no tok/s. The
`Backend` protocol (`veystrix/backends/base.py`) and the `batch_*` hooks used by `batch_scheduler.py`
(`flashnext.py:458-564`) are already the right seam. Plan:
- `cpp/engine/capi/veystrix_c.h` and `.cpp`:
  - `vx_engine_open(model_dir, profile_json)`;
  - `vx_session_open` and `vx_session_close`;
  - `vx_session_prefill(tokens, n)`;
  - `vx_session_step(sampler_params) → token`;
  - `vx_snapshot_save`, `vx_snapshot_restore`;
  - `vx_cancel`, `vx_engine_stats`, `vx_engine_close`.
  Use plain C types, return error codes plus `vx_last_error()`, and never let a C++ exception cross the ABI.
- The engine owns its GPU-driving thread and multi-session scheduler. Python only enqueues work and receives tokens
  (a callback or a blocking per-request queue), so the GIL never sits in the hot loop.
- `veystrix/backends/cpp_engine.py` implements `Backend` plus the `batch_*` hooks over `ctypes` (standard library, no
  new build dependency).
- Selector: `VEYSTRIX_ENGINE=python|cpp`. Default stays `python` until the gate in section 6 passes.

**A3 — Shadow and rollback after cutover.** Both engines cannot run at once on 16 GB. Use **record and replay**
instead: the C++ backend logs (prompt token ids, frozen profile, emitted tokens, per-step digests) for sampled
greedy and seeded-sampling requests. An owner-run replay job feeds them to the Python engine later and diffs them.
Put the hook in `cpp_engine.py`'s generate path. Keep it off by default and make it one branch when off. Keep the
Python engine importable for at least two releases behind `VEYSTRIX_ENGINE=python`.

**A1 — Minimum C++ feature set before Python can be dropped.**

Day 1 (required):
1. The shipped default profile for each model, including its KV mode. Confirm the shipped profile (F8): Qwen bf16
   or tiered4/exact-speed, and Nemotron INT8 KV.
2. Greedy plus THINKING/INSTRUCT sampling with the default-key draw.
3. Batched bit-exact prefill.
4. Exact prefix cache save/restore wired into serving, with fail-closed checks.
5. Context-window enforcement that returns the same errors (`ContextLimitError`, `OversizedPrefillError`).
6. Cancellation mid-prefill and mid-decode.
7. Streaming tokens.
8. Guard parity with `veystrix/engine_guard.py`: thermal, swap, battery and memory reclaim.
9. Stats used by the UI: tok/s, prompt rate and usage counts.
10. Session save/load/list/delete.
11. Admission.

Can follow: MTP (opt-in), s3fifo and other pool policies beyond the one that wins Phase 1, warm cache, cache-aware
routing, lookup drafting. Several of these are rejected in `DO_NOT_IMPLEMENT.md` anyway; check before porting.

**A4 — Drop criterion:** section 6.

---

## 4. Hard rules (non-negotiable)

- **Bit-exact or reject.** Tokens and per-step float32 logit SHA-256 must match Python. Never weaken a gate to
  "tokens only" or "close enough", and never rebaseline a reference to make a gate pass.
- **Everything opt-in, with rollback.** Every new path sits behind an env flag. The current path stays the default
  until the owner promotes it with evidence. Disabled diagnostics cost one branch only.
- **No hardware run without "go".** This covers any real-checkpoint run, benchmark or timing. Always use the guard:
  `scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 5120 --compressor-growth-mib 8192
  --thermal-level 2 --max-seconds 900`. Check `sysctl vm.swapusage` before runs and leave cool-downs (≥20 s) between
  them.
- **Check `DO_NOT_IMPLEMENT.md` before any performance idea.** A rejected idea comes back only with a new reason
  and the owner's approval.
- **Threads and sync:** only those listed in C1. Any other new thread, mutex, worker-count or QoS change → stop and
  ask.
- **Measurement rules** (AGENT_INVARIANTS): the 32-token arm, interleaved and order-reversed arms, a resolution band,
  physical MB/token, active RAM, and generation and tail rates. Limit any negative conclusion to what was actually
  measured.
- **Stop and ask when:**
  - a digest changes and you cannot localize why within one session;
  - a design needs a new thread or sync primitive not in C1;
  - a guard trips twice;
  - a phase gate fails;
  - the shipped default profile is ambiguous (F8).

---

## 5. Phased plan

Effort figures are ESTIMATES in opencode working sessions.

### Phase 0 — Instrument and baseline (1 session; branch `cpp-p0-profile`)
1. Add `VEYSTRIX_CPP_PROFILE` JSONL counters (B1) to `real_checkpoint_generate.cpp`, `nemotron_generate.cpp`,
   `moe_stream.cpp`, `cross_shard_moe.cpp` and `native_read.c`.
2. Make diagnostics single-branch (F6): hoist every `getenv` into a startup struct, and do the name formatting inside
   the enabled branch. Add `VEYSTRIX_CPP_NO_DIGEST=1` for timing runs: parity gates keep digests on and timing arms
   turn them off.
3. Offline: build, run ctests (expect 37/38; `native_generate` is the known `.venv` path artifact), and run the
   parity tools in dry mode.
4. **Owner-run** (ask first): Qwen and Nemotron, C++ vs Python, 32-token arm with a 1K prompt, 3 interleaved pairs,
   profile on. Deliverable: a per-token time breakdown table plus physical MB/token and hit rate for each engine.

**Gate:** counters add no measurable time when disabled (digests are unchanged, which is trivially true), and the
breakdown is reported.

### Phase 1 — Solo speed parity (2–3 sessions; `cpp-p1-solo-speed`)
1. **Nemotron pool (F1):** wire `NemoPool` into `nemotron_generate.cpp` behind `VEYSTRIX_CPP_NEMO_POOL=1`. Slots
   are zero-copy wrapped (no iterator copy). Reads use the 8-lane batched reader. Keep the current compact-unique
   path as the rollback. The digest decides: if the pooled gather (slot indices into the pool buffer) changes the
   digest compared with the compact-unique stack, build the compact first-occurrence stack *from pool slots* (copy
   rows into a small contiguous buffer, the way Qwen P5 does) instead.
2. **Qwen pool (F2):** report the C++ hit rate and MB/token against Python. If C++ reads more bytes, port Python's
   S3-FIFO as one budgeted cross-layer pool behind `VEYSTRIX_CPP_POOL=s3fifo`, and size it from measured free RAM.
3. **Prefill readers (F5):** build MoE readers once at load, cache the shard index, and reuse slot buffers.
4. Owner-run gates: Qwen gate (b) 3/3, Qwen 1024-token run 3/3, Nemotron INT8 gate 3/3. Then the paired timing arms.

**Gate:** all digests identical, and C++ decode tok/s ≥ Python within the resolution band for both models.

### Phase 2 — Resumable per-layer decode plus async I/O, solo (3–4 sessions; `cpp-p2-layer-cursor`)
1. Split `run_step` into an explicit per-layer state machine (`LayerCursor`: layer index, hidden array, pending
   wait-set, phase ∈ {PreRoute, AwaitReads, PostMoE}). Use plain structs, not C++20 coroutines, so it is easy to
   debug. Do the same for the Nemotron loop.
2. Add the async read variant (C1), the shared `plan → pin → submit → wait → build → eval → release` pool interface
   (C3), and pins and fences under the pool mutex (invariant 3).
3. Synthetic tests: one per invariant in C1 (items 1–6), plus a stress test of 1,000 randomized interleavings with
   a fake reader and random delays, asserting byte-exact outputs.

**Gate:** with N=1, digests are identical to Phase 1 and tok/s is not lower within the band (owner-run).

### Phase 3 — Real multi-session overlap (3–4 sessions; `cpp-p3-overlap`)
1. Drive `LayerCursor`s from `MsScheduler`: soft lockstep, furthest-behind first, `max_skew_layers`; park sessions on
   pool exhaustion. Make PLE state truly per session.
2. Admission from measured numbers (C5). Shared immutable prefix snapshots (C4).
3. Owner-run:
   - N ∈ {1, 2, 3, 4} for Nemotron and {1, 2, 3} for Qwen;
   - `run_multisession_parity.py` (solo vs concurrent, tokens plus digests) at every N;
   - then timing with aggregate tok/s, per-session tok/s, bytes per aggregate token, coalesced reads, pin hold time,
     and the guard envelope.

**Gate:** every session bit-identical to solo, and aggregate tok/s higher than at N=1. Qwen decision point: if the
best N still needs more than 0.75 GB per aggregate token, report the ceiling and stop pushing Qwen tok/s. Focus Qwen
on TTFT via the prefix cache instead, and the owner decides.

### Phase 4 — Dylib plus Python backend plus shadow (3–5 sessions; `cpp-p4-capi`; can start after Phase 1)
Build the C ABI, `veystrix/backends/cpp_engine.py`, `VEYSTRIX_ENGINE`, and the record/replay shadow log (A2, A3).
Run the existing server test suite with the C++ backend for `/v1/chat/completions`, `/v1/responses` and
`/v1/messages`, including streaming, cancel, tool calls and context-limit errors.

**Gate:** the protocol suite passes; live single-request generation matches CLI digests (owner-run).

### Phase 5 — Feature tail and drop gate
Finish the A1 day-1 list. Validate the shipped default profile end to end (F8). Run section 6.

---

## 6. Drop-Python gate (all required)

| # | Gate | Threshold |
|---|---|---|
| G1 | Parity, both models, **shipped default profile** | prompts of 16/128/1024/4096 tokens plus 2 multi-turn tool conversations; 32 greedy tokens; tokens and digests 3/3. Seeded THINKING and INSTRUCT sampling 3/3. |
| G2 | Protocol | full server suite passes with `VEYSTRIX_ENGINE=cpp`, all 3 endpoints, streaming, cancel, tool calls, errors |
| G3 | Speed | decode tok/s ≥ Python and TTFT ≤ Python at 1K/4K, within the resolution band (3 interleaved pairs); peak RSS ≤ Python |
| G4 | Safety | guard parity: thermal stop, swap cap, battery, memory reclaim, and cancellation under load, each with a test |
| G5 | Soak | owner-chosen period of real app use with shadow logs; replay 100% identical |
| G6 | Rollback | `VEYSTRIX_ENGINE=python` works in the release build |

---

## 7. Evidence format (every owner-run result)

Write `results/cpp/<date>-<phase>-<name>/verdict.json` and a short `README.md` next to it:

```json
{
  "branch": "...", "commit": "...", "model": "qwen|nemotron", "profile": {"kv": "...", "mtp": false, "flags": {}},
  "arms": [{"engine": "cpp|python", "order": 1, "tokens": 32, "tok_s": 0.0, "tail_tok_s": 0.0,
            "ttft_s": 0.0, "physical_mb_per_token": 0.0, "hit_rate": 0.0, "active_gib": 0.0}],
  "band": {"method": "...", "value": 0.0},
  "parity": {"tokens_equal": true, "digests_equal": true, "runs": "3/3", "first_divergence": null},
  "multi_session": {"n": 0, "aggregate_tok_s": 0.0, "per_session_tok_s": [], "bytes_per_agg_token_gb": 0.0,
                    "coalesced_reads": 0, "max_pin_hold_ms": 0.0},
  "guard": {"stopped": null, "swap_growth_mib": 0, "floor_gib": 0.0, "thermal_max": 0},
  "labels": {"<claim>": "MEASURED|CITED|ESTIMATE|IDEA"}
}
```

Also update `docs/performance/CPP_FULL_PORT_STATUS.md` after each phase, and add any rejected idea to
`DO_NOT_IMPLEMENT.md` with its number and source.
