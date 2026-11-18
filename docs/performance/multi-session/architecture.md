# Multi-session architecture — Phase 0 design contract

Language-neutral design contract for a **future, opt-in** path in which one user request
fans out to N = 2–4 concurrent sub-agents [idea: task-brief parameter] on the same 16 GB
fanless M4 Air [cited: `docs/performance/overnight-2026-11-16.md`].
Aggregate decode targets: **≥5 tok/s on Qwen Flash-Next** and **≥15 tok/s on Nemotron**
[idea: task-brief target]. Solo baselines today: Qwen ≈2.5 tok/s, Nemotron 7–8.8 tok/s
[cited: `docs/performance/overnight-2026-11-16.md`].

**Phase 0 scope boundary.** This document specifies a contract only. It proposes **no
engine or server code changes now**; every mechanism below is a future implementation
behind an opt-in flag, with all current defaults unchanged. It is not a benchmark report
and no real-checkpoint or benchmark command was run for it.

**Hard constraint.** Each session's output must stay bit-identical to running alone.
Rows from different sessions are **never** batched into one matmul. Only the *fetch* is
shared; compute stays single-row. The two mechanisms are (a) **read sharing** — an expert
read once is reused by every session that needs it at that layer (in-flight coalescing +
shared cache), and (b) **overlap** — one session's GPU work runs while another waits on
SSD reads.

**Label legend.** `[measured]` = recorded in a live run; `[cited]` = taken from the named
existing doc (itself a recorded measurement); `[estimate]` = arithmetic or an
author-provided anchor, not measured; `[idea]` = proposed design decision, no evidence.
At authoring time `results/multisession/<ts>-sim/` did **not** exist, so every Task 1
number below is an `[estimate]` anchor from the task brief. Replace them with simulator
output before implementation.

---

## 1. Components

### 1.1 SessionState (per session, exclusively owned)

- KV cache (with its frozen tier/mode from admission — see §3).
- GDN / Mamba recurrent state.
- Absolute position counter.
- Sampler with its **own PRNG key** — an independent stream, never reseeded or advanced
  by another session's sampling [idea: bit-identity requirement].
- Prefix-cache handle, with solo-equivalent restoration semantics (§3).
- Admission-time frozen numerics profile (prefill chunk size, KV tier/mode, adaptive
  top-k), immutable for the session's lifetime (§3).

Nothing in SessionState is shared. Sharing happens only through the pool and the
scheduler's coordination of reads.

### 1.2 SharedExpertPool (one process-wide pool, keyed by `(layer, expert)`)

Starting point, read-only: `cpp/engine/src/nemo_pool.cpp` on branch
`nemotron-pool-nocache-fc2skip`. Every bullet in this list describes existing source
behaviour [cited: `cpp/engine/src/nemo_pool.cpp`]:

- Key = `(layer, expert)`; a single fixed-stride, aligned buffer holds all slots.
- **In-flight shared futures already exist**: the in-flight check precedes the resident
  check so a same-key arrival mid-fill coalesces instead of hitting half-written data;
  one leader performs the single physical read and publishes the result to all waiters.
- Two descriptor maps (cached / uncached) keep a cached reuse from being contaminated by
  `F_NOCACHE` fills; LRU victim selection never evicts a slot whose fill is in flight.
- A failed fill returns its slot to the free list (never the hit path) and wakes waiters
  with an error.

Additions required for multi-session (all `[idea]`):

- **Pin counts per slot.** A pin records that a session has queued GPU work which will
  read that slot. A pinned slot is never evicted or overwritten. Pins are released only
  after that session's **completion fence** for the consuming layer/step — not when the
  read returns, and not when another session finishes. This extends the existing
  "never evict an in-flight fill" guarantee to the post-fill, pre-fence window.
- **Per-layer union fetch.** For the set of `(layer, expert)` keys needed by all sessions
  waiting at layer L, issue one fetch per *distinct key* and let every waiter use it. The
  union is across sessions at the same layer; it is **not** a union across rows and does
  not change how many rows each expert is applied to.
- **New sync / thread / worker accounting.** Pin counters and fences live under the
  existing pool mutex; no new mutex is required `[idea]`. No new I/O worker is required;
  the I/O worker count is unchanged `[idea]`. **Any dedicated completion/fence thread,
  additional mutex, or I/O worker-count change is NEEDS OWNER APPROVAL.**

### 1.3 Scheduler

- **One GPU-driving thread** `[idea: matches the current single driving thread; adding a
  second is NEEDS OWNER APPROVAL]`.
- Sessions are state machines that **yield at "waiting for expert reads at layer L"**; a
  yielded session consumes no GPU time until its reads (and pins) are satisfied.
- The driver advances runnable sessions, skips waiting ones, and is work-conserving
  (§2). The **I/O worker count is unchanged** `[idea]`.
- Prefill is exclusive in v1 (§3) `[idea: task-brief rule]`.

### 1.4 Admission and memory budget

Governed by §5. Admission is the only place a session may be refused; it freezes the
session's numerics profile at that moment (§3).

---

## 2. Scheduler choice — lockstep vs interleaved

Comparison and recommendation are `[idea]`; the supporting numbers are `[estimate]` or
`[cited]` as tagged below.

| Policy | Read sharing | Utilisation / fairness | Verdict |
|---|---|---|---|
| Strict lockstep (all sessions advance exactly one layer together) | Maximum union per layer; best coalescing | Stalls on the slowest read; a single cold layer idles every session | Too brittle |
| Pure interleaved (run one session as far as it can) | Union windows collapse; little coalescing | Work-conserving but unbounded skew; one session can starve others and hold pins | Rejected |
| **Soft lockstep** | Preserves per-layer union windows | Work-conserving; bounded skew | **Recommended** |

**Recommendation (default expectation) `[idea]`:** soft lockstep — round-robin at **layer
boundaries**, work-conserving (skip a session whose reads are not ready rather than
stalling the round), with **bounded skew** (a session may not lead the round by more than
a small number of layers; exact bound is an open question, §9).

Supporting anchors, all `[estimate]`:

- Nemotron, arrangement B, LRU, N = 4, 6.6 GB pool: ≈112 unique experts per aggregate
  token vs ≈138 per solo token (≈19% fewer); read working set ≈190–200 MB per aggregate
  token. Source: Task 1 anchor; simulator not present at authoring time.
- Qwen, arrangement C, 349-slot pool, N = 3: ≈0.9 GB per aggregate token vs ≈1.17 GB per
  solo token (≈23% fewer). Source: Task 1 anchor.
- Caveat on the Qwen anchor: same-layer expert overlap between two conversations is only
  6.3% [cited: `docs/performance/handoff-2026-11-14.md`, item B], so **in-flight coalescing
  alone cannot produce that reduction** on Qwen. Most of it must come from shared-pool /
  page-cache temporal reuse, which soft lockstep should preserve; this needs Task 1
  confirmation before it is relied on (§8, §9).

The skew bound and the round cadence are policy knobs, not defaults to be frozen here;
each would be opt-in `[idea]`.

---

## 3. Bit-identity rules (non-negotiable)

1. **No cross-session row batching.** Each session's experts are applied to its own rows
   in its own single-row compute. The pool shares bytes, never matmuls. `[idea: hard
   constraint from the task brief]`
2. **Per-session PRNG.** Sampler PRNG keys are independent (§1.1); no shared or
   cross-advanced randomness. `[idea]`
3. **Freeze numerics at admission.** Prefill chunk size, KV tier/mode, and adaptive top-k
   are fixed for a session at admission and **never changed under memory pressure**. If
   the budget cannot satisfy a new session's frozen profile, **refuse the new session**
   (§5) rather than mutate an admitted one. `[idea]`
4. **Prefix cache must match the solo reference.** A prefix-cache hit must be bit-identical
   to the corresponding solo reference run. If a digest test cannot establish that, the
   prefix cache is **disabled for that configuration** rather than trusted. Existing
   Nemotron caveats apply: continuation digests already depend on chunk edges and on
   session-identity flags such as SDPA/fused-scan markers
   [cited: `docs/performance/handoff-2026-11-14.md`, item D;
   cited: `docs/performance/overnight-2026-11-16.md`]. `[idea: the gating rule]`
5. **Prefill is exclusive in v1** `[idea]`. No other session's decode is interleaved inside
   a session's prefill, and no prefill overlaps another prefill. Only decode is a candidate
   for multi-session overlap. `[idea]`
6. **All of the above are opt-in with defaults unchanged.** `[idea]`

---

## 4. Failure isolation

Base behaviour from `nemo_pool.cpp` is the starting point: a failed fill invalidates its
slot to the free list, erases the in-flight entry, and sets the promise so **all coalesced
waiters wake with the error** [cited: source file]. Required multi-session contract,
all `[idea]`:

- A failed shared read wakes **all** its waiters with the error; no waiter is left blocked
  on a future that will never complete.
- Each waiter **retries exactly once with its own read** (direct, non-pool read path),
  independent of the leader and of other sessions.
- A **second** failure fails **only that session**. It must not fail the pool, the
  scheduler, or sibling sessions; their pins are released through their own fences.
- **Cancelling one session releases only its own pins**, and only after its fence drains;
  other sessions' pins and in-flight reads are untouched.
- Python-era failure modes to avoid by design (historical references, **not** Phase 0
  fixes) [cited: `docs/performance/handoff-2026-11-14.md`, item E]:
  - `decode_many` is **not transactional** — a mid-layer exception invalidates every
        member of a grouped pass. The new design must scope failure to one session.
  - `should_cancel` sits **inside the group's try block** — one client's disconnect check
        can fail the whole group. Cancellation must be checked per session, outside any
        shared commit point.

---

## 5. Memory budget and admission

Total budget (all `[idea]` as a formula):

```
total = model resident
      + shared expert pool
      + page-cache floor for the target
      + N × per-session state (KV + GDN/Mamba + prefix handle + samplers)
```

Page-cache floor anchors, both `[estimate]` (Task 1; simulator absent at authoring):

- Nemotron target: ≈190–200 MB of read working set per **aggregate** token at N = 4,
  arrangement B, LRU, 6.6 GB pool (≈112 unique experts/aggregate token vs ≈138 solo).
- Qwen target: ≈0.9 GB per **aggregate** token at N = 3, arrangement C, 349-slot pool
  (vs ≈1.17 GB/token solo).

Admission rule `[idea]`:

- Compute the budget for the prospective session N+1 using its **frozen** profile (§3).
- **Refuse session N+1 if admitting it would push the page cache below the target's
  floor**, or below the standing 3 GiB available-memory rule
  [cited: `docs/performance/measurement-standard.md`, "Keep at least 3 GiB available";
  cited: `docs/performance/handoff-2026-11-14.md`].
- Count **effective** pool capacity, not nominal slots: pinned slots from already-queued
  GPU work cannot be evicted, so they are unavailable to new admissions. This is the
  multi-session-specific pressure that does not exist in solo decode.
- Refusal is the pressure valve — never a numerics change, never eviction of a pinned
  slot, never shrinking an admitted session's state (§3, rule 3).

---

## 6. Thermal behaviour

The fanless Air reaches thermal level 2 after ≈100 s of sustained load
[cited: `docs/performance/overnight-2026-11-16.md`; the same note reports guard stops of
16K cold runs and brief level-2 appearances that never lasted the ≈15 s stop threshold].

Guard behaviour `[idea]`:

- On sustained thermal level 2, **stop admitting new sessions** and degrade gracefully:
  finish admitted work or fail it per-session (§4), never silently continue hot.
- Report, rather than hide, the degradation: thermal level and time at level,
  aggregate tok/s and per-session tok/s, bytes/token and hits/coalesced from the pool,
  memory floor, swap, compressor, and which sessions were skipped by the work-conserving
  scheduler.
- No default change to existing run-guard rules; thermal handling is opt-in `[idea]`.

---

## 7. Nemotron vs Qwen — where the difference comes from

**Nemotron (latency- and copy-bound; overlap + cache size matter most).**

- Per token ≈115 ms total: ≈38 ms GPU work, ≈75.5 ms expert dispatch and wait
  [cited: `docs/performance/overnight-2026-11-16.md`]. Decode-bound rate 8.4–8.8 tok/s
  at a 3,000-token prompt, 7.7 at 8K context [cited: same].
- Physical reads are only ≈179 MB/token against ≈773 MB of expert bytes touched (≈77%
  served from page cache); decode is latency-bound per layer, not bandwidth-bound
  [cited: same]. Misses ≈1.4 experts per layer [cited: same].
- Cached-copy floor ≈1.28 ms per 6 experts (8-thread pool); splitting records further
  does not go below ≈1.21 ms [cited: same]. Decode is a strict per-layer dependency
  chain (router → experts → next layer), so a single session cannot overlap its own
  compute with its own reads; **cross-session overlap is the only overlap available**.
- GPU-bound ceiling with one driving thread ≈1 / 0.038 ≈26 tok/s aggregate
  [estimate: arithmetic on cited 38 ms GPU]. A per-aggregate-token read floor of
  ≈112/6 ≈18.7 layer-equivalents × ≈1.28 ms ≈24 ms [estimate: Task 1 expert anchor ×
  cited copy floor] must be hidden behind other sessions' GPU work.

**Qwen (near its SSD limit; overlap alone cannot reach the target).**

- SSD reaches 2.9 GB/s for the 345 KB expert reads decode produces; decode averages
  ≈2.5 GB/s and is bytes-bound [cited: `docs/performance/overnight-2026-11-16.md`;
  the task brief rounds to ≈2.9 GB/s, ≈2.5 tok/s solo].
- Per token ≈385 ms: ≈289 ms I/O wait, ≈51 ms router sync, ≈70 ms GPU compute
  [cited: same]. A 349-slot pool reads ≈0.95 GB/token at ≈24% hit rate [cited: same];
  larger pools are RAM-bound [cited: same].
- GPU-bound ceiling ≈1 / 0.070 ≈14 tok/s aggregate [estimate: cited 70 ms GPU], so
  **compute is not the limiter** for the 5 tok/s target [idea: task-brief target].
- SSD-bound ceiling with the shared-pool anchor ≈2.9 / 0.9 ≈3.2 tok/s aggregate
  [estimate: cited SSD rate ÷ Task 1 anchor]. To reach 5 tok/s the stream must fall to
  ≈2.9 / 5 ≈0.58 GB per aggregate token [estimate]. Overlap cannot create bytes that are
  not read; the target needs fewer bytes per aggregate token.

---

## 8. Feasibility verdict per model

**Nemotron — target ≥15 tok/s aggregate: plausible but unproven.** `[estimate]`

- Reasoning: cited GPU ceiling ≈26 tok/s aggregate [estimate from §7] leaves headroom
  above 15 tok/s; the ≈24 ms/aggregate-token read floor [estimate] must be overlapped,
  and Task 1's ≈112 vs ≈138 unique-expert reduction [estimate] shrinks what must be read.
- Limiting resource if it misses: **per-layer read/copy latency and copy bandwidth**,
  plus scheduler skew (a stalled session holds pins and shrinks effective pool capacity).
- Next lever: larger shared cache / page-cache floor and tighter soft-lockstep alignment
  to widen per-layer unions `[idea]`; these are memory-bound, so admission decides.
- Evidence that would change the verdict upward: Task 1 simulator showing the ≈112
  unique-expert anchor holds at the admitted cache size, and a measured overlap run
  where read wait is hidden behind sibling GPU work. Downward: measured skew stalls or
  copy-floor time that cannot be hidden.

**Qwen — target ≥5 tok/s aggregate: looks unreachable under the current design.**
`[estimate]`

- Reasoning: SSD ceiling ≈3.2 tok/s aggregate from the ≈0.9 GB/token anchor
  [estimate, §7]; the anchor needs to reach ≈0.58 GB/token to hit 5 tok/s [estimate].
  Overlap cannot fix a bytes problem, and cited same-layer overlap is only 6.3%
  [cited], so in-flight coalescing contributes little.
- Limiting resource: **bytes per aggregate token** (SSD throughput ÷ bytes/token),
  constrained by RAM for the pool/page cache, not by GPU or I/O worker count.
- Next lever: reduce bytes per aggregate token — a larger resident expert set (more RAM),
  or a checkpoint with fewer experts per token / better route prediction. Reducing
  experts read is a numerics/quality change and is out of scope for bit-identity `[idea]`.
- Evidence that would change the verdict: a Task 1 result showing ≤≈0.58 GB/aggregate
  token within the admitted memory budget [estimate], or a measured SSD/pool configuration
  that hides all non-I/O time so the ceiling is purely bytes/token. Constructive note:
  N = 2–3 sharing [estimate] should still be measured — even if the target [idea] is missed, moving from
  ≈2.5 tok/s solo [cited] toward the ≈3.2 tok/s ceiling [estimate] is a real gain and
  should not be discarded.

Both verdicts are estimates, not claims of success or failure; Task 1 simulator output is
the first thing that should replace them (§9).

---

## 9. Open questions for the owner

- Approve the soft-lockstep default and the bounded-skew value; is fairness or peak
  aggregate throughput the priority? (Any new sync/thread/worker count also needs
  approval — §1.2, §1.3.)
- Should Task 1 simulator output replace the `[estimate]` anchors before any C++ work
  starts, and who runs it?
- Reconcile the Qwen solo anchor ≈1.17 GB/token [estimate] with the cited ≈0.95 GB/token
  at 349 slots [cited] before using either for admission.
- Is a memory budget that refuses new sessions acceptable UX, and what should the client
  see on refusal?
- Should Qwen be pursued at all if the measured ceiling stays below the 5 tok/s target
  [idea], and if so is the target to be restated?
- Any constraints on thermal guard behaviour beyond "stop admitting and report" (§6)?
