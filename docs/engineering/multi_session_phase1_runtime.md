# Multi-session Phase 1 runtime — SessionState, admission, one-driver scheduler

Scope: implement Phase 1 of the multi-session architecture contract
`docs/performance/multi-session/architecture.md` (on branch
`multi-session-phase0`) §1.1, §1.3, §1.4, §3 and §5, on top of the existing
read-only `cpp/engine/src/cooperative_decode.hpp`, `sampling.hpp` and
`prefix_cache.hpp`. This is a **synthetic CPU implementation**: no checkpoint,
no benchmark, no SSD, no MLX in the new test. No defaults elsewhere are changed;
the path is opt-in by construction (nothing in the existing engine calls the
new scheduler).

Label legend (same as the contract): `[measured]` = recorded in a live run;
`[cited]` = taken from a named existing doc/artifact; `[estimate]` = arithmetic
or an author-provided anchor, not measured; `[idea]` = proposed design decision,
no evidence. This document reports only `[estimate]`/`[cited]` anchors and
`[idea]` decisions, plus the `[measured]`-style *test* output below (which is a
CPU fixture result, not a performance measurement).

## What was added

| File | Role |
|---|---|
| `cpp/engine/src/session_state.hpp` / `.cpp` | `SessionState`: opaque model-cache vector, independent PRNG key/state, frozen `NumericsProfile`, `PrefixCacheHandle`, position counter, session id, explicit yield handshake. |
| `cpp/engine/src/admission.hpp` / `.cpp` | Memory budget from args; refuse session N+1 below the page-cache floor or the standing 3 GiB rule. |
| `cpp/engine/src/ms_scheduler.hpp` / `.cpp` | One GPU-driving-thread state-machine scheduler built on `cooperative_run`; work-conserving round-robin with bounded skew, explicit yield at layer L, exclusive prefill in v1. |
| `cpp/engine/tests/test_ms_scheduler.cpp` | Five tiny CPU fixtures (a)–(e), all synthetic. |
| `cpp/engine/CMakeLists.txt` | New `.cpp` files added to `veystrix_engine`; guarded `test_ms_scheduler` target and `ms_scheduler` ctest. |

### SessionState (contract §1.1)

Everything is exclusively owned; the type is move-only (copy deleted) so a PRNG
stream or cache vector cannot be duplicated across sessions. The PRNG is an
independent splitmix64 stream `[idea]`: `prng_key()` is constant for the
session and `draw_u64()` advances only that session's own state — no global RNG
and no other session ever advances it. The real sampler key can be mapped onto
this state later; in Phase 1 it is a scalar CPU fixture stream. The model cache
is a `std::vector<std::shared_ptr<void>>` of opaque layer handles owned only by
this session. `NumericsProfile` (prefill chunk, KV tier/mode, adaptive top-k) is
frozen at admission and immutable afterwards (§3 rule 3). `PrefixCacheHandle`
carries a chain key and an `enabled` flag; a configuration whose prefix hit is
not bit-identical must leave it disabled (§3 rule 4). No prefix lookup happens
here.

### Admission / memory budget (contract §1.4, §5)

Budget formula `[idea]` (SI MB, matching the simulator's 1 MB = 1e6 B):

```
capacity = model resident + effective expert pool + page cache + N x session state
effective expert pool = nominal pool - pinned pool
```

`pinned_pool_mb` implements §5's "count effective pool capacity, not nominal
slots" `[idea]`; all inputs are read from `AdmissionArgs` (from args — no real
measurement). Session N+1 is refused if the projected page cache drops below
`page_cache_floor_mb`, or below the standing 3 GiB available-memory rule
(`min_available_mb = 3072`, `[cited]`
`docs/performance/measurement-standard.md`; `docs/performance/handoff-2026-11-14.md`).
An unfrozen profile is also refused rather than mutated (§3 rule 3).

Page-cache floor anchors `[estimate]`, from the Phase 0 simulator
`models/shared/tests/bench/multisession_sim.py` (branch `multi-session-phase0`,
which cites the same numbers in `docs/performance/multi-session/architecture.md`
§2/§5):

- Nemotron: **≈190–200 MB per aggregate token** (N=4, arrangement B, LRU, 6.6 GB
  pool; ≈112 unique experts/aggregate token vs ≈138 solo). Represented here as
  `kNemotronFloorMbPerAggToken = 195.0`.
- Qwen: **≈0.9 GB per aggregate token** (N=3, arrangement C, 349-slot pool;
  vs ≈1.17 GB/token solo). Represented here as
  `kQwenFloorMbPerAggToken = 900.0`.

`page_cache_floor_from_anchor(model, tokens)` multiplies the anchor by a target
aggregate-token count; it is still `[estimate]`. Callers may pass any explicit
floor in `AdmissionArgs::page_cache_floor_mb`. No simulator was re-run for this
work; these numbers are cited anchors, not measurements of this runtime.

### Scheduler (contract §1.3, §2, §3 rule 5)

`MsScheduler::drive()` executes synchronously on the **calling** thread — the
single GPU-driving thread. There is **no** new thread, mutex, I/O worker, or
worker-count change. Any future dedicated completion/fence thread, extra mutex,
or I/O worker-count change is **NEEDS OWNER APPROVAL** and is not added here;
the header and implementation carry that comment.

Each step is executed through a one-shot proxy passed to the existing
`cooperative_run()`, so step indexing, seeding and round-robin semantics are
exactly the existing single-thread ones; rows are never batched across sessions
(§3 rule 1). Sessions are state machines: `Ready`, `Prefill`,
`WaitingForExpertReads`, `Done`. A step calls
`SessionState::request_yield(L)` when its expert reads at layer L are missing;
the scheduler consumes that request after the step and parks the session. The
caller-supplied `ReadsReadyFn(session_id, layer)` callback wakes it; the
callback does no I/O and spawns nothing. The driver:

1. wakes waiting sessions whose reads are ready;
2. selects runnable sessions (skips waiting/done → work-conserving);
3. enforces exclusive prefill in v1 (only the first prefill session advances;
   no decode interleaves a prefill) `[idea]`;
4. applies bounded skew `[idea]` (`max_skew_steps = 1` default): a session may
   not lead by more than the bound when another runnable session is behind,
   unless skipping would empty the round;
5. advances each runnable session once via `cooperative_run()`.

Skew bound and round cadence are policy knobs `[idea]`, not frozen defaults;
the contract leaves the exact bound as an open question (§9).

## Tests (synthetic CPU only)

`cpp/engine/tests/test_ms_scheduler.cpp` (guarded ctest name `ms_scheduler`):

- **(a)** N = 2, 3, 4 sessions interleaved by one driver vs solo
  `cooperative_run()`: per-session token streams are bit-identical and step
  counts match.
- **(b)** A session that yields at layer 7 consumes exactly one step; further
  `drive()` calls with `reads_ready = false` consume zero steps, and one step
  runs once the callback reports ready. A waiting-only driver does not spin.
- **(c)** Work-conserving: with one session parked at its yield, the other
  advances three steps in three rounds.
- **(d)** Admission admits session 1, refuses session 2 when the projected page
  cache falls below the floor; the 3 GiB rule and the unfrozen-profile refusal
  are checked separately.
- **(e)** Per-session PRNG independence: session B's drawn sequence equals its
  solo sequence even when session A draws interleaved with it; neither key nor
  state is advanced across sessions.

### Exact test output (this worktree, Release build)

Built directly (standalone, no MLX) and via CMake/ctest. Binary stdout,
verbatim:

```
test_ms_scheduler OK: (a) bit-identity N=2,3,4
test_ms_scheduler OK: (b) yield at layer 7 held until reads-ready
test_ms_scheduler OK: (c) work-conserving while one session waits
test_ms_scheduler OK: (d) admission refuses below page-cache floor
test_ms_scheduler OK: (e) per-session PRNG independence
test_ms_scheduler OK (5/5)
```

ctest summary, verbatim:

```
    Start 2: ms_scheduler
1/1 Test #2: ms_scheduler .....................   Passed    0.30 sec

100% tests passed out of 1

Total Test time (real) =   0.30 sec
```

(No checkpoint, SSD or GPU was used; these are CPU fixtures, `[measured]` as
test results only, not a performance measurement.)

## Non-goals / limitations

- No `SharedExpertPool` pin/fence implementation, no cross-session union fetch:
  those belong to the pool workstream. The scheduler only reports yields and
  takes a reads-ready callback.
- No real sampler integration: the PRNG is a CPU scalar stream; mapping it to
  `sampling.hpp`'s `mx::array` key is future work and must preserve bit
  identity.
- No real prefix-cache hit/restore is exercised; `PrefixCacheHandle` is a
  handle only. Per §3 rule 4, any configuration without a bit-identical digest
  test must keep it disabled.
- No thermal guard, no new sync/thread/worker counts, no numerics changes, no
  defaults changed outside this opt-in path.
- No commit was made.
