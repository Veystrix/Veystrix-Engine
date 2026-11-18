# Phase 1 SharedExpertPool — implementation note

**Scope:** port the Nemotron expert slot pool and add the multi-session fetch
primitives from `docs/performance/multi-session/architecture.md` §1.2 (failure
isolation from §4). This is an opt-in, CPU-testable building block only: no
scheduler, no server, no real checkpoint, no benchmark, no default change.

**Label legend (per the contract):** `[measured]` = observed in the run below;
`[cited]` = taken from the named source/doc; `[estimate]` = arithmetic/anchor,
not measured; `[idea]` = design decision, no evidence. No real-checkpoint or
speed number appears here.

## What was built

| File | Status |
|---|---|
| `cpp/engine/src/nemo_pool.hpp` / `.cpp` | `[cited]` verbatim port from `nemotron-pool-nocache-fc2skip`, plus two clearly-marked **additive** hooks (see below). In-flight shared futures, two fd maps (cached/uncached, `F_NOCACHE` fail-closed), LRU, and failed-fill invalidation are unchanged. |
| `cpp/engine/src/shared_expert_pool.hpp` / `.cpp` | new: wraps `NemoPool`; pin counts per slot, session/layer completion fences, per-layer union fetch, documented failure isolation. |
| `cpp/engine/tests/test_shared_expert_pool.cpp` | new: CPU/tiny synthetic fixtures, injectable fake `ReadFn` + resolver, no checkpoint/MLX. |
| `cpp/engine/tools/gen_shared_expert_pool_vectors.py` | new: tiny deterministic golden payloads (LCG, CPU only). |
| `cpp/engine/CMakeLists.txt` | added both `.cpp` to `veystrix_engine`; added `test_shared_expert_pool` target and a guarded `add_test` (only registered when `tests/vectors_shared_expert_pool/manifest.json` exists). |

### Additive hooks (why `NemoPool` changed at all)

The ported pool's bookkeeping mutex is private. To obey §1.2's **"pin counters
and fences live under the existing pool mutex; no new mutex is required"**,
`NemoPool` exposes two additive members only:

- `std::mutex& mutex() const` — returns the **existing** bookkeeping mutex. The
  pin/fence state is guarded by this lock; `SharedExpertPool` has no mutex.
- `set_eviction_veto(std::function<bool(int slot)>)` — consulted inside the
  existing `allocate_slot_locked()` LRU victim scan, **with the pool lock held**.
  The callback reads pin state guarded by that same lock, so pinned slots are
  never selected as victims. A failed/empty veto keeps the original behaviour.

No ported line of the fill/coalescing/fd/LRU/fail-closed logic was altered
other than this two-line victim-scan hook. `[idea]`

### Semantics

- `acquire_and_pin(session, layer, expert)` / `pin(session, layer, expert)` —
  ensure residency (one coalesced fill on a miss) and increment that slot's pin
  count. `pin(layer, expert)` is the contract's literal shorthand and pins
  under `kDefaultSessionId`. A pin is **not** released when a read returns.
- `release_fence(session, layer)` — the **only** pin-release path. It releases
  exactly that session's references recorded at that layer; other sessions'
  references (even on the same slot) stay until their own fences. Returns the
  number of references released.
- `fetch_union(layer, experts)` — dedups the multiset in first-seen order and
  issues one `pool_.acquire` per distinct key. NemoPool's existing shared
  futures coalesce concurrent same-key acquires into one physical read. It is a
  union **across sessions at one layer**, never across rows; it does not pin.
- Failure isolation — NemoPool already invalidates a failed slot to the free
  list, erases the in-flight entry and sets the promise so **all** waiters wake
  with the error. Per §4, `SharedExpertPool` does not swallow or widen it:
  **retry-once-then-fail-session is the caller's responsibility** (retry once
  via `NemoPool::read_direct`; a second failure fails only that session). No
  retry/cancellation mechanism is added here.
- Eviction is fail-closed: if every slot is pinned and a new key is needed, the
  ported "pool exhausted; refusing partial remap" error is thrown; a pinned
  slot is never silently overwritten. `[idea]`

### Sync / thread / worker accounting

- New mutexes: **0** — pin/fence state uses `NemoPool::mutex()` (§1.2).
- New engine threads: **0**. New I/O workers: **0**; worker count unchanged.
- **NEEDS OWNER APPROVAL** for any dedicated completion/fence thread, extra
  mutex, or I/O worker-count change — none added.
- Known narrow window `[idea]`: `acquire_and_pin` establishes residency, then
  takes the lock to record the pin; under today's single driver thread (§1.3)
  there is no competing evictor between those steps. A future multi-threaded
  driver must hold `pool_.mutex()` across both. Pin state itself is always under
  the lock.

## Verification (all CPU-only, synthetic)

Standalone build (MLX is not needed by these sources or this test):

```
clang++ -std=c++20 -O2 -Wall -Wextra -pthread -I cpp/engine/src \
  cpp/engine/src/nemo_pool.cpp cpp/engine/src/shared_expert_pool.cpp \
  cpp/engine/tests/test_shared_expert_pool.cpp -o <tmp>/test_shared_expert_pool
```

The only compiler diagnostic is the verbatim port's pre-existing
`unused parameter 'key'` at `nemo_pool.cpp:165`; no warning from the new code.

Exact test output `[measured]`:

```
synthetic shard: /var/folders/37/fn6fxqyd2rgfmvc4954dggbm0000gn/T/sep_pool_lp3hER/synthetic.bin (4096 bytes, no checkpoint)
[a] in-flight coalescing: concurrent sessions share one read
  ok: one physical read for the shared key
  ok: both sessions got the same slot
  ok: coalesced payload is byte-exact
  ok: misses == 1
  ok: coalesced_reads == sessions-1
  ok: physical_reads == 1
  ok: physical_bytes_read == one record
  ok: hits == 0 (no false hit on half-written data)
[b] pin prevents eviction until fence
  ok: pinned slot has one pin
  ok: pinned slot survives LRU pressure (victim was the unpinned slot)
  ok: unpinned slot was evicted instead
  ok: pinned payload still byte-exact after pressure
  ok: exactly one eviction occurred
  ok: fence released the pin
  ok: slot becomes evictable after fence
[c] fence releases only the calling session's pins
  ok: pin counts are per slot (a=2, b=1)
  ok: session 1 fence released one reference
  ok: slot a still pinned by session 2
  ok: slot b (session 2) untouched by session 1 fence
  ok: both session-2-pinned slots survive after session 1 fence
  ok: still-pinned payload is byte-exact
  ok: session 2 fence released both references
  ok: all pins cleared only after their owner's fence
  ok: previously pinned slots are evictable once all fences drain
[d] per-layer union fetch dedups distinct keys
  ok: one mapping per distinct key
  ok: first-seen order preserved
  ok: one physical read per distinct key
  ok: every union payload is byte-exact
  ok: repeat keys are pure hits (no extra reads)
  ok: hits == 2 on the second fetch
[e] failed shared read wakes every waiter with an error
  ok: all sessions woke with an error (no hang)
  ok: only the leader issued a read
  ok: failed fill returned its slot to the free list
  ok: failed key is not on the hit path
  ok: retry after failure succeeds and is byte-exact
test_shared_expert_pool OK (all checks passed)
```

Cross-language golden check `[measured]`:

```
wrote records.bin (4096 bytes) + union_gather.bin + manifest.json
records sha256: f2fcfc939175330150872a4290440eb73a6cd83cd213ddc4cd1d1b8a6b9370f5
union gather sha256: 49295551fc6112ec5ea1b16581d7ba388788698de5ee33e3b9ab89adfbe057b6
  ok: generator records.bin matches in-test LCG bytes (cross-language)
```

Determinism `[measured]`: 25 consecutive runs of the binary, 0 failures
(the leader's fake read blocks until all waiters register, so the coalescing
and failure tests are barrier-driven, not timing-driven).

Byte-exactness `[measured]`: `std::memcmp` plus a 32-bit word pass
(the `mx::view -> uint32` repo convention without an MLX dependency) on every
payload asserted in (a), (b), (d) and (e).

Coverage of the required cases: (a) one physical read for two concurrent
sessions sharing a key; (b) pin blocks LRU eviction until fence; (c) fence
releases only the calling session's pins; (d) union fetch dedups; (e) failed
read wakes all waiters with an error and leaves no stuck in-flight entry.

## Not done / owner gates

- No real checkpoint, no SSD/benchmark, no model run, no tok/s claim.
- No scheduler changes, no new default, no server integration — out of scope.
- `[idea]` Union fetch currently dedups only within one call; cross-session
  coalescing across *separate* calls relies on the resident slot (hit) after
  the first call. The §2 soft-lockstep driver that groups sessions at a layer
  boundary is not implemented here.
- **NEEDS OWNER APPROVAL** before adding a fence thread, a second mutex, or an
  I/O worker-count change (§1.2 / §1.3).
