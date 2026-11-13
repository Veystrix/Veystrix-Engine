"""Flash-Next pool capacity planning at prefill and decode boundaries."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
import time

GIB = 2**30
GB = 10**9
POLICY = "VEYSTRIX_MEMORY_POLICY"
POOL_STEPS = (2 * GIB, GIB, GIB // 2, 0)
GROW_MARGIN = GIB
GROW_COOLDOWN_SECONDS = 120
KV_BYTES_PER_TOKEN = 64 * 1024
DECODE_SLACK = GIB // 2
# Each verification row can select ten different experts. 8 MiB covers the
# larger supported G32/G64 record plus an expert-major staging copy.
UNION_BYTES_PER_ROUTE = 8 * 2**20


def enabled(environ=None, physical_bytes=None):
    env = os.environ if environ is None else environ
    mode = env.get(POLICY, "auto").lower()
    if mode == "legacy" or env.get("FLASHNEXT_MEMORY_MANAGER") == "1":
        return False
    if mode == "elastic":
        return True
    if physical_bytes is None:
        physical_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return physical_bytes <= 16 * GIB


@dataclass(frozen=True)
class MemoryBudget:
    available_bytes: int
    resident_model_bytes: int
    live_state_bytes: int
    prefix_ram_bytes: int
    pending_state_bytes: int
    prefill_scratch_bytes: int
    speculative_workspace_bytes: int
    decode_slack_bytes: int
    expert_pool_bytes: int
    pressure: int

    @property
    def projected_available_bytes(self):
        return (self.available_bytes - self.pending_state_bytes - self.prefill_scratch_bytes
                - self.speculative_workspace_bytes - self.decode_slack_bytes - self.expert_pool_bytes)


def plan(*, available, pressure, resident, state, prefix, pending_tokens, prefill_scratch,
         verification_width, existing_pool, cap, previous, last_change, now):
    """Choose optional pool capacity. Already-resident bytes are reported separately."""
    workspace = min(8, max(0, verification_width)) * 10 * UNION_BYTES_PER_ROUTE
    state_growth = max(0, pending_tokens) * KV_BYTES_PER_TOKEN
    slack = DECODE_SLACK if not prefill_scratch else 0
    unreserved = available + existing_pool - state_growth - prefill_scratch - workspace - slack
    size = next(step for step in POOL_STEPS if step <= max(0, min(cap, unreserved)))
    if pressure >= 2:
        size = 0
    if previous is not None and size > previous:
        if now - last_change < GROW_COOLDOWN_SECONDS or unreserved - size < GROW_MARGIN:
            size = previous
    return MemoryBudget(
        available + existing_pool, resident, state, prefix, state_growth,
        prefill_scratch, workspace, slack, size, pressure,
    )


def requested_cap(environ=None):
    env = os.environ if environ is None else environ
    raw = float(env.get("FLASHNEXT_EXPERT_POOL_GB", "6") or 0)
    return max(0, round(raw * GB)) if math.isfinite(raw) else 0


def cache_bytes(cache):
    """Account for live recurrent and KV arrays without evaluating them."""
    seen = set()

    def size(value):
        if isinstance(value, (tuple, list)):
            return sum(size(child) for child in value)
        if value is None or id(value) in seen:
            return 0
        seen.add(id(value))
        return int(getattr(value, "nbytes", 0))

    return sum(size(entry.state) for entry in cache)


def apply(store, *, phase, tokens, chunk, ring, verification_width, resident, cache,
          prefix_store=None, synchronize=None, clear_cache=None):
    """Resize an optional pool between phases, under the backend request lock."""
    from . import memory_manager, pool_lifecycle

    previous = getattr(store, "_flashnext_elastic_pool_bytes", None)
    changed = getattr(store, "_flashnext_elastic_changed", float("-inf"))
    pool = getattr(store, "_flashnext_expert_pool", None)
    # An anonymous mmap can be sparsely touched. Count at most the
    # populated slots as reclaimable when comparing alternative capacities.
    existing = min(pool_lifecycle.pool_bytes(pool),
                   len(getattr(pool, "_slot_of", ())) * getattr(pool, "stride", 0))
    now = time.monotonic()
    sample = memory_manager.read_system()
    scratch = 0
    if phase == "prefill":
        scratch = memory_manager.estimate_prefill(
            memory_manager.Request(tokens, chunk_tokens=chunk, ring_eligible=ring),
            pool_resident=bool(existing),
        )
    budget = plan(
        available=sample["available"], pressure=sample["pressure"], resident=resident,
        state=cache_bytes(cache), prefix=prefix_store.usage()["ram_bytes"] if prefix_store else 0,
        pending_tokens=tokens, prefill_scratch=scratch, verification_width=verification_width,
        existing_pool=existing, cap=requested_cap(), previous=previous, last_change=changed, now=now,
    )
    # The mmap is rounded down to whole slots. Compare against the configured
    # size recorded by this planner, so a warm pool is retained across turns.
    if pool is not None and budget.expert_pool_bytes != previous:
        freed = pool_lifecycle.release_decode_pool(store, synchronize=synchronize, clear_cache=clear_cache)
        if not freed:
            return replace(budget, expert_pool_bytes=previous or pool_lifecycle.pool_bytes(pool))
    store._flashnext_elastic_pool_bytes = budget.expert_pool_bytes
    if budget.expert_pool_bytes != previous:
        store._flashnext_elastic_changed = now
    if prefix_store is not None and sample["pressure"] >= 2:
        prefix_store.shrink_ram(0)
    return budget
