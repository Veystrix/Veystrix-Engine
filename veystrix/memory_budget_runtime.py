"""Runtime memory plan for the C++ engine, computed in Python before engine open.

Single source of truth for both engines: the tier table and the 5 GiB hard swap
ceiling live in veystrix.memory_tiers; this module reads the live machine state
(the same ctypes calls as scripts/flashnext/real_run_guard.py) and emits the
values the C++ profiles consume.

Nothing here enforces a limit: userland cannot stop the kernel from swapping.
The plan FAILS CLOSED to streaming (pool=0, resident=False, max_sessions=1)
when memory is short. The guard enforces the hard ceiling by stopping the run.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from dataclasses import dataclass

from veystrix.memory_tiers import (
    GIB,
    current_tier,
    flashnext_pool_cap_bytes,
    full_resident_bytes,
    swap_ceiling_bytes,
    swap_under_ceiling,
)


@dataclass(frozen=True)
class RuntimeBudget:
    total_bytes: int
    available_bytes: int
    swap_used_bytes: int
    pressure: int
    pool_bytes: int              # optional expert pool (0 = SSD streaming)
    nemotron_slots: int          # 0 = use the C++ default
    resident: bool               # model fits: full-resident, no SSD streaming
    max_sessions: int            # admission cap under the swap ceiling

    def profile_updates(self) -> dict:
        """Keys merged into the C ABI profile_json by the cpp backend."""
        return {
            "expert_pool_bytes": int(self.pool_bytes),
            "nemotron_slots": int(self.nemotron_slots),
            "resident": bool(self.resident),
            "max_sessions": int(self.max_sessions),
        }


class _VMStatistics64(ctypes.Structure):
    _fields_ = [
        ("free_count", ctypes.c_uint32), ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32), ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64), ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64), ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64), ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64), ("hits", ctypes.c_uint64), ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32), ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64), ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64), ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32), ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32), ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


class _XswUsage(ctypes.Structure):
    _fields_ = [("total", ctypes.c_uint64), ("avail", ctypes.c_uint64),
                ("used", ctypes.c_uint64), ("pagesize", ctypes.c_uint32),
                ("encrypted", ctypes.c_uint32)]


def read_machine() -> tuple[int, int, int, int]:
    """(total, available, swap_used, pressure). Falls back to conservative values."""
    total = int(os.environ.get("VEYSTRIX_PHYSICAL_MEMORY_GIB", "0")) or 0
    if total:
        total = int(total * GIB)
    else:
        try:
            total = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=5).strip())
        except Exception:
            try:
                total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            except Exception:
                total = 16 * GIB
    page = os.sysconf("SC_PAGE_SIZE")
    available = total // 4  # conservative fallback
    swap_used = 0
    pressure = 1
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        stats = _VMStatistics64()
        count = ctypes.c_uint32(ctypes.sizeof(stats) // 4)
        host = libc.mach_host_self()
        libc.host_statistics64.argtypes = [ctypes.c_uint32, ctypes.c_int,
                                           ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        libc.host_statistics64.restype = ctypes.c_int
        if libc.host_statistics64(host, 4, ctypes.byref(stats), ctypes.byref(count)) == 0:
            available = (stats.free_count + stats.inactive_count + stats.speculative_count) * page
        swap = _XswUsage()
        size = ctypes.c_size_t(ctypes.sizeof(swap))
        if libc.sysctlbyname(b"vm.swapusage", ctypes.byref(swap), ctypes.byref(size),
                             None, 0) == 0:
            swap_used = int(swap.used)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            text=True, timeout=5).strip()
        pressure = int(out)
    except Exception:
        pass
    return total, available, swap_used, pressure


def plan(*, total=None, available=None, swap_used=None, pressure=None,
         dense_bytes=0, routed_expert_bytes=0, record_bytes=0, kv_bytes_per_token=0,
         session_count=1, context_tokens=0, environ=None) -> RuntimeBudget:
    """Size the expert pool from live memory and the model, not from a fixed table.

    pool = what is actually free beside the resident dense core, the KV for the admitted
    sessions and the OS reserve, bounded by the bytes the model can use (all routed experts).
    With the model unknown it falls back to the old tier cap so behaviour never gets looser
    than before. Safety margins (reserve, swap ceiling, pressure) are never scaled up.
    ``VEYSTRIX_POOL_CAP_BYTES`` caps the pool explicitly.
    """
    env = environ if environ is not None else os.environ
    if total is None or available is None or swap_used is None or pressure is None:
        machine = read_machine()
        total = machine[0] if total is None else total
        available = machine[1] if available is None else available
        swap_used = machine[2] if swap_used is None else swap_used
        pressure = machine[3] if pressure is None else pressure

    reserve = max(3 * GIB, total // 4)
    ceiling = swap_ceiling_bytes()

    # Fail closed under critical VM pressure or swap pressure: no extra pool, no extra sessions.
    # Note: pressure == 2 (macOS warning) is normal on active desktops and must not disable
    # dynamic allocation on higher-RAM Macs when plenty of RAM is available.
    if not swap_under_ceiling(swap_used) or swap_used > ceiling - GIB:
        return RuntimeBudget(total, available, swap_used, pressure, 0, 0, False, 1)
    if pressure >= 4 or available < reserve:
        return RuntimeBudget(total, available, swap_used, pressure, 0, 0, False, 1)

    # Full residency when the whole routed expert set fits beside the dense
    # core and the OS reserve (same rule as models/nemotron35/admission.py).
    resident_bytes = full_resident_bytes(
        routed_expert_bytes=routed_expert_bytes, dense_bytes=dense_bytes,
        total_bytes=total)
    # Resident needs the experts to fit in what is free NOW (page cache that can be dropped
    # counts as available), not merely in the machine's nominal size.
    resident = resident_bytes > 0 and resident_bytes + dense_bytes + GIB // 2 <= available

    working = dense_bytes + session_count * context_tokens * kv_bytes_per_token
    headroom = max(0, min(available - reserve, total - reserve - working) - GIB // 2)
    if routed_expert_bytes:
        cap = routed_expert_bytes                     # nothing beyond "every expert" is useful
    else:
        cap = flashnext_pool_cap_bytes(current_tier(env))   # model unknown: stay conservative
    if total < 20 * GIB:
        # The 16 GB Mac is the only validated machine: its optional pool stays at the measured
        # 2 GB cap no matter how much page cache happens to be free right now.
        cap = min(cap, flashnext_pool_cap_bytes(current_tier(env)))
    override = env.get("VEYSTRIX_POOL_CAP_BYTES")
    if override and override.isdigit():
        cap = min(cap, int(override))
        resident = resident and resident_bytes <= cap     # an explicit cap also forbids residency
    pool = min(cap, headroom)
    unit = record_bytes if record_bytes > 0 else 64 * 2**20
    pool = (int(pool) // unit) * unit                  # whole records only
    if resident:
        pool = max(pool, resident_bytes // unit * unit)

    # Each concurrent session needs a bounded share of the swap headroom.
    per_session = max(0, ceiling - swap_used) // GIB
    max_sessions = max(1, min(4, per_session))
    # Nemotron's slot pool only pays off when (nearly) everything fits: a small pool measured
    # harmful on 16 GB. Resident uses exact keys; a big partial pool is allowed from 6 GiB.
    if resident and routed_expert_bytes and record_bytes:
        slots = int(resident_bytes // record_bytes)
    elif resident and routed_expert_bytes:
        slots = int(resident_bytes // 5_612_544)
    elif record_bytes and pool >= 6 * GIB:
        slots = int(pool // record_bytes)
    else:
        slots = 0
    return RuntimeBudget(int(total), int(available), int(swap_used), int(pressure),
                         int(pool), int(slots), resident, max_sessions)
