"""Physical-memory tiers for pool sizing on Apple-silicon Macs.

The validated baseline is a 16 GB Mac. Larger Macs can allocate a larger
optional expert pool, but the *safety* thresholds (start gate, swap ceiling,
guard floor) are deliberately NOT scaled: they came from real-machine
measurements, not from installed RAM, and lowering them would be unsafe.

Pool caps are about 12-17% of usable unified memory. Larger capacities have
more bandwidth as well as more RAM, but the table stays conservative because
the larger tiers are not validated on hardware yet.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

GIB = 2 ** 30


@dataclass(frozen=True)
class MemoryTier:
    name: str
    pool_cap_gb: float          # user-facing GB (decimal), as in --pool-gb
    nemotron_pool_cap_gb: float  # auto Nemotron pool cap in GiB


BASELINE = MemoryTier("16 GB", 2.0, 4.0)
TIERS: tuple[tuple[int, MemoryTier], ...] = (
    (112, MemoryTier("128 GB", 12.0, 8.0)),
    (80, MemoryTier("96 GB", 8.0, 8.0)),
    (48, MemoryTier("64 GB", 6.0, 6.0)),
    (28, MemoryTier("32 GB", 4.0, 4.0)),
    (20, MemoryTier("24 GB", 3.0, 4.0)),
    (0, BASELINE),
)


def physical_bytes(environ=None) -> int:
    env = os.environ if environ is None else environ
    override = env.get("VEYSTRIX_PHYSICAL_MEMORY_GIB")
    if override:
        try:
            return max(0, int(float(override) * GIB))
        except ValueError:
            pass
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 16 * GIB


def tier_for(total_bytes: int) -> MemoryTier:
    gib = total_bytes / GIB
    for minimum, tier in TIERS:
        if gib >= minimum:
            return tier
    return BASELINE


def current_tier(environ=None) -> MemoryTier:
    return tier_for(physical_bytes(environ))
