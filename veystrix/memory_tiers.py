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


# MLX shape dimensions are 32-bit; Flash-Next passes its decode pool to Metal
# kernels as a single uint32 array, so no pool can exceed ~8 GiB until the
# kernel addressing is redesigned. This is a hardware/kernel ceiling, not a
# memory-size decision, and applies equally to every tier.
FLASHNEXT_KERNEL_POOL_CAP_BYTES = 8 * GIB

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


def flashnext_pool_cap_bytes(tier: MemoryTier) -> int:
    """Largest Flash-Next pool the Metal kernel can address on this tier."""
    return min(round(tier.pool_cap_gb * 1_000_000_000), FLASHNEXT_KERNEL_POOL_CAP_BYTES)


def full_resident_bytes(*, routed_expert_bytes: int, dense_bytes: int, total_bytes: int,
                        os_reserve_bytes: int | None = None) -> int:
    """Bytes needed to keep every routed expert resident beside the dense core.

    Returns 0 when the machine cannot hold them plus the OS reserve — the caller
    must then fall back to SSD streaming. The reserve is the same rule Nemotron
    already uses (3 GiB floor or a quarter of RAM), never lowered to make a fit.
    """
    if os_reserve_bytes is None:
        os_reserve_bytes = max(3 * GIB, total_bytes // 4)
    if total_bytes < dense_bytes + routed_expert_bytes + os_reserve_bytes:
        return 0
    return routed_expert_bytes
