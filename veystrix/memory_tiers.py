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
    prefill_limit: int = 32768  # safe single-request cold/incremental prefill ceiling
    prefill_chunk: int = 2048   # default C++ prefill chunk tokens
    mlx_cache_mb: int = 256     # default MLX free-buffer cache cap (MiB)
    kv_budget_fraction: float = 0.35
    parked_sessions: int = 1
    parked_tokens: int = 16384
    prefix_cache_quota_gb: float = 4.0


# MLX shape dimensions are 32-bit; Flash-Next passes its decode pool to Metal
# kernels as a single uint32 array, so no pool can exceed ~8 GiB until the
# kernel addressing is redesigned. This is a hardware/kernel ceiling, not a
# memory-size decision, and applies equally to every tier.
FLASHNEXT_KERNEL_POOL_CAP_BYTES = 8 * GIB

BASELINE = MemoryTier(
    "16 GB", 2.0, 4.0, prefill_limit=32768, prefill_chunk=1024, mlx_cache_mb=256,
    kv_budget_fraction=0.35, parked_sessions=1, parked_tokens=16384, prefix_cache_quota_gb=4.0,
)
# Every Apple-silicon unified memory tier (16/24/32/48/64/96/128 GB) gets tuned
# prefill admission ceilings, chunk sizes, MLX cache caps, KV budget shares,
# RAM-parked session pools, and persistent SSD prompt-cache quotas.
TIERS: tuple[tuple[int, MemoryTier], ...] = (
    (112, MemoryTier("128 GB", 12.0, 8.0, prefill_limit=1_048_576, prefill_chunk=8192, mlx_cache_mb=2048,
                     kv_budget_fraction=0.50, parked_sessions=12, parked_tokens=524_288, prefix_cache_quota_gb=12.0)),
    (80,  MemoryTier("96 GB",  8.0,  8.0, prefill_limit=1_048_576, prefill_chunk=8192, mlx_cache_mb=2048,
                     kv_budget_fraction=0.50, parked_sessions=12, parked_tokens=524_288, prefix_cache_quota_gb=12.0)),
    (56,  MemoryTier("64 GB",  6.0,  6.0, prefill_limit=524_288,   prefill_chunk=8192, mlx_cache_mb=1024,
                     kv_budget_fraction=0.50, parked_sessions=8,  parked_tokens=262_144, prefix_cache_quota_gb=10.0)),
    (43,  MemoryTier("48 GB",  5.0,  5.0, prefill_limit=262_144,   prefill_chunk=4096, mlx_cache_mb=1024,
                     kv_budget_fraction=0.45, parked_sessions=6,  parked_tokens=131_072, prefix_cache_quota_gb=8.0)),
    (28,  MemoryTier("32 GB",  4.0,  4.0, prefill_limit=131_072,   prefill_chunk=4096, mlx_cache_mb=512,
                     kv_budget_fraction=0.45, parked_sessions=4,  parked_tokens=65_536,  prefix_cache_quota_gb=8.0)),
    (20,  MemoryTier("24 GB",  3.0,  4.0, prefill_limit=65_536,    prefill_chunk=2048, mlx_cache_mb=512,
                     kv_budget_fraction=0.40, parked_sessions=2,  parked_tokens=32_768,  prefix_cache_quota_gb=6.0)),
    (0,   BASELINE),
)

# Hard swap ceiling (owner: 5 GB). This is a PROCESS limit, never scaled by
# installed RAM — a larger machine must not be allowed to swap more.
SWAP_HARD_LIMIT_BYTES = 5 * GIB


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


def swap_ceiling_bytes() -> int:
    """Hard process swap ceiling. Never scaled by installed RAM."""
    return SWAP_HARD_LIMIT_BYTES


def swap_under_ceiling(swap_used_bytes: int) -> bool:
    """True while the process is inside the 5 GB hard swap ceiling."""
    return 0 <= int(swap_used_bytes) < SWAP_HARD_LIMIT_BYTES


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
