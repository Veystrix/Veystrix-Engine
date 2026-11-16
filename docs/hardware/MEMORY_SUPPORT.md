# Unified-memory support (16–128 GB)

## Status

The **16 GB Mac is the validated baseline**: every recorded pool, prefill and guard decision came from an M4 MacBook Air with 16 GB. Macs with 24, 32, 64, 96 and 128 GB are now *supported by configuration* — the app offers larger optional decode pools and the engine sizes them by a memory tier — but those tiers have **not** been validated on real hardware. Treat larger pools as an opt-in performance choice, not a measured guarantee.

## Why capacity alone is not enough

Two Macs with the same RAM can have very different bandwidth:

| Memory | Example chips | Bandwidth |
|---|---|---|
| 24 GB | M2/M3/M4/M5/M6 base | 100–170 GB/s |
| 32 GB | M4/M5 base, M1/M2 Pro | 120–200 GB/s |
| 64 GB | M4/M5 Pro and Max | 273–614 GB/s |
| 96 GB | M2/M3 Max, M3 Ultra | 300–819 GB/s |
| 128 GB | M3 Max, M4/M5 Max | 400–614 GB/s |

The tier table is therefore conservative and based on **physical memory**, not chip names.

## The tier table

`veystrix/memory_tiers.py` is the single source of truth:

| Tier (physical GiB) | Flash-Next pool cap (GB) | Nemotron auto-pool cap (GiB) | ~% of usable memory |
|---|---:|---:|---:|
| 16 (validated) | 2 | 4 | ~17% |
| 24 | 3 | 4 | ~17% |
| 32 | 4 | 4 | ~17% |
| 64 | 6 | 6 | ~12% |
| 96 | 8 | 8 | ~11% |
| 128 | 12 | 8 | ~12% |

Caps are about 12–17% of usable unified memory. The 16 GB baseline keeps its exact 2 GB cap; every large tier is bounded further by the headroom checks at the first decode.

## What was deliberately NOT changed

These are **absolute** limits from measured behaviour, not RAM fractions. They stay the same on every machine:

- 9 GiB start gate (`modes.START_GATE_GB`, `admission_model.MANDATORY_GATE_GIB`, `prefill_readiness.MANDATORY_START_GATE_BYTES`)
- 3 GiB guard floor (`prefill_scheduler.GUARD_FLOOR_BYTES`)
- 3 GiB swap-growth ceiling and 2.5 GiB swap-growth stop (`memory_policy`)
- 12.7 GB Metal working-set limit (`prefill_budget`)
- Nemotron OS reserve `max(3 GiB, total/4)` and `pool_pays_off`

Raising these would weaken safety guards to make a larger pool load — explicitly against the project rule.

## Environment override

Set `VEYSTRIX_PHYSICAL_MEMORY_GIB` to simulate a tier in tests or on unusual hardware; valid values are 16, 24, 32, 64, 96, 128 (or any number, mapped to the nearest lower tier).

## Verification

- `veystrix/tests/test_memory_tiers.py` — tier mapping, override, monotonic caps
- `models/flashnext/tests/unit/test_memory_tiers.py` — planner availability, fixed-size caps, tier-aware planner steps, Nemotron cap, unchanged start gate
- Swift `KitTests` — DecodePool options and the validated baseline label
