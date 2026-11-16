# Architecture decision log (ADR)

Short, durable records of decisions that shape the project. Each entry states
the decision, the reason, and the consequences. Superseded entries stay here
with a note pointing to their replacement.

## ADR-001 — Support exactly two model families (2026-11-09)

**Decision.** The engine and app support Qwen 3.8 Flash-Next (`flashnext`) and
NVIDIA Nemotron 3.5 Lightning (`nemotron35-lightning`). Removed families
(`qwen27b`, `k2_horizon`, `bonsai2`) stay rejected in `veystrix/preferences.py`.

**Why.** Each extra family adds a backend, a memory plan, an app settings path
and a validation surface. Two families are what the owner actually runs.

**Consequence.** New model requests are evaluated as a swap of an existing
family, not as a third backend.

## ADR-002 — Keep Flash-Next as the default until Qwen 4 Flash is validated (2026-11-16)

**Decision.** Qwen 3.8 Flash-Next remains the working default. Qwen 4 Flash is
adopted only after it is downloaded, loaded and validated on the owner's Mac.
Rollback keeps the old checkpoint folder and alias; no weights or caches are
deleted. Cutover details and the intake spec are prepared on the `qwen-4-flash`
branch in `docs/transition/qwen-4-flash.md`.

**Why.** The owner uses this as a daily coding model. Prose that merely looks
right is not enough — token digests, memory behaviour, tool calls and the
quality suite must pass on the target hardware first.

**Consequence.** Do not weaken preflight profiles, expert bounds, MTP checks or
memory admission just to make a new checkpoint load.

## ADR-003 — One identity descriptor per family, consumed by engine and app (2026-11-16)

**Decision.** `veystrix/model_identity.py` and the app's
`VeystrixKit/ModelIdentity.swift` are the single sources of truth for
`model_type` allowlists, display names, served IDs, download repos, checkpoint
aliases/folders and context limits. Preflight architecture data lives in
`models/flashnext/preflight_profiles/*.json`.

**Why.** Before this, a model swap touched dozens of scattered literals, which
made drift likely and cutover risky.

**Consequence.** A compatible successor checkpoint is onboarded with a new
profile/manifest JSON and an identity update, not a hunt through the tree.

## ADR-004 — Linux CI fast lane for pure-Python tests (2026-11-16)

**Decision.** An `ubuntu-latest` job runs the 39 shared test modules that need
no MLX (~17s); macOS CI keeps the full suite plus the app build and all 16
Nemotron unit modules.

**Why.** The macOS job is the bottleneck for fast feedback; most shared logic
is pure Python and needs no GPU.

**Consequence.** A new shared test that imports MLX must be added to the
documented runtime set in `veystrix/tests/_linux.py`, or it is skipped on Linux
and still runs on macOS.
