# Exact speed profile (both models)

A single app toggle, **"Qwen exact speed (bit-identical, 24/24 quality gate)"**, enables the measured optimisations that were validated *together* on the real 16 GB M4 checkpoint. It is **on by default** on the `neural-engine` branch; the older "faster prefill and decode (experimental)" toggle (`short-throughput`) is unchanged and stays off. They are size-independent, so the same set works on larger Macs. Nothing here changes precision, routing, tokens or the OS/swap safety guards.

Because the profile requires the tiered4 hybrid KV cache, the app's KV selector also defaults to **Tiered INT4 / INT8** so the UI matches what the engine runs.

## What it enables

Flash-Next (`--enable exact-speed`): the curated set `kv-tiered4`, `layer-major-hybrid-4k`, `single-chunk-layer-major`, `merge-small-tail`, `qsa-shared-dense-kv`, `qsa-fused-gather`, `qsa-fused-sdpa`, `ngram-dedup`, `dense-gemm`, `s3fifo-pool`, plus `FLASHNEXT_QSA_GATHER_TILE=32`.

Nemotron: `VEYSTRIX_NEMOTRON_DENSE_GEMM=1` (bit-equal for shape-verified projections). Nemotron's other accepted speed defaults (prefetch order, expert pipelining, fused Mamba scan, SDPA prefill) are already on regardless of the toggle.

## Measured evidence (fanless 16 GB M4, 4-bit weights, full routing)

- Flash-Next prefill: single-chunk layer-major +14–30%, merge-small-tail +36% on a 4.3K cold prompt, shared dense KV −6%, fused gather −6.5%, fused SDPA −3.5%, ngram-dedup −3%, dense GEMM −2.4%.
- Flash-Next decode: S3-FIFO pool +5% tok/s with 5% fewer bytes read.
- Nemotron prefill: shape-verified dense GEMM +4% at 8K/16K.
- Combined quality gate: **24/24 token-identical cases, PASS** (Tier A bit-identical; Tier C no regression).

Source: `docs/performance/overnight-2026-11-16.md`.

## Rollback

Every member is an env-var-backed flag owned by the mode (`OWNED` in `modes.py`), so a saved `~/.frankenstein/env` cannot silently change it. Turning the toggle off restores the reference path. The older "faster prefill and decode (experimental)" toggle (`short-throughput`) is unchanged.

No live A/B run was launched for this change — the gains above are the existing paired measurements, and `AGENT_INVARIANTS.md` requires explicit permission for new benchmarks.
