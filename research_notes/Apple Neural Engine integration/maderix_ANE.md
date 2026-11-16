# maderix/ANE — the "matrix" ANE backprop project (2026-11-17)

The user's "some guy named matrix" is **maderix** (Manjeet Singh). Primary
sources: https://github.com/maderix/ANE (MIT), the three-part Substack series
"Inside the M4 ANE", and the fork https://github.com/m0at/ANEtransformers
(Mistral 7B ANE prefill, README read in full).

## What he actually did
Full transformer **training** (forward + backward + Adam) on the ANE via
reverse-engineered private APIs (`_ANEClient`, `_ANECompiler`,
`_ANEInMemoryModel`), **bypassing Core ML entirely**. Training results on M4:
Stories110M (109M) 91 ms/step; Qwen3-0.6B (596M) 412 ms/step. He is explicit
that utilization is low (~5-9% of peak), many elementwise ops fall back to CPU,
and this is research code, not a production framework.

## Why this matters more than his training demo
1. **Direct private-API ANE dispatch is ~60-80 µs round-trip**, not the ~21 ms
   we measured through `coremltools`/Core ML. This is ~250-350x less overhead —
   it attacks the exact blocker we identified (the Python/Core ML marshalling),
   not the ANE math.
2. **Working GPU↔ANE zero-copy via shared IOSurface** — files `gpu_ane_share.m`
   and `gpu_prefill_ane_decode.m` in his repo. This is the concrete answer to
   "I don't want two allocations". Same IOSurface shared by Metal and ANE.
3. **ANE baked-weight prefill 76.6 tok/s** on Mistral 7B Q4_0, M5 24 GB
   (`m0at/ANEtransformers`). Single layer ANE vs BLAS: 23.9 ms vs 118.1 ms
   (4.9x). Note: still below MLX ~200+ tok/s on same chip — ANE wins on
   batched matmul but loses to a tuned Metal GEMM end-to-end. Decode stays on
   CPU SDOT (19.7 tok/s) because dispatch overhead dominates at S=1.
4. **2-phase loading**: cold compile of 128 programs ~80 s, warm start ~220 ms
   by forging the compiled program's `hexStringIdentifier` from the `aned`
   daemon cache. Cache is in-memory, lost on reboot.
5. **Compiled programs live in the `aned` daemon**, not the user process. User
   process holds only IOSurface mappings (~5 MB) — a big memory advantage.

## Hard constraints relevant to Veystrix
- Via private API, **INT8/INT4/LUT all FAIL to compile** — weights must be fp16.
  (Core ML quantized models work by an unknown internal path.) So the INT8 win
  we found through Core ML is NOT available with this approach.
- **Baked weights only** — runtime weight IOSurface override is dead. Each
   weight matrix needs its own compiled program; Mistral 7B = 128 programs.
- **No reduction ops** (`reduce_sum`, `rsqrt`) → RMSNorm cannot run on ANE →
  2 CPU↔ANE round-trips per layer, the main bottleneck.
- **Channel-first `[1,C,1,S]` layout** requires transposes; wrong layout gives
  silently wrong output (cosine 0.02 vs 0.998 with correct transpose).
- **Max 2 outputs per program** (3+ silently corrupt all but the last) — K+V
  fused, Q separate.
- **Minimum sequence S=16 (fp32) / S=32 (fp16)** — below this, silent all-zeros.
  Confirms our batch-1 finding.
- **~119 compile limit** per process (resource leak) → `exec()` restart or
  2-phase forge.
- **Kernel chaining (`_ANEChainingRequest`) fails**, Error Code=15, likely
  needs an entitlement. Every dispatch is a full CPU→ANE→CPU round-trip.
- **Numerics**: full 32-layer prefill cosine 0.998, max_err 0.61 — NOT
  bit-identical. Would need opt-in Tier B with a measured quality check.

## Owner rule (2026-11-17)
Validate every feature on the 16 GB M4 Mac first; larger Macs only after it
succeeds there. The 32 GB+ statements below are therefore deferred, not next
steps.

## Fit with Veystrix
- Flash-Next: 75.5 GB experts → fp16 baked copies impossible; route/gather is
  unsupported on ANE. Only the dense core (~3.4 GB → 6.8 GB fp16) could be a
  target, and only on large-Mac tiers.
- Nemotron: dense core 1.25 GB → 2.5 GB fp16, feasible on 16 GB. Its
  attention projections (q/k/v/o, hidden 2688) are conv1x1-shaped and are the
  cleanest first target.
- Shipping risk: private undocumented APIs may break with any macOS update,
  and affect App Store / notarization. Veystrix uses free ad-hoc distribution,
  which reduces (does not eliminate) that risk.

## Verdict
This is the native path identified as necessary in REPORT_NEMOTRON.md, except
it uses private ANE APIs instead of Core AI/MLTensor. It removes the marshalling
cost and provides zero-copy GPU↔ANE IOSurface sharing. It does NOT solve the
SSD-bound expert-read bottleneck, and it introduces fp16 numerics plus private
API fragility. Any adoption must be opt-in, quality-gated, and benchmarked
under the guard with the owner's explicit permission.
