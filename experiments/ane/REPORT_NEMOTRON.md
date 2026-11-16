# Nemotron prefill on the ANE — 16 GB M4 Air, 2026-11-17

Owner-directed retry after `REPORT.md` (2026-11-07). Everything below was
measured on the owner's 16 GB M4 MacBook Air, AC power, under
`scripts/flashnext/real_run_guard.py`, in an isolated venv with coremltools
9.0 / MLX 0.32.2 / mlx-lm 0.31.3. No production file was changed. The repo
`.venv` was not modified.

## Per-projection result (real checkpoint weights)

Layer 5 `q_proj` (`backbone.layers.5.mixer.q_proj`, Q4 G64 affine) was read
from the real safetensors shard, dequantized once to fp16, and run as a Core ML
`inner_product` on ANE versus the same GEMM in MLX:

| Prefill rows | ANE | MLX GPU | ANE speedup | max abs diff |
|---|---|---|---|---|
| 512 | 3.47 ms | 4.63 ms | 1.33× | 0.0034 |
| 2048 | 11.43 ms | 17.63 ms | 1.54× | 0.0040 |
| 4096 | 22.33 ms | 35.10 ms | 1.57× | 0.0041 |

So the kernel itself is genuinely faster. Numerics differ by ~4e-4 in absolute
terms (outputs ~10) — **not bit-identical**, so any use would be an opt-in
Tier B path with a measured quality check.

## End-to-end prefill (13 attention layers, q_proj + o_proj on ANE)

Prototype hook: `ane_nemotron_hook.py` bakes each live projection to a Core ML
model on first use, routes prefill calls through it, and leaves decode and
k/v projections on MLX. Harness: the project's own `qualification.py`,
2,048-token cold prompt, 128 output tokens.

| Arm | prefill tok/s | TTFT | token digest | guard |
|---|---|---|---|---|
| baseline | **239.1** | 8.59 s | `02467b73…` | clean, floor 4.58 GiB |
| ANE (cold) | **93.8** | 21.88 s | `1985f667…` (differs) | clean, floor 3.18 GiB, compressor 6.2 GB |
| baseline (warm) | **288.8** | 7.11 s | — | clean |
| ANE (warm) | — | — | — | **STOPPED: memory floor crossed** (2.95 GiB, compressor 6.4 GB) |

The warm run amortises the 26 Core ML compiles, so compilation is not the
whole story: the integrated path still cannot run inside the 3 GiB guard.
Per-call marshalling (GPU→CPU copy, fp16 conversion, Core ML predict, result
copy back) costs far more in the live engine than in the isolated benchmark,
and the duplicate fp16 weights (26 × ~21 MB) plus Core ML runtime push the
compressor from ~2.8 GB to ~6.4 GB. The output digest also changes, so the
bit-identical gate would fail even if speed were acceptable.

## Follow-up: weight format and native caller experiments (same day)

Everything above was re-attacked at the integration layer, since the isolated
kernel is genuinely faster:

| Experiment | Result |
|---|---|
| Native PyObjC `MLMultiArray` over an MLX/DLPack buffer | **Segfault** — MLX's buffer is not CPU-safe for Core ML |
| Native PyObjC with an external NumPy pointer | Segfaults during predict; Core ML does not retain the buffer |
| 4-bit palettized (`kmeans`/`uniform`, mlprogram) | **Silently falls back to GPU**, 16 ms, mean error 0.10–0.21 (fp16: 0.00027) |
| fp16 **mlprogram** (not neuralNetwork) | ANE, **6.3 ms** vs MLX GPU 17.6 ms — best isolated kernel |
| INT8 mlprogram, per-channel | ANE, **5.24 ms**, half fp16's weight bytes, mean error 0.0093 |
| End-to-end INT8, 13 attention layers, warm | Memory fixed (floor 3.6 GiB) but **279 vs 289 tok/s — 3.3% slower** |
| End-to-end INT8, 4 attention layers, warm | 274 tok/s; still slower |
| End-to-end INT8, Mamba `in_proj`, 8 layers | 276 tok/s; full 39 layers **tripped memory guard** (compressor 7.3 GB) |
| Baseline warm, for reference | **288.8 tok/s**, floor 4.65 GiB |

The kernel is 3.4× faster but each ANE call costs roughly **21 ms of marshalling**
(MLX GPU sync, fp16/fp32 conversion, Core ML predict, copy back), versus about
12 ms of GPU compute saved per projection. With 26 calls the overhead exceeds
the saving. Memory pressure from the duplicate weights adds a second penalty.

## Verdict

ANE offload makes Nemotron prefill **slower in every end-to-end configuration
tested on the 16 GB Mac**: 2.5× slower cold with fp16, 3.3–4.4% slower warm with
INT8, and memory-guard failures at full coverage. The blocker is the
MLX↔Core ML boundary, not the ANE kernel. Overcoming it requires either a
native unified runtime where tensors never leave Core ML/MLTensor land (Apple's
Core AI on macOS 27+, not available for these custom MoE/INT8-attention ops),
or a single ANE call per layer covering every projection with no host copies.
Neither is achievable in the current Python/MLX engine. This does not rule out
ANE on a larger Mac where duplicate weights and memory headroom are not
constraints; that remains the 32 GB+ track in `REOPEN_PLAN.md`.

## Files

- `export_nemotron_qproj.py` — read-only export of one real projection.
- `ane_nemotron_qproj.py` — isolated ANE vs MLX measurement.
- `ane_nemotron_hook.py` — runtime prototype hook (experiments only).
- `run_nemotron_prefill_ane.py` — qualification harness wrapper.
- `results/ane_nemotron_qproj.json`, `results/ane-e2e-*.jsonl`,
  `results/ane-warm-*.jsonl` — raw measurements.
- `results/ane_baseline_prefill.json`, `results/ane_baseline_contention.json`
  — earlier baseline reproduction on this machine.
