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

## Verdict

ANE offload makes Nemotron prefill **2.5× slower cold and memory-unsafe warm**
on the 16 GB Mac. This closes the 16 GB ANE prefill route with direct
end-to-end evidence, not just the earlier microbenchmarks. It does not rule
out ANE on a larger Mac where duplicate weights and memory headroom are not
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
