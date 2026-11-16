# FusionML forensic verification (offline, 2026-11-17)

The subagent recommended FusionML (arXiv 2607.22785) as the top actionable item:
"1.15–1.38× prefill, token-identical, measured on M4." I verified the primary
source at the git-object level. **The recommendation does not survive.**

## What I did
Cloned `ommo007/FusionML` and fetched full history (`git fetch --unshallow`).
Recovered the benchmark JSONs that were removed from `main` (the only commit on
main is `2e36b64 chore: ignore benchmarks folder on main branch`).

## Evidence found in git objects (all paths offline-verifiable)
- `benchmarks/results/reproducibility/Apple_M4_24GB/head_to_head_v2.json`
  (blob `051a28bf`), system: macOS 26.3, Apple M4, 24 GB, mlx 0.30.6.
- Same for M1 8GB, M3 Pro 18GB, M4 Pro 24GB.
- `vs_mlx.json`, `tri_compute.json`, `head_to_head.json`, `matmul.json`,
  `training.json` — all **synthetic** matmul/softmax/elementwise/forward_pass
  microbenchmarks.

## Numbers (Apple M4 24GB, recovered JSON)
- matmul 128: fusionml median 0.0525 ms with `backend_used: "cpu"`,
  `_fusionml_beats_best: false`; winner `pt_cpu` 0.0048 ms.
- forward_pass 32: fusionml 0.602 ms vs mlx 0.381 ms (numpy wins at 0.217).
- forward_pass 128: fusionml 0.763 ms vs **mlx 0.357 ms** (winner: mlx).
- forward_pass 512: mlx also wins (winner field `mlx`).

So FusionML's own released M4 numbers show it is **slower than MLX** at the
forward-pass sizes that matter for prefill. The one fast path (`fusionml_pipe`)
only approaches numpy/mlx, never beats them.

## The claimed LLM results are not in the repository
The paper claims 1.18–1.25× TTFT and token-identical output on Qwen2.5-7B through
stock MLX-LM. Searching **every object in every ref**: no LLM benchmark code, no
Qwen checkpoint reference, no generation/TTFT/token artifacts. The only `token`
match is an unrelated vendored `setuptools/.../_tokenizer.py`. The main branch's
`ANEEngine.swift` throws `ANEError.modelNotAvailable` (placeholder).

## Veystrix already has the "eager boundary" idea
FusionML's mechanism: insert `mx.eval` before a CPU+GPU row split so MLX's lazy
scheduler doesn't serialize cross-stream work. Veystrix's
`models/flashnext/layer_major.py` already calls `mx.eval` at the analogous
points (lines 726, 740, 759, 776) before dependent graphs proceed. The idea is
not new to us, and nothing in the recovered FusionML data suggests it beats MLX.

## Verdict
**Do not adopt FusionML.** Its released raw data contradicts its paper's headline
on the very chip (M4) that mattered; the LLM claims have no artifacts. Its
"eager boundary" mechanism is already present in Veystrix's layer-major prefill.
The right next step is the non-ANE expert-wise K=2 read sharing, not FusionML.
