# Nemotron real-checkpoint logit-digest gap: localization and fix

**Status (measured): FIXED — Nemotron parity gate PASS, 4/4 tokens and 4/4
logit SHA-256 digests.**
All real-checkpoint commands were run under `scripts/flashnext/real_run_guard.py
--floor-gib 1.5 --swap-growth-mib 5120 --compressor-growth-mib 8192
--thermal-level 2`; every guarded run reported `stopped: null`, zero swap
growth (`swapout_mib: 0`), and exit 0 except the intentional gate failures.

## Symptom

`nemotron_generate` produced the correct greedy tokens
(1010, 1044, 1032, 1050) but the wrong logit digests: C++ step 0
`7acc8fc914cd…` vs Python `549ff4ec65a8…`.

## Method

1. New diagnostic `cpp/engine/tools/dump_nemo_py_layers.py` mirrors
   `gen_nemotron_reference.py` (pinned env: `KV=bf16`, `GATHER=1`,
   `FAST_SSM=0`, `PARALLEL_MAMBA=0`, chunk_tokens=1, zero-seeded caches) and
   reads the exact `prompt_ids` from `/tmp/nemo_ref.json`. It class-hooks
   `mlx_lm.models.nemotron_h.NemotronHBlock.__call__` (all 52 layers). The
   captured tensor is the same one C++ digests: block output `x + mixer(norm(x))`
   (`nemotron_h.py:455`) == C++ `h = h + delta` (`nemotron_generate.cpp:674`),
   after the residual add and before `norm_f`/`lm_head`. Digest = SHA-256 of
   float32 little-endian bytes (`digest_array`, `nemotron_generate.cpp:128`).
2. C++ side: `NEMOTRON_DUMP_LAYERS=1` prints `[NEMOLAYER step L=XX] <sha256>`
   for every layer of every step. With 6 prompt tokens there are 6 groups of 52
   lines; group 6 (last prefill step) produces step-0 logits.
3. For op-level narrowing, temporary instrumented copies of
   `nemotron_generate.cpp` were built in `/tmp` (no shared-file edits for the
   probes) to print intermediate digests; the Python dumper gained matching
   `--dump-moe-layer` / `--dump-mamba-layer` modes.

## First divergent layers (measured)

| Stage | First divergent layer | Which op |
|---|---|---|
| Before fixes | L1 (MoE), step 5 | relu2 / routed + shared expert |
| After fix 1 | L28 (mamba) step 4, L30 step 5 | `compute_dt` |
| After fix 2 | L10 (MoE), decode step 2 | router score normalization |
| After fix 3 | none | full parity |

Step-0 input/embedding and layer 0 (mamba) matched from the start, so the gap
was never in the embedding, norm, or mamba primitive in isolation.

## Root causes and fixes

All three are the same pattern: Python runs the computation through
`@mx.compile` (fused kernel, weak scalar `0`), while C++ used eager ops and/or a
strongly typed `mx::array(0.0f)`. The fused kernels differ by ~1 ULP, which is
enough to change bf16 outputs and, eventually, greedy logits.

1. **relu2 dtype promotion.** `mx::square(mx::maximum(u, mx::array(0.0f)))`
   promoted bf16 `u` to f32; Python `nn.relu2` uses a weak scalar `0` and stays
   bf16. Evidence: at L1 step 5, Python `su` digest `d5802e83…` matched C++,
   C++ `sa` `6e786410…` matched Python's `relu2(su, f32-zero)` variant, while
   Python's real `sa` was `fd692b75…`.
   Fix: `mx::astype(mx::array(0.0f), u.dtype())` in `engine.cpp`
   (`moe_block_relu2`) and in `nemotron_generate.cpp` (shared expert `sa`).
2. **`compute_dt` fusion.** Python's `compute_dt` is `@mx.compile` around the
   compiled `nn.softplus`; the C++ eager `logaddexp`/`clip` chain differs by
   ~1 ULP (measured 391/4096 elements on a representative f32 tensor).
   Evidence: Python `dt_py` `37b0b5bc…` vs C++-style `dt_cpp` `f13ca642…` at
   layer 28. Fix: compile the same graph in `mamba_kernel.cpp`
   (`mx::compile(..., /*shapeless=*/true)`), mirroring `hc_silu`.
3. **Router `group_expert_select` fusion.** Gate logits and top-k indices
   matched exactly, but C++ scores `3de4c771…` vs Python `2d144401…` at L10
   decode step 2. Evidence: in-process replay showed unfused scores reproduce
   the C++ digest and compiled scores reproduce Python's. Fix: compile the
   sigmoid→bias→argpartition→take_along_axis→normalize graph in
   `nemotron_route` (`nemotron_generate.cpp`). `mx::compile` could not infer
   `mx::slice` shapes, so the top-k selection uses `take_along_axis` with a
   statically shaped `[1,1,k]` slot index array.

## Final parity gate (measured)

```
step  0: token OK (got   1010, ref   1010)  digest OK (got 549ff4ec65a8, ref 549ff4ec65a8)
step  1: token OK (got   1044, ref   1044)  digest OK (got b5292f2f4cbf, ref b5292f2f4cbf)
step  2: token OK (got   1032, ref   1032)  digest OK (got 0826df9290ab, ref 0826df9290ab)
step  3: token OK (got   1050, ref   1050)  digest OK (got 87618bbe56e3, ref 87618bbe56e3)
PASS: 4/4 tokens and 4/4 logit digests match the reference
guard: {"event": "end", "stopped": null, "returncode": 0, "swapout_mib": 0, ...}
```

Synthetic regression tests (no guard needed): `ctest -R 'relu2|mamba|engine_parity'`
— 5/5 passed (`engine_parity`, `engine_relu2`, `mamba_kernel`, `mamba_layer`,
`engine_parity_real`).

## Files changed

- `cpp/engine/src/engine.cpp` — relu2 zero dtype in `moe_block_relu2`.
- `cpp/engine/src/mamba_kernel.cpp` — compiled `compute_dt`.
- `cpp/engine/tools/nemotron_generate.cpp` — shared-expert relu2 zero dtype,
  compiled `nemotron_route`.
- `cpp/engine/tools/dump_nemo_py_layers.py` (new) — guarded Python per-layer
  digest dumper with optional MoE/mamba intermediate modes.
- This document (new).

No commit was made. CMakeLists and other subagent files were not touched.
