# Veystrix native decode engine (C++)

A C++ port of Veystrix's MoE **decode compute path**, built against the **same
MLX 0.32.2** as the Python engine so outputs can be bit-identical. This is the
foundation of the engine rewrite: the per-token MoE block (the dominant compute
and SSD-read path) is implemented natively and verified byte-for-byte against
the real Python ops and real checkpoint weights.

## Status (2026-11-17) — core MoE compute DONE and bit-exact

Implemented in `src/`:
- `router()` — softmax(precise) + argpartition + take_along_axis + renorm,
  matching `adaptive_topk._moe_call` at threshold 1.0.
- `q4_projection()` — `mx.gather_qmm` affine Q4 (G32/G64), matching
  `StreamingSwitchLinear.__call__`.
- `swiglu()`, `weighted_combine()` — matching the expert activation + score
  combine in `expert_cache.py` / `metal_runtime.py`.
- `moe_block()` / `moe_block_from_logits()` — the composition, one MoE layer.
- `moe_block_relu2()` — Nemotron-style relu2 MLP (single up projection, no gate).
- `rms_norm()` — one-centered production RMSNorm (`patch_rmsnorm.py`).
- `gdn_normalize_qk()`, `gdn_gate_sigmoid()`, `gdn_gate_silu()` — GDN chains
  (`compile_gdn.py`), used in every linear-attention layer.
- `gated_residual()` — Qwen4 hyper-connection / gated residual, used twice per
  decoder layer (`Qwen4ExpGatedResidual.__call__`).
- `sample_greedy()` — production greedy sampler (`veystrix/sampling.py`).
- `src/expert_store.cpp` — native safetensors expert loader (`load_layer`),
  loads a real 512-expert Qwen layer straight from the checkpoint with no
  Python feeding weights.

## Build & test

```sh
cd cpp/engine
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure        # 4 tests, all bit-exact
./build/test_engine_parity tests/vectors           # 3 shapes vs Python
./build/test_native_layer <shard> <prefix> tests/vectors/real
```

Regenerate parity vectors (Python side, same MLX 0.32.2):
```sh
.venv/bin/python cpp/engine/tools/gen_vectors.py --out cpp/engine/tests/vectors
.venv/bin/python cpp/engine/tools/gen_real_vectors.py \
  --checkpoint /Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --out cpp/engine/tests/vectors/real
```

Vectors are generated artifacts (`.gitignore`); only the generators are tracked.

## Test evidence (latest ctest run)

```
1/5 engine_smoke .......... Passed
2/5 engine_parity ......... Passed   (small, Qwen G32 [2560->640], Nemotron G64 [2688->1856])
3/5 engine_relu2 .......... Passed   (Nemotron relu2 MoE)
4/5 engine_parity_real .... Passed   (real Qwen expert weights)
5/5 engine_native_layer ... Passed   (C++ loads real layer, bit-identical)
100% tests passed
```

## Bit-exactness methodology
Compare raw bytes (`view` to uint32), not tolerances — the same technique the
repo uses (`phase2-engineering-evidence.md`). Parity is checked for router
indices/scores, the full MoE block output, RMSNorm, and greedy sampling.

## What remains for a COMPLETE engine rewrite

The MoE compute is done; these are still Python and must be ported for a
standalone native decode loop:
1. **Attention**: GatedDeltaNet (linear attention, `compile_gdn.py`) and QSA
   sparse attention (`qsa_chunk.py`) with KV-cache state.
2. **Hyper-connections / residual stream** in `Qwen4ExpDecoderLayer`
   (`.venv/.../qwen4_exp/language.py:744-772`) and the 48-layer loop (`:792-829`).
3. **KV cache** management (INT8 KV, Mamba/SSM recurrent state).
4. **Prefill** path (`prefill.py`, layer-major ring).
5. **Sampling** beyond greedy (temperature/top-p/top-k) — greedy is done.
6. **I/O integration**: wire `src/expert_store` + `cpp/decode/io_pager` into the
   layer loop (streamed experts, pool, native batched pread).
7. **Server**: HTTP/Swift bridge (`veystrix/server.py`) — optional; the engine
   can be driven as a library without it.

The dominant per-token compute (routed MoE) is already native and bit-exact;
attention + the layer loop is the remaining large piece. See
`docs/performance/ENGINE_LANGUAGE_ANALYSIS.md` for scope/estimates.

## Design notes
- C++ namespace `mlx::core` (Python `mx` is an alias); `namespace mx = mlx::core`.
- `MLX_ROOT` pinned to the venv MLX package so kernels are bit-identical.
- CMake 4.4 emits a harmless macOS-14 / built-for-26.2 deployment-target warning.
