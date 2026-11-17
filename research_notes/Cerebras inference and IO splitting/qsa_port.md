# QSA primitive C++ port — status & integration note

## What was ported

The **pre-RoPE projection + normalization path of the QSA indexer** for
Qwen3.8-Flash-Next. In the installed model package this lives in:

```
.venv/lib/python3.12/site-packages/mlx_vlm/models/qwen4_exp/language.py
  class Qwen4ExpQSAIndexer.__call__   (around lines 275-360)
  class Qwen4ExpRMSNorm.__call__      (around lines 215-237)
```

The referenced `models/flashnext/qsa_chunk.py` candidate does **not exist** in
this environment (no `flashnext` package under `mlx_vlm/models/`), so I ported
candidate #1 from the real Qwen4-exp source instead.

The primitive is genuinely on the production decode path: it is the part of
`Qwen4ExpQSAIndexer.__call__` that turns the layer input `x` into (a) the
normalized indexer **query** and (b) the normalized, block-mean-pooled indexer
**keys**, which are then RoPE'd and scored to select sparse KV blocks.

RoPE application and block selection (`argpartition`, `where`, `selected_tokens`)
were deliberately **not** included — see "Why pre-RoPE only" below.

## Why this primitive (smallest bit-exact, testable piece)

- It uses only standard, deterministic MLX ops: `matmul`, `reshape`, `slice`,
  `squeeze`, `astype`, `square`, `mean(keepdims)`, `rsqrt`, scalar add/mul,
  `transpose`, `expand_dims`. No custom/fused Metal kernels, no tie-breaking.
- The two RMSNorm calls use the REAL `Qwen4ExpRMSNorm` class (verified by the
  generator importing it from `mlx_vlm.models.qwen4_exp.language`), so the
  reference vectors come from the exact Python op path.
- Excludes the two risky stages: `_apply_rope` goes through `MRoPERotaryEmbedding`
  with `mrope_section` frequency layout and a `@mx.compile`-fused Metal apply
  kernel (fused-apply fast path gated on `_HAS_METAL`), and `argpartition` has
  device-dependent tie ordering. Neither is suitable for a small, stable,
  bit-exact standalone test. The chosen prefix is fully bit-exact on CPU/Metal
  MLX 0.32.2.

## Exact op order (mirrored one-for-one in qsa.cpp)

Python reference (no cache, no RoPE) and C++ mapping:

```
1. qk       = x @ proj_weight.T                      -> mx::matmul(x, mx::transpose(proj_weight))
2. qk       = qk.reshape(B, S, Hq + Hk, D)           -> mx::reshape
3. query    = qk[:, :, :Hq]                          -> mx::slice(qk, {0,0,0,0},{B,S,Hq,D},{1,1,1,1})
4. raw_keys = qk[:, :, Hq:].squeeze(2)               -> mx::slice(...,{0,0,Hq,0},{B,S,Hq+Hk,D},...) + mx::squeeze(...,2)
5. query    = q_layernorm(query).transpose(0,2,1,3)  -> qsa_rms_norm + mx::transpose({0,2,1,3})
6. blocks   = S // compress_ratio;  complete_len = blocks * ratio
7. pooled   = raw_keys[:, :complete_len].reshape(B, blocks, ratio, D)
8. pooled   = mean(pooled.astype(float32), axis=2).astype(raw_keys.dtype)   -> mx::astype(mx::mean(...,2), raw_keys.dtype())
9. pooled   = k_layernorm(pooled)                    -> qsa_rms_norm
10.pooled_k = expand_dims(pooled, 1)                 -> mx::expand_dims(pooled, 1)
```

`q_layernorm` / `k_layernorm` are `Qwen4ExpRMSNorm` (group_size is None in the
indexer). Its exact op order, also mirrored:

```
dtype = x.dtype
y = x.astype(mx.float32)
y = y * mx.rsqrt(mx.mean(mx.square(y), axis=-1, keepdims=True) + eps)
y = y * (1.0 + weight.astype(mx.float32))   # "one-centered" gain
return y.reshape(x.shape).astype(dtype)
```

Note the `1.0 + weight` Qwen4 convention (checkpoint weights are centered at
zero) — this is NOT the standard `mx.fast::rms_norm(x, weight, eps)`.

Dtype flow: `matmul` output dtype follows the input (`float32` / `bfloat16` /
`float16` all tested); RMSNorm computes in `float32` then casts back to the
input dtype; key mean-pooling casts to `float32` then back to `raw_keys.dtype`.

## Files created (all within my ownership)

- `cpp/engine/src/qsa.hpp` — `QSAConfig`, `QSARoPEConfig`, `QSAIndexerOutput`,
  `qsa_rms_norm`, `qsa_indexer_pre_rope`.
- `cpp/engine/src/qsa.cpp` — implementation.
- `cpp/engine/tools/gen_qsa_vectors.py` — generates vectors via the real
  Python path (uses the actual `Qwen4ExpRMSNorm`), output defaults to
  `cpp/engine/tests/vectors_qsa`.
- `cpp/engine/tests/test_qsa.cpp` — loads vectors with `mx::load`, compares via
  uint32 byte view (`bit_equal`), prints `test_qsa OK`.
- `cpp/engine/tests/vectors_qsa/` — generated npy vectors + `cases.json` +
  per-case `meta.json` (data artifacts, regenerable).

## Verification (standalone, no CMake edits)

```
cd /Users/vincent/Documents/Veystrix
.venv/bin/python cpp/engine/tools/gen_qsa_vectors.py --out cpp/engine/tests/vectors_qsa
MLXROOT=.venv/lib/python3.12/site-packages/mlx
xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
  cpp/engine/src/qsa.cpp cpp/engine/tests/test_qsa.cpp \
  -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$(pwd)/$MLXROOT/lib" -o /tmp/test_qsa
/tmp/test_qsa cpp/engine/tests/vectors_qsa
```

Output:

```
  [case0_float32_small] OK (query + pooled_keys bit-identical)
  [case1_bfloat16_decode] OK (query + pooled_keys bit-identical)
  [case2_float16_batch2] OK (query + pooled_keys bit-identical)
test_qsa OK (3 cases bit-identical to Python)
```

## Integration (for the owner)

1. **CMakeLists.txt** — add `src/qsa.cpp` to the engine library sources and add
   a `test_qsa` target built from `tests/test_qsa.cpp` (link `${MLX_LIBRARIES}`
   and the same include dirs as the other tests). Add an optional custom target
   that runs `gen_qsa_vectors.py` before the test if vectors are missing.
2. **engine.hpp / engine.cpp** — no changes are strictly required; this is a
   self-contained namespace `veystrix::engine` API. To expose it through the
   engine facade, `#include "qsa.hpp"` and forward `QSAConfig` from the Rust/FFI
   config layer when the full indexer is wired in.
3. No changes to existing tests or generators are needed.

The API is deliberately generic (see below); nothing in the signature or
implementation hardcodes a Qwen-only dimension or head count.

## Portability to Nemotron / future models

Nemotron 3.5 Lightning uses a different attention architecture — Mamba/SSM
(state-space) mixing plus attention — and **does not use QSA**. QSA is a
Qwen-family adapter: the indexer, compress-ratio block pooling and the
"one-centered" `Qwen4ExpRMSNorm` convention are specific to Qwen4-exp /
Qwen3.8-Flash-Next.

Model-specific parts of this primitive:
- The **Qwen4 `1.0 + weight` RMSNorm** (`qsa_rms_norm`) — specific to Qwen4-exp
  checkpoints (standard models store the gain directly).
- The **indexer's q/k split + compress-ratio block mean-pooling** geometry —
  specific to `Qwen4ExpQSAIndexer`.

Reusable parts (shared with Nemotron and future models):
- `matmul`, `reshape`, `slice`, `transpose`, `astype`, `mean`, `rsqrt` — plain
  MLX ops; the generic plumbing and the `QSAConfig` parameterization.
- The decode-side router / expert MoE / sampling primitives already in the C++
  engine (softmax, gather, gather_qmm, top-k) are model-agnostic.

The API avoids baked-in constants: `QSAConfig` carries `n_heads`, `kv_heads`,
`head_dim`, `compress_ratio`, `token_budget`, `block_topk`, `eps`, and a
`QSARoPEConfig` (`base`, `max_position_embeddings`, `mrope_section`, `style`,
`attention_scaling`). The current primitive consumes the dims/head counts,
compress ratio and eps; the budget/topk and RoPE fields are reserved for the
block-selection and RoPE stages and for future models that share the backbone
but differ in rotary layout. No Qwen literal (e.g. `indexer_n_heads`,
`indexer_compress_ratio`) is hardcoded in `qsa.cpp`.

## What remains for full QSA (not in scope here)

1. `qsa_apply_rope` with `MRoPERotaryEmbedding` (mrope_section frequency layout,
   `rotate_half`, cos/sin cast to input dtype) — higher parity risk due to the
   fused `@mx.compile` Metal path; recommend a dedicated bit-exact task.
2. Block scoring: `scores = query @ pooled_keys.transpose(0,1,3,2)`,
   `sum(maximum(scores.astype(float32), 0), axis=1)`, `/ sqrt(head_dim)`.
3. Causal masking + `argpartition(scores, kth=-block_topk)` selection and
   `selected_tokens` construction (`where`, `any`, `tail`).
4. `QSAKVCache` integration (`update_indexer`, `offset`, `index_position_ids`).
5. Wiring the resulting sparse mask into `Qwen4ExpAttention.__call__`.
