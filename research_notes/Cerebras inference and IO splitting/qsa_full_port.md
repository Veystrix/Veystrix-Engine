# QSA full-path C++ port — status & integration note

This completes the **remaining half** of the `Qwen4ExpQSAIndexer` port. The
pre-RoPE projection/normalization primitive is already in `qsa.hpp/.cpp`; this
note covers **RoPE, block scoring/selection, and the KV-cache indexer state**.

## Production source ported

```
.venv/lib/python3.12/site-packages/mlx_vlm/models/qwen4_exp/language.py
  Qwen4ExpQSAIndexer.__call__        lines 306-380
    _apply_rope                      lines 300-304
  QSAKVCache.update_indexer          lines 54-63
  _append_indexer_positions          lines 21-38
.venv/lib/python3.12/site-packages/mlx_vlm/models/qwen3_5/language.py
  Qwen3_5RotaryEmbedding             lines 27-42
.venv/lib/python3.12/site-packages/mlx_vlm/models/rope_utils.py
  MRoPERotaryEmbedding.apply_rotary / __call__ / compute_inv_freq /
  compute_mrope_frequencies / _apply_interleaved_rotary_pos_emb_axis1
```

## Files created (all within ownership)

- `cpp/engine/src/qsa_full.hpp` — `QSARotaryParams`, `QSAFullConfig`,
  `QSACache`, `qsa_compute_inv_freq`, `qsa_apply_rope`, `qsa_full_indexer`.
- `cpp/engine/src/qsa_full.cpp` — implementation. Depends only on `qsa.hpp`
  (reuses `QSAConfig`, `qsa_rms_norm`) and the MLX C++ API. `qsa.cpp/.hpp`
  were NOT modified.
- `cpp/engine/tools/gen_qsa_full_vectors.py` — drives the REAL
  `Qwen4ExpQSAIndexer`, `QSAKVCache` and `Qwen3_5RotaryEmbedding`, runs a
  one-step decode with a pre-populated indexer cache, saves all inputs, the
  expected `qsa_mask` and the updated cache to `npy`, plus `meta.json`.
- `cpp/engine/tests/test_qsa_full.cpp` — loads vectors, compares via uint32
  byte view (same `bit_equal` pattern as `test_qsa.cpp`), covers **both** the
  `None` path and the selection path.
- `cpp/engine/tests/vectors_qsa_full/` — generated data (regenerable).

## What is implemented

1. **`qsa_apply_rope`** — partial rotary matching
   `MRoPERotaryEmbedding.apply_rotary`'s eager branch. `inv_freq =
   1/base**(arange(0,dim,2)/dim)`; `angle = pos.f32[...,None]*inv_freq`;
   `emb = concat([angle,angle])`; `cos/sin` cast to the input dtype;
   `rotate_half` on the first `dim` channels; remaining channels pass through.
   2-D text positions and 3-D MRoPE positions (interleaved/chunked section
   selector) are both supported.
2. **Block scoring** — full raw keys from the cache, mean-pool + `k_layernorm`,
   rope on pooled keys with `block_position_ids`, `query @ pooled.T`,
   `sum(max(scores.f32,0))/sqrt(head_dim)`, causal `-inf` masking.
3. **`argpartition` selection** — `argpartition(kth=-block_topk)[..., -topk:]`,
   `selected_tokens` via `any`/`tail`/`causal`/`where`, returns
   `selected_tokens[:,None]`; `std::nullopt` when
   `max_complete_blocks <= block_topk` (production `None`).
4. **`QSACache::update_indexer`** — concatenate keys on axis 1, append
   position ids on axis -1 with the 2-D <-> 3-D broadcasting rules of
   `_append_indexer_positions`. Runs **before** the None early-return, exactly
   as in production (so the cache is updated even when `None` is returned).

## Bit-exactness caveats

### 1. Fused-Metal RoPE path (handled)
On Apple Silicon `MRoPERotaryEmbedding.fused_apply` is True and `apply_rotary`
uses a hand-written Metal kernel with `metal::cos`/`metal::sin`. That path
differs from the eager `mx.cos`/`mx.sin` path by ~1 float32 ULP. The generator
therefore forces `rot.fused_apply = False`, i.e. the real eager branch of the
real module; the C++ port reproduces that math bit-identically (verified for
float32 and float16). Production's fused path only perturbs low bits of the
rotated values and does not change the boolean QSA mask for the well-separated
scores used here. If you need parity with the fused kernel specifically, note
that `mx::cos` (C++) and `metal::cos` are different implementations and a
C++-side fused path would have to call the same Metal kernel.

### 2. argpartition tie order (documented, not masked)
`mx.argpartition` has device-dependent ordering among tied scores. The final
`selected_tokens` is an OR over the selected blocks, so ordering within the
top-k set is irrelevant — only the **cut** between the k-th and (k+1)-th scores
matters. The generated vectors have a large gap at that cut:
case1 = 0.224, case2 = 0.336, case3 = 0.510 (several scores clamp to exactly
0.0 via `max(...,0)`, but those zeros are strictly below the top-k cut, so
their order cannot change the selected set). Scores with genuine ties spanning
the top-k boundary are not covered by these vectors; if a future model produces
such a case, bit-exact selection is not guaranteed across devices and should be
reported rather than papered over.

### 3. MLX C++ integer-division promotion (bug found & fixed during this port)
`mx::arange(n)` is int32 in C++, but `int32_array / int_scalar` promotes to
**float32** (e.g. `0, 0.25, 0.5, ...`), whereas Python's `//` floor-divides.
This initially produced a wrong `token_blocks` (only the first token of each
selected block was marked). Fixed by using `mx::floor_divide` with an int32
divisor for `token_blocks` and `complete_counts`, and by casting `arange`
results used as indices to int32. All selection cases then pass bit-exactly.

## Verification (standalone, no CMake edits)

```
cd /Users/vincent/Documents/Veystrix
.venv/bin/python cpp/engine/tools/gen_qsa_full_vectors.py --out cpp/engine/tests/vectors_qsa_full
MLXROOT=.venv/lib/python3.12/site-packages/mlx
xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
  cpp/engine/src/qsa.cpp cpp/engine/src/qsa_full.cpp \
  cpp/engine/tests/test_qsa_full.cpp \
  -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$(pwd)/$MLXROOT/lib" -o /tmp/test_qsa_full
/tmp/test_qsa_full cpp/engine/tests/vectors_qsa_full
```

Output:

```
  [case0_none] OK (None + cache bit-identical)
  [case1_select] OK (mask + cache bit-identical, shape (1,1,1,25))
  [case2_select_batch2] OK (mask + cache bit-identical, shape (2,1,2,22))
  [case3_select_float16] OK (mask + cache bit-identical, shape (1,1,1,25))
test_qsa_full OK (4 cases bit-identical to Python)
```

## Integration (for the owner)

1. **CMakeLists.txt** — add `src/qsa_full.cpp` to the engine library sources
   (keep `src/qsa.cpp` too) and add a `test_qsa_full` target built from
   `tests/test_qsa_full.cpp`, linking `${MLX_LIBRARIES}` with the same include
   dirs as the other tests. Optionally add a target that runs
   `gen_qsa_full_vectors.py` when vectors are missing.
2. **engine.hpp / engine.cpp** — no changes strictly required; the API lives in
   `namespace veystrix::engine`. To wire full QSA into the model, `#include
   "qsa_full.hpp"`, populate `QSAFullConfig` (`QSAConfig` base + `QSARotaryParams`
   rope), carry a `QSACache` per layer, and use the returned mask in
   `Qwen4ExpAttention.__call__` in place of the Python indexer call.
3. No changes to existing tests, generators, or source files are needed.
