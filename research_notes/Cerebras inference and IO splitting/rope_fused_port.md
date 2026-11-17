# Fused-Metal MRoPE C++ port — status & integration note

This completes the **fused RoPE** parity gap left open by `qsa_full_port.md`.
That note's `qsa_apply_rope` matches only production's *eager* rotary branch;
this port reproduces the **fused-Metal kernel** bit-for-bit so the QSA indexer
can be made token-exact against the path production actually takes on Apple
Silicon.

## Production source ported

```
.venv/lib/python3.12/site-packages/mlx_vlm/models/rope_utils.py
  MRoPERotaryEmbedding.apply_rotary            lines 1244-1287
  _compiled_mrope_apply / _fast_mrope_apply    lines 1007-1025, 693-742
  _mrope_apply_kernel                          lines 568-652   <-- exact kernel
  compute_inv_freq                             lines 1043-1044
  mrope_position_selector / _interleaved / _chunked  lines 536-539, 512-527
.venv/lib/python3.12/site-packages/mlx_vlm/models/qwen3_5/language.py
  Qwen3_5RotaryEmbedding (style="interleaved") lines 27-42
.venv/lib/python3.12/site-packages/mlx_vlm/models/qwen4_exp/language.py
  Qwen4ExpQSAIndexer._apply_rope               lines 300-304
```

## Files created (all within ownership)

- `cpp/engine/src/rope_fused.hpp` — `rope_fused_compute_inv_freq`,
  `rope_fused_position_selector`, `rope_fused_cos_sin` (reference),
  `rope_fused_apply`, `rope_fused_apply_precomputed`.
- `cpp/engine/src/rope_fused.cpp` — implementation. Calls the **exact** Metal
  kernel source via `mx::fast::metal_kernel` (same technique as
  `gdn_kernel.cpp`). No existing file modified.
- `cpp/engine/tools/gen_rope_fused_vectors.py` — drives the REAL
  `Qwen3_5RotaryEmbedding` with `fused_apply = True`, saves q/k, position_ids,
  inv_freq, position_selector and the real fused q_out/k_out to `npy`, plus
  `meta.json`. Covers float32, float16, bfloat16 and both 2-D (text) and 3-D
  (MRoPE) position ids.
- `cpp/engine/tests/test_rope_fused.cpp` — loads vectors, byte-compares via
  raw-bit `view` (uint32 for f32, uint16 for f16/bf16), and additionally checks
  the inv_freq/selector helpers against the production values.
- `cpp/engine/tests/vectors_rope_fused/` — generated data (regenerable).

## What I determined by reading the source (the key result)

The fused kernel is **not** merely a cos/sin multiply-add in a different order.
`_mrope_apply_kernel` (rope_utils.py 607-645) computes the angle *inside the
kernel*, per frequency slot, with Metal stdlib intrinsics:

```
float pos   = static_cast<float>(position_ids[...]);       // axis-selected for 3-D ids
float angle = pos * static_cast<float>(inv_freq[freq_idx]);
float c = metal::cos(angle);
float s = metal::sin(angle);
x_out[idx]         = static_cast<T>(xv * c - xp * s);       // xv = x[d],   xp = x[d+half]
x_out[base+pair_d] = static_cast<T>(xp * c + xv * s);       // half-split pairing
```

`metal::cos`/`metal::sin` are different implementations from `mx.cos`/`mx.sin`,
so reproducing the fused result with MLX ops is impossible in general (~1 ULP
difference; more after cancellation). The only bit-exact route is to call the
same kernel source. `rope_fused.cpp` reconstructs the verbatim f-string from
`_mrope_apply_kernel` for `pairing == "half_split"` (which is what
`style="interleaved"` uses) and submits it through `mx::fast::metal_kernel`
with grid `(B*H*T*slots, 1, 1)`, threadgroup `(256, 1, 1)`, template `T =
x.dtype()`. `slots = half_dim + D - rotary_dim`.

Two production quirks deliberately reproduced:

1. The fused kernel **ignores `attention_scaling`** (no such kernel input).
   QSA uses `attention_scaling == 1.0`, so there is no impact on the indexer.
2. The rotation is computed in **float32 and cast to `T` at the end**, whereas
   the eager path casts cos/sin to `T` *before* multiplying. For f16/bf16 this
   is an additional source of divergence beyond the cos/sin intrinsics.

## Bit-exactness results

### C++ port vs production fused kernel — **0 ULP (bit-identical)**

All six generated cases pass a raw-bit comparison:

```
  [case_f32_pos2] OK (dtype=float32, pos_ndim=2, native, bit-identical)
  [case_f32_pos3] OK (dtype=float32, pos_ndim=3, native, bit-identical)
  [case_f16_pos2] OK (dtype=float16, pos_ndim=2, native, bit-identical)
  [case_f16_pos3] OK (dtype=float16, pos_ndim=3, native, bit-identical)
  [case_bf16_pos2] OK (dtype=bfloat16, pos_ndim=2, bf16-bits, bit-identical)
  [case_bf16_pos3] OK (dtype=bfloat16, pos_ndim=3, bf16-bits, bit-identical)
test_rope_fused OK (6 cases bit-identical to Python fused kernel)
```

The test also verifies `rope_fused_compute_inv_freq` and
`rope_fused_position_selector` are bit-identical to `rot.inv_freq` /
`rot.position_selector` from production for every case.

### Eager vs fused (the original gap) — exact deltas and mask impact

Measured on 20 random seeds with real QSA shapes (B=1, S=1, Hq=2, D=128,
rotary_dim=32, past_len=24, ratio=4, budget=16), toggling only
`rot.fused_apply`:

| dtype   | masks identical | rotary elements differing (max) | max monotonic ULP | max abs diff (f32 view) |
|---------|-----------------|---------------------------------|-------------------|-------------------------|
| float32 | 20/20           | 16 / 256                        | 15                | 1.19e-07                |
| float16 | 20/20           | 17 / 256                        | 54 (uint16-order) | 9.77e-04 (= 1 f16 ULP)  |

"Monotonic ULP" is the absolute difference of sign-mapped bit patterns
(`bits < 0 ? 0x8000… − (bits & 0x7fff…) : bits`), so values near zero can show
larger ULP counts; the absolute differences stay at the bottom-bit scale.

**Mask impact: none.** The boolean QSA mask (`selected_tokens[:, None]`, the
only quantity that affects token parity) is identical for fused and eager in
all 20/20 float32 and 20/20 float16 trials. The reason is structural: the
fused/eager delta lives only in the low bits of rotated q/k; the subsequent
`sum(max(scores.f32,0)) / sqrt(head_dim)` and the `argpartition` top-k cut are
well-separated in these workloads, so the selected block set never changes.
This does **not** prove immunity for pathological inputs where two block
scores straddle the top-k boundary within ~1e-7; such a case would need a
large-gap analysis of its own. The fused port above removes the concern anyway
by matching production's actual path exactly.

## Verification (standalone, no CMake edits)

```
cd /Users/vincent/Documents/Veystrix
.venv/bin/python cpp/engine/tools/gen_rope_fused_vectors.py \
    --out cpp/engine/tests/vectors_rope_fused
MLXROOT=.venv/lib/python3.12/site-packages/mlx
xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
  cpp/engine/src/rope_fused.cpp cpp/engine/tests/test_rope_fused.cpp \
  -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$(pwd)/$MLXROOT/lib" -o /tmp/test_rope_fused
/tmp/test_rope_fused cpp/engine/tests/vectors_rope_fused
```

(Output shown in the previous section.)

## Integration (for the owner)

1. **CMakeLists.txt** — add `src/rope_fused.cpp` to the engine library sources
   and `tests/test_rope_fused.cpp` to the test target. (Not done here: shared
   file, out of my ownership.)
2. **Wiring into the QSA indexer** — replace the indexer's rotary call with the
   fused port. The production trigger is
   `MRoPERotaryEmbedding.apply_rotary(q, k, position_ids, unsqueeze_dim=1)`
   with `fused_apply == True` and `q/k.ndim == 4`; in C++:
   ```cpp
   inv_freq = rope_fused_compute_inv_freq(rp.dim, rp.base);
   selector = rope_fused_position_selector(rp.mrope_section, rp.dim/2,
                                           rp.style == "interleaved");
   rotated  = rope_fused_apply(query, position_ids, inv_freq, selector,
                               rp.dim);
   ```
   Use this for both `query` and the pooled keys in `qsa_full_indexer`. Keep
   `qsa_apply_rope` as the eager/reference path; do not delete it.
3. **Generator/test vectors** are regenerable with the command above.

## Caveats

- **Metal required.** If `mx::fast::metal_kernel` cannot run (no Metal), the
  C++ function falls back to `rope_fused_cos_sin` + `rope_fused_apply_precomputed`,
  which uses `mx.cos/mx.sin` and is only ~1 ULP close, not bit-exact. The
  production path is Metal-only anyway.
- **bfloat16 storage.** numpy/PEP-3118 cannot represent bfloat16, so the
  generator bit-casts bfloat16 arrays to `uint16` via `mx.view` before `mx.save`;
  the test reinterprets with `mx::view(arr, mx::bfloat16)`. `meta.json` records
  `stored_as_uint16 = true` for those cases.
- **Kernel source coupling.** The kernel f-string is reconstructed from
  rope_utils.py 607-645 for `pairing == "half_split"` / interleaved style only.
  If a future model uses even/odd pairing (`split_select`, `ernie_3d`,
  `sectioned_even_odd`), a new pair-source branch is needed. inv_freq and the
  selector are passed straight through, so other rotary_dim values work
  without source changes.
- No shared files, other branches, or commits were touched.
