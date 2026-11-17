# Full GDN layer C++ port — status & integration note

## What was ported

A **complete single GDN (gated-delta-net) linear-attention layer forward** — the
`Qwen3_5GatedDeltaNet.__call__` decode step (S == 1) as specialized by
`Qwen4ExpGatedDeltaNet`. It is composed entirely from the already-verified
bit-exact primitives; **no existing file was modified and no primitive was
reimplemented**.

Python reference (installed `mlx_vlm`, MLX C++ 0.32.2):

```
.venv/lib/python3.12/site-packages/mlx_vlm/models/qwen3_5/language.py
  Qwen3_5GatedDeltaNet.__call__            lines 1109-1196
  Qwen3_5GatedDeltaNet.__init__            lines 1047-1089
.venv/lib/python3.12/site-packages/mlx_vlm/models/qwen4_exp/language.py
  Qwen4ExpGatedDeltaNet._normalize_qk       lines 266-272   (override)
  Qwen4ExpRMSNormGated.__call__             lines 239-254   (self.norm)
.venv/lib/python3.12/site-packages/mlx_vlm/models/qwen3_5/gated_delta.py
  _compute_g_beta / compute_g               lines 261-268
  gated_delta_update -> gated_delta_kernel  lines 382-408 / 170-210
```

The reused primitives (all called, none changed):

- `engine.hpp/.cpp`: `compute_g_beta`, `gdn_normalize_qk`,
  `rms_norm_gated`, and `gated_delta_step_kernel`.
- `gdn_conv.h/.cpp`: `gdn_causal_conv_decode_silu`.

## Exact op order (mirrored one-for-one in `gdn_layer.cpp`)

```
 1. mixed_qkv = in_proj_qkv(x) = matmul(x, Wqkv.T)   -> [B,1,conv_dim]
    z          = in_proj_z(x)                        -> [B,1,value_dim]
    b          = in_proj_b(x)                        -> [B,1,Hv]
    a          = in_proj_a(x)                        -> [B,1,Hv]
    z          = reshape(z, [B,1,Hv,Dv])
 2. conv_input  = concat([conv_state, mixed_qkv], axis=1)         [B,K,conv_dim]
    cache[0]    = contiguous(conv_input[:, -(K-1):, :])
    conv_out    = silu( depthwise_conv(conv_input, conv1d_weight) ) [B,1,C]
        via gdn_causal_conv_decode_silu (preps raw [C,K,1] weight once)
 3. q,k,v = split(conv_out, [key_dim, 2*key_dim], -1)
            -> reshape q,k [B,1,Hk,Dk], v [B,1,Hv,Dv]
 4. q,k = _normalize_qk(q,k):  scale = Dk^-0.5
            q = q*rsqrt(sum(q^2,-1,keepdims)+1e-6)*scale ; k = k*rsqrt(...)
        via gdn_normalize_qk
 5. g,beta = compute_g_beta(A_log, a, b, dt_bias)     via compute_g_beta
    y,state = gated_delta_step_kernel(q,k,v,g,beta,state)  (fused Metal kernel)
 6. out = Qwen4ExpRMSNormGated(y, z)  (sigmoid gate)  via rms_norm_gated
 7. output = out_proj(reshape(out,[B,1,value_dim])) = matmul(flat, Wout.T)
```

Returned cache: the new `conv_state` (`cache[0]`) and the new recurrent `state`
(`cache[1]`).

Note on step 4/5: the kernel maps the `Hv` value heads onto the `Hk` key heads
internally via `hk_idx = hv_idx / (Hv / Hk)`. Python's `gated_delta_update` does
**not** explicitly `repeat` q/k on the kernel path (repeat only exists in the
ops/chunked fallbacks), so q/k are passed at `Hk` heads — matching Python
exactly. The task's "repeat to Hv" instruction describes the ops fallback and
would be redundant here.

## Files created (all within my ownership)

- `cpp/engine/src/gdn_layer.hpp` — `GDNConfig`, `GDNWeights`, `GDNCacheState`,
  `GDNLayerOutput`, `GDNActivation`, and `gdn_layer_forward`.
- `cpp/engine/src/gdn_layer.cpp` — implementation (composition only).
- `cpp/engine/tools/gen_gdn_layer_vectors.py` — generates vectors from the REAL
  `Qwen4ExpGatedDeltaNet` (default `cpp/engine/tests/vectors_gdn_layer`). It also
  writes intermediate tensors (`i_*.npy`) for isolating any future divergence.
- `cpp/engine/tests/test_gdn_layer.cpp` — loads vectors, compares output +
  both new states **byte-for-byte** (true bit view via `mx::view`), prints
  `test_gdn_layer OK (bit-exact)`.
- `cpp/engine/tests/vectors_gdn_layer/` — generated npy vectors + `meta.json`
  (data artifacts, fully regenerable).
- this note.

## Verification (standalone, no CMake edits)

```
cd /Users/vincent/Documents/Veystrix
.venv/bin/python cpp/engine/tools/gen_gdn_layer_vectors.py \
    --out cpp/engine/tests/vectors_gdn_layer
MLXROOT=.venv/lib/python3.12/site-packages/mlx
xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
  cpp/engine/src/engine.cpp cpp/engine/src/gdn_kernel.cpp \
  cpp/engine/src/gdn_conv.cpp cpp/engine/src/gdn_layer.cpp \
  cpp/engine/tests/test_gdn_layer.cpp \
  -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$(pwd)/$MLXROOT/lib" -o /tmp/test_gdn_layer
/tmp/test_gdn_layer cpp/engine/tests/vectors_gdn_layer
```

Output:

```
test_gdn_layer OK (bit-exact)
```

Also verified bit-exact on three additional geometries/seeds (generator flags):

```
--seed 7   --hidden 48  --hk 1 --hv 2 --dk 32 --dv 32 --kernel 3   -> OK
--seed 99  --hidden 128 --hk 2 --hv 4 --dk 32 --dv 64 --kernel 4   -> OK
--seed 123 --hidden 96  --hk 2 --hv 6 --dk 64 --dv 32 --kernel 5   -> OK
```

## Integration (for the owner)

1. **CMakeLists.txt** — add `src/gdn_layer.cpp` to the engine library sources and
   add a `test_gdn_layer` target built from `tests/test_gdn_layer.cpp` (link the
   same MLX libs/include dirs as `test_conv`/`test_gdn_delta`). Optionally add a
   custom target that runs `gen_gdn_layer_vectors.py` when vectors are missing.
2. **engine.hpp / engine.cpp** — no changes required; this is a self-contained
   `veystrix::engine` API. To expose it through the facade, `#include
   "gdn_layer.hpp"` and populate `GDNConfig`/`GDNWeights`/`GDNCacheState` from the
   Rust/FFI config and checkpoint loader. The caller owns the cache and feeds the
   returned `conv_state`/`recurrent_state` back on the next decode step.
3. No changes to existing tests, generators, or primitives are needed.

The API is generic: `GDNConfig` carries every geometry field (hidden/key/value
dims, head counts, head dims, conv kernel, eps, output activation), and
`GDNActivation` selects sigmoid vs silu for the Qwen4ExpRMSNormGated gate.

## Caveats / important findings

- **Dtype: f32 input required for bit-exactness.** The verified C++
  `gdn_normalize_qk` is f32-only, while Python's `_normalize_qk` works in the
  input dtype: with a **bf16** input it produces bf16 q/k and the fused kernel
  sees `InT=bf16`. The generator therefore uses **f32 `x` and f32 `conv_state`**;
  then the entire post-projection path (conv, q/k norm, delta kernel, gated
  RMSNorm, out_proj) is f32 and every reused primitive is bit-exact. Linear /
  conv / out_proj weights remain bf16 (production dtype); MLX promotes the
  matmul to f32 exactly as `nn.Linear` does. Supporting a bf16 input path
  bit-exactly would need a bf16 variant of `gdn_normalize_qk` (out of scope —
  `engine.hpp` is not owned here).
- **`model.eval()` is mandatory in the generator.** `nn.Module.training`
  defaults to `True`, so `Qwen3_5GatedDeltaNet.__call__` passes
  `use_kernel = not self.training = False` and takes the *ops* fallback. The
  production decode path uses the fused Metal kernel, so the generator calls
  `model.eval()` before the step. (This was the source of an initial 1-ULP
  mismatch: the Python side used `gated_delta_ops` while C++ used
  `gated_delta_step_kernel`.)
- **Bit comparison uses `mx::view`, not `astype`.** `mx::astype(float, uint32)`
  is a numeric *conversion* (e.g. -1.5 -> 0), not a bit reinterpretation; the
  test uses `mx::view(mx::reshape(a, {-1}), mx::uint32)` for a true byte-for-byte
  comparison. (Existing older tests use `astype`, which is a weaker check.)
- **S > 1 is supported as a token-by-token loop** through the same decode path.
  This is identical to Python for S == 1 (the verified case); for S > 1 Python
  uses a different full-conv + chunked-scan prefill path, so the loop is a
  reasonable extension but is **not** claimed bit-exact to that prefill path.
- **eps** is passed through `GDNConfig.eps`; the test reads the other geometry
  from the saved tensor shapes and uses the generator's `1e-6`.
- **No shared files touched, no commits made.** Stayed on branch `cpp-engine`.
