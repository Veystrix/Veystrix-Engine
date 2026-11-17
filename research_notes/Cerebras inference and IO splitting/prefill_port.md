# GDN prefill (S > 1) C++ port — research notes

Date: 2026-11-17. Branch: `cpp-engine`. MLX C++ 0.32.2, M4 GPU (Metal).

## Goal

Port production's GDN linear-attention **prefill** (S > 1 tokens) bit-exactly so
the C++ engine can ingest real prompts (the app always prefills before decode).
Existing `gdn_layer.cpp` only covers S == 1.

## Findings: what production actually does for S > 1 on this M4

### The delta-rule kernel, not the chunked scan

`mlx_vlm/models/qwen3_5/gated_delta.py::gated_delta_update` (L382-408):

```python
g, beta = _compute_g_beta(A_log, a, b, dt_bias)
if not use_kernel or mx.default_device() != mx.gpu or not mx.metal.is_available():
    if q.shape[1] > 1 and g.ndim == 3:
        return gated_delta_chunked(...)   # only for non-kernel backends (e.g. CUDA)
    return gated_delta_ops(...)
return gated_delta_kernel(...)            # <-- M4 GPU, eval mode
```

`Qwen3_5GatedDeltaNet.__call__` passes `use_kernel = not self.training`; the
real layer is in eval mode, so `use_kernel=True`. On this M4
`mx.default_device()==gpu` and `mx.metal.is_available()==True`, so the `not
use_kernel or ...` branch is FALSE and it calls `gated_delta_kernel` directly.

Instrumented the real `Qwen4ExpGatedDeltaNet` (spy on `gated_delta_kernel` /
`gated_delta_chunked` / `gated_delta_ops`): for S in {2,5,16} the call list is
`['kernel']` every time — never `chunked`, never `ops`.

`gated_delta_kernel` compiles `_make_gated_delta_kernel()` whose source contains
`for (int t = 0; t < T; ++t) { ... }` — i.e. **production prefill is a
per-token loop fused inside the Metal kernel**. It is mathematically the same
recurrence as repeated S==1 steps, and the existing `kGdnDeltaKernelSource`
(gdn_kernel.h) is that exact source. So the S==1 kernel primitive already does
the right thing at S>1; no chunked-scan port is needed. `gated_delta_chunked`
is dead code for this deployment.

### The other S>1 differences inside `__call__`

The GDN layer does NOT just loop the S==1 decode step. For S>1:

1. full projections over `[B,S,H]` (one matmul, not S matmuls);
2. optional pre-conv mask: `mixed_qkv = where(mask[...,None], mixed_qkv, 0)`;
3. `conv_input = concat([conv_state, mixed_qkv])`; `cache[0] = conv_input[:,-(K-1):]`;
4. **full depthwise `nn.Conv1d` over the sequence** + `nn.silu` — NOT the cached
   per-token decode conv (`_qwen3_5_decode_depthwise_conv`). Verified the two
   differ numerically (~3e-8 max abs), so the full `mx.conv1d` primitive must
   be used for bit-exactness;
5. q/k/v split, Qwen4Exp `_normalize_qk` (L2 + `1/sqrt(Dk)` query scale);
6. `gated_delta_update(..., mask)` -> fused kernel (unmasked source when
   `mask is None`; `_gated_delta_kernel_masked` when a mask is passed);
7. Qwen4Exp gated RMSNorm + full out_proj over `[B,S,value_dim]`.

The SSM mask is `None` in the normal single-prompt prefill
(`_create_qwen3_5_ssm_mask` returns None unless `cache.left_padding` is set),
but the layer also accepts an explicit `[B,S]` bool mask which selects the
masked kernel (masked positions: y=0, state unchanged). Both are covered.

### Masking in the SM mask/kernel

Python masked kernel guard: `if (mask[b_idx * T + t])`, inputs
`[q,k,v,g,beta,state_in,T,mask]`, kernel name `gated_delta_step_mask`. The C++
masked source in `prefill.cpp` was verified byte-for-byte equal to
`_make_gated_delta_kernel(has_mask=True, vectorized=False)` output.

### MoE path

`Qwen3_5MoeSparseMoeBlock.__call__(x)` takes **no mask** and processes `[B,S,H]`
uniformly (softmax/argpartition router, `SwitchGLU` gather, shared expert).
There is no S>1-specific mask handling in MoE, so no `moe_prefill` variant is
needed: flatten `B*S` into the token axis and use the existing
`moe_block` / `moe_block_from_logits` / `q4_projection` primitives (they already
accept arbitrary token counts). Only the GDN (and upstream PLE) layers consume
an SSM mask.

## Deliverables (owned files only)

- `cpp/engine/src/prefill.hpp` — API + extensive rationale comments.
- `cpp/engine/src/prefill.cpp` — `gdn_prefill`, full-sequence path + masked kernel.
- `cpp/engine/tools/gen_prefill_vectors.py` — real-model vector generator (meta.json).
- `cpp/engine/tests/test_prefill.cpp` — bit-exact test (uint32 view compare).
- `cpp/engine/tests/vectors_prefill/` — generated vectors (s2, s5, s16, s5_masked).
- this note.

No shared files touched (`gdn_layer.*`, `engine.*`, `decoder*`, `qsa*`,
`CMakeLists.txt`, existing tests all unchanged). No commit, no branch switch.

## Verification (all bit-exact)

```
.venv/bin/python cpp/engine/tools/gen_prefill_vectors.py --out cpp/engine/tests/vectors_prefill
MLXROOT=.venv/lib/python3.12/site-packages/mlx && \
xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
  cpp/engine/src/engine.cpp cpp/engine/src/gdn_kernel.cpp cpp/engine/src/gdn_conv.cpp \
  cpp/engine/src/gdn_layer.cpp cpp/engine/src/prefill.cpp cpp/engine/tests/test_prefill.cpp \
  -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$(pwd)/$MLXROOT/lib" -o /tmp/test_prefill
/tmp/test_prefill cpp/engine/tests/vectors_prefill
```

Output:
```
  case s5 (S=5, mask=no): OK (bit-exact)
  case s2 (S=2, mask=no): OK (bit-exact)
  case s5_masked (S=5, mask=yes): OK (bit-exact)
  case s16 (S=16, mask=no): OK (bit-exact)
test_prefill OK (4 cases, bit-exact)
```

## Integration instructions

1. Add `cpp/engine/src/prefill.cpp` to the engine build (the stand-alone
   compile command above is the exact source list; CMakeLists is owned by the
   parent agent, so wire it there).
2. Include `prefill.hpp` and call `gdn_prefill(x[B,S,H], weights, cache, cfg)`
   for a prompt; pass `&mask` (a bool `[B,S]` array) only when the layer is
   given an SSM/padding mask, otherwise `nullptr`.
3. Reuse `GDNWeights`/`GDNConfig`/`GDNCacheState`/`GDNLayerOutput` from
   `gdn_layer.hpp` (no new types). `GDNLayerOutput.output` is `[B,S,H]`;
   `.conv_state` / `.recurrent_state` are the advanced cache to feed the next
   chunk or decode step.
4. Keep x and the cache f32 (as the S==1 layer vectors do); weights stay
   bf16/f32 as in production, MLX promotes matmul/conv to f32.
5. Constraints: Dk multiple of 32 (Metal kernel tile), Hv multiple of Hk.

## What remains / caveats

- Dtype coverage is f32 activations (same convention as the verified S==1
  layer). A bf16-activation probe would hit the f32-only `gdn_normalize_qk`
  primitive; not required for the C++ engine's current f32 accumulator path.
- No chunked-scan implementation: unnecessary because production uses the
  fused kernel on this GPU. If the engine is ever deployed on a non-Metal
  backend, `gated_delta_chunked` would need its own port (documented in
  gated_delta.py, only rel-err < 1e-3, not bit-exact).
- CPU fallback in `run_delta_kernel` calls the ops reference and cannot honour
  kernel mask semantics; only relevant without Metal.
- Long prompts are split into chunks by `models/flashnext/prefill.py`; chunking
  changes kernel shapes and is numerically *equivalent*, not bit-identical, to
  a single call (documented in that file). `gdn_prefill` handles one chunk.
