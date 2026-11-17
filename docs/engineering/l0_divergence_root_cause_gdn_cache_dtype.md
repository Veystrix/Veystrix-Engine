# L0 divergence root cause: GDN conv-state cache initialized as f32 (Python uses bf16)

**Date:** 2026-11-18
**Scope:** end-to-end batched prefill, REAL checkpoint, layer 0. Resolves the
contradiction between the controlled replay (95.05% exact L0 hidden) and the
end-to-end binary (38.62% exact). **No end-to-end bit-exact claim.**

## The fix

In `cpp/engine/tools/real_checkpoint_generate.cpp` (line ~1251), the GDN cache
is initialized as:

```cpp
GDNCacheState zero(
    mx::zeros(mx::Shape{1, dcfg.gdn.conv_kernel - 1, conv_dim}, mx::float32),  // WRONG dtype
    mx::zeros(mx::Shape{1, dcfg.gdn.num_v_heads, dcfg.gdn.head_v_dim,
                        dcfg.gdn.head_k_dim}, mx::float32));                    // correct: f32
```

Python (`Qwen3_5GatedDeltaNet.__call__`, `language.py:1132-1136`) creates the conv
state with **`dtype=inputs.dtype`**, which is **bf16** on the real checkpoint:

```python
conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim),
                      dtype=inputs.dtype)
```

Change the first tensor's dtype to bf16 (i.e. `mixed_input.dtype()` / the
activation dtype), leaving `recurrent_state` f32 (Python also uses f32 there):

```cpp
GDNCacheState zero(
    mx::zeros(mx::Shape{1, dcfg.gdn.conv_kernel - 1, conv_dim}, mx::bfloat16),
    mx::zeros(mx::Shape{1, dcfg.gdn.num_v_heads, dcfg.gdn.head_v_dim,
                        dcfg.gdn.head_k_dim}, mx::float32));
```

A defensive dtype-generic fix in `gdn_prefill` (`prefill.cpp`) is also valid:
cast `cache.conv_state` to `x.dtype()` before `mx::concatenate`, so the conv
input can never be promoted by a wrongly-typed cache. Do both for robustness.
This is a **shared file** — not edited here; apply in the owner's tree.

## Why this explains everything

`gdn_prefill` does `mx::concatenate({cache.conv_state, mixed_qkv}, 1)` where
`mixed_qkv` is bf16. Measured dtype promotion:

| cache conv_state dtype | resulting `conv_input` dtype |
|---|---|
| bf16 (Python, controlled replay) | bfloat16 |
| f32 (end-to-end tool cache init) | float32 |

An f32 `conv_input` makes `conv1d`, `silu`, q/k/v, `_normalize_qk`, and the
fused delta kernel all run in f32 instead of bf16. Every intermediate after the
concat shifts by ~1 bf16 ULP, exactly the observed residual.

## Decisive measurement

`probe_gdn_cache_dtype.cpp` runs `gdn_prefill` at L0 on the **same**
`mixed_input` / recurrent state / real weights, changing only the conv-state
dtype (same zero values), and compares to Python's real GDN output:

| GDN forward | n | maxdiff | corr | exact |
|---|---:|---:|---:|---:|
| conv_state **bf16** | 40960 | **0** | 1.000000000 | **100.00%** |
| conv_state **f32** | 40960 | 0.0078125 | 0.999975508 | **26.87%** (29955 differ) |

Feeding that f32-cache branch into the L0 replay
(`probe_moe_intermediates.cpp`, `gdn_out_f32cache.bin` override) reproduces
the end-to-end statistic:

| Full L0 hidden [1,16,10240] vs Python | maxdiff | exact |
|---|---:|---:|
| controlled replay (bf16 cache) | 0.00195 | 95.05% |
| replay with f32-cache GDN branch | 0.00391 | **37.67%** |
| end-to-end binary (`/tmp/bdbg5/L.L00.f32` last token) | 0.00293 | **38.62%** |

The 37.67% vs 38.62% match (and the matching maxdiff scale) confirms the cache
dtype is the dominant source. Residual sub-percent differences come from the
replay reusing Python's router decisions instead of recomputing them from the
slightly shifted `m2d`.

## Ruled out (with evidence)

- **Embed**: C++ `embed_all` (gather packed rows then `mx::dequantize`) is
  bit-identical to Python's `layer_L00_input.f32` — `maxdiff=0, 100% exact`
  for all 10240 elements. Python's `nn.QuantizedEmbedding.__call__` also does
  gather-then-dequantize, so the hypothesis from the task brief is false.
- **Weights**: every L0 weight loaded by the C++ `DenseLoader` (hyper-connection
  norm gains, all six quantized hyper projections, router gate, all four shared
  expert triplets, all nine GDN projections + conv/A_log/dt_bias/norm/out_proj)
  matches Python's dumped weights bit-for-bit (uint32 packed weights compared
  as uint32; bf16 scales/norms as f32 values — 100% exact).
- **GDN primitives, MoE, router, shared expert**: all bit-exact with identical
  inputs, as established in prior reports.

## Scope / honesty

The single decisive variable tested is the conv-state dtype. The fix restores
the GDN branch to 100% exact at L0. I have **not** rebuilt the full end-to-end
binary with the shared-file fix applied (not permitted here), so I do not claim
full end-to-end bit-exactness. Please apply the two-line dtype change and re-run
the 48-layer digest.

## Files (new only, no shared edits, no commit)

- `cpp/engine/tools/probe_embed.cpp` — dump exact C++ embed / tiled embed.
- `cpp/engine/tools/probe_l0_weights.cpp` — dump C++ `DenseLoader` L0 weights.
- `cpp/engine/tools/probe_gdn_cache_dtype.cpp` — bf16-vs-f32 conv-cache test.
- `cpp/engine/tools/probe_moe_intermediates.cpp` — replay with branch override.
- this report.
