# L0 MoE drift localization: MoE compute is bit-exact; divergence is upstream

**Date:** 2026-11-18
**Scope:** REAL checkpoint `Qwen3.8-Flash-Next-MLX-4bit-MTP`, layer 0 (GDN /
linear_attention, top_k=10, 512 experts), 16-token prefill, C++ vs Python/MLX.
All comparisons use arrays dumped from one side and loaded into the other, so
routing and weight loading are held constant. **No end-to-end bit-exact claim.**

## Result in one line

At L0, given identical `m2d`, router indices/scores, and stacked expert weights,
**every MoE intermediate is byte-identical** — gate/up `gather_qmm`, SwiGLU,
down `gather_qmm`, weighted combine, and the entire dense shared expert. Router
logits/indices/scores are also bit-exact. The `sorted_indices` flag and the
scale/bias bf16 reinterpret are not the source.

## Hypothesis tests (both negative)

1. **`sorted_indices`**: Python's prefill takes `indices.size = 16*10 = 160 >=
   64`, so `_issue` calls `_gather_sort` and passes `sorted_indices=True` to
   `gather_qmm`; C++ `moe_block` passes `false`. Measured: C++ with `false` and
   with `true` produce **identical** outputs (`maxdiff=0` vs Python in both).
   Not a kernel-numeric difference.
2. **scale/bias bf16 bytes**: C++ reinterprets uint16 bytes from `mx::load_
   safetensors`, Python reinterprets `store.rows_np` uint16 via `.view(
   bfloat16)`. A direct raw `pread` of expert 49's `gate_proj.scales` and
   `.biases` from the shard matches Python's dumped rows bit-for-bit
   (`maxdiff=0, 100% exact`, 51200 elements each). Not a byte-loading issue.

## Measurements (C++ vs Python, identical inputs, layer 0, S=16)

### Router — all bit-exact
| Intermediate | maxdiff | exact |
|---|---:|---:|
| gate_logits (from identical m2d + gate_w) | 0 | 100% |
| router indices | 0 mismatches / 160 | 100% |
| router scores | 0 | 100% |

### Routed experts — all bit-exact (Python unsorted reference replay)
| Intermediate | n | maxdiff | exact |
|---|---:|---:|---:|
| gate `gather_qmm` | 102400 | 0 | 100% |
| up `gather_qmm` | 102400 | 0 | 100% |
| SwiGLU | 102400 | 0 | 100% |
| down `gather_qmm` | 409600 | 0 | 100% |
| weighted product (`d * scores`) | 409600 | 0 | 100% |
| weighted combine (`sum(-2)`) | 40960 | 0 | 100% |
| `moe_block` routed combine | 40960 | 0 | 100% |

### Shared expert — all bit-exact
`sgl`, `sg`, `se_g`, `se_u`, `se_swiglu`, `se_s`, `sg*s`: every one `maxdiff=0,
100% exact`. Final `moe2d = routed + sg*s` is also `maxdiff=0, 100% exact`.

**Conclusion: no MoE intermediate diverges.**

## Where the residual L0 gap actually is

Replaying the full layer from the identical tiled embed `layer_input`
(`[1,16,10240]` bf16) with C++ GDN (now fixed) and C++ MoE:

| Intermediate | maxdiff | exact | note |
|---|---:|---:|---|
| attn `hc_norm` output | 0 | 100% | |
| attn `silu(low)` | 6.1e-5 | 99.77% | 12/5120 elements, 1 bf16 ULP |
| attn `mix` / `prod` | 3.9e-3 / 7.8e-3 | 100% | 3/163840 elements |
| attn `mixed_input` | 7.8e-3 | ~100% | **1/40960 element** (1 ULP) |
| hidden after attn inject | 0 | 100% | (uses bit-exact GDN output) |
| mlp `mixed_input` (= MoE `m2d`) | 3.9e-3 | 99.98% | **8/40960 elements** (1 ULP) |
| MoE branch from C++ `m2d` | 9.8e-4 | 85.46% | amplified from those 8 elements |
| **FULL L0 hidden** | **1.95e-3** | **95.05%** | 8103/163840 elements |

So the remaining L0 gap is **rare 1-ULP bf16 differences in
`Qwen4ExpGatedResidual`** (`gated_residual` in `engine.cpp`), specifically in
`silu(down(normed)/hc)` and the subsequent `mean(mix*streams, -2)`:
`normed` itself is bit-exact, and the hyper-connection gains / down / up / inject
weighted outputs are otherwise exact. These are isolated rounding ties, not a
systematic dtype promotion. They compound because a single changed element in
`m2d` changes a token's gate ranking / expert outputs downstream.

This controlled end-to-end replay gives **95% exact at L0**, which is far
better than the previously reported 38.6% exact. That mismatch suggests the
end-to-end binary used for the earlier number was built before the two GDN
dtype fixes (or took a different path). Re-run the full 48-layer comparison
with a freshly rebuilt binary before drawing conclusions about layers 1..47.

## Concrete fix status

- MoE: **no fix needed** — `q4_projection`, `swiglu`, `weighted_combine`,
  `gated_residual` (for the dominant path) and the shared expert are already
  bit-exact when fed the same inputs and weights.
- `sorted_indices`: **no change needed** — it does not change numerics; do not
  add sorting to the C++ MoE to "match" Python.
- The only remaining issue is the handful of 1-ULP hyper-connection ties. If
  they matter, the place to look is `gated_residual` in `engine.cpp`
  (`mx::array(hc_count)` division and the bf16 `mean`). No dtype-generic fix
  was found necessary here; the two GDN fixes from the previous investigation
  already cover the systematic divergence.

## Files (new only, no shared edits, no commit)

- `cpp/engine/tools/dump_python_moe_intermediates.py`
- `cpp/engine/tools/probe_moe_intermediates.cpp`
- this report.

Build/run recipe is in the header comments of the C++ probe. The Python dump
captures 87 raw files (including the per-projection gather outputs, the
unsorted reference replay, the hyper-connection stages, and the stacked expert
weights); the C++ probe reads them and prints all comparisons above.
