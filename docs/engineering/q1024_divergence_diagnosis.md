# Why the C++ engine is bit-exact at S=16 but not at S>=128/1024

Status: **measured** (small guarded runs + MLX-only probes), 2026-11-18.
All real-checkpoint runs were guarded with the owner's flags; every log reports
`"stopped": null` and thermal level 0 throughout.

## Result in one line

The first divergent op is **not** PLE, GDN, conv1d, SDPA, or MoE chunking. It is
the **sorted-vs-unsorted `gather_qmm` path inside the routed MoE**:
Python `SwitchGLU.__call__` (`mlx_vlm/models/switch_layers.py:172-193`) applies
`_gather_sort` and calls `gather_qmm(..., sorted_indices=True)` whenever
`indices.size >= 64` (always true for prefill), while C++ `moe_block`
(`cpp/engine/src/engine.cpp:97-110`) calls `q4_projection` with the default
`sorted_indices=False` and no gather-sort. The two MLX kernels are bit-identical
for small gathered-row counts but diverge by ~1 bf16 ULP once
`M = S * top_k` is large (S=128 -> M=1280). This is the source of the 1024-token
logit drift.

## Small-run results (S=128, same seed 20261117)

- Python ref `/tmp/qwen_128_ref.json`: step0 token=271, digest
  `c6f51dae37658d7bc3b2f8a05dd239a282b8cbc8a4d47d4d89dc7feaf11fe9bc`.
- C++ `/tmp/qd_finalgate` with generated config: step0 token=271 but digest
  `5dbb28ccbd278529ece9716189f9aa4c46d921174fe9642adde677495a0dc82c`.
  => divergence already present at S=128 (S=16 gate (b) is 8/8 exact).
- C++ layer dumps `/tmp/s128_cpp_layer.*.f32`, Python layer dumps
  `/tmp/l128_py/layer_L##.f32`, Python GDN intermediates `/tmp/i128_py`,
  Python MoE intermediates `/tmp/m128_py`.

### Layer-0 localisation (all measured, bit comparisons)

| quantity | C++ vs Python |
|---|---|
| `gdnin.L00` (input to GDN, full [1,128,2560]) | 0 / 327680 differ |
| GDN `branch.L00` last token | 0 / 2560 differ |
| `m2d` (MoE input) | 0 / 327680 differ |
| `gate_logits` | 0 / 65536 differ |
| router indices (1280) / scores | 0 mismatches |
| `mlpinj` (MLP hyper injection) | 0 / 512 differ |
| **`moe2d`** | **216971 / 327680 differ, maxabs 0.00195** |

Layer 0 is a GDN layer with **no PLE** (PLE is layer 1), so PLE is ruled out as
the first divergent op.

### Python's own two MoE paths differ at S=128

`dump_python_moe_intermediates.py` records both:
- `ref_comb.bin` = unsorted reference (`gather_qmm(sorted_indices=False)`, no
  `_gather_sort`),
- `combined.bin` = production path (`_gather_sort` + `sorted_indices=True`).

At S=128 they differ by 259851 / 327680 elements, maxabs 0.001953125 — the
same magnitude/signature as the end-to-end C++/Python gap. C++ end-to-end
matches `ref_comb + sg_s` far more closely than `combined + sg_s`
(134783 vs 34738 exact elements), confirming C++ is on the unsorted kernel.
At S=16 both paths are bit-identical, which is why gate (b) passes.

## Probes (MLX-only, no checkpoint, no guard needed)

- `probe_gather_qmm_modes.cpp`: batched `gather_qmm` vs T separate M=1 calls is
  **0 differing elements** for T = 16, 64, 128, 512, 1024 (single- and
  64-expert stacks, `sorted_indices=false`). Per-token MoE chunking is fine.
- `/tmp/probe_real_shapes.cpp` (new): same test at the real expert shapes
  gate/up `[E,640,320]` and down `[E,2560,80]` — 0 differing elements at
  T = 16/128/1024. Shape is not the issue.
- `/tmp/probe_moe_T128` (replay of `probe_moe_intermediates.cpp` with
  `T=128`): with **identical m2d / routes / scores / Python-dumped weights**,
  every MoE intermediate (gate/up/down gather_qmm, swiglu, weighted combine,
  shared expert, final `moe2d`) is 100% bit-exact. MoE compute is correct;
  only the kernel selected by `sorted_indices` differs.

## Why it is S-dependent

`SwitchGLU.__call__` hard-codes `do_sort = indices.size >= 64`, so Python always
takes the sorted path for prefill. MLX 0.32.2's sorted `gather_qmm` kernel and
the unsorted kernel produce identical results for small gathered-row counts
(S=16 -> 160 rows), but from roughly S>=128 (M>=1280 rows) they choose different
tiling/reduction and differ by 1 bf16 ULP. C++ never takes the sorted path, so
it silently follows the unsorted kernel and drifts once the prompt is long.

## Proposed fix — NOT a one-line change; top 2 approaches

**Approach A (bit-exact, moderate ~150-250 LOC, +~1-2 GiB): batched sorted MoE
for prefill.** In the prefill path (new tool under `cpp/engine/tools/` or a new
function in `engine.cpp`), mirror `SwitchGLU.__call__` exactly:
1. flatten routes, `order = argsort(flat)`, `inv = argsort(order)`;
2. permute activation rows `x[order // K]` and use `idx = flat[order]`;
3. make one `gather_qmm(..., sorted_indices=true)` call over all `S*K` rows;
4. scatter back with `inv`, then weighted-combine and add the shared expert.

This requires all distinct experts of the prompt chunk to be resident at once
(up to 512 experts ~= 1.3 GiB of Q4 weights + scales/biases, which fits in the
~3 GiB current budget). The existing 16-slot `StreamingMoE` cannot supply this;
a resident-expert variant (or a prefill-sized slot pool, `num_slots>=512`) is
needed. Tradeoff: extra memory and a second MoE code path; exact at any S.

**Approach B (cheap but changes the reference): force Python's unsorted path.**
Change `SwitchGLU.__call__`'s `do_sort` rule (`models/flashnext/...` vendored
copy or an env flag) so the Python reference also uses
`sorted_indices=False`. Gate (b) at S=16 is unaffected because the two kernels
are identical there. Tradeoff: it is a reference-side change, not an engine
fix, and requires regenerating the 1024 reference — acceptable only if the goal
is a like-for-like parity gate rather than reproducing stock Python behaviour.

Rejected: increasing `num_slots` alone (does not change the kernel choice),
per-token chunking (probe shows it is already bit-exact), PLE/GDN/SDPA changes
(proven bit-exact by layer dumps).

## Files read (key ones)

- `cpp/engine/src/prefill_batched.cpp`, `prefill.cpp`, `ple.cpp`,
  `gdn_kernel.h`, `engine.cpp` (`moe_block`, `q4_projection`),
  `moe_stream.cpp`, `slot_plan.cpp`, `cross_shard_moe.cpp`
- `cpp/engine/tools/real_checkpoint_generate.cpp`,
  `probe_gather_qmm_modes.cpp`, `probe_moe_intermediates.cpp`,
  `dump_python_moe_intermediates.py`, `dump_python_intermediates.py`,
  `dump_python_layers.py`, `capture_real_reference.py`, `gen_real_config.py`
- `mlx_vlm/models/qwen4_exp/language.py` (`.venv`),
  `mlx_vlm/models/qwen3_5/language.py`, `mlx_vlm/models/qwen3_5/gated_delta.py`,
  `mlx_vlm/models/qwen3_5_moe/language.py`,
  `mlx_vlm/models/switch_layers.py`, `mlx_vlm/models/base.py`
- docs/engineering/moe_drift_localization.md (prior S=16 analysis)
