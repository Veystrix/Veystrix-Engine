# Model-agnostic adapter/config layer for the C++ engine

**Date:** 2026-11-17
**Branch:** `cpp-engine`
**Status:** implemented + standalone verify passing (958 checks)

## What this is

The C++ decode primitives (`cpp/engine/src/engine.hpp`, `qsa.hpp`, `gdn_conv.h`)
are already model-agnostic — every shape, epsilon and head count is an argument.
What was missing is one place that turns a *checkpoint config* into the exact set
of primitives (and exact configs) a decode step must call. This layer is that
place, so the decode loop is written once and dispatches through one function.

Files created (all new; no existing file modified):

| File | Role |
|---|---|
| `cpp/engine/src/model_adapter.hpp` | Enums, `ModelSpec`, `GDNConfig`, `ExpertConfig`, `HyperConnectionConfig`, `LayerDispatch`, builders, `dispatch_layer()`, `describe()`, flat-format docs |
| `cpp/engine/src/model_adapter.cpp` | Pure-C++ implementation (no MLX calls) |
| `cpp/engine/tools/gen_model_adapter_vectors.py` | Reads the real `config.json` files with Python `json`, emits flat key=value vectors |
| `cpp/engine/tests/test_model_adapter.cpp` | `verify_against_python`: rebuilds each `ModelSpec` from the flat file and asserts every field bit-exactly |
| this note | Design / integration / future-model guidance |

## Design

### Enums
- `ModelFamily { QWEN4_EXP, NEMOTRON_H, GENERIC }`.
- `LayerKind { LINEAR_ATTN_GDN, FULL_ATTN_QSA, MAMBA_SSM, MOE }`.
- `Activation { SWIGLU, RELU2 }`.

Nemotron's plain dense-attention layers are represented as `FULL_ATTN_QSA`
(the full-attention family) with `use_qsa_indexer=false` and `qsa.n_heads=0`, so
`dispatch_layer()` returns `use_qsa=false` for them. This keeps the enum to the
four values the engine actually dispatches on while distinguishing sparse QSA
from dense attention.

### ModelSpec fields (everything the engine needs to dispatch)
- `family`, `hidden_size`, `n_layers`.
- `layer_kinds` (per-layer attention/recurrent kind) and `has_moe` (per-layer
  routed-expert flag). Qwen is all-MoE; Nemotron is MoE only on `moe` blocks.
- `expert`: `n_experts`, `top_k`, `group_size`, `moe_intermediate`, `activation`,
  and `tensor_prefix_format` — the exact streamed-expert naming convention.
- `norm_eps`, `hc` (`hc_count`, `hc_lowrank`, `output_gate_type`).
- `qsa` (reused from the unmodified sibling `qsa.hpp`) plus `use_qsa_indexer`.
- `gdn` (`GDNConfig`, defined locally because no sibling provides one; see
  integration note below).

### Exact values captured from the real checkpoints

Qwen3.8-Flash-Next (`QWEN4_EXP`):
- hidden 2560, 48 layers; `layer_types` = 3×`linear_attention` then
  1×`full_attention` (interval 4); every layer has MoE.
- experts 512 / top-10 / group 32 / intermediate 640 / SwiGLU.
- prefix `language_model.model.layers.{L}.mlp.switch_mlp.{P}.{T}`.
- norm_eps 1e-6; hyper-connections hc_count=4, hc_lowrank=320,
  output_gate_type=sigmoid.
- QSA: n_heads 4, kv_heads 1, head_dim 128, compress_ratio 4,
  token_budget 2048 → block_topk 512 (derived), eps 1e-6.
- GDN: conv_kernel 4, 16 key heads, 48 value heads, key/value head_dim 128.

Nemotron 3.5 Lightning (`NEMOTRON_H`):
- hidden 2688, 52 layers from `layers_block_type` (mamba / attention / moe).
- experts 128 / top-6 / group 64 / intermediate 1856 / relu2.
- prefix `backbone.layers.{L}.mixer.switch_mlp.{P}.{T}`.
- norm_eps (layer_norm_epsilon) 1e-5; no hyper-connections; no QSA indexer.

### Dispatch
`LayerDispatch dispatch_layer(const ModelSpec&, int layer)` returns, for one
layer, which primitives apply and the exact configs to pass:

| kind | primitives | config used |
|---|---|---|
| `LINEAR_ATTN_GDN` | `gdn_normalize_qk`, `compute_g_beta`, `gated_delta_step_kernel`, `gdn_causal_conv_decode_silu`, `rms_norm_gated` | `spec.gdn` |
| `FULL_ATTN_QSA` (Qwen) | `qsa_indexer_pre_rope`, `qsa_rms_norm`, attention | `spec.qsa` |
| `FULL_ATTN_QSA` (Nemotron "attention") | dense attention (no indexer) | — |
| `MAMBA_SSM` | Mamba/SSM scan (primitive not ported yet) | — |
| any layer with `has_moe` | `router`, `q4_projection`, activation (`swiglu`/relu2), `weighted_combine` | `spec.expert` |

`GDNConfig` is defined in `model_adapter.hpp` because no sibling currently
defines one. **Integration:** when a `gdn.hpp` adapter lands, delete that struct
and `#include "gdn.hpp"`; keep the field names identical so no call site changes.
`QSAConfig` is reused from the unmodified `qsa.hpp`.

## Flat key=value config format

One file per model, UTF-8, one `key=value` line each. `#` comments / blank lines
ignored; vector values are comma-separated with no spaces; floats are decimal or
scientific (parsed with `std::stof`). `expected_*` keys are the Python-side
ground truth. The `qsa_*` / `gdn_*` / `hc_*` keys are `0`/empty for families that
do not use them.

```
model_family=QWEN4_EXP
hidden_size=2560
n_layers=48
layer_types=linear_attention,linear_attention,linear_attention,full_attention,...
expected_layer_kinds=LINEAR_ATTN_GDN,LINEAR_ATTN_GDN,...,FULL_ATTN_QSA,...
expected_has_moe=1,1,1,1,...
n_experts=512
top_k=10
group_size=32
moe_intermediate=640
expert_activation=SWIGLU
expert_prefix=language_model.model.layers.{L}.mlp.switch_mlp.{P}.{T}
norm_eps=1e-06
hc_count=4
hc_lowrank=320
output_gate_type=sigmoid
use_qsa_indexer=1
qsa_n_heads=4
qsa_kv_heads=1
qsa_head_dim=128
qsa_compress_ratio=4
qsa_token_budget=2048
qsa_eps=1e-06
gdn_conv_kernel=4
gdn_n_key_heads=16
gdn_n_value_heads=48
gdn_key_head_dim=128
gdn_value_head_dim=128
```

`{L}` = layer index, `{P}` = projection part, `{T}` = tensor part (`weight` |
`scales` | `biases`). Nemotron's `expert_prefix` is
`backbone.layers.{L}.mixer.switch_mlp.{P}.{T}`.

## Verify commands + output

```console
$ cd /Users/vincent/Documents/Veystrix
$ python3 cpp/engine/tools/gen_model_adapter_vectors.py /tmp/model_adapter_flat
wrote /tmp/model_adapter_flat/qwen_model_adapter.txt
wrote /tmp/model_adapter_flat/nemotron_model_adapter.txt

$ MLXROOT=.venv/lib/python3.12/site-packages/mlx && \
  xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
    cpp/engine/src/model_adapter.cpp cpp/engine/tests/test_model_adapter.cpp \
    -o /tmp/test_model_adapter && \
  /tmp/test_model_adapter /tmp/model_adapter_flat
test_model_adapter OK (958 checks)
```

Also passes with the task's exact `-lmlx` / rpath link line (the adapter itself
needs no MLX linkage, but linking MLX is harmless):

```console
$ MLXROOT=.venv/lib/python3.12/site-packages/mlx && \
  xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
    cpp/engine/src/model_adapter.cpp cpp/engine/tests/test_model_adapter.cpp \
    -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$(pwd)/$MLXROOT/lib" \
    -o /tmp/test_model_adapter_mlx && \
  /tmp/test_model_adapter_mlx /tmp/model_adapter_flat
test_model_adapter OK (958 checks)
```

## Integration instructions

1. Build `model_adapter.cpp` into the engine target (no CMake edit was made
   here — add it to your engine source list when reconciling the build).
2. In the Python loader (or a small generator), read the checkpoint `config.json`
   and produce the flat file via `gen_model_adapter_vectors.py` (or pass the same
   scalars directly to the builders from C++ integration code).
3. Build the `ModelSpec` once at engine init (`from_qwen_config` /
   `from_nemotron_config`); store it alongside the 48/52-element `LayerState`
   vector. Each decode step, loop `l = 0..n_layers-1`, call
   `dispatch_layer(spec, l)` and drive the primitives indicated by its flags.
4. Use `d.expert.tensor_prefix_format` to derive the per-layer streamed-expert
   tensor names: substitute `{L}` with the layer index, `{P}` with each
   projection part, `{T}` with `weight`/`scales`/`biases` — feed those to
   `expert_store` / `io_pager`.
5. `describe(spec)` is available for debug logging / parity-harness diagnostics.
6. When a `gdn.hpp` adapter appears, replace the local `GDNConfig` with its
   definition (see integration note above).

## Why this makes the engine future-model-ready

The engine no longer hard-codes Qwen or Nemotron constants. All family-specific
knowledge is isolated in two builder functions and a flat scalar format; the
decode loop consumes only `ModelSpec` + `dispatch_layer`. A new model must
provide exactly:

1. **Flat scalars**: `model_family`, `hidden_size`, `n_layers`, expert params
   (`n_experts`, `top_k`, `group_size`, `moe_intermediate`, `activation`),
   `norm_eps`, and any hyper-connection / QSA / GDN scalars it uses.
2. **An ordered layer list**: `layer_types` (or a `layers_block_type`-style list)
   that maps to `LayerKind` values, plus a per-layer MoE mask.
3. **A tensor-name convention**: `expert_prefix` with `{L}`/`{P}`/`{T}`
   placeholders, so streaming needs no family-specific string code.
4. **A builder** (or reuse `from_generic_config` + fill the spec) that translates
   its config schema into a `ModelSpec`. For a genuinely new primitive family
   (e.g. a new recurrent mixer), add a `LayerKind` value and a branch in
   `dispatch_layer`; no existing builder or primitive changes.

The test harness is the gate: add a generator branch that emits the new model's
flat file (including `expected_*` ground truth), and a `verify_*` function in
`test_model_adapter.cpp`. Bit-exact integer/float comparison and exact string
comparison are already enforced for the two current families.
