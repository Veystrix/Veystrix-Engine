# Nemotron 3.5 Lightning C++ decode loop — dataflow and checkpoint naming

This document records the exact tensor names, shapes, dtypes and Python
dataflow the C++ `nemotron_generate` tool assembles. Everything below is
**measured** from the real checkpoint
`/Users/vincent/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit-mlx-community`
(4 shards, `model.safetensors.index.json`, 729 tensors) or **cited** from the
installed `mlx_lm` / Veystrix sources. No estimate is used for parity claims.

## 1. Config (cited from `config.json`)

- hidden = 2688, vocab = 131072, layers = 52
- `layers_block_type`: 23 `mamba`, 23 `moe`, 6 `attention`
- attention: 32 q heads, 2 kv heads, head_dim 128, no RoPE, no q/k norm
- mamba: 64 heads × 64 dim (`I = 4096`), `n_groups = 8`, `ssm_state = 128`,
  `conv_kernel = 4`, `conv_dim = I + 2·G·Ds = 6144`,
  `in_proj` logical width `I + conv_dim + H = 10304`
- MoE: 128 routed experts, top-6, `moe_intermediate = 1856`, 1 shared expert
  with intermediate 3712, `relu2`, `routed_scaling_factor = 2.5`,
  `norm_topk_prob = true`, `n_group = 1`, `topk_group = 1`
- quantization: affine 4-bit, group 64 (`packed_in = logical_in / 8`,
  `n_groups = logical_in / 64`)
- **time-step limits**: `mlx_lm.models.nemotron_h.ModelArgs.__post_init__`
  sets `time_step_limit = (0.0, inf)` when config.json has no `time_step_limit`
  key. The checkpoint's `time_step_min/max/floor` keys are **not read** by this
  mlx-lm revision, so the C++ tool passes `(0, inf)` to `mamba_ssm_step_kernel`.

## 2. Exact per-layer tensor names (measured from the index)

Prefix: `backbone.layers.{L}.`. Every layer has `norm.weight` (bf16 [2688]).

### Mamba layer (`mixer.`)
| suffix | dtype | shape |
|---|---|---|
| `norm.weight` | BF16 | [4096] |
| `in_proj.weight` / `.scales` / `.biases` | U32 / BF16 / BF16 | [10304,336] / [10304,42] / [10304,42] |
| `conv1d.weight` | BF16 | [6144,4,1] |
| `conv1d.bias` | BF16 | [6144] |
| `A_log`, `dt_bias`, `D` | BF16 | [64] each (Python loads them into f32 params; kernel casts to f32) |
| `out_proj.weight` / `.scales` / `.biases` | U32 / BF16 / BF16 | [2688,512] / [2688,64] / [2688,64] |

### Attention layer (`mixer.`)
`q_proj`, `k_proj`, `v_proj`, `o_proj`, each a Q4-G64 triplet
`.weight` (U32), `.scales` (BF16), `.biases` (BF16). Logical widths:
q = 4096, k = v = 256, o in = 4096, all out = 2688 (except k/v = 256).

### MoE layer (`mixer.`)
| suffix | dtype | shape |
|---|---|---|
| `gate.weight` | BF16 | [128,2688] (dense, no `.scales`) |
| `gate.e_score_correction_bias` | F32 | [128] |
| `shared_experts.up_proj.*` | U32/BF16/BF16 | [3712,336] / [3712,42] / [3712,42] |
| `shared_experts.down_proj.*` | U32/BF16/BF16 | [2688,464] / [2688,58] / [2688,58] |
| `switch_mlp.fc1.weight` | U32 | [128,1856,336] |
| `switch_mlp.fc1.scales`/`.biases` | BF16 | [128,1856,42] |
| `switch_mlp.fc2.weight` | U32 | [128,2688,232] |
| `switch_mlp.fc2.scales`/`.biases` | BF16 | [128,2688,29] |

### Head / embeddings
- `backbone.embeddings.{weight,scales,biases}`: U32 [131072,336], BF16 [131072,42] ×2
- `backbone.norm_f.weight`: BF16 [2688]
- `lm_head.{weight,scales,biases}`: U32 [131072,336], BF16 [131072,42] ×2

Note: the routed experts live in different shards per layer, and some layers
split `fc1` and `fc2` across adjacent shards. `gen_nemotron_config.py` emits an
exact `shard.<tensor>` map so the C++ `ExpertReader` can `pread` just the
selected rows by byte offset from the safetensors header.

## 3. Python decode dataflow (cited from `models/nemotron35/backend_core.py`)

`_step(tokens) = model.lm_head(model.backbone([tokens], cache)[:, -1:, :])[:, -1, :]`.

`NemotronHBlock.__call__` (mlx_lm `nemotron_h.py`): `x + mixer(norm(x))`,
where `norm = nn.RMSNorm(hidden, eps)` == `mx.fast.rms_norm(x, weight, eps)`.

### Mamba branch
`in_proj -> split(gate, conv_input, dt) -> depthwise conv+bias+silu ->
split(hidden_ssm, B, C) -> ssm_update_kernel (S=1) -> reshape -> MambaRMSNormGated
(y, gate) -> out_proj`. Caches: conv `[1,3,6144]` bf16, ssm `[1,64,64,128]` f32.
S=1 with a non-None state selects the fused `ssm_update_kernel` (the reference
capture zero-seeds the cache so the first token takes this path too).

### Attention branch (bf16 KV)
`q = q_proj(x).reshape(B,1,Hq,D).transpose(0,2,1,3)`, likewise k/v with Hkv;
`KVCache.update_and_fetch`; `mx.fast.scaled_dot_product_attention(q, k, v,
scale=head_dim^-0.5, mask=None)`; transpose+reshape; `o_proj`. No RoPE.

### MoE branch
1. `gates = x @ gate.weight.T` (dense bf16);
   `orig = sigmoid(gates.f32)`; `scores = orig + e_score_correction_bias` (f32).
2. `n_group=1`, so no group masking. `inds = argpartition(-scores, kth=5)[..., :6]`.
3. `scores = take_along_axis(orig, inds) / (sum + 1e-20) * 2.5` (f32).
4. `switch_mlp`: with `VEYSTRIX_NEMOTRON_GATHER=1` the Python path uses
   `gather_qmm` over the selected experts — the same kernel as the validated
   C++ `moe_block_relu2()` primitive (up -> relu2 -> down -> weighted sum).
5. Routed output is cast back to bf16, then
   `+ shared_experts(x)` = `down_proj(relu2(up_proj(x)))` (both Q4-G64).

### Final head
`norm_f(hidden)` then `lm_head` (Q4-G64 quantized linear), last position only.
Greedy = `argmax(flatten(logits))`. Digest = SHA-256 of f32-LE logits.

## 4. C++ tool assembly (measured)

`cpp/engine/tools/nemotron_generate.cpp` drives `dispatch_layer(ModelSpec, L)`
from `from_nemotron_config()` (extended overload), so each layer runs exactly
one of the validated primitives. Dense weights use `DenseLoader::linear()`
(QPacked G64, kept packed) and routed experts are streamed by `ExpertReader`
(only the selected rows are `pread`). One token is fed per `run_step`: prefill
is `prompt.size()` sequential steps, then N greedy decode steps with SHA-256
digests. Output lines match `check_nemotron_parity.py`'s strict grammar.

## 5. Validation gate

`gen_nemotron_reference.py` captures a short greedy reference from the Python
backend (bf16 KV, `chunk_tokens=1`, `GATHER=1`, zero-seeded Mamba state) under
`real_run_guard.py`. `check_nemotron_parity.py` compares tokens + digests
strictly. See the files for the exact env pinning and citations.
