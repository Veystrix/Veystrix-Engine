# Remaining bit-exactness gaps — quantized shared expert + INDEXER-NONE fallback

Two small gaps left by `decoder_full.cpp` are closed here, in new self-contained
functions that do **not** modify `decoder*`, `engine.*`, `qsa*`, or CMake:
`moe_block_shared_quantized` and `full_attention_dense_fallback`.

## Production source investigated

```
.venv/lib/python3.12/site-packages/mlx_vlm/models/qwen3_5_moe/language.py
  Qwen3_5MoeSparseMoeBlock.__call__        lines 31-46   (router + SwitchGLU + shared + gate)
.venv/lib/python3.12/site-packages/mlx_vlm/models/qwen4_exp/language.py
  Qwen4ExpQSAIndexer.__call__              lines 306-380 (None early return at 329-332)
  Qwen4ExpAttention.__call__               lines 390-417 (qsa_mask None -> mask untouched -> super().__call__)
.venv/lib/python3.12/site-packages/mlx_vlm/models/qwen3_5/language.py
  Qwen3_5Attention.__call__ / _prepare_projected_qkv   lines 922-1033
.venv/lib/python3.12/site-packages/mlx_vlm/models/base.py
  create_attention_mask                    lines 214-228 (N == 1 -> None)
  scaled_dot_product_attention             lines 305-373 (mask=None -> mx.fast SDPA, no mask)
.venv/lib/python3.12/site-packages/mlx_vlm/models/cache.py
  create_attention_mask / KVCache.update_and_fetch    lines 73-83 / 345-367
models/flashnext/adaptive_topk.py
  _moe_call                                lines 257-285 (shared + sigmoid(shared_gate) combine)
models/flashnext/preflight.py
  quantized-tensor inventory               lines 96-106
models/flashnext/loader.py
  nn.quantize(class_predicate=...)         lines 206-236
```

## Is the shared expert quantized in production? YES (documented finding)

`Qwen3_5MoeSparseMoeBlock` is constructed with `self.shared_expert =
Qwen3_5MoeMLP(...)` (a dense MLP) **in the mlx-vlm model class**, but that is
only the placeholder. The real checkpoint stores `mlp.shared_expert.{gate,up,down}_proj`
as **WSB quantized triplets** (`preflight.py:105`), and the runtime loader's
`nn.quantize(...)` with a shape-driven `class_predicate` (`loader.py:206-236`)
replaces those `nn.Linear` modules with `nn.QuantizedLinear`. So production
compute for the shared GLU is `mx.quantized_matmul`, not dense matmul — exactly
the gap `moe_block_shared_quantized` closes by using `q4_projection` (single
expert, `rhs_indices=0`).

A standalone check confirmed that `mx.gather_qmm` over a one-expert Q4 triplet
is **bit-identical** to `nn.QuantizedLinear` / `mx.quantized_matmul` for the
decode-shaped bf16 inputs used here (same MLX 0.32.2 build):

```
lin == qmm        True
lin == gqmm       True   (after squeezing the slot dim)
uint32 equal      True
max abs diff      0.0
```

**Caveat — the sigmoid gate stays dense.** `preflight.py:104` also lists
`mlp.shared_expert_gate` as WSB, so production quantizes the gate too. The
helper intentionally keeps `shared_gate_logits_w` as a **dense** `[1, H]` weight
(matching `decoder_full.cpp::moe_block_with_shared`'s existing convention and
the task's function signature). The generator therefore seeds that gate as a
dense `nn.Linear`. If byte-exactness of the gate itself is ever required, pass
a `Q4Projection` for it and replace the final
`sigmoid(matmul(x, transpose(shared_gate_logits_w)))` with a `q4_projection`
call; the GLU path documented here is unaffected. This is the only part of the
shared-expert computation not covered bit-for-bit against production.

The `SharedQuantConfig::shared_quantized` flag handles both worlds: `true`
(Q4/WSB, the documented production gap), `false` (dense GLU fallback for
checkpoints whose shared expert is not quantized).

## What is implemented

1. **`moe_block_shared_quantized`** — routed experts via the already-verified
   `moe_block` (Q4 `gather_qmm`), plus the shared expert's gate/up/down computed
   with `q4_projection` over a one-expert triplet + `swiglu` + down, combined as
   `routed + sigmoid(x @ gate_w.T) * shared` (`adaptive_topk._moe_call`).
2. **`full_attention_dense_fallback`** — same q/k/v projection, one-centered
   q/k norms, RoPE and KV-cache append as `full_attention_forward`, but with
   **no QSA mask**: for a decode step (`S == 1`) the MLX fused SDPA is called
   with `mask_mode=""` and no mask array (production `mask=None`), and for
   `S > 1` with `mask_mode="causal"` (production `"causal"`). It asserts, via a
   copy of the indexer cache, that `qsa_full_indexer` really returns `None`,
   and throws otherwise.

The generator forces the eager RoPE branch (`rotary_emb.fused_apply = False`)
and uses the real module's one-centered gain setting for q/k norms, so both
primitives and the C++ code reproduce the same op order bit-for-bit.

## Files created (all within ownership)

- `cpp/engine/src/parity_gaps.hpp` — `SharedQuantConfig` + the two function declarations.
- `cpp/engine/src/parity_gaps.cpp` — implementations; depends only on
  `engine.cpp` (`q4_projection`, `moe_block`, `swiglu`) and `qsa*.cpp`
  (`qsa_rms_norm`, `qsa_apply_rope`, `qsa_full_indexer`). No shared file modified.
- `cpp/engine/tools/gen_parity_gaps_vectors.py` — builds the real quantized
  `Qwen3_5MoeSparseMoeBlock` and the real indexer-None `Qwen4ExpAttention`,
  asserts real == primitive (bit-exact) for both, saves inputs/weights/expected.
- `cpp/engine/tests/test_parity_gaps.cpp` — loads vectors, compares via the
  uint32 byte view, prints `test_parity_gaps OK`.
- `cpp/engine/tests/vectors_parity_gaps/` — generated data (regenerable).

## Verification (standalone, no CMake edits)

```
cd /Users/vincent/Documents/Veystrix
.venv/bin/python cpp/engine/tools/gen_parity_gaps_vectors.py \
    --out cpp/engine/tests/vectors_parity_gaps
MLXROOT=.venv/lib/python3.12/site-packages/mlx
xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
  cpp/engine/src/engine.cpp cpp/engine/src/qsa.cpp cpp/engine/src/qsa_full.cpp \
  cpp/engine/src/parity_gaps.cpp cpp/engine/tests/test_parity_gaps.cpp \
  -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$(pwd)/$MLXROOT/lib" -o /tmp/test_pg
/tmp/test_pg cpp/engine/tests/vectors_parity_gaps
```

Output:

```
self-check OK: real modules == C++ primitive formulas (both gaps, bit-exact)
wrote parity-gaps vectors to cpp/engine/tests/vectors_parity_gaps
wrote 39 arrays
test_parity_gaps OK
```

Also verified with a second seed (`--seed 777`): both self-check and C++ test
pass. Both gaps are therefore **bit-exact for the one-token decode step**.

## Remaining scope / caveats

- The vectors cover `S == 1` decode. The `S > 1` branch uses `mask_mode="causal"`
  (MLX fused causal kernel) but was not exercised here; a prefill generator
  with 3-D MRoPE positions and fused RoPE would be needed to prove that path.
- The inherited fused-Metal RoPE caveat from `qsa_full_port.md` still applies:
  the generator forces the eager branch, which the C++ `qsa_apply_rope`
  reproduces bit-for-bit.
- The sigmoid `shared_expert_gate` is dense in these vectors (see caveat above);
  production quantizes it as WSB.
