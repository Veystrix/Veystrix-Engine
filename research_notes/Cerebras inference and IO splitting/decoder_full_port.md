# decoder_full C++ port — status & integration note

This closes the two gaps left by `decoder.cpp` (verified bit-exact only for an
all-linear tiny config with a zeroed shared expert and a QSA cache carried but
never composed):

1. **Non-zero shared expert** in the MoE block.
2. **Full-attention (QSA sparse) branch** composed into the decoder loop, so a
   mixed stack (GDN linear layers + full-attention layers) runs in one step.

No existing source, test, generator, `CMakeLists.txt`, or branch was touched.
Everything new lives in `cpp/engine/src/decoder_full.{hpp,cpp}`,
`cpp/engine/tools/gen_decoder_full_vectors.py`,
`cpp/engine/tests/test_decoder_full.cpp`, and this note.

## Python reference (all paths real, MLX C++ 0.32.2)

```
models/flashnext/adaptive_topk.py
  _moe_call                          lines 54-289
    shared      = self.shared_expert(x)                    (line 264)
    shared_gate = mx.sigmoid(self.shared_expert_gate(x))   (line 265)
    return y + shared_gate * shared                        (line 278)
.venv/.../mlx_vlm/models/qwen3_5_moe/language.py
  Qwen3_5MoeSparseMoeBlock.__call__  lines 31-46
    shared_expert = Qwen3_5MLP(dim, shared_expert_intermediate_size)
    swiglu(gate, up) = nn.silu(gate) * up                  (models/activations.py:8)
.venv/.../mlx_vlm/models/qwen4_exp/language.py
  Qwen4ExpAttention.__call__         lines 390-417  (indexer mask -> dense attn)
  Qwen3_5Attention.__call__          qwen3_5/language.py:922-963
  Qwen3_5Attention._prepare_projected_qkv                :965-1033
  Qwen4ExpDecoderLayer.__call__      lines 744-772  (branch/injection)
  Qwen4ExpQSAIndexer.__call__        lines 306-380  (bit-exact qsa_full_indexer)
.venv/.../mlx_vlm/models/base.py
  scaled_dot_product_attention      lines 305-373  (mx.fast.scaled_dot_product_attention)
```

## What is implemented

### `moe_block_with_shared` (decoder_full.cpp)
Routed experts use the already-verified `moe_block()` (Q4 gather_qmm). The
shared expert is the **dense** `Qwen3_5MoeMLP` GLU:
```
g     = x @ W_gate.T ;  u = x @ W_up.T
act   = u * (g * sigmoid(g))          # engine's explicit swiglu
s     = act @ W_down.T
sg    = sigmoid(x @ W_shared_gate.T)
out   = routed + sg * s
```
The combine is exactly `adaptive_topk._moe_call`'s `y + shared_gate*shared`
with `renorm_blend = 0` (production threshold 1.0 path).

**Quantized shared expert.** In the real checkpoint the three shared-expert
projections are stored as Q4 WSB triplets (`preflight.py:101-106`), not dense
`nn.Linear` weights. The engine's `Q4Projection` / `q4_projection()` already
compute those matmuls bit-exactly; a production integration should build the
same `g/u/down` path with `q4_projection(...)` instead of `mx::matmul`. The
combine math (`sg*s`, `routed + ...`) is unchanged. The bit-exact vector here
uses the dense `Qwen3_5MoeMLP` that `Qwen3_5MoeSparseMoeBlock` actually
constructs; that is the module the Python reference runs.

### `full_attention_forward` (decoder_full.cpp)
One `Qwen4ExpAttention` decode branch, mirroring the Python op-for-op:
1. `qsa_full_indexer` -> boolean sparse mask `[B,1,S,key_len]` (the already
   bit-exact full QSA indexer; `qsacache.offset` kept in sync with the KV cache).
2. `q/k/v_proj` (dense `nn.Linear`, bias=False); `q_proj` split into queries
   and the gating half (`Hq*D*2` -> two `[B,S,Hq,D]`).
3. `q_norm` / `k_norm` via `qsa_rms_norm` — the one-centered/zero-centered
   gain used by the verified QSA primitives (see the RMSNorm caveat below).
4. RoPE via `qsa_apply_rope` (eager branch), attention + indexer share params.
5. KV cache `update_and_fetch` = concatenate on axis 2; mask sliced to
   `kv_seq_len = S + offset + 1` exactly as `_prepare_projected_qkv`.
6. `mx::fast::scaled_dot_product_attention(q,k,v,scale,"",mask,nullopt)` —
   **the same MLX 0.32.2 fused kernel** Python calls, so it is bit-exact.
7. `out.transpose(0,2,1,3).reshape(B,S,Hq*D) * sigmoid(gate)` then `o_proj`.

### `run_decoder_step_full`
Extends `run_decoder_step`: per-layer `AttnKind` selects GDN (linear) or the
full-attention branch; the MoE step uses `moe_block_with_shared` (non-zero
shared expert); `DecoderFullDebug` exposes per-layer branch/moe/hidden and the
QSA mask. `GDNCacheState` and `FullAttentionCache` are updated in place.

## Bit-exact subset (proven by the standalone test)

Tiny real config (seed 20261117; also 1, 12345, 999 verified):
```
H=32, hc_count=2, hc_lowrank=16, 3 layers: 2 GDN (Hk=1,Hv=2,Dk=Dv=32,K=4)
+ 1 QSA full-attention (2 q heads, 1 kv head, head_dim=16, partial rotary=1.0),
indexer 2 heads, ratio=2, budget=4, topk=2, rope base=10000, mrope_section=[6,6,4],
prepopulated KV+indexer length 8 (past_len), 2-D text positions, eager RoPE,
8 experts top-2, group_size=32 bits=4, NON-ZERO dense shared expert (inter=32),
vocab=64, rms_norm_eps=1e-6, all modules eval()
```
The generator runs the step twice — real modules vs a primitive-formula
reference that mirrors `decoder_full.cpp` exactly — and **asserts byte
identity** before saving. The C++ test then re-checks, via raw `uint32` views:

* final logits,
* every layer's attention branch output (GDN or full attention),
* every layer's MoE+shared output,
* every layer's hidden state,
* the full layer's boolean QSA mask (compared byte-wise as `uint8`, since a
  1-byte dtype cannot be viewed as `uint32`).

All comparisons pass bit-exactly for the four seeds above. `test_decoder`
(original all-linear, zero-shared) still passes unchanged.

### RMSNorm convention caveat (important when integrating)
`patch_rmsnorm.apply()` makes **all** `Qwen4ExpRMSNorm` instances use the raw
weight as the gain (no `1+`) — that is what the decoder's `gated_residual` /
`final_head` and `rms_norm` implement. The QSA primitives (`qsa_rms_norm`) use
the **zero-centered** gain `1 + weight`. A single process therefore needs two
conventions. The generator applies the production patch for the hc/head norms
and sets `_flashnext_one_centered = False` on the full layer's q/k norms and
the indexer's q/k layernorms (the patch's own per-instance `configure`
mechanism), so the real modules use `1 + weight` exactly where the C++ QSA
primitives do. This is why real and primitive paths are byte-identical; a
checkpoint whose q/k norms use the patched convention would need the
`1 + weight` factors folded into the saved weights before calling the QSA
primitives (documented here rather than papered over).

## What remains for full production parity (NOT implemented)

1. **Quantized shared expert weights.** `moe_block_with_shared` uses dense
   `mx::matmul` for gate/up/down; production stores WSB triplets. Substitute
   `q4_projection` (no new math), then re-verify against a real quantized
   checkpoint.
2. **Fused-Metal RoPE.** `qsa_apply_rope` reproduces the eager `mx.cos`/`mx.sin`
   path; production's `MRoPERotaryEmbedding.fused_apply=True` uses a Metal
   kernel differing by ~1 ULP. The generator forces the eager branch. Matching
   the fused kernel requires calling that Metal kernel from C++ (same caveat as
   `qsa_full_port.md`).
3. **3-D MRoPE (multimodal) positions.** `qsa_apply_rope` supports them, but
   this vector uses 2-D text positions only; a multimodal decode vector is not
   included.
4. **Prefill / S>1.** `mx::fast.scaled_dot_product_attention` works for S>1,
   but the QSA mask, the GDN branch, and cache semantics for S>1 are not
   verified by these vectors (see `decoder_loop_port.md`).
5. **Dense fallback when the indexer returns None** (`max_complete_blocks <=
   block_topk`): `full_attention_forward` throws rather than silently running
   an unmasked dense attention. Wire the `None` path with the production
   causal mask if needed.
6. **Quantized KV cache** (`QSAQuantizedKVCache`) is not composed; the dense
   `KVCache` concatenation path is what is verified.
7. **CMake / model adapter integration**: add `src/decoder_full.cpp` to the
   engine library and a `test_decoder_full` target (no CMake edits were made
   here). `model_adapter.hpp`'s second `GDNConfig` still collides with
   `gdn_layer.hpp`'s; map scalars explicitly as in `decoder_loop_port.md`.

## Verification (standalone, no CMake edits)

```bash
cd /Users/vincent/Documents/Veystrix
.venv/bin/python cpp/engine/tools/gen_decoder_full_vectors.py \
    --out cpp/engine/tests/vectors_decoder_full
MLXROOT=.venv/lib/python3.12/site-packages/mlx
xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
  cpp/engine/src/engine.cpp cpp/engine/src/gdn_kernel.cpp cpp/engine/src/gdn_conv.cpp \
  cpp/engine/src/gdn_layer.cpp cpp/engine/src/qsa.cpp cpp/engine/src/qsa_full.cpp \
  cpp/engine/src/decoder.cpp cpp/engine/src/decoder_full.cpp \
  cpp/engine/tests/test_decoder_full.cpp \
  -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$(pwd)/$MLXROOT/lib" -o /tmp/test_decoder_full
/tmp/test_decoder_full cpp/engine/tests/vectors_decoder_full
```

Generator output:
```
self-check OK: real modules == C++ primitive formulas (bit-exact)
wrote decoder_full vectors to cpp/engine/tests/vectors_decoder_full
logits shape=(1, 1, 64), first row head=[-0.30656815  0.585963    0.22350328  0.00657212  0.14886919]
layer kinds = ['linear', 'linear', 'full']
```
Test output:
```
test_decoder_full OK (bit-exact, 3 layers, 1 full-attn)
```
Same result for seeds `1`, `12345`, `999`.

### Prior-attempt bug fixed here
`HEAD` already contained these four files (commit `9136e4c`), but
`test_decoder_full.cpp` compared the boolean QSA mask with a `uint32` view,
which throws (`[view] ... last axis must be a multiple of the requested type
size`). This deliverable adds a `uint8` byte-view comparison for the mask; the
implementation itself was unchanged. No commit was made.
