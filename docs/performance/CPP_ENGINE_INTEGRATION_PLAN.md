# C++ engine integration plan — completing the bit-exact decode rewrite

**Date:** 2026-11-17
**Branch:** `cpp-engine` (plan only — no code changed by this document)
**Goal:** a concrete, code-grounded plan to take `cpp/engine/` from "bit-exact MoE
compute core" to a full standalone, bit-exact C++ decode loop (and prefill), with
an end-to-end parity gate on the real checkpoint.

> **Parallel work in flight (2026-11-17).** The working tree also contains
> **untracked, unverified** sibling ports not yet represented in `engine.hpp`:
> `cpp/engine/src/gdn_conv.{h,cpp}` (GDN causal-conv1d decode chain),
> `cpp/engine/src/qsa.{hpp,cpp}` (QSA indexer **pre-RoPE** projection + norms only —
> RoPE and block selection explicitly out of scope), `cpp/engine/src/sampling.{hpp,cpp}`
> (full non-greedy sampler incl. categorical RNG), plus their vector generators and
> tests. This plan is written against the **verified** primitives in `engine.hpp`
> (§1) and treats those three files as work to reconcile at M2/M3/M6 (§8); their
> signatures are summarized in §1.6. Do not assume they pass the parity gate until
> their own ctests are run.

**Grounding sources.** Current state: `cpp/engine/README.md`, `cpp/engine/STATUS.md`.
Primitive signatures: `cpp/engine/src/engine.hpp`. Native loader:
`cpp/engine/src/expert_store.{hpp,cpp}`. Pager scaffold: `cpp/decode/src/io_pager.hpp`.
Python reference: `.venv/lib/python3.12/site-packages/mlx_vlm/models/qwen4_exp/language.py`
(hereafter `qwen4_exp/language.py`), `.../qwen3_5/language.py` (hereafter
`qwen3_5/language.py`), `.../qwen3_5_moe/language.py` (hereafter
`qwen3_5_moe/language.py`), `veystrix/backends/flashnext.py`, `veystrix/sampling.py`,
`models/flashnext/prefill.py`, `models/flashnext/prefill_ring.py`,
`models/flashnext/qsa_chunk.py`. Scope: `docs/performance/ENGINE_LANGUAGE_ANALYSIS.md`.
Bit-exact I/O rules: `docs/performance/OPUS_E0_E1_SPEC.md`,
`docs/performance/OPUS_MIRACLE_PROMPT.md`. Test guard:
`scripts/flashnext/real_run_guard.py`.

---

## 1. What already exists in C++, exactly

`cpp/engine/src/engine.hpp` (namespace `veystrix::engine`, `mx = mlx::core`,
`engine.hpp:24-26`). All primitives are built against the same MLX 0.32.2 as the
Python engine so reduce order matches (`README.md:3-7, 87-89`). Verified: 5/5 ctests,
raw-byte (uint32 view) parity against Python vectors and real Qwen expert weights
(`README.md:30-59`, `STATUS.md:7-33`).

### 1.1 MoE / router / projection primitives

| Signature (file:line) | Mirrors |
|---|---|
| `Q4Projection{mx::array weight, scales, biases}` — `engine.hpp:28-35` | packed affine Q4 gate/up/down tensors |
| `ExpertWeights{Q4Projection gate, up, down; int group_size=32; int bits=4}` — `engine.hpp:37-48` | one layer's 3 expert projections |
| `struct RouterOutput{mx::array indices; mx::array scores}; RouterOutput router(const mx::array& logits, int top_k, float renorm_blend)` — `engine.hpp:56-60` | `adaptive_topk._moe_call` at threshold 1.0; returns int32 `[tokens,k]` + float `[tokens,k]` |
| `mx::array q4_projection(const mx::array& x, const Q4Projection& p, const mx::array& route, int group_size, int bits, bool slot_input=false, bool sorted_indices=false)` — `engine.hpp:67-69` | `mx.gather_qmm(..., rhs_indices=route, transpose=True, mode="affine")` / `StreamingSwitchLinear.__call__` |
| `mx::array swiglu(const mx::array& gate, const mx::array& up)` — `engine.hpp:72` | `up * (gate * sigmoid(gate))` |
| `mx::array weighted_combine(const mx::array& expert_out, const mx::array& scores)` — `engine.hpp:75` | `(out * scores[...,None]).sum(-2)` |
| `mx::array moe_block(const mx::array& x, const mx::array& route, const mx::array& scores, const ExpertWeights& w)` — `engine.hpp:81-82` | full routed MoE layer; **routes/scores are external** |
| `mx::array moe_block_from_logits(const mx::array& x, const mx::array& logits, int top_k, float renorm_blend, const ExpertWeights& w)` — `engine.hpp:85-87` | router + MoE in one call |
| `mx::array moe_block_relu2(const mx::array& x, const mx::array& route, const mx::array& scores, const Q4Projection& up, const Q4Projection& down)` — `engine.hpp:91-93` | Nemotron relu2 MLP (separate engine; not on Qwen path) |

**Critical scope note:** `moe_block()` is explicitly "routed experts only, no shared
expert" (`engine.hpp:77-82`). The production Python `Qwen3_5MoeSparseMoeBlock.__call__`
also computes a **shared expert** and a sigmoid-gated contribution
(`qwen3_5_moe/language.py:43-46`) and adds it to the routed result (`:46`). That
shared path is **not** in any C++ primitive — it must be ported (see §2, §8).

### 1.2 Norm / gating / residual primitives

| Signature (file:line) | Mirrors |
|---|---|
| `mx::array rms_norm(const mx::array& x, const mx::array& weight, float eps)` — `engine.hpp:98` | one-centered production RMSNorm (`patch_rmsnorm.py`); `Qwen4ExpRMSNorm` (`qwen4_exp/language.py:215-236`) |
| `mx::array rms_norm_gated(const mx::array& x, const mx::array& weight, const mx::array& gate, float eps, bool sigmoid_act)` — `engine.hpp:103-104` | `Qwen4ExpRMSNormGated` (`qwen4_exp/language.py:239-254`); GDN block output norm |
| `struct GatedResidualOutput{mx::array mixed_input; mx::array injection_weights}; GatedResidualOutput gated_residual(const mx::array& hyper_input, const mx::array& norm_weight, const mx::array& down_w, const mx::array& up_w, const mx::array& inject_w, int hc_count, int hidden_size, float eps)` — `engine.hpp:153-162` | `Qwen4ExpGatedResidual.__call__` (`qwen4_exp/language.py:442-454`) |

### 1.3 GDN (linear attention) primitives

| Signature (file:line) | Mirrors |
|---|---|
| `std::pair<mx::array,mx::array> gdn_normalize_qk(const mx::array& q, const mx::array& k)` — `engine.hpp:112` | `Qwen4ExpGatedDeltaNet._normalize_qk` (`qwen4_exp/language.py:266-272`) |
| `mx::array gdn_gate_sigmoid(const mx::array& y, const mx::array& gate)` / `gdn_gate_silu(...)` — `engine.hpp:116-117` | `compile_gdn._gate_sigmoid` / `_gate_silu` chains |
| `struct DeltaStepOutput{mx::array y; mx::array state}; DeltaStepOutput gated_delta_step(const mx::array& q, const mx::array& k, const mx::array& v, const mx::array& g, const mx::array& beta, const mx::array& state)` — `engine.hpp:124-130` | ops-based delta-rule step; `_gated_delta_step_ops`; `q,k:[B,H,Dk]`, `v:[B,H,Dv]`, `g,beta:[B,H]`, `state:[B,H,Dv,Dk]` f32 |
| `DeltaStepOutput gated_delta_step_kernel(const mx::array& q, const mx::array& k, const mx::array& v, const mx::array& g, const mx::array& beta, const mx::array& state)` — `engine.hpp:137-139` | bit-exact fused Metal kernel `gdn_kernel.h` (simd_sum reduction order); `q,k:[B,T,Hk,Dk]`, `v:[B,T,Hv,Dv]`, `state:[B,Hv,Dv,Dk]`; falls back to `gated_delta_step` if Metal unavailable |
| `std::pair<mx::array,mx::array> compute_g_beta(const mx::array& A_log, const mx::array& a, const mx::array& b, const mx::array& dt_bias)` — `engine.hpp:143-146` | `g = exp(-exp(A_log) * softplus(a + dt_bias)); beta = sigmoid(b)` |

### 1.4 Sampler primitive

`mx::array sample_greedy(const mx::array& logits)` — `engine.hpp:108` — production
greedy (`veystrix/sampling.py:177-180`: `mx.argmax(...).reshape(1).astype(mx.uint32)`).
Everything beyond greedy is unported (`README.md:76`, `STATUS.md:46`).

### 1.5 Native loader and pager (scaffolds, not primitives)

- `ExpertWeights load_layer(const std::string& shard_path, const std::string& prefix, int group_size=32, int bits=4)` — `expert_store.hpp:22-24`; implemented `expert_store.cpp:37-45` via `mx::load_safetensors` (`:22`), with `uint16 → bfloat16` view for scales/biases (`:9-16`). Loads all 512 experts of one layer, verified bit-exact (`STATUS.md:24-26`).
- `Q4Projection load_projection(shard_path, prefix, proj, group_size)` — `expert_store.hpp:16-19`.
- `PagerPlan plan_remap(route, expert_slot_pairs, free_slots, fd, header, row_bytes, slot_base, slot_stride)` — `io_pager.hpp:36-40`; returns `PagerPlan{slots (route order), reads (dedup misses)}` (`:27-30`).
- `int run_batched_reads(const std::vector<PagerRead>& reads, int width)` — `io_pager.hpp:45`, batched positioned pread via `native_read.c`. `PagerRead` is ABI-matched (`io_pager.hpp:17-23`). **Excludes** O_DIRECT/F_NOCACHE bypass and KICK/WAIT overlap — both measured slower (`io_pager.hpp:5-8`, `docs/performance/DO_NOT_IMPLEMENT.md`).

### 1.6 In-flight (untracked, unverified) sibling ports — reconcile before relying on them

These are **not** in `engine.hpp` and are **not** covered by the 5/5 ctest claim
(`README.md:50-59`); they are listed so the milestones can absorb them:

- `gdn_conv.h:19-32` — `gdn_decode_depthwise_conv(conv_input [B,K,C], weight [K,C] f32)`
  → `[B,1,C]`; `gdn_causal_conv_decode(conv_input, weight [C,K,1])`; and
  `gdn_causal_conv_decode_silu(...)` adding the fused `silu`. Mirrors
  `_qwen3_5_decode_depthwise_conv` / `_causal_conv1d_decode` (`qwen3_5/language.py:72-74, 1091-1101`).
- `qsa.hpp:48-96` — `QSARoPEConfig`, `QSAConfig`, `qsa_rms_norm(x, weight, eps)`,
  `QSAIndexerOutput qsa_indexer_pre_rope(x, proj_weight, q_weight, k_weight, cfg)`
  returning `query [B,Hq,S,D]` and `pooled_keys [B,1,S//ratio,D]`. Covers only
  `qwen4_exp/language.py:317-344`; **MRoPE, scoring, argpartition block selection,
  and KV cache are explicitly deferred** (`qsa.hpp:10-14`).
- `sampling.hpp:24-61` — `SamplingParams`, `Distribution`,
  `sampling_distribution(logits, params, seen)`, `sampling_mask(logits, min_p, top_p)`,
  `sample_non_greedy(logits, params, seen, key)`. Mirrors `sampling.py:117-185`;
  notes shared PRNG parity via `mx.random.seed` / `mlx::core::random::seed`
  (`sampling.hpp:8-11`). This is exactly the §6 extension, pending verification.

---

## 2. Per-token decode dataflow: Python → C++ mapping

The Python outer decode step is `standard_tokens()` in `veystrix/backends/flashnext.py:1267-1311`:
yield the current token, then `step = self.language(token[None], cache=self.cache)`
(`:1307`), `token = sampler(step.logits[:, -1, :])` (`:1308`), `mx.eval(token)`
(`:1309`). The final token is committed with `skip_logits=True` (`:1301-1302`).
`self.language` is the `LanguageModel` whose `model` is `Qwen4ExpModel`
(`qwen4_exp/language.py:832-843`).

The forward itself is `Qwen4ExpModel.__call__` (`qwen4_exp/language.py:792-829`):
embed then `mx.tile(hidden_states, (1,1,hc_count))` (`:804-807`), build `fa_mask` /
`ssm_mask` (`:811-814`), the 48-layer loop (`:817-825`), and the final
`hyper_connection_mixer(hidden_states)` (`:829`). Per-layer body is
`Qwen4ExpDecoderLayer.__call__` (`:744-772`).

Mapping (legend: **PORTED** = C++ primitive exists; **PORT** = re-express with MLX
C++ ops; **STATE** = needs persistent per-layer state):

| # | Python step (cite) | Status / C++ target |
|---|---|---|
| 0 | Embed + tile to `hc_count` (`qwen4_exp/language.py:804-807`) | **PORT** (embedding gather + tile; no primitive) |
| 0b | `fa_mask` / `ssm_mask` construction (`:811-814`, `_create_qwen3_5_*_mask`) | **PORT**; attention mask is sparse QSA mask (step 4c) |
| 1 | PLE per-layer embedding, if layer in `ple_layer_ids` (`:734-740, 752-755`) | **PORT + STATE** (`ArraysCache` extra 2 slots, `:848`) |
| 2 | `attn_hyper_connection = Qwen4ExpGatedResidual.__call__` (`:757-759`, `:442-454`) | **PORTED** `gated_residual()` (`engine.hpp:157`). *Hyper-connection residual is the `hidden_states` tensor itself* (`hyper_input` returned at `:454`). |
| 3a | GDN `in_proj_qkv / z / b / a` (`qwen3_5/language.py:1116-1121`), `z` reshape (`:1123`) | **PORT** (plain `mx::Linear`/matmul) |
| 3b | Causal conv + conv state (`qwen3_5/language.py:1125-1159`) | **PORT (WIP) + STATE** — `gdn_causal_conv_decode_silu` exists in `gdn_conv.h:31-32` (untracked; verify first). `conv_state = cache[0]`, `[B,K-1,conv_dim]` (`:1129`); decode uses `_causal_conv1d_decode` (`:1152-1157`, helper `:72-74`) |
| 3c | `q,k` L2 norm (`qwen3_5/language.py:1173`) | **PORTED** `gdn_normalize_qk()` (`engine.hpp:112`) |
| 3d | `g,beta` from `a,b,A_log,dt_bias` | **PORTED** `compute_g_beta()` (`engine.hpp:143`) |
| 3e | Gated-delta recurrence (`qwen3_5/language.py:1175-1186`) | **PORTED** `gated_delta_step_kernel()` / `gated_delta_step()` (`engine.hpp:137/128`). **STATE** `state = cache[1]`, `[B,Hv,Dv,Dk]` f32 (`:1170,1188-1189`) |
| 3f | `norm(out, z)` + `out_proj` (`qwen3_5/language.py:1195-1196`) | `rms_norm_gated()` **PORTED** (`engine.hpp:103`); `out_proj` **PORT** |
| 4a | QSA indexer: `index_qk_proj`, q/k norms, MRoPE, pooled keys, `argpartition` block selection, `selected_tokens` mask (`qwen4_exp/language.py:306-380`, `qsa_chunk.py:160-203, 419-527`) | **PARTLY PORTED (WIP)** — `qsa_indexer_pre_rope` (`qsa.hpp:91-96`) covers `:317-344` only; MRoPE/scoring/block selection still **PORT**. **STATE** `index_keys`, `index_position_ids` in `QSAKVCache` (`qwen4_exp/language.py:41-63`). Highest-risk item (§8). |
| 4b | `q/k/v_proj` with gate split; `q_norm`/`k_norm` (`qwen3_5/language.py:931-944, 976-987`); Qwen4 one-centered norms (`qwen4_exp/language.py:386-388, 215-236`) | `rms_norm()` **PORTED** (`engine.hpp:98`); projections/reshape **PORT** |
| 4c | MRoPE (`qwen3_5/language.py:1010-1015`), KV cache update (`:1031-1032`), mask combination (`qwen4_exp/language.py:398-410`), SDPA / QSA gather (`qwen3_5/language.py:957-960`, `qsa_chunk.py:211-334`) | **PORT + STATE**. `QSAKVCache` K/V + index state (`qwen4_exp/language.py:65-79`); INT8/quantized KV (`qsa_chunk.py:316`) |
| 4d | `o_proj(output * sigmoid(gate))` (`qwen3_5/language.py:963`) | **PORT** |
| 5 | Attention injection: `hyper_input + (branch[...,None,:] * inject_w[...,None])` (`qwen4_exp/language.py:766-767`) | **PORT** (elementwise; injection weights from step 2) |
| 6 | `mlp_hyper_connection = gated_residual` (`:769`) | **PORTED** `gated_residual()` (`engine.hpp:157`) |
| 7a | MoE gate logits → softmax/precise → argpartition → renorm (`qwen3_5_moe/language.py:32-38`; production `adaptive_topk._moe_call` via loader) | **PORTED** `router()` (`engine.hpp:60`, with `renorm_blend` for the production `_RENORM_BLEND`) or `moe_block_from_logits()` (`:85`) |
| 7b | `switch_mlp(x, inds)` Q4 gather + SwiGLU + combine (`qwen3_5_moe/language.py:40-41`) | **PORTED** `q4_projection()`+`swiglu()`+`weighted_combine()` / `moe_block()` (`engine.hpp:67,72,75,81`) + I/O wiring (§4) |
| 7c | Shared expert `shared_expert(x)` + `sigmoid(shared_expert_gate(x))` (`:43-46`) | **PORT** — *not covered by any C++ primitive* (`engine.hpp:77`) |
| 8 | MLP injection: `hyper_input + (branch[...,None,:] * inject_w[...,None])` (`qwen4_exp/language.py:771-772`) | **PORT** (elementwise) |
| 9 | Final `hyper_connection_mixer` with `use_combine=False` (`:784, 829`, returns only `mixed_input` at `:449-450`) | **PORTED-with-adapter**: `gated_residual()` always returns `injection_weights` (`engine.hpp:153-156`); call with `hc_count` etc. and discard that field; confirm no `block_inject_weight` path |
| 10 | `lm_head` / tied `embed_tokens.as_linear` (`qwen4_exp/language.py:841-842`, `prefill.py:562-565`) | **PORT** (matmul) |
| 11 | Sampler greedy (`:1308`, `sampling.py:177-180`) | **PORTED** `sample_greedy()` (`engine.hpp:108`); non-greedy **PORT (WIP)** — `sample_non_greedy` in `sampling.hpp:58-61` (untracked; verify RNG parity, §6) |

Net: the MoE bottleneck and most elementwise/norm chains are done; the remaining
work is attention/recurrence **with state**, the two GatedResidual call sites' glue,
the shared expert, the head, and the pager integration — consistent with
`README.md:66-84` and `STATUS.md:35-48`.

---

## 3. The 48-layer decoder loop in C++

### 3.1 State that must be carried

`Qwen4ExpModel.make_cache()` (`qwen4_exp/language.py:844-851`) creates one cache per
layer: `ArraysCache(size=4 if PLE else 2)` for linear-attention layers and
`QSAKVCache()` for full-attention layers. The C++ loop needs a matching
`std::vector<LayerState>` of length `num_hidden_layers` (48), indexed identically to
`self.layers` (`:780-783`), with `is_linear = layer_types[layer_idx] == "linear_attention"`
(`:728`).

Per layer:

- **Hyper-connection residual (all layers):** the loop's `hidden_states` tensor has
  shape `[B, 1, hc_count*hidden]` after tiling (`qwen4_exp/language.py:807`). Each
  `gated_residual()` call (`engine.hpp:157`) consumes it and returns `hyper_input`
  (the same buffer) + `mixed_input` + `injection_weights`; the layer's branch output
  is injected back (`qwen4_exp/language.py:757-772`). No separate hidden buffer is
  needed: `hidden_states` is threaded through all 48 iterations (`:819-825`) and
  only the final mixer consumes it (`:829`).
- **GDN conv state (linear layers):** `[B, K-1, conv_dim]`, `conv_dim = key_dim*2+value_dim`
  (`qwen3_5/language.py:1065,1129`), `cache[0]`; updated every token to
  `conv_input[:, -n_keep:, :]` (`:1151`). Element dtype = inputs dtype (bf16).
- **GDN recurrent state (linear layers):** `[B, Hv, Dv, Dk]` float32, `cache[1]`
  (`qwen3_5/language.py:1170-1172,1188-1189`); shape must match
  `gated_delta_step_kernel`'s `state` contract (`engine.hpp:135`).
- **GDN PLE state (subset of linear layers):** two extra `ArraysCache` slots when
  `layer_idx+1 in ple_layer_ids` (`qwen4_exp/language.py:734-740, 848`).
- **QSA KV state (full-attention layers):** `QSAKVCache.keys/values` with `offset`
  (`qwen4_exp/language.py:41-42,65-79`), plus `index_keys` and
  `index_position_ids` updated by `update_indexer()` (`:54-63`); INT8/quantized KV
  where enabled (`qsa_chunk.py:316`). The indexer needs the full key history to
  compute pooled block scores (`qwen4_exp/language.py:329-380`).

### 3.2 Allocation and reuse

- Allocate `LayerState` once at engine init (mirroring `make_cache()`), **pre-sized
  to max context** for KV and zero-initialized for conv/recurrent state, so no
  per-token allocation. Track a per-layer `offset` (Python `cache.offset`,
  `qwen4_exp/language.py:992,1007`), advanced after each update
  (`qwen3_5/language.py:1190-1193`).
- Recurrent state is float32 and fixed-shape — reuse the same `mx::array` buffer via
  the `DeltaStepOutput.state` return (`engine.hpp:124-127`); avoid reallocating it
  each token. Hold one scratch `[B,T,Hv,Dv]` output buffer for the fused kernel.
- Because MLX is lazy and `standard_tokens()` forces materialization only at
  `mx.eval(token)` (`flashnext.py:1309`), the C++ driver should likewise `eval` the
  sampled token (and the cache state, cf. `flashnext.py:1302`) at the step boundary,
  not per op — otherwise the latency advantage is lost.
- Expert weights: for a small/resident run, `ExpertStore::load_layer()`
  (`expert_store.hpp:22-24`) can hold all 48 layers' `ExpertWeights` in memory; for
  the real 4-bit checkpoint, use the pager (§4) and keep only slots + a small LRU.

### 3.3 Layer iteration

```
for l in 0..num_hidden_layers-1:            # qwen4_exp/language.py:817
    if has_ple[l]: hidden += ple(...)        # :752-755
    (mixed, hyper, inj) = gated_residual(hidden, ...)   # :757  (engine.hpp:157)
    if is_linear[l]:
        branch = gdn_forward(mixed, conv_state[l], rec_state[l], ...)  # :761
    else:
        branch = qsa_attention(mixed, kv[l], pos_ids, ...)              # :763
    hidden = hyper + (branch[...,None,:] * inj[...,None]).reshape(...)  # :766-767
    (mixed, hyper, inj) = gated_residual(hidden, ...)   # :769
    moe_out = moe_forward(mixed, ...)                    # :770  (§4)
    hidden = hyper + (moe_out[...,None,:] * inj[...,None]).reshape(...) # :771-772
final = gated_residual(hidden, ..., use_combine_false)   # :829
logits = lm_head(final)                                   # sampling.py:1308
```

The branch on `is_linear` mirrors `Qwen4ExpDecoderLayer.__init__`
(`qwen4_exp/language.py:728-732`). `ssm_idx`/`fa_idx` masks
(`qwen4_exp/language.py:785-790, 811-814`) must be prepared once per step and passed
to the right branch (`:818`).

---

## 4. Wiring `expert_store` + `io_pager` into the MoE step

### 4.1 Two loading strategies

1. **Resident / full-layer:** `ExpertStore::load_layer(shard, prefix, group_size, bits)`
   (`expert_store.hpp:22-24`, `.cpp:37-45`) loads gate/up/down for one layer straight
   from safetensors (`mx::load_safetensors`, `.cpp:22`) and reinterprets uint16
   scales/biases as bf16 (`.cpp:9-16`). Feed the returned `ExpertWeights` directly to
   `moe_block()` (`engine.hpp:81`) or `moe_block_from_logits()` (`:85`). This is the
   already-verified path (`STATUS.md:24-26`) and the right first integration.
2. **Streamed / paged:** use the pager for the real checkpoint, where a layer's 512
   experts cannot all be resident on 16 GB.

### 4.2 Concrete streamed MoE flow per layer

1. `router(logits, top_k, _RENORM_BLEND)` (`engine.hpp:60`) → `indices [tokens,k]`,
   `scores`. This决策 is *never* altered by prefetch (bit-exact rule, §4.3).
2. Flatten routed expert ids in route order; call
   `plan_remap(route, expert_slot_pairs, free_slots, fd, header, row_bytes, slot_base, slot_stride)`
   (`io_pager.hpp:36-40`). It returns `PagerPlan{slots (one per route, route order),
   reads (deduplicated misses)}` (`:27-30`), assigning fresh slots from `free_slots`
   only for non-resident experts (`:32-35`). Pure/deterministic — same inputs → same
   plan, no GPU, no I/O.
3. `run_batched_reads(plan.reads, width)` (`io_pager.hpp:45`) issues the misses via
   the GIL-released native batched pread. Only the 9 part rows for each unique
   missing expert are read (mirrors the Python nine-parallel-reads-per-layer design,
   `docs/archive/flashnext/brief-2026-08-30.md:35,70`).
4. Build `Q4Projection` views in **slot order** from the pool's zero-copy buffers
   (`expert_store.hpp:16-19` signature; cf. Python `prefill_ring._views`,
   `prefill_ring.py:471-503`) and call `q4_projection(x, p, route_slots, ...)`
   (`engine.hpp:67`, with `slot_input=true`) for gate/up/down → `swiglu()` (`:72`) →
   `weighted_combine()` (`:75`), or `moe_block(x, route_slots, scores, w)` (`:81`).
5. Add the shared expert (`qwen3_5_moe/language.py:43-46`) — a plain MLP + sigmoid
   gate that still needs porting (§2 step 7c).
6. Inject into the hyper-connection residual (`qwen4_exp/language.py:771-772`).

### 4.3 Streaming/pool behavior and bit-exact slot rules

The authoritative bit-exact gate is in `OPUS_E0_E1_SPEC.md:12-14`: 100% identical
token IDs, identical SHA-256 of raw logits at every step, per-layer residual
checksums for the first 64 steps, plus a chaos mode (random 0–5 ms I/O-thread delays).
A divergence under this gate is a **slot-reuse race, not a result**
(`OPUS_E0_E1_SPEC.md:43-45`). The pipeline design this must satisfy:

- "Refresh all 48 layer tables in ONE sync; slot remap bit-exact (0.000e+00 diff)"
  — single-sync on-device slot resolution (`OPUS_MIRACLE_PROMPT.md:127-129`,
  `OPUS_E0_E1_SPEC.md:77`). So C++ should batch the remap for all 48 layers into one
  host sync/pin, not one per layer.
- The predictor is **hint-only — never committed routing** — so that routing (and
  thus output) is bit-exact by construction (`OPUS_E0_E1_SPEC.md:50-52, 77-78`).
  Prefetch may fill the page cache but must not change `router()` inputs/results.
- Concrete refcount rules the C++ pool must enforce (derived from the above + the
  Python ring contract):
  - A slot is **reserved** when a routed expert id is first assigned in
    `plan_remap` and stays reserved (refcount ≥ 1) until the GPU kernel consuming its
    bytes has completed; release only after the sync/fence (`prefill_ring.py:532-548`
    `_retire` / `finish`, and `OPUS_MIRACLE_PROMPT.md:127-129`). Never free a slot an
    in-flight kernel still views.
  - Slot assignment must be **deterministic and first-seen by route order** so
    `plan.reads`/`plan.slots` are reproducible (`io_pager.hpp:32-35`; Python
    `prefill_ring.py:453-469` sorts by shard position but writes to the original
    routing slot). Chaos delays must not change this order.
  - The current layer's slot set must not be overwritten by the next layer's reads
    until a sync proves the previous layer's compute is done (one ring half suffices
    in the serial scheme, `prefill_ring.py:14-18, 532-542`).
  - Do **not** add O_DIRECT/F_NOCACHE bypass or KICK/WAIT early overlap — explicitly
    excluded as slower (`io_pager.hpp:5-8`, `DO_NOT_IMPLEMENT.md`).

For prefill, the native equivalent of the two-half ring overlap
(`prefill_ring.py:117-124, 373-433`) is a later milestone (§5), not required for the
first decode loop.

---

## 5. Prefill plan (layer-major ring + chunking)

Prefill differs from decode in three ways: it runs `T > 1` tokens per call, it uses
chunking to bound memory, and it can run **layer-major** (all chunks of layer *l*
before layer *l+1*) to overlap the next layer's expert reads with the current
layer's compute. Purity/bit-exactness must be re-established because chunking changes
kernel shapes: "numerically equivalent but NOT bit-identical to the single call"
(`prefill.py:23-25`).

Concrete steps, mirroring the Python reference:

1. **Chunk planning.** Port `plan_chunks()` policies `grid`/`balanced`/`budgeted`
   (`prefill.py:31-46, 176-216`) and chunk caps (`MIN_CHUNK_TOKENS=64`,
   `MAX_CHUNK_TOKENS=2048`, `prefill.py:26-30`). Spans share the cache; non-final
   chunks advance every cache and keep nothing (`_prefill_chunk`, `:219-230`); the
   final chunk projects logits and samples (`_prefill_target`, `:232-356`).
2. **Forward shape.** Prefill uses `return_hidden`/`skip_logits` to avoid materializing
   full-sequence logits for long prompts, with a fused head for the last row
   (`prefill.py:257-345`, `:556-566`). Short prompts keep full logits (`:70-72`).
   The C++ loop must reproduce both branches to match Python numerically.
3. **Layer-major ring.** Port `_run_layer_major()` (`prefill.py:579-704`): windows cut
   at system/message anchors via `layer_major.plan_windows` (`:590`), an
   `ExpertWindow` over `expert_cache._PREFILL_RING[0]` (`:597`), `peel_tail` for a tiny
   trailing chunk (`:609`), frozen cache snapshots at internal boundaries
   (`:639-646`), and `hyper_connection_mixer(states[-1])` before the head (`:691`).
   The C++ analogue leases a two-half ring, issues `preadv`-style batched reads into
   slot buffers (`prefill_ring.py:246-259`), and hands zero-copy views to
   `q4_projection()` (`prefill_ring.py:471-503`).
4. **Overlap + safety.** Issue next-layer reads while the current layer computes
   (`prefill_ring.py:373-433` `stage_begin`/`stage_finish`), but honor the working-set
   check (`:445-451, 589-717`) and release ordering: GPU fence → close ring → free
   scratch pool (`:721-750`). A capacity miss, busy half, or memory pressure falls
   back to the plain path for that layer (`:319-365`).
5. **Bit-exactness gate.** Because chunking is not bit-identical to one call
   (`prefill.py:23-25`), the C++ prefill must be validated against the *chunked Python
   reference with the same plan* (same spans, same head choice), not against a single
   long call. Digest logits and first token under the guard (§7).

---

## 6. Sampling integration

- **Done:** `sample_greedy()` (`engine.hpp:108`) matches production greedy
  (`veystrix/sampling.py:177-180`: `argmax` → `(1,)` `uint32`). The decode loop samples
  `step.logits[:, -1, :]` and calls `mx.eval(token)` (`flashnext.py:1308-1309`).
- **To extend** (the parallel/speculative sampling work already landed in Python):
  `Sampler.distribution()` (`sampling.py:117-146`) applies presence penalty
  (`:127-133`), temperature (`:135`), top-k via `argpartition` (`:137-145`), then
  `_mask()` (`:148-170`) applies min-p (`:155-158`) and top-p nucleus (`:160-169`);
  `__call__` draws with `mx.random.categorical` and maps survivors back to token ids
  (`:182-185`). Settings are `Sampling` (`sampling.py:46-83`) with `THINKING` /
  `INSTRUCT` presets (`:29-43`). The lossless speculative/parallel path is
  `Sampler.verify_point_mass()` (`sampling.py:187-216`), landed with lookup drafting
  (`docs/performance/lookup-drafting-2026-11-15.md:12-20`, commit `8f38af0`).
- **Plan:** port `distribution` + `_mask` + categorical draw into a C++ `Sampler`,
  keeping an ordered `_seen` history for presence penalty and mirroring the exact op
  order (presence → temperature → top-k → softmax → min-p → top-p → categorical).
  An untracked WIP implementation already mirrors this in `sampling.hpp:24-61`
  (`sampling_distribution`, `sampling_mask`, `sample_non_greedy`); validate it with
  seeded draw-for-draw parity before folding it in — do not duplicate it.
  Reuse `sample_greedy()` when `temperature <= 0` (`sampling.py:57-58`). Keep a
  verification entry point analogous to `verify_point_mass()` for later MTP/lookup.
- **RNG parity is a named risk (§8):** `mx.random.categorical` uses MLX's global RNG;
  the C++ port must drive the same RNG with the same seed/state on the same op order,
  otherwise sampled (T>0) trajectories diverge even with identical logits. Greedy is
  the safe first gate.

---

## 7. Bit-exact end-to-end parity test

Use the real checkpoint at
`/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP` (path used in
`README.md:44`) and the existing vector generators as primitives tests
(`cpp/engine/tools/gen_vectors.py`, `gen_real_vectors.py`; `README.md:40-46`).
Raw-byte comparison (uint32 view), not tolerances, is the repo standard
(`README.md:61-64`, `STATUS.md:28-30`).

End-to-end harness design (follow `OPUS_E0_E1_SPEC.md:11-19, 98-108`):

1. **Reference:** run the Python decode with the exact same prompt, prefill plan,
   cache config, and sampling settings; at every step record the sampled token id and
   the SHA-256 of the raw final logits row (float32 little-endian, as in
   `logit_reference.py`). Gate = 100% token-id match + identical logit SHA-256 every
   step, plus per-layer residual checksums for the first 64 steps
   (`OPUS_E0_E1_SPEC.md:12-14, 105-107`).
2. **C++ arm:** drive the native engine as a library (no HTTP/Swift needed —
   `ENGINE_LANGUAGE_ANALYSIS.md:25-38, 49`), emitting the same digest stream. Compare
   token-by-token; on the first mismatch, bisect by layer using the residual
   checksums to identify which primitive/state diverged.
3. **Guard:** wrap every run with `scripts/flashnext/real_run_guard.py`
   (`OPUS_E0_E1_SPEC.md:16-19`):
   ```
   DIR=$(.venv/bin/python -c 'from veystrix.results import new_run_directory; print(new_run_directory("flashnext","cpp-engine-parity"))')
   scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 2048 \
     --compressor-growth-mib 4096 --log "$DIR/guard.jsonl" -- <cpp-parity-runner>
   ```
   The guard samples memory/swap/compressor/thermal/power out of process and stops
   the command group on violations (`real_run_guard.py:1-20, 126-223`).
4. **Chaos mode:** inject random 0–5 ms I/O-thread delays (as in the E0/E1 harness,
   `OPUS_E0_E1_SPEC.md:13-14`) and require the gate to pass in both arms — a
   divergence indicates a slot-reuse race (`:43-45`), not a result.
5. **Prefill gate:** separately compare C++ prefill against Python with the *same*
   chunk plan (§5), since chunking is not bit-identical to a single call
   (`prefill.py:23-25`). Preserve all `results/**` outputs (`OPUS_E0_E1_SPEC.md:19`).

---

## 8. Honest milestone sequence, effort, and risks

Scope reference: ~8,900 lines of hot path (`ENGINE_LANGUAGE_ANALYSIS.md:39-54`), a
full Qwen port estimated 6–10 weeks for one agent (`:70-83`), with ~30% extra for the
bit-exactness tax (`:80-83`). The C++ primitives already done remove part of the MoE
work, but attention/recurrence + loop + prefill remain. Sequence:

| M | Scope | Rough effort | Gate |
|---|---|---|---|
| M0 | Current primitives (§1) | **done** | 5/5 ctests (`README.md:50-59`) |
| M1 | Wire `expert_store.load_layer` + `io_pager` into `moe_block`; shared expert; single-sync slot remap (§2, §4) | 1–2 wk | per-step MoE logit SHA on real layer |
| M2 | GDN linear-attention: conv state + recurrence via `gated_delta_step_kernel`, gated norm, out_proj (§2 3b-3f). Fold in/verify WIP `gdn_conv` (`gdn_conv.h:19-32`) | 1–2 wk | per-layer state + logit parity |
| M3 | QSA attention: indexer, MRoPE, sparse block selection, KV cache, o_proj (§2 4a-4d). Fold in/verify WIP pre-RoPE `qsa_indexer_pre_rope` (`qsa.hpp:91-96`); RoPE/selection remain | 2–3 wk | **hardest**; attention logit parity |
| M4 | 48-layer loop + hyper-connections + final mixer + lm_head + greedy sampler (§3, §6) | 1 wk | full greedy decode token parity |
| M5 | Prefill: chunking + layer-major ring + overlap (§5) | 2–3 wk | same-plan prefill digest parity |
| M6 | Sampling beyond greedy + RNG parity (§6) — verify/adopt WIP `sampling.hpp:24-61` | 1 wk | seeded sampled-token parity |
| M7 | End-to-end guarded parity harness + optional server bridge (§7) | 1–2 wk | full gate under guard + chaos |

Total ≈ 9–13 weeks for one agent, consistent with the 6–10 week full-Qwen estimate
plus the 30% bit-exactness tax (`ENGINE_LANGUAGE_ANALYSIS.md:76, 80-83`).

**Specific blockers / risks**

- **Stateful kernels (M2/M3).** GDN recurrence must match the fused Metal kernel's
  reduction order; `gated_delta_step_kernel` uses the exact `gdn_kernel.h`
  (`simd_sum`) and only falls back to the ops path if Metal is unavailable
  (`engine.hpp:132-139`) — a fallback would break bit-exactness. Conv state dtype and
  the `conv_state[:, -n_keep:, :]` update (`qwen3_5/language.py:1151`) must be exact.
  The installed `gated_delta_kernel` already runs the recurrence as one Metal kernel
  per call, so the risk is *faithfully re-driving it*, not writing a Python-style
  per-token loop (`docs/performance/inference-blueprint-handoff.md:29`).
- **QSA block selection (M3).** `argpartition` tie-breaking, pooled-key
  normalization, MRoPE positions, the `valid_blocks`/`complete_counts` logic
  (`qwen4_exp/language.py:329-380`), INT8/quantized KV (`qsa_chunk.py:316`), and the
  gather-tile kernel mode (`qsa_chunk.py:141-155`) all affect which tokens are read.
  Any mismatch compounds over 48 layers and long contexts.
- **RNG parity (M6).** `mx.random.categorical` global RNG state and op order
  (`sampling.py:183, 210, 214`); a separate seeded RNG or different softmax/categorical
  reduction diverges sampled trajectories even at identical logits. Seed 42 is a known
  sensitivity (`docs/flashnext/research.md:119`). Mitigate by gating greedy first.
- **Shared expert + mixer adapter (M1/M4).** Neither is covered by existing primitives
  (`engine.hpp:77`; `:153-156` vs `qwen4_exp/language.py:449-450`).
- **Bit-exactness tax (~30%)** — subtle, recurring kernel-shape and state traps are
  expected (`ENGINE_LANGUAGE_ANALYSIS.md:80-83`); budget for them.
- **Physical limits / expectations.** C++ alone does not remove the 16 GB SSD I/O wall;
  the recommendation is a future-model / reuse investment, not a 16 GB speed fix
  (`ENGINE_LANGUAGE_ANALYSIS.md:85-94, 135-138`). Excluded optimizations
  (`io_pager.hpp:5-8`) must stay excluded.

**Do not** switch branches, commit, or modify code to execute this plan — it is a
plan only. Keep all real-run evidence under `results/**` and every run behind
`real_run_guard.py` (`OPUS_E0_E1_SPEC.md:16-19`).
