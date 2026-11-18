# MTP sparse indexer mask for long contexts

**Status (measured):** the C++ MTP draft head now calls the shared
`qsa_full_indexer()` primitive to build the selected-token mask for long
contexts and applies **masked SDPA** instead of dense attention. Head-level
parity against the checkpoint's own `Qwen4ExpMTPModule` is **bit-exact** on a
synthetic tiny config (0 / 32 per-head SDPA elements, 0 / 25 mask, 0 / 64 logits)
and on the **real checkpoint at `past_len = 2052`** (0 / 6144 per-head SDPA
elements, 0 / 2053 mask, 0 / 248320 logits), with identical SHA-256 on every
tensor. The short-context dense path is unchanged: synthetic position 0
(0 / 64), and real positions 7 and 11 (0 / 248320 each). No commit was made.

All results below are labelled **measured**, **cited**, or **honest limitation**.

## Scope of edits (only these files in this worktree)

| File | Change |
|---|---|
| `cpp/engine/src/mtp.hpp` | `MTPConfig::indexer_kv_heads` and `sparse_threshold` (default 2048); optional QSA indexer weights on `MTPWeights` / `MTPRealWeights` with `set_indexer_weights()` / `has_indexer()`; `MTPDraftCache::indexer` (a `QSACache` carrying raw index keys + positions) and `has_indexer_state()`; `MTPSparseDebug` (mask + per-head SDPA capture); sparse-capable overloads of `mtp_draft_forward` / `mtp_draft_forward_real`. |
| `cpp/engine/src/mtp.cpp` | Sparse helpers (`mtp_sparse_active`, `mtp_qsa_config`, `mtp_indexer_positions`, `mtp_sparse_mask`, `mtp_masked_sdpa`, `one_centered_to_qsa_gain`); `sparse_attention` / `sparse_attention_real` (copy the validated dense projection/RoPE path, then apply the QSA mask); overloads wired so the dense path is byte-identical. |
| `cpp/engine/tools/check_mtp_sparse_parity.py` | New: synthetic patched-threshold golden + real-checkpoint injected-cache golden; byte-exact compare of per-head SDPA output, QSA mask and logits. C++ drivers are generated into the output directory, never committed. |
| `docs/engineering/mtp_sparse_indexer.md` | This document. |

`qsa_full.cpp`, `qsa.cpp`, `decoder_full.*`, `real_checkpoint_generate.cpp`,
`engine.cpp` and every other shared primitive were **not** modified. No new
CMake target was needed (`cpp/engine/CMakeLists.txt` unchanged).

## What changed and why it is correct

### Detection and fallback (measured / cited)

* The QSA indexer itself decides `None` vs mask: `max_complete_blocks =
  key_len / compress_ratio <= block_topk` → `None` (`cited`:
  `mlx_vlm/.../qwen4_exp/language.py::Qwen4ExpQSAIndexer.__call__`). With the
  real config (`indexer_budget=2048`, `compress_ratio=4` → `block_topk=512`) the
  mask first appears at `key_len = 2053` (`past_len = 2052`, S=1). The task's
  "past_len >= 2048" is the coarse `QSA_CHUNK_THRESHOLD` prefill constant; the
  exact indexer condition is used here so C++ never diverges at the boundary.
* `mtp_sparse_active()` only gates **calling** the indexer; when the indexer
  returns `None` the code falls back to the untouched dense path, exactly like
  Python (`Qwen4ExpAttention.__call__`: `if qsa_mask is not None: ...`).
  `VEYSTRIX_MTP_FORCE_SPARSE=1` forces the attempt (test hook).
* Sparse is only taken when the caller supplies both indexer weights and
  `MTPDraftCache::indexer` state. A short cache with only past K/V keeps the
  validated dense path, which is why positions 7/11 remain byte-identical.

### Reuse of existing primitives (measured)

* `mtp_sparse_mask()` calls `qsa_full_indexer()` verbatim (scoring, argpartition
  block selection, token expansion). No indexer math was reimplemented.
* `mtp_masked_sdpa()` mirrors `full_attention_forward()` in `decoder_full.cpp`:
  truncate the mask to `kv_seq_len = S + past_len + 1`, then the same
  `mx::fast::scaled_dot_product_attention(q, k, v, scale, "", mask, nullopt)`.
* The indexer rotary dim is `int(head_dim * partial_rotary_factor)` (the
  attention's rotary instance, shared by Python's indexer), and text position
  ids are 2-D `[B, S] = past_len + j` (`_default_position_ids`); both are
  reproduced by `mtp_qsa_config()` / `mtp_indexer_positions()`.

### Real-checkpoint norm convention (measured)

The production loader applies `patch_rmsnorm` to **every**
`Qwen4ExpRMSNorm`, including the indexer's `q_layernorm` / `k_layernorm`
(`cited`: `models/flashnext/loader.py::load_streaming` calls
`configure_rmsnorm(model, one_centered=...)` after attaching the MTP module;
`patch_rmsnorm.py` measures `q/k_layernorm` mean +0.9306, min +0.500). But
`qsa_full_indexer()`/`qsa_rms_norm()` compute gain = `1 + weight`.
`one_centered_to_qsa_gain()` passes `f32(w) - 1` so `1 + (f32(w)-1) == f32(w)`
exactly for the measured gain range `[0.5, ~1.4]`. The real parity run below
confirms the resulting mask is byte-identical (0 / 2053).

The real indexer weights **are present** in the MTP sidecar (shard 22):
`language_model.mtp.layers.0.self_attn.indexer.index_qk_proj.{weight,scales,biases}`,
`.q_layernorm.weight`, `.k_layernorm.weight`. No missing-weight blocker.

## Validation (all **measured** unless labelled otherwise)

Every run compared raw f32 bytes (uint32 view) — no tolerance. SHA-256 is over
the exact raw file bytes; diff counts are differing 4-byte elements.

### 1. Synthetic tiny config, sparse path (no checkpoint)

`check_mtp_sparse_parity.py` drives the real `Qwen4ExpMTPModule` + real
`Qwen4ExpQSAIndexer` with the tiny config from `gen_mtp_vectors.py`, fixed seeds
and a deterministically re-seeded weight set. The synthetic indexer's
`block_topk = 8 // 4 = 2` already selects blocks at short contexts, so warming
a QSA cache to `past_len = 24` (`key_len = 25`, 6 complete blocks > 2) activates
the sparse path without patching any production constant; the C++ config uses
`sparse_threshold = 1` (opt-in). RoPE is forced to the deterministic eager
branch in both.

| Tensor | Elements | Python SHA-256 | C++ SHA-256 | Diffs |
|---|---|---|---|---|
| per-head SDPA `[1,4,1,8]` | 32 | `c8c6d31f7404482ac1743d305e270e224759bcf65ad8b8cd6dcc972ff96a7202` | identical | 0 / 32 |
| QSA mask `[1,1,1,25]` | 25 | `d930a7861a219aaf5c846b7dc658118560e153ee1909acef54c80fc496b64b61` | identical | 0 / 25 |
| logits `[1,1,64]` | 64 | `15829b021967c8c9ded0ecf5e4cbd65159bc5edec30bfa258fb5553fa290604f` | identical | 0 / 64 |

Result: **BIT-EXACT PASS**. Reproduced deterministically across processes and
output directories.

### 2. Real checkpoint, sparse path at `past_len = 2052` (injected cache)

Same tool with `--real --checkpoint ...`. To avoid the prohibitive 2050-token
backbone prefill, a **synthetic prepopulated MTP QSA cache** (random already-
RoPE'd K/V and random raw indexer keys in the real bf16 shapes/dtypes) is
injected into the real module's `QSAKVCache`; one real draft step then runs with
the real checkpoint weights (including the quantized `index_qk_proj`, the real
one-centered indexer norms, the real masked SDPA, the 512 streamed experts and
the untied LM head). `past_len = 2052` → `key_len = 2053`, 513 blocks > 512.

| Tensor | Elements | Python SHA-256 | C++ SHA-256 | Diffs |
|---|---|---|---|---|
| per-head SDPA `[1,24,1,256]` | 6144 | `54782dc4b2c9aa1a3513b1b0434007f0d5ede5835266b28aa2d8fc2b8340104d` | identical | 0 / 6144 |
| QSA mask `[1,1,1,2053]` | 2053 | `011cc2e8c17448c6d0127cf2d50874ff1221243d2ac997b8b1964fdc9786830e` | identical | 0 / 2053 |
| logits `[1,1,248320]` | 248320 | `770afcbe95c880a046538f6f42d95284361f7022c53e7ac651ea8877a1547b13` | identical | 0 / 248320 |

Result: **BIT-EXACT PASS**.

### 3. Short-context regressions (unchanged dense path)

* Synthetic position 0 (`test_mtp.cpp` on `vectors_mtp`): `fuse_inputs` and
  draft logits both 0 / 64 differing f32 elements (256 bytes bit-identical);
  all 11 point-mass verify cases pass.
* Real position 7 (`check_mtp_full_parity.py --prompt-tokens 8`): 0 / 248320
  differing elements; SHA-256 `a9152fa1374be61d26caaf1202c8636272435c2a1336b0f9950dd9a65bf86df3`
  on both sides; argmax 220.
* Real position 11 (`--prompt-tokens 12`): 0 / 248320; SHA-256
  `fc9cceffa6cd9a51653fd5a860b0524771cbdbeabd29f02306030f6246daf094`;
  argmax 510.

### Guard / resource observations (measured)

From `/Users/vincent/Documents/Veystrix`, every real-checkpoint run used:

```
python3 scripts/flashnext/real_run_guard.py --floor-gib 1.5 \
  --swap-growth-mib 5120 --compressor-growth-mib 8192 --thermal-level 2 \
  --max-seconds 900 --log <log> -- .venv/bin/python <tool> ...
```

Guard `stopped` was `null` on every run (returncode 0), swap stayed at
2849–2897 MiB (no growth), thermal level 0 throughout, peak active memory
~2.27 GiB. A 30 s cool-down and swap re-check separated each real run.

## Honest limitations

* **The real sparse run used an injected (synthetic) prepopulated cache, not a
  genuine 2050-token prompt prefill** (`measured` limitation). The sparse
  *computation* is validated with real weights at the real long-context geometry;
  what is **not** measured end-to-end is the backbone prefill that would produce
  those past K/V and index keys from a real prompt. **Estimate:** a 2051-token
  prefill is ~170x the token work of the 12-token regression (~29 s each), i.e.
  on the order of tens of minutes to over an hour, with KV state across 48 layers
  and 512 experts — excessive under the guard's memory/time budget, so it was
  deliberately not attempted. The synthetic validation plus the real injected-
  cache validation together cover every stage of the sparse draft step.
* **RoPE branch:** both parity tools force the eager `mx.cos`/`mx.sin` branch
  (`fused_apply=False`), matching the C++ port. Production's fused Metal kernel
  differs by ~1 f32 ULP at nonzero positions (`cited`:
  `research_notes/.../qsa_full_port.md`); the position-7/11 regression base
  follows the same established convention.
* **Multi-step state write-back:** the non-const sparse overload updates
  `MTPDraftCache::indexer` in place so callers can chain steps. The parity tools
  exercise a single step; multi-step cache continuity was not separately
  measured.
* The `one_centered_to_qsa_gain` conversion is exact for the measured
  `[0.5, ~1.4]` indexer gain range; a hypothetical checkpoint with gains outside
  `[0.5, 2]` would need a one-centered variant of `qsa_rms_norm` (out of scope;
  `qsa_full.cpp` was not modified).
