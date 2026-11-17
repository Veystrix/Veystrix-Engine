# C++ parity-gap audit — `real_checkpoint_generate.cpp` header

**Date:** 2026-11-18
**Scope:** the six known divergences listed in
`cpp/engine/tools/real_checkpoint_generate.cpp:23-31` against the production
values that the reference capture actually used. "Gate (b)" is the full
48-layer real-checkpoint cross-check defined in
`docs/performance/OPUS_PROMPT_CPP_MEMORY_GATE.md:24` (exact token IDs + raw
logit SHA-256 vs `results/flashnext/20261117-222244-real-reference/reference.json`).
The C++ side currently fails that gate 0/8 (`CPP_ENGINE_REAL_CROSSCHECK.md:12-14`).

Production values below come from the **defaults** (no opt-in mode flags),
because `capture_real_reference.py` constructs `FlashNextBackend(model_path=...)`
with no mode overrides. Where an INT4/INT8/feature is opt-in only, that is
stated explicitly — an opt-in mode is *not* the production default the captured
reference used.

## Table

| Gap (tool header line) | Production value (file:line) | C++ value (header:23-31) | Changes step-0 digest? | Required for gate (b)? | Smallest synthetic ctest |
|---|---|---|---|---|---|
| Sequential single-token prefill vs chunked/layer-major (`:23-25`) | `FLASHNEXT_PREFILL_CHUNK_TOKENS` default `0` = one whole-prompt call (`models/flashnext/prefill.py:26,82-93`); chat/agentic profiles opt in to 128/2048 (`models/flashnext/prefill_16gb.py:6`, `models/flashnext/settings/modes.py`), layer-major ring (`prefill_ring.py`). | Sequential S=1 steps; `run_decoder_step` MoE reshape is `[1,H]`-only (`:24-25`). | **yes** — step-0 logits come from prefill; different chunk shapes change reduction order. | **yes** — gate needs prefill logits to match. | `prefill` (`test_prefill`, `tests/vectors_prefill`) — full-sequence S>1, *not* chunked; no chunked ctest exists. |
| Dense KV vs INT8 (`:26-28`) | KV dtype default **`bf16`** (`models/flashnext/quantized_kv.py:19,33-39`; `settings/setting_research.py:249-251`; `settings/modes.py:98`). INT4/INT8/tiered KV is opt-in (`:29`, `modes.py:131-171`). Indexer returns `None` for <2048 past tokens → dense causal attention (`:26-27`). | Dense KV, dense causal fallback for indexer-`None` (`:27-28`); no quantized KV. | **unknown** — reference used bf16, so C++ dense KV is representationally close; INT8 would *diverge* from the captured reference. | **no** (for the default-bf16 reference); INT8/q-KV is opt-in. | `parity_gaps` (`test_parity_gaps`, indexer-None dense fallback); no INT8-KV ctest. |
| PLE at layer 2 missing (`:29-30`) | PLE applied when `layer_idx+1 in ple_layer_ids` (`qwen4_exp/language.py:734-740,752-755`; cache gets 2 extra slots at `:848`); "layer 2" = 0-based layer index 1 (`CPP_ENGINE_REAL_CROSSCHECK.md:24`). | PLE not reproduced (`:29-30`). | **yes** — PLE adds a residual at layer index 1, shifting all later logits. | **yes** — a missing layer cannot match. | **none** — no ctest currently covers PLE (needs a new synthetic case). |
| Eager RoPE (`:29`) | Fused Metal `mx.fast.rope` via the upstream attention/indexer `_apply_rope` (`models/flashnext/qsa_chunk.py:352,402`; `qwen4_verifier.py:61,76`). | Eager explicit cos/sin RoPE; fused-Metal path not reproduced (`:29`). | **yes** — fused vs eager differ in rounding order. | **yes** — bit-exact gate requires the same kernel. | `rope_fused` (`test_rope_fused`, `tests/vectors_rope_fused`). |
| Dequantized dense matmul / shared expert (`:29-30`) | Dense projections stay packed and use `nn.quantize` + `mx.quantized_matmul` (`models/flashnext/loader.py:209-236`; `mlx.nn.QuantizedLinear.__call__`); shared expert gate/up/down are quantized (`CPP_ENGINE_INTEGRATION_PLAN.md:154`, `gen_parity_gaps_vectors.py:4-11`). | `dequantize` + `mx::matmul` on bf16 dense weights; dequantized shared expert (`:29-30`; `OPUS_PROMPT_CPP_MEMORY_GATE.md:41-46`). | **yes** — probe shows thousands of differing float32 elements (`tools/probe_quantized.cpp`). | **yes** — the largest documented divergence source. | `engine_parity` (dequantized dense GEMM); `parity_gaps` (quantized shared expert). |
| No prefix cache / MTP (`:30-31`) | Prefix cache opt-in `FLASHNEXT_PREFIX_CACHE=0` default (`models/flashnext/prefix_session.py:29,42-43`); native MTP `FLASHNEXT_NATIVE_MTP=off` default (`veystrix/backends/flashnext.py:62-79,163`), greedy-only (`:211`, `sampling.py:61-62`). | Not reproduced (`:30-31`). | **no** for the captured fresh greedy prompt (both off by default). | **no** for gate (b) as captured; would be required for a prefix/MTP-arm gate. | **none** — no synthetic ctest for prefix cache / MTP. |

## Notes on "Changes step-0 digest?"

- The C++ engine currently produces a *different* step-0 token on every run
  (`CPP_ENGINE_REAL_CROSSCHECK.md:15-18`), so "changes the digest" is answered
  as the *expected* effect of closing the gap, not a measured delta.
- `quantized_matmul` vs `dequantize`+`matmul` is demonstrably non-bit-equal:
  the synthetic probe (`cpp/engine/tools/probe_quantized.cpp`) reports, for
  `[2560,2560]` Q4 affine with M=1, gs=32/64, 2233/2236 of 2560 outputs
  differing; with M=16, 38982/39135 of 40960 differing.
- The KV row is the subtle one: the captured reference used **bf16** KV, not
  INT8. The header's "dense KV vs INT8" label describes a divergence that only
  exists when a quantized-KV mode is enabled; under the reference defaults the
  C++ dense KV is already the closest match.
