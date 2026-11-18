# Nemotron INT8 KV attention (production default) — C++ port

Status: **implemented, bit-exact on synthetic vectors and on the real
checkpoint.** Measured results are labeled `[measured]`; design facts are
labeled `[cited]` from `models/nemotron35/attention.py` and
`mlx_lm/models/cache.py`; anything not validated is labeled `[estimate]` or
`[limitation]`.

## What changed

* `cpp/engine/src/nemo_attention.hpp/.cpp`
  * Added `NemoInt8Cache` / `NemoInt8Block`, the C++ mirror of
    `AttentionCache` + `mlx_lm.models.cache.QuantizedKVCache` state.
  * Added `nemo_attention_int8_forward()` alongside the untouched bf16
    `nemo_attention_forward()`.
* `cpp/engine/tools/nemotron_generate.cpp`
  * Default KV mode is now INT8. `VEYSTRIX_NEMOTRON_KV=bf16` opts into the
    legacy bf16 KV path; unset / `int8` / any other value uses INT8.
* `cpp/engine/tools/gen_nemo_int8_vectors.py`, `cpp/engine/tests/test_nemo_int8.cpp`,
  `cpp/engine/CMakeLists.txt` — golden vectors, byte-for-byte test, ctest target
  `nemo_int8`.

## Reference behavior implemented `[cited]`

`attention.py` constants: `FORMAT=affine-int8-g64`, `GROUP_SIZE=64`,
`BLOCK_TOKENS=4096`, `QUERY_TOKENS=128`.

* q/k/v/o projections are unchanged from the bf16 path (no RoPE, no q/k norm).
* `AttentionCache.append` quantizes each appended K/V slice with
  `mx.quantize(group_size=64, bits=8, "affine")` and stores:
  * `packed` uint32 `[B, Hkv, n, D/4]`,
  * `scales` bf16 `[B, Hkv, n, D/64]`,
  * `biases` bf16 `[B, Hkv, n, D/64]`,
  per block (`n <= BLOCK_TOKENS`). The C++ cache keeps the exact same arrays;
  each decode step concatenates the newly quantized token along axis 2.
* `attend()`:
  1. **Single block** (decode and short prompts, `< 4096` tokens):
     `mx.dequantize(K), mx.dequantize(V)` → `mx.fast.scaled_dot_product_attention`
     with `mask="causal"` for S>1, `mask=None` for S=1.
  2. **Multi-block**: per 128-query chunk, `q.f32 * scale` reshaped to
     `[B, Hkv, Hq/Hkv, Q, D]`; for each block insert a size-1 kv-head axis and
     run `mx.quantized_matmul` for scores and for the weighted value sum, with
     per-block causal masking, running max, exp/correction accumulation, and a
     final `astype(q.dtype)`.

A third production variant (`attend()` lines 106-110, dequant + fused SDPA over
all blocks for multi-block S>1 when `VEYSTRIX_NEMOTRON_ATTENTION_PREFILL_SDPA=1`)
is intentionally not implemented here; decode always uses (1) or (2).

## Validation

### Synthetic golden vectors `[measured]`

Generator uses the **real** `StreamedAttention` + `AttentionCache` classes from
the repo venv (no checkpoint), tiny config H=4, Hkv=2, D=128 (two affine groups),
synthetic bf16 weights:

* `decode`: 4 sequential S=1 steps (one block).
* `prompt`: one S=6 causal chunk + 2 S=1 steps (one block).
* `multiblock`: `BLOCK_TOKENS` monkeypatched to 4 **in the imported real module**
  (only the constant differs from production), one S=5 chunk (causal masking
  across two blocks, `PREFILL_SDPA=0`) + 3 S=1 steps; two blocks, 12 state arrays.

`test_nemo_int8` compares output, q/k/v, the pre-o_proj `attended` tensor, and
**every** packed/scales/biases array of every block, by raw uint32 bits:

```
$ cd cpp/engine/build && ctest -R nemo --output-on-failure
100% tests passed out of 2   # nemo_attention (bf16) + nemo_int8
$ ./test_nemo_int8 ../tests/vectors_nemo_int8
nemo_attention_int8: BIT-EXACT PASS (decode + prompt single-block + multiblock loop cases;
                                     output + q/k/v + attended + packed cache state)
```

### Real-checkpoint parity `[measured]`

Guard: `real_run_guard.py --floor-gib 1.5 --swap-growth-mib 5120
--compressor-growth-mib 8192 --thermal-level 2 --max-seconds 1200`, cwd
`/Users/vincent/Documents/Veystrix`; `sysctl vm.swapusage` checked before each
run (~3.6 GiB used by unrelated WebKit processes; **zero swap growth**, guard
`stopped: null` on every run, exit 0).

The stock `gen_nemotron_reference.py` pins `VEYSTRIX_NEMOTRON_KV=bf16` and
`int8_kv=False` internally. The INT8 reference was captured with a `/tmp`
variant that changes exactly those two lines to `int8` / `True` (same seed,
6-token prompt, 4 tokens, zero-primed Mamba caches, `chunk_tokens=1`):
`/tmp/gen_nemotron_ref_int8.py` → `/tmp/nemo_ref_int8.json`. Flat config built
with `gen_nemotron_config.py --reference /tmp/nemo_ref_int8.json`.

C++ tool run with `VEYSTRIX_NEMOTRON_KV` unset (exercises the new default):

| step | Python INT8 logit SHA-256 (first 12) | C++ (first 12) | match |
|---|---|---|---|
| 0 | `604002f8f201` | `604002f8f201` | yes |
| 1 | `36c59423f698` | `36c59423f698` | yes |
| 2 | `0213fa02c028` | `0213fa02c028` | yes |
| 3 | `47ebc57a133e` | `47ebc57a133e` | yes |

All 4/4 logit digests bit-identical. Locally selected tokens `1010, 1044, 1032,
1050` match the bf16 reference's tokens but the logits differ, as expected from
the different KV numerics. The `VEYSTRIX_NEMOTRON_KV=bf16` opt-in was re-run
against the original `/tmp/nemo_ref.json`: 4/4 digests still match, so the env
switch is bidirectional.

### Limitations / scope

* `[limitation]` The real-checkpoint parity used a 6-token prompt, so only the
  **single-block** path ran on the real model. A production multi-block
  (>4096-token) real-checkpoint run was not attempted: prefilling >4096 tokens
  token-by-token is expensive, and the guard budget was better spent on the
  bit-exact synthetic multi-block case (which runs the identical loop code with
  `BLOCK_TOKENS=4`).
* `[limitation]` The `PREFILL_SDPA=1` multi-block prefill variant (dequant +
  fused SDPA, not bit-identical to the loop by design — see attention.py:23-33)
  is not ported. Decode and short prompts are fully covered.
* `[measured]` `NemoInt8Cache::block_tokens` defaults to 4096 (production). The
  multiblock test lowers it to 4 to make the loop cheap; no production code
  path changes this value.

## Reproduction

```bash
# golden vectors (real Python module, repo venv)
VEYSTRIX_ROOT=/Users/vincent/Documents/Veystrix \
  /Users/vincent/Documents/Veystrix/.venv/bin/python \
  cpp/engine/tools/gen_nemo_int8_vectors.py cpp/engine/tests/vectors_nemo_int8

# synthetic bit-exact test
cd cpp/engine/build && cmake --build . -j4 && ctest -R nemo --output-on-failure
```
