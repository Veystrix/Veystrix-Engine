# KV-tier + MTP integration into the real checkpoint decode loop (phase 1)

**Status:** implemented and measured, with one **bit-exact** claim (primitive
round-trip on GPU) and two deliberately **non-bit-exact** reports
(generator-level KV mode comparison and MTP vs Python native MTP). Every number
below is labelled **measured**, **cited**, or **estimate**. The opt-in features
are OFF by default; with neither env var set, the generator's gate-(b) output
is unchanged (**measured** step-line identity, below).

Scope of edits (only these were changed in this worktree):
`cpp/engine/tools/real_checkpoint_generate.cpp` (integration + a selftest entry
point), new `cpp/engine/tools/check_kv_tier_parity.py`, new
`cpp/engine/tools/check_mtp_real.py`, and this document. The read-only
primitives `kv_quant.{hpp,cpp}` and `mtp.{hpp,cpp}`, `engine.cpp`, and all other
shared files were NOT modified. No new binary target was added, so
`cpp/engine/CMakeLists.txt` needed no change. No commit was made.

## What is wired

### KV tier (`VEYSTRIX_CPP_KV_TIER=tiered4|affine4|affine8`)

* `cpp/engine/src/kv_quant.hpp` primitives are called, not reimplemented:
  `kv_quantize` / `kv_dequantize` (affine4/affine8) and `Tiered4KV` (tiered4).
* `full_attention_dense()` now routes every full-attention KV write through a
  per-layer `KVCacheWriter`. When the mode is unset, `KVCacheWriter::append`
  performs the original raw `mx::concatenate`, byte-for-byte.
* The default path uses batched prefill (`run_prefill_batched` in a shared file
  that may not be edited). After prefill, `KVCacheWriter::prime_full` applies
  the quantize/dequantize round-trip to the finished cache (or primes the
  `Tiered4KV` state for tiered4), so subsequent `run_step` decode writes
  continue the same representation. Sequential prefill (`VEYSTRIX_CPP_SEQ_PREFILL`)
  is wired directly through `run_step`.
* Optional knobs: `VEYSTRIX_CPP_KV_GROUP_SIZE` (32/64, default 64),
  `VEYSTRIX_CPP_KV_RECENT_TOKENS` (multiple of 256, default 8192). These mirror
  the Python `FLASHNEXT_KV_GROUP_SIZE` / `FLASHNEXT_KV_RECENT_TOKENS` defaults
  (**cited**: `docs/engineering/kv_quant_port.md`, `models/flashnext/mixed_kv.py`).

### MTP (`VEYSTRIX_CPP_MTP=1`)

* `run_step` optionally exposes its last `[B,S,HC*H]` hidden stream; the decode
  loop calls `mtp_draft_forward` to propose a draft and `mtp_verify` to
  accept/reject using the session-seeded PRNG (`VEYSTRIX_CPP_MTP_SEED`, default
  0) and a near-argmax greedy `SamplingParams`. Optional depth
  `VEYSTRIX_CPP_MTP_DEPTH` (1..4, default 1).
* The verify loop is lossless by construction: an accepted draft is consumed
  through the real target (`run_step`, advancing the KV cache) and supplies the
  row for the next draft; a rejected draft emits `mtp_verify`'s redraw from the
  current target row **without** consuming a target step. No cache rollback is
  needed.
* **What is NOT wired (honest):** the draft head is a small deterministic
  *synthetic* `MTPWeights` set (H=32, HC=2, 1 expert, vocab=256), not the
  checkpoint's MTP weights. The real hidden stream `[1,1,4*2560]` is sliced to
  the synthetic `[1,1,2*32]` (a shape adaptation, not a learned projection).
  Real token ids are folded modulo 256 for the draft embedding. The primitive's
  validated path is position-0, fresh-cache, dense 1-token attention; real
  draft positions, the sparse indexer branch, the untied LM head
  (`tie_word_embeddings=False` in this checkpoint), and the 512 quantized
  routed MTP experts are NOT loaded. Loading those into `MTPWeights` would need
  ~5 GiB of bf16 expert arrays (**estimate**: 512×640×2560×3×2 B) plus a
  full embed/LM-head matrix, and would still not be bit-faithful to the
  primitive's tied-head / dense-MoE contract. Therefore **full real-MTP
  (token-level) parity is not claimed**; only losslessness is.

## Device investigation — why the bit-exact claim is GPU↔GPU only

**Cited** (`docs/engineering/kv_quant_port.md`, "Investigation" section): with
the same mlx 0.32.2 build, CPU vs GPU `mx.quantize` differed in 1/128 packed
words and 0/16 scales/biases; CPU vs GPU `mx.dequantize` differed in 569/1024
outputs. The C++ primitive is the same op as Python; the original `test_kv_quant`
bit-exactness was obtained only after forcing the CPU device to match its
CPU-generated golden vectors. The real generator runs attention on the **GPU**.
Consequently:

* We did **not** reuse the CPU golden vectors for an end-to-end claim.
* `cpp/engine/tools/check_kv_tier_parity.py` generates fresh golden vectors on
  the **GPU** with the real Python ops (`mx.quantize`/`mx.dequantize` and
  `models.flashnext.mixed_kv.MixedPrecisionQSAKVCache`), and drives a new
  `VEYSTRIX_CPP_KV_TIER_SELFTEST=<dir>` entry point in the C++ binary that
  runs the SAME integration functions on the GPU.
* **No cross-device bit-exactness is claimed.** The claim is limited to the
  GPU C++ integration path vs GPU Python reference on the same fixed input.

### Bit-exact test (measured, exact output)

Command (run from `/Users/vincent/Documents/Veystrix`, no checkpoint, no guard
needed because no real checkpoint is loaded):

```
.venv/bin/python /Users/vincent/.local/share/opencode/worktree/dd9a0e/ms-phase1-engine/cpp/engine/tools/check_kv_tier_parity.py \
    --out /tmp/kvparity_p1 --binary /tmp/real_gen_p1
```

Output:

```
device=Device(gpu, 0) (goldens generated on GPU)
wrote golden vectors under /private/tmp/kvparity_p1
[selftest] affine4  gs=64 bits=4 elems=1024 diffs=0 maxabs=0 BIT-EXACT PASS
[selftest] affine8  gs=64 bits=8 elems=1024 diffs=0 maxabs=0 BIT-EXACT PASS
[selftest] tiered4  gs=64 bits=4 elems=131072 diffs=0 maxabs=0 BIT-EXACT PASS
kv_tier_selftest: BIT-EXACT PASS

selftest exit=0
```

This covers the tiered4 history/recent split with an 8,192→256-token window
(3 chunks of 128/256/128; the last 256 tokens age into INT4 history) and both
affine bit widths. It is a **primitive-level** claim: the full generator still
has the pre-existing logit gaps listed below.

## Real-checkpoint runs (all measured)

Every real-checkpoint run below used
`scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 5120
--compressor-growth-mib 8192 --thermal-level 2 --max-seconds 600` from
`/Users/vincent/Documents/Veystrix`, with `sysctl vm.swapusage` checked before
each (baseline `used = 3595.38M` throughout) and 10–12 s cool-down gaps.
**No guard stop occurred in any run** (`"stopped": null`), swap growth was
0 MiB in every run, thermal level 0 throughout. The binary was built with the
same MLX 0.32.2 include/lib as the Python engine (`xcrun clang++ -std=c++20 -O2`,
all `cpp/engine/src/*.cpp` + `io_pager.cpp` + `native_read.c`). PLE loading was
disabled with `VEYSTRIX_CPP_NO_PLE=1` because the older flat config in
`results/flashnext/20261117-222244-real-reference` lacks the newer PLE tensor
map keys; this is orthogonal to the two features.

Same 16-token prompt as the stored reference, 3 greedy tokens, token-in of
step 0 is 93460:

| C++ arm | tokens (step 0/1/2) | digests (first 12 hex) | guard |
|---|---|---|---|
| baseline (features unset) | 220 / 220 / 103910 | `6be10d95b4d0`, `213adfe577f6`, `ec41771b5596` | null, 8.0 s |
| explicit `KV_TIER=off` | 220 / 220 / 103910 | identical step lines to baseline | null, 8.0 s |
| `affine4` | 220 / 220 / 103910 | `6be10d95b4d0`, `a1a43d52fd53`, `1a0356b73ef8` | null, 8.1 s |
| `affine8` | 220 / 220 / 103910 | `6be10d95b4d0`, `c9eb6164913f`, `031049a5baa7` | null, 8.0 s |
| `tiered4` | 220 / 220 / 103910 | `6be10d95b4d0`, `c9eb6164913f`, `031049a5baa7` | null, 8.0 s |
| `MTP=1`, depth 2 | 220 / 220 / 103910 | `6be10d95b4d0`, `213adfe577f6`, `ec41771b5596` | null, 9.1 s |

Notes (measured): step-0 digests are identical in all arms because batched
prefill logits are produced before the tier priming changes the cache; the
modes first affect step 1. `tiered4` C++ digests equal `affine8` C++ digests
because 16 prompt + 3 decode tokens ≪ 8192 recent window, so every live token
is in the INT8 recent tier — the INT4 history tier is never exercised in this
short run (the standalone selftest above does exercise it). The MTP arm's
losslessness summary was `proposed=2 accepted=0 rejected=2`; both drafts were
rejected and redrawn to the plain greedy token by design of the synthetic head.

`check_mtp_real.py` output (measured, exact):

```
comparison: 3 baseline steps vs 3 MTP steps
step  0: token OK (baseline 220, mtp 220)  digest OK (6be10d95b4d0 vs 6be10d95b4d0)
step  1: token OK (baseline 220, mtp 220)  digest OK (213adfe577f6 vs 213adfe577f6)
step  2: token OK (baseline 103910, mtp 103910)  digest OK (ec41771b5596 vs ec41771b5596)

========================================================================
LOSSLESS: PASS — tokens 3/3, digests 3/3 (MTP output == plain greedy output)
PYTHON NATIVE MTP (reported, not claimed bit-exact): C++ MTP vs Python token agreement 0/3 (C++ [220, 220, 103910], Python [271, 248068, 198])
```

### Python side, matching modes (measured, deliberately NOT a bit-exact gate)

Python references were captured with the newer
`cpp/engine/tools/capture_real_reference.py` under the same guard, forcing
greedy, same 16-token prompt, 3 tokens:

| Python arm (`FLASHNEXT_KV_CACHE_DTYPE`) | effective mode / bits (recorded) | tokens |
|---|---|---|
| `affine4` | affine4 / 4 | 271 / 96347 / 103853 |
| `affine8` | affine8 / 8 | 271 / 96347 / 103853 |
| `tiered4` (+ `FLASHNEXT_QSA_KERNEL=gather`) | tiered4 / 4 | 271 / 248045 / 271 |
| `FLASHNEXT_NATIVE_MTP=on`, depth 2 | native=on, active=True | 271 / 248068 / 198 (0 logit digests — MTP bypasses `Sampler.__call__`) |

`check_kv_tier_parity.py --compare-arms` output (measured):

```
generator-level KV mode comparison (measured; bit-exactness NOT claimed)
  affine4  C++ tokens=[220, 220, 103910] digests=['6be10d95b4d0', 'a1a43d52fd53', '1a0356b73ef8']
           Python tokens=[271, 96347, 103853] digests=['a9b0e49dad17', '2c87eb341f0e', 'ffebad4f924c']  (ran)
  affine8  C++ tokens=[220, 220, 103910] digests=['6be10d95b4d0', 'c9eb6164913f', '031049a5baa7']
           Python tokens=[271, 96347, 103853] digests=['b099c6573b41', 'ea417adb3c6c', '56257a32a4ba']  (ran)
  tiered4  C++ tokens=[220, 220, 103910] digests=['6be10d95b4d0', 'c9eb6164913f', '031049a5baa7']
           Python tokens=[271, 248045, 271] digests=['c8538fc5a890', '1c63caa3275f', '662b4ec3d47c']  (ran)
```

**Honest reading:** C++ and Python do not agree token-for-token in ANY mode,
including the existing bf16 reference (the stored gate already reports 0/8
token and 0/8 digest matches — **cited**:
`research_notes/Cerebras inference and IO splitting/real_checkpoint_generate.md`).
The pre-existing gaps are PLE, quantized-vs-dense matmul, unfused RoPE, dense
vs int8 KV, and sequential-vs-chunked prefill. The mode arms nevertheless show
the intended behaviour: each C++ mode changes logits from baseline, and the
Python arms show affine4≈affine8 tokens with tiered4 (gather kernel) distinct.
**We do not claim cross-language bit-exactness for any generator-level mode.**

## Reproduce

```
# Build (from the worktree root), MLX from the owner's venv:
MLXROOT=/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages/mlx
xcrun clang -O2 -c models/flashnext/native_read.c -o /tmp/native_read.o
xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src -Icpp/decode/src \
    cpp/engine/src/*.cpp cpp/decode/src/io_pager.cpp /tmp/native_read.o \
    cpp/engine/tools/real_checkpoint_generate.cpp \
    -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$MLXROOT/lib" -o /tmp/real_gen_p1

# Primitive bit-exact check (GPU, no checkpoint):
cd /Users/vincent/Documents/Veystrix
.venv/bin/python /Users/vincent/.local/share/opencode/worktree/dd9a0e/ms-phase1-engine/cpp/engine/tools/check_kv_tier_parity.py \
    --out /tmp/kvparity_p1 --binary /tmp/real_gen_p1

# Real-checkpoint runs: always via
# scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 5120 \
#   --compressor-growth-mib 8192 --thermal-level 2 --max-seconds 600 --log <log> -- \
#   env VEYSTRIX_CPP_NO_PLE=1 VEYSTRIX_CPP_KV_TIER=<mode> /tmp/real_gen_p1 \
#     <checkpoint> <reference.json> 3 <real_config.txt>
# (check sysctl vm.swapusage before each; stop if the guard prints STOP.)
```
