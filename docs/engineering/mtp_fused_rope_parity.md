# Production fused-RoPE parity for the C++ MTP draft head

**Status (measured):** the C++ real-checkpoint MTP draft head is **bit-exact**
against the checkpoint's own `Qwen4ExpMTPModule` with **Python's PRODUCTION
FUSED Metal RoPE kernel** (the default; `fused_apply` untouched) at **nonzero
draft positions 7 and 11** with a prepopulated QSA KV cache: **0 / 248320**
differing f32 elements and identical SHA-256 / argmax in both cases. The
legacy **EAGER** branch remains available as an opt-in and is *also* bit-exact
against Python's `fused_apply=False` golden (0 / 248320 both positions). The
position-0 fresh-cache gate and the synthetic `test_mtp` gate still pass. This
closes the "C++ MTP uses eager `qsa_apply_rope`, ~225220/248320 elements ~1 ULP
from production" gap recorded in `mtp_full_parity.md` §"Eager RoPE branch".
Everything below is labelled **measured**, **cited**, or **honest limitation**.
**No commit was made.**

## Scope of edits (only these files in this worktree)

| File | Change |
|---|---|
| `cpp/engine/src/mtp.cpp` | `#include "rope_fused_apply.hpp"`; new `mtp_apply_rope()` helper in the anonymous namespace; both eager `qsa_apply_rope` call sites (`dense_attention` synthetic path and `dense_attention_real` real path) now call the helper. Default = `rope_apply_fused` (the verbatim Metal kernel port in `rope_fused.cpp`); opt-in `VEYSTRIX_CPP_MTP_EAGER_ROPE=1` keeps `qsa_apply_rope`. |
| `cpp/engine/tools/check_mtp_fused_rope_parity.py` | New: dual-mode (fused / eager) nonzero-position golden + byte-exact C++ compare. The fused golden does **not** override `fused_apply` and asserts the module really is on the fused path. The C++ driver is generated into the output dir, never committed. |
| `docs/engineering/mtp_fused_rope_parity.md` | This document. |

`engine.cpp`, `qsa.cpp`, `qsa_full.cpp`, `rope_fused.*`, `rope_fused_apply.hpp`,
`decoder_full.*`, `real_checkpoint_generate.cpp` and every other shared
primitive were NOT modified. `cpp/engine/CMakeLists.txt` needed no change (the
new tool generates its own driver; the existing `test_mtp` target already picks
up the edited `mtp.cpp`).

## What changed and why it is correct

### C++ side (measured compile + run)

`mtp_apply_rope(x, position_ids, rp)` dispatches on the env var:

```
VEYSTRIX_CPP_MTP_EAGER_ROPE=1  -> qsa_apply_rope(x, position_ids, rp)
unset / "0"                    -> rope_apply_fused(x, position_ids, rp)
```

The convenience overload `rope_apply_fused(x, position_ids, const QSARotaryParams&)`
in `rope_fused_apply.hpp` builds `inv_freq` and the MRoPE position selector with
the exact production helpers and forwards to `rope_fused_apply()`, which invokes
the verbatim `_mrope_apply_kernel` f-string via `mx::fast::metal_kernel` — the
same path already used by the backbone full attention in
`real_checkpoint_generate.cpp` (`rope_apply_fused(q, position_ids, cfg.rope)`).
`rope_fused.cpp` was previously proven 0-ULP vs production by `test_rope_fused`
/ `test_rope_fused_mrope` (`cited` in that header).

The position ids are the unchanged `[3, B, S]` `draft_positions()` layout: all
three MRoPE axes carry the same scalar text position, so the interleaved
selector is irrelevant to the angle and `rp.style = "interleaved"` /
`pairing = "half_split"` is reproduced exactly. Measured rotary config from the
real module (`python_meta_fused.json`): `fused_apply=true`, `style=interleaved`,
`pairing=half_split`, `dim=64`, `inv_freq_len=32`,
`mrope_section=[11,11,10]`, interleaved `position_selector`. No position-layout,
`mrope_section` or interleaved-flag adjustment was needed — the swap alone
closed the gap. **No residual root cause remains.**

### Python golden (measured)

`--rope-mode fused` leaves `language.mtp.layers[0].self_attn.rotary_emb`
completely untouched; it asserts `rot.fused_apply` is true and records the
rotary details in `python_meta_fused.json`. `--rope-mode eager` sets
`rot.fused_apply = False; rot._compiled_apply = None` before *any* MTP forward,
so both the warmed cache keys and the final draft step use the eager branch —
exactly the convention in `check_mtp_full_parity.py`. Both modes use the same
real backbone hidden stream and the same target-argmax next token; the shared
inputs (hidden + token) hash identically, while the RoPE-rotated cache keys
differ by design (separate `cache_sha256` recorded per mode).

## Reproduction

Build is automatic inside the tool's `compare` phase (a small C++ driver is
generated into the output dir and compiles the engine sources + MLX). From
`/Users/vincent/Documents/Veystrix`, every real-checkpoint run below used the
required guard with `sysctl vm.swapusage` checked beforehand (~2.9 GiB baseline;
no guard stop — `"stopped": null` in every summary) and 15 s cool-downs:

```
VEYSTRIX_CPP_MTP_EAGER_ROPE=1 \   # only for the EAGER gate; omit for FUSED
.venv/bin/python scripts/flashnext/real_run_guard.py \
  --floor-gib 1.5 --swap-growth-mib 5120 --compressor-growth-mib 8192 \
  --thermal-level 2 --max-seconds <budget> --log <log> -- \
  .venv/bin/python <worktree>/cpp/engine/tools/check_mtp_fused_rope_parity.py \
  --checkpoint /Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --worktree <worktree> --out <out> \
  [--prompt-tokens N --prompt-seed S] --rope-mode fused|eager|both
```

`--rope-mode both` (default) orchestrates `gen`→`compare` for fused then eager
as four child processes so the heavy Python model is released before each C++
loader allocates packed weights; `--phase gen|compare --rope-mode fused|eager`
runs a single phase. The tool reuses an up-to-date driver binary.

To keep the legacy gate passing, it must now opt in explicitly (the gate's own
source was not edited):

```
VEYSTRIX_CPP_MTP_EAGER_ROPE=1 <guard ...> -- \
  .venv/bin/python <worktree>/cpp/engine/tools/check_mtp_full_parity.py ...
```

## Validation — exact measured output

### 1. Fused golden vs C++ default — position 7 (8-token prompt, seed 20261118)

Golden provenance (measured): real backbone hidden `[1,1,10240]` bf16;
prepopulated cache K/V `[2,7,256]`; next token 248044 (target argmax); argmax
220. Python golden SHA and C++ SHA both equal:

```
python golden [fused]: position=7 past_len=7 token=248044 argmax=220
  sha256=39a6478dd24217e5e38570372afbacf821fbf3ddcce9399e3d682621d17b8bff
MTP_FUSED_ROPE_PARITY mode=fused cpp_rope=fused past_len=7 argmax=220
  logits_sha256=39a6478dd24217e5e38570372afbacf821fbf3ddcce9399e3d682621d17b8bff elems=248320
byte-level differing f32 elements: 0 / 248320
MTP RoPE PARITY: BIT-EXACT PASS (tokens OK, logits OK)
guard(gen):     stopped=null returncode=0 seconds=6.1  swap 2905->2905 MiB
guard(compare): stopped=null returncode=0 seconds=23.1 swap 2905->2905 MiB
```

### 2. Fused golden vs C++ default — position 11 (12-token prompt, seed 777)

```
python golden [fused]: position=11 past_len=11 token=271 argmax=248058
  sha256=b7958a8791619d4e963810397c19bbbcfe8290ad926c0c8920cf565729c0210a
MTP_FUSED_ROPE_PARITY mode=fused cpp_rope=fused past_len=11 argmax=248058
  logits_sha256=b7958a8791619d4e963810397c19bbbcfe8290ad926c0c8920cf565729c0210a
byte-level differing f32 elements: 0 / 248320
guard(both): stopped=null returncode=0 seconds=35.3 swap 2897->2897 MiB
```

### 3. Eager regression — Python `fused_apply=False` vs `VEYSTRIX_CPP_MTP_EAGER_ROPE=1`

Measured — both positions bit-exact, and the SHAs equal the pre-existing eager
gate's published values (`mtp_full_parity.md` §§1–2):

| Position | Eager Python == C++ SHA-256 | Diffs | Argmax |
|---|---|---|---|
| 7 | `a9152fa1374be61d26caaf1202c8636272435c2a1336b0f9950dd9a65bf86df3` | 0 / 248320 | 220 |
| 11 | `15981ef4ed62a1e7668df2e6427b1ad9a1d11d794ff2b5b21cbd55cc084b0bcc` | 0 / 248320 | 248058 |

`guard(eager gen)` p7: stopped=null, 5.0 s; `guard(eager compare)` p7:
stopped=null, 1.0 s. Combined `--rope-mode both` at each position:
`all_bit_exact=true`, shared inputs identical, cache digests differ by design.

### 4. Quantified production gap (measured, for contrast)

Raw f32 byte comparison of the two *Python* goldens (no tolerance):

| Position | Fused-vs-eager differing elements | Mean abs diff | Argmax |
|---|---|---|---|
| 7 | 227526 / 248320 | 0.037950 | 220 (both) |
| 11 | 223145 / 248320 | 0.033186 | 248058 (both) |

This is the ~1-ULP drift from `metal::cos/sin` (fused kernel, f32 multiply-add
then cast) vs `mx.cos/sin` (eager, cos/sin cast to bf16 first). Argmax is
unchanged — consistent with the original 225220/248320 observation. After the
swap, the C++ **fused** path matches the fused golden with **0** diffs; the
opt-in C++ **eager** path matches the eager golden with **0** diffs.

### 5. Existing gates — still passing (measured)

* Position-0 fresh-cache `check_mtp_real_parity.py` on a freshly compiled
  `real_gen_mtp` (no env var needed; at position 0 `cos=1`/`sin=0`, so fused and
  eager are identical): SHA
  `d6df9782a6c395575abf58f0eb8bfdd191534f929a4399e57b8173eac7d9217a`
  for both Python and C++, 0 / 248320 diffs, argmax 220. Guard: stopped=null,
  7.1 s.
* Nonzero-position `check_mtp_full_parity.py` (eager golden) **with**
  `VEYSTRIX_CPP_MTP_EAGER_ROPE=1`: SHA
  `a9152fa1374be61d26caaf1202c8636272435c2a1336b0f9950dd9a65bf86df3`
  for both, 0 / 248320 diffs, argmax 220. Guard: stopped=null, 28.2 s.
  **Without** the env var it correctly FAILS (230887 / 248320 diffs) because
  the C++ default is now fused — this is the intended behavior change, not a
  regression; the gate must opt in to the eager branch it validates.
* Synthetic `test_mtp` (CMake target, position 0, no checkpoint): `fuse_inputs`
  and draft logits 256 bytes bit-identical; all 11 verify cases OK.

### 6. Safety record (measured)

All nine real-checkpoint invocations used the exact guard flags above; every
guard summary had `"stopped": null`. Swap baseline 2897–2905 MiB, peak
2897–2905 MiB (growth 0—8 MiB, far under the 5120 MiB cap); compressor growth
well under 8192 MiB; thermal level 0 throughout; 15 s cool-downs between runs.
C++ peak active memory 2.26 GiB for the head. No OOM, no thermal or swap stop.

## Honest scope / limitations

* **Head-level, same-input parity** — Python and C++ received the identical
  backbone hidden, token id and (mode-specific) prepopulated K/V. No
  generator-level C++ ↔ Python token parity is claimed (`cited`:
  `docs/engineering/kv_tier_mtp_integration.md`).
* **Dense path only** — validated at `past_len = 7` and `11`, far below the
  `key_len > 2048` threshold where the QSA indexer emits a sparse mask. The
  sparse-mask draft step is an open limitation inherited from
  `mtp_full_parity.md`; not implemented or claimed here.
* **Shape / batch** — `B=1`, `S=1` (the draft step). `S>1` would need a causal
  mask; out of scope.
* **Eager gate invocation changed** — because the default is now fused, the old
  eager gate must set `VEYSTRIX_CPP_MTP_EAGER_ROPE=1`. This is documented above
  and is the only caller-visible behavior change.
* **Fused kernel port is Metal-only for bit-exactness** — `rope_fused_apply`
  falls back to an approximate `mx.cos/sin` reference without Metal (`cited`:
  `rope_fused.cpp` catch branch). All measurements here were on Apple Silicon
  with MLX 0.32.2.
* **Cited vs measured** — the 0-ULP claim for `rope_fused.cpp` against
  production is `cited` from the header/tests; the 0-diff results above are
  **measured** on the real shard-22 head. No estimate is used for any parity
  claim.
