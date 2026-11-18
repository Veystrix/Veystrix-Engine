# Nonzero-position MTP draft parity (prepopulated QSA cache)

**Status (measured):** the C++ real-checkpoint MTP draft head is **bit-exact**
against the checkpoint's own `Qwen4ExpMTPModule` at **nonzero draft positions**
(7 and 11) with a **prepopulated QSA KV cache**: 0 / 248320 differing f32
elements, identical logit SHA-256 and identical argmax in both cases. The
position-0 fresh-cache regression still passes byte-identically (0 / 248320),
and the synthetic MTP test still passes. This closes the "only position-0
fresh-cache validated" gap documented in `mtp_real_weights.md`. Everything
below is labelled **measured**, **cited**, or **honest limitation**. No commit
was made.

## Scope of edits (only these files in this worktree)

| File | Change |
|---|---|
| `cpp/engine/src/mtp.hpp` | New `MTPDraftCache` (past K/V `[B,nKV,past_len,D]`, `past_len`); new 5-arg `mtp_draft_forward_real(..., const MTPDraftCache&)` declaration. The 4-arg fresh-cache declaration is kept. |
| `cpp/engine/src/mtp.cpp` | `draft_positions()` helper; `dense_attention_real()` now takes the cache, evaluates MRoPE at `cache.past_len` and concatenates past K/V before dense SDPA; 4-arg `mtp_draft_forward_real` delegates to the 5-arg overload with an empty cache. |
| `cpp/engine/tools/check_mtp_full_parity.py` | New: Python nonzero-position golden (real backbone hidden + prepopulated MTP cache) and byte-exact C++ compare; the C++ driver is generated into the output dir, never committed. |
| `docs/engineering/mtp_full_parity.md` | This document. |

`engine.cpp`, `qsa.cpp`, `qsa_full.cpp`, `decoder_full.*`,
`real_checkpoint_generate.cpp` and every other shared primitive were NOT
modified. `cpp/engine/CMakeLists.txt` needed no change (no new target).

## What changed and why it is correct

### C++ side

* RoPE: the fresh path used `cfg.position` (0). The prepopulated path uses
  `cache.past_len` (= Python `QSAKVCache.offset`), matching Python's
  `_prepare_projected_qkv`: with `position_ids=None` it builds
  `position_ids = arange(cache.offset, cache.offset + L)` and tiles to 3 axes.
  `draft_positions()` reproduces that `[3,B,S]` layout.
* Attention KV: Python stores keys **after** rotary via
  `cache.update_and_fetch(keys, values)` and returns the full
  `keys[..., :offset, :]` sequence. The C++ path concatenates the supplied
  (already-rotated) `past_keys`/`past_values` ahead of the current key/value
  on the sequence axis, then runs the same no-mask dense SDPA as before.
* No sparse indexer state is carried: for `key_len <= 2048` the QSA indexer
  returns `None` and `_create_qwen3_5_attention_mask` returns `None` for `S=1`
  (`cited`: `mlx_vlm/models/qwen4_exp/language.py::Qwen4ExpQSAIndexer.__call__`,
  `mlx_vlm/models/cache.py::create_attention_mask`,
  `mlx_vlm/models/base.py::create_attention_mask`). With `indexer_budget=2048`,
  `compress_ratio=4` → `block_topk=512` → sparse fires only when
  `key_len > 2048` (`cited`: config.py + the indexer's early return). Dense is
  therefore the correct path for the short contexts tested here.

### Python golden (the fused-RoPE caveat)

At position 0, `cos=1`/`sin=0`, so the eager `mx.cos`/`mx.sin` path and
production's fused Metal `metal::cos`/`metal::sin` kernel are identical — this
is why the earlier head-level parity passed at position 0. At a **nonzero**
position they differ by ~1 float32 ULP (`cited`:
`research_notes/Cerebras inference and IO splitting/qsa_full_port.md`,
"Fused-Metal RoPE path": "differs from the eager `mx.cos`/`mx.sin` path by ~1
float32 ULP"; the C++ port reproduces the eager branch). The first run with
the default fused kernel gave **225220 / 248320 differing elements** (mean
abs diff 0.033, argmax still 220) — an honest 1-ULP-level divergence, not a
logic error. The golden therefore forces the real module's eager branch with
`language.mtp.layers[0].self_attn.rotary_emb.fused_apply = False`, the exact
convention already used by `cpp/engine/tools/gen_qsa_full_vectors.py`. The
indexer shares that rotary object (`Qwen4ExpQSAIndexer.__init__` stores it), so
both attention and indexer use the eager branch. `rope_branch` is recorded in
`python_meta.json`.

## Reproduction

Build is automatic inside the tool's `compare` phase (it generates a small C++
driver into the output dir and compiles the engine sources + MLX). From
`/Users/vincent/Documents/Veystrix`, every real-checkpoint run below used:

```
scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 2 --max-seconds <budget> \
  --log <log> -- .venv/bin/python <worktree>/cpp/engine/tools/check_mtp_full_parity.py \
  --checkpoint /Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --worktree <worktree> --out <out> [--prompt-tokens N --prompt-seed S] \
  --phase gen|--phase compare
```

`sysctl vm.swapusage` was checked before each run; 12–15 s cool-down gaps were
observed between guarded runs. **No guard stop occurred** (`"stopped": null`
in every guard summary); thermal level was 0 in every run. The tool runs `gen`
and `compare` as separate child processes (so Python's full model is released
before the C++ loader allocates the packed MTP weights).

## Validation — exact measured output

### 1. Nonzero position A — position 7, 8-token prompt, seed 20261118

Golden provenance (measured): real backbone hidden `[1,1,10240]` bf16,
min −3.406, max 2.484, std 0.332; prepopulated cache K/V `[2,7,256]`
(`nKV=2`, `past_len=7`, `head_dim=256`); next token 248044 from target argmax;
argmax 220.

```
python golden: position=7 past_len=7 token=248044 logits=(1, 1, 248320) argmax=220 sha256=a9152fa1374be61d
MTP_FULL_PARITY past_len=7 argmax=220 logits_sha256=a9152fa1374be61d26caaf1202c8636272435c2a1336b0f9950dd9a65bf86df3 elems=248320
[mem mtp-full-parity] active=2.26 GiB cache=0.00 GiB peak=2.26 GiB
python logits: 248320 f32  sha256=a9152fa1374be61d26caaf1202c8636272435c2a1336b0f9950dd9a65bf86df3  argmax=220
C++    logits: 248320 f32  sha256=a9152fa1374be61d26caaf1202c8636272435c2a1336b0f9950dd9a65bf86df3  argmax=220
byte-level differing f32 elements: 0 / 248320
REAL MTP NONZERO-POSITION PARITY: BIT-EXACT PASS (tokens OK, logits OK)
guard(gen):     {"stopped": null, "returncode": 0, "seconds": 8.0,  "available_floor_gib": 2.27, "swap_baseline_mib": 3577, "swap_peak_mib": 3763, "thermal_seconds": {"0": 8}}
guard(compare): {"stopped": null, "returncode": 0, "seconds": 26.2, "available_floor_gib": 2.34, "swap_baseline_mib": 3724, "swap_peak_mib": 3724, "thermal_seconds": {"0": 26}}
```

### 2. Nonzero position B — position 11, 12-token prompt, seed 777

```
python golden: position=11 past_len=11 token=271 logits=(1, 1, 248320) argmax=248058 sha256=15981ef4ed62a1e7
MTP_FULL_PARITY past_len=11 argmax=248058 logits_sha256=15981ef4ed62a1e7668df2e6427b1ad9a1d11d794ff2b5b21cbd55cc084b0bcc elems=248320
[mem mtp-full-parity] active=2.26 GiB cache=0.00 GiB peak=2.26 GiB
python logits: 248320 f32  sha256=15981ef4ed62a1e7668df2e6427b1ad9a1d11d794ff2b5b21cbd55cc084b0bcc  argmax=248058
C++    logits: 248320 f32  sha256=15981ef4ed62a1e7668df2e6427b1ad9a1d11d794ff2b5b21cbd55cc084b0bcc  argmax=248058
byte-level differing f32 elements: 0 / 248320
REAL MTP NONZERO-POSITION PARITY: BIT-EXACT PASS (tokens OK, logits OK)
guard(gen):     {"stopped": null, "returncode": 0, "seconds": 6.1,  "available_floor_gib": 5.80, "swap_baseline_mib": 3281, "swap_peak_mib": 3281, "thermal_seconds": {"0": 6}}
guard(compare): {"stopped": null, "returncode": 0, "seconds": 24.2, "available_floor_gib": 8.22, "swap_baseline_mib": 3281, "swap_peak_mib": 3281, "thermal_seconds": {"0": 24}}
```

### 3. Position-0 regression (legacy 4-arg path) — unchanged

The existing `check_mtp_real_parity.py` gate, run with the rebuilt worktree
binary, still passes byte-identically (measured):

```
python golden: token=141029 logits=(1, 1, 248320) argmax=220 sha256=d6df9782a6c39557
MTP_REAL_PARITY argmax=220 logits_sha256=d6df9782a6c395575abf58f0eb8bfdd191534f929a4399e57b8173eac7d9217a elems=248320
byte-level differing f32 elements: 0 / 248320
REAL MTP DRAFT PARITY: BIT-EXACT PASS (tokens OK, logits OK)
guard: {"stopped": null, "returncode": 0, "seconds": 7.1, "available_floor_gib": 5.57, "swap_baseline_mib": 3281, "swap_peak_mib": 3281, "thermal_seconds": {"0": 7}}
```

### 4. Synthetic primitive regression — unchanged

```
/tmp/test_mtp_wt cpp/engine/tests/vectors_mtp
  fuse_inputs OK (256 bytes bit-identical)
  draft logits OK (256 bytes bit-identical)
  ... 11 verify cases OK ...
test_mtp OK (fuse_inputs + draft logits bit-identical; 11 verify cases)
```

## Honest scope / limitations

* **Head-level, same-input parity.** Python and C++ received the identical
  `[1,1,10240]` bf16 hidden stream, the identical token id and the identical
  prepopulated K/V cache. We do NOT claim generator-level C++ ↔ Python token
  parity (the backbone already differs for the documented PLE / quantized-vs-
  dense matmul / RoPE / KV / prefill reasons; `cited`:
  `docs/engineering/kv_tier_mtp_integration.md`).
* **Dense path only — sparse mask is a genuine open limitation.** Validated at
  `past_len = 7` and `11`, far below the `key_len > 2048` threshold where the
  QSA indexer emits a sparse boolean mask and Python's attention mask becomes
  sparse. The C++ `MTPDraftCache` carries only past K/V, **not** indexer state
  (`index_keys` / `index_position_ids`) or indexer weights, and no sparse-mask
  SDPA path is implemented. A draft step at `past_len >= 2048` would need that
  sparse path; it is out of scope and NOT claimed. The two measured positions
  exercise the dense regime correctly (`cited`: indexer early return +
  `create_attention_mask` above).
* **Eager RoPE branch.** Bit-exactness is against the real module's eager RoPE
  branch (`fused_apply=False`). Production's fused Metal kernel differs by
  ~1 f32 ULP (`cited`: `qsa_full_port.md` caveat 1), so exact parity with the
  *default* production fused path at nonzero positions is NOT claimed; the
  first run above quantified that divergence (225220 / 248320 elements).
* **Shape / batch.** Validated for `B=1`, `S=1` (the MTP draft step). `S>1`
  would require a causal mask in C++ (Python's `create_attention_mask` returns
  `"causal"` when `N>1`); not implemented or claimed here. `draft_positions()`
  is written for general `S` but the mask path is deliberately dense/no-mask.
* **Indexer state not updated by this path.** The C++ draft step does not call
  `qsa_full_indexer` nor append index keys, so it cannot be chained to produce
  a valid sparse cache for a later step. A full sparse-capable draft loop would
  need indexer state wiring (out of scope).
* **Resident packed experts / same device / same as prior caveats** —
  unchanged from `mtp_real_weights.md`: C++ keeps all 512 experts packed
  (~2.26 GiB active for the head), Python streams them from SSD; both ran on
  GPU with MLX 0.32.2, no cross-device bit-exactness claimed.
