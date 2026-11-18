# Real-checkpoint MTP draft head in C++ (shard 22) — wire-up and parity

**Status (measured):** the C++ draft head is **bit-exact** against the
checkpoint's own `Qwen4ExpMTPModule` on two different real hidden states, on the
same device (GPU, MLX 0.32.2). The synthetic-weight MTP integration in the
decode loop has been replaced by the real shard-22 head; `mtp_verify` still
makes the emitted stream lossless. Everything below is labelled **measured**,
**cited**, or **honest limitation**. No commit was made.

## Scope of edits (only these files in this worktree)

| File | Change |
|---|---|
| `cpp/engine/src/mtp.hpp` | `MTPConfig` gains `one_centered_norms` / `tie_word_embeddings`; new `MTPRealWeights` (QPacked/Linear/ExpertWeights) + `mtp_draft_forward_real()` declaration. Synthetic `MTPWeights` / `mtp_draft_forward` are untouched. |
| `cpp/engine/src/mtp.cpp` | One-centered RMSNorm, real gated residual / QSA attention / packed SwitchGLU+shared MoE / real quantized-embedding row / untied `quantized_matmul` LM head; new `mtp_draft_forward_real()`. Fail-closed on the wrong norm/head convention. |
| `cpp/engine/tools/real_checkpoint_generate.cpp` | `VEYSTRIX_CPP_MTP_REAL_PARITY=<dir>` head-only parity entry point; `VEYSTRIX_CPP_MTP=1` decode loop now loads and calls the real head (synthetic fallback: `VEYSTRIX_CPP_MTP_SYNTHETIC=1`); derives MTP shard entries from `lm_head` for the older flat config. |
| `cpp/engine/tools/check_mtp_real_parity.py` | New: drives Python real MTP + C++ on the SAME hidden state/token; raw-byte logit + argmax comparison. |
| `docs/engineering/mtp_real_weights.md` | This document. |

`engine.cpp`, `kv_quant.*`, `qsa*`, `decoder_full.*` and every other shared
primitive were NOT modified. `cpp/engine/CMakeLists.txt` needed no change (no
new target; the existing manual build of the tool is used).

## What is wired

### Real weights (`MTPRealWeights`)

All 76 `language_model.mtp.*` tensors of
`model-00022-of-00022.safetensors` are loaded through `DenseLoader`'s
`linear()` (keeps affine-Q4 weights **packed** as `QPacked`/`Linear`) and
`raw()` (norm gains, router gate). The routed experts are an `ExpertWeights`
triplet consumed by the already-validated `q4_projection`/`moe_block`
(`mx::gather_qmm`, no sort for one token). The tokenizer embedding and the
**untied** `lm_head` are `QPacked` tables; the head applies one gathered and
dequantized embedding row and the head matmul via `mx::quantized_matmul`.
Nothing is eagerly upcast to bf16: measured active memory of the head-only
parity process is **2.26 GiB** (vs the ~5 GiB estimate for fully resident bf16
experts in the earlier integration doc).

The math difference from the synthetic path is only in conventions:
`patch_rmsnorm.py` patches every `Qwen4ExpRMSNorm` to **one-centered** gains
(`gain = weight`, not `1 + weight`) — `oc_rms_norm()` mirrors `_rms_norm`'s op
order exactly — and `tie_word_embeddings=False` selects `lm_head`. The sidecar
layer is `layer_types=["qwen_sparse_attention"]`, so the fresh-cache single
token path is a dense softmax (indexer returns `None`), the same path the
synthetic primitive validates.

### Decode loop

`VEYSTRIX_CPP_MTP=1` (depth 1..4, session seed) now calls
`mtp_draft_forward_real` on the last hidden stream `[1,1,4*2560]` from
`run_step` and the **real** token id (no modulo-256 folding, no shape slicing).
`mtp_verify` remains lossless: an accepted draft is consumed through the target,
a rejected draft is redrawn from the current target row.
`VEYSTRIX_CPP_MTP_SYNTHETIC=1` restores the old tiny head for smoke tests only.

### Parity harness

`check_mtp_real_parity.py` runs in two child processes so Python's full model is
released before the C++ loader allocates:

1. `--phase gen`: `load_streaming(..., use_mtp=True)`, a short deterministic
   prompt, `language(..., capture_layer_ids=[])` exposes the last layer's raw
   `[1,1,10240]` hidden stream; `mtp.forward(language, hidden, token,
   mtp.make_cache(language))` produces the golden logits. Dumps `hidden.f32`,
   `token.txt`, `py_logits.f32` and a flat `config.txt` (shard map).
2. `--phase compare`: runs the C++ binary with `VEYSTRIX_CPP_MTP_REAL_PARITY`
   and compares `cpp_logits.f32` to `py_logits.f32` by **f32 byte equality**,
   plus argmax/token equality. No tolerance.

## Reproduction

Build (from the worktree root, MLX from the engine venv):

```
MLXROOT=/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages/mlx
xcrun clang -O2 -c models/flashnext/native_read.c -o /tmp/native_read.o
xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src -Icpp/decode/src \
    cpp/engine/src/*.cpp cpp/decode/src/io_pager.cpp /tmp/native_read.o \
    cpp/engine/tools/real_checkpoint_generate.cpp \
    -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$MLXROOT/lib" -o /tmp/real_gen_mtp
```

Every real-checkpoint run below used, from `/Users/vincent/Documents/Veystrix`:

```
scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 2 --max-seconds <budget> \
  --log <log> -- <command>
```

`sysctl vm.swapusage` was checked before each run (`used = 3403.38M`, i.e. ~3.3
GiB baseline throughout), with 12 s cool-down gaps. **No guard stop occurred**
(`"stopped": null` in every guard summary); swap peak equalled baseline (0 MiB
growth) and thermal level was 0 in every run.

## Validation — exact test output

### 1. Real MTP head parity, case A (8-token prompt, token 141029)

```
python golden: token=141029 logits=(1, 1, 248320) argmax=220 sha256=d6df9782a6c39557
MTP_REAL_PARITY argmax=220 logits_sha256=d6df9782a6c395575abf58f0eb8bfdd191534f929a4399e57b8173eac7d9217a elems=248320
[mem mtp-real-parity] active=2.26 GiB cache=0.00 GiB peak=2.26 GiB
========================================================================
python logits: 248320 f32  sha256=d6df9782a6c395575abf58f0eb8bfdd191534f929a4399e57b8173eac7d9217a  argmax=220
C++    logits: 248320 f32  sha256=d6df9782a6c395575abf58f0eb8bfdd191534f929a4399e57b8173eac7d9217a  argmax=220
byte-level differing f32 elements: 0 / 248320
REAL MTP DRAFT PARITY: BIT-EXACT PASS (tokens OK, logits OK)
guard: {"event": "end", "stopped": null, "returncode": 0, "seconds": 2.0, "available_floor_gib": 6.12, "swap_baseline_mib": 3403, "swap_peak_mib": 3403, "swap_end_mib": 3403, "compressor_baseline_mib": 1176, "compressor_peak_mib": 1176, "swapout_mib": 0, "compressed_gib_total": 0.0, "thermal_seconds": {"0": 2}}
```

### 2. Real MTP head parity, case B (independent 12-token prompt, token 106836)

```
python golden: token=106836 logits=(1, 1, 248320) argmax=198 sha256=a8d8fee337a34826
MTP_REAL_PARITY argmax=198 logits_sha256=a8d8fee337a34826e79e624d279b1d9b2e5e6b5ff4bb5db6ae582cab95724121 elems=248320
[mem mtp-real-parity] active=2.26 GiB cache=0.00 GiB peak=2.26 GiB
========================================================================
python logits: 248320 f32  sha256=a8d8fee337a34826e79e624d279b1d9b2e5e6b5ff4bb5db6ae582cab95724121  argmax=198
C++    logits: 248320 f32  sha256=a8d8fee337a34826e79e624d279b1d9b2e5e6b5ff4bb5db6ae582cab95724121  argmax=198
byte-level differing f32 elements: 0 / 248320
REAL MTP DRAFT PARITY: BIT-EXACT PASS (tokens OK, logits OK)
guard: {"event": "end", "stopped": null, "returncode": 0, "seconds": 7.1, "available_floor_gib": 4.44, ... "swap_peak_mib": 3403, "swapout_mib": 0, "thermal_seconds": {"0": 7}}
```

The hidden streams were genuine, not degenerate: case A `hidden.f32` had 10240
elements, min −5.938, max 3.953, std 0.4135 (8 zeros); logits min −7.31, max
9.5, std 1.346, top-5 argmax `[220, 271, 198, 17, 30]`.

### 3. Synthetic primitive regression (unchanged path)

```
/tmp/test_mtp_mtp cpp/engine/tests/vectors_mtp
  fuse_inputs OK (256 bytes bit-identical)
  draft logits OK (256 bytes bit-identical)
  ... 11 verify cases OK ...
test_mtp OK (fuse_inputs + draft logits bit-identical; 11 verify cases)
```

### 4. Decode-loop wire-up (real head, load-only + 3 tokens)

```
[mtp] real shard-22 head loaded (H=2560 HC=4 E=512 topk=10 vocab=248320 one_centered=1 tied=0)
[mem post-load] active=4.47 GiB cache=0.00 GiB peak=4.47 GiB
guard: {"event": "end", "stopped": null, ... "available_floor_gib": 5.77, ...}
```

```
[step 0] in=93460 out=220 sha=6be10d95b4d03725b67829d6c7a6b14cd29c3008d0b389fdf00b5f24c5a6dc7c
[step 1] in=220 out=220 sha=213adfe577f681b13de80b0e05e8f31ee769ecf5dee0f2e087c734c54665571a
[step 2] in=220 out=103910 sha=ec41771b55960ae519a3e556c7650d02c55557c2f8dec4e9f33fdfa9185002b6
[mtp] summary: proposed=2 accepted=0 rejected=2 (lossless greedy: output equals plain greedy)
guard: {"event": "end", "stopped": null, "returncode": 0, "seconds": 9.1, "available_floor_gib": 2.53, "swap_baseline_mib": 3403, "swap_peak_mib": 3403, "swapout_mib": 0, "thermal_seconds": {"0": 9}}
```

The step digests are byte-identical to the documented baseline greedy arm in
`kv_tier_mtp_integration.md` (the two drafts were rejected and redrawn from the
target, as expected under near-argmax verification).

## Honest scope / limitations

* **Bit-exact claim is head-level, on the same input.** Python and C++ received
  the identical `[1,1,10240]` bf16 hidden stream and the identical token id.
  We do **not** claim generator-level C++ ↔ Python token parity: the backbone
  already differs for the documented PLE / quantized-vs-dense matmul / RoPE /
  KV / prefill reasons (cited: `docs/engineering/kv_tier_mtp_integration.md`).
* **Fresh-cache path only.** The bit-exact draft step is S=1, QSA indexer
  `None`, MRoPE position 0, i.e. `mtp.make_cache(language)` with
  `position_ids=None` (`cited`: `models/flashnext/mtp.py`, `Qwen4ExpAttention`).
  A prepopulated MTP cache / multi-token draft (sparse indexer mask, nonzero
  position) is **not implemented**; the decode loop deliberately evaluates real
  drafts at position 0. Verification is lossless, so this limits draft quality,
  not emitted-token correctness.
* **Resident packed experts, not SSD streaming.** Python's
  `StreamingSwitchGLU` reads 10 routed experts per step from disk; C++ keeps all
  512 experts packed in memory (~2.26 GiB active for the head alone). The
  arithmetic (`gather_qmm` with `affine`, unsorted, SwiGLU order) is the same and
  is what the byte comparison validates; the I/O path is not reproduced.
* **All sidecar projections in this checkpoint are 4-bit / group-32.**
  `DenseLoader::linear()` derives gs/bits from shapes per tensor; the embedding
  and lm_head use the config's group_size/bits (32/4 here). A mixed-bit sidecar
  would need per-tensor derivation for those two tables too.
* **Same device.** Both goldens and C++ ran on the GPU with MLX 0.32.2; no
  cross-device (CPU↔GPU) bit-exactness is claimed (the 1-ULP device difference
  is documented in `docs/engineering/kv_quant_port.md`).
* **MTP quality caveat unchanged** (`setting_mtp.py`: native MTP off by default,
  "experimental, exact greedy only"). This work ports the head faithfully; it
  does not enable MTP by default.
