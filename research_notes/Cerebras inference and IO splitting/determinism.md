# Determinism investigation — `real_checkpoint_generate.cpp`

Owner note for `cpp/engine/tools/determinism_probe.cpp`,
`cpp/engine/tools/check_determinism.py`, and this file. No shared engine source
was modified. `real_checkpoint_generate.cpp` was NOT patched (see "Decision").

## Question

Three runs of the same binary on the real 105 GB checkpoint produced three
different step-0 tokens (186204, 145166, 229910). A nondeterministic decode
cannot pass a bit-exact gate. The owner has confirmed **Python is deterministic
on the same checkpoint (3/3 identical token sequences + digests)**, so this is
a fixable C++ port bug, not a hardware limit.

## Method

`determinism_probe.cpp` generates SYNTHETIC inputs with the exact shapes/dtypes
of the real ops (verified against the safetensors headers: `lm_head.weight` U32
[248320,320], `.scales`/`.biases` BF16 [248320,80]) and evaluates each op 5x in
one process, printing a SHA-256 of the float32-LE output plus the number of
differing elements vs. run 0. It loads no checkpoint and finishes in seconds.

Build:
```bash
cd /Users/vincent/Documents/Veystrix
MLXROOT=.venv/lib/python3.12/site-packages/mlx
xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
  cpp/engine/src/engine.cpp cpp/engine/src/gdn_kernel.cpp \
  cpp/engine/tools/determinism_probe.cpp \
  -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$(pwd)/$MLXROOT/lib" -o /tmp/det_probe
.venv/bin/python cpp/engine/tools/check_determinism.py --probe /tmp/det_probe
```

## Probe results — every isolated op is DETERMINISTIC

`check_determinism.py` summary (5 evals per sub-case, 1 distinct hash each):

| mode | shapes | run-0 SHA-256 (first 16 hex) | verdict |
|---|---|---|---|
| `qmm` f32 / bf16 | x[1,1,2560] @ w[248320,320] packed, gs32 b4 | `8d8439dc15be20b6` / `04cbccbc7619823f` | DETERMINISTIC |
| `dqmm` f32 / bf16 | dequantize head then dense `mx::matmul` | `ad24cd5c2e726561` / `c73bea01cf6622c2` | DETERMINISTIC |
| `gdn` | fused delta kernel, q/k[1,1,16,128] v[1,1,48,128] state[1,48,128,128] | `598c44d4983b1085` | DETERMINISTIC |
| `sdpa` | fused SDPA q[1,24,1,256] k/v[1,2,16,256] causal | `b4d6942d0109c8fe` | DETERMINISTIC |
| `router` | softmax+argpartition(512,10), argmax(248320), incl. exact-tie stress | `29e23344b7a0a944` (v0) | DETERMINISTIC |
| `gqmm` | MoE `gather_qmm`, host-backed slot w[16,2560,320], rhs[1,10] | `cd40a2ea1fb0a19e` | DETERMINISTIC |
| `nan` | qmm with +Inf/NaN scales; argmax on NaN-containing row | `0560e44bab9d2e10` / argmax idx=585 | DETERMINISTIC |
| `race` | gather_qmm then immediately mutate host buffer (eval synchronicity) | A/B matched expected 5/5 | mx::eval is SYNCHRONOUS |

The `qmm` and `dqmm` hashes differ from each other (different numerics), but
each is stable across all 5 evals **and** across 3 separate processes.

## The lm_head split-K hypothesis is DISPROVEN

MLX 0.32.2 `mlx/backend/metal/quantized.cpp`, `QuantizedMatmul::eval_gpu`:

* `M = x.size()/K`. For the lm_head, `mixed` is [1,1,2560], so **M = 1**.
* `vector_limit = get_qmv_batch_limit(K=2560, N=248320, d)`. On Apple M4
  (`arch_gen==17`, size char != 'd') this returns **13**.
* `M (1) < 13`, so the `M >= vector_limit` branch — the only one that calls
  `qmm_splitk` (atomic partial sums) — is **never taken**.
* Dispatch falls to `transpose_ → dispatch_qmv → qmv`; with `N%8==0` and
  `K % (pack_factor*2*32) == 2560 % 512 == 0`, the kernel is
  `affine_qmv_fast_*`. Each output row is fully reduced inside one SIMD group;
  **no split-K, no atomics**.
* The NAX split-K path does not exist on M4 (`is_nax_available()` false).

So `mx::quantized_matmul` for the lm_head is deterministic on this machine for
this shape. The original note's "most likely source" was a plausible guess but
is wrong. **Dequantizing the head to dense `mx::matmul` is therefore not a
determinism fix** — it is deterministic already, and it changes the logits away
from production's `nn.quantize` path. Not applied.

## Evidence that the nondeterminism is in the multi-step composition

Low-memory reduced configs (only a few layers loaded, so no 13 GB load and no
swap) isolated when it appears. Each case = 3 independent processes:

| config | step-0 token | digests |
|---|---|---|
| 1 prompt token, 48 layers | 59 (3/3) | identical (3/3) |
| 16 prompt tokens, 4/5/6 layers | stable | identical (3/3) |
| 16 prompt tokens, 7–11 layers | stable | **3 different digests** |
| 16 prompt tokens, 12/24/36 layers | **3 different tokens** | 3 different digests |
| full: 16 tokens, 48 layers | 212192 / 25625 / 96774 | 3 different digests |

Forcing `mx::eval(logits)` after every prefill step (to break the large lazy
graph) did **not** make it deterministic. Re-evaluating the same lazy logits
graph 5x inside one process gave identical hashes. So the variation is
cross-process and grows with layer count — consistent with a small numerical
difference accumulating through the composed stack, not with a single
nondeterministic kernel.

## Localization run (granted, 2 guarded runs, reduced 10-layer config)

Instrumentation added to `real_checkpoint_generate.cpp` behind
`REALGEN_DEBUG_HASHES=1` (off by default): hashes branch / hidden / MoE after
every layer plus final `mixed` and `logits`, and counts NaN/Inf, on the last
prefill step.

Guarded invocation (both runs, exact envelope):
```bash
REALGEN_DEBUG_HASHES=1 python3 scripts/flashnext/real_run_guard.py \
  --floor-gib 1.5 --swap-growth-mib 2048 --compressor-growth-mib 4096 \
  --log /tmp/determinism_guard/guard_runN.jsonl -- \
  /tmp/real_gen_localize <ckpt> <ref> 1 /tmp/real_config_L10_P16.txt
```
Guard result: both `returncode=0`, `stopped=None`, ~3–4 s each,
`available_floor_gib` 4.87 / 4.74 (floor 1.5), `swapout_mib=0`, compressor peak
963 / 997 MiB (limit +4096). No swap; no guard intervention.

Per-layer comparison of the 42 captured arrays:

* **Layers 0–5 (incl. GDN layers 0,1,2,4,5 and QSA layer 3): all 24 hashes
  IDENTICAL** across the two runs.
* **First divergence: `L06_branch_gdn`** (layer 6 is `linear_attention` / GDN).
  From there `L06_hidden_attn`, `L06_moe`, `L06_hidden`, all of layers 7–9,
  `final_mixed`, and `final_logits` diverge.
* **NaN/Inf count = 0 for all 42 arrays in both runs.**
* Step-0 token was 17324 in both; digests differed
  (`07a30017…` vs `c3198eb2…`).

Additional synthetic probes added after localization: `gdnsub` (fast::rms_norm,
rms_norm_gated, gdn_normalize_qk, compute_g_beta) and `reduce`
(`mx::mean(square(x),-1)` over 2560/10240 and `mx::mean(x,-2)`) — **all
deterministic** (5/5 identical hashes). `race` confirmed `mx::eval` is
synchronous with host-backed MoE buffers.

## Most likely cause / proposed fix

The divergence is in the **GDN path at layer 6**, not in the lm_head, not in
QSA (layer 3 QSA branch was bit-identical), not in MoE (layers 0–5 MoE outputs
identical), and not in any single MLX op (all isolated deterministically).
Because layers 0–5 are bit-identical at the last step, layer 6's input hidden
state is identical; the differing quantity is the **per-layer GDN cache state
(conv_state / recurrent_state)** threaded through the 16 sequential prefill
steps, or a value-dependent path inside the fused `mx::fast::metal_kernel`
gated-delta kernel for layer 6. This is consistent with the earlier threshold:
a 6-layer config (layers 0–5) was deterministic, adding layer 6 made it vary.

## Verification run of the lazy-aliasing fix — FAILED (hypothesis disproven)

Applied, behind `REALGEN_DEBUG_HASHES`, a tool-local `gdn_layer_forward_local`
that (a) replaces the fused `gated_delta_step_kernel` with a tool-local
ops-based delta step (the shared `engine.cpp` version calls
`mx::expand_dims(g,{-1,-1})`, which MLX 0.32.2 rejects, and needs Hk->Hv head
replication) and (b) `mx::eval`s conv_state and recurrent_state at every step
boundary. Two guarded 10-layer runs (same envelope, plus `--thermal-level 99`):

```
run A: returncode=0 stopped=None seconds=5 floor=5.06GiB swapout_mib=0
       step0 out=281    sha=60514eb380c3...
run B: returncode=0 stopped=None seconds=5 floor=4.64GiB swapout_mib=0
       step0 out=8821   sha=bfff846dcd9d...
```

(NB: one earlier invocation aborted in 1 s with `[expand_dims] Received duplicate
axes.` before any compute; the ops-step shape bug was fixed before runs A/B.)

**Result: the fix did NOT remove the nondeterminism.** All 42 arrays: layers 0–5
still bit-identical, **first divergence still `L06_branch_gdn`**, NaN/Inf still
0 everywhere. The step-0 token/digest still differ. Replacing the fused kernel
and adding materialization barriers changed the *values* (L05_hidden hash moved
from `50dd638d…` to `76220810…`) but not the *divergence pattern*. Therefore:

* the fused GDN `metal_kernel` is NOT the cause;
* lazy state aliasing / missing materialization barriers are NOT the cause;
* the divergence is upstream of the delta step in layer 6's input path.

**New clue:** layer 6 is the first layer whose GDN weights are loaded from
`model-00009-of-00022.safetensors` (layers 0–5 use shards 1/7/8). The leading
hypothesis is now in the weight-loading/dequantize path for that shard, or a
value-dependent reduction in `gated_residual` for layer 6's specific weight
values — not in the recurrent kernel. Next cheap diagnostic (no decode): hash
the dequantized layer-6 GDN weights across processes to see whether the loaded
weights themselves differ. No further fix was applied.

## Files touched

* NEW `cpp/engine/tools/determinism_probe.cpp` — synthetic op probe.
* NEW `cpp/engine/tools/check_determinism.py` — hash verifier.
* NEW this note.
* `real_checkpoint_generate.cpp` — **unchanged** (no proven fix to apply).
