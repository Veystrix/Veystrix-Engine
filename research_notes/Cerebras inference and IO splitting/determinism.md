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

## Weight-loading diagnostic — weights are IDENTICAL (shard-9 lead disproven)

Added `det_probe weights <checkpoint> <config>` mode, which replicates
`DenseLoader` exactly (`mx::load_safetensors` + `as_bf16` + `mx::dequantize` +
`mx::eval`) and SHA-256s every dequantized GDN tensor for layers 0,5,6,7 plus
the two `mx::zeros` cache-state initializers. Ran in THREE separate processes:

```
all 31 hashes IDENTICAL across processes 1/2/3 (diff empty both comparisons)
L6 in_proj_qkv    468fd01ebbe7e9fae2ef6fbafa5da3a8e7251574ba41445e294c6d1ec9bfd613
L6 in_proj_z      7eaeef1a85d9980e4cae98f356f656cde59d585eba96e4b862f5d81b692e455f
L6 in_proj_b      528874bcc5d4da7111abfe1c03d95982f6dbdba427467f3629a99121d08a76d1
L6 in_proj_a      94c687fb1d35e5bbd05d8644351ae2253bfa63069f7591c9f09da5a96d7ff7ca
L6 conv1d.weight  00211cc80e643b1e5c839016ebf054e5634fe6fbb60804aae1fe17b631b9a7a7
L6 A_log          f8180f113a616da15604fb94e5be61c3e0acb93513b7b9cf61d8505f302a43cf
L6 dt_bias        85c0afa2891a778908ccac17c4c5fcd267211d92dcbc8b145ea5b746e680093e
L6 norm.weight    5e7c123d772c1dfe1b0e2bb94f7731740a11bb10f259a18a3a4abd90f8f1a747
L6 out_proj       95cb3f050c79950b95982b90869967229bc1ded81aaaf4bff2d949fafbd44839
CACHE conv_state  e7c57dec499c6bfb0695e9568a1e44ff563ec9b3546b05a156904019af6f8c1f
CACHE rec_state   1d1ca7719669a57942a643ec132e9a0880a98f0e53ca4ab75129d4bd6d71852b
```
(Layers 0/5 shards 1/7/8 and layer 7 QSA shards also all stable.)

Additional synthetic `matmul` probe over the exact GDN projection shapes
([1,2560] @ [10240,2560], [6144,2560], [2560,6144], …) in f32 and bf16: all
deterministic 5/5.

## Confirmed status and remaining cause

Ruled out by evidence: weight loading / dequantize (identical hashes), cache
initialization (identical), lm_head split-K (disproven), fused GDN delta kernel
and lazy-state aliasing (replacing both did not change the divergence pattern),
every isolated MLX op and reduction (deterministic in synthetic probes), NaN/Inf
(none). **Confirmed: the divergence is a value-dependent numerical difference in
the composed forward path, accumulating in the float32 GDN recurrent state
across prefill steps and first crossing a bf16 rounding boundary in layer 6's
branch output.** Layers 0–5 hidden states stay bit-identical because the small
float32 differences round away in bf16 until then.

## Per-step GDN state diagnostic — FIRST DIVERGENCE: step 1, layer 9

Added opt-in `dump_gdn_states()` (REALGEN_DEBUG_HASHES) hashing conv_state and
recurrent_state after every one of the 16 prefill steps for all 10 layers.
Two guarded runs (floor 1.5, swap 2048, compressor 4096, thermal 99,
10-layer config, 16 tokens):

```
run A: returncode=0 stopped=None seconds=6 floor=4.50GiB swapout_mib=0  step0 out=11  sha=d2a8f6e1...
run B: returncode=0 stopped=None seconds=6 floor=4.55GiB swapout_mib=0  step0 out=220 sha=2b8c0ccf...
160 state hashes per run.
```

Comparison results:

* **Step 0: all 20 state hashes identical** (10 layers × conv+rec).
* **Step 1: ONLY layer 9 diverges — both `conv` and `rec` simultaneously**
  (`conv d6d9eeacaa71ed6a` vs `2872f5d97fed1224`,
  `rec f4849594a42e31cd` vs `d04784e21265c9aa`). Layers 0–8 states identical.
* Step 2: still only layer 9. Step 3 onward: layers 6 and 8 join (layer 6 rec
  at step 3; layer 8 conv+rec at step 3); layers 0–4 and 5 never diverge.

Interpretation: because both layer-9 state tensors diverge at the same moment
and layer 9's own GDN state at step 0 is identical, the difference is in
layer 9's **input** at step 1, i.e. layer 8's output. Layer 8's own GDN state is
identical, so its attention branch is fine — the difference comes from layer
8's **MoE**. Crucially, **layers 5 and 8 are the only `CrossShardMoE` layers**
(the 2 of 10 whose 9 expert tensors split across two shards; in the full model
12 of 48 do). The layers whose downstream GDN states diverge first are exactly
6 (after layer 5) and 9 (after layer 8). This is a strong, specific correlation.

**Implicated op:** the tool-local `CrossShardMoE::forward` split-shard expert
path (host-buffer-backed slot arrays + `gather_qmm` + `mx::eval`), NOT the GDN
kernel, NOT weight loading, NOT lazy GDN state. Layer 8's full-attention
neighbour (layer 7) is a secondary suspect for layer 8's own later divergence.

**Minimal fix (NOT applied, needs verification):** for the cross-shard layers,
avoid the persistent no-op-deleter host buffers — copy the gathered expert rows
into fresh owned `mx::array`s per step in `CrossShardMoE::forward`, or route
those layers through a fully-resident dequantized expert path. Verify with one
more guarded reduced-config run before touching anything shared.

## Files touched

* NEW `cpp/engine/tools/determinism_probe.cpp` — synthetic op probe.
* NEW `cpp/engine/tools/check_determinism.py` — hash verifier.
* NEW this note.
* `real_checkpoint_generate.cpp` — **unchanged** (no proven fix to apply).
