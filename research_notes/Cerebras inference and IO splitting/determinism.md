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

## Open question / proposed next step

Every candidate op is deterministic in isolation, including with NaN inputs and
with host-buffer reuse. The divergence therefore arises only in the full
composed path. To pinpoint it without repeated 13 GB loads, the next diagnostic
should be a SINGLE instrumented run (under the owner's guard) that, during the
last prefill step, materializes and hashes the hidden state after each of the 48
layers plus the final `mixed` and `logits`, and counts NaN/Inf. The first layer
whose hidden hash is reproducible-but-different (or where NaN first appears)
identifies the responsible op/composition. Two such runs are enough to localize
it. I have not run this yet at the owner's request.

## Files touched

* NEW `cpp/engine/tools/determinism_probe.cpp` — synthetic op probe.
* NEW `cpp/engine/tools/check_determinism.py` — hash verifier.
* NEW this note.
* `real_checkpoint_generate.cpp` — **unchanged** (no proven fix to apply).
