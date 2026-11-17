# GDN per-layer drift localized: `_normalize_qk` + `compute_g_beta` dtype promotion

**Date:** 2026-11-18
**Scope:** REAL checkpoint `Qwen3.8-Flash-Next-MLX-4bit-MTP`, layer 20 (`linear_attention` / GDN),
16-token prefill, C++ batched path vs Python/MLX path. **No bit-exact claim outside the
measurements below.**

## Result in one line

The fused gated-delta Metal kernel is **bit-exact**. The first divergence inside the GDN
branch is `Qwen4ExpGatedDeltaNet._normalize_qk` (q/k L2+scale), followed by
`compute_g_beta` (softplus). Both diverge **only because the C++ primitives promote
bf16 operands to f32**, while the Python/MLX ops keep bf16. Feeding the kernel identical
inputs gives byte-identical `y` and recurrent state; applying the two fixes makes the
**entire GDN branch at layer 20 byte-identical** (maxdiff 0 for every intermediate).

## How it was measured

1. `cpp/engine/tools/dump_python_intermediates.py` (new file) hooks the *real* model's
   `Qwen4ExpGatedDeltaNet.__call__` at layer 20, captures the pre-forward GDN inputs
   (`mixed_input`, `conv_state`, `recurrent_state`), runs a replica of the real forward
   that calls the layer's real submodules, and dumps all intermediates as raw f32 (plus
   the uint32/bf16 weights needed by C++). The replica output is **bit-exact** against
   the genuine forward: `replica-vs-real maxdiff = 0.0`.
   - Hook note: the backend wraps `Qwen4ExpGatedDeltaNet.__call__` in
     `models/flashnext/linear_state.py` and captures the original base-class function at
     init, so patching `Qwen3_5GatedDeltaNet.__call__` is bypassed. Patch the concrete
     class's wrapper instead.
2. `cpp/engine/tools/probe_layer_intermediates.cpp` (new file) reads those dumps and
   replays each C++ primitive on the **identical Python inputs**, then compares every
   intermediate. Upstream ops (concat, conv1d+silu, split/reshape) came out 100% exact,
   which validates the probe.
3. Both runs used the owner's memory guard.

Run under guard:
```
cd /Users/vincent/Documents/Veystrix
.venv/bin/python scripts/flashnext/real_run_guard.py \
  --floor-gib 1.5 --swap-growth-mib 5120 --compressor-growth-mib 8192 \
  --thermal-level 99 --max-seconds 600 --log /tmp/xshard_guard/gdn_probe_l20.jsonl -- \
  .venv/bin/python <worktree>/cpp/engine/tools/dump_python_intermediates.py \
    --checkpoint /Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP \
    --out /tmp/gdn_probe_l20 --layer 20 --tokens 2
/tmp/probe_gdn /tmp/gdn_probe_l20
```

## Measurements (C++ vs Python, identical inputs, layer 20, S=16)

| Intermediate | maxdiff | corr | exact | verdict |
|---|---:|---:|---:|---|
| conv_input / conv_out / q_raw / k_raw | 0 | 1.0 | 100% | bit-exact |
| **q_norm (`_normalize_qk`)** | **4.52e-4** | 0.999995541 | 0% | **first divergence** |
| **k_norm (`_normalize_qk`)** | **5.34e-3** | 0.999996823 | 0% | **first divergence** |
| **g (`compute_g_beta`)** | **2.09e-3** | 0.999999953 | 0.39% | **second divergence** |
| beta (`sigmoid(b)`) | 0 | 1.0 | 100% | bit-exact |
| delta_y, kernel fed **Python** q/k/g/beta | **0** | 1.0 | **100%** | **kernel is bit-exact** |
| state, kernel fed **Python** q/k/g/beta | **0** | 1.0 | **100%** | **kernel is bit-exact** |
| delta_y / state, kernel fed C++ q/k/g | 1.28e-4 / 1.30e-3 | ~0.999995 | 0% | propagated from above |
| RMSNormGated output (`normed`) | 7.81e-3 | 0.999993474 | 54.58% | propagated |
| **GDN branch `out`** | **7.81e-3** | **0.999979431** | 27.19% | propagated |

The 7.8e-3 branch-level gap at layer 20 is consistent with the owner's full-run layer-20
hidden maxdiff ~0.014 (the branch is injected with hyper-connection weights up to 2 and
then passes through the MLP block).

## Root cause #1: `gdn_normalize_qk` (cpp/engine/src/engine.cpp:223-230)

Current C++ casts q/k to f32 and normalizes in f32, returning f32. Python keeps bf16
all the way (`mx.square`, `mx.sum`, `+1e-6`, `rsqrt`, `q*rsqrt` all return bf16 for a bf16
input, and `q * scale` with a Python float also stays bf16). The checkpoint's real
conv output is bf16, so C++ and Python round at different points.

Note the Qwen4 override applies `scale` **only to q**, not k
(`language.py:266-272: return q * scale, k`) — the current C++ also does this, so only
the dtype needs fixing.

Fix (dtype-generic, keeps the f32 synthetic path bit-exact too):

```cpp
std::pair<mx::array, mx::array> gdn_normalize_qk(const mx::array& q,
                                                 const mx::array& k) {
  const double scale = std::pow(static_cast<double>(q.shape().back()), -0.5);
  auto norm = [&](const mx::array& t) {
    // Python: t * rsqrt(sum(t*t, keepdims) + 1e-6), all in t.dtype().
    // C++ array/float promotes bf16 -> f32, so the scalar must be an ARRAY of
    // the same dtype (1e-6f -> bf16 rounds exactly as MLX's Python path does).
    mx::array eps = mx::array(1e-6f, t.dtype());
    mx::array y = t;
    y = y * mx::rsqrt(mx::sum(mx::square(y), -1, /*keepdims=*/true) + eps);
    return y;  // caller applies scale to q only
  };
  mx::array qn = norm(q);
  mx::array kn = norm(k);
  mx::array sc = mx::array(scale, q.dtype());
  return {qn * sc, kn};
}
```

## Root cause #2: `compute_g_beta` (cpp/engine/src/engine.cpp:139-148)

Current C++: `mx::logaddexp(a + dt_bias, mx::array(0.0f))`. The f32 zero promotes the
bf16 `a+dt_bias` to f32, so the softplus is computed (and rounded) in f32. Python calls
`nn.softplus(a + dt_bias)` on bf16 (both `a` and `dt_bias` are BF16 in the checkpoint) and
only promotes to f32 when multiplying by `exp(A_log)`.

Verified: `nn.softplus(x) == mx.logaddexp(x, mx.array(0, dtype=mx.bfloat16))` exactly
(maxdiff 0), whereas `logaddexp(x, f32(0))` differs by up to 8e-3.

Fix (works for both bf16 and f32 inputs — with f32 `a`, MLX promotes back to f32):

```cpp
std::pair<mx::array, mx::array> compute_g_beta(const mx::array& A_log,
                                               const mx::array& a,
                                               const mx::array& b,
                                               const mx::array& dt_bias) {
  mx::array A = mx::exp(mx::astype(A_log, mx::float32));
  mx::array sp = mx::logaddexp(a + dt_bias,
                               mx::array(0.0f, (a + dt_bias).dtype()));
  mx::array g = mx::exp(-A * sp);
  return {g, mx::sigmoid(b)};
}
```

## Verification of the fix

With only these two primitives changed (implemented as FIX variants inside the probe,
so no shared file was edited), every intermediate from `q_norm` through the final GDN
branch output is byte-identical to Python on the real layer-20 inputs:

```
FIX q_norm C++ vs Python       maxdiff=0  corr=1.000000000 exact=100.00%
FIX k_norm C++ vs Python       maxdiff=0  corr=1.000000000 exact=100.00%
FIX g C++ vs Python            maxdiff=0  corr=1.000000000 exact=100.00%
FIX delta_y C++ vs Python      maxdiff=0  corr=1.000000000 exact=100.00%
FIX state C++ vs Python        maxdiff=0  corr=1.000000000 exact=100.00%
FIX normed C++ vs Python       maxdiff=0  corr=1.000000000 exact=100.00%
FIX out (branch) C++ vs Python maxdiff=0  corr=1.000000000 exact=100.00%
```

The fused kernel source, grid `(32, Dv, B*Hv)`, threadgroup `(32,4,1)` and template args
already match Python; no kernel/tiling/constant change is needed.

## Honest scope / next steps

- This measurement isolates the GDN **branch** at layer 20 with identical inputs. It
  proves the two dtype promotions above are sufficient to make that branch bit-exact.
- It does **not** re-measure the attention hyper-connection, MoE combine, shared expert,
  or full-attention SDPA on the real checkpoint (owner previously verified those in
  isolation). After applying these two fixes, a full 48-layer C++ vs Python run is the
  decisive end-to-end check for remaining sources.
- Files created (no shared source file edited, no commit):
  - `cpp/engine/tools/probe_layer_intermediates.cpp`
  - `cpp/engine/tools/dump_python_intermediates.py`
  - `cpp/engine/tools/check_gdn_dtype_divergence.py` (synthetic, no checkpoint)
  - this report.
