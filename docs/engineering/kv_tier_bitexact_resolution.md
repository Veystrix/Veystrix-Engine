# KV-tier bit-exactness resolution: the GPU-vs-CPU 1-ULP gap

**Verdict (measured + cited):** The GPU↔CPU difference is an **irreducible MLX
backend implementation difference inside `mx.dequantize`** (different number of
bf16 rounding steps), not a defect in the C++ primitives or the generator writer.
On the **same device** (GPU, which is what real serving uses) the KV-tier output
is **bit-exact** between the C++ generator path and the Python reference — the
existing `VEYSTRIX_CPP_KV_TIER_SELFTEST` proves this with exact output below.
No change to `kv_quant.{hpp,cpp}`, the writer, or `CMakeLists.txt` is needed;
changing them would break the "same MLX op as Python" contract and would only
move the discrepancy.

Every number below is **measured** on this machine (Apple Silicon, mlx 0.32.2,
`/Users/vincent/Documents/Veystrix/.venv`), unless labelled **cited** (source
code inspected) or **estimate**. All claims of bit-exactness show exact outputs.

---

## 1. Exact cause — which op, which rounding

### 1.1 `mx.quantize`: scales/biases are device-independent; packed data differs only at ties

**Measured** (`cpp/engine/tools/check_kv_tier_parity.py --device-gap`, fixed seed
20261118, shape `[1,2,4,128]`, group 64):

```
affine4: quantize scales diff=0/16, biases diff=0/16, packed-word diff=0/128
affine8: quantize scales diff=0/16, biases diff=0/16, packed-word diff=3/256
```

So the scale and bias arrays are **bit-identical** across devices. The rare
packed-word differences come from the rounding function at exact half-way points:

* **Cited** — GPU kernel `affine_quantize`
  (`mlx/backend/metal/kernels/quantized.h:2622-2724`, installed at
  `.../site-packages/mlx/include/mlx/backend/metal/kernels/quantized.h`) uses
  MSL `round(...)` (round-half-away-from-zero):
  `q0 = round(edge / scale)` and `val = min(round((w - bias) / scale), n_bins)`.
* **Cited** — CPU implementation `quantize<T,U>`
  (`mlx/backend/cpu/quantized.cpp:1158-1220`) uses `std::rint(...)` (round-half-
  to-even): `q0 = std::rint(edge / scale)` and
  `w_el = std::rint((w_el - bias) / scale)`.

These agree except at exact `.5` values, hence only a few packed words differ.
This is a minor, data-dependent effect.

### 1.2 `mx.dequantize`: the dominant difference — ONE bf16 rounding on GPU vs TWO on CPU

This is the real cause of the ~1-ULP gap.

* **GPU — cited** (`mlx/ops.cpp:5335-5343`): for `s.device == gpu`,
  `affine_dequantize` creates a `fast::Quantize` primitive with `dequantize=true`
  and dispatches the fused Metal kernel `affine_dequantize`
  (`mlx/backend/metal/kernels/quantized.h:2727-2786`). Its body is
  (`T` = output dtype = bf16):

  ```metal
  T scale = scales[gindex];   // bf16
  T bias  = biases[gindex];   // bf16
  out[i]  = scale * d + bias; // d = integer quantization level
  ```

  `d` is an integer and the operands are promoted to float32 for the
  arithmetic; the single store to `out[i]` (bf16) performs **one**
  round-to-nearest-even. Net effect: `round_bf16(f32(scale·q + bias))`.

* **CPU — cited** (`mlx/ops.cpp:5291-5345`): there is no fused kernel; the op
  falls back to a decomposition:

  ```cpp
  w = multiply(w, expand_dims(scales, -1));  // uint32 packed * bf16 scales
  w = add(w, expand_dims(biases, -1));       // + bf16 biases
  ```

  **Measured**: `mlx.multiply(mx.array([3],uint32), mx.array([0.1],bf16)).dtype
  == mlx.core.bfloat16`. MLX type promotion gives `uint32 * bf16 -> bf16`, so
  the product is **rounded to bf16**, then the addition is **rounded to bf16**
  again. Net effect: `round_bf16( round_bf16(f32(scale·q)) + bias )` — **two**
  bf16 roundings.

The doc `kv_quant_port.md` ("r = float(scale)*q + float(bias), rounded once")
describes the **GPU** kernel correctly; it is not the CPU path's behavior.

**Measured proof of the two models** (script reconstructs `q`, bf16 `scale`,
bf16 `bias` from the byte-identical affine4 golden and simulates both schemes):

```
quantize byte-identical? True True True
CPU-single-round model vs measured CPU: 477 diffs      (model is WRONG for CPU)
GPU-double-round model vs measured GPU: 477 diffs      (model is WRONG for GPU)
single-round model vs measured GPU:      0 diffs       (GPU = ONE rounding)
double-round model vs measured CPU:      0 diffs       (CPU = TWO roundings)
mlx uint32*bf16 result dtype: mlx.core.bfloat16
```

Two bf16 roundings vs one differ by ~1 bf16 ULP for most elements. In
near-cancellation cases (`scale·q ≈ -bias`) the first rounding changes which
side of zero the result lands on, so the difference can be much larger than one
ULP — **measured** affine4 examples with identical inputs:

```
idx=512: x=-0.0176  cpu=0x0000 (+0)        gpu=0xbb40 (-0.00292969)
idx=643: x=-0.0580  cpu=0x0000 (+0)        gpu=0xbb80 (-0.00390625)
```

### 1.3 Aggregate measured gap

`cpp/engine/tools/check_kv_tier_parity.py --device-gap` (reproducible):

```
affine4: quantize scales diff=0/16, biases diff=0/16, packed-word diff=0/128
  dequantize CPU vs GPU: 574/1024 bf16-bit outputs differ
  (559 within <=8 bf16 bits, 15 larger near cancellations)
  quantize output byte-identical => dequant diff is PURELY the dequantize kernel
affine8: quantize scales diff=0/16, biases diff=0/16, packed-word diff=3/256
  dequantize CPU vs GPU: 639/1024 bf16-bit outputs differ
  (597 within <=8 bf16 bits, 42 larger near cancellations)
```

Because the affine4 quantize triplets are **byte-identical**, the 574/1024
dequant difference is entirely attributable to the dequantize kernel — this is
the decisive isolation.

---

## 2. Same-device (GPU↔GPU) KV-tier parity — BIT-EXACT (measured)

The generator writer calls the same primitives the selftest drives:
`phase1::kv_roundtrip` (affine4/affine8) and `Tiered4KV` (tiered4, via the same
`append` + `dequantize(history) ‖ dequantize(recent)` path). Goldens are
generated on the GPU with the real Python ops
(`mx.quantize`/`mx.dequantize`, `MixedPrecisionQSAKVCache`) and the C++ binary
runs the same functions on the GPU.

Command (no checkpoint → no run guard needed):

```
cd /Users/vincent/Documents/Veystrix
.venv/bin/python <worktree>/cpp/engine/tools/check_kv_tier_parity.py \
    --out /tmp/kvparity_kvreal --binary /tmp/real_gen_p1
```

Exact output (measured):

```
device=Device(gpu, 0) (goldens generated on GPU)
wrote golden vectors under /private/tmp/kvparity_kvreal
[selftest] affine4  gs=64 bits=4 elems=1024 diffs=0 maxabs=0 BIT-EXACT PASS
[selftest] affine8  gs=64 bits=8 elems=1024 diffs=0 maxabs=0 BIT-EXACT PASS
[selftest] tiered4  gs=64 bits=4 elems=131072 diffs=0 maxabs=0 BIT-EXACT PASS
kv_tier_selftest: BIT-EXACT PASS

selftest exit=0
```

This is the property that matters for real serving: C++ and Python both run on
GPU and produce **identical K/V arrays** for every KV-tier mode.

### Why the writer's call pattern does not introduce a divergence

* **Cited** — Python affine cache stores packed data and dequantizes the whole
  cache at gather time (`models/flashnext/quantized_kv.py:147-165`). The C++
  writer quantizes the finished cache once in `prime_full` and each new decode
  chunk separately in `append`. MLX groups are along the feature dimension D
  (group_size 64), **not** the token dimension, so quantizing a chunk alone vs.
  as part of a whole tensor yields identical packed bytes; `dequantize` is
  elementwise, so results are identical regardless of partitioning. No
  divergence.
* **Cited** — Python tiered4 `gather_selected`
  (`models/flashnext/mixed_kv.py:444-475`) unpacks via `mx.dequantize` (or a
  fused kernel documented bit-identical to it) and selects history vs recent
  with `mx.where`. The C++ writer materializes
  `concat(dequantize(history), dequantize(recent))`. The tiered4 selftest above
  (all 512 tokens selected, 131072 outputs) is bit-exact, confirming the
  numerics match. `HISTORY_GROWTH=4096` is a capacity detail and affects no
  packed byte (**cited**: `kv_quant_port.md`).

---

## 3. Why the gap is irreducible — no fix applied

1. The C++ wrappers call `mx::quantize` / `mx::dequantize` **directly** — the
   same primitives Python calls (**cited**: `kv_quant.cpp:45-70`). Therefore
   C++(GPU) == Python(GPU) and C++(CPU) == Python(CPU), each bit-exact. There is
   no C++-side rounding to "fix".
2. Forcing cross-device equality would require reimplementing one device's
   rounding inside `kv_quant.cpp` (e.g. hand-rolling the CPU double-round on
   GPU). That would (a) no longer be the same MLX op, and (b) make GPU C++
   **disagree** with GPU Python — directly breaking the serving-relevant claim.
   Rejected by design.
3. The difference is in Apple's MLX kernels/decomposition, not in the Veystrix
   repo; it cannot be removed from this worktree. It is documented here as
   measured + cited rather than claimed as success.

**Files changed in this worktree:** this document and
`cpp/engine/tools/check_kv_tier_parity.py` (added a read-only `--device-gap`
characterization flag; default selftest behavior unchanged — re-verified
bit-exact after the edit). `kv_quant.{hpp,cpp}`, the writer in
`real_checkpoint_generate.cpp`, and `CMakeLists.txt` were **not** modified and
need no change. No commit was made.

---

## 4. Scope note — generator-level logit/token parity

The bit-exact claim here is about the **KV-tier arrays** the writer hands to
SDPA (what the selftest drives). Full generator logit/token identity between C++
and Python still has pre-existing, **non-KV** gaps — PLE, quantized-vs-dense
matmul, unfused RoPE, dense-vs-int8 KV, sequential-vs-chunked prefill
(**cited**: `docs/engineering/kv_tier_mtp_integration.md`). Those are out of
scope for this task and are **not** claimed bit-exact. No real-checkpoint run was
performed for this resolution: the same-device KV-tier claim is established by
the checkpoint-free selftest above, and the cross-device claim is a documented
characterization, not a gate.
