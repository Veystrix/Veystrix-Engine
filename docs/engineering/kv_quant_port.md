# KV-cache tier quantization primitives — C++ port (foundation stage)

**Status (measured):** bit-exact C++ ↔ Python parity for the packing/dequant
primitives, on CPU, with synthetic golden vectors. Full prefill attention
integration is **not** done — see "Remaining work".

All numbers below are **measured** on this machine (mlx 0.32.2, Apple Silicon,
`/Users/vincent/Documents/Veystrix/.venv`), not estimates. Python behavior claims
about the reference ops are **cited** to `models/flashnext/quantized_kv.py` and
`models/flashnext/mixed_kv.py`.

## Files added / changed

| File | Role |
|---|---|
| `cpp/engine/src/kv_quant.hpp` / `.cpp` | `KVPacked`, `kv_quantize`, `kv_dequantize`, `Tiered4KV` |
| `cpp/engine/tools/gen_kv_quant_vectors.py` | synthetic, CPU-only golden vectors from the **real** Python ops/modules |
| `cpp/engine/tests/test_kv_quant.cpp` | byte-for-byte comparison (packed/scales/biases + dequant) |
| `cpp/engine/CMakeLists.txt` | add `kv_quant.cpp` to lib, `test_kv_quant` target, guarded `add_test` |
| `cpp/engine/tests/vectors_kv_quant/` | generated golden vectors (regenerate with the tool above) |

## Dataflow

```
bf16 K/V tensor x [..., T, D]        (synthetic, fixed seed)
        │
        ├─ kv_quantize(x, gs, bits)  ──> mx::quantize(x, gs, bits, "affine")
        │                                 ├─ data    uint32  [..., T, D/(32/bits)]
        │                                 ├─ scales  bf16    [..., T, D/gs]
        │                                 └─ biases  bf16    [..., T, D/gs]
        │
        └─ kv_dequantize(data,scales,biases,gs,bits)
                                      ──> mx::dequantize(...)  -> bf16 [..., T, D]
```

Affine convention (cited: `mixed_kv.py` fused-kernel source lines 54–57 and
76–87): `r = float(scale) * q + float(bias)`, rounded **once** to bfloat16.
MLX's `mx::dequantize` implements this directly, so the C++ wrapper does not
re-implement the arithmetic.

`tiered4` (`Tiered4KV`, mirrors `MixedPrecisionQSAKVCache(mode="tiered4")`,
cited: `mixed_kv.py` lines 399–442):

```
append(keys, values):
  history tier  <- mx.quantize(original keys/values, gs, bits=4)   # aged-out rows
  recent  tier  <- mx.quantize(original keys/values, gs, bits=8)   # last `recent_tokens`
  recent_n = (offset - history_tokens) + T
  aged     = max(0, (recent_n - recent_tokens) // 256 * 256)
  if aged: drop first `aged` INT8 rows; history_tokens += aged
  offset += T
```

Both tiers are written from the **original activations** (no INT8→INT4
cascade) — this is the property `mixed_kv.py` calls out in its module docstring.
History storage in Python uses `HISTORY_GROWTH=4096` capacity pages; the C++
container stores arrays at their logical token length because capacity is a pure
allocation detail and does not affect any packed byte. The exposed
`history_keys()/history_values()` are the slice `[0, history_tokens)` — exactly
the rows attention reads via `old_at = min(indices, history_tokens - 1)`.

## Golden vectors (synthetic, CPU only)

Regenerate:

```
/Users/vincent/Documents/Veystrix/.venv/bin/python \
    cpp/engine/tools/gen_kv_quant_vectors.py
```

Fixed seed 20261118; no checkpoint or benchmark is read. Cases (measured
shapes/dtypes from the generator output):

| Case | Input | gs | bits | packed | scales/biases | dequant |
|---|---|---|---|---|---|---|
| `affine4` | bf16 `[1,2,4,128]` | 64 | 4 | uint32 `[1,2,4,16]` | bf16 `[1,2,4,2]` | bf16 `[1,2,4,128]` |
| `affine8` | bf16 `[1,2,4,128]` | 64 | 8 | uint32 `[1,2,4,32]` | bf16 `[1,2,4,2]` | bf16 `[1,2,4,128]` |
| `affine4_wide` | bf16 `[1,1,3,256]` | 64 | 4 | uint32 `[1,1,3,32]` | bf16 `[1,1,3,4]` | bf16 `[1,1,3,256]` |
| `tiered4` | bf16 chunks `[128,256,128]` | 64 | 4 hist / 8 recent | hist `[1,2,256,16]`, recent `[1,2,256,32]` | bf16 `[1,2,256,2]` | bf16 `[1,2,256,128]` |

The tiered case drives the real `MixedPrecisionQSAKVCache`; final state is
`offset=512`, `history_tokens=256`, recent window 256 (the minimum legal value;
production default 8192 only changes the aging point, not the pack numerics).
The affine cases additionally cross-check `AffineQuantizedQSAKVCache.dequantize()`
against the standalone dequant bit-for-bit in the generator itself.

## Investigation: C++ vs Python `mx::quantize` — device, not implementation

**Measured** (same fixed bf16 tensor, same mlx 0.32.2 build, Python):

| comparison | differing elements |
|---|---|
| CPU `quantize` packed words vs GPU `quantize` | 1 / 128 |
| CPU scales vs GPU scales | 0 / 16 |
| CPU biases vs GPU biases | 0 / 16 |
| CPU `dequantize` output vs GPU `dequantize` | 569 / 1024 |

So the first C++ test run (C++ default device = GPU vs CPU-generated golden)
showed 1-ULP mismatches — **the C++ op is not different from the Python op**;
the two were evaluated on different backends. **Measured** fix: the test calls
`mx::set_default_device(mx::Device(mx::Device::DeviceType::cpu))` to match the
CPU-only generator. With that, the test is bit-exact.

Exact final test output (measured):

```
  affine4: gs=64 bits=4 packed/scales/biases/dequant OK
  affine8: gs=64 bits=8 packed/scales/biases/dequant OK
  affine4_wide: gs=64 bits=4 packed/scales/biases/dequant OK
  tiered4: chunks=3 offset=512 history_tokens=256 (INT4 history + INT8 recent) OK
test_kv_quant: BIT-EXACT PASS (affine4/affine8/affine4_wide pack+dequant, tiered4 history/recent split)
```

`ctest -R kv_quant` → `1/1 Test #19: kv_quant .... Passed` (measured).

**Consequence for later integration:** golden vectors must be generated on the
same backend that the C++ attention path will use (CPU here). If a future stage
compares a GPU C++ attention path against these vectors, regenerate with GPU
`mx.set_default_device(mx.gpu)` or add device-matched vectors; do not treat the
1-ULP GPU/CPU drift as a correctness bug in the primitive.

## Remaining work (full attention integration — not in this stage)

1. Wire `KVPacked` into the QSA cache update path: append newly projected K/V as
   INT4 history + INT8 recent, track `history_tokens`/`offset`, and implement
   page aging exactly as `MixedPrecisionQSAKVCache.update_and_fetch`.
2. Implement `gather_selected` over packed rows (cited: `mixed_kv.py` lines
   444–475): gather packed words/scales/biases, dequantize per selected row, and
   `mx.where(old, hk, rk)` between the INT4 history and INT8 recent tiers.
3. Feed dequantized (or fused-kernel) K/V into SDPA; validate against Python
   `gather_selected` + `mx.fast.scaled_dot_product_attention` with bit-exact
   vectors (the Python fused-SDPA path claims bit-identity to that reference).
4. Snapshot/restore of the tiered state (schema version in `mixed_kv.py` lines
   517–589) if C++ needs prefix-cache persistence.
5. Regenerate device-matched vectors if/when the parity target moves to GPU.
