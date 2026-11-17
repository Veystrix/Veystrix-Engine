# C++ decode path — implementation checklist

**Branch:** `speedup-roadmap`
**Method:** #1 (native / sync-removed decode path) — promoted to first implementation
**Target:** Qwen3.8-Flash-Next decode 2.3 → 5+ tok/s on 16 GB M4, bit-identical
**Reference:** `Jakevin/moe-offload` `mlx-qwen38/cpp/` — same model, bit-exact (token ids, raw logits every step, 122/122 cache tensors), 9.0–9.6 tok/s on M4 Pro
**Gate:** raw logits + full cache-tensor equality, not just final token digest

## Investigation
- [ ] Map the current Python decode path: where `mx.eval` / syncs occur per token and per layer
- [ ] Identify the minimal set of kernels/ops that must be fused into C++ to remove per-layer syncs
- [ ] Study the reference C++ runner (MLX C++ API) and what it ports vs the Python loader
- [ ] Determine whether Veystrix's model checkpoint format is compatible with the MLX C++ API loader
- [ ] Record the baseline: token digest, tok/s, physical MB/token, guard floor (requires run permission)

## Implementation
- [ ] Scaffold `cpp/decode/` with CMake target and MLX C++ API linkage
- [ ] Port expert I/O (explicit merged pread) into the C++ path
- [ ] Port decode kernels (attention, norm, router, Q4 affine G32 GEMV, shared expert)
- [ ] Port cache state handling with exact tensor-by-tensor parity
- [ ] Expose a small interface to the Python engine (opt-in flag, off by default)
- [ ] Add a bit-exact gate test: raw logits + all cache tensors vs Python path

## Validation (only with explicit run permission)
- [ ] Guard envelope check + ask permission
- [ ] A/B candidate vs current baseline, same prompt/seed/token count
- [ ] Compare token digest, raw logits, cache tensors, tok/s, physical MB/token
- [ ] If quality differs → drop method entirely, log in `DO_NOT_IMPLEMENT.md`
- [ ] If slower → keep flag off, record NO GAIN
- [ ] If faster + bit-exact → ACCEPTED, promote baseline, update `SPEEDUP_ROADMAP.md`

## Status
- **Current phase:** investigation
- **Blocked on:** hardware run permission (for baseline profile and A/B)
