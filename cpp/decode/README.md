# C++ decode path (Method #1, first from the speedup roadmap)

Goal: a bit-exact native decode path for Qwen3.8-Flash-Next (`qwen4_exp`) on MLX C++,
removing per-layer Python host syncs. This is **research code, off by default**; it is not
wired into the engine yet.

## Toolchain status — VERIFIED 2026-11-17

`cmake -S . -B build && cmake --build build -j` succeeds against the **same MLX 0.32.2**
the Python engine uses. `./build/mlx_probe` output:

```
6 8 10 12
mlx 0.32.2, metal=1, mlx_probe OK
```

Notes:
- The C++ namespace is `mlx::core` (Python's `mx` is an alias): `namespace mx = mlx::core;`
- `MLX_ROOT` defaults to the repo venv's MLX package so kernels are bit-identical.
- CMake 4.4.3 emits a harmless macOS-14 / built-for-26.2 deployment-target warning.

## Critical finding before porting the full layer stack

Local profiling infrastructure already exists and contradicts the premise that host sync
is the top lever here:

- `models/flashnext/hostwindow.py:7-8`: deliberately adding `mx.eval(scores, inds)`, a
  `tolist()` and host list work measured **498.5 ms/token vs 499.3 ms/token baseline — free**,
  because both the drive and GPU were idle during that host work.
- `models/flashnext/adaptive_topk.py:89-103`: production already does **one sync per decode
  layer** (`FLASHNEXT_ONE_SYNC=1`, set in `settings/launch.py:53`), bringing `inds` over with
  `scores` so the expert block skips its own eval.
- The reference repo's 41%-sync profile came from its own Python loader, which is a different
  implementation from Veystrix's already-optimized one.

Therefore the full 330 KB port (reference `mlx-qwen38/cpp/`: loader 82 KB, kernels 61 KB,
engine 53 KB, I/O scheduler 39 KB, layers 34 KB) may deliver little here. The gating step is
to measure Veystrix's actual per-token sync/SSD/GPU split with the existing profiler
(`FLASHNEXT_HOST_WINDOW=1`) before committing to the port. **That requires a guarded hardware
run and explicit owner permission.**

## Precise sync sites (from the decode-path map)

Per token (`veystrix/backends/flashnext.py`):
- `:1270` `token.item()` — GPU→host sync
- `:1302` `mx.eval([entry.state for entry in self.cache])` — final token
- `:1309` `mx.eval(token)`

Per layer (the "41%" analogue):
- `models/flashnext/adaptive_topk.py:95-103` router `mx.eval(scores, inds)` / `mx.eval(scores)`
- `models/flashnext/expert_cache.py:787-788` `build_plan`: `mx.eval(flat)` + `flat.tolist()`
- `models/flashnext/expert_cache.py:1781-1787` `StreamingSwitchGLU`: `mx.eval(flat)` + `routed.tolist()`

Compute to port: `.venv/.../mlx_vlm/models/qwen4_exp/language.py:725-829` (layer loop),
GDN `:127-260`, QSA `models/flashnext/qsa_chunk.py:222-274`.
I/O to port: `models/flashnext/store.py` (`SafeTensorStore`, mmap + `pread`/`preadv`),
`models/flashnext/native_read.c` (libdispatch batched pread, GIL-releasing template).
No MLX C++ safetensors loader exists — it must be reimplemented.

## Milestones
- [x] Build toolchain verified (`mlx_probe`)
- [ ] Measure actual sync share with `FLASHNEXT_HOST_WINDOW=1` (needs run permission)
- [ ] Decide: full port vs narrower native I/O/router kernel
- [ ] Bit-exact parity harness: raw logits + all cache tensors vs Python path
- [ ] Opt-in integration behind a flag, off by default
- [ ] Guarded A/B + digest gate
