# maderix/ANE — actual source-code deep dive

**Research date:** 2026-11-17
**Primary source:** `git clone https://github.com/maderix/ANE`, full history fetched, HEAD = `d91c984` ("Update README.md", MIT, manjeet singh). Work copy: `/private/var/folders/37/fn6fxqyd2rgfmvc4954dggbm0000gn/T/opencode/ANE`. No Veystrix code was touched.
**Secondary sources read in full:** the four articles at `maderix.github.io/articles/inside-the-m4-ane-part-{1,2,3,4}`; Orion paper (arXiv:2603.06728); `thebasedcapital/ane-infer` (`docs/ane-internals.md`); `skyfallsin/ane.cpp` + `apple-neural-engine-field-guide` (`tests/README.md`); Pradeep Elankumaran's article (pradeep.md, 2026-03-30); GitHub issue #44.

---

## 0. CRITICAL CORRECTIONS BEFORE ANYTHING ELSE

1. **`gpu_ane_share.m` and `gpu_prefill_ane_decode.m` DO NOT EXIST.** The README lists them at lines 120–121 of `README.md`:

   ```
   ├── gpu_ane_share.m             # GPU↔ANE zero-copy IOSurface demo
   ├── gpu_prefill_ane_decode.m    # GPU prefill → ANE decode pipeline
   ```

   and reports results for them at `README.md:210-215`:

   ```
   GPU↔ANE inference pipeline (M4, seq=256):
   | Model         | GPU Prefill | ANE Decode | Total  |
   | Stories110M   | 6.7ms       | 1.9ms      | 8.8ms  |
   | Qwen3-0.6B    | 9.7ms       | 2.3ms      | 12.0ms |
   ```

   I verified exhaustively: `git rev-list --all | xargs git ls-tree -r --name-only` (every tree of every commit, including deleted files) and `git log --all --diff-filter=ACDMRT --name-only` return **no such paths**. They were never committed. Likewise `bridge/Makefile` has a `test` target for `test_bridge.m`, which is also absent. **Question 1 therefore cannot be answered from maderix code; the README numbers are unreplicated, source-less claims.** I give the closest real evidence below and flag it as such.

2. **Direction of the pipeline is inverted in the README vs. every working implementation.** README says "GPU prefill → ANE decode". Part 2 and Part 4 of the author's own articles say the opposite: *"ANE for prefill and anything over ~25 µs of matrix work; CPU/SME for single-token decode where the 90 µs floor would dominate."* `ane-infer` (issue #44) is explicitly "hybrid ANE+Metal inference engine … 32 tok/s **GPU decode**" with ANE prefill; `ane.cpp` likewise uses ANE only for GEMMs and CPU for attention/norm. Treat the README table as unsubstantiated.

3. **"INT8/INT4 weights fail to compile via `_ANEInMemoryModel`" is false for INT8.** `ane_int8_bench.m` compiles int8 weights through `constexpr_affine_dequantize` successfully. INT4 has no support anywhere. Details in §3.

4. **The "S≥16/32 all-zeros cliff" is not in maderix source or articles.** The documented symptom is a hard eval error **`0x1d`**, not silent all-zeros. The numbers (S≥16 / W≥32 / ~49 KB) come from Orion and the skyfallsin field guide, not maderix. Details in §4.

---

## 1. GPU↔ANE shared IOSurface (the missing files, and the closest real evidence)

### 1.1 What actually exists in maderix/ANE

Every kernel in the repo allocates its **own flat, 1-D IOSurface** and hands it to ANE via `_ANEIOSurfaceObject`. The canonical implementation is `create_surface()` in `bridge/ane_bridge.m:59-68` (identical to `ane_create_surface()` in `training/ane_runtime.h:34-43` and the inline surfaces in `inmem_bench.m:83-90`):

```objc
static IOSurfaceRef create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}
```

Note: `width = byte count`, `height = 1`, `bytesPerElement = 1`, `pixelFormat = 0`. It is a byte bag, not a Metal-compatible pixel format. The logical tensor shape exists only in the MIL signature, e.g. `func main<ios18>(tensor<fp32, [1, ch, 1, sp]> x)` (`inmem_bench.m:28`).

In `ane_bridge_compile_multi_weights()` (`bridge/ane_bridge.m:70-201`), one surface is created per input/output (`:171-177`), each is wrapped:

```objc
[wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
    g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[i])];
```

(`:183-185`), and bound into the request at `:194-197`:

```objc
k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
    g_ANEReq,
    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
    wIns, iIdx, wOuts, oIdx, nil, nil, @0);
```

### 1.2 Ownership / locking / synchronization mechanics that ARE implemented

| Operation | Function / lines | Mechanic |
|---|---|---|
| Write input | `ane_bridge_write_input`, `bridge/ane_bridge.m:233-239` | `IOSurfaceLock(surf, 0, NULL)` → `memcpy(IOSurfaceGetBaseAddress(surf), …)` → `IOSurfaceUnlock(surf, 0, NULL)`. Plain CPU `memcpy`; no GPU involvement. |
| Read output | `ane_bridge_read_output`, `:241-247` | `IOSurfaceLock(surf, kIOSurfaceLockReadOnly, NULL)` → `memcpy` → `Unlock`. |
| Destroy | `ane_bridge_free`, `:249-276` | `unloadWithQoS:21`, `CFRelease()` each surface, delete `tmpDir`, nil ARC objects, `free()` struct. |
| Request lifetime | same `k->request` reused for every `ane_bridge_eval` | `evaluateWithQoS:options:request:error:` at `:223-231`, QoS 21. |

There is **no cross-agent synchronization code anywhere in maderix/ANE** — no `IOSurfaceSharedEvent`, no Metal interop, no fence, no `MTLBuffer`↔IOSurface mapping.

### 1.3 The closest real statements about GPU↔ANE zero-copy

- **Part 1:** *"The I/O uses IOSurfaces — the same shared memory mechanism used for GPU textures. This means zero-copy transfers between GPU and ANE are theoretically possible if you share the same IOSurfaceRef."* — explicitly *theoretically*, no demo.
- **Part 4:** *"The CPU, GPU, and ANE use different programming models but share unified memory. A GPU-produced tensor can reach the ANE through another mapping of the same IOSurface. Each ANE submission adds approximately 90 µs of host-side dispatch time."* and table: `IOSurface | one buffer, three mappings, zero copies`.
- **`training/m5result.md:98-104`** (M5, macOS 26.3) — the only place sync primitives are probed:

  ```
  ### SharedEvents — Classes Exist, Need IOSurfaceSharedEvent
  - `_ANESharedEvents`, `_ANESharedSignalEvent`, `_ANESharedWaitEvent` all exist
  - `alloc/init` returns nil — they need `IOSurfaceSharedEvent` objects (Metal shared events)
  - `_ANESharedSignalEvent` has `symbolIndex` and `agentMask` — for GPU↔ANE sync
  - Signal API: `+signalEventWithValue:symbolIndex:eventType:sharedEvent:`
  - Wait API: `+waitEventWithValue:sharedEvent:eventType:`
  ```

  So the private classes for GPU↔ANE ordering exist but were never made to work in this repo, and are single-source (M5 only).
- **`thebasedcapital/ane-infer/docs/ane-internals.md`** (M5/macOS 26.3, single-source): `_ANEChainingRequest` mapped; `prepareChainingWithModel:` succeeds with correct factory `+objectWithstatsSurRef:outputBuffer:`; `buffersReadyWithModel:` **fails silently**; chaining sequence `prepareChaining → buffersReady → enqueueSets` is mandatory. `_ANEIOSurfaceOutputSets`, `memoryPoolId`, `signalEvents` exist. This is a Rust engine doing ANE prefill + **Metal GPU decode**, not shared hidden states.

### 1.4 Verdict for "16 GB Mac, hidden states GPU→ANE without a second allocation"

- **No working implementation exists in maderix/ANE.** The README table is the entire evidence base for it and the source files are absent.
- What would be required, from the evidence: (a) create one IOSurface and map it both as a Metal texture/buffer and wrap it with `_ANEIOSurfaceObject`; (b) respect the ANE's packed `[1,C,1,S]` fp16/fp32 layout — Orion constraint #20: *"the ANE reads the flat buffer as packed [1,C,1,S] data starting from byte 0, ignoring the surface's nominal dimensions"*; (c) order the handoff with `IOSurfaceSharedEvent` via `_ANESharedSignalEvent`/`_ANESharedWaitEvent`; (d) amortize ~90–119 µs dispatch. None of (a)–(c) is implemented in any file I could find.
- Practical consequence: **do not plan Veystrix around zero-copy GPU→ANE hidden-state handoff based on this repo.** It is a research idea at "theoretically possible" maturity. Zero-copy CPU↔ANE is real (shared IOSurface + `memcpy` into it) but that is a different claim.

---

## 2. `bridge/ane_bridge.h` / `.m` — exact C API, costs, cache, requirements

### 2.1 Exact C-callable API (`bridge/ane_bridge.h:15-94`)

```c
typedef struct ANEKernelHandle ANEKernelHandle;          // opaque

int  ane_bridge_init(void);                               // 0 ok, -1 fail
ANEKernelHandle *ane_bridge_compile(
        const char *mil_text, size_t mil_len,
        const uint8_t *weight_data, size_t weight_len,
        int n_inputs, const size_t *input_sizes,
        int n_outputs, const size_t *output_sizes);
ANEKernelHandle *ane_bridge_compile_multi_weights(
        const char *mil_text, size_t mil_len,
        const char **weight_names, const uint8_t **weight_datas,
        const size_t *weight_lens, int n_weights,
        int n_inputs, const size_t *input_sizes,
        int n_outputs, const size_t *output_sizes);
bool ane_bridge_eval(ANEKernelHandle *kernel);
void ane_bridge_write_input(ANEKernelHandle*, int idx, const void*, size_t);
void ane_bridge_read_output(ANEKernelHandle*, int idx, void*, size_t);
void ane_bridge_free(ANEKernelHandle*);
int  ane_bridge_get_compile_count(void);
void ane_bridge_reset_compile_count(void);
uint8_t *ane_bridge_build_weight_blob(const float *src, int rows, int cols, size_t *out_len);
uint8_t *ane_bridge_build_weight_blob_transposed(...);
uint8_t *ane_bridge_build_weight_blob_int8(const int8_t *src, int rows, int cols, size_t *out_len);
uint8_t *ane_bridge_build_weight_blob_quantized(const float*, int, int, float *out_scale, size_t *out_len);
void ane_bridge_free_blob(void *ptr);
```

Header line 1 states intent: *"C-callable bridge to ANE private APIs for Python ctypes"*. `bridge/Makefile` builds `libane_bridge.dylib` with `xcrun clang -O2 -fobjc-arc -fPIC -dynamiclib … -framework Foundation -framework IOSurface -ldl`. A prebuilt `bridge/libane_bridge.dylib` is committed.

### 2.2 Compile → load → evaluate sequence (exact)

`ane_bridge_compile_multi_weights()` (`bridge/ane_bridge.m:70-201`):

1. `NSData *milData = [NSData dataWithBytes:mil_text length:mil_len];` (`:83`) — **NSData, not NSString**.
2. Build `wdict`: `wdict[name] = @{@"offset": @0, @"data": data};` (`:87-92`).
3. `modelWithMILText:weights:optionsPlist:` with `wdict.count > 0 ? wdict : @{}` (`:94-96`).
4. `_ANEInMemoryModel inMemoryModelWithDescriptor:` (`:102-103`).
5. `id hx = [mdl hexStringIdentifier]`, `tmpDir = NSTemporaryDirectory()/hx`; create `tmpDir/weights`, write `model.mil` and each weight file (`:109-129`). Weight names like `@model_path/weights/wq.bin` are stripped to relative paths by `substringFromIndex:12` (`:119-123`).
6. **`compileWithQoS:21 options:@{} error:&e`** (`:131-138`).
7. **`loadWithQoS:21 options:@{} error:&e`**, with exactly one retry after `usleep(100000)` (100 ms) "for ANE slot reclamation" (`:140-156`).
8. `g_compile_count++` (`:158`), then allocate the handle, surfaces, and request (`:160-199`).

`ane_bridge_eval()` (`:223-231`): `evaluateWithQoS:21 options:@{} request:kernel->request error:&e`.
`ane_bridge_free()` (`:249-276`): `unloadWithQoS:21`, release surfaces, delete `tmpDir`.

### 2.3 Measured costs (with provenance)

| Cost | Number | Source |
|---|---|---|
| Compile, small kernel (256×256 conv, sp=64) | **~11 ms** (10.8–13.9 ms across QoS 0–63); load **~1.7 ms** (QoS 0 first use: 15.6 ms); eval **0.06–0.08 ms** avg / 0.17–0.22 ms single | `training/m5result.md:55-78` (M5, macOS 26.3, single-source) |
| Compile+load, 64×64 conv sp=32 | **33.3 ms**; unload **0.5 ms**; reload **3.8 ms** | `training/m5result.md:15-21` |
| First compile (general) | **~20–40 ms**; cache hits "effectively free" | Part 1, "Compilation Cache" |
| Static pipeline, 72 kernels | **7.6 s** compile per restart | `training/README.md:26-34,44-54` |
| Static + ANE extras, 86 kernels | **9.6 s** compile per restart | `training/README.md:30-34` |
| Dynamic pipeline, 10 kernels | **0.4 s** one-time | `training/README.md:37-42`; Part 3 |
| Cross-generation static compile/10 steps | M1 ~7.1–8.5 s, M4 ~3.5 s, M5 3.2–3.4 s | `benchmarks/ANE_BENCHMARK_REPORT.md:31-39` |
| Per-eval dispatch floor | **~0.095 ms** (Part 2), **~90 µs** (Part 4), **117 µs standard / 106 µs direct** (`ane-infer`, M5), **~119 µs** (`ane.cpp` field guide, M3 Max), ~160 µs cited in Part 3 | multiple sources, different chips/paths |
| Real-time path | entitlement-gated; `beginRealTimeTask` → NO | `ane-infer` docs (single-source) |

QoS sweep conclusion (`m5result.md:59`): *"All QoS values 0-63 compile, load, and eval successfully. No measurable latency difference — ANE appears to run at fixed frequency regardless of QoS."*

### 2.4 Cache behavior across process restarts

Part 1 gives the exact on-disk cache:

```
~/Library/Caches/<app>/com.apple.e5rt.e5bundlecache/<build>/<hash>/
  <hash>.bundle/
  └── H16G.bundle/           ← H16G = M4 ANE target
      ├── H16G.e5            ← FlatBuffer binary (2-3 KB)
      └── main/main_ane/model.anehash
```

*"First compile takes ~20-40ms. Cache hits are effectively free."* Part 4 adds the daemon cache (`aned cache com.apple.aned/…/model.hwx`) and notes compile/load go through XPC while **evaluation takes a direct path** `ANEServicesProgramProcessRequestDirect` without the daemon. Orion §5.1 adds that `hexStringIdentifier` is *"a composite of three SHA-256 hashes"*, so identical MIL + identical weight-dict keys ⇒ identical identifier ⇒ same temp dir and same cache identity across reloads/restarts. Net: **compiled E5/HWX programs persist across process restarts**; the in-memory weight BLOBFILEs are rewritten to `tmpDir` each compile, and the E5 binary is only ~2–3 KB (1024×1024 matmul = 2,688 bytes; 128×128 = 2,680 bytes, Part 1).

### 2.5 macOS / entitlement requirements stated

- `README.md:153`: *"Requires macOS 15+ on Apple Silicon (tested on M4)."*
- `benchmarks/ANE_BENCHMARK_REPORT.md:143-148`: macOS 26.x `[MLModel compileModelAtURL:]` broken for standalone benchmarks (fixed in PR #27 by switching to in-memory MIL); macOS 15.x works for all M-series with correct MIL format.
- `m5result.md:3`: M5 / macOS 26.3 (Darwin 25.3.0), ANE family H16 (same as M4).
- **No entitlement file, code-signing recipe, or privilege request exists anywhere in maderix/ANE.** Private classes are resolved at runtime via `dlopen(...AppleNeuralEngine.framework..., RTLD_NOW)` + `NSClassFromString` (`ane_bridge.m:33-57`). The only access caveat stated is Part 1's *"Temp directory workaround … If you don't have write access to the default location, compilation fails with an opaque error."* `sudo` appears only for `powermetrics` in `training/dashboard.py:940`. The real-time path is entitlement-gated per `ane-infer` (single-source).
- `README.md:217-219` disclaimer: private undocumented APIs, no stability guarantee, fair-use/interoperability rationale; no Apple code included.

### 2.6 Reusable via Python ctypes?

Yes in principle: it is a C ABI dylib explicitly designed for ctypes. Caveats:
- No Python ctypes example is committed. The commit named "Python Bridge+Memory leak fix+More functions" (`ebac5dd`) added only `bridge/*.{h,m}`, the dylib, and ObjC training files — no `.py`.
- `bridge/Makefile`'s `test` target references a missing `test_bridge.m`.
- `input_sizes`/`output_sizes` are byte counts; you must generate the MIL text yourself (no MIL generator in `bridge/`).
- Multi-input constraints from Orion (§6) apply: uniform allocation sizes (#18), alphabetical parameter ordering (#19), packed flat reads (#20). maderix's own README:192 states *"multi-input ANE requests cause 0x1d error; inputs packed into spatial dimension instead"*.

---

## 3. Weight handling: INT8/INT4 compilation, runtime updates, "dynamic weights"

### 3.1 INT8 weights DO compile; INT4 has no path

**Evidence — `ane_int8_bench.m:70-77`** (real, working MIL in the repo):

```objc
// constexpr_affine_dequantize: int8 weights → fp16 at compile time
[m appendFormat:
    @"        tensor<fp16, [%d, %d, 1, 1]> W%d = constexpr_affine_dequantize()"
    @"[axis = int32(0), name = string(\"W%d\"), "
    @"quantized_data = tensor<int8, [%d, %d, 1, 1]>"
    @"(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu))), "
    @"scale = fp16(0x1p-3), zero_point = int8(0)];\n",
    ch, ch, i, i, ch, ch, (unsigned long)(64 + i * cs)];
```

The int8 blob builder (`ane_int8_bench.m:18-32` and `bridge/ane_bridge.m:330-344`) uses a **64-byte header + 1 byte/element**, with `chunk[10]=0x08` as the 8-bit marker; fp16 uses `0x10`. `ane_bridge_build_weight_blob_quantized()` (`:346-371`) computes **global** symmetric scale `max_abs/127` (the header comment at `ane_bridge.h:88` inaccurately says per-channel axis=0).

Part 4 (Aug 2026) refines the performance story and explains why the README's 1.88× and Part 2's "no INT8 speedup" are both partly right:

| Weights | Activations | Measured | Mechanism |
|---|---|---|---|
| fp16 | fp16 | 1.00× baseline | 2-byte inter-layer transfers |
| int8 | fp16 | **1.00×** | weights expanded before compute; chain stays fp16 |
| int8 | int8 | **up to 1.95×** | packed compute, 1-byte inter-layer transfers |

*"W8A8 is a chain property. … Int8 weights without int8 activation edges stay on the fp16 path."* Middle blocks use compute mode `0xb1418005` and edge DMA word `0x80049240`; entry `0x93418005`/`0x80041240`, output `0x91418005`. A single 1024ch/32² conv: int8 260.4 µs vs fp16 260.5 µs. Deep 48L chain: 36.01 TOPS W8A8 vs 18.45 TFLOPS FP16; W8A8 peak power 12.25 W vs FP16 4.57 W. **INT4: no support in maderix/ANE, no path in Orion (*"quantization (INT8/INT4) is not yet supported"*), and explicitly listed among failed experiments in `ane.cpp`'s field guide (M3 Max).**

### 3.2 Weights are baked at compile time; `weightsBuffer` is dead — CONFIRMED in code

Every request in the repo passes `nil` for `weightsBuffer`:
- `bridge/ane_bridge.m:197` — `wIns, iIdx, wOuts, oIdx, nil, nil, @0`
- `training/ane_runtime.h:123-125`, `inmem_bench.m:93-95`, `inmem_basic.m:86-88`, `ane_int8_bench.m:181-183` — same.

`training/m5result.md:83-92` is the direct experiment:

```
### weightsBuffer IOSurface — Does NOT Override
Passing a `weightsBuffer` IOSurface with different weights to the request does not change output.
Baseline (1x identity):    [0.1000, 0.2000, 0.3000, 0.3999]
weightsBuffer (3x identity): [0.1000, 0.2000, 0.3000, 0.3999]
```

`test_weight_reload.m:1-5` poses the file-swap question; `m5result.md:9-23` gives the result on M5/macOS 26.3:

```
**Result: No.** Weights are baked at compile time. Overwriting weights/weight.bin in tmpDir
and doing unload→load produces identical output — the ANE ignores the file change.
Compile+load: 33.3ms | Unload: 0.5ms | Reload: 3.8ms
Max A-B diff: 0.000000
```

**CONFLICT (unreplicated both ways):** Orion (arXiv:2603.06728, §5, M4 Max / macOS 15) reports the opposite — *delta compilation*: `unloadWithQoS(21)` → write BLOBFILEs → `loadWithQoS(21)` yields **494 ms for 60 kernels** vs 4,200 ms full recompile (8.5×), ~9 ms/kernel vs ~70 ms/kernel, and bypasses ANECCompile entirely and the ~119 limit. Possible explanations are hardware (M4 vs M5), macOS (15 vs 26.3), or blob/toolchain differences; **neither result has been independently replicated.** Test on the actual target M4 before relying on either.

The only *proven* paths to changing weights without re-emitting MIL are:
1. **Dynamic weights via IOSurface input** (below) — maderix's production solution; and
2. **LoRA adapter-as-input** (Orion §6) — bake `W_base`, pass low-rank A/B as input surfaces, hot-swap with zero recompilation. Constraints #18–20 apply.

### 3.3 The "dynamic weights" trick — exact mechanics

`training/training_dynamic/mil_dynamic.h:14-35`, `gen_dyn_matmul()`:

```objc
// slice activations: sp[act_sp_off : act_sp_off+seq]
[m appendFormat:@"... %s_act = slice_by_size(x=%s,begin=%s_ba,size=%s_sa)...", ...];
// slice weights from the SAME input: sp[w_sp_off : w_sp_off+oc]
[m appendFormat:@"... %s_wt = slice_by_size(x=%s,begin=%s_bw,size=%s_sw)...", ...];
```

`gen_dyn_matmul_mil()` (`:37-45`) declares the whole thing as one input:

```objc
// Simple dynamic matmul kernel: y = x @ W, input [1,IC,1,SEQ+OC], output [1,OC,1,SEQ]
int sp = seq + oc;
[m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, sp];
gen_dyn_matmul(m, "mm", ic, oc, seq, 0, seq, "x");
```

`test_dynamic_matmul.m:1-19` gives the full layout with fp32 I/O:

```
// Layout: input [1, D, 1, S+D] — activations in sp[0:S], weight rows in sp[S:S+D]
// fp32 input is cast to fp16 in-kernel; output cast back to fp32.
```

For attention, `gen_sdpa_fwd_dynamic()` (`mil_dynamic.h:48-116`) packs **all three weight matrices** into the spatial axis:

```
Input: [1, DIM, 1, SEQ + Q_DIM + KV_DIM + KV_DIM] fp16
  sp[0:SEQ]                    = xnorm [DIM, SEQ]
  sp[SEQ:SEQ+Q_DIM]            = Wq [DIM, Q_DIM]
  sp[SEQ+Q_DIM:...KV_DIM]      = Wk [DIM, KV_DIM]
  sp[...]                      = Wv [DIM, KV_DIM]
```

Result: 10 kernels compiled once (≈0.4 s), reused for every layer; weights rewritten into the pre-allocated per-layer IOSurfaces before each dispatch, updated after Adam. Costs stated in Part 3: *"dynamic weight packing adds CPU overhead for the fp32→fp16 conversion and interleaved write into the spatial dimension. And the ANE can't fuse the weight load with the computation the way it can with baked constants."* `ane.cpp` field guide quantifies the analogous tradeoff for runtime matvec: **1.65 ms at 2560×2560, correct, 5× slower than const-weight conv but zero recompile cost**; runtime-weight **conv** fails with `0x1d` on M3 Max (test 5, test 40).

For a 16 GB machine with 4-bit MoE weights, note the systemic problem: the dynamic trick still requires every active weight tensor to be written into the IOSurface **every eval**, and the ANE's fp16 surface doubles 4-bit data back to 16 bits; plus `matmul` (used by maderix dynamic path) is ~3× slower than `conv` (Part 2 #17). This does not scale to tens of billions of MoE parameters.

---

## 4. Minimum practical sequence length / batch shape; 2–4 token prefill; workarounds

**Nothing in maderix source or articles states a minimum sequence length or an all-zeros cliff.** The relevant evidence is external and mutually consistent on the *number*, not on the *symptom*:

| Constraint | Exact statement | Source |
|---|---|---|
| Min IOSurface ~49 KB; **pad seq dim ≥ 16** | *"There is a minimum IOSurface size of approximately 49 KB (#4), meaning single-token tensors with shape [1, 768, 1, 1] (3,072 bytes in fp16) must be padded to at least [1, 768, 1, 16] (24,576 bytes)."* Symptom: **`0x1d` at eval**. | Orion Table 3 #4, §3 (single-source; internally inconsistent 49 KB vs 24 KB — 768×32×2 = 49,152 B, so ≥32 is the safer reading) |
| **W ≥ 32 (SP=32)** | *"The ANE requires IOSurface tensors with a minimum spatial width (W dimension) of 32. For single-token decode, 31 of those lanes are unused."* | Pradeep article, fn 11 (M3 Max) |
| Inputs **and outputs** need W ≥ 32 | test 31: identity `[1,1,1,SP]` works; slicing output to `[1,1,1,1]` *"compiles and then fails at eval with ANE status 0x1d. On the tested paths, all runtime IOSurfaces, including outputs, needed W >= 32."* test 32: runtime input width must be **exactly 32**; W=33 fails 0x1d; spatial kernels compile only to `kW=15`. | skyfallsin field guide tests 31–32 (M3 Max, single-source) |
| Decode uses min seq dim 16 | *"Each subsequent token is processed through the full model on ANE with a minimum sequence dimension of 16 (to satisfy constraint #4)."* | Orion §4.4 |

**Corrected answer:** the documented failure is error `0x1d`, not silent all-zeros. A 1-token `[1,768,1,1]` fp16 surface (3,072 B) is below the floor; pad the spatial dimension to 16 (Orion) or 32 (field guide/pradeep). The "all zeros" framing is likely a garbled retelling of these notes.

**Can 2–4 token prefill / speculative verification run on ANE?** Yes, with caveats:
- **W-lane batching is the demonstrated mechanism.** Field-guide test 37: packing 4 distinct samples into W lanes 0..3 of one IOSurface produced correct results for `matmul` (max_err ≈ 0.00195) and `oproj_add` (max_err ≈ 0.00391); throughput **5.0×** (1406.8→280.0 µs) for matmul and **2.93×** (1596.6→544.3 µs) for `oproj_add`. Pradeep's article: 4 prompt tokens batched across W lanes gives ≈2× prefill throughput.
- Padding to S=16/32 with dummy tokens works but wastes 31/32 of the lanes for real single-token work.
- **But:** Part 4 says single-token decode loses to CPU/SME because ~90 µs dispatch dominates; crossover is ≈23 µs of ANE work. Field-guide test 29: a working ANE decode-attention block matched CPU numerically (max_err < 0.001) but was **12.6 / 24.3 / 84.2 ms** at seq_len 64/256/1024 vs **0.021 / 0.043 / 0.137 ms** for CPU GQA — **~560–610× slower**. Attention and norms must stay off ANE.
- **fp32 I/O does NOT fix the small-S limit.** maderix kernels already use fp32 surfaces with in-MIL casts (`inmem_bench.m:28,36,40`; `test_weight_reload.m:45-60`), and README:182 says fp16 direct I/O is ~37% faster. The limit is allocation size / W-lane width, not dtype. Orion #20 adds that over-allocated surfaces are read packed from byte 0, so padding must be written correctly at the front.
- **Batch shape everywhere in maderix is N=1**, `[1,C,1,S]`. The `HEADS`/`KV_HEADS` constraints are in `training/README.md:172-175`.

**Bit-identical caveat (critical for Veystrix):** ANE fp16 matmul max_err ≈0.002–0.004 (field guide test 37), and the fp16 RMSNorm has an error floor of **~0.01095** that did not improve with grouped two-pass reduction (test 38). Orion reports 100% top-1 agreement over 64 greedy tokens vs CPU fp32 and max logit error 0.073 — that is *token* parity, **not bit-identical logits**. Under a hard bit-identical-token constraint, ANE-produced logits must be validated per-token or kept out of the token-selection path.

---

## 5. Drafter/speculative code? Reusable C bridge?

- **maderix/ANE has NO small drafter, no speculative decoding, no KV-cache inference engine.** It is transformer *training* (static `train_large.m`, `train_large_ane.m`, dynamic `training_dynamic/train.m`) plus microbenchmarks. The only generation code is `training/dashboard.py` (a TUI that got a CPU KV cache in commit `06535fc`) for demonstrating training checkpoints.
- Community work built on it:
  - `johnmai-dev/ANE-LM` — first ANE LLM runtime; **forked into** `skyfallsin/ane.cpp` (Qwen3.5-4B/9B: 9.21/11.66 tok/s int8 generate on M3 Max; 113 ANE↔CPU dispatches per token; `latency ≈ 119 µs + bytes / 78 GB/s`).
  - `thebasedcapital/ane-infer` (issue #44) — Rust + ObjC + Metal, Qwen3.5-2B hybrid, 32 tok/s **GPU** decode; ANE fused FFN mega-kernel 3.6 TFLOPS; direct eval 106 µs vs 117 µs standard; `doEvaluateDirectWithModel:`.
  - Part 3 "What's next": mega-kernel layer fusion (issue #24): 12-layer 5081 µs vs 24 separate evals 15227 µs; 4-layer partial fusion 7.7×; "weights are still `const()` so you'd need gradient accumulation + double-buffered compilation."
  - **Speculative decode / self-draft is explicitly a FAILED experiment** in `ane.cpp`'s field guide and Pradeep's "what didn't work" list (M3 Max). No ANE speculative implementation is published.
- **Reusable C bridge:** `bridge/libane_bridge.dylib` + `ane_bridge.h` is exactly a ctypes-ready C API (functions listed in §2.1), but it exposes only compile/eval/I-O/weight-blob helpers — **no MIL generator, no model, no sampler**. You supply MIL text and byte sizes. `bridge/Makefile`'s `test` target points at a nonexistent `test_bridge.m`, and no Python example exists. So: reusable as a *transport layer*, not as an engine.

---

## 6. Every stated limitation, with numbers and replication status

Legend: **R** = appears in multiple independent sources/code paths; **S** = single-source/unreplicated (author or one contributor); **C** = contested (direct contradiction between sources).

| # | Limitation (exact numbers where given) | Source | Status |
|---|---|---|---|
| 1 | SDPA causal mask silently ignored by ANE → Q@Kᵀ (ANE) → mask+softmax (CPU) → scores@V (ANE) | README:189; training/README:18; Part 3; Orion #6 | **R** |
| 2 | ~119 compiles per process, then silent fail/crash; workaround `exec()` restart | README:108,190; Part 3; Orion #5; community | **R** |
| 3 | FP16 gradient underflow in backward; fix loss scaling `256 * NLAYERS` | README:191; Part 3 | **R** (code) |
| 4 | Single-input constraint; multi-input requests cause `0x1d`; pack inputs into spatial dimension | README:192; Orion #18–20 | **R** |
| 5 | `weightsBuffer` does not override compiled weights | `m5result.md:83-92`; code request `weightsBuffer:nil` | **S** (one probe, code-consistent) |
| 6 | Weight reload by file swap fails (M5/macOS 26.3): A–B diff 0.000000; compile+load 33.3 ms, unload 0.5 ms, reload 3.8 ms | `m5result.md:9-23` / `test_weight_reload.m` | **C** — Orion reports delta reload works on M4 Max/macOS 15 (494 ms/60 kernels) |
| 7 | Apple "38 TOPS" misleading; true FP16 ≈19 TFLOPS, INT8 dequantized to FP16 | Part 2; Orion Table 1 | **R** |
| 8 | Part 4 revision: W8A8 packed mode reaches 36.01 TOPS but only with int8 activation edges; int8-weights-only = 1.00×; W8A8 peak 12.25 W vs FP16 4.57 W | Part 4 | **S** (refines #7) |
| 9 | SRAM cliff: 24 MB fits → 5.7 TFLOPS; 96 MB spills → ~30% drop; inferred ~32 MB SRAM | Part 2 | **S** (author) |
| 10 | Part 4 revision: streaming threshold is **2 MiB per operand** (`HAT[0x1b8]`), 64 KiB resident weight store, up to 16 MiB streamed in 16 DMA chunks, 64-bank SRAM / 16 B interleave `bank=floor(addr/16) mod 64`; "24–32 MB cliff reflects tiled streams, not a single capacity boundary" | Part 4 | **S** (author, later) |
| 11 | Dispatch overhead ~0.095 ms / ~90 µs; under ~1 ms ops are dispatch-limited; crossover ~23 µs of ANE work vs SME | Part 2, Part 4; community: 117/106 µs, 119 µs, 160 µs | **R** (magnitude), exact value varies |
| 12 | Training utilization only ~5–9% of peak; many elementwise ops fall back to CPU; not a replacement for GPU training | README:32-34; benchmark report util 3.4–9.2% | **R** |
| 13 | Conv 1×1 ≈3× faster than matmul; deep graphs 16–64 ops reach 94–98.8% of peak vs ~30% for single op | Part 2; Part 4; Orion #17 | **R** |
| 14 | SpaceToDepth(4) on 64ch tensor: 3.92→16.37 TFLOPS (4.18× end-to-end); avoid S2D block_size=2 (2.8–3.1% overhead) | Part 4 | **S** |
| 15 | Max graph depth: 383 conv blocks compile/load/eval; 384 makes `ANECompilerService` disconnect | Part 4 | **S** |
| 16 | No FP8: H16G E4M3 capability gate is zero; no INT4 path anywhere | Part 4; Orion limitations; field guide | **R** |
| 17 | No branches/loops/dynamic control flow; atomic graph execution; can't pause/inspect intermediates | Part 1, Part 3, Part 4 | **R** |
| 18 | Multi-output buffers must be uniform size, ordered alphabetically; `0x1d` / silent wrong data | Orion #2,#3 | **S** |
| 19 | Multi-input buffers uniform size, ordered alphabetically, packed flat read from byte 0 | Orion #18–20 | **S** |
| 20 | Min IOSurface ~49 KB / W≥32 / pad S≥16; symptom `0x1d` | Orion #4; pradeep fn11; field guide tests 31–32 | **R** (2 independent sources), exact threshold differs |
| 21 | `concat` MIL op rejected → split programs | Orion #1 | **C** — `mil_dynamic.h:149,164,180-181` uses `concat` successfully on M4 |
| 22 | 32K-channel convolutions rejected → CPU fallback | Orion #16 | **C** — `training/ane_classifier.h` + PR#19 run a 32000-output classifier conv on ANE (`training/README.md:30-34`) |
| 23 | BLOBFILE offset is `uint64(64)`, not 128; gelu invalid; conv has no `bias=`; MIL text must be NSData; weight dict `@{}` not nil; matmul transpose flags need named consts | Orion #8,#9,#10,#11,#12,#13; Part 1 confirms NSData | **R** for NSData/offset; rest **S** |
| 24 | `exec()` restart overhead ~50 ms | Orion #15 | **S** |
| 25 | macOS 26.x `[MLModel compileModelAtURL:]` broken; macOS 15.x fine | benchmark report:143-148; PR#27 | **S** |
| 26 | M1 standalone benchmarks fail (single-blob weight format); training works via `stories_mil.h` | benchmark report:105-109; community (moriwang, andyg5000) | **R** |
| 27 | M3 Pro: only `ch=512` compiles (52 channel values tested); peak 16.77 TFLOPS, sustained 15.04 over 5 s | benchmark report:111-116; community (D-Ogi) | **S** |
| 28 | `sram_bench`/`inmem_bench` fail on M4 Pro (MIL compat); `inmem_peak` and training work | benchmark report:118-122; community | **S** |
| 29 | Chaining: `prepareChaining` succeeds, `buffersReady` fails silently; correct output-sets factory `objectWithstatsSurRef:outputBuffer:` | `ane-infer` docs | **S** (M5) |
| 30 | Real-time path entitlement-gated (`beginRealTimeTask` → NO) | `ane-infer` docs | **S** (M5) |
| 31 | ANE fp16 RMSNorm error floor ~0.01095; grouped reduction doesn't help | field guide test 38 | **S** (M3 Max) — **direct threat to bit-identical tokens** |
| 32 | ANE decode attention 560–610× slower than CPU GQA | field guide test 29 | **S** (M3 Max) |
| 33 | Speculative decode/self-draft, runtime-weight conv, ANE-only attention, 3-runtime-input kernels, INT4 all failed | field guide + pradeep article | **S** (M3 Max) |
| 34 | DMA serialized: 2→8 independent input planes ≈ +995 µs each, no concurrency knee | Part 4 | **S** |
| 35 | FP16 W8A8 power 4.57 W vs 12.25 W; earlier 2.79 W / 6.6 TFLOPS/W measurements used persistent low-power mode | Part 4 measurement note | **S** (revision of Part 2) |

Also note: the two headline files for GPU↔ANE and all of `README.md:210-215`'s GPU-prefill/ANE-decode table are **source-less (S+, effectively unverifiable)** — see §0.

---

## ACTIONABLE FOR VEYSTRIX 16 GB

**Constraint framing:** target is a 4-bit MoE LLM (~15 GB of weights for a 30B model on a 16 GB M4) with a **hard bit-identical-token constraint**. ANE bakes weights into compiled programs (§3.2); there is no published way to bake or dynamically stage tens of GB of 4-bit MoE weights, and INT4 has no ANE path at all (§3.1, §6 #16). So ANE cannot be the main executor for these models. The only defensible uses are narrow, measurable ones.

### Idea 1 — Keep the MoE on Metal/CPU; at most use ANE as a prefill accelerator for a *small dense* helper, never for single-token decode.
**Evidence:** Part 4's measured crossover is ≈23 µs of ANE work vs the ≈90 µs/host dispatch floor, with the explicit finding *"ANE for prefill … CPU/SME for single-token decode"*; Part 2 adds *"Use SME for: Single-token decode (zero dispatch overhead)"*. `ane-infer` gets its 32 tok/s from **GPU decode**, and `ane.cpp` uses 113 ANE↔CPU dispatches per token with `latency ≈ 119 µs + bytes / 78 GB/s`. Runtime-weight conv fails with `0x1d` and dynamic matvec is 5× slower than baked-weight conv (field guide tests 5/40 and the dynamic-matvec result), so swapping MoE experts through the ANE is not viable. For any ANE prefill, use baked `const()` weights, 1×1 conv (≈3× matmul), deep fused graphs (16–64 ops), 512–1024 channels, and `SpaceToDepth(4)` if channels are shallow (4.18×, Part 4).

### Idea 2 — Before touching a model, stand up a ctypes harness around the existing `libane_bridge.dylib` and measure *your* M4 at the exact shapes you need, including a bit-exactness gate.
**Evidence:** `bridge/ane_bridge.h:1` explicitly targets Python ctypes; §2.1 lists the full ABI, §2.3 gives expected costs (~11 ms compile, ~1.7 ms load, 0.06–0.08 ms eval for a 256×256 conv on M5; 20–40 ms first compile / free cache hits per Part 1). Use `input_sizes = C*S*2` (fp16 direct I/O is ~37% faster than fp32, README:182), start at **S=32 / N=1** to clear the W≥32 / 0x1d floor (§4), and compare every output against the MLX/Metal fp32 result. Gate: if RMSNorm is in the graph, expect ~0.01 error (field guide test 38) and therefore no bit-identical guarantee — keep norm and token selection on CPU. Also re-test the two contested items locally: file-swap weight reload (M5 says fail, Orion M4 says works, §3.2) and `concat` (Orion says banned, maderix uses it, §6 #21). Cost is hours, not a rewrite.

### Idea 3 — If you insist on multi-token ANE verification, use W-lane batching of ≤4 tokens with baked weights; do not pad S=1 and do not put attention/norm on ANE.
**Evidence:** field-guide test 37 is the only demonstrated multi-token-per-dispatch path: 4 samples in W lanes 0..3 gave correct matmul (max_err ≈ 0.00195) and `oproj_add` (max_err ≈ 0.00391) results with **5.0× / 2.93×** throughput gains; Pradeep's article reports ~2× prefill from the same trick. That fits a 2–4-token speculative-verification window. But the errors above and the ~0.01 RMSNorm floor (test 38) mean ANE logits are not bit-identical — you must verify against the target model's fp32 logits and keep argmax sampling on CPU; Orion's 100% top-1 parity is token-level, not bitwise. Field-guide test 29 (ANE attention 560–610× slower than CPU GQA) and the failed speculative-decode experiments mean there is no published ANE self-drafter to reuse.
