# Nemotron-3.5-Lightning Mamba/SSM decode path — C++ porting plan

**Scope of study:** read-only. No files modified, no git commands, no checkpoints run.
**Bottom line:** the decode SSM primitive is **one small fused Metal kernel** (~55 lines of MSL) plus a cached depthwise conv and a gated RMSNorm. There is **no `mx::fast::ssm`/`ssd`/`scan` primitive in MLX C++ 0.32.2**, so the MSL source must be ported **verbatim** (exact same technique as `gdn_kernel.cpp`). The two "fast" Python scans (`fast_ssm.py`, `mamba_scan.py`) are **prefill-only and explicitly non-bit-exact** — they must NOT be used as the decode reference.

---

## 1. Exact Mamba-2 / SSM recurrence math used in Python

### 1.1 Which path actually runs in decode (S = 1)

`mlx_lm.models.ssm.ssm_update` — `/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages/mlx_lm/models/ssm.py:217-261`, dispatch at **lines 230-236**:

```python
seq_len = hidden_states.shape[1]
if (seq_len > 1 or state is None or mx.default_device() != mx.gpu
        or not mx.metal.is_available()):
    return ssm_attn(...)          # chunked scan (lines 115-214)
else:
    return ssm_update_kernel(...) # FUSED METAL KERNEL (lines 67-97)
```

For decode (S=1, state present, Metal GPU) the **fused kernel** at `ssm.py:67-97` is used. The chunked `ssm_attn` (`ssm.py:115-214`) matters only to *produce the incoming f32 state during prefill* and as the non-Metal fallback.

The "faster" research paths are off the decode path:
- `fast_ssm.py:13-17` docstring: *"Same arithmetic up to floating-point reassociation and **not bit-identical**… Only used for several-hundred-row calls without a mask; **decode… keep the stock path**."* Gate: `fast_ssm.py:28` `MIN_ROWS=64`, dispatch `:90-93`, installed by `streamed_model.py:123-124`.
- `mamba_scan.py:59-60`: `if length < 2*BLOCK (=128) … return super()._ssm(...)`; `mamba_scan.py:14-53` `parallel_ssm` is prefill block-parallel with FP32 boundary states.

**⇒ Reference for bit-exactness must be `mlx_lm.models.ssm.ssm_update_kernel` (stock), not either research scan.**

### 1.2 The fused kernel — verbatim MSL

`ssm.py:14-61` `make_ssm_kernel()` returns `mx.fast.metal_kernel(name="ssm_kernel", …)` with this exact source (`ssm.py:17-55`):

```cpp
auto n = thread_position_in_grid.z;
auto h_idx = n % H;                 // head index        (H = num_heads)
auto g_idx = n / G;                 // group index       (G = heads/groups = repeats)
constexpr int n_per_t = Ds / 32;    // Ds/32 = 128/32 = 4

auto x = X + n * Dh;  out += n * Dh;
auto i_state = state_in  + n * Dh * Ds;
auto o_state = state_out + n * Dh * Ds;
auto C_ = C + g_idx * Ds;   auto B_ = B + g_idx * Ds;

auto ds_idx = thread_position_in_threadgroup.x;      // 0..31
auto d_idx  = thread_position_in_grid.y;             // 0..Dh-1 (64)

auto dt_ = static_cast<float>(dt[n]);                // dt is ALREADY softplus+clip, f32
auto A   = -fast::exp(static_cast<float>(A_log[h_idx]));
auto dA  =  fast::exp(A * dt_);                      // per-step decay

float acc = 0.0f;
auto x_ = static_cast<float>(x[d_idx]);
for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * ds_idx + i;               // 0..Ds-1
    auto idx   = d_idx * Ds + s_idx;                 // state row-major [Dh, Ds]
    auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
    auto state = dA * i_state[idx] + dB_by_x;
    o_state[idx] = static_cast<U>(state);            // U = f32
    acc += state * static_cast<float>(C_[s_idx]);
}
acc = simd_sum(acc);                                  // reduce 32 lanes (state dim)
if (thread_index_in_simdgroup == 0)
    out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]); // T = input dtype (bf16)
```

**Recurrence, written out (one token, head h, state row d, state col s):**

| quantity | formula | citation |
|---|---|---|
| dt | `clip(softplus(dt_raw[h] + dt_bias[h]), t_min, t_max)` in **f32** | `ssm.py:7-11` |
| decay | `dA[h] = exp(-exp(A_log[h]) * dt[h])` using **`fast::exp`** | kernel `ssm.py:37-38` |
| state update | `S'[h,d,s] = dA[h] * S[h,d,s] + x[h,d] * dt[h] * B[g,s]` (f32) | kernel `:46-48` |
| output | `y[h,d] = cast_T( Σ_s S'[h,d,s] * C[g,s] + x[h,d] * D[h] )` | kernel `:49-54` |
| reduction order | 4 sequential `acc +=` per lane, then `simd_sum` over 32 lanes | kernel `:43-51` |

This is the standard Mamba-2 selective scan (`dA = exp(dt·A)`, `dB = dt·x·B`, `y = C·S + D·x`), with **`B,C` shared per group** (`g_idx = n/repeats`, `repeats = heads/groups = 64/8 = 8`).

**Launch configuration** (`ssm.py:83-97`, must be reproduced exactly for bit-exactness):
- input names `["X","A_log","B","C","D","dt","state_in"]`; outputs `["out","state_out"]` (`ssm.py:57-60`)
- template: `T=<input dtype>`, `U=<state dtype>`, `Dh=64`, `Ds=128`, `H=64`, `G=8` (`ssm.py:85-92`)
- `grid=(32, d, h*n)` = `(32, 64, batch*64)`, `threadgroup=(32, 8, 1)` (`ssm.py:93-94`)
- output shapes `[(n,1,h,d), state.shape]`, dtypes `[input dtype, state dtype]` (`ssm.py:95-96`)

**Constraints from the kernel:** `Ds % 32 == 0` (128 ✓), `H % groups == 0` (64/8 ✓); grid.y divisible by 8 (64 ✓).

### 1.3 Shapes and dtypes (Nemotron-Lightning public config, `config.py:8-16`)

| tensor | shape (decode, B=1) | dtype | source |
|---|---|---|---|
| x / `hidden_states` | `[1,1,64,64]` | bf16 (input activation) | `nemotron_h.py:179-181` |
| `A_log` | `[64]` | **f32** (`mx.log(mx.arange(..., dtype=f32))`) | `nemotron_h.py:121` |
| `dt_bias` | `[64]` | f32 init; **checkpoint dtype must be confirmed** | `nemotron_h.py:120`, `cast_predicate :557-562` |
| `dt` raw | `[1,1,64]` | bf16 from `in_proj` split | `nemotron_h.py:216-220` |
| `dt` to kernel | `[1,1,64]` | **f32** (after `compute_dt`) | `ssm.py:9-11` |
| `D` to kernel | `[64]` | **`hidden_states.dtype` (bf16)** — mixer explicitly casts `self.D.astype(hidden_states.dtype)` at `nemotron_h.py:195`; kernel reads `D[h_idx]` as bf16 then promotes in `x_ * D[h_idx]` | `nemotron_h.py:122,195` |
| `B`, `C` | `[1,1,8,128]` | bf16 (from conv output split) | `nemotron_h.py:182-183, 222-229` |
| SSM state in/out | `[1,64,64,128]` = 524,288 floats (2 MiB/layer) | **float32** | produced by `dtxdecay @ B` in `ssm.py:183-187` (f32 dtx × bf16 B → f32); `parallel_ssm` casts explicitly at `mamba_scan.py:34`; kernel `U = state.dtype` at `ssm.py:80,87` |
| conv state (cache[0]) | `[1, 3, 6144]` | bf16 (`dtype=conv_input.dtype`) | `nemotron_h.py:143-159` |
| conv1d weight | `[6144, 4, 1]` (after `sanitize` `moveaxis(2,1)`, `nemotron_h.py:538-542`) | bf16 | `nemotron_h.py:106-113` |
| y from kernel | `[1,1,64,64]` | bf16 (`T`) | kernel `out[d_idx]`, `ssm.py:96` |

Derived sizes (from `config.py:8-16` and `nemotron_h.py:94-115`):
`intermediate_size = 64*64 = 4096`; `conv_dim = 4096 + 2*8*128 = 6144`; `in_proj` output `= 4096 + 6144 + 64 = 10304`; `n_groups=8`, `heads_per_group=8`.

### 1.4 Is there an MLX C++ primitive?

**No.** I inspected the installed MLX C++ headers (`…/mlx/include/mlx/fast.h`): the only custom-kernel entry point is `mx::fast::metal_kernel` (`fast.h:77-85`, plus `precompiled_cuda_kernel` at `:96`). There is no `ssm`, `ssd`, `scan`, or `selective_scan` in `fast.h`. Available and needed ops do exist: `mx::fast::rms_norm(x, optional<array> weight, float eps)` (`fast.h:14-18`), `mx::logaddexp` (`ops.h:1089`), `mx::clip` (`ops.h:350`), `mx::cumsum` (`ops.h:1375`).

**⇒ Decision: port the `ssm_kernel` MSL source verbatim into a `mamba_kernel.h` raw string (exact analogue of `gdn_kernel.h:7-76`) and launch it with `mx::fast::metal_kernel` exactly like `gdn_kernel.cpp:23-41`. Do not re-express the recurrence with MLX ops for the production path** — `fast::exp` and the `simd_sum`/4-accumulate order cannot be reproduced by eager ops.

---

## 2. Per-layer dataflow of one Mamba layer in decode (S=1)

Reference: `NemotronHMamba2Mixer.__call__` — `nemotron_h.py:207-234`; `_conv` `:134-166`; `_ssm` `:168-205`; block residual `NemotronHBlock.__call__` `:443-455`. The block first applies `nn.RMSNorm` to the residual stream (`nemotron_h.py:449`).

| # | Op (Python) | C++ equivalent | Shape / dtype flow | Cite |
|---|---|---|---|---|
| 1 | `in_proj(hidden)` — affine-Q4 G64 linear | `linear(x, Linear::quantized(..., gs=64, bits=4))` | `[1,1,2688] bf16 → [1,1,10304] bf16` | `nemotron_h.py:116-118,214`; `engine.cpp:10-28` |
| 2 | `mx.split` → `gate`, `conv_input`, `dt` | `mx::split` / slices | `[1,1,4096]`, `[1,1,6144]`, `[1,1,64]`, all bf16 | `:216-220` |
| 3 | `_conv`: `padded = concat([conv_state, conv_input], axis=1)`; `cache[0] = padded[:,-(K-1):,:]` | cast conv_state to input dtype, `mx::concatenate`, `mx::slice` tail | `[1,3,6144] + [1,1,6144] → [1,4,6144]` bf16; new cache `[1,3,6144]` | `:143-159`; `gdn_layer.cpp:43-53` |
| 4 | `conv1d(padded)` depthwise (bias if `use_conv_bias`) + `nn.silu` | prepped weight `[K,C]`, f32 accumulate, add bias, cast, `hc_silu` | `[1,4,6144]→[1,1,6144] bf16 → silu bf16` | `:165-166`; `gdn_conv.cpp:11-45` (adapt: +bias) |
| 5 | `mx.split` conv_out → `hidden_ssm`, `B`, `C` | slices | `[1,1,4096]`, `[1,1,1024]`, `[1,1,1024]` | `:222-229` |
| 6 | `_ssm` reshape + `ssm_update` → fused kernel | reshape + `mamba_ssm_step_kernel` | hidden `[1,1,64,64]`, B/C `[1,1,8,128]`; state `[1,64,64,128] f32`; y `[1,1,64,64] bf16` | `:179-205`; `ssm.py:67-97` |
| 7 | `compute_dt`: `softplus(dt_f32 + dt_bias)` then `clip` | `mx::logaddexp(dt_f32, 0.f)` + `mx::clip` (f32) | `[1,1,64] bf16 → f32` | `ssm.py:7-11`; `engine.cpp:194-207` |
| 8 | kernel recurrence (§1.2) | verbatim MSL kernel | writes `state_out`, `y` | kernel `:36-54` |
| 9 | `y.reshape([1,1,4096])`; `cache.advance(1)` | reshape; cache bookkeeping | bf16 | `:202-205,232` |
| 10 | `MambaRMSNormGated(y, gate)`: `swiglu(gate,y)=silu(gate)*y`; `unflatten(-1,(-1,512))`; `mx.fast.rms_norm(weight=None, eps)`; `* self.weight` | `hc_silu(gate)` × y; `mx::reshape`; `mx::fast::rms_norm(x, std::nullopt, eps)`; multiply by f32/bf16 weight (dtype confirm) | `[1,1,4096]` → `[1,1,8,512]` → norm → `[1,1,4096]` | `nemotron_h.py:76-88,124-129,233`; `activations.py:10-11` |
| 11 | `out_proj` — affine-Q4 G64 linear | `linear(..., Linear::quantized)` | `[1,1,4096] → [1,1,2688] bf16` | `:130-132,234` |
| 12 | `NemotronHBlock`: `x + hidden_states` | `mx::add` | `[1,1,2688]` residual | `:443-455` |

Note on step 10: `swiglu(gate, x)` is `nn.silu(gate) * x` (`mlx_lm/models/activations.py:10-11`) — reuse `hc_silu` (`engine.cpp:73-88`) then multiply; the existing `engine.cpp swiglu()` at `:66-71` is identical semantically.

---

## 3. How the existing C++ patterns apply

- **`gdn_kernel.h` / `gdn_kernel.cpp` — the exact template.** `gdn_kernel.h:7-76` stores the Python MSL as a raw string `R"GDNKERNEL(… )GDNKERNEL"`; `gdn_kernel.cpp:23-41` builds `mx::fast::metal_kernel(name, input_names, output_names, source)`, passes `grid=std::make_tuple(32, Dv, B*Hv)`, `threadgroup=(32,4,1)`, a `std::vector<std::pair<std::string,mx::fast::TemplateArg>>` of template args, output shapes/dtypes, and falls back to an ops reference on exception (`:43-46`). The Mamba kernel uses the *same* launch style with `grid=(32,Dh,B*H)`, `threadgroup=(32,8,1)`, templates `{T,U,Dh,Ds,H,G}`, and **no scalar-array input** (unlike GDN's `Tarr` at `gdn_kernel.cpp:29`).
- **`gdn_conv.cpp:11-45` — cached conv.** `gdn_decode_depthwise_conv` = `astype(x,f32) * weight[None]`, `sum(axis=1)`, cast back, `expand_dims` (`:19-24`); `gdn_causal_conv_decode` preps raw `[C,K,1] → [K,C]` f32 (`:32-34`); `gdn_causal_conv_decode_silu` adds `hc_silu` (`:38-44`). For Mamba: same helper but **add conv bias before silu** (Nemotron `use_conv_bias` likely true — owner confirm) and keep conv state in bf16 (`gdn_layer.cpp:46-53` shows the defensive cast + tail-cache pattern to copy).
- **`gdn_layer.cpp:13-93` — full-layer composition.** Token-slicing loop (`:32-34`), projections (`:37-40`), conv+cache (`:43-56`), splits/reshapes (`:58-67`), recurrent kernel (`:76-81`, note the explicit `astype(y, xt.dtype())` at `:81` — the Mamba kernel already returns `T`, so no extra cast needed), norm (`:84`), out_proj (`:87-88`). Mirror this structure for `mamba_layer.cpp`.
- **`engine.cpp` dtype conventions.** `rms_norm` uses f32 math + cast back (`:168-178`); **`compute_g_beta` documents the critical softplus rule** (`:198-205`): a f32 zero in `logaddexp` promotes bf16 and diverges (~8e-3), so use same-dtype zero. For Mamba `compute_dt` the input is **already f32**, so `mx::logaddexp(dt_f32, mx::array(0.0f))` + `mx::clip` is the direct port. `linear()` (`:10-28`) handles quantized and dense paths with bias added last. `hc_silu` (`:73-88`) compiles the identical fused graph by default.
- **`moe_block_relu2` (`engine.cpp:155-166`)** already covers the 23 MoE layers: up → `square(max(u,0))` → down → `weighted_combine`, with `q4_projection` G64/4-bit. No new work for MoE; just ensure `from_nemotron_config` keeps `Activation::RELU2` (`model_adapter.cpp:148`) and the tensor prefix (`model_adapter.cpp:149-150`).
- **Adapter plumbing.** `model_adapter.hpp:57` already has `LayerKind::MAMBA_SSM`; `:181` has `bool use_mamba`; `:195-197` (`model_adapter.cpp`) sets `d.use_mamba=true` with the literal comment *"primitive not ported yet"*. There is **no `MambaConfig` struct** and `from_nemotron_config` (`model_adapter.cpp:124-157`, declared `.hpp:155-160`) carries no Mamba geometry — this must be added (§4).

---

## 4. Concrete file-by-file implementation plan

Legend: **NUMERIC** = must be bit-exact (verified byte-for-byte); **NON-NUMERIC** = plumbing/lifecycle. LOC are estimates.

| # | File (new unless noted) | Action | Est. LOC | Difficulty | Class |
|---|---|---|---|---|---|
| 1 | `src/mamba_kernel.h` (new) | Copy the `ssm_kernel` MSL source from `ssm.py:17-55` into `R"MAMBAKERNEL(…)MAMBAKERNEL"`; generator `tools/gen_mamba_kernel.py` mirroring `gen_gdn_kernel.py:1-124` to prevent drift | ~60 + 40 gen | Trivial | **NUMERIC** (source must be byte-identical) |
| 2 | `src/mamba_kernel.cpp` + `.h` (new) | `mx::fast::metal_kernel("ssm_kernel", {"X","A_log","B","C","D","dt","state_in"}, {"out","state_out"}, kSource)`; call with shapes `{(B,1,H,Dh), state.shape()}`, dtypes `{x.dtype(), state.dtype()}`, `grid=(32,Dh,B*H)`, `threadgroup=(32,8,1)`, tmpl `{T,U,Dh,Ds,H,G}`. Add eager ops fallback (documented **non**-bit-exact) | ~90 | Moderate | **NUMERIC** (launch config) + NON-NUMERIC (fallback) |
| 3 | `src/mamba_conv.h/.cpp` (new) | Depthwise conv helper `[B,K,C]×[K,C] (+ bias)` with f32 accumulate→cast→`hc_silu`, and conv-state prep/tail update. Start from `gdn_conv.cpp:11-45`; **verify bf16 conv accumulation order against `nn.Conv1d`** | ~80 | Moderate | **NUMERIC** |
| 4 | `src/engine.hpp/.cpp` (edit) | Add `compute_dt(dt, dt_bias, tmin, tmax)` f32 = `clip(logaddexp(dt+bias, 0f), …)` next to `compute_g_beta` (`engine.cpp:194-207`); declare at `engine.hpp:201` area | ~15 | Easy | **NUMERIC** |
| 5 | `src/mamba_layer.hpp/.cpp` (new) | `MambaConfig`, `MambaWeights` (in_proj QPacked, conv1d raw weight, conv bias, A_log, dt_bias, D f32, norm_weight, out_proj QPacked), `MambaCacheState{conv_state, ssm_state}`, `MambaLayerOutput`; `mamba_layer_forward()` reproducing §2 steps 1-12, token loop like `gdn_layer.cpp:32-92` | ~200 | Moderate–Hard | **NUMERIC** (ops+order+dtypes) |
| 6 | `src/model_adapter.hpp/.cpp` (edit) | Add `MambaConfig` (mamba_num_heads, mamba_head_dim, n_groups, ssm_state_size, conv_kernel, intermediate_size, conv_dim, eps, time_step_limit, use_conv_bias, mamba_proj_bias); add to `ModelSpec` (`.hpp:112-131`) and `LayerDispatch` (`.hpp:176-188`); extend `from_nemotron_config` signature + body (`.cpp:124-157`); set `d.mamba` in `dispatch_layer` (`.cpp:195-197`) and remove the "not ported" comment; add `to_string`/`describe` fields | ~60 | Easy | NON-NUMERIC (config plumbing) |
| 7 | `CMakeLists.txt` (edit) | Add new `.cpp` to `veystrix_engine` (`:26`), `add_executable`/`target_link_libraries` blocks (pattern `:51-56`), `add_test` guarded by vector-dir existence (pattern `:263-266`) | ~25 | Easy | NON-NUMERIC |
| 8 | `tools/gen_mamba_vectors.py` (new) | Kernel-level vectors via real `ssm_update_kernel`; layer-level vectors via real `NemotronHMamba2Mixer` with tiny config. Mirror `gen_vectors.py:168-199` and `gen_gdn_layer_vectors.py:56-120` | ~180 | Moderate | Tooling (produces NUMERIC goldens) |
| 9 | `tests/test_mamba_kernel.cpp` (new) | Byte-for-byte `uint32` view compare of `y` **and** `state_out` against Python kernel; copy `bit_equal_report` from `test_gdn_delta.cpp:25-53` | ~100 | Easy | Test harness |
| 10 | `tests/test_mamba_layer.cpp` (new) | Compare layer `output`, `conv_state_out`, `ssm_state_out` bit-exactly; isolate intermediates on failure like `test_gdn_layer.cpp` | ~130 | Moderate | Test harness |

**Ordering:** 1 → 2 → 8(kernel vectors) → 9 (kernel must pass first, exactly like the delta-kernel-first discipline in `test_gdn_delta.cpp:1-7`) → 3 → 4 → 5 → 8(layer vectors) → 10 → 6 → 7.

**Numeric risk concentration:** steps 1,2,3,4,5. Steps 6,7,9,10 are mechanical. Total ≈ **880 LOC**, difficulty dominated by dtype/accumulation-order matching, not by code volume.

---

## 5. Synthetic bit-exact test design + required config fields

### 5.1 Kernel-level test (the essential one)

Generator (`gen_mamba_vectors.py`) must call the **real stock path**:

```python
from mlx_lm.models.ssm import ssm_update_kernel, compute_dt
# inputs with real kernel geometry (small but legal: Ds % 32 == 0):
B,H,Dh,G,Ds,K = 1, 64, 64, 8, 128, 4
X        = <bf16> [B,1,H,Dh]
A_log    = <f32>  [H]
Bv,Cv    = <bf16> [B,1,G,Ds]
Dv       = <bf16> [H]           # IMPORTANT: mixer casts D to hidden dtype
dt_raw   = <bf16> [B,1,H]
dt_bias  = <f32>  [H]
state    = <f32>  [B,H,Dh,Ds]
y, state_out = ssm_update_kernel(X, A_log, Bv, Cv, Dv, dt_raw, dt_bias,
                                 state, time_step_limit)
mx.eval(y, state_out)
# save all inputs + y + state_out as .npy; meta.json = shapes/dtypes/limits + mlx_version
```

C++ test `test_mamba_kernel.cpp` calls `mamba_ssm_step_kernel(...)` on the loaded inputs and compares **both outputs** via `mx::astype(mx::reshape(a,{-1}), mx::uint32)` + `mx::eval` + raw word loop (`test_gdn_delta.cpp:25-53`) — **zero tolerance**, not an RMS threshold. Repeat over cases with bf16/f32/f16 `T`, multiple seeds, B=1 and B=2 (verify `g_idx = n/G` batch indexing), and `time_step_limit` both `(0,inf)` and `(0.001,100)`.

### 5.2 Layer-level test

Build a real `nemotron_h.NemotronHMamba2Mixer` with a **tiny config** (e.g. `hidden=256, mamba_num_heads=16, mamba_head_dim=64, n_groups=2, ssm_state_size=128, conv_kernel=4`), overwrite all weights deterministically (weights bf16; `A_log` f32; `dt_bias`/`D`/`norm.weight` exactly as the checkpoint would provide — see risk §6), set `.eval()`, run `mixer(x, mask=None, cache=[conv_state, ssm_state])`, save inputs + intermediates (`projected`, `gate`, `conv_input`, `conv_output`, `hidden_ssm/B/C`, `dt_post_compute_dt`, kernel `y`, `normed`) + `expected_output`, `expected_conv_state`, `expected_ssm_state`. Mirror `gen_gdn_layer_vectors.py:70-120`. Compare each saved tensor byte-for-byte to isolate the first diverging op.

### 5.3 Real-checkpoint vector stage (owner-gated)

Extend `gen_nemotron_real.py:22-94` pattern to a real Mamba layer: read the first `"mamba"` layer's dense weights from `streamed.reader`, take a real decode cache produced by the real Python path, run one `mixer(...)` step, and compare C++ output/logit digest. This is the only way to settle the dtype unknowns in §6. Note `config.py:96` currently reports `inference_ready: False` — real-checkpoint numerical validation has not been performed yet.

### 5.4 Config fields needed from `config.json` (owner inputs — cannot be read here)

Geometry (public values from `config.py:8-16` are known, but **these extra fields are not validated there** and must be supplied/confirmed):

| field | used for | note |
|---|---|---|
| `mamba_num_heads`, `mamba_head_dim`, `n_groups`, `ssm_state_size`, `conv_kernel` | kernel shapes/templates | `64,64,8,128,4` per `config.py:10-11` |
| `layer_norm_epsilon` | `fast.rms_norm` eps in MambaRMSNormGated (`nemotron_h.py:126`) | not in `config.py` EXPECTED |
| **`time_step_min` / `time_step_max` / `time_step_limit`** | `compute_dt` clip range | **critical**: `ModelArgs.__post_init__` defaults to `(0.0, inf)` if absent (`nemotron_h.py:63-65`); `config.py` never checks it. Decode numerics depend on the actual value |
| `use_conv_bias` | whether to add conv bias before silu | `nemotron_h.py:112` |
| `mamba_proj_bias` | in_proj/out_proj bias presence | `nemotron_h.py:117,131` |
| `quantization.group_size`, `bits` | Q4 affine for in/out_proj and MoE | G64/4 per `streamed_model.py:96,107` |
| runtime **dtypes** of `A_log`, `dt_bias`, `D`, `norm.weight`, and the SSM **state** | kernel template `U`, D cast, norm multiply | A_log is excluded from dtype casting (`nemotron_h.py:557-562`); others may be bf16 in the checkpoint. State assumed f32 (§1.3) — confirm |
| `n_group`, `topk_group` | MoE gate `group_expert_select` (`nemotron_h.py:326-334`) — `config.py` doesn't list them; if `None`, Python would fail, so they must exist | needed for full decode loop |
| `max_position_embeddings`, `rope_theta`, `rope_scaling/parameters`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `attention_bias` | the 6 attention layers | `config.py:75-79` |
| `n_routed_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `moe_shared_expert_intermediate_size`, `n_shared_experts`, `norm_topk_prob`, `routed_scaling_factor`, `moe_latent_size` | MoE + shared expert | `config.py:12-14`; `moe_latent_size=None` simplifies |

Also needed: a small set of **real per-tensor weight/cache dumps** (one Mamba layer) with documented dtypes, for the §5.3 stage.

---

## 6. Risks / unknowns

1. **No MLX C++ SSM primitive** — confirmed by inspecting `fast.h`; verbatim MSL port is mandatory, not optional. Identical MLX version (0.32.2) on both sides is a hard prerequisite (same as `STATUS.md:10` discipline).
2. **Determinism of the fused scan:** the kernel has **no atomics** and a fixed `simd_sum` over 32 lanes with `threadgroup=(32,8,1)` and the 4-iteration accumulate order (`ssm.py:43-51`) — deterministic for fixed shapes. Risk is *reduction-order drift*, not nondeterminism: changing grid/threadgroup/source breaks bit-exactness. `fast::exp` (`:37-38`) is an approximate but deterministic MSL builtin — eager `mx::exp` fallback would **not** match.
3. **Research paths must stay off:** if `VEYSTRIX_NEMOTRON_FAST_SSM=1` or `VEYSTRIX_NEMOTRON_PARALLEL_MAMBA=1` were set while generating goldens, the vectors would be non-bit-exact (`fast_ssm.py:13-17`, `mamba_scan.py:1,34`). Force stock path in the generator (S=1 automatically does this, but be explicit).
4. **Conv bf16 accumulation:** unlike GDN (`gdn_conv.cpp:13-24` has a verified helper), Mamba calls `self.conv1d(padded_input)` directly (`nemotron_h.py:165`). The f32-accumulate-then-cast formula must be validated against MLX `nn.Conv1d` for bf16 + K=4 + bias. Highest-risk non-kernel op.
5. **Runtime dtype uncertainty:** `D` is cast to hidden dtype at `nemotron_h.py:195`; `dt_bias`/`D`/`norm.weight` may be bf16 in the checkpoint while `A_log` is protected from casting (`:557-562`). The GDN port needed a dedicated divergence tool (`tools/check_gdn_dtype_divergence.py`) for exactly this class of issue — expect a Mamba equivalent.
6. **`compute_dt` compiled vs eager:** Python `@mx.compile` on f32 `softplus=logaddexp(x,0)`+clip (`ssm.py:7-11`, `activations.py:96-102`); C++ eager `logaddexp`+`clip` may fuse differently. GDN showed the same-dtype-zero rule matters (`engine.cpp:198-201`) — here the operand is f32 so risk is low, but verify.
7. **State dtype assumption:** f32 is inferred from `ssm.py:183-187` and `mamba_scan.py:34`; if a real checkpoint cache is bf16, `U` and the 2 MiB/layer budget (`memory.py:27-30` uses `state_element_bytes=4`) change.
8. **Batch>1 indexing:** `g_idx = n/G` with `n = z-index` flattens `[batch, heads]` and relies on `G = heads/groups` mapping to `[batch, groups]` (`ssm.py:19-20`); include a B=2 parity case.
9. **Fallback path is not bit-exact** (same caveat as `gdn_kernel.cpp:43-46` / `test_gdn_delta.cpp:3-7`); production must require Metal, and tests must run the kernel path.
10. **Scope boundary:** this plan delivers the Mamba primitive only. The full Nemotron C++ decode loop still needs the 6 dense-attention layers (Python uses INT8 block KV cache — `attention.py:37-172`, not yet a C++ primitive) and the 52-layer loop/final head; those are separate tasks. `STATUS.md:57,146-166` is Qwen-centric and should be updated once the Mamba primitive lands (replace `model_adapter.cpp:196`'s "not ported yet").

---

### Key citation index
- Stock decode kernel + dispatch: `ssm.py:7-11, 14-61, 67-97, 217-261`
- Non-bit-exact prefill paths: `fast_ssm.py:13-17,28,90-93`; `mamba_scan.py:1,14-53,59-60`
- Mamba layer dataflow/cache/dtypes: `nemotron_h.py:76-88,91-132,134-166,168-234,538-542,557-562`
- Config: `config.py:8-16,37-52,96`; `streamed_model.py:96,104-108,123-124`; `memory.py:20-30`
- C++ templates: `gdn_kernel.h:7-76`; `gdn_kernel.cpp:11-47`; `gdn_conv.cpp:11-45`; `gdn_layer.cpp:13-93`; `engine.cpp:10-28,66-88,155-166,168-178,194-207`; `engine.hpp:88,144-148,187-194`; `model_adapter.hpp:57,112-131,155-160,176-188`; `model_adapter.cpp:124-157,175-209`; `CMakeLists.txt:26,51-56,263-266`
- Tests/tools: `test_gdn_delta.cpp:1-7,25-75`; `gen_vectors.py:168-199`; `gen_gdn_layer_vectors.py:56-120`; `gen_nemotron_real.py:22-94`; `gen_gdn_kernel.py:1-124`
