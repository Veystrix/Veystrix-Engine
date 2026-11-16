# Veystrix internal inference-engine map (for ANE integration feasibility research)

Branch: `neural-engine`, HEAD `92e0459` ("Make the bit-identical exact-speed profile the default", 2026-11-17).
Scope: read-only code/doc research. No files modified, no benchmarks, no inference, no servers started. All numbers below are **quoted from repository docs/code**, not measured in this session.

> Important pre-existing finding: ANE work is already recorded as **evaluated and rejected** in `experiments/ane/REPORT.md`. See §6. No production ANE/Core ML code exists in the tree.

---

## 1. Where model computation actually runs — entry-point map

### 1.1 Control plane (pure Python / Swift, no tensors)

| Layer | File:lines | Role |
|---|---|---|
| macOS app | `apps/macos/Veystrix/Sources/VeystrixKit/LaunchPlan.swift:79-92` | Builds `python -m veystrix.model_router --host 127.0.0.1 --port … --pool-gb … --checkpoint … --model … --flashnext-kv-cache … --flashnext-memory-policy absolute-swap-only`, plus `--flashnext-speed` / `--flashnext-exact-speed` / `--flashnext-bounded-prefill` (`:87-89`) and `--overnight` / `--prefix-dir` (`:90-92`). Environment assembled at `:94-146`. |
| App settings defaults | `apps/macos/Veystrix/Sources/VeystrixKit/Settings.swift:130,131,133,136` | `flashNextKVCache = .tiered4`, `flashNextSpeed = false`, `flashNextExactSpeed = true`, `flashNextBoundedPrefill = false`. |
| Router | `veystrix/model_router.py:323-444` (`main`), `:127-140` (`RouterServer`), `:142-266` (`RouterHandler.do_GET/do_POST`), `:285-320` (`_forward`) | One stable local HTTP port; forwards OpenAI/Anthropic-compatible requests to the single engine process on an ephemeral port. Ownership/handoff/coordinator at `:94-125`, `:245-249`; supervisor loop `:376-431`. |
| Process driver | `veystrix/model_process.py:20-162` | `ProcessDriver._command` (`:44-71`) builds the engine CLI via `models.flashnext.overnight.server_command`; `load` (`:91-125`) starts `python -m veystrix.engine_guard …` with a private API key/instance id; `unload` (`:137-162`) proves the whole process group exited. |
| Engine CLI builder | `models/flashnext/overnight.py:55-91` | `server_command(args)` emits `--mode agentic --pool-gb … --host … --port … --context-limit … --max-cold-prefill … --max-incremental-prefill … --model flashnext --checkpoint …` and one `--enable NAME` per feature (`:81-82`). Nemotron branch `:58-69`. |
| CLI | `veystrix/cli.py:233-353` (`command`), `:356-390` (`main`) | Parses `--model`, `--mode`, `--enable`, `--pool-gb`, `--native-mtp`, `--mtp-depth`, `--context-limit`, etc. (`:238-258`); chooses `CHAT`/`AGENTIC` plan unless benchmarking (`:263-276`); applies env with `modes.apply_mode` (`:309-312`); final exec is `/usr/bin/caffeinate -i … veystrix.session` (`:280-296`, `:353`, `:388-389`). Pinned runtime versions `RUNTIME_VERSIONS` (`:27-35`): MLX 0.32.2, mlx-lm 0.31.3, mlx-vlm 0.6.17, transformers 5.16.1, numpy 2.5.2, requests 2.34.2, huggingface-hub 1.29.0; probes at `:36-46`, `:83-103`. |
| Session / backend factory | `veystrix/session.py:459-524` (`build_backend`) | The only place the two heavy backends are imported: `FlashNextBackend` (`:461`, constructed `:492`) and `NemotronBackend` (`:513`, constructed `:516-520`). Also sets GPU keep-warm (`:504-507`). |
| Backend contract | `veystrix/backends/base.py:53-79` | `class Backend(Protocol)` — `open_conversation`, `append_user`, `generate(...)`, `check_invariant`, `reset`, `save_session`/`load_session`/`list_sessions`/`delete_session`, `configure`, `settings_registry`/`settings_state`. `DecodeTimer` `:18-50`; `CANCELLABLE_PREFILL_STEP_SIZE = 64` `:15`. |
| MLX isolation rule | `veystrix/backends/__init__.py:1-6` | Verbatim: *"These are the only modules that import MLX. The rest of veystrix stays pure Python so it can be imported by every environment, which matters here because the two models need different ones."* (In practice a few server/bench/import sites also do local `import mlx.core as mx`, e.g. `veystrix/server.py:961` area, `veystrix/session.py:402,547,1321,1354`, but all tensor compute classes live in `veystrix/backends/` and `models/`.) |

### 1.2 Tensor compute plane (MLX + custom Metal)

**Flash-Next (Qwen3.8-Flash-Next, 4-bit Q4):**
- Orchestration: `veystrix/backends/flashnext.py` — `FlashNextBackend` (`:122`), `__init__` (`:138-244`), `generate` (`:671-758`), `_generate` (`:982-1430`), batch helpers (`:458-573`), speculative decoders (`:584-661`).
- Model load: `models/flashnext/loader.py:117-286` `load_streaming()` uses `mlx_vlm.utils.get_model_and_args/load_config/...` (`:21-26`) and patches the stock model with: streaming MoE `StreamingSwitchGLU` (`:28`), `adaptive_topk.apply` (`:29`), RMSNorm fix (`:30-31`), `qsa_chunk.apply` (`:32`), compile glue/GDN (`:33-34`), `linear_state.apply` (`:35`), `quantized_kv.apply` (`:36`), streaming n-gram embedding (`:37-40`). `mx.set_wired_limit` proven at `:65-80`. The resulting `self.language` object is an `mlx.nn.Module` whose forward is executed by MLX on Metal; per-token call site is `veystrix/backends/flashnext.py:1307` (`self.language(token[None], cache=self.cache)`).
- Custom GPU kernels: `models/flashnext/metal_runtime.py` (57,658 B) — embedded Metal source templating (`_render` `:539`, `_template_head` `:544`, `_require` `:572`, `_slice_qmv_header` `:579`, `_mlx_qmv_header` `:654`), `Q4G32Projection` `:46`, `MetalCapabilities`/`probe_capabilities` `:55-103`, `block_plan` `:151`, `weighted_combine` `:181`, `MetalMoEExecutor` `:665` with `_kernel` `:743`, `_launch` `:877`, `_metal_projection` `:914`, `_metal_block` `:943`, `_metal_fused_up_swiglu` `:994`, `_metal_fused_down_combine` `:1023`, `execute` `:1105`.
- Other kernels/graph patches: `models/flashnext/qsa_chunk.py` (`_gather_kv` `:211`, `_gathered_call` `:222`, `_selected_blocks` `:160`, `_prepare_indexer` `:375`, `_chunked_call` `:529`, `apply` `:622`); `models/flashnext/mixed_kv.py` (`FLAG_FUSED_GATHER` `:34`, `FLAG_FUSED_SDPA` `:124`); `models/flashnext/quantized_kv.py`; `models/flashnext/linear_state.py`; `models/flashnext/compiled.py`; `models/flashnext/ngram.py`; `models/shared/dense_gemm.py`.

**Nemotron 3.5 Lightning:**
- Orchestration: `veystrix/backends/nemotron35.py` — `NemotronBackend` (`:158`), `generate` (`:460`), `_generate` (`:548-809`), buffer release (`:810-824`), health (`:475`).
- Core: `models/nemotron35/backend_core.py:31-509` `EngineCore` — `_step` `:82`, `_forward` `:115` (calls `self.model.backbone(mx.array([tokens]), cache=self.cache)` at `:117`), `_prenorm` `:122`, `_segment_layer_major` `:157`, `prefill` `:237`/`_prefill` `:251`, `decode` `:311`, `decode_many` `:329`, `verify_decode` `:369`, `native_mtp_tokens` `:426`, `prompt_lookup_tokens` `:484`.
- Model load: `models/nemotron35/streamed_model.py:80+` `load_streamed()`; base model is `mlx_lm.models.nemotron_h` with `mlx_lm.models.cache` `KVCache`/`ArraysCache` (CLI probe at `veystrix/cli.py:41-45`; `backend_core.py:52`, `:123-124`). `OverlappedSharedMoE` `:139`, `_overlap_shared_experts` `:159`.
- Kernels: `models/nemotron35/attention.py` (affine INT8 block-backed attention, `:1-25`), `fast_ssm.py` (compiled Mamba-2 scan, default on, not bit-identical), `mamba_scan.py` (opt-in block-parallel scan), `dense_gemm.py` (`FLAG = "VEYSTRIX_NEMOTRON_DENSE_GEMM"` `:7`), `mtp.py` (standalone affine-Q4 draft head), `streamed_experts.py` (`StreamedSwitchMLP` `:814-986` with `mx.gather_qmm` `:860-864`, `mx.quantized_matmul` `:892-895`, optional reuse of Flash-Next's `MetalMoEExecutor.affine_singleton` `:884-891`; `ExpertPool.fetch` `:530`; `StackedExpertReader` `:97`).

**Device APIs present (all MLX/Metal only):** `mx.device_info()` — `veystrix/backends/flashnext.py:798`, `models/nemotron35/admission.py:98`, `models/flashnext/prefill_budget.py:22-38`, `models/flashnext/hardware_probe.py:38`, `models/flashnext/prefill_ring.py:235-237`; `mx.set_wired_limit` — `models/flashnext/loader.py:65-80`; `mx.new_stream(mx.gpu)` — `models/flashnext/expert_cache.py:705`, bench files, `veystrix/tests/test_model_thread.py:15`; `mx.metal.*` (cache/get_active memory/capture) — `models/flashnext/chat_guard.py:95-96`, `models/flashnext/tests/bench/*`. **No `set_default_device`, no device enum/selection layer, no alternate compute backend.**

---

## 2. Decode & prefill pipelines and the current measured bottlenecks

### 2.1 Flash-Next (Qwen) — concrete functions

- **Prefill**: `FlashNextBackend._generate` calls `prefill_language(...)` at `veystrix/backends/flashnext.py:1195-1197` (or prefix-cache restore at `:1184-1189`). Implementation: `models/flashnext/prefill.py:359` `prefill_language()` → `_run_chunks` (`:472`) for chunk-major or `_run_layer_major` (`:579`) when `_single_span_layer_major` (`:569-577`) fires; chunk planning in `plan_chunks` `:176`, `balanced_edges` `:154`, `budgeted_edges` `:131`. `FLASHNEXT_PREFILL_CHUNK_TOKENS` default 2,048 in agentic mode (`settings/modes.py:85`).
- **Decode step**: `standard_tokens()` `veystrix/backends/flashnext.py:1267-1311`. Hot path is `step = self.language(token[None], cache=self.cache)` (`:1307`) → `token = sampler(step.logits[:, -1, :])` (`:1308`) → `mx.eval(token)` (`:1309`). Alternatives: prompt lookup `:1316-1330`, MTP/fused draft decoders `:604-661`, `decoder.generate` `:1332`. Shared sampler: `veystrix/sampling.py:86-200` (`__call__` `:172-185`, greedy `mx.argmax` `:176-179`).
- **Attention / KV cache**: QSA gather vs reference selected in `models/flashnext/qsa_chunk.py:86-90`, gather tile `:141-146`; tiered INT4-history/INT8-recent KV in `models/flashnext/mixed_kv.py` (recent tokens default 8,192 — `settings/modes.py:97`); `quantized_kv.py` fingerprint parts.
- **Linear attention (GatedDeltaNet)**: 36 linear layers (per `models/flashnext/linear_state.py` docstring); `FLASHNEXT_EVAL_LINEAR_STATE=1` is in `COMMON` (`settings/modes.py:71`), implementation `linear_state.apply`; compiled recurrence upstream (`compiled.py:11-14` notes `gated_delta.py` already uses `@partial(mx.compile, shapeless=True)`).
- **Router / expert selection**: `models/flashnext/routing.py:353-863` (`RoutingProfile`, `begin_decode` `:539`, `_observe` `:563`, `after_token` `:582`), `models/flashnext/adaptive_topk.py`, exact top-10 of 512 with exact softmax/renorm is a non-negotiable (`docs/flashnext/quality-validation-policy.md:12-13`).
- **Expert GEMM / SSD stream**: `models/flashnext/expert_cache.py:StreamingSwitchGLU` `:1336`, `__call__` `:1742`, `_pool_pass` `:1887`, `_one_pass` `:2018`, packed-pass `:1812`, `_executor` `:1582`; custom Metal block GEMM `metal_runtime.py:MetalMoEExecutor.execute` `:1105`; read pool/physical read accounting `expert_cache.py:68-226`, `:580-784`; disk store `models/flashnext/store.py`.
- **KV cache**: per-layer state arrays in `self.cache` (MLX arrays), `mixed_kv.py`, `quantized_kv.py`; exact snapshot/restore in `models/flashnext/prefix_cache.py` + `prefix_session.py`.

**Bottleneck numbers (quoted from `docs/performance/`):**
- `docs/performance/overnight-2026-11-16.md:418`: *"Qwen, 385 ms per token: expert-read wait 289 ms (75%), router synchronisation 51 ms, GPU compute about 70 ms."* (numbers as written; components are overlapping/rounded in the source).
- `:253`: baseline 2.39 tok/s; `FLASHNEXT_PROFILE_IO=1` showed *"77% of decode wall in `io_wait`"*, critical read task waited 11.4 s in the 16-worker queue vs 4.6 s in `pread`, ~2.3 GB/s of ~3 GB/s SSD; *"Each token reads about 0.95 GB (hit rate about 24% in a 349-slot pool)."*
- `:286`: real decode trace 480 experts/token (48 layers × 10), 14,400 calls; LRU pool hit rate 0.0% (349 slots), S3-FIFO 11.5%.
- `:432`: 2 GB pool 2.62–2.69 tok/s vs 1 GiB pool 2.50–2.53; real server (1 GiB pool) 2.40 tok/s. `:273-278`: S3-FIFO +5% tok/s, 5% fewer bytes (2.39 → 2.62 tok/s; 157.7 GB → ~223 GB read in the paired arms).
- `docs/performance/stage-11.md:24`: five sustained runs, 103-token prompt/512 output: stream p10/median/p90 **2.65/2.66/2.69 tok/s**, median **742 MB/token** at ~2.79 GB/s ≈ **266 ms/token** I/O + ~110 ms/token outside; *"Perfect expert residency would still cap that execution path near 9 tok/s."* The 20 tok/s budget needs ~30 ms compute and ~56 MB/token reads (`:24`).
- Prefill: `overnight-2026-11-16.md:169-170` — 4,096-token cold prefill 48.0 s (85 tok/s) → 41.3 s (99 tok/s, exact dense-prefix path) → 33.9 s (121 tok/s, rejected `qsa-fast-prefill`); 8,192 tokens 95.1 s (86 tok/s) → 72.4 s (113 tok/s). `:148`: dense-prefix first chunk 2,048 tokens 607 → 121 ms per layer. `:11`: agentic turns 1.4–1.7K new tokens 51 → 27 s; 3.3–4.5K new over 3–11K cached 105–112 s → 42–69 s; cold 3.4–3.8K 37–43 s; minimum available 2.46 GiB (baseline 3.18 GiB).
- `EXACT_SPEED.md:13-18`: Flash-Next prefill +14–30% (single-chunk layer-major), +36% on 4.3K cold (merge-small-tail), −6% shared dense KV, −6.5% fused gather, −3.5% fused SDPA, −3% ngram-dedup, −2.4% dense GEMM; decode +5% tok/s.

### 2.2 Nemotron 3.5 Lightning — concrete functions

- **Prefill**: `NemotronBackend._generate` (`veystrix/backends/nemotron35.py:548+`) → `EngineCore.prefill` / `_prefill` (`models/nemotron35/backend_core.py:237,251`) → layer-major segmentation `_segment_layer_major` (`:157`), standard 2,048-token chunks; per-layer routing/expert streaming as in §1.2.
- **Decode**: `EngineCore.decode` (`backend_core.py:311-326`) → `_step` (`:82`) → `_forward` (`:115-120`) → `model.backbone(...)` + `model.lm_head` (`:85`). Speculative: `verify_decode` `:369-424`, `native_mtp_tokens` `:426`, `prompt_lookup_tokens` `:484`, `decode_many` `:329-367`. Sampler is the same `veystrix/sampling.py` class (`:319`).
- **Linear attention / Mamba**: 23 Mamba layers; `models/nemotron35/fast_ssm.py` (default on, compiled kernel, *"Same arithmetic up to floating-point reassociation and not bit-identical"*, **39 → 13 ms per 2,048-token chunk**, ~3×); `models/nemotron35/mamba_scan.py` (opt-in block-parallel, FP32 boundary states, rejected per `overnight-2026-11-16.md:13`); attention via `models/nemotron35/attention.py` over 6 layers with affine INT8 cache and fused SDPA.
- **Router / expert selection / GEMM**: `models/nemotron35/streamed_experts.py:StreamedSwitchMLP.__call__` `:932-986`; next-layer route prediction `_prefetch_next` `:826-845`; `_gather` `:847-867` (`mx.gather_qmm`, 4-bit affine, group_size from layer), `_expert` `:881-895` (`mx.quantized_matmul`; `VEYSTRIX_NEMOTRON_METAL_FMA_DEQUANT=1` uses Flash-Next's `MetalMoEExecutor.affine_singleton`), pipelined prefill `_pipelined` `:897-922`; async reads and pool `ExpertPool` `:339-805` / `StackedExpertReader` `:97-330`.
- **KV cache**: `EngineCore._make_cache` `:51-55` (`KVCache` or INT8 `AttentionCache`), `models/nemotron35/cache_routing.py`, `mlx_lm.models.cache`.
- **Bottleneck numbers**:
  - `overnight-2026-11-16.md:418`: *"Nemotron, about 120 ms per token (8.2 tok/s …): expert wait about 75 ms, GPU compute and dispatch about 26 ms (23.4 ms by component), the rest router sync and copy."*
  - `:426`: per token **115.6 ms** — routing sync/copy 37.6 ms (1.64 ms per MoE layer), expert dispatch/wait 71.9 ms (3.13 ms per layer), rest ~6 ms; **179 MB/token** physical SSD reads vs 773 MB expert bytes touched, **77% from page cache**; cached 6 × 5.6 MB `preadv` floor 2.14 ms single-thread / 1.28 ms on 8 threads (~26 GB/s).
  - `:39`: baseline 8,192-token prefill **252 tok/s** (32.3 s split: Mamba 9.7 s/23 layers, routed+shared experts 18.2 s of which 4.6 s SSD wait, attention 4.3 s/6 layers); after default speed changes **355.6 tok/s** (`:9`, `:126` Mamba scan 319.6 → 355.6 tok/s, +11%).
  - `:108`: harness 8,192-token cell legacy 231.9 tok/s (TTFT 35.4 s) → new 293.5 tok/s (TTFT 27.9 s), decode 7.66/7.70 tok/s, memory floor 5.13/5.11 GiB.
  - Attention block-eval (`handoff-2026-11-15.md:27`): per token over six attention layers **4.1 → 1.7 ms at 4K, 32.0 → 9.8 ms at 20K, 54.5 → 16.9 ms at 30K**; real 8K 182.8 → 179.2 ms/token; real 16K 155.7 → 143.4 ms/token (6.53 → 7.09 tok/s).
  - `:87`: cold SSD reads capped at **3.0–3.3 GB/s** at any thread count; `:86` shared-expert start gave decode −10% so it is prefill-only.
  - `admission.py:143-145` (code comment): 16 GB M4, INT8 KV, 128 tokens after prefix hit, 14 Nov 2026 — no pool 7.1–7.4 tok/s and 195–202 MB/token; 2 GiB pool 6.4–7.2 and 245–260 MB; 3 GiB 5.3–5.5 and 290 MB.
  - `docs/performance/stage-11.md:25`: 512-output BF16 arm **163 ms/token**, **261 MB/token** (~94 ms I/O at 2.79 GB/s + ~69 ms conditional estimate).
- Both models are **SSD/I/O-bound** at their 16 GB floors (`overnight-2026-11-16.md:13,31`: *"Decode is 75-90% SSD wait; the GPU part is 41 ms of 137 ms"*; `research-survey-2026-11-15.md:9`: Qwen ~740 MB/token at ~2.7 tok/s ≈ 2 GB/s, Nemotron ~41 ms GPU + 80–90 ms SSD/token).

---

## 3. Exact-speed profile — what commit `92e0459` turned on by default

### 3.1 The four files changed in `92e0459`
1. `apps/macos/Veystrix/Sources/VeystrixKit/Settings.swift`: `flashNextKVCache` `.bf16 → .tiered4` (now `:130`), `flashNextExactSpeed` `false → true` (now `:133`); explanatory comments `:128-129,132`.
2. `apps/macos/Veystrix/Sources/Veystrix/SettingsView.swift:30` (toggle help text now says "On by default").
3. `apps/macos/Veystrix/Tests/VeystrixKitTests/KitTests.swift:294-315` (`exactSpeedToggleIsIndependentAndPersists` now asserts default on, tiered4, `--flashnext-exact-speed` present, `VEYSTRIX_NEMOTRON_DENSE_GEMM=1`, and that toggling off removes both).
4. `docs/performance/EXACT_SPEED.md:3-5`.

### 3.2 Full flag/env chain (app → engine)
`Settings.swift:130,133` → `LaunchPlan.swift:85-89` (`--flashnext-kv-cache tiered4`, `--flashnext-exact-speed`, `--flashnext-memory-policy absolute-swap-only`) and `:118-121` (`env["VEYSTRIX_NEMOTRON_DENSE_GEMM"] = settings.flashNextExactSpeed ? "1" : "0"`) → `veystrix/model_router.py:329-335` flags parsed, passed at `:346-350` → `veystrix/model_process.py:46-53` builds `features = ["kv-tiered4"]` plus `"exact-speed"` → `models/flashnext/overnight.py:81-82` emits `--enable kv-tiered4 --enable exact-speed` → `veystrix/cli.py:248,271` → `models/flashnext/settings/modes.py:443-446` merges each feature's env dict → `modes.apply_mode` `veystrix/cli.py:310`, `modes.py:494-516`.

### 3.3 Every flag/env var set by the profile
`models/flashnext/settings/modes.py:236-247` `EXACT_SPEED_FEATURES` (10 features):
`kv-tiered4`, `layer-major-hybrid-4k`, `single-chunk-layer-major`, `merge-small-tail`, `qsa-shared-dense-kv`, `qsa-fused-gather`, `qsa-fused-sdpa`, `ngram-dedup`, `dense-gemm`, `s3fifo-pool`.
`modes.py:249` `EXACT_SPEED_ENV = {"FLASHNEXT_QSA_GATHER_TILE": "32"}`.
`modes.py:250-254` `EXPERIMENTAL["exact-speed"]` = union of the members' dicts `| EXACT_SPEED_ENV`.

Expanded env var → value → consumer (from `EXPERIMENTAL` entries `:130-235` and the consumption grep):

| Feature | Env vars and values | Primary consumer |
|---|---|---|
| `kv-tiered4` | `FLASHNEXT_KV_CACHE_DTYPE=tiered4`, `FLASHNEXT_QSA_KERNEL=gather` | `models/flashnext/quantized_kv.py:19` (`FLAG_KV_DTYPE`), validation `:192`; `models/flashnext/qsa_chunk.py:24` (`FLAG_KERNEL`), `:86-90` |
| `layer-major-hybrid-4k` | `FLASHNEXT_PREFILL_ORDER=layer`, `FLASHNEXT_PREFILL_WINDOW_TOKENS=4096`, `FLASHNEXT_PREFILL_HYBRID_SMALL_WINDOWS=1`, `FLASHNEXT_LAYER_MAJOR_MIN_TOKENS=2049` | `models/flashnext/layer_major.py:55-62`, `:183`; `prefill.py:569-579` |
| `single-chunk-layer-major` | `FLASHNEXT_LAYER_MAJOR_SINGLE=1`, `FLASHNEXT_LAYER_MAJOR_MIN_TOKENS=1024` (this min wins via `modes.py:447-448`) | `models/flashnext/prefill.py:570-577` |
| `merge-small-tail` | `FLASHNEXT_PREFILL_MERGE_TAIL=2048` | `models/flashnext/layer_major.py:62,183` |
| `qsa-shared-dense-kv` | `FLASHNEXT_QSA_SHARED_DENSE_KV=1` | `models/flashnext/qsa_chunk.py:30` |
| `qsa-fused-gather` | `FLASHNEXT_QSA_FUSED_GATHER=1` | `models/flashnext/mixed_kv.py:34` |
| `qsa-fused-sdpa` | `FLASHNEXT_QSA_FUSED_SDPA=1` | `models/flashnext/mixed_kv.py:124` |
| `ngram-dedup` | `FLASHNEXT_NGRAM_DEDUP=1` | `models/flashnext/ngram.py:20` |
| `dense-gemm` | `FLASHNEXT_DENSE_GEMM=1` | `models/flashnext/loader.py:144` → `models/shared/dense_gemm.py:8,23` |
| `s3fifo-pool` | `FLASHNEXT_EXPERT_POOL_POLICY=s3fifo` | `models/flashnext/expert_pool.py:47` |
| `EXACT_SPEED_ENV` | `FLASHNEXT_QSA_GATHER_TILE=32` | `models/flashnext/qsa_chunk.py:25,141-146` |
| Nemotron side (app only) | `VEYSTRIX_NEMOTRON_DENSE_GEMM=1` | `models/nemotron35/dense_gemm.py:7` → `models/shared/dense_gemm.py` |

Also always set for Flash-Next by the mode plan (not specific to exact-speed): `COMMON` (`modes.py:68-72`) `FLASHNEXT_CHAT_GUARD=1`, `FLASHNEXT_NATIVE_MTP=off`, `FLASHNEXT_EVAL_LINEAR_STATE=1`; `PREFILL_OFF` (`:75-81`); `EXPERIMENTAL_OFF` (`:94-121`) includes `FLASHNEXT_QSA_GATHER_TILE=8` (overridden to 32), `FLASHNEXT_KV_RECENT_TOKENS=8192`, `FLASHNEXT_KV_CACHE_DTYPE=bf16` (overridden to tiered4), `FLASHNEXT_EXPERT_POOL_POLICY=lru` (overridden to s3fifo), etc.; AGENTIC adds `AGENTIC_PREFILL` (`:82-93`) and watch dog/swap settings (`:406-441`). `OWNED` (`modes.py:264-270`) lists every owned key; `apply_mode` (`:494-516`) pops all `OWNED` keys before writing the plan, so a saved `~/.frankenstein/env` cannot silently change any of them.

Side effects of the gather/kv modes in `plan_mode` (`modes.py:449-463`): unless the launch used `absolute-swap-only`, `FLASHNEXT_MEMORY_POLICY=strict-staged` and `FLASHNEXT_MEMORY_WAIVER=0` are forced (the app passes `absolute-swap-only`, so this branch is skipped); inherited `FLASHNEXT_KV_LAYER_MODES`, `FLASHNEXT_KV_GROUP_SIZE`, `FLASHNEXT_KV_RECENT_TOKENS`, `FLASHNEXT_QSA_GATHER_TILE` are passed through if present (`:455-463`). Dependency validation: `:368-377` requires `kv-tiered4` for fused-gather/fused-sdpa and a layer-major mode for single-chunk/merge-tail — all satisfied by the bundle.

### 3.4 Evidence and rollback (as documented)
`docs/performance/EXACT_SPEED.md:3` — on by default on `neural-engine`; older `short-throughput` unchanged/off; tiered4 KV selector default to match the engine (`:5`). `:9-11` feature list (Nemotron also gets bit-equal dense GEMM; other Nemotron speed defaults always on). `:13-20` measured evidence (fanless 16 GB M4, 4-bit weights, full routing; **24/24 token-identical cases, PASS**, Tier A bit-identical + Tier C no regression; source `overnight-2026-11-16.md`). `:22-24` rollback: every member is an env-var-backed flag in `OWNED`, so turning the toggle off restores the reference path. `:26`: *"No live A/B run was launched for this change — the gains above are the existing paired measurements, and `AGENT_INVARIANTS.md` requires explicit permission for new benchmarks."*

Pinned by tests: `models/flashnext/tests/unit/test_modes.py:252-268` (asserts at 16/32/128 GB and CHAT/AGENTIC: `FLASHNEXT_QSA_GATHER_TILE=32`, `FLASHNEXT_EXPERT_POOL_POLICY=s3fifo`, `FLASHNEXT_PREFILL_ORDER=layer`, `FLASHNEXT_DENSE_GEMM=1`, and every exact-speed flag is in `OWNED`); `veystrix/tests/test_model_process.py:57-62`.

---

## 4. Hard rules any integration must respect (verbatim quotes)

**`docs/flashnext/AGENT_INVARIANTS.md`**
- `:10-12`: *"Keep the canonical control path unchanged unless we explicitly approve promotion with recorded evidence."* / *"Make every optimization opt-in until it passes the promotion rules below."*
- `:42`: *"Preserve the exact token digest. Any digest change rejects the optimization."*
- `:43-44`: *"Preserve BF16 rounding boundaries. A small numerical difference is not an acceptable quality result."*
- `:45-49`: no `mx.eval`/warmup in decode path without controlled measurement and explicit approval; no buffer merging to reduce object count; no worker/thread increases without controlled measurement.
- `:51-53`: *"Every disabled diagnostic feature must reduce to a branch on the decode hot path. It must allocate no closures, dictionaries, timers, counters, strings, callbacks, or context managers when disabled."*
- `:94-106`: every optimization experiment must keep the control path, use interleaved reversed arms, pass the exact token-digest gate, report resolution band/raw arms, physical MB/token and active RAM, generation and tail rate (and queue residence/etc. for I/O work).
- `:155`: ***"Do not run benchmarks, sweeps, or tests without our explicit permission."***

**`docs/flashnext/quality-validation-policy.md`**
- `:10-18` non-negotiables: keep Q4 affine G32 weights, quantization, routing (*"exact top-10 of 512, exact softmax and renorm"*), model mathematics; *"No approximate attention, no expert skipping or substitution, no lossy state compression (KV or GatedDeltaNet recurrent/conv state), no reduced-precision state that the reference keeps at higher precision."*; *"Every optimization is opt-in (a flag or setting that is off by default). Turning it on must not change the code path of the verified default."*; *"An explicit fallback to the last verified path always exists (section 7)."*
- `:20-28`: three evidence tiers — **A bit-identical** (same token ids, complete continuation state, positions; KV hash/state tensors equal exactly), **B numerically close** (never reported as A), **C task level**; *"A task-level pass never excuses a failed tier A or an unexplained tier B finding."*
- `:44-46` determinism: greedy temperature 0, thinking off, fixed chat template, `mx.random.seed(0)`, same checkpoint/pool settings, fresh process per arm, prompt token ids and digest recorded.
- `:125-133` gate: `same-plan` cached vs uncached **FAIL on any token difference in any turn of any case**; `reordered` fails if a passing reference case fails or any category loses passes; `kv-precision` needs complete arms and matching digests; missing cases fail.
- `:138-143`: default enablement only if tier A + B + C + clean memory guards + evidence covering the sizes enabled for (*"2K evidence does not cover 10K"*). `:144-150` fallback ladder (unset chunk policy → chunk 512/unchunked → prefix cache off → all experiment flags off). `:157`: *"No real-model arm runs without explicit approval per workload, no automatic retries, no automatic move to a larger size."*

**`docs/measurement-standard.md`**
- `:3-27`: one project terminal `./tests/run.sh`, append-only JSONL under `results/<runtime>/YYYYMMDD-HHMMSS-<experiment>/record.jsonl` + `output.log`; scratch logs never support published claims (`:29-31`).
- `:33-49`: schema 1; write `run` before arms, persist raw `arm` before validation; preserve failures/interruptions; record checkpoint, fingerprints, harness, prompt, template, sampler, effort, limits, ordering, environment, cache conditions; *"Greedy comparisons require exact token digests"*; fresh child processes, ≥3 arms per condition, alternate order; report paired effect and resolution band (inside band = unresolved).
- `:50-54`: *"Do not hammer the machine with tests… Use the smallest relevant controlled run, avoid redundant reruns and warmups, and stop a branch once its correctness, execution-path, or feasibility gate decisively fails."*
- `:56-57`: missing provenance is `unknown`, never reconstructed from filenames/timestamps.

Also `docs/performance/EXACT_SPEED.md:26` and `docs/performance/overnight-2026-11-16.md:5` restate permission/no-precision-change rules; `docs/flashnext/AGENT_INVARIANTS.md:17-21` keeps generic MLX Q4/G64 as the rollback reference (`FLASHNEXT_METAL_G64=0`) and G64 Metal default (`FLASHNEXT_METAL_G64=1`).

---

## 5. Memory tiers, kernel constraints, full-residency rules

### 5.1 Tier table (`veystrix/memory_tiers.py`)
- `FLASHNEXT_KERNEL_POOL_CAP_BYTES = 8 * GIB` (`:31`) with comment `:27-30`: MLX shape dims are 32-bit; Flash-Next passes its decode pool to Metal as a single `uint32` array, so no pool can exceed ~8 GiB until kernel addressing is redesigned — *"a hardware/kernel ceiling, not a memory-size decision, and applies equally to every tier."*
- `BASELINE = MemoryTier("16 GB", 2.0, 4.0)` (`:33`); `TIERS` (`:34-41`): **24 GB → (3.0, 4.0)**, **32 GB → (4.0, 4.0)**, **64 GB → (6.0, 6.0)**, **96 GB → (8.0, 8.0)**, **128 GB → (12.0, 8.0)**, else 16 GB baseline. Thresholds in GiB of physical RAM: 20, 28, 48, 80, 112, 0 (`:35-40`).
- `physical_bytes()` `:44-55` honors `VEYSTRIX_PHYSICAL_MEMORY_GIB` override; `tier_for()` `:58-63`; `flashnext_pool_cap_bytes()` `:70-72` = `min(pool_cap_gb × 1e9, 8 GiB)`.
- `full_resident_bytes()` `:75-87`: returns 0 (stream from SSD) when `total_bytes < dense_bytes + routed_expert_bytes + os_reserve_bytes`; default reserve `max(3 GiB, total_bytes // 4)` — *"never lowered to make a fit"*.

`docs/hardware/MEMORY_SUPPORT.md` restates the table (`:25-32`: 16→2/4 GB, 24→3/4, 32→4/4, 64→6/6, 96→8/8, 128→12/8), notes larger tiers **not validated on real hardware** (`:5`). `:41`: all Flash-Next 48×512 experts = 24,576 records × 3,072,000 B = **75.5 GB (70.31 GiB)**; pool passed as single `uint32` MLX array → **8 GiB** cap; full residency needs multi-buffer/64-bit offsets, out of scope. `:40`: Nemotron routed experts **15.38 GiB**, dense core **1.25 GiB**, OS reserve `max(3 GiB, total/4)`; on 16 GB it does not fit, so it streams. `:48-54` unchanged guards: 9 GiB start gate, 3 GiB guard floor, 3 GiB swap ceiling / 2.5 GiB swap-growth stop, 12.7 GB Metal working-set fallback, Nemotron OS reserve. `:58` override values 16/24/32/64/96/128.

### 5.2 Flash-Next budgeting
- `models/flashnext/memory_budget.py`: `POOL_STEPS = (2 GiB, 1 GiB, 0.5 GiB, 0)` (`:12`), `KV_BYTES_PER_TOKEN = 64 KiB` (`:31`), `DECODE_SLACK = 0.5 GiB` (`:32`), `UNION_BYTES_PER_ROUTE = 8 MiB` (`:33-35`); `pool_steps` `:16-28` adds one step at tier cap on larger Macs; `plan` `:70-91` (pressure ≥ 2 ⇒ size 0; grow cooldown 120 s and 1 GiB margin); `requested_cap` `:94-97` (`FLASHNEXT_EXPERT_POOL_GB`); `apply` `:115-154`.
- `models/flashnext/prefill_budget.py`: `METAL_LIMIT = 12_713_115_648` (12.7 GB measured max recommended working set on validated 16 GB M4, `:18`), `GUARD_HEADROOM = 500_000_000` (`:19`); `device_metal_limit()` `:22-38` queries MLX; model shape `:40-49` (`HIDDEN=2560`, `HC=4`, `LAYERS=48`, `FULL_LAYERS=12`, `LINEAR_LAYERS=36`, `CONV_DIM=10,240`, `GDN_STATE_BYTES=48*128*128*4 + 3*CONV_DIM*2`, `KV_BYTES_PER_TOKEN=2*2*256*2`, `VOCAB=248,320`); measured constants `:51-55` (`WEIGHTS=2_794_742_880`, `SCRATCH_RING=1_622_032_384`, `DECODE_POOL=2_000_000_000`, `ALLOCATOR_CACHE_PEAK=1_350_000_000`); `fits()` `:89-93` requires `peak <= metal_limit - 500 MB`; `budget()` `:111-136`; incremental `chunk_increment_bytes()` `:148-157`.
- Modes `choose_pool_gb` `settings/modes.py:297-328` enforces the tier cap (`flashnext_pool_cap_bytes`), refuses > cap, 9 GiB start gate, fixed 2 GB caveat.

### 5.3 Nemotron admission (`models/nemotron35/admission.py`, fail-closed)
- Constants `:20-30`: `ACTIVATION_BYTES_PER_CHUNK_TOKEN = 256 KiB`, `MIN_OS_RESERVE = 3 GiB`, `SWAP_REFUSE_BYTES = 2.5 GiB`, `MAX_CONTEXT/DEFAULT_CONTEXT = 1,048,576`, `BF16_CONTEXT = 262,144`, `PREFILL_LIMIT = 65,536`, `DEFAULT_POOL_BYTES = 2 GiB`, `AUTO_POOL_MAX = 4 GiB`, `AUTO_POOL_HEADROOM = 1.25 GiB`, `INT8_ATTENTION_SCRATCH = 384 MiB`.
- `auto_pool_max()` `:33-41` uses the tier's Nemotron cap; `read_machine()` `:76-104` raises `AdmissionRefused` if any `vm_stat`/`sysctl`/`mx.device_info()` probe is missing (fail closed); `budget()` `:124-127` = `min(gpu_limit, total − max(3 GiB, total/4), available)`.
- `make_plan()` `:130-201`: refuses critical pressure (`:147-148`) and swap at safety line (`:149-150`); context clamps `:152-155`; **full residency** path `:163-174` calls `full_resident_bytes(...)` then re-checks working-set, else 0 = stream; auto sizing `:175-181` shrinks on existing swap; shrink loop `:185-196` (pool −256 MiB steps, then halve chunk to 256) else `AdmissionRefused` `:197-201`. `pool_pays_off()` `:204-206`; per-request `check_request()` `:209-217`; `pool_bytes_from_env()` `:220-231` (`VEYSTRIX_NEMOTRON_POOL_GB`, 0–6 GB).

---

## 6. Alternate-backend/device abstractions, and existing ANE code

- **Backend abstraction**: only the `Backend` Protocol (`veystrix/backends/base.py:53-79`) plus a settings registry (`veystrix/backend_settings.py:8-80`, `Setting`/`SettingRegistry` with `lifecycle` in `live|next-turn|startup|read-only` at `:29-31`). These abstract *conversation/config*, not hardware. `build_backend()` `veystrix/session.py:459-524` is a hard-coded two-way `if name == "flashnext" / == NEMOTRON` switch (`:460,510`); there is no device parameter.
- **MLX device/backend APIs in tree**: only telemetry/limits/streams listed in §1.2 (`mx.device_info`, `mx.metal.*`, `mx.set_wired_limit`, `mx.new_stream(mx.gpu)`). No `set_default_device`, no `GPU/CPU/ANE` selection, no MLModel/Core ML/MPS/Metal device abstraction, no Swift↔Python tensor bridge. The Swift app only launches and supervises the Python engine (`LaunchPlan.swift`, `EngineProcess.swift`).
- **There IS a completed ANE evaluation, but no ANE implementation**: `experiments/ane/REPORT.md` (tracked) plus six result JSONs in `experiments/ane/results/` (`e1_plan`, `e2_verify`, `e3_latency`, `e4_overlap`, `e4b_alone`, `e5`). `git ls-files experiments/ane` shows exactly those 7 files. The scripts named in `:104` (`e1_placement.py`, `e2_verify.py`, `e3_latency.py`, `e4_overlap.py`, `e4b_alone.py`, `e5_numerics_and_shape.py`, `e5_memory.py`) and their `out/` models are **not present** in this branch; `:10-11` says they were *"untracked and uncommitted"* on branch `development3.8`, no production file changed, no weights downloaded/read.
  - `:3-7` status: ***"evaluated and rejected (2026-11-07). ANE development is stopped."*** Reasons: batch-1 predictors never reach ANE; predictor is ~0.1% of a 1K prefill; ANE traffic slows bandwidth-bound GPU work by 3–17%; ANE is not bit-exact for routing; dense offload needs duplicate fp16 weights that do not fit.
  - `:30-35` placement: batch 1 every op planned on CPU (35 µs, 0 MB neural footprint); batch ≥16 linear/siLU planned on ANE, `topk` cast CPU-only; ANE footprint 2.72–18.4 MB per model; DI driver power up 100% for ANE-loaded models.
  - `:43-56` latency (p50 ms/call, router 2560→512, top-10): batch 1 ANE 0.035 (ran on CPU), CoreML CPU 0.036, MLX GPU 0.22, MLX CPU 0.019, numpy 0.010; batch 16 ANE 0.173; batch 256 ANE **0.409** vs CPU 0.421 vs MLX GPU 0.49; batch 1024 ANE **1.41** vs CPU 1.61 vs GPU 1.37. ANE dispatch ~0.15 ms/call; a paced ANE predictor (one call/5 ms) took 1.64 ms (clocks down). Large fp16 matmul: ANE 4.2 TFLOPS vs MLX GPU 3.1 (1024×2560×2560).
  - `:58-62` memory: Core ML import +59 MB RSS; one ANE router +25 MB RSS first / +2 MB after, 2.72 MB ANE footprint; 48 per-layer routers ≈ 130 MB ANE + ~120 MB RSS + 59 MB ≈ **310 MB**.
  - `:64-76` overlap: ANE predictor 200 Hz +3.5%, flat-out +17.6%, ANE matmul +14.9% on bandwidth-bound GPU (8.13 ms alone); compute-bound GPU unaffected. ANE matmul slowed 1.22 ms → 2.8–3.0 ms beside GPU load.
  - `:78-94`: ANE top-10 set matched fp32 reference on 98.8% of 2,048 random rows (CPU fp16 89.9%) — **not exact, so it must not make routing decisions** (`:80`); a perfect predictor only ties baseline (`:84`); prefill 1K router ~1.4 ms/layer ≈ 68 ms over 48 layers = **0.14%** of the recorded 49.3 s prefill (`:86`); dense prefill matmuls need own fp16 weights (3.4 GB dense core cannot be duplicated) and Q4 `gather_qmm` is unavailable in Core ML (`:91`).
  - `:96-101` recommendation: do not build ANE route prediction/prefetch; do not use ANE for output-affecting computation (not bit-exact, duplicate fp16 weights, I/O-bound prefill).
  - `:14-22` conditions: M4 Air 16 GB, macOS 27.2, coremltools 9.0, MLX 0.32.2, isolated venv, random weights, battery 75→68%, preliminary, short loops; `:21` notes Python Core ML marshalling is included and a native Swift/ObjC caller may be cheaper — not measured.
- `docs/performance/overnight-2026-11-16.md:30` table row: **"ANE / AMX prefill | Not attempted: fp16 ANE numerics would change outputs and the routed experts are streamed"**; `:21,28` note the M4 has no neural accelerators.

---

## 7. What the test suite pins

Counts (static file/`def test_` counts at HEAD; no tests were run):
| Directory | `.py` files | `def test_` occurrences |
|---|---:|---:|
| `veystrix/tests/` | 92 (90 `test_*.py` + `__init__.py`, `_linux.py`) | 991 |
| `models/flashnext/tests/unit/` | 137 | 2,300 |
| `models/flashnext/tests/bench/` | 96 (benchmark/case harnesses, not unit tests) | — |
| `models/flashnext/tests/cases/` | 78 terminal cases (`case_*.py`) | — |
| `models/nemotron35/tests/unit/` | 17 | 203 |
| `models/nemotron35/tests/bench/` | 6 | — |
| `models/nemotron35/tests/cases/` | 6 | — |
| `apps/macos/Veystrix/Tests/` | 2 Swift files (`KitTests.swift`, `WorkflowSmoke.swift`) | Swift Testing `@Test` |

Test runner/docs: `tests/run.sh`, `tests/README.md`, `docs/testing.md`, `docs/measurement-standard.md:9-27`; results API enforced by `veystrix/tests/test_results_policy.py`, `test_results_index.py`, `test_measurement.py`. Linux CI skips MLX tests (`veystrix/tests/_linux.py:1-4`).

**Tests that would break if compute paths changed:**
- **Exact-speed configuration pins**: `models/flashnext/tests/unit/test_modes.py:252-268` (bundle env values + every flag in `OWNED`); `veystrix/tests/test_model_process.py:57-62`; `apps/macos/…/KitTests.swift:294-315` (default-on toggle and env/argument contract).
- **Nemotron bit-identical compute**: `models/nemotron35/tests/unit/test_speed_features.py` — prefill vs decode `mx.array_equal` (`:65-67`), attention/state equality (`:111`), streamed vs stock model (`:131`), fused ops (`:172-175,205-207,231-233`), rounding-level paths `mx.allclose` (`:294,321`), layer-major logits/state bit-identical to chunk-major (`:774-814`), pipelined experts (`:819`), streamed model bit-identical to stock in-memory (`:927`), exact rows (`:945`). Also `test_dense_gemm.py:35-71` (`mx.array_equal` bit equality, shape-verified gating), `test_int8_attention.py:41-107` (single-token decode bit-identical), `test_backend.py:138`, `test_cache_routing.py:64-125`.
- **Flash-Next bit/digest pins**: `models/flashnext/tests/unit/test_prefix_cache.py:5,126-208,349,398-401,519-522,591-604` (restore bit-identical, fingerprint refusal, RAM/SSD identity); `test_prefill_chunks.py:4-5,328-330` (restored run bit-identical with same spans; chunking numerically equivalent but **not** bit-identical to one call); `test_intelligent_memory.py:7-8,124-180` (bit-exact KV reconstruction and fingerprint isolation); `test_prefill_ring.py:181,220-223,265,302,361,714,764,777,844-855` (byte-identical staging/ring/temporary-pool results); `test_qsa_chunk.py:35`, `test_prefill_qsa_tiles.py:35` (`mx.array_equal` baseline vs candidate); `test_expert_pool.py:363,387+` (byte-identical buffered fill; policy equivalence class), `test_swiglu_contract.py:23` (bf16 bit round-trip).
- **Quality gates**: `models/flashnext/tests/bench/quality_suite.py` + `quality_compare.py` (24-case Tier A/C gate; `docs/flashnext/quality-validation-policy.md:99-136`), `numeric_diagnostics.py` (Tier B, `policy.md:74-98`). `nexus/tests/test_qualification.py:14-24` (token_sha256 gate), `test_concurrency_bench.py:246-297` (digests equal to single reference / `token_sha256`), `test_sampling.py` (sampler semantics), `test_flashnext_backend.py`, `test_flashnext_mtp_lifecycle.py`, `test_model_thread.py`, `test_session.py:164`, `test_phase3/5_remediations.py` all import MLX and exercise the runtime.

---

## 8. Factual seams (no ANE speculation)

The only hardware-agnostic seam in the code is the `Backend` Protocol plus `build_backend()`; below it, tensor compute is inseparable from MLX model objects and custom Metal kernel strings. Any alternate compute device would necessarily intersect: (a) `self.language(...)` at `veystrix/backends/flashnext.py:1307` / `model.backbone(...)` at `models/nemotron35/backend_core.py:117`; (b) the MoE executors `models/flashnext/metal_runtime.py:MetalMoEExecutor` and `models/nemotron35/streamed_experts.py:StreamedSwitchMLP`; (c) attention/QSA kernels `models/flashnext/qsa_chunk.py`, `models/flashnext/mixed_kv.py`, `models/nemotron35/attention.py`, `fast_ssm.py`; (d) KV/state arrays and their exact snapshot rules (`prefix_cache.py`, `test_prefix_cache.py`); all under the digest/precision/permission rules in §4. The repository's own recorded position on ANE is `experiments/ane/REPORT.md:96-101` (§6) and `docs/performance/overnight-2026-11-16.md:30`.
