# Engine language composition & C++ port scope analysis

**Date:** 2026-11-17
**Question:** How much of the engine is Swift / Python / C / other; if we choose C++, how much
change is required and how long will it take?

## 1. Measured composition (first-party code, excluding venv/dist/build/tests)

| Layer | Language | Files | Lines | Role |
|---|---|---|---|---|
| macOS app (UI + controller) | **Swift** | 38 | **~5,550** | UI, settings, process launch, HTTP client |
| Engine orchestration (`veystrix/`) | Python | 174 | ~37,700 | CLI, server, process mgmt, 17k of it tests |
| Flash-Next engine (`models/flashnext/`) | Python | 73 prod | ~28,100 prod | decode/prefill/expert cache (rest is ~65k test/bench) |
| Nemotron engine (`models/nemotron35/`) | Python | 60 | ~9,750 | decode/prefill/expert streaming |
| Shared | Python | 14 | ~1,700 | expert pack etc. |
| Native readers / shield | **C** | 3 | ~600 | `native_read.c`, `shield.c`, one `.mm` (~477) |
| C++ scaffold (ours) | **C++** | 5 | ~382 | `cpp/decode/` pager + tests (new) |
| Metal shaders | — | 0 own | 0 | All GPU work via MLX/mlx-vlm, no in-repo `.metal` |

**Bottom line:** the engine is over **90% Python**. Swift is only the app shell (~5.5k lines),
talking to Python over HTTP (`127.0.0.1:port/v1/chat/completions`, `ChatClient.swift:117-130`) and
launching it as a subprocess (`EngineProcess.swift:32-69`, `SystemEngineLauncher`). C is ~600 lines
of native helpers. The compute kernels are not ours — they are MLX/MLX-VLM (C++/Metal under the hood).

## 2. The architecture boundary (what a port does NOT touch)

```
Swift app  ──HTTP/JSON──>  Python server (veystrix/server.py)  ──in-process──>  Python engine
   (UI)                     (process boundary, stable API)        (models/*)
                                    ^
                            Python CLI / process mgmt (veystrix/cli.py, model_process.py)
```

Crucially, **the Swift side talks HTTP to the Python server**. A C++ engine can replace everything
below the HTTP boundary without changing a single line of Swift — as long as the C++ binary serves
the same `/v1/chat/completions` etc. on the same port. So the Swift 5.5k lines are **not** part of
the port. The C `native_read.c` helper is reusable as-is (already ABI-matched to our pager).

## 3. What a full C++ engine port actually requires

Direct compute hot path (the part that must be ported for speed):

| Module | Lines | What it does |
|---|---|---|
| `models/flashnext/expert_cache.py` | 2,117 | expert LRU, routing, fetch, shared reads |
| `veystrix/backends/flashnext.py` | 1,430 | backend, generate loop, prefill/decode orchestration |
| `models/flashnext/layer_major.py` | 991 | layer-major prefill scheduler |
| `models/nemotron35/streamed_experts.py` | 996 | Nemotron expert streaming |
| `models/flashnext/store.py` | 752 | safetensors index + positioned reads |
| `models/flashnext/qsa_chunk.py` | 642 | QSA sparse attention / gather |
| `models/nemotron35/backend_core.py` | 525 | Nemotron engine core |
| `models/flashnext/adaptive_topk.py` | 457 | router, top-k, host sync |
| `models/flashnext/loader.py` | 497 | model load / monkey-patch (no C++ equivalent) |
| **Total core hot path** | **~8,900** | |

Plus the supporting surface (not all must be ported for a first speed-focused build):
- `veystrix/server.py` (2,734) — HTTP API. Could stay Python/other and call the C++ engine as a
  library, OR be replaced by a C++ HTTP server (much larger).
- MLX-VLM model definitions (`language.py` for `qwen4_exp`, GDN, attention) come from the venv —
  these must be re-expressed with MLX C++ ops (the reference C++ runner did exactly this: ~330 KB
  across 10 files, 1:1 with these modules).
- Quality/digest harnesses, mode/profile system, preferences — stay Python; only need a bridge.

**Required for a bit-exact, speed-focused first port:** ~9,000 lines of Python hot path translated
to MLX C++ + a safetensors C++ loader + a C++/Python bridge for the HTTP server. The proven
reference (`Jakevin/moe-offload/mlx-qwen38/cpp/`) is ~330 KB of C++ for ONE model's decode path
(kernels, layers, loader, engine, io scheduler) and it is dense attention only (≤2,048 tokens) —
it does not cover QSA sparse attention, long context, prefill chunking, or Nemotron.

## 4. Time estimate (honest, one agent)

| Scope | Estimate | Notes |
|---|---|---|
| **Pragmatic #11 hook** (ctypes call to `build_plan` via our dylib) | 0.5–1 day + A/B | Small; host-time ceiling ≤6%/39%; needs guard permission |
| **Single-model decode port** (Qwen decode path only, like the reference) | 1–3 weeks | ~330 KB C++ from reference, adapted to Q4-G32 safetensors; dense-only, ≤2k context |
| **Full Qwen engine port** (decode + prefill + QSA + cache + modes) | 6–10 weeks | ~9k lines hot path; QSA sparse attention and GDN state are the hard parts |
| **Both models** (Qwen + Nemotron) | +4–8 weeks | Nemotron's Mamba/SSM + separate expert store is a second port |
| HTTP server in C++ (optional) | +2–4 weeks | Or keep Python server + bridge (cheaper) |

**Critical bit-exactness tax:** every ported kernel must be validated against the Python reference
(raw logits + all cache tensors). The reference C++ runner's own note says the `k_hdr` prefill fix
changed outputs for some prompts — these are subtle, recurring correctness traps. Budget ~30% of
the time for bit-exactness gating, not just translation.

## 5. Recommendation

- **Do NOT port for a speed win on this 16 GB M4.** We measured the ceilings: 1.065× (Qwen),
  1.386× (Nemotron) from full sync removal, and the I/O wall (~7 tok/s) remains. A C++ engine
  cannot reach 5+/15+ here — only more memory or a second Mac can.
- **If you want the port for future-model reuse / the long-term plan you stated:** scope it as a
  **single-model decode-first port** (~1–3 weeks) reusing the reference C++ runner, keep the Swift
  app and HTTP server unchanged, and treat bit-exactness as the primary gate. The pager scaffold we
  built (`cpp/decode/`) is the first tested module.
- **Cheapest real win available now:** enable the measured #2 TTFT improvement (flag off, bit-identical).
