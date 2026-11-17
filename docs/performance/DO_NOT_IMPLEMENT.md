# Do-not-implement registry — measured dead ends

This is the single place to check before trying a performance idea again. Every
entry has a **measured** result on Veystrix's own hardware (16 GB M4, guarded
runs). "Rejected" means it did not clear the resolution band or failed a quality
gate — not that it was untested. Numbers come from the cited docs; do not add a
new entry without a number and a source.

Owner note: this registry exists so a dead end is never re-implemented. The code
flags remain in the tree as explicitly-required rollback/comparison paths
(AGENT_INVARIANTS.md: "preserve generic MLX Q4/G64 as our comparison reference and
explicit rollback"; "every optimization opt-in with fallback"). Removing them
would violate those invariants and break reproducibility — see the owner's
pending decision at the bottom.

## Decode — rejected

| Idea | Measured result | Source |
|---|---|---|
| Expert prefetch `FLASHNEXT_EXPERT_PREFETCH` | 2.39 → 2.38 tok/s | overnight-2026-11-16.md:253 |
| Decode overlap `FLASHNEXT_DECODE_OVERLAP` | 2.39 → 2.41 tok/s (within band) | :253 |
| I/O workers 6/8/16/24/32 | all 2.55–2.57 tok/s — SSD saturated | :280-282 |
| Pool policies other than S3-FIFO | cost-aware 2.45 vs lru 2.39 vs s3fifo 2.54; least-stale within band | :267-278 |
| Larger pool (2/3/6 GB) | clamped to 1 GiB; 3 GB stops at 2.59 GiB floor; 6 GB refused | :292-294 |
| Perfect expert predictor + overlapped reads | +0.1 / −2.9 / −0.7% (tied); early submit −8.2% | real-runs-2026-11-14.md |
| Native MTP all depths (Qwen) | depth 1 2.36 vs plain 2.58; depth 2 2.17; depth 3 1.35; adaptive 1.45 | real-runs-2026-11-14.md:68-107 |
| **Native MTP depth 2 (Qwen, exact verifier)** | **2.298 → 1.924 tok/s (0.837×, 16% SLOWER); digest identical** | experiments/ane A/B 2026-11-17, commit c68ec90 |
| Native MTP (Nemotron) | depth 2 4.94 vs 4.94; adaptive 5.03→4.71; 8K 3.90→2.25 cold | stage-9.md, real-runs:180-181, native-mtp-8k |
| Lookup drafting v2 | 7.30 → 6.15 tok/s Nemotron (11–16% slower) | lookup-drafting-2026-11-15.md:26-37 |
| Nemotron decode read-ahead / stale-drop | 5.14/5.04/4.96 vs 6.67 kept | handoff-2026-11-15.md:54-82 |
| Nemotron parallel Mamba | failed exactness gate; 3.66 vs 3.91 tok/s | inference-blueprint-handoff.md:193-208 |
| Nemotron 4K chunks | MLX peak 3.64→4.47 GB, compressor 692→1778 MB | overnight-2026-11-16.md:128-130 |
| `FLASHNEXT_COMPILE_GDN` | 2.41 → 1.88 tok/s | inference-blueprint-handoff.md:130 |
| Cache-aware routing | +8.3% but failed trajectory gate (repeated question) | brief.md:92-101 |
| Expert packing, Qwen | +9% in-process but 66.6 vs 66.1 s through server; 70.5 GiB extra disk | overnight-2026-11-16.md:420-424 |
| Expert packing + DLPack, Nemotron | +6.3% but 15.5 GiB disk; pack alone +6% | overnight-2026-11-16.md:424-428 |
| Nemotron zero-copy shards | +2.3% (8.14 vs 7.95) — marginal, owner decision | overnight-2026-11-16.md:430 |
| `F_NOCACHE` for experts | 8% slower | research.md:304 |
| `F_RDAHEAD=0` | 1.3% in 8/8 pairs, below resolution band | research.md:308 |
| Wired-limit sysctl | +13.5% standalone, −0.4% in harness | brief.md:130-133 |

## Prefill — rejected

| Idea | Measured result | Source |
|---|---|---|
| Route-union prefill | 126/112 s → 43 s only vs wrong baseline; through router 30.5 vs 29.9 s | lookup-drafting-2026-11-15.md:39 |
| PLE row prefetch | 33.8 s → 36.8 s; +87.3 GB reads | overnight-2026-11-16.md:188-190 |
| QSA gather tile 16/32 vs 8 (pre-fusion) | no difference; tile 128 slower, more memory | :326, :350-352, :402-404 |
| `qsa-fast-prefill` | changes 4/24 token streams, fails tool_2 quality gate | :11, :314-322 |
| Pipelined attention tiles | 0.507→0.400 s chunk but 42.8/43.1 vs 43.7/43.0 s end-to-end | :358-360 |
| Truncating invisible keys | bit-equal but time unchanged | :374-376 |
| Compiling prefill injection | bit-equal, ~25 ms over 4.3K prefill | :378-380 |
| Dequantize-all-history + bf16 gather | 0.98 s vs 0.73 s (4× bytes) | :330 |
| Fused gather rounding product separately | not bit-identical (3.4M elements differ) | :338 |
| `FLASHNEXT_COMPILE_GDN` | 41.2 → 41.3 ms scan; step 128 15% faster but changes rounding | :85 |
| `mx.async_eval` for layer-major layers | 311 vs 314 tok/s | :84 |
| Early shared expert for single-token decode | 5.85 vs 6.6 tok/s (−10%) | :86 |
| Nemotron 4096 chunk | +6% but memory pressure (see decode table) | :128-130 |
| Block-parallel Mamba | digest mismatch, slower | stage notes |

## ANE — all rejected on this 16 GB M4 (2026-11-17 research)

| Idea | Measured result | Source |
|---|---|---|
| ANE as main executor / expert cache | weights baked, INT4 unsupported, ~2 GB program cap; experts cannot be staged | maderix_CODE_deepdive.md |
| ANE single-token decode | GPU 74.9 vs ANE 23.2 tok/s (base M4, batch 1); ANE bandwidth 7.7 vs GPU 94 GB/s | BASE_M4_ANEFORGE_DATA.md |
| ANE attention / norm / router / LM head | attention 560–610× slower than CPU GQA; RMSNorm ~0.01 error; fp16 flips ties | maderix_CODE_deepdive.md |
| Core ML ANE offload (Python path) | 3.3% slower end-to-end (279 vs 289 tok/s), ~21 ms/call marshalling | REPORT_NEMOTRON.md |
| Palettized 4-bit Core ML | silently on GPU, accuracy ruined | palettized_qproj.json |
| Native PyObjC zero-copy buffer | segfault — MLX buffer unsafe for Core ML | IOSURFACE_PATH.md |
| ANE drafter + exact verify at 16 GB | no published case; MacPaw/Orbit are dense/larger chips | ane_decode_final.md |
| FusionML CPU+GPU row split (1.18–1.25× claim) | own M4 raw data: LOSES to MLX (0.763 vs 0.357 ms batch 128); no LLM artifacts | FUSIONML_FORENSICS.md |
| Bit-exact ANE intermediate layer | no cross-runtime bitwise identity exists; tie-margin is final-head, token-ID only | ane_bitexact_path.md |

## Still open (not rejected — do not add to this list)
- Full residency on 32 GB+ for Nemotron (~24 tok/s arithmetic estimate, unvalidated)
- Multi-buffer / 64-bit offsets to exceed Flash-Next's 8 GiB single-`uint32` pool cap
- Batch-K (K>1) expert-wise verification with shared reads (union ratio 0.790 Qwen / 0.826 Nemotron; K=2 measured slower, larger K untested)
- Last-row-only prefill below 2,048 tokens (TTFT/memory candidate)

## Owner decision pending
The user asked (2026-11-17) to "remove everything in the engine which is not a
speedup, but document it." The documentation half is this file. The removal half
is **not executed** because it conflicts with two hard invariants: preserve the
generic MLX Q4/G64 rollback and comparison reference, and keep an explicit
fallback for every optimization. Those paths are referenced by 18 test/doc
files. Options to decide:
1. Keep flags in code (off by default, zero runtime cost) + this registry — recommended.
2. Delete rejected flags and their code/tests — violates the fallback invariant; needs explicit owner override.
3. Move rejected flags to a separate `models/flashnext/rejected/` archive module, not imported by the runtime — middle ground.
