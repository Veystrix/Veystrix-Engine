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

## Investigation results (2026-11-17)
- [x] MLX C++ toolchain verified: `cpp/decode/` builds with venv MLX 0.32.2; `mlx_probe` runs (metal=1).
- [x] Decode path mapped: per-token syncs `flashnext.py:1270,1302,1309`; per-layer router sync `adaptive_topk.py:95-103`; expert route `expert_cache.py:787-788,1781-1787`.
- [x] **Critical contrary evidence:** `hostwindow.py:7-8` recorded that adding `mx.eval` + `tolist()` + host work was **free** (498.5 vs 499.3 ms/token) because both devices were idle. Production already does one sync/layer (`FLASHNEXT_ONE_SYNC=1`, `settings/launch.py:53`). The reference's 41% sync share may not apply to this codebase.
- [x] No C++ safetensors loader exists; porting the full reference stack (~330 KB) is high effort with unproven payoff here.

## Guard profile — MEASURED 2026-11-17 (owner authorized)
Guarded run clean: floor 3.86 GiB, swap growth 0, compressor peak 4593 MB, no stop.
Harness: `bench_host_window.py --tokens 24 --passes 3`, `FLASHNEXT_HOST_WINDOW=1`.
Token IDs identical across all 3 passes. Saved:
`results/flashnext/20261117-host-window-profile/` (log + guard JSONL).

Per token (300.1 ms, 3.33 tok/s on this short-prompt harness):
| Component | ms/token | share |
|---|---|---|
| `io_await` (shared, blocked on drive) | 156.86 | 52.3% |
| `route_tolist` | 12.50 | 4.2% |
| `to_mx_host` | 2.89 | 1.0% |
| `keep_loop` | 1.93 | 0.6% |
| `moe_issue_host` | 0.54 | 0.2% |
| `plan_host` | 0.36 | 0.1% |
| **all exclusive host time** | **18.22** | **6.1%** |

**Upper bound if 100% of host work were removed:** 300.1 → 281.9 ms/token =
3.33 → 3.55 tok/s = **1.065×**. A full C++ decode port cannot reach 5 tok/s on
this hardware even in the impossible best case. The dominant cost is SSD I/O
wait (52.3%), which C++ decode does not address.

## DECISION — pivot
Full C++ decode port is **DEFERRED / low priority**. The highest-value work is the
native I/O path: Method #2 (explicit sorted, gap-merged pread) and Method #3
(per-expert coalescing / sidecar), which target the 156.86 ms/token I/O wait.

This is not a quality or correctness rejection — it is a measured value judgment.
The C++ scaffold remains for later if I/O work changes the profile.

## Nemotron C++ — separate question, NOT covered by the Qwen profile

Owner asked (2026-11-17) whether C++ decode applies to Nemotron. It is a different case:

- **Different model:** local checkpoint is `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit-mlx-community`,
  `model_type = nemotron_h`, `NemotronHForCausalLM`, 52 layers, hidden 2688, 6 experts/token,
  **Q4 affine G64** (vs Qwen's Q4 affine G32). The reference C++ repo's Nemotron work is for
  74B-A9B **Bolt** and is only a llama.cpp expert-cache patch — no C++ runner to adapt.
- **Different engine:** Nemotron uses `models/nemotron35/` (own backend, `backend_core.py`,
  `streamed_experts.py`), not Qwen's `qwen4_exp`. No code or profile transfers.
- **Different sync structure:** `backend_core.py:311-327` has 2 evals/token;
  `streamed_experts.py:915,961,973` does `unique.tolist()` per layer. Host work exists but its
  share is **unmeasured**. The Qwen 1.065× ceiling must NOT be assumed for Nemotron.
- **Existing harness:** `models/nemotron35/tests/bench/bench_decode_profile.py` records per-layer
  `route_sync_and_copy_ms` (host/sync) vs `expert_dispatch_and_wait_ms` (I/O wait), and enables
  `VEYSTRIX_BENCH_TOKEN_DIGEST=1` + `VEYSTRIX_BENCH_LOGIT_DIGEST=1` quality gates.

**GUARDED PROFILE — MEASURED 2026-11-17 (owner authorized "run Nemotron teste").**
Harness `bench_decode_profile.py --tokens 96 --pool-gb 2`. Guard clean: floor 3.63 GiB,
swap growth 0. Token + logit digests captured, status `diagnostic`. Saved:
`results/nemotron35/20261117-140232-decode-profile-manual/`.

| Per token (181.4 ms, 5.51 tok/s) | ms | share |
|---|---|---|
| `expert_dispatch_and_wait_ms` (I/O) | 123.1 | 67.9% |
| `route_sync_and_copy_ms` (host sync + route copy) | 50.5 | 27.8% |
| other | 7.8 | 4.3% |

2208 MoE layer calls / 96 tokens = 23/token. Cache **hit rate only 43.7%**
(5792 hits / 7456 misses) with a 2 GB pool; **436 MB read per token**.

**Upper bound if ALL host route sync were removed:** 181.4 → 130.9 ms =
5.51 → **7.64 tok/s (1.386×)**. Better than Qwen's 1.065×, but still far short of 15.
I/O wait (67.9%) dominates, and the 43.7% hit rate shows the cache is undersized.

**Nemotron decision:** C++ / sync removal is a real but modest lever here (~1.39× ceiling),
NOT the path to 15 tok/s. Priority is increasing the expert cache hit rate (break-even rule,
Method #5) and reducing I/O wait (Methods #2/#3/#5). C++ remains optional/low priority.

## Owner decision (2026-11-17)
- **Full C++ engine port: PLANNED, NOT NOW.** The owner explicitly wants it eventually for
  future-model reuse. Keep this checklist and `cpp/decode/` scaffold as the entry point.
  Do not start the full port until the owner says so.
- **Now: I/O improvements** (Methods #2 merged pread, #3 expert coalescing, #5 cache sizing),
  targeting the measured I/O wait (Qwen 52.3%, Nemotron 67.9%) and Nemotron's 43.7% cache hit rate.

## Status
- **Current phase:** I/O improvements active; C++ full port deferred by owner decision
- **Unblocked:** owner authorized both profile runs; future A/B runs still need permission
