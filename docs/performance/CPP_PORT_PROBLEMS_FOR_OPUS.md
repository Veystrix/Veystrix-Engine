# C++ core-engine port: problems encountered (owner will forward to Opus)

Branch `cpp-engine-qdense`. This is a living list of concrete blockers/problems found while
porting the Veystrix Python inference engine's core + profitable features to C++.
Label: [measured] / [estimate] / [cited]. Update as work proceeds.

## P0 — Core engine not bit-exact beyond the captured 16-token shape [measured]

- Gate (b) passes at 16-token prefill: 3/3 guarded runs, 8/8 tokens + 8/8 logit SHA-256,
  no env vars (cpp-engine-qdense @ `104ee38`+). Canonical tool `check_real_parity.py`.
- At a 1024-token prompt (seed 20261117, config `/tmp/xshard_guard/real_config_1024.txt`,
  Python ref `/tmp/qwen_1024_ref`), C++ diverges:
  - step 0: token 271 = 271 [OK] but logit digest `db1d2671…` vs Python `269821a8…` [DIFF]
  - step 1: token 248046 vs Python 96220 [DIFF]; steps 2-3 also differ.
- Python config confirmed identical (prefill_chunk_tokens=0 one-call, bf16 KV, MTP off,
  fused RoPE) — not a config mismatch.

### Root-cause analysis so far
- The divergence is **systematic, not a race**: two C++ runs at S=128 are 100% identical
  run-to-run (all 48 layer hidden dumps). [measured]
- First divergence at S=128 is **L00**, a GDN (linear-attention) layer, maxdiff 0.00195
  (~1-2 bf16 ULP), 72% exact. At S=16 L00 is 100% exact. [measured]
- Therefore some op in the GDN/attention/MoE chain has kernel-dispatch or reduction
  behaviour that changes with S (16 vs 128+). Candidates under investigation:
  1. GDN `conv1d` over S tokens + fused delta kernel reduction order.
  2. Full-attention SDPA with 128 queries (MLX may pick a different kernel than 1/16).
  3. MoE per-token chunking (C++ 16-slot pool, M=1 gather_qmm) vs Python's batched
     gather_qmm over all S rows at once (different kernel / reduction at large M).
  4. PLE n-gram embedding (fixed eager-eval; see P1).
- A diagnostic subagent is localizing this from existing dumps; result pending.

## P1 — Lazy IO nodes inside the GPU graph (race) [measured, fixed]

- Dense weights: `DenseLoader::raw()` returned lazy `mx::load_safetensors` Load nodes;
  their first read happened inside prefill GPU graph via cross-stream fences → ~1.89
  maxdiff at L11, ~5-10% nondeterministic runs. Fixed by eager `mx::eval` at load
  (Opus diagnosis). A/B `VEYSTRIX_CPP_LAZY_DENSE=1` reproduces (9/10).
- PLE n-gram embedding: streamed path wraps malloc'd `pread` buffers in lazy arrays,
  same race at long S. Fixed by eager `mx::eval(embeddings)`.
- Token embed dequant: eager eval added in tool.

## P2 — Performance: C++ slower than Python at long context [measured]

- 1024-token prefill+4 decode: C++ 174 s vs Python 30 s (5.8x slower). Before perf fixes.
- PLE streamed reader issued one `pread()` per row (~49k syscalls at S=1024). Fixed by
  coalescing sorted contiguous runs into batched reads (bit-exact; commit `a599c8a`).
  At S=128, wall 23 s with `sys=20 s` → I/O syscalls still dominate; likely MoE
  per-token re-reads (48 layers x S tokens) are the remaining bottleneck.
- Need to measure end-to-end after the coalescing fix at 1024 (blocked on memory, see P3).

## P3 — Test environment / guard limits [measured]

- Repeated real-checkpoint runs push swap baseline up (2665 -> 3328 MiB), so the
  owner-approved 5 GB swap-growth guard stops new 1024-token runs even though thermal
  stays at level 0. Must let memory settle (2746 MiB) before long runs.
- A localization re-run was stopped by thermal guard (level 2 for 15 s) after back-to-back
  1024-token runs. Heavy runs need cool-down gaps.

## P4 — Feature gaps vs Python core engine [cited from feature inventory]

- Nemotron: NO C++ decode path (nemo_pool.cpp is a pool component only).
- Prefix cache: not implemented in C++.
- KV tiers (INT8/INT4/tiered4), MTP, continuous batching, non-greedy sampling,
  expert-pool policies (s3fifo/etc), prefetch: not in C++ (see
  `docs/performance/CPP_PORT_PROBLEMS` inventory for full NUMERIC/NON-NUMERIC split).
- Priority per owner: core engine + features that give profit, incrementally, with
  tests; multi-session Phase 0 is done on a separate branch but on hold.

## Questions for Opus

1. What is the most likely S-dependent op causing the systematic 1-2 ULP divergence at
   L00 between S=16 (exact) and S=128 (divergent), given every primitive is bit-exact
   with identical inputs? How to isolate fastest?
2. Is the MLX `mx::compile` fused silu graph (used for nn.silu parity) safe to reuse as a
   static shapeless function across 48 layers, or can it introduce shape-dependent
   rounding?
3. How to make C++ faster than Python at long context — the batched-vs-per-token MoE
   gather_qmm kernel dispatch, and PLE/MoE I/O batching — without breaking bit-exactness?
4. Given Python's non-greedy sampler uses global RNG state, what is the cleanest way to
   match it bit-for-bit in C++?
