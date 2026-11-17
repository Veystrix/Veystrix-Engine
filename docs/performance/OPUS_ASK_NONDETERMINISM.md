# Opus: help needed — residual MLX nondeterminism in C++ prefill (gate (b) 3/3)

Paste to Opus. Repo https://github.com/joyboyreturns/Veystrix-Core branch `cpp-engine-qdense` (tip `a531e23`). Self-contained. **Read code + evidence, give diagnosis + minimal fix. Do NOT edit shared files without a bit-exact verification plan.**

## Status: bit-exact ACHIEVED, but ~5-10% of runs fail a race

- We reached **8/8 tokens AND 8/8 logit SHA-256 digests match** the reference exactly (`results/flashnext/20261117-222244-real-reference/reference.json`, tokens `[271,248068,198,760,1156,579,1876,7701]`). Canonical gate tool: `cpp/engine/tools/check_real_parity.py`.
- BUT repeated guarded runs are **nondeterministic**: e.g. 15/15 once, then 34/35, 19/20, 19/20. A failing run has step-0 (prefill) digest differing (all 8 tokens may still be OK, or step 1 flips). Gate (b) requires 3/3 clean, so this race blocks us.

## What is SOLVED and verified (do not re-investigate)
- Memory/nondeterminism gate (a): packed dense Q4 + streamed PLE + shared `plan_slots`; real run ~11 s, zero swap.
- Bit-exact arithmetic fixes (each proven by synthetic dump/probe):
  - GDN conv_state cache MUST be bf16 not f32 (`real_checkpoint_generate.cpp` ~1252). With f32, whole conv→qkv→delta chain promotes to f32.
  - Python `nn.silu` is `@partial(mx.compile, shapeless=True) x*sigmoid(x)`. C++ must emit the SAME compiled kernel — see `hc_silu` in `cpp/engine/src/engine.cpp`. Unfused `x*mx::sigmoid(x)` differs by 1 ULP on ~12/5120 elements. Applied to: gated_residual, final_head, final_mixed_no_combine (prefill_batched.cpp), swiglu, GDN prefill conv (prefill.cpp), GDN decode conv (gdn_conv.cpp), rms_norm_gated gate, PLE conv (ple.cpp).
  - `mx::sigmoid`, bf16 `/int`, `mx::mean` are bit-exact (ruled out).
  - MoE (router, gather_qmm, swiglu, weighted combine, shared expert) all byte-exact with identical inputs (probe in `cpp/engine/tools/probe_moe_intermediates.cpp`, report `docs/engineering/moe_drift_localization.md`).
  - Head kernel: step-0 uses a custom metal kernel in Python but it is bit-equal to C++ `quantized_matmul` (script verified).

## The race — precise evidence
- Failing vs passing guarded run, full per-layer dumps (`REALGEN_DUMP_BATCH_LAYERS=<prefix>`): **first divergence is always L11**, a full-attention layer. L3/L7 (also full-attn) are stable.
- At L11: `gdnin` (attention input = `ar.mixed_input`) is **100% bit-exact** between runs; `branch` (attention output from `full_attention_dense_batched`, `prefill_batched.cpp:26-83`) diverges with **maxdiff ~1.89** (LARGE, not 1 ULP — suggests reading stale/uninitialized KV cache or q/k/v, not rounding).
- `mx::fast::scaled_dot_product_attention` is deterministic in ISOLATION for the exact prefill shape `[1,24,16,256]` q / `[1,2,16,256]` kv (tested 6× in 3 processes, identical hashes).
- So: identical attention INPUT, deterministic SDPA kernel, yet attention OUTPUT differs run-to-run → the q/k/v projections or the KV cache the SDPA reads differ. Suspect MLX lazy-eval / buffer-reuse aliasing with compiled-silu graphs across the 48-layer chain.

## Things tried (efficacy)
- `REALGEN_DUMP_BATCH_LAYERS` dumps ON (holds refs to all intermediates + evals f32 casts of each per layer in the tool): 6/6 deterministic.
- In-loop `mx::eval` of `ar.mixed_input`, `branch`, `mr.mixed_input`, `moe2d`, `hidden` inside `run_prefill_batched`: up to 15/15 once, then 34/35, 19/20 — reduces but does not eliminate.
- Tool-side eval of f32 casts of all `BatchedPrefillDebug` arrays after graph built: 16/20 (worse).
- Non-static `mx::compile` in hc_silu: 9/10 (no help).
- `mx::eval(out)` inside hc_silu: testing now.
- `MLX_METAL_FAST_SYNCH=1`, `MLX_MAX_OPS_PER_BUFFER=1` env: no reliable fix (MAX_OPS also changes numerics → 0/20).

## Questions
1. Given identical `gdnin` and a deterministic SDPA kernel, what is the most likely source of the large (~1.89) run-to-run difference in the full-attention `branch` at L11? Is it the KV cache (`caches[l].full.keys/values` built via `mx::concatenate`?), the q/k/v `linear()` (quantized_matmul on QPacked weights), or the `rope_apply_fused`?
2. How do I make this deterministic in a way that matches Python's evaluation pattern (Python is deterministic with the same MLX 0.32.2)? Is there an MLX C++ API (`mx::eval` placement, `mx::synchronize`, streams, `mx::fast` compile mode) that forces the needed serialization?
3. Is the shared static `mx::compile` shapeless graph in `hc_silu` fundamentally racy when invoked many times across a deep lazy graph? If so, what's the correct thread-safe / deterministic pattern that still emits the fused kernel?
4. Could the issue be `mx::array` views aliasing persistent host buffers (the streamed MoE slot pool in `moe_stream.cpp`, or QPacked weight buffers in `DenseLoader`)? What is the standard MLX fix (copy on use? eval before reuse?)
5. What is the cleanest way to guarantee 3/3 (ideally 100%) determinism for this gate without changing numerics?

Constraints: no quality loss (must keep 8/8 bit-exact digests when it passes); Python stays serving engine until gate (b) passes 3/3; all real runs under `scripts/flashnext/real_run_guard.py`, 5 GB swap cap. Never push to main.

Build: MLX=`/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages/mlx`; `xcrun clang++ -std=c++20 -O2 -I$MLX/include -Icpp/engine/src -Icpp/decode/src <all src/*.cpp except decoder.cpp,model_adapter.cpp> cpp/decode/src/io_pager.cpp /tmp/nr_qd2.o cpp/engine/tools/real_checkpoint_generate.cpp -L$MLX/lib -lmlx -Wl,-rpath,$MLX/lib -o /tmp/probe`.
Run one: `cd /Users/vincent/Documents/Veystrix && .venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 5120 --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 600 --log /tmp/xshard_guard/opus.jsonl -- .venv/bin/python cpp/engine/tools/check_real_parity.py --binary /tmp/probe --reference results/flashnext/20261117-222244-real-reference/reference.json --checkpoint /Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP --config /tmp/xshard_guard/real_config_ple.txt --tokens 8`
