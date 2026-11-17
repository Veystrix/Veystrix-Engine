# Opus prompt: C++ real-checkpoint bit-exactness — final blocker

Paste to Opus. Repo: https://github.com/joyboyreturns/Veystrix-Core , branch `cpp-engine-qdense` (tip `6ab4e0d`). Self-contained.

## Request
Help me achieve bit-exact parity of the C++ engine vs the Python reference on the real Qwen3.8-Flash-Next checkpoint. Read the code and evidence, then give your diagnosis + minimal fixes. We have already fixed memory and nondeterminism; this is purely the remaining arithmetic divergence. Answer my 5 doubts at the end explicitly.

## Goal / hard gate
C++ greedy decode must match Python reference `results/flashnext/20261117-222244-real-reference/reference.json` exactly: tokens `[271, 248068, 198, 760, 1156, 579, 1876, 7701]` and per-step logit SHA-256, 3/3 runs. Python is deterministic. No quality loss. All real runs under `scripts/flashnext/real_run_guard.py`, 5 GB swap cap.

## What is FIXED and verified
- Nondeterminism (shared `plan_slots`, gate (a) passed).
- Memory: packed dense weights + streamed PLE n-gram embedding; real run 11 s, zero swap, 3.2 GiB peak.
- Many dtype/convention bugs found by comparing to Python source:
  - all Qwen4ExpRMSNorm gains are one-centered (patch_rmsnorm) — fixed hc_norm, attention q_norm/k_norm, PLE norms;
  - whole stack must stay bf16: fixed int-scalar divides (C++ `x/float` upcasts bf16→f32), delta-kernel output cast, PLE conv_state dtype, PLE gate scalars, gdn_normalize_qk and compute_g_beta (proven bit-exact at layer 20 with identical inputs).
- Synthetic bit-exact ctests: 27/28 pass (the one failure is an unrelated worktree `.venv` path artifact).

## Current measured state (real checkpoint, guarded)
- Step-0 token 271 OK, step-1 token 248068 OK. Steps 2–7 wrong.
- Step-2 is an EXACT TIE in Python (tokens 198 and 271 both logit 25.25, gap 0.0). C++ residual drift breaks the tie toward 271, cascading.
- Per-layer hidden correlation (C++ vs Python, last prefill token): L0=0.99998 (38.6% exact elements, maxdiff 0.0029), L1=0.99996, L3=0.9999, L5=0.9983, L10=0.998, L20=0.998, L30=0.989, L40=0.974, L47=0.986. All hidden states now 100% bf16-representable.
- Logits at step 0: corr 0.98, maxdiff ~1.5 (bf16 values, but different).

## Key contradiction to resolve
GDN branch is byte-exact at layer 20 when fed IDENTICAL inputs (proved by `cpp/engine/tools/probe_layer_intermediates.cpp` + `dump_python_intermediates.py`). Yet L0 diverges (maxdiff 0.0029) with IDENTICAL embed input (embed is bit-exact bf16). So a non-GDN op at L0 diverges. A subagent is probing the MoE branch at L0 (`probe_moe_intermediates.cpp`, `dump_python_moe_intermediates.py`) — its result may not be ready.

## Files / commands
- Tool: `cpp/engine/tools/real_checkpoint_generate.cpp`; batched prefill `cpp/engine/src/prefill_batched.{hpp,cpp}`; primitives `cpp/engine/src/{engine,gdn_layer,prefill,ple,qsa,qsa_full,decoder_full,moe_stream,cross_shard_moe}.cpp`.
- Dumps: `REALGEN_DUMP_LAYER_HIDDEN=<dir>`, `REALGEN_DUMP_LOGITS=<prefix>`; Python `dump_python_layers.py`, `dump_python_head.py`.
- Diagnostic notes: `research_notes/Cerebras inference and IO splitting/determinism.md`, `docs/engineering/gdn_bit_exact_drift_localization.md`, `docs/performance/CPP_PARITY_GAPS.md`.
- Build: `MLX=<venv>/site-packages/mlx`; xcrun clang++ -std=c++20 -O2 -I$MLX/include -Icpp/engine/src -Icpp/decode/src <all src/*.cpp except decoder.cpp,model_adapter.cpp> cpp/decode/src/io_pager.cpp /tmp/nr_qd2.o cpp/engine/tools/real_checkpoint_generate.cpp -L$MLX/lib -lmlx -Wl,-rpath,$MLX/lib -o /tmp/qd.

## My 5 doubts
1. Given all primitives are individually bit-exact, what is the most likely source of the residual ~0.003 maxdiff at L0 with identical bf16 embed input — MoE router scores, gather_qmm scales loading, weighted_combine reduction, or attention SDPA mask? How do I isolate it fastest?
2. Python `gather_qmm` uses `sorted_indices` option; C++ `q4_projection` passes false. Could that change routing/expert output at L0 even with identical inputs? (Subagent is probing.)
3. Step 2 is an exact Python tie. Even if we get within 1 ULP, argmax may still flip. Is bit-exact logit parity sufficient to reproduce Python's tie-breaking, or is there an explicit index-order rule we are missing?
4. We feed bf16 x to quantized_matmul and get bf16 logits, matching Python. But per-layer hidden maxdiff grows with depth. Is this accumulation inherent to any 1-ULP kernel difference, meaning we must match EVERY op exactly — and what is the realistic path?
5. Is there any chance the Python reference used a different effective config (prefill chunk, KV mode, PLE params) than we assume? What would prove/disprove that fastest?

## Constraints
- No quality loss; bit-identical tokens + digests. Python remains serving engine until gate (b) passes. Do NOT write engine code myself without a bit-exact verification plan.
