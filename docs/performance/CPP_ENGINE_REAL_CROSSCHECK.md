# C++ engine — real-checkpoint cross-check result (2026-11-17)

## Verdict: FAILED. Do NOT drop the Python engine.

The C++ engine is bit-exact on tiny synthetic configs (20/20 ctests, full native
token generation from npy weights). But the decisive gate — matching Python's
greedy decode **on the real 125B Qwen checkpoint** — FAILED:

- Guarded Python reference (`results/flashnext/20261117-222244-real-reference/reference.json`,
  clean: floor 2.93 GiB, swap flat, 8 tokens):
  `tokens=[271, 248068, 198, 760, 1156, 579, 1876, 7701]` + per-step logit SHA-256.
- C++ `real_checkpoint_generate` on the same checkpoint/prompt: **0/8 tokens and
  0/8 logit digests match**. First divergence at step 0 (C++ token 145166 vs
  Python 271). Strict gate exits 1 — NOT weakened.
- **Nondeterminism**: three C++ runs gave three different step-0 tokens
  (186204, 145166, 229910). Likely `mx::quantized_matmul` split-K/atomics in
  `lm_head`; GDN kernel / fused SDPA are other candidates. A stable SHA-256 gate
  is not achievable this way even if the math were numerically close.
- A later rerun crashed with SIGBUS (-10) at ~13.5 GB peak during weight loading
  — the full 48-layer load is near the memory ceiling on this 16 GB M4.

## Why it diverges (documented, not papered over)
- quantized vs dense matmul across ~48 layers of projections
- missing PLE layer at layer index 1
- sequential single-token prefill vs production's chunked/layer-major ring
- dense KV cache vs production INT8 KV
- eager vs fused RoPE in the real path
- dequantized shared expert
- no prefix-cache / MTP replay

## Decision
The precondition for dropping Python ("engine works completely and cross-checks
everything including the app") is NOT met. **Keep the Python engine.** The C++
engine remains a validated native core/library on synthetic configs; making it
a drop-in for the real model requires closing the divergence list above and
resolving nondeterminism — substantial further work.

Artifacts: `cpp/engine/tools/{real_checkpoint_generate.cpp,gen_real_config.py,check_real_parity.py}`,
`research_notes/Cerebras inference and IO splitting/real_checkpoint_generate.md`,
and the preserved reference dir above. All on branch `cpp-engine`.
