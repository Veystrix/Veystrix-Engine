# Real-checkpoint determinism finding (2026-11-17)

## Python reference is DETERMINISTIC
Three independent guarded runs of `capture_real_reference.py` on the real 125B
checkpoint (same prompt/fixed token ids) produced identical output:
- tokens: `[271, 248068, 198, 760, 1156, 579, 1876, 7701]`
- step-0 logit SHA-256: `1638ab11b2debceb...`
Runs: `results/flashnext/{20261117-222244-real-reference,20261117-225329-pydet,20261117-225341-pydet}`.

## Implication
The C++ real-checkpoint generator's nondeterminism (3 different step-0 tokens:
186204, 145166, 229910) is a BUG in the C++ port, NOT a hardware property of
this M4. Bit-exact parity on the real model is therefore achievable in principle
— the target (Python digest) is stable. The fix belongs in the C++ port:
- likely `mx::quantized_matmul` split-K/atomics in lm_head/projections, OR
- uninitialized cache/array memory, OR
- map-iteration/order dependence in weight loading.

A subagent is isolating the exact op. Once deterministic, the real-checkpoint
divergence list (PLE layer, chunked prefill, INT8 KV, fused RoPE, dequantized
shared expert, quantized-vs-dense matmul) can be closed one at a time against a
stable reference.

Python engine stays until the guarded real-checkpoint cross-check passes.
