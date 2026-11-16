# ANEForge source verification (2026-11-17, overnight)

I cloned github.com/sbryngelson/ANEForge and read the actual source. Key findings
that refine (not overturn) the earlier "no ANE win" conclusion.

## What is actually implemented (verified in source)

- **`aneforge/moe.py`** — a complete sparse-MoE implementation on ANE from GGUF:
  routing matches HF (softmax → top-k → renorm); every expert is **baked resident
  and computed densely** (one batched matmul over the expert axis, non-selected
  experts zero-weighted); top-k without native topk/sort (peel row-max k times).
  **Expert cap: ≤2049** (expert indices carried as fp16 integers, exact to 2048).
  Both of Veystrix's models fit: Nemotron 128 experts, Flash-Next 512.
- **`aneforge/speculative.py::spec_generate`** — exact greedy speculative
  decoding: draft proposes n_draft tokens, target verifies all in one forward,
  accept matching prefix + 1 correction. Docstring: *"Greedy speculative is
  exact: it emits the same tokens as greedy."* README claims **2.28× (7.4→16.8
  tok/s)** on Qwen3-8B + 0.6B draft, verify(K)≈verify(1).
- **`aneforge/_paired.py`** — paired-fp16 (double-fp16) compensated arithmetic:
  Knuth TwoSum, Dekker TwoProduct, error-free transforms, compensated dot
  through `@ ones` (never `reduce_sum`). This is a real mechanism for higher
  precision on a fp16-only engine.
- **`aneforge/llm.py`** — the tie-margin fallback is real and in code:
  `lmhead_tie_margin = 3e-3`; if top-2 gap < margin × max|logit|, recompute logits
  on host fp32. Greedy matches host exactly; sampling stays on ANE.

## Critical caveats (from the same source/docs)

- **ANE MoE is slow and low-fidelity at our scale:** README table — Qwen1.5-MoE-A2.7B
  full model on pure ANE ≈ **2 tok/s (int8)**; Qwen3.5-27B hybrid decodes coherent
  but *"fp16-bound vs llama.cpp"*. docs/llm.md: on-device int8/fp16 decode is
  "coherent yet lower-fidelity than llama.cpp's fp32-accumulated Q4; harder prompts
  come out shallower." That fails Veystrix's bit-identical rule.
- **Exactness claims are narrow:** "greedy matches HF" / "16/16 greedy match" /
  "correlation 1.0 at every position" — these are token-level/top-1 or logit
  correlation, **not bitwise identity** of hidden states. The tie-margin only
  guards the FINAL head; an error in an intermediate layer is not caught.
- **Dense compute of all experts** is the right trade only because ANE decode is
  latency-bound and the hidden state stays on-chip — for a 512-expert Qwen layer
  computed densely, that is 512× the FFN math per token, which is why it is slow.
- Single-author project (Bryngelson); many "bit-exact vs numpy" claims are for
  isolated ops (conv, resize, slice), not end-to-end LLM token identity.

## Implication for Veystrix

ANEForge proves a full MoE and exact-by-construction speculative decoding CAN run
on ANE, but at 2 tok/s with lower fidelity at 30B-scale — far below MLX's ~8 tok/s
at higher fidelity. The directly reusable ideas are the **tie-margin fallback**
(as a bit-exactness guard for any low-precision path) and the **expert-wise dense
grouping** concept, which mirrors the K=2 read-sharing candidate already found
(but ANEForge needs all experts resident, which our SSD-streamed setup cannot do).

No change to the owner-rule target: still validate on 16 GB first; ANE remains an
opt-in island at most, not a replacement for the Metal/CPU MoE path.
