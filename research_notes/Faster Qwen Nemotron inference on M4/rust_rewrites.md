# Why AI is being rewritten in Rust — and relevance to Veystrix (2026-11-17)

## Why Rust is spreading in AI
- The rewrite is mostly **around the model**, not the model math. Rust owns the
  always-on layers: tokenizers (HF `tokenizers`), file format (`safetensors`), and
  serving frontends (vLLM v0.27.0 and SGLang v0.5.17 rewrote frontends in Rust).
  Motivation: Python frontend became CPU-bound, no GIL, no GC, single binary,
  memory safety. Sources: dev.to "Why software is converging on Rust", acingai
  "Your LLM Serving Bottleneck Moved to the CPU", fratepietro.com Rust inference guide.
- vLLM RFC #40846: 1 Rust frontend 837 req/s vs 32 Python API processes 786 req/s;
  TTFT 166 → 50.5 ms. Cloudflare Infire: up to 7% faster than vLLM on H100 but CPU
  25% vs 140%.
- Rust is **not** winning GPU kernels, research code, or games. Kernels remain
  C++/CUDA/Triton. HF TGI (Rust core) is in maintenance mode, pointing to vLLM/SGLang
  (Python/C++/CUDA). "Writing an engine in Rust doesn't make it fast — speed comes
  from the kernels." (fratepietro.com)

## Directly relevant findings for our case
1. **th-engine (Rust/candle) vs Splash on M5 Max** (benebsworth.com, 2026-09-28):
   went 0.56x → 1.20x Splash decode on same 27B 4-bit model. The gap was NOT kernels:
   "the kernels were not the gap." Fixes were host-sync/command-buffer reductions —
   blocking host syncs per round **28 → 1**, command buffers per round **86 → 1**,
   GPU idle **25 ms of 87 ms**, cold 4 MiB allocations 48 → 7. Also prefix cache
   repeated TTFT 2.5 s → 22 ms; bitwise-identical prefill GEMM tile 36.6 → 43.9 TFLOPS.
   This confirms our measured sync-removal lever is real but bounded.
2. **Ferrox (Rust GGUF, Metal)** MoE prefill: cut command buffers from >100 to ~1 per
   prefill → within ~1.1x of llama.cpp (note.com/h3adeu, 2026-09-07). Same lever as
   our roadmap Method #7 (MLX command-buffer limits), still untested.
3. **mlxcel (Rust binary calling MLX C++ runtime)**: same kernels as mlx-lm, decode
   within a couple percent (kubesimplify, 2026-05-29). Confirms wrapper language
   does not matter when kernels are shared.
4. **MetalRT (C++!, not Rust)** 658 tok/s Qwen3-0.6B on M4 Max, 1.10–1.19x mlx-lm
   using the *same model files*, 1.35–2.14x llama.cpp (runanywhere.ai, 2026-03-03).
   The fastest decode engine here is C++, not Rust.
5. Frink validates every architecture against llama.cpp logits and refuses to load
   without that evidence — same philosophy as our bit-identical digest gate.

## Verdict for Veystrix
- Rust per se will not fix our bottleneck. Our measured splits are I/O wait
  52.3% (Qwen) / 67.9% (Nemotron), host work 6.1% / 27.8%. Rust targets host/serving
  overhead, which is the smaller share here — and for Qwen it is already free
  (hostwindow.py:7-8).
- The transferable lessons are language-independent and already in our roadmap:
  #7 command-buffer limits, #9 memory residency, #13 prefix caching, #1 native port.
- Rust adds nothing for the SSD read volume (~408 MB/token Qwen, ~436 MB/token
  Nemotron). It cannot change the 16 GB bandwidth ceiling.
- Conclusion: no new method to add. Rust is a packaging/serving choice, not a
  speedup for this specific workload. Do not divert from the roadmap.
