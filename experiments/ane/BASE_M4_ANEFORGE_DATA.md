# Base 16 GB M4 — ANEForge's raw measured data (offline, 2026-11-17)

This is the first source that measures ANE vs GPU on the **exact machine class**
the owner target: Apple M4, `Mac16,12`, **16 GB**, 8 GPU cores, macOS (26.x),
from ANEForge's own committed results:
`bench/results/rooflines/roofline-apple-m4-Mac16_12-d4109cd41a75-6779bad2.json`.

## GEMM peak (GFLOP/s)
| Device | GEMM peak | Conv peak | Memory BW |
|---|---:|---:|---:|
| ANE | 5,435 | 9,495 | **7.7 GB/s** |
| GPU | 2,821 | 7,069 | **93.9 GB/s** |
| CPU | 1,542 | 10 | 36.0 GB/s |

ANE has ~1.9× the GEMM compute of GPU, but **12× less bandwidth**. This is the
whole story: ANE wins at compute-bound, SRAM-resident matmul; it loses badly
whenever weights/activations must stream.

## Full 8-layer decode (d_model 2048, 8 layers, 16 heads, d_ff 8192, vocab 32K)
| Batch | GPU tok/s | ANE tok/s | ANE8 tok/s | ANE relerr |
|---:|---:|---:|---:|---:|
| 1 | **74.9** | 23.2 | 45.2 | 2.5e-2 |
| 2 | **84.0** | 45.3 | 83.6 | 2.4e-2 |
| 4 | **158.9** | 81.1 | 140.8 | 2.5e-2 |
| 8 | **291.2** | 137.2 | 213.5 | 2.5e-2 |
| 16 | **499.7** | 207.5 | 269.0 | 2.5e-2 |
| 32 | **736.8** | 275.9 | 286.5 | 2.5e-2 |

**GPU beats ANE at every batch size**, even where ANE should shine (large batch).
Even the 8-way replicated ANE8 loses to GPU at all sizes. ANE relative error
2.5–5% — far beyond bit-identical.

## LLM decode (Qwen3-0.6B fp16 on ANE, `bench/mlperf/results/llm_decode_fp16.json`)
p50 18.68 ms/token ≈ 53.5 tok/s on the *contributor's* machine (not necessarily
this M4); relerr of the ANE body is high. ANEForge's own README says decode
"coherent yet lower-fidelity than llama.cpp."

## Verdict — this ends the ANE question on 16 GB
On the owner's exact hardware class, the independent raw data shows:
1. ANE GEMM is faster but ANE bandwidth is 12× lower — MoE experts are streamed
   from SSD (the dominant cost), which ANE cannot accelerate.
2. In complete decode, GPU is 1.6–3.2× faster than ANE at every batch size.
3. ANE error is 2.5–5%, disqualifying bit-identical outputs by a wide margin.

There is **no ANE-based prefill or decode win** for Veystrix on 16 GB. The
productive path remains the non-ANE expert-wise K=2 read sharing.
