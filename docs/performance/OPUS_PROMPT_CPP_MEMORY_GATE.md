# Opus prompt: C++ real-checkpoint memory blowup blocks the final parity gate

Paste this to Opus. It is self-contained; the repo is public.

---

## Request

You are looking at the Veystrix C++ inference engine. I need a strong second opinion on a **memory** problem that blocks the last validation gate. Please read the repo and give me: (1) a diagnosis you can defend from the code, (2) the least-risky concrete fix with file/line references, (3) explicit answers to my 6 doubts at the end. Point to exact files/lines, not just hypotheses.

## Repo / branches

- Repo: https://github.com/joyboyreturns/Veystrix-Core
- Branch `cpp-engine` (local tip `be81251`, remote `ef690b7`): C++ engine + the standalone real-checkpoint generator `cpp/engine/tools/real_checkpoint_generate.cpp` where the memory problem lives.
- Branch `cpp-engine-xshard-fix` (tip `5e3a5bf`, **PR #2**): a merged-ready slot-bookkeeping fix that already solved the nondeterminism. Read for context.
- Never touch `main`.

## Background — where we are

Goal: C++ engine must be bit-identical to the Python engine on the real ~105 GB Qwen3.8-Flash-Next MLX-4bit checkpoint (exact tokens, logit SHA-256, BF16 rounding, routing). Python is deterministic and remains the serving engine until parity passes. Python ground truth: `results/flashnext/20261117-222244-real-reference/reference.json`, tokens `[271, 248068, 198, 760, 1156, 579, 1876, 7701]`.

**Good news, already done:** the nondeterminism is FIXED by PR #2 (shared `plan_slots()` in `cpp/engine/src/slot_plan.cpp`, used by both `StreamingMoE` and `CrossShardMoE`). Gate (a) passed: two guarded 10-layer runs were bit-identical across all 204 captured hashes. So this is no longer a correctness-of-bookkeeping question.

**The blocker:** gate (b), the full 48-layer cross-check, cannot complete on this 16 GB M4 because the C++ tool thrashes swap. Every real run is under `scripts/flashnext/real_run_guard.py`; do NOT run the real checkpoint yourself.

## The memory problem — measured evidence

Guarded full run (swap-growth limit 5120 MiB, compressor 8192 MiB, floor 1.5 GiB, thermal 99):
- ran 270 s, then guard stopped it: `swap grew 5156 MiB over launch`, `swap_peak_mib=8906` (~8.7 GB), `swapout_mib=53772` (52 GB of swapouts), `compressed_gib_total=202`, `available_floor_gib=1.93`.
- Python's reference capture on the same machine/checkpoint used the SAME envelope and finished in 11 s with `swapout_mib=0`, compressor peak +4435 MiB.

I quantified the anonymous-memory footprint directly from the safetensors headers:

| | packed (mmap, evictable) | dequantized bf16 (anonymous, swapped) |
|---|---|---|
| C++ dense weights | 1.71 GiB | **6.83 GiB** |
| Python dense weights (keeps packed) | 1.71 GiB | 0 |
| expert slot pools (both) | — | 2.20 GiB |
| **total anonymous** | | **C++ 9.03 GiB vs Python ~3.91 GiB** |

Root cause (confirmed from code): `DenseLoader::dense()` (`real_checkpoint_generate.cpp:274-340`, especially 302-315) calls `mx::dequantize(w, s, b, group_size, bits, "affine")` then `mx::eval(out)` for **741 dense quantized tensors**, and the decode loop (`~1084-1172`) holds every layer's dequantized bf16 copy in `std::vector<RealLayer> layers` for the whole run. Python keeps these packed and multiplies with `nn.quantize` / `quantized_matmul`. The ~5 GiB difference is exactly what causes the thrashing, and it is also the documented "quantized vs dense matmul" logit-divergence gap in `docs/performance/CPP_ENGINE_REAL_CROSSCHECK.md`.

## Relevant code

- `cpp/engine/tools/real_checkpoint_generate.cpp:274-340` — `DenseLoader`: `raw()` uses `mx::load_safetensors` (mmap, fine); `dense()` does the eager dequantize + eval (the problem); `quantized()` keeps embed/lm_head packed.
- Every dense projection is a plain `mx::matmul(x, mx::transpose(W))`: `cpp/engine/src/engine.cpp:154-185`, `gdn_layer.cpp:37-83`, `decoder_full.cpp:55-217`, `prefill.cpp:165-220`, `qsa.cpp:36`, `qsa_full.cpp:165-206`, `parity_gaps.cpp:40-157`.
- The packed expert path already exists and works bit-exactly: `engine.cpp` `q4_projection()` uses `mx::gather_qmm` on packed weights/scales/biases; `engine.hpp:28-48` defines `Q4Projection`/`ExpertWeights`.
- MLX C++ API (0.32.2): `mx::quantized_matmul(x, w, scales, biases=..., transpose=true, group_size=..., bits=..., mode="affine")` at `ops.h:1567`. Also `mx::quantize` / `mx::dequantize`.
- Build of the tool adds `src/slot_plan.cpp src/cross_shard_moe.cpp`; command in `research_notes/Cerebras inference and IO splitting/real_checkpoint_generate.md:56-66`.

## What I already ruled out (do not re-propose)

- Raising guard limits (swap 5120 → still thrashed; compressor 8192 → still stopped on swap). The machine is otherwise stable; this is a real 5 GiB anonymous-memory excess, not a guard-tuning issue.
- Memory-mapping the dequantized weights: MLX copies weights into anonymous Metal buffers when they are used in `matmul`, so they become swappable again. Does not help.
- The slot-bookkeeping nondeterminism (PR #2) — solved and gate-(a)-verified.

## My 6 doubts — answer each explicitly

1. **Will routing dense projections through `mx::quantized_matmul` on packed weights be bit-identical to Python's `nn.quantize` + quantized matmul, and to the current dequantize+`matmul`?** Python parity is the hard gate. If `quantized_matmul` and `dequantize`+`matmul` differ by even 1 ULP, which one does production Python actually use for these dense projections? Check `models/flashnext/loader.py` (`nn.quantize` at ~230) and the mlx-lm layers.

2. **What is the exact layout/parameter mapping?** Packed safetensors weight is `[out, in/8]` uint32, scales `[out, in/group]` bf16. For `mx::quantized_matmul`, what `transpose`, `group_size`, `bits`, `mode` reproduce `matmul(x, transpose(dequantized_w))`? The current code pre-transposes the dequantized weight; does `quantized_matmul`'s `transpose=true` expect the packed `[out,in/8]` layout directly?

3. **Is there a lower-risk fix that keeps the current arithmetic bit-for-bit but cuts memory?** Idea A: dequantize one layer's dense weights on demand, run all 24 steps through that layer, free, next layer — but generated tokens are unknown until all layers finish, so layer-major only works for the known prompt tokens. Idea B: keep packed weights resident (1.71 GiB mmap) and dequantize+eval a single layer's weights per token, free immediately (peak anonymous ~150 MB, cost ~1152 small evals over 24 steps). Are either of these sound, and which preserves the exact per-layer math? Is there a better variant?

4. **Will the full run even fit after the fix?** Expert pools 2.20 GiB + KV caches + lm_head packed + activations. Estimate the post-fix peak anonymous memory and whether gate (b) completes under a 1.5 GiB floor / 2048 MiB swap-growth envelope.

5. **Does gate (b) actually require all 48 layers, or is a smaller faithful config acceptable?** The task wants exact token/digest parity vs the Python reference, which used all 48 layers. Confirm there is no shortcut that still counts as the parity gate.

6. **What is the minimal change set with the best risk/bit-exactness tradeoff?** Options I see: (i) full quantized-matmul rewrite of all dense projections + regenerate parity vectors; (ii) on-demand per-layer dequantization (Idea B) touching only the tool's decode loop; (iii) something you propose. Which do you recommend, and exactly which files change?

## Constraints

- No quality loss: bit-identical tokens, logits, BF16 rounding, routing. "Close enough" is not acceptable.
- 16 GB M4 validation; all real-checkpoint runs need owner permission + guard. Do not run them.
- I verify any proposal against the bit-exact gate before touching shared primitives.
