# Direct search leads — second miracle (no new hardware), 2026-11-17

Constraints: 16 GB M4 Air ONLY (no SSD/Mac/RAM purchase), no quality loss, ANE allowed.

## 1. ANE — MacPaw research MATERIALLY overturns our earlier rejection
- MacPaw "LLM inference on Apple Neural Engine" (research.macpaw.com/projects/neural-engine, 2026-10): M4 Pro ANE decode 32.2 tok/s plain, 98.1 chain, **136.9 tree** vs GPU 8-bit MLX 81.5, +DSpark 144.9. ANE-only used 3.4× less energy, lower latency. ANE prefill 1.9× faster than GPU.
- CRITICAL for us: "Dividing the weight reads of a decoding step between ANE and GPU: at most 5-11% more bandwidth — the two units share one memory system." So ANE cannot beat the memory/SSD wall; it can only help if it removes COMPUTE or enables better speculation, not more bandwidth.
- Quality: per-channel INT4 11-46% faster but top-1 agreement fell to 57%; group-wise INT4 (preserves quality) does NOT run on ANE. INT8 activations 2.4× FFN but ~10× error. Palettized LUT supported single-layer only, full model fell back to GPU.
- M4 ANE internals (maderix.github.io/articles/inside-the-m4-ane-part-4): fp16 default, W8A8 packed mode 36 TOPS, int8 weights+fp16 activations stay fp16 path (no speedup). Group-wise/affine int4 NOT supported bit-exactly.
- Verdict: ANE can do fast fp16/int8 SPECULATIVE decoding (tree verification) but (a) it does not change SSD bytes, (b) our Q4 affine G32/G64 MoE has no bit-exact ANE path, (c) it shares the memory bus. Only plausible role: exact speculative verification with ANE running the draft head, IF that reduces bytes (it doesn't) or enables sharing (unproven). Low payoff for our byte-bound problem but worth asking Opus given owner interest.

## 2. int4 GEMV kernel layout — a real, bit-exact compute win (but compute isn't our wall)
- M4 Max profiling (john-rocky/apple-silicon-llm-bench): one-simdgroup-per-row int4 GEMV streams bytes 22-28% faster than K-sliced threadgroup; int4-blockwise-gs32 is dequant/ALU-limited not DRAM-bound on small dense models. Bigger quant blocks made it WORSE. Not directly our Q4 G32 MoE, but a bit-exact kernel-layout idea for the compute slice (~81 ms Qwen) — which overlaps with I/O (E0), so limited payoff.
- CPU SME units (M4): 3.3-3.7 TFLOPS fp32, usable via MLX CPU stream. Another bit-exact compute island; shares memory bus, so cannot beat byte wall.

## 3. Implication for the "second miracle"
The owner forbids more SSD/Mac/RAM. E0/E1/E3 proved: overlap helps +25% max, prediction can't reach targets, byte wall (873 MB @ 3.36 GB/s = 260 ms) is the blocker. ANE cannot add memory bandwidth (shares bus) and has no bit-exact Q4 path. So the ONLY remaining physical levers under the constraint are:
  (a) FEWER BYTES via a lossless representation (scale/bias entropy coding ~10% cap; cross-token exact union sharing; compute-on-compressed) — see lossless research.
  (b) SERVER PATH GAP: 2.3 production vs 3.33 short harness ~135 ms/token unaccounted — see compute_server research. This is the most promising "free" win because it does not require beating the SSD wall, just removing non-SSD overhead.
  (c) ANE speculative *verification* as exact decode acceleration IF it shares expert reads across tokens — but bytes unchanged, so likely small.

These are the leads to press Opus on. Explicitly forbid it from proposing second SSD/Mac/RAM.
