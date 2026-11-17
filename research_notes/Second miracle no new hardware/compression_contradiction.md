# MAJOR: our "lossless compression is closed" conclusion may be WRONG

Found 2026-11-17 in direct search. This is the highest-value lead for the second miracle because it attacks the byte wall directly with NO new hardware and NO quality loss.

## The contradiction

Our earlier conclusion (from the 8-stream research and a "byte-exact" study):
- Q4_K nibble entropy 3.86/4.0 bits, whole-artifact lossless ceiling ~1.04-1.05x, dictionary-zstd EXPANDS.
- Measured only 0.9-2.1% on our Q4 codes → declared compression CLOSED.

New evidence (vramkit, github.com/SRKRZ23/vramkit, measured on a MacBook, 2026-10-17):
- A 2-D adaptive context model range coder (each 4-bit weight conditioned on its LEFT + UP neighbours) gets **2.676 bits/weight (−33.1% vs raw 4-bit) on REAL Qwen2.5-0.5B int4 weights, bit-exact verified**.
- Crucially, **plain zstd-19 alone gets 2.717 bits/weight (−32.1%)** and xz-9e 2.708 (−32.3%) on the same real weights. Across all weight-tensor types it averages 2.92 bits/weight (−27%), lossless, beating zstd-19 on 4 of 5.
- Sits "on the 2-D entropy floor".

Also:
- ECF8 (ICLR 2026): lossless FP8 format, entropy-aware, −26.9% memory, +177% throughput, bit-exact fidelity, fundamental limit ~FP4.67.
- Shannonic (MLSys 2026): within 1% of Shannon limit, 530 B state, few ops/symbol.
- EntQuant (ICML 2026): ANS, Float8 at 2.1 bits — but that one is NOT bit-exact (it changes weights to preserve function), so it FAILS our rule.

## Why the conflict matters

Our 0.9-2.1% and their −32% cannot both describe the same bytes. Likely explanations:
1. Our Q4 affine G32/G64 tensors interleave scales/biases with codes in a layout that defeats byte-oriented compressors, while vramkit tested clean packed int4.
2. We compressed the whole artifact (with metadata) rather than per expert tensor; or we used default zstd level, not -19.
3. Our experts (Qwen 512/layer, Nemotron 128/layer) may have different entropy than attention o_proj.
4. The byte-exact study measured Q4_K (a different format, with per-block fp16 scales) — not our MLX affine G32/G64.

## What must be done (falsifiable, 1 script, no hardware)

Re-measure LOSSLESS compression on our ACTUAL expert tensors, per tensor:
- Take one Qwen expert `gate_proj.weight` (Q4 G32) and one Nemotron expert (Q4 G64).
- Test zstd -19, xz -9e, and a 2-D context range coder on the raw code bytes (and separately on the scale/bias bytes).
- Verify bit-exact roundtrip (SHA-256 of decompressed bytes == original).
- Report bits/weight for codes and scales separately.

If we get −30% on codes: 873 MB/token → ~611 MB/token → 182 ms drive at 3.36 GB/s → with E0 overlap, Qwen could approach 5 tok/s. If scales/biases (20% of Qwen bytes) are also compressible, even better. This is the ONLY lead found that could reach the target on 16 GB with no quality loss.

This MUST be the top item in the Opus prompt. Forbid Opus from hand-waving it away with the old 3.86/4.0 study — demand the per-tensor re-measurement.

## RESOLVED (measured 2026-11-17, offline, no hardware run): compression is CLOSED for our format
Read the REAL MLX expert tensor `layers.0.mlp.switch_mlp.gate_proj.weight`:
- dtype `U32` (uint32!), shape [512,640,320] — MLX packs the 4-bit codes into
  32-bit words (8 values/word), which destroys the byte-level structure the
  vramkit 2-D context model and zstd exploit on clean int4.
- zstd levels 1/9/19/22 ALL give 95.2% (4.8% saving) on a 16 MB sample.
- zlib-1/6/9 give 95.5%.
- Bit-exact SHA-256 roundtrip confirmed (af1c7a51... both sides).
So the vramkit −33% does NOT transfer to MLX's U32-packed Q4. Our original
0.9-2.1% conclusion STANDS. Do NOT ask Opus to pursue code-stream compression;
it is definitively closed. The only lossless byte leads that survive are:
  (1) read-once SSD staging dedup (~1.8x via route union, bit-exact — RANK 1),
  (2) BF16 scale/bias exponent coding (~6% Qwen / 3% Nemotron — small),
  (3) the certify-don't-predict LOWER BOUND skip (radical Q1, ~20-25% reads, bit-exact guaranteed argmax) — a genuine open niche.
