# Phase 0 result — Nemotron squared-ReLU zero-column lever (2026-11-17)

Opus's second-miracle "new lever": skip reading W_down columns whose relu2
activation is exactly zero. Confirmed architecture (`mlp_hidden_act: relu2`,
`models/nemotron35/streamed_experts.py:862`), ran the guarded offline probe
`bench_relu2_sparsity.py` on the real full-size checkpoint, 96-token greedy
decode, 512-token prompt, hook on `nn.relu2`. Guard clean (floor 3.19 GiB,
swap flat, 0 swapouts). Evidence: `results/nemotron35/20261117-195345-relu2-sparsity/phase0.json`.

## Measured
| metric | value | Opus kill criterion |
|---|---|---|
| exact-zero fraction of relu2 activations | **0.9432** | kill if <0.60 → PASS |
| nonzero fraction | 0.0568 | — |
| 64-neuron groups with any nonzero | **0.8249** | kill if needed-64group >0.6 → **FAIL** |
| sample: 4,000 relu2 calls, 177.8 M elements | — | — |

## Verdict: DROP the byte-saving lever (but the zeros are real)

The activations ARE ~94% exactly zero — more than Opus's " >90% sparse" literature
claim. BUT the nonzeros are scattered: 82.5% of 64-neuron quantization groups
contain at least one nonzero. In the real W_down layout, a 64-column group shares
one scale/bias and its codes must be read if any column is live. So even with an
optimistic group-reorderable layout, ~82.5% of the down projection must still be
read — a total-expert saving of only ~9% (W_up + up-sb always read; down is 82.5%
read), nowhere near the 35-45% Opus projected from an assumed f≈0.3-0.35 clustered
nonzero pattern.

Note on Phase 0a's "needed_chunk_fraction ≈ 1.0": that was a BUG in my probe — it
chunked the 1856-element activation vector by `chunk_bytes//4` columns (4096 for
16 KB), so the whole vector was one chunk. Invalid; do not use. The valid number
is the 64-group figure 0.8249. A corrected layout-aware probe (`bench_relu2_chunks.py`)
is committed but not needed: 0.8249 already exceeds the 0.6 kill line.

## The compute side is separate
The 94% zeros DO mean ~94% of the down-projection GEMM work is on zero activations.
A bit-exact sparse down-kernel could cut that compute (~tens of ms), but compute is
not the bottleneck (E0 shows it overlaps with I/O; ~58 ms floor). At most a few
tok/s, and Nemotron's 15 target needs the I/O bytes cut, which this does not do.

## Conclusion
Opus's second-miracle lever fails on the actual model: the zeros are real but too
scattered to skip SSD byte ranges. Combined with Lead 1 and Lead 2 DROP (accepted),
Lead 3 small (~6%), and E1's +25% prediction ceiling, this confirms Opus's own
bottom line: **no no-quality-loss software method reaches Qwen 5 or Nemotron 15 on
this 16 GB M4.** The only remaining untested software A/B is pool-as-sole-cache +
F_NOCACHE (partial gain toward ~3.2 Qwen, not the target). Targets require more
memory / faster storage, which are excluded by the owner.
