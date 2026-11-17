# Opus response — second miracle (pasted by owner, 2026-11-17)

Full text of Opus's reply to `OPUS_SECOND_MIRACLE_PROMPT.md`.

---

I'll check the cited papers and the Nemotron expert architecture first, then work through the byte arithmetic for each lead.

Nemotron 3 Nano's experts are confirmed as squared-ReLU. That changes the analysis, so I'll check the intra-expert sparsity data and look once more for the 2610.34440 paper.

I couldn't find a software method that gets Qwen to 5 tok/s on this machine, and I can show why. For Nemotron I found one new lossless lever, and it's specific to Nemotron's activation function. Two of your three leads fail on first principles.

## Summary
- Lead 1 (read-once dedup): saves zero SSD bytes once any cache exists. Proof below.
- Lead 2 (certificates): can't skip expert reads, because expert outputs become the state that every later token reads. It still works on the output head (LM head), which is a small gain.
- Lead 3 (scale/bias coding): real, about 6% for Qwen. Keep it.
- New lever: Nemotron's experts use squared ReLU, so most of their internal activations are exactly zero. The matching columns of the second expert matrix (the down projection) never need to be read. The kernel and arithmetic stay unchanged, so it's bit-exact.
- ANE: cannot help.

## What I could verify
- MonoMoE is real. It's a weight-major persistent megakernel for quantized MoE decode, built into vLLM. MonoMoE is "a weight-major persistent megakernel for block-wise quantized MoE decode". It targets GPU memory bandwidth, not SSD. Its starting observation is that the fused MoE matrix multiplies sustain only ≤22% of peak DRAM bandwidth at batch size 1. It's no precedent for SSD dedup.
- arXiv:2610.34440: I couldn't locate it. A direct search returned nothing, so the r(n)/u(n) figures are unverified. My conclusion below doesn't depend on them.
- Nemotron's activation is confirmed. The experts per MoE layer are 128 routed + 1 shared, the hidden dimension is 2,688, and the activation function is Squared ReLU (ReLU²). This matches your expert size exactly: 2 matrices of 2688×1856 at 4.5 bits per weight is 5,612,544 bytes. Qwen's tensor [512, 640, 320] means hidden size 2560, intermediate size 640, and three matrices per expert (SwiGLU style). That also matches 3,072,000 bytes, with scale/bias making up 20%.

## Lead 1: zero bytes saved (proof)
Let D_t be the set of experts token t routes to. In ordinary sequential decode with any cache that holds one token's routed set (1.47 GB for Qwen, which both the page cache and your pool exceed), an expert used by both t and t+1 is read once: it misses at t and hits at t+1. A read-once batch over {t, t+1} also reads D_t ∪ D_{t+1} minus what's already resident. That's the same set, so physical bytes are identical.
The "~1.8x route-union" figure counts logical routed bytes, but the cache already removes that sharing. Speculation only adds bytes, from the experts of rejected drafts. Your E1 result confirms this: knowing the next token's routes perfectly changed physical bytes by only −1.8%. Comparing r_ssd ≈ 1 against u(2) mixes physical cost with logical sharing; measured physically, u(2) ≈ 1.
The dedup only appears to win in a harness with no cache at all. Your Nemotron number points that way: 857 MB/token is more than the logical routed bytes (23 layers × 6 experts × 5.61 MB = 774.5 MB). At 3.36 GB/s, 857 MB would also cap production at 3.9 tok/s, yet you see about 7. So the "physical" byte counter is recording something other than production SSD reads. Fix it before trusting any byte projection, this one included.
A published negative result reports the same effect: on Qwen3-235B, measured decode matched a bytes-per-token / bandwidth model, while a batching scheme meant to amortize one disk sweep collapsed at batch 32 from paging. DROP.

## Lead 2: works for the output head, not for experts
The certificate guarantees token t's argmax. But each expert's output goes into the residual stream, which becomes token t's KV cache and recurrent (GDN/Mamba) state. Token t+1's layer-1 attention reads that state. Skip any expert read at t and the state is no longer bit-exact for every later token. Recomputing it exactly means reading the same bytes before t+1 can start, so nothing is saved.
Composed error bounds across 48 layers would be loose anyway. Even within a single attention layer, a certified method found the mass-derived output bound exceeds the measured local output error by 11-13x.
The output head is different because logits feed nothing but the argmax. Prior work holds up: CSV-Decode gives exact top-k for top-k tasks and ε-bounded softmax using Cauchy-Schwarz bounds on logits for whole clusters. The certhead implementation returns the identical token to argmax(Wh), always; this is structural, because of the dense fallback.
Two caveats on the head certificate:
- Your logit SHA-256 gate can't run on it. Unopened logits are never computed. Gate it on token IDs plus a final-hidden-state digest, and turn it off whenever log-probabilities are requested.
- It saves memory bandwidth and compute, not SSD bytes, if the head is resident. That's a few ms.
KEEP the head certificate (small). DROP the per-layer certificate.

## The new lever: skip zero columns of Nemotron's down projection
Mechanism. A Nemotron expert computes y = W_down · relu(W_up·x)². Wherever (W_up·x)_j ≤ 0, the activation a_j is exactly +0 in BF16, and column j of W_down contributes nothing. Sparsity for trained ReLU² models is very high: a 2026 paper states that activation matrices are naturally more than 90% sparse with the squared ReLU activation function. That figure comes from dense models; MoE experts may be less sparse, so measure it first.
The closest prior work, intra-expert sparsity on SwiGLU models, is lossy. It retains about 95% of baseline benchmark accuracy, not 100% and contains no Nemotron or squared-ReLU results. Apple's LLM-in-a-Flash work used neuron predictors, which are also approximate. Here nothing is predicted: the zero set comes from the exact W_up output.
Bit-exactness proof (no kernel change):
1. Read all of W_up and compute up-projection plus ReLU² with the production kernels. These values are exact.
2. Read back the nonzero mask (1,856 bits per expert).
3. Read only the down-projection chunks containing a nonzero neuron, plus the scale/bias of each quantization group (64 neurons) that has any nonzero neuron.
4. Fill every unread byte of the expert's buffer with zeros, or any finite value.
5. Run the unchanged down-projection kernel on the full buffer.
Every unread weight is multiplied by exactly ±0, including the group scale and bias terms. Accumulators start at +0, and under round-to-nearest +0 + (−0) = +0, so an accumulator never becomes −0. Adding ±0 therefore changes nothing, bit for bit. The reduction order is the kernel's own and doesn't move. You can lay the 64 columns of each quantization group out on disk in any order, because the code places each column back at its original position in memory. Quantization groups and arithmetic never change.
One thing to check: if production fuses up, activation and down into one kernel, the split version must reproduce the intermediate's exact dtype. The digest gate will catch any mismatch.

Expected bytes (estimates; Phase 0 replaces them):
- Expert split: W_up codes 2.49 MB + scale/bias 0.31 MB; W_down the same.
- Reads per cache miss: 2.81 MB (W_up) + active scales + f × 2.49 MB, where f is the fraction of down chunks that are needed. At 10% nonzero activations and f ≈ 0.3–0.35 (16 KB chunks, clustered within groups), that's about 4.0 MB instead of 5.61 (−29%).
- Cache footprint: caching partial down projections raises effective capacity by an estimated 1.25–1.4×.
- Combined: −35 to −45% of SSD bytes.
What that means for Nemotron:
- At your serial compute floor of ~58 ms (my earlier figure: your old 181 ms profile minus 123 ms of I/O), today's ~123 ms of I/O falls to roughly 70–80 ms, which gives about 7 → 10–11.5 tok/s.
- With a 7–7.5 GB sole-cache pool, scaling your S3-FIFO table gives about 10 GB of effective capacity, roughly 16 ms of I/O per token, and ~13.5 tok/s.
- Reaching 15 also needs the non-I/O floor cut by about 8 ms. Squared-ReLU sparsity also removes most of the down-projection compute, which helps.

The experiment:
Phase 0 (½ day, offline): Hook the 3 × 1,200-token traces and record exact-zero masks per (layer, expert, token). Report: fraction of nonzero activations; fully-zero 64-neuron groups; needed-chunk fraction at 4/16/64 KB, both in original order and after offline co-activation reordering within each group (fit on separate text, not the eval traces); neuron hotness. Also benchmark random-read throughput against read size (4–256 KB, queue depth 8–64, F_NOCACHE). Count Qwen's exact zeros too (expect ≈0). Kill criterion: Fewer than 60% zeros, or needed 16 KB chunks above 0.6 → DROP
Phase 1 (A/B): Two-phase fetch in the C++ pager, with a per-chunk valid bitmap in each slot. Invariant: for every nonzero activation, its column and its group's scale/bias are valid before the down kernel runs. Kill: Below 9 tok/s → DROP
Gate: Token IDs plus per-step logit SHA-256 identical to the reference (logits are fully computed, so this gate works here). Chaos mode adds random I/O delays, random evictions, and random finite garbage in unread regions. Bit-identical output under garbage fill proves skipped bytes never matter. Don't use NaN poisoning: NaN × 0 = NaN, so even a correct implementation would fail it. | Any mismatch → bug

Predicted Phase 1: I/O bytes −35 to −45%; io_await from ~123 to ~70 ms on today's pool, or ~16–25 ms with a 7.5 GB sole-cache pool; 10–14 tok/s, with 15 as a stretch.

## Lead 3
Scale/bias is 20% of Qwen's bytes (G32) and 11% of Nemotron's (G64). Exponent and sign coding saves roughly a third of that, so about 6% and 3%, matching your figures. Decoding about 175 MB/token for Qwen at 3–4 tok/s needs about 0.6 GB/s on the CPU, which fits on 2 P-cores. KEEP, low priority. I didn't verify 2606.15789.

## ANE: cannot help
- It can't reduce bytes. It reads from the same memory and the same SSD.
- It can't take over anything that feeds model state. It would have to match the GPU's fp32 accumulation bit for bit, which a different unit can't do.
- The certificate-tolerant work is too small. A certificate can absorb ANE's fp16 error if the bound includes it, but the only such work is the head shadow, worth about 1 ms.
Explicitly: no bit-exact ANE role improves tok/s here.

## Qwen 5 tok/s: no software solution exists (proof)
With routes unpredictable (E3 recall 0.36), layer L's reads can't start before layer L's router. Only that layer's expert compute can overlap with them.
1. Lower bound on time per token: t ≥ (130 ms cached step − expert compute) + B/3.36 GB/s. At 5 tok/s (200 ms), even hiding 60 ms of expert compute, B ≤ ~435 MB/token.
2. Qwen has no exact zeros, since SwiGLU doesn't produce them, so the new lever doesn't apply.
3. The cache can't get B that low. B is at least the misses of Belady's optimal policy (OPT) at a realizable pool. On 16 GB with Qwen's resident weights and a 14–31K-token KV cache, that's about 8 GB or less. Your table puts OPT at 8 GB at 4.43 tok/s. Even OPT, which can't run online without future knowledge, plus the 6% from Lead 3, gives about 4.7 < 5.
4. Best realistic: an S3-FIFO sole cache of 8 GB (3.22 tok/s in your simulator), plus scale/bias coding and streaming each expert's compute as its bytes land: about 3.5–3.9 tok/s.
The one combination never tested is pool as sole cache + F_NOCACHE. Your 2→4 GB regression and the 8% F_NOCACHE loss each tested half of it, and double caching explains both. It's the largest untested bit-exact Qwen gain (2.3 → ~3.2) and needs no new code.

## Nemotron 15 tok/s: not ruled out on 16 GB
At the current ~58 ms floor, 15 tok/s needs ≤29 MB of SSD reads per token. Without the squared-ReLU lever, that means OPT with 10 GB, which isn't realizable. With it, about 7.5 GB of sole-cache pool under S3-FIFO comes within roughly 10% of the target. Phase 0 settles it in half a day.

## KEEP / DROP
| Lead | Verdict | Expected gain |
| Squared-ReLU zero-column skip (Nemotron) | KEEP, rank 1. Run Phase 0 now | 7 → 10–14 tok/s |
| Pool as sole cache + F_NOCACHE | KEEP (untested combination) | Qwen → ~3.2, Nemotron enabler |
| Lead 3 scale/bias coding | KEEP, low priority | ~6% Qwen / 3% Nemotron |
| Lead 2 on the LM head only | KEEP, small. Gate on token IDs plus hidden-state digest | a few ms |
| Lead 1 read-once dedup | DROP (proof above) | 0 |
| Lead 2 per-layer/per-expert certificate | DROP: expert outputs are state | 0 |
| ANE | DROP: no bit-exact role | 0 (energy only) |
| Qwen 5 tok/s | No software solution under these constraints; best ≈ 3.5–3.9 | — |

If you'd like, I can write the Phase 0 trace-hook and chunk-analysis script to hand to Codex.
