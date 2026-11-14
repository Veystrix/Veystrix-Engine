# fable 5.2

You are working on Veystrix, an Apple Silicon inference engine for SSD-streamed sparse MoE checkpoints. Start with [Stage 10](stage-10.md), [Stage 11](stage-11.md), and the [14 November experiments](real-runs-2026-11-14.md). Inspect the code and raw measurements before choosing an approach. Pick the experiment with the strongest measured chance of lowering single-request latency. Produce a patch and a falsifiable measurement, or an artifact that rules out the idea.

## 1. Make Qwen prefill fit with stable allocation

The guarded 512-input run now fits with 128-token chunks, a 256-expert scratch ring, and uncached prefill reads. A colder startup still fell to 2.68 GiB and the 2K probe stopped on heat despite a 4.67 GiB memory floor. Trace transient allocations through `loader.py`, `prefill_ring.py`, `expert_cache.py` and `prefill.py`. Measure how often the ring falls back or rereads the same expert across chunks. Compare a reuse plan at identical chunk edges. Require identical continuation state, at least 3 GiB available throughout, flat swap and fewer thermal-pressure samples.

## 2. Get below 50 ms per accepted output token

The sustained Qwen baseline used about 742 MB/token, 2.79 GB/s effective SSD bandwidth and 110 ms/token outside measured I/O. Improving only one term cannot reach the 20 tok/s gate. Qwen depth-1 verification costs 660 ms versus 381 ms plain and accepts 0.78 drafts/round. Extra rows currently buy little because they request mostly different experts. Build a paired experiment that reduces both blocking I/O and verifier cost. Compare lossless expert packing and route-union scheduling to the strongest same-budget cache policy. Test fixed workspaces and kernel launch reduction using the measured shape corpus. Publish physical MB/accepted token, end-to-end verify/repair time, accepted tokens/round, GPU activity where observable, and the resolution band. Falsify the design if those changes leave the measured critical path above 50 ms/token.

## 3. Localize Nemotron numerical drift

The real streamed-vs-stock BF16 check agreed on 17 forced greedy decisions but reached 1.129 maximum logit error. Reproducing the stock expert kernel did not resolve the discrepancy. Instrument both paths layer by layer for the same checkpoint, IDs, positions and cache state. Compare Mamba/conv recurrence, RMSNorm, attention reductions and token packing in that order only where the first-divergence trace warrants it. Capture the smallest reproducer and the first operation whose output diverges. Re-run the stock reference and a real 512-token continuation. Evaluate INT8 KV separately with a declared error budget.

Use the 512-token Stage 11 prompt as a second numerical probe. Merely removing the prefix store changed the system-anchor split and the 128-token continuation digest. Hold every chunk edge fixed while isolating cache reuse, then vary edges explicitly to find the first divergence.

## 4. Make scheduling genuinely decode-first

The opt-in cooperative HTTP path already admits separate request states and favors ready decode work. Nemotron shares experts across compatible ready rows. Flash-Next interleaves singleton forwards. Extend Flash-Next with layer-local expert reuse while preserving per-request QSA positions and GDN state. Add resumable speculative rounds with memory reserved for verifier and rollback workspace. Inject disconnects inside verification and at every output boundary, then compare subsequent logits, prefix restore and the next request. Check concurrency 1 overhead against the 3% budget and measure aggregate throughput and per-stream intertoken tails at concurrency 2 and 4 under the same 16 GiB gate.

## Return

Return the code change, benchmark command, result-directory links, exact model and executable identities, and enough raw evidence to reproduce the result. Keep exact greedy parity, sampled-distribution parity and approximate quantization results identifiable by configuration. Freeze flags only after the release gate passes on both checkpoints.
