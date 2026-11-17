# Opus response — miracle hunt (pasted by owner, 2026-11-17)

Full text of Opus's reply to `docs/performance/OPUS_MIRACLE_PROMPT.md`.
Opus states it worked only from our document and did NOT verify the arXiv papers;
its quoted paper numbers are our notes' figures, not independent findings.

---

I couldn't find a new compression or routing trick that hits the targets. What I found is a mistake in the model behind the "impossible on 16 GB" ceilings. Your own numbers suggest Qwen 5+ is reachable on this machine without any cache gains, and Nemotron 15 is borderline. I worked only from your document. I haven't checked the arXiv papers, so where I use their numbers they're your notes' figures, not mine.

## 1. Section 2 assumes I/O and compute run back to back

Every ceiling in Section 2 uses **t = compute + bytes/BW**. Your OPT tables only fit that formula, with a fixed floor of about 58 ms for Nemotron (for example, 13.82 tok/s at 8 GB gives 72 ms, which is 58 ms plus about 43 MB/token). In a properly pipelined decode the time per token is **max(GPU, bytes/BW) plus whatever stall can't be hidden**.

Taking your profile numbers at face value:

| | GPU | I/O at 3.0 GB/s | Serial (today) | Overlapped ceiling |
|---|---|---|---|---|
| Qwen, 408 MB/token as-is | ~81 ms | 136 ms | 300 ms | **~136–150 ms → 6.7–7.3 tok/s** |
| Nemotron, 436 MB as-is | ~20 ms* | 145 ms | 181 ms | ~145 ms → 6.9 tok/s |
| Nemotron, 6 GB cache, S3-FIFO (~177 MB) | ~20 ms* | 59 ms | — | **~59–66 ms → 15–17 tok/s** |

\*Nemotron's GPU share isn't broken out. The 50 ms of host time has to come off the critical path (see section 2).

The physics floor on 16 GB is far higher than the targets. With realistic caches and perfect eviction (OPT), Qwen at 8 GB needs about 248 MB/token, roughly 83 ms of I/O, so the floor is about 12 tok/s. Nemotron at 6 GB needs about 74 MB/token, so its floor is above 30 tok/s. **The ~3 GB/s wall stops you only because I/O and compute currently run one after the other.**

The key anomaly in your own data is "perfect predictor + overlap tied." With an oracle predictor and per-layer I/O (~3.3 ms) about equal to per-layer compute (~3 ms), real overlap should have cut io_await roughly in half. A tie means the I/O and compute never actually ran at the same time. Early submit (−8.2%) and early shared expert (−10%) fit the same story: reordering work on a path that is still serial only adds overhead. Likely causes:
- a host sync and router readback on every layer;
- MLX evaluating each layer to completion before the next starts;
- Python GIL contention between the I/O, decode and HTTP threads;
- memcpy from the page cache counted inside io_await.

**Two numbers to reconcile first:**
- Qwen's 300 ms/token profile implies 3.3 tok/s, but you report 2.3. That's about 135 ms/token unaccounted for, and it matches "packing +9% in-process, no server gain." The server path itself may be costing more than any lead on your list.
- Nemotron's 181 ms profile implies 5.5 tok/s, but you report ~7.
- The hard rule says "top-9/240" while Model B is described as 128 experts, top-6. Which is it?

## 2. Mechanism: a pipeline where the GPU waits on events, with routing left untouched

1. **Build the whole token's graph once (in C++ or a custom MLX primitive), not per layer from Python.** Each MoE layer gets an `ExpertFetch(topk)` primitive on the CPU stream. It writes into slots of one persistent pool array and outputs slot indices. The GPU's quantized gather reads `pool[slot]`, and MLX's cross-stream events make the GPU wait only on that layer's fetch. Check that MLX 0.32.2 can have the GPU wait on CPU work without a host sync. This is the useful part of lead F; your host time is 18 ms, so F's 340 ms saving doesn't apply here.
2. **A predictor supplies prefetch hints only. They are never committed as routing.** That's the fix for Edge0 (lead A). The real router still picks the experts. A wrong hint costs bandwidth, and a missed expert becomes a demand read that jumps the queue. Routing is never changed, so it's bit-exact by construction. Start with no training: apply layer L+1's (and L+2's) router weights to layer L's post-attention residual. For layer 0 of the next token, a lookup table from token id to experts.
3. **The pool is the only cache: F_NOCACHE direct reads into page-aligned slots, sized as large as memory allows.** Two of your results point the same way. Moving the pool from 2 to 4 GB regressed, and F_NOCACHE was 8% slower with pool = 0. Both look like the pool and the page cache holding the same bytes twice, plus memory pressure. Your hit rates support this: Qwen is already at 72% (408 of 1,475 MB logical) through the page cache, while Nemotron is at only 44% (436 of 775 MB). Nemotron is getting about 2 GB of effective cache when 5–6 GB looks available. Lead E's slot cache is this mechanism. Prefetches land in a small staging area and are promoted only when used, so wrong hints don't evict useful experts.
4. **The I/O engine is a C++ thread pool with 8–16 requests in flight, or F_RDADVISE (lead D).** Your probe got 3.6–3.8 GB/s against 2.6 effective today. F_RDADVISE fills the page cache, which brings back the double-caching problem, so it has to win on bandwidth to justify that.
5. **Eviction is S3-FIFO, but experts the predictor expects soon are protected (lead J, partial).** With about one token of lookahead instead of J's 16, expect a 5–15% byte cut, not 2×. Quantify it in your simulator.

**Realistic numbers with a predictor that catches 80% of misses and over-fetches 30%:**
- **Qwen:** ~530 MB of I/O per token → 177 ms at 3.0 GB/s, 143 ms at 3.7 → **5.2–6.5 tok/s, so the 5 tok/s target looks reachable.**
- **Nemotron, 6 GB pool:** ~230 MB → 77 ms at 3.0 GB/s, 62 ms at 3.7 → **12–16 tok/s.** Reaching 15 needs all three: about 3.6 GB/s or better, a 6 GB pool with zero swap, and the 50 ms of host work moved off the critical path.

**Bit-exactness verdict: bit-exact**, as long as three conditions hold:
- **Slots are pinned** with a reference count until the command buffers using them complete. I suspect your earlier "speculative reads caused divergence" was a slot being reused or evicted while a kernel was still reading it.
- **Expert outputs are combined in router-rank order** no matter which slot they sit in.
- **There are no "summed partials" or dummy-slot tricks** unless they keep the exact summation order. This is a real risk in lead E's design.

## 3. Experiments on the 16 GB M4

**Bit-exact gate, applied to every experiment:**
- Reference: the current production sync path.
- Runs: greedy decoding, 3 prompts (chat, code, agentic), 1,200 tokens each, scored after token 100 to get steady state.
- Pass condition: 100% identical token IDs and an identical SHA-256 of the raw logits at every step.
- Per-layer residual checksums for the first 64 steps, so a failure can be localized.
- **Chaos mode:** inject random 0–5 ms delays into the I/O threads. The output must stay bit-identical. This is what catches the race class.

**Report for each run:** physical MB/token split into demand, prefetch-used and prefetch-wasted; io_await on the critical path (time the GPU spends blocked on the fetch event); sustained tok/s after 5 minutes (the Air is fanless); vm_stat swapouts, which must be 0.

| # | Experiment | Prediction | Kill criterion |
|---|---|---|---|
| E0 (½ day) | **Concurrency proof.** Run a fully cached decode step (re-run an identical step) in a loop while a C++ thread reads 408 MB/token of cold expert ranges at full speed. | GPU step inflates <10%; read rate stays 3.0–3.7 GB/s | Inflation >25% or bandwidth drop >20% means the overlap ceiling is lower; recompute the tables |
| E1 | **Oracle pipeline.** Record exact routes in pass 1. In pass 2, use them as hints issued 1–3 layers ahead while the real router still decides. | Qwen ≥5.5 tok/s (profile basis), critical-path io_await ≤30 ms | Oracle below 4.5 tok/s means no predictor can save this path; go to hardware (section 5) |
| E2 | **Nemotron pool sweep**, sole cache with F_NOCACHE: 0/2/4/6/8 GB | Bytes/token falls with pool size; no regression without double caching | Swap appears at 6 GB |
| E3 | **Predictor recall, offline from traces.** Recall@10 for prefetching the predicted top 10, 14 and 20 experts | Zero-training pre-gate recall@14 ≈ 0.7–0.9 (unmeasured estimate) | Below 0.6 means train the head (section 4) |
| E4 | E1 with the real predictor | Qwen 5.2–6.5, Nemotron 12–16 (6 GB pool) | — |

## 4. Training, only if E3 recall is too low

The trained head is a **hint, never routing**, so quality is untouched.
- **Architecture:** one linear head per layer from the hidden size to 512 experts, warm-started from layer L+1's router weights. Optionally a second head for L+2. About 50M parameters (~100 MB in BF16).
- **Data:** route logs from batched prefill over 0.5–1M tokens of representative text. Prefill is far faster than 2.3 tok/s decode, and tiny batch-vs-single numeric differences don't matter for a hint.
- **Training:** BCE loss on the routed set, in MLX, about an hour on the M4.
- **Metric:** recall at the chosen prefetch width. Edge0's recovery LoRA isn't needed.

## 5. If E1 fails: the hardware fallback

- **A second SSD, which looks like the cheapest fix.** Stripe or replicate experts across the internal SSD and a USB4/Thunderbolt NVMe enclosure, with the pager splitting each token's reads between them. Expected aggregate is about 5.5–6.5 GB/s. Nemotron's 436 MB at 6 GB/s is about 73 ms per token, near 14 tok/s with today's cache and comfortably 15+ with a 6 GB pool. Qwen would become compute-bound at about 7+ tok/s. It's bit-exact trivially.
- **A second Mac.** The M4 Air has Thunderbolt 4, not 5, so the Thunderbolt 5 RDMA path in your notes doesn't apply. Expert parallelism over a Thunderbolt bridge still works: only about 5 KB of hidden state moves per layer. Two 16 GB Airs could hold almost all of Nemotron's 16.5 GB of experts, which makes it compute-bound.
- **More memory.** A 24 GB Mac gives a 12 GB pool; even your serial S3-FIFO model then reaches 14.4 tok/s, and overlap takes it past 20. At 32 GB, Nemotron is fully resident. Qwen needs 24–32 GB, where a 20 GB pool reaches 5.2 tok/s even in the serial model.

## 6. Compression reconciliation (leads C and K)

Your 3.86/4.0-bit result applies to the 4-bit codes, and it was measured on llama.cpp's **Q4_K format, not MLX's affine G32/G64**. In MLX's format the BF16 scale and bias are 20% of Qwen's bytes (5 bits per weight) and 11% of Nemotron's. The Nemotron expert size matches exactly: 2 matrices of 2688×1856 at 4.5 bits per weight is 5,612,544 bytes.

ZipMoE-style exponent coding only applies to that scale/bias stream. Expect it to save about a third of that stream (my estimate, to be measured). Upper bound: **about 10% total for Qwen, about 7% for Nemotron.** Measure it per stream in an hour. Lead K (base expert plus residuals): the difference of two near-uniform 4-bit codes has more entropy than either, so expect more than 4 bits and a DROP. Run a one-script check on the most similar expert pair per layer.

## KEEP / DROP

- **KEEP, phase 1:** the event-gated C++ pipeline (F generalized), the oracle test (E1), the pool as sole cache with F_NOCACHE (E), the C++ thread-pool or F_RDADVISE I/O engine (D), chaos-mode gating, and reconciling the server-vs-profile gap.
- **KEEP, phase 2:** A as a hint-only predictor, and J-style protection of predicted experts.
- **KEEP, hardware fallback:** a second SSD, a second Mac over Thunderbolt 4, or 24 GB of memory.
- **DEFER:** B (SpecFit), G (MTP), H (DraftExpert). While I/O-bound they raise the unique expert bytes per step, which your M4 sharing analysis already predicts loses. Revisit once decode is compute-bound.
- **DROP:**
  - A as committed routing: it breaks the hard rule.
  - I: the savings only exist when batching several tokens.
  - K: expected above 4 bits.
  - M's PagedMoE kernel: MLX already works directly on the Q4 codes.
  - C as a headline win: it's capped at about 10%. Keep it only as a low-priority measurement.

If you'd like, I can turn this into a spec doc for Codex, with the E0/E1 harness and the bit-exact gate written out.

---

## Owner-side validation (2026-11-17) — checked against the actual repo data

Verdict: Opus found a REAL flaw in how the ceilings are framed, but overstated the
consequence. Its E0/E1 experiments are worth running. Details:

**1. CORRECT that the ceilings use a serial model — but the "12/30 tok/s floor"
numbers drop the non-I/O term.** `STEP_B_LONG_TRACE_FINAL.md` uses
`t = non-I/O + bytes/BW` (Qwen 143.24 ms + bytes/2.603 GB/ms; Nemotron 58.3 ms +
bytes/3.541 GB/ms). With perfect overlap the floor is `max(non-I/O, bytes/BW)`, so
Opus is right the formula matters. But its "Qwen floor ~12 tok/s / Nemotron >30"
quietly sets non-I/O to zero. The real max-overlap floors are **~7 tok/s Qwen**
(max(143, 83) ms) and **~17 tok/s Nemotron at 6 GB** (max(58, 25) ms), and less if
the 50 ms of host route sync stays on the critical path. Still above the targets in
principle — far less dramatic than claimed, and Nemotron 15 stays tight.

**2. PARTIALLY CORRECT that our overlap result is inconclusive — this is the
important open question.** `bench_read_ceiling.py:42-46` does exactly what Opus asks:
it issues the *exact* next-layer cold rows one layer early into the page cache
(same bytes, one layer earlier) — the upper bound of an exact predictor. Result was
tied (+0.1/−2.9/−0.7%). Two readings: (a) byte-bound — total bytes unchanged, so
total time unchanged; (b) Opus's — no real overlap was achieved. Evidence that (b)
may be true: 1bit.gg's oracle on this SAME model got 10.2→11.5 tok/s vs 7.3→8.8
demand-only *from the same bytes* by stressing queue depth (16 threads × 256 KiB),
while our own worker sweep (6–32) was flat. The two engines reached different SSD
operating points, so our negative result may be implementation-specific. This is
genuinely unresolved — E0/E1 are the right way to settle it and should be run
before any C++ port or hardware purchase.

**3. CORRECT and fixed: the top-9/240 inconsistency.** It is a stale line in our
briefs referring to the OTHER Nemotron (Bolt 74B-A9B, 240 experts, top-9), not
Model B (Lightning 30B-A3B, 128 experts, top-6). Corrected in OPUS_MIRACLE_PROMPT.md.

**4. The profile-vs-reported gaps are workload differences, not hidden bugs.**
The 300.1 ms Qwen profile is the short-prompt harness (3.33 tok/s); sustained
512-token server runs are 2.65–2.69 tok/s (`stage-11.md:24`) and production
long-context 5.2–6.2. Nemotron's real-checkpoint 512-token decode is 125–142 ms
(7–8 tok/s, `handoff-2026-11-15.md:10`); the 181 ms profile is a different config.
Opus was right to flag them, but they are not 135 ms of missing performance.

**5. Strongest new cross-check from our own research:** 1bit.gg's Flash-Next oracle
(10.2–11.5 vs 7.3–8.8 tok/s from identical bytes) is independent support for Opus's
"latency, not bandwidth" thesis — conditional on our engine having a low-queue-depth
problem. Our flat 6–32 worker sweep is the counter-evidence. E0 resolves it.

**Recommendation:** run E0 (concurrency proof) and E1 (oracle pipeline) first —
cheap, decisive, bit-exact-gated. Defer B/G/H (SpecFit/MTP/DraftExpert) while
byte-bound, as Opus says. Its second-SSD fallback has independent support in our
notes (~18 GB/s combined, decode 2.4–2.9→5.8–6.0). Any hardware run needs fresh
owner permission and the guard.
