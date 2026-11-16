# ANE research decision summary (2026-11-17, overnight session)

Owner rule in force: **16 GB M4 first; larger Macs only after success.**

## The one concrete candidate found: expert-wise K=2 read sharing

Three independent lines of research converged on ONE mechanism that is
bit-identical, untested by Veystrix, and addresses the real bottleneck:

- **Mechanism** (Specter §3.4.2, EStream §3.2, BigMoMo): instead of token-row-wise
  verification (each drafted row triggers its own SSD reads — why MTP lost at
  150–200 ms/row), group the K rows by expert, load each distinct expert ONCE,
  run a 2-row GEMM, accumulate in the baseline expert order. Routing and
  arithmetic are unchanged → token-identical by construction. This is *read
  sharing*, not the rejected *read overlap/prefetch*.
- **Decision statistic** (expert_streaming_research.md §7.2): consecutive-token
  expert-set union ratio. >0.92 = kill; 0.80–0.92 = implement; <0.80 =
  promising but check draft acceptance.

## Our offline measurement (no hardware run, `route_union_analysis.py`)

Using the 36,864 per-token expert sets already in the saved route trace:

| Statistic | Value | Threshold verdict |
|---|---:|---|
| K=2 weighted mean union ratio | **0.796** | just under "promising" (<0.80) |
| K=2 median | 0.794 | |
| K=2 p10 / p90 | 0.701 / 0.889 | |
| Implied read-wait saving | **~59 ms of 289 ms/token** | before draft cost |
| K=4 raw | 0.625 | |
| K=6 raw | 0.542 | |
| K=6 miss-adjusted | 0.621 | borderline vs Fable's 0.6 |

This is the **first measured evidence from Veystrix's own data that a concrete,
bit-identical speedup mechanism is viable on 16 GB.** Caveats: trace is from a
different checkpoint with 8 routed experts/token (current config is 10);
miss-adjustment is approximate; draft acceptance unknown; batch-2 reduction
order must be verified bit-identical (model on `dense_gemm._bit_equal`).

## FusionML — credibility downgrade (important correction to the subagent)

The subagent recommended FusionML (arXiv 2607.22785) as the top actionable item
("1.15–1.38× prefill, token-identical, measured on M4"). I verified the primary
source:
- Paper is real (arXiv:2607.22785, 1 author, Jul 2026, "community-replicated").
- GitHub repo `ommo007/FusionML` has **6 stars**, no benchmarks branch on main,
  no result artifacts, and one commit ("chore: ignore benchmarks folder").
- The row-split code exists in Swift (`WorkSplitter.swift`, `SmartScheduler.swift`)
  and is a generic CPU/GPU split; **`ANEEngine.swift` is a placeholder that
  throws `ANEError.modelNotAvailable`**.
- The Python `hybrid_backend.py` only picks NumPy for small ops and MLX for large
  ops — no row split, no eager boundary, no ANE.
- The paper's specific Qwen2.5-7B 1.18–1.25× TTFT numbers have **no reproduction
  artifacts in the repo**.

Verdict: treat FusionML's numbers as **unverified single-author claims**, not a
green light. Its *mechanism* (eager `mx.eval` boundary before a CPU+GPU row split
to avoid MLX's lazy scheduler serializing cross-stream work) is plausible and
cheap to test independently on Veystrix's own matmuls — but do not adopt it as a
default path on the strength of the paper alone.

## ANEForge tie-margin fallback (second candidate)

ANEForge (arXiv 2606.17090, 1 author) reports an fp16 head with a tie-margin
fallback: if top-1/top-2 gap < 3e-3 × max logit, recompute logits on the fp32
head. It restored exact greedy equality with host fp32 in their test. This is a
principled way to use fast low-precision kernels under a bit-identical contract
and could later guard any ANE output. Unreplicated single-author result; needs
a Veystrix-specific trigger-rate and exactness measurement.

## Source-level corrections from maderix CODE deep-dive (same day)

I read the actual maderix/ANE source (not just its README). Corrections to our
earlier notes:
- **`gpu_ane_share.m` / `gpu_prefill_ane_decode.m` do not exist** in any commit;
  only the README mentions them. The README's GPU-prefill→ANE-decode direction is
  inverted vs every working implementation (ANE prefill → GPU/CPU decode). So the
  "working GPU↔ANE zero-copy demo" I reported earlier is **README-only, unverified**.
- **INT8 weights DO compile** via `constexpr_affine_dequantize` (`ane_int8_bench.m`);
  the ~1.95× needs int8 *activation edges* in a deep chain, not INT8 weights alone.
  INT4 has no path.
- **S<16/32 is a hard `0x1d` eval error, not silent all-zeros** (our earlier note
  was wrong). Min IOSurface ~49 KB, W≥32, pad spatial dim.
- **ANE attention is 560–610× slower than CPU GQA** (M3 Max field guide, S), and
  ANE RMSNorm has an ~0.01 error floor — both are direct bit-identical threats.
- **W-lane batching of ≤4 tokens** is the only demonstrated multi-token-per-dispatch
  ANE path (5.0× matmul / 2.93× oproj_add, max_err ~0.002–0.004) — but norms,
  attention and argmax must stay off ANE.
- **Delta weight reload is contested**: maderix M5 says file-swap reload fails;
  Orion M4 says 494 ms/60 kernels works. Neither replicated. Test locally.

Net effect: ANE remains unsuitable as the main executor; the K=2 *software*
read-sharing candidate is unaffected by these corrections (it runs on the
CPU/Metal path, not ANE).

## Bottom line for the owner

- There IS now one measured, bit-identical, 16 GB-relevant candidate: expert-wise
  K=2 read sharing, with a 0.796 union ratio and ~59 ms/token potential save.
- No published ANE technique beats MLX GPU for prefill on the base M4; ANE's
  strongest credible role remains an opt-in island or a drafter whose inexact
  output never reaches logits.
- FusionML's headline result is unverified; its *idea* (eager boundary) is worth
  a cheap independent probe, not adoption.
- Next actions are offline/code design only until the owner authorizes a guarded
  hardware run.
