# ANE Research Index — consolidated for the build decision

Compiled overnight 2026-11-17 for the owner. Owner rule in force: **16 GB M4
first; larger Macs only after success.** No code changes in this research phase.

## The one-line answer

There is **no proven ANE-over-MLX speedup on the base M4 for this 4-bit MoE
workload**, and ANE cannot execute the MoE experts. But the research produced
**one concrete, bit-identical, 16 GB-relevant candidate** (expert-wise K=2 read
sharing) and a clear inventory of what is and is not viable.

## FINAL VERDICT (all research complete, 2026-11-17)

**No bit-identical ANE path improves Veystrix prefill and decode on 16 GB.**
All eight research threads are complete (see `experiments/ane/FINAL_ANSWER.md`).
The one bit-identical candidate that survives — expert-wise K=2 read sharing —
runs on CPU/Metal, not ANE. ANE is conclusively excluded as the speed mechanism
by its baked weights, fp16-only numerics, ~90 µs dispatch floor, and the
SSD-bound nature of the workload. Owner decision pending on whether to relax the
bit-identical rule (which still yields no meaningful ANE speedup) or build the
non-ANE K=2 candidate.

## Start here (in order)

0. **`experiments/ane/FINAL_ANSWER.md`** — the definitive answer to the objective.
1. **`experiments/ane/DECISION_SUMMARY.md`** — the synthesis: the K=2 candidate,
   the FusionML credibility downgrade, the maderix source corrections, and the
   bottom line. Read this first.
2. **`experiments/ane/ROUTE_UNION_FINDING.md`** — the offline measurement that
   produced the candidate: K=2 union ratio 0.796 → ~59 ms/token potential save.
3. **`experiments/ane/REPORT_NEMOTRON.md`** — our own failed end-to-end ANE tests
   (slower cold, memory-guard failure warm) and why.

## The candidate (what to build after the owner authorizes a run)

**Expert-wise K=2 verification with shared reads** — bit-identical by
construction:
- Group two consecutive tokens' rows by expert, load each distinct expert once
  (union, not sum), run a 2-row GEMM, accumulate in baseline expert order.
- Our measured union ratio (0.796 mean, p10 0.701) is below the literature's
  "promising" threshold (0.80) and implies ~59 ms of the 289 ms read wait saved
  before draft cost.
- Next: design the verifier on paper, model the draft acceptance cost, then ask
  for ONE guarded 16 GB run. Do not implement on the strength of offline data alone.

## ANE: viable vs not (from all sources)

| Role | Verdict | Evidence |
|---|---|---|
| Main executor for 30B/75GB MoE | **No** | baked weights, INT4 unsupported, 32 MB SRAM, can't stage experts |
| Single-token decode | **No** | ~90 µs dispatch floor vs ~23 µs crossover; ANE attention 560× slower than CPU |
| Attention/norm/argmax on ANE | **No** | causal mask ignored; RMSNorm error ~0.01; not bit-identical |
| Small dense prefill "island" (opt-in) | **Maybe** | 1×1 conv ~3× matmul, W-lane 4-token batching 5×/2.93×; must verify fp32 |
| Drafter whose output never reaches logits | **Possible** | exact-verify loophole; needs route-union + acceptance data |
| GPU↔ANE zero-copy | **Unverified** | the maderix demo files don't exist (README-only) |

## The research notes (all in this directory)

- **`ane_complete_catalog.md`** — 418-line verified constraint/capability catalog
  with 13 preserved source contradictions. The reference before any ANE design.
- **`maderix_CODE_deepdive.md`** — source-level read of maderix/ANE: exact C ABI,
  compile/load/eval costs, INT8-vs-INT4 truth, 0x1d cliff, W-lane batching.
- **`direct_ANE_runtimes.md`** — ANEForge / Orion / FusionML / others: install
  paths, Python APIs, accuracy tables, FusionML credibility warning.
- **`expert_streaming_research.md`** — 2026 MoE papers; the K=2 read-sharing
  mechanism and its decision thresholds.
- **`ane_hardware.md`, `ane_llm_evidence.md`, `ane_software_stack.md`,
  `hybrid_feasibility.md`** — the first-pass deep research (hardware, papers,
  Core ML APIs, contention).
- **`internal_engine_map.md`, `internal_app_map.md`** — maps of Veystrix's own
  hot paths and packaging/integration seams.
- **`hollance_repo.md`** — cross-check of the community reference.
- **`CREATIVE_SOLUTION_PROMPT.md`, `SECOND_OPINION_PROMPT.md`** — the prompts used
  for outside second opinions (the latter has an erratum about the 345 KB error).

## Key corrections made during research

- 345 KB = one of nine expert *regions*, not a whole expert (~3 MB). No 4.5×
  read amplification; our byte figures are self-consistent.
- The route trace DOES contain per-token expert IDs (`events`), so the gating
  test ran offline — my earlier "blocked" claim was wrong.
- FusionML's headline numbers lack reproduction artifacts (6-star repo, ANE
  engine throws); its eager-boundary *idea* is worth a cheap independent probe.
- maderix's GPU↔ANE demo files don't exist; INT8 weights do compile; S<16 is a
  0x1d error, not silent zeros.
- Owner legal name recorded as sole copyright holder in both repos (commits
  `65f8361` core, `703ebcd` public).

## Status / next gate

- Research: complete for a build decision.
- Blocked on owner authorization for: any guarded hardware run; installing
  ANEForge/FusionML; native Swift/private-API work.
- Not blocked (can do on wake): design the K=2 verifier on paper; extend the
  route-union script to Nemotron's trace if one exists; spec the bit-identical
  batch-2 reduction-order test.
