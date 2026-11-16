# FINAL ANSWER: can ANE improve Veystrix prefill/decode on 16 GB? (2026-11-17)

Owner objective: search ANE completely, document it, and stop only if a way is
found to improve prefill AND decode speed using ANE, validated on 16 GB first.

## Verdict

**No. There is no bit-identical ANE path that improves Veystrix's prefill and
decode on the 16 GB M4.** Eight research threads (papers, community references,
and source-level reads of maderix/ANE, ANEForge, Orion, FusionML) converge on
this. The blocker is not effort — it is hardware + the bit-identical rule:

1. **ANE cannot execute the MoE models.** Weights are baked at compile time (~2 GB
   program ceiling), INT4 has no ANE path, experts are streamed from SSD and cannot
   be staged into the engine. ANEForge's own MoE runs ~2 tok/s at lower fidelity;
   Qwen3.5-27B is "fp16-bound" and "shallower on hard prompts" than MLX.
2. **The bit-identical rule disqualifies ANE numerics.** ANE is fp16-only; the
   ANEForge tie-margin guard restores greedy *token-ID* equality only at the final
   head, fires 2–3% of tokens, is decode-only as shipped, and cannot repair error
   in intermediate layers. No engine (ORT-MLX, tritify, ExBLAS, runtime-certified
   attention) provides cross-runtime bitwise BF16 identity.
3. **Decode is SSD-bound, not compute-bound.** 289 ms of 385 ms is expert reads;
   ANE's advantage (fused fp16 matmul on a separate path) cannot touch it. Every
   proposed ANE decode role (drafter, router, LM head) inserts a cross-engine hop
   into an already memory-bound path.
4. **Prefill's remaining time is non-MoE kernels** (attention/GDN/HC) that ANE
   cannot run bit-identically — ANE attention is 560× slower than CPU GQA, and
   RMSNorm has an ~0.01 error floor.

## What the research DID produce

**One bit-identical, 16 GB-relevant candidate — but it is NOT ANE:**
**expert-wise K=2 read sharing** (group two tokens by expert, load each distinct
expert once, accumulate in baseline order). Our offline measurement of the saved
route trace gave a K=2 union ratio of **0.796 mean** → ~59 ms/token of the 289 ms
read wait potentially saved. It runs on CPU/Metal, is bit-identical by
construction, and targets the real bottleneck. This is the highest-value next
build, but it does not satisfy "using ANE."

**Reusable infrastructure:** ANEForge's tie-margin fp32 fallback can be a general
bit-exactness guard for any future low-precision path (zero speedup on its own).

## What is NOT viable (evidence in research notes)
- ANE as main executor / expert cache / runtime-weight staging — impossible.
- ANE single-token decode / attention / norm / router / LM head — slower or
  non-bit-identical (or both).
- FusionML's 1.18–1.25× prefill claim — unverified single-author (6-star repo,
  ANE engine throws); its eager-boundary *idea* is worth a cheap independent probe.
- maderix GPU↔ANE zero-copy demo — README-only; files don't exist in any commit.

## Decision for the owner

- **ANE does not meet the objective on 16 GB.** It cannot be the speed mechanism
  under the bit-identical rule. If the rule is relaxed to "token-ID identical",
  the only defensible ANE feature is head-only prefill with a ported tie-margin —
  expected ~1.0× prefill speedup — not worth it.
- **The productive path is the non-ANE K=2 read sharing.** Recommend asking
  permission for ONE guarded hardware run to validate it, with a bit-identical
  digest check and physical-byte counters, before any code adoption.
- Higher-Mac work stays deferred per the owner rule until a 16 GB feature succeeds.

## Evidence index
- `research_notes/Apple Neural Engine integration/INDEX.md` — master entry point
- `experiments/ane/DECISION_SUMMARY.md` — synthesis and corrections
- `experiments/ane/ROUTE_UNION_FINDING.md` — the K=2 measurement
- `experiments/ane/ANEFORGE_SOURCE_FINDING.md`, `maderix_CODE_deepdive.md`,
  `ane_decode_final.md`, `ane_bitexact_path.md`, `ane_complete_catalog.md` — the
  source-level and literature evidence
- 9 commits this session; working tree clean apart from preserved `results/`.
