# Local-repository leads: untried speedup ideas in Veystrix's own research/experiment notes

Scope: mined only this repository's own notes and experiment records (no web research). Method: read `docs/performance/*.md`, `research_notes/Apple Neural Engine integration/*.md`, `experiments/ane/*.md`, `models/flashnext/settings/modes.py`, `models/flashnext/settings/setting_research.py`, and cross-checked every candidate against `docs/performance/DO_NOT_IMPLEMENT.md`. Citations are local file paths with line numbers / section names (this is a local-repo task, so there are no URLs).

Hard constraint applied throughout: no quality loss / bit-identical output. "Untried" means the notes explicitly frame the idea as open, pending, future work, hypothesis, next-work order, or not-yet-validated; it does NOT mean it clears the quality gate.

---

## Key Question 1 — Which ideas appear in the notes as open/pending/untried and could plausibly give a decode or prefill win with no quality loss?

### Takeaway

There are at least 8 genuinely untried leads with file citations. The strongest is **expert-wise K>1 read-sharing verification** (a software/CPU-Metal mechanism, explicitly called "the one bit-identical candidate" by the ANE research). Several others are explicitly "open" in `DO_NOT_IMPLEMENT.md`'s *Still open* list (which is the authoritative "not rejected" registry): multi-buffer/64-bit offsets for the 8 GiB pool cap, last-row-only prefill below 2,048 tokens, and larger-K verification. Most remaining leads are small prefill/TTFT wins (autotune, paged-KV, fragment cache, prefill→decode seeding) or infrastructure (continuous batching), not 2× levers.

### Cited Findings

**LEAD 1 — Expert-wise multi-token verification with shared reads (K>1), bit-identical by construction. Highest-value untried lead.**
- Premise: instead of token-row-wise verification (each drafted row triggers its own SSD expert reads — the reason MTP lost at 150–200 ms/row), group the K rows by expert, load each distinct expert ONCE (union, not sum), run a K-row GEMM, accumulate in baseline expert order. Routing and arithmetic unchanged → token-identical by construction. — [experiments/ane/DECISION_SUMMARY.md §The one concrete candidate, lines 5–18]; [research_notes/Apple Neural Engine integration/expert_streaming_research.md §7.1, lines 181–193]; [experiments/ane/ROUTE_UNION_FINDING.md §Why this matters, lines 73–79]
- Expected gain: measured K=2 union ratio 0.796 weighted mean / 0.812 raw on Veystrix's own saved route trace → ~59 ms of the 289 ms per-token read wait saved before draft cost; potential ~2.6 → 4+ tok/s. For larger K: K=6 raw union ratio 0.542 (clears Fable's 0.6 gate), miss-adjusted 0.621 (borderline) → ~38–46% fewer miss reads/token. — [experiments/ane/ROUTE_UNION_FINDING.md §Measured result and §Miss-adjusted result, lines 18–71]; [experiments/ane/DECISION_SUMMARY.md table, lines 20–33]; [research_notes/Apple Neural Engine integration/INDEX.md §The candidate, lines 35–45]
- Open premise-gate: (a) batch-K verification must reproduce batch-1 argmax exactly — MLX matmul accumulation order changes between <192 and ≥192 rows, and `dense_gemm._bit_equal` has only been validated at ≥384 rows, not K=4–6; (b) cheap-drafter greedy acceptance must be ≳70% or accepted-tokens-per-round eats the saving; (c) re-record a trace on the current 10-experts/token config (existing trace is 8 experts/token); (d) needs a guarded paired A/B run with physical-byte counters. — [experiments/ane/ROUTE_UNION_FINDING.md §Critical caveats, lines 34–51]; [experiments/ane/K2_QWEN_DESIGN.md §Honest expected outcome, lines 45–52]; [docs/performance/DO_NOT_IMPLEMENT.md §Still open, line 78]
- Documented: `DO_NOT_IMPLEMENT.md:78` explicitly lists it as OPEN: "Batch-K (K>1) expert-wise verification with shared reads (union ratio 0.790 Qwen / 0.826 Nemotron; K=2 measured slower, larger K untested)". Also `INDEX.md:16-22`, `DECISION_SUMMARY.md:96-105`, `FINAL_ANSWER.md:30-38`, `expert_streaming_research.md §7–8, lines 179–238`, `REOPEN_PLAN.md` closing note.
- IMPORTANT epistemic caveat / do not re-litigate K=2 blindly: `K2_QWEN_DESIGN.md §Critical finding, lines 6–24` says the safe K=2 read-sharing mechanism is already substantially implemented in Qwen's exact speculative verifier (`models/flashnext/qwen4_verifier.py:315-345` unions experts across all K rows in one `switch` call), and native MTP depth 2 with the exact verifier was measured SLOWER (2.298 → 1.924 tok/s, 0.837×, digest identical — [DO_NOT_IMPLEMENT.md:27]). So the genuinely open part is **larger K (4/6) with a cheap drafter and measured union/acceptance**, not re-testing K=2 through the existing MTP path. The notes themselves disagree on whether K=2 is "unmeasured" (ANE research) or "already implemented and slower" (K2_QWEN_DESIGN); treat that as unresolved.

**LEAD 2 — Full residency on 32 GB+ for Nemotron (~24 tok/s arithmetic estimate). Deferred only by the 16 GB-first rule.**
- Premise: Nemotron's 15.38 GiB routed experts become fully resident at 32 GB+ (`higher-mac-support` path in `models/nemotron35/admission.py`, `veystrix/memory_tiers.py`), removing the 75–92 ms expert-read wait; decode becomes compute-bound at ~41 ms of GPU work. — [experiments/ane/REOPEN_PLAN.md §The thesis, lines 16–24]; [docs/performance/lookup-drafting-2026-11-15.md §The Nemotron GPU question, lines 41–43]
- Expected gain: ~24 tok/s by arithmetic (1000/41), unvalidated; current 16 GB Nemotron is 6.5–8.5 tok/s. — [docs/performance/handoff-2026-11-15.md lines 7–18]; [research-survey-2026-11-15.md §What is left, lines 32–36]
- Open premise-gate: confirm on a 32 GB+ host that expert-read wait actually drops before adding any offload; owner rule "16 GB first; larger Macs only after success". — [REOPEN_PLAN.md §Validation order, lines 27–32]; [FINAL_ANSWER.md line 60]
- Documented: `DO_NOT_IMPLEMENT.md:76` (Still open), `handoff-2026-11-15.md:16`, `research-survey:34`, `lookup-drafting:43`, `REOPEN_PLAN.md:20,29-32`, `FINAL_ANSWER.md:60`.

**LEAD 3 — Multi-buffer / 64-bit offsets to exceed Flash-Next's 8 GiB single-`uint32` pool cap. Deferred only by the 16 GB-first rule / kernel redesign.**
- Premise: Flash-Next passes its decode pool to Metal as one `uint32` array → hard 8 GiB kernel-addressable cap (`FLASHNEXT_KERNEL_POOL_CAP_BYTES = 8 * GIB`), so its 75.5 GB experts can never be fully resident even on big Macs without kernel addressing redesign. Multi-buffer/64-bit offsets would remove that ceiling. — [research_notes/Apple Neural Engine integration/internal_engine_map.md §lines 161–166]; [experiments/ane/CREATIVE_SOLUTION_PROMPT.md lines 15–16]; [SECOND_OPINION_PROMPT.md line 47]
- Expected gain: enables full Qwen residency on large Macs (Qwen 75.5 GB); on 16 GB the value is questionable because larger pools are already RAM-bound. — [overnight-2026-11-16.md:292-294]; [inference-blueprint-baseline.md:131-148]
- Open premise-gate: kernel addressing change (out of scope in current tiers); needs a validated memory budget. — [DO_NOT_IMPLEMENT.md:77]
- Documented: `DO_NOT_IMPLEMENT.md:77` (Still open), `docs/hardware/MEMORY_SUPPORT.md:41` as quoted in `internal_engine_map.md:166`.

**LEAD 4 — Last-row-only prefill logits below the 2,048-token threshold (TTFT / temporary-memory candidate).**
- Premise: for prompts ≤2,048 tokens prefill computes vocabulary logits for EVERY prompt position although generation consumes only the last row; the last-row-only path already protects larger prompts. Gate must compare exact final logits because matrix-shape changes select different numerical kernels. — [docs/flashnext/research.md §lines 189–197]; [docs/flashnext/handoff.md:578-579]; [docs/flashnext/brief.md:202-204]
- Expected gain: prefill latency and transient-memory reduction, "not a decode claim" (no tok/s number measured). — [research.md:191-193]
- Open premise-gate: bit-exact final logits — the last-row projection was recorded as changing logits by up to 0.125, so this is NOT yet lossless. — [docs/flashnext/prefill-memory-investigation.md:26,69]; [docs/flashnext/long-context-10k-plan.md:78]
- Documented: `DO_NOT_IMPLEMENT.md:79` (Still open), `docs/flashnext/research.md:189-204`, `handoff.md:578`, `brief.md:202`, `prefill-memory-investigation.md:26,69`.

**LEAD 5 — Fix parallel Mamba exactness (PRF-01) and then evaluate the chunk-parallel Qwen GDN kernel (KER-08).**
- Premise: block-parallel Mamba prefill gave 3.66 → 3.91 tok/s on a 6.5K prompt but FAILED the greedy-token gate (`e42ebe…` vs `ce5d95…`); find the chunk-boundary divergence and compare against an FP32 reference at 2K–8K. Qwen's installed `gated_delta_kernel` already fuses the GDN recurrence, so only the chunk-parallel variant (KER-08) remains. — [docs/performance/inference-blueprint-handoff.md §Parallel Mamba at real context, lines 193–208 and §Remaining work #2, lines 297–301]; [inference-blueprint-baseline.md:182-192]
- Expected gain: ~7% prefill from parallel Mamba if made exact; KER-08 unquantified. — [handoff §Parallel Mamba table, lines 198–206]
- Open premise-gate: bit-exact FP32 reference at real context; keep off until greedy IDs match. — [inference-blueprint-handoff.md:203-206]
- Documented: `inference-blueprint-handoff.md:193-208,297-301`; currently in `DO_NOT_IMPLEMENT.md:31,59` as rejected *for exactness* — so it is a "correctness-gated future work" item, not presently eligible.

**LEAD 6 — Continuous batching / decode-first multi-client (CON-03 expert-union batching, CON-04 context affinity, CON-05 warm handover).**
- Premise: Nemotron groups ready decode rows from the same model in one layer-major pass keeping routed experts alive across rows; Flash-Next co-schedules singleton decode steps. `VEYSTRIX_CONTINUOUS_BATCHING=1` and `veystrix/bench/concurrency.py` exist; tiny-checkpoint tests pass; real runs not done. — [docs/performance/stage-10.md §Decode-first server, lines 11–17]; [handoff-2026-11-14.md item E, lines 116–125]; [inference-blueprint-handoff.md §Remaining work #5, lines 310–315]
- Expected gain: aggregate throughput at concurrency 2–4 and per-stream inter-token tails; NOT a single-stream 2× lever. — [stage-10.md:17]; [handoff-2026-11-14.md:118]
- Open premise-gate: concurrency-1 overhead vs the 3% budget; admission needs >4 GiB headroom beyond the request so a second conversation gets a 503 (bench has `--ignore-admission` for measurement only); `decode_many` is not transactional and one client's disconnect can fail the group. — [handoff-2026-11-14.md:120-125]
- Documented: `stage-10.md:11-17`, `handoff-2026-11-14.md:116-125`, `inference-blueprint-handoff.md:310-315`, `fable 5.2.md §4`.

**LEAD 7 — Paged KV live integration (LCT-01/07) plus LCT-03 SSD cold pages and LCT-04 QSA sparse fetch.**
- Premise: `models/shared/paged_kv.py` implements a 64-token block table, copy-on-write forks, sink/recent pinning and spill/restore; unit-tested but not on a live runtime path. — [inference-blueprint-handoff.md §LCT-01/07 implementation, line 35 and §Verification, lines 237–238 and §Remaining work #3, lines 302–305]
- Expected gain: memory/TTFT at long context; unquantified. — [inference-blueprint-handoff.md:302-305]
- Open premise-gate: needs a batched backend owning more than one conversation state, then LCT-03/LCT-04. — [inference-blueprint-handoff.md:302-305]
- Documented: `inference-blueprint-handoff.md:35,237-238,302-305`.

**LEAD 8 — Prefill autotune / shape profile (`FLASHNEXT_PREFILL_AUTOTUNE`, `FLASHNEXT_PREFILL_SHAPE_PROFILE`).**
- Premise: choose the widest balanced chunk once per target prefill from host headroom + a per-width measured `seconds_by_chunk` corpus; width and plan version are part of prefix-cache identity. — [docs/performance/stage-6.md §lines 3–28]; [setting_research.py:153-157]
- Expected gain: TTFT / computed-prefill win via the p90 paired delta; not yet reported. — [stage-6.md:67-69]
- Open premise-gate: real paired run with a per-width shape corpus from the same checkpoint and execution order. — [stage-6.md:11-17]
- Documented: `stage-6.md:3-35,67-69`, `setting_research.py:153-157`.

**Lower-priority but explicitly open (unlikely to reach 2× on their own; listed for completeness):**
- `PREFILL_SEED_DECODE` / `VEYSTRIX_NEMOTRON_PREFILL_SEED_DECODE`: copy one hot expert per layer from completed prefill ring reads into a free decode slot. Unmeasured because the guard declined the decode pool in the only combined measurement. — [docs/performance/prompt-lookup-and-prefill-2026-11-13.md:19-23,39-42,68-69]; [setting_research.py:134]
- Fragment/universal cache: warm TTFT at 8K–64K prompts; only measured at 2K with a small gain because the prefix cache already served most of the work. — [inference-blueprint-handoff.md §Tokenized fragment cache, lines 210–222 and §Remaining work #4, lines 306–309]
- Exact-kernel fusion family (KER-06 fused norm+router+top-k, KER-05 4-branch residual, KER-03 fused quantized expert) and QNT-05 tiered KV beyond the current opt-in: "need quality gates; nothing in this pass promotes an approximate path." — [inference-blueprint-handoff.md §Remaining work #6, lines 316–320]. Caveat: an explicit cost split found fusing Qwen's hyper-connection elementwise pieces worth only ~1% of a 4.3K prefill, so some fusion ideas are already bounded low. — [overnight-2026-11-16.md §Hyper-connection cost split, lines 398–400]
- `gather_qmm` for Nemotron routed experts (~5–8% prefill, needs memory): listed in "What is left (levers, not done)". — [overnight-2026-11-16.md:447]. Conflicting evidence: a one-round screen was slower (GATHER 3.52 vs 3.80 tok/s) and `GATHER_MAX_EXPERTS=24` silently sends larger chunks down the per-expert path. — [inference-blueprint-handoff.md:125]; [handoff-2026-11-14.md:114]. Treat as a low-confidence lead gated on bit-exactness and a raised cap.
- Qwen routed-expert prediction to read fewer prefill bytes: gain bounded by 6–24% of unused bytes; needs high recall with on-demand misses. — [overnight-2026-11-16.md:446]. Weak: route-union prefill showed no gain over the app's profile. — [lookup-drafting-2026-11-15.md:39]; [DO_NOT_IMPLEMENT.md:46]
- ANE "output-neutral work during SSD stalls" (Track A Phase 3) and a small dense prefill "island"/W-lane ≤4-token batching: explicitly "Maybe"/"Possible" and "worth a cheap independent probe" with expected gain small, and must be fp32/bit-verified. — [experiments/ane/REOPEN_PLAN.md:62-66]; [INDEX.md:54-55]; [maderix_CODE_deepdive.md:325,403]; [FINAL_ANSWER.md:50-56]. These do NOT meet the 2× bar and ANE numerics are disqualified under the bit-identical rule.
- ANEForge tie-margin fp32 fallback is reusable bit-exactness *infrastructure* (zero speedup on its own). — [DECISION_SUMMARY.md:62-69]; [FINAL_ANSWER.md:40-41]

### Inferences

- The repository's own highest-upside untried mechanism is a software read-sharing verifier, not ANE; it attacks the actual bottleneck (SSD expert bytes) while preserving bit-exactness by construction.
- Most prefill ideas in the notes are already measured and adopted into the `exact-speed` bundle (`EXACT_SPEED_FEATURES` in `modes.py:236-254`); genuinely new prefill wins are small (autotune, last-row logits, paged KV, seeding) rather than 2×.
- Several "open" items are gated less by feasibility than by permission to run a guarded A/B and by exactness checks — the notes repeatedly say "just needs ONE guarded run" / "needs the owner's authorization" (e.g., `K2_QWEN_DESIGN.md:77-81`, `INDEX.md:90-97`, `stage-6.md:37`).

### Gaps

- No note gives a measured tok/s for larger-K (4/6) expert-wise verification; all gains are derived from offline union ratios.
- The trace used for the K>1 statistics is from a different checkpoint (8 vs 10 experts/token) and may not transfer; `K2_QWEN_DESIGN.md` and `INDEX.md` disagree on whether K=2 has already been tested.
- "Last-row-only prefill" is explicitly recorded as changing logits by up to 0.125, so no lossless version is documented yet.
- No note quantifies the gain of paged-KV live integration, fragment cache at 8K–64K, or prefill→decode seeding under an actual decode pool.

---

## Key Question 2 — Which leads are explicitly deferred only because of the "16 GB first" rule?

### Takeaway

Two hardware-residency leads and one ANE-offload track are deferred purely by the owner's "16 GB M4 first; larger Macs only after success" rule, not by a measured rejection. They remain open in `DO_NOT_IMPLEMENT.md` and the ANE decision docs.

### Cited Findings

- Nemotron full residency on 32 GB+ (~24 tok/s arithmetic): open, unvalidated — [DO_NOT_IMPLEMENT.md:76]; [REOPEN_PLAN.md:20,27-32]; [FINAL_ANSWER.md:60]; [research-survey-2026-11-15.md:34-36]; [lookup-drafting-2026-11-15.md:43]
- Flash-Next multi-buffer / 64-bit offsets to exceed the 8 GiB single-`uint32` pool cap (full Qwen residency impossible even on big Macs without this): open — [DO_NOT_IMPLEMENT.md:77]; [internal_engine_map.md:161-166]; [CREATIVE_SOLUTION_PROMPT.md:15-16]; [SECOND_OPINION_PROMPT.md:47]
- ANE Track B (32 GB+), i.e. ANE offload of static dense attention/FFN of a fully-resident model: "the credible path" physically, but deferred until the 16 GB path succeeds — [REOPEN_PLAN.md:16-32]. Track A (16 GB) stays conservative/output-neutral only.
- Owner rule stated verbatim: "16 GB M4 first; larger Macs only after success" — [INDEX.md:3-4]; [DECISION_SUMMARY.md:3]; [REOPEN_PLAN.md:1,29-32]
- Research survey's "What is left": more RAM (32 GB+) is "the only measured lever"; a GPU with resident weights is where HyperQwen/EVICT/DraftExpert-style methods apply — [research-survey-2026-11-15.md:32-36]
- On 16 GB the arithmetic is explicitly hopeless for full residency: 85%/95% Nemotron residency needs 15.4/17.0 GiB before core/KV; Qwen 55% needs 41.5 GiB — [inference-blueprint-handoff.md:104-115]; [inference-blueprint-baseline.md:131-148]

### Inferences

- These are not "dead ends" — the owner must not discard them when pruning rejected flags; they are park-until-larger-hardware items.
- Any future 32 GB+ validation must first confirm the residency premise (ms/token and MB/token drop) before attributing gains to ANE or other offload, per `REOPEN_PLAN.md:55-60`.

### Gaps

- No 32 GB+ hardware run exists anywhere in the notes; all gains there are arithmetic estimates.
- No note states how the 8 GiB pool-cap redesign would be tested for bit-exactness.

---

## Key Question 3 — Which candidate ideas are already rejected (do NOT re-list as new leads), and which are ineligible under the no-quality-loss constraint?

### Takeaway

`DO_NOT_IMPLEMENT.md` is the authoritative measured-dead-end registry. Cross-checking every lead above against it: none of Leads 1–8 appear in its rejected tables, except Lead 5 (parallel Mamba) and `gather_qmm`, which are flagged as correctness-gated/conflicting rather than clean speedup wins. Everything in the ANE tables is rejected on this 16 GB M4.

### Cited Findings

Rejected decode (do not re-list): expert prefetch, decode overlap, I/O worker counts, pool policies other than S3-FIFO, larger pools, perfect-predictor+overlap, native MTP at all depths (incl. depth 2 exact-verifier 0.837×), lookup drafting v2, Nemotron read-ahead/stale-drop, parallel Mamba, 4K chunks, `FLASHNEXT_COMPILE_GDN`, cache-aware routing (trajectory gate), expert packing, zero-copy shards, `F_NOCACHE`, `F_RDAHEAD=0`, wired-limit sysctl — [DO_NOT_IMPLEMENT.md §Decode — rejected, lines 16–40]
Rejected prefill (do not re-list): route-union prefill, PLE row prefetch, QSA gather tiles (pre-fusion), `qsa-fast-prefill` (fails tool_2 quality gate and is NOT bit-identical), pipelined attention tiles at end-to-end level, truncating invisible keys, compiling injection, dequantize-all-history, fused gather rounding product, `FLASHNEXT_COMPILE_GDN`, `mx.async_eval`, early shared expert, Nemotron 4K chunk, block-parallel Mamba — [DO_NOT_IMPLEMENT.md §Prefill — rejected, lines 42–59]
Rejected ANE (all on this 16 GB M4): ANE as main executor/expert cache, single-token decode, attention/norm/router/LM head, Core ML ANE offload, palettized 4-bit Core ML, native PyObjC zero-copy, ANE drafter+exact verify, FusionML CPU+GPU row split, bit-exact ANE intermediate layer — [DO_NOT_IMPLEMENT.md §ANE, lines 61–73]; [FINAL_ANSWER.md:8-28]
- FusionML's 1.18–1.25× claim is explicitly downgraded: its headline numbers are unverified single-author claims (6-star repo, ANE engine throws), and Veystrix's own raw M4 data LOSES to MLX (0.763 vs 0.357 ms at batch 128). The eager-boundary *idea* is "worth a cheap independent probe", not adoption. — [DECISION_SUMMARY.md:40-60]; [FUSIONML_FORENSICS.md:49]; [DO_NOT_IMPLEMENT.md:72]. Note an older note still recommends it (`direct_ANE_runtimes.md:44,337`); treat the newer forensics/synthesis as authoritative.
- Explicitly OPEN (not rejected): full residency 32 GB+, multi-buffer/64-bit offsets, batch-K K>1 read-sharing, last-row-only prefill <2,048 tokens — [DO_NOT_IMPLEMENT.md §Still open, lines 75–79]
- Ineligible under the no-quality-loss rule (not "untried promising"): cache-aware routing for Nemotron (swap breaks tool-call JSON at ε=0.05/0.1; ε=0.02 unrun but only ~+3% and still changes routing), 2-bit/mixed-precision experts, expert budgeting/pruning, S2-MoE/Skliar reranking, DraftExpert (trained experts) — [research-survey-2026-11-15.md:19-25,54-72]; [DO_NOT_IMPLEMENT.md:34,38]; [expert_streaming_research.md:140-150,230-236]

### Inferences

- The `EXACT_SPEED_FEATURES` bundle in `modes.py:236-254` plus `EXACT_SPEED.md` represent the adopted, measured, bit-identical wins; re-proposing any of those as "new" would be wrong.
- The owner's planned flag-removal decision (`DO_NOT_IMPLEMENT.md:81-90`) is about code hygiene, not new speed — rejected flags are preserved as rollback/comparison paths, which is why they still appear in `modes.py`/`setting_research.py`.

### Gaps

- `DO_NOT_IMPLEMENT.md` requires a number and source per entry; the "Still open" items (lines 75–79) have no measured number by definition, which is why they are leads.
- No note reconciles the `direct_ANE_runtimes.md` pro-FusionML recommendation with the later forensics; I defer to the later, source-verified documents.
