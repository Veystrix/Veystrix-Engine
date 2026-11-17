# Validation of Opus second-miracle response (2026-11-17)

Checked against actual checkpoint + engine code + our measurements. Verdict: the
response is HIGH QUALITY and its central new lever is real. One of its challenges
to our data is correct and corrects an error in the prompt we sent it.

## Validated
1. **squared-ReLU CONFIRMED.** Checkpoint config `mlp_hidden_act: relu2`; engine
   `models/nemotron35/streamed_experts.py:862` does `mx.gather_qmm(nn.relu2(h), ...)`.
   Expert = 2× 2688×1856 Q4 = 5,612,544 B. Exact zero mask comes from the exact
   up-projection output — no prediction, so bit-exactness is structural. The lever
   stands. Qwen uses SwiGLU (no exact zeros) — lever does not apply to Qwen, correct.
2. **Opus's byte-counter challenge is CORRECT.** The 857 MB/token I gave Opus for
   Nemotron was a no-pool profile arm (`FABLE_IO_BRIEF_FULL.md:74-78`, 436 MB at
   96 tokens; long-trace sim constants). Production app-like cells read only
   **193–216 MB/token** at ~7 tok/s (`handoff-2026-11-15.md:24,29,76`) because the
   page cache is already at ~74% hit. Logical routed bytes = 23×6×5.61 = 774.5 MB;
   OPT@2 GB = 265 MB. So Nemotron is NOT 857 MB bound — it is ~200 MB bound. This
   materially changes the arithmetic: Opus's 15 tok/s analysis (needs ≤29 MB/token
   at 58 ms floor) is in the right framework, and the squared-ReLU lever could help.
   CORRECTION to carry forward: use ~200 MB/token (production, page-cache warm),
   not 857 MB, for Nemotron decode.
3. **Lead 1 (read-once dedup) DROP — proof accepted.** Cache already realizes union
   sharing: an expert in D_t ∩ D_{t+1} is read once (miss then hit). E1 confirmed
   perfect future knowledge changed physical bytes only −1.8%. A batch over
   {t,t+1} reads the same set minus resident. No byte gain; speculation adds bytes.
4. **Lead 2 per-layer certificate DROP — proof accepted.** Expert outputs feed the
   residual/KV/GDN state that later tokens read; skipping reads breaks bit-exactness
   for all later tokens; recomputing exactly reads the same bytes. Head-only
   certificate is structurally exact (dense fallback) but saves only a few ms and
   needs a different gate (hidden-state digest, logits not all computed).
5. **ANE DROP — consistent with our own 6-stream research.** No bit-exact role.
6. **Qwen 5 tok/s no-software-solution — consistent with our E1/E3/STEP_B findings.**
   Best realistic ~3.5–3.9 tok/s. The one untested combination (pool as sole cache
   + F_NOCACHE) is worth an A/B for a partial gain toward ~3.2, not the target.

## Decision
The squared-ReLU zero-column skip is the first genuinely NEW bit-exact byte lever
that (a) has a real architectural basis, (b) is not killed by the E1 prediction
ceiling, and (c) targets Nemotron, where 15 tok/s is still open. Run Opus's Phase 0
(offline, no hardware, no guard): hook the existing 3×1,200-token traces, record
exact-zero masks per (layer, expert, token), measure nonzero fraction and needed
chunk fraction. Kill if <60% zeros or needed-16KB-chunks >0.6. This is cheap and
decisive — do it next.
