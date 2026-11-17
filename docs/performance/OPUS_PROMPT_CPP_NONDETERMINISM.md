# Opus prompt: C++ engine nondeterminism on the real checkpoint

Paste this to Opus. It is self-contained; the repo is public.

---

## Request

You are looking at the Veystrix inference engine. I need a strong second opinion on a subtle numerical bug. Please read the repo and give me:

1. your diagnosis of the root cause,
2. a minimal, **provably bit-exact** fix (concrete patch or precise algorithm with file/line references),
3. explicit answers to the 7 doubts at the end.

Do not just offer a hypothesis — point to exact files and lines.

## Repo / branches

- Repo: https://github.com/joyboyreturns/Veystrix-Core
- Branch `cpp-engine` (tip `349fc03`): the C++ rewrite plus every determinism diagnostic. **The bug lives here.**
- Branch `nemotron-pool-nocache-fc2skip` (tip `ccdcfa1`): Task A/B C++ optimizations. Read for context only — Task A (`cpp/engine/src/nemo_pool.cpp`) uses the *same* persistent host-buffer slot pattern as the implicated code, so it may share the bug.
- Never modify `main`.

## The goal / hard gate

The C++ engine must be **bit-identical** to the Python engine on the real ~105 GB Qwen3.8-Flash-Next / Nemotron checkpoint: exact token IDs, exact logit SHA-256, exact BF16 rounding, exact MoE routing. Python is deterministic (3/3 identical runs); C++ is not. Until parity passes, Python remains the serving engine.

Python ground truth: `results/flashnext/20261117-222244-real-reference/reference.json` — tokens `[271, 248068, 198, 760, 1156, 579, 1876, 7701]`, identical across 3 guarded runs.

## What already works

- 20/20 bit-exact ctests on synthetic configs; `native_generate` matches Python token IDs + logit SHA byte-for-byte on synthetic weights.
- Every isolated MLX op is deterministic 5/5 in `cpp/engine/tools/determinism_probe.cpp`: `qmm`, dequantize+`mx::matmul`, fused GDN delta, fused SDPA, router (incl. exact-tie stress), `gather_qmm`, NaN/Inf handling.
- The `race` probe indicates `mx::eval` is synchronous w.r.t. host-backed MoE buffers.

## Evidence chain (eliminations)

1. **Real cross-check fails**: 0/8 tokens, 0/8 logit digests; first divergence at token 0. C++ gives a different step-0 token each run; one run SIGBUS at ~13.5 GB peak.
2. **lm_head split-K disproven**: on Apple M4, shape `[1,1,2560] @ [248320,320]` gives M=1 < `vector_limit`=13, so dispatch is `affine_qmv_fast` (one SIMD group per output row, no atomics). Dequantizing the head is not a fix.
3. **Layer-hash localization** (42 arrays, last prefill step, 10-layer reduced config, guarded 2 runs): layers 0–5 all bit-identical; first divergence `L06_branch_gdn`; NaN/Inf = 0 everywhere.
4. **Fused kernel + lazy-aliasing fix FAILED**: replacing the fused GDN delta step with an ops-based step and adding `mx::eval` barriers on conv/recurrent state at every layer boundary changed the values but left the divergence pattern unchanged at L06. Both hypotheses ruled out.
5. **Weight hashes identical**: SHA-256 of every dequantized GDN weight for layers 0,5,6,7 plus cache-state init, across 3 separate processes — all 31 hashes identical. Weight loading / shard-9 ruled out.
6. **Per-step state hashing** (16 steps × 10 layers × {conv,rec} = 160 hashes/run, guarded 2 runs):
   - step 0: all states identical;
   - **step 1: ONLY layer 9 diverges, both `conv` and `rec` simultaneously**; layers 0–8 identical;
   - step 2: still only layer 9; step 3+: layers 6 and 8 join; layers 0–4 and 5 never diverge.
   - **Layers 5 and 8 are the only `CrossShardMoE` layers in this config** (12 of 48 in the full model). The layers whose downstream GDN states diverge first are exactly 6 (after 5) and 9 (after 8) — a strong, specific correlation.

## Implicated code (please read it)

`cpp/engine/tools/real_checkpoint_generate.cpp`:

- `class CrossShardMoE` at line 413; `forward()` at line 473.
- Lines 451–461: persistent `buf_` — `posix_memalign(64)` + `shared_ptr<uint8_t>` with a normal `free` deleter; zero-initialized once at construction.
- Lines 516–553: `plan_remap` across the 9 expert tensors (per-tensor fds, shard boundaries); `run_batched_reads(reads, 8)` fills `buf_`; LRU metadata (`slot_expert_`, `free_slots_`, `slot_ref_`, `slot_last_`) mutated around the read.
- **Lines 555–564, key suspect**: the persistent buffer pointer is wrapped in an `mx::array` with a **no-op deleter**, so MLX neither owns nor copies the storage:
  ```cpp
  auto keep = buf_[i];
  a.emplace_back(buf_[i].get(), std::move(shape), dtype,
                 [keep](void*) {});
  ```
- Lines 572–573: `moe_block(x, route_slots, scores, ew); mx::eval(out);`

Full written evidence: `research_notes/Cerebras inference and IO splitting/determinism.md` (from "Per-step GDN state diagnostic" onward), `docs/performance/CPP_ENGINE_REAL_CROSSCHECK.md`, `docs/performance/CPP_DETERMINISM_FINDING.md`, `cpp/engine/STATUS.md`.

## My doubts — please answer each explicitly

1. **Why is Python deterministic on the same hardware and weights while C++ diverges?** What does Python's MoE path do differently (fully-resident dequantized experts? different buffer ownership? different read batching)? Which Python file/function should I compare against?

2. **Where exactly is the divergence in the MoE path** — (a) the expert weights read into `buf_`, (b) the route/scores fed into `moe_block`, or (c) `gather_qmm` numerics on a no-copy external host buffer? The synthetic probe says `gather_qmm` on a host-backed slot array is deterministic — but it never exercised *cross-step buffer reuse*. Could MLX memoize/reuse a graph keyed on the same external buffer pointer across steps?

3. **How can overwriting `buf_` at step N+1 change step N's result if `mx::eval` is synchronous?** Is there any lazy path inside `moe_block` that `mx::eval(out)` does not force? Or is the real bug in `plan_remap`/`run_batched_reads` — e.g. duplicate reads, LRU bookkeeping updated before reads finish, or read-completion order deciding which expert lands in which slot?

4. **Is this the same root cause as Task A?** `nemo_pool.cpp` uses the same persistent `posix_memalign` slot buffers + coalesced batched reads. Its 27-check bit-exact test forces eval and does *not* exercise cross-step reuse or multi-thread completion. If the persistent-buffer pattern is itself unsafe, Task A needs the same fix. Please assess.

5. **Layer 8's own GDN state diverges at step 3 (secondary).** Is that downstream of the same MoE bug, or a *separate* full-attention KV-cache/SDPA aliasing bug at layer 7?

6. **What is the minimal fix that can be proven bit-exact?** Options I see: (i) copy gathered expert rows into fresh owned `mx::array`s each step (deep copy before `moe_block`); (ii) route cross-shard layers through a fully-resident dequantized expert path (kills the streaming benefit); (iii) add an explicit ownership/materialization barrier. Which is correct, and which preserves the I/O benefit?

7. **The full 48-layer model has 12 cross-shard layers.** Do we expect the same divergence pattern there, and will your fix scale?

## Constraints

- No quality loss: bit-identical tokens, logits, BF16 rounding, routing. "Close enough" is not acceptable.
- 16 GB M4 validation first. Real-checkpoint runs require owner permission + the guard.
- I will verify whatever you propose against the bit-exact gate before applying it to shared primitives.
