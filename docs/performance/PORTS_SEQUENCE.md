# Sequential port plan (after config/method sweep)

**Branch:** `speedup-roadmap`
**Language decision (owner, 2026-11-17):** C++ with the MLX C++ API for the pager and
eventual full engine port. Reuse the existing C `native_read.c` for batched reads. No Rust.
**Order:** highest expected value first. One port at a time; each gets unit tests (no hardware),
then a guarded A/B with digest gate (needs owner permission). A port is only kept if it is
bit-identical AND faster; otherwise it is documented and the flag stays off.

| Order | Port | Source claim | Our realistic expectation on 16 GB | Status |
|---|---|---|---|---|
| 1 | #11 io_uring/O_DIRECT pager + KICK/WAIT | 1.47→16–22 tok/s, byte-exact | **Downgraded:** 2 of its 3 mechanisms already rejected here. Only id-remap + async-pool + zero-copy handoff portable, ≤6% Qwen / ≤39% Nemotron ceiling. | CORE DONE; hookup gated on full port |
| 2 | #15 full-layer staging + 2nd SSD mirror | prefill 29→240 t/s | **HARDWARE BLOCKED:** only 1 physical SSD (`diskutil list`: disk0 + synthesized). Portable part (two-half ring staging, read l+1 while l computes) already in `prefill_ring.py` and ON in agentic mode | BLOCKED / already partly deployed |
| 3 | #12 lossless rANS + 3-tier cache | 2.78×, decode 2.10–15.58× | **DEAD END:** 4-bit weights have 0.9–2.1% compressibility (RPI 2025 arXiv:2503.18869); rANS adds CPU cost; ratio too small to cut 408 MB/token | REJECTED (research) |
| 4 | #22 SwiftLM GPU-vs-SSD rewrite | 8.5× from bad baseline | **WEAK:** source's 8.5× was 0.58→4.95 from a single-queue baseline; our engine already runs 2.3–7 tok/s (past that baseline). Little headroom | LOW VALUE |
| 5 | #19 last-row-only prefill | repo lead, no number | **GATE FAILS:** currently changes logits (not bit-exact). No published number. Needs proof of exactness before any A/B | BLOCKED (quality gate) |
| — | #1 full C++ engine port | 1.065×/1.386× ceilings | Owner wants it later for future-model reuse | LATER (owner) |

## Port 1 (#11) scoping note (2026-11-17)
The source's 16–22 tok/s came from three things. We can only port one:
- **O_DIRECT / page-cache bypass** — EXCLUDED. Our `F_NOCACHE` measured 8% slower
  (`DO_NOT_IMPLEMENT.md:38`); macOS page cache helps decode's expert reuse.
- **KICK/WAIT overlap of shared-expert compute with misses** — EXCLUDED. Our decode-overlap
  and early-submit were within band / −8.2% (`:21,25`); early shared expert −10% (`:86`).
- **CPU id-remap op + async batched-read pool + compact slot pool, handed to GPU with no per-layer
  Python round-trip** — PORTABLE. This is the remaining ~6–28% host-time target and the seed of
  the full engine port. Build this as `cpp/decode/io_pager.*` (C++/MLX) calling the existing C
  `native_read.c`, behind an opt-in flag. Expect at most ~1.065× on Qwen, ~1.386× on Nemotron —
  i.e. it will NOT hit 5+/15+ alone. Keep only if bit-identical and faster.

## Port 1 progress (2026-11-17)
- [x] Scoped to portable mechanisms; O_DIRECT + overlap excluded with citations.
- [x] `cpp/decode/io_pager.{hpp,cpp}` — C++ id-remap: deduplicates routed experts,
  assigns slots deterministically, emits one batched-read record per unique miss,
  fails closed (-1) when the slot pool is exhausted. Zero GPU, pure function.
- [x] `cpp/decode/tests/test_io_pager.cpp` — bit-exact/determinism/fail-closed test:
  **ctest 1/1 passed**.
- [x] `run_batched_reads` wired to the existing `native_read.c` `flashnext_batch_pread`
  (ABI-matched 32-byte PagerRead struct), bit-exact.
- [x] `test_pager_read.cpp`: writes a fake expert file, plans a route with dupes +
  a resident expert, reads via native batched pread, wraps a slot zero-copy as
  `mx::array`, confirms bytes identical. **ctest 2/2 passed** (`io_pager`, `pager_read`).
- [x] BUT: engine hookup is gated. `build_plan` (`expert_cache.py:784-794`) already does the
  same dedup/slot map in Python; replacing that call site with the C++ pager IS part of the
  full C++ engine port the owner deferred. The standalone pager cannot be wired in without it.
- Honest expectation: this alone cannot reach 5+/15+; it is the seed of the full port.
- Decision: keep pager as tested seed; do not force a risky mid-engine ctypes hook. Move to Port 2.

## Overall port verdict (2026-11-17)
After investigating all 5, **none is a viable path to the targets on this 16 GB M4**:
- Port 1 (#11 pager): core built + bit-exact tests pass, but only ~≤6%/≤39% host-time ceiling,
  and live hookup requires the deferred full engine port. 16–22 tok/s depended on O_DIRECT +
  CUDA hardware that are rejected here.
- Port 2 (#15): needs a 2nd SSD that doesn't exist on this machine; portable staging is already
  deployed.
- Port 3 (#12): 4-bit weights are 0.9–2.1% compressible — cannot cut the SSD bus meaningfully.
- Port 4 (#22): 8.5× was from a baseline far slower than our engine.
- Port 5 (#19): fails the bit-exact gate.

Root cause remains physical: ~408–436 MB/token SSD reads on a ~3 GB/s bus cap decode near
7 tok/s. The only levers that reach 5+/15+ are more memory (#17), a 2nd Mac (#18) — both deferred
by owner rule — or a lossy method (forbidden). Recommend stop ports and consolidate.

## Per-port gate (all ports)
1. Design note + C++/MLX scaffold in `cpp/decode/`.
2. Bit-exact unit test (raw bytes / logits / cache tensors) — no hardware.
3. Opt-in env flag, off by default; no change to the default path.
4. Guarded A/B vs current baseline, token + logit digest gate (owner permission before run).
5. Keep only if bit-identical AND faster; otherwise document and leave flag off.
