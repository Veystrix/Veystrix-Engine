# C++ full-port status (living document)

Goal: replace the Python serving engine with the C++ engine, feature-for-feature,
with bit-identical tokens/logits; then stand up multi-session end to end.
Python remains the serving engine until every item below is validated.

## LEGEND
- [DONE-VALIDATED] bit-exact verified by test + real guarded runs
- [DONE-OFFLINE] implemented/tested offline only (no real-checkpoint gate)
- [IN PROGRESS] actively being ported
- [TODO] not started
- [BLOCKED] needs owner decision/input

## Core engine (Qwen Flash-Next) — [DONE-VALIDATED]
Branch `cpp-engine-qdense`, PR #3.
- 16-token gate (b): 3/3 clean runs, 8/8 tokens + 8/8 logit SHA-256.
- S=128: all 48 layer hidden states bit-exact vs Python.
- 1024-token + 4 decode steps: all tokens + digests match; 3/3 deterministic;
  ~31 s (was 174 s), swap_peak 3636 MiB.
- Key fixes: bf16 GDN conv cache; compiled `nn.silu`; eager load-time eval of
  dense weights + PLE/token embeds (IO race); coalesced PLE reads;
  first-occurrence compact expert stack with sorted `gather_qmm` for prefill
  (single- and cross-shard).
- ctests 28/29 (only `native_generate` fails — pre-existing worktree `.venv` path artifact).

## Feature port matrix
| Feature | Python behavior | C++ status | Class |
|---|---|---|---|
| Greedy sampling | argmax | [DONE-VALIDATED] `sample_greedy` | NUMERIC |
| Non-greedy sampling (THINKING/INSTRUCT) | distribution + seeded categorical | [DONE-OFFLINE] 11-case bit-exact test; wired into native_generate opt-in | NUMERIC |
| Prefix cache | exact snapshot/restore + SHA chain | [DONE-OFFLINE] `prefix_cache` module + synthetic test (bit-exact capture/save/load/restore, COW, fail-closed); NOT yet wired into the real generator | NON-NUMERIC |
| Variable context window / admission | refusal above server window | [TODO] protocol-level; C++ tool currently accepts any length | NON-NUMERIC |
| KV tiers (affine4/8 + tiered4) | lossy KV modes | [DONE-OFFLINE] quantize/dequantize + tiered4 split primitives bit-exact vs Python (CPU device); full attention integration remains | NUMERIC |
| MTP / speculative | native draft head | [DONE-OFFLINE] fuse_inputs + draft forward + 11 verify cases bit-exact; real-checkpoint/QSA/expert integration remains | NUMERIC |
| Cooperative decode (batching foundation) | cooperative decode co-step | [DONE-OFFLINE] single-thread interleaved==solo bit-exact (4 sessions); full server integration remains | NON-NUMERIC |
| Expert pool policies (s3fifo/etc) | LRU default + alternatives | [TODO] C++ has LRU slot pool | NON-NUMERIC |
| **Nemotron Mamba layer** | Mamba-2 mixer decode | [DONE-OFFLINE] bit-exact output+conv+ssm state vs real Python mixer | NUMERIC |
| **Nemotron bf16 attention** | dense bf16 KV decode | [DONE-OFFLINE] bit-exact output+q/k/v+cache vs real StreamedAttention (INT8 path still TODO) | NUMERIC |
| **Nemotron 52-layer loop (INT8 KV, production default)** | full decode + final head | [DONE-VALIDATED] real 24 GB checkpoint: 4/4 tokens + 4/4 digests, 3/3 runs; bf16 opt-in also validated | mixed |

## Nemotron port plan
`docs/engineering/nemotron_mamba_port_plan.md` on `cpp-engine-qdense`:
verbatim MSL `ssm_kernel` port (no MLX primitive exists), ~880 LOC total,
kernel-first then layer then loop. Kernel stage [IN PROGRESS].
Also still needed: 6 dense-attention layers with INT8 block KV (no C++ primitive)
and the 52-layer loop + final head.

## Multi-session
PR #4 (`multi-session-phase0`): offline simulator + design docs + measurement
protocol, 12/12 tests, feasibility verdict (Nemotron plausible, Qwen byte-limited).
True end-to-end (shared pool/scheduler in C++) is explicitly Phase 1+ and was
deferred by the owner; requires the engine full port first.

## Cross-language + non-greedy + MTP gaps CLOSED 2026-11-18 [measured]
- **Cross-language on arbitrary prompts**: reviewer premise DISPROVEN. Fresh
  Python refs at N=16/64/256/512/1024 all bit-exact C++ vs Python (4/4 tokens
  + digests each). N=128 strict gate fails on EOS early-stop COUNT only (Python
  stops at EOS 248046); digests verified bit-exact. Doc:
  `docs/engineering/xlang_parity_multilen.md`.
- **Non-greedy real-checkpoint**: THINKING/INSTRUCT opt-in sampling matches
  Python 8/8 sampled tokens + 8/8 digests e2e; 11/11 isolated sampler cases.
  Keyless default-key draw required. Doc: `docs/engineering/nongreedy_real_parity.md`.
- **MTP full**: nonzero draft positions (7, 11) with prepopulated QSA KV cache
  bit-exact vs Python (0/248320 diff elements, SHA match); position-0 regression
  passes. Remaining: sparse indexer >=2048, fused-RoPE production path (~1 ULP).
  Doc: `docs/engineering/mtp_full_parity.md`.
- **Multi-session end-to-end re-verified on fully integrated tree**: N=3, all
  sessions bit-identical to solo (12 digests), guard clean. Artifact:
  `results/flashnext/ms-real-parity-integrated-final/verdict.json`.

## MTP production-path gaps CLOSED 2026-11-18 [measured]
- **Fused Metal RoPE**: MTP draft head now calls `rope_apply_fused` (verbatim
  Metal kernel) by default; `VEYSTRIX_CPP_MTP_EAGER_ROPE=1` keeps eager path.
  Bit-exact vs Python PRODUCTION fused kernel at positions 7/11: 0/248320 diff,
  SHA match (`39a6478d...` / `b7958a87...`). Eager path also 0/248320. The old
  ~225220/248320 (~1 ULP) gap is eliminated. Doc + tool:
  `docs/engineering/mtp_fused_rope_parity.md`,
  `check_mtp_fused_rope_parity.py`.
- **Sparse indexer mask (past_len>=2048)**: MTP draft attention calls
  `qsa_full_indexer` + masked SDPA; one-centered indexer norm gain conversion.
  Bit-exact vs Python at past_len=2052: 0/6144 SDPA, 0/2053 mask, 0/248320
  logits (SHA `770afcbe...`); synthetic past_len=24 also 0 diffs. Dense/short
  regressions preserved. Doc + tool: `docs/engineering/mtp_sparse_indexer.md`,
  `check_mtp_sparse_parity.py`.
- Final merged-tree validation: Qwen gate 3/3, Nemotron INT8 gate 3/3, ctests
  37/38, multi-session N=3 end-to-end PASS (`ms-real-parity-merged-final`).

## Blocking decisions / owner inputs
- [BLOCKED] Dropping Python: will do only after the full matrix above is validated.
- [BLOCKED] Nemotron config.json fields (head dims, ssm_state_size, time_step_limit,
  dt_bias/D dtypes) — listed in the port plan's "owner inputs needed".
- Multi-session Phase 1 scope/flags (new sync/thread counts = NEEDS OWNER APPROVAL).
