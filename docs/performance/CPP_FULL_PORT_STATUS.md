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
| KV tiers (INT8/INT4/tiered4) | lossy KV modes | [TODO] reference gate uses bf16; C++ dense KV is bit-exact for that | NUMERIC |
| MTP / speculative | native draft head | [TODO] | NUMERIC |
| Continuous batching / concurrency | cooperative decode co-step | [TODO] | NON-NUMERIC |
| Expert pool policies (s3fifo/etc) | LRU default + alternatives | [TODO] C++ has LRU slot pool | NON-NUMERIC |
| **Nemotron Mamba layer** | Mamba-2 mixer decode | [DONE-OFFLINE] bit-exact output+conv+ssm state vs real Python mixer | NUMERIC |
| **Nemotron bf16 attention** | dense bf16 KV decode | [DONE-OFFLINE] bit-exact output+q/k/v+cache vs real StreamedAttention (INT8 path still TODO) | NUMERIC |
| **Nemotron 52-layer loop (bf16 KV)** | full decode + final head | [DONE-VALIDATED] real 24 GB checkpoint: 4/4 tokens + 4/4 digests, 3/3 runs | mixed |

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

## Blocking decisions / owner inputs
- [BLOCKED] Dropping Python: will do only after the full matrix above is validated.
- [BLOCKED] Nemotron config.json fields (head dims, ssm_state_size, time_step_limit,
  dt_bias/D dtypes) — listed in the port plan's "owner inputs needed".
- Multi-session Phase 1 scope/flags (new sync/thread counts = NEEDS OWNER APPROVAL).
