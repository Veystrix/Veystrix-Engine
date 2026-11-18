# Non-greedy real-checkpoint parity — C++ sampler vs `veystrix.sampling.Sampler`

Status: **PASS (measured)** — both the isolated sampler test and the full
end-to-end sampled decode match token-for-token (and digest-for-digest on the
e2e arm) for THINKING and INSTRUCT at seed 7 on the real Qwen3.8-Flash-Next
checkpoint. No files outside the three allowed were edited; nothing was committed.

Throughout, **MEASURED** = observed in this worktree on 2026-11-18,
**CITED** = read from a file/line in the repo, **ESTIMATE** = inference.

## 1. Claim and scope

`veystrix::engine::sample_non_greedy` (`cpp/engine/src/sampling.cpp`) must return
the same token id as `veystrix.sampling.Sampler.__call__` (non-greedy branch,
`veystrix/sampling.py:172-185`) for the same logit row, the same preset and the
same seed. `sampling.{hpp,cpp}` is unchanged: it was already bit-exact against
Python for 11 synthetic cases. Only the real generator's decode loop, a new
checker and this doc were added.

## 2. What was added

### 2.1 Opt-in env in `cpp/engine/tools/real_checkpoint_generate.cpp`

In `run_generation`, before the decode loop:

| env | effect |
| --- | --- |
| `VEYSTRIX_CPP_SAMPLING=thinking\|instruct` | off by default; when set, use `sample_non_greedy(decided, params, seen_tokens)` instead of `sample_greedy(decided)` |
| `VEYSTRIX_CPP_SAMPLING_SEED=<int>` | required when sampling is on; seeds the MLX default key once with `mx::random::seed(seed)` |

Presets mirror `THINKING` / `INSTRUCT` in `veystrix/sampling.py:29-43`:

- thinking: temp 1.0, top_p 0.95, top_k 20, min_p 0.0, presence 0.0
- instruct: temp 0.7, top_p 0.80, top_k 20, min_p 0.0, presence 1.5

`seen_tokens` is appended at the sampling boundary when presence ≠ 0, mirroring
`Sampler._note_sampled` (`veystrix/sampling.py:104-115`). A stderr banner prints
the effective mode/seed/params.

**RNG investigation (MEASURED + CITED).** Python does
`mx.random.seed(seed)` (`veystrix/session.py:545-549`) and then
`mx.random.categorical(masked)` with **no** key (`sampling.py:183`); the default
`KeySequence` splits and advances its key on each draw (`mlx/random.h:12-33`,
`categorical` overloads `:220-234`). The C++ mirror is therefore
`mx::random::seed(seed)` once + `sample_non_greedy(..., key = nullopt)` per step
(`sampling.cpp:89-99`). Passing `mx::random::key(seed)` as an *explicit* key on
every step is **wrong**: an explicit key is stateless, so the same categorical
draw repeats. MEASURED with mlx 0.32.2: `seed 7` + default key draws
`[0, 3, 1, 1, 1, 0]` on a fixed 4-way row, while the same explicit key repeated
gave `[3, 3, 3]`. This is why the flag seeds once and uses the default key.

### 2.2 Build change (required, documented)

The historical standalone build command
(`research_notes/Cerebras inference and IO splitting/real_checkpoint_generate.md:52-66`)
omits `sampling.cpp` and other newer engine sources. The opt-in call needs
`cpp/engine/src/sampling.cpp` linked in. `check_nongreedy_parity.py::build_binary`
compiles **all** `cpp/engine/src/*.cpp` except `decoder.cpp` and
`model_adapter.cpp` (the exclusions cited in
`docs/performance/OPUS_PROMPT_CPP_FINAL.md:32`), plus `io_pager.cpp`,
`native_read.o` and the tool. No build file in the repo was changed.

### 2.3 New checker `cpp/engine/tools/check_nongreedy_parity.py`

Two arms:

1. **`--mode sampler` (decisive, isolates the sampler).** A guarded greedy real
   run dumps raw per-step logits with `REALGEN_DUMP_LOGITS` (existing hook at
   `real_checkpoint_generate.cpp:2141-2146`). Those exact float32 rows (vocab
   248320, CITED from `real_config.txt`) are replayed through Python `Sampler`
   and through a tiny C++ driver compiled from stdin against `sampling.cpp`
   (no repo source added). Identical input ⇒ any difference is the sampler.
2. **`--mode e2e` (whole decode).** C++ real gen with the two env vars vs the
   Python FlashNext backend with the same preset and `mx.random.seed(seed)`.
   Both use prompt seed 20261117 / 16 prompt tokens (CITED from
   `capture_real_reference.py:183-184`). The checker clears `backend.stops`
   because the C++ tool has no stop logic (it always emits N tokens) while the
   chat backend stops on EOS; a sampled EOS at step 1 would otherwise truncate
   Python and create a fake diff. This is a harness alignment, not a weakening.

Every real-checkpoint run is wrapped in
`scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 5120
--compressor-growth-mib 8192 --thermal-level 2 --max-seconds <budget>` from
`/Users/vincent/Documents/Veystrix`, as required. Preflight swap was checked
before each batch (baseline 3395–4106 MiB, within the stated ~3.4 GiB user
baseline). **All guard logs ended with `"stopped": null`** (MEASURED); no
cool-down-triggering stop occurred.

## 3. Results (all MEASURED on 2026-11-18, mlx 0.32.2)

### 3.1 Engine-logit sanity check

The guarded greedy C++ run on the current branch matched the stored Python
reference `results/flashnext/20261117-222244-real-reference/reference.json`
**8/8 logit SHA-256 digests and 8/8 tokens** (MEASURED). Note: the older
`research_notes/.../real_checkpoint_generate.md:86-99` reports 0/8 for an
earlier state; that note is stale for this branch. Because the logits agree,
an end-to-end sampled comparison is meaningful here.

### 3.2 Sampler parity on identical real logits (8 steps, seed 7)

| preset | Python sampled tokens | C++ sampled tokens | result |
| --- | --- | --- | --- |
| thinking | `[271, 248044, 198, 760, 1156, 579, 1876, 7701]` | identical | **PASS** (0 diffs) |
| instruct | `[271, 248044, 198, 760, 1156, 579, 1876, 7701]` | identical | **PASS** (0 diffs) |

Extra multi-seed replay on the same real logits (MEASURED, 5 seeds × 2 presets,
all 0 diffs). Examples showing genuinely different draws, not just argmax:

| seed | preset | sampled tokens (Python == C++) |
| --- | --- | --- |
| 1 | thinking / instruct | `[220, 248045, 198, 760, 1156, 579, 1876, 7701]` |
| 42 | thinking | `[271, 220, 271, 760, 1156, 579, 1876, 7701]` |
| 42 | instruct | `[271, 220, 198, 760, 1156, 579, 1876, 7701]` |
| 123 | thinking / instruct | `[271, 96220, 198, 760, 1156, 579, 1876, 7701]` |
| 2026 | thinking / instruct | `[271, 248068, 198, 760, 1156, 579, 1876, 7701]` |

### 3.3 End-to-end sampled decode (both presets, seed 7, 8 tokens)

Both arms produced the **exact same sampled token sequence** and the **exact
same 8 logit digests**. The sequence differs from greedy from step 1 onward,
confirming real sampling was exercised (greedy step 1 is 248068, not 248044):

```
C++ sampled:    [271, 248044, 248045, 271, 248068, 198, 760, 1156]
Python sampled: [271, 248044, 248045, 271, 248068, 198, 760, 1156]
token sequence: MATCH (8/8)
logit digests:  MATCH (8/8)
```

Full digest row (identical on both sides, MEASURED):
`1638ab11b2de…`, `9c3a9173ade7…`, `717c25450d47…`, `2031e8e9aa31…`,
`4017d077e2f2…`, `42eb26ff2da4…`, `af4250a7009f…`, `cfbd10707f38…`.

C++ stderr confirmed the preset was applied, e.g.
`[sampling] non-greedy mode=instruct seed=7 temp=0.7 top_p=0.8 top_k=20 min_p=0 presence=1.5`;
the Python report recorded the matching `Sampling` fields. Guard summaries:
`"stopped": null` on all four real runs; available-memory floors 1.65–4.81 GiB
(above the 1.5 GiB floor), swap peaks within +5120 MiB of baseline, thermal 0.

## 4. Honest limitations

- **MEASURED** horizon is 8 tokens; determinism is argued from the shared
  default-key PRNG, not from a long-run statistical test.
- **ESTIMATE**: the isolated sampler arm is the stronger evidence because it
  removes engine-logit coupling; the e2e arm additionally depends on both
  engines' logits agreeing, which happens to hold on this branch for this prompt.
- **CITED**: the chat normally stops on EOS (`flashnext.py:239-243,1278`);
  the checker deliberately disables stops only for the e2e comparison.
- The `VEYSTRIX_CPP_SAMPLING` path is mutually exclusive with the MTP decode
  path (`mtp.enabled && hidden_ready` branch at `real_checkpoint_generate.cpp:2169`);
  MTP was off in every run here.

## 5. Reproduce

```
cd /Users/vincent/Documents/Veystrix
.venv/bin/python <worktree>/cpp/engine/tools/check_nongreedy_parity.py \
  --mode sampler --preset thinking --seed 7 --tokens 8 \
  --budget 900 --outdir <scratch>/nongreedy-parity-thinking
.venv/bin/python <worktree>/cpp/engine/tools/check_nongreedy_parity.py \
  --mode sampler --preset instruct --seed 7 --tokens 8 \
  --budget 900 --outdir <scratch>/nongreedy-parity-instruct \
  --binary <scratch>/nongreedy-parity-thinking/bin/real_gen_sampling \
  --reuse-logits <scratch>/nongreedy-parity-thinking/logits
.venv/bin/python <worktree>/cpp/engine/tools/check_nongreedy_parity.py \
  --mode e2e --preset thinking --seed 7 --tokens 8 \
  --budget 900 --outdir <scratch>/nongreedy-e2e-thinking \
  --binary <scratch>/nongreedy-parity-thinking/bin/real_gen_sampling
```
