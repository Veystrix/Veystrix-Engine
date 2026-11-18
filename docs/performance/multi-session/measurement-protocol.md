# Multi-session Phase 0 — measurement protocol (commands only, OWNER RUN)

**Status: commands only. Nothing in this document was executed by the agent that wrote it.** Every
command below is an **OWNER RUN** command for the owner's test terminal. No git command appears here,
and no measurement, checkpoint load, or guarded run was performed in this task. This file follows
[`docs/measurement-standard.md`](../../measurement-standard.md) (shared JSONL run records, fresh child
processes, at least three arms per condition, forward/reverse ordering, paired effects with a
resolution band) and cites the baselines in
[`docs/performance/handoff-2026-11-14.md` §E](../handoff-2026-11-14.md) (read-only).

**Number labels.** Every number below is tagged `measured` (from an existing run record), `cited`
(from a doc or source in this repo), `estimate` (a planning choice), or `idea` (not built or agreed
here). Numbers in the results template are placeholders to be filled by the owner and are marked
`measured — pending OWNER RUN`.

## Guard, ordering, and opt-in conventions

All commands are wrapped in `scripts/flashnext/real_run_guard.py`
([source](../../../scripts/flashnext/real_run_guard.py)), which samples once per second and stops the
command's process group when any rule is broken: available memory under `--floor-gib` for
`--floor-samples` consecutive samples (default `3` samples, `cited` from the guard source); swap
growth over `--swap-growth-mib` (`5120` MiB = `5` GiB growth cap here, protocol choice, `estimate`);
sustained compressor growth over `--compressor-growth-mib` (`8192` MiB here, `estimate`); thermal
pressure at or above `--thermal-level`; Low Power Mode; low battery; or runtime over `--max-seconds`.
The guard also records the thermal level every sample. These commands raise `--thermal-level` to
`99` (`estimate`) so the guard **logs** thermal level without terminating on it; lower it to `2`
(`cited`: the guard default and the level that stopped long runs in
[`handoff-2026-11-15.md`](../handoff-2026-11-15.md)) to restore a thermal stop. The `--floor-gib 3`
used below matches the harness's own `FLOOR = 3 GiB` (`cited`: `veystrix/bench/qualification.py`) and
the guard's default floor. `--max-seconds 900` / `600` are `estimate` planning budgets, not measured
durations. Guard logs go under `/tmp/multi-session-phase0/...` as required; copy the chosen ones into
the run's `results/` folder if the result is to support a claim (`cited`: measurement-standard).

Arms are interleaved and reversed as in `docs/measurement-standard.md`: condition **A** =
single-session baseline (`dedicated`, `cooperative-1`), condition **B** = multi-session
(`cooperative-2`, `cooperative-4`). A forward pass runs `A B A B` arms in one guarded child process;
the reversed pass runs the same arm list with `--order reversed`, which reverses it to `B A B A` in a
fresh child process (`cited`: `veystrix/bench/concurrency.py` parses `--order given|reversed`).
Repeat the forward/reversed pair until three fresh child processes per condition exist (`cited`:
measurement-standard "at least three arms per condition"); report paired effects and the measured
resolution band (`cited`). All experimental behaviour is **opt-in with defaults unchanged**:
`VEYSTRIX_CONTINUOUS_BATCHING=1`, `--ignore-admission`, `--qwen-prefill-16gb`,
`VEYSTRIX_NEMOTRON_DECODE_PROFILE_DIR`, `FLASHNEXT_DECODE_PROFILE_DIR`, and
`FLASHNEXT_PREFILL_READ_STATS=1` are flags/env for these runs only (`cited`: §E, stage-10, and the
respective source files).

## 1. Baseline — the existing Python (continuous-batching) path

### Evidence to inherit, not re-measure here

- `VEYSTRIX_CONTINUOUS_BATCHING=1` and `veystrix/bench/concurrency.py` exist but were **not run on a
  real checkpoint**: the overhead at one conversation against the `3%` budget, and throughput and
  inter-token tails at two and four conversations, are open (`cited`: §E). The `1`, `2`, `4`
  concurrency points and the `3%` budget are all `cited` from §E.
- Production admission needs more than `4` GiB available beyond a request's need, so on this `16` GiB
  Mac a second conversation is refused with a `503`; the bench's `--ignore-admission` replaces host
  headroom with no limit **for measurement only**, while the harness's own `3` GiB floor still stops
  an arm (`cited`: §E and `concurrency.py`). These are `cited` numbers, not measured in this task.
- The benchmark writes `results/<model>/<stamp>-stage10-concurrency-manual/concurrency.json`
  after every arm (`cited`: `concurrency.py` docstring). Its fields supply the requested evidence:
  `aggregate_tok_s`; per-stream `stream_tok_s`, `gap_p50_ms`, `gap_p95_ms`, `token_sha256` and
  `digests_match_single`; arm `physical_mb_per_token` (bytes from
  `models/flashnext/diskio.disk_bytes_read` over the arm's first-to-last-token interval per emitted
  token, i.e. per aggregate token); `restored_tokens` (for hit rate `restored_tokens / prompt_tokens`);
  `available_floor_bytes`; `swap_growth_bytes` / `swapout_growth_bytes`; plus the guard log's
  `thermal` samples (`cited`: `concurrency.py`, `diskio.py`, guard source).

### OWNER RUN — Nemotron, forward (`A B A B`) then reversed (`B A B A`)

Each block is one guarded fresh child process. Use distinct run directories / guard logs per round;
`$NEMOTRON` is the owner's checkpoint path (`cited` variable convention from existing docs).

```sh
# OWNER RUN — Nemotron, round r1, forward A B A B. Not executed.
VEYSTRIX_CONTINUOUS_BATCHING=1 \
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 900 \
  --log /tmp/msp0-nemotron-baseline-fwd-r1.guard.jsonl -- \
.venv/bin/python -m veystrix.bench.concurrency --model nemotron35 --checkpoint "$NEMOTRON" \
  --prompt-tokens 512 --output-tokens 128 \
  --arms dedicated cooperative-2 cooperative-1 cooperative-4 \
  --ignore-admission
```

```sh
# OWNER RUN — Nemotron, round r2, reversed B A B A. Not executed.
VEYSTRIX_CONTINUOUS_BATCHING=1 \
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 900 \
  --log /tmp/msp0-nemotron-baseline-rev-r2.guard.jsonl -- \
.venv/bin/python -m veystrix.bench.concurrency --model nemotron35 --checkpoint "$NEMOTRON" \
  --prompt-tokens 512 --output-tokens 128 \
  --arms dedicated cooperative-2 cooperative-1 cooperative-4 --order reversed \
  --ignore-admission
```

```sh
# OWNER RUN — Nemotron, round r3, forward A B A B (third fresh child process). Not executed.
VEYSTRIX_CONTINUOUS_BATCHING=1 \
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 900 \
  --log /tmp/msp0-nemotron-baseline-fwd-r3.guard.jsonl -- \
.venv/bin/python -m veystrix.bench.concurrency --model nemotron35 --checkpoint "$NEMOTRON" \
  --prompt-tokens 512 --output-tokens 128 \
  --arms dedicated cooperative-2 cooperative-1 cooperative-4 \
  --ignore-admission
```

### OWNER RUN — Qwen, forward (`A B A B`) then reversed (`B A B A`)

`--qwen-prefill-16gb` is the opt-in low-transient launch path for this `16` GiB host (`cited`:
stage-11 / §E command). Same arm pattern, model `flashnext`.

```sh
# OWNER RUN — Qwen (flashnext), round r1, forward A B A B. Not executed.
VEYSTRIX_CONTINUOUS_BATCHING=1 \
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 900 \
  --log /tmp/msp0-qwen-baseline-fwd-r1.guard.jsonl -- \
.venv/bin/python -m veystrix.bench.concurrency --model flashnext --checkpoint "$QWEN" \
  --prompt-tokens 512 --output-tokens 128 --qwen-prefill-16gb \
  --arms dedicated cooperative-2 cooperative-1 cooperative-4 \
  --ignore-admission
```

```sh
# OWNER RUN — Qwen (flashnext), round r2, reversed B A B A. Not executed.
VEYSTRIX_CONTINUOUS_BATCHING=1 \
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 900 \
  --log /tmp/msp0-qwen-baseline-rev-r2.guard.jsonl -- \
.venv/bin/python -m veystrix.bench.concurrency --model flashnext --checkpoint "$QWEN" \
  --prompt-tokens 512 --output-tokens 128 --qwen-prefill-16gb \
  --arms dedicated cooperative-2 cooperative-1 cooperative-4 --order reversed \
  --ignore-admission
```

```sh
# OWNER RUN — Qwen (flashnext), round r3, forward A B A B (third fresh child process). Not executed.
VEYSTRIX_CONTINUOUS_BATCHING=1 \
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 900 \
  --log /tmp/msp0-qwen-baseline-fwd-r3.guard.jsonl -- \
.venv/bin/python -m veystrix.bench.concurrency --model flashnext --checkpoint "$QWEN" \
  --prompt-tokens 512 --output-tokens 128 --qwen-prefill-16gb \
  --arms dedicated cooperative-2 cooperative-1 cooperative-4 \
  --ignore-admission
```

Record for every arm: the fields listed above, with `physical_mb_per_token` computed from the
`disk_bytes_read` difference (`cited`: `models/flashnext/diskio.py`), and mark each entry with its
condition (`A` or `B`), arm name, run ID, git SHA, and guard log path. An arm that stops at the floor,
at the swap cap, or on digest mismatch is preserved as incomplete/diagnostic, never as a promotion
result (`cited`: measurement-standard).

## 2. Real multi-session route traces for the simulator

Goal: give the offline cache simulator expert routes from `N` concurrent sub-agent-style prompts that
share one system prefix, rather than from synthetic or replayed traces (`idea`: this is the intended
input; no trace was recorded in this task). The concurrency harness's `prompt_variants()` already
builds such prompts — a shared `<|im_start|>system` header with distinct short bodies (`cited`:
`veystrix/bench/concurrency.py`) — so run the multi-session arms with route recording enabled.
`N = 2` and `N = 4` are the `cited` multi-session points from §E.

### Nemotron — `VEYSTRIX_NEMOTRON_DECODE_PROFILE_DIR`

The flag is opt-in; when set, the Nemotron backend writes one
`nemotron-decode-<time_ns>.json` per request into the directory, with per-token routes and timing
(`cited`: `models/nemotron35/decode_profile.py`, `models/flashnext/decode_profile.py`).

```sh
# OWNER RUN — Nemotron multi-session route trace, N=2 and N=4, forward then reversed. Not executed.
VEYSTRIX_CONTINUOUS_BATCHING=1 VEYSTRIX_NEMOTRON_DECODE_PROFILE_DIR=/tmp/multi-session-phase0/nemotron/routes/fwd \
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 900 \
  --log /tmp/msp0-nemotron-routes-fwd.guard.jsonl -- \
.venv/bin/python -m veystrix.bench.concurrency --model nemotron35 --checkpoint "$NEMOTRON" \
  --prompt-tokens 512 --output-tokens 128 \
  --arms cooperative-2 cooperative-4 --ignore-admission

# OWNER RUN (not executed)
VEYSTRIX_CONTINUOUS_BATCHING=1 VEYSTRIX_NEMOTRON_DECODE_PROFILE_DIR=/tmp/multi-session-phase0/nemotron/routes/rev \
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 900 \
  --log /tmp/msp0-nemotron-routes-rev.guard.jsonl -- \
.venv/bin/python -m veystrix.bench.concurrency --model nemotron35 --checkpoint "$NEMOTRON" \
  --prompt-tokens 512 --output-tokens 128 \
  --arms cooperative-2 cooperative-4 --order reversed --ignore-admission
```

The forward block runs `A B` (here `N=2` then `N=4`) and the reversed block runs `B A` — the
forward/reversed pair from `docs/measurement-standard.md` (`cited`). Repeat in fresh child processes
to reach three arms per condition (`cited`).

### Qwen — `bench_route_trace.py`

`models/flashnext/tests/bench/bench_route_trace.py` records real chat-turn expert routes as events
`[turn, phase, token, layer, [experts]]` and writes `route-trace.json.gz`; the simulator reads that
format (`cited`: `bench_route_trace.py`, `bench_read_replay.py`, `cache_sim.py`). Its stock prompts are
`photosynthesis`, `sketchup`, `open` (`cited`: source), and its `--tokens` default is `256` (`cited`).

```sh
# OWNER RUN — Qwen real route trace for the simulator, forward then reversed prompt order. Not executed.
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 600 \
  --log /tmp/msp0-qwen-routes-fwd.guard.jsonl -- \
.venv/bin/python -m models.flashnext.tests.bench.bench_route_trace \
  --tokens 256 --prompts photosynthesis sketchup open

# OWNER RUN (not executed)
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 600 \
  --log /tmp/msp0-qwen-routes-rev.guard.jsonl -- \
.venv/bin/python -m models.flashnext.tests.bench.bench_route_trace \
  --tokens 256 --prompts open sketchup photosynthesis
```

**Caveat (`cited` from source / `idea` for the gap).** `bench_route_trace.py` runs its turns
sequentially in one process; it does not itself expose a concurrent shared-prefix mode. The Nemotron
block above does produce concurrent shared-prefix traces through the concurrency harness. Extending
`bench_route_trace.py` (or its route callback) to emit interleaved `turn` ids for `N` concurrent
shared-prefix prompts is an `idea` for a later task, not something built or measured here. Until
then, record which harness produced each trace in the results table.

## 3. Qwen solo physical MB/token — settling 390 vs 880

- The production figure is **about `390` MB/token physically read from SSD** (`cited`:
  [`docs/MLX/MLX 0.32.2 Metal backend, read against the Flash-N.md`](../../MLX/MLX%200.32.2%20Metal%20backend%2C%20read%20against%20the%20Flash-N.md),
  and `docs/flashnext/research.md`). The `390` figure is `cited`.
- The other figure is **about `880` MB/token of cached expert rows copied out of the page cache**,
  about `37` ms/token in the `2026-09-23` read replay (`cited`: `models/flashnext/expert_pool.py` and
  `docs/flashnext/research.md`). The `880` and `37` figures are `cited`.

Method (`idea`, until the owner runs it): same checkpoint, sampler and prompt, two guarded solo
generations that differ only in output length — `1` token and `N` tokens (`N = 128`, `estimate`) —
and take the difference of the `diskio.disk_bytes_read` totals divided by `N - 1`:

```text
physical_mb_per_decode_token = (N * mb_per_token(N) - 1 * mb_per_token(1)) / (N - 1)
```

where `mb_per_token(k)` is `qualification`'s `physical_mb_per_token` from
`disk_bytes_read` over the request (`cited`: `veystrix/bench/qualification.py`,
`models/flashnext/diskio.py`). A result near `390` (`cited`) supports the physical-read figure for
solo decode; a result near `880` (`cited`) shows the read meter is counting page-cache copies. The
outcome is `measured — pending OWNER RUN`; neither value is asserted by this document.

`--set-env FLASHNEXT_PREFILL_READ_STATS=1` exposes the prefill byte counter used by the §E command
(`cited`: §E). Interleave as `A B A B` then `B A B A` with `A = 1` token, `B = N` tokens, fresh child
processes, with each cell writing to its own default `results/` run directory (`cited`: measurement-standard).

```sh
# OWNER RUN — Qwen solo physical MB/token, round r1, forward A(1) B(128) A(1) B(128). Not executed.
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 600 \
  --log /tmp/msp0-qwen-solo-physical-a1-fwd.guard.jsonl -- \
.venv/bin/python -m veystrix.bench.qualification --model flashnext --checkpoint "$QWEN" \
  --prompt-tokens 512 --output-tokens 1 --mode cold-process --qwen-prefill-16gb \
  --set-env FLASHNEXT_PREFILL_READ_STATS=1

# OWNER RUN (not executed)
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 600 \
  --log /tmp/msp0-qwen-solo-physical-b128-fwd.guard.jsonl -- \
.venv/bin/python -m veystrix.bench.qualification --model flashnext --checkpoint "$QWEN" \
  --prompt-tokens 512 --output-tokens 128 --mode cold-process --qwen-prefill-16gb \
  --set-env FLASHNEXT_PREFILL_READ_STATS=1

# OWNER RUN (not executed)
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 600 \
  --log /tmp/msp0-qwen-solo-physical-a1-fwd-r2.guard.jsonl -- \
.venv/bin/python -m veystrix.bench.qualification --model flashnext --checkpoint "$QWEN" \
  --prompt-tokens 512 --output-tokens 1 --mode cold-process --qwen-prefill-16gb \
  --set-env FLASHNEXT_PREFILL_READ_STATS=1

# OWNER RUN (not executed)
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 600 \
  --log /tmp/msp0-qwen-solo-physical-b128-fwd-r2.guard.jsonl -- \
.venv/bin/python -m veystrix.bench.qualification --model flashnext --checkpoint "$QWEN" \
  --prompt-tokens 512 --output-tokens 128 --mode cold-process --qwen-prefill-16gb \
  --set-env FLASHNEXT_PREFILL_READ_STATS=1
```

```sh
# OWNER RUN — Qwen solo physical MB/token, round r2, reversed B(128) A(1) B(128) A(1). Not executed.
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 600 \
  --log /tmp/msp0-qwen-solo-physical-b128-rev.guard.jsonl -- \
.venv/bin/python -m veystrix.bench.qualification --model flashnext --checkpoint "$QWEN" \
  --prompt-tokens 512 --output-tokens 128 --mode cold-process --qwen-prefill-16gb \
  --set-env FLASHNEXT_PREFILL_READ_STATS=1

# OWNER RUN (not executed)
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 600 \
  --log /tmp/msp0-qwen-solo-physical-a1-rev.guard.jsonl -- \
.venv/bin/python -m veystrix.bench.qualification --model flashnext --checkpoint "$QWEN" \
  --prompt-tokens 512 --output-tokens 1 --mode cold-process --qwen-prefill-16gb \
  --set-env FLASHNEXT_PREFILL_READ_STATS=1

# OWNER RUN (not executed)
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 600 \
  --log /tmp/msp0-qwen-solo-physical-b128-rev-r2.guard.jsonl -- \
.venv/bin/python -m veystrix.bench.qualification --model flashnext --checkpoint "$QWEN" \
  --prompt-tokens 512 --output-tokens 128 --mode cold-process --qwen-prefill-16gb \
  --set-env FLASHNEXT_PREFILL_READ_STATS=1

# OWNER RUN (not executed)
.venv/bin/python scripts/flashnext/real_run_guard.py --floor-gib 3 --swap-growth-mib 5120 \
  --compressor-growth-mib 8192 --thermal-level 99 --max-seconds 600 \
  --log /tmp/msp0-qwen-solo-physical-a1-rev-r2.guard.jsonl -- \
.venv/bin/python -m veystrix.bench.qualification --model flashnext --checkpoint "$QWEN" \
  --prompt-tokens 512 --output-tokens 1 --mode cold-process --qwen-prefill-16gb \
  --set-env FLASHNEXT_PREFILL_READ_STATS=1
```

The pair above is `A B A B` then `B A B A`; append one more fresh-process repetition per cell to
reach three arms per condition (`cited`: measurement-standard). Report the paired difference and its
resolution band (`cited`).

## 4. Results template

One row per guarded arm. All value cells are `measured — pending OWNER RUN`; reference figures are
`cited`. The `Label` column carries the required tag for each row's numbers. All fields from task
items 1–3 are present, plus run ID, git SHA, and guard log path.

| Run ID | git SHA | Guard log path | Label | Model | Source (task item) | Order | Cond. / arm | Concurrency (N) | Aggregate tok/s | Per-session tok/s (s1 / s2 / s3 / s4) | Inter-token p50 ms | Inter-token p95 ms | Physical MB / aggregate token | Hit rate | Memory floor (GiB) | Swap growth (MiB) | Thermal level (max) | Digest vs solo | Route trace path | Solo physical MB/token (390 vs 880) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TBD | TBD | TBD | measured — pending OWNER RUN | nemotron35 | 1 — baseline | fwd `A B A B` | A / dedicated | 1 (`cited`) | TBD | TBD / — / — / — | TBD | TBD | TBD | TBD | TBD | TBD | TBD | reference (self) | — | — |
| TBD | TBD | TBD | measured — pending OWNER RUN | nemotron35 | 1 — baseline | fwd `A B A B` | B / cooperative-2 | 2 (`cited`) | TBD | TBD / TBD / — / — | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD (vs solo) | — | — |
| TBD | TBD | TBD | measured — pending OWNER RUN | nemotron35 | 1 — baseline | fwd `A B A B` | B / cooperative-4 | 4 (`cited`) | TBD | TBD / TBD / TBD / TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD (vs solo) | — | — |
| TBD | TBD | TBD | measured — pending OWNER RUN | flashnext | 1 — baseline | rev `B A B A` | B / cooperative-2 | 2 (`cited`) | TBD | TBD / TBD / — / — | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD (vs solo) | — | — |
| TBD | TBD | TBD | measured — pending OWNER RUN | nemotron35 | 2 — routes | fwd `A B` | cooperative-2/4 | 2, 4 (`cited`) | — | — | — | — | — | — | TBD | TBD | TBD | — | TBD (`nemotron-decode-*.json`) | — |
| TBD | TBD | TBD | measured — pending OWNER RUN | flashnext | 2 — routes | rev `B A` | route-trace turns | 3 prompts (`cited`) | — | — | — | — | — | — | TBD | TBD | TBD | — | TBD (`route-trace.json.gz`) | — |
| TBD | TBD | TBD | measured — pending OWNER RUN | flashnext | 3 — solo physical | fwd `A B A B` / rev `B A B A` | 1-token vs 128-token | 1 | — | TBD (1 tok) | — | — | TBD | — | TBD | TBD | TBD | — | — | TBD (near `390` or `880`, both `cited`) |

Fill every `TBD` only from the arm record (`concurrency.json` / `qualification.json`) and the matching
guard JSONL; keep incomplete or stopped runs in the file marked diagnostic (`cited`:
measurement-standard). The `2` and `4` concurrency values, `512`/`128` prompt/output token choices
in the commands, and the `1`/`128` solo output lengths are `cited` (§E and existing harness defaults)
or `estimate` (the `128` solo length); none were measured in this task.

## Appendix — what this document did not do

- No command in this document was executed; there are no measured numbers here. Every command is
  marked `OWNER RUN` and wrapped in `scripts/flashnext/real_run_guard.py` with the agreed flags
  (`--floor-gib 3 --swap-growth-mib 5120 --compressor-growth-mib 8192 --thermal-level 99
  --max-seconds <budget> --log /tmp/... -- <command>`).
- No git operation of any kind was performed; no files outside this protocol document were created
  or modified by this task.
- The `3%` overhead budget, `4` GiB admission headroom, `16` GiB host, `390`/`880` MB/token figures,
  and `2`/`4`-session baselines are `cited`, not measured here. The guard thresholds, timeouts,
  `N = 128` solo length, and the shared-prefix simulator extension are `estimate` or `idea`.
