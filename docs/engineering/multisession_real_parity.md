# Multi-session REAL-checkpoint parity — end-to-end bit-identity validation

Scope: validate the hard contract rule "each session's per-step output is
bit-identical to that session run SOLO" on the **real** 113 GB
Qwen3.8-Flash-Next MLX-4bit checkpoint, on real hardware, with N concurrent
sessions driven on **one** thread. This is the decisive end-to-end counterpart
of the synthetic CPU fixture in
[multi_session_phase1_runtime.md](multi_session_phase1_runtime.md).

Label legend: `[measured]` = recorded in a guarded live run below; `[cited]` =
from an existing doc/artifact; `[idea]` = design decision; all numeric claims
about the runs are `[measured]`.

## Key insight (why bit-identity is expected, and what was tested)

The real model weights are **read-only and shared** across sessions. Only the
mutable decode state must be per-session:

| Per-session (exclusive) | Shared (read-only) |
|---|---|
| `std::vector<DecoderFullLayerCache>` (GDN conv/recurrent + dense full-attn KV) | `DenseLoader`, `RealLayer` weights, `RealHead`, embed Q4 triplet |
| `std::vector<PLECache>` (n-gram history + conv state) | `StreamingMoE` / `CrossShardMoE` weight bytes (scratch slot buffers written per step, see limitations) |
| position counter, PRNG key/state (`SessionState`) | the single MLX device / GPU-driving thread |
| current logits, step/token counters, digest records | |

Prefill is **exclusive** in v1: each session's prompt is prefilled into its own
caches on the calling thread, one session at a time. Decode steps are then
interleaved by `MsScheduler` (which drives `cooperative_run()` on the same
thread). **Rows are never batched across sessions**: every step is one
independent single-row `run_step()` call, and its logits are fully materialized
(`mx::eval`) before the next session's step runs. No new thread, mutex, I/O
worker or worker-count change was introduced.

## What was added (all in this worktree, branch `ms-real-parity`)

| File | Role |
|---|---|
| `cpp/engine/tools/multisession_real.cpp` | New CLI: `<checkpoint> <num_sessions> <tokens> <config0> <config1> ...`. Loads weights once, per-session prefill, interleaved decode, emits `session=<i> step=<s> token=<t> sha256=<digest>` to stderr. |
| `cpp/engine/tools/run_multisession_parity.py` | OWNER-RUN harness: distinct fixed-seed prompts → N flat configs → N guarded SOLO runs → 1 guarded concurrent run → strict token+digest comparison. |
| `docs/engineering/multisession_real_parity.md` | This document. |

`cpp/engine/CMakeLists.txt` was **not** modified: like
`real_checkpoint_generate`, the tool is built manually against the engine
library (build command below). `real_checkpoint_generate.cpp` and all shared
primitives are **unchanged**.

### Reuse (no numeric code rewritten)

`multisession_real.cpp` uses the include-with-renamed-main pattern of
`probe_embed.cpp`:

```cpp
#define main realgen_main_disabled
#include "real_checkpoint_generate.cpp"
#undef main
```

All helpers live in anonymous namespaces, so this is collision-free. The tool
calls the existing `DenseLoader`, `load_ple`, `run_step`, `embed_token`,
`run_prefill_batched`, `sample_greedy` and `sha256_hex` directly. The
weight-loading (~210 lines) and batched-prefill (~80 lines) orchestration is
copied verbatim from `run_generation()` so the SOLO and CONCURRENT computations
are the same. Notably:

* The batched prefill (not sequential S=1) is reused **per session**, with the
  always-on f32 materialization barriers. This keeps post-prefill cache bytes
  identical to the SOLO reference; a sequential prefill would not (see
  limitations).
* Digest convention is the exact float32-little-endian SHA-256 of the decided
  logit row, identical to `real_checkpoint_generate` / `capture_real_reference.py`.
* Sampling is `sample_greedy` (default). KV tier, MTP, debug hashes and the
  local-GDN/cross-shard-copy A/B flags are left at their defaults in both paths.
* Per-session PLE caches are swapped into the single PLE layer object
  immediately before each `run_step` and copied back after; on one thread with
  one fully-evaluated step at a time this keeps PLE state strictly per-session.

## Build (verified in this worktree)

```bash
cd /Users/vincent/.local/share/opencode/worktree/dd9a0e/ms-real-parity
MLX=/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages/mlx
OBJ=/tmp/msreal_obj && mkdir -p "$OBJ"
xcrun clang -O2 -c models/flashnext/native_read.c -o "$OBJ/native_read.o"
ls cpp/engine/src/*.cpp cpp/decode/src/io_pager.cpp | \
  xargs -P 8 -I{} sh -c 'f="{}"; o="'$OBJ'/$(basename "${f%.cpp}").o"; \
    xcrun clang++ -std=c++20 -O2 -I"'$MLX'/include" -Icpp/engine/src \
    -Icpp/decode/src -c "$f" -o "$o"'
xcrun ar rcs "$OBJ/libvey.a" "$OBJ"/*.o
for tool in real_checkpoint_generate multisession_real; do
  xcrun clang++ -std=c++20 -O2 -I"$MLX/include" -Icpp/engine/src \
    -Icpp/decode/src -Icpp/engine/tools cpp/engine/tools/$tool.cpp \
    "$OBJ/libvey.a" -L"$MLX/lib" -lmlx -Wl,-rpath,"$MLX/lib" \
    -o /tmp/$( [ "$tool" = real_checkpoint_generate ] && echo real_gen_ms || echo multisession_real )
done
```

## Measured validation (guarded, this machine)

All runs used
`scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 5120
--compressor-growth-mib 8192 --thermal-level 2 --max-seconds 900`, from
`/Users/vincent/Documents/Veystrix`. `sysctl vm.swapusage` was checked before
every run (baseline `used = 3403 MiB`, the expected ~3.4 GiB WebKit baseline);
20 s cool-downs separated runs. **Every guard summary had `stopped: null`, exit
0, zero swap growth and thermal level 0 for the whole run.** No run was
stopped or refused.

Harness invocations:

```bash
/Users/vincent/Documents/Veystrix/.venv/bin/python \
  cpp/engine/tools/run_multisession_parity.py \
  --sessions 2 --tokens 3 --prompt-len 16 --budget 900 --cooldown 20
/Users/vincent/Documents/Veystrix/.venv/bin/python \
  cpp/engine/tools/run_multisession_parity.py \
  --sessions 3 --tokens 8 --prompt-len 32 --budget 900 --cooldown 20
```

Artifacts (configs, per-run guard logs and stdout/stderr, `verdict.json`):

* `results/flashnext/ms-real-parity-20261118-101843/`
* `results/flashnext/ms-real-parity-20261118-102002/`

### Run 1 — N=2, 3 tokens, 16-token prompts: **PASS**

```
PASS session 0: 3/3 tokens and digests bit-identical to solo (step0 sha256=b6ea9d2b02fe4155a538232f9aff0abd15fa3518df134eee45a12278acfdf329)
PASS session 1: 3/3 tokens and digests bit-identical to solo (step0 sha256=dfdfbdbf6d55d0032fb5d2751dda9bab8f9e8b9592a08390376b7b0fd9e570ca)
PASS: all 2 sessions bit-identical to SOLO (3 tokens each).
```

Guard resource envelope: solo runs 9.1 s each (available floor 2.91 / 3.36 GiB);
concurrent run 14.2 s (floor 3.12 GiB, compressor peak 4741 MiB, swap unchanged).

### Run 2 — N=3, 8 tokens, 32-token prompts: **PASS**

```
PASS session 0: 8/8 tokens and digests bit-identical to solo (step0 sha256=a4105b123b84300a0981f1182e1f7b08284d5e100e98bf718f2f2924ab38e128)
PASS session 1: 8/8 tokens and digests bit-identical to solo (step0 sha256=6c2f1db9e30621b88e4daef55a7ed2ee029b33080918f2ed244f6e55bea7cffa)
PASS session 2: 8/8 tokens and digests bit-identical to solo (step0 sha256=e1b18c6f0d2eec3d912875d25e38a3e3cb5a75527bdd0c76c8a7d1d48525cf47)
PASS: all 3 sessions bit-identical to SOLO (8 tokens each).
```

Guard resource envelope: solo runs 12.1 / 13.2 / 12.1 s (available floors 3.02 /
3.16 / 2.87 GiB); concurrent run 32.3 s (floor 2.76 GiB, compressor peak 6373
MiB, swap unchanged, thermal level 0 throughout). All 24 per-step digests match
exactly; the full solo-vs-concurrent digest arrays are in `verdict.json`. First
and last digests per session:

| Session | step 0 sha256 (first 16) | step 7 sha256 (first 16) |
|---|---|---|
| 0 | `a4105b123b84300a` | `bb21eabf13046b75` |
| 1 | `6c2f1db9e30621b8` | `c9cca770cecf133e` |
| 2 | `e1b18c6f0d2eec3d` | `6da8ce88fbb12520` |

## Honest limitations

* **This is C++ SOLO vs C++ CONCURRENT parity, not C++ vs Python.** The known
  C++/Python gaps (quantized-vs-dense matmul, PLE, sequential-vs-batched prefill,
  dense-vs-int8 KV, unfused RoPE; see the `real_checkpoint_generate` note) are
  unchanged and out of scope. A PASS here does not claim Python parity.
* **Prefill reuse matters.** The concurrent tool deliberately reuses the
  **batched** prefill per session, matching SOLO. If a future variant prefills
  with sequential S=1 `run_step` calls, its post-prefill cache bytes (and hence
  decode digests) will differ from the batched SOLO reference — that would be a
  known, honest limitation, not a scheduler regression. No such reuse was used
  in the passing runs.
* **Shared expert scratch buffers.** `StreamingMoE` / `CrossShardMoE` keep
  reusable per-layer slot buffers; they are shared across sessions by design
  (weight bytes are read-only). Bit-identity holds because each step fully
  materializes its logits (`mx::eval` in `emit_record` / `sample_greedy`) before
  another session's step can overwrite those buffers. This ordering is a hard
  invariant of the single-thread driver, not an assumption about MLX laziness.
* **PRNG is present but not exercised numerically.** `SessionState` gives each
  session an independent splitmix64 stream that is advanced only by that
  session. `sample_greedy` is deterministic, so the stream does not affect these
  results; multi-session categorical sampling with per-session keys is future
  work and must be validated separately.
* **Scale of the test.** Two configurations were measured (N=2/3, 3/8 tokens,
  16/32-token prompts). More sessions, longer contexts or a KV-tier/MTP-enabled
  configuration would need a fresh guarded run; with KV tier enabled each
  session already gets its own `KVCacheWriter`, but that path was not exercised
  (both SOLO and CONCURRENT used the default KV tier off).
* No numerics, defaults, CMake target or shared source file was changed. No
  commit was made. No guard run was stopped (all `stopped: null`).
