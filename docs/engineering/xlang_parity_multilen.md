# Cross-language parity on FRESH prompts at multiple lengths

Status: **measured** (2026-11-18). All figures below are labelled
**measured**, **estimate**, or **cited** at first use. Bottom line: the
reviewer's premise is **disproven** — every length is bit-exact on every logit
row Python actually computed.

## Premise under test

A reviewer claimed the integrated C++ real-checkpoint generator matches Python
only on the captured gate reference (`results/flashnext/20261117-222244-real-reference/reference.json`,
`prompt_tokens=16`) and diverges on arbitrary real prompts. This experiment
tests that claim directly: capture **fresh** Python references at six distinct
prompt lengths and compare C++ greedy token ids **and** per-step logit SHA-256
digests bit-for-bit.

Each length N is a *different* prompt: `capture_real_reference.py` uses fixed
seed `20261117` but calls `rng.integers(0, 200000, size=N)`, so the token stream
(and its SHA-256) changes with N. N=16 reproduces the known gate prompt
(`prompt_sha256=f01215c52941e19b67a00ccaec1fb9b42bab3dfd7108be26222812dc7af33104`).

Bit-exactness claim: identical greedy token ids **and** identical
float32-little-endian SHA-256 of each decided logit row, for every requested
step. Nothing is relaxed to "tokens only" or a tolerance.

## Build (measured)

`real_checkpoint_generate` is not a CMake target; it is compiled manually
against the integrated `veystrix_engine` sources (all 31 `cpp/engine/src/*.cpp`
plus `cpp/decode/src/io_pager.cpp`, `models/flashnext/native_read.c`), with MLX
pinned to the venv build:

```
MLX=/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages/mlx
xcrun clang -O2 -c models/flashnext/native_read.c -o /tmp/native_read.o
xcrun clang++ -std=c++20 -O2 -I"$MLX/include" -Icpp/engine/src -Icpp/decode/src \
  <all src/*.cpp> cpp/decode/src/io_pager.cpp /tmp/native_read.o \
  cpp/engine/tools/real_checkpoint_generate.cpp \
  -L"$MLX/lib" -lmlx -Wl,-rpath,"$MLX/lib" \
  -o cpp/engine/build/real_checkpoint_generate
```

The integrated source already contains the batched **sorted** `gather_qmm`
prefill path (`cpp/engine/src/engine.cpp:114-140`), which prior work
(`docs/engineering/q1024_divergence_diagnosis.md`) identified as the fix for the
S>=128 drift seen on an earlier branch. The build here is from the
`xlang-parity` worktree (off `ms-phase1-integration`), so this measurement is of
the *integrated* code, not the earlier divergent build.

## Method (measured)

For each N in {16, 64, 128, 256, 512, 1024}, with `--tokens 4`:

1. `capture_real_reference.py --tokens 4 --prompt-tokens N` under
   `scripts/flashnext/real_run_guard.py` from `/Users/vincent/Documents/Veystrix`.
2. `gen_real_config.py` from the fresh reference (JSON-only; no tensor load,
   no guard needed). It regenerates the prompt from seed 20261117 and verifies
   its SHA-256 against the reference before writing the flat config.
3. `real_checkpoint_generate <ckpt> <reference.json> 4 <config>` under the same
   guard; stdout step lines captured to disk.
4. In-process bit-for-bit comparison of token ids and digests (the harness
   deliberately does **not** call `check_real_parity.py`, which would re-run the
   binary outside the guard).

Guard flags (each real-checkpoint run):
`--floor-gib 1.5 --swap-growth-mib 5120 --compressor-growth-mib 8192
--thermal-level 2 --max-seconds 600`. `sysctl vm.swapusage` is sampled before
every run; 18 s cool-downs separate consecutive guarded runs. The harness
aborts on any non-zero guard exit or non-null `stopped`.

Orchestration: `cpp/engine/tools/check_xlang_multilen.py`. Artifacts:
`results/flashnext/xlang_multilen_20261118/len_*/{reference.json,real_config.txt,py_guard.jsonl,cpp_guard.jsonl,cpp_stdout.txt,verdict.json}`
and `aggregate.json`.

Pre-run swap baseline (measured, `sysctl vm.swapusage`): `used = 3395 MiB`,
the expected ~3.4 GiB WebKit baseline.

## Results (measured)

Six distinct prompts, `--tokens 4`, integrated binary. Every guard summary had
`stopped: null`, exit 0, thermal level 0 for the whole run, and zero or modest
swap growth (largest: N=512 Python, 3387→5616 MiB, +2229 MiB, well under the
5120 MiB guard threshold). No run was refused or stopped.

| N | prompt sha256 (first 16) | Python finish | C++ steps emitted | tokens (C++=Python) | digests (C++=Python) | strict gate | bit-exact compute |
|---|---|---|---|---|---|---|---|
| 16 | `f01215c52941e19b` | length | 4 | 4/4 | 4/4 | PASS | PASS |
| 64 | `e2d8b36b7e407941` | length | 4 | 4/4 | 4/4 | PASS | PASS |
| 128 | `5c9d73b2a0e9a9ff` | stop | 4 (Python emitted 1) | 1/1 | 1/1, plus unpaired digest verified | FAIL (count only) | PASS |
| 256 | `2175c9c0486709fd` | length | 4 | 4/4 | 4/4 | PASS | PASS |
| 512 | `036e507968f8446c` | length | 4 | 4/4 | 4/4 | PASS | PASS |
| 1024 | `424b1903f729d9e1` | length | 4 | 4/4 | 4/4 | PASS | PASS |

Strict gate = count must match and every emitted step's token and digest must
match (the semantics of `check_real_parity.py`). Bit-exact compute = every
logit row Python actually evaluated has an identical float32-LE SHA-256 on the
C++ side; a Python early stop on EOS is not a numeric divergence.

### Exact digests per length (measured)

Each line is `step — token — sha256` from the C++ stdout; the Python reference
step tokens and digests are byte-identical for every step listed.

**N=16** (`prompt_sha256=f01215c52941e19b67a00ccaec1fb9b42bab3dfd7108be26222812dc7af33104`)
```
0  271    1638ab11b2debcebf8b8fbe30138766154e28d25d3479d92c7c3dfbac2196102
1  248068 9c3a9173ade7397c434ff35fab68a193aeec4de76a824def2451713652aa1319
2  198    4d5d49e8cef4b6cf06bc0a41f3c42c0a11e5a16920711fa48ebd634eac714a00
3  760    f41be938282d135a6d54969da7e999129d2efd01f21623e0ca4057c1cf724179
```
This reproduces the known gate reference digests exactly (cited:
`results/flashnext/20261117-222244-real-reference/reference.json`).

**N=64** (`prompt_sha256=e2d8b36b7e40794139e5976ee88bb5ec6861f73afbdbefc7586e810c2662fa53`)
```
0  271     c7d07880100a2096dbcbb4ccc27972d3e84a157b0dd543e905900eb9cbf4d85b
1  104980  3403ad3a793b468e378381e023c95cb09237d7db5478ed327cd019b3888faeae
2  99943   ffa2c5cb130486dc7131b20c1bdfbe7d101b9f3ea7735b94f42de15ab4ff1e0b
3  103916  8fc481dfac374ca6405bb1ac423a25b2eb6775cdbc33df77069bd8d3f157df1a
```

**N=128** (`prompt_sha256=5c9d73b2a0e9a9ffa6b08ce1f5610498472eec23c47de88844e140a7042333f9`)
```
0  271      c6f51dae37658d7bc3b2f8a05dd239a282b8cbc8a4d47d4d89dc7feaf11fe9bc
1  248046   5ab68c5caf814a55d5a664cffad7c3734e91b284b3cdd4b8f7c5478e7483000b
2  198      52fe57896fce31cf53f49f8f5a95e00d2ca926bfc86a3b9ced3612ca17203308
3  248045   aa1bc43c9798e96ea8be949fa911bfa557f8d8197ab0d7412b5dfe60a2db8888
```
CITED contrast: the earlier branch (`docs/engineering/q1024_divergence_diagnosis.md:22-26`)
reported C++ step 0 = `5dbb28ccbd278529ece9716189f9aa4c46d921174fe9642adde677495a0dc82c`
vs Python `c6f51dae…`. Here C++ step 0 is `c6f51dae…` — **bit-exact**. The
integrated sorted-`gather_qmm` path fixed that drift.

**N=256** (`prompt_sha256=2175c9c0486709fddaf856062b21faf43e17c14c74dcd0d80ca5793a0060e741`)
```
0  271     a010ff0d9b5ff8871dbb069f894fb4adb1b8661f4b7541f80201da58bdcfff83
1  248068  02a8690e31eca53f6bd06c53206a9570dca075afdd1e4e801dbad5c8067bfe87
2  271     5d802cf69a06ec21038276701178dee45ef56c2527dec21ea146e9512047a3c9
3  248069  aa7633af2fc164383dbe432eaff32212de19631c30657736768ffddda1532c77
```

**N=512** (`prompt_sha256=036e507968f8446c37bf657e1cd7f40c1d892485a25d1852af6754a38744c37f`)
```
0  271     0c0e6daf4ed0669b458b42b8a36f7fbad768177171d977e72ede46ad193c83fe
1  104980  f1f2b218c409e08805286e88b72ab8e7afc27bfe71c62577c2e3bf9d9e89ad46
2  109120  8ff2eeeb7faa373810576f51de2657ea99ad340467d36b96b8a3f839eacab19f
3  103916  0345abae546880e85b40bcefa68c4535bf5698a9c8e4e660411d0a49c2a8109d
```

**N=1024** (`prompt_sha256=424b1903f729d9e1228f55db060c5e516d3286d439191136c6872e1d803e5e9e`)
```
0  271     269821a88a272923e0b6af83adcd806a4bd57e597f6984622659620f1305f1d7
1  96220   ff96a476f70baf1f7df3a765b845fff6ad2681b240280f3998620f7b35c331ad
2  98920   b34a4a1c5307375e6fe7d6dee607e5483b50825e82a62151f6a5fc647163209a
3  96810   26fd39cffb91476b3cd3bb1ca9ab4aa99813c5959fca1db2f4a207e4cdeb4ab5
```

## N=128 early stop — classification (measured)

The strict gate reports FAIL at N=128 only because of a **step count**
mismatch, not a digest mismatch: Python emitted 1 token (`finish="stop"`), C++
emitted 4. The single paired step is bit-exact (token 271, digest
`c6f51dae…`). Cause, measured from the checkpoint's `generation_config.json`
(`eos_token_id = [248046, 248044]`): C++'s step-1 token is **248046**, an EOS
id. Python's sampler sampled 248046 at step 1, hit EOS, and did not emit it;
the C++ generator does not implement EOS stopping and emitted it, then
continued.

To rule out a numeric divergence at the stop step, a results-local derived
capture (`results/flashnext/xlang_multilen_20261118/_capture_all_digests.py`,
one added output line on `capture_real_reference.py`) recorded **all** logit
digests Python evaluated at N=128 (guarded run, `stopped=null`):

| step | Python logit sha256 | C++ logit sha256 | match |
|---|---|---|---|
| 0 | `c6f51dae37658d7bc3b2f8a05dd239a282b8cbc8a4d47d4d89dc7feaf11fe9bc` | same | BIT-EXACT |
| 1 | `5ab68c5caf814a55d5a664cffad7c3734e91b284b3cdd4b8f7c5478e7483000b` | same | BIT-EXACT |

Steps 2-3 have no Python counterpart (Python had already stopped), so they
cannot diverge. The harness `--eos-aware` mode reproduces this classification
automatically: measured `strict=False bit_exact=True eos_early_stop=True
(all py digests 2/2)` in
`results/flashnext/xlang_multilen_20261118_eosverify/aggregate_eos_aware.json`.

## Localization (if any divergence)

**No numeric divergence was found at any length**, so no first-divergent
layer/op localization was required. The only non-PASS was the N=128 EOS stop
condition, localized above to `generation_config.json`'s EOS set and C++'s
lack of EOS stopping — not to GDN, RoPE, PLE, the prefill kernel, or any
numeric kernel. Diagnostic tooling was prepared and verified regardless:
`REALGEN_DUMP_BATCH_LAYERS` in `real_checkpoint_generate.cpp:1998-2040` and
`dump_python_layers.py --prompt-tokens N` (self-test passed), with a comparison
helper at `/tmp/xlang_compare_layers.py`.

## Conclusion (measured)

The reviewer's premise — that the integrated C++ generator is bit-exact only
on the captured gate reference and diverges on arbitrary real prompts — is
**disproven**. On six *fresh, distinct* prompts spanning 16 to 1024 tokens, the
C++ generator reproduced Python's greedy token ids and every per-step
float32-little-endian logit SHA-256 bit-for-bit. This includes N=128, the exact
length and the exact step-0 digest on which an earlier branch diverged; the
integrated sorted-`gather_qmm` prefill path (`cpp/engine/src/engine.cpp:114-140`,
cited from `docs/engineering/q1024_divergence_diagnosis.md`) closed that gap.

The single caveat is behavioral, not numeric: C++ does not stop generation on
EOS, so at N=128 it emits four tokens where Python emits one. This is a
stop-condition difference (tracked as a known gap) and does not affect the
bit-exactness of the computed logits or tokens.

## Cited references

- `cpp/engine/tools/capture_real_reference.py` — deterministic prompt +
  logit-digest capture (EFFECTIVE_CONFIG_SOURCES, lines 37-61).
- `cpp/engine/tools/gen_real_config.py` — prompt regeneration / SHA check
  (lines 96-104).
- `cpp/engine/tools/check_real_parity.py` — strict step-line comparison format
  (LINE_RE, line 27).
- `docs/engineering/q1024_divergence_diagnosis.md` — earlier-branch S-dependence
  root cause (`SwitchGLU.__call__` `_gather_sort` + `sorted_indices=True`).
- `docs/performance/CPP_PARITY_GAPS.md` — the six documented header gaps.
