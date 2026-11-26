# Veystrix terminal tool — QUICKSTART

How to try the terminal tool from this checkout. This file was authored read-only:
every command below was checked with `--help` / offline diagnostics **except**
`veystrix start`, which loads a real model and was deliberately NOT run.
Tags: `MEASURED` = I ran a read-only command; `CITED` = exact file:line in this repo.

## 1. Launch from source (no install / bundle needed)

Use the checked-in venv interpreter; `python -m veystrix` is the same CLI as the
installed console script (`veystrix = veystrix.cli:main`, `pyproject.toml:35`;
`python -m veystrix` resolves via `veystrix/__main__.py:1-5`) [CITED].

    cd /Users/vincent/Documents/veystrix-cli-wt
    .venv/bin/python --version        # Python 3.12.16   [MEASURED]
    .venv/bin/python -m veystrix ...  # same as `veystrix ...`

If the venv is missing/stale, recreate it with `./chat.sh setup` (installs the
shared runtime) [CITED: veystrix/cli.py:174-195].

Two checkpoints are installed under `/Users/vincent/models/` (default discovery
root `~/models`, `veystrix/tool/commands.py:72`) [MEASURED via `models`]:

    flashnext             /Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP          (Qwen3.8)
    nemotron35-lightning  /Users/vincent/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit-mlx-community

`start` accepts `--model {flashnext,nemotron35-lightning}` and `--checkpoint PATH`
(`veystrix/tool/cli.py:38-39`) [CITED].

## 2. Doctor first (safe, offline)

    .venv/bin/python -m veystrix doctor

    ok  python: 3.12.16 [MEASURED]
    ok  macOS >= 26.2 (MLX 0.32.2): 27.2 [CITED (brief 2026-11-18)]
    ok  Apple silicon: arm64 [MEASURED]
    ok  free disk >= 20 GB: 131.6 GiB [MEASURED]
    warn  available memory >= 3 GiB: 2.2 GiB [MEASURED]
    ok  port free: 127.0.0.1:8080 [MEASURED]
    ok  checkpoint present: 2 installed [MEASURED]
    ok  bundle integrity: no installed bundle (source mode) [MEASURED]
    ok  engine selector: selected=python, active=python [MEASURED]

(The per-row labels above are emitted by the tool itself, `commands.py:164-200`.)

## 3. Command sequence: start → status → connect → stop

Subcommand names come from `TOOL_COMMANDS` / `_parser` in
`veystrix/tool/cli.py:24-64` [CITED]; flags were confirmed with each `--help`
[MEASURED]. `start` was NOT run here (it loads a model), so its output is
illustrative; the other three outputs below are MEASURED.

    # 1) start the local engine (background + supervised by default)
    .venv/bin/python -m veystrix start --model flashnext \
        --checkpoint /Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP --port 8080
    # -> JSON with status=running, port=8080, engine=python

    # 2) check status
    .venv/bin/python -m veystrix status --json
    -> {"configured_engine": "python", "engine": "python", "model": null,
        "port": 8080, "status": "stopped"}          # stopped because start was not run

    # 3) connect a harness (writes its provider config; --no-apply previews only)
    .venv/bin/python -m veystrix connect opencode --port 8080 --no-apply
    -> {"harness": "opencode", "result": "ready to update",
        "path": "/Users/vincent/.config/opencode/opencode.json", ...}
    # supported: opencode, claude-code, codex, qwen-code, kimi, hermes,
    #            pi, aider, cline, continue       [MEASURED: `harnesses`]

    # 4) stop
    .venv/bin/python -m veystrix stop
    # -> stop: status=not running

Run a harness without editing its config: `veystrix run <harness> --port 8080`
(`--dry-run` prints the launch command + env only) [CITED cli.py:125-135].
Other useful subcommands: `restart`, `logs -n 100`, `models`, `config get`,
`cache status`, `diagnostics`, `doctor`, `update check` [CITED cli.py:24-29].

## 4. One command, and switching engines

Install the launcher once (puts `veystrix` in `~/.local/bin`, pointing at this checkout):

    scripts/install_cli.sh
    veystrix                    # starts the configured engine if nothing is running, opens the UI,
                                # stops the engine again on exit (config keep-engine-running true to keep it)

Swap engines or models with one command, from the shell or inside the UI:

    veystrix engine             # selected / active engine, and why each one is or is not available
    veystrix engine cpp         # switch to the C++ engine (restarts a running engine)
    veystrix engine python      # and back
    veystrix model nemotron35-lightning
    /engine cpp   /model flashnext      # the same, from inside the UI

- `python` is the default and always available.
- `cpp` runs from this checkout when `cpp/engine/build/libveystrix_c.dylib` exists
  (`cmake -S cpp/engine -B cpp/engine/build -DVEYSTRIX_BUILD_CAPI=ON && cmake --build cpp/engine/build -j`).
  An installed engine bundle takes precedence and keeps its `cpp_parity: pass` gate
  (`docs/engineering/m6_cpp_toggle.md`). A swap that cannot run is refused before anything is stopped or saved.
- Switching is always a restart, never a hot swap.
- Measured on a 16 GB Mac, real text: Python is the faster engine today
  (`docs/performance/ENGINE_AB_2026-11-20.md`).

## 4b. Adding a model

`veystrix models inspect <checkpoint>` reports the architecture primitives a model needs, which engine has them,
and how it would fit on 16/24/32/48/64/96/128 GB Macs. See `docs/engineering/ADDING_A_MODEL.md`.

## 4c. Memory

The expert pool is sized from live free RAM and the model's real expert bytes (`veystrix/memory_budget_runtime.py`),
never above the 5 GiB swap ceiling's safety margins; the 16 GB Mac stays at its validated 2 GB pool cap.
`VEYSTRIX_POOL_CAP_BYTES` caps it explicitly.

## 5. Honest notes

- This checkout is a **source tree in source mode**: no engine bundle is installed
  (doctor: "no installed bundle (source mode)") [MEASURED].
- No source is shipped to the public repo; the public release is binary-only.
  Public home is `joyboyreturns/Veystrix` (`README.md:3-7`) [CITED].
- Bare `veystrix` / `.venv/bin/python -m veystrix` opens the **Veystrix-branded**
  coding-agent TUI (`veystrix/cli.py:358-360`); `veystrix tui --harness <name>`
  is the explicit form. It is a white-label TUI, not a generic shell [CITED].
