# Using Veystrix: the TUI, the endpoint alone, or the endpoint with any harness

Veystrix serves one local model through one endpoint that speaks the OpenAI API (`/v1/chat/completions`, `/v1/responses`), the
Anthropic API (`/v1/messages`), plus `/health`, `/ready` and `/v1/models`. Reasoning models (Ornith, Flash-Next, Nemotron) return their
`<think>…</think>` text as reasoning content and their tool calls as structured tool calls, on every route. Pick the workflow you like;
they share the same endpoint and settings.

## 1. The TUI

```sh
veystrix                      # starts the configured model if nothing is running, then opens the coding-agent UI
```

Inside: `/model ornith`, `/engine cpp`, `/kv tiered4+`, `/status`, `/endpoint`. When the UI started the endpoint it stops it on exit
unless `veystrix config set keep-engine-running true`; it never stops an endpoint you started yourself.

## 2. The endpoint alone

```sh
veystrix models pull ornith                   # once: ~19.5 GB to ~/models/Ornith-1.5-35B-A3B-MLX-4bit (checks space, resumes, verifies)
veystrix models port-check                    # once per C++ build: offline conformance check that unlocks --engine cpp
veystrix start --model ornith                 # background, supervised, guard on; prints the block below when ready
veystrix start --model ornith --foreground    # same, attached to this terminal (Ctrl-C stops it)
veystrix status                               # engine, runtime (generic/custom), model, KV mode, context
veystrix stop
```

Options: `--context N` (ceiling; Ornith's own is 262,144), `--kv auto|bf16|int8|k8v4|tiered|tiered-k8v4|tiered4|tiered4+` (`auto` lets
the memory plan choose; see `docs/engineering/UNIVERSAL_CPP_ENGINE.md` 11.2), `--port N`, `--host 0.0.0.0` (LAN; refused unless
`veystrix config set require-api-key true`). For other devices prefer Tailscale Serve (`docs/REMOTE_ACCESS.md`), which needs no LAN
listener. When ready, `start` prints:

```
Veystrix endpoint (ready)
  OpenAI base URL:     http://127.0.0.1:8080/v1
  Anthropic base URL:  http://127.0.0.1:8080
  API key:             local  (not required; any value works)
  model id:            veystrix-ornith-<installation hash>   (alias veystrix-ornith)
  model:               Ornith 1.5 35B-A3B
  engine / runtime:    cpp / generic
  KV:                  tiered  (10,880 B/token)
  context ceiling:     262,144
  sampling defaults:   temperature 0.6, top_p 0.95, top_k 20
  smoke test:
    curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -H 'Authorization: Bearer local' -d '…'
```

`veystrix endpoint` prints the same block at any time (running or not) plus the harness snippets; `veystrix endpoint --json` is the
same data for scripts.

## 3. The endpoint with any harness

Configure once, then use the harness exactly as you normally would:

```sh
veystrix connect opencode       # writes the harness's own config (backup + idempotent); also codex, claude-code, qwen-code, pi, kimi, hermes, ...
opencode                        # run the harness yourself, any time the endpoint is up
```

Or launch it pre-wired for one session without touching its config:

```sh
veystrix run claude-code        # sets ANTHROPIC_BASE_URL/…/ANTHROPIC_MODEL for this process only
veystrix run codex -- --help    # arguments after -- go to the harness
```

Neither needs the TUI, and closing a harness never stops the endpoint. `veystrix endpoint` lists, per harness, whether it is installed,
the one-time `connect` command, the `run` command and a direct environment-variable line; and for any other client:

```sh
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1 OPENAI_API_KEY=local OPENAI_MODEL=veystrix-ornith-<hash>
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=local ANTHROPIC_MODEL=veystrix-ornith-<hash>
```

Sampling: Ornith defaults to its model card's temperature 0.6, top-p 0.95, top-k 20; changing sampling in preferences (`/settings`)
overrides them. Per-request sampling fields are listed in `X-Veystrix-Ignored-Fields` (the server fixes sampling per process).

Client quirks the server absorbs: Codex sends a developer message next to its instructions and Claude Code sends a `system` message
after the first user turn; the Qwen-family chat templates accept one system message, first, so leading system messages are joined and a
later one is rendered as a user message in place. `veystrix run claude-code` (and `connect`) set `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to the
server's window, because Claude Code assumes 200k tokens for a model id it does not know. `/v1/models` lists the loaded model first.

Status of this page: the commands and outputs above are covered by offline tests (`veystrix/tests/test_generic_runtime.py`,
`test_harness_adapters.py`, `test_system_message_folding.py`). Real harness matrix (21 Nov 2026, `--context 262144`, guarded, each
harness: turn 1 reads `note.txt` with its file tool and reports the word, turn 2 continues the session and recalls it):

| Harness | Ornith (generic runtime) | Flash-Next (custom runtime) |
|---|---|---|
| opencode | pass / pass | pass / pass (turn 1 220 s) |
| codex | pass / pass | pass / pass (turn 1 193 s) |
| claude-code | pass / pass | not completed: guard stop in turn 1 (see below) |
| qwen-code | pass / pass | not run |
| pi | pass / pass | not run |
| kimi | pass / pass | not run |
| hermes | pass / pass | not run |

Logs: `results/cpp/harness-matrix/`. Flash-Next's first run grew swap to 2856 MiB during Claude Code's first prefill (MLX active peak
9.7 GiB). Cause: the custom C++ prefill kept per-layer debug tensors for all 48 layers. Fixed: a 2048-token chunk now peaks at 6.0 GiB,
12K tokens at 6.6 GiB, swap unchanged, outputs identical. The retry on the fixed build kept swap flat (1674 -> 1610 MiB, available memory
3.56 GiB or more). The run guard still stopped it after 152 s on its sustained-compression rule (compressor 2-4.3 GiB, already 3.1 GiB
while the model loaded). Flash-Next claude-code, qwen-code, pi, kimi and hermes stay unmeasured until another guarded run is approved.

Conversation switching: qwen-code sends a side request with its own system prompt after every turn. The server keeps the other
conversation's C session parked (`VEYSTRIX_CPP_PARKED_SESSIONS`, default 1; `VEYSTRIX_CPP_PARKED_TOKENS`, default 16,384; 0 turns it
off). Nothing is parked while available memory is under `VEYSTRIX_CPP_PARKED_MIN_AVAILABLE_GIB` (default 4), because a parked session
keeps its KV and recurrent state (~20-24 KB per token). The main conversation then resumes without a re-prefill. In the final Ornith run,
qwen-code's turn-2 main request took 7 s; before the fix it re-prefilled for 33 s or more. The rest of that turn (82 s) is qwen-code's own
side agent: three chained requests, which the server already serves as live continuations.

Final Ornith run: `results/cpp/harness-matrix/ornith-262k-final3`, 7/7 harnesses × 2 turns pass, 601 s, swap flat (1570 -> 1490 MiB),
no guard stop. An earlier run of the same code with 2 parked sessions / 32K tokens was stopped by the run guard's sustained-compression
rule during qwen-code (`ornith-262k-final2`). That is why the defaults are now 1 session / 16K tokens with a memory gate.
