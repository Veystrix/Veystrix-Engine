# Harness Integration Guide

Veystrix is engineered to power local agentic workflows on Apple Silicon Macs. You can use Veystrix through its built-in interactive terminal UI (`direct`), drive it with curated external coding-agent harnesses, or connect any tool that speaks the OpenAI or Anthropic API protocols to its local loopback endpoints.

---

## Supported Harnesses

In this commercial release, external harness integrations are curated and locked to provide maximum reliability, strict session KV cache stability, and zero configuration overhead:

| Harness | Mode / Command | Description & Capabilities | Auto-Start |
|:---|:---|:---|:---:|
| **Built-in (`direct`)** | `veystrix` | Built directly into Veystrix. Full terminal chat UI, syntax highlighting, diff viewing, model downloading, and real-time status rail. | Built-in |
| **OpenCode** | `veystrix run opencode` | Full-screen terminal coding agent with multi-file editing, planning, and tool use. | Yes |
| **Pi Coding Agent** | `veystrix run pi` | Minimalist, high-speed terminal coding agent with native tool execution and streaming. | Yes |
| **Hermes Agent** | `veystrix run hermes` | NousResearch agent with structured function calling, tool use, and multi-turn workflows. | Yes |
| **OpenAI / Anthropic APIs** | Loopback endpoints | Any editor extension, script, or CLI speaking `/v1/chat/completions` or `/v1/messages`. | Manual / Background |

> [!NOTE]
> External harnesses other than OpenCode, Pi, and Hermes (e.g. legacy or third-party agent harnesses) are locked in this public release. If a locked harness is requested, `veystrix` exits cleanly and directs you to the supported harnesses.

---

## Zero-Configuration Launch (`veystrix run`)

You do not need to start the engine manually before running a harness. Running `veystrix run <harness>` automatically inspects the engine state, boots the background engine on your configured port if stopped, verifies model readiness, and launches the harness with optimal environment parameters:

```sh
# Launch OpenCode pre-wired to Veystrix
veystrix run opencode

# Launch Pi Coding Agent pre-wired to Veystrix
veystrix run pi

# Launch Hermes Agent using the Veystrix profile
veystrix run hermes
```

Any arguments passed after `--` are forwarded directly to the underlying harness:

```sh
veystrix run opencode -- --model veystrix-inkling2-mini
```

---

## One-Time Provider Configuration (`veystrix connect`)

If you prefer to launch the harness directly from your shell without the `veystrix run` wrapper, configure its local provider file once:

```sh
# Configure OpenCode provider settings (~/.config/opencode/opencode.json)
veystrix connect opencode

# Configure Pi agent extension (~/.pi/agent/extensions/veystrix.ts)
veystrix connect pi

# Configure Hermes profile (~/.hermes/profiles/veystrix/config.yaml)
veystrix connect hermes
```

After running `veystrix connect <harness>`, ensure the Veystrix engine is running (`veystrix start`) and launch your harness normally.

---

## Switching Harnesses in the TUI (`/settings`)

Inside the interactive Veystrix interface (`veystrix`):

1. Press `Ctrl-K` or type `/settings` and press **Enter**.
2. Navigate down to the **Harness** row and press **Enter** to open the harness picker.
3. Select **OpenCode**, **Pi**, **Hermes**, or **Built-in (`direct`)**.
4. Press **Enter** to save your selection. The choice is persisted across sessions in `~/.veystrix/config.json`.

---

## Local OpenAI & Anthropic API Endpoints

Veystrix binds exclusively to loopback (`127.0.0.1`) by default, ensuring your code and prompts never leave your Mac:

- **OpenAI Compatible Endpoint:** `http://127.0.0.1:8080/v1/chat/completions`
- **OpenAI Models Endpoint:** `http://127.0.0.1:8080/v1/models`
- **Anthropic Compatible Endpoint:** `http://127.0.0.1:8080/v1/messages`

### Environment Variables for Generic Clients

To use Veystrix with any OpenAI-compatible CLI or SDK:

```sh
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="local"
export OPENAI_MODEL="veystrix-inkling2-mini"
```

To use Veystrix with any Anthropic-compatible client:

```sh
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"
export ANTHROPIC_API_KEY="local"
export ANTHROPIC_MODEL="veystrix-inkling2-mini"
```

### Quick Smoke Test

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local" \
  -d '{
    "model": "veystrix-inkling2-mini",
    "messages": [{"role": "user", "content": "Hello from Veystrix!"}],
    "max_tokens": 32
  }'
```
