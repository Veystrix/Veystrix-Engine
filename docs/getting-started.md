# Getting Started with Veystrix (v0.1.0)

This guide covers installing Veystrix, downloading or pointing to a model checkpoint, running the terminal UI or local API server, and managing KV cache & SSD storage.

## 1. Requirements & Installation

- **Hardware**: Apple Silicon Mac (`arm64`), macOS 14+ (16 GB unified memory minimum; 24 GB+ for full in-RAM residency).
- **Python**: 3.12+ (if installing from source).

Install via pip from the repository root:

```sh
python3 -m venv .venv
.venv/bin/pip install .
```

Verify the installation:

```sh
veystrix --version
veystrix doctor
```

## 2. Model Setup (`Inkling 2 Mini`)

Veystrix v0.1.0 ships with support for **Inkling 2 Mini** (`inkling2-mini`) on the generic C++ runtime.

- Check detected checkpoints (default directory: `~/models`, or set `VEYSTRIX_MODEL_ROOT`):
  ```sh
  veystrix models list
  ```
- Download the checkpoint:
  ```sh
  veystrix models pull inkling2-mini
  ```
- Or register an existing local checkpoint folder without downloading:
  ```sh
  veystrix models add /path/to/Inkling-2-Mini-oQ4e-mtp
  ```

## 3. Running Veystrix

### Interactive Terminal UI (TUI)

Run `veystrix` inside any project directory:

```sh
veystrix
```

Inside the TUI you can use `/models`, `/kv`, `/config`, `/cache`, `/doctor`, `/update`, and `/quit`.

### Background API Server (OpenAI & Anthropic Compatible)

Start the local loopback server on `127.0.0.1:8080`:

```sh
veystrix start --background
veystrix status
veystrix endpoint    # prints base URLs, model ID, and client snippets
```

Test with `curl`:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local" \
  -d '{
    "model": "veystrix-inkling2-mini",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 64
  }'
```

Stop the server when finished:

```sh
veystrix stop
```

### Driving External Coding Agents

Launch supported harnesses pre-wired to your local engine (auto-starts the engine if stopped):

```sh
veystrix run opencode
veystrix run pi
veystrix run hermes
```

See [Harness Integration](harness-integration.md) for one-time `veystrix connect` setup and client details.

## 4. KV Cache & SSD Storage

By default (`kv = auto`), Veystrix picks the most precise KV mode that fits your memory budget — selecting `sparse-paged` at long context on 16 GB Macs (exact BF16 prefix + recent window in RAM, 4-bit quantized 256-token cold history pages deduplicated on SSD).

```sh
veystrix kv                           # show current policy and available modes
veystrix kv sparse-paged              # set KV mode explicitly (restart to apply)
veystrix config set cache-quota-gb 16 # set SSD cache quota (1–2000 GB)
veystrix storage                      # inspect SSD cache usage
veystrix storage claim 5GB            # reclaim oldest snapshots (use --dry-run to preview)
```

## 5. Troubleshooting

Run `veystrix doctor` to check Python version, Apple Silicon architecture, free disk/RAM, port availability, checkpoint integrity, and engine bundle status. To generate a sanitized diagnostic bundle:

```sh
veystrix diagnostics
```
