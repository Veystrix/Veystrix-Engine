# Documentation

> **Status (updated 2026-11-16):** two models are supported — Qwen 3.8 Flash-Next (the default) and NVIDIA Nemotron 3.5 Lightning (experimental). CHAT and AGENTIC modes (`--mode chat|agentic`, supervised `./serve.sh`) are in the repository; see the [mode and agentic validation guide](flashnext/agentic-validation.md) for tested CHAT behavior and the AGENTIC acceptance checks. Unit suites on `main`: 988 shared, 2,289 Flash-Next, 203 Nemotron; the Linux fast lane runs 455 no-MLX tests in ~17s and the macOS app's Swift logic tests run in CI. Real-checkpoint harness runs remain owner-validated, not claimed here. Model decisions are recorded in the [ADR log](decisions/README.md). Release, packaging and security guides: [release process](release/RELEASE_PROCESS.md), [free channel](release/FREE_CHANNEL.md), [validation status](release/VALIDATION.md), [packaging](packaging/PACKAGING.md), [security audit](security/SECURITY_AUDIT_2026-11-10.md), [manual launch](MANUAL_LAUNCH.md).

Use the component handoff before changing code or running a benchmark. Read
the research record when you need prior measurements or rejected approaches.
[`CONTRIBUTING.md`](../CONTRIBUTING.md) defines the shared workflow.

## Document structure

Every active component uses the same three documents:

| Document | Purpose |
|---|---|
| `brief.md` | Purpose, scope, support status, and main results |
| `research.md` | Dated measurements, decisions, and rejected approaches |
| `handoff.md` | Current operation, validation, constraints, and next work |

Measurement directories index raw evidence. The project-wide
[measurement standard](measurement-standard.md) defines the shared test and
record rules.

## Components

| Component | Brief | Research | Handoff |
|---|---|---|---|
| Shared chat | [Brief](chat/brief.md) | [Research](chat/research.md) | [Handoff](chat/handoff.md) |
| Flash-Next | [Brief](flashnext/brief.md) | [Research](flashnext/research.md) | [Handoff](flashnext/handoff.md) |
| Nemotron 3.5 Lightning | [Plan](nemotron35/plan.md) | [Architecture](nemotron35/architecture.md) | [Implementation](nemotron35/implementation.md) |

Flash-Next also has [agent invariants](flashnext/AGENT_INVARIANTS.md), which
collect runtime-specific correctness and experiment constraints.

## Supporting material

| Path | Purpose |
|---|---|
| [`decisions/README.md`](decisions/README.md) | Architecture decision record (ADR) log for model and runtime decisions |
| [`security/ANTI_REVERSE_ENGINEERING.md`](security/ANTI_REVERSE_ENGINEERING.md) | Honest limits of local tamper resistance and the commercial protection path |
| [`performance/SPEEDUP_ROADMAP.md`](performance/SPEEDUP_ROADMAP.md) | Sequential 22-method execution ledger — check current step and status |
| [`performance/DO_NOT_IMPLEMENT.md`](performance/DO_NOT_IMPLEMENT.md) | Measured dead ends — check before retrying any speed idea |
| [`performance/EXACT_SPEED.md`](performance/EXACT_SPEED.md) | Curated bit-identical speed profile for both models |
| [`hardware/MEMORY_SUPPORT.md`](hardware/MEMORY_SUPPORT.md) | Supported unified-memory tiers and pool caps |
| [`measurement-standard.md`](measurement-standard.md) | Shared live-test and JSONL evidence rules |
| [`REMOTE_ACCESS.md`](REMOTE_ACCESS.md) | Using the engine from other devices over Tailscale |
| [`LICENSE`](../LICENSE) | Veystrix License: free for personal use (incl. benchmarking/research); paid commercial license required for business use |
| [`results/flashnext/`](../results/flashnext/) | Retained Flash-Next records |
| [`flashnext/graphics/`](flashnext/graphics/) | Flash-Next trace graphics and plots |
| [`MLX/`](MLX/) | MLX Metal backend source notes; reference material, not workflow instructions |

The [archive](archive/README.md) preserves superseded runbooks and historical
records. Archived documents may contain obsolete paths or commands; active
documents are the source of current instructions.
