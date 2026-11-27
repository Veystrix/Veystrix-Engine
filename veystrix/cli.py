#!/usr/bin/env python3
"""Select one model runtime, then execute the shared chat in its environment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from veystrix import model_identity, preferences
from veystrix.checkpoints import installed_checkpoints
from veystrix.tool.cli import TOOL_COMMANDS as _TOOL_COMMANDS, dispatch_tool as _dispatch_tool


PYTHON_ENV = {
    "shared": "VEYSTRIX_PYTHON",
    "inkling2-mini": "VEYSTRIX_INKLING2_PYTHON",
}
MANAGED_ENV = ROOT / ".venv"
MANAGED_PYTHON = MANAGED_ENV / "bin" / "python"
RUNTIME_VERSIONS = {
    "mlx": "0.32.2",
    "mlx-lm": "0.31.3",
    "mlx-vlm": "0.6.17",
    "transformers": "5.16.1",
    "numpy": "2.5.2",
    "requests": "2.34.2",
    "huggingface-hub": "1.29.0",
}
RUNTIME_PROBES = {
    "inkling2-mini": """
import mlx.core
from transformers import AutoTokenizer
""",
}


def branch_sync_warning(root: Path = ROOT) -> str:
    """Warn when a checkout lacks commits from its known origin/main."""
    if not (root / ".git").exists():
        return ""              # the packaged app is not a checkout (and must not poke at git)
    try:
        current = subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor",
             "origin/main", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return ""
    if current.returncode != 1:
        return ""
    branch = subprocess.run(
        ["git", "-C", str(root), "branch", "--show-current"],
        capture_output=True,
        check=False,
        text=True,
    ).stdout.strip() or "current branch"
    return (
        f"warning: {branch} does not include known origin/main; "
        "fetch and merge origin/main before validating chat behavior"
    )


def _split_build(argv: list[str]) -> tuple[str | None, list[str]]:
    if argv and not argv[0].startswith("-"):
        return argv[0], argv[1:]
    return None, argv


def _runtime_probe(model: str) -> str:
    versions = repr(RUNTIME_VERSIONS)
    # On the C++ generic runtime, Flash-Next does not import mlx_vlm; avoid the slow/unnecessary probe.
    probe_key = "ornith" if (model == "flashnext" and os.environ.get("VEYSTRIX_ENGINE", "").strip().lower() == "cpp") else model
    return f"""
from importlib.metadata import PackageNotFoundError, version
import sys

expected = {versions}
for package, wanted in expected.items():
    try:
        found = version(package)
    except PackageNotFoundError:
        raise SystemExit(f"missing package {{package}}")
    if found != wanted:
        raise SystemExit(f"{{package}} {{found}}, expected {{wanted}}")

import mlx.core as mx
import mlx.nn as nn
if not hasattr(mx, "array") or not hasattr(mx, "fast") or not hasattr(nn, "Module"):
    raise SystemExit("MLX backend API is incomplete")
{RUNTIME_PROBES[probe_key]}
"""


def _python_runtime_error(path: Path, model: str) -> str | None:
    try:
        result = subprocess.run(
            [str(path), "-c", _runtime_probe(model)],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    if result.returncode == 0:
        return None
    detail = (result.stderr or result.stdout).strip().splitlines()
    return detail[-1][:240] if detail else f"probe exited with {result.returncode}"


def _supports_python(path: Path, model: str) -> bool:
    return _python_runtime_error(path, model) is None


def _setup_failure(operation: str, error: BaseException) -> SystemExit:
    detail = getattr(error, "stderr", None) or str(error)
    return SystemExit(
        f"Veystrix setup failed during {operation}: {detail}\n"
        "Retry with './chat.sh setup'."
    )


def _run_setup_step(operation: str, command: list[str]) -> None:
    print(f"Veystrix setup: {operation}", flush=True)
    try:
        subprocess.check_call(command)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise _setup_failure(operation, exc) from exc


def _interpreter(model: str) -> Path:
    override_name = next(
        (name for name in (PYTHON_ENV[model], PYTHON_ENV["shared"])
         if os.environ.get(name)),
        None,
    )
    override = os.environ.get(override_name) if override_name else None
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise SystemExit(f"missing Python environment override {override_name}: {path}")
        if (os.environ.get("VEYSTRIX_SKIP_RUNTIME_PROBE") == "1"
                or path.resolve() == Path(sys.executable).resolve()):
            return path
        if not _supports_python(path, model):
            reason = _python_runtime_error(path, model) or "runtime probe failed"
            raise SystemExit(
                f"Python environment override {override_name} is incompatible "
                f"with {model}: {reason}"
            )
        return path

    if not _supports_python(MANAGED_PYTHON, model):
        setup_environment(["--venv", str(MANAGED_ENV)])
    if not _supports_python(MANAGED_PYTHON, model):
        reason = _python_runtime_error(MANAGED_PYTHON, model) or "runtime probe failed"
        raise SystemExit(
            f"managed Veystrix environment is incompatible with {model}: {reason}\n"
            "Retry with './chat.sh setup'."
        )
    return MANAGED_PYTHON


def setup_environment(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="veystrix setup")
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    args = parser.parse_args(argv)
    target = args.venv.expanduser().resolve()
    creator = sys.executable
    if sys.version_info < (3, 12):
        creator = shutil.which("python3.12") or ""
    if not creator:
        raise SystemExit("Python 3.12 is required; install it, then run setup again")
    _run_setup_step("create the managed Python environment", [creator, "-m", "venv", str(target)])
    python = target / "bin" / "python"
    _run_setup_step(
        "upgrade pip",
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
    )
    _run_setup_step(
        "install Veystrix and its shared runtime",
        [str(python), "-m", "pip", "install", "-e", str(ROOT)],
    )
    print(f"Veystrix environment ready: {target}")
    return 0


def _default_checkpoint_args(argv: list[str]) -> list[str]:
    """Select the only installed checkpoint, or ask before loading one."""
    if argv and (argv[0] == "setup" or not argv[0].startswith("-")):
        return argv
    if any(
        value in ("--model", "--model-path", "--checkpoint")
        or value.startswith(("--model=", "--model-path=", "--checkpoint="))
        for value in argv
    ):
        return argv

    choices = installed_checkpoints()
    if not choices:
        return argv
    if len(choices) == 1:
        model, path = choices[0]
        return [*argv, "--model", model, "--checkpoint", str(path)]

    print("Multiple compatible checkpoints found:", file=sys.stderr)
    for index, (model, path) in enumerate(choices, 1):
        print(f"  {index}. {model}: {path}", file=sys.stderr)
    if not sys.stdin.isatty():
        raise SystemExit(
            "choose a checkpoint with --model and --checkpoint before loading"
        )
    try:
        answer = input(f"Choose a checkpoint [1-{len(choices)}]: ").strip()
        index = int(answer) - 1
    except (EOFError, ValueError):
        raise SystemExit("invalid checkpoint choice") from None
    if not 0 <= index < len(choices):
        raise SystemExit("invalid checkpoint choice")
    model, path = choices[index]
    return [*argv, "--model", model, "--checkpoint", str(path)]


def command(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    argv = ["--server" if value == "/server" else value for value in argv]
    build, argv = _split_build(list(argv))
    if build:
        raise SystemExit(f"veystrix: unexpected argument {build!r}; model selection is --model {'|'.join(model_identity.MODEL_CHOICES)}")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", choices=model_identity.MODEL_CHOICES)
    parser.add_argument("--kv", help="KV precision for the generic C++ runtime (auto, bf16, int8, tiered, ...)")
    parser.add_argument("--profile", choices=("plain", "agent"))
    parser.add_argument("--model-path", "--checkpoint", dest="model_path")
    parser.add_argument("--preferences-file", default=preferences.DEFAULT_PATH)
    parser.add_argument("--v4", action="store_true")
    parser.add_argument("--context-limit", type=int)
    parser.add_argument("--prefix-dir")
    known, remaining = parser.parse_known_args(argv)

    model = known.model or model_identity.MODEL_CHOICES[0]
    interpreter = _interpreter(model)

    # A source checkout runs the session script; the packaged app ships compiled modules only, so it runs the module.
    session_script = ROOT / "veystrix" / "session.py"
    run_session = [str(interpreter), "-u", str(session_script)] if session_script.is_file() \
        else [str(interpreter), "-u", "-m", "veystrix.session"]
    chat_args = [
        *run_session,
        "--model", model,
        "--preferences-file", known.preferences_file,
    ]
    if known.profile:
        chat_args += ["--profile", known.profile]
    if known.model_path:
        chat_args += ["--model-path", known.model_path]
    if os.environ.get("VEYSTRIX_WORKSPACE") and "--workspace" not in remaining:
        chat_args += ["--workspace", os.environ["VEYSTRIX_WORKSPACE"]]
    chat_args.extend(remaining)

    environment = dict(os.environ)
    if environment.get("VEYSTRIX_MLX_AUTO_BUFFER_CAPS") == "1":
        # MLX reads these once in the child, before its first Metal command buffer.
        environment.setdefault("MLX_MAX_OPS_PER_BUFFER", "200")
        environment.setdefault("MLX_MAX_MB_PER_BUFFER", "1024")
    if known.prefix_dir:
        environment["VEYSTRIX_CACHE_ROOT"] = known.prefix_dir
    environment["VEYSTRIX_ENGINE"] = "cpp"
    if "--server" in chat_args:
        environment.setdefault("VEYSTRIX_MODE", "agentic")
        environment.setdefault("FLASHNEXT_WATCHDOG", "1")
        environment.setdefault("FLASHNEXT_WATCHDOG_MEMORY", "1")
    environment.setdefault("FLASHNEXT_CHAT_GUARD", "1")
    environment.setdefault("FLASHNEXT_PREFILL_CHUNK_TOKENS", "2048")
    if known.context_limit is not None:
        environment["VEYSTRIX_CONTEXT_LIMIT"] = str(known.context_limit)
    if known.kv:
        environment["VEYSTRIX_KV_MODE"] = known.kv
    if not session_script.is_file():
        environment["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), environment.get("PYTHONPATH", "")) if p)
    return ["/usr/bin/caffeinate", "-i", *chat_args], environment


def _product_version() -> str:
    """The running build's version. In a sealed bundle the manifest sits at the engine root
    next to this code (…/engine/lib/veystrix/cli → …/engine/BUNDLE-MANIFEST.json); in a source
    checkout fall back to pyproject.toml. Never read it from $HOME, which may hold another build."""
    import json
    for candidate in (Path(__file__).resolve().parents[2] / "BUNDLE-MANIFEST.json",):
        try:
            version = str(json.loads(candidate.read_text()).get("version") or "")
            if version:
                return version
        except Exception:
            pass
    try:
        import tomllib
        return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    except Exception:
        return "dev"


def main() -> int:
    # Help/version must work offline, with no model selection or setup side
    # effects. Handle them before checkpoint discovery (terminal-tool UX).
    if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-V"):
        print(f"veystrix {_product_version()}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(
            f"veystrix {_product_version()} — a local coding agent for your Mac\n"
            "\nUsage:\n"
            "  veystrix                 launch the coding-agent terminal UI\n"
            "  veystrix start           start the local model server (OpenAI- and Anthropic-compatible\n"
            "                           API) in the background\n"
            "  veystrix status          show whether the model is loaded and ready\n"
            "  veystrix model NAME      choose the active model (browse and download with /models in the UI)\n"
            "  veystrix endpoint        print the local API base URL, key and client snippets\n"
            "  veystrix storage         show SSD prompt-cache storage usage and reclaim space\n"
            "  veystrix update          check for and install the latest version\n"
            "  veystrix <command>       other commands "
            "(install, connect, run, harnesses, doctor, uninstall, ...)\n"
            "  veystrix <command> --help   detailed help for a command\n"
            "\nRun `veystrix harnesses` to connect a coding agent to the local API."
        )
        return 0
    # Bare `veystrix` launches the Veystrix-branded coding-agent TUI like Claude Code.
    if len(sys.argv) == 1:
        return _dispatch_tool(["tui"])
    # Terminal-tool subcommands take precedence over the legacy chat launcher.
    # Nothing here loads a model or touches harness config files; M2 works offline.
    if len(sys.argv) > 1 and sys.argv[1] in _TOOL_COMMANDS:
        return _dispatch_tool(sys.argv[1:])
    if len(sys.argv) > 1 and sys.argv[1] == "workflow":
        from veystrix.workflows.__main__ import main as workflow_main
        return workflow_main(sys.argv[2:])

    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        return setup_environment(sys.argv[2:])
    warning = branch_sync_warning()
    if warning:
        print(warning, file=sys.stderr)
    executable, environment = command(_default_checkpoint_args(sys.argv[1:]))
    os.execvpe(executable[0], executable, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
