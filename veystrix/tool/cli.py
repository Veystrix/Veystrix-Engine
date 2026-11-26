"""Subcommand dispatch for the Veystrix terminal tool.

M2 implements the front end and supervisor. Harness adapters (`connect`, `run`,
`harnesses`) land in M3 and the branded white-label TUI in M3b; until then they
fail with a clear message rather than being mistaken for chat arguments.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from veystrix.tool import engine as engine_mod
from veystrix.tool.commands import (cmd_cache, cmd_config, cmd_diagnostics, cmd_doctor,
                                    cmd_install, cmd_models, cmd_uninstall, cmd_update)
from veystrix.tool.config import Config, LocalKeyStore, home_dir, resolve_default_model
from veystrix.tool.harnesses import (
    DOC, HARNESS_ALIASES, HARNESS_CHOICES, NAMES, PUBLIC_HARNESS_CHOICES,
    canonical_harness_name, configure, is_locked, resolve_model_id, run_command, run_environment,
)
from veystrix.tool.supervisor import StartOptions, health, logs, restart, start, status, stop
from veystrix.tool.tui import DEFAULT_HARNESS, HARNESS_NAMES, tui

TOOL_COMMANDS = {
    "install", "uninstall",
    "models", "start", "stop", "restart", "status", "logs",
    "config", "cache", "storage", "doctor", "diagnostics", "update",
    "connect", "run", "harnesses", "tui", "engine", "model", "endpoint", "kv",
}


DESCRIPTIONS = {
    "start": "Start the endpoint (OpenAI + Anthropic API) in this terminal (stops when closed; --background detaches).",
    "stop": "Stop the endpoint.",
    "restart": "Stop and start the endpoint with the saved engine and model.",
    "status": "One line: running or not, model, KV mode and port.",
    "logs": "Print the endpoint log.",
    "connect": "Write a harness's config so it talks to this endpoint.",
    "run": "Launch a harness against the running endpoint (arguments after -- go to the harness).",
    "models": "List, inspect, download and port-check models (veystrix models --help).",
    "engine": "Show the engine, or switch it (python | cpp) and restart the endpoint.",
    "model": "Show the model, or switch it and restart the endpoint.",
    "kv": "Show the KV precision policy, or save one (applied at the next start).",
    "storage": "Show SSD prompt-cache storage usage and reclaim space (e.g. veystrix storage claim 1GB).",
}
JSON_HELP = "machine-readable output"
PORT_HELP = "endpoint port (default: the saved one)"


def _parser(command: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"veystrix {command}", description=DESCRIPTIONS.get(command))
    if command == "start":
        parser.add_argument("--engine", choices=engine_mod.ENGINE_CHOICES)
        parser.add_argument("--engine-profile", default=engine_mod.CPP_DEFAULT_PROFILE,
                            help="cpp parity profile to gate (default: agent, the only attested profile)")
        from veystrix.model_identity import MODEL_CHOICES
        from veystrix.kv_policy import CHOICES as KV_CHOICES
        parser.add_argument("--model", choices=MODEL_CHOICES, help="model family (default: the saved one)")
        parser.add_argument("--checkpoint", default="", help="checkpoint directory (default: the installed one)")
        parser.add_argument("--port", type=int, help=PORT_HELP)
        parser.add_argument("--context", type=int, help="context ceiling in tokens (default: the model's own)")
        parser.add_argument("--kv", choices=KV_CHOICES,
                            help="KV precision (generic runtime); auto = chosen by the memory plan")
        parser.add_argument("--host", choices=("127.0.0.1", "0.0.0.0"),
                            help="listen address; 0.0.0.0 (LAN) requires require-api-key")
        parser.add_argument("--json", action="store_true", help="print the status as JSON instead of the endpoint block")
        parser.add_argument("--foreground", action="store_true", help="run in this terminal (default on a TTY; Ctrl-C or closing stops it)")
        parser.add_argument("--background", "-d", action="store_true", help="detach into the background until veystrix stop")
        parser.add_argument("--timeout", type=float, default=300.0, help="seconds to wait for readiness")
        parser.add_argument("--from-bundle", action="store_true")
        parser.add_argument("--command", help="test-only engine command override")
    elif command == "logs":
        parser.add_argument("-n", "--lines", type=int, default=200)
    elif command == "models":
        parser.add_argument("rest", nargs=argparse.REMAINDER)
    elif command in ("config", "cache", "storage", "doctor", "diagnostics", "update"):
        parser.add_argument("rest", nargs=argparse.REMAINDER)
    elif command in ("install", "uninstall"):
        parser.add_argument("rest", nargs=argparse.REMAINDER)
    elif command in ("status", "stop", "restart"):
        parser.add_argument("--json", action="store_true", help=JSON_HELP)
    elif command in ("connect", "run"):
        parser.add_argument("harness", choices=PUBLIC_HARNESS_CHOICES, metavar="HARNESS",
                            help=f"target harness ({', '.join(PUBLIC_HARNESS_CHOICES)})")
        parser.add_argument("--port", type=int, help=PORT_HELP)
        parser.add_argument("--model", help="installed model id (default: the discovered model)")
        if command == "connect":
            parser.add_argument("--no-apply", action="store_true", help="preview without writing config")
        else:
            parser.add_argument("--dry-run", action="store_true", help="print the launch command and env only")
    return parser


def _cmd_swap(command: str, rest: list[str], home: Path) -> int:
    """``veystrix engine [python|cpp]`` and ``veystrix model [name]``: report or swap in one command."""
    from veystrix.tool.swap import MODELS, describe, swap
    parser = argparse.ArgumentParser(prog=f"veystrix {command}", description=DESCRIPTIONS[command])
    parser.add_argument("target", nargs="?", choices=(engine_mod.ENGINE_CHOICES if command == "engine"
                                                      else MODELS))
    parser.add_argument("--no-restart", action="store_true",
                        help="save the selection but leave a running engine alone")
    parser.add_argument("--json", action="store_true", help=JSON_HELP)
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for the restarted endpoint")
    args = parser.parse_args(rest)
    if args.target is None:
        report = describe(home)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        print(f"engine: selected={report['selected']} active={report['active'] or 'not running'}")
        print(f"model:  {report['model']}")
        for name, info in report["available"].items():
            mark = "ok " if info["available"] else "no "
            print(f"  {mark}{name}" + (f" - {info['reason']}" if info["reason"] else ""))
        print("switch with: veystrix engine cpp | veystrix engine python")
        return 0
    try:
        outcome = swap(home, **{command: args.target}, restart=not args.no_restart,
                       timeout=args.timeout, echo=lambda m: print("veystrix: " + m))
    except ValueError as exc:
        raise SystemExit(f"veystrix: {exc}") from exc
    if args.json:
        print(json.dumps(outcome, indent=2, sort_keys=True))
    else:
        verb = "restarted on" if outcome["restarted"] else "now using"
        print(f"veystrix: {verb} {outcome['engine']} / {outcome['model']}"
              + ("" if outcome["changed"] else " (already selected)"))
    return 0


def _start_engine(args, config: Config, model: str) -> str:
    """--engine / VEYSTRIX_ENGINE / config, except that a model with a single engine gets that one
    when nothing explicit asked otherwise (``veystrix start --model ornith`` runs the C++ engine)."""
    from veystrix.tool.swap import engine_for
    return engine_for(model, getattr(args, "engine", None) or os.environ.get("VEYSTRIX_ENGINE"), config.home)


def _default_checkpoint(model: str) -> str:
    """The single installed checkpoint for ``model`` (the router needs an explicit path)."""
    from veystrix.tool.swap import default_checkpoint
    return default_checkpoint(model)


def _print_ready_block(home: Path, port: int, *, foreground: bool = False) -> None:
    from veystrix.tool.endpoint import describe, render
    try:
        print(render(describe(home, port=port), harnesses=False), flush=True)
        tail = ("(Ctrl-C or closing this terminal stops the endpoint)" if foreground
                else "(the endpoint keeps running until veystrix stop)")
        print(f"\nharness snippets: veystrix endpoint   {tail}", flush=True)
    except Exception as exc:                       # never fail a successful start on the printout
        print(f"veystrix: endpoint ready on 127.0.0.1:{port} ({exc})", flush=True)


def _cmd_kv(rest: list[str], home: Path, config: Config) -> int:
    """``veystrix kv`` shows the KV policy; ``veystrix kv MODE`` saves it (next start applies it)."""
    from veystrix.kv_policy import CHOICES, MODES
    parser = argparse.ArgumentParser(prog="veystrix kv", description=DESCRIPTIONS["kv"])
    parser.add_argument("mode", nargs="?", choices=CHOICES)
    parser.add_argument("--json", action="store_true", help=JSON_HELP)
    args = parser.parse_args(rest)
    if args.mode:
        config.values["kv"] = args.mode
        config.save()
    report = {"kv": config.get("kv"), "modes": {k: dict(zip(
        ("prefix_tokens", "prefix_bits", "recent_tokens", "recent_bits", "hist_k_bits", "hist_v_bits"), v))
        for k, v in MODES.items()}}
    running = health(config.get("port"), api_key=LocalKeyStore(home).load())
    if running and isinstance(running.get("runtime"), dict):
        report["active"] = running["runtime"].get("kv")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"kv policy: {report['kv']}" + (" (saved; restart the endpoint to apply)" if args.mode else ""))
        if report.get("active"):
            print(f"active:    {report['active'].get('mode')} ({report['active'].get('plan')})")
        print("modes: " + ", ".join(CHOICES))
    return 0


def status_line(result: dict) -> str:
    """``veystrix status`` for people: one line, the facts a harness user needs."""
    state = result.get("status") or "stopped"
    port = result.get("port")
    if state != "running":
        extra = f"; last model {result['model']}" if result.get("model") else ""
        return f"veystrix: {state} (port {port}{extra}); start it with: veystrix start"
    parts = [f"veystrix: running on 127.0.0.1:{port}", f"model {result.get('model') or '?'}"]
    if result.get("kv"):
        parts.append(f"kv {result['kv']}")
    if result.get("context_ceiling"):
        used = result.get("context_in_use")
        parts.append(f"context {used if used is not None else '?'}/{result['context_ceiling']}")
    return " | ".join(parts)


def _served_ids(port: int, key: str | None) -> list[str] | None:
    from veystrix.tool.endpoint import _get_json
    data = _get_json(f"http://127.0.0.1:{port}/v1/models", key)
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        return None
    return [str(m["id"]) for m in data["data"] if isinstance(m, dict) and m.get("id")]


def _selected_engine(args, config: Config) -> str:
    explicit = getattr(args, "engine", None)
    return engine_mod.resolve_engine(explicit, os.environ, config.values)


def dispatch_tool(argv: list[str]) -> int:
    if not argv:
        raise SystemExit("veystrix: expected a command; see --help")
    command, rest = argv[0], argv[1:]
    home = home_dir()
    config = Config(home)
    if command not in TOOL_COMMANDS:
        raise SystemExit(f"veystrix: unknown command {command!r}")
    if command == "tui":
        parser = argparse.ArgumentParser(prog="veystrix tui")
        parser.add_argument("--harness", default=None,
                            help=f"default: the saved harness (config), else {DEFAULT_HARNESS} ({', '.join(PUBLIC_HARNESS_CHOICES)})")
        parser.add_argument("--port", type=int)
        parser.add_argument("--model")
        parser.add_argument("--resume")
        parser.add_argument("--once", action="store_true", help="run one non-interactive session (tests)")
        parser.add_argument("--cancel-after", type=float)
        parser.add_argument("--command", help="test-only harness command override")
        tui_args = parser.parse_args(rest)
        if tui_args.harness:
            c_name = canonical_harness_name(tui_args.harness)
            if is_locked(c_name) and c_name != "direct":
                raise SystemExit(
                    f"veystrix: harness '{tui_args.harness}' is locked and unavailable in this release.\n"
                    f"Available harnesses in this release: {', '.join(PUBLIC_HARNESS_CHOICES)}."
                )
            tui_args.harness = c_name
        tui(tui_args.harness, home=home, port=tui_args.port or config.get("port"),
                   model=tui_args.model, resume=tui_args.resume, once=tui_args.once,
                   cancel_after=tui_args.cancel_after,
                   command=shlex.split(tui_args.command) if tui_args.command else None)
        return 0
    if command in ("engine", "model"):
        return _cmd_swap(command, rest, home)
    if command == "endpoint":
        from veystrix.tool.endpoint import main as endpoint_main
        return endpoint_main(rest, home)
    if command == "kv":
        return _cmd_kv(rest, home, config)
    if command == "storage":
        from veystrix.tool.storage import cmd_storage
        return cmd_storage(rest, home, config)
    if command == "install":
        apply = "--apply" in rest
        return cmd_install([arg for arg in rest if arg != "--apply"], home, config, apply=apply)
    if command == "uninstall":
        apply = "--apply" in rest
        return cmd_uninstall([arg for arg in rest if arg != "--apply"], home, apply=apply)
    if command == "harnesses":
        import shutil
        from veystrix.tool.harness_adapters import structured_available
        for name in PUBLIC_HARNESS_CHOICES:
            try:
                binary = run_command(name, "model")[0]
            except ValueError:
                binary = None                      # config-only (editor extensions)
            found = shutil.which(binary) if binary else None
            state = ("installed, own UI" if structured_available(name)
                     else "installed" if found else "config only" if binary is None else "not installed")
            print(f"{name:<12} {state:<18} {DOC[name][0]} (checked {DOC[name][1]})")
        return 0

    # `veystrix run <harness> [options] [-- harness args]`: keep options out of the passthrough.
    passthrough: list[str] = []
    if command == "run" and "--" in rest:
        split = rest.index("--")
        passthrough = rest[split + 1:]
        rest = rest[:split]
    if command in ("connect", "run") and rest and not rest[0].startswith("-"):
        raw_harness = rest[0]
        canonical = canonical_harness_name(raw_harness)
        if is_locked(canonical) or is_locked(raw_harness):
            raise SystemExit(
                f"veystrix: harness '{raw_harness}' is locked and unavailable in this release.\n"
                f"Available harnesses in this release: {', '.join(PUBLIC_HARNESS_CHOICES)}."
            )
        if canonical in PUBLIC_HARNESS_CHOICES:
            rest[0] = canonical
    args = _parser(command).parse_args(rest)
    if command == "connect":
        model_id = args.model
        if not model_id:
            try:
                model_id = resolve_default_model(config, interactive=sys.stdin.isatty())
            except SystemExit:
                model_id = None          # no installed checkpoint (offline/CI): adapter uses its placeholder id
        result = configure(args.harness, home=home, port=args.port or config.get("port"),
                           model_id=model_id, key=LocalKeyStore(home).load() or "local",
                           apply=not args.no_apply)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if command == "run":
        import shutil
        port = args.port or config.get("port")
        model_id = resolve_model_id(port, args.model,
                                    api_key=LocalKeyStore(home).load(),
                                    explicit=args.model is not None)
        report = health(port, api_key=LocalKeyStore(home).load())
        context = (report.get("context_limit") if isinstance(report, dict)
                   and report.get("installed_model_id") == model_id else None)
        env = run_environment(args.harness, port, model_id, LocalKeyStore(home).load() or "local", context=context)
        command_line = [*run_command(args.harness, model_id), *passthrough]
        if args.dry_run:
            print(json.dumps({"command": command_line, "environment": env}, indent=2, sort_keys=True))
            return 0
        if report is None:
            if shutil.which(command_line[0]) is None:
                raise SystemExit(
                    f"veystrix: harness {args.harness!r} ({command_line[0]!r}) is not installed or not on PATH; "
                    "run 'veystrix harnesses' to check."
                )
            from veystrix.tool.swap import ensure_engine
            print(f"veystrix: engine is not running on 127.0.0.1:{port}; starting it automatically...", file=sys.stderr)
            ensure_engine(
                home,
                port=port,
                model=args.model,
                explicit_model=args.model is not None,
                interactive=sys.stdin.isatty(),
                echo=lambda m: print(f"veystrix: {m}", file=sys.stderr),
            )
            report = health(port, api_key=LocalKeyStore(home).load())
            if report is None:
                raise SystemExit(f"veystrix: no engine answering on 127.0.0.1:{port}; failed to auto-start.")
            model_id = resolve_model_id(port, args.model,
                                        api_key=LocalKeyStore(home).load(),
                                        explicit=args.model is not None)
            context = (report.get("context_limit") if isinstance(report, dict)
                       and report.get("installed_model_id") == model_id else None)
            env = run_environment(args.harness, port, model_id, LocalKeyStore(home).load() or "local", context=context)
            command_line = [*run_command(args.harness, model_id), *passthrough]
        served = _served_ids(port, LocalKeyStore(home).load())
        if served and model_id not in served:
            norm_id = model_id.removeprefix("veystrix-")
            if not any(s in (model_id, norm_id, f"veystrix-{norm_id}") for s in served):
                # The router answers an unknown id with 404; say so before the harness starts and fails later.
                raise SystemExit(f"veystrix: the endpoint on 127.0.0.1:{port} serves {', '.join(served)}; "
                                 f"{model_id!r} is not one of them (use --model with one of those ids).")
        try:
            os.execvpe(command_line[0], command_line, {**os.environ, **env})
        except FileNotFoundError as exc:
            raise SystemExit(
                f"veystrix: harness {args.harness!r} ({command_line[0]!r}) is not installed or not on PATH; "
                "run 'veystrix harnesses' to check."
            ) from exc
        return 0                                               # only reached when exec is stubbed (tests)
    if command == "start":
        from veystrix.tool.swap import resolve_model_and_checkpoint
        model = args.model or config.get("model")
        checkpoint = args.checkpoint
        engine = _start_engine(args, config, model)
        engine_profile = getattr(args, "engine_profile", engine_mod.CPP_DEFAULT_PROFILE)
        if getattr(args, "engine", None):
            engine_mod.check_engine(engine, home, model, engine_profile)
        if not args.command:
            model, checkpoint = resolve_model_and_checkpoint(
                model,
                checkpoint=checkpoint,
                explicit_model=args.model is not None,
                interactive=sys.stdin.isatty() and not args.json,
                echo=lambda m: print(f"veystrix: {m}", file=sys.stderr),
                resolver_fn=_default_checkpoint,
            )
            engine = _start_engine(args, config, model)
        engine_mod.check_engine(engine, home, model, engine_profile)
        port = args.port or config.get("port")
        foreground = bool(args.foreground or (not args.background and not args.json and sys.stdin.isatty()))
        announce = None if args.json else (lambda _state: _print_ready_block(home, port, foreground=foreground))
        options = StartOptions(
            home=home, port=port, engine=engine,
            model=model, checkpoint=checkpoint,
            command=shlex.split(args.command) if args.command else None,
            foreground=foreground, timeout=args.timeout, from_bundle=args.from_bundle,
            engine_profile=engine_profile, kv=args.kv or "", context=args.context or 0,
            host=args.host or config.get("host"), on_ready=announce)
        result = start(options)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        elif foreground:
            print("veystrix: endpoint stopped", flush=True)
        return 0
    if command == "stop":
        result = stop(home)
    elif command == "restart":
        from veystrix import model_identity as _mi
        from veystrix.tool.swap import resolve_model_and_checkpoint
        old = status(home)
        model = old.get("model") or config.get("model")
        if _mi.is_locked(model) or model not in _mi.PUBLIC_MODEL_CHOICES:
            model = config.get("model")
        model, checkpoint = resolve_model_and_checkpoint(
            model,
            explicit_model=False,
            interactive=sys.stdin.isatty() and not args.json,
            echo=lambda m: print(f"veystrix: {m}", file=sys.stderr),
        )
        stop(home)
        engine = _start_engine(args, config, model)
        options = StartOptions(home=home, port=old.get("port") or config.get("port"), engine=engine,
                               model=model, checkpoint=checkpoint, timeout=300.0)
        result = start(options)
    elif command == "status":
        result = status(home)
        if not args.json:
            print(status_line(result))
            return 0
    elif command == "logs":
        print(logs(home, args.lines), end="")
        return 0
    elif command == "models":
        return cmd_models(args.rest, home, config)
    elif command == "config":
        return cmd_config(args.rest, home, config, LocalKeyStore(home))
    elif command == "cache":
        return cmd_cache(args.rest, home, config)
    elif command == "doctor":
        return cmd_doctor(args.rest, home, config, _selected_engine(args, config))
    elif command == "diagnostics":
        return cmd_diagnostics(args.rest, home, config, LocalKeyStore(home))
    elif command == "update":
        return cmd_update(args.rest, home, config)
    else:                                                      # unreachable: kept defensive
        raise SystemExit(f"veystrix: unknown command {command!r}")
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"{command}: " + ", ".join(f"{key}={value}" for key, value in sorted(result.items())
                                         if key not in ("command",)))
    return 0
