"""Veystrix-branded terminal coding-agent UI (M3b).

The Veystrix TUI is the only surface the user sees. It drives an external agentic harness
underneath, with the harness's model endpoint pointed at the local Veystrix engine. The
harness keeps its own tools and agent loop; Veystrix supplies the model and the UI.

White-label strategy, in order:
  1. documented suppression flags/env vars (cited per harness below);
  2. a conservative output filter that drops harness banners, update and telemetry lines;
  3. rewriting any surviving harness name to "Veystrix".

Where a harness publishes no branding flag, that limitation is recorded and filtering is
the authority. No flag or key is guessed.
"""
from __future__ import annotations

import json
import os
import queue
import re
import select
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from veystrix.tool.config import Config, LocalKeyStore, home_dir, tool_home
from veystrix.tool.harnesses import (
    PUBLIC_HARNESS_CHOICES, is_locked, resolve_model_id, run_command, run_environment,
)
from veystrix.tool.supervisor import health

CHECKED = "2026-11-18"

# Doc-verified suppression settings. `doc` is the URL used for the check on CHECKED.
# `flags` are only used when a harness documents them; no guessed flags.
WHITELABEL = {
    "opencode": {
        "doc": "https://opencode.ai/docs/config/",
        "settings": {"update": "disable"},     # config: `update: "disable"` skips update checks
        "flags": [], "env": {},
        "note": "No documented banner-suppression flag; output filtering is the authority.",
    },
    "claude-code": {
        "doc": "https://code.claude.com/docs/en/env-vars",
        "settings": {}, "flags": [],
        "env": {"DISABLE_TELEMETRY": "1", "DISABLE_UPDATES": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_HIDE_CWD": "1"},
        "note": "Env vars documented; startup logo persists, so output filtering removes it.",
    },
    "codex": {
        "doc": "https://developers.openai.com/codex/config-reference",
        "settings": {}, "flags": [], "env": {"DO_NOT_TRACK": "1"},
        "note": "No documented banner/update-suppression flag; output filtering is the authority.",
    },
    "qwen-code": {
        "doc": "https://qwenlm.github.io/qwen-code-docs/en/users/configuration/settings",
        "settings": {}, "flags": ["--no-telemetry"],
        "env": {"QWEN_TELEMETRY_ENABLED": "0", "QWEN_USAGE_STATISTICS_ENABLED": "0"},
        "note": "--no-telemetry and the two env vars are documented; banner still filtered.",
    },
    "deepseek": {
        "doc": "https://github.com/deepseek-ai/DeepSeek-TUI",
        "settings": {}, "flags": [],
        "env": {"DO_NOT_TRACK": "1", "DSH_PROFILE": "headless"},
        "note": "DeepSeek (dsh) headless profile; Veystrix provider via ~/.dsh/profiles/headless/cordis.patch.yml.",
    },
    "kimi": {
        "doc": "https://moonshotai.github.io/kimi-code/en/configuration/providers",
        "settings": {}, "flags": [], "env": {"DO_NOT_TRACK": "1"},
        "note": "No documented branding flag; output filtering is the authority.",
    },
    "hermes": {
        "doc": "https://hermes-agent.nousresearch.com/docs/integrations/providers",
        "settings": {}, "flags": [], "env": {"DO_NOT_TRACK": "1"},
        "note": "No documented branding flag; output filtering is the authority.",
    },
    "pi": {
        "doc": "https://pi.dev/docs/latest/settings",
        "settings": {"quietStartup": True, "enableInstallTelemetry": False, "enableAnalytics": False},
        "flags": [], "env": {},
        "note": "quietStartup + enableInstallTelemetry:false + enableAnalytics:false are documented.",
    },
    "aider": {
        "doc": "https://aider.chat/docs/llms/openai-compat.html",
        "settings": {}, "flags": [], "env": {"DO_NOT_TRACK": "1"},
        "note": "No documented branding flag; output filtering is the authority.",
    },
    "cline": {
        "doc": "https://docs.cline.bot/provider-config/openai-compatible",
        "settings": {}, "flags": [], "env": {"DO_NOT_TRACK": "1"},
        "note": "No standalone CLI; TUI drives the generic OpenAI-compatible path.",
    },
    "continue": {
        "doc": "https://docs.continue.dev/customize/model-providers/top-level/openai",
        "settings": {}, "flags": [], "env": {"DO_NOT_TRACK": "1"},
        "note": "No standalone CLI; TUI drives the generic OpenAI-compatible path.",
    },
}
DEFAULT_HARNESS = "opencode"                 # cleanest documented white-label path
HARNESS_NAMES = PUBLIC_HARNESS_CHOICES

# Lines that are harness surface rather than Veystrix output. Conservative: whole-line matches.
# Only harness UI chrome is filtered. Patterns are ANCHORED to the line start:
# a model answer containing e.g. "sign in" or "continue" must survive intact
# (Sonnet finding 23).
_BRANDING = [
    # Require a harness-style separator (colon / version / `vN`) after the
    # name, so ordinary model text like "Pi is 3.14" survives (Sonnet).
    re.compile(r"(?i)^\s*(claude code|opencode|codex|qwen code|kimi code|hermes|pi|aider|cline)\s*(?:[:\-—]|v?\d)[^\n]{0,60}$"),
    re.compile(r"(?i)^\s*(update available|new version available|auto-?update|changelog:|release notes:)\b"),
    re.compile(r"(?i)^\s*(telemetry|analytics|anonymous usage|usage statistics|error reporting)\s*[:=]"),
    re.compile(r"(?i)^\s*(sign in|log in|oauth|subscription)\s*(to|with|required|now)"),
    re.compile(r"(?i)^\s*(welcome to|version\s*v?\d|tips:|press ctrl|type /help|╭|╰|─{4,})\s*"),
]
_TOOL_LINE = re.compile(
    r"(?i)\b(read_file|edit_file|write_file|run_shell|bash|grep|find|ls|apply_patch|str_replace_editor)\b"
    r"[\s(\{]*([^\n)}]{0,120})")
# "continue" is excluded: it is an ordinary English word, not a brand.
_NAMES = re.compile(r"(?i)\b(claude code|opencode|codex|qwen code|kimi code|hermes|aider|cline)\b")


def banner() -> list[str]:
    return ["", "  ◆ V E Y S T R I X  ·  local coding agent", "  ─────────────────────────────────",
            "  /config  /cache  /doctor  /update  /quit   Ctrl-C to cancel", ""]


def passthrough_banner(harness: str, *, engine: str, model: str | None, running: bool,
                       port: int = 8080) -> list[str]:
    """What Veystrix prints before a harness without a structured mode takes the terminal.

    Only facts that are true here: no token counters (Veystrix cannot see the harness's
    traffic) and no Veystrix slash commands (the harness owns the keyboard)."""
    from veystrix import model_identity
    from veystrix.tool import renderer as R
    from veystrix.tool.harnesses import family_of
    theme = R.Theme.detect(tty=sys.stdout.isatty())
    label = HARNESS_LABELS.get(harness, harness)
    family = family_of(model)
    name = model_identity.for_family(family).display_name if family else (model or "no model")
    dot = theme.sym("dot")
    rule = ("─" if not theme.ascii_only else "-") * 52
    state = (theme.paint("ok", f"{theme.sym('on')} ready on 127.0.0.1:{port}") if running
             else theme.paint("warn", f"{theme.sym('off')} engine stopped"))
    lines = ["",
             "  " + theme.paint("brand", theme.bold(f"{theme.sym('brand')} V E Y S T R I X"))
             + "  ·  local coding agent",
             "  " + theme.paint("dim", rule),
             f"  {theme.paint('dim', 'model')} {name} "
             f"{dot} {theme.paint('dim', 'harness')} {label} {dot} {state}"]
    if not running:
        lines.append("  " + theme.paint("warn", "start the engine with `veystrix start` in another terminal; "
                                                "requests fail until it is running"))
    lines += ["  " + theme.paint("dim", f"{label} opens in its own screen below; quit it to return. "
                                        "The full Veystrix screen runs on OpenCode or Codex:"),
              "  " + theme.paint("dim", "veystrix tui --harness opencode  (or /settings → Connect a harness)"),
              ""]
    return lines


def branding_clean(line: str, harness: str | None = None) -> str | None:
    """Drop harness surface lines; rewrite surviving harness names to Veystrix."""
    stripped = line.rstrip("\n")
    if not stripped.strip():
        return None
    for pattern in _BRANDING:
        if pattern.search(stripped):
            return None
    cleaned = _NAMES.sub("Veystrix", stripped)
    return cleaned if cleaned.strip() else None


def format_tool_call(name: str, detail: str = "") -> str:
    detail = detail.strip().strip("'\"")
    return f"  ◆ Veystrix · {name}" + (f"  {detail[:80]}" if detail else "")


def whitelabel_environment(harness: str, port: int, model_id: str, key: str) -> dict:
    """Endpoint + suppression env for a harness, entirely composed from cited settings."""
    if harness not in WHITELABEL:
        raise ValueError(f"unknown harness {harness!r}")
    spec = WHITELABEL[harness]
    env = run_environment(harness, port, model_id, key)
    env.update(spec["env"])
    env["DO_NOT_TRACK"] = "1"
    env["VEYSTRIX_BRANDED_UI"] = "1"
    return env


def _filter_lines(text: str, harness: str) -> list[str]:
    out = []
    for line in text.splitlines():
        match = _TOOL_LINE.search(line)
        if match:
            out.append(format_tool_call(match.group(1), match.group(2)))
        else:
            cleaned = branding_clean(line, harness)
            if cleaned is not None:
                out.append(cleaned)
    return out


# How each harness reopens a saved session (checked against the installed CLIs' --help).
RESUME_ARGS = {
    "claude-code": lambda sid: ["--resume", sid], "hermes": lambda sid: ["--resume", sid],
    "codex": lambda sid: ["resume", sid], "opencode": lambda sid: ["--session", sid],
    "kimi": lambda sid: ["--session", sid], "pi": lambda sid: ["--session", sid],
    "qwen-code": lambda sid: ["--resume", sid],
}


def launch_command(harness: str, model_id: str, resume: str | None = None) -> list[str]:
    command = run_command(harness, model_id)
    if resume:
        build = RESUME_ARGS.get(harness)
        if build is None:
            raise SystemExit(f"veystrix: {harness} cannot reopen a saved session; start without --resume")
        command = [*command, *build(resume)]
    command = [*command, *WHITELABEL[harness]["flags"]]
    return command


def run_once(command: list[str], env: dict, *, harness: str, cancel_after: float | None = None,
             timeout: float = 10.0, stdin: str | None = None) -> dict:
    """Run a harness process, filter its output into Veystrix-branded lines.

    Used by the interactive TUI and by offline tests. No TTY is required.
    """
    process = subprocess.Popen(command, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               env={**os.environ, **env})
    if stdin is not None:
        try:
            process.stdin.write(stdin)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    output = []
    cancelled = False
    if process.poll() is not None:
        try:
            process.stdout.close()
        except OSError:
            pass
        raise SystemExit(f"veystrix: harness command could not start: {' '.join(command)}")
    deadline = time.monotonic() + timeout
    cancel_at = time.monotonic() + cancel_after if cancel_after is not None else None
    try:
        while time.monotonic() < deadline:
            if cancel_at is not None and time.monotonic() >= cancel_at and process.poll() is None:
                process.send_signal(signal.SIGINT)
                cancelled = True
                cancel_at = None
            ready, _, _ = select.select([process.stdout], [], [], 0.05)
            if not ready:
                continue
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue
            filtered = _filter_lines(line, harness)
            output.extend(filtered)
    finally:
        try:
            process.stdout.close()
        except OSError:
            pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
    if cancelled:
        output.append("  ◆ Veystrix · generation cancelled")
    return {"lines": output, "exit_code": process.returncode, "cancelled": cancelled}


def slash(command: str, *, home: Path, port: int) -> list[str]:
    """Veystrix-native commands available inside the branded UI."""
    from veystrix.tool.commands import cmd_cache, cmd_config, cmd_doctor
    name, _, rest = command.strip().partition(" ")
    config = Config(home)
    key = LocalKeyStore(home)
    if name in ("/config", "config"):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cmd_config(rest.split() if rest else [], home, config, key)
        return buffer.getvalue().splitlines()
    if name in ("/cache", "cache"):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cmd_cache(rest.split() if rest else ["status"], home, config)
        return buffer.getvalue().splitlines()
    if name in ("/doctor", "doctor"):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cmd_doctor([], home, config, "python")
        return buffer.getvalue().splitlines()
    if name in ("/engine", "engine"):
        from veystrix.tool.swap import describe, swap
        target = rest.strip().lower()
        if not target:
            report = describe(home)
            lines = [f"  ◆ Veystrix · engine={report['selected']} "
                     f"(active: {report['active'] or 'not running'}) model={report['model']}"]
            for engine_name, info in report["available"].items():
                lines.append(f"      {'ok' if info['available'] else 'no'} {engine_name}"
                             + (f" - {info['reason']}" if info["reason"] else ""))
            lines.append("      switch: /engine cpp  or  /engine python")
            return lines
        notes: list[str] = []
        try:
            outcome = swap(home, engine=target, echo=lambda m: notes.append("  ◆ Veystrix · " + m))
        except (ValueError, SystemExit) as exc:
            return [*notes, f"  ◆ Veystrix · cannot switch: {str(exc).removeprefix('veystrix: ')}"]
        return [*notes, f"  ◆ Veystrix · engine is now {outcome['engine']} / {outcome['model']}"
                + (" (restarted)" if outcome["restarted"] else "")]
    if name in ("/model", "model"):
        from veystrix.tool.swap import MODELS, swap
        target = rest.strip().lower()
        if target not in MODELS:
            return [f"  ◆ Veystrix · usage: /model {' | '.join(MODELS)}"]
        notes = []
        try:
            outcome = swap(home, model=target, echo=lambda m: notes.append("  ◆ Veystrix · " + m))
        except (ValueError, SystemExit) as exc:
            return [*notes, f"  ◆ Veystrix · cannot switch: {str(exc).removeprefix('veystrix: ')}"]
        return [*notes, f"  ◆ Veystrix · model is now {outcome['model']} on {outcome['engine']}"
                + (" (restarted)" if outcome["restarted"] else "")]
    if name in ("/theme", "theme"):
        from veystrix.tool import renderer as R
        theme = R.Theme.detect(tty=True)
        return [f"  ◆ Veystrix · theme color={theme.color} ascii={theme.ascii_only} "
                f"(NO_COLOR / VEYSTRIX_UI_ASCII=1 to override)"]
    if name in ("/speed", "speed"):
        from veystrix import speed as speed_mod
        parts = rest.split()
        if not parts:
            return speed_mod.describe(config.get("speed_mode"), config.get("speed_knobs"),
                                      config.get("model"), os.environ)
        if parts[0] in speed_mod.MODES:
            config.set("speed_mode", parts[0])
            return speed_mod.describe(config.get("speed_mode"), config.get("speed_knobs"),
                                      config.get("model"), os.environ)
        if len(parts) == 2 and parts[0] in speed_mod.KNOBS:
            try:
                value = int(parts[1])
            except ValueError:
                return [f"  ◆ Veystrix · invalid value {parts[1]!r}; expected an integer"]
            knobs = dict(config.get("speed_knobs") or {})
            knobs[parts[0]] = value
            config.values["speed_knobs"] = knobs
            config.values["speed_mode"] = "custom"
            config.save()
            return speed_mod.describe("custom", knobs, config.get("model"), os.environ)
        return ["  ◆ Veystrix · usage: /speed [max|recipe|custom] | /speed <knob> <value>",
                "      knobs: " + ", ".join(speed_mod.KNOBS)]
    if name in ("/status", "status"):
        from veystrix.tool import renderer as R
        from veystrix.tool.endpoint import describe
        theme = R.Theme.detect(tty=True)
        info = describe(home, port=port)
        lines = [R.status_line(R.Status(
            engine=info.get("engine") or os.environ.get("VEYSTRIX_ENGINE") or "python",
            model=info.get("served_model_id") or os.environ.get("VEYSTRIX_MODEL", ""),
            runtime=info.get("runtime") or "", kv=info.get("kv_mode") or "",
            window=int(info.get("context_ceiling") or 0), used=int(info.get("context_in_use") or 0)),
            theme, width=100)]
        lines.append(f"  ◆ Veystrix · {info['display_name']} · "
                     f"KV {info['kv_mode']}" + (f" ({info['kv_bytes_per_token']:,} B/token)"
                                                if info.get("kv_bytes_per_token") else "")
                     + f" · context {info.get('context_in_use') or 0:,}/{int(info['context_ceiling']):,}")
        return lines
    if name in ("/endpoint", "endpoint"):
        from veystrix.tool.endpoint import describe, render
        return render(describe(home, port=port)).splitlines()
    if name in ("/kv", "kv"):
        from veystrix.kv_policy import CHOICES
        target = rest.strip().lower()
        if not target:
            return [f"  ◆ Veystrix · KV policy {config.get('kv')} (modes: {', '.join(CHOICES)}); "
                    "set with /kv MODE, applies at the next start"]
        if target not in CHOICES:
            return [f"  ◆ Veystrix · unknown KV mode {target}; use {', '.join(CHOICES)}"]
        config.values["kv"] = target
        config.save()
        return [f"  ◆ Veystrix · KV policy saved: {target} (restart the endpoint to apply: /stop then /start)"]
    if name in ("/reasoning", "reasoning"):
        return ["  ◇ Veystrix · reasoning pane toggled (D6 renderer)"]
    if name in ("/compact", "compact"):
        return ["  ◆ Veystrix · compact: drop or summarise old tool results on the client"]
    if name in ("/start", "start"):
        from veystrix.tool.swap import ensure_engine
        notes: list[str] = []
        try:
            result = ensure_engine(home, port=port, echo=lambda m: notes.append("  ◆ Veystrix · " + m))
        except (ValueError, SystemExit) as exc:
            return [*notes, f"  ◆ Veystrix · cannot start: {str(exc).removeprefix('veystrix: ')}"]
        if not result.get("started"):
            return [f"  ◆ Veystrix · already running: {result.get('engine')} / {result.get('model')}"]
        return [*notes, f"  ◆ Veystrix · ready: {result.get('engine')} / {result.get('model')}"]
    if name in ("/stop", "stop"):
        from veystrix.tool.supervisor import stop as _stop_engine
        _stop_engine(home)
        return ["  ◆ Veystrix · engine stopped"]
    if name in ("/restart", "restart"):
        from veystrix.tool.supervisor import stop as _stop_engine
        from veystrix.tool.swap import ensure_engine
        _stop_engine(home)
        notes = []
        try:
            result = ensure_engine(home, port=port, echo=lambda m: notes.append("  ◆ Veystrix · " + m))
        except (ValueError, SystemExit) as exc:
            return [*notes, f"  ◆ Veystrix · cannot restart: {str(exc).removeprefix('veystrix: ')}"]
        return [*notes, f"  ◆ Veystrix · restarted: {result.get('model')}"]
    if name in ("/update", "update"):
        # Same trust chain as `veystrix update`: signed SHA256SUMS then archive.
        # Homebrew installs are upgraded by brew itself; everything else is the
        # curl/script install handled by the signed updater.
        from veystrix.tool.updater import update
        check_only = rest.strip() in ("--check", "check")
        lines: list[str] = []
        if os.environ.get("VEYSTRIX_INSTALLER") == "brew" and not check_only:
            import shutil as _shutil
            if _shutil.which("brew") is None:
                return ["  ◆ Veystrix · Homebrew install detected but `brew` is not on PATH"]
            for args in (("update",), ("upgrade", "veystrix")):
                try:
                    proc = subprocess.run(["brew", *args], capture_output=True, text=True, timeout=300)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    return ["  ◆ Veystrix · brew " + " ".join(args) + " failed: " + str(exc)]
                if proc.returncode != 0:
                    tail = [ln.strip() for ln in (proc.stdout + "\n" + proc.stderr).splitlines() if ln.strip()][-4:]
                    return ["  ◆ Veystrix · brew " + " ".join(args) + " failed:", *["    " + ln for ln in tail]]
            return ["  ◆ Veystrix · Homebrew upgrade finished; restart the terminal tool"]
        rc = update(home, check_only=check_only,
                    echo=lambda m: lines.append("  ◆ Veystrix · " + m))
        return lines if rc == 0 else [*lines, "  ◆ Veystrix · update did not complete"]
    if name in ("/port-check", "port-check"):
        proc = subprocess.run(
            [sys.executable, "-m", "veystrix", "models", "port-check", "--tiny"],
            capture_output=True, text=True, timeout=180,
        )
        tail = [ln.strip() for ln in (proc.stdout + "\n" + proc.stderr).splitlines() if ln.strip()][-6:]
        if proc.returncode == 0:
            return ["  ◆ Veystrix · port-check --tiny passed (engine verified)", *["    " + ln for ln in tail[-2:]]]
        return ["  ◆ Veystrix · port-check --tiny failed:", *["    " + ln for ln in tail]]
    if name in ("/quit", "quit", "/exit", "exit"):
        return ["bye"]
    return [f"  ◆ Veystrix · unknown command {name}; use /settings, /start, /stop, /model, /download, "
            "/effort, /harness, /config, /cache, /doctor, /kv, /endpoint, /theme, /status, /speed, /update, /quit"]


# =====================================================================================
# D6 full UI: Veystrix owns the screen and drives the harness headless through
# harness_adapters (normalised events). The legacy --once and passthrough paths above and
# below are unchanged; this runs only for a structured adapter on a real TTY.
# =====================================================================================
FRAME_INTERVAL = 0.016      # event-driven redraw cap (~60 fps); keypresses redraw immediately
SPINNER_INTERVAL = 0.1
STATUS_INTERVAL = 2.0
BUSY_STATUS_INTERVAL = 0.4
CANCEL_GRACE = 2.0          # a hung harness is force-released this long after Ctrl-C
QUIT_WINDOW = 2.0
ALT_ENTER = "\033[?1049h\033[2J\033[?25l\033[?2004h"
ALT_LEAVE = "\033[?2004l\033[?25h\033[?1049l"
_SEQ = {"[A": "up", "[B": "down", "[C": "right", "[D": "left", "[5~": "pgup", "[6~": "pgdn",
        "[H": "home", "[F": "end", "[1~": "home", "[4~": "end", "OH": "home", "OF": "end",
        "[3~": "delete", "OA": "up", "OB": "down",
        "[13;2u": "newline", "[27;2;13~": "newline", "\r": "newline", "\n": "newline"}
_CTRL = {"\r": "enter", "\n": "enter", "\x7f": "backspace", "\x08": "backspace", "\x03": "ctrl-c",
         "\x01": "home", "\x05": "end",
         "\x04": "ctrl-d", "\x0b": "ctrl-k", "\x0c": "ctrl-l", "\x12": "ctrl-r", "\x14": "ctrl-t",
         "\x15": "ctrl-u", "\t": "tab"}
SLASH_COMMANDS = (
    "/new", "/restart", "/settings", "/start", "/stop", "/update", "/speed", "/port-check", "/model", "/models", "/effort", "/download", "/harness",
    "/kv", "/status",
    "/endpoint", "/config", "/cache", "/doctor", "/resume", "/reasoning",
    "/tools", "/compact", "/clear", "/theme", "/help", "/quit",
)
_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist", ".mypy_cache"}


class KeyParser:
    """Terminal bytes (as text) -> normalised key tokens. A pasted block (bracketed paste)
    becomes one ``paste:<text>`` token so embedded newlines never submit the prompt."""

    def __init__(self) -> None:
        self.pending = ""

    def feed(self, text: str) -> list[str]:
        text, self.pending = self.pending + text, ""
        keys: list[str] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "\x1b":
                if text.startswith("\x1b[200~", i):
                    end = text.find("\x1b[201~", i)
                    if end < 0:
                        self.pending = text[i:]
                        break
                    keys.append("paste:" + text[i + 6:end])
                    i = end + 6
                    continue
                rest = text[i + 1:]
                match = next((seq for seq in sorted(_SEQ, key=len, reverse=True)
                              if rest.startswith(seq)), None)
                if match:
                    keys.append(_SEQ[match])
                    i += 1 + len(match)
                elif rest and any(seq.startswith(rest) for seq in _SEQ):
                    self.pending = text[i:]  # sequence split across two reads
                    break
                else:
                    keys.append("esc")
                    i += 1
            elif ch in _CTRL:
                keys.append(_CTRL[ch])
                i += 1
            elif ch >= " ":
                keys.append(ch)
                i += 1
            else:
                i += 1
        return keys


def sanitize_text(text: str) -> str:
    """Hide harness surface text (banner/update/telemetry/sign-in) from adapter errors."""
    kept = [cleaned for line in text.splitlines()
            if (cleaned := branding_clean(line)) is not None]
    return "\n".join(kept)


class SessionIndex:
    """Veystrix-owned record of the last session per harness+cwd (powers ``--resume``)."""

    def __init__(self, home: Path):
        self.path = Path(home) / "ui_sessions.json"

    def load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def last(self, harness: str, cwd: str) -> str | None:
        return (self.load().get(harness) or {}).get(cwd)

    def save(self, harness: str, cwd: str, session_id: str | None) -> None:
        if not session_id:
            return
        data = self.load()
        data.setdefault(harness, {})[cwd] = session_id
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, self.path)
        except OSError:
            pass


def fetch_engine_status(port: int, key: str = "", timeout: float = 0.3) -> dict | None:
    """GET http://127.0.0.1:<port>/veystrix/status (loopback only). None when unreachable."""
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(f"http://127.0.0.1:{int(port)}/veystrix/status",
                                 headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with opener.open(req, timeout=timeout) as resp:  # noqa: S310 (loopback)
            return json.loads(resp.read())
    except (OSError, ValueError):
        return None


class StatusPoller:
    """Background poll so a slow /veystrix/status never delays a keypress."""

    def __init__(self, fetch, interval: float = BUSY_STATUS_INTERVAL, max_misses: int = 2):
        import threading
        self.fetch, self.interval, self.latest = fetch, interval, None
        self.max_misses = max(1, int(max_misses))
        self._misses = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="veystrix-status")

    def _poll_once(self) -> None:
        data = self.fetch()
        if data is not None:
            self._misses = 0
            self.latest = data
        else:
            self._misses += 1
            if self._misses >= self.max_misses:
                self.latest = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self.interval)

    def start(self) -> "StatusPoller":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def __call__(self):
        return self.latest


def friendly_model(model_id: str) -> str:
    """Display name for a served id or family key (``veystrix-ornith-<hash>`` -> ``Ornith 1.5 35B-A3B``)."""
    from veystrix import model_identity
    from veystrix.tool.harnesses import family_of
    family = family_of(model_id)
    if family is None:
        return model_id
    return model_identity.for_family(family).display_name.removeprefix("Veystrix ")


def apply_engine_status(status, data: dict | None) -> bool:
    """Copy /veystrix/status into the renderer Status; True when anything visible changed."""
    if not data:
        return False
    before = repr(status)
    status.engine = str(data.get("engine") or status.engine)
    if data.get("model"):
        status.model_id = str(data["model"])
        status.model = friendly_model(status.model_id)
    if data.get("context_window"):
        status.window = int(data["context_window"])
    if data.get("used_tokens") is not None:
        status.used = int(data["used_tokens"])
    if data.get("cache_state") in ("cold", "cached"):
        status.cache = data["cache_state"]
    if data.get("sampling"):
        status.sampling = str(data["sampling"])
    rate = data.get("decode_tokens_per_second")
    # 0 is a real reading (idle engine); accepting it clears a stale rate. The
    # value is always the engine's measured decode speed, never an SSE chunk
    # estimate, so every harness shows the same number for the same model.
    if isinstance(rate, (int, float)) and not isinstance(rate, bool):
        status.tok_s = float(rate)
    activity = data.get("activity") if isinstance(data.get("activity"), dict) else {}
    if "active" in data or "stage" in data or activity:
        active = bool(data.get("active") if "active" in data else activity.get("active"))
        stage = str(data.get("stage") or activity.get("stage") or "") if active else ""
        if stage in ("idle", "none"):
            stage = ""
        status.stage = stage
        if stage == "prefill":
            done = data.get("prefill_done") if data.get("prefill_done") is not None else activity.get("prefill_done")
            total = data.get("prefill_total") if data.get("prefill_total") is not None else activity.get("prefill_total")
            if total is None:
                total = activity.get("prompt_tokens")
            status.prefill_done = int(done or 0)
            status.prefill_total = int(total or 0)
        else:
            status.prefill_done = 0
            status.prefill_total = 0
    status.thermal = int(data.get("thermal") or 0)
    runtime = data.get("runtime") if isinstance(data.get("runtime"), dict) else {}
    if runtime.get("runtime") in ("generic", "custom"):
        status.runtime = runtime["runtime"]
    kv = runtime.get("kv") if isinstance(runtime.get("kv"), dict) else {}
    if kv.get("mode"):
        status.kv = str(kv["mode"])
    return repr(status) != before


HARNESS_LABELS = {
    "direct": "Built-in",
    "opencode": "OpenCode", "claude-code": "Claude Code", "codex": "Codex", "qwen-code": "Qwen Code",
    "deepseek": "DeepSeek", "kimi": "Kimi Code", "hermes": "Hermes", "pi": "Pi", "aider": "Aider",
    "cline": "Cline", "continue": "Continue",
}
KV_DESCRIPTIONS = {
    "auto": "Automatic (chosen by memory plan)",
    "bf16": "Exact full precision (highest memory)",
    "int8": "8-bit keys and values",
    "k8v4": "8-bit keys, 4-bit values",
    "int4": "4-bit keys and values (lowest memory)",
    "tiered": "Exact recent window + 8-bit history",
    "tiered-k8v4": "Exact recent window + 8-bit K / 4-bit V",
    "tiered4": "Exact recent window + 4-bit history",
    "tiered4+": "Exact recent window + 4-bit history (extended)",
    "sparse-paged": "Exact prefix/recent + Quest top-32 sparse INT4 history",
}
EFFORT_DESCRIPTIONS = {
    "auto": "Let the harness / model decide",
    "off": "No thinking (fastest answers)",
    "low": "Light reasoning",
    "medium": "Balanced reasoning (default)",
    "high": "Deep reasoning",
    "xhigh": "Maximum reasoning depth",
}


def harness_inventory(home: Path, active: str = "", include_locked: bool = False) -> list[dict]:
    """Every harness `connect` knows, with install / connection state (cheap: PATH + one read)."""
    import shutil
    from veystrix.tool import harnesses as H
    from veystrix.tool.harness_adapters import structured_available
    items = []
    names = H.NAMES if include_locked else H.PUBLIC_HARNESS_CHOICES
    for name in names:
        try:
            binary = H.run_command(name, "veystrix")[0]
        except ValueError:
            binary = None                       # editor extension: config snippet only
        items.append({
            "key": name, "label": HARNESS_LABELS.get(name, name),
            "installed": bool(binary and shutil.which(binary)), "config_only": binary is None,
            "connected": H.is_connected(name, home), "active": name == active,
            "structured": structured_available(name),
        })
    items.append({
        "key": "direct", "label": HARNESS_LABELS["direct"],
        "installed": True, "config_only": False,
        "connected": True, "active": active == "direct",
        "structured": True,
    })
    return items


def _dir_size(path: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def model_inventory(root: Path | None = None, include_locked: bool = False) -> list[dict]:
    """Every known model family with its local state: installed (complete), partial or absent.

    Headers only (the family's structural check); nothing is loaded or downloaded."""
    from veystrix import checkpoints, model_identity
    from veystrix.model_pull import resolve_target
    root = Path(root) if root is not None else checkpoints.model_root()
    items = []
    for family, ident in model_identity.IDENTITIES.items():
        if ident.locked and not include_locked:
            continue
        try:
            complete = list(checkpoints.installed_for_family(family, root))
        except Exception:                       # e.g. a verifier unavailable in this build
            complete = []
        dest = resolve_target(family, root, allow_locked=True)[1] if ident.download_repo else None
        resolved = {p.resolve() for p in complete}
        if dest is not None and dest.resolve() in resolved:
            path = dest
        else:
            path = complete[0] if complete else dest
        installed = bool(complete)
        partial = (not installed and dest is not None and dest.is_dir()
                   and any(p.name != ".cache" for p in dest.iterdir()))
        items.append({
            "key": family, "name": ident.display_name.removeprefix("Veystrix "),
            "context": ident.context_limit, "repo": ident.download_repo,
            "installed": installed, "partial": partial,
            "path": str(path) if path is not None else "",
            "size": _dir_size(path) if (installed or partial) and path is not None else 0,
            "locked": bool(ident.locked),
        })
    return items


def model_matches(item: dict, query: str) -> bool:
    """Search-bar filter: every word must appear in the name, family, repo or state."""
    words = (query or "").lower().split()
    if not words:
        return True
    state = "installed" if item.get("installed") else "partial" if item.get("partial") else "download"
    hay = " ".join(str(item.get(k) or "") for k in ("name", "key", "repo")).lower() + " " + state
    return all(w in hay for w in words)


def match_model(name: str, include_locked: bool = False) -> str | None:
    """Family key for a family, checkpoint alias or served id (case-insensitive)."""
    from veystrix import model_identity
    key = name.strip().lower()
    for family, ident in model_identity.IDENTITIES.items():
        if ident.locked and not include_locked:
            continue
        if key in (family, ident.checkpoint_alias.lower(), ident.served_id.lower()) and key:
            return family
    return None


class UILoop:
    """The input/event loop. All I/O is injected (``write``, ``size``) and time is passed in,
    so the same code runs on a real TTY and under the synthetic clock in tests.

    ``step(now, keys)`` is one iteration: apply keys, drain adapter events, tick the spinner,
    then redraw only the rows that changed. Keypresses redraw immediately; event-driven
    changes are coalesced to FRAME_INTERVAL but never discarded (no dropped content).

    Anything slow (engine start/stop, port-check, model scans, downloads, deletes, harness
    connects) runs on a daemon thread and reports back through ``_bg_events``; the loop
    only ever applies the result, so a keypress never waits on disk or network.
    """

    def __init__(self, adapter, *, write, size, theme=None, status=None, home: Path | None = None,
                 port: int = 8080, harness: str = DEFAULT_HARNESS, cwd: str = "",
                 status_fn=None, slash_fn=None, index: SessionIndex | None = None,
                 running: bool | None = None, adapter_factory=None, inventory_fn=None,
                 harness_fn=None, pull_fn=None):
        from veystrix.tool import renderer as R
        self.R = R
        self.adapter, self.write, self.size = adapter, write, size
        self.theme = theme or R.Theme.detect(tty=True)
        cols, rows = size()
        self.state = R.UIState(cols=cols, rows=rows, status=status or R.Status(), cwd=cwd, sanitize=sanitize_text,
                               harness=harness, harness_label=HARNESS_LABELS.get(harness, harness))
        try:
            saved = Config(home or home_dir())
            self.state.status.effort = saved.get("effort")
            self.state.show_reasoning = bool(saved.get("show_reasoning"))
        except Exception:
            pass
        self.home, self.port, self.harness, self.cwd = home, port, harness, cwd
        self.status_fn, self.slash_fn, self.index = status_fn, slash_fn or slash, index
        self.adapter_factory = adapter_factory          # harness -> started adapter | None
        self.inventory_fn = inventory_fn or (lambda: model_inventory())
        self.harness_fn = harness_fn or harness_inventory
        self.pull_fn = pull_fn                          # None -> veystrix.model_pull.pull
        self._engine_running: bool | None = running
        self._bg_events: queue.Queue = queue.Queue()
        self._bg_busy = False
        self._bg_action = ""
        self._update_busy = False
        self._needs_port_check = False
        self._inventory: list[dict] | None = None
        self._scanning = False
        self._after_scan: tuple | None = None
        self._downloading: str | None = None
        self._connecting = False
        self._overlay_return: str | None = None
        self._settings_index: int = 0
        self._now = 0.0
        self.done = False
        self.prev: list[str] | None = None
        self.dirty = True
        self.last_frame_at = -1e9
        self.last_spin = 0.0
        self.last_status = -1e9
        self.cancel_at: float | None = None
        self.last_ctrl_c = -1e9
        self.history: list[str] = []
        self.hist_pos = 0
        self.frames = 0
        self.last_frame: list[str] = []

    def _home(self) -> Path:
        return self.home or home_dir()

    def _cursor(self) -> int:
        st = self.state
        return len(st.input) if st.cursor is None else max(0, min(st.cursor, len(st.input)))

    def _set_cursor(self, pos: int) -> None:
        st = self.state
        pos = max(0, min(pos, len(st.input)))
        st.cursor = None if pos == len(st.input) else pos

    def _insert_text(self, chunk: str) -> None:
        st = self.state
        pos = self._cursor()
        st.input = st.input[:pos] + chunk + st.input[pos:]
        self._set_cursor(pos + len(chunk))

    def _complete_tab(self) -> None:
        st = self.state
        pos = self._cursor()
        before = st.input[:pos]
        start = pos
        while start > 0 and not before[start - 1].isspace():
            start -= 1
        token = before[start:pos]
        if not token:
            return
        matches: list[tuple[str, str, int, int]] = []
        if token.startswith("/") and start == 0:
            q = token.lower()
            for cmd in SLASH_COMMANDS:
                if cmd.startswith(q) or q[1:] in cmd[1:]:
                    matches.append((cmd, cmd + " ", start, pos))
        elif token.startswith("@"):
            q = token[1:].lower()
            root = Path(self.cwd or os.getcwd())
            found: list[str] = []
            try:
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
                    rel_dir = os.path.relpath(dirpath, root)
                    for fname in sorted(filenames):
                        rel = fname if rel_dir == "." else os.path.join(rel_dir, fname)
                        if not q or q in rel.lower():
                            found.append(rel)
                        if len(found) >= 200:
                            break
                    if len(found) >= 200:
                        break
            except OSError:
                found = []
            found.sort(key=lambda p: (not p.lower().startswith(q), len(p), p))
            for rel in found[:24]:
                matches.append((f"@{rel}", f"@{rel} ", start, pos))
        else:
            return
        if not matches:
            st.notice = f"no completions for {token}"
            return
        if len(matches) == 1:
            _, repl, s, e = matches[0]
            st.input = st.input[:s] + repl + st.input[e:]
            self._set_cursor(s + len(repl))
            st.notice = ""
            return
        st.completions = matches
        st.overlay, st.index = "complete", 0

    # ---- input
    def handle_key(self, key: str, now: float) -> None:
        st = self.state
        self._now = st.now = max(now, st.now)
        self.dirty = True
        if key.startswith("paste:"):
            if st.overlay == "palette":
                st.query += key[6:].replace("\n", " ")
            elif st.permission is None and st.overlay is None:
                self._insert_text(key[6:].replace("\r\n", "\n").replace("\r", "\n"))
            return
        if key == "ctrl-l":
            self.prev = None
            return
        if st.permission is not None:
            self._permission_key(key, now)
            return
        if st.overlay:
            self._overlay_key(key, now)
            return
        if key == "ctrl-c":
            self._ctrl_c(now)
        elif key == "ctrl-k":
            st.overlay, st.query, st.index = "palette", "", 0
        elif key == "ctrl-r":
            self._set_show_reasoning(not st.show_reasoning)
        elif key == "ctrl-t":
            self._toggle_tools()
        elif key == "ctrl-u":
            st.input, st.cursor = "", None
        elif key == "ctrl-d":
            if not st.input:
                self.done = True
        elif key == "enter":
            self.submit(now)
        elif key == "newline":
            self._insert_text("\n")
        elif key == "tab":
            self._complete_tab()
        elif key == "left":
            self._set_cursor(self._cursor() - 1)
        elif key == "right":
            self._set_cursor(self._cursor() + 1)
        elif key == "home":
            self._set_cursor(0)
        elif key == "end":
            self._set_cursor(len(st.input))
        elif key == "backspace":
            pos = self._cursor()
            if pos > 0:
                st.input = st.input[:pos - 1] + st.input[pos:]
                self._set_cursor(pos - 1)
        elif key == "delete":
            pos = self._cursor()
            if pos < len(st.input):
                st.input = st.input[:pos] + st.input[pos + 1:]
                self._set_cursor(pos)
        elif key == "pgup":
            st.scroll += max((st.rows - 5) // 2, 1)
        elif key == "pgdn":
            st.scroll = max(st.scroll - max((st.rows - 5) // 2, 1), 0)
        elif key in ("up", "down"):
            self._history(key)
        elif key == "esc":
            st.input, st.cursor, st.scroll = "", None, 0
        elif key == "?" and not st.input:
            st.overlay = "help"
        elif len(key) == 1:
            self._insert_text(key)
            st.notice = ""

    def _history(self, key: str) -> None:
        if not self.history:
            return
        self.hist_pos = (min(self.hist_pos + 1, len(self.history)) if key == "up"
                         else max(self.hist_pos - 1, 0))
        self.state.input = "" if self.hist_pos == 0 else self.history[-self.hist_pos]
        self.state.cursor = None

    def _toggle_tools(self) -> None:
        st = self.state
        tools = [e for e in st.entries if e.kind == "tool"]
        st.tools_collapsed = any(not e.collapsed for e in tools) if tools else not st.tools_collapsed
        for entry in tools:
            entry.collapsed = st.tools_collapsed

    def _ctrl_c(self, now: float) -> None:
        st = self.state
        if st.busy:
            st.activity, self.cancel_at = "cancelling", now
            try:
                self.adapter.cancel()
            except Exception as exc:  # adapter failure must not kill the screen
                self.R.apply_event(st, _error_event(exc), now)
        elif st.input:
            st.input, st.cursor = "", None
            return                      # clearing the line is not the first half of "press twice"
        elif now - self.last_ctrl_c <= QUIT_WINDOW:
            self.done = True
        else:
            st.notice = "press Ctrl-C again to quit"
        self.last_ctrl_c = now

    def _permission_key(self, key: str, now: float) -> None:
        st = self.state
        choice = {"y": "once", "a": "always", "n": "reject", "esc": "reject"}.get(key)
        if key == "ctrl-c":
            choice = "reject"
            self._ctrl_c(now)
        if choice is None:
            return
        try:
            self.adapter.reply_permission(str(st.permission.get("id", "")), choice)
        except Exception as exc:
            self.R.apply_event(st, _error_event(exc), now)
        st.permission = None
        st.notice = f"permission: {choice}"

    # ---- overlays
    def _open_overlay(self, name: str, *, back: str | None = None) -> None:
        st = self.state
        if back == "settings" and st.overlay == "settings":
            self._settings_index = st.index
        self._overlay_return = back
        st.overlay, st.index, st.query = name, 0, ""

    def _close_overlay(self, *, back: bool = True) -> None:
        st = self.state
        target = self._overlay_return if back else None
        self._overlay_return = None
        st.completions = []
        st.confirm = None
        st.overlay = None
        if target == "settings":
            self._open_settings(index=self._settings_index)
        elif target in ("models", "model"):
            self._open_models(manager=target == "models", back=None)

    def _overlay_key(self, key: str, now: float) -> None:
        st = self.state
        ov = st.overlay
        if ov == "confirm":
            self._confirm_key(key, now)
            return
        if key in ("esc", "ctrl-c") or (ov == "palette" and key == "ctrl-k"):
            self._close_overlay(back=key == "esc")
            return
        if ov == "help":
            if key in ("enter", "?", "q"):
                st.overlay = None
            return
        if key == "up":
            st.index -= 1
            return
        if key == "down":
            st.index += 1
            return
        if ov == "palette":
            if key == "backspace":
                st.query, st.index = st.query[:-1], 0
            elif key == "enter":
                items = self.R.palette_items(st.query)
                if items:
                    name = items[st.index % len(items)][1]
                    st.overlay = None
                    self.run_command("/" + name, now)
            elif len(key) == 1:
                st.query, st.index = st.query + key, 0
            return
        if ov == "complete":
            if key in ("enter", "tab"):
                if st.completions:
                    _, repl, s, e = st.completions[st.index % len(st.completions)]
                    st.input = st.input[:s] + repl + st.input[e:]
                    self._set_cursor(s + len(repl))
                st.overlay, st.completions = None, []
            return
        if ov == "settings":
            if key == "enter":
                self._settings_activate(now)
            return
        if ov == "kv":
            if key == "enter" and st.picker:
                self._choose_kv(st.picker[st.index % len(st.picker)]["key"])
            return
        if ov == "effort":
            if key == "enter" and st.picker:
                self._set_effort(st.picker[st.index % len(st.picker)]["key"])
                self._close_overlay()
            return
        if ov == "harness":
            if key == "enter" and st.picker:
                self._connect_harness(st.picker[st.index % len(st.picker)]["key"])
            return
        if ov in ("model", "models") and (key == "backspace" or (len(key) == 1 and key.isprintable())):
            st.query = st.query[:-1] if key == "backspace" else st.query + key
            st.index = 0
            self._fill_models()                     # live search
            return
        if ov == "model":
            if key == "enter" and st.picker:
                self._choose_model(st.picker[st.index % len(st.picker)]["key"])
            return
        if ov == "models":
            if not st.picker:
                return
            item = st.picker[st.index % len(st.picker)]
            if key == "enter":
                if item.get("installed"):
                    st.notice = f"{item['name']} is installed — Ctrl-D deletes it"
                else:
                    self._download_async(item["key"])
            elif key in ("ctrl-d", "delete"):
                self._request_delete(item["key"], back="models")
            return
        if ov == "resume" and key == "enter" and st.sessions:
            chosen = st.sessions[st.index % len(st.sessions)]
            st.overlay = None
            self.resume(str(chosen.get("id")), now)

    def _ask(self, title: str, lines: list[str], on_yes: tuple, *, yes: str = "yes", no: str = "no",
             danger: bool = False, back: str | None = None, path: str = "") -> None:
        st = self.state
        st.confirm = {"title": title, "lines": lines, "yes": yes, "no": no, "danger": danger,
                      "action": on_yes, "path": path}
        self._overlay_return = back
        st.overlay = "confirm"

    def _confirm_key(self, key: str, now: float) -> None:
        st = self.state
        if key in ("n", "N", "esc", "ctrl-c"):
            st.notice = "cancelled — nothing changed"
            self._close_overlay(back=key != "ctrl-c")
            return
        if key not in ("y", "Y"):
            return
        action = (st.confirm or {}).get("action") or ()
        back = self._overlay_return
        self._overlay_return = None
        st.confirm, st.overlay = None, None
        kind = action[0] if action else ""
        if kind == "delete":
            self._delete_async(action[1], action[2])
            if back:
                self._overlay_return = back
                self._close_overlay()
        elif kind == "download":
            self._download_async(action[1])
        elif kind == "start":
            self._start_engine_async(restart=False)
        elif kind == "restart":
            self._start_engine_async(restart=True)

    # ---- commands
    def submit(self, now: float) -> None:
        st = self.state
        text = st.input.strip()
        if not text:
            return
        if st.busy and not text.startswith("/"):
            st.notice = "busy: Ctrl-C cancels the current turn"
            return
        st.input, st.cursor, self.hist_pos = "", None, 0
        self.history.append(text)
        if text.startswith("/"):
            self.run_command(text, now)
            return
        if self._engine_running is False:
            st.notice = "engine not started — open /settings or run /start"
            self.say("engine is not running — pick a model in /settings and start it, or run /start")
            return
        R = self.R
        st.add(R.Entry("user", text=text))
        st.busy, st.scroll, st.tokens, st.first_token_at, st.turn_started = True, 0, 0, None, now
        st.status.cancelled, st.activity, st.notice = False, "sending", ""
        try:
            self.adapter.send(text)
        except Exception as exc:
            R.apply_event(st, _error_event(exc), now)
            R.apply_event(st, _event("turn.end", status="error"), now)
        if self.index is not None:
            self.index.save(self.harness, self.cwd, getattr(self.adapter, "session_id", None))

    def say(self, *lines: str) -> None:
        for line in lines:
            self.state.add(self.R.Entry("system", text=line))

    def run_command(self, text: str, now: float) -> None:
        st = self.state
        name, _, rest = text.strip().partition(" ")
        rest = rest.strip()
        if name in ("/quit", "/exit"):
            self.done = True
        elif name == "/help":
            st.overlay = "help"
        elif name == "/reasoning":
            self._set_show_reasoning(not st.show_reasoning)
        elif name == "/effort":
            if rest:
                self._set_effort(rest.lower())
            else:
                self._open_effort_picker(back=None)
        elif name == "/kv":
            if rest:
                self._choose_kv(rest.lower(), from_overlay=False)
            else:
                self._open_kv_picker(back=None)
        elif name == "/tools":
            self._toggle_tools()
            st.notice = f"tool output {'collapsed' if st.tools_collapsed else 'expanded'}"
        elif name == "/new":
            self._new_conversation(now)
        elif name == "/clear":
            st.entries, st.scroll, self.prev = [], 0, None
        elif name == "/resume":
            if rest:
                self.resume(rest, now)
            else:
                try:
                    st.sessions = list(self.adapter.list_sessions())
                except Exception:
                    st.sessions = []
                st.overlay, st.index = "resume", 0
        elif name == "/compact":
            self.say("compact: asking the harness to compact context")
            try:
                self.adapter.send("/compact")
            except Exception as exc:
                self.R.apply_event(st, _error_event(exc), now)
        elif name == "/settings":
            self._open_settings()
        elif name == "/harness":
            self._open_harness_picker(back=None)
        elif name == "/model" and not rest:
            self._open_models(manager=False, back=None)
        elif name in ("/download", "/models"):
            self._download_command(rest)
        elif name == "/start":
            self._start_engine_async(restart=False)
        elif name == "/restart":
            self._start_engine_async(restart=True)
        elif name == "/stop":
            self._stop_engine_async()
        elif name == "/port-check":
            self._port_check_async()
        elif name == "/update":
            self._update_async(rest)
        elif name == "/status":
            self.say(self.R.status_line(st.status, self.theme, width=max(st.cols - 4, 40)))
        else:
            try:
                lines = self.slash_fn(text, home=self._home(), port=self.port)
            except Exception as exc:
                lines = [f"command failed: {exc}"]
            self.say(*lines)

    # ---- background work
    def _spawn(self, kind: str, work, name: str) -> None:
        """Run ``work()`` on a daemon thread; its return value arrives as ``(kind, value)``."""
        def runner() -> None:
            try:
                value = work()
            except BaseException as exc:          # surfaced on the UI thread, never lost
                value = exc
            self._bg_events.put((kind, value))
        threading.Thread(target=runner, daemon=True, name=name).start()

    def _engine_job(self, action: str, activity: str, notice: str, work) -> None:
        st = self.state
        if self._bg_busy:
            st.notice = "engine action already in progress"
            return
        self._bg_busy, self._bg_action = True, action
        st.busy, st.activity, st.notice, st.turn_started = True, activity, notice, self._now
        self._spawn("engine_done", lambda: (action, *work()), f"veystrix-engine-{action}")

    def _start_engine_async(self, *, restart: bool = False) -> None:
        home = self._home()

        def work():
            try:
                if restart:
                    self.slash_fn("/stop", home=home, port=self.port)
                lines = list(self.slash_fn("/start", home=home, port=self.port))
                ok = any("ready:" in line or "already running:" in line for line in lines)
            except Exception as exc:
                return [f"  ◆ Veystrix · cannot start: {exc}"], False
            if ok:
                self._follow_engine_model(home)
            return lines, ok
        self._engine_job("restart" if restart else "start",
                         "restarting engine" if restart else "starting engine",
                         "loading model in background...", work)

    def _follow_engine_model(self, home: Path) -> None:
        """Point a model-pinned harness session at the running engine's model id."""
        try:
            from veystrix import model_identity
            from veystrix.tool.harnesses import family_of, sync_hermes_model
            report = health(self.port, api_key=LocalKeyStore(home).load())
            installed = report.get("installed_model_id") if isinstance(report, dict) else None
            family = family_of(installed) or family_of(Config(home).get("model"))
            alias = model_identity.served_id(family) if family else None
            if self.harness == "hermes" and alias:
                try:
                    sync_hermes_model(home, self.port, alias)
                except (OSError, ValueError):
                    pass
            set_model = getattr(self.adapter, "set_model", None)
            if not callable(set_model):
                return
            target = alias if self.harness == "opencode" and alias else (installed or alias)
            if target and target != getattr(self.adapter, "model", None):
                set_model(str(target))
        except Exception:
            pass                               # the next prompt reports any real problem

    def _stop_engine_async(self) -> None:
        home = self._home()

        def work():
            try:
                return list(self.slash_fn("/stop", home=home, port=self.port)), True
            except Exception as exc:
                return [f"  ◆ Veystrix · stop failed: {exc}"], False
        self._engine_job("stop", "stopping engine", "stopping engine...", work)

    def _port_check_async(self) -> None:
        home = self._home()

        def work():
            try:
                lines = list(self.slash_fn("/port-check", home=home, port=self.port))
                return lines, any("passed" in line for line in lines)
            except Exception as exc:
                return [f"  ◆ Veystrix · port-check failed: {exc}"], False
        self._engine_job("port-check", "running port-check --tiny", "verifying the engine (seconds)...", work)

    def _update_async(self, rest: str = "") -> None:
        """Check GitHub Releases (or brew) and upgrade the terminal tool in the background."""
        if self._update_busy:
            self.state.notice = "an update is already in progress"
            return
        self._update_busy = True
        home = self._home()
        command = "/update --check" if rest.strip() in ("--check", "check") else "/update"
        self.say("  ◆ Veystrix · checking for updates ...")
        self._spawn("update_done",
                    lambda: list(self.slash_fn(command, home=home, port=self.port)),
                    "veystrix-update")

    def _on_update_done(self, lines) -> None:
        self._update_busy = False
        if isinstance(lines, BaseException):
            lines = [f"  ◆ Veystrix · update failed: {lines}"]
        self.say(*lines)

    def probe_async(self) -> None:
        """Background startup probe: selected engine and whether C++ still needs a port-check."""
        home = self._home()

        def work():
            from veystrix.tool.swap import describe
            return describe(home)
        self._spawn("probe", work, "veystrix-probe")

    def _on_engine_done(self, data) -> None:
        st = self.state
        if isinstance(data, BaseException):
            data = (self._bg_action or "start", [f"  ◆ Veystrix · engine action failed: {data}"], False)
        action, lines, ok = data
        self._bg_busy, self._bg_action = False, ""
        st.busy, st.activity = False, ""
        self.say(*lines)
        if action in ("start", "restart"):
            if ok:
                self._engine_running = True
            st.notice = "engine ready" if ok else "engine start failed"
        elif action == "stop":
            if ok:
                self._engine_running = False
            st.notice = "engine stopped" if ok else "engine stop failed"
        elif action == "port-check":
            if ok:
                self._needs_port_check = False
            st.notice = "port-check passed — ready for /start" if ok else "port-check failed"

    def _on_probe(self, info) -> None:
        if isinstance(info, BaseException) or not isinstance(info, dict):
            return
        cpp = (info.get("available") or {}).get("cpp") or {}
        self._needs_port_check = (not cpp.get("available", True)) and "port-check" in str(cpp.get("reason") or "")
        if self._engine_running is not True and info.get("selected"):
            self.state.status.engine = str(info["selected"])

    def _sync_engine_state(self) -> None:
        st = self.state
        if self._bg_busy:
            value = {"start": "starting", "restart": "starting", "stop": "stopping",
                     "port-check": "verifying"}.get(self._bg_action, "starting")
        elif self._engine_running:
            value = "ready"
        elif st.status.engine == "cpp" and self._needs_port_check:
            value = "needs port-check"
        else:
            value = "stopped"
        if value != st.engine_state:
            st.engine_state = value
            self.dirty = True

    # ---- settings
    def _refresh_settings(self) -> None:
        from veystrix.tool.swap import describe
        home = self._home()
        config = Config(home)
        info = describe(home)
        eng = info["selected"]
        mod = info["model"]
        kv = config.get("kv") or "auto"
        running = bool(info["running"])
        cpp_info = (info.get("available") or {}).get("cpp") or {}
        cpp_reason = str(cpp_info.get("reason") or "")
        needs_pc = (not cpp_info.get("available", True)) and ("port-check" in cpp_reason)
        self._needs_port_check = needs_pc
        eng_badge = "  [needs port-check]" if (eng == "cpp" and needs_pc) else ""
        state_txt = "running" if running else "stopped (not loaded)"
        harness = self.harness
        connected = False
        try:
            from veystrix.tool.harnesses import is_connected
            connected = is_connected(harness, home)
        except Exception:
            pass
        label = HARNESS_LABELS.get(harness, harness)
        effort = config.get("effort")
        thinking = "shown" if self.state.show_reasoning else "hidden"
        items = [
            (f"Model     {mod}", "model", "Runtime", "Enter: choose"),
            (f"Effort    {effort}", "effort", "Runtime", "Enter: choose"),
            (f"Thinking  {thinking}", "reasoning", "Runtime", "Enter: show / hide"),
            (f"KV cache  {kv}", "kv", "Runtime", "Enter: choose"),
            (f"Harness   {label}" + ("  (connected)" if connected else ""), "harness", "Harness",
             "Enter: connect"),
            (f"{'Restart' if running else 'Load'} model  {mod}{eng_badge}  [{state_txt}]", "start", "Engine", ""),
        ]
        if needs_pc:
            items.append(("Verify runtime (port-check --tiny)", "port-check", "Engine", "one-time check"))
        if running:
            items.append(("Unload / stop model", "stop", "Engine", ""))
        items.append(("Install or delete models", "download", "Models", "/models"))
        items.append(("Close settings", "close", "Models", "Esc"))
        self.state.settings_items = items
        if not running:
            self.state.status.engine = eng
        self.state.status.kv = kv if kv != "auto" else self.state.status.kv

    def _open_effort_picker(self, *, back: str | None) -> None:
        from veystrix.tool.config import EFFORTS
        st = self.state
        cur = Config(self._home()).get("effort")
        st.picker = [{"key": e, "label": e, "desc": EFFORT_DESCRIPTIONS.get(e, ""),
                      "active": e == cur} for e in EFFORTS]
        self._open_overlay("effort", back=back)
        st.index = next((i for i, item in enumerate(st.picker) if item["active"]), 0)

    def _open_kv_picker(self, *, back: str | None) -> None:
        from veystrix.kv_policy import CHOICES
        st = self.state
        cur = Config(self._home()).get("kv") or "auto"
        st.picker = [{"key": m, "label": m, "desc": KV_DESCRIPTIONS.get(m, ""),
                      "active": m == cur} for m in CHOICES]
        self._open_overlay("kv", back=back)
        st.index = next((i for i, item in enumerate(st.picker) if item["active"]), 0)

    def _choose_kv(self, mode: str, *, from_overlay: bool = True) -> None:
        from veystrix.kv_policy import CHOICES
        st = self.state
        if mode not in CHOICES:
            st.notice = f"unknown KV mode {mode}; use {', '.join(CHOICES)}"
            return
        config = Config(self._home())
        previous = config.get("kv") or "auto"
        config.values["kv"] = mode
        config.save()
        st.status.kv = mode
        st.notice = f"KV cache set to {mode}" + (" — restart to apply" if self._engine_running else "")
        if from_overlay:
            back = self._overlay_return
            if self._engine_running and mode != previous:
                self._ask("Restart to apply KV cache?",
                          [f"KV cache mode changed to {mode}.", "", "Restart the running model now to apply it?"],
                          ("restart",), yes="restart now", no="later", back=back)
                return
            self._overlay_return = back
            self._close_overlay()
        elif st.overlay == "settings":
            self._refresh_settings()

    def _set_effort(self, level: str | None) -> None:
        """Set (or, with no level, cycle) the reasoning effort; the router applies it to every
        harness from the next request on. Saved, so it survives restarts."""
        from veystrix.tool.config import EFFORTS
        st = self.state
        config = Config(self._home())
        cur = config.get("effort")
        if level is None:
            level = EFFORTS[(EFFORTS.index(cur) + 1) % len(EFFORTS)] if cur in EFFORTS else "medium"
        level = {"none": "off", "max": "xhigh", "med": "medium"}.get(level, level)
        if level not in EFFORTS:
            st.notice = f"effort must be one of: {', '.join(EFFORTS)}"
            return
        config.values["effort"] = level
        config.save()
        st.status.effort = level
        st.notice = {"off": "effort off — the model answers without thinking",
                     "auto": "effort auto — each harness decides"}.get(
            level, f"effort {level} — applies from the next message")
        if st.overlay == "settings":
            self._refresh_settings()

    def _set_show_reasoning(self, show: bool) -> None:
        st = self.state
        st.show_reasoning = show
        st.notice = f"reasoning {'shown' if show else 'hidden'}"
        try:
            config = Config(self._home())
            config.values["show_reasoning"] = show
            config.save()
        except Exception:
            pass
        if st.overlay == "settings":
            self._refresh_settings()

    def _open_settings(self, index: int = 0) -> None:
        self._refresh_settings()
        self._overlay_return = None
        self.state.overlay, self.state.index = "settings", index

    def _settings_activate(self, now: float) -> None:
        st = self.state
        if not st.settings_items:
            st.overlay = None
            return
        action = st.settings_items[st.index % len(st.settings_items)][1]
        home = self._home()
        if action == "close":
            st.overlay = None
        elif action == "effort":
            self._open_effort_picker(back="settings")
        elif action == "reasoning":
            self._set_show_reasoning(not st.show_reasoning)
        elif action == "model":
            self._open_models(manager=False, back="settings")
        elif action == "kv":
            self._open_kv_picker(back="settings")
        elif action == "harness":
            self._open_harness_picker(back="settings")
        elif action == "download":
            self._open_models(manager=True, back="settings")
        elif action == "start":
            from veystrix.tool.supervisor import status as _status
            st.overlay = None
            cur_state = _status(home)
            restart = cur_state.get("status") in ("running", "unhealthy") or self._engine_running is True
            self._start_engine_async(restart=restart)
        elif action == "port-check":
            st.overlay = None
            self._port_check_async()
        elif action == "stop":
            st.overlay = None
            self._stop_engine_async()

    # ---- harness picker
    def _open_harness_picker(self, *, back: str | None) -> None:
        st = self.state
        try:
            st.picker = self.harness_fn(self._home(), self.harness)
        except Exception as exc:
            st.picker, st.picker_note = [], f"could not list harnesses: {exc}"
        st.picker_note = "connecting..." if self._connecting else ""
        self._open_overlay("harness", back=back)
        st.index = next((i for i, item in enumerate(st.picker) if item.get("active")), 0)

    def _new_conversation(self, now: float) -> None:
        st = self.state
        if st.busy:
            st.notice = "finish or cancel (Ctrl-C) the current turn before /new"
            return
        reset_fn = getattr(self.adapter, "reset", None)
        if callable(reset_fn):
            try:
                reset_fn()
            except Exception as exc:
                self.R.apply_event(st, _error_event(exc), now)
                return
        st.entries, st.scroll, self.prev = [], 0, None
        st.status.used, st.status.cancelled, st.tokens = 0, False, 0
        st.notice = "started a new conversation"
        if self.index is not None:
            self.index.save(self.harness, self.cwd, getattr(self.adapter, "session_id", None))

    def _connect_harness(self, name: str) -> None:
        st = self.state
        if is_locked(name) and name != "direct":
            st.notice = f"{name} is locked in this release"
            return
        if self._connecting:
            st.notice = "a harness connection is already in progress"
            return
        if st.busy and name != self.harness:
            st.notice = "finish or cancel (Ctrl-C) the current turn before switching harness"
            return
        home, port = self._home(), self.port
        label = HARNESS_LABELS.get(name, name)
        self._connecting = True
        st.picker_note = f"switching to {label}..."
        st.notice = f"switching to {label}..."

        def work():
            import shutil
            from veystrix import model_identity
            from veystrix.tool import harnesses as H
            from veystrix.tool.harness_adapters import DirectAdapter, structured_available
            config = Config(home)
            result = {}
            installed = True
            structured = True
            if name != "direct":
                model_id = model_identity.served_id(config.get("model")) or None
                result = H.configure(name, home=home, port=port, model_id=model_id,
                                     key=LocalKeyStore(home).load() or "local")
                if name == "opencode":
                    H.ensure_opencode_models(home, port)
                try:
                    binary = H.run_command(name, "veystrix")[0]
                    installed = bool(shutil.which(binary))
                except ValueError:
                    installed = False
                structured = structured_available(name)
            config = Config(home)
            config.values["harness"] = name
            config.save()
            adapter = None
            live_harness = self.harness
            if name == self.harness:
                live_harness = name
            elif self.adapter_factory is not None:
                adapter = self.adapter_factory(name)
                if adapter is not None:
                    live_harness = name
                elif self.harness != "direct":
                    # Selected harness has no in-TUI structured driver: stop the old external
                    # harness and switch the live TUI back to the built-in DirectAdapter.
                    key_val = LocalKeyStore(home).load() or ""
                    model_id = model_identity.served_id(config.get("model")) or ""
                    adapter = DirectAdapter(port=port, key=key_val, model=model_id if self._engine_running else "")
                    live_harness = "direct"
            return {"name": name, "result": result, "adapter": adapter,
                    "live_harness": live_harness, "installed": installed, "structured": structured}
        self._spawn("harness_done", work, "veystrix-connect")

    def _on_harness_done(self, data) -> None:
        st = self.state
        self._connecting = False
        st.picker_note = ""
        if isinstance(data, BaseException):
            st.notice = "harness switch failed"
            self.R.apply_event(st, _error_event(data), self._now)
            return
        name = data["name"]
        label = HARNESS_LABELS.get(name, name)
        where = short_home(str((data.get("result") or {}).get("path") or ""))
        adapter = data.get("adapter")
        live_harness = data.get("live_harness") or self.harness
        if adapter is not None:
            old, self.adapter = self.adapter, adapter
            threading.Thread(target=_quiet_close, args=(old,), daemon=True, name="veystrix-close").start()
            self.harness = live_harness
        if live_harness == name:
            if adapter is not None:
                st.notice = f"switched to {label} — now driving this session"
            else:
                st.notice = f"{label} connected" + (f" ({where})" if where else "")
        elif not data.get("installed", True):
            st.notice = f"{label} config written, but {name} is not installed (Built-in active)"
        else:
            st.notice = f"{label} configured — run 'veystrix run {name}' to launch it (Built-in active)"
        st.harness, st.harness_label = (self.harness, HARNESS_LABELS.get(self.harness, self.harness))
        if st.overlay == "harness":
            try:
                st.picker = self.harness_fn(self._home(), self.harness)
            except Exception:
                pass
            for i, item in enumerate(st.picker):
                item["active"] = item.get("key") == self.harness
                if item.get("key") == name:
                    st.index = i
        elif st.overlay == "settings":
            self._refresh_settings()

    # ---- models: picker, manager, download, delete
    def _open_models(self, *, manager: bool, back: str | None) -> None:
        st = self.state
        self._open_overlay("models" if manager else "model", back=back)
        self._fill_models()
        if not manager:
            st.index = next((i for i, item in enumerate(st.picker) if item.get("active")), 0)
        self._scan_models()

    def _fill_models(self) -> None:
        st = self.state
        try:
            active = Config(self._home()).get("model")
        except Exception:
            active = ""
        is_manager = st.overlay in ("models", "manager")
        items = [dict(item, active=item.get("key") == active) for item in (self._inventory or [])
                 if (is_manager or not item.get("locked")) and model_matches(item, st.query)]
        if self._downloading:
            for item in items:
                if item.get("key") == self._downloading:
                    item["busy"] = "downloading..."
        st.picker = items
        st.picker_note = "scanning local models..." if self._scanning and not items else ""

    def _scan_models(self, then: tuple | None = None) -> None:
        if then is not None:
            self._after_scan = then
        if self._scanning:
            return
        self._scanning = True
        if self._inventory is None:
            self.state.picker_note = "scanning local models..."
        self._spawn("inventory", self.inventory_fn, "veystrix-models")

    def _on_inventory(self, items) -> None:
        st = self.state
        self._scanning = False
        if isinstance(items, BaseException):
            st.picker_note = f"could not scan models: {items}"
            items = self._inventory or []
        self._inventory = list(items)
        if st.overlay in ("model", "models"):
            keep = st.index
            self._fill_models()
            st.index = keep
        follow, self._after_scan = self._after_scan, None
        if follow and follow[0] == "delete":
            self._request_delete(follow[1], confirmed=follow[2], back=follow[3])

    def _model_item(self, family: str) -> dict | None:
        return next((item for item in (self._inventory or []) if item.get("key") == family), None)

    def _choose_model(self, family: str) -> None:
        from veystrix import model_identity
        st = self.state
        config = Config(self._home())
        previous = config.get("model")
        ident = model_identity.IDENTITIES.get(family)
        if ident is None or ident.locked:
            st.notice = f"{family} is locked and unavailable in this build" if ident else f"unknown model {family}"
            return
        from veystrix.tool.swap import engine_for
        config.values["model"] = family
        config.values["engine"] = engine_for(family, os.environ.get("VEYSTRIX_ENGINE"), self._home())
        engine_note = ""
        config.save()
        if not self._engine_running:
            st.status.model, st.status.model_id = friendly_model(family), family
            st.status.engine = config.get("engine")
        item = self._model_item(family) or {}
        name = ident.display_name
        st.notice = f"model set to {family}{engine_note}"
        back = self._overlay_return
        if item and not item.get("installed"):
            self._ask("Model not downloaded",
                      [f"{name} ({family}) is not on this Mac yet.", "",
                       f"Download it now from {ident.download_repo}?" if ident.download_repo
                       else "It has no published download."],
                      ("download", family), yes="download", no="later", back=back)
            return
        if self._engine_running and not self._serves(family, st.status.model_id or st.status.model) \
                and family != previous:
            self._ask("Restart engine?",
                      ["Another model is currently loaded.", "",
                       f"Restart now to load {name}?"],
                      ("restart",), yes="restart now", no="later", back=back)
            return
        if not self._engine_running and item and item.get("installed"):
            self._ask("Load model?",
                      [f"{name} ({family}) is installed and ready.", "",
                       "Load it now?"],
                      ("start",), yes="load now", no="later", back=back)
            return
        st.notice += " — loads on /start" if not self._engine_running else ""
        self._overlay_return = back
        self._close_overlay()

    @staticmethod
    def _serves(family: str, served: str) -> bool:
        from veystrix import model_identity
        ident = model_identity.IDENTITIES.get(family)
        served = (served or "").lower()
        if not served or ident is None:
            return False
        return served in (family, ident.served_id) or served.startswith(f"veystrix-{family}")

    def _serving(self, family: str) -> bool:
        """Is the running engine (as far as this UI knows) serving ``family``? Unknown = yes."""
        if not self._engine_running:
            return False
        served = self.state.status.model_id or self.state.status.model
        return not served or self._serves(family, served)

    def _download_command(self, rest: str) -> None:
        st = self.state
        words = rest.split()
        if not words:
            self._open_models(manager=True, back=None)
            return
        if words[0] in ("delete", "rm", "remove"):
            yes = any(w in ("--yes", "-y") for w in words[1:])
            names = [w for w in words[1:] if not w.startswith("-")]
            family = match_model(names[0]) if names else None
            if family is None:
                self.say("usage: /download delete <model> [--yes]   (models: " + ", ".join(_model_keys()) + ")")
                return
            self._request_delete(family, confirmed=yes, back=None)
            return
        family = match_model(words[0])
        if family is None:
            self.say(f"unknown model {words[0]}; use /download <model> with one of: " + ", ".join(_model_keys()))
            st.notice = "unknown model"
            return
        self._download_async(family)

    def _download_async(self, family: str) -> None:
        st = self.state
        if self._downloading:
            st.notice = f"already downloading {self._downloading}; one download at a time"
            return
        item = self._model_item(family)
        if item and item.get("installed"):
            st.notice = f"{family} is already installed"
            return
        self._downloading = family
        st.task, st.task_started = f"downloading {family}: checking files and disk space", self._now
        st.notice = ""
        if st.overlay in ("model", "models"):
            self._fill_models()
        pull = self.pull_fn

        def work():
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # tqdm would scribble on the screen
            try:
                from huggingface_hub.utils import disable_progress_bars
                disable_progress_bars()
            except Exception:
                pass
            from veystrix.checkpoints import model_root
            puller = pull
            if puller is None:
                from veystrix.model_pull import pull as puller
            root = model_root()
            plan = puller(family, root, dry_run=True)
            total = int(plan.get("to_download") or 0)
            seen = [0]

            def echo(message: str) -> None:
                seen[0] += 1
                self._bg_events.put(("download_progress", (family, seen[0], total, str(message))))
            report = puller(family, root, echo=echo, inspect_after=False)
            return family, report
        self._spawn("download_done", work, "veystrix-download")

    def _on_download_progress(self, data) -> None:
        family, i, total, message = data
        detail = message.removeprefix("downloading ").strip()
        count = f"{i}/{total}" if total else str(i)
        self.state.task = f"downloading {family} {count} · {detail}"

    def _on_download_done(self, data) -> None:
        st = self.state
        family = self._downloading or "model"
        self._downloading = None
        st.task = ""
        if isinstance(data, BaseException):
            exc = data
            missing = isinstance(exc, ImportError) and str(getattr(exc, "name", "") or "").startswith("huggingface_hub")
            if missing:
                text = "cannot download: huggingface_hub is not installed (pip install huggingface-hub)"
            else:
                text = f"download of {family} failed: {exc}"
            self.state.add(self.R.Entry("error", text=text))
            st.notice = "download failed — run /download again to resume"
        else:
            family, report = data
            files = report.get("files", 0)
            dest = short_home(str(report.get("destination") or ""))
            self.say(f"downloaded {family}: {files} files verified in {dest}")
            st.notice = f"{family} downloaded — /start to load it"
        self._scan_models()

    def _request_delete(self, family: str, *, confirmed: bool = False, back: str | None = None) -> None:
        st = self.state
        if self._serving(family):
            st.notice = f"{family} is loaded in the running engine — /stop it first"
            self.say(f"refusing to delete {family}: the engine is serving it. Run /stop first, then delete.")
            return
        if self._downloading == family:
            st.notice = f"{family} is downloading — wait for it to finish"
            return
        item = self._model_item(family)
        if item is None:                         # not scanned yet: scan, then come back here
            st.notice = "scanning local models..."
            self._scan_models(then=("delete", family, confirmed, back))
            return
        if not (item.get("installed") or item.get("partial")) or not item.get("path"):
            st.notice = f"{family} is not downloaded — nothing to delete"
            return
        path = str(item["path"])
        if confirmed:
            self._delete_async(family, path)
            return
        size = self.R.human_size(int(item.get("size") or 0))
        self._ask("Delete model files",
                  [f"Delete {item.get('name') or family} ({family})?",
                   f"  {size} will be freed. This cannot be undone."],
                  ("delete", family, path), yes="delete", no="keep", danger=True, back=back,
                  path=short_home(path))

    def _delete_async(self, family: str, path: str) -> None:
        st = self.state
        if self._serving(family):                # re-checked: the engine may have started meanwhile
            st.notice = f"{family} is loaded in the running engine — /stop it first"
            return
        home = self._home()
        st.task, st.task_started = f"deleting {family}", self._now

        def work():
            import shutil
            from veystrix.checkpoints import model_root
            from veystrix.tool.supervisor import status as engine_status
            root = model_root().resolve()
            target = Path(path).resolve()
            if target == root or root not in target.parents or not target.is_dir():
                raise ValueError(f"refusing to delete {target}: not a model folder under {root}")
            live = engine_status(home)
            if live.get("status") in ("running", "unhealthy") and live.get("model") in (family, None):
                raise ValueError(f"the engine is serving {family}; run /stop first")
            shutil.rmtree(target)
            return family, str(target)
        self._spawn("delete_done", work, "veystrix-delete")

    def _on_delete_done(self, data) -> None:
        st = self.state
        st.task = ""
        if isinstance(data, BaseException):
            self.state.add(self.R.Entry("error", text=f"delete failed: {data}"))
            st.notice = "delete failed — nothing removed"
        else:
            family, target = data
            self.say(f"deleted {family} ({short_home(target)})")
            st.notice = f"{family} deleted"
        self._scan_models()

    def resume(self, session_id: str, now: float) -> None:
        st = self.state
        try:
            self.adapter.resume(session_id)
        except Exception as exc:
            self.R.apply_event(st, _error_event(exc), now)
            return
        st.entries, st.scroll = [], 0
        self.say(f"resumed session {session_id}")
        if self.index is not None:
            self.index.save(self.harness, self.cwd, session_id)

    # ---- loop
    def _drain_background(self) -> None:
        while True:
            try:
                kind, data = self._bg_events.get_nowait()
            except queue.Empty:
                return
            handler = getattr(self, "_on_" + kind, None)
            if handler is not None:
                handler(data)
            self.dirty = True

    def step(self, now: float, keys=()) -> bool:
        """One iteration; returns True when a frame was written."""
        st = self.state
        self._now = st.now = now
        for key in keys:
            self.handle_key(key, now)
        self._drain_background()
        for event in self.adapter.poll(0):
            self.R.apply_event(st, event, now)
            self.dirty = True
        if (st.busy or st.task) and now - self.last_spin >= SPINNER_INTERVAL:
            st.spinner += 1
            self.last_spin = now
            self.dirty = True
        if self.cancel_at is not None:
            if not st.busy:
                self.cancel_at = None
            elif now - self.cancel_at >= CANCEL_GRACE:  # hung harness: release the UI
                self.R.apply_event(st, _event("turn.end", status="cancelled"), now)
                self.cancel_at = None
        poll_interval = BUSY_STATUS_INTERVAL if st.busy else STATUS_INTERVAL
        if self.status_fn is not None and now - self.last_status >= poll_interval:
            self.last_status = now
            data = self.status_fn()
            if data is not None and self._engine_running is False:
                self._engine_running = True
            elif data is None and self._engine_running is True and not self._bg_busy:
                self._engine_running = False
                st.status.stage = ""
                st.status.tok_s = 0.0
                try:
                    from veystrix.tool.supervisor import _read_state
                    sup = _read_state(self._home()) or {}
                except Exception:
                    sup = {}
                if sup.get("status") == "guard_stopped":
                    st.notice = "engine stopped by memory safety guard — run /start"
                else:
                    st.notice = "engine stopped — run /start"
                self.dirty = True
            rate = data.get("decode_tokens_per_second") if isinstance(data, dict) else None
            if isinstance(rate, (int, float)) and not isinstance(rate, bool):
                # Latch on the presence of engine telemetry, not on a non-zero
                # reading: a 0.0 during idle must still stop chunk-based guesses
                # from overwriting the engine's real tok/s later.
                st.server_tok_s = True
            if apply_engine_status(st.status, data):
                self.dirty = True
        self._sync_engine_state()
        size = self.size()
        if size != (st.cols, st.rows):
            st.cols, st.rows = size
            self.prev, self.dirty = None, True
        if not self.dirty:
            return False
        if not keys and now - self.last_frame_at < FRAME_INTERVAL:
            return False
        return self.render(now)

    def next_wakeup(self, now: float) -> float:
        """Seconds the driver may sleep before it must call ``step`` again."""
        waits = [0.05]
        if self.dirty:
            waits.append(max(self.last_frame_at + FRAME_INTERVAL - now, 0.0))
        if self.state.busy or self.state.task:
            waits.append(max(self.last_spin + SPINNER_INTERVAL - now, 0.0))
        return min(waits)

    def render(self, now: float) -> bool:
        st, R = self.state, self.R
        frame = R.compose_frame(st, self.theme, st.cols, st.rows)
        full = self.prev is None
        changed = R.diff_rows(self.prev, frame)
        self.prev, self.last_frame = frame, frame
        self.dirty, self.last_frame_at = False, now
        if not changed:
            return False
        self.frames += 1
        self.write(("\033[2J" if full else "") + R.paint_rows(changed))
        return True


def prepare_harness(harness: str, home: Path, port: int, model: str | None) -> str | None:
    """Get a harness ready to talk to the local engine; return the model id to pin its session to.

    OpenCode: its Veystrix provider must list the family aliases (written once, with a backup),
    and the session is pinned to ``veystrix-<family>`` so it never falls back to another model.
    Pi: ensure its Veystrix provider extension is installed and return the model id so
    ``pi --model veystrix/<id>`` always targets the local Veystrix engine.
    Headless CLIs (claude-code, qwen-code, kimi): return the resolved model id."""
    from veystrix import model_identity
    from veystrix.tool.harnesses import (configure, ensure_opencode_models, family_of,
                                         is_connected, sync_hermes_model)
    family = family_of(model) or family_of(Config(home).get("model"))
    alias = model_identity.served_id(family) if family else None
    resolved = (str(model) if (model and str(model).startswith("veystrix-")) else alias)
    if harness == "hermes" and alias:
        # Hermes has no per-session model switch: its profile's default must follow the engine.
        try:
            sync_hermes_model(home, port, alias)
        except (OSError, ValueError):
            pass
        return alias
    if harness == "pi":
        try:
            if not is_connected("pi", home):
                configure("pi", home=home, port=port, model_id=resolved)
        except (OSError, ValueError):
            pass
        return resolved
    if harness in ("claude-code", "qwen-code", "kimi"):
        return resolved
    if harness != "opencode":
        return None
    try:
        ensure_opencode_models(home, port)
    except (OSError, ValueError):
        pass                                   # a hand-edited provider stays as the user wrote it
    return alias


def _model_keys() -> list[str]:
    from veystrix import model_identity
    return list(model_identity.PUBLIC_MODEL_CHOICES)


def short_home(path: str) -> str:
    from veystrix.tool.renderer import short_path
    return short_path(path)


def _quiet_close(adapter) -> None:
    try:
        adapter.close()
    except Exception:
        pass


def _error_event(exc: Exception):
    return _event("error", message=str(exc) or exc.__class__.__name__)


def _event(kind: str, **data):
    from veystrix.tool.harness_adapters import ev
    return ev(kind, **data)


def run_ui(adapter, *, home: Path, port: int, harness: str, key: str = "", engine: str = "python",
           model: str = "", resume: str | None = None, cwd: str | None = None, running: bool = True) -> dict:
    """Own the terminal: alternate screen, raw keys, bounded redraws; restore on every exit path."""
    import atexit
    import shutil as _shutil
    import termios
    import tty
    from veystrix.tool import renderer as R
    cwd = cwd or os.getcwd()
    index = SessionIndex(home)
    if resume:
        adapter.session_id = resume
    status = R.Status(engine=engine, model=friendly_model(model or ""), model_id=model or "")
    poller = StatusPoller(lambda: fetch_engine_status(port, key)).start()
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    out = sys.stdout

    def write(data: str) -> None:
        out.write(data)
        out.flush()

    def size() -> tuple[int, int]:
        got = _shutil.get_terminal_size((80, 24))
        return got.columns, got.lines

    def restore() -> None:
        try:
            write(ALT_LEAVE)
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except (termios.error, OSError, ValueError):
            pass
    atexit.register(restore)
    # Restore the terminal on SIGTERM/SIGHUP too, not only on normal exit or
    # KeyboardInterrupt (Sonnet finding 25).
    import signal
    previous_signals = {}

    def _stop_engine_on_exit(timeout: float = 5.0) -> None:
        try:
            from veystrix.tool.supervisor import stop as _stop_engine
            _stop_engine(home, timeout=timeout)
        except Exception:
            pass

    def _signal_restore(signum, _frame):
        try:
            ui.adapter.close()
        except Exception:
            pass
        _stop_engine_on_exit(timeout=5.0)
        restore()
        os._exit(128 + signum)

    for _sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            previous_signals[_sig] = signal.signal(_sig, _signal_restore)
        except (OSError, ValueError):
            pass
    def adapter_factory(name: str):
        """A started structured adapter for ``name`` (from /settings -> Harness), or None."""
        from veystrix import model_identity
        from veystrix.tool.builtin_adapter import BuiltinAdapter
        from veystrix.tool.harness_adapters import make_adapter
        cur_model = Config(home).get("model")
        served = model_identity.served_id(cur_model) or model or "veystrix-inkling2-mini"
        cur_key = LocalKeyStore(home).load() or key or "local"
        if name == "direct":
            # Veystrix's own Built-in harness: read/write/bash + simple commands.
            return BuiltinAdapter(port=port, key=cur_key, cwd=cwd,
                                  model=served if ui._engine_running else "")
        env = whitelabel_environment(name, port, served, cur_key)
        new = make_adapter(name, env=env, cwd=cwd,
                           model=prepare_harness(name, home, port, cur_model),
                           log_path=tool_home(home) / "logs" / f"harness-{name}.log")
        if new is not None:
            new.start()
        return new

    ui = UILoop(adapter, write=write, size=size, status=status, home=home, port=port,
                harness=harness, cwd=cwd, status_fn=poller, index=index, running=running,
                adapter_factory=adapter_factory)
    # Stray Python-level stderr (library warnings, download progress) would scribble over the
    # alternate screen; send it to a log for the life of the UI.
    saved_stderr = sys.stderr
    try:
        from veystrix.tool.config import tool_home
        log_dir = tool_home(home) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stderr_log = open(log_dir / "tui-stderr.log", "a", encoding="utf-8", errors="replace")
    except OSError:
        stderr_log = None
    parser = KeyParser()
    import codecs
    import select as _select
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    try:
        adapter.start()
        tty.setraw(fd)
        write(ALT_ENTER)
        if stderr_log is not None:
            sys.stderr = stderr_log
        if resume:
            ui.say(f"resumed session {resume}")
        if not running:
            # The home dashboard shows no model loaded until the user picks and loads one.
            ui.state.notice = "no model loaded — /model to pick & load, /models to install/delete, /kv for KV cache"
        ui.probe_async()
        while not ui.done:
            now = time.monotonic()
            ui.step(now, ())
            ready, _, _ = _select.select([fd], [], [], ui.next_wakeup(time.monotonic()))
            keys: list[str] = []
            if ready:
                data = os.read(fd, 4096)
                if not data:
                    break
                keys = parser.feed(decoder.decode(data))
            if keys:
                ui.step(time.monotonic(), keys)
    except KeyboardInterrupt:
        pass
    finally:
        for _sig, _handler in previous_signals.items():
            try:
                signal.signal(_sig, _handler)
            except (OSError, ValueError):
                pass
        poller.stop()
        sys.stderr = saved_stderr
        if stderr_log is not None:
            stderr_log.close()
        try:
            ui.adapter.close()
        finally:
            _stop_engine_on_exit(timeout=10.0)
            restore()
            atexit.unregister(restore)
    if index is not None:
        index.save(ui.harness, cwd, getattr(ui.adapter, "session_id", None))
    return {"status": "done", "frames": ui.frames}


def tui(harness: str | None = None, *, home: Path | None = None, port: int = 8080,
        model: str | None = None, resume: str | None = None, once: bool = False,
        cancel_after: float | None = None, command: list[str] | None = None,
        ui: str | None = None) -> dict:
    home = home or home_dir()
    config = Config(home)
    harness = harness or config.get("harness") or DEFAULT_HARNESS   # /settings -> Harness persists it
    if harness != "direct" and is_locked(harness):
        harness = DEFAULT_HARNESS
    port = port or config.get("port")
    running = health(port, api_key=LocalKeyStore(home).load()) is not None
    if not running and (once or command is not None):
        raise SystemExit(f"veystrix: engine is not running on 127.0.0.1:{port}; start it with 'veystrix start' first.")
    return _tui_session(harness, home=home, config=config, port=port, model=model, resume=resume,
                        once=once, cancel_after=cancel_after, command=command, ui=ui, running=running)


def _tui_session(harness: str, *, home: Path, config: Config, port: int, model: str | None,
                 resume: str | None, once: bool, cancel_after: float | None,
                 command: list[str] | None, ui: str | None, running: bool = True) -> dict:
    if running:
        model_id = resolve_model_id(port, model or config.get("model"),
                                    api_key=LocalKeyStore(home).load(),
                                    explicit=model is not None)
    else:
        model_id = model or config.get("model") or "inkling2-mini"
    command_given = command
    from veystrix.tool.swap import engine_for as _engine_for
    selected_engine = _engine_for(model or config.get("model"), os.environ.get("VEYSTRIX_ENGINE"), home)
    if harness == "direct" and command_given is None and not once:
        from veystrix.tool.builtin_adapter import BuiltinAdapter
        key_val = LocalKeyStore(home).load() or ""
        display_model = model_id if (running or model is not None) else ""
        adapter = BuiltinAdapter(port=port, key=key_val, cwd=os.getcwd(),
                                 model=model_id if running else "")
        return run_ui(adapter, home=home, port=port, harness="direct",
                      key=key_val, engine=selected_engine,
                      model=display_model, resume=resume, running=running)
    env = whitelabel_environment(harness, port, model_id, LocalKeyStore(home).load() or "local")
    command = command or launch_command(harness, model_id, resume)
    if once:
        result = run_once(command, env, harness=harness, cancel_after=cancel_after)
        result["lines"] = [*banner(), *result["lines"]]
        if result["cancelled"]:
            result["lines"].append("  ◆ Veystrix · session can be resumed with --resume <id>")
        for line in result["lines"]:
            print(line)
        return result
    # D6 full UI: always own the screen on a real TTY unless --command or VEYSTRIX_UI=passthrough.
    # When the selected harness has a structured adapter (OpenCode/Codex), drive it headless;
    # otherwise use the built-in DirectAdapter so the Veystrix logo, /model, /models, /effort,
    # /kv, and /settings are always available. Never load a model by default.
    mode = (ui or os.environ.get("VEYSTRIX_UI") or "auto").lower()
    if (command_given is None and mode in ("auto", "headless") and sys.stdin.isatty()
            and sys.stdout.isatty()):
        from veystrix.tool.builtin_adapter import BuiltinAdapter
        from veystrix.tool.harness_adapters import make_adapter
        key_val = LocalKeyStore(home).load() or ""
        adapter = make_adapter(harness, env=env, cwd=os.getcwd(),
                               model=prepare_harness(harness, home, port, model_id),
                               log_path=tool_home(home) / "logs" / f"harness-{harness}.log")
        ui_harness = harness
        if adapter is None:
            adapter = BuiltinAdapter(port=port, key=key_val, cwd=os.getcwd(),
                                     model=model_id if running else "")
            ui_harness = "direct"
        display_model = model_id if (running or model is not None) else ""
        return run_ui(adapter, home=home, port=port, harness=ui_harness,
                      key=key_val, engine=selected_engine,
                      model=display_model, resume=resume, running=running)
    prepare_harness(harness, home, port, model_id)
    lines = passthrough_banner(harness, engine=selected_engine,
                               model=model_id, running=running, port=port)
    print("\n".join(lines))
    try:
        process = subprocess.Popen(command, env={**os.environ, **env})
    except (OSError, ValueError) as exc:
        raise SystemExit(f"veystrix: could not start harness {harness!r}: {exc}. "
                         "Use --command to supply a harness command, or connect one with 'veystrix connect'.") from exc
    prev_sigs = {}

    def _passthrough_signal(signum, _frame):
        try:
            process.terminate()
        except OSError:
            pass
        try:
            from veystrix.tool.supervisor import stop as _stop_engine
            _stop_engine(home, timeout=5.0)
        except Exception:
            pass
        os._exit(128 + signum)

    for _sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            prev_sigs[_sig] = signal.signal(_sig, _passthrough_signal)
        except (OSError, ValueError):
            pass
    try:
        process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        print("\n  ◆ Veystrix · generation cancelled")
    finally:
        for _sig, _handler in prev_sigs.items():
            try:
                signal.signal(_sig, _handler)
            except (OSError, ValueError):
                pass
        try:
            from veystrix.tool.supervisor import stop as _stop_engine
            _stop_engine(home, timeout=10.0)
        except Exception:
            pass
    return {"status": "done"}
