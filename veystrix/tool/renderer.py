"""Veystrix-owned terminal renderer (D6).

The user sees ONLY Veystrix branding. This module draws the screen; an external
agentic harness runs behind a headless adapter and emits normalized events. The
renderer is pure (no I/O, no MLX, no harness import), so it can be golden-tested
offline at any terminal size and with no color.

Design rules from the owner brief:
  * truecolor -> 256-color -> plain fallback (NO_COLOR / TERM=dumb)
  * never rely on color alone: every semantic marker also has a symbol/word
  * 80x24 minimum, ASCII fallback for non-UTF-8 locales
  * status line shows model, window gauge, reasoning effort, cache state, tok/s, TTFT and
    thermal/guard state. The engine is one product to the user: which runtime serves a model
    is never shown (it is chosen automatically, see swap.engine_for)
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field

# ---- palette -------------------------------------------------------------
_TRUE = {
    "brand": (124, 92, 255), "text": (232, 232, 238), "dim": (140, 140, 152),
    "ok": (80, 220, 140), "warn": (255, 190, 80), "err": (255, 100, 110),
    "accent": (90, 210, 255), "reason": (170, 150, 220),
}
_256 = {
    "brand": 99, "text": 255, "dim": 245, "ok": 42, "warn": 214,
    "err": 203, "accent": 81, "reason": 141,
}
_TRUE_LIGHT = {
    "brand": (88, 52, 214), "text": (28, 28, 36), "dim": (96, 96, 108),
    "ok": (16, 128, 72), "warn": (176, 100, 0), "err": (190, 30, 50),
    "accent": (0, 110, 170), "reason": (110, 80, 170),
}
_256_LIGHT = {
    "brand": 56, "text": 235, "dim": 241, "ok": 28, "warn": 130,
    "err": 124, "accent": 25, "reason": 97,
}
_SYMBOLS = {
    "brand": "◆", "ok": "✓", "err": "✗", "warn": "!", "tool": "▸",
    "reason": "◇", "cancelled": "⊘", "cached": "↺", "cold": "↓",
    "dot": "·", "bar_full": "█", "bar_empty": "░", "spinner": "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
    "on": "●", "off": "○", "busy": "◐", "alert": "▲", "sel": "▸", "bullet": "•",
}
_ASCII = {
    "brand": "*", "ok": "+", "err": "x", "warn": "!", "tool": ">",
    "reason": "o", "cancelled": "X", "cached": "~", "cold": "v",
    "dot": ".", "bar_full": "#", "bar_empty": "-", "spinner": "|/-\\",
    "on": "*", "off": "o", "busy": "~", "alert": "!", "sel": ">", "bullet": "*",
}
# Selection-row background (colour themes only; no-color/ascii rely on the "▸" marker).
_SEL_BG = {True: (44, 38, 78), False: (229, 224, 252)}
_SEL_BG_256 = {True: 236, False: 254}


@dataclass
class Theme:
    ascii_only: bool = False
    color: str = "none"  # none | 256 | true
    dark: bool = True

    @classmethod
    def detect(cls, tty: bool = True) -> "Theme":
        ascii_only = (os.environ.get("TERM", "").lower() in ("dumb", "")
                      or os.environ.get("LC_ALL", "").lower().endswith("ascii")
                      or os.environ.get("VEYSTRIX_UI_ASCII") == "1")
        if not tty or os.environ.get("NO_COLOR") is not None:
            color = "none"
        elif os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
            color = "true"
        else:
            color = "256"
        dark = os.environ.get("VEYSTRIX_UI_THEME", "dark").lower() != "light"
        return cls(ascii_only=ascii_only, color=color, dark=dark)

    def sym(self, name: str) -> str:
        table = _ASCII if self.ascii_only else _SYMBOLS
        value = table.get(name, name)
        if name == "spinner":
            return value
        return value

    def paint(self, name: str, text: str) -> str:
        if self.color == "none":
            return text
        if self.color == "256":
            code = (_256 if self.dark else _256_LIGHT).get(name, 255)
            prefix = f"\033[38;5;{code}m"
        else:
            r, g, b = (_TRUE if self.dark else _TRUE_LIGHT).get(name, (255, 255, 255))
            prefix = f"\033[38;2;{r};{g};{b}m"
        return f"{prefix}{text}\033[0m"

    def bold(self, text: str) -> str:
        """Bold weight in colour themes; plain text otherwise (text parity across themes)."""
        return text if self.color == "none" or not text else f"\033[1m{text}\033[0m"

    def blend(self, a: str, b: str, t: float, text: str) -> str:
        """Foreground interpolated between two palette entries (truecolor), else nearest entry."""
        if self.color != "true":
            return self.paint(a if t < 0.5 else b, text)
        table = _TRUE if self.dark else _TRUE_LIGHT
        ca, cb = table.get(a, (255, 255, 255)), table.get(b, (255, 255, 255))
        r, g, bl = (round(x + (y - x) * min(max(t, 0.0), 1.0)) for x, y in zip(ca, cb))
        return f"\033[38;2;{r};{g};{bl}m{text}\033[0m"

    def select(self, text: str) -> str:
        """Highlight a whole (already padded) selection row with a background band."""
        if self.color == "none":
            return text
        if self.color == "256":
            prefix = f"\033[48;5;{_SEL_BG_256[self.dark]}m"
        else:
            prefix = "\033[48;2;{};{};{}m".format(*_SEL_BG[self.dark])
        return prefix + text.replace("\033[0m", "\033[0m" + prefix) + "\033[0m"


def _fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    clean = re.sub(r"\033\[[0-9;?]*[a-zA-Z]", "", text)
    if len(clean) <= width:
        return text
    return clean[: max(width - 3, 0)] + "..."


def _fit_left(text: str, width: int) -> str:
    """Like ``_fit`` but keeps the END (paths: the folder name is the useful part)."""
    if width <= 3 or len(text) <= width:
        return text if len(text) <= width else text[:max(width, 0)]
    return "..." + text[-(width - 3):]


def fmt_tokens(n: int) -> str:
    """Format token counts compactly using k and M suffixes (e.g. 12.4k, 256k, 1M)."""
    n = int(n)
    if n < 1000:
        return str(n)
    if n >= 1_000_000:
        if n % 1_048_576 == 0:
            return f"{n // 1_048_576}M"
        val = f"{n / 1_000_000:.1f}"
        return (val[:-2] if val.endswith(".0") else val) + "M"
    if n >= 4096 and n % 1024 == 0:
        return f"{n // 1024}k"
    val = f"{n / 1000:.1f}"
    return (val[:-2] if val.endswith(".0") else val) + "k"


def gauge(used: int, window: int, theme: Theme, width: int = 10) -> str:
    """Context-window gauge with a compact k/M numeric label (never color-only)."""
    u_txt = fmt_tokens(used)
    if window <= 0:
        return f"{u_txt}/?"
    w_txt = fmt_tokens(window)
    if width <= 0:
        return f"{u_txt}/{w_txt}"
    ratio = min(max(used / window, 0.0), 1.0)
    filled = round(ratio * width)
    color = "warn" if ratio > 0.85 else "accent"
    bar = theme.sym("bar_full") * filled + theme.sym("bar_empty") * (width - filled)
    return f"{theme.paint(color, bar)} {u_txt}/{w_txt}"


@dataclass
class Status:
    engine: str = "python"
    model: str = ""
    used: int = 0
    window: int = 0
    sampling: str = "thinking"
    cache: str = "cold"  # cold | cached
    tok_s: float = 0.0
    ttft_s: float = 0.0
    thermal: int = 0
    cancelled: bool = False
    runtime: str = ""       # "generic" | "custom" | "" (python); internal, never displayed
    kv: str = ""            # active KV mode, shown when the runtime reports one
    effort: str = ""        # reasoning effort setting; replaces `sampling` on the line when set
    model_id: str = ""      # raw served id / family key behind `model` (which is for display)
    stage: str = ""         # "" | "idle" | "preparing" | "prefill" | "decode"
    prefill_done: int = 0
    prefill_total: int = 0


def _phase_segment(status: Status, theme: Theme, level: int) -> str | None:
    """Live model phase for the bottom bar: prefill length + %, or decode speed."""
    stage = (status.stage or "").lower()
    compact, tiny = level >= 1, level >= 4
    if stage == "prefill":
        total = max(int(status.prefill_total or 0), 0)
        done = max(0, min(int(status.prefill_done or 0), total)) if total else 0
        pct = round(100 * done / total) if total > 0 else 0
        if total > 0:
            t_txt = fmt_tokens(total)
            if tiny:
                label = f"pf {t_txt} {pct}%"
            else:
                label = f"prefill {t_txt} ({pct}%)"
        else:
            label = f"pf {pct}%" if tiny else f"prefill {pct}%"
        return theme.paint("accent", label)
    if stage == "preparing":
        return theme.paint("accent", "pf ..." if tiny else "prefill ...")
    if stage == "decode":
        if tiny:
            label = f"{status.tok_s:.1f} t/s"
        elif compact:
            label = f"decode {status.tok_s:.1f} t/s"
        else:
            label = f"decode {status.tok_s:.1f} tok/s"
        return theme.paint("ok", label)
    if level <= 2:
        return f"{status.tok_s:.1f} " + ("t/s" if compact else "tok/s")
    return None


def status_line(status: Status, theme: Theme, width: int = 80) -> str:
    """One line with the model, context gauge, effort, live prefill/decode phase and safety state.

    When the full line does not fit, optional fields are dropped in a fixed order
    (TTFT, then idle tok/s) and labels are abbreviated; model, gauge, effort, cache, active
    prefill/decode telemetry, and thermal are never dropped. The runtime engine is absent.
    """
    dot = theme.paint("dim", theme.sym("dot"))
    cache = (theme.sym("cached") if status.cache == "cached" else theme.sym("cold")) + " " + status.cache

    def build(level: int) -> str:
        # level 0 full .. 5 tiny; each step only shortens or drops optional fields
        compact, tiny = level >= 1, level >= 4
        effort = (("effort " if not compact else "") + status.effort) if status.effort else status.sampling
        parts = [theme.paint("brand", theme.sym("brand")), theme.paint("brand", theme.bold("VEYSTRIX")),
                 dot, theme.paint("accent", theme.bold(_fit(status.model, 6 if level >= 5 else (10 if compact else 18)))),
                 dot, gauge(status.used, status.window, theme, width=0 if tiny else (4 if compact else 10)),
                 dot, effort, dot, cache]
        if status.kv and level <= 2:
            parts += [dot, ("kv " if compact else "KV ") + status.kv]
        phase = _phase_segment(status, theme, level)
        if phase:
            parts += [dot, phase]
        if level <= 1 and (status.stage or "").lower() not in ("prefill", "preparing", "decode"):
            parts += [dot, ("ttft " if compact else "TTFT ") + f"{status.ttft_s:.1f}s"]
        parts += [dot, ("therm " if compact else "thermal ") + str(status.thermal)]
        if status.cancelled:
            parts.append(theme.sym("cancelled") + " cancelled")
        return " ".join(parts)

    for level in range(6):
        line = build(level)
        if visible_len(line) <= width:
            return line
    return _fit(line, width)


def banner(theme: Theme, width: int = 80) -> list[str]:
    return [
        "",
        "  " + theme.paint("brand", f"{theme.sym('brand')} V E Y S T R I X") +
        "  ·  local coding agent",
        "  " + "─" * min(52, max(20, width - 4)) if not theme.ascii_only
        else "  " + "-" * min(52, max(20, width - 4)),
        "  /config  /cache  /doctor  /update  /effort  /theme  /resume  /reasoning  /compact  /quit",
        "",
    ]


def tool_block(name: str, args: dict, result: str | None, theme: Theme,
               collapsed: bool = False, width: int = 80) -> list[str]:
    """Readable tool call: symbol + word, args, optional result with line numbers."""
    head = f"  {theme.sym('tool')} {theme.paint('accent', name)}"
    if args:
        head += "  " + _fit(" ".join(f"{k}={v}" for k, v in args.items()), max(width - 12, 8))
    lines = [head]
    if collapsed or result is None:
        return lines
    for index, line in enumerate(result.splitlines(), 1):
        lines.append(f"      {index:4d} │ {_fit(line, max(width - 14, 8))}")
    return lines


def diff_block(diff: str, theme: Theme, width: int = 80) -> list[str]:
    """Unified diff: +/- markers AND color, line numbers never stripped."""
    out = []
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            out.append(theme.paint("ok", "  + " + _fit(line[1:], max(width - 6, 8))))
        elif line.startswith("-") and not line.startswith("---"):
            out.append(theme.paint("err", "  - " + _fit(line[1:], max(width - 6, 8))))
        elif line.startswith("@@"):
            out.append(theme.paint("dim", "  @ " + _fit(line, max(width - 6, 8))))
        else:
            out.append("    " + _fit(line, max(width - 6, 8)))
    return out


def stream_line(text: str, theme: Theme, spinner: int = 0, width: int = 80) -> str:
    spin = theme.sym("spinner")[spinner % len(theme.sym("spinner"))]
    return "  " + theme.paint("brand", spin) + " " + _fit(text, max(width - 6, 8))


def reasoning_block(text: str, theme: Theme, width: int = 80) -> list[str]:
    return [theme.paint("reason", "  " + theme.sym("reason") + " " + _fit(line, max(width - 6, 8)))
            for line in text.splitlines()]


def cancelled_line(theme: Theme) -> str:
    return "  " + theme.sym("cancelled") + " " + theme.paint("warn", "cancelled")


# =====================================================================================
# D6 full UI: pure model + frame composer (no I/O). tui.py owns the terminal and the loop.
# A frame is ALWAYS exactly ``rows`` strings of exactly ``cols`` visible characters, so a
# bounded redraw can compare rows and rewrite only the ones that changed.
# =====================================================================================
import textwrap  # noqa: E402

MIN_COLS, MIN_ROWS = 80, 24
_ANSI = re.compile(r"\033\[[0-9;?]*[a-zA-Z]")
TOOL_OUTPUT_LINES = 12
PALETTE = (
    ("Settings (model / effort / KV / harness)", "settings"),
    ("New conversation", "new"),
    ("Set reasoning effort", "effort"),
    ("Choose KV cache mode", "kv"),
    ("Choose / load a model", "model"),
    ("Install or delete models", "models"),
    ("Download or delete models", "download"),
    ("Connect / switch harness", "harness"),
    ("Start engine", "start"),
    ("Restart engine", "restart"),
    ("Stop engine", "stop"),
    ("Verify the engine (port-check)", "port-check"),
    ("Show / hide reasoning", "reasoning"),
    ("Toggle tool output (collapse / expand)", "tools"),
    ("Resume a previous session", "resume"),
    ("Show status", "status"),
    ("Show help", "help"),
    ("Compact context", "compact"),
    ("Clear screen", "clear"),
    ("Quit", "quit"),
)
HELP_LINES = (
    "Keys",
    "  Enter / Shift+Enter send / newline   Tab     complete /cmd & @file",
    "  Ctrl-K     command palette           ?       help (empty input)",
    "  Ctrl-R     show/hide reasoning       Ctrl-T  collapse/expand tools",
    "  PgUp/PgDn  scroll                    Ctrl-C  cancel (2x quit)",
    "  y / a / n  allow once / always / reject a permission request",
    "Commands",
    "  /new       start new chat            /restart  restart model",
    "  /settings  model, effort, KV, harness  /model  pick & load model",
    "  /models    install or delete models  /effort   reasoning effort",
    "  /kv        choose KV cache           /harness  switch harness",
    "  /start /stop /status /update /endpoint /resume [id] /clear /quit",
)
# Home dashboard quick-start grid: (keys, what it does).
SHORTCUTS = (
    ("/model", "pick & load a model"), ("/download", "install / delete models"),
    ("/effort", "reasoning effort"), ("/kv", "choose KV cache"),
    ("/settings", "all runtime settings"), ("/start", "load selected model"),
    ("@file Tab", "attach a file"), ("Ctrl-K", "command palette"),
    ("?", "all shortcuts"), ("Ctrl-C", "cancel, twice quits"),
)
TAGLINE = "local coding agent  ·  your models, your machine"

# Block-letter hero mark (6 rows, 62 columns). "█" is the letter body, the box-drawing
# characters are its drop shadow; the ASCII fallback keeps only the body.
_LOGO_GLYPHS = {
    "V": ("██╗   ██╗", "██║   ██║", "██║   ██║", "╚██╗ ██╔╝", " ╚████╔╝ ", "  ╚═══╝  "),
    "E": ("███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"),
    "Y": ("██╗   ██╗", "╚██╗ ██╔╝", " ╚████╔╝ ", "  ╚██╔╝  ", "   ██║   ", "   ╚═╝   "),
    "S": ("███████╗", "██╔════╝", "███████╗", "╚════██║", "███████║", "╚══════╝"),
    "T": ("████████╗", "╚══██╔══╝", "   ██║   ", "   ██║   ", "   ██║   ", "   ╚═╝   "),
    "R": ("██████╗ ", "██╔══██╗", "██████╔╝", "██╔══██╗", "██║  ██║", "╚═╝  ╚═╝"),
    "I": ("██╗", "██║", "██║", "██║", "██║", "╚═╝"),
    "X": ("██╗  ██╗", "╚██╗██╔╝", " ╚███╔╝ ", " ██╔██╗ ", "██╔╝ ██╗", "╚═╝  ╚═╝"),
}
LOGO_WORD = "VEYSTRIX"


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def char_width(ch: str) -> int:
    """Terminal cells occupied by a character (CJK/emoji = 2, combining = 0)."""
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def visible_len(text: str) -> int:
    return sum(char_width(ch) for ch in strip_ansi(text))


def fit_exact(text: str, cols: int) -> str:
    """Pad (or truncate) to exactly ``cols`` terminal cells (Sonnet finding 26)."""
    n = visible_len(text)
    if n == cols:
        return text
    if n < cols:
        return text + " " * (cols - n)
    out, used = [], 0
    for ch in strip_ansi(text):
        w = char_width(ch)
        if used + w > cols:
            break
        out.append(ch)
        used += w
    return "".join(out) + " " * (cols - used)


def _spread(left: str, right: str, width: int) -> str:
    """``left`` then ``right`` flush against the right edge of ``width`` cells."""
    room = width - visible_len(right) - 1
    if not right or room < 8:
        return fit_exact(left, width)
    return fit_exact(_fit(left, room), room) + " " + right


@dataclass
class Entry:
    kind: str  # user | assistant | reasoning | tool | diff | system | error | cancelled
    text: str = ""
    id: str = ""
    name: str = ""
    args: dict = field(default_factory=dict)
    ok: bool | None = None
    collapsed: bool = False
    open: bool = True


@dataclass
class UIState:
    cols: int = 80
    rows: int = 24
    status: Status = field(default_factory=Status)
    entries: list = field(default_factory=list)
    input: str = ""
    cursor: int | None = None
    cwd: str = ""
    busy: bool = False
    spinner: int = 0
    show_reasoning: bool = False
    tools_collapsed: bool = False
    # help | palette | resume | settings | complete | harness | model | models | confirm
    overlay: str | None = None
    query: str = ""
    index: int = 0
    sessions: list = field(default_factory=list)
    settings_items: list = field(default_factory=list)
    completions: list = field(default_factory=list)
    permission: dict | None = None
    scroll: int = 0
    activity: str = ""
    notice: str = ""
    turn_started: float = 0.0
    first_token_at: float | None = None
    tokens: int = 0
    server_tok_s: bool = False  # True once real engine telemetry from /veystrix/status is active
    sanitize: object = None  # callable(str) -> str; hides harness surface text
    now: float = 0.0          # clock of the frame being composed (elapsed timers)
    harness: str = ""         # active harness key (home dashboard / settings only)
    harness_label: str = ""
    engine_state: str = ""    # ready | stopped | starting | stopping | verifying | needs port-check
    task: str = ""            # background job (model download) shown on the activity rail
    task_started: float = 0.0
    picker: list = field(default_factory=list)   # rows for the harness / model / models overlays
    picker_note: str = ""     # e.g. "scanning ~/models ..." while the inventory loads
    confirm: dict | None = None  # {"title", "lines", "yes", "no", "danger"} for the confirm modal

    def add(self, entry: Entry) -> Entry:
        self.entries.append(entry)
        return entry


def _open(state: UIState, kind: str) -> Entry:
    last = state.entries[-1] if state.entries else None
    if last is not None and last.kind == kind and last.open:
        return last
    for entry in state.entries[-3:]:
        if entry.kind in ("assistant", "reasoning"):
            entry.open = False
    return state.add(Entry(kind))


def apply_event(state: UIState, event, now: float = 0.0) -> None:
    """Reduce one normalised harness event into the UI state (pure; clock passed in)."""
    kind, data = event.kind, event.data
    if kind == "message.delta":
        entry = _open(state, "assistant")
        entry.text += data.get("text", "")
        state.tokens += 1
        if state.first_token_at is None:
            state.first_token_at = now
            state.status.ttft_s = max(now - state.turn_started, 0.0)
        span = now - state.first_token_at
        if span >= 0.25 and not state.server_tok_s:
            state.status.tok_s = (state.tokens - 1) / span
        state.activity = "responding"
    elif kind == "reasoning.delta":
        entry = _open(state, "reasoning")
        entry.text += data.get("text", "")
        if state.first_token_at is None:
            state.first_token_at = now
            state.status.ttft_s = max(now - state.turn_started, 0.0)
        state.activity = "thinking"
    elif kind == "tool.start":
        for e in state.entries[-3:]:
            if e.kind in ("assistant", "reasoning"):
                e.open = False
        state.add(Entry("tool", id=data.get("id", ""), name=data.get("name", "tool"),
                        args=dict(data.get("args") or {}), collapsed=state.tools_collapsed))
        state.activity = "running " + str(data.get("name", "tool"))
    elif kind in ("tool.output", "tool.end"):
        tool = next((e for e in reversed(state.entries)
                     if e.kind == "tool" and e.id == data.get("id")), None)
        if tool is None:
            return
        if kind == "tool.output":
            tool.text = (tool.text + ("\n" if tool.text and not tool.text.endswith("\n") else "")
                         + data.get("text", ""))[-20000:]
            return
        tool.ok, tool.open = bool(data.get("ok", True)), False
        if len(tool.text.splitlines()) > TOOL_OUTPUT_LINES:
            tool.collapsed = True
        if data.get("diff"):
            if tool.text.strip() == str(data["diff"]).strip():
                tool.text = ""          # the diff block below IS the output; do not print it twice
            state.add(Entry("diff", text=data["diff"]))
        state.activity = "working"
    elif kind == "permission.request":
        state.permission = dict(data)
    elif kind == "usage":
        output = int(data.get("output", 0))
        total = int(data.get("input", 0)) + output
        state.status.used = total or state.status.used
        if output > 0 and state.first_token_at is not None and not state.server_tok_s:
            span = now - state.first_token_at
            if span >= 0.5:
                state.status.tok_s = output / span
        if data.get("window"):
            state.status.window = int(data["window"])
        if data.get("cached"):
            state.status.cache = "cached"
    elif kind == "error":
        text = str(data.get("message", "error"))
        if callable(state.sanitize):
            text = state.sanitize(text) or "the harness reported an error"
        state.add(Entry("error", text=text))
    elif kind == "turn.end":
        state.busy = False
        state.permission = None
        state.activity = ""
        for e in state.entries:
            e.open = False
        if data.get("status") == "cancelled":
            state.status.cancelled = True
            state.add(Entry("cancelled"))


# ---- composition ---------------------------------------------------------------------
def _wrap(text: str, width: int, indent: str = "") -> list[str]:
    out: list[str] = []
    width = max(width, 8)
    for raw in text.split("\n"):
        if not raw.strip():
            out.append("")
            continue
        lead = indent or " " * (len(raw) - len(raw.lstrip(" ")))
        wrapped = textwrap.wrap(raw.rstrip(), width, initial_indent="", subsequent_indent=lead,
                                replace_whitespace=False, drop_whitespace=False,
                                break_long_words=True, break_on_hyphens=False) or [""]
        # drop_whitespace=False keeps the break's space at the start of a continuation line
        out.extend([wrapped[0].rstrip() or wrapped[0]]
                   + [lead + line[len(lead):].lstrip(" ").rstrip() for line in wrapped[1:]])
    return out


_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"\*\*([^*\n]+)\*\*")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_NUMBERED = re.compile(r"^(\s*)(\d{1,3}[.)])\s+(.*)$")
_RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_CODE_TOKEN = re.compile(
    r"(?P<comment>(?<![:\w])#(?![!\[{]).*$|//.*$)"
    r"|(?P<string>\"(?:[^\"\\]|\\.)*\"?|'(?:[^'\\]|\\.)*'?)"
    r"|(?P<number>\b\d+(?:\.\d+)?\b)"
    r"|(?P<word>[A-Za-z_][A-Za-z0-9_]*)")
_KEYWORDS = frozenset((
    "def class return if elif else for while in not and or is import from as with try except "
    "finally raise yield lambda pass break continue None True False async await global nonlocal "
    "let const var function fn pub impl struct enum match use mut new this export default "
    "interface type extends implements package func defer chan range select switch case "
    "then fi do done esac local static void int char bool string public private protected "
    "include null nil true false self Self where trait mod crate").split())


def highlight_code(line: str, theme: Theme) -> str:
    """Keyword / string / comment / number colouring for one code line (colour themes only)."""
    if theme.color == "none" or not line.strip():
        return line

    def paint(m: re.Match) -> str:
        if m.group("comment"):
            return theme.paint("dim", m.group(0))
        if m.group("string"):
            return theme.paint("ok", m.group(0))
        if m.group("number"):
            return theme.paint("warn", m.group(0))
        word = m.group("word")
        return theme.paint("brand", word) if word in _KEYWORDS else word
    return _CODE_TOKEN.sub(paint, line)


def _inline(text: str, theme: Theme) -> str:
    """**bold** loses its markers (weight in colour themes); `code` keeps its backticks so the
    cue survives without colour, and is tinted in colour themes."""
    text = _BOLD.sub(lambda m: theme.bold(m.group(1)), text)
    return _INLINE_CODE.sub(lambda m: theme.paint("accent", "`" + m.group(1) + "`"), text)


def _code_frame(lang: str, code: list[str], theme: Theme, w: int) -> list[str]:
    """A fenced block as a rounded frame with the language in the top border."""
    h, v, tl, tr, bl, br = (("-", "|", "+", "+", "+", "+") if theme.ascii_only
                            else ("─", "│", "╭", "╮", "╰", "╯"))
    width = max(w, 24)
    inner = width - 4
    label = f" {lang} "
    top = (theme.paint("dim", tl + h) + theme.paint("accent", label)
           + theme.paint("dim", h * max(width - 3 - len(label), 0) + tr))
    out = ["  " + top]
    for raw in code:
        text = fit_exact(_fit(raw.replace("\t", "    ").rstrip(), inner), inner)
        out.append("  " + theme.paint("dim", v) + " " + highlight_code(text, theme) + " " + theme.paint("dim", v))
    out.append("  " + theme.paint("dim", bl + h * (width - 2) + br))
    return out


def _assistant_lines(text: str, theme: Theme, w: int) -> list[str]:
    """Markdown-ish assistant typography: headings, framed fenced code with a language
    header and highlighting, bullets, numbered lists, quotes, rules, bold and inline code."""
    out = [""]
    code: list[str] | None = None
    lang = "code"
    bullet = theme.sym("bullet")
    for raw in text.split("\n"):
        stripped = raw.strip()
        if stripped.startswith("```"):
            if code is None:
                lang, code = stripped[3:].strip() or "code", []
            else:
                out.extend(_code_frame(lang, code, theme, w))
                code = None
            continue
        if code is not None:
            code.append(raw)
            continue
        heading = _HEADING.match(raw)
        if heading:
            level, title = len(heading.group(1)), _inline(heading.group(2), theme)
            color = "brand" if level == 1 else ("accent" if level == 2 else "text")
            out.append("  " + theme.paint(color, theme.bold(_fit(title, w))))
            continue
        if _RULE.match(raw):
            out.append("  " + theme.paint("dim", ("-" if theme.ascii_only else "─") * min(w, 72)))
            continue
        item = _BULLET.match(raw)
        number = _NUMBERED.match(raw) if not item else None
        if item or number:
            pad = (item or number).group(1)
            mark = (theme.paint("accent", bullet) if item else theme.paint("accent", number.group(2)))
            body = item.group(2) if item else number.group(3)
            mark_w = 1 if item else len(number.group(2))
            hang = " " * (len(pad) + mark_w + 1)
            for i, wrapped in enumerate(_wrap(body, w - len(hang))):
                out.append("  " + (pad + mark + " " if i == 0 else hang) + _inline(wrapped, theme))
            continue
        if stripped.startswith(">"):
            bar = theme.paint("dim", "|" if theme.ascii_only else "│")
            for wrapped in _wrap(stripped[1:].lstrip(), w - 2):
                out.append("  " + bar + " " + theme.paint("dim", _inline(wrapped, theme)))
            continue
        for wrapped in _wrap(raw, w):
            out.append("  " + _inline(wrapped, theme))
    if code is not None:                       # still streaming: frame what has arrived
        out.extend(_code_frame(lang, code, theme, w))
    return out


def _entry_lines(entry: Entry, state: UIState, theme: Theme, cols: int) -> list[str]:
    w = cols - 4
    if entry.kind == "user":
        bar = theme.paint("brand", "|" if theme.ascii_only else "┃")
        lead = theme.paint("brand", ">" if theme.ascii_only else "❯")
        body = _wrap(entry.text, w - 4)
        return ["", *[f"  {bar} " + (f"{lead} " if i == 0 else "  ") + theme.bold(line)
                      for i, line in enumerate(body)]]
    if entry.kind == "assistant":
        return _assistant_lines(entry.text, theme, w)
    if entry.kind == "reasoning":
        lines = entry.text.splitlines() or [""]
        if not state.show_reasoning:
            return [theme.paint("reason", f"  {theme.sym('reason')} reasoning ({len(lines)} lines, Ctrl-R to show)")]
        return reasoning_block("\n".join(_wrap(entry.text, w - 2)), theme, cols)
    if entry.kind == "tool":
        output = entry.text if entry.text else None
        lines = entry.text.splitlines()
        shown = output
        extra = 0
        if output is not None and len(lines) > TOOL_OUTPUT_LINES and not entry.collapsed:
            shown, extra = "\n".join(lines[-TOOL_OUTPUT_LINES:]), len(lines) - TOOL_OUTPUT_LINES
        block = tool_block(entry.name, entry.args, shown, theme, collapsed=entry.collapsed, width=cols - 2)
        marker = ("…" if entry.ok is None else (theme.paint("ok", theme.sym("ok")) if entry.ok
                                                 else theme.paint("err", theme.sym("err") + " failed")))
        if entry.ok is None:
            marker = theme.sym("spinner")[state.spinner % len(theme.sym("spinner"))]
        block[0] = _fit(block[0], cols - 14) + "  " + marker
        if entry.collapsed and lines:
            block[0] += theme.paint("dim", f"  (+{len(lines)} lines)")
        if extra:
            block.insert(1, theme.paint("dim", f"      … {extra} earlier lines"))
        gutter = theme.paint("dim", "|" if theme.ascii_only else "│")
        return [block[0]] + [line.replace(" │ ", f" {gutter} ", 1) if not theme.ascii_only else line
                             for line in block[1:]]
    if entry.kind == "diff":
        return diff_block(entry.text, theme, cols)
    if entry.kind == "error":
        return ["  " + theme.paint("err", theme.sym("err") + " " + _fit(entry.text, w))]
    if entry.kind == "cancelled":
        return [cancelled_line(theme)]
    dot = theme.paint("dim", "." if theme.ascii_only else "·")
    return [f"  {dot} " + theme.paint("dim", line) for line in _wrap(entry.text, w - 2)]


_ASCII_FOLD = str.maketrans({"│": "|", "┃": "|", "─": "-", "╭": "+", "╮": "+", "╰": "+", "╯": "+",
                             "┌": "+", "┐": "+", "└": "+", "┘": "+", "❯": ">", "●": "*",
                             "…": "...", "↓": "v", "↑": "^", "·": ".", "⏎": "\\n",
                             "█": "#", "░": "-", "◆": "*", "▸": ">", "◇": "o", "✓": "+", "✗": "x",
                             "○": "o", "◐": "~", "▲": "!", "•": "*", "━": "=", "—": "-", "–": "-",
                             "→": "->", "×": "x"})


def _asciify(text: str) -> str:
    """Pure-ASCII fallback for non-UTF-8 terminals: known glyphs mapped, the rest folded or '?'."""
    mapped = text.translate(_ASCII_FOLD)
    folded = "".join(c for c in unicodedata.normalize("NFKD", mapped) if not unicodedata.combining(c))
    return folded.encode("ascii", "replace").decode("ascii")


# ---- modal boxes ---------------------------------------------------------------------
def _box_width(cols: int) -> int:
    return min(cols - 6, 78)


def _box(title: str, rows_: list, cols: int, theme: Theme, height: int, *,
         footer: str = "", width: int | None = None, tone: str = "brand") -> list[str]:
    """Rounded modal: title in the top border, optional key footer in the bottom border.

    A row is a string, or ``("sel", text)`` for the highlighted selection band.
    """
    width = width or _box_width(cols)
    h, v, tl, tr, bl, br = (("-", "|", "+", "+", "+", "+") if theme.ascii_only
                            else ("─", "│", "╭", "╮", "╰", "╯"))
    inner = width - 2
    title_txt = _fit(f" {title} ", inner - 2)
    right_fill = max(inner - visible_len(title_txt) - 1, 0)
    top_bar = (theme.paint("dim", tl + h) + theme.paint(tone, theme.bold(title_txt))
               + theme.paint("dim", h * right_fill + tr))
    lines = [top_bar]
    edge = theme.paint("dim", v)
    for row in rows_[: max(height - 2, 1)]:
        selected = isinstance(row, tuple)
        text = row[1] if selected else row
        if visible_len(text) > inner - 2:       # keep a right margin: never touch the border
            text = _fit(text, inner - 2)
        cell = fit_exact(" " + text, inner)
        lines.append(edge + (theme.select(cell) if selected else cell) + edge)
    if footer:
        foot = " " + _fit(footer, inner - 4) + " "
        lines.append(theme.paint("dim", bl + h) + theme.paint("dim", foot)
                     + theme.paint("dim", h * max(inner - 1 - visible_len(foot), 0) + br))
    else:
        lines.append(theme.paint("dim", bl + h * inner + br))
    pad = " " * ((cols - width) // 2)
    return [pad + line for line in lines]


def _windowed(rows_: list, focus: int, limit: int) -> list:
    """At most ``limit`` rows, scrolled so row ``focus`` stays visible."""
    if len(rows_) <= limit or limit <= 0:
        return rows_
    start = min(max(focus - limit // 2, 0), len(rows_) - limit)
    return rows_[start:start + limit]


def _sep(theme: Theme) -> str:
    return " " + theme.sym("dot") + " "


def _keys(theme: Theme, *parts: str) -> str:
    return _sep(theme).join(parts)


# ---- home dashboard ------------------------------------------------------------------
def hero_logo(theme: Theme) -> list[str]:
    """The VEYSTRIX block-letter mark: gradient body + dim shadow, or plain '#' in ASCII."""
    letters = [_LOGO_GLYPHS[ch] for ch in LOGO_WORD]
    height = len(letters[0])
    if theme.ascii_only:
        rows_ = ["".join("#" if c == "█" else " " for c in "".join(glyph[r] for glyph in letters))
                 for r in range(height - 1)]
        return [row.rstrip() for row in rows_]
    out = []
    span = max(len(letters) - 1, 1)
    for r in range(height):
        parts = []
        for i, glyph in enumerate(letters):
            for run in re.findall(r"█+|[^█ ]+| +", glyph[r]):
                if run[0] == "█":
                    parts.append(theme.blend("brand", "accent", i / span, theme.bold(run)))
                elif run[0] == " ":
                    parts.append(run)
                else:
                    parts.append(theme.paint("dim", run))
        out.append("".join(parts))
    return out


def logo_width(theme: Theme) -> int:
    return max(visible_len(line) for line in hero_logo(theme))


_PILL = {"ready": ("on", "ok"), "starting": ("busy", "accent"), "stopping": ("busy", "accent"),
         "verifying": ("busy", "accent"), "needs port-check": ("alert", "warn"),
         "stopped": ("off", "dim")}


def engine_pill(state_name: str, theme: Theme) -> str:
    glyph, tone = _PILL.get(state_name or "stopped", ("off", "dim"))
    return theme.paint(tone, theme.sym(glyph) + " " + (state_name or "stopped"))


def short_path(path: str) -> str:
    home = os.path.expanduser("~")
    if path and home not in ("", "/", "~") and (path == home or path.startswith(home + os.sep)):
        return "~" + path[len(home):]
    return path


def _home_card(state: UIState, theme: Theme, cols: int) -> list[str]:
    width = _box_width(cols)
    inner = width - 3                      # _box prefixes one space
    half = inner // 2
    label = lambda text: theme.paint("dim", f"{text:<9}")   # noqa: E731

    def pair(left: str, right: str) -> str:
        return fit_exact(left, half) + right

    model = state.status.model or "not loaded"
    harness = state.harness_label or state.harness or "none"
    rows_ = [
        pair(label("Engine") + engine_pill(state.engine_state, theme),
             label("Model") + theme.paint("accent", _fit(model, half - 10))),
        pair(label("Effort") + (state.status.effort or "auto"),
             label("Harness") + _fit(harness, half - 10)),
        pair(label("KV mode") + (state.status.kv or "auto"),
             label("Thinking") + ("shown" if state.show_reasoning else "hidden")),
    ]
    if state.cwd:
        rows_.append(label("Folder") + theme.paint("dim", _fit(short_path(state.cwd), inner - 10)))
    rows_.append(theme.paint("dim", ("-" if theme.ascii_only else "─") * (inner - 1)))
    cells = [theme.paint("accent", f"{key:<11}") + theme.paint("dim", _fit(desc, half - 12))
             for key, desc in SHORTCUTS]
    for i in range(0, len(cells), 2):
        rows_.append(pair(cells[i], cells[i + 1] if i + 1 < len(cells) else ""))
    footer = "type a message and press Enter to start"
    return _box("Home", rows_, cols, theme, len(rows_) + 2, footer=footer)


def chat_header(theme: Theme, cols: int) -> list[str]:
    """The hero mark at the top of a conversation: part of the transcript, so it scrolls
    away as the chat grows."""
    logo = hero_logo(theme)
    lw = max(visible_len(line) for line in logo)
    tag = " " * max((cols - len(TAGLINE)) // 2, 0) + theme.paint("dim", TAGLINE)
    return ["", *[" " * ((cols - lw) // 2) + line for line in logo], tag]


def _home_screen(state: UIState, theme: Theme, cols: int, height: int) -> list[str]:
    """Hero logo + dashboard card, centred; degrades (tagline, then logo) when short."""
    card = _home_card(state, theme, cols)
    logo = hero_logo(theme)
    lw = max(visible_len(line) for line in logo)
    logo = [" " * ((cols - lw) // 2) + line for line in logo]
    tag = " " * max((cols - len(TAGLINE)) // 2, 0) + theme.paint("dim", TAGLINE)
    word = theme.paint("brand", theme.bold(f"{theme.sym('brand')} V E Y S T R I X"))
    word = " " * max((cols - 18) // 2, 0) + word
    for block in ([*logo, tag, "", *card], [*logo, tag, *card], [*logo, "", *card], [*logo, *card],
                  [word, "", *card], card):
        if len(block) <= height:
            break
    top = max((height - len(block)) // 2, 0)
    body = ([""] * top + block)[:height]
    return body + [""] * (height - len(body))


# ---- overlays ------------------------------------------------------------------------
def _human_size(n: int) -> str:
    size = float(max(n, 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


human_size = _human_size


def _model_rows(state: UIState, theme: Theme, inner: int, manager: bool) -> tuple[list, int]:
    rows_: list = []
    focus = 0
    items = state.picker
    sel = state.index % max(len(items), 1)
    for i, item in enumerate(items):
        active = bool(item.get("active"))
        mark = theme.paint("ok", theme.sym("on")) if active else " "
        name = str(item.get("name") or item.get("key"))
        if item.get("busy"):
            status_txt = theme.paint("accent", str(item["busy"]))
        elif item.get("installed"):
            status_txt = theme.paint("ok", theme.sym("ok") + " installed") + theme.paint(
                "dim", f"  {_human_size(int(item.get('size') or 0))}")
        elif item.get("partial"):
            status_txt = theme.paint("warn", theme.sym("warn") + " partial") + theme.paint(
                "dim", f"  {_human_size(int(item.get('size') or 0))}")
        else:
            status_txt = theme.paint("dim", theme.sym("cold") + " not downloaded")
        lead = theme.sym("sel") + " " if i == sel else "  "
        title = lead + mark + " " + (theme.paint("accent", theme.bold(name)) if i == sel else theme.bold(name))
        line1 = _spread(title, status_txt, inner)
        detail = str(item.get("key"))
        if item.get("context"):
            detail += f"  {theme.sym('dot')}  {int(item['context']) // 1024}K context"
        where = ""
        if manager and (item.get("installed") or item.get("partial")) and item.get("path"):
            where = short_path(str(item["path"]))
        elif manager and item.get("repo"):
            where = "hf: " + str(item["repo"])
        if where:
            sep = f"  {theme.sym('dot')}  "
            detail += sep + _fit_left(where, max(inner - 6 - len(detail) - len(sep), 12))
        line2 = "     " + theme.paint("dim", _fit(detail, inner - 6))
        if i == sel:
            focus = len(rows_)
            rows_ += [("sel", line1), ("sel", line2)]
        else:
            rows_ += [line1, line2]
    return rows_, focus


def _harness_rows(state: UIState, theme: Theme, inner: int) -> tuple[list, int]:
    rows_: list = []
    sel = state.index % max(len(state.picker), 1)
    for i, item in enumerate(state.picker):
        mark = theme.paint("ok", theme.sym("on")) if item.get("active") else " "
        lead = theme.sym("sel") + " " if i == sel else "  "
        label = f"{str(item.get('label') or item.get('key')):<13}"
        name = theme.paint("accent", theme.bold(label)) if i == sel else theme.bold(label)
        key = theme.paint("dim", f"{str(item.get('key')):<12}")
        if item.get("connected"):
            link = theme.paint("ok", theme.sym("ok") + " connected")
        else:
            link = theme.paint("dim", "  not connected")
        where = ("config only" if item.get("config_only")
                 else "installed" if item.get("installed") else "not installed")
        if item.get("installed") and not item.get("config_only"):
            where += " " + theme.sym("dot") + (" full UI" if item.get("structured") else " own screen")
        row = _spread(lead + mark + " " + name + key + link, theme.paint("dim", where), inner)
        rows_.append(("sel", row) if i == sel else row)
    return rows_, sel


def _settings_rows(state: UIState, theme: Theme, inner: int) -> tuple[list, int]:
    rows_: list = []
    focus = 0
    section = None
    items = state.settings_items
    sel = state.index % max(len(items), 1)
    for i, item in enumerate(items):
        label = item[0] if isinstance(item, (tuple, list)) else str(item)
        sec = item[2] if isinstance(item, (tuple, list)) and len(item) > 2 else None
        hint = item[3] if isinstance(item, (tuple, list)) and len(item) > 3 else ""
        if sec and sec != section:
            if section is not None:
                rows_.append("")
            rows_.append(theme.paint("brand", theme.bold(sec.upper())))
            section = sec
        lead = theme.sym("sel") + " " if i == sel else "  "
        text = theme.paint("accent", lead + label) if i == sel else lead + label
        row = _spread(text, theme.paint("dim", hint), inner) if hint else text
        if i == sel:
            focus = len(rows_)
            rows_.append(("sel", row))
        else:
            rows_.append(row)
    return rows_, focus


def _choice_rows(state: UIState, theme: Theme, inner: int) -> tuple[list, int]:
    rows_: list = []
    sel = state.index % max(len(state.picker), 1)
    for i, item in enumerate(state.picker):
        mark = theme.paint("ok", theme.sym("on")) if item.get("active") else " "
        lead = theme.sym("sel") + " " if i == sel else "  "
        key_txt = f"{str(item.get('label') or item.get('key')):<14}"
        name = theme.paint("accent", theme.bold(key_txt)) if i == sel else theme.bold(key_txt)
        desc = theme.paint("dim", str(item.get("desc") or ""))
        row = _spread(lead + mark + " " + name, desc, inner)
        rows_.append(("sel", row) if i == sel else row)
    return rows_, sel


def _overlay(state: UIState, theme: Theme, cols: int, height: int) -> list[str]:
    sel_mark = theme.sym("sel") + " "
    inner = _box_width(cols) - 4          # one cell of padding on each side of the content
    limit = max(height - 2, 1)
    nav = "↑↓ move"
    if state.overlay == "kv":
        rows_, focus = _choice_rows(state, theme, inner)
        rows_ = [theme.paint("dim", "Choose the KV cache precision (applies when a model starts)."), ""] + rows_
        return _box("KV cache mode", _windowed(rows_, focus + 2, limit), cols, theme, height,
                    footer=_keys(theme, nav, "Enter select", "Esc back"))
    if state.overlay == "effort":
        rows_, focus = _choice_rows(state, theme, inner)
        rows_ = [theme.paint("dim", "Choose reasoning effort (applies from the next message)."), ""] + rows_
        return _box("Reasoning effort", _windowed(rows_, focus + 2, limit), cols, theme, height,
                    footer=_keys(theme, nav, "Enter select", "Esc back"))
    if state.overlay == "help":
        rows_ = [theme.paint("brand", theme.bold(line)) if not line.startswith(" ") else line
                 for line in HELP_LINES]
        return _box("Veystrix help", rows_, cols, theme, height, footer=_keys(theme, "Esc close", "? toggle"))
    if state.overlay == "palette":
        items = palette_items(state.query)
        sel = state.index % max(len(items), 1)
        rows_ = [theme.paint("brand", theme.sym("tool")) + " " + state.query + ("_" if theme.ascii_only else "█"),
                 theme.paint("dim", ("-" if theme.ascii_only else "─") * (inner - 1))]
        listing = []
        for i, (label, name) in enumerate(items):
            row = _spread(theme.paint("accent", sel_mark + label) if i == sel else "  " + label,
                          theme.paint("dim", "/" + name), inner)
            listing.append(("sel", row) if i == sel else row)
        if not items:
            listing.append(theme.paint("dim", "  no matching command"))
        rows_ += _windowed(listing, sel, limit - 2)
        return _box("Command palette (Ctrl-K)", rows_, cols, theme, height,
                    footer=_keys(theme, nav, "Enter run", "Esc close"))
    if state.overlay == "resume":
        rows_ = []
        for i, s in enumerate(state.sessions[:10]):
            active = i == state.index % max(len(state.sessions), 1)
            label = str(s.get("title") or s.get("id"))
            rows_.append(("sel", theme.paint("accent", sel_mark + label)) if active else "  " + label)
        if not state.sessions:
            rows_.append(theme.paint("dim", "  no previous sessions"))
        return _box("Resume session", rows_, cols, theme, height,
                    footer=_keys(theme, nav, "Enter resume", "Esc cancel"))
    if state.overlay == "settings":
        rows_, focus = _settings_rows(state, theme, inner)
        return _box("Veystrix settings", _windowed(rows_, focus, limit), cols, theme, height,
                    footer=_keys(theme, nav, "Enter change / run", "Esc close"))
    if state.overlay == "harness":
        rows_, focus = _harness_rows(state, theme, inner)
        rows_ = [theme.paint("dim", "Pick the agent that drives Veystrix; Enter writes its config."), ""] + rows_
        if state.picker_note:
            rows_.append(theme.paint("dim", state.picker_note))
        return _box("Connect a harness", _windowed(rows_, focus + 2, limit), cols, theme, height,
                    footer=_keys(theme, nav, "Enter connect", "Esc back"))
    if state.overlay in ("model", "models"):
        manager = state.overlay == "models"
        cursor = "_" if theme.ascii_only else "█"
        search = (theme.paint("accent", "Search ") + state.query + theme.paint("brand", cursor)
                  if state.query else
                  theme.paint("accent", "Search ") + theme.paint("brand", cursor)
                  + theme.paint("dim", " type a name, family or repo to filter"))
        rows_, focus = _model_rows(state, theme, inner, manager)
        rows_ = [search, theme.paint("dim", ("-" if theme.ascii_only else "─") * (inner - 1)), *rows_]
        focus += 2
        if not state.picker:
            rows_.append(theme.paint("dim", "  " + (state.picker_note or (
                f"no model matches {state.query!r}" if state.query else "no models known"))))
        elif state.picker_note:
            rows_ += ["", theme.paint("dim", state.picker_note)]
        if manager:
            return _box("Model manager", _windowed(rows_, focus, limit), cols, theme, height,
                        footer=_keys(theme, nav, "Enter download", "Ctrl-D delete", "Esc close"))
        return _box("Choose a model", _windowed(rows_, focus, limit), cols, theme, height,
                    footer=_keys(theme, nav, "Enter select", "Esc back"))
    if state.overlay == "confirm" and state.confirm:
        c = state.confirm
        tone = "err" if c.get("danger") else "accent"
        width = min(_box_width(cols), 66)
        rows_ = [""] + [theme.bold(str(c["lines"][0]))] if c.get("lines") else [""]
        if c.get("path"):
            rows_.append("  " + theme.paint("dim", _fit_left(str(c["path"]), width - 8)))
        rows_ += [str(line) for line in list(c.get("lines", ()))[1:]] + [""]
        yes, no = c.get("yes", "confirm"), c.get("no", "cancel")
        return _box(str(c.get("title", "Confirm")), rows_, cols, theme, height, tone=tone,
                    footer=f"[y] {yes}   [n] {no}   Esc {no}", width=width)
    if state.overlay == "complete":
        shown = state.completions[:12]
        rows_ = []
        for i, item in enumerate(shown):
            label = item[0] if isinstance(item, (tuple, list)) else str(item)
            active = i == state.index % max(len(shown), 1)
            rows_.append(("sel", theme.paint("accent", sel_mark + label)) if active else "  " + label)
        return _box("Autocomplete", rows_, cols, theme, height,
                    footer=_keys(theme, nav, "Enter/Tab insert", "Esc close"))
    return []


def palette_items(query: str) -> list[tuple[str, str]]:
    q = query.strip().lower()
    return [item for item in PALETTE if not q or q in item[0].lower() or q in item[1]]


# ---- composer ------------------------------------------------------------------------
def _composer_rail(left_text: str, cols: int, theme: Theme, *, top: bool, right: str = "") -> str:
    """Rounded top or bottom border of the input composer, with an optional right label."""
    h = "-" if theme.ascii_only else "─"
    lcorner = "+" if theme.ascii_only else ("╭" if top else "╰")
    rcorner = "+" if theme.ascii_only else ("╮" if top else "╯")
    tail = (" " + right + " " + theme.paint("dim", h)) if right else ""
    trimmed = _fit(left_text, max(cols - 9 - visible_len(tail), 8))
    fill = max(cols - 7 - visible_len(trimmed) - visible_len(tail), 0)
    return (" " + theme.paint("dim", lcorner + h) + " " + trimmed + " "
            + theme.paint("dim", h * fill) + tail + theme.paint("dim", rcorner) + " ")


def _elapsed(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    return f"{seconds}s" if seconds < 60 else f"{seconds // 60}m {seconds % 60:02d}s"


def _activity_rail(state: UIState, theme: Theme, cols: int) -> str:
    spin = theme.sym("spinner")[state.spinner % len(theme.sym("spinner"))]
    dot = _sep(theme)
    if state.permission:
        p = state.permission
        text = (theme.paint("warn", f"{theme.sym('warn')} allow {p.get('action', 'action')}: "
                            f"{_fit(str(p.get('detail', '')), max(cols - 54, 10))}")
                + "  [y] once  [a] always  [n] reject")
    elif state.busy:
        label = state.activity or "working"
        if state.notice:
            label += f"  ({state.notice})"
        text = (theme.paint("brand", spin) + " " + theme.paint("accent", _fit(label, max(cols - 24, 10)))
                + theme.paint("dim", dot + _elapsed(state.now - state.turn_started)))
    elif state.task:
        text = (theme.paint("brand", spin) + " " + theme.paint("accent", _fit(state.task, max(cols - 24, 10)))
                + theme.paint("dim", dot + _elapsed(state.now - state.task_started)))
    elif state.status.cancelled:
        text = (theme.sym("cancelled") + " " + theme.paint("warn", "cancelled")
                + theme.paint("dim", "  send a message to continue, /resume for other sessions"))
    else:
        bullet = theme.paint("ok" if not state.notice else "accent", theme.sym("on"))
        text = bullet + " " + theme.paint("dim", state.notice or "ready")
    return _composer_rail(text, cols, theme, top=True)


def _input_row(state: UIState, theme: Theme, cols: int) -> str:
    nl = "\\n" if theme.ascii_only else "⏎"
    cur_glyph = "_" if theme.ascii_only else "█"
    vbar = theme.paint("dim", "|" if theme.ascii_only else "│")
    lead = theme.paint("brand", ">" if theme.ascii_only else "❯")
    pos = len(state.input) if state.cursor is None else max(0, min(state.cursor, len(state.input)))
    room = max(cols - 10, 10)
    ell = "..." if theme.ascii_only else "…"
    if pos == len(state.input):
        shown = state.input.replace("\n", nl)
        if len(shown) > room:
            shown = ell + shown[-(room - 1):]
        inner_prompt = f" {lead} " + shown + theme.paint("brand", cur_glyph)
    else:
        left = state.input[:pos].replace("\n", nl)
        right = state.input[pos:].replace("\n", nl)
        combined = left + cur_glyph + right
        if len(combined) > room:
            start = max(0, len(left) - room // 2)
            combined = ell + combined[start:start + room - 1]
        inner_prompt = f" {lead} " + combined
    return f" {vbar}" + fit_exact(inner_prompt, cols - 4) + f"{vbar} "


def _hint_rail(state: UIState, theme: Theme, cols: int) -> str:
    """Context-aware shortcut hints (what the next key does here) + context-window usage."""
    text = state.input
    token = text.rsplit(None, 1)[-1] if text.strip() else ""
    if state.busy:
        hints = ["Ctrl-C cancel", "PgUp scroll", "Ctrl-R reasoning", "Ctrl-T tools"]
    elif text.startswith("/") and " " not in text.strip():
        hints = ["Tab complete", "Enter run", "Esc clear"]
    elif token.startswith("@"):
        hints = ["Tab attach file", "Enter send", "Esc clear"]
    elif text:
        hints = ["Enter send", "Shift+Enter newline", "Esc clear", "Ctrl-U clear line"]
    else:
        hints = ["? help", "Ctrl-K palette", "/settings", "/model", "Tab complete", "Shift+Enter newline"]
    if state.scroll:
        hints.insert(0, f"{theme.sym('cold')} scrolled {state.scroll} lines (PgDn)")
    right = ""
    if state.status.window > 0:
        pct = min(round(100 * state.status.used / state.status.window), 999)
        right = theme.paint("warn" if pct >= 85 else "dim", f"ctx {pct}%")
    room = cols - 12 - (visible_len(right) + 3 if right else 0)
    sep = " | " if theme.ascii_only else " · "
    shown: list[str] = []
    for hint in hints:
        if visible_len(sep.join([*shown, hint])) > room:
            break
        shown.append(hint)
    return _composer_rail(theme.paint("dim", sep.join(shown)), cols, theme, top=False, right=right)


def compose_frame(state: UIState, theme: Theme, cols: int | None = None, rows: int | None = None) -> list[str]:
    """Pure: UI state -> ``rows`` lines of exactly ``cols`` visible chars."""
    cols = cols or state.cols
    rows = rows or state.rows
    if cols < MIN_COLS or rows < MIN_ROWS:
        msg = f"{theme.sym('warn')} terminal too small: need {MIN_COLS}x{MIN_ROWS}, have {cols}x{rows}"
        frame = [" " * cols for _ in range(rows)]
        frame[rows // 2] = fit_exact(_fit(msg, cols), cols)
        return frame
    head = status_line(state.status, theme, width=cols)
    rule = ("-" if theme.ascii_only else "─") * cols
    body_h = rows - 5
    # transcript, newest at the bottom
    flat: list[str] = []
    for entry in state.entries:
        flat.extend(_entry_lines(entry, state, theme, cols))
    if flat:
        flat[:0] = chat_header(theme, cols)
    if not flat:
        body = _home_screen(state, theme, cols, body_h)
    else:
        state.scroll = max(0, min(state.scroll, max(len(flat) - body_h, 0)))
        end = len(flat) - state.scroll
        window = flat[max(end - body_h, 0):end]
        body = ([""] * (body_h - len(window))) + window
    if state.overlay:
        over = _overlay(state, theme, cols, body_h)
        top = max((body_h - len(over)) // 2, 0)
        if not flat:
            body = [""] * body_h            # a modal over the dashboard replaces it (no bleed-through)
        else:                               # breathing room: blank the transcript rows that touch it
            for edge in (top - 1, top + len(over)):
                if 0 <= edge < body_h:
                    body[edge] = ""
        for i, line in enumerate(over[:body_h]):
            body[top + i] = line
    # composer: activity rail (row -5), input (row -4), hint rail (row -3), rule (row -2), status bar (row -1)
    lines = [*body, _activity_rail(state, theme, cols),
             _input_row(state, theme, cols), _hint_rail(state, theme, cols),
             theme.paint("dim", rule), head]
    if theme.ascii_only:
        lines = [_asciify(line) for line in lines]
    return [fit_exact(line, cols) for line in lines[:rows]]


def diff_rows(prev: list[str] | None, new: list[str]) -> list[tuple[int, str]]:
    """Bounded redraw: only rows whose content changed (all rows when ``prev`` is unknown or resized)."""
    if prev is None or len(prev) != len(new):
        return list(enumerate(new))
    return [(i, line) for i, (a, line) in enumerate(zip(prev, new)) if a != line]


def paint_rows(changed: list[tuple[int, str]], cursor: tuple[int, int] | None = None) -> str:
    """Terminal bytes for a bounded redraw (one synchronized-output burst)."""
    out = ["\033[?2026h"]
    for row, text in changed:
        out.append(f"\033[{row + 1};1H{text}\033[0m")
    if cursor:
        out.append(f"\033[{cursor[0] + 1};{cursor[1] + 1}H")
    out.append("\033[?2026l")
    return "".join(out)


# ---- capture helpers (SVG snapshot, asciicast) ---------------------------------------
def _xterm256(n: int) -> tuple[int, int, int]:
    if n < 16:
        base = [(0, 0, 0), (205, 0, 0), (0, 205, 0), (205, 205, 0), (0, 0, 238), (205, 0, 205),
                (0, 205, 205), (229, 229, 229), (127, 127, 127), (255, 0, 0), (0, 255, 0),
                (255, 255, 0), (92, 92, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255)]
        return base[n]
    if n < 232:
        n -= 16
        level = (0, 95, 135, 175, 215, 255)
        return level[n // 36], level[(n // 6) % 6], level[n % 6]
    v = 8 + (n - 232) * 10
    return v, v, v


def frame_to_svg(lines: list[str], dark: bool = True, font_px: int = 14) -> str:
    """Render an ANSI frame as a standalone SVG (colour runs preserved)."""
    from xml.sax.saxutils import escape
    cw, ch = font_px * 0.6, font_px * 1.25
    cols = max((visible_len(line) for line in lines), default=0)
    bg, fg = ("#0f0f14", "#e8e8ee") if dark else ("#fafafa", "#1c1c24")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{cols * cw + 16:.0f}" '
           f'height="{len(lines) * ch + 16:.0f}" font-family="Menlo,Consolas,monospace" font-size="{font_px}">',
           f'<rect width="100%" height="100%" fill="{bg}"/>']
    for r, line in enumerate(lines):
        runs, color, pos = [], fg, 0
        for m in _ANSI.finditer(line):
            if m.start() > pos:
                runs.append((color, line[pos:m.start()]))
            pos = m.end()
            code = m.group(0)[2:-1]
            if code in ("", "0"):
                color = fg
            elif code.startswith("38;2;"):
                red, green, blue = (int(x) for x in code.split(";")[2:5])
                color = f"#{red:02x}{green:02x}{blue:02x}"
            elif code.startswith("38;5;"):
                red, green, blue = _xterm256(int(code.split(";")[2]))
                color = f"#{red:02x}{green:02x}{blue:02x}"
        if pos < len(line):
            runs.append((color, line[pos:]))
        spans = "".join(f'<tspan fill="{c}">{escape(t)}</tspan>' for c, t in runs if t)
        out.append(f'<text x="8" y="{8 + (r + 1) * ch - 4:.1f}" xml:space="preserve">{spans}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def asciicast(records: list[tuple[float, str]], cols: int, rows: int, title: str = "Veystrix UI") -> str:
    """asciicast v2 text from ``(seconds, terminal_bytes)`` records (deterministic: no wall clock)."""
    header = {"version": 2, "width": cols, "height": rows, "timestamp": 0, "title": title,
              "env": {"TERM": "xterm-256color"}}
    lines = [json.dumps(header)]
    lines += [json.dumps([round(t, 4), "o", data]) for t, data in records]
    return "\n".join(lines) + "\n"
