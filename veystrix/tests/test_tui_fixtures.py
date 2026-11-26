"""Shared offline fixtures for the D6 UI tests (no harness binary, no model, no GPU).

* recorded event streams under ``golden_ui/`` (shaped per the documented opencode v2 SSE and
  codex app-server schemas; NOT captured from a live model)
* ``FakeClock``/``ScreenMirror``/``drive`` : a synthetic clock and a virtual terminal that
  applies the exact bytes UILoop writes, so "no dropped frames" is checkable
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from veystrix.tool import renderer as R
from veystrix.tool.harness_adapters import (Event, ReplayAdapter, parse_sse, translate_codex,
                                            translate_opencode)
from veystrix.tool.tui import UILoop

GOLDEN = Path(__file__).parent / "golden_ui"
SIZES = ((80, 24), (120, 40), (200, 60))
THEMES = {
    "dark": R.Theme(color="true", dark=True),
    "light": R.Theme(color="true", dark=False),
    "no-color": R.Theme(color="none"),
    "ascii": R.Theme(color="none", ascii_only=True),
}


def opencode_events(name: str = "opencode_turn.sse", session: str = "ses_demo") -> list[Event]:
    state = {"session": session}
    out: list[Event] = []
    with open(GOLDEN / name) as handle:
        for payload in parse_sse(handle):
            out.extend(translate_opencode(payload, state))
    return out


def codex_events(name: str = "codex_turn.jsonl") -> list[Event]:
    state: dict = {}
    out: list[Event] = []
    for line in (GOLDEN / name).read_text().splitlines():
        if line.strip():
            out.extend(translate_codex(json.loads(line), state))
    return out


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t


class ScreenMirror:
    """Virtual terminal fed with the bytes UILoop writes (cursor-addressed rows only)."""

    ROW = re.compile(r"\033\[(\d+);1H(.*?)\033\[0m(?=\033\[\d+;1H|\033\[\?2026l|$)", re.S)

    def __init__(self, rows: int) -> None:
        self.rows = [""] * rows
        self.writes: list[tuple[float, str]] = []
        self.clock: FakeClock | None = None

    def write(self, data: str) -> None:
        self.writes.append((self.clock.now() if self.clock else 0.0, data))
        if "\033[2J" in data:
            self.rows = [""] * len(self.rows)
            data = data.replace("\033[2J", "")
        for match in self.ROW.finditer(data):
            index = int(match.group(1)) - 1
            if index >= len(self.rows):
                self.rows.extend([""] * (index + 1 - len(self.rows)))
            self.rows[index] = match.group(2)


def make_loop(adapter, *, cols=120, rows=40, theme=None, engine="cpp", status=None, sizes=None, **kw):
    clock = FakeClock()
    box = {"size": (cols, rows)} if sizes is None else sizes
    mirror = ScreenMirror(rows)
    mirror.clock = clock
    status = status or R.Status(engine=engine, model="flashnext", used=0, window=32768,
                                sampling="thinking", cache="cold")
    loop = UILoop(adapter, write=mirror.write, size=lambda: box["size"], theme=theme or THEMES["no-color"],
                  status=status, **kw)
    return loop, clock, mirror, box


def drive(loop, clock, keys_at=(), until=10.0):
    """Run the loop on the synthetic clock. ``keys_at`` is ``[(t, key)]``.

    The real elapsed compute time of each step is added to the synthetic clock, so a slow
    compose/diff shows up in the latency numbers while sleeping costs nothing. The driver
    wakes for a key the instant it arrives (as ``select`` on stdin does).
    Returns ``[(key, seconds from arrival until the frame containing it was written)]``.
    """
    pending = sorted(keys_at, key=lambda item: item[0])   # stable: same-time keys keep their order
    latencies: list[tuple[str, float]] = []
    while clock.t < until:
        due = [(t, k) for t, k in pending if t <= clock.t]
        pending = [(t, k) for t, k in pending if t > clock.t]
        began = time.perf_counter()
        frames = loop.frames
        loop.step(clock.t, [k for _, k in due])
        clock.t += time.perf_counter() - began
        if due and loop.frames > frames:
            latencies.extend((k, clock.t - t) for t, k in due)
        quiet = (not loop.state.busy and not loop.dirty and not pending
                 and not getattr(loop.adapter, "_pending", []))
        if loop.done or quiet:
            break
        horizon = loop.next_wakeup(clock.t)
        if pending:
            horizon = min(horizon, max(pending[0][0] - clock.t, 0.0))
        clock.t += max(horizon, 0.0005)
    return latencies


def scenario_adapter(events: list[Event] | None = None) -> ReplayAdapter:
    return ReplayAdapter([events if events is not None else opencode_events()],
                         sessions=[{"id": "ses_1", "title": "fix parser bug"},
                                   {"id": "ses_2", "title": "add tests for cache"}])


def run_turn(adapter, *, cols, rows, theme, engine="cpp", prompt="fix the bug in app.py"):
    """Type a prompt, replay the recorded turn, return (loop, mirror) once the turn ended."""
    loop, clock, mirror, _ = make_loop(adapter, cols=cols, rows=rows, theme=theme, engine=engine)
    keys = [(0.01 * (i + 1), ch) for i, ch in enumerate(prompt)]
    keys.append((0.01 * (len(prompt) + 2), "enter"))
    drive(loop, clock, keys, until=30.0)
    clock.t += 0.2
    loop.step(clock.t)
    return loop, mirror
