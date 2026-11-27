"""Pure-Python supervisor restart policy, swap check, and server launch command."""
from __future__ import annotations

import re
import subprocess
import sys
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

GUARD_EXIT = 75
ADMISSION_EXIT = 69
CONFIG_EXIT = 70
PORT_BUSY_EXIT = 71
POWER_EXIT = 72


def bundled() -> bool:
    """True inside the packaged app: there is no chat.sh or source tree, only compiled modules and a bundled Python."""
    return Path(sys.executable).resolve().is_relative_to((REPO.parent / "python").resolve())


def server_command(args) -> list[str]:
    """Build the CLI server launch command for the C++ engine."""
    launcher = [sys.executable, "-m", "veystrix.cli"] if bundled() else [str(REPO / "chat.sh")]
    model_name = getattr(args, "model", "inkling2-mini")
    command = [
        *launcher, "--server", "--host", args.host, "--port", str(args.port),
        "--model", model_name, "--checkpoint", args.checkpoint,
    ]
    if getattr(args, "context_limit", None):
        command += ["--context-limit", str(args.context_limit)]
    if getattr(args, "kv", None):
        command += ["--kv", str(args.kv)]
    if getattr(args, "prefix_dir", None):
        command += ["--prefix-dir", args.prefix_dir]
    if getattr(args, "preferences_file", None):
        command += ["--preferences-file", args.preferences_file]
    return command


def swap_used_mb() -> float:
    out = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
        capture_output=True, text=True, timeout=5,
    ).stdout
    match = re.search(r"used = ([\d.]+)M", out)
    return float(match.group(1)) if match else 0.0


class Policy:
    """Pure restart decisions (no clock, no process)."""

    def __init__(self, *, max_restarts=8, window_s=1800.0, max_guard_stops=4, guard_window_s=21600.0,
                 base_cooldown_s=20.0, guard_cooldown_s=180.0, max_cooldown_s=600.0):
        self.max_restarts, self.window_s = max_restarts, window_s
        self.max_guard_stops, self.guard_window_s = max_guard_stops, guard_window_s
        self.base_cooldown_s, self.guard_cooldown_s, self.max_cooldown_s = base_cooldown_s, guard_cooldown_s, max_cooldown_s
        self.restarts = deque()
        self.guard_stops = deque()
        self.consecutive_failures = 0
        self.last_failure_time = None

    @staticmethod
    def _prune(queue, now, window):
        while queue and now - queue[0] > window:
            queue.popleft()

    def decide(self, exit_code: int, now: float, *, owner_stop: bool = False) -> tuple:
        """Return ``(action, cooldown_seconds, reason)`` with action in {"stop", "restart"}."""
        if owner_stop or exit_code == 0:
            self.consecutive_failures = 0
            self.last_failure_time = None
            return "stop", 0.0, "stopped by the owner" if owner_stop else "server exited cleanly"
        if exit_code == PORT_BUSY_EXIT:
            return "stop", 0.0, "the port is already served by another process: stop it first"
        if exit_code == POWER_EXIT:
            return "stop", 0.0, "the battery is below the minimum (exit 72): plug in or charge, then start again; restarting cannot fix it"
        if exit_code == CONFIG_EXIT:
            return "stop", 0.0, "launch configuration refused (exit 70): restarting cannot fix it"

        if self.last_failure_time is not None and now - self.last_failure_time > self.window_s:
            self.consecutive_failures = 0
        self.last_failure_time = now
        self.consecutive_failures += 1

        self._prune(self.restarts, now, self.window_s)
        self._prune(self.guard_stops, now, self.guard_window_s)
        if exit_code in (GUARD_EXIT, ADMISSION_EXIT):
            self.guard_stops.append(now)
            if len(self.guard_stops) > self.max_guard_stops:
                return "stop", 0.0, (f"{len(self.guard_stops)} memory stops within {self.guard_window_s / 3600:g} h "
                                     f"(limit {self.max_guard_stops}): the machine cannot sustain this load, so the supervisor stops")
            self.restarts.append(now)
            return "restart", self.guard_cooldown_s, f"memory guard stop (exit {exit_code}); restarting after cooldown"
        self.restarts.append(now)
        failures = max(self.consecutive_failures, len(self.restarts))
        if failures > self.max_restarts:
            return "stop", 0.0, f"{failures} restarts within {self.window_s / 60:g} min (limit {self.max_restarts}): crash loop"
        cooldown = min(self.max_cooldown_s, self.base_cooldown_s * (2 ** (self.consecutive_failures - 1)))
        return "restart", cooldown, f"server exited with code {exit_code}; restarting after cooldown"
