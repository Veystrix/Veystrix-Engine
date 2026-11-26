"""Engine supervisor for the terminal tool (ported from EngineController/EngineProcess).

Key rules carried over from the app:
  * loopback only, always (127.0.0.1);
  * a guard stop is NOT a crash: never auto-restart after one;
  * crash restarts are bounded (at most 3) with backoff;
  * battery below the configured floor (default 25%) refuses a start, AC always passes;
  * a memory preflight and bundle integrity are checked at start;
  * every engine path goes through the ``--engine`` / ``VEYSTRIX_ENGINE`` selector.

The watcher is a detached process so ``veystrix start`` can return while supervision
continues. Tests inject power/memory state with VEYSTRIX_TEST_POWER / VEYSTRIX_TEST_MEMORY
and run a mock server with ``--command``; no real checkpoint or harness config is touched.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from veystrix.tool.config import Config, LocalKeyStore, home_dir, sys_is_macos, tool_home
from veystrix.tool.engine import bundle_integrity, check_engine, check_protocol, report_engine
from veystrix.tool.engine import bundle_root, source_cpp_ready

MAX_RESTARTS = 3
_WATCHERS: dict[int, subprocess.Popen] = {}     # reaped in-process by stop() to avoid zombies
GUARD_MARKERS = (
    "memory_admission_refused", "guard stop", "safety stop", "safety guard", "memory guard",
    "swap ceiling", "nothing was loaded", "refused", "memory pressure", "watchdog",
)
GUARD_EXIT = 86
GUARD_EXIT_CODES = (86, 75)
_MIN_AVAILABLE_BYTES = 3 * 1024**3


@dataclass
class StartOptions:
    home: Path | None = None
    port: int = 8080
    engine: str = "python"
    engine_profile: str = "agent"  # cpp parity profile to gate (Track A attests 'agent')
    model: str = "inkling2-mini"
    checkpoint: str = ""
    command: list[str] | None = None
    foreground: bool = False
    timeout: float = 30.0
    from_bundle: bool = False
    kv: str = ""                 # generic runtime KV mode ("" = config / auto)
    context: int = 0             # context ceiling (0 = the model's own)
    host: str = "127.0.0.1"      # 0.0.0.0 exposes the endpoint on the LAN (API key required)
    on_ready: object = None      # callable(status dict) once /health is ready (prints the endpoint block)
    environ: dict = field(default_factory=lambda: dict(os.environ))


def run_dir(home: Path | None = None) -> Path:
    return tool_home(home) / "run"


def state_path(home: Path | None = None) -> Path:
    return run_dir(home) / "supervisor.json"


def log_path(home: Path | None = None) -> Path:
    return tool_home(home) / "logs" / "engine.log"


def stop_marker(home: Path | None = None) -> Path:
    return run_dir(home) / "stop"


def health(port: int, timeout: float = 1.0, api_key: str | None = None) -> dict | None:
    """GET /health on loopback only, never through a proxy. None means unreachable."""
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", "/health", headers=headers)
        response = conn.getresponse()
        if response.status != 200:
            return None
        data = json.loads(response.read().decode("utf-8", "replace"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, http.client.HTTPException):
        return None
    finally:
        conn.close()


def power_state(environ=None, runner=None) -> dict:
    environ = os.environ if environ is None else environ
    override = environ.get("VEYSTRIX_TEST_POWER")
    if override:
        try:
            data = json.loads(override)
            if isinstance(data, dict):
                return {"on_ac": data.get("on_ac"), "percent": data.get("percent")}
        except ValueError:
            pass
    if not sys_is_macos():
        return {"on_ac": None, "percent": None}
    runner = runner or subprocess.run
    try:
        result = runner(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return {"on_ac": None, "percent": None}
    text = result.stdout or ""
    on_ac = "AC Power" in text if "Power" in text else None
    percent = None
    import re
    match = re.search(r"(\d+)%", text)
    if match:
        percent = int(match.group(1))
    return {"on_ac": on_ac, "percent": percent}


def battery_refusal(state: dict, min_percent: int) -> str | None:
    """AC power always passes; unreadable power refuses before launching anything."""
    if min_percent <= 0 or state.get("on_ac") is True:
        return None
    if state.get("on_ac") is None:
        return "Power source could not be read; plug in and try again."
    percent = state.get("percent")
    if not isinstance(percent, int):
        return "On battery but charge could not be read; plug in and try again."
    if percent < min_percent:
        return f"Battery is {percent}%, below the {min_percent}% minimum. Plug in or charge before starting."
    return None


def memory_state(environ=None, runner=None) -> dict:
    environ = os.environ if environ is None else environ
    override = environ.get("VEYSTRIX_TEST_MEMORY")
    if override:
        try:
            data = json.loads(override)
            if isinstance(data, dict):
                return {"available_bytes": data.get("available_bytes")}
        except ValueError:
            pass
    if not sys_is_macos():
        return {"available_bytes": None}
    runner = runner or subprocess.run
    try:
        stat = runner(["vm_stat"], capture_output=True, text=True, timeout=5, check=False).stdout or ""
        pages = {}
        for line in stat.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                digits = "".join(ch for ch in value.strip() if ch.isdigit())
                if digits:
                    pages[key.strip()] = int(digits)
        # Apple silicon pages are 16 KiB; vm_stat states the size on its first line.
        match = re.search(r"page size of (\d+) bytes", stat)
        page = int(match.group(1)) if match else 4096
        available = sum(pages.get(key, 0) for key in
                        ("Pages free", "Pages inactive", "Pages purgeable")) * page
        return {"available_bytes": available}
    except (OSError, subprocess.SubprocessError):
        return {"available_bytes": None}


def memory_refusal(state: dict, minimum: int = _MIN_AVAILABLE_BYTES) -> str | None:
    available = state.get("available_bytes")
    if not isinstance(available, int):
        return None                      # unknown is an advisory, not a refusal
    if available < minimum:
        return f"Only {available / 1024**3:.1f} GiB of memory looks available; the engine needs about {minimum / 1024**3:.0f} GiB."
    return None


def guard_stop(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in GUARD_MARKERS)


def next_action(log_tail: str, exit_code: int, restarts: int, supervised: bool) -> str:
    """Crash vs guard stop. A guard stop is NEVER a crash and is never restarted."""
    if guard_stop(log_tail) or exit_code in GUARD_EXIT_CODES:
        return "guard_stopped"
    if not supervised or restarts >= MAX_RESTARTS:
        return "failed"
    return "restart"


def build_command(*, port: int, model: str, checkpoint: str, environ: dict,
                  engine: str = "python", home: Path | None = None, host: str = "127.0.0.1") -> list[str]:
    """Launch contract for the engine-side process (loopback only).

    ``python`` (the default) is untouched: the source checkout is launched as
    ``python -m veystrix.model_router`` exactly as before, and ``environ`` is
    not modified.

    ``cpp`` is binary-only and comes from the installed bundle. It is launched
    through the bundle's engine-side CLI, ``engine/bin/veystrix-engine serve
    ... --engine cpp`` (the documented front-end/engine contract). The bundle
    root is ``<home>/.veystrix/engine``; ``home`` falls back to ``VEYSTRIX_HOME``
    then the user home, matching ``veystrix.tool.engine.bundle_root``. The cpp
    selection is mirrored into the child env by ``start()`` (``engine_env``), not
    here, so this function is pure. A refused cpp request never reaches here:
    ``check_engine()`` runs in ``start()`` before any command is built.
    """
    resolved_home = home or Path(environ.get("VEYSTRIX_HOME") or Path.home())
    # Developer checkout with its own C++ build and no installed bundle: serve through the
    # same router as python, with VEYSTRIX_ENGINE=cpp selecting the C ABI backend.
    source_cpp = engine == "cpp" and source_cpp_ready(resolved_home, environ)
    if engine == "cpp" and not source_cpp:
        engine_cli = bundle_root(resolved_home) / "bin" / "veystrix-engine"
        command = [str(engine_cli), "serve",
                   "--port", str(port), "--model", model]
        if checkpoint:
            command += ["--checkpoint", checkpoint]
        command += ["--engine", "cpp"]
        # The child selection is carried in `state["engine_env"]` by start() and
        # applied in watch(); build_command stays pure and does not mutate the
        # caller's environ (which start() reuses as the foreground env seed).
        return command

    # Default python path: byte-for-byte the original source-checkout contract.
    root = Path(__file__).resolve().parents[2]
    chosen = environ.get("VEYSTRIX_PYTHON") or str(root / ".venv" / "bin" / "python")
    python = chosen if Path(chosen).is_file() else sys.executable
    command = [python, "-u", "-m", "veystrix.model_router",
               "--host", host if host in ("127.0.0.1", "0.0.0.0") else "127.0.0.1",
               "--port", str(port), "--model", model]
    if checkpoint:
        command += ["--checkpoint", checkpoint]
    return command


def engine_env_for(engine: str, environ: dict) -> dict:
    """Engine-side selector vars carried into the supervised child process.

    ``VEYSTRIX_CPP_BATCH`` is opt-in and not a credential. It is normally
    inherited through os.environ, but mirroring it into ``engine_env`` for cpp
    also covers a custom ``options.environ`` or a ``--command`` override. The
    default python path adds nothing (byte-for-byte unchanged).
    """
    if engine != "cpp":
        return {}
    env = {"VEYSTRIX_ENGINE": "cpp"}
    from veystrix.speed import KNOBS
    for name in ("VEYSTRIX_KV_MODE", "VEYSTRIX_CONTEXT_LIMIT", *KNOBS.values()):
        value = environ.get(name)
        if value:
            env[name] = str(value)
    flag = environ.get("VEYSTRIX_CPP_BATCH") or os.environ.get("VEYSTRIX_CPP_BATCH")
    if str(flag) == "1":
        env["VEYSTRIX_CPP_BATCH"] = "1"
    return env


def apply_cpp_worker_env(env: dict) -> dict:
    """Process-level settings the C++ engine worker must start with (read once at process start).

    ``MallocLargeCache=0``: macOS malloc keeps freed large blocks resident. Loading the dense weights
    left 1.1 GB of them (Nemotron) that the page cache could not use, so every decode token read
    ~20% more experts from the SSD; with the cache off the worker's footprint drops by ~1 GB and decode
    is faster. A value the user already set wins. The Python engine is not touched.
    """
    if env.get("VEYSTRIX_ENGINE") == "cpp":
        env.setdefault("MallocLargeCache", "0")
    return env


def _write_state(home: Path, data: dict) -> None:
    path = state_path(home)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _read_state(home: Path) -> dict | None:
    try:
        data = json.loads(state_path(home).read_text())
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _tail(path: Path, count: int = 60) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-count:])


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.terminate()
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass


def start(options: StartOptions) -> dict:
    home = options.home or home_dir()
    config = Config(home)
    port = options.port or config.get("port")
    from veystrix.tool.swap import engine_for
    from veystrix import model_identity
    host = options.host or config.get("host")
    if host != "127.0.0.1" and not (config.get("require_api_key") and LocalKeyStore(home).load()):
        raise SystemExit("veystrix: --host 0.0.0.0 exposes the endpoint to the local network; turn on the API key "
                         "first: veystrix config set require-api-key true (Tailscale Serve needs no LAN exposure: "
                         "docs/REMOTE_ACCESS.md)")
    ident = model_identity.for_family(options.model)
    if options.command:
        engine = options.engine or "python"
    elif options.engine and options.engine in ident.engines:
        engine = options.engine
    else:
        engine = engine_for(options.model, home=home)
    if not options.command:
        check_engine(engine, home, options.model, options.engine_profile)
    kv = options.kv or (config.get("kv") if config.get("kv") != "auto" else "")
    context = options.context or config.get("context")
    if kv:
        options.environ["VEYSTRIX_KV_MODE"] = kv
    if context:
        options.environ["VEYSTRIX_CONTEXT_LIMIT"] = str(context)
    # Maximum-speed switches (benchmark settings) are process-level: forward
    # them into both the child command and the supervised watcher's env seed.
    from veystrix.speed import speed_env
    for name, value in speed_env(config.get("speed_mode"), config.get("speed_knobs"),
                                 options.model, options.environ).items():
        options.environ.setdefault(name, value)
    existing = health(port, api_key=LocalKeyStore(home).load())
    if existing is not None:
        if existing.get("ready", True) and existing.get("status") != "fault":
            raise SystemExit(f"veystrix: port {port} is already answering /health; stop that engine first.")
        stop(home)

    refusal = battery_refusal(power_state(options.environ), config.get("minimum_battery_percent"))
    if refusal:
        raise SystemExit("veystrix: " + refusal)
    refusal = memory_refusal(memory_state(options.environ))
    if refusal:
        raise SystemExit("veystrix: " + refusal)
    # cpp is bundle-only by definition: always run the bundle checks, even
    # without --from-bundle, so a tampered/missing engine cannot be launched.
    if not options.command and ((engine == "cpp" and not source_cpp_ready(home, options.environ)) or options.from_bundle):
        problems = bundle_integrity(home)
        if problems:
            raise SystemExit("veystrix: bundle integrity failed:\n  " + "\n  ".join(problems))
        check_protocol(home)

    # The engine process is launched by watch() (or subprocess.run in the
    # foreground), not by this process. build_command() selects the cpp bundle
    # CLI for engine=cpp; carry the cpp selection through in the child
    # environment seed too, including for --command test overrides (which skip
    # build_command). The default python path leaves the seed untouched.
    command_env = dict(options.environ)
    # `veystrix start --model X` serves X: a request naming another installed model gets a clear 409 instead of
    # silently unloading X (the router's coordinator is pinned). VEYSTRIX_MODEL_HANDOFF=auto still allows switching.
    command_env.setdefault("VEYSTRIX_MODEL_PIN", "1")
    if engine == "cpp":
        command_env["VEYSTRIX_ENGINE"] = "cpp"
        apply_cpp_worker_env(command_env)
    command = list(options.command) if options.command else build_command(
        port=port, model=options.model, checkpoint=options.checkpoint,
        environ=command_env, engine=engine, home=home, host=host)
    if options.foreground:
        key = LocalKeyStore(home).load()
        if key and host != "127.0.0.1":
            command_env["VEYSTRIX_SERVER_API_KEY"] = key
        stop_marker(home).unlink(missing_ok=True)
        process = subprocess.Popen(command, cwd=str(Path(__file__).resolve().parents[2]), env=command_env)
        state = {"port": port, "engine": engine, "model": options.model, "command": command,
                 "host": host, "kv": kv or "auto", "context": context or 0,
                 "engine_env": engine_env_for(engine, options.environ),
                 "pin_model": command_env["VEYSTRIX_MODEL_PIN"] == "1",
                 "started_at": time.time(), "restarts": 0, "status": "starting",
                 "pid": os.getpid(), "engine_pid": process.pid, "supervised": False,
                 "foreground": True}
        _write_state(home, state)
        stopping = False
        prev_signals = {}

        def _on_term(_signum, _frame):
            nonlocal stopping
            stopping = True

        for _sig in (signal.SIGHUP, signal.SIGTERM):
            try:
                prev_signals[_sig] = signal.signal(_sig, _on_term)
            except (OSError, ValueError):
                pass
        announced = False
        try:
            while not stopping and process.poll() is None:
                if not announced:
                    report = health(port, timeout=0.5, api_key=key)
                    if report is not None and report.get("ready", True):
                        announced = True
                        state["status"] = "running"
                        _write_state(home, state)
                        if callable(options.on_ready):
                            options.on_ready({"status": "running", "port": port, "engine": engine,
                                              "model": options.model, "host": host, "kv": kv, "context": context,
                                              "foreground": True})
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass
        finally:
            for _sig, _handler in prev_signals.items():
                try:
                    signal.signal(_sig, _handler)
                except (OSError, ValueError):
                    pass
            _terminate(process)
            state_path(home).unlink(missing_ok=True)
            stop_marker(home).unlink(missing_ok=True)
        return {"status": "stopped", "exit_code": process.returncode}

    log = log_path(home)
    log.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    stop_marker(home).unlink(missing_ok=True)
    state = {"port": port, "engine": engine, "model": options.model, "command": command,
             "host": host, "kv": kv or "auto", "context": context or 0,
             "engine_env": engine_env_for(engine, options.environ),
             "pin_model": command_env["VEYSTRIX_MODEL_PIN"] == "1",
             "started_at": time.time(), "restarts": 0, "status": "starting",
             "pid": None, "engine_pid": None, "supervised": config.get("supervised_restarts")}
    _write_state(home, state)
    watcher = subprocess.Popen(
        [sys.executable, "-m", "veystrix.tool.supervisor", "_watch", "--home", str(home)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    state["pid"] = watcher.pid
    _WATCHERS[watcher.pid] = watcher
    _write_state(home, state)

    api_key = LocalKeyStore(home).load()
    deadline = time.monotonic() + options.timeout
    while time.monotonic() < deadline:
        if watcher.poll() is not None:
            st = _read_state(home) or {}
            reason = st.get("status") or f"exit {watcher.returncode}"
            detail = _tail(log, 12).strip()
            msg = f"veystrix: engine failed to start ({reason}); see {log}"
            if detail:
                last_lines = [ln.strip() for ln in detail.splitlines() if ln.strip()][-3:]
                if last_lines:
                    msg += "\n  " + "\n  ".join(last_lines)
            raise SystemExit(msg)
        report = health(port, timeout=0.5, api_key=api_key)
        if report is not None:
            if report.get("status") == "fault" and not report.get("ready", False):
                fault = report.get("fault") or "model load failed"
                raise SystemExit(f"veystrix: {fault}; see {log}")
            if report.get("ready", True):
                state = _read_state(home) or state
                state["status"] = "running"
                _write_state(home, state)
                result = status(home)
                if callable(options.on_ready):
                    options.on_ready(result)
                return result
        time.sleep(0.05)
    st = _read_state(home) or {}
    probe_err = ""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
    try:
        conn.request("GET", "/health", headers={"Authorization": "Bearer " + api_key} if api_key else {})
        resp = conn.getresponse()
        probe_err = f"http={resp.status}"
    except Exception as exc:
        probe_err = f"probe={exc!r}"
    finally:
        conn.close()
    detail = _tail(log, 12).strip()
    msg = (f"veystrix: engine did not become healthy within {options.timeout:g}s "
           f"(watcher_poll={watcher.poll()}, status={st.get('status')}, "
           f"engine_pid={st.get('engine_pid')}, {probe_err}); see {log}")
    if detail:
        last_lines = [ln.strip() for ln in detail.splitlines() if ln.strip()][-3:]
        if last_lines:
            msg += "\n  " + "\n  ".join(last_lines)
    raise SystemExit(msg)


def stop(home: Path | None = None, timeout: float = 15.0) -> dict:
    home = home or home_dir()
    state = _read_state(home)
    if state is None:
        return {"status": "not running"}
    stop_marker(home).write_text("stop\n")
    watcher_pid = state.get("pid")
    engine_pid = state.get("engine_pid")
    if isinstance(watcher_pid, int) and watcher_pid > 0:
        try:
            os.kill(watcher_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif isinstance(engine_pid, int) and engine_pid > 0:
        try:
            os.kill(engine_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = False
        for pid in (watcher_pid, engine_pid):
            if isinstance(pid, int) and pid > 0:
                try:
                    os.kill(pid, 0)
                    alive = True
                except ProcessLookupError:
                    pass
        if not alive:
            break
        time.sleep(0.1)
    if isinstance(engine_pid, int) and engine_pid > 0:
        try:
            os.kill(engine_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    watcher = _WATCHERS.pop(watcher_pid, None)
    if watcher is not None:
        try:
            watcher.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    state["status"] = "stopped"
    state["engine_pid"] = None
    state["pid"] = None
    _write_state(home, state)
    state_path(home).unlink(missing_ok=True)
    stop_marker(home).unlink(missing_ok=True)
    return {"status": "stopped"}


def status(home: Path | None = None) -> dict:
    home = home or home_dir()
    state = _read_state(home) or {}
    port = state.get("port") or Config(home).get("port")
    report = health(port, api_key=LocalKeyStore(home).load())
    state_engine = state.get("engine") or "python"
    if report is None:
        state.setdefault("status", "stopped" if not state else state.get("status"))
    elif report.get("ready", True):
        state["status"] = "running"
    else:
        state["status"] = "unhealthy"
    state.update({"port": port, "engine": report_engine(report, state_engine),
                  "configured_engine": state_engine, "model": report.get("model") if report else state.get("model")})
    runtime = report.get("runtime") if report else None
    if isinstance(runtime, dict) and runtime:
        kv = runtime.get("kv") if isinstance(runtime.get("kv"), dict) else {}
        state["runtime"] = runtime.get("runtime")
        if kv:
            state["kv"] = kv.get("mode")
            state["kv_plan"] = kv.get("plan")
        for key in ("kv_bytes_per_token", "context_in_use", "context_ceiling"):
            if runtime.get(key) is not None:
                state[key] = runtime[key]
        stats = runtime.get("stats") if isinstance(runtime.get("stats"), dict) else {}
        if stats:
            state["engine_stats"] = stats
    return state


def logs(home: Path | None = None, count: int = 200) -> str:
    return _tail(log_path(home or home_dir()), count)


def restart(options: StartOptions) -> dict:
    """Switching engines or models is a restart, never a hot swap."""
    home = options.home or home_dir()
    stop(home)
    return start(options)


def watch(home: Path) -> int:
    """Detached supervision loop. Invoked as ``python -m veystrix.tool.supervisor _watch``."""
    stopping = False

    def on_signal(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    state = _read_state(home)
    if state is None:
        return 1
    log = log_path(home)
    log.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    handle = log.open("a", buffering=1)
    supervised = bool(state.get("supervised", True))
    command = [str(part) for part in state.get("command", [])]
    port = int(state.get("port", 8080))
    key = LocalKeyStore(home).load()
    env = dict(os.environ)
    for name in list(env):
        if any(secret in name.upper() for secret in ("TOKEN", "SECRET", "API_KEY", "PASSWORD")):
            env.pop(name)
    env["PYTHONUNBUFFERED"] = "1"
    env["VEYSTRIX_PORT"] = str(port)
    # Engine-specific selection (currently VEYSTRIX_ENGINE=cpp) is stored in
    # state at start(); it is applied after secret stripping because it is a
    # selector, never a credential. The default python path adds nothing.
    for name, value in (state.get("engine_env") or {}).items():
        env[str(name)] = str(value)
    if state.get("pin_model"):
        env["VEYSTRIX_MODEL_PIN"] = "1"
    apply_cpp_worker_env(env)
    if key:
        env["VEYSTRIX_API_KEY"] = key
        if state.get("host", "127.0.0.1") != "127.0.0.1":
            env["VEYSTRIX_SERVER_API_KEY"] = key     # the router refuses a LAN listener without one
    try:
        while not stopping and not stop_marker(home).exists():
            state = _read_state(home) or state
            if state.get("status") in ("guard_stopped", "failed"):
                break
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                                       stderr=subprocess.STDOUT, cwd=str(Path(__file__).resolve().parents[2]),
                                       env=env)
            state["engine_pid"] = process.pid
            state["status"] = "starting"
            _write_state(home, state)
            ever_ready = False
            while process.poll() is None:
                if stopping or stop_marker(home).exists():
                    _terminate(process)
                    state["status"] = "stopped"
                    state["engine_pid"] = None
                    _write_state(home, state)
                    return 0
                report = health(port, timeout=0.5, api_key=key)
                if report and report.get("memory_policy", {}).get("stop_reason"):
                    _terminate(process)
                    state["status"] = "guard_stopped"
                    state["engine_pid"] = None
                    _write_state(home, state)
                    return 0
                if ever_ready and report and report.get("status") == "fault" and not report.get("ready", True):
                    _terminate(process)
                    tail = _tail(log, 80)
                    state["status"] = "guard_stopped" if guard_stop(tail) else "failed"
                    state["engine_pid"] = None
                    _write_state(home, state)
                    return 0 if state["status"] == "guard_stopped" else 1
                if report and report.get("ready", True) and state.get("status") == "starting":
                    ever_ready = True
                    state["status"] = "running"
                    _write_state(home, state)
                time.sleep(0.25)
            tail = _tail(log, 80)
            action = next_action(tail, int(process.returncode or 0), int(state.get("restarts", 0)),
                                 supervised and (ever_ready or "VEYSTRIX_TEST_BACKOFF" in os.environ))
            if action != "restart":
                state["status"] = action              # guard_stopped is terminal: no auto-restart
                state["engine_pid"] = None
                _write_state(home, state)
                return 0 if action == "guard_stopped" else 1
            state["restarts"] = int(state.get("restarts", 0)) + 1
            _write_state(home, state)
            refusal = battery_refusal(power_state(), Config(home).get("minimum_battery_percent"))
            if refusal:
                state["status"] = "stopped"
                state["engine_pid"] = None
                _write_state(home, state)
                return 0
            backoff = float(os.environ.get("VEYSTRIX_TEST_BACKOFF", "1"))
            time.sleep(max(0.0, min(2 ** (state["restarts"] - 1), 8)) * backoff)
    finally:
        handle.close()
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_watch":
        import argparse
        parser = argparse.ArgumentParser(prog="veystrix supervisor")
        parser.add_argument("--home", type=Path, default=home_dir())
        args = parser.parse_args(argv[1:])
        return watch(args.home)
    from veystrix.tool.cli import dispatch_tool
    return dispatch_tool(argv)


if __name__ == "__main__":
    raise SystemExit(main())
