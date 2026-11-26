"""Headless harness adapters (D6): drive an agentic harness through its documented
structured interface and normalise everything into ONE Veystrix event stream, so the
Veystrix renderer owns the whole screen and no harness UI/branding is ever shown.

Normalised events (``Event.kind``)::

    message.delta     {text}
    reasoning.delta   {text}
    tool.start        {id, name, args}
    tool.output       {id, text}
    tool.end          {id, ok, text?, diff?}
    permission.request{id, action, detail, choices}
    turn.end          {status: ok | cancelled | error}
    usage             {input, output, reasoning, cached, window?}
    error             {message}

Adapters (all stdlib only, localhost only, no runtime network beyond the loopback harness):

  * ``OpenCodeAdapter``  ``opencode serve`` (HTTP) + SSE event stream. Bound to 127.0.0.1,
    protected with a per-run random password (``OPENCODE_SERVER_PASSWORD``, HTTP Basic,
    user ``opencode``). Stream path is ``/api/event`` on v2 servers and ``/event`` on v1.
  * ``CodexAdapter``     ``codex app-server`` JSON-RPC (newline-delimited JSON) over stdio.
    Notifications are streamed; approval *server requests* are answered by the client
    (the UI), never auto-approved.
  * ``PtyPassthrough``   generic fallback for harnesses with no documented structured mode.

Pinned harness versions (the protocol shapes below were verified against these builds;
a different version is allowed but reported by ``version_warning``)::

    PINNED_VERSIONS = {"opencode": "2.3.2", "codex": "0.166.0"}

License / rebranding caveat
---------------------------
Harnesses run as separate, unmodified processes that the user installed; Veystrix ships no
harness source and does not patch or repackage their binaries. Hiding a harness' interface
(because we do not render it) is NOT the same as removing its attribution: the harness
licences (opencode MIT, codex Apache-2.0) still require their notices to be kept wherever
the harness itself is redistributed, and Veystrix lists them in NOTICE/LICENSES. This layer
only replaces the *screen*. For ``PtyPassthrough`` the harness draws the screen itself, so
its branding CANNOT be fully replaced there: Veystrix only owns one status line and filters
what it can; that limitation is deliberate and documented rather than hidden.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import secrets
import select
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

PINNED_VERSIONS = {"opencode": "2.3.2", "codex": "0.166.0"}
PROTOCOL_CHECKED = "2026-11-19"
EVENT_KINDS = ("message.delta", "reasoning.delta", "tool.start", "tool.output", "tool.end",
               "permission.request", "turn.end", "usage", "error")
PERMISSION_CHOICES = ("once", "always", "reject")
LOOPBACK = "127.0.0.1"


def _loopback_opener() -> urllib.request.OpenerDirector:
    """Loopback HTTP opener that never routes 127.0.0.1 through system proxies."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def friendly_url_error(exc: Exception, *, target: str = "engine", port: int | None = None,
                       log_hint: str = "") -> str:
    """Turn raw urllib/socket errors into actionable messages for the TUI."""
    reason = getattr(exc, "reason", exc)
    text = str(reason or exc)
    refused = isinstance(reason, ConnectionRefusedError) or "connection refused" in text.lower() or "errno 61" in text.lower()
    if target == "engine":
        where = f" on 127.0.0.1:{int(port)}" if port else ""
        if refused:
            return f"engine is not running{where} — run /start to load a model"
        return f"could not reach engine{where}: {text}"
    if refused:
        return f"harness server is not reachable{log_hint} — switch harness in /settings or restart Veystrix"
    return f"harness connection error{log_hint}: {text}"


@dataclass(frozen=True)
class Event:
    kind: str
    data: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, "data": self.data}, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "Event":
        raw = json.loads(line)
        return cls(raw["kind"], raw.get("data", {}))


def ev(kind: str, **data) -> Event:
    if kind not in EVENT_KINDS:
        raise ValueError(f"unknown event kind {kind!r}")
    return Event(kind, data)


def version_warning(harness: str, found: str | None) -> str | None:
    pinned = PINNED_VERSIONS.get(harness)
    if pinned and found and pinned not in found:
        return f"{harness} {found} differs from pinned {pinned}; structured events may differ"
    return None


# ---- SSE ------------------------------------------------------------------
def parse_sse(lines: Iterable) -> Iterator[dict]:
    """Parse Server-Sent-Events lines into JSON payloads. Comments/heartbeats are skipped;
    a payload that is not JSON is yielded as ``{"type": "raw", "data": text}``."""
    buf: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw
        line = line.rstrip("\r\n")
        if not line:
            if buf:
                text = "\n".join(buf)
                buf = []
                try:
                    obj = json.loads(text)
                    yield obj if isinstance(obj, dict) else {"type": "raw", "data": obj}
                except ValueError:
                    yield {"type": "raw", "data": text}
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            buf.append(line[5:].lstrip(" ") if line[5:6] == " " else line[5:])
    if buf:
        try:
            obj = json.loads("\n".join(buf))
            yield obj if isinstance(obj, dict) else {"type": "raw", "data": obj}
        except ValueError:
            pass


# ---- translators (pure; replayed offline from recorded streams) -----------
def _content_text(content) -> str:
    out = []
    for part in content or []:
        if isinstance(part, dict) and part.get("type") == "text":
            out.append(str(part.get("text", "")))
    return "".join(out)


def _looks_like_diff(text: str) -> bool:
    return "\n@@" in "\n" + text or text.startswith(("--- ", "diff --git"))


def translate_opencode(msg: dict, state: dict | None = None) -> list[Event]:
    """opencode v2 SSE envelope ``{"id","type","data"}`` -> normalised events.

    ``state`` carries ``session`` (events from other sessions are dropped), ``tools`` (id ->
    name/args). Unknown or housekeeping events (server.connected, installation.*,
    tui.toast.show, heartbeats ...) are dropped: they are harness surface, not Veystrix output.
    """
    state = state if state is not None else {}
    kind = msg.get("type", "")
    data = msg.get("data") or {}
    sid = data.get("sessionID")
    if state.get("session") and sid and sid != state["session"]:
        return []
    tools = state.setdefault("tools", {})
    if kind == "session.text.delta":
        return [ev("message.delta", text=data.get("delta", ""))]
    if kind == "session.reasoning.delta":
        return [ev("reasoning.delta", text=data.get("delta", ""))]
    if kind == "session.tool.input.started":
        tools[data["id"]] = {"name": data.get("name", "tool"), "args": {}}
        return []
    if kind == "session.tool.called":
        info = tools.setdefault(data["id"], {"name": "tool", "args": {}})
        info["args"] = data.get("input") or {}
        return [ev("tool.start", id=data["id"], name=info["name"], args=info["args"])]
    if kind == "session.tool.progress":
        meta = data.get("metadata") or {}
        text = meta.get("output") or meta.get("text") or ""
        return [ev("tool.output", id=data["id"], text=str(text))] if text else []
    if kind in ("session.tool.success", "session.tool.failed"):
        ok = kind.endswith("success")
        text = _content_text(data.get("content"))
        if not ok and not text:
            text = str((data.get("error") or {}).get("message", "tool failed"))
        out = [ev("tool.output", id=data["id"], text=text)] if text else []
        end = {"id": data["id"], "ok": ok}
        if _looks_like_diff(text):
            end["diff"] = text
        out.append(Event("tool.end", end))
        return out
    if kind in ("permission.asked",):
        resources = data.get("resources") or []
        return [ev("permission.request", id=data.get("id", ""), action=data.get("action", ""),
                   detail=data.get("message") or ", ".join(map(str, resources)),
                   choices=list(PERMISSION_CHOICES))]
    if kind in ("session.usage.updated", "session.step.ended", "session.usage.recorded"):
        tokens = data.get("tokens")
        if isinstance(tokens, dict):
            return [ev("usage", input=int(tokens.get("input", 0)), output=int(tokens.get("output", 0)),
                       reasoning=int(tokens.get("reasoning", 0)),
                       cached=int((tokens.get("cache") or {}).get("read", 0)))]
        return []
    if kind == "session.execution.succeeded":
        return [ev("turn.end", status="ok")]
    if kind == "session.execution.interrupted":
        return [ev("turn.end", status="cancelled")]
    if kind in ("session.execution.failed", "session.step.failed"):
        err = data.get("error") or {}
        return [ev("error", message=str(err.get("message", "model error"))),
                ev("turn.end", status="error")]
    return []


def _codex_file_diff(changes) -> str:
    parts = []
    for change in changes or []:
        parts.append(f"--- a/{change.get('path', '')}\n+++ b/{change.get('path', '')}\n{change.get('diff', '')}")
    return "\n".join(parts)


def translate_codex(msg: dict, state: dict | None = None) -> list[Event]:
    """codex app-server notification/request -> normalised events (schema-checked 0.166.0)."""
    state = state if state is not None else {}
    method = msg.get("method", "")
    params = msg.get("params") or {}
    if method == "item/agentMessage/delta":
        return [ev("message.delta", text=params.get("delta", ""))]
    if method in ("item/reasoning/textDelta", "item/reasoning/summaryTextDelta"):
        return [ev("reasoning.delta", text=params.get("delta", ""))]
    if method == "item/started":
        item = params.get("item") or {}
        if item.get("type") == "commandExecution":
            return [ev("tool.start", id=item["id"], name="shell", args={"command": item.get("command", "")})]
        if item.get("type") == "fileChange":
            paths = [c.get("path", "") for c in item.get("changes") or []]
            return [ev("tool.start", id=item["id"], name="apply_patch", args={"files": ", ".join(paths)})]
        if item.get("type") in ("mcpToolCall", "dynamicToolCall"):
            return [ev("tool.start", id=item["id"], name=item.get("tool", "tool"),
                       args=item.get("arguments") if isinstance(item.get("arguments"), dict) else {})]
        return []
    if method == "item/commandExecution/outputDelta":
        return [ev("tool.output", id=params.get("itemId", ""), text=params.get("delta", ""))]
    if method == "item/completed":
        item = params.get("item") or {}
        ok = item.get("status") in ("completed", None)
        if item.get("type") == "commandExecution":
            return [Event("tool.end", {"id": item["id"], "ok": ok and item.get("exitCode") in (0, None)})]
        if item.get("type") == "fileChange":
            diff = _codex_file_diff(item.get("changes"))
            return [Event("tool.end", {"id": item["id"], "ok": ok, "diff": diff})]
        if item.get("type") in ("mcpToolCall", "dynamicToolCall"):
            return [Event("tool.end", {"id": item["id"], "ok": ok})]
        return []
    if method == "thread/tokenUsage/updated":
        usage = params.get("tokenUsage") or {}
        total = usage.get("total") or {}
        return [ev("usage", input=int(total.get("inputTokens", 0)), output=int(total.get("outputTokens", 0)),
                   reasoning=int(total.get("reasoningOutputTokens", 0)),
                   cached=int(total.get("cachedInputTokens", 0)),
                   window=int(usage.get("modelContextWindow") or 0))]
    if method == "turn/completed":
        turn = params.get("turn") or {}
        status = {"completed": "ok", "interrupted": "cancelled", "failed": "error"}.get(turn.get("status"), "ok")
        out = []
        if status == "error":
            out.append(ev("error", message=str((turn.get("error") or {}).get("message", "turn failed"))))
        out.append(ev("turn.end", status=status))
        return out
    if method == "error":
        return [ev("error", message=str((params.get("error") or {}).get("message", params.get("message", "error"))))]
    if method == "item/commandExecution/requestApproval":
        return [ev("permission.request", id=str(msg.get("id")), action="run command",
                   detail=str(params.get("command") or params.get("reason") or ""),
                   choices=list(PERMISSION_CHOICES))]
    if method == "item/fileChange/requestApproval":
        return [ev("permission.request", id=str(msg.get("id")), action="change files",
                   detail=str(params.get("reason") or params.get("grantRoot") or ""),
                   choices=list(PERMISSION_CHOICES))]
    return []


# ---- adapter base ---------------------------------------------------------
class Adapter:
    """Contract used by the UI loop. ``poll`` never blocks longer than ``timeout``."""

    name = "adapter"
    structured = True

    def start(self) -> None: ...
    def send(self, text: str) -> None: ...
    def cancel(self) -> None: ...
    def reply_permission(self, request_id: str, choice: str) -> None: ...
    def list_sessions(self) -> list[dict]: return []
    def resume(self, session_id: str) -> None: ...
    def reset(self) -> None: pass
    def close(self) -> None: ...
    session_id: str | None = None

    def __init__(self) -> None:
        self._events: "queue.Queue[Event]" = queue.Queue()

    def _emit(self, events: Iterable[Event]) -> None:
        for event in events:
            self._events.put(event)

    def poll(self, timeout: float = 0.0) -> list[Event]:
        out: list[Event] = []
        try:
            out.append(self._events.get(timeout=timeout) if timeout > 0 else self._events.get_nowait())
            while True:
                out.append(self._events.get_nowait())
        except queue.Empty:
            pass
        return out


class ReplayAdapter(Adapter):
    """Offline fake: replays recorded event batches. Each ``send`` releases the next scripted
    turn, one batch per ``poll`` so tests can interleave redraws and keypresses."""

    name = "replay"

    def __init__(self, turns: list[list[Event]] | None = None, sessions: list[dict] | None = None,
                 batch: int = 1):
        super().__init__()
        self.turns = list(turns or [])
        self.sessions = sessions or []
        self.batch = max(1, batch)
        self.sent: list[str] = []
        self.cancelled = 0
        self.replies: list[tuple[str, str]] = []
        self.resumed: str | None = None
        self.closed = False
        self._pending: list[Event] = []
        self.session_id = "ses_replay"

    def send(self, text: str) -> None:
        self.sent.append(text)
        if self.turns:
            self._pending.extend(self.turns.pop(0))

    def cancel(self) -> None:
        self.cancelled += 1
        self._pending = [ev("turn.end", status="cancelled")]

    def reply_permission(self, request_id: str, choice: str) -> None:
        self.replies.append((request_id, choice))

    def list_sessions(self) -> list[dict]:
        return list(self.sessions)

    def resume(self, session_id: str) -> None:
        self.resumed = session_id
        self.session_id = session_id

    def poll(self, timeout: float = 0.0) -> list[Event]:
        out, self._pending = self._pending[:self.batch], self._pending[self.batch:]
        return out

    def close(self) -> None:
        self.closed = True


# ---- opencode ---------------------------------------------------------------
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOOPBACK, 0))
        return sock.getsockname()[1]


class OpenCodeAdapter(Adapter):
    """``opencode serve`` + SSE. ``base_url``/``password`` may be injected (tests point this
    at an in-process loopback fake); otherwise the server is spawned on 127.0.0.1 with a
    random per-run password and stopped on ``close``."""

    name = "opencode"
    USER = "opencode"

    def __init__(self, *, env: dict | None = None, cwd: str | None = None,
                 base_url: str | None = None, password: str | None = None,
                 binary: str = "opencode", event_paths: tuple[str, ...] = ("/api/event", "/event"),
                 request_timeout: float = 10.0, model: str | None = None, provider: str = "veystrix",
                 log_path: str | os.PathLike | None = None):
        super().__init__()
        self.log_path = Path(log_path) if log_path else None   # server output; DEVNULL when None
        self.env, self.cwd, self.binary = env or {}, cwd, binary
        # Pin every session to the local Veystrix model; otherwise OpenCode uses its own default,
        # which may be another installation (refused by a pinned engine) or a cloud model.
        self.model, self.provider = model, provider
        self.base_url, self.password = base_url, password
        self.event_paths = event_paths
        self.request_timeout = request_timeout
        self.proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: dict = {}
        self.connected = threading.Event()

    # -- plumbing
    def _auth(self) -> dict:
        token = base64.b64encode(f"{self.USER}:{self.password}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    def _check_loopback(self) -> None:
        host = urllib.parse.urlparse(self.base_url).hostname
        if host not in (LOOPBACK, "localhost", "::1"):
            raise ValueError(f"refusing non-loopback harness endpoint {self.base_url!r}")

    def _request(self, method: str, path: str, body: dict | None = None):
        """JSON request; OpenCode 2.x wraps replies as ``{"data": ...}``, which is unwrapped.
        An HTTP error carries the server's own message instead of a bare status line."""
        data = json.dumps(body).encode() if body is not None else None
        headers = {**self._auth(), "Content-Type": "application/json"}
        req = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
        try:
            with _loopback_opener().open(req, timeout=self.request_timeout) as resp:  # noqa: S310 (loopback only)
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                info = json.loads(exc.read() or b"{}")
                detail = str(info.get("message") or info.get("error") or "") if isinstance(info, dict) else ""
                field = info.get("field") if isinstance(info, dict) else None
                detail += f" ({field})" if field and detail else ""
            except (OSError, ValueError):
                pass
            raise RuntimeError(f"harness request {method} {path.split('?')[0]} failed: "
                               f"HTTP {exc.code}" + (f" — {detail}" if detail else "")) from None
        except (urllib.error.URLError, OSError) as exc:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError("harness server exited unexpectedly" + self._log_hint()) from None
            raise RuntimeError(friendly_url_error(exc, target="harness", log_hint=self._log_hint())) from None
        value = json.loads(raw) if raw else {}
        if isinstance(value, dict) and set(value) == {"data"}:
            value = value["data"]
        return value

    def _log_hint(self) -> str:
        if self.log_path is None:
            return ""
        try:
            tail = [l for l in self.log_path.read_text(errors="replace").splitlines() if l.strip()][-1:]
        except OSError:
            tail = []
        return (f": {tail[0][:160]}" if tail else "") + f" (see {self.log_path})"

    def model_ref(self, model: str | None = None) -> dict | None:
        model = model or self.model
        return {"providerID": self.provider, "id": model} if model else None

    def command(self, port: int) -> list[str]:
        return [self.binary, "serve", "--hostname", LOOPBACK, "--port", str(port)]

    def start(self) -> None:
        if self.base_url is None:
            if shutil.which(self.binary) is None:
                raise FileNotFoundError(f"{self.binary} not found")
            port = _free_port()
            self.password = self.password or secrets.token_urlsafe(24)
            env = {**os.environ, **self.env, "OPENCODE_SERVER_PASSWORD": self.password}
            sink = subprocess.DEVNULL
            if self.log_path is not None:
                try:
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                    sink = open(self.log_path, "ab")
                except OSError:
                    sink = subprocess.DEVNULL
            try:
                self.proc = subprocess.Popen(self.command(port), env=env, cwd=self.cwd,
                                             stdout=sink, stderr=subprocess.STDOUT)
            finally:
                if sink is not subprocess.DEVNULL:
                    sink.close()                     # the child keeps its own descriptor
            self.base_url = f"http://{LOOPBACK}:{port}"
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                try:
                    self._request("GET", "/api/info")
                    break
                except (urllib.error.URLError, OSError, RuntimeError):
                    if self.proc.poll() is not None:
                        raise RuntimeError("harness server exited during startup" + self._log_hint())
                    time.sleep(0.1)
            else:
                raise TimeoutError("harness server did not start within 20s" + self._log_hint())
        self.password = self.password or ""
        self._check_loopback()
        if self.session_id is None:
            body = {"model": self.model_ref()} if self.model else {}
            created = self._request("POST", "/api/session", body)
            self.session_id = created.get("id") if isinstance(created, dict) else None
            if not self.session_id:
                raise RuntimeError("harness did not return a session id")
        elif self.model:
            self.set_model(self.model)
        self._state["session"] = self.session_id
        self._thread = threading.Thread(target=self._pump, daemon=True, name="veystrix-sse")
        self._thread.start()
        self.connected.wait(timeout=5)

    def _pump(self) -> None:
        # Bound retries per path so a dropped stream is actually reconnected
        # (Sonnet finding 5) without spinning forever.
        max_retries = 3
        for path in self.event_paths:
            for attempt in range(max_retries + 1):
                if self._stop.is_set():
                    return
                req = urllib.request.Request(
                    self.base_url + path,
                    headers={**self._auth(), "Accept": "text/event-stream"})
                # A local cold prefill emits nothing for minutes: a 10 s read
                # timeout would silently end the stream and leave the UI `busy`
                # forever (Sonnet finding 22). Use a generous idle timeout.
                idle_timeout = max(self.request_timeout, 900)
                try:
                    resp = urllib.request.urlopen(req, timeout=idle_timeout)  # noqa: S310
                except (urllib.error.URLError, OSError):
                    if self._stop.wait(timeout=1.0):
                        return
                    continue
                if "text/event-stream" not in (resp.headers.get("Content-Type") or ""):
                    resp.close()
                    break
                self.connected.set()
                closed_unexpectedly = False
                try:
                    for payload in parse_sse(resp):
                        if self._stop.is_set():
                            break
                        self._emit(translate_opencode(payload, self._state))
                    else:
                        closed_unexpectedly = not self._stop.is_set()
                except (OSError, ValueError):
                    closed_unexpectedly = not self._stop.is_set()
                finally:
                    resp.close()
                if not closed_unexpectedly or self._stop.is_set():
                    return
                if attempt >= max_retries:
                    break  # this path keeps dropping: try the next path
                self._emit([ev("error", message="harness event stream closed; reconnecting"),
                            ev("turn.end")])
                if self._stop.wait(timeout=1.0):
                    return
        self.connected.set()
        self._emit([ev("error", message="harness event stream unavailable")])

    # -- commands
    def send(self, text: str) -> None:
        self._request("POST", f"/api/session/{self.session_id}/prompt", {"text": text})

    def cancel(self) -> None:
        try:
            self._request("POST", f"/api/session/{self.session_id}/interrupt")
        except (urllib.error.URLError, OSError, RuntimeError):
            pass

    def reply_permission(self, request_id: str, choice: str) -> None:
        decision = choice if choice in PERMISSION_CHOICES else "reject"
        self._request("POST", f"/api/session/{self.session_id}/permission/{request_id}/reply",
                      {"decision": decision})

    def list_sessions(self) -> list[dict]:
        try:
            data = self._request("GET", "/api/session?limit=20&order=desc")
        except (urllib.error.URLError, OSError, ValueError, RuntimeError):
            return []
        items = data.get("items", data.get("data", data)) if isinstance(data, dict) else data
        return [{"id": s.get("id"), "title": s.get("title") or s.get("id")}
                for s in items if isinstance(s, dict)]

    def resume(self, session_id: str) -> None:
        self.session_id = session_id
        self._state["session"] = session_id
        if self.model:
            try:
                self.set_model(self.model)
            except (RuntimeError, urllib.error.URLError, OSError):
                pass                       # an older session may refuse; the prompt still goes out

    def reset(self) -> None:
        """Start a fresh OpenCode session."""
        if not self.base_url:
            return
        body = {"model": self.model_ref()} if self.model else {}
        created = self._request("POST", "/api/session", body)
        new_id = created.get("id") if isinstance(created, dict) else None
        if new_id:
            self.session_id = new_id
            self._state["session"] = new_id
            self._state.pop("tools", None)

    def set_model(self, model: str) -> None:
        """Point the current session at another local model (after /model + restart)."""
        self.model = model
        if self.session_id and self.base_url:
            self._request("POST", f"/api/session/{self.session_id}/model", {"model": self.model_ref(model)})

    def close(self) -> None:
        self._stop.set()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# ---- codex ------------------------------------------------------------------
class CodexAdapter(Adapter):
    """``codex app-server`` over stdio (newline-delimited JSON-RPC). ``popen`` may be injected
    so offline tests can drive a fake process; approvals surface as ``permission.request``
    and are answered only by ``reply_permission``."""

    name = "codex"
    DECISION = {"once": "accept", "always": "acceptForSession", "reject": "decline"}

    def __init__(self, *, env: dict | None = None, cwd: str | None = None, binary: str = "codex",
                 command: list[str] | None = None, client_version: str = "0"):
        super().__init__()
        self.env, self.cwd = env or {}, cwd
        self._command = command or [binary, "app-server", "--listen", "stdio://"]
        self.client_version = client_version
        self.proc: subprocess.Popen | None = None
        self._next_id = 0
        self._lock = threading.Lock()
        self._waiters: dict[int, "queue.Queue[dict]"] = {}
        self._approval_ids: dict[str, object] = {}
        self._thread: threading.Thread | None = None
        self._turn_id: str | None = None
        self._state: dict = {}

    def _write(self, obj: dict) -> None:
        with self._lock:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()

    def _call(self, method: str, params: dict | None = None, timeout: float = 20.0) -> dict:
        with self._lock:
            self._next_id += 1
            rid = self._next_id
        waiter: "queue.Queue[dict]" = queue.Queue()
        self._waiters[rid] = waiter
        self._write({"id": rid, "method": method, "params": params or {}})
        try:
            reply = waiter.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"codex {method} timed out") from None
        finally:
            self._waiters.pop(rid, None)
        if "error" in reply:
            raise RuntimeError(str(reply["error"].get("message", reply["error"])))
        return reply.get("result") or {}

    def start(self) -> None:
        if self.proc is None:
            if shutil.which(self._command[0]) is None and not os.path.isabs(self._command[0]):
                raise FileNotFoundError(f"{self._command[0]} not found")
            self.proc = subprocess.Popen(self._command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                         env={**os.environ, **self.env}, cwd=self.cwd)
        self._thread = threading.Thread(target=self._pump, daemon=True, name="veystrix-codex")
        self._thread.start()
        self._call("initialize", {"clientInfo": {"name": "veystrix", "title": "Veystrix",
                                                  "version": self.client_version}})
        self._write({"method": "initialized"})
        if self.session_id is None:
            result = self._call("thread/start", {"cwd": self.cwd} if self.cwd else {})
            self.session_id = (result.get("thread") or {}).get("id")
        else:
            self._call("thread/resume", {"threadId": self.session_id})

    def _pump(self) -> None:
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            self._dispatch(msg)
        self._emit([ev("error", message="harness exited"), ev("turn.end", status="error")])

    def _dispatch(self, msg: dict) -> None:
        if "method" not in msg:  # response to one of our calls
            waiter = self._waiters.get(msg.get("id"))
            if waiter is not None:
                waiter.put(msg)
            return
        if "id" in msg:  # server request: only approvals are surfaced; others are rejected
            if msg["method"] in ("item/commandExecution/requestApproval", "item/fileChange/requestApproval"):
                self._approval_ids[str(msg["id"])] = msg["id"]
                self._emit(translate_codex(msg, self._state))
            else:
                self._write({"id": msg["id"], "error": {"code": -32601, "message": "unsupported by client"}})
            return
        if msg["method"] == "turn/started":
            self._turn_id = ((msg.get("params") or {}).get("turn") or {}).get("id")
        self._emit(translate_codex(msg, self._state))

    def send(self, text: str) -> None:
        threading.Thread(target=self._turn, args=(text,), daemon=True).start()

    def _turn(self, text: str) -> None:
        try:
            result = self._call("turn/start", {"threadId": self.session_id,
                                               "input": [{"type": "text", "text": text}]})
            self._turn_id = (result.get("turn") or {}).get("id") or self._turn_id
        except (RuntimeError, TimeoutError, OSError) as exc:
            self._emit([ev("error", message=str(exc)), ev("turn.end", status="error")])

    def cancel(self) -> None:
        if not (self.session_id and self._turn_id):
            return

        def interrupt() -> None:
            try:
                self._call("turn/interrupt", {"threadId": self.session_id, "turnId": self._turn_id}, timeout=5)
            except (RuntimeError, TimeoutError, OSError):
                pass
        threading.Thread(target=interrupt, daemon=True).start()

    def reply_permission(self, request_id: str, choice: str) -> None:
        rid = self._approval_ids.pop(str(request_id), None)
        if rid is None:
            return
        self._write({"id": rid, "result": {"decision": self.DECISION.get(choice, "decline")}})

    def list_sessions(self) -> list[dict]:
        try:
            data = self._call("thread/list", {"limit": 20}, timeout=5)
        except (RuntimeError, TimeoutError, OSError):
            return []
        return [{"id": t.get("id"), "title": t.get("name") or t.get("preview") or t.get("id")}
                for t in data.get("data", [])]

    def resume(self, session_id: str) -> None:
        self.session_id = session_id
        self._call("thread/resume", {"threadId": session_id})

    def reset(self) -> None:
        """Start a fresh Codex thread."""
        result = self._call("thread/start", {"cwd": self.cwd} if self.cwd else {})
        new_id = (result.get("thread") or {}).get("id")
        if new_id:
            self.session_id = new_id
            self._turn_id = None

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# ---- generic fallback -------------------------------------------------------
class PtyPassthrough:
    """Fallback for harnesses without a documented structured mode: run the harness in a PTY and
    copy bytes through, reserving the LAST terminal row for a Veystrix status line.

    Limitation (by design, documented): the harness still draws the screen, so its own
    branding/banner cannot be fully replaced here. Use a structured adapter when one exists.
    """

    structured = False
    name = "pty"

    def __init__(self, command: list[str], env: dict | None = None, status: Callable[[], str] | None = None):
        self.command, self.env, self.status = command, env or {}, status or (lambda: "VEYSTRIX")

    @staticmethod
    def reserve_status_row(rows: int) -> str:
        """Set the scroll region to all rows but the last and park the cursor in it."""
        return f"\033[1;{max(rows - 1, 1)}r\033[{max(rows - 1, 1)};1H"

    def paint_status(self, cols: int, rows: int) -> str:
        text = self.status()[:cols].ljust(cols)
        return f"\0337\033[{rows};1H\033[7m{text}\033[0m\0338"

    def run(self, in_fd: int, out_fd: int, size: tuple[int, int] = (80, 24)) -> int:
        import pty
        import struct
        import fcntl
        import termios
        cols, rows = size
        pid, master = pty.fork()
        if pid == 0:  # child
            os.environ.update(self.env)
            os.execvp(self.command[0], self.command)
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", max(rows - 1, 1), cols, 0, 0))
        os.write(out_fd, self.reserve_status_row(rows).encode())
        os.write(out_fd, self.paint_status(cols, rows).encode())
        stdin_open = True
        try:
            while True:
                watch = [master, in_fd] if stdin_open else [master]
                ready, _, _ = select.select(watch, [], [], 0.25)
                if master in ready:
                    try:
                        data = os.read(master, 65536)
                    except OSError:
                        break
                    if not data:
                        break
                    os.write(out_fd, data + self.paint_status(cols, rows).encode())
                if stdin_open and in_fd in ready:
                    data = os.read(in_fd, 4096)
                    if data:
                        os.write(master, data)
                    else:
                        stdin_open = False
        finally:
            os.write(out_fd, f"\033[r\033[{rows};1H\033[K".encode())
            try:
                _, status = os.waitpid(pid, 0)
            except ChildProcessError:
                status = 0
        return os.waitstatus_to_exitcode(status)


# ---- built-in direct local API adapter -------------------------------------
class DirectAdapter(Adapter):
    """Built-in adapter that talks directly to the local Veystrix OpenAI-compatible
    server (127.0.0.1:<port>/v1/chat/completions) via streaming SSE. Used when no
    external structured harness (OpenCode/Codex) is installed so the full Veystrix
    TUI (logo, /model, /models, /effort, /kv, /settings) always opens."""

    name = "direct"
    structured = True

    def __init__(self, *, port: int = 8080, key: str = "", model: str | None = None,
                 request_timeout: float = 900.0):
        super().__init__()
        self.port = int(port)
        self.key = key or "local"
        self.model = model or ""
        self.request_timeout = request_timeout
        self._messages: list[dict] = []
        self._cancel = threading.Event()
        self._resp = None
        self._lock = threading.Lock()
        self.session_id = "local"

    def start(self) -> None:
        pass

    def set_model(self, model: str) -> None:
        self.model = model or ""

    def reset(self) -> None:
        self.cancel()
        self._messages.clear()

    def send(self, text: str) -> None:
        self._cancel.clear()
        self._messages.append({"role": "user", "content": text})
        threading.Thread(target=self._run_turn, daemon=True, name="veystrix-direct").start()

    def _resolve_model(self) -> str:
        if self.model:
            return self.model
        try:
            url = f"http://{LOOPBACK}:{self.port}/health"
            headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
            req = urllib.request.Request(url, headers=headers)
            with _loopback_opener().open(req, timeout=2.0) as resp:  # noqa: S310 (loopback)
                data = json.loads(resp.read().decode("utf-8", "replace"))
            if isinstance(data, dict):
                chosen = data.get("installed_model_id") or data.get("model") or ""
                if chosen:
                    self.model = str(chosen)
                    return self.model
        except Exception:
            pass
        return "veystrix"

    def _run_turn(self) -> None:
        url = f"http://{LOOPBACK}:{self.port}/v1/chat/completions"
        payload = {
            "model": self._resolve_model(),
            "messages": list(self._messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST", headers=headers)
        reply_parts: list[str] = []
        try:
            resp = _loopback_opener().open(req, timeout=self.request_timeout)  # noqa: S310 (loopback)
            with self._lock:
                self._resp = resp
            try:
                for raw in resp:
                    if self._cancel.is_set():
                        self._emit([ev("turn.end", status="cancelled")])
                        return
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except ValueError:
                        continue
                    if isinstance(chunk, dict) and chunk.get("error"):
                        err = chunk["error"]
                        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                        self._emit([ev("error", message=msg), ev("turn.end", status="error")])
                        return
                    usage = chunk.get("usage") if isinstance(chunk, dict) else None
                    if isinstance(usage, dict):
                        self._emit([ev("usage",
                                       input=int(usage.get("prompt_tokens") or 0),
                                       output=int(usage.get("completion_tokens") or 0),
                                       cached=int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0))])
                    choices = chunk.get("choices") if isinstance(chunk, dict) else None
                    if not choices:
                        continue
                    delta = (choices[0] or {}).get("delta") or {}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if reasoning:
                        self._emit([ev("reasoning.delta", text=str(reasoning))])
                    content = delta.get("content") or ""
                    if content:
                        reply_parts.append(str(content))
                        self._emit([ev("message.delta", text=str(content))])
            finally:
                with self._lock:
                    self._resp = None
                resp.close()
            if reply_parts:
                self._messages.append({"role": "assistant", "content": "".join(reply_parts)})
            self._emit([ev("turn.end", status="cancelled" if self._cancel.is_set() else "ok")])
        except (urllib.error.URLError, OSError) as exc:
            if self._cancel.is_set():
                self._emit([ev("turn.end", status="cancelled")])
            else:
                self._emit([ev("error", message=friendly_url_error(exc, target="engine", port=self.port)),
                            ev("turn.end", status="error")])
        except Exception as exc:
            if self._cancel.is_set():
                self._emit([ev("turn.end", status="cancelled")])
            else:
                self._emit([ev("error", message=str(exc) or exc.__class__.__name__),
                            ev("turn.end", status="error")])

    def cancel(self) -> None:
        self._cancel.set()
        with self._lock:
            resp = self._resp
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    def reply_permission(self, request_id: str, choice: str) -> None:
        pass

    def close(self) -> None:
        self.cancel()


# ---- selection --------------------------------------------------------------
STRUCTURED = {"opencode": OpenCodeAdapter, "codex": CodexAdapter}
# Print-mode CLI drivers (headless.py) have no long-running local server, but
# they still speak a structured JSON stream and are driven by this same contract.
HEADLESS = ("claude-code", "qwen-code", "kimi", "pi", "hermes")


def _headless_available(harness: str) -> bool:
    from veystrix.tool.headless import available
    return available(harness)


def structured_available(harness: str) -> bool:
    if harness in HEADLESS:
        return _headless_available(harness)
    binary = {"opencode": "opencode", "codex": "codex"}.get(harness)
    return harness in STRUCTURED and binary is not None and shutil.which(binary) is not None


def make_adapter(harness: str, *, env: dict | None = None, cwd: str | None = None,
                 model: str | None = None, log_path=None) -> Adapter | None:
    """Structured adapter for ``harness`` or ``None`` when only passthrough exists."""
    if harness in HEADLESS:
        if not _headless_available(harness):
            return None
        from veystrix.tool.headless import HeadlessCLIAdapter
        return HeadlessCLIAdapter(harness, env=env, cwd=cwd, model=model, log_path=log_path)
    cls = STRUCTURED.get(harness)
    if cls is None or not structured_available(harness):
        return None
    if cls is OpenCodeAdapter:
        return cls(env=env, cwd=cwd, model=model, log_path=log_path)
    return cls(env=env, cwd=cwd)
