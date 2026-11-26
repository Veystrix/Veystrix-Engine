"""Veystrix's own Built-in agent harness.

Unlike the plain ``DirectAdapter`` (chat only), this adapter declares the local
tool schemas to the engine, walks the streamed OpenAI tool-call loop, asks the
TUI for permission before any mutating tool, and executes everything through
``Repo.call`` (path confinement + sanitised shell environment). It is the
``Built-in`` harness in the picker.

The engine's measured ``decode_tokens_per_second`` still comes from the shared
``/veystrix/status`` poller in the TUI, so tok/s looks identical here and in
every other harness.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

from veystrix.tool.harness_adapters import (
    LOOPBACK, PERMISSION_CHOICES, DirectAdapter, _loopback_opener, ev, friendly_url_error,
)
from veystrix.tools import MUTATING_TOOLS, TOOLS, render_tool_result
from veystrix.tools.repo import Repo

# read / write / bash plus the small navigation commands; no network tools.
CORE_TOOLS = ("find_files", "list_dir", "read_file", "search",
              "write_file", "replace_text", "run_command")
TOOL_SCHEMAS = [t for t in TOOLS if t["function"]["name"] in CORE_TOOLS]
PERMISSION_TIMEOUT = 300.0


class BuiltinAdapter(DirectAdapter):
    """Tool-capable Built-in harness driving the local engine over SSE."""

    name = "direct"          # same picker entry / SessionIndex key as before
    structured = True

    def __init__(self, *, port: int = 8080, key: str = "", model: str | None = None,
                 request_timeout: float = 900.0, cwd: str | None = None,
                 max_tool_rounds: int = 8):
        super().__init__(port=port, key=key, model=model, request_timeout=request_timeout)
        chosen = Path(cwd or ".").expanduser()
        if not chosen.is_dir():
            chosen = Path.cwd()
        self.cwd = str(chosen.resolve())
        self.max_tool_rounds = max(1, int(max_tool_rounds))
        self._permissions: dict[str, tuple[threading.Event, list[str]]] = {}
        self._always: set[str] = set()
        self._perm_lock = threading.Lock()

    def reset(self) -> None:
        super().reset()
        with self._perm_lock:
            self._permissions.clear()
            self._always.clear()

    def reply_permission(self, request_id: str, choice: str) -> None:
        if choice not in PERMISSION_CHOICES:
            return
        with self._perm_lock:
            pending = self._permissions.pop(str(request_id), None)
            if pending is None:
                return
            gate, answer = pending
        answer.append(choice)
        gate.set()

    def list_sessions(self) -> list[dict]:
        return [{"id": self.session_id, "title": "Built-in Veystrix"}] if self.session_id else []

    def _ask(self, name: str, args: dict) -> str:
        """Return 'once' / 'always' / 'reject' for a mutating tool."""
        with self._perm_lock:
            if name in self._always:
                return "always"
            request_id = f"builtin-{name}-{threading.get_ident()}-{id(args)}"
            gate = threading.Event()
            answer: list[str] = []
            self._permissions[request_id] = (gate, answer)
        detail = args.get("command") or args.get("path") or json.dumps(args, ensure_ascii=False)
        self._emit([ev("permission.request", id=request_id, action=name,
                       detail=str(detail)[:400], choices=list(PERMISSION_CHOICES))])
        if not gate.wait(PERMISSION_TIMEOUT):
            with self._perm_lock:
                self._permissions.pop(request_id, None)
            return "reject"
        choice = answer[0] if answer else "reject"
        if choice == "always":
            with self._perm_lock:
                self._always.add(name)
        return choice

    def send(self, text: str) -> None:
        self._cancel.clear()
        self._messages.append({"role": "user", "content": text})
        threading.Thread(target=self._run_turn, daemon=True, name="veystrix-builtin").start()

    def _post(self, messages: list[dict]):
        url = f"http://{LOOPBACK}:{self.port}/v1/chat/completions"
        payload = {"model": self._resolve_model(), "messages": messages, "stream": True,
                   "stream_options": {"include_usage": True}, "tools": TOOL_SCHEMAS}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers=headers)
        resp = _loopback_opener().open(req, timeout=self.request_timeout)  # noqa: S310 (loopback)
        with self._lock:
            self._resp = resp
        return resp

    def _stream(self, resp) -> tuple[str, str, dict, str | None, bool]:
        """Consume one SSE response -> (content, reasoning, calls, finish_reason, had_error)."""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: dict[int, dict] = {}
        finish: str | None = None
        had_error = False
        try:
            for raw in resp:
                if self._cancel.is_set():
                    self._emit([ev("turn.end", status="cancelled")])
                    return "", "", {}, finish, False
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error"):
                    err = chunk["error"]
                    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    self._emit([ev("error", message=msg)])
                    had_error = True
                    return "", "", {}, finish, True
                usage = chunk.get("usage")
                if isinstance(usage, dict) and usage:
                    self._emit([ev("usage",
                                   input=int(usage.get("prompt_tokens") or 0),
                                   output=int(usage.get("completion_tokens") or 0),
                                   cached=int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0))])
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0] or {}).get("delta") or {}
                finish = (choices[0] or {}).get("finish_reason") or finish
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if reasoning:
                    reasoning_parts.append(str(reasoning))
                    self._emit([ev("reasoning.delta", text=str(reasoning))])
                text = delta.get("content") or ""
                if text:
                    content_parts.append(str(text))
                    self._emit([ev("message.delta", text=str(text))])
                for call in delta.get("tool_calls") or []:
                    index = int(call.get("index") or 0)
                    slot = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    if call.get("id"):
                        slot["id"] = str(call["id"])
                    fn = call.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = str(fn["name"])
                    if fn.get("arguments"):
                        slot["arguments"] += str(fn["arguments"])
        finally:
            with self._lock:
                self._resp = None
            try:
                resp.close()
            except Exception:
                pass
        return "".join(content_parts), "".join(reasoning_parts), calls, finish, had_error

    def _run_turn(self) -> None:
        repo = None
        try:
            repo = Repo(self.cwd)
        except SystemExit as exc:
            self._emit([ev("error", message=str(exc)), ev("turn.end", status="error")])
            return
        messages = list(self._messages)
        for _ in range(self.max_tool_rounds):
            if self._cancel.is_set():
                self._emit([ev("turn.end", status="cancelled")])
                return
            try:
                resp = self._post(messages)
            except (urllib.error.URLError, OSError) as exc:
                self._emit([ev("error", message=friendly_url_error(exc, target="engine", port=self.port)),
                            ev("turn.end", status="error")])
                return
            except Exception as exc:
                self._emit([ev("error", message=str(exc) or exc.__class__.__name__),
                            ev("turn.end", status="error")])
                return
            content, reasoning, calls, finish, had_error = self._stream(resp)
            if self._cancel.is_set():
                return
            if had_error:
                self._emit([ev("turn.end", status="error")])
                return
            ordered = [calls[i] for i in sorted(calls) if calls[i].get("name")]
            assistant: dict = {"role": "assistant", "content": content or None}
            if reasoning:
                assistant["reasoning_content"] = reasoning
            if ordered:
                assistant["tool_calls"] = [
                    {"id": c["id"] or f"call_{i}", "type": "function",
                     "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                    for i, c in enumerate(ordered)
                ]
            messages.append(assistant)
            if not ordered or (finish not in ("tool_calls", None)):
                if content:
                    self._messages.append({"role": "assistant", "content": content})
                self._emit([ev("turn.end", status="ok")])
                return
            for i, call in enumerate(ordered):
                call_id = call["id"] or f"call_{i}"
                name = call["name"]
                try:
                    args = json.loads(call["arguments"] or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("tool arguments must be an object")
                except ValueError as exc:
                    args, result, ok = {}, f"invalid tool arguments: {exc}", False
                    self._emit([ev("tool.start", id=call_id, name=name, args=args)])
                    self._emit([ev("tool.output", id=call_id, text=result),
                                ev("tool.end", id=call_id, ok=False)])
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
                    continue
                self._emit([ev("tool.start", id=call_id, name=name, args=args)])
                if name in MUTATING_TOOLS and self._ask(name, args) == "reject":
                    result, ok = "rejected by user", False
                else:
                    try:
                        raw = repo.call(name, args, should_cancel=self._cancel.is_set)
                        result, ok = render_tool_result(name, raw), True
                    except Exception as exc:
                        result, ok = f"{exc.__class__.__name__}: {exc}", False
                self._emit([ev("tool.output", id=call_id, text=str(result)[:20000]),
                            ev("tool.end", id=call_id, ok=ok,
                               diff=str(result)[:20000] if ok and name in ("write_file", "replace_text") else None)])
                messages.append({"role": "tool", "tool_call_id": call_id, "content": str(result)})
        self._emit([ev("error", message="tool loop limit reached"),
                    ev("turn.end", status="error")])
