"""Headless CLI harness drivers: Claude Code, Qwen Code, Kimi Code, Pi and Hermes.

Each harness is run once per turn in its documented non-interactive JSON mode and the next
turn resumes the same harness session, so the harness keeps its own agent loop, tools and
history while Veystrix owns the screen (same contract as the OpenCode/Codex adapters).

Stream shapes were captured from the installed CLIs on 2026-11-22 against a loopback fake
endpoint (one tool call, then a text answer) and are replayed in tests::

    claude 2.1.330   claude -p --output-format stream-json --verbose --include-partial-messages
    qwen   0.27.0    qwen --output-format stream-json --include-partial-messages   (Claude shape)
    kimi   2.5.3     kimi --prompt=… --output-format stream-json                    (OpenAI messages)
    pi     1.5.2     pi -p --mode json                                              (pi agent events)
    hermes 0.23.1    hermes -p veystrix chat --query=… --format stream-json -Q      (hermes events)

Resume: claude/qwen ``--resume <id>``, kimi ``--session <id>``, pi ``--session <id>``,
hermes ``--resume <id>`` (all verified to continue the same session id).

Approvals: these print modes cannot stop mid-turn to ask, so each runs with its documented
auto-approve mode inside the working directory — the same default OpenCode uses (its tools
run without prompting unless the user configures a permission rule).
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
from pathlib import Path

from veystrix.tool.harness_adapters import Adapter, Event, ev

STDERR_TAIL = 4000


# ---- translators (pure) ------------------------------------------------------
def _block_text(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for part in content or []:
        if isinstance(part, dict):
            if part.get("type") == "text":
                out.append(str(part.get("text", "")))
        elif isinstance(part, str):
            out.append(part)
    return "".join(out)


def _end(tool_id: str, ok: bool, text: str) -> list[Event]:
    out = [ev("tool.output", id=tool_id, text=text)] if text else []
    end = {"id": tool_id, "ok": bool(ok)}
    if text and ("\n@@" in "\n" + text or text.startswith(("--- ", "diff --git"))):
        end["diff"] = text
    out.append(Event("tool.end", end))
    return out


def translate_claude_stream(msg: dict, state: dict) -> list[Event]:
    """Claude Code / Qwen Code ``stream-json`` (with partial messages) -> events."""
    kind = msg.get("type")
    if msg.get("session_id"):
        state["session"] = msg["session_id"]
    if kind == "stream_event":
        event = msg.get("event") or {}
        etype = event.get("type")
        if etype == "message_start":
            state["streamed_text"] = False
            state["streamed_thinking"] = False
        elif etype == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                state["streamed_text"] = True
                return [ev("message.delta", text=delta["text"])]
            if delta.get("type") == "thinking_delta" and delta.get("thinking"):
                state["streamed_thinking"] = True
                return [ev("reasoning.delta", text=delta["thinking"])]
        return []
    if kind == "assistant":
        out: list[Event] = []
        seen = state.setdefault("tools_started", set())
        for block in (msg.get("message") or {}).get("content") or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and not state.get("streamed_text") and block.get("text"):
                out.append(ev("message.delta", text=block["text"]))
            elif btype == "thinking" and not state.get("streamed_thinking") and block.get("thinking"):
                out.append(ev("reasoning.delta", text=block["thinking"]))
            elif btype == "tool_use" and block.get("id") not in seen:
                seen.add(block.get("id"))
                args = block.get("input") if isinstance(block.get("input"), dict) else {}
                out.append(ev("tool.start", id=str(block.get("id", "")), name=str(block.get("name", "tool")),
                              args=args))
        return out
    if kind == "user":
        out = []
        content = (msg.get("message") or {}).get("content")
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                out += _end(str(block.get("tool_use_id", "")), not block.get("is_error"),
                            _block_text(block.get("content")))
        return out
    if kind == "result":
        out = []
        usage = msg.get("usage") or {}
        if isinstance(usage, dict) and usage:
            out.append(ev("usage", input=int(usage.get("input_tokens") or 0),
                          output=int(usage.get("output_tokens") or 0),
                          reasoning=int((usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0),
                          cached=int(usage.get("cache_read_input_tokens") or 0)))
        if msg.get("is_error"):
            out.append(ev("error", message=str(msg.get("result") or msg.get("subtype") or "harness error")))
        state["finished"] = True
        return out
    return []


def translate_kimi(msg: dict, state: dict) -> list[Event]:
    """Kimi Code ``--output-format stream-json``: one OpenAI-style message per line."""
    role = msg.get("role")
    if role == "meta":
        if msg.get("session_id"):
            state["session"] = msg["session_id"]
        return []
    if role == "assistant":
        out: list[Event] = []
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("think", "thinking", "reasoning"):
                    text = part.get("think") or part.get("thinking") or part.get("text") or ""
                    if text:
                        out.append(ev("reasoning.delta", text=str(text)))
                elif part.get("type") == "text" and part.get("text"):
                    out.append(ev("message.delta", text=str(part["text"])))
        elif content:
            out.append(ev("message.delta", text=str(content)))
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {"arguments": fn.get("arguments")}
            out.append(ev("tool.start", id=str(call.get("id", "")), name=str(fn.get("name", "tool")),
                          args=args if isinstance(args, dict) else {"value": args}))
        return out
    if role == "tool":
        text = _block_text(msg.get("content"))
        return _end(str(msg.get("tool_call_id", "")), not msg.get("is_error"), text)
    return []


def translate_pi(msg: dict, state: dict) -> list[Event]:
    """pi ``--mode json`` agent events -> events."""
    kind = msg.get("type")
    if kind == "session" and msg.get("id"):
        state["session"] = msg["id"]
        return []
    if kind == "message_start":
        state["streamed_text"] = False
        state["streamed_thinking"] = False
        return []
    if kind == "message_update":
        event = msg.get("assistantMessageEvent") or {}
        etype = event.get("type")
        if etype == "text_delta" and event.get("delta"):
            state["streamed_text"] = True
            return [ev("message.delta", text=str(event["delta"]))]
        if etype == "thinking_delta" and event.get("delta"):
            state["streamed_thinking"] = True
            return [ev("reasoning.delta", text=str(event["delta"]))]
        if etype == "error":
            err = event.get("error") or event.get("message") or event.get("errorMessage") or "pi error"
            if isinstance(err, dict):
                err = err.get("errorMessage") or err.get("message") or str(err)
            return [ev("error", message=str(err))]
        return []
    if kind == "tool_execution_start":
        args = msg.get("args") if isinstance(msg.get("args"), dict) else {}
        return [ev("tool.start", id=str(msg.get("toolCallId", "")), name=str(msg.get("toolName", "tool")),
                   args=args)]
    if kind == "tool_execution_end":
        result = msg.get("result") or {}
        return _end(str(msg.get("toolCallId", "")), not msg.get("isError"), _block_text(result.get("content")))
    if kind == "message_end":
        message = msg.get("message") or {}
        if message.get("role") != "assistant":
            return []
        out: list[Event] = []
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "thinking" and not state.get("streamed_thinking") and block.get("thinking"):
                    out.append(ev("reasoning.delta", text=str(block["thinking"])))
                elif btype == "text" and not state.get("streamed_text") and block.get("text"):
                    out.append(ev("message.delta", text=str(block["text"])))
        usage = message.get("usage") or {}
        if isinstance(usage, dict) and (usage.get("totalTokens") or usage.get("input") or usage.get("output")):
            out.append(ev("usage", input=int(usage.get("input") or 0), output=int(usage.get("output") or 0),
                          reasoning=int(usage.get("reasoning") or 0), cached=int(usage.get("cacheRead") or 0)))
        if message.get("stopReason") == "error" or message.get("errorMessage"):
            out.append(ev("error", message=str(message.get("errorMessage") or "pi model error")))
        return out
    if kind == "error":
        return [ev("error", message=str(msg.get("message") or msg.get("error") or "pi error"))]
    if kind == "agent_end":
        state["finished"] = True
    return []


def translate_hermes(msg: dict, state: dict) -> list[Event]:
    """Hermes ``chat --format stream-json`` -> events."""
    kind = msg.get("type")
    if msg.get("session_id"):
        state["session"] = msg["session_id"]
    if kind == "text" and msg.get("text"):
        text = str(msg["text"])
        if not state.get("any_text"):
            text = text.lstrip("\n")
        state["any_text"] = True
        return [ev("message.delta", text=text)] if text else []
    if kind in ("reasoning", "thinking") and (msg.get("text") or msg.get("reasoning")):
        return [ev("reasoning.delta", text=str(msg.get("text") or msg.get("reasoning")))]
    if kind == "tool_use":
        args = msg.get("input") if isinstance(msg.get("input"), dict) else {}
        return [ev("tool.start", id=str(msg.get("tool_call_id", "")), name=str(msg.get("name", "tool")), args=args)]
    if kind == "tool_result":
        output = msg.get("output")
        text = output if isinstance(output, str) else json.dumps(output)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "output" in parsed:
                text = str(parsed.get("output") or parsed.get("error") or "")
        except ValueError:
            pass
        return _end(str(msg.get("tool_call_id", "")), not msg.get("is_error"), text)
    if kind == "result":
        tokens = msg.get("tokens") or {}
        out = [ev("usage", input=int(tokens.get("input") or 0), output=int(tokens.get("output") or 0),
                  cached=int(tokens.get("cache_read") or 0))] if tokens else []
        if int(msg.get("exit_code") or 0) != 0:
            out.append(ev("error", message=str(msg.get("error") or msg.get("text") or "hermes failed")))
        state["finished"] = True
        return out
    if kind == "error":
        return [ev("error", message=str(msg.get("message") or msg.get("error") or "hermes error"))]
    return []


# ---- per-harness launch specs ------------------------------------------------
def _safe(prompt: str) -> str:
    # A prompt that starts with "-" would be parsed as an option by every CLI here.
    return " " + prompt if prompt.startswith("-") else prompt


def _claude_argv(binary: str, prompt: str, session: str | None, model: str | None) -> list[str]:
    argv = [binary, "-p", _safe(prompt), "--output-format", "stream-json", "--verbose",
            "--include-partial-messages", "--permission-mode", "bypassPermissions"]
    if model:
        argv += ["--model", model]
    if session:
        argv += ["--resume", session]
    return argv


def _qwen_argv(binary: str, prompt: str, session: str | None, model: str | None) -> list[str]:
    argv = [binary, _safe(prompt), "--output-format", "stream-json", "--include-partial-messages",
            "--approval-mode", "yolo"]
    if model:
        argv += ["--model", model]
    if session:
        argv += ["--resume", session]
    return argv


def _kimi_argv(binary: str, prompt: str, session: str | None, model: str | None) -> list[str]:
    # `--auto` cannot be combined with --prompt; print mode already runs tools without asking.
    argv = [binary, f"--prompt={prompt}", "--output-format", "stream-json"]
    if session:
        argv += ["--session", session]
    return argv


def _pi_argv(binary: str, prompt: str, session: str | None, model: str | None) -> list[str]:
    argv = [binary, "-p", "--mode", "json"]
    if model:
        argv += ["--model", f"veystrix/{model}"]
    if session:
        argv += ["--session", session]
    return argv + [_safe(prompt)]


def _hermes_argv(binary: str, prompt: str, session: str | None, model: str | None) -> list[str]:
    argv = [binary, "-p", "veystrix", "chat", f"--query={prompt}", "--format", "stream-json", "-Q", "--yolo"]
    if session:
        argv += ["--resume", session]
    return argv


SPECS = {
    "claude-code": {"binary": "claude", "argv": _claude_argv, "translate": translate_claude_stream,
                    "env": {}},
    "qwen-code": {"binary": "qwen", "argv": _qwen_argv, "translate": translate_claude_stream,
                  "env": {"QWEN_CODE_SUPPRESS_YOLO_WARNING": "1"}},
    "kimi": {"binary": "kimi", "argv": _kimi_argv, "translate": translate_kimi, "env": {}},
    "pi": {"binary": "pi", "argv": _pi_argv, "translate": translate_pi, "env": {}},
    "hermes": {"binary": "hermes", "argv": _hermes_argv, "translate": translate_hermes, "env": {}},
}


def available(harness: str) -> bool:
    spec = SPECS.get(harness)
    return spec is not None and shutil.which(spec["binary"]) is not None


class HeadlessCLIAdapter(Adapter):
    """Runs one harness process per turn in its JSON print mode and resumes its session."""

    structured = True

    def __init__(self, harness: str, *, env: dict | None = None, cwd: str | None = None,
                 model: str | None = None, binary: str | None = None,
                 log_path: str | os.PathLike | None = None):
        super().__init__()
        if harness not in SPECS:
            raise ValueError(f"no headless driver for {harness!r}")
        self.name = harness
        self.spec = SPECS[harness]
        self.binary = binary or self.spec["binary"]
        self.env = {**self.spec["env"], **(env or {})}
        self.cwd = cwd
        self.model = model or self.env.get("ANTHROPIC_MODEL") or self.env.get("OPENAI_MODEL") or None
        self.log_path = Path(log_path) if log_path else None
        self.session_id: str | None = None
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._cancelled = False

    def start(self) -> None:
        if shutil.which(self.binary) is None and not os.path.isabs(self.binary):
            raise FileNotFoundError(f"{self.binary} not found")

    def set_model(self, model: str) -> None:
        if not model:
            return
        self.model = model
        if self.name == "claude-code":
            for key in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
                        "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                        "CLAUDE_CODE_SUBAGENT_MODEL"):
                if key in self.env:
                    self.env[key] = model
        if "OPENAI_MODEL" in self.env:
            self.env["OPENAI_MODEL"] = model

    def reset(self) -> None:
        self.cancel()
        self.session_id = None

    def resume(self, session_id: str) -> None:
        self.session_id = session_id

    def list_sessions(self) -> list[dict]:
        return [{"id": self.session_id, "title": self.session_id}] if self.session_id else []

    def argv(self, prompt: str) -> list[str]:
        return self.spec["argv"](self.binary, prompt, self.session_id, self.model)

    def send(self, text: str) -> None:
        with self._lock:
            busy = self.proc is not None and self.proc.poll() is None
        if busy:
            self._emit([ev("error", message="a turn is still running; wait or press Ctrl-C"),
                        ev("turn.end", status="error")])
            return
        self._cancelled = False
        threading.Thread(target=self._turn, args=(text,), daemon=True, name=f"veystrix-{self.name}").start()

    def _log(self, text: str) -> None:
        if self.log_path is None or not text:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8", errors="replace") as fh:
                fh.write(text if text.endswith("\n") else text + "\n")
        except OSError:
            pass

    def _turn(self, text: str) -> None:
        state: dict = {"session": self.session_id}
        stderr_chunks: list[str] = []
        try:
            proc = subprocess.Popen(self.argv(text), cwd=self.cwd, env={**os.environ, **self.env},
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, bufsize=1, start_new_session=True)
        except OSError as exc:
            self._emit([ev("error", message=f"{self.binary} could not start: {exc}"), ev("turn.end", status="error")])
            return
        with self._lock:
            self.proc = proc

        def drain() -> None:
            for line in proc.stderr:
                stderr_chunks.append(line)
                if sum(map(len, stderr_chunks)) > STDERR_TAIL * 4:
                    del stderr_chunks[: len(stderr_chunks) // 2]
        err_thread = threading.Thread(target=drain, daemon=True)
        err_thread.start()
        errors = 0
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self._log(line)
                continue
            if not isinstance(msg, dict):
                continue
            events = self.spec["translate"](msg, state)
            errors += sum(1 for e in events if e.kind == "error")
            self._emit(events)
        code = proc.wait()
        err_thread.join(timeout=2)
        with self._lock:
            self.proc = None
        if state.get("session"):
            self.session_id = state["session"]
        stderr = "".join(stderr_chunks)
        self._log(stderr[-STDERR_TAIL:])
        if self._cancelled:
            self._emit([ev("turn.end", status="cancelled")])
        elif code != 0 and not state.get("finished"):
            tail = [l for l in stderr.strip().splitlines() if l.strip()][-2:]
            detail = " ".join(tail)[:300] or f"exit code {code}"
            self._emit([ev("error", message=f"{self.name} stopped: {detail}"), ev("turn.end", status="error")])
        else:
            self._emit([ev("turn.end", status="error" if errors else "ok")])

    def cancel(self) -> None:
        with self._lock:
            proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        self._cancelled = True
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except (OSError, ProcessLookupError):
            pass

        def reap() -> None:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
        threading.Thread(target=reap, daemon=True).start()

    def reply_permission(self, request_id: str, choice: str) -> None:
        pass

    def close(self) -> None:
        self.cancel()
