"""Local model router: one process owns the active model and stable API port.

Compatible requests can share the resident worker. Cross-session switches need
explicit handoff. Model loading and inference are never retried automatically.
"""
from __future__ import annotations

import argparse
import errno
import http.client
import json
import os
import re
import signal
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from veystrix.quick_bind import QuickBindMixin
from urllib.parse import urlsplit

from veystrix import context_budget, model_identity, model_registry
from veystrix.model_coordinator import ModelCoordinator, LifecycleError, ModelNotFound, ModelBusy, HandoffRequired, PRELOAD_OWNER
from veystrix.model_process import ProcessDriver
from veystrix.server import VeystrixHandler, RequestError
from veystrix.sessions import derive_key, explicit_id

HEARTBEAT_SECONDS = 15.0
PORT_IN_USE_EXIT = 98
INPUT_CACHE_BYTES = 4 * 1024 * 1024
INPUT_CACHE_SECONDS = 120


def _same_json(left, right):
    """Preserve JSON types, key order, and the sign of zero when reusing wire bytes."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return len(left) == len(right) and all(
            lk == rk and _same_json(lv, rv) for (lk, lv), (rk, rv) in zip(left.items(), right.items())
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right))
    if isinstance(left, float):
        return repr(left) == repr(right)
    return left == right


class _InputCache:
    """Keep one recent authenticated session's model-independent JSON encoding."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last = None
        self._cross_model_hits = 0

    def encode(self, path, scope, payload):
        if scope is None or "model" not in payload or len(payload) == 1:
            return json.dumps(payload).encode()
        model = payload["model"]
        fields = {key: value for key, value in payload.items() if key != "model"}
        key = (path, scope)
        now = time.monotonic()
        with self._lock:
            entry = self._last
            if entry is not None and now >= entry[0]:
                self._last = entry = None
            if entry is not None and key == entry[1]:
                try:
                    hit = _same_json(fields, entry[2])
                except RecursionError:
                    hit = False
                if hit:
                    if model != entry[4]:
                        self._cross_model_hits += 1
                    encoded = entry[3]
                    self._last = (now + INPUT_CACHE_SECONDS, key, fields, encoded, model)
                else:
                    encoded = None
            else:
                encoded = None
        if encoded is None:
            encoded = json.dumps(fields).encode()
            if len(encoded) <= INPUT_CACHE_BYTES:
                with self._lock:
                    self._last = (now + INPUT_CACHE_SECONDS, key, fields, encoded, model)
        return b"".join((b'{"model": ', json.dumps(model).encode(), b", ", memoryview(encoded)[1:]))

    def snapshot(self):
        with self._lock:
            return {"cross_model_hits": self._cross_model_hits}


MODEL_HANDOFF_ENV = "VEYSTRIX_MODEL_HANDOFF"
MODEL_PIN_ENV = "VEYSTRIX_MODEL_PIN"


def auto_handoff() -> bool:
    """``auto``: a request for the other installed model takes over the engine once the current request has finished.

    The coordinator otherwise refuses a switch unless the caller is the session that owns the resident model, to protect a
    long-running coding session from being evicted. A client with one conversation per bot (Rakazo) has a different
    session identity for every bot, so it could never move between the Qwen and Nemotron bots. A request is never
    interrupted: the switch waits for the running one. Default ``off`` keeps the strict rule; the app sets ``auto``."""
    return os.environ.get(MODEL_HANDOFF_ENV, "off").strip().lower() == "auto"


def session_owner(headers, payload):
    # A dual-model workflow keeps lifecycle ownership across its agents while
    # each agent still supplies a distinct session ID to the inference server.
    workflow = headers.get("X-Veystrix-Workflow") if headers is not None else None
    if workflow and re.fullmatch(r"[0-9a-f]{32}", workflow):
        return "workflow:" + workflow
    hint = explicit_id(headers, payload)
    if hint:
        return "explicit:" + hint
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
        return None
    if not any(m.get("role") == "user" for m in messages):
        return None
    try:
        return derive_key(messages, tools=payload.get("tools") or [])
    except (TypeError, AttributeError):
        return None


_EFFORT_CACHE: dict = {"mtime": None, "value": None}


def configured_effort() -> str | None:
    """The user's Veystrix effort setting, re-read only when config.json changes. None = auto."""
    try:
        from veystrix.tool.config import Config, home_dir
        config = Config(home_dir()) if _EFFORT_CACHE["mtime"] is None else None
        path = config.path if config else _EFFORT_CACHE["path"]
        mtime = path.stat().st_mtime_ns if path.exists() else 0
        if mtime != _EFFORT_CACHE["mtime"]:
            _EFFORT_CACHE.update(path=path, mtime=mtime, value=(config or Config(home_dir())).get("effort"))
    except Exception:
        return None
    value = _EFFORT_CACHE["value"]
    return None if value in (None, "auto") else value


def apply_effort(payload: dict, effort: str | None) -> None:
    """Replace every reasoning control in ``payload`` with ``effort`` (``off`` = thinking off).
    One setting then means the same thing for every harness, whatever wire form it speaks."""
    if not effort or not isinstance(payload, dict):
        return
    for key in ("reasoning", "reasoning_effort", "enable_thinking"):
        payload.pop(key, None)
    kwargs = payload.get("chat_template_kwargs")
    if isinstance(kwargs, dict):
        kwargs.pop("enable_thinking", None)
        if not kwargs:
            payload.pop("chat_template_kwargs")
    payload["reasoning_effort"] = "none" if effort == "off" else effort


class RouterServer(QuickBindMixin, ThreadingHTTPServer):
    daemon_threads = True
    # Permit immediate restart after recently closed connections. The OS still
    # rejects binding over another active listener (no SO_REUSEPORT).
    allow_reuse_address = True
    def __init__(self, addr, coordinator, driver, *, api_key=None, inventory=model_registry.discover):
        super().__init__(addr, RouterHandler)
        self.coordinator = coordinator
        self.driver = driver
        self.api_key = api_key
        self.allowed_origins = ()
        self.inventory = inventory
        self.input_cache = _InputCache()


class RouterHandler(VeystrixHandler):
    protocol_version = "HTTP/1.1"
    def _access(self):
        if self._reject_foreign_origin():
            return False
        if not self._authorized():
            self._error("invalid API key", 401, "invalid_api_key")
            return False
        return True

    def _catalog(self):
        models = self.server.inventory()
        active = self.server.coordinator.snapshot().model_id
        driver = self.server.driver
        # The loaded backend's admitted window beats the configured estimate, but only while
        # its process is alive and its /health still names this exact model.
        try:
            health = driver.health() if getattr(driver, "process", None) is not None else None
        except Exception:
            health = None                                          # a catalog never fails on telemetry
        entries = []
        # The loaded model first: many clients take data[0] as the default.
        models = sorted(models, key=lambda m: m.id != active)
        for m in models:
            window = model_registry.catalog_context(m)
            if isinstance(health, dict) and health.get("installed_model_id") == m.id:
                limit = health.get("context_limit")
                if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
                    window = limit
            entries.append({"id": m.id, "object": "model", "created": 0, "owned_by": "veystrix",
                            "name": m.display_name, "family": m.family,
                            "loaded": m.id == active, "verification": m.verification,
                            # What this engine will accept for the model, not the checkpoint's maximum.
                            "context_window": window,
                            "suggested_client_context_window": context_budget.suggested_client_window(window),
                            "max_output_tokens": max(1, window - context_budget.MIN_OUTPUT_RESERVE)})
        return entries

    def do_GET(self):
        if not self._access(): return
        path = urlsplit(self.path).path
        if path == "/v1/models":
            self._json({"object": "list", "data": self._catalog()})
        elif path == "/v1/installed-models":
            active = self.server.coordinator.snapshot().model_id
            self._json({"object": "list", "data": [dict(m.as_dict(), loaded=m.id == active,
                                                        context_window=model_registry.catalog_context(m))
                                                    for m in self.server.inventory()]})
        elif path.startswith("/v1/models/"):
            entry = next((m for m in self._catalog() if m["id"] == path[len("/v1/models/"):]), None)
            if entry: self._json(entry)
            else: self._error("model not found", 404, "model_not_found")
        elif path == "/veystrix/status":
            snap = self.server.coordinator.snapshot()
            data = self.server.driver.get_json("/veystrix/status") if snap.state == "ready" else None
            self._json(data or {"model": snap.model_id, "engine": None, "router_state": snap.state})
        elif path in ("/health", "/ready"):
            snap = self.server.coordinator.snapshot()
            health = self.server.driver.health() if snap.state == "ready" else None
            if health and health.get("installed_model_id") == snap.model_id:
                health["router_state"] = snap.state
                health["busy"] = snap.active
                health["input_reuse"] = self.server.input_cache.snapshot()
                self._json(health, 200 if path == "/health" or health.get("ready") else 503)
            else:
                self._json({"status": snap.state, "ready": False, "model": snap.model_id,
                            "busy": snap.active, "fault": snap.error}, 200 if path == "/health" else 503)
        else:
            self._error("route not found", 404, "not_found")

    def do_POST(self):
        if not self._access(): return
        path = urlsplit(self.path).path
        if path not in ("/v1/chat/completions", "/v1/responses", "/v1/messages"):
            self._error("route not found", 404, "not_found"); return
        self._sse_open = False
        heartbeat_stop = None
        try:
            payload = self._payload()
            requested = payload["model"] if "model" in payload else self.server.coordinator.snapshot().model_id
            if not isinstance(requested, str) or not requested:
                raise ModelNotFound("choose an installed model")
            # Family aliases remain accepted only when the family has exactly one installation.
            legacy = {f"veystrix-{fam}": fam for fam in model_identity.IDENTITIES}
            if requested in legacy:
                matches = [m for m in self.server.inventory() if m.family == legacy[requested]]
                if len(matches) != 1:
                    raise ModelNotFound("model family is absent or ambiguous; select an installation ID")
                if requested == "veystrix-flashnext-nothink":
                    # Preserve the existing no-thinking shortcut explicitly.
                    payload["reasoning_effort"] = "none"
                requested = matches[0].id
                payload["model"] = requested
            apply_effort(payload, configured_effort())
            if self.server.coordinator.resolve(requested) is None:
                served = sorted(m.id for m in self.server.inventory())
                raise ModelNotFound(f"model {requested!r} is not installed; this endpoint serves "
                                    + (", ".join(served) if served else "no installed model")
                                    + " (see GET /v1/models)")
            if payload.get("stream") is True:
                self._sse_start()
                self.close_connection = True
                self._response_started = True
                heartbeat_stop = threading.Event()
                self._router_heartbeat_stop = heartbeat_stop
                def heartbeat():
                    while not heartbeat_stop.wait(HEARTBEAT_SECONDS):
                        try: self._heartbeat()
                        except (OSError, ValueError): return
                self._router_heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
                self._router_heartbeat_thread.start()
            owner = session_owner(self.headers, payload)
            with self.server.coordinator.request(requested, owner=owner, timeout=20, cancelled=self._client_gone,
                                                 handoff=auto_handoff(),
                                                 exclusive_owner=bool(owner and owner.startswith("workflow:"))):
                self._forward(path, payload)
        except RequestError as exc:
            self._error(exc, exc.status, exc.code)
        except LifecycleError as exc:
            status = 404 if isinstance(exc, ModelNotFound) else 409 if isinstance(exc, HandoffRequired) else 503
            self._router_error(path, exc, status, exc.code)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (OSError, http.client.HTTPException) as exc:
            if not getattr(self, "_response_started", False):
                self._router_error(path, f"engine connection failed: {exc}", 503, "engine_unavailable")
            elif self._sse_open:
                self._router_error(path, f"engine connection failed: {exc}", 503, "engine_unavailable")
            else:
                self.close_connection = True
        finally:
            if heartbeat_stop is not None:
                heartbeat_stop.set()

    def _router_error(self, path, message, status, code):
        if not self._sse_open:
            self._error(message, status, code, headers={"x-should-retry": "false"})
            return
        stop = getattr(self, "_router_heartbeat_stop", None)
        if stop is not None: stop.set()
        error = {"message": str(message), "type": "server_error", "code": code}
        if path == "/v1/messages":
            self._sse({"type": "error", "error": error}, event="error")
            self._sse({"type": "message_stop"}, event="message_stop")
        elif path == "/v1/responses":
            self._sse({"type": "response.failed", "response": {"status": "failed", "error": error}}, event="response.failed")
        else:
            self._sse({"error": error})
            with self._write_lock():
                self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()

    def _forward(self, path, payload):
        driver = self.server.driver
        session = explicit_id(self.headers, payload)
        scope = (self.headers.get("X-Veystrix-Workflow"), session) if self.server.api_key and session else None
        body = self.server.input_cache.encode(path, scope, payload)
        conn = http.client.HTTPConnection("127.0.0.1", driver.port, timeout=7200)
        try:
            headers = {"Content-Type": "application/json", "Authorization": "Bearer " + driver.api_key}
            for name in ("X-Session-Id", "X-Veystrix-Session", "X-Cache-Id", "anthropic-version"):
                if self.headers.get(name): headers[name] = self.headers[name]
            conn.request("POST", path, body=body, headers=headers)
            response = conn.getresponse()
            if self._sse_open:
                # Stop heartbeats before relaying: a comment must never split an SSE frame.
                self._router_heartbeat_stop.set()
                self._router_heartbeat_thread.join(timeout=1)
                if response.status >= 400:
                    detail = response.read().decode("utf-8", "replace")
                    self._router_error(path, detail, response.status, "engine_error")
                    return
            else:
                self.send_response(response.status)
                self._response_started = True
                for name in ("Content-Type", "Content-Length", "Retry-After", "x-should-retry", "X-Veystrix-Reasoning"):
                    value = response.getheader(name)
                    if value is not None: self.send_header(name, value)
                self.send_header("Connection", "close")
                self.end_headers()
            self.close_connection = True
            while True:
                chunk = response.read1(65536)
                if not chunk: break
                self.wfile.write(chunk)
                self.wfile.flush()
        finally:
            conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local Veystrix model router")
    # Loopback by default. 0.0.0.0 (LAN) only with an API key: the supervisor refuses it otherwise.
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--pool-gb", default="0")
    parser.add_argument("--prefix-dir")
    parser.add_argument("--flashnext-kv-cache", choices=("bf16", "tiered4"), default="bf16")
    parser.add_argument("--flashnext-speed", action="store_true", help="enable the experimental short throughput profile")
    parser.add_argument("--flashnext-exact-speed", action="store_true",
                        help="enable the curated bit-identical exact-speed flag set (24/24 quality gate)")
    parser.add_argument("--flashnext-bounded-prefill", action="store_true",
                        help="enable the experimental low-transient prefill profile for 16 GB hosts")
    parser.add_argument("--flashnext-memory-policy", choices=("absolute-swap-only",))
    parser.add_argument("--overnight", action="store_true", help="restart after a worker failure within bounded supervisor limits")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=model_identity.MODEL_CHOICES, default="flashnext")
    parser.add_argument("--pin-model", action="store_true",
                        help=f"serve only the --model installation: a request naming another installed model gets 409 "
                             f"instead of a model swap, unless {MODEL_HANDOFF_ENV}=auto (also {MODEL_PIN_ENV}=1)")
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1" and not os.environ.get("VEYSTRIX_SERVER_API_KEY"):
        parser.error("--host 0.0.0.0 requires an API key (VEYSTRIX_SERVER_API_KEY); refusing to listen on the LAN without one")
    if args.flashnext_speed and args.flashnext_bounded_prefill:
        parser.error("--flashnext-speed and --flashnext-bounded-prefill plan the prefill differently; choose one")
    initial_id = model_registry.installation_id(args.model, args.checkpoint)
    # Do not start the listener with an installation that discovery cannot validate.
    if model_registry.resolve(initial_id) is None:
        parser.error("selected model was not found in the configured model root or is incomplete")
    driver = ProcessDriver(pool_gb=args.pool_gb, prefix_dir=args.prefix_dir,
                           flashnext_kv_cache=args.flashnext_kv_cache,
                           flashnext_memory_policy=args.flashnext_memory_policy, flashnext_speed=args.flashnext_speed,
                           flashnext_exact_speed=args.flashnext_exact_speed,
                           flashnext_bounded_prefill=args.flashnext_bounded_prefill)
    coordinator = ModelCoordinator(model_registry.resolve, driver,
                                   pinned=args.pin_model or os.environ.get(MODEL_PIN_ENV) == "1")
    try:
        server = RouterServer((args.host, args.port), coordinator, driver, api_key=os.environ.get("VEYSTRIX_SERVER_API_KEY"))
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        print(f"Cannot start Veystrix: {args.host}:{args.port} is already in use. "
              "Stop the listener or choose another port.", file=sys.stderr, flush=True)
        return PORT_IN_USE_EXIT
    stopping = threading.Event()
    def shutdown(_signum, _frame):
        stopping.set()
        threading.Thread(target=server.shutdown, daemon=True).start()
    previous = {sig: signal.signal(sig, shutdown) for sig in (signal.SIGINT, signal.SIGTERM)}
    def startup():
        try:
            with coordinator.request(initial_id, owner=PRELOAD_OWNER, cancelled=stopping.is_set):
                pass
        except LifecycleError as exc:
            print(f"initial model load failed: {exc}", flush=True)
            if not stopping.is_set():
                coordinator._set(state="fault", error=str(exc))
                threading.Thread(target=server.shutdown, daemon=True).start()
    thread = threading.Thread(target=startup, daemon=True)
    thread.start()
    def supervise():
        from models.flashnext.overnight import Policy, swap_used_mb
        policy = Policy()
        while not stopping.wait(2):
            snap = coordinator.snapshot()
            process = driver.process
            if snap.state != "ready" or snap.active or process is None:
                continue
            code = process.poll()
            if code is None or not coordinator._gate.acquire(blocking=False):
                continue
            try:
                snap = coordinator.snapshot()
                if snap.state != "ready" or snap.active or driver.process is not process:
                    continue
                action, cooldown, reason = policy.decide(code, time.time(), owner_stop=False)
                try:
                    driver.unload()
                except Exception as exc:
                    coordinator._set(state="fault", error=f"restart cleanup failed: {exc}")
                    continue
                if action != "restart":
                    coordinator._set(state="fault", model=None, owner=None, error=reason)
                    continue
                coordinator._set(state="cooldown", model=None, owner=None, error=reason)
            finally:
                coordinator._gate.release()
            print(f"engine exited ({code}); {reason}", flush=True)
            if stopping.wait(cooldown): return
            # Match the original supervisor's swap-settle requirement; never purge or change limits.
            try:
                previous_swap = swap_used_mb()
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline and not stopping.wait(15):
                    current_swap = swap_used_mb()
                    if abs(current_swap - previous_swap) < 64: break
                    previous_swap = current_swap
            except Exception as exc:
                coordinator._set(state="fault", error=f"could not verify swap settlement: {exc}")
                continue
            if stopping.is_set(): return
            if not coordinator._gate.acquire(timeout=1):
                coordinator._set(state="fault", error="restart could not acquire model gate")
                continue
            try:
                if coordinator.snapshot().state != "cooldown": continue
                coordinator._set(state="unloaded", error=None)
            finally:
                coordinator._gate.release()
            try:
                with coordinator.request(snap.model_id, owner=snap.owner, cancelled=stopping.is_set):
                    pass
            except LifecycleError as exc:
                print(f"supervised restart failed: {exc}", flush=True)
    if args.overnight:
        threading.Thread(target=supervise, daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        stopping.set()
        server.server_close()
        thread.join(timeout=5)
        if coordinator._gate.acquire(timeout=90):
            try:
                driver.unload()
            finally:
                coordinator._gate.release()
        for sig, handler in previous.items(): signal.signal(sig, handler)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
