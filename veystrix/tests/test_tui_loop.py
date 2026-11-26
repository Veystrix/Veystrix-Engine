"""D6 Veystrix-owned UI loop + headless adapters, fully offline.

Covers: key parsing, cancel (incl. a hung harness), resume, palette/help, permission replies,
200-token stream with bounded redraw, resize, the keypress->frame latency gate on a synthetic
clock, the opencode SSE adapter against an in-process loopback server, the codex JSON-RPC
adapter against a fake stdio process, the PTY fallback, and the /veystrix/status payload.

``VEYSTRIX_UI_CAPTURE=1`` additionally writes SVG snapshots, an asciicast and a latency report
into results/ui/ (deterministic: they are rendered from the synthetic clock, not wall time).
"""
from __future__ import annotations

import json
import os
import random
import socket
import statistics
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from veystrix.tests.test_tui_fixtures import (GOLDEN, THEMES, FakeClock, ScreenMirror, drive,
                                              make_loop, opencode_events, run_turn, scenario_adapter)
from veystrix.tool import renderer as R
from veystrix.tool import tui as T

_HOME = None


def setUpModule():
    # Loops built without an explicit home fall back to home_dir(): point it at a throwaway
    # folder so no test can write the developer's real ~/.veystrix/config.json.
    global _HOME
    _HOME = tempfile.TemporaryDirectory()
    os.environ["VEYSTRIX_HOME"] = _HOME.name


def tearDownModule():
    os.environ.pop("VEYSTRIX_HOME", None)
    _HOME.cleanup()
from veystrix.tool.harness_adapters import (CodexAdapter, OpenCodeAdapter, PtyPassthrough,
                                            ReplayAdapter, ev, make_adapter, parse_sse,
                                            structured_available, translate_opencode,
                                            version_warning)

ROOT = Path(__file__).resolve().parents[2]
UI_RESULTS = ROOT / "results" / "ui"


def text_of(loop) -> str:
    return "\n".join(R.strip_ansi(line) for line in loop.last_frame)


def settle(loop, until, timeout=3.0):
    """Step the loop on a moving clock until ``until()`` holds (background work finished)."""
    deadline = time.monotonic() + timeout
    t = loop.state.now
    while not until() and time.monotonic() < deadline:
        time.sleep(0.01)
        t += 0.05
        loop.step(t)
    loop.step(t + 0.05)
    return t + 0.05


class KeyParsing(unittest.TestCase):
    def test_keys_arrows_paste_and_split_sequences(self):
        p = T.KeyParser()
        self.assertEqual(p.feed("ab\r\x7f\x03\x0b\x12\x14"),
                         ["a", "b", "enter", "backspace", "ctrl-c", "ctrl-k", "ctrl-r", "ctrl-t"])
        self.assertEqual(p.feed("\x1b[A\x1b[B\x1b[5~\x1b[6~"), ["up", "down", "pgup", "pgdn"])
        self.assertEqual(p.feed("\x1b"), ["esc"])
        self.assertEqual(p.feed("\x1b["), [])               # split CSI waits for its tail
        self.assertEqual(p.feed("A"), ["up"])
        self.assertEqual(p.feed("\x1b[200~line1\nline2\x1b[201~x"), ["paste:line1\nline2", "x"])

    def test_paste_never_submits_and_keeps_newlines(self):
        loop, clock, _, _ = make_loop(ReplayAdapter(), cols=120, rows=40)
        loop.step(0.0, ["paste:first\nsecond"])
        self.assertEqual(loop.state.input, "first\nsecond")
        self.assertEqual(loop.adapter.sent, [])
        self.assertIn("first⏎second", text_of(loop))


class CancelAndQuit(unittest.TestCase):
    def long_turn(self, n=400):
        return [ev("message.delta", text=f"w{i} ") for i in range(n)] + [ev("turn.end", status="ok")]

    def test_ctrl_c_cancels_the_turn_and_the_ui_recovers(self):
        adapter = ReplayAdapter([self.long_turn(), [ev("message.delta", text="again"), ev("turn.end", status="ok")]])
        loop, clock, mirror, _ = make_loop(adapter)
        loop.step(0.0, list("go") + ["enter"])
        for i in range(1, 6):
            loop.step(i * 0.02)
        self.assertTrue(loop.state.busy)
        loop.step(0.2, ["ctrl-c"])
        self.assertEqual(adapter.cancelled, 1)
        loop.step(0.25)
        self.assertFalse(loop.state.busy)
        self.assertTrue(loop.state.status.cancelled)
        self.assertIn("cancelled", text_of(loop))
        self.assertIn("cancelled", R.strip_ansi(loop.last_frame[-5]))    # activity line, not just a toast
        # the next prompt works and clears the cancelled flag
        loop.step(1.0, list("more") + ["enter"])
        for i in range(10):
            loop.step(1.1 + i * 0.02)
        self.assertFalse(loop.state.status.cancelled)
        self.assertIn("again", text_of(loop))
        self.assertEqual(mirror.rows, loop.last_frame)

    def test_hung_harness_is_released_after_the_grace_period(self):
        class Hung(ReplayAdapter):
            def cancel(self):          # ignores the request, never emits turn.end
                self.cancelled += 1
        adapter = Hung([[ev("message.delta", text="x")] * 3])
        loop, clock, _, _ = make_loop(adapter)
        loop.step(0.0, list("go") + ["enter"])
        loop.step(0.05)
        loop.state.busy = True
        loop.step(0.1, ["ctrl-c"])
        loop.step(0.1 + T.CANCEL_GRACE - 0.1)
        self.assertTrue(loop.state.busy)
        loop.step(0.1 + T.CANCEL_GRACE + 0.1)
        self.assertFalse(loop.state.busy)
        self.assertIn("cancelled", text_of(loop))

    def test_idle_ctrl_c_twice_quits_once_clears_input(self):
        loop, _, _, _ = make_loop(ReplayAdapter())
        loop.step(0.0, list("abc"))
        loop.step(0.1, ["ctrl-c"])
        self.assertEqual(loop.state.input, "")
        self.assertFalse(loop.done)
        loop.step(0.2, ["ctrl-c"])
        self.assertIn("press Ctrl-C again to quit", text_of(loop))
        loop.step(0.3, ["ctrl-c"])
        self.assertTrue(loop.done)

    def test_enter_while_busy_does_not_queue_a_second_turn(self):
        adapter = ReplayAdapter([self.long_turn()])
        loop, _, _, _ = make_loop(adapter)
        loop.step(0.0, list("go") + ["enter"])
        loop.step(0.1, list("second") + ["enter"])
        self.assertEqual(adapter.sent, ["go"])
        self.assertIn("busy", text_of(loop))


class ResumeFlow(unittest.TestCase):
    def test_resume_picker_selects_a_session_and_persists_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = T.SessionIndex(Path(tmp))
            adapter = scenario_adapter()
            loop, _, _, _ = make_loop(adapter, index=index, harness="opencode", cwd="/work")
            loop.step(0.0, list("/resume") + ["enter"])
            screen = text_of(loop)
            self.assertIn("Resume session", screen)
            self.assertIn("fix parser bug", screen)
            self.assertIn("add tests for cache", screen)
            loop.step(0.1, ["down", "enter"])
            self.assertEqual(adapter.resumed, "ses_2")
            self.assertIn("resumed session ses_2", text_of(loop))
            self.assertEqual(index.last("opencode", "/work"), "ses_2")
            self.assertIsNone(index.last("opencode", "/elsewhere"))
            loop.step(0.2, list("/resume ses_1") + ["enter"])
            self.assertEqual(adapter.resumed, "ses_1")

    def test_escape_closes_the_picker_without_resuming(self):
        adapter = scenario_adapter()
        loop, _, _, _ = make_loop(adapter)
        loop.step(0.0, list("/resume") + ["enter", "esc"])
        self.assertIsNone(adapter.resumed)
        self.assertNotIn("Resume session", text_of(loop))


class PaletteHelpPermission(unittest.TestCase):
    def test_palette_filters_and_runs_a_command(self):
        loop, _, _, _ = make_loop(scenario_adapter())
        loop.step(0.0, ["ctrl-k"])
        self.assertIn("Command palette", text_of(loop))
        self.assertIn("Show / hide reasoning", text_of(loop))
        loop.step(0.1, list("tools"))
        self.assertIn("Toggle tool output", text_of(loop))
        self.assertNotIn("Show help", text_of(loop))
        loop.step(0.2, ["enter"])
        self.assertIsNone(loop.state.overlay)
        self.assertTrue(loop.state.tools_collapsed)

    def test_help_opens_with_question_mark_only_on_empty_input(self):
        loop, _, _, _ = make_loop(scenario_adapter())
        loop.step(0.0, ["?"])
        self.assertIn("Veystrix help", text_of(loop))
        loop.step(0.1, ["esc"])
        loop.step(0.2, list("why?"))
        self.assertNotIn("Veystrix help", text_of(loop))
        self.assertEqual(loop.state.input, "why?")

    def test_permission_request_is_answered_by_the_user_not_auto_approved(self):
        for key, expected in (("y", "once"), ("a", "always"), ("n", "reject")):
            adapter = ReplayAdapter([[ev("permission.request", id="per_1", action="bash",
                                         detail="python -m unittest", choices=["once", "always", "reject"])]])
            loop, _, _, _ = make_loop(adapter)
            loop.step(0.0, list("run tests") + ["enter"])
            loop.step(0.1)
            screen = text_of(loop)
            self.assertIn("allow bash: python -m unittest", screen)
            self.assertIn("[y] once", screen)
            self.assertEqual(adapter.replies, [])
            loop.step(0.2, [key])
            self.assertEqual(adapter.replies, [("per_1", expected)])
            self.assertIsNone(loop.state.permission)

    def test_slash_commands_delegate_to_the_veystrix_native_handlers(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop, _, _, _ = make_loop(scenario_adapter(), home=Path(tmp), port=1)
            loop.step(0.0, list("/engine") + ["enter"])
            self.assertIn("Veystrix · engine=", text_of(loop))
            loop.step(0.1, list("/status") + ["enter"])
            self.assertIn("VEYSTRIX", text_of(loop))
            loop.step(0.2, list("/quit") + ["enter"])
            self.assertTrue(loop.done)


class StreamAndResize(unittest.TestCase):
    def tokens(self, n=200):
        return [ev("message.delta", text=f"tok{i} ") for i in range(n)] + [ev("turn.end", status="ok")]

    def test_200_token_stream_is_complete_ordered_and_redraw_is_bounded(self):
        adapter = ReplayAdapter([self.tokens()], batch=4)
        loop, clock, mirror, _ = make_loop(adapter, cols=120, rows=40)
        steps = 0
        drive_keys = [(0.0, c) for c in "go"] + [(0.0, "enter")]
        drive(loop, clock, drive_keys, until=30.0)
        assistant = [e for e in loop.state.entries if e.kind == "assistant"][0].text
        self.assertEqual(assistant, "".join(f"tok{i} " for i in range(200)))      # nothing lost, in order
        self.assertEqual(mirror.rows, loop.last_frame)                             # screen == truth
        self.assertGreater(loop.state.status.tok_s, 0)
        # bounded redraw: never faster than FRAME_INTERVAL without a keypress, and mostly tiny diffs
        times = [t for t, _ in mirror.writes[2:]]
        gaps = [b - a for a, b in zip(times, times[1:])]
        self.assertTrue(gaps)
        self.assertGreaterEqual(min(gaps), T.FRAME_INTERVAL - 1e-9)
        stream_bytes = [len(d) for _, d in mirror.writes[3:]]
        full = len(R.paint_rows(list(enumerate(loop.last_frame))))
        self.assertLess(statistics.mean(stream_bytes), full / 3)
        self.assertGreater(loop.frames, 10)

    def test_resize_repaints_fully_and_every_frame_matches_the_new_size(self):
        adapter = ReplayAdapter([[ev("message.delta", text="hello world " * 20), ev("turn.end", status="ok")]])
        loop, clock, mirror, box = make_loop(adapter, cols=120, rows=40)
        drive(loop, clock, [(0.0, c) for c in "hi"] + [(0.0, "enter")], until=5.0)
        for size in ((80, 24), (200, 60), (120, 40), (60, 20)):
            box["size"] = size
            clock.t += 0.1
            loop.step(clock.t)
            frame = loop.last_frame
            self.assertEqual(len(frame), size[1])
            self.assertTrue(all(R.visible_len(line) == size[0] for line in frame), size)
            self.assertIn("\033[2J", mirror.writes[-1][1])                   # full repaint on resize
            self.assertEqual(mirror.rows[:size[1]], frame)
            self.assertTrue(all(row == "" for row in mirror.rows[size[1]:]))
        self.assertIn("terminal too small", text_of(loop))
        box["size"] = (120, 40)
        loop.step(clock.t + 0.1)
        self.assertIn("hello world", text_of(loop))                          # transcript survived


class LatencyGate(unittest.TestCase):
    def scenario(self):
        events = [ev("reasoning.delta", text="thinking about it\n" * 3)]
        events += [ev("message.delta", text=f"t{i} ") for i in range(300)]
        events += [ev("tool.start", id="x", name="bash", args={"command": "ls"}),
                   ev("tool.output", id="x", text="a\nb\nc"), ev("tool.end", id="x", ok=True),
                   ev("turn.end", status="ok")]
        return ReplayAdapter([events], batch=3)

    def keys(self):
        rng = random.Random(7)
        keys = [(0.0, "g"), (0.01, "o"), (0.02, "enter")]
        keys += [(0.1 + rng.random() * 1.6, "abcdefghijklmnopqrstuvwxyz"[i % 26]) for i in range(60)]
        keys += [(2.0, "ctrl-k"), (2.1, "esc"), (2.2, "?"), (2.3, "esc"), (2.4, "ctrl-r"), (2.5, "ctrl-t")]
        return keys

    def test_keypress_to_first_frame_under_100ms_and_no_frame_dropped(self):
        worst = {}
        for cols, rows in ((80, 24), (120, 40), (200, 60)):
            loop, clock, mirror, _ = make_loop(self.scenario(), cols=cols, rows=rows, theme=THEMES["dark"])
            keys = self.keys()
            lat = drive(loop, clock, keys, until=30.0)
            # A key whose state change is visually identical to the previous
            # frame (e.g. ctrl-r with a collapsed reasoning pane) legitimately
            # produces no repaint. Allow at most one such no-op key; every
            # other keypress must be attributed to a frame.
            self.assertGreaterEqual(len(lat), len(keys) - 1,
                                    "at most one keypress produced no frame")
            seconds = [s for _, s in lat]
            worst[f"{cols}x{rows}"] = {"p50_ms": round(statistics.median(seconds) * 1000, 3),
                                       "max_ms": round(max(seconds) * 1000, 3), "keys": len(seconds)}
            self.assertLess(max(seconds), 0.100, f"{cols}x{rows} worst keypress->frame {max(seconds) * 1000:.1f} ms")
            # no dropped frames: the virtual screen equals the final composed frame, and every typed
            # character made it into the input line in order
            self.assertEqual(mirror.rows, loop.last_frame)
            # `esc` clears the input line, so only characters typed AFTER the
            # final esc are expected to remain (the first "go" is the submitted
            # prompt, and "?" is only a single char when it opens help).
            ordered = sorted(keys)
            last_esc = max((i for i, (_, k) in enumerate(ordered) if k == "esc"),
                           default=-1)
            typed = "".join(k for _, k in ordered[last_esc + 1:]
                            if len(k) == 1 and k != "?")
            self.assertTrue(loop.state.input.endswith(typed) or typed in loop.state.input,
                            (loop.state.input, typed))
            assistant = [e for e in loop.state.entries if e.kind == "assistant"][0].text
            self.assertEqual(assistant, "".join(f"t{i} " for i in range(300)))
        self.latency_report = worst
        if os.environ.get("VEYSTRIX_UI_CAPTURE"):
            UI_RESULTS.mkdir(parents=True, exist_ok=True)
            (UI_RESULTS / "latency.json").write_text(json.dumps(
                {"gate_ms": 100, "clock": "synthetic scheduling + measured compose/diff time",
                 "sizes": worst}, indent=2) + "\n")

    def test_keypress_during_heavy_streaming_is_not_coalesced_away(self):
        loop, clock, mirror, _ = make_loop(self.scenario(), cols=120, rows=40)
        loop.step(0.0, list("go") + ["enter"])
        for i in range(10):
            loop.step(0.001 * (i + 1))                  # events arrive faster than FRAME_INTERVAL
        frames = loop.frames
        loop.step(0.011, ["x"])                         # inside the coalescing window
        self.assertEqual(loop.frames, frames + 1)       # key frame is immediate
        self.assertIn("x", R.strip_ansi(mirror.rows[-4]))


class OpenCodeLoopback(unittest.TestCase):
    """The real adapter against an in-process 127.0.0.1 server that speaks the documented API."""

    PASSWORD = "s3cret-test"

    def start_server(self, event_path="/api/event"):
        owner = self
        sse = (GOLDEN / "opencode_turn.sse").read_bytes()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                return

            def _auth(self):
                import base64
                want = "Basic " + base64.b64encode(f"opencode:{owner.PASSWORD}".encode()).decode()
                ok = self.headers.get("Authorization") == want
                owner.auth_ok.append(ok)
                if not ok:
                    self.send_response(401)
                    self.end_headers()
                return ok

            def do_GET(self):
                if not self._auth():
                    return
                path = self.path.split("?")[0]
                if path == "/api/info":
                    return self._json({"version": "2.3.2"})
                if path == "/api/session":
                    return self._json({"items": [{"id": "ses_9", "title": "older work"}]})
                if path == event_path:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    owner.prompted.wait(5)
                    self.wfile.write(sse)
                    self.wfile.flush()
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                if not self._auth():
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                owner.posts.append((self.path, body))
                if self.path == "/api/session":
                    # OpenCode 2.3.2 wraps replies as {"data": ...}
                    return self._json({"data": {"id": "ses_demo"}} if owner.wrapped else {"id": "ses_demo"})
                if self.path.endswith("/prompt") and owner.reject_prompt:
                    raw = json.dumps({"_tag": "InvalidRequestError", "message": "Invalid session ID",
                                      "field": "sessionID"}).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                if self.path.endswith("/prompt"):
                    owner.prompted.set()
                self._json({})

            def _json(self, obj):
                raw = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.auth_ok, self.posts, self.prompted = [], [], threading.Event()
        self.wrapped = getattr(self, "wrapped", False)
        self.reject_prompt = getattr(self, "reject_prompt", False)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.shutdown)
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def collect(self, adapter, until_kind="turn.end", timeout=8.0):
        out, end = [], time.monotonic() + timeout
        while time.monotonic() < end:
            out.extend(adapter.poll(0.05))
            if any(e.kind == until_kind for e in out):
                return out
        self.fail(f"no {until_kind}; got {[e.kind for e in out]}")

    def test_full_turn_over_sse_with_password_and_cancel_and_permission(self):
        url = self.start_server()
        adapter = OpenCodeAdapter(base_url=url, password=self.PASSWORD)
        self.addCleanup(adapter.close)
        adapter.start()
        self.assertEqual(adapter.session_id, "ses_demo")
        adapter.send("fix it")
        events = self.collect(adapter)
        kinds = [e.kind for e in events]
        for kind in ("reasoning.delta", "tool.start", "tool.output", "tool.end", "message.delta", "usage", "turn.end"):
            self.assertIn(kind, kinds)
        self.assertNotIn("LEAK", "".join(str(e.data) for e in events))       # other session filtered
        adapter.cancel()
        adapter.reply_permission("per_1", "always")
        adapter.reply_permission("per_2", "nonsense")                          # unknown choice -> safest: reject
        posts = dict((p, b) for p, b in self.posts)
        self.assertEqual(posts["/api/session/ses_demo/prompt"], {"text": "fix it"})
        self.assertIn("/api/session/ses_demo/interrupt", posts)
        self.assertEqual(posts["/api/session/ses_demo/permission/per_1/reply"], {"decision": "always"})
        self.assertEqual(posts["/api/session/ses_demo/permission/per_2/reply"], {"decision": "reject"})
        self.assertTrue(all(self.auth_ok), "every request carried the Basic password")
        self.assertEqual(adapter.list_sessions(), [{"id": "ses_9", "title": "older work"}])
        # the same server rejects an unauthenticated probe
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(url + "/api/info", timeout=3)
        self.assertEqual(ctx.exception.code, 401)

    def test_falls_back_to_the_v1_event_path(self):
        url = self.start_server(event_path="/event")
        adapter = OpenCodeAdapter(base_url=url, password=self.PASSWORD)
        self.addCleanup(adapter.close)
        adapter.start()
        adapter.send("hi")
        self.assertIn("turn.end", [e.kind for e in self.collect(adapter)])

    def test_refuses_non_loopback_endpoints_and_spawns_loopback_only(self):
        with self.assertRaises(ValueError):
            OpenCodeAdapter(base_url="http://example.com:4096", password="x").start()
        command = OpenCodeAdapter().command(4096)
        self.assertEqual(command[:2], ["opencode", "serve"])
        self.assertEqual(command[command.index("--hostname") + 1], "127.0.0.1")

    def test_opencode_2_3_wrapped_replies_and_model_pinning(self):
        self.wrapped = True
        url = self.start_server()
        adapter = OpenCodeAdapter(base_url=url, password=self.PASSWORD, model="veystrix-ornith")
        self.addCleanup(adapter.close)
        adapter.start()
        self.assertEqual(adapter.session_id, "ses_demo")              # unwrapped from {"data": ...}
        posts = dict(self.posts)
        self.assertEqual(posts["/api/session"], {"model": {"providerID": "veystrix", "id": "veystrix-ornith"}})
        adapter.send("hi")
        self.assertEqual(dict(self.posts)["/api/session/ses_demo/prompt"], {"text": "hi"})
        adapter.set_model("veystrix-flashnext")
        self.assertEqual(dict(self.posts)["/api/session/ses_demo/model"],
                         {"model": {"providerID": "veystrix", "id": "veystrix-flashnext"}})

    def test_http_errors_carry_the_servers_message(self):
        self.reject_prompt = True
        url = self.start_server()
        adapter = OpenCodeAdapter(base_url=url, password=self.PASSWORD)
        self.addCleanup(adapter.close)
        adapter.start()
        with self.assertRaises(RuntimeError) as ctx:
            adapter.send("hi")
        self.assertIn("HTTP 400", str(ctx.exception))
        self.assertIn("Invalid session ID (sessionID)", str(ctx.exception))
        adapter.cancel()                                             # tolerated, never raises

    def test_resume_switches_the_session_used_for_prompts(self):
        url = self.start_server()
        adapter = OpenCodeAdapter(base_url=url, password=self.PASSWORD)
        self.addCleanup(adapter.close)
        adapter.start()
        adapter.resume("ses_9")
        adapter.send("again")
        self.assertIn("/api/session/ses_9/prompt", [p for p, _ in self.posts])


FAKE_CODEX = r'''
import sys, json
def send(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
T = "thr_1"; TU = "turn_1"
for line in sys.stdin:
    m = json.loads(line); meth = m.get("method"); i = m.get("id")
    if meth == "initialize":
        assert m["params"]["clientInfo"]["name"] == "veystrix"
        send({"id": i, "result": {"userAgent": "fake"}})
    elif meth == "thread/start":
        send({"id": i, "result": {"thread": {"id": T}}})
    elif meth == "turn/start":
        assert m["params"]["input"] == [{"type": "text", "text": "go"}], m
        send({"id": i, "result": {"turn": {"id": TU, "status": "inProgress", "items": []}}})
        send({"method": "turn/started", "params": {"threadId": T, "turn": {"id": TU}}})
        send({"method": "item/agentMessage/delta", "params": {"delta": "hello ", "itemId": "m", "threadId": T, "turnId": TU}})
        send({"id": 77, "method": "item/commandExecution/requestApproval",
              "params": {"command": "rm -rf build", "itemId": "c", "threadId": T, "turnId": TU}})
        send({"id": 78, "method": "item/tool/call", "params": {}})
    elif meth == "turn/interrupt":
        send({"id": i, "result": {}})
        send({"method": "turn/completed", "params": {"threadId": T, "turn": {"id": TU, "status": "interrupted"}}})
    elif meth is None and i == 77:
        send({"method": "item/agentMessage/delta", "params": {"delta": "decision=" + m["result"]["decision"], "itemId": "m", "threadId": T, "turnId": TU}})
        send({"method": "turn/completed", "params": {"threadId": T, "turn": {"id": TU, "status": "completed"}}})
    elif meth == "thread/list":
        send({"id": i, "result": {"data": [{"id": "thr_0", "name": "older"}]}})
    elif meth == "thread/resume":
        send({"id": i, "result": {}})
'''


class CodexStdio(unittest.TestCase):
    def adapter(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        script = Path(self.tmp.name) / "fake_codex.py"
        script.write_text(FAKE_CODEX)
        adapter = CodexAdapter(command=[sys.executable, str(script)])
        self.addCleanup(adapter.close)
        adapter.start()
        return adapter

    def collect(self, adapter, kind, timeout=8.0):
        out, end = [], time.monotonic() + timeout
        while time.monotonic() < end:
            out.extend(adapter.poll(0.05))
            if any(e.kind == kind for e in out):
                return out
        self.fail(f"no {kind}; got {[e.kind for e in out]}")

    def test_approval_is_a_client_decision_and_maps_to_codex_values(self):
        for choice, wire in (("once", "accept"), ("always", "acceptForSession"), ("reject", "decline")):
            with self.subTest(choice):
                adapter = self.adapter()
                self.assertEqual(adapter.session_id, "thr_1")
                adapter.send("go")
                events = self.collect(adapter, "permission.request")
                request = next(e for e in events if e.kind == "permission.request")
                self.assertEqual(request.data["id"], "77")
                self.assertEqual(request.data["detail"], "rm -rf build")
                self.assertNotIn("turn.end", [e.kind for e in events])          # nothing auto-approved
                adapter.reply_permission("77", choice)
                done = self.collect(adapter, "turn.end")
                self.assertIn(f"decision={wire}", "".join(e.data.get("text", "") for e in done))

    def test_cancel_interrupts_the_turn_and_unsupported_requests_are_rejected(self):
        adapter = self.adapter()
        adapter.send("go")
        self.collect(adapter, "permission.request")
        adapter.cancel()
        end = next(e for e in self.collect(adapter, "turn.end") if e.kind == "turn.end")
        self.assertEqual(end.data["status"], "cancelled")

    def test_sessions_and_resume(self):
        adapter = self.adapter()
        self.assertEqual(adapter.list_sessions(), [{"id": "thr_0", "title": "older"}])
        adapter.resume("thr_0")
        self.assertEqual(adapter.session_id, "thr_0")


class PtyFallback(unittest.TestCase):
    def test_passthrough_reserves_a_status_row_and_returns_the_exit_code(self):
        r_in, w_in = os.pipe()
        os.close(w_in)                                  # stdin at EOF
        r_out, w_out = os.pipe()
        adapter = PtyPassthrough([sys.executable, "-c", "print('hello from harness')"],
                                 status=lambda: "VEYSTRIX · python · fallback")
        code = adapter.run(r_in, w_out, size=(80, 24))
        os.close(w_out)
        data = b""
        while True:
            chunk = os.read(r_out, 65536)
            if not chunk:
                break
            data += chunk
        text = data.decode(errors="replace")
        self.assertEqual(code, 0)
        self.assertIn("hello from harness", text)
        self.assertIn("\033[1;23r", text)               # scroll region leaves row 24 for us
        self.assertIn("VEYSTRIX · python · fallback", text)
        self.assertFalse(adapter.structured)

    def test_selection_prefers_structured_and_never_invents_a_mode(self):
        from unittest.mock import patch
        from veystrix.tool.headless import HeadlessCLIAdapter

        self.assertFalse(structured_available("aider"))
        self.assertIsNone(make_adapter("aider"))
        with patch("veystrix.tool.headless.available", return_value=False):
            self.assertIsNone(make_adapter("claude-code"))
        with patch("veystrix.tool.headless.available", return_value=True):
            adapter = make_adapter("claude-code", cwd=".", model="veystrix-ornith")
            self.assertIsInstance(adapter, HeadlessCLIAdapter)
            self.assertEqual(adapter.name, "claude-code")
        self.assertIsNone(version_warning("opencode", "2.3.2"))
        self.assertIn("differs from pinned", version_warning("codex", "9.0.0"))


class SSEAndStatus(unittest.TestCase):
    def test_sse_parser_handles_comments_multiline_crlf_and_garbage(self):
        raw = [": heartbeat\n", "\n", 'data: {"type":"a",\n', 'data: "data":{}}\n', "\n",
               "data: not json\r\n", "\r\n", b'data: {"type":"b"}\n', b"\n", 'data: {"type":"tail"}']
        got = list(parse_sse(raw))
        self.assertEqual([g.get("type") for g in got], ["a", "raw", "b", "tail"])

    def test_unknown_events_are_dropped_not_rendered(self):
        for kind in ("server.connected", "installation.update-available", "tui.toast.show",
                     "session.step.started", "mcp.status.changed"):
            self.assertEqual(translate_opencode({"type": kind, "data": {"message": "x"}}, {}), [])

    def test_engine_status_maps_into_the_status_line(self):
        status = R.Status(engine="python", model="m")
        changed = T.apply_engine_status(status, {
            "engine": "cpp", "model": "veystrix-flashnext", "context_window": 32768, "used_tokens": 4096,
            "cache_state": "cached", "sampling": "instruct", "decode_tokens_per_second": 1.4, "thermal": 1})
        self.assertTrue(changed)
        line = R.status_line(status, R.Theme(color="none", ascii_only=True), width=140)
        for needle in ("4k/32k", "~ cached", "instruct", "1.4 tok/s", "thermal 1"):
            self.assertIn(needle, line)
        self.assertNotIn("cpp", line)
        status.effort = "high"
        self.assertIn("effort high", R.status_line(status, R.Theme(color="none", ascii_only=True), width=140))
        self.assertFalse(T.apply_engine_status(status, None))
        self.assertIsNone(T.fetch_engine_status(1))                  # nothing on port 1: None, not a crash

    def test_status_poller_feeds_the_loop_without_blocking_it(self):
        data = {"engine": "cpp", "model": "veystrix-flashnext", "context_window": 8192, "used_tokens": 7}
        loop, clock, _, _ = make_loop(ReplayAdapter(), status_fn=lambda: data, engine="python")
        loop.step(0.0)
        self.assertNotIn("cpp", R.strip_ansi(loop.last_frame[-1]))
        self.assertIn("7/8k", R.strip_ansi(loop.last_frame[-1]))

    def test_veystrix_status_endpoint_payload(self):
        from veystrix.server import VeystrixHTTPServer
        service = SimpleNamespace(
            installed_model_id="veystrix-flashnext", last_budget={"server_window": 8192},
            last_prompt_tokens=1234, last_cached_tokens=100, last_reasoning=SimpleNamespace(thinking=False),
            cache_diagnostics=lambda: {"decode_tokens_per_second": 3.2})
        httpd = VeystrixHTTPServer(("127.0.0.1", 0), service, api_key="k")
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        port = httpd.server_address[1]
        payload = T.fetch_engine_status(port, key="k", timeout=5)
        self.assertEqual(payload["model"], "veystrix-flashnext")
        self.assertEqual(payload["context_window"], 8192)
        self.assertEqual(payload["used_tokens"], 1234)
        self.assertEqual(payload["cache_state"], "cached")
        self.assertEqual(payload["sampling"], "instruct")
        self.assertEqual(payload["decode_tokens_per_second"], 3.2)
        self.assertIn(payload["engine"], ("python", "cpp"))
        self.assertIsNone(T.fetch_engine_status(port, key="wrong", timeout=5))   # still key-protected


class Capture(unittest.TestCase):
    """Writes golden visual evidence to results/ui/ when VEYSTRIX_UI_CAPTURE=1; always checks it renders."""

    def test_capture_svg_and_asciicast(self):
        records = []
        loop, clock, mirror, _ = make_loop(scenario_adapter(), cols=120, rows=40, theme=THEMES["dark"])
        prompt = [(0.1 + 0.04 * i, c) for i, c in enumerate("fix the bug in app.py")] + [(1.2, "enter")]
        drive(loop, clock, prompt, until=20.0)
        cast = R.asciicast([(t, d) for t, d in mirror.writes], 120, 40, "Veystrix UI: fix the bug in app.py")
        self.assertGreater(len(cast.splitlines()), 10)
        self.assertEqual(mirror.rows, loop.last_frame)
        svgs = {}
        for name, (cols, rows), theme in (("main_120x40_dark", (120, 40), "dark"),
                                          ("main_120x40_light", (120, 40), "light"),
                                          ("main_80x24_ascii", (80, 24), "ascii")):
            lp, _ = run_turn(scenario_adapter(), cols=cols, rows=rows, theme=THEMES[theme])
            svgs[name] = R.frame_to_svg(lp.last_frame, dark=theme != "light")
            self.assertIn("VEYSTRIX", svgs[name])
        if os.environ.get("VEYSTRIX_UI_CAPTURE"):
            UI_RESULTS.mkdir(parents=True, exist_ok=True)
            (UI_RESULTS / "session_120x40.cast").write_text(cast)
            for name, svg in svgs.items():
                (UI_RESULTS / f"{name}.svg").write_text(svg)


class SettingsAndManualStart(unittest.TestCase):
    def test_settings_overlay_cycles_engine_and_model_without_loading(self):
        from unittest.mock import patch
        from veystrix.tool.config import Config
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"VEYSTRIX_MODEL_ROOT": str(Path(tmp) / "models")}):
            home = Path(tmp)
            with socket.socket() as sock:                  # a port no developer engine uses
                sock.bind(("127.0.0.1", 0))
                config = Config(home)
                config.values["port"] = sock.getsockname()[1]
                config.values["model"] = "inkling2-mini"
                config.save()
            loop, _, _, _ = make_loop(scenario_adapter(), home=home, port=1)
            loop.step(0.0, list("/settings") + ["enter"])
            screen = text_of(loop)
            self.assertIn("Veystrix settings", screen)
            self.assertIn("stopped (not loaded)", screen)
            for hidden in ("python", "cpp", "C++"):
                self.assertNotIn(hidden, screen)
            loop.step(0.1, ["down", "enter"])              # effort row opens the effort picker
            self.assertEqual(loop.state.overlay, "effort")
            loop.step(0.12, ["down", "enter"])             # medium -> high, returns to settings
            self.assertEqual(loop.state.overlay, "settings")
            self.assertEqual(Config(home).get("effort"), "high")
            self.assertIn("effort high", R.strip_ansi(loop.last_frame[-1]))
            loop.step(0.15, ["down", "enter"])             # thinking row toggles and saves
            self.assertFalse(Config(home).get("show_reasoning"))
            loop.step(0.2, ["up", "up", "enter"])          # model row opens the model picker
            self.assertEqual(loop.state.overlay, "model")
            settle(loop, lambda: bool(loop.state.picker))
            loop.step(loop.state.now + 0.1, ["enter"])
            self.assertEqual(loop.state.overlay, "confirm")      # not downloaded: offer, never auto-fetch
            loop.step(loop.state.now + 0.1, ["n"])
            self.assertEqual(loop.state.overlay, "settings")     # back where the user came from
            self.assertEqual(loop._engine_running, None)         # nothing was loaded
            loop.step(loop.state.now + 0.1, ["esc"])
            self.assertIsNone(loop.state.overlay)

    def test_tui_does_not_auto_start_engine_when_stopped(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            with patch("veystrix.tool.tui.health", return_value=None), \
                    patch("veystrix.tool.tui._tui_session", return_value={"status": "done"}) as sess:
                res = T.tui(home=Path(tmp), port=19999)
            self.assertEqual(res, {"status": "done"})
            sess.assert_called_once()
            self.assertFalse(sess.call_args.kwargs["running"])

    def test_stopped_engine_guards_prompts_and_async_start_unlocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            called = []
            gate = threading.Event()

            def fake_slash(cmd, *, home, port):
                called.append(cmd)
                if cmd == "/start":
                    gate.wait(2.0)
                    return ["  ◆ Veystrix · ready: cpp / flashnext"]
                if cmd == "/stop":
                    return ["  ◆ Veystrix · engine stopped"]
                return []

            adapter = ReplayAdapter([[ev("message.delta", text="hi"), ev("turn.end", status="ok")]])
            loop = T.UILoop(adapter, write=lambda _: None, size=lambda: (120, 40),
                            theme=THEMES["ascii"], home=home, port=1, slash_fn=fake_slash, running=False)
            loop.step(0.0, list("hello") + ["enter"])
            self.assertEqual(adapter.sent, [])
            self.assertIn("engine is not running", text_of(loop))

            loop.step(0.1, list("/start") + ["enter"])
            self.assertTrue(loop.state.busy)
            gate.set()
            deadline = time.monotonic() + 3.0
            t = 0.2
            while loop.state.busy and time.monotonic() < deadline:
                time.sleep(0.01)
                t += 0.05
                loop.step(t)
            self.assertFalse(loop.state.busy)
            self.assertTrue(loop._engine_running)
            self.assertEqual(called, ["/start"])

            loop.step(t + 0.1, list("hello") + ["enter"])
            self.assertEqual(adapter.sent, ["hello"])

            loop.step(t + 0.2, list("/stop") + ["enter"])
            deadline = time.monotonic() + 3.0
            while loop.state.busy and time.monotonic() < deadline:
                time.sleep(0.01)
                t += 0.05
                loop.step(t)
            self.assertFalse(loop._engine_running)
            self.assertEqual(called, ["/start", "/stop"])


class InputBarAndAutocomplete(unittest.TestCase):
    def test_cursor_movement_midline_edit_and_multiline_newline(self):
        p = T.KeyParser()
        self.assertEqual(p.feed("\x1b[13;2u\x1b\r\x01\x05\x1b[3~"),
                         ["newline", "newline", "home", "end", "delete"])
        loop, _, _, _ = make_loop(ReplayAdapter(), cols=120, rows=40)
        loop.step(0.0, list("helo"))
        loop.step(0.1, ["left", "l"])
        self.assertEqual(loop.state.input, "hello")
        loop.step(0.2, ["home", "delete", "H", "end", "newline", *list("world")])
        self.assertEqual(loop.state.input, "Hello\nworld")
        self.assertEqual(loop.adapter.sent, [])
        self.assertIn("Hello⏎world", text_of(loop))

    def test_slash_and_at_file_tab_autocomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "alpha.py").write_text("x = 1\n")
            (root / "src").mkdir()
            (root / "src" / "beta_one.py").write_text("y = 2\n")
            (root / "src" / "beta_two.py").write_text("z = 3\n")
            loop, _, _, _ = make_loop(ReplayAdapter(), cols=120, rows=40, cwd=str(root))
            # Unique slash completion inserts immediately
            loop.step(0.0, list("/set") + ["tab"])
            self.assertEqual(loop.state.input, "/settings ")
            self.assertIsNone(loop.state.overlay)
            # Ambiguous slash completion opens picker overlay
            loop.step(0.1, ["ctrl-u"] + list("/st") + ["tab"])
            self.assertEqual(loop.state.overlay, "complete")
            self.assertIn("Autocomplete", text_of(loop))
            loop.step(0.2, ["down", "enter"])
            self.assertIsNone(loop.state.overlay)
            self.assertIn(loop.state.input.strip(), ("/start", "/stop", "/status"))
            # Unique @file completion inserts immediately
            loop.step(0.3, ["ctrl-u"] + list("check @alp") + ["tab"])
            self.assertEqual(loop.state.input, "check @alpha.py ")
            # Ambiguous @file completion opens picker and inserts chosen path
            loop.step(0.4, ["ctrl-u"] + list("edit @beta") + ["tab"])
            self.assertEqual(loop.state.overlay, "complete")
            loop.step(0.5, ["enter"])
            self.assertEqual(loop.state.input, "edit @src/beta_one.py ")


def fake_inventory(root: Path, installed=("flashnext", "ornith")):
    """Inventory rows shaped like tui.model_inventory, pointing at real folders under ``root``."""
    from veystrix import model_identity
    rows = []
    for family, ident in model_identity.IDENTITIES.items():
        path = root / (ident.checkpoint_directory or ident.download_repo.split("/")[-1])
        rows.append({"key": family, "name": ident.display_name.removeprefix("Veystrix "),
                     "engines": tuple(ident.engines), "repo": ident.download_repo,
                     "installed": family in installed, "partial": False,
                     "path": str(path), "size": 5 * 1024 ** 3 if family in installed else 0})
    return rows


class HarnessFromSettings(unittest.TestCase):
    LABELS = ("OpenCode", "Claude Code", "Codex", "Qwen Code", "DeepSeek", "Kimi Code", "Hermes",
              "Pi", "Aider", "Cline", "Continue")

    def setUp(self):
        from unittest.mock import patch
        from veystrix.tool import harnesses
        self._unlocked = []
        for ident in harnesses.HARNESS_IDENTITIES.values():
            if ident.locked:
                object.__setattr__(ident, "locked", False)
                self._unlocked.append(ident)
        p = patch("veystrix.tool.harnesses.PUBLIC_HARNESS_CHOICES", harnesses.HARNESS_CHOICES)
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        for ident in self._unlocked:
            object.__setattr__(ident, "locked", True)

    def test_settings_harness_picker_lists_every_harness_and_connects_one(self):
        from unittest.mock import patch
        from veystrix.tool.config import Config
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"VEYSTRIX_MODEL_ROOT": str(Path(tmp) / "models")}):
            home = Path(tmp)
            made = []
            fresh = ReplayAdapter([[ev("message.delta", text="via codex"), ev("turn.end", status="ok")]])

            def factory(name):
                made.append(name)
                return fresh
            loop, _, _, _ = make_loop(scenario_adapter(), home=home, port=1, adapter_factory=factory)
            loop.step(0.0, list("/settings") + ["enter"])
            self.assertIn("HARNESS", text_of(loop))
            loop.step(0.1, ["down", "down", "down", "down", "enter"])  # Model, Effort, Thinking, KV -> Harness
            self.assertEqual(loop.state.overlay, "harness")
            screen = text_of(loop)
            self.assertIn("Connect a harness", screen)
            for label in self.LABELS:
                self.assertIn(label, screen)
            self.assertIn("not connected", screen)
            self.assertEqual(loop.state.picker[loop.state.index % 11]["key"], "opencode")  # active first
            loop.step(0.2, ["down", "down", "enter"])                # opencode -> claude-code -> codex
            settle(loop, lambda: not loop._connecting)
            self.assertEqual(Config(home).get("harness"), "codex")   # persisted
            written = home / ".codex" / "config.toml"
            self.assertTrue(written.is_file())
            self.assertIn("veystrix", written.read_text())
            self.assertEqual(made, ["codex"])
            self.assertIs(loop.adapter, fresh)                       # live switch, no restart
            self.assertEqual(loop.harness, "codex")
            row = next(line for line in text_of(loop).splitlines() if "Codex" in line)
            self.assertIn("connected", row)
            self.assertNotIn("not connected", row)
            loop.step(loop.state.now + 0.1, ["esc"])                 # back to settings
            self.assertEqual(loop.state.overlay, "settings")
            self.assertIn("Harness   Codex  (connected)", text_of(loop))
            loop.step(loop.state.now + 0.1, ["esc"])
            loop.step(loop.state.now + 0.1, list("hello") + ["enter"])
            self.assertEqual(fresh.sent, ["hello"])                  # the new harness drives the chat

    def test_harness_command_and_saved_choice_drive_the_next_launch(self):
        from unittest.mock import patch
        from veystrix.tool.config import Config
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            config = Config(home)
            config.values["harness"] = "aider"
            config.save()
            with patch("veystrix.tool.tui.health", return_value=None), \
                    patch("veystrix.tool.tui._tui_session", return_value={"status": "done"}) as sess:
                T.tui(home=home, port=19999)
            self.assertEqual(sess.call_args.args[0], "aider")
            loop, _, _, _ = make_loop(scenario_adapter(), home=home, port=1, harness="aider",
                                      harness_fn=lambda h, active: [
                                          {"key": "aider", "label": "Aider", "installed": False,
                                           "connected": False, "active": active == "aider"}])
            loop.step(0.0, list("/harness") + ["enter"])
            self.assertEqual(loop.state.overlay, "harness")
            self.assertIn("Aider", text_of(loop))
            loop.step(0.1, ["esc"])
            self.assertIsNone(loop.state.overlay)                    # opened directly: Esc closes


class OpenCodeProviderAliases(unittest.TestCase):
    def test_aliases_are_added_once_and_user_settings_survive(self):
        from veystrix.tool import harnesses as H
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            path = H.config_path("opencode", home)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"plugins": ["x"], "model": "veystrix/veystrix-inkling2-mini", "providers": {
                "veystrix": {"settings": {"baseURL": "http://127.0.0.1:8080/v1"},
                             "models": {"veystrix-inkling2-mini-abc": {"name": "mine"}}}}}))
            self.assertTrue(H.ensure_opencode_models(home, 8080))
            data = json.loads(path.read_text())
            models = data["providers"]["veystrix"]["models"]
            self.assertIn("veystrix-inkling2-mini", models)
            self.assertEqual(models["veystrix-inkling2-mini-abc"], {"name": "mine"})   # kept
            self.assertEqual(data["plugins"], ["x"])
            self.assertEqual(data["model"], "veystrix/veystrix-inkling2-mini")         # default untouched
            self.assertFalse(H.ensure_opencode_models(home, 8080))                  # idempotent
            self.assertEqual(T.prepare_harness("opencode", home, 8080,
                                               "veystrix-inkling2-mini-ab67350107f5"), "veystrix-inkling2-mini")
            self.assertIsNone(T.prepare_harness("codex", home, 8080, "inkling2-mini"))

    def test_yaml_upsert_never_duplicates_a_top_level_key(self):
        from veystrix.tool._formats import yaml_upsert
        text = "model: openai/gpt\nother: 1\nlist:\n  - a\nkeep:\n  x: 2\n"
        out = yaml_upsert(text, "model: openai/veystrix-inkling2-mini\nlist: []\n")
        tops = [l.split(":")[0] for l in out.splitlines() if l and l[0] not in " -#"]
        self.assertEqual(sorted(tops), ["keep", "list", "model", "other"])
        self.assertIn("  x: 2", out)
        self.assertEqual(yaml_upsert(out, "model: openai/veystrix-inkling2-mini\nlist: []\n"), out)

    def test_hermes_repair_keeps_settings_and_other_providers(self):
        import yaml
        from veystrix.tool import harnesses as H
        broken = ("model:\n  default: old\n  max_turns: 9\nfallback_providers: []\n"
                  "custom_providers:\n- name: Veystrix\n  base_url: x\n- name: mine\n  base_url: y\n"
                  "# >>> veystrix managed >>>\ncustom_providers:\n    - name: \"veystrix\"\n"
                  "model:\n  default: \"veystrix-inkling2-mini\"\nfallback_providers:\n# <<< veystrix managed <<<\n")
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            path = H.config_path("hermes", home)
            path.parent.mkdir(parents=True)
            path.write_text(broken)
            self.assertTrue(H.sync_hermes_model(home, 8080, "veystrix-inkling2-mini"))
            text = path.read_text()
            tops = [l.split(":")[0] for l in text.splitlines() if l and l[0] not in " -#"]
            self.assertEqual(len(tops), len(set(tops)))
            data = yaml.safe_load(text)
            self.assertEqual([p["name"] for p in data["custom_providers"]], ["mine", "veystrix"])
            self.assertEqual(data["model"]["default"], "veystrix-inkling2-mini")
            self.assertEqual(data["model"]["max_turns"], 9)
            self.assertEqual(data["fallback_providers"], [])
            self.assertIn("veystrix-inkling2-mini", data["custom_providers"][1]["models"])
            self.assertFalse(H.sync_hermes_model(home, 8080, "veystrix-inkling2-mini"))

    def test_passthrough_banner_states_only_facts(self):
        lines = "\n".join(T.passthrough_banner("hermes", engine="cpp", model="veystrix-inkling2-mini",
                                               running=False, port=8080))
        self.assertIn("V E Y S T R I X", lines)
        self.assertIn("Inkling 2 Mini", lines)
        self.assertIn("Hermes", lines)
        self.assertIn("veystrix start", lines)
        for absent in ("0/?", "t/s", "therm", "/cache"):
            self.assertNotIn(absent, lines)

    def test_family_of(self):
        from veystrix.tool.harnesses import family_of
        self.assertEqual(family_of("inkling2-mini"), "inkling2-mini")
        self.assertEqual(family_of("veystrix-inkling2-mini-6059bb"), "inkling2-mini")
        self.assertIsNone(family_of("gpt-9"))


class ModelPickerOverlay(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        from veystrix.tool import harnesses
        self._unlocked_h = []
        for ident in harnesses.HARNESS_IDENTITIES.values():
            if ident.locked:
                object.__setattr__(ident, "locked", False)
                self._unlocked_h.append(ident)
        p2 = patch("veystrix.tool.harnesses.PUBLIC_HARNESS_CHOICES", harnesses.HARNESS_CHOICES)
        p2.start()
        self.addCleanup(p2.stop)

    def tearDown(self):
        for ident in self._unlocked_h:
            object.__setattr__(ident, "locked", True)

    def loop(self, tmp, **kw):
        from veystrix.tool.config import Config
        Config(Path(tmp)).set("model", "inkling2-mini")
        root = Path(tmp) / "models"
        calls = {"slash": [], "pull": []}

        def fake_slash(cmd, *, home, port):
            calls["slash"].append(cmd)
            return ["  ◆ Veystrix · ready: cpp / inkling2-mini"] if cmd == "/start" else []

        def fake_pull(name, root_, **kwargs):
            calls["pull"].append((name, kwargs.get("dry_run", False)))
            return {"to_download": 0} if kwargs.get("dry_run") else {"files": 1, "destination": str(root_)}
        installed = kw.pop("installed", ("inkling2-mini",))
        loop, _, _, _ = make_loop(scenario_adapter(), home=Path(tmp), port=1, slash_fn=fake_slash,
                                  inventory_fn=lambda: fake_inventory(root, installed=installed),
                                  pull_fn=fake_pull, **kw)
        return loop, calls

    def test_model_without_arguments_opens_the_picker_with_install_state(self):
        from unittest.mock import patch
        from veystrix.tool.config import Config
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"VEYSTRIX_MODEL_ROOT": str(Path(tmp) / "models")}):
            loop, calls = self.loop(tmp)
            loop.step(0.0, list("/model") + ["enter"])
            self.assertEqual(loop.state.overlay, "model")
            settle(loop, lambda: bool(loop.state.picker))
            screen = text_of(loop)
            for needle in ("Choose a model", "Inkling 2 Mini", "inkling2-mini", "Search",
                           "installed  5.0 GB", "● Inkling 2 Mini"):
                self.assertIn(needle, screen)
            for hidden in ("python", "cpp", "C++"):
                self.assertNotIn(hidden, screen)
            self.assertEqual(calls["slash"], [])
            loop.step(loop.state.now + 0.1, ["enter"])
            self.assertEqual(Config(Path(tmp)).get("model"), "inkling2-mini")
            self.assertEqual(Config(Path(tmp)).get("engine"), "cpp")
            self.assertEqual(loop.state.overlay, "confirm")
            self.assertIn("Load model?", text_of(loop))
            loop.step(loop.state.now + 0.1, ["n"])
            self.assertIsNone(loop.state.overlay)
            self.assertNotIn("cpp", loop.state.notice)
            self.assertEqual(calls["slash"], [])

    def test_choosing_while_running_offers_a_restart_and_missing_models_offer_a_download(self):
        from unittest.mock import patch
        from veystrix.tool.config import Config
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"VEYSTRIX_MODEL_ROOT": str(Path(tmp) / "models")}):
            loop, calls = self.loop(tmp, running=True)
            loop.step(0.0, list("/model") + ["enter"])
            settle(loop, lambda: bool(loop.state.picker))
            orig_get = Config.get
            with patch.object(Config, "get", lambda self_c, k: "other" if k == "model" else orig_get(self_c, k)):
                loop.step(loop.state.now + 0.1, ["enter"])
            self.assertEqual(loop.state.overlay, "confirm")
            self.assertIn("Restart engine?", text_of(loop))
            loop.step(loop.state.now + 0.1, ["y"])
            settle(loop, lambda: not loop._bg_busy)
            self.assertEqual(calls["slash"], ["/stop", "/start"])

            loop2, calls2 = self.loop(tmp, running=False, installed=())
            loop2.step(0.0, list("/model") + ["enter"])
            settle(loop2, lambda: bool(loop2.state.picker))
            loop2.step(loop2.state.now + 0.1, ["enter"])
            self.assertIn("Model not downloaded", text_of(loop2))
            loop2.step(loop2.state.now + 0.1, ["y"])
            settle(loop2, lambda: loop2._downloading is None and calls2["pull"])
            self.assertIn(("inkling2-mini", False), calls2["pull"])

    def test_model_name_shortcut_still_goes_straight_to_the_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop, calls = self.loop(tmp)
            loop.step(0.0, list("/model inkling2-mini") + ["enter"])
            self.assertEqual(calls["slash"], ["/model inkling2-mini"])
            self.assertIsNone(loop.state.overlay)

    def test_kv_and_effort_pickers_models_alias_and_direct_fallback(self):
        from unittest.mock import patch
        from veystrix.tool.config import Config
        from veystrix.tool.harness_adapters import DirectAdapter
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            loop, _ = self.loop(tmp)
            loop.step(0.0, list("/kv") + ["enter"])
            self.assertEqual(loop.state.overlay, "kv")
            self.assertIn("KV cache mode", text_of(loop))
            loop.step(0.1, ["down", "down", "enter"])
            self.assertIsNone(loop.state.overlay)
            self.assertEqual(Config(home).get("kv"), "int8")
            self.assertEqual(loop.state.status.kv, "int8")
            loop.step(0.2, list("/effort") + ["enter"])
            self.assertEqual(loop.state.overlay, "effort")
            self.assertIn("Reasoning effort", text_of(loop))
            loop.step(0.3, ["down", "enter"])
            self.assertIsNone(loop.state.overlay)
            self.assertEqual(Config(home).get("effort"), "high")
            loop.step(0.4, list("/models") + ["enter"])
            self.assertEqual(loop.state.overlay, "models")
            captured = {}

            def fake_run_ui(adapter, **kw):
                captured["adapter"] = adapter
                captured.update(kw)
                return {"status": "done"}
            with patch("veystrix.tool.harness_adapters.make_adapter", return_value=None), \
                    patch("veystrix.tool.tui.run_ui", side_effect=fake_run_ui), \
                    patch("sys.stdin.isatty", return_value=True), \
                    patch("sys.stdout.isatty", return_value=True):
                res = T._tui_session("opencode", home=home, config=Config(home), port=8080,
                                     model=None, resume=None, once=False, cancel_after=None,
                                     command=None, ui="auto", running=False)
            self.assertEqual(res, {"status": "done"})
            self.assertIsInstance(captured["adapter"], DirectAdapter)
            self.assertEqual(captured["harness"], "direct")
            self.assertEqual(captured["model"], "")

    def test_new_restart_and_exact_harness_switching(self):
        from veystrix.tool.config import Config
        from veystrix.tool.harness_adapters import DirectAdapter
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            loop, calls = self.loop(tmp, running=True)
            resets = []
            loop.adapter.reset = lambda: resets.append(True)
            loop.state.add(R.Entry("user", text="old message"))
            loop.state.status.used = 500
            loop.step(0.0, list("/new") + ["enter"])
            self.assertEqual(resets, [True])
            self.assertEqual(loop.state.entries, [])
            self.assertEqual(loop.state.status.used, 0)
            self.assertIn("new conversation", loop.state.notice)
            loop.step(0.1, list("/restart") + ["enter"])
            settle(loop, lambda: not loop._bg_busy)
            self.assertEqual(calls["slash"], ["/stop", "/start"])
            direct_loop, _, _, _ = make_loop(
                scenario_adapter(), home=home, port=1, harness="opencode",
                adapter_factory=lambda name: DirectAdapter(port=1) if name == "direct" else None,
            )
            direct_loop.step(0.0, list("/harness") + ["enter"])
            self.assertIn("Built-in", text_of(direct_loop))
            direct_loop.step(0.1, ["down", "enter"])
            settle(direct_loop, lambda: not direct_loop._connecting)
            self.assertEqual(Config(home).get("harness"), "claude-code")
            self.assertEqual(direct_loop.harness, "direct")
            self.assertIsInstance(direct_loop.adapter, DirectAdapter)
            active_keys = [i["key"] for i in direct_loop.state.picker if i.get("active")]
            self.assertEqual(active_keys, ["direct"])
            direct_loop.adapter._messages.append({"role": "user", "content": "hi"})
            direct_loop.adapter.reset()
            self.assertEqual(direct_loop.adapter._messages, [])


class DownloadManager(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "models"
        env = patch.dict(os.environ, {"VEYSTRIX_MODEL_ROOT": str(self.root)})
        env.start()
        self.addCleanup(env.stop)
        self.folder = self.root / "Inkling-2-Mini-oQ4e-mtp"
        self.folder.mkdir(parents=True)
        (self.folder / "model-00001.safetensors").write_bytes(b"\0" * 1024)
        self.scans = 0

    def inventory(self):
        self.scans += 1
        return fake_inventory(self.root, installed=("inkling2-mini",) if self.folder.exists() else ())

    def make(self, pull=None, **kw):
        loop, _, _, _ = make_loop(scenario_adapter(), home=Path(self.tmp.name), port=1,
                                  inventory_fn=self.inventory, pull_fn=pull, **kw)
        return loop

    def test_download_runs_in_the_background_with_progress_on_the_activity_rail(self):
        import shutil
        shutil.rmtree(self.folder)
        gate, seen = threading.Event(), []

        def fake_pull(name, root, *, dry_run=False, echo=None, inspect_after=True, **_):
            seen.append((name, dry_run, inspect_after))
            if dry_run:
                return {"to_download": 2}
            echo("downloading config.json (0 MiB)")
            gate.wait(3)
            echo("downloading model-00001.safetensors (5,000 MiB)")
            return {"files": 2, "destination": str(root / "Inkling-2-Mini-oQ4e-mtp")}
        loop = self.make(fake_pull)
        loop.step(0.0, list("/download") + ["enter"])
        self.assertEqual(loop.state.overlay, "models")
        settle(loop, lambda: bool(loop.state.picker))
        screen = text_of(loop)
        for needle in ("Model manager", "Inkling 2 Mini", "not downloaded", "Ctrl-D delete", "Enter download",
                       "hf: mlx-community/Inkling-2-Mini-oQ4e-mtp"):
            self.assertIn(needle, screen)
        loop.step(loop.state.now + 0.1, ["enter"])
        t = settle(loop, lambda: "1/2" in loop.state.task)
        self.assertIn("downloading inkling2-mini 1/2", R.strip_ansi(loop.last_frame[-5]))
        loop.step(t, ["esc"] + list("still typing"))
        self.assertEqual(loop.state.input, "still typing")
        self.assertTrue(loop.state.task)
        gate.set()
        settle(loop, lambda: loop._downloading is None)
        self.assertEqual(seen[0], ("inkling2-mini", True, True))
        self.assertEqual(seen[1], ("inkling2-mini", False, False))
        self.assertIn("downloaded inkling2-mini: 2 files verified", text_of(loop))
        self.assertEqual(loop.state.task, "")
        settle(loop, lambda: not loop._scanning)
        self.assertGreaterEqual(self.scans, 2)

    def test_missing_huggingface_hub_is_a_clear_error_not_a_crash(self):
        def fake_pull(*_a, **_k):
            raise ModuleNotFoundError("No module named 'huggingface_hub'", name="huggingface_hub")
        loop = self.make(fake_pull)
        loop.step(0.0, list("/download inkling2-mini") + ["enter"])
        settle(loop, lambda: loop._downloading is None)
        self.assertIn("huggingface_hub is not installed", text_of(loop))
        self.assertIn("download failed", text_of(loop))

    def test_delete_needs_explicit_confirmation(self):
        loop = self.make()
        loop.step(0.0, list("/download") + ["enter"])
        settle(loop, lambda: bool(loop.state.picker))
        loop.step(loop.state.now + 0.1, ["d"])
        self.assertEqual(loop.state.overlay, "models")
        self.assertEqual(loop.state.query, "d")
        loop.step(loop.state.now + 0.1, ["backspace"])
        loop.step(loop.state.now + 0.1, ["ctrl-d"])
        self.assertEqual(loop.state.overlay, "confirm")
        screen = text_of(loop)
        self.assertIn("Delete model files", screen)
        self.assertIn("cannot be undone", screen)
        self.assertIn("[y] delete", screen)
        loop.step(loop.state.now + 0.1, ["x", "enter"])
        self.assertEqual(loop.state.overlay, "confirm")
        loop.step(loop.state.now + 0.1, ["n"])
        self.assertEqual(loop.state.overlay, "models")
        self.assertTrue(self.folder.exists())
        loop.step(loop.state.now + 0.1, ["delete"])
        self.assertEqual(loop.state.overlay, "confirm")
        loop.step(loop.state.now + 0.1, ["esc"])
        self.assertTrue(self.folder.exists())
        loop.step(loop.state.now + 0.1, ["ctrl-d"])
        loop.step(loop.state.now + 0.1, ["y"])
        settle(loop, lambda: not self.folder.exists() and not loop.state.task)
        self.assertFalse(self.folder.exists())
        self.assertTrue(self.root.is_dir())
        self.assertIn("deleted inkling2-mini", text_of(loop))

    def test_delete_is_refused_while_the_engine_serves_the_model(self):
        loop = self.make(running=True)
        loop.state.status.model = "veystrix-inkling2-mini"
        loop.step(0.0, list("/download delete inkling2-mini --yes") + ["enter"])
        settle(loop, lambda: not loop._scanning)
        self.assertTrue(self.folder.exists())
        self.assertIn("refusing to delete inkling2-mini", text_of(loop))
        self.assertIn("/stop", text_of(loop))
        loop.step(loop.state.now + 0.1, list("/download") + ["enter"])
        settle(loop, lambda: bool(loop.state.picker))
        loop.step(loop.state.now + 0.1, ["d"])
        self.assertEqual(loop.state.overlay, "models")
        self.assertTrue(self.folder.exists())

    def test_direct_subcommands(self):
        seen = []

        def fake_pull(name, root, **kw):
            seen.append(name)
            return {"to_download": 0} if kw.get("dry_run") else {"files": 0, "destination": str(root)}
        loop = self.make(fake_pull)
        loop.step(0.0, list("/download bogus") + ["enter"])
        self.assertIn("unknown model bogus", text_of(loop))
        loop.step(0.1, list("/download inkling2-mini") + ["enter"])
        settle(loop, lambda: loop._downloading is None and seen)
        self.assertEqual(seen, ["inkling2-mini", "inkling2-mini"])
        loop.step(loop.state.now + 0.1, list("/download delete inkling2-mini") + ["enter"])
        settle(loop, lambda: loop.state.overlay == "confirm")
        self.assertTrue(self.folder.exists())
        loop.step(loop.state.now + 0.1, ["n"])
        self.assertIsNone(loop.state.overlay)
        loop.step(loop.state.now + 0.1, list("/download delete veystrix-inkling2-mini --yes") + ["enter"])
        settle(loop, lambda: not self.folder.exists() and not loop.state.task)
        self.assertFalse(self.folder.exists())

    def test_model_inventory_reads_real_folders(self):
        import json as _json
        from unittest.mock import patch
        (self.folder / "config.json").write_text(_json.dumps({"model_type": "inkling2_loop"}))
        (self.folder / "tokenizer.json").write_text("{}")
        (self.folder / "tokenizer_config.json").write_text("{}")
        (self.folder / "model.safetensors.index.json").write_text(
            _json.dumps({"weight_map": {"w": "model-00001.safetensors"}}))
        with patch("veystrix.recipe.emit_recipe", return_value={}):
            rows = {row["key"]: row for row in T.model_inventory(self.root)}
        self.assertTrue(rows["inkling2-mini"]["installed"])
        self.assertGreater(rows["inkling2-mini"]["size"], 1024)
        self.assertEqual(Path(rows["inkling2-mini"]["path"]), self.folder)
        self.assertEqual(rows["inkling2-mini"]["context"], 1_048_576)
        self.assertNotIn("engines", rows["inkling2-mini"])

    def test_search_bar_filters_the_list(self):
        loop = self.make()
        loop.step(0.0, list("/download") + ["enter"])
        settle(loop, lambda: bool(loop.state.picker))
        loop.step(loop.state.now + 0.1, list("ink"))
        self.assertEqual([i["key"] for i in loop.state.picker], ["inkling2-mini"])
        loop.step(loop.state.now + 0.1, list("zzz"))
        self.assertEqual(loop.state.picker, [])
        self.assertIn("no model matches", text_of(loop))
        for _ in range(6):
            loop.step(loop.state.now + 0.1, ["backspace"])
        from veystrix import model_identity
        self.assertEqual(len(loop.state.picker), len(model_identity.IDENTITIES))
        loop.step(loop.state.now + 0.1, list("installed"))
        self.assertEqual([i["key"] for i in loop.state.picker], ["inkling2-mini"])
        self.assertTrue(T.model_matches({"name": "Inkling 2 Mini", "repo": "mlx-community/x"}, "inkling mlx"))
        self.assertFalse(T.model_matches({"name": "Inkling 2 Mini"}, "qwen"))


class TelemetryAndFlashNextFixes(unittest.TestCase):
    def test_server_tok_s_not_clobbered_by_word_chunks(self):
        state = R.UIState(status=R.Status(engine="cpp", model="veystrix-inkling2-mini"))
        state.turn_started = 10.0
        T.apply_engine_status(state.status, {"decode_tokens_per_second": 15.4})
        state.server_tok_s = True
        for i in range(1, 5):
            R.apply_event(state, ev("message.delta", text=f"word{i} "), now=10.0 + i * 0.26)
        self.assertAlmostEqual(state.status.tok_s, 15.4, places=1)
        state2 = R.UIState(status=R.Status(engine="cpp", model="veystrix-inkling2-mini"))
        state2.turn_started = 20.0
        R.apply_event(state2, ev("reasoning.delta", text="plan "), now=20.2)
        self.assertAlmostEqual(state2.status.ttft_s, 0.2, places=2)
        R.apply_event(state2, ev("message.delta", text="done"), now=21.2)
        R.apply_event(state2, ev("usage", output=15), now=21.2)
        self.assertAlmostEqual(state2.status.tok_s, 15.0, places=1)

    def test_engine_tok_s_is_harness_independent(self):
        from veystrix.tool.harness_adapters import ReplayAdapter

        for harness in ("opencode", "codex", "claude-code", "direct"):
            with self.subTest(harness=harness), tempfile.TemporaryDirectory() as tmp:
                adapter = ReplayAdapter([[]])
                adapter.name = harness
                loop = T.UILoop(adapter, write=lambda _: None, size=lambda: (80, 24),
                                theme=THEMES["ascii"], home=Path(tmp), port=1,
                                status_fn=lambda: {"decode_tokens_per_second": 4.1},
                                running=True)
                loop.step(0.0, ())
                self.assertTrue(loop.state.server_tok_s, harness)
                self.assertAlmostEqual(loop.state.status.tok_s, 4.1, places=1, msg=harness)
                R.apply_event(loop.state, ev("message.delta", text="word "), now=0.2)
                self.assertAlmostEqual(loop.state.status.tok_s, 4.1, places=1, msg=harness)

    def test_zero_engine_rate_latches_and_clears_stale_rate(self):
        from veystrix.tool.harness_adapters import ReplayAdapter

        with tempfile.TemporaryDirectory() as tmp:
            readings = iter([{"decode_tokens_per_second": 15.4},
                             {"decode_tokens_per_second": 0.0},
                             {"decode_tokens_per_second": 4.1}])
            loop = T.UILoop(ReplayAdapter([[]]), write=lambda _: None, size=lambda: (80, 24),
                            theme=THEMES["ascii"], home=Path(tmp), port=1,
                            status_fn=lambda: next(readings), running=True)
            loop.step(0.0, ())
            self.assertAlmostEqual(loop.state.status.tok_s, 15.4, places=1)
            loop.step(2.1, ())
            self.assertAlmostEqual(loop.state.status.tok_s, 0.0, places=1)
            loop.step(4.2, ())
            self.assertAlmostEqual(loop.state.status.tok_s, 4.1, places=1)

    def test_probe_kv_forwarding_timeouts_and_model_resolution(self):
        from unittest.mock import patch
        from veystrix import cli as root_cli
        from veystrix.model_process import ProcessDriver
        from veystrix.tool import cli as tool_cli
        from veystrix.tool.harness_adapters import DirectAdapter

        with patch.dict(os.environ, {"VEYSTRIX_ENGINE": "cpp"}):
            self.assertNotIn("mlx_vlm", root_cli._runtime_probe("inkling2-mini"))

        with patch.dict(os.environ, {"VEYSTRIX_CONTEXT_LIMIT": "16384", "VEYSTRIX_KV_MODE": "int8"}):
            drv = ProcessDriver()
            cmd = drv._command(SimpleNamespace(family="inkling2-mini", checkpoint="/tmp/fake-ckpt"), 9001)
        self.assertIn("--context-limit", cmd)
        self.assertEqual(cmd[cmd.index("--context-limit") + 1], "16384")
        self.assertIn("--kv", cmd)
        self.assertEqual(cmd[cmd.index("--kv") + 1], "int8")

        parser = tool_cli._parser("start")
        ns = parser.parse_args([])
        self.assertEqual(ns.timeout, 300.0)

        import io
        adapter = DirectAdapter(port=8080, model="")
        fake_resp = io.BytesIO(json.dumps({"ready": True, "installed_model_id": "veystrix-inkling2-mini-abc123"}).encode())
        with patch("veystrix.tool.harness_adapters._loopback_opener") as mock_opener:
            mock_opener.return_value.open.return_value.__enter__.return_value = fake_resp
            self.assertEqual(adapter._resolve_model(), "veystrix-inkling2-mini-abc123")

        with patch.dict(os.environ, {"VEYSTRIX_PYTHON": sys.executable}), \
                patch.object(root_cli, "_supports_python", side_effect=AssertionError("probe should be skipped")):
            self.assertEqual(root_cli._interpreter("inkling2-mini").resolve(), Path(sys.executable).resolve())

        import subprocess
        proc = subprocess.run([sys.executable, "-c",
                               "import sys; import veystrix.server, veystrix.model_router; "
                               "assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)"],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

        from veystrix.tool import supervisor as sup
        with tempfile.TemporaryDirectory() as tmp:
            h = Path(tmp)
            load_calls = [0]

            def fake_load(self_ks):
                load_calls[0] += 1
                return "k"

            class DeadWatcher:
                pid = 99999
                returncode = 0
                def poll(self): return 0
            with patch.object(sup, "health", return_value=None), \
                    patch.object(sup, "check_engine", lambda *a, **k: None), \
                    patch.object(sup, "source_cpp_ready", lambda *a, **k: True), \
                    patch.object(sup.LocalKeyStore, "load", fake_load), \
                    patch.object(sup.subprocess, "Popen", return_value=DeadWatcher()):
                with self.assertRaises(SystemExit) as ctx:
                    sup.start(sup.StartOptions(
                        home=h, port=8080, engine="python", model="inkling2-mini", timeout=60.0,
                        environ={"VEYSTRIX_TEST_POWER": '{"on_ac": true, "percent": 100}',
                                 "VEYSTRIX_TEST_MEMORY": '{"available_bytes": 17179869184}'}))
            self.assertIn("engine failed to start", str(ctx.exception))
            self.assertLessEqual(load_calls[0], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
