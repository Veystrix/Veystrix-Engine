"""Offline tests for the Built-in Veystrix agent harness.

A tiny loopback OpenAI-compatible server streams a tool call and then a final
answer. No real model, no network, no writes outside a temporary directory.
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from veystrix.tool.builtin_adapter import BuiltinAdapter
from veystrix.tool.harness_adapters import ev


class _Handler(BaseHTTPRequestHandler):
    bodies: list[dict] = []
    reply = None

    def log_message(self, *_args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        type(self).bodies.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in type(self).reply(body):
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def _sse(content="", calls=None, finish=None, usage=None):
    chunks = [{"choices": [{"delta": {"content": content}, "finish_reason": finish}]}]
    if calls:
        chunks = [{"choices": [{"delta": {"tool_calls": calls}, "finish_reason": "tool_calls"}]}]
    if usage:
        chunks.append({"usage": usage, "choices": []})
    return chunks


class BuiltinHarnessTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        (Path(self._tmp.name) / "README.md").write_text("hello from repo\n")
        self.server = None

    def tearDown(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()

    def _serve(self, replies):
        states = iter(replies)
        _Handler.bodies = []
        _Handler.reply = lambda _body: next(states)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self.server.server_address[1]

    def _drain(self, adapter, timeout=5.0):
        end = time.time() + timeout
        events = []
        while time.time() < end:
            events += adapter.poll(0.05)
            if any(e.kind == "turn.end" for e in events):
                return events
        return events

    def test_read_tool_loop_uses_local_tools(self):
        call = {"index": 0, "id": "call_1", "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"README.md"}'}}
        port = self._serve([_sse(calls=[call]), _sse(content="answer", finish="stop")])
        adapter = BuiltinAdapter(port=port, cwd=self._tmp.name, model="veystrix-ornith")
        adapter.start()
        adapter.send("read the readme")
        events = self._drain(adapter)
        kinds = [e.kind for e in events]
        self.assertIn("tool.start", kinds)
        self.assertIn("tool.output", kinds)
        self.assertIn("tool.end", kinds)
        self.assertEqual(events[-1], ev("turn.end", status="ok"))
        self.assertTrue(any("hello from repo" in str(e.data) for e in events if e.kind == "tool.output"))
        self.assertEqual(len(_Handler.bodies), 2)
        first, second = _Handler.bodies
        names = {t["function"]["name"] for t in first["tools"]}
        self.assertTrue({"read_file", "write_file", "run_command"}.issubset(names))
        self.assertEqual(second["messages"][-1]["role"], "tool")
        self.assertEqual(second["messages"][-1]["tool_call_id"], "call_1")

    def test_mutating_tool_honours_permission_reject(self):
        call = {"index": 0, "id": "call_2", "type": "function",
                "function": {"name": "run_command", "arguments": '{"command":"echo hi"}'}}
        port = self._serve([_sse(calls=[call]), _sse(content="done", finish="stop")])
        adapter = BuiltinAdapter(port=port, cwd=self._tmp.name, model="veystrix-ornith")
        adapter.start()
        adapter.send("run a command")
        end = time.time() + 5.0
        request = None
        while time.time() < end and request is None:
            request = next((e for e in adapter.poll(0.05) if e.kind == "permission.request"), None)
        self.assertIsNotNone(request)
        self.assertEqual(request.data["action"], "run_command")
        adapter.reply_permission(request.data["id"], "reject")
        events = self._drain(adapter)
        self.assertTrue(any(e.kind == "tool.end" and e.data.get("ok") is False for e in events))
        self.assertTrue(any("rejected" in str(e.data) for e in events if e.kind == "tool.output"))
        self.assertEqual(events[-1], ev("turn.end", status="ok"))


if __name__ == "__main__":
    unittest.main()
