"""Pre-caching every effort: what is remembered, what ./serve.sh --warm sends, and that it stops at a safety stop."""
import http.server
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from veystrix import warm as W

SYSTEM = {"role": "system", "content": "You are a coding agent. cwd=/Users/x/repo"}
TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {}}}}]


class RememberTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "warm"

    def test_tests_never_use_the_owners_real_directory(self):
        self.assertNotIn(".flashnext/warm", str(W.WARM_DIR))

    def test_warm_body_targets_requested_model(self):
        entry = {"system": SYSTEM, "tools": TOOLS}
        body = W.request_body(entry, "medium", "veystrix-nemotron35-lightning")
        self.assertEqual(body["model"], "veystrix-nemotron35-lightning")

    def test_family_names(self):
        self.assertEqual(W.family("QwenCode/0.27.0 (darwin; arm64)"), "qwencode")
        self.assertEqual(W.family("deepseek-harness/0.2.2-rc.1 (+https://x)"), "deepseek-harness")
        self.assertEqual(W.family(""), "unknown")
        self.assertNotIn("/", W.family("../../etc/passwd"))

    def test_a_tool_using_client_is_remembered_privately_and_only_when_it_changes(self):
        self.assertTrue(W.remember("QwenCode/0.27.0", [SYSTEM, {"role": "user", "content": "hi"}], TOOLS, self.dir))
        path = self.dir / "qwencode.json"
        self.assertEqual(oct(path.stat().st_mode & 0o777), "0o600")
        self.assertEqual(oct(self.dir.stat().st_mode & 0o777), "0o700")
        saved = json.loads(path.read_text())
        self.assertEqual(saved["system"], SYSTEM)
        self.assertEqual(saved["tools"], TOOLS)
        self.assertNotIn("hi", path.read_text())          # the user turn is not stored
        self.assertFalse(W.remember("QwenCode/0.27.0", [SYSTEM, {"role": "user", "content": "other"}], TOOLS, self.dir))
        changed = {"role": "system", "content": "different"}
        self.assertTrue(W.remember("QwenCode/0.27.0", [changed], TOOLS, self.dir))

    def test_requests_without_tools_or_a_system_message_are_not_remembered(self):
        self.assertFalse(W.remember("x", [SYSTEM], [], self.dir))
        self.assertFalse(W.remember("x", [{"role": "user", "content": "a"}], TOOLS, self.dir))
        self.assertFalse(W.remember("x", [], TOOLS, self.dir))
        self.assertEqual(W.load(self.dir), [])

    def test_a_write_failure_never_raises(self):
        blocked = Path(self.tmp.name) / "file"
        blocked.write_text("x")
        self.assertFalse(W.remember("x", [SYSTEM], TOOLS, blocked / "sub"))


class StubServer:
    """Serves /health, /v1/cache/diagnostics and a streaming chat endpoint; records every chat request."""

    def __init__(self, fail_at=None, error_event=None, busy=False):
        self.chats, self.fail_at, self.error_event, self.busy = [], fail_at, error_event, busy
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj, status=200):
                raw = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/health":
                    self._json({"status": "ok", "ready": True, "worker": {"busy": outer.busy}})
                else:
                    self._json({"last_prefix": {"prompt_tokens": 100, "hit_tokens": 0, "prefilled_tokens": 100, "outcome": "MISS"}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.chats.append({"body": body, "headers": {k.lower(): v for k, v in self.headers.items()}})
                if outer.fail_at is not None and len(outer.chats) == outer.fail_at:
                    return self._json({"error": {"message": "Do NOT retry automatically", "code": "memory_critical_stop"}}, 503)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if outer.error_event:
                    self.wfile.write(("data: " + json.dumps({"error": outer.error_event}) + "\n\n").encode())
                else:
                    self.wfile.write(b'data: {"choices":[{"delta":{"content":"r"}}]}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_port
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class WarmRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        W.remember("QwenCode/0.27.0", [SYSTEM], TOOLS, self.dir)
        W.remember("deepseek-harness/0.2", [{"role": "system", "content": "other agent"}], TOOLS, self.dir)
        self.lines = []

    def run_warm(self, stub, *args):
        return W.main(["--port", str(stub.port), *args], directory=self.dir, out=self.lines.append)

    def serve(self, **kw):
        stub = StubServer(**kw)
        self.addCleanup(stub.close)
        return stub

    def test_one_prefill_only_request_per_client_and_effort(self):
        stub = self.serve()
        self.assertEqual(self.run_warm(stub), 0)
        self.assertEqual(len(stub.chats), 2 * 4)
        efforts = [c["body"]["reasoning_effort"] for c in stub.chats]
        self.assertEqual(sorted(set(efforts)), ["high", "low", "medium", "xhigh"])
        for chat in stub.chats:
            self.assertEqual(chat["headers"]["x-veystrix-warm"], "1")
            self.assertEqual(chat["body"]["tools"], TOOLS)
            self.assertEqual(chat["body"]["messages"][0]["role"], "system")
            self.assertEqual(len(chat["body"]["messages"]), 2)
            self.assertTrue(chat["body"]["stream"])
        self.assertTrue(any("done." in line for line in self.lines))

    def test_efforts_clients_and_last_effort_can_be_chosen(self):
        stub = self.serve()
        self.assertEqual(self.run_warm(stub, "--efforts", "low,xhigh,medium", "--client", "qwencode", "--last", "medium"), 0)
        self.assertEqual([c["body"]["reasoning_effort"] for c in stub.chats], ["low", "xhigh", "medium"])
        self.assertEqual({c["headers"]["user-agent"][:8] for c in stub.chats}, {"QwenCode"})

    def test_off_is_not_a_separate_variant_and_unknown_efforts_are_refused(self):
        stub = self.serve()
        self.assertEqual(self.run_warm(stub, "--efforts", "off"), 2)
        self.assertEqual(stub.chats, [])

    def test_a_memory_safety_stop_ends_the_run_and_nothing_is_retried(self):
        stub = self.serve(fail_at=2)
        self.assertEqual(self.run_warm(stub), 5)
        self.assertEqual(len(stub.chats), 2)
        self.assertTrue(any("Nothing was retried" in line for line in self.lines))

    def test_a_cut_off_tool_call_after_one_token_counts_as_a_finished_prefill(self):
        stub = self.serve(error_event={"message": "the model began a tool call but the output limit of 1 tokens ended it first"})
        self.assertEqual(self.run_warm(stub, "--efforts", "low", "--client", "qwencode"), 0)

    def test_any_other_server_error_stops_the_run(self):
        stub = self.serve(error_event={"message": "something else broke"})
        self.assertEqual(self.run_warm(stub, "--efforts", "low"), 5)
        self.assertEqual(len(stub.chats), 1)

    def test_a_busy_server_is_left_alone(self):
        stub = self.serve(busy=True)
        self.assertEqual(self.run_warm(stub), 4)
        self.assertEqual(stub.chats, [])

    def test_dry_run_sends_nothing_and_nothing_remembered_says_so(self):
        stub = self.serve()
        self.assertEqual(self.run_warm(stub, "--dry-run"), 0)
        self.assertEqual(stub.chats, [])
        empty = tempfile.TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        lines = []
        self.assertEqual(W.main(["--port", str(stub.port)], directory=empty.name, out=lines.append), 1)
        self.assertIn("nothing to warm", lines[0])

    def test_an_unreachable_server_is_reported_not_crashed(self):
        stub = self.serve()
        port = stub.port
        stub.close()
        self.assertEqual(W.main(["--port", str(port)], directory=self.dir, out=self.lines.append), 3)


if __name__ == "__main__":
    unittest.main()
