"""Offline translator/argv coverage for the five headless CLI harness drivers."""
from __future__ import annotations

import unittest

from veystrix.tool.headless import (
    _claude_argv, _hermes_argv, _kimi_argv, _pi_argv, _qwen_argv,
    translate_claude_stream, translate_hermes, translate_kimi, translate_pi,
)


class HeadlessTranslatorTest(unittest.TestCase):
    def test_claude_shape(self):
        state = {}
        self.assertEqual(
            [e.kind for e in translate_claude_stream(
                {"type": "stream_event", "event": {"type": "content_block_delta",
                                                  "delta": {"type": "text_delta", "text": "hi"}}}, state)],
            ["message.delta"])
        self.assertEqual(
            [e.kind for e in translate_claude_stream(
                {"type": "assistant", "message": {"content": [
                    {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "x"}}]}}, state)],
            ["tool.start"])
        kinds = [e.kind for e in translate_claude_stream(
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "hello"}]}}, state)]
        self.assertEqual(kinds, ["tool.output", "tool.end"])
        translate_claude_stream({"type": "result", "usage": {"input_tokens": 10, "output_tokens": 4}}, state)
        self.assertTrue(state["finished"])

    def test_kimi_pi_hermes_shapes(self):
        state = {}
        kinds = [e.kind for e in translate_kimi({"role": "assistant", "content": [
            {"type": "text", "text": "hi"}], "tool_calls": [
            {"id": "t1", "function": {"name": "read_file", "arguments": '{"path":"x"}'}}]}, state)]
        self.assertEqual(kinds, ["message.delta", "tool.start"])
        self.assertEqual([e.kind for e in translate_kimi(
            {"role": "tool", "tool_call_id": "t1", "content": "ok"}, state)],
            ["tool.output", "tool.end"])

        self.assertEqual([e.kind for e in translate_pi(
            {"type": "tool_execution_start", "toolCallId": "p1", "toolName": "run_command",
             "args": {"command": "echo hi"}}, {})], ["tool.start"])
        self.assertEqual([e.kind for e in translate_pi(
            {"type": "tool_execution_end", "toolCallId": "p1", "isError": False,
             "result": {"content": "hi"}}, {})], ["tool.output", "tool.end"])
        pi_state = {}
        translate_pi({"type": "agent_end"}, pi_state)
        self.assertTrue(pi_state["finished"])
        # Pi errors must surface even when totalTokens == 0.
        pi_err = translate_pi({"type": "message_end", "message": {
            "role": "assistant", "stopReason": "error", "errorMessage": "model not found",
            "usage": {"totalTokens": 0}}}, {})
        self.assertEqual([e.kind for e in pi_err], ["error"])
        self.assertIn("model not found", pi_err[0].data["message"])
        # Non-streamed Pi assistant text in message_end is still emitted.
        pi_text = translate_pi({"type": "message_end", "message": {
            "role": "assistant", "content": [{"type": "text", "text": "hello"}]}}, {})
        self.assertEqual([e.kind for e in pi_text], ["message.delta"])
        self.assertEqual(pi_text[0].data["text"], "hello")

        self.assertEqual([e.kind for e in translate_hermes({"type": "text", "text": "hi"}, {})],
                         ["message.delta"])
        self.assertEqual([e.kind for e in translate_hermes(
            {"type": "tool_result", "tool_call_id": "h1", "output": "ok"}, {})],
            ["tool.output", "tool.end"])
        hermes_state = {}
        translate_hermes({"type": "result", "exit_code": 0, "tokens": {"input": 3, "output": 2}},
                         hermes_state)
        self.assertTrue(hermes_state["finished"])

    def test_argv_builders_resume_and_pin_model_where_supported(self):
        self.assertIn("--resume", _claude_argv("claude", "hi", "s1", "veystrix-ornith"))
        self.assertIn("--model", _qwen_argv("qwen", "hi", "s1", "veystrix-ornith"))
        self.assertIn("--session", _kimi_argv("kimi", "hi", "s1", "veystrix-ornith"))
        self.assertIn("--session", _pi_argv("pi", "hi", "s1", "veystrix-ornith"))
        self.assertIn("veystrix/veystrix-ornith", _pi_argv("pi", "hi", "s1", "veystrix-ornith"))
        self.assertIn("--resume", _hermes_argv("hermes", "hi", "s1", "veystrix-ornith"))
        # A prompt starting with '-' must never be parsed as an option.
        self.assertTrue(_claude_argv("claude", "-danger", None, None)[2].startswith(" "))

    def test_prepare_harness_pins_pi_and_headless_models(self):
        import tempfile
        from pathlib import Path
        from veystrix.tool import tui as T
        from veystrix.tool.headless import HeadlessCLIAdapter

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            # Pi auto-installs its extension and gets a non-None model id.
            pi_model = T.prepare_harness("pi", home, 8080, "veystrix-ornith-abc123")
            self.assertEqual(pi_model, "veystrix-ornith-abc123")
            ext = home / ".pi" / "agent" / "extensions" / "veystrix.ts"
            self.assertTrue(ext.is_file())
            self.assertIn("veystrix-ornith", ext.read_text())
            adapter = HeadlessCLIAdapter("pi", model=pi_model)
            self.assertIn("--model", adapter.argv("hello"))
            self.assertIn("veystrix/veystrix-ornith-abc123", adapter.argv("hello"))
            # HeadlessCLIAdapter.set_model updates env vars for claude-code / qwen-code.
            cc = HeadlessCLIAdapter("claude-code", env={"ANTHROPIC_MODEL": "old"}, model="old")
            cc.set_model("veystrix-flashnext")
            self.assertEqual(cc.model, "veystrix-flashnext")
            self.assertEqual(cc.env["ANTHROPIC_MODEL"], "veystrix-flashnext")


if __name__ == "__main__":
    unittest.main()
