"""Clients that send more than one system message (21 Nov 262k harness matrix: Codex and Claude Code got
"System message must be at the beginning" from the Qwen-family chat templates)."""
from __future__ import annotations

import unittest
from pathlib import Path

from veystrix.server import _anthropic_messages, _normalize_messages, _responses_messages


class SystemFoldingTests(unittest.TestCase):
    def test_leading_system_and_developer_messages_are_joined(self):
        out = _normalize_messages([{"role": "system", "content": "A"}, {"role": "developer", "content": "B"},
                                   {"role": "user", "content": "hi"}])
        self.assertEqual(out, [{"role": "system", "content": "A\n\nB"}, {"role": "user", "content": "hi"}])

    def test_later_system_message_becomes_user_in_place(self):
        out = _normalize_messages([{"role": "system", "content": "S"}, {"role": "user", "content": "u1"},
                                   {"role": "assistant", "content": "a1"}, {"role": "system", "content": "env"},
                                   {"role": "user", "content": "u2"}])
        self.assertEqual([m["role"] for m in out], ["system", "user", "assistant", "user", "user"])
        self.assertEqual(out[3]["content"], "env")

    def test_claude_code_system_after_user(self):
        # Claude Code (Nov 2026): top-level system plus a role "system" message after the first user turn.
        out = _anthropic_messages({"system": [{"type": "text", "text": "prompt"}], "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "system", "content": [{"type": "text", "text": "# Environment"}]}]})
        self.assertEqual([m["role"] for m in out], ["system", "user", "user"])

    def test_codex_instructions_plus_developer(self):
        out = _responses_messages({"instructions": "I", "input": [
            {"type": "message", "role": "developer", "content": "D"},
            {"type": "message", "role": "user", "content": "hi"}]})
        self.assertEqual(out[0], {"role": "system", "content": "I\n\nD"})
        self.assertEqual(len(out), 2)

    def test_qwen_template_renders_folded_messages(self):
        template = Path("/Users/vincent/models/Ornith-1.5-35B-A3B-MLX-4bit/chat_template.jinja")
        if not template.exists():
            self.skipTest("Ornith checkpoint not installed")
        from jinja2 import Environment

        def fail(msg):
            raise ValueError(msg)
        env = Environment()
        env.globals["raise_exception"] = fail
        out = _normalize_messages([{"role": "user", "content": "hi"}, {"role": "system", "content": "env"},
                                   {"role": "developer", "content": "d"}])
        text = env.from_string(template.read_text()).render(messages=out, add_generation_prompt=True)
        self.assertIn("env", text)


if __name__ == "__main__":
    unittest.main()
