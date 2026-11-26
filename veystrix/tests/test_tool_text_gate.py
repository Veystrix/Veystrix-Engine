"""Text streams while a request declares tools, and a tool call never leaks into it (VEYSTRIX_STREAM_TEXT_WITH_TOOLS)."""
import os
import unittest
from unittest.mock import patch

from veystrix.server import STREAM_TEXT_WITH_TOOLS_ENV, _parse_tool_calls, _stream_text_with_tools
from veystrix.text import CompletedTextBuffer, ToolCallTextGate

TOOL = {"type": "function", "function": {"name": "list_files", "description": "", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
CALL = '<tool_call>{"name": "list_files", "arguments": {"path": "/tmp"}}</tool_call>'


def stream(raw: str, size: int):
    """What the server does: words through the buffer, the gate, then the finished reply's remainder."""
    gate, words, sent = ToolCallTextGate(), CompletedTextBuffer(), []
    for start in range(0, len(raw), size):
        for word in words.feed(raw[start:start + size]):
            released = gate.feed(word)
            if released:
                sent.append(released)
    for word in words.finish():
        released = gate.feed(word)
        if released:
            sent.append(released)
    visible, calls = _parse_tool_calls(raw, [TOOL])
    rest = gate.remainder(visible) if visible else ""
    return sent, rest, visible, calls, gate


class ToolCallTextGateTests(unittest.TestCase):
    def test_plain_text_streams_in_pieces_and_adds_up(self):
        raw = "Hello there. Here is a longer answer, in several sentences.\nIt has a second line."
        for size in (1, 2, 3, 5, 11, 1000):
            sent, rest, visible, calls, gate = stream(raw, size)
            self.assertEqual("".join(sent) + rest, visible)
            self.assertEqual(visible, raw)
            self.assertEqual(calls, [])
            if size <= 11:
                self.assertGreater(len(sent), 3, "text must arrive before the end")

    def test_a_tool_call_never_reaches_the_client_as_text(self):
        raw = "I will look at the folder.\n\n" + CALL
        for size in (1, 2, 4, 7, 1000):
            sent, rest, visible, calls, gate = stream(raw, size)
            joined = "".join(sent) + rest
            self.assertNotIn("<tool_call", joined)
            self.assertNotIn("list_files", joined)
            self.assertEqual(joined, "I will look at the folder.")
            self.assertEqual([c["name"] for c in calls], ["list_files"])
            self.assertTrue(gate.blocked)

    def test_text_after_a_tool_call_is_sent_at_the_end(self):
        raw = "First part.\n" + CALL + "\nSecond part."
        sent, rest, visible, calls, gate = stream(raw, 3)
        self.assertEqual("".join(sent), "First part.")
        self.assertEqual("".join(sent) + rest, visible)
        self.assertIn("Second part.", rest)
        self.assertNotIn("<tool_call", visible)

    def test_marker_split_across_pieces_is_held_back(self):
        gate = ToolCallTextGate()
        self.assertEqual(gate.feed("Sure <tool_"), "Sure")
        self.assertEqual(gate.feed('call>{"name": "x"}'), "")
        self.assertTrue(gate.blocked)
        self.assertEqual(gate.feed("more"), "")

    def test_a_lookalike_that_is_not_a_marker_is_released_later(self):
        gate = ToolCallTextGate()
        self.assertEqual(gate.feed("use the <tool"), "use the")
        self.assertEqual(gate.feed(" box"), " <tool box")      # the space between the words goes out with the next piece
        self.assertFalse(gate.blocked)
        self.assertEqual(gate.sent, "use the <tool box")

    def test_whitespace_matches_the_stripped_final_text(self):
        sent, rest, visible, calls, gate = stream("\n\n  Hello world  \n", 2)
        self.assertEqual("".join(sent) + rest, visible)
        self.assertEqual(visible, "Hello world")

    def test_only_a_tool_call_sends_no_text(self):
        sent, rest, visible, calls, gate = stream(CALL, 5)
        self.assertEqual(sent, [])
        self.assertEqual(rest, "")
        self.assertEqual(len(calls), 1)

    def test_a_disagreeing_final_text_is_reported_not_duplicated(self):
        gate = ToolCallTextGate()
        gate.feed("Hello wor")
        gate.feed("ld ")
        self.assertEqual(gate.remainder("Hello there"), "there")
        self.assertTrue(gate.mismatched)

    def test_flag_default_on_and_zero_restores_holding(self):
        env = {k: v for k, v in os.environ.items() if k != STREAM_TEXT_WITH_TOOLS_ENV}
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(_stream_text_with_tools())
        for off in ("0", "false", "off", "NO"):
            with patch.dict(os.environ, {STREAM_TEXT_WITH_TOOLS_ENV: off}):
                self.assertFalse(_stream_text_with_tools())


if __name__ == "__main__":
    unittest.main()
