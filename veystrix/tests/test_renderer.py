"""D6 offline golden tests for the Veystrix-owned renderer (no harness, no GPU)."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from veystrix.tool import renderer as R


class RendererTests(unittest.TestCase):
    def test_no_color_and_ascii_are_deterministic(self):
        with patch.dict("os.environ", {"NO_COLOR": "1", "VEYSTRIX_UI_ASCII": "1"}):
            theme = R.Theme.detect(tty=True)
        self.assertEqual(theme.color, "none")
        self.assertTrue(theme.ascii_only)
        line = R.status_line(R.Status(engine="cpp", model="flashnext", used=4096,
                                      window=32768, sampling="thinking", cache="cached",
                                      tok_s=1.4, ttft_s=2.3), theme, width=80)
        self.assertIn("VEYSTRIX", line)
        self.assertNotIn("cpp", line)
        self.assertIn("4k/32k", line)  # gauge always has a numeric label
        self.assertIn("~ cached", line)
        self.assertNotIn("\033", line)  # no ANSI escapes in no-color mode
        self.assertLessEqual(len(line), 80)

    def test_engine_is_hidden_between_python_and_cpp_frames(self):
        theme = R.Theme(color="none", ascii_only=True)
        base = dict(model="flashnext", used=1000, window=8192, sampling="thinking",
                    cache="cold", tok_s=1.0, ttft_s=1.0)
        py = R.status_line(R.Status(engine="python", **base), theme, width=120)
        cpp = R.status_line(R.Status(engine="cpp", **base), theme, width=120)
        self.assertEqual(py, cpp)
        self.assertNotIn("python", py)
        self.assertNotIn("cpp", cpp)

    def test_tool_block_has_line_numbers_and_diff_has_markers(self):
        theme = R.Theme(color="none", ascii_only=True)
        block = "\n".join(R.tool_block("edit_file", {"path": "app.py"},
                                       "line one\nline two", theme))
        self.assertIn("edit_file", block)
        self.assertIn("1 │", block)
        self.assertIn("2 │", block)
        diff = "\n".join(R.diff_block("--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new", theme))
        self.assertIn("+ new", diff)
        self.assertIn("- old", diff)
        self.assertIn("@ @@", diff)

    def test_stream_and_cancel(self):
        theme = R.Theme(color="none", ascii_only=False)
        first = R.stream_line("hello", theme, spinner=0)
        second = R.stream_line("hello", theme, spinner=1)
        self.assertNotEqual(first, second)  # spinner animates
        self.assertIn("cancelled", R.cancelled_line(theme))

    def test_sixty_four_color_fallback_has_no_truecolor_escape(self):
        with patch.dict("os.environ", {"TERM": "xterm-256color", "COLORTERM": ""}):
            os.environ.pop("NO_COLOR", None)
            theme = R.Theme.detect(tty=True)
        self.assertEqual(theme.color, "256")
        painted = theme.paint("brand", "X")
        self.assertIn("\033[38;5;", painted)
        self.assertNotIn("\033[38;2;", painted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
