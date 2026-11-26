"""D6 golden frames for the Veystrix-owned screen: 3 sizes x 4 themes x 4 cases, branding
suppression, tool/diff rendering, engine=python vs cpp (exactly one changed region).

Offline: recorded event streams replayed through the real translators and the real UILoop; no
harness binary and no model. Regenerate hashes after an intentional visual change with
``VEYSTRIX_UPDATE_GOLDEN=1 python -m unittest veystrix.tests.test_renderer_ui``.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest

from veystrix.tests.test_tui_fixtures import (GOLDEN, SIZES, THEMES, codex_events, make_loop,
                                              opencode_events, run_turn, scenario_adapter)
from veystrix.tool import renderer as R
from veystrix.tool.harness_adapters import ReplayAdapter

HASHES = GOLDEN / "frames.json"
CASES = ("main", "help", "palette", "permission", "welcome")
TEXT_GOLDENS = {("main", 80, 24, "ascii"): "main_80x24_ascii.txt",
                ("main", 120, 40, "no-color"): "main_120x40_no-color.txt",
                ("help", 80, 24, "no-color"): "help_80x24_no-color.txt",
                ("welcome", 80, 24, "ascii"): "welcome_80x24_ascii.txt",
                ("welcome", 120, 40, "no-color"): "welcome_120x40_no-color.txt",
                ("welcome", 200, 60, "no-color"): "welcome_200x60_no-color.txt"}


def welcome_frame(cols: int, rows: int, theme_name: str, engine: str = "cpp", **state) -> list[str]:
    """The home dashboard: no transcript yet, so the hero logo + dashboard card are shown."""
    loop, _, _, _ = make_loop(scenario_adapter(), cols=cols, rows=rows, theme=THEMES[theme_name],
                              engine=engine, cwd="/work/veystrix", harness="opencode")
    for key, value in state.items():
        setattr(loop, key, value)
    loop.step(0.0)
    return loop.last_frame


def case_frame(case: str, cols: int, rows: int, theme_name: str, engine: str = "cpp") -> list[str]:
    theme = THEMES[theme_name]
    if case == "welcome":
        return welcome_frame(cols, rows, theme_name, engine)
    if case == "permission":
        adapter = ReplayAdapter([opencode_events("opencode_permission.sse")])
        loop, mirror = run_turn(adapter, cols=cols, rows=rows, theme=theme, engine=engine)
        loop.step(99.0)
        return loop.last_frame
    loop, mirror = run_turn(scenario_adapter(), cols=cols, rows=rows, theme=theme, engine=engine)
    if case == "help":
        loop.step(100.0, ["?"])
    elif case == "palette":
        loop.step(100.0, ["ctrl-k", "r", "e"])
    return loop.last_frame



_HOME = None


def setUpModule():
    # Frames must not depend on the developer's ~/.veystrix (effort, thinking): use a blank home.
    global _HOME
    _HOME = tempfile.TemporaryDirectory()
    os.environ["VEYSTRIX_HOME"] = _HOME.name


def tearDownModule():
    os.environ.pop("VEYSTRIX_HOME", None)
    _HOME.cleanup()

class GoldenFrames(unittest.TestCase):
    updates: dict = {}

    @classmethod
    def setUpClass(cls):
        cls.golden = json.loads(HASHES.read_text()) if HASHES.exists() else {}

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("VEYSTRIX_UPDATE_GOLDEN") and cls.updates:
            HASHES.write_text(json.dumps(dict(sorted(cls.updates.items())), indent=2) + "\n")

    def test_frame_matrix_sizes_themes_cases(self):
        stripped: dict = {}
        for case in CASES:
            for cols, rows in SIZES:
                for theme_name in THEMES:
                    key = f"{case}/{cols}x{rows}/{theme_name}"
                    with self.subTest(key):
                        frame = case_frame(case, cols, rows, theme_name)
                        self._invariants(frame, cols, rows, theme_name)
                        digest = hashlib.sha256("\n".join(frame).encode()).hexdigest()
                        self.updates[key] = digest
                        if not os.environ.get("VEYSTRIX_UPDATE_GOLDEN"):
                            self.assertEqual(digest, self.golden.get(key), f"golden drift in {key}")
                        text_name = TEXT_GOLDENS.get((case, cols, rows, theme_name))
                        if text_name:
                            text = "\n".join(line.rstrip() for line in frame) + "\n"
                            path = GOLDEN / text_name
                            if os.environ.get("VEYSTRIX_UPDATE_GOLDEN"):
                                path.write_text(text)
                            self.assertEqual(text, path.read_text(), f"text golden drift {text_name}")
                        stripped[(case, cols, rows, theme_name)] = [R.strip_ansi(x) for x in frame]
        # dark and light differ ONLY in colour: identical text, different escapes
        for case in CASES:
            for cols, rows in SIZES:
                self.assertEqual(stripped[(case, cols, rows, "dark")], stripped[(case, cols, rows, "light")])
                self.assertEqual(stripped[(case, cols, rows, "dark")], stripped[(case, cols, rows, "no-color")])
        self.assertEqual(len(self.updates), len(CASES) * len(SIZES) * len(THEMES))

    def _invariants(self, frame, cols, rows, theme_name):
        self.assertEqual(len(frame), rows)
        for line in frame:
            self.assertEqual(R.visible_len(line), cols, line)
        joined = "\n".join(frame)
        if theme_name in ("no-color", "ascii"):
            self.assertNotIn("\033", joined)
        if theme_name == "ascii":
            self.assertTrue(joined.isascii(), [c for c in joined if ord(c) > 127][:5])
        else:
            self.assertFalse(theme_name == "dark" and "\033[38;2;" not in joined)
        footer = R.strip_ansi(frame[-1])
        self.assertIn("VEYSTRIX", footer)
        for word in ("cpp", "python"):        # one engine to the user: the runtime is never shown
            self.assertNotIn(word, footer)
        self.assertIn("medium", footer)       # reasoning effort never dropped
        self.assertIn("therm", footer)        # safety state never dropped
        self.assertRegex(footer, r"[\d.]+[kM]?/32k")  # numeric gauge label, never colour only


class BrandingSuppression(unittest.TestCase):
    FORBIDDEN = ("opencode", "codex", "claude code", "update available", "9.9.9", "sign in",
                 "telemetry", "anthropic account")

    def _screen(self, events):
        loop, mirror = run_turn(ReplayAdapter([events]), cols=120, rows=40, theme=THEMES["no-color"])
        return "\n".join(R.strip_ansi(x) for x in loop.last_frame).lower()

    def test_harness_surface_events_never_reach_the_screen(self):
        # the recorded stream contains installation.update-available and a harness toast
        raw = (GOLDEN / "opencode_turn.sse").read_text()
        self.assertIn("update-available", raw)
        self.assertIn("OpenCode 9.9.9", raw)
        screen = self._screen(opencode_events())
        for word in self.FORBIDDEN:
            self.assertNotIn(word, screen)
        self.assertNotIn("leak from another session", screen)   # other-session events dropped
        self.assertIn("veystrix", screen)

    def test_harness_error_text_is_scrubbed(self):
        screen = self._screen(opencode_events("opencode_error.sse"))
        for word in self.FORBIDDEN:
            self.assertNotIn(word, screen)
        self.assertIn("the harness reported an error", screen)

    def test_codex_stream_renders_the_same_screen_contract(self):
        screen = self._screen(codex_events())
        self.assertIn("python -m unittest", screen)
        self.assertIn("ran 3 tests", screen)
        self.assertIn("all tests pass now.", screen)
        for word in self.FORBIDDEN:
            self.assertNotIn(word, screen)


class ToolAndDiffRendering(unittest.TestCase):
    def setUp(self):
        loop, _ = run_turn(scenario_adapter(), cols=120, rows=40, theme=THEMES["ascii"])
        self.loop = loop
        self.text = "\n".join(loop.last_frame)

    def test_tool_blocks_have_status_marker_and_line_numbers(self):
        self.assertIn("> read  path=app.py  +", self.text)
        self.assertRegex(self.text, r"1 \| def total")
        self.assertIn("> edit  path=app.py  +", self.text)

    def test_diff_is_rendered_once_with_markers(self):
        self.assertRegex(self.text, r"(?m)^  \+ +return sum\(xs\) *$")
        self.assertRegex(self.text, r"(?m)^  - +return sum\(xs\[:-1\]\) *$")
        self.assertIn("@ @@ -1,2 +1,2 @@", self.text)
        self.assertEqual(self.text.count("+++ b/app.py"), 1)

    def test_reasoning_shown_by_default_and_collapsible(self):
        self.assertIn("Checking the failing test first.", self.text)   # thinking is visible
        self.assertIn("then patch the off-by-one.", self.text)
        self.loop.step(50.0, ["ctrl-r"])
        hidden = "\n".join(self.loop.last_frame)
        self.assertIn("reasoning (2 lines", hidden)
        self.assertNotIn("Checking the failing test first.", hidden)
        self.loop.step(50.5, ["ctrl-r"])
        self.assertIn("Checking the failing test first.", "\n".join(self.loop.last_frame))

    def test_tool_output_collapses_and_expands(self):
        self.loop.step(51.0, ["ctrl-t"])
        collapsed = "\n".join(self.loop.last_frame)
        self.assertIn("(+2 lines)", collapsed)
        self.assertNotIn("1 | def total", collapsed)
        self.loop.step(52.0, ["ctrl-t"])
        self.assertIn("1 | def total", "\n".join(self.loop.last_frame))

    def test_long_tool_output_is_bounded(self):
        from veystrix.tool.harness_adapters import ev
        events = [ev("tool.start", id="b", name="bash", args={"command": "seq 100"}),
                  ev("tool.output", id="b", text="\n".join(str(i) for i in range(100))),
                  ev("tool.end", id="b", ok=True), ev("turn.end", status="ok")]
        loop, _ = run_turn(ReplayAdapter([events]), cols=120, rows=40, theme=THEMES["ascii"])
        text = "\n".join(loop.last_frame)
        self.assertIn("(+100 lines)", text)       # auto-collapsed past TOOL_OUTPUT_LINES
        loop.step(60.0, ["ctrl-t"])
        text = "\n".join(loop.last_frame)
        self.assertIn("88 earlier lines", text)
        self.assertEqual(len(loop.last_frame), 40)


class EngineIndicator(unittest.TestCase):
    """Veystrix is one engine to its users: which runtime serves a model is never on screen."""

    def test_python_and_cpp_render_identically(self):
        for case in CASES:
            for cols, rows in SIZES:
                for theme_name in THEMES:
                    with self.subTest(f"{case}/{cols}x{rows}/{theme_name}"):
                        py = case_frame(case, cols, rows, theme_name, engine="python")
                        cpp = case_frame(case, cols, rows, theme_name, engine="cpp")
                        self.assertEqual(py, cpp)
                        footer = R.strip_ansi(cpp[-1])
                        for word in ("python", "cpp", "C++"):
                            self.assertNotIn(word, footer)

    def test_status_line_never_exceeds_width_and_keeps_safety_fields(self):
        for width in (80, 100, 120, 200):
            for engine in ("python", "cpp"):
                line = R.status_line(R.Status(engine=engine, model="Qwen3-Next-80B-A3B", used=123456,
                                              window=262144, cache="cached", tok_s=12.3, ttft_s=1.2,
                                              thermal=2, effort="medium"), R.Theme(color="none"), width=width)
                self.assertLessEqual(len(line), width)
                for needle in ("123.5k/256k", "therm", "cached", "medium"):
                    self.assertIn(needle, line)
                self.assertNotIn(engine, line)


class HelpOverlay(unittest.TestCase):
    def test_help_text_fits_the_80_column_box(self):
        for line in R.HELP_LINES:
            self.assertLessEqual(len(line), 68, line)
        frame = case_frame("help", 80, 24, "no-color")
        text = "\n".join(frame)
        for needle in ("Veystrix help", "Ctrl-K", "/resume", "allow once"):
            self.assertIn(needle, text)


class BoundedRedraw(unittest.TestCase):
    def test_diff_rows_and_paint_are_minimal(self):
        a = ["x" * 80] * 24
        b = list(a)
        b[10] = "y" * 80
        self.assertEqual([row for row, _ in R.diff_rows(a, b)], [10])
        self.assertEqual(R.diff_rows(a, a), [])
        self.assertEqual(len(R.diff_rows(None, b)), 24)
        self.assertEqual(len(R.diff_rows(a, b[:20])), 20)        # resized: repaint everything
        bytes_ = R.paint_rows(R.diff_rows(a, b))
        self.assertEqual(bytes_.count(";1H"), 1)
        self.assertLess(len(bytes_), 200)

    def test_too_small_terminal_gets_a_clear_message_not_a_crash(self):
        state = R.UIState(cols=60, rows=10)
        frame = R.compose_frame(state, R.Theme(color="none", ascii_only=True), 60, 10)
        self.assertEqual(len(frame), 10)
        self.assertTrue(any("terminal too small" in line for line in frame))

    def test_svg_and_asciicast_helpers(self):
        frame = case_frame("main", 80, 24, "dark")
        svg = R.frame_to_svg(frame, dark=True)
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("VEYSTRIX", svg)
        self.assertIn("#7c5cff", svg)         # brand colour survived the ANSI -> SVG conversion
        cast = R.asciicast([(0.0, "a"), (0.5, "b")], 80, 24).splitlines()
        self.assertEqual(json.loads(cast[0])["version"], 2)
        self.assertEqual(json.loads(cast[2]), [0.5, "o", "b"])


class MarkdownRendering(unittest.TestCase):
    def test_fenced_code_blocks_headings_and_inline_styles_render_cleanly(self):
        md = "## Fix summary\nUse `sum(xs)` for **all** items:\n```python\ndef total(xs):\n    return sum(xs)\n```\n"
        for theme_name in ("dark", "no-color", "ascii"):
            with self.subTest(theme_name):
                theme = THEMES[theme_name]
                state = R.UIState(cols=100, rows=30)
                state.add(R.Entry("assistant", text=md))
                frame = R.compose_frame(state, theme, 100, 30)
                self.assertEqual(len(frame), 30)
                self.assertTrue(all(R.visible_len(line) == 100 for line in frame))
                joined = "\n".join(R.strip_ansi(line) for line in frame)
                self.assertIn("Fix summary", joined)
                self.assertIn("python", joined)
                self.assertIn("return sum(xs)", joined)
                if theme_name == "ascii":
                    self.assertTrue("\n".join(frame).isascii())

    def test_code_frame_bullets_and_highlighting(self):
        md = ("# Plan\n- first **bold** item\n- second `x = 1`\n1. numbered\n> quoted\n"
              "```python\nfor i in range(3):  # loop\n    print('hi', i)\n```\ndone")
        frames = {}
        for theme_name in THEMES:
            state = R.UIState(cols=100, rows=30)
            state.add(R.Entry("assistant", text=md))
            frames[theme_name] = R.compose_frame(state, THEMES[theme_name], 100, 30)
            top, bottom, left, right = box_columns(frames[theme_name], "python")
            self.assertEqual(bottom - top, 3)                         # two code lines, framed
        plain = "\n".join(frames["no-color"])
        for needle in ("Plan", "• first bold item", "• second `x = 1`", "1. numbered", "│ quoted",
                       "for i in range(3):  # loop", "print('hi', i)", "done"):
            self.assertIn(needle, plain)
        self.assertNotIn("**", plain)
        self.assertNotIn("```", plain)
        self.assertIn("* first bold item", "\n".join(frames["ascii"]))
        dark = "\n".join(frames["dark"])
        self.assertIn("\033[1m", dark)                                # bold weight in colour
        self.assertEqual([R.strip_ansi(x) for x in frames["dark"]], frames["no-color"])
        keyword = R.highlight_code("for x in y:  # c", THEMES["dark"])
        self.assertNotEqual(keyword, "for x in y:  # c")
        self.assertEqual(R.strip_ansi(keyword), "for x in y:  # c")
        self.assertEqual(R.highlight_code("for x", THEMES["no-color"]), "for x")


def box_columns(frame: list[str], top_marker: str) -> tuple[int, int, int, int]:
    """(top row, bottom row, left col, right col) of the rounded box whose top border contains
    ``top_marker``; asserts every row between has its vertical edges in exactly those columns."""
    text = [R.strip_ansi(line) for line in frame]
    for top, row in enumerate(text):
        for tl, tr in (("╭", "╮"), ("+", "+")):
            if tl in row and top_marker in row[row.index(tl):] and row.rstrip().endswith(tr):
                left, right = row.index(tl), row.rindex(tr)
                bottom = next(i for i in range(top + 1, len(text))
                              if len(text[i]) > right and text[i][left] in "+╰" and text[i][right] in "+╯")
                for i in range(top + 1, bottom):
                    assert text[i][left] in "|│" and text[i][right] in "|│", (i, text[i])
                return top, bottom, left, right
    raise AssertionError(f"no box titled {top_marker!r}")


class WelcomeDashboard(unittest.TestCase):
    NEEDLES = ("Home", "Engine", "stopped", "Model", "flashnext", "Effort", "medium", "Thinking", "shown",
               "KV mode", "auto", "Harness",
               "OpenCode", "Folder", "/work/veystrix", "/settings", "/model", "/download", "/start",
               "@file Tab", "Ctrl-K", "?", "local coding agent")

    def test_hero_logo_and_dashboard_at_every_size_and_theme(self):
        for cols, rows in SIZES:
            for theme_name in THEMES:
                with self.subTest(f"{cols}x{rows}/{theme_name}"):
                    frame = welcome_frame(cols, rows, theme_name)
                    text = [R.strip_ansi(line) for line in frame]
                    joined = "\n".join(text)
                    self.assertEqual(len(frame), rows)
                    self.assertTrue(all(R.visible_len(line) == cols for line in frame))
                    for needle in self.NEEDLES:
                        self.assertIn(needle, joined)
                    ascii_only = theme_name == "ascii"
                    if ascii_only:
                        self.assertTrue("\n".join(frame).isascii())
                        logo = [i for i, line in enumerate(text) if line.count("#") >= 12]
                        self.assertEqual(len(logo), 5)
                    else:
                        logo = [i for i, line in enumerate(text) if "██" in line or "╚═╝" in line]
                        self.assertEqual(len(logo), 6)
                        self.assertIn("██╗   ██╗███████╗██╗   ██╗███████╗████████╗██████╗ ██╗██╗  ██╗",
                                      text[logo[0]])
                    self.assertEqual(logo, list(range(logo[0], logo[0] + len(logo))))   # contiguous
                    # centred: equal margins (to within one column) on both sides of the mark
                    width = R.logo_width(THEMES[theme_name])
                    left = len(text[logo[1]]) - len(text[logo[1]].lstrip())
                    self.assertLessEqual(abs(left - (cols - width - left)), 1)
                    top, bottom, l, r = box_columns(frame, "Home")
                    self.assertGreater(top, logo[-1])                 # card sits under the mark
                    self.assertLessEqual(abs(l - (cols - 1 - r)), 1)  # card centred too
                    self.assertIn("type a message", text[bottom])
                    self.assertLess(bottom, rows - 5)                 # composer + bottom status untouched
                    self.assertIn("❯" if not ascii_only else ">", text[-4])

    def test_text_is_identical_across_dark_light_and_no_color(self):
        for cols, rows in SIZES:
            plain = [welcome_frame(cols, rows, name) for name in ("dark", "light", "no-color")]
            stripped = [[R.strip_ansi(x) for x in frame] for frame in plain]
            self.assertEqual(stripped[0], stripped[1])
            self.assertEqual(stripped[0], stripped[2])
            self.assertNotEqual(plain[0], plain[1])                   # but the colours differ

    def test_engine_status_pill(self):
        for kwargs, needle in (({"_engine_running": True}, "● ready"),
                               ({"_engine_running": False}, "○ stopped"),
                               ({"_needs_port_check": True}, "▲ needs port-check"),
                               ({"_bg_busy": True, "_bg_action": "start"}, "◐ starting")):
            with self.subTest(needle):
                text = "\n".join(R.strip_ansi(x) for x in welcome_frame(120, 40, "dark", **kwargs))
                self.assertIn(needle, text)
        ascii_text = "\n".join(welcome_frame(80, 24, "ascii", _needs_port_check=True))
        self.assertIn("! needs port-check", ascii_text)

    def test_logo_glyphs_are_rectangular(self):
        for ch, rows_ in R._LOGO_GLYPHS.items():
            self.assertEqual(len(rows_), 6, ch)
            self.assertEqual(len({R.visible_len(row) for row in rows_}), 1, ch)
        self.assertEqual(R.logo_width(THEMES["dark"]), 62)
        self.assertTrue(all(line.isascii() for line in R.hero_logo(THEMES["ascii"])))

    def test_overlay_replaces_the_dashboard_instead_of_bleeding_through(self):
        loop, _, _, _ = make_loop(scenario_adapter(), cols=80, rows=24, theme=THEMES["no-color"])
        loop.step(0.0, ["?"])
        text = "\n".join(R.strip_ansi(x) for x in loop.last_frame)
        self.assertIn("Veystrix help", text)
        self.assertNotIn("Home", text)
        self.assertNotIn("██", text)


class ChatHeader(unittest.TestCase):
    def test_logo_heads_a_new_chat_and_scrolls_away(self):
        for theme_name in THEMES:
            with self.subTest(theme_name):
                state = R.UIState(cols=120, rows=40)
                state.add(R.Entry("user", text="first prompt"))
                text = [R.strip_ansi(x) for x in R.compose_frame(state, THEMES[theme_name], 120, 40)]
                mark = "#######" if theme_name == "ascii" else "███████╗"
                logo_rows = [i for i, line in enumerate(text) if mark in line]
                prompt_row = next(i for i, line in enumerate(text) if "first prompt" in line)
                self.assertTrue(logo_rows)
                self.assertLess(max(logo_rows), prompt_row)           # above the first message
                self.assertNotIn("Home", "\n".join(text))             # the dashboard card is gone
                for i in range(60):
                    state.add(R.Entry("system", text=f"line {i}"))
                text = "\n".join(R.strip_ansi(x) for x in R.compose_frame(state, THEMES[theme_name], 120, 40))
                self.assertNotIn(mark, text)                          # scrolled away with the chat
                state.scroll = 10_000                                 # PgUp to the top: it is still there
                text = "\n".join(R.strip_ansi(x) for x in R.compose_frame(state, THEMES[theme_name], 120, 40))
                self.assertIn(mark, text)


class ComposerAndOverlays(unittest.TestCase):
    def test_busy_rail_shows_spinner_activity_and_elapsed_time(self):
        state = R.UIState(cols=100, rows=30, busy=True, activity="responding", turn_started=10.0, now=73.0)
        state.add(R.Entry("user", text="hi"))
        frame = [R.strip_ansi(x) for x in R.compose_frame(state, THEMES["no-color"], 100, 30)]
        self.assertIn("responding · 1m 03s", frame[-5])
        self.assertIn("Ctrl-C cancel", frame[-3])
        self.assertTrue(frame[-5].lstrip().startswith("╭─") and frame[-5].rstrip().endswith("╮"))
        self.assertIn("VEYSTRIX", frame[-1])

    def test_hints_follow_what_is_being_typed(self):
        for text, needle in (("/se", "Tab complete"), ("look at @ap", "Tab attach file"),
                             ("hello", "Shift+Enter newline"), ("", "? help")):
            state = R.UIState(cols=120, rows=30, input=text)
            state.add(R.Entry("user", text="x"))
            self.assertIn(needle, R.strip_ansi(R.compose_frame(state, THEMES["no-color"], 120, 30)[-3]))

    def test_bottom_bar_shows_prefill_progress_and_decode_speed(self):
        st_pf = R.Status(model="Ornith 1.5", used=4096, window=262144, effort="medium",
                         stage="prefill", prefill_done=1720, prefill_total=4096)
        for width in (80, 120, 200):
            line = R.strip_ansi(R.status_line(st_pf, THEMES["no-color"], width=width))
            self.assertLessEqual(len(line), width)
            self.assertIn("42%", line)
            self.assertIn("4k", line)
            self.assertNotIn("tok/s", line)
        st_dec = R.Status(model="Ornith 1.5", used=4096, window=262144, effort="medium",
                          stage="decode", tok_s=15.3)
        for width in (80, 120, 200):
            line = R.strip_ansi(R.status_line(st_dec, THEMES["no-color"], width=width))
            self.assertLessEqual(len(line), width)
            self.assertIn("15.3", line)
            self.assertNotIn("prefill", line)

    def test_selection_band_is_colour_only(self):
        state = R.UIState(cols=100, rows=30, overlay="palette")
        colour = R.compose_frame(state, THEMES["dark"], 100, 30)
        plain = R.compose_frame(state, THEMES["no-color"], 100, 30)
        self.assertTrue(any("\033[48;2;" in line for line in colour))
        self.assertEqual([R.strip_ansi(x) for x in colour], plain)
        self.assertTrue(any("▸ Settings" in line for line in plain))   # marker, not colour alone


if __name__ == "__main__":
    unittest.main(verbosity=2)
