"""Inkling 2 Mini's response format and the swap-safety regressions behind it."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from veystrix import kv_policy
from veystrix.text import (
    InklingSegmentFilter,
    ReasoningStreamFilter,
    ThinkingStreamFilter,
    extract_inkling_reasoning_and_content,
    extract_reasoning_and_content,
    response_format_for,
)

INKLING_REPLY = (
    "<|message_model|><|content_thinking|>Let me reason first.<|end_message|>"
    "<|message_model|><|content_text|>THE ANSWER<|end_message|>"
    "<|content_model_end_sampling|>"
)


class InklingFormatTests(unittest.TestCase):
    def test_format_detection(self):
        self.assertEqual(response_format_for("inkling2-mini"), "inkling")
        self.assertEqual(response_format_for("veystrix-inkling2-mini"), "inkling")
        self.assertEqual(response_format_for("flashnext"), "think")
        self.assertEqual(response_format_for(None), "think")

    def test_whole_text_puts_answer_in_content_not_reasoning(self):
        reasoning, content = extract_reasoning_and_content(INKLING_REPLY, True, format="inkling")
        self.assertEqual(reasoning, "Let me reason first.")
        self.assertEqual(content, "THE ANSWER")
        self.assertNotIn("<|", content)

    def test_no_thinking_leaves_answer_in_content(self):
        reasoning, content = extract_inkling_reasoning_and_content(INKLING_REPLY, False)
        self.assertEqual(content, "THE ANSWER")

    def test_stream_splits_markers_across_pieces(self):
        stream = ReasoningStreamFilter(True, format="inkling")
        pieces = [
            "<|message_model|><|content_th", "inking|>Let me ",
            "reason first.<|end_message|><|message_model|><|content_text|>THE ",
            "ANSWER<|end_message|>",
        ]
        merged: list[tuple[str, str]] = []
        items = [item for piece in pieces for item in stream.feed(piece)] + stream.finish()
        for kind, text in items:
            if merged and merged[-1][0] == kind:
                merged[-1] = (kind, merged[-1][1] + text)
            elif text:
                merged.append((kind, text))
        self.assertEqual(merged, [("thinking", "Let me reason first."), ("content", "THE ANSWER")])

    def test_thinking_filter_hides_reasoning_but_shows_answer(self):
        shown = ThinkingStreamFilter(True, False, format="inkling")
        visible = "".join(shown.feed(piece) for piece in
                          ["<|content_thinking|>secret<|end_message|>", "<|content_text|>answer<|end_message|>"])
        self.assertEqual(visible, "answer")

    def test_tool_payload_parsing(self):
        from veystrix.server import _parse_tool_calls

        body = '{"name":"send_message","args":{"to":"team","text":"hi"}}'
        visible, calls = _parse_tool_calls(body, None, format="inkling")
        self.assertEqual(visible, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "send_message")
        self.assertEqual(calls[0]["arguments"], '{"to": "team", "text": "hi"}')

    def test_segment_filter_never_leaks_a_partial_marker(self):
        stream = InklingSegmentFilter(True)
        emitted = [text for piece in ["<|content_text|>hi<|end_mes", "sage|>"] for _, text in stream.feed(piece)]
        self.assertEqual("".join(emitted), "hi")

    def test_unprefixed_reply_defaults_to_content_not_reasoning(self):
        from veystrix.text import inkling_initial_kind

        self.assertEqual(inkling_initial_kind("<|message_model|>"), "content")
        self.assertEqual(inkling_initial_kind("<|message_model|><|content_thinking|>"), "thinking")
        self.assertEqual(inkling_initial_kind("<|message_model|><|content_invoke_tool_json|>"), "tool")

        stream = ReasoningStreamFilter(True, format="inkling", initial_kind=inkling_initial_kind("<|message_model|>"))
        items = stream.feed("Hello directly!") + stream.finish()
        self.assertEqual(items, [("content", "Hello directly!")])

        reasoning, content = extract_reasoning_and_content("Hello directly!", True, format="inkling", initial_kind="content")
        self.assertEqual(reasoning, "")
        self.assertEqual(content, "Hello directly!")

    def test_tool_json_block_never_streams_as_content_or_reasoning(self):
        from veystrix.text import extract_inkling_parts
        from veystrix.server import _parse_tool_calls

        raw = (
            "<|message_model|><|content_thinking|>Need to read file.<|end_message|>"
            "<|message_model|><|content_invoke_tool_json|>{\"name\": \"read_file\", \"args\": {\"path\": \"a.py\"}}<|end_message|>"
            "<|content_model_end_sampling|>"
        )
        stream = ReasoningStreamFilter(True, format="inkling", initial_kind="content")
        items = stream.feed(raw) + stream.finish()
        streamed_content = "".join(t for k, t in items if k == "content")
        streamed_thinking = "".join(t for k, t in items if k == "thinking")
        self.assertEqual(streamed_content, "")
        self.assertEqual(streamed_thinking, "Need to read file.")

        reasoning, visible, tool_blocks = extract_inkling_parts(raw, True, initial_kind="content")
        self.assertEqual(reasoning, "Need to read file.")
        self.assertEqual(visible, "")
        self.assertEqual(len(tool_blocks), 1)
        clean, calls = _parse_tool_calls(visible, None, format="inkling", tool_blocks=tool_blocks)
        self.assertEqual(clean, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read_file")
        self.assertEqual(calls[0]["arguments"], '{"path": "a.py"}')

    def test_pi_extension_includes_inkling2_mini_reasoning(self):
        from veystrix.pi_extension import EXTENSION_SOURCE

        self.assertIn("veystrix-inkling2-mini", EXTENSION_SOURCE)
        self.assertIn('m.family === "inkling2-mini"', EXTENSION_SOURCE)


class SwapSafetyTests(unittest.TestCase):
    def test_backend_context_limit_is_not_a_prefill_limit(self):
        from veystrix.session import _apply_backend_limits

        class Backend:
            context_limit = 1_048_576

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VEYSTRIX_MAX_COLD_PREFILL", None)
            _apply_backend_limits(Backend())
            self.assertEqual(os.environ["VEYSTRIX_MAX_COLD_PREFILL"], "32768")
            self.assertEqual(os.environ["VEYSTRIX_MAX_INCREMENTAL_PREFILL"], "32768")

    def test_explicit_kv_mode_is_clamped_to_budget(self):
        geometry = kv_policy.KVGeometry(8, 8, 128)
        huge_context = 1_048_576
        plan = kv_policy.plan(geometry, huge_context, 2 * kv_policy.GIB, requested="int8")
        self.assertNotEqual(plan.mode, "int8")
        self.assertLessEqual(plan.bytes_at_context, 2 * kv_policy.GIB)
        self.assertIn("int8 needs", plan.reason)

    def test_requested_kv_mode_that_fits_is_kept(self):
        geometry = kv_policy.KVGeometry(8, 8, 128)
        plan = kv_policy.plan(geometry, 4096, 4 * kv_policy.GIB, requested="int8")
        self.assertEqual(plan.mode, "int8")
        self.assertEqual(plan.reason, "requested")

    def test_guard_stop_is_terminal_with_backoff(self):
        from veystrix.tool import swap

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            now = 1_000_000.0
            with patch.object(swap.time, "time", return_value=now), \
                 patch.object(swap, "health", return_value={"status": "fault", "ready": False}), \
                 patch.object(swap, "status", return_value={"status": "guard_stopped"}), \
                 patch.object(swap, "logs", return_value="safety guard stopped"), \
                 patch.object(swap, "start") as started:
                with self.assertRaises(ValueError) as first:
                    swap.ensure_engine(home, port=1)
                self.assertIn("memory safety guard", str(first.exception))
                with self.assertRaises(ValueError) as second:
                    swap.ensure_engine(home, port=1)
                self.assertIn("retry in", str(second.exception))
                started.assert_not_called()

    def test_status_poller_clears_latest_after_engine_stops(self):
        from veystrix.tool.tui import StatusPoller

        responses = [{"engine": "cpp", "model": "inkling2-mini"}, None, None]
        it = iter(responses)
        poller = StatusPoller(lambda: next(it, None), interval=10.0, max_misses=2)
        poller._poll_once()
        self.assertIsNotNone(poller())
        # One transient miss keeps the last reading (no flicker).
        poller._poll_once()
        self.assertIsNotNone(poller())
        # Second consecutive miss clears latest so UILoop flips _engine_running to False.
        poller._poll_once()
        self.assertIsNone(poller())

    def test_friendly_url_error_replaces_raw_connection_refused(self):
        import urllib.error
        from veystrix.tool.harness_adapters import friendly_url_error

        err = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
        msg_engine = friendly_url_error(err, target="engine", port=8080)
        self.assertIn("engine is not running on 127.0.0.1:8080", msg_engine)
        self.assertIn("/start", msg_engine)
        self.assertNotIn("urlopen", msg_engine)

        msg_harness = friendly_url_error(err, target="harness", log_hint=" (see log)")
        self.assertIn("harness server is not reachable", msg_harness)
        self.assertNotIn("urlopen", msg_harness)

    def test_uiloop_detects_stopped_engine_on_poll(self):
        from veystrix.tool.harness_adapters import ReplayAdapter
        from veystrix.tool.tui import UILoop

        with tempfile.TemporaryDirectory() as tmp:
            ui = UILoop(
                ReplayAdapter(),
                write=lambda _s: None,
                size=lambda: (100, 30),
                home=Path(tmp),
                status_fn=lambda: None,
                running=True,
            )
            ui.step(10.0, ())
            self.assertFalse(ui._engine_running)
            self.assertIn("engine stopped", ui.state.notice)

    def test_memory_tiers_cover_16_to_128_gb_macs(self):
        from veystrix.backends.cpp_generic import _prefill_chunk
        from veystrix.memory_tiers import GIB, tier_for
        from veystrix.session import _apply_backend_limits

        expected = [
            (16, "16 GB", 32_768, 1024, 256),
            (24, "24 GB", 65_536, 2048, 512),
            (32, "32 GB", 131_072, 4096, 512),
            (48, "48 GB", 262_144, 4096, 1024),
            (64, "64 GB", 524_288, 8192, 1024),
            (96, "96 GB", 1_048_576, 8192, 2048),
            (128, "128 GB", 1_048_576, 8192, 2048),
        ]
        for gib, name, prefill_lim, chunk, cache_mb in expected:
            t = tier_for(gib * GIB)
            self.assertEqual(t.name, name)
            self.assertEqual(t.prefill_limit, prefill_lim)
            self.assertEqual(t.prefill_chunk, chunk)
            self.assertEqual(t.mlx_cache_mb, cache_mb)
            self.assertGreaterEqual(t.parked_sessions, 1)
            self.assertGreaterEqual(t.prefix_cache_quota_gb, 4.0)
            self.assertEqual(_prefill_chunk({"VEYSTRIX_PHYSICAL_MEMORY_GIB": str(gib)}), chunk)

        class Backend:
            context_limit = 1_048_576

        with patch.dict(os.environ, {"VEYSTRIX_PHYSICAL_MEMORY_GIB": "64"}, clear=False):
            os.environ.pop("VEYSTRIX_MAX_COLD_PREFILL", None)
            os.environ.pop("VEYSTRIX_MAX_INCREMENTAL_PREFILL", None)
            _apply_backend_limits(Backend())
            self.assertEqual(os.environ["VEYSTRIX_MAX_COLD_PREFILL"], "524288")
            self.assertEqual(os.environ["VEYSTRIX_MAX_INCREMENTAL_PREFILL"], "524288")

    def test_persistent_prompt_cache_and_cancel_preservation(self):
        from veystrix.backends import cpp_engine
        from veystrix.backends.base import GenerationCancelled

        class FakeTokenizer:
            unk_token_id = -1

            def convert_tokens_to_ids(self, name):
                return 999 if name == "<|end_message|>" else None

            def decode(self, ids):
                return "".join(f"t{i}" for i in ids)

            def encode(self, text, add_special_tokens=False):
                return [ord(c) for c in text]

        class FakeBinding:
            has_v2 = True

            def __init__(self):
                self.prefills = []
                self.appends = []
                self.restores = []
                self.saves = []
                self.sessions = {}
                self.next_sid = 1

            def abi_version(self):
                return 2

            def mode(self):
                return "live"

            def open_engine(self, _d, _p):
                return object()

            def close_engine(self, _e):
                pass

            def open_session(self, _e):
                sid = self.next_sid
                self.next_sid += 1
                self.sessions[sid] = []
                return sid

            def close_session(self, sid):
                self.sessions.pop(sid, None)

            def prefill(self, sid, toks):
                self.prefills.append(list(toks))
                self.sessions[sid] = list(toks)

            def prefill_ex(self, sid, toks, flags):
                self.appends.append(list(toks))
                self.sessions[sid].extend(toks)

            def step(self, sid, _s):
                self.sessions[sid].append(42)
                return 42

            def tokens(self, sid):
                return len(self.sessions[sid])

            def save(self, sid, directory):
                p = Path(directory)
                p.mkdir(parents=True, exist_ok=True)
                (p / "meta.txt").write_text("ok\n", encoding="utf-8")
                self.saves.append((directory, list(self.sessions[sid])))

            def restore(self, sid, directory):
                self.restores.append(directory)
                for d, toks in reversed(self.saves):
                    if d == directory:
                        self.sessions[sid] = list(toks)
                        return
                toks = json.loads((Path(directory).parent / "tokens.json").read_text())["tokens"]
                self.sessions[sid] = list(toks)

            def cancel(self, _sid):
                pass

            def stats(self, _e):
                return {}

        import json
        with tempfile.TemporaryDirectory() as tmp:
            cpp_engine._CPP_PREFIX_CACHE.clear()
            binding = FakeBinding()
            backend = cpp_engine.CppEngineBackend(
                model_dir="/tmp/fake-model",
                profile_json='{"prefill_chunk": 256}',
                binding=binding,
                tokenizer=FakeTokenizer(),
                session_dir=tmp,
                stop_tokens={42},
            )
            # 1) System prompt (600 tokens) + <|end_message|> (999) + "hello" (5 tokens)
            sys_part = list(range(1, 601)) + [999]
            prompt_hello = sys_part + [10, 11, 12, 13, 14]
            backend.append_tokens(prompt_hello)
            backend.generate(max_tokens=4)
            # Both the static system prefix (601) and full prompt (606) are cached on SSD.
            self.assertEqual(backend.prefix_cache_reusable_tokens(prompt_hello), len(prompt_hello))
            # 2) Same "hello" prompt in a fresh session -> 100% exact SSD restore, 0 tokens prefilled!
            backend.reset()
            prefills_before = len(binding.prefills)
            appends_before = len(binding.appends)
            backend.append_tokens(prompt_hello)
            backend.generate(max_tokens=4)
            self.assertEqual(len(binding.prefills), prefills_before)
            self.assertEqual(len(binding.appends), appends_before)
            self.assertEqual(backend.last_prefix.hit_tokens, len(prompt_hello))
            # 3) Different user turn sharing the same 601-token system prompt -> hits 601, appends only 3!
            backend.reset()
            prompt_hi = sys_part + [20, 21, 22]
            backend.append_tokens(prompt_hi)
            backend.generate(max_tokens=4)
            self.assertEqual(backend.last_prefix.hit_tokens, len(sys_part))
            self.assertEqual(binding.appends[-1], [20, 21, 22])
            # 4) Restart hydration: clearing RAM dict and opening a new backend reloads SSD cache!
            cpp_engine._CPP_PREFIX_CACHE.clear()
            backend2 = cpp_engine.CppEngineBackend(
                model_dir="/tmp/fake-model",
                profile_json='{"prefill_chunk": 256}',
                binding=binding,
                tokenizer=FakeTokenizer(),
                session_dir=tmp,
                stop_tokens={42},
            )
            self.assertEqual(backend2.prefix_cache_reusable_tokens(prompt_hello), len(prompt_hello))
            # 5) Mid-prefill cancel preserves already-prefilled chunks in tape + _live_at!
            backend2.reset()
            cpp_engine._CPP_PREFIX_CACHE.clear()
            long_prompt = list(range(5000, 5800))  # 800 tokens -> chunks of 256
            backend2.append_tokens(long_prompt)
            calls = [0]
            def cancel_after_first():
                calls[0] += 1
                return calls[0] >= 2
            with self.assertRaises(GenerationCancelled):
                backend2.generate(max_tokens=4, on_prefill_progress=lambda _d, _t: None,
                                  should_cancel=cancel_after_first)
            self.assertEqual(len(backend2.tape), 512)
            self.assertEqual(backend2._live_at, 512)
            cpp_engine._CPP_PREFIX_CACHE.clear()

    def test_strict_probe_native_mach_readings_match_schema(self):
        from models.flashnext import chat_guard

        probe = chat_guard.StrictProbe()
        vm = probe.machine()
        self.assertIsInstance(vm, dict)
        for key in (
            "page_bytes", "available_mb", "free_mb", "compressor_mb",
            "wired_mb", "active_mb", "file_backed_mb", "anonymous_mb", "counters",
        ):
            self.assertIn(key, vm)
            self.assertIsNotNone(vm[key], f"missing native vm field: {key}")
        self.assertGreater(vm["page_bytes"], 0)
        self.assertGreaterEqual(vm["available_mb"], vm["free_mb"])
        for ckey in ("swapin", "swapout", "pagein", "pageout", "compress", "decompress"):
            self.assertIn(ckey, vm["counters"])
            self.assertIsInstance(vm["counters"][ckey], int)
        rss = probe.rss_bytes()
        self.assertIsInstance(rss, int)
        self.assertGreater(rss, 0)
        swap_mb = probe.swap_used_mb()
        self.assertIsInstance(swap_mb, float)
        self.assertGreaterEqual(swap_mb, 0.0)
        pressure = probe.pressure_level()
        self.assertIn(pressure, (1, 2, 4))


if __name__ == "__main__":
    unittest.main()
