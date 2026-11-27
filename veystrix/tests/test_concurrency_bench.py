"""The Stage 10 concurrency harness, driven with fake backends (no checkpoint, no server, nothing outside ``results/``)."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from veystrix import results
from veystrix.backends.base import GenerationCancelled
from veystrix.bench import concurrency, qualification
from veystrix.tests.test_cooperative_disconnect import FakeModel, GroupedBackend, fold, following

GIB = 2**30
PROMPT_TOKENS = 400
OUTPUT_TOKENS = 8


class ScriptedClock:
    """A clock the fakes advance, so token timestamps are exactly what the test scripted."""

    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def advance_ms(self, milliseconds):
        self.now += round(milliseconds * 1e6)


class BenchBackend(GroupedBackend):
    """A fake that speaks the qualification-facing surface as well as the cooperative one."""

    def __init__(self, model, config):
        super().__init__(model)
        self.config = config
        self.corrupt = False

    # -- what qualification.prompt_ids and the harness use ----------------------------------------------
    def encode(self, text):
        return [5 + ord(character) % 50 for character in text]

    def append_tokens(self, ids):
        self.pending.extend(ids)

    def reset(self):
        self.state, self.pending, self.tape = 0, [], []

    def check_invariant(self):
        return True

    def fork_for_batch(self):
        if not self.config.eligible:
            return None
        fork = BenchBackend(self.model, self.config)
        fork.corrupt = self.config.corrupt_forks
        return fork

    # -- the dedicated path ----------------------------------------------------------------------------
    def generate(self, max_tokens, out=None, on_prefilled=None, on_prefill_progress=None, on_decode_token=None,
                 should_cancel=None):
        cancel = should_cancel or (lambda: False)
        self.state = fold(0, self.pending)
        self.tape.extend(self.pending)
        self.pending.clear()
        if on_prefilled is not None:
            on_prefilled()
        value, produced = following(self.state), 0
        for _ in range(max_tokens):
            if cancel():
                raise GenerationCancelled
            self.config.pause(self.config.dedicated_ms)
            if self.config.interrupt_dedicated == produced:
                raise KeyboardInterrupt
            on_decode_token(value, "")
            self.state = fold(self.state, [value])
            value = following(self.state)
            produced += 1
        return "", SimpleNamespace(finish="length", tokens=produced, rate=0.0, prefill_seconds=0.0)

    # -- the cooperative path --------------------------------------------------------------------------
    def batch_decode(self, value, sampler, want_logits):
        config = self.config
        with config.lock:
            config.decodes += 1
            number = config.decodes
        if config.interrupt_cooperative == number:
            raise KeyboardInterrupt
        config.pause(config.cooperative_ms)
        token = super().batch_decode(value, sampler, want_logits)
        if self.corrupt and token is not None:
            token = 5 + (token - 5 + 1) % 50
        return token


class Config(SimpleNamespace):
    def pause(self, milliseconds):
        if self.clock is not None:
            self.clock.advance_ms(milliseconds)
        if self.sleep:
            time.sleep(self.sleep)


def make_backend(*, clock=None, **options):
    config = Config(clock=clock, dedicated_ms=100, cooperative_ms=125, sleep=0, eligible=True, corrupt_forks=False,
                    interrupt_dedicated=None, interrupt_cooperative=None, decodes=0, lock=threading.Lock())
    for key, value in options.items():
        setattr(config, key, value)
    return BenchBackend(FakeModel(), config)


class FakeMeter:
    total = 64_000_000

    def reset(self):
        self.started = True

    def bytes_since(self):
        return FakeMeter.total


def healthy_memory():
    return {"available_bytes": 8 * GIB, "compressor_bytes": 0, "swap_used_bytes": 100, "swapin_bytes_total": 0,
            "swapout_bytes_total": 0}


class StatisticsTests(unittest.TestCase):
    def test_percentiles_use_linear_interpolation(self):
        self.assertIsNone(concurrency.percentile([], 0.5))
        self.assertEqual(concurrency.percentile([7], 0.95), 7)
        self.assertEqual(concurrency.percentile([10, 20, 30, 40, 50], 0.50), 30)
        self.assertAlmostEqual(concurrency.percentile([50, 10, 40, 20, 30], 0.95), 48.0)

    def test_stream_metrics_from_injected_timestamps(self):
        start = 5_000_000_000
        times = [start + int(ms * 1e6) for ms in (100, 200, 330, 430, 560)]
        metrics = concurrency.stream_metrics(times, start)
        self.assertEqual(metrics["output_tokens"], 5)
        self.assertAlmostEqual(metrics["ttft_ms"], 100.0)
        self.assertAlmostEqual(metrics["stream_tok_s"], 4 / 0.460)                  # (n - 1) / (t_last - t_first)
        self.assertAlmostEqual(metrics["gap_p50_ms"], 115.0)                         # gaps 100, 130, 100, 130
        self.assertAlmostEqual(metrics["gap_p95_ms"], 130.0)
        self.assertAlmostEqual(metrics["gap_max_ms"], 130.0)

    def test_degenerate_streams_have_no_rate(self):
        self.assertIsNone(concurrency.stream_metrics([], 0)["ttft_ms"])
        one = concurrency.stream_metrics([300_000_000], 100_000_000)
        self.assertAlmostEqual(one["ttft_ms"], 200.0)
        self.assertIsNone(one["stream_tok_s"])
        self.assertIsNone(one["gap_max_ms"])

    def test_aggregate_is_all_tokens_over_first_to_last_wall_time(self):
        first = [int(ms * 1e6) for ms in (100, 200, 300)]
        second = [int(ms * 1e6) for ms in (150, 250, 350)]
        aggregate = concurrency.aggregate_metrics([first, second, []])
        self.assertEqual(aggregate["aggregate_tokens"], 6)
        self.assertAlmostEqual(aggregate["aggregate_span_s"], 0.25)
        self.assertAlmostEqual(aggregate["aggregate_tok_s"], 24.0)
        self.assertIsNone(concurrency.aggregate_metrics([[]])["aggregate_tok_s"])

    def test_overhead_percentage(self):
        self.assertAlmostEqual(concurrency.overhead_pct(10.0, 8.0), 20.0)
        self.assertAlmostEqual(concurrency.overhead_pct(10.0, 11.0), -10.0)
        for dedicated, cooperative in ((None, 8.0), (10.0, None), (0.0, 1.0)):
            self.assertIsNone(concurrency.overhead_pct(dedicated, cooperative))


class PromptTests(unittest.TestCase):
    def test_variants_are_distinct_equal_length_and_extend_qualifications_two(self):
        backend = make_backend()
        prompts = concurrency.prompt_variants(backend, PROMPT_TOKENS, 4)
        self.assertEqual(len({row["prompt_sha256"] for row in prompts}), 4)
        self.assertTrue(all(len(row["ids"]) == PROMPT_TOKENS for row in prompts))
        for variant in (0, 1):
            self.assertEqual(prompts[variant]["ids"], qualification.prompt_ids(backend, PROMPT_TOKENS, variant=variant))
        self.assertEqual(concurrency.prompt_variants(backend, PROMPT_TOKENS, 2)[1]["ids"], prompts[1]["ids"])
        with self.assertRaises(ValueError):
            concurrency.prompt_variants(backend, PROMPT_TOKENS, 5)
        with self.assertRaises(ValueError):
            concurrency.prompt_variants(backend, 40, 1)


class HarnessCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="concurrency-results-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.enterContext(mock.patch.object(results, "RESULTS_ROOT", self.root))
        self.enterContext(mock.patch.dict(os.environ))
        os.environ.pop("VEYSTRIX_RESULTS_DIR", None)
        self.out = self.root / "nemotron35" / "stamp-stage10-concurrency" / "concurrency.json"

    def run_arms(self, backend, arms, *, streams=None, memory_read=healthy_memory, memory_interval=0.01, **options):
        streams = streams or max(concurrency.STREAMS[name] for name in arms)
        prompts = concurrency.prompt_variants(backend, PROMPT_TOKENS, streams)
        report = {"model": "fake"}
        options.setdefault("clock", backend.config.clock or time.perf_counter_ns)
        options.setdefault("read_meter", FakeMeter)
        code = concurrency.run_benchmark(
            backend, report, self.out, arms=arms, prompts=prompts, output_tokens=OUTPUT_TOKENS,
            memory_read=memory_read, memory_interval=memory_interval, headroom=lambda: 1 << 62, **options)
        return code, report

    def evidence(self):
        return json.loads(self.out.read_text())


class ArmTests(HarnessCase):
    def test_single_stream_arms_record_timing_exactness_and_the_overhead(self):
        clock = ScriptedClock()
        backend = make_backend(clock=clock, dedicated_ms=100, cooperative_ms=125)
        code, report = self.run_arms(backend, ["dedicated", "cooperative-1"], clock=clock)
        self.assertEqual(code, concurrency.EXIT_CLEAN)
        self.assertEqual(report["status"], "completed")
        dedicated, cooperative = report["arms"]
        for arm in (dedicated, cooperative):
            self.assertEqual(arm["status"], "completed")
            (stream,) = arm["streams"]
            self.assertEqual(stream["output_tokens"], OUTPUT_TOKENS)
            self.assertTrue(stream["digests_match_single"])
            self.assertEqual(len(stream["token_times_ms"]), OUTPUT_TOKENS)
        self.assertAlmostEqual(dedicated["streams"][0]["stream_tok_s"], 10.0)
        self.assertAlmostEqual(cooperative["streams"][0]["stream_tok_s"], 8.0)
        self.assertAlmostEqual(dedicated["streams"][0]["gap_p50_ms"], 100.0)
        self.assertAlmostEqual(cooperative["streams"][0]["gap_max_ms"], 125.0)
        self.assertAlmostEqual(report["cooperative_1_overhead_pct"], 20.0)
        self.assertTrue(report["cooperative_1_matches_dedicated"])
        self.assertEqual(cooperative["scheduler"]["max_decode_group"], 1)
        self.assertIsNone(dedicated["scheduler"])
        # 64 MB over the tokens emitted after the first
        self.assertAlmostEqual(dedicated["physical_mb_per_token"], 64.0 / (OUTPUT_TOKENS - 1))
        self.assertAlmostEqual(dedicated["aggregate_tok_s"], OUTPUT_TOKENS / (0.1 * (OUTPUT_TOKENS - 1)))
        self.assertEqual(dedicated["available_floor_bytes"], 8 * GIB)
        self.assertEqual(dedicated["swap_growth_bytes"], 0)
        self.assertEqual(self.evidence()["arms"][1]["arm"], "cooperative-1")

    def test_two_and_four_streams_complete_with_digests_equal_to_their_single_references(self):
        backend = make_backend(sleep=0.0005)
        arms = ["dedicated", "cooperative-1", "cooperative-2", "cooperative-4"]
        code, report = self.run_arms(backend, arms)
        self.assertEqual(code, concurrency.EXIT_CLEAN)
        two, four = report["arms"][2], report["arms"][3]
        self.assertEqual((len(two["streams"]), len(four["streams"])), (2, 4))
        for arm in (two, four):
            self.assertEqual(arm["status"], "completed")
            self.assertEqual([row["output_tokens"] for row in arm["streams"]], [OUTPUT_TOKENS] * len(arm["streams"]))
            self.assertTrue(all(row["digests_match_single"] for row in arm["streams"]))
            self.assertEqual(len({row["token_sha256"] for row in arm["streams"]}), len(arm["streams"]))
            self.assertGreaterEqual(arm["scheduler"]["max_decode_group"], 1)
            self.assertLessEqual(arm["scheduler"]["max_decode_group"], len(arm["streams"]))
            self.assertIn("decode_groups", arm["scheduler"])
            self.assertGreater(arm["aggregate_tok_s"], 0)
        # variants 1..3 had no single-stream reference: one reference run each, through the dedicated path
        self.assertEqual(len(report["reference_runs"]), 3)
        self.assertEqual({row["streams"][0]["variant"] for row in report["reference_runs"]}, {1, 2, 3})
        sources = {row["reference_source"] for arm in (two, four) for row in arm["streams"]}
        self.assertEqual(sources, {"dedicated", "reference-run"})
        self.assertEqual(report["exactness"]["mismatches"], 0)
        self.assertTrue(report["exactness"]["all_match"])
        # the streams really are different prompts
        self.assertEqual(len({row["prompt_sha256"] for row in four["streams"]}), 4)

    def test_without_single_references_only_known_prompts_are_checked(self):
        backend = make_backend()
        code, report = self.run_arms(backend, ["cooperative-2", "dedicated"], single_references=False)
        two = report["arms"][0]
        self.assertEqual(report["reference_runs"], [])
        # reversed order: the dedicated arm ran last, yet it is the reference for prompt 0
        self.assertTrue(two["streams"][0]["digests_match_single"])
        self.assertIsNone(two["streams"][1]["digests_match_single"])
        self.assertEqual(code, concurrency.EXIT_CLEAN)

    def test_reversed_order_gives_the_same_exactness_and_records_the_arms_in_run_order(self):
        backend = make_backend()
        arms = ["dedicated", "cooperative-1", "cooperative-2"][::-1]
        code, report = self.run_arms(backend, arms)
        self.assertEqual([arm["arm"] for arm in report["arms"]], arms)
        self.assertEqual(code, concurrency.EXIT_CLEAN)
        self.assertTrue(report["exactness"]["all_match"])
        self.assertTrue(all(row["digests_match_single"] for arm in report["arms"] for row in arm["streams"]))

    def test_a_concurrent_stream_that_differs_from_its_single_reference_is_flagged(self):
        backend = make_backend(corrupt_forks=True)
        code, report = self.run_arms(backend, ["dedicated", "cooperative-2"])
        self.assertEqual(code, concurrency.EXIT_MISMATCH)
        self.assertEqual(report["status"], "completed")
        self.assertFalse(report["exactness"]["all_match"])
        self.assertTrue(all(row["digests_match_single"] is False for row in report["arms"][1]["streams"]
                            if row["variant"] == 0))
        self.assertGreaterEqual(report["exactness"]["mismatches"], 1)

    def test_an_ineligible_configuration_is_reported_and_nothing_runs(self):
        backend = make_backend(eligible=False)
        code, report = self.run_arms(backend, ["dedicated", "cooperative-1"])
        self.assertEqual(code, concurrency.EXIT_USAGE)
        self.assertEqual(report["status"], "ineligible")
        self.assertIn("fork_for_batch() returned None", report["error"])
        self.assertEqual(self.evidence()["status"], "ineligible")
        self.assertEqual([arm["arm"] for arm in report["arms"]], ["dedicated"])   # the dedicated arm still ran first


class StopTests(HarnessCase):
    def low_after(self, healthy_calls):
        calls = [0]

        def read():
            calls[0] += 1
            row = healthy_memory()
            if calls[0] > healthy_calls:
                row["available_bytes"] = 2 * GIB
            return row

        return read

    def test_available_memory_below_three_gib_cancels_a_dedicated_arm_and_is_recorded(self):
        backend = make_backend(sleep=0.004)
        prompts = concurrency.prompt_variants(backend, PROMPT_TOKENS, 1)
        report = {"model": "fake"}
        code = concurrency.run_benchmark(
            backend, report, self.out, arms=["dedicated", "cooperative-1"], prompts=prompts, output_tokens=400,
            memory_read=self.low_after(3), memory_interval=0.002, read_meter=FakeMeter, headroom=lambda: 1 << 62)
        self.assertEqual(code, concurrency.EXIT_STOPPED)
        (arm,) = report["arms"]                           # the run ended; cooperative-1 never started
        self.assertEqual(arm["status"], "stopped")
        self.assertEqual(arm["stopped"]["available_bytes"], 2 * GIB)
        self.assertEqual(arm["stopped"]["floor_bytes"], qualification.FLOOR)
        self.assertIn("below the floor", arm["stopped"]["reason"])
        stream = arm["streams"][0]
        self.assertTrue(stream["cancelled"])
        self.assertFalse(stream["completed"])
        self.assertLess(stream["output_tokens"], 400)
        self.assertIsNone(stream["digests_match_single"])
        self.assertEqual(arm["available_floor_bytes"], 2 * GIB)
        saved = self.evidence()
        self.assertEqual(saved["status"], "stopped")
        self.assertEqual(saved["arms"][0]["stopped"]["available_bytes"], 2 * GIB)

    def test_a_cooperative_arm_with_two_streams_stops_both_and_the_loop_unwinds(self):
        backend = make_backend(sleep=0.004)
        prompts = concurrency.prompt_variants(backend, PROMPT_TOKENS, 2)
        report = {"model": "fake"}
        code = concurrency.run_benchmark(
            backend, report, self.out, arms=["cooperative-2"], prompts=prompts, output_tokens=400,
            memory_read=self.low_after(3), memory_interval=0.002, read_meter=FakeMeter, headroom=lambda: 1 << 62)
        self.assertEqual(code, concurrency.EXIT_STOPPED)
        (arm,) = report["arms"]
        self.assertEqual(arm["status"], "stopped")
        self.assertEqual(arm["stopped"]["available_bytes"], 2 * GIB)
        self.assertTrue(all(row["cancelled"] for row in arm["streams"]))
        self.assertTrue(all(row["output_tokens"] < 400 for row in arm["streams"]))
        self.assertEqual(self.evidence()["status"], "stopped")

    def test_low_memory_before_inference_refuses_the_arm(self):
        backend = make_backend()
        code, report = self.run_arms(backend, ["dedicated"], memory_read=lambda: {**healthy_memory(),
                                                                                  "available_bytes": GIB})
        self.assertEqual(code, concurrency.EXIT_STOPPED)
        self.assertEqual(report["arms"][0]["status"], "refused")
        self.assertEqual(report["arms"][0]["stopped"]["available_bytes"], GIB)

    def test_an_unsampleable_host_is_treated_as_below_the_floor(self):
        backend = make_backend()
        code, report = self.run_arms(backend, ["dedicated"], memory_read=lambda: {**healthy_memory(),
                                                                                  "available_bytes": None})
        self.assertEqual(code, concurrency.EXIT_STOPPED)
        self.assertEqual(report["arms"][0]["status"], "refused")

    def test_an_interrupt_in_a_dedicated_arm_stops_the_run_with_evidence(self):
        backend = make_backend(interrupt_dedicated=3)
        code, report = self.run_arms(backend, ["dedicated", "cooperative-1"])
        self.assertEqual(code, concurrency.EXIT_STOPPED)
        self.assertEqual(report["status"], "stopped")
        self.assertIn("KeyboardInterrupt", report["error"])
        self.assertEqual(self.evidence()["status"], "stopped")

    def test_an_interrupt_on_the_model_thread_unwinds_every_request_thread(self):
        backend = make_backend(interrupt_cooperative=4)
        before = threading.active_count()
        code, report = self.run_arms(backend, ["cooperative-2"], single_references=False)
        self.assertEqual(code, concurrency.EXIT_STOPPED)
        self.assertEqual(report["status"], "stopped")
        deadline = time.monotonic() + 5
        while threading.active_count() > before and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertLessEqual(threading.active_count(), before)


class MainTests(HarnessCase):
    def checkpoint(self):
        path = Path(tempfile.mkdtemp(prefix="concurrency-checkpoint-"))
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def run_main(self, backend, *arguments):
        for stub in (mock.patch.object(concurrency, "load_backend", return_value=backend),
                     mock.patch.object(concurrency, "collect", return_value={"host": "stub"}),
                     mock.patch.object(concurrency, "checkpoint_metadata", return_value={"files": "stub"}),
                     mock.patch("veystrix.bench.memory.sample", healthy_memory)):
            self.enterContext(stub)
        with contextlib.redirect_stdout(io.StringIO()):
            return concurrency.main([
                "--model", "nemotron35", "--checkpoint", str(self.checkpoint()), "--prompt-tokens", str(PROMPT_TOKENS),
                "--output-tokens", str(OUTPUT_TOKENS), "--ignore-admission", *arguments])

    def run_directories(self):
        return sorted(path for path in (self.root / "nemotron35").iterdir())

    def test_main_runs_the_arms_and_writes_only_one_evidence_file_under_results(self):
        backend = make_backend()
        code = self.run_main(backend, "--arms", "dedicated", "cooperative-2", "--order", "reversed")
        self.assertEqual(code, concurrency.EXIT_CLEAN)
        (run,) = self.run_directories()
        self.assertRegex(run.name, r"^\d{8}-\d{6}-stage10-concurrency-manual$")
        self.assertEqual([path.name for path in run.iterdir()], ["concurrency.json"])    # the prefix scratch dir is gone
        report = json.loads((run / "concurrency.json").read_text())
        self.assertEqual(report["status"], "completed")
        self.assertEqual([arm["arm"] for arm in report["arms"]], ["cooperative-2", "dedicated"])
        self.assertEqual(report["arms_requested"], ["cooperative-2", "dedicated"])
        self.assertEqual(report["harness_sha256"], concurrency.harness_sha256())
        self.assertEqual(report["reference_host"], {"host": "stub"})
        self.assertEqual(report["admission"], "disabled by --ignore-admission")
        self.assertFalse(report["prefix_cache"])
        self.assertEqual(report["runtime_environment"]["FLASHNEXT_PREFIX_CACHE"], "0")
        self.assertEqual(report["runtime_environment"]["FLASHNEXT_PREFILL_CACHE_LIMIT_MB"], "512")
        self.assertTrue(all(key.startswith(concurrency.ENGINE_PREFIXES) for key in report["runtime_environment"]))
        self.assertNotIn("FLASHNEXT_PREFIX_CACHE_DIR", report["runtime_environment"])
        self.assertEqual(len(report["prompts"]), 2)

    def test_main_exits_nonzero_with_a_clear_message_when_forking_is_not_possible(self):
        backend = make_backend(eligible=False)
        with mock.patch("sys.stderr") as stderr:
            code = self.run_main(backend, "--arms", "dedicated", "cooperative-1")
        self.assertEqual(code, concurrency.EXIT_USAGE)
        message = "".join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn("fork_for_batch() returned None", message)
        (run,) = self.run_directories()
        report = json.loads((run / "concurrency.json").read_text())
        self.assertEqual(report["status"], "ineligible")
        self.assertEqual(report["arms"], [])

    def test_prefix_cache_flag_mirrors_qualification_when_on(self):
        backend = make_backend()
        self.run_main(backend, "--arms", "dedicated", "--prefix-cache", "on")
        (run,) = self.run_directories()
        report = json.loads((run / "concurrency.json").read_text())
        self.assertTrue(report["prefix_cache"])
        self.assertEqual(report["runtime_environment"]["FLASHNEXT_PREFIX_CACHE"], "1")
        self.assertEqual(report["runtime_environment"]["VEYSTRIX_NEMOTRON_PREFIX_CACHE"], "1")

    def test_evidence_cannot_be_written_outside_results(self):
        with self.assertRaises(ValueError):
            concurrency.write_evidence(Path(tempfile.gettempdir()) / "concurrency-escape.json", {})
        with self.assertRaises(ValueError):
            concurrency.write_evidence(self.root.parent / "elsewhere" / "concurrency.json", {})
        self.assertFalse((Path(tempfile.gettempdir()) / "concurrency-escape.json").exists())

    def test_the_environment_block_matches_qualifications(self):
        """qualification.main owns the block; this harness mirrors it, so a change there must show up here."""
        source = Path(qualification.__file__).read_text()
        start = source.index("os.environ.update({", source.index("def main"))
        block = source[start:source.index("})", start)]
        expected = dict(re.findall(r'"([A-Z0-9_]+)":\s*"([^"]*)"', block))
        self.assertGreater(len(expected), 10, "could not find the environment block in qualification.main")
        with tempfile.TemporaryDirectory() as directory:
            for key in list(expected):
                os.environ.pop(key, None)
            concurrency.apply_harness_environment(directory, prefix_cache=True)
            observed = {key: os.environ.get(key) for key in expected}
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
