from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from models.flashnext.tests.bench import bench_production
from veystrix.bench import roofline
from veystrix.bench.memory import system_sample
from veystrix.bench.telemetry import Trace, current, mlx_shapes


FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "inputs" / "stage0-telemetry.json").read_text())


class TelemetryTests(unittest.TestCase):
    def test_roofline_and_repeat_resolution(self):
        row = roofline.arm_metrics(FIXTURE["events"], FIXTURE["computed_prefill_tokens"],
                                   FIXTURE["physical_read_bytes"], FIXTURE["io_wait_ns"])
        self.assertEqual(row["ttft_ns"], 60_000_000)
        self.assertEqual(row["tokenization_ns"], 1_000_000)
        self.assertEqual(row["prefix_lookup_restore_ns"], 3_000_000)
        self.assertEqual(row["computed_prefill_tok_s"], 400)
        self.assertAlmostEqual(row["stream_decode_tok_s"], 2 / 0.09)
        self.assertEqual(row["engine_decode_tok_s"], 30)
        self.assertEqual(row["physical_mb_per_token"], 40)
        self.assertEqual(row["effective_storage_gb_s"], 2)
        self.assertAlmostEqual(row["non_io_ms_per_token"], 40 / 3)
        self.assertEqual(row["accepted_tokens_per_verification"], 2)
        self.assertEqual(row["median_target_round_ns"], 50_000_000)
        self.assertEqual(row["upper_bound_at_3gb_s_tok_s"], 37.5)
        band = roofline.resolution(FIXTURE["repeated_stream_rates"])
        self.assertEqual(band["median"], 22)
        self.assertAlmostEqual(band["resolution_half_width"], 1.6)
        self.assertIsNone(roofline.resolution([1, 2, 3, 4]))

    def test_sidecar_marks_memory_floor_and_retains_matching_provenance(self):
        metadata = dict.fromkeys(("stage0_provenance", "checkpoint_identity", "source_fingerprints",
                                  "prompt_sha256", "sampling", "token_limit", "protocol"))
        metrics = roofline.arm_metrics(FIXTURE["events"], 16, 120_000_000, 60_000_000)
        row = {"token_sha256": "a" * 64, "gen_rate": 25, "stage0": {
            "metrics": metrics, "memory": {"sampled_available_floor_bytes": 2 * 2**30},
        }}
        report = roofline.report(metadata, {"baseline": [row] * 5}, 0)
        self.assertEqual(report["schema_version"], FIXTURE["schema_version"])
        self.assertEqual(report["provenance"], metadata)
        self.assertFalse(report["conditions"]["baseline"]["sampled_memory_floor_pass"])
        self.assertEqual(report["conditions"]["baseline"]["resolution"]["engine_decode_tok_s"]["median"], 30)
        self.assertIsNone(report["paired_overhead"])

    def test_incomplete_observations_are_null(self):
        row = roofline.arm_metrics([], None, None, None)
        for name in ("ttft_ns", "computed_prefill_tok_s", "stream_decode_tok_s",
                     "effective_storage_gb_s", "accepted_tokens_per_verification",
                     "upper_bound_compute_only_tok_s"):
            self.assertIsNone(row[name])

    def test_overhead_requires_matching_digests(self):
        baseline = [{"gen_rate": 20, "token_sha256": "same"}] * 5
        measured = [{"gen_rate": 19, "token_sha256": "same"}] * 5
        self.assertFalse(roofline.paired_overhead(baseline, measured)["under_one_percent"])
        self.assertTrue(roofline.paired_overhead(baseline, baseline)["under_one_percent"])
        with self.assertRaisesRegex(ValueError, "digests"):
            roofline.paired_overhead(baseline, [{"token_sha256": "changed"}] * 5)

    def test_dispatch_trace_is_scoped_and_shape_only(self):
        def dispatch(*args, **kwargs):
            return 123

        mx = SimpleNamespace(**dict.fromkeys(
            ("quantized_matmul", "gather_qmm", "eval", "async_eval", "synchronize"), dispatch
        ))
        trace = Trace(capture_shapes=True)
        operand = SimpleNamespace(shape=(2, 16), secret="never serialize")
        with self.assertRaisesRegex(ValueError, "restore"):
            with mlx_shapes(mx, trace):
                mx.quantized_matmul(operand, operand)
                with trace.activate():
                    self.assertEqual(mx.quantized_matmul(operand, operand, bits=4), 123)
                    mx.eval(operand)
                    raise ValueError("restore")
        self.assertIsNone(current.get())
        self.assertIs(mx.quantized_matmul, dispatch)
        self.assertEqual(trace.counts["mlx_eval_calls"], 1)
        self.assertEqual(trace.shape_rows(), [{
            "operation": "mlx.quantized_matmul", "operand_shapes": ((2, 16), (2, 16)),
            "options": {"bits": 4}, "calls": 1,
        }])

    def test_memory_parse_preserves_unavailable_fields(self):
        vm = ("Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
              "Pages free:  4.\nPages inactive:  8.\nPages speculative:  0.\n"
              "Pages occupied by compressor: 2.\nSwapouts: 0.\n")
        with patch("veystrix.bench.memory.output", return_value=vm), patch(
            "models.flashnext.watchdog.swap_used_mib", side_effect=OSError,
        ):
            row = system_sample()
        self.assertEqual(row["available_bytes"], 12 * 4096)
        self.assertEqual(row["compressor_bytes"], 2 * 4096)
        self.assertEqual(row["swapout_bytes_total"], 0)
        self.assertIsNone(row["swap_used_bytes"])
        self.assertIsNone(row["swapin_bytes_total"])

    def test_production_arm_preserves_legacy_token_digest(self):
        ids = FIXTURE["ids"]

        class Backend:
            def reset(self):
                self.tape, self.pending = [], [1, 2]

            def append_text(self, prompt):
                pass

            def generate(self, max_tokens, on_prefilled, **kwargs):
                trace = current.get()
                if trace is not None:
                    trace.prefill_tokens = len(self.pending)
                    trace.mark("prefill_start")
                    trace.mark("first_logits_ready")
                on_prefilled()
                if trace is not None:
                    trace.mark("decode_start")
                for token in ids:
                    self.tape.append(token)
                    if "on_decode_token" in kwargs:
                        kwargs["on_decode_token"](token, "")
                if trace is not None:
                    trace.decode_commit()
                return "", SimpleNamespace(tokens=len(ids), rate=20)

        target = Backend()
        meter = SimpleNamespace(reset=lambda: None, bytes_since=lambda: 1000)
        class Sampler(nullcontext):
            def prepare(self):
                pass

            def summary(self):
                return {}

        sampler = Sampler()
        with patch.object(bench_production, "memory_snapshot", return_value={}), patch(
            "models.flashnext.diskio.vm_counters", return_value={},
        ), patch("models.flashnext.diskio.free_memory_mb", return_value=100), patch(
            "veystrix.bench.memory.MemorySampler", return_value=sampler,
        ) as memory:
            off = bench_production.arm(target, len(ids), meter, 0)
            memory.assert_not_called()
            on = bench_production.arm(target, len(ids), meter, 0, telemetry="timing")
        expected = hashlib.sha256(str(tuple(ids)).encode()).hexdigest()
        self.assertEqual(off["token_sha256"], expected)
        self.assertEqual(on["token_sha256"], expected)
        self.assertEqual(off["ids"], on["ids"])
        self.assertNotIn("stage0", off)
        self.assertEqual(on["stage0"]["metrics"]["computed_prefill_tokens"], 2)
        self.assertEqual(sum(e["event"] == "token_emitted" for e in on["stage0"]["events"]), len(ids))


if __name__ == "__main__":
    unittest.main()
