import unittest

from models.flashnext.tests.bench.bench_stage6_prefill import summarize


class Stage6SummaryTests(unittest.TestCase):
    def test_resolution_and_acceptance_are_separate_from_correctness(self):
        rows = []
        for index in range(5):
            for arm, ttft, decode in (("reference", 120, 5), ("candidate", 100, 5)):
                rows.append({
                    "size": 2048, "round": index + 1, "cache": "cold", "arm": arm,
                    "token_sha256": "tokens", "state_sha256": "state",
                    "metrics": {"ttft_ns": ttft, "prefill_ns": ttft - 10,
                                "stream_decode_tok_s": decode, "computed_prefill_tok_s": 20},
                })
        summary = summarize(rows)["2048/cold"]
        self.assertTrue(summary["ttft_win"])
        self.assertTrue(summary["prefill_win"])
        self.assertTrue(summary["decode_within_reference_noise"])
        self.assertTrue(summary["state_parity"])
        self.assertIsNone(summarize(rows[:2])["2048/cold"]["ttft_win"])
        self.assertIsNone(summarize(rows)["2048/prefix-hit"]["token_parity"])
