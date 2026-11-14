"""Stage 4 shape corpus: extraction, step inference, and the committed fixture."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from models.flashnext.tests.bench import kernel_corpus as kc

OPTIONS = {"bits": 4, "group_size": 32, "mode": "affine", "transpose": True}
METAL = {"bits": 4, "group_size": 32, "output_shape": [1, 4, 32], "grid": [32, 8, 4], "threadgroup": [32, 2, 1]}
LAYERS, STEPS, PROMPT = 3, 2, 5


def dense(rows, in_width, out_width, calls):
    return {"operation": kc.DENSE, "calls": calls, "options": dict(OPTIONS),
            "operand_shapes": [[1, rows, in_width], [out_width, in_width // 8],
                               [out_width, in_width // 32], [out_width, in_width // 32]]}


def metal(name, x, experts, out_width, in_width, calls):
    return {"operation": name, "calls": calls, "options": dict(METAL),
            "operand_shapes": [x, [experts, out_width, in_width // 8], [experts, out_width, in_width // 32],
                               [experts, out_width, in_width // 32], [1, experts]]}


def gather(x, experts, out_width, in_width, indices, calls):
    return {"operation": kc.GATHER, "calls": calls, "options": dict(OPTIONS),
            "operand_shapes": [x, [experts, out_width, in_width // 8], [experts, out_width, in_width // 32],
                               [experts, out_width, in_width // 32], indices]}


def arm(rounds=STEPS, with_events=True, scale=1):
    """One arm of a 3-layer model, 2 decode rounds, the second cache_commit_only."""
    rows = [
        # shared-expert gate (out 1): once per layer per forward pass
        dense(1, 64, 1, LAYERS * STEPS), dense(PROMPT, 64, 1, LAYERS),
        # another dense projection, two per layer
        dense(1, 64, 16, 2 * LAYERS * STEPS), dense(PROMPT, 64, 16, 2 * LAYERS),
        # vocabulary head: one round computes logits, the other is cache_commit_only
        dense(1, 64, 32, 1),
        # custom metal path covers layers 1..2 (layer 0 runs gather_qmm)
        metal("metal.gate_qmv", [1, 64], 4, 32, 64, (LAYERS - 1) * STEPS),
        metal("metal.fused_down", [1, 4, 32], 4, 64, 32, (LAYERS - 1) * STEPS),
        gather([1, 1, 1, 1, 64], 4, 32, 64, [1, 1, 4], 2 * STEPS),
        gather([1, 1, 4, 1, 32], 4, 64, 32, [1, 1, 4], STEPS),
        gather([10, 1, 64], 4, 32, 64, [10], 2),
    ]
    for row in rows:
        row["calls"] *= scale
    events = None
    if with_events:
        events = [{"event": "decode_start"}, {"event": "target_round", "positions": 1},
                  {"event": "target_round", "positions": 1, "cache_commit_only": True}]
    return {"stage0": {"shape_trace": rows, "events": events,
                       "metrics": {"target_rounds": rounds, "computed_prefill_tokens": PROMPT}}}


def artifact(arms=2, **kwargs):
    made = [arm(**kwargs) for _ in range(arms)]
    return {"token_limit": STEPS, "raw_arms": {"baseline": copy.deepcopy(made)},
            "conditions": [{"condition": "baseline", "arms": copy.deepcopy(made)}]}


class ExtractTests(unittest.TestCase):
    def extract(self, data):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "production.json"
            path.write_text(json.dumps(data))
            return kc.extract(path, repo_root=folder)

    def decode_row(self, extracted, operation, **match):
        rows = [r for r in extracted["decode"] if r["operation"] == operation
                and all(r.get(k) == v for k, v in match.items())]
        self.assertEqual(len(rows), 1, (operation, match, rows))
        return rows[0]

    def test_arms_are_counted_once_although_the_artifact_stores_each_twice(self):
        extracted = self.extract(artifact(arms=2))
        self.assertEqual(extracted["source"]["unique_arms"], 2)
        self.assertEqual(extracted["source"]["duplicate_views_ignored"], 2)
        gate = self.decode_row(extracted, kc.DENSE, out_width=1)
        self.assertEqual(gate["total_calls"], 2 * LAYERS * STEPS)
        self.assertEqual(gate["calls_per_step"], LAYERS)
        self.assertEqual(sorted(gate["calls_by_arm"]), ["baseline/0", "baseline/1"])

    def test_aggregation_across_arms_sums_calls(self):
        data = artifact(arms=3)
        extracted = self.extract(data)
        row = self.decode_row(extracted, kc.DENSE, out_width=16)
        self.assertEqual(row["total_calls"], 3 * 2 * LAYERS * STEPS)
        self.assertEqual(row["calls_per_step"], 2 * LAYERS)

    def test_decode_and_prefill_classification(self):
        extracted = self.extract(artifact())
        decode_ops = {(r["operation"], tuple(r["operand_shapes"][0])) for r in extracted["decode"]}
        self.assertIn((kc.DENSE, (1, 1, 64)), decode_ops)
        self.assertIn(("metal.gate_qmv", (1, 64)), decode_ops)
        self.assertIn((kc.GATHER, (1, 1, 1, 1, 64)), decode_ops)
        for row in extracted["decode"]:
            self.assertEqual(row["class"], "decode")
        prefill = {(r["operation"], tuple(r["operand_shapes"][0])) for r in extracted["prefill"]}
        self.assertEqual(prefill, {(kc.DENSE, (1, PROMPT, 64)), (kc.GATHER, (10, 1, 64))})
        self.assertTrue(all(r["class"] == "prefill" for r in extracted["prefill"]))

    def test_steps_inferred_two_ways_and_layers_recovered(self):
        steps = self.extract(artifact(arms=2))["decode_steps"]
        self.assertEqual(steps["steps"], 4)
        self.assertEqual(steps["layers"], LAYERS)
        self.assertEqual(steps["by_artifact"]["total_steps"], steps["by_invariant"]["total_steps"])
        self.assertEqual(steps["by_artifact"]["decode_rounds_per_arm"], STEPS)
        self.assertEqual(steps["by_invariant"]["decode_passes_per_arm"], STEPS)

    def test_disagreement_between_artifact_and_call_counts_raises(self):
        with self.assertRaisesRegex(kc.CorpusError, "decode steps disagree"):
            self.extract(artifact(rounds=3))

    def test_views_of_one_arm_that_differ_raise(self):
        data = artifact()
        data["conditions"][0]["arms"][0]["stage0"]["shape_trace"][0]["calls"] += 1
        with self.assertRaisesRegex(kc.CorpusError, "disagree"):
            self.extract(data)

    def test_metal_plus_gather_layers_must_cover_the_layer_count(self):
        extracted = self.extract(artifact())
        coverage = extracted["layer_coverage"]
        self.assertEqual((coverage["metal_layers_per_step"], coverage["gather_layers_per_step"]), (LAYERS - 1, 1))
        data = artifact()
        for made in data["raw_arms"]["baseline"] + data["conditions"][0]["arms"]:
            made["stage0"]["shape_trace"][5]["calls"] += STEPS     # one more metal layer per step
        with self.assertRaisesRegex(kc.CorpusError, "metal layers"):
            self.extract(data)

    def test_derived_widths_and_weight_bytes(self):
        row = self.decode_row(self.extract(artifact()), kc.DENSE, out_width=16)
        self.assertEqual((row["in_width"], row["out_width"]), (64, 16))
        # 16*64/2 packed bytes, plus scales and biases at 16*64/32 groups * 2 bytes each
        self.assertEqual(row["weight_bytes"], 512 + 2 * 64)
        gather_row = self.decode_row(self.extract(artifact()), kc.GATHER, in_width=64)
        self.assertEqual(gather_row["weight_bytes"], 4 * (32 * 64 // 2 + 2 * (32 * 64 // 32 * 2)))
        self.assertEqual(kc.q4_bytes(640, 2560)["total"], 640 * 2560 // 2 + 2 * 640 * 2560 // 32 * 2)

    def test_packing_mismatch_is_an_error(self):
        data = artifact()
        row = data["raw_arms"]["baseline"][0]["stage0"]["shape_trace"][0]
        row["operand_shapes"][1] = [1, 7]
        data["conditions"][0]["arms"][0] = copy.deepcopy(data["raw_arms"]["baseline"][0])
        with self.assertRaisesRegex(kc.CorpusError, "does not pack"):
            self.extract(data)

    def test_fractional_calls_are_explained_by_cache_commit_only_rounds(self):
        row = self.decode_row(self.extract(artifact(arms=2)), kc.DENSE, out_width=32)
        self.assertEqual(row["total_calls"], 2)
        self.assertEqual(row["calls_per_step"], 0.5)
        self.assertIn("cache_commit_only", row["fraction_explained"])

    def test_fractional_calls_without_an_explanation_raise(self):
        with self.assertRaisesRegex(kc.CorpusError, "fractional"):
            self.extract(artifact(with_events=False))

    def test_fractional_call_that_is_not_the_vocabulary_head_raises(self):
        data = artifact()
        for made in data["raw_arms"]["baseline"] + data["conditions"][0]["arms"]:
            made["stage0"]["shape_trace"][2]["calls"] += 1    # out=16 projection no longer divides by the step count
        with self.assertRaisesRegex(kc.CorpusError, "fractional"):
            self.extract(data)

    def test_chunked_prefill_is_refused_rather_than_guessed(self):
        data = artifact()
        for made in data["raw_arms"]["baseline"] + data["conditions"][0]["arms"]:
            made["stage0"]["shape_trace"].append(dense(3, 64, 1, LAYERS))
        with self.assertRaisesRegex(kc.CorpusError, "one prefill pass"):
            self.extract(data)

    def test_fixture_view_keeps_decode_rows_and_summarises_prefill(self):
        extracted = self.extract(artifact())
        view = kc.fixture_view(extracted)
        self.assertNotIn("prefill", view)
        self.assertEqual(view["prefill_summary"]["rows"], len(extracted["prefill"]))
        text = kc.dump(view)
        self.assertEqual(json.loads(text), view)
        self.assertTrue(text.endswith("\n") and "\n" not in text[:-1])


class FixtureTests(unittest.TestCase):
    ARTIFACT = kc.REPO_ROOT / "results/flashnext/20261113-175927-bench-production-manual/production.json"

    def test_committed_fixture_loads_and_is_internally_consistent(self):
        fixture = kc.load()
        steps = fixture["decode_steps"]
        self.assertEqual(steps["steps"], steps["by_artifact"]["total_steps"])
        self.assertEqual(steps["steps"], steps["by_invariant"]["total_steps"])
        coverage = fixture["layer_coverage"]
        self.assertEqual(coverage["metal_layers_per_step"] + coverage["gather_layers_per_step"], steps["layers"])
        self.assertTrue(all(r["class"] == "decode" for r in fixture["decode"]))
        self.assertEqual(fixture["schema_version"], kc.SCHEMA_VERSION)
        self.assertIn("sha256", fixture["source"])

    def test_fixture_matches_extraction_from_the_real_artifact(self):
        if not self.ARTIFACT.is_file():
            self.skipTest(f"source artifact not present: {self.ARTIFACT}")
        self.assertEqual(kc.DEFAULT_FIXTURE.read_text(), kc.dump(kc.fixture_view(kc.extract(self.ARTIFACT))))

    def test_wrong_schema_version_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "corpus.json"
            path.write_text(json.dumps({"schema_version": 999}))
            with self.assertRaisesRegex(kc.CorpusError, "schema_version"):
                kc.load(path)


if __name__ == "__main__":
    unittest.main()
