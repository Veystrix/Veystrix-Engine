"""The layer-by-layer divergence tracer, exercised on a tiny synthetic checkpoint."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlx.nn as nn

from models.nemotron35 import trace_real, verify_real
from models.nemotron35.tiny_checkpoint import TINY_CONFIG, permissive_config_check, write_checkpoint

# tiny layout: 0 mamba, 1 moe, 2 attention, 3 moe
TOKENS, CHUNK = 21, 8


class Scaled(nn.Module):
    """A deliberate deviation: the wrapped module's output times ``factor``. (1.0001 would vanish in bfloat16.)"""

    def __init__(self, inner, factor=1.05):
        super().__init__()
        self.inner = inner
        self.factor = factor

    def __call__(self, *args, **kwargs):
        return self.inner(*args, **kwargs) * self.factor


class TraceRealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="nemotron-trace-")
        write_checkpoint(Path(cls.dir) / "ck")
        cls.root = Path(cls.dir) / "ck"
        write_checkpoint(Path(cls.dir) / "ck64", config=dict(TINY_CONFIG, head_dim=64))    # INT8 KV needs head_dim % 64 == 0

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def run_trace(self, *, decode=2, mutate=None, root=None, **kw):
        real = trace_real.load_streamed

        def patched(*args, **kwargs):
            streamed = real(*args, **kwargs)
            if mutate is not None:
                mutate(streamed.model.backbone.layers)
            return streamed

        with mock.patch.object(trace_real, "load_streamed", patched):
            return trace_real.trace(root or self.root, tokens=TOKENS, decode=decode, chunk=CHUNK, swap_limit_mb=1e9,
                                    config_check=permissive_config_check, **kw)

    # (a) structure and agreement with verify -------------------------------------------------------------------
    def test_structure_and_free_logits_match_verify(self):
        report = self.run_trace()
        verified = verify_real.verify(self.root, config_check=permissive_config_check, tokens=TOKENS, decode=2,
                                      chunk=CHUNK, swap_limit_mb=1e9)
        self.assertEqual([p["phase"] for p in report["phases"]], ["prefill", "decode1", "decode2"])
        kinds = ["mamba", "moe", "attention", "moe"]
        for phase in report["phases"]:
            self.assertEqual([e["layer"] for e in phase["layers"]], [0, 1, 2, 3])
            self.assertEqual([e["kind"] for e in phase["layers"]], kinds)
            for entry in phase["layers"]:
                for side in ("forced", "free"):
                    self.assertEqual(
                        set(entry[side]) >= {"max_abs_diff", "mean_abs_diff", "differing_elements", "elements",
                                             "bit_identical", "max_abs_ref"}, True)
                self.assertIn("reference_dtype", entry)
                self.assertIn("forced_dtype", entry)
                self.assertIn("free_dtype", entry)
        self.assertEqual(report["phases"][0]["chunk_sizes"], [8, 8, 5])
        self.assertEqual(report["phases"][1]["chunk_sizes"], [1])
        # state is compared for the BF16 configuration on mamba and attention layers only
        for entry in report["phases"][0]["layers"]:
            if entry["kind"] in ("mamba", "attention"):
                self.assertEqual(entry["state_comparison"], "compared")
                self.assertIn(entry["state_bit_identical"], (True, False))
            else:
                self.assertNotIn("state_bit_identical", entry)
        for phase, step in zip(report["phases"], verified["steps"]):
            free = phase["logits"]["free_vs_reference"]
            self.assertAlmostEqual(free["max_abs_diff"], step["max_abs_diff"], delta=1e-6)
            self.assertEqual(free["greedy_token"], step["greedy_token"])
            self.assertEqual(phase["reference_greedy_token"], step["greedy_token"])
        for key in ("first_forced_divergence", "forced_divergent_layers_by_kind", "free_max_abs_by_layer",
                    "all_forced_bit_identical"):
            self.assertIn(key, report)
        self.assertEqual(report["free_max_abs_by_layer"]["prefill"].__len__(), 4)
        total = report["forced_divergent_layers_by_kind"]["prefill"]
        self.assertEqual({k: v["total"] for k, v in total.items()}, {"mamba": 1, "moe": 2, "attention": 1})
        # Without any injection the tiny model is consistent: no forced divergence implies no drill-down.
        if report["first_forced_divergence"] is None:
            self.assertTrue(report["all_forced_bit_identical"])
            self.assertFalse(any("suboperations" in e for p in report["phases"] for e in p["layers"]))

    # (b) injected deviations -------------------------------------------------------------------------------------
    def test_injected_moe_deviation_is_localised_to_the_routed_experts(self):
        def mutate(layers):
            layers[3].mixer.switch_mlp = Scaled(layers[3].mixer.switch_mlp)

        report = self.run_trace(decode=1, mutate=mutate)
        self.assertEqual(report["first_forced_divergence"], {"phase": "prefill", "layer": 3, "kind": "moe"})
        self.assertFalse(report["all_forced_bit_identical"])
        prefill = report["phases"][0]["layers"]
        for entry in prefill[:3]:
            self.assertTrue(entry["forced"]["bit_identical"], entry["layer"])
            self.assertNotIn("suboperations", entry)
        culprit = prefill[3]
        self.assertFalse(culprit["forced"]["bit_identical"])
        self.assertGreater(culprit["forced"]["max_abs_diff"], 0.0)
        self.assertEqual(culprit["first_differing_suboperation"], "routed_experts")
        names = [s["name"] for s in culprit["suboperations"]]
        self.assertEqual(names[:3], ["norm", "gate_indices", "gate_scores"])
        by_name = {s["name"]: s for s in culprit["suboperations"]}
        for clean in ("norm", "gate_indices", "gate_scores"):
            self.assertTrue(by_name[clean]["bit_identical"], clean)
        self.assertEqual(by_name["routed_experts"]["engine_class"], "Scaled")
        self.assertIn("reference_dtype", by_name["routed_experts"])
        self.assertIn("engine_shape", by_name["routed_experts"])
        self.assertFalse(by_name["mixer_output"]["bit_identical"])
        # the first decode phase drills down as well, and only the injected layer is forced-divergent anywhere
        self.assertEqual(report["phases"][1]["layers"][3]["first_differing_suboperation"], "routed_experts")
        by_kind = report["forced_divergent_layers_by_kind"]
        for phase in ("prefill", "decode1"):
            self.assertEqual(by_kind[phase]["moe"]["divergent_layers"], [3])
            self.assertEqual(by_kind[phase]["mamba"]["divergent"], 0)
            self.assertEqual(by_kind[phase]["attention"]["divergent"], 0)
        # the free walk carries the deviation to the logits
        self.assertGreater(report["phases"][0]["logits"]["free_vs_reference"]["max_abs_diff"], 0.0)

    def test_injected_attention_deviation_is_localised_to_the_mixer_output(self):
        def mutate(layers):
            layers[2].mixer.o_proj = Scaled(layers[2].mixer.o_proj)

        report = self.run_trace(decode=1, mutate=mutate)
        self.assertEqual(report["first_forced_divergence"], {"phase": "prefill", "layer": 2, "kind": "attention"})
        prefill = report["phases"][0]["layers"]
        self.assertTrue(all(e["forced"]["bit_identical"] for e in prefill[:2]))
        culprit = prefill[2]
        self.assertEqual(culprit["first_differing_suboperation"], "mixer_output")
        by_name = {s["name"]: s for s in culprit["suboperations"]}
        for clean in ("norm", "q_proj", "k_proj", "v_proj"):
            self.assertTrue(by_name[clean]["bit_identical"], clean)
        self.assertTrue(by_name["cache_state_before"]["bit_identical"])
        # the engine's attention is a subclass of the stock class that defers to it for a BF16 KVCache
        self.assertEqual(by_name["mixer_output"]["reference_class"], "NemotronHAttention")
        self.assertIn(by_name["mixer_output"]["engine_class"], ("NemotronHAttention", "StreamedAttention"))
        self.assertEqual(report["forced_divergent_layers_by_kind"]["prefill"]["moe"]["divergent"], 0)

    def test_injected_mamba_deviation_is_localised_to_the_mixer_output(self):
        def mutate(layers):
            layers[0].mixer.out_proj = Scaled(layers[0].mixer.out_proj)

        report = self.run_trace(decode=1, mutate=mutate)
        self.assertEqual(report["first_forced_divergence"], {"phase": "prefill", "layer": 0, "kind": "mamba"})
        culprit = report["phases"][0]["layers"][0]
        self.assertEqual(culprit["first_differing_suboperation"], "mixer_output")
        by_name = {s["name"]: s for s in culprit["suboperations"]}
        self.assertTrue(by_name["in_proj"]["bit_identical"])
        self.assertTrue(by_name["cache_state_after_mixer"]["bit_identical"])     # out_proj does not touch the state

    # (c) INT8 attention cache -------------------------------------------------------------------------------------
    def test_int8_kv_skips_the_attention_state_comparison(self):
        report = self.run_trace(decode=1, int8_kv=True, root=Path(self.dir) / "ck64")
        self.assertTrue(report["int8_kv"])
        for phase in report["phases"]:
            for entry in phase["layers"]:
                if entry["kind"] == "attention":
                    self.assertTrue(entry["state_comparison"].startswith("skipped"), entry["state_comparison"])
                    self.assertIsNone(entry["state_bit_identical"])
                    self.assertIsNone(entry["state_max_abs_diff"])
                elif entry["kind"] == "mamba":
                    self.assertEqual(entry["state_comparison"], "compared")
        # the drill-down of an attention layer (when INT8 makes it diverge) names the two cache classes
        for phase in report["phases"]:
            for entry in phase["layers"]:
                for step in entry.get("suboperations", ()):
                    if step["name"] == "cache_state_before" and entry["kind"] == "attention":
                        self.assertEqual((step["reference_class"], step["engine_class"]), ("KVCache", "AttentionCache"))
                        self.assertTrue(step["state_comparison"].startswith("skipped"))


if __name__ == "__main__":
    unittest.main()
