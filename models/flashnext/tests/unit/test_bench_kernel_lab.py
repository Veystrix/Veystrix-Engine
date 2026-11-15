"""Stage 4 kernel lab: arithmetic, helpers, synthetic pool, and argument handling.

Timings are fabricated; the GPU is used only where the existing metal runtime
unit tests use it (tiny pools and one chain of trivial kernels).
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest

import mlx.core as mx
import numpy as np

from models.flashnext.metal_runtime import MetalMoEExecutor
from models.flashnext.tests.bench import bench_kernel_lab as lab
from models.flashnext.tests.bench import kernel_corpus as kc


def measured(independent, dependent=None, extra=0.0):
    """Minimal shape of a `measure()` result as the model and annotations read it."""
    dependent = independent if dependent is None else dependent
    return {"independent": {"eval_us": {"median": independent}, "total_us": {"median": independent + extra}},
            "dependent": {"net_eval_us": {"median": dependent}, "net_total_us": {"median": dependent + extra}}}


def entry(kind, in_width, out_width, calls, weight_bytes=1000, operation=None):
    return {"kind": kind, "in_width": in_width, "out_width": out_width, "calls_per_step": calls,
            "weight_bytes": weight_bytes, "operation": operation or kc.DENSE}


def fabricated():
    corpus = {"decode": [
        entry("dense", 2560, 1, 48), entry("dense", 2560, 640, 100, 1_000_000),
        entry("dense", 2560, 248320, 0.5, 400_000_000), entry("gather", 2560, 640, 2, 10_000_000, kc.GATHER),
        entry("metal", 2560, 640, 47, operation="metal.gate_qmv"), entry("metal", 2560, 640, 47, operation="metal.swiglu"),
        entry("metal", 640, 2560, 47, operation="metal.fused_down"),
        entry("dense", 2560, 6144, 36, 9_000_000)]}
    dense = {
        "dense_2560x1": {"entry": corpus["decode"][0], "by_rows": {1: measured(2.0, 10.0), 2: measured(2.5, 11.0)}},
        "dense_2560x640": {"entry": corpus["decode"][1], "by_rows": {1: measured(30.0, 40.0), 2: measured(50.0, 60.0)}},
        "dense_2560x248320": {"entry": corpus["decode"][2], "by_rows": {1: measured(4000.0, 4100.0),
                                                                     2: measured(4100.0, 4200.0)}},
        "dense_2560x6144": {"entry": corpus["decode"][7], "by_rows": {1: measured(80.0, 90.0), 2: measured(75.0, 85.0)}},
        "gather_2560x640": {"entry": corpus["decode"][3], "by_rows": {1: measured(100.0, 110.0)}},
    }
    def moe_record(ind, dep, distinct, rows):
        record = measured(ind, dep)
        record.update(bytes_distinct=distinct * 3_072_000, bytes_addressed=rows * 10 * 3_072_000, launches_per_call=3)
        return record

    moe = {
        "pool bf16 fused M=1 single": moe_record(300.0, 330.0, 10, 1),
        "pool bf16 fused M=2 same": moe_record(500.0, 560.0, 10, 2),
        "pool bf16 fused M=2 disjoint": moe_record(600.0, 620.0, 20, 2),
        "streamed bf16 fused M=1 single": moe_record(290.0, 320.0, 10, 1),
    }
    results = {"floor": {"metal_copy4": measured(2.0, 3.0)}, "dense": dense, "moe": moe,
               "gdn": {"_geometry": {"layers_per_step": 36}, "M=1": measured(50.0, 60.0), "M=2": measured(90.0, 100.0)}}
    return corpus, results


class LinearityTests(unittest.TestCase):
    def test_perfectly_linear_chain(self):
        check = lab.linearity_check(100.0, 10, 200.0, 20)
        self.assertAlmostEqual(check["ratio"], 2.0)
        self.assertAlmostEqual(check["marginal"], 10.0)
        self.assertAlmostEqual(check["intercept"], 0.0)
        self.assertTrue(check["ok"])

    def test_fixed_submit_cost_shows_as_intercept_and_can_fail_the_check(self):
        check = lab.linearity_check(130.0, 10, 230.0, 20)      # 30 fixed + 10 per repetition
        self.assertAlmostEqual(check["intercept"], 30.0)
        self.assertAlmostEqual(check["marginal"], 10.0)
        self.assertAlmostEqual(check["average"], 13.0)
        self.assertFalse(check["ok"])
        self.assertLess(check["deviation"], -0.10)

    def test_sublinear_and_superlinear_are_both_flagged(self):
        self.assertFalse(lab.linearity_check(100.0, 10, 100.0, 20)["ok"])      # time did not grow: folded or cached
        self.assertFalse(lab.linearity_check(100.0, 10, 260.0, 20)["ok"])
        self.assertTrue(lab.linearity_check(100.0, 10, 209.0, 20)["ok"])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            lab.linearity_check(100.0, 20, 200.0, 10)
        with self.assertRaises(ValueError):
            lab.linearity_check(0.0, 10, 200.0, 20)

    def test_stat(self):
        s = lab.stat([3, 1, 2, 10, 4])
        self.assertEqual((s["median"], s["min"], s["max"]), (3.0, 1.0, 10.0))


class TokenModelTests(unittest.TestCase):
    def setUp(self):
        self.corpus, self.results = fabricated()

    def test_m1_independent_arithmetic_by_class(self):
        model = lab.token_model(self.corpus, self.results, rows_list=(1,))
        body = model["modes"]["independent"]
        self.assertAlmostEqual(body["floor_us_per_launch"], 2.0)
        v = body["by_rows"][1]["single"]
        c = v["classes"]
        self.assertAlmostEqual(c["dense_tiny"]["us"], 48 * 2.0)
        self.assertAlmostEqual(c["dense_other"]["us"], 100 * 30.0 + 36 * 80.0 + 0.5 * 4000.0)
        self.assertAlmostEqual(c["gather_layer0"]["us"], 2 * 100.0)
        self.assertAlmostEqual(c["moe"]["us"], 47 * 300.0)
        self.assertAlmostEqual(c["gdn"]["us"], 36 * 50.0)
        expected = 48 * 2 + (3000 + 2880 + 2000) + 200 + 47 * 300 + 36 * 50
        self.assertAlmostEqual(v["total_ms"], expected / 1000.0)
        self.assertAlmostEqual(c["moe"]["ms"], 47 * 300.0 / 1000.0)

    def test_floor_only_counts_launches_times_the_trivial_kernel(self):
        v = lab.token_model(self.corpus, self.results, rows_list=(1,))["modes"]["independent"]["by_rows"][1]["single"]
        launches = 48 + 100 + 36 + 0.5 + 2 + 47 * 3 + 36 * 2
        self.assertAlmostEqual(v["launches"], launches)
        self.assertAlmostEqual(v["floor_only_ms"], launches * 2.0 / 1000.0)
        self.assertAlmostEqual(v["classes"]["moe"]["floor_ms"], 47 * 3 * 2.0 / 1000.0)

    def test_dependent_mode_uses_net_figures_and_its_own_floor(self):
        model = lab.token_model(self.corpus, self.results, rows_list=(1,))
        body = model["modes"]["dependent"]
        self.assertAlmostEqual(body["floor_us_per_launch"], 3.0)
        c = body["by_rows"][1]["single"]["classes"]
        self.assertAlmostEqual(c["dense_tiny"]["us"], 48 * 10.0)
        self.assertAlmostEqual(c["moe"]["us"], 47 * 330.0)

    def test_verification_rounds_use_m_row_timings_and_both_routing_cases(self):
        model = lab.token_model(self.corpus, self.results, rows_list=(1, 2))
        v2 = model["modes"]["independent"]["by_rows"][2]
        self.assertEqual(sorted(v2), ["disjoint", "same"])
        self.assertAlmostEqual(v2["same"]["classes"]["moe"]["us"], 47 * 500.0)
        self.assertAlmostEqual(v2["disjoint"]["classes"]["moe"]["us"], 47 * 600.0)
        self.assertAlmostEqual(v2["same"]["classes"]["dense_tiny"]["us"], 48 * 2.5)
        self.assertAlmostEqual(v2["same"]["classes"]["gdn"]["us"], 36 * 90.0)

    def test_gather_is_excluded_beyond_m1_and_said_so(self):
        v2 = lab.token_model(self.corpus, self.results, rows_list=(2,))["modes"]["independent"]["by_rows"][2]["same"]
        self.assertNotIn("gather_layer0", v2["classes"])
        self.assertTrue(any("gather_2560x640" in e for e in v2["excluded"]))

    def test_moe_variant_selects_the_streamed_timings(self):
        model = lab.token_model(self.corpus, self.results, rows_list=(1,), moe_variant="streamed bf16 fused")
        self.assertAlmostEqual(model["modes"]["independent"]["by_rows"][1]["single"]["classes"]["moe"]["us"], 47 * 290.0)

    def test_missing_moe_timing_is_reported_not_zero_filled_silently(self):
        del self.results["moe"]["pool bf16 fused M=2 disjoint"]
        v2 = lab.token_model(self.corpus, self.results, rows_list=(2,))["modes"]["independent"]["by_rows"][2]["disjoint"]
        self.assertNotIn("moe", v2["classes"])
        self.assertFalse(v2["complete"])
        self.assertTrue(any("moe" in e for e in v2["excluded"]))

    def test_unequal_metal_kernel_counts_are_an_error(self):
        self.corpus["decode"][5]["calls_per_step"] = 46
        with self.assertRaisesRegex(ValueError, "not called equally"):
            lab.token_model(self.corpus, self.results, rows_list=(1,))

    def test_model_states_what_it_omits(self):
        model = lab.token_model(self.corpus, self.results, rows_list=(1,))
        text = " ".join(model["omitted"])
        for word in ("norms", "attention", "router", "sampling", "round trips"):
            self.assertIn(word, text)

    def test_total_metric_adds_host_build_time(self):
        corpus, results = fabricated()
        results["dense"]["dense_2560x1"]["by_rows"][1] = measured(2.0, 10.0, extra=1.0)
        eval_ms = lab.token_model(corpus, results, rows_list=(1,))["modes"]["independent"]["by_rows"][1]["single"]
        total_ms = lab.token_model(corpus, results, rows_list=(1,), metric="total")["modes"]["independent"]["by_rows"][1]["single"]
        self.assertAlmostEqual(total_ms["classes"]["dense_tiny"]["us"] - eval_ms["classes"]["dense_tiny"]["us"], 48 * 1.0)

    def test_annotate_derives_bandwidth_and_floor_multiples(self):
        corpus, results = fabricated()
        lab.annotate(results, results["floor"]["metal_copy4"])
        d = results["dense"]["dense_2560x640"]["by_rows"][1]["derived"]["independent"]
        self.assertAlmostEqual(d["gb_per_s"], 1_000_000 / 30.0 / 1e3)
        self.assertAlmostEqual(d["x_floor"], 30.0 / 2.0)
        self.assertAlmostEqual(results["dense"]["dense_2560x640"]["by_rows"][1]["derived"]["dependent"]["x_floor"], 40.0 / 3.0)
        m = results["moe"]["pool bf16 fused M=2 same"]["derived"]["independent"]
        self.assertAlmostEqual(m["gb_per_s_distinct"], 10 * 3_072_000 / 500.0 / 1e3)
        self.assertAlmostEqual(m["gb_per_s_addressed"], 2 * 10 * 3_072_000 / 500.0 / 1e3)
        self.assertAlmostEqual(m["x_floor_per_launch"], 500.0 / 3 / 2.0)

    def test_moe_scaling_reports_growth_and_per_row_cost(self):
        corpus, results = fabricated()
        scaling = lab.moe_scaling(results)
        same = scaling["same"][2]["independent"]
        self.assertAlmostEqual(same["x_m1"], 500.0 / 300.0)
        self.assertAlmostEqual(same["us_per_row"], 250.0)
        self.assertAlmostEqual(scaling["disjoint"][2]["dependent"]["x_m1"], 620.0 / 330.0)
        self.assertEqual(scaling["same"][1]["independent"]["x_m1"], 1.0)


class HelperTests(unittest.TestCase):
    def test_routing_same_and_disjoint(self):
        same = lab.routing_for(4, "same")
        self.assertEqual(len({e for row in same for e in row}), lab.TOP_K)
        disjoint = lab.routing_for(8, "disjoint")
        self.assertEqual(len({e for row in disjoint for e in row}), 8 * lab.TOP_K)
        self.assertEqual(max(e for row in disjoint for e in row), 79)
        self.assertEqual(lab.routing_for(1, "single"), [list(range(10))])
        with self.assertRaises(ValueError):
            lab.routing_for(2, "other")

    def test_production_configuration_is_measured_first(self):
        first = lab.moe_configs()[0]
        self.assertEqual((first.mode, first.rows, first.fused), ("pool", 1, True))
        self.assertEqual(first.dtype, mx.bfloat16)
        self.assertEqual(first.label, "pool bf16 fused M=1 single")
        self.assertEqual(lab.MoEConfig("streamed", mx.float32, 4, "disjoint", False).label,
                         "streamed f32 unfused M=4 disjoint")

    def test_compare_accepts_equal_and_rejects_perturbed_and_scale_hidden_errors(self):
        a = mx.array(np.linspace(-1, 1, 64, dtype=np.float32)).reshape(1, 64)
        self.assertTrue(lab.compare(a, a)["ok"])
        self.assertFalse(lab.compare(a * 1.10, a)["ok"])
        self.assertFalse(lab.compare(a, a.reshape(2, 32))["ok"])
        # small outputs: a 5% error is inside atol everywhere, but the relative RMS check still catches it
        small = a * 1e-3
        self.assertFalse(lab.compare(small * 1.05, small)["ok"])

    def test_float32_dense_skip_rule_protects_memory(self):
        self.assertIsNone(lab.f32_skip_reason({"in_width": 2560, "out_width": 640}))
        self.assertIn("exceed", lab.f32_skip_reason({"in_width": 2560, "out_width": 248320}))

    def test_settle_returns_at_once_when_pressure_is_at_baseline(self):
        t = lab.Thermal()
        from unittest import mock
        with mock.patch.object(lab, "thermal_level", return_value=0):
            self.assertLess(lab.settle(t, 0, 5.0), 0.5)

    def test_thermal_summary_flags_nonzero_start_and_rise(self):
        t = lab.Thermal()
        t.samples = [{"label": "a", "level": 0}, {"label": "b", "level": 2}, {"label": "c", "level": 1}]
        s = t.summary()
        self.assertTrue(s["rose_during_run"])
        self.assertFalse(s["nonzero_at_start"])
        self.assertEqual((s["start"], s["end"], s["max"]), (0, 1, 2))
        t.samples = [{"label": "a", "level": 1}, {"label": "b", "level": 1}]
        self.assertTrue(t.summary()["nonzero_at_start"])
        self.assertFalse(t.summary()["rose_during_run"])

    def test_budget_records_what_it_skipped(self):
        budget = lab.Budget(0.0)
        self.assertTrue(budget.exhausted("a config"))
        self.assertEqual(budget.skipped, ["a config"])
        self.assertFalse(lab.Budget(3600.0).exhausted("x"))

    def test_audit_compares_launch_rows_with_corpus_rows(self):
        rows = [{"operation": "metal.gate_qmv", "operand_shapes": [[1, 2560]], "options": {"grid": [32, 160, 10]}}]
        corpus = {"decode": [{"kind": "metal", "operation": "metal.gate_qmv", "operand_shapes": [[1, 2560]],
                              "options": {"grid": [32, 160, 10]}}]}
        self.assertTrue(lab.audit_against_corpus({"rows": rows}, corpus)["matches"])
        corpus["decode"][0]["options"] = {"grid": [32, 80, 10]}
        self.assertFalse(lab.audit_against_corpus({"rows": rows}, corpus)["matches"])

    def test_linearity_summary_finds_nested_checks(self):
        good, bad = lab.linearity_check(100, 10, 200, 20), lab.linearity_check(130, 10, 230, 20)
        summary = lab.linearity_summary({"a": {"independent": {"linearity": good}}, "b": {"x": {"linearity": bad}}})
        self.assertEqual((summary["checks"], summary["outside_tolerance"]), (2, 1))
        self.assertTrue(summary["worst"][0]["path"].startswith("b"))

    def test_gdn_geometry_follows_the_corpus_and_refuses_a_corpus_without_it(self):
        geometry = lab.gdn_geometry(kc.load())
        self.assertEqual(geometry["layers_per_step"], 36)
        self.assertEqual((geometry["Hv"], geometry["Dv"], geometry["Hk"], geometry["Dk"]), (48, 128, 16, 128))
        corpus = kc.load()
        corpus["decode"] = [e for e in corpus["decode"] if not (e["kind"] == "dense" and e["out_width"] == 6144)]
        with self.assertRaisesRegex(ValueError, "lacks the GDN projections"):
            lab.gdn_geometry(corpus)


class RouteWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.windows, cls.meta = lab.load_route_windows()

    def write(self, mutate):
        import json
        import tempfile

        data = json.loads(lab.WINDOWS_FIXTURE.read_text())
        mutate(data)
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        self.addCleanup(os.unlink, handle.name)
        json.dump(data, handle)
        handle.close()
        return handle.name

    def test_the_committed_fixture_has_24_windows_of_m_rows_of_eight_ids(self):
        self.assertEqual(sorted(self.windows), [2, 4, 8])
        for rows, windows in self.windows.items():
            self.assertEqual(len(windows), 24)
            for window in windows:
                self.assertEqual(len(window), rows)
                self.assertTrue(all(len(row) == lab.BLOCK_TOP_K for row in window))
        self.assertEqual(self.meta["windows"], {2: 24, 4: 24, 8: 24})
        self.assertEqual(len(self.meta["sha256"]), 64)

    def test_every_window_maps_onto_the_synthetic_pool_unchanged(self):
        largest = 0
        for windows in self.windows.values():
            for window in windows:
                self.assertEqual(lab.map_to_slots(window), window)
                largest = max(largest, len({e for row in window for e in row}))
        self.assertEqual(largest, 47)
        self.assertLessEqual(largest, lab.POOL_EXPERTS)

    def test_ids_past_the_pool_are_refused(self):
        with self.assertRaises(ValueError):
            lab.map_to_slots([[0, 1, 80]])
        with self.assertRaises(ValueError):
            lab.map_to_slots([[0, 1, 2]], experts=2)
        self.assertEqual(lab.map_to_slots([[0, 79]]), [[0, 79]])

    def test_unique_per_pair_is_the_mean_over_windows(self):
        windows = [[[0, 1], [0, 1]], [[0, 1], [2, 3]]]
        self.assertAlmostEqual(lab.unique_per_pair(windows), (2 / 4 + 4 / 4) / 2)
        for rows, recorded in self.meta["fixture_unique_per_pair"].items():
            self.assertAlmostEqual(lab.unique_per_pair(self.windows[rows]), recorded, places=3)

    def test_patterns_build_the_windows_each_chain_cycles_through(self):
        fixture = self.windows
        self.assertEqual(len(lab.block_windows(4, "trace", fixture)), 24)
        same, = lab.block_windows(4, "same", fixture)
        self.assertEqual(len({e for row in same for e in row}), lab.BLOCK_TOP_K)
        disjoint, = lab.block_windows(8, "disjoint", fixture)
        self.assertEqual(len({e for row in disjoint for e in row}), 64)
        self.assertEqual(lab.BLOCK_REPS % 24, 0)
        with self.assertRaises(ValueError):
            lab.block_windows(2, "other", fixture)

    def test_a_malformed_fixture_is_refused(self):
        def schema(data):
            data["schema_version"] = 2

        def missing(data):
            del data["by_rows"]["4"]

        def narrow(data):
            data["by_rows"]["2"]["windows"][0]["routes"][0].pop()

        def unique(data):
            data["by_rows"]["8"]["windows"][3]["unique"] += 1

        for mutate in (schema, missing, narrow, unique):
            with self.subTest(mutate=mutate.__name__), self.assertRaises(ValueError):
                lab.load_route_windows(self.write(mutate))


class BlockHelperTests(unittest.TestCase):
    def test_plan_arrays_are_uint32_and_cover_every_pair(self):
        host = lab.pool_routes([[0, 1, 2, 3], [0, 1, 4, 5]])
        self.assertTrue(all(value & lab.SLOT_BIT for value in host))
        routes, pairs = lab.plan_arrays(host, 2)
        self.assertEqual((routes.dtype, pairs.dtype), (mx.uint32, mx.uint32))
        live = [p for p in np.asarray(pairs).reshape(-1).tolist() if p != 0xFFFFFFFF]
        self.assertEqual(sorted(live), list(range(8)))
        self.assertEqual(lab.groups_mean([[[0, 1, 2, 3], [0, 1, 4, 5]]], 2), 6)

    def test_plan_cost_is_a_positive_microsecond_statistic(self):
        cost = lab.plan_cost([[[0, 1, 2, 3, 4, 5, 6, 7]] * 2], 2, batches=5, rounds=2)
        self.assertEqual(len(cost["values"]), 5)
        self.assertGreater(cost["median"], 0.0)

    def test_bit_identity_compares_bytes_dtype_and_shape(self):
        a = mx.array([1.0, 2.0, 3.0]).astype(mx.bfloat16)
        self.assertTrue(lab.bit_identical(a, mx.array([1.0, 2.0, 3.0]).astype(mx.bfloat16)))
        self.assertFalse(lab.bit_identical(a, mx.array([1.0, 2.0, 3.01]).astype(mx.float32).astype(mx.bfloat16)
                                           + mx.array([0.0, 0.0, 0.02]).astype(mx.bfloat16)))
        self.assertFalse(lab.bit_identical(a, a.astype(mx.float32)))
        self.assertFalse(lab.bit_identical(a, a.reshape(1, 3)))
        one = mx.array([1.0], dtype=mx.float32)
        self.assertFalse(lab.bit_identical(one, mx.array([1.0000001], dtype=mx.float32)))

    def test_bit_identity_refuses_integer_dtypes(self):
        with self.assertRaises(TypeError):
            lab.bit_identical(mx.array([1]), mx.array([1]))


def block_results(pair, block2, plan, layers_rows=(2, 4)):
    """Fabricated `results["block"]` records for the verification-round arithmetic."""
    out = {}
    for rows in layers_rows:
        record = {"per_pair": measured(pair * rows, pair * rows + 10, extra=5.0), "block": {}}
        for r in lab.BLOCK_ROWS:
            entry_ = measured(block2 * rows, block2 * rows + 10, extra=5.0)
            entry_["plan_us"] = {"median": plan}
            record["block"][str(r)] = entry_
        out[f"M={rows} trace"] = record
    return out


class VerificationRoundTests(unittest.TestCase):
    def setUp(self):
        self.corpus, self.results = fabricated()
        self.results["dense"]["dense_2560x1"]["by_rows"][4] = measured(3.0, 12.0)
        self.results["block"] = block_results(pair=100.0, block2=80.0, plan=20.0)
        for name in ("dense_2560x640", "dense_2560x248320", "dense_2560x6144"):
            self.results["dense"][name]["by_rows"][4] = measured(60.0, 70.0)
        self.results["gdn"]["M=4"] = measured(120.0, 130.0)
        self.model = lab.token_model(self.corpus, self.results, rows_list=(2, 4))

    def test_round_states_both_totals_and_the_difference(self):
        rounds = lab.block_rounds(self.corpus, self.results, self.model)
        r = rounds["independent"][2]
        classes = self.model["modes"]["independent"]["by_rows"][2]["same"]["classes"]
        other = sum(c["ms"] for name, c in classes.items() if name != "moe")
        self.assertAlmostEqual(r["other_classes_ms"], other)
        self.assertAlmostEqual(r["moe_per_pair_ms"], 47 * 200.0 / 1000)
        self.assertAlmostEqual(r["moe_block_kernels_ms"], 47 * 160.0 / 1000)
        self.assertAlmostEqual(r["host_plan_ms"], 47 * 20.0 / 1000)
        self.assertAlmostEqual(r["round_per_pair_ms"], other + 47 * 200.0 / 1000)
        self.assertAlmostEqual(r["round_block_ms"], other + 47 * 180.0 / 1000)
        self.assertAlmostEqual(r["difference_ms"], -47 * 20.0 / 1000)
        self.assertAlmostEqual(r["difference_pct"], (r["round_block_ms"] / r["round_per_pair_ms"] - 1) * 100)

    def test_dependent_mode_uses_the_net_figures(self):
        r = lab.block_rounds(self.corpus, self.results, self.model)["dependent"][4]
        self.assertAlmostEqual(r["moe_per_pair_ms"], 47 * 410.0 / 1000)
        self.assertAlmostEqual(r["moe_block_kernels_ms"], 47 * 330.0 / 1000)

    def test_a_block_that_loses_shows_a_positive_difference(self):
        self.results["block"] = block_results(pair=100.0, block2=105.0, plan=20.0)
        r = lab.block_rounds(self.corpus, self.results, self.model)["independent"][2]
        self.assertGreater(r["difference_ms"], 0.0)
        self.assertGreater(r["difference_pct"], 0.0)

    def test_total_metric_adds_build_time_to_both_sides(self):
        model = lab.token_model(self.corpus, self.results, rows_list=(2,), metric="total")
        r = lab.block_rounds(self.corpus, self.results, model)["independent"][2]
        self.assertAlmostEqual(r["moe_per_pair_ms"], 47 * 205.0 / 1000)

    def test_missing_measurements_are_left_out_not_zero_filled(self):
        del self.results["block"]["M=4 trace"]
        rounds = lab.block_rounds(self.corpus, self.results, self.model)
        self.assertEqual(sorted(rounds["independent"]), [2])
        self.results["block"] = {}
        self.assertEqual(lab.block_rounds(self.corpus, self.results, self.model)["independent"], {})

    def test_a_corpus_without_metal_rows_has_no_rounds(self):
        corpus = {"decode": [e for e in self.corpus["decode"] if e["kind"] != "metal"]}
        self.assertEqual(lab.block_rounds(corpus, self.results, self.model), {})
        self.assertIsNone(lab.moe_layer_count(corpus))
        self.assertEqual(lab.moe_layer_count(self.corpus), 47)


class SectionHandlingTests(unittest.TestCase):
    def test_a_raising_section_is_recorded_and_does_not_propagate(self):
        budget, failures = lab.Budget(3600.0), []

        def broken():
            raise RuntimeError("MLX 0.32.4 quantized.h changed")

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(lab.run_section("moe", broken, budget, failures))
        self.assertEqual(budget.skipped, [{"section": "moe", "exception": "RuntimeError: MLX 0.32.4 quantized.h changed"}])
        self.assertEqual(len(failures), 1)
        self.assertIn("moe", failures[0])

    def test_systemexit_from_a_section_is_recorded_too(self):
        budget, failures = lab.Budget(3600.0), []
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(lab.run_section("block", lambda: sys.exit("custom Metal is unavailable"), budget, failures))
        self.assertEqual(budget.skipped[0]["exception"], "SystemExit: custom Metal is unavailable")

    def test_keyboard_interrupt_is_not_swallowed(self):
        def interrupt():
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            lab.run_section("dense", interrupt, lab.Budget(3600.0), [])

    def test_a_finishing_section_records_nothing(self):
        budget, failures, ran = lab.Budget(3600.0), [], []
        self.assertTrue(lab.run_section("gdn", lambda: ran.append(1), budget, failures))
        self.assertEqual((budget.skipped, failures, ran), ([], [], [1]))

    def test_partial_results_survive_a_failure_in_the_same_section(self):
        results, budget, failures = {}, lab.Budget(3600.0), []

        def partial():
            results["M=2 same"] = {"done": True}
            raise ValueError("second config failed")

        with contextlib.redirect_stdout(io.StringIO()):
            lab.run_section("block", partial, budget, failures)
        self.assertEqual(results, {"M=2 same": {"done": True}})

    def test_arguments_start_with_no_failures_and_know_the_block_section(self):
        args = lab.parse_args([])
        self.assertEqual(args.failures, [])
        self.assertIn("block", args.sections)
        self.assertEqual(lab.parse_args(["--only", "block"]).sections, ("block",))
        self.assertEqual(args.windows, lab.WINDOWS_FIXTURE)


class DriftWordingTests(unittest.TestCase):
    def test_slower_at_the_end_reads_as_slower(self):
        text = lab.drift_wording(12.0)
        self.assertIn("SLOWER", text)
        self.assertNotIn("FASTER", text)
        self.assertIn("12.0%", text)

    def test_faster_at_the_end_reads_as_faster(self):
        text = lab.drift_wording(-8.0)
        self.assertIn("FASTER", text)
        self.assertNotIn("SLOWER", text)
        self.assertIn("8.0%", text)

    def test_small_drift_has_no_warning(self):
        for drift in (-5.0, -1.0, 0.0, 5.0):
            self.assertEqual(lab.drift_wording(drift), "")

    def test_the_canary_docstring_states_the_unit(self):
        self.assertIn("microseconds per call", lab.canary.__doc__)
        self.assertIn("higher end figure", lab.canary.__doc__)


class ArgumentTests(unittest.TestCase):
    def test_defaults(self):
        args = lab.parse_args([])
        self.assertEqual(args.corpus, kc.DEFAULT_FIXTURE)
        self.assertEqual(args.sections, lab.SECTIONS)
        self.assertTrue(args.json)
        self.assertEqual((args.ms, args.chains, args.reps, args.label), (25.0, 5, None, ""))

    def test_flags(self):
        args = lab.parse_args(["--only", "dense", "--only", "gdn", "--no-json", "--label", "mlx-0.32.2-pinned",
                               "--reps", "64", "--ms", "10", "--corpus", "x.json", "--chains", "7"])
        self.assertEqual(args.sections, ("dense", "gdn"))
        self.assertFalse(args.json)
        self.assertEqual((args.label, args.reps, args.ms, args.chains), ("mlx-0.32.2-pinned", 64, 10.0, 7))
        self.assertEqual(str(args.corpus), "x.json")

    def test_invalid_values_are_refused(self):
        for argv in (["--only", "nonsense"], ["--chains", "4"], ["--ms", "0"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                lab.parse_args(argv)


def _metal_available():
    return MetalMoEExecutor(lab.TOP_K, lab.HIDDEN, lab.TOP_K).available


class SyntheticPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("GITHUB_ACTIONS") == "true" or sys.platform != "darwin":
            raise unittest.SkipTest("custom Metal integration needs a local Apple GPU")
        cls.pool = lab.SyntheticPool(3)

    @classmethod
    def tearDownClass(cls):
        cls.pool.close()

    def test_records_sit_at_header_plus_stride_in_the_slab_layout(self):
        layout = lab.slab_layout()
        self.assertEqual(self.pool.stride, layout.record_stride)
        self.assertEqual(layout.record_stride, 3_072_000)
        self.assertFalse(self.pool.array[:lab.POOL_HEADER].any())
        self.assertEqual(self.pool.buffer_mx.dtype, mx.uint32)
        self.assertGreaterEqual(self.pool.buffer_mx.size * 4, lab.POOL_HEADER + 3 * self.pool.stride)

    def test_every_part_reads_back_byte_exact(self):
        for expert in range(3):
            for projection in lab.PROJECTIONS:
                w, s, b = lab.expert_projection(expert, projection)
                for part, array in (("weight", w), ("scales", s), ("biases", b)):
                    raw = array if part == "weight" else array.view(mx.uint16)
                    expected = np.asarray(raw).view(np.uint8).reshape(-1)
                    np.testing.assert_array_equal(self.pool.read(expert, projection, part), expected)

    def test_part_sizes_match_the_layout_constants(self):
        sizes = {(p, part): self.pool.read(0, p, part).size for p in lab.PROJECTIONS
                 for part in ("weight", "scales", "biases")}
        self.assertEqual(sizes[("gate_proj", "weight")], 819_200)
        self.assertEqual(sizes[("gate_proj", "scales")], 102_400)
        self.assertEqual(sizes[("down_proj", "biases")], 102_400)

    def test_executor_reference_path_accepts_records_read_back_from_the_pool(self):
        projections = {}
        for projection, (out_width, in_width) in lab.PROJECTIONS.items():
            parts = ([], [], [])
            for expert in range(3):
                w = np.array(self.pool.read(expert, projection, "weight")).view(np.uint32)
                parts[0].append(mx.array(w.reshape(out_width, in_width // 8)))
                for index, part in ((1, "scales"), (2, "biases")):
                    raw = np.array(self.pool.read(expert, projection, part)).view(np.uint16)
                    parts[index].append(mx.array(raw.reshape(out_width, in_width // lab.GROUP)).view(mx.bfloat16))
            projections[projection] = tuple(mx.stack(p) for p in parts)
        x = (mx.random.normal((1, lab.HIDDEN)) * lab.X_STD).astype(mx.bfloat16)
        routes = mx.arange(3, dtype=mx.uint32).reshape(1, 3)
        scores = mx.array([[0.5, 0.3, 0.2]], dtype=mx.bfloat16)
        shared = mx.zeros((1, lab.HIDDEN), dtype=mx.bfloat16)
        reference = MetalMoEExecutor(3, lab.HIDDEN, 3, backend="reference", group_size=lab.GROUP)
        actual = reference.execute(x, routes, projections, scores=scores, shared_y=shared)
        expected = reference.execute(x, routes, lab.routed_projections([0, 1, 2]), scores=scores, shared_y=shared)
        mx.eval(actual, expected)
        self.assertEqual(reference.last_path, "reference")
        self.assertEqual(actual.shape, (1, lab.HIDDEN))
        np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)), np.asarray(expected.astype(mx.float32)))


@unittest.skipUnless(sys.platform == "darwin" and os.environ.get("GITHUB_ACTIONS") != "true",
                     "custom Metal integration needs a local Apple GPU")
class MetalVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _metal_available():
            raise unittest.SkipTest("custom Metal unavailable")
        cls.pool = lab.SyntheticPool(20)

    @classmethod
    def tearDownClass(cls):
        cls.pool.close()

    def verify(self, config):
        executor = MetalMoEExecutor(lab.TOP_K, lab.HIDDEN, lab.TOP_K, group_size=lab.GROUP, fused_up_swiglu=config.fused)
        experts, scores, shared = lab.build_moe_inputs(config)
        call, _ = lab.moe_call_factory(executor, config, experts, scores, shared, self.pool, None)
        return call, experts, scores, shared, lab.verify_moe(call, config, experts, scores, shared)

    def test_executor_over_the_synthetic_pool_matches_the_reference(self):
        for dtype in (mx.bfloat16, mx.float32):
            with self.subTest(dtype=str(dtype)):
                config = lab.MoEConfig("pool", dtype, 2, "disjoint", True)
                result = self.verify(config)[-1]
                self.assertTrue(result["ok"], result)
                self.assertLess(result["rel_rms"], 1e-2)

    def test_a_corrupted_record_is_refused(self):
        config = lab.MoEConfig("pool", mx.bfloat16, 2, "disjoint", True)
        base = lab.POOL_HEADER + 3 * self.pool.stride + lab.slab_layout().offset("gate_proj", "weight")
        saved = np.array(self.pool.array[base:base + 819_200])
        self.pool.array[base:base + 819_200] = 0x5A
        try:
            result = self.verify(config)[-1]
        finally:
            self.pool.array[base:base + 819_200] = saved
        self.assertFalse(result["ok"])

    def test_launch_audit_counts_three_kernels_for_the_fused_executor(self):
        config = lab.MoEConfig("pool", mx.bfloat16, 1, "single", True)
        experts, scores, shared = lab.build_moe_inputs(config)
        executor = MetalMoEExecutor(lab.TOP_K, lab.HIDDEN, lab.TOP_K, group_size=lab.GROUP, fused_up_swiglu=True)
        call, _ = lab.moe_call_factory(executor, config, experts, scores, shared, self.pool, None)
        audit = lab.launch_audit(call, config)
        self.assertEqual(audit["metal_launches"], 3)
        self.assertEqual(audit["kernels"], ["metal.fused_down", "metal.gate_qmv", "metal.swiglu"])


@unittest.skipUnless(sys.platform == "darwin" and os.environ.get("GITHUB_ACTIONS") != "true",
                     "needs a local Apple GPU")
class MeasureTests(unittest.TestCase):
    def test_measure_returns_every_figure_and_a_linear_chain(self):
        case = lab.floor_cases()[0]
        result = lab.measure(case, target_ms=1.0, chains=5, fixed_reps=64, roundtrip_samples=5)
        self.assertEqual(result["reps"], 64)
        for key in ("eval_us", "build_us", "total_us", "linearity"):
            self.assertIn(key, result["independent"])
        self.assertEqual(len(result["independent"]["eval_us"]["values"]), 5)
        self.assertIn("net_eval_us", result["dependent"])
        self.assertTrue(result["dependent"]["link_is_free"])
        self.assertGreater(result["roundtrip_us"]["median"], 0.0)
        self.assertIn("never to be summed", result["roundtrip_note"])
        self.assertEqual(result["independent"]["linearity"]["reps"], [64, 128])

    def test_dense_dependent_chain_prices_its_link_separately(self):
        entry_ = {"in_width": 64, "out_width": 32}
        weights = lab.synthetic_q4(32, 64)
        result = lab.measure(lab.dense_case(entry_, weights, 1, mx.bfloat16), target_ms=1.0, chains=5, fixed_reps=32,
                             roundtrip_samples=5)
        dep = result["dependent"]
        self.assertFalse(dep["link_is_free"])
        self.assertGreater(dep["link_eval_us"]["median"], 0.0)
        self.assertGreaterEqual(dep["gross_eval_us"]["median"], dep["net_eval_us"]["median"])

    def test_nonfinite_dependent_chain_is_refused(self):
        w, s, b = lab.synthetic_q4(32, 64)
        s = s * mx.array(float("nan"), dtype=mx.bfloat16)
        case = lab.dense_case({"in_width": 64, "out_width": 32}, (w, s, b), 1, mx.bfloat16)
        with self.assertRaises(FloatingPointError):
            lab.measure(case, target_ms=1.0, chains=5, fixed_reps=8)

    def test_synthetic_weights_are_bounded(self):
        w, s, b = lab.synthetic_q4(4096, 64)
        self.assertEqual((w.shape, s.dtype), ((4096, 8), mx.bfloat16))
        with self.assertRaises(MemoryError):
            lab.synthetic_q4(2_000_000, 2560)


if __name__ == "__main__":
    unittest.main()
