import contextlib
import gzip
import io
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from models.flashnext.tests.bench import bench_pool_policy_trace as B
from models.shared.cost_aware import CostWeights, RouteFrequency


RECORDED_TRACE = "results/flashnext/20260922-221213-route-trace-manual/route-trace.json.gz"


def decode(*layers_experts):
    return [[0, "decode", i, layer, list(experts)] for i, (layer, experts) in enumerate(layers_experts)]


class PolicyTests(unittest.TestCase):
    def test_lru_hand_counted(self):
        # 2 slots, one layer. Accesses: [1,2] miss miss; [1] hit; [3] evicts 2; [2] misses again.
        events = decode((0, (1, 2)), (0, (1,)), (0, (3,)), (0, (2,)))
        hit, _ = B.replay(events, 2, "lru")
        self.assertEqual(hit, 1 / 5)

    def test_lru_cyclic_scan_larger_than_pool_never_hits_but_bip_does(self):
        # Textbook scan: 10 experts walked cyclically one per event against 8 slots, so LRU always evicts the next one.
        events = decode(*[(0, (i % 10,)) for i in range(200)])
        lru, _ = B.replay(events, 8, "lru")
        bip, _ = B.replay(events, 8, "bip8")
        self.assertEqual(lru, 0.0)
        self.assertGreater(bip, 0.1)

    def test_belady_is_an_upper_bound(self):
        events = decode(*[(i % 3, tuple((i * j) % 7 for j in range(1, 4))) for i in range(200)])
        best = B.replay(events, 6, "belady")[0]
        for name in ("lru", "arc", "bip8", "least-stale"):
            self.assertLessEqual(B.replay(events, 6, name)[0], best + 1e-12, name)

    def test_least_stale_preserves_ahead_layer_entries(self):
        events = decode((0, (1, 2)), (1, (1, 2)), (0, (3, 4)), (1, (1, 2)))
        self.assertEqual(B.replay(events, 4, "least-stale")[0], 2 / 8)

    def test_collision_counts_only_experts_evicted_before_their_use(self):
        events = [[0, "decode", i // 2, layer, experts] for i, (layer, experts) in
                  enumerate(((0, [1, 2]), (1, [1, 2]), (0, [3, 4]), (1, [1, 2])))]
        lru, least_stale = [], []
        B.replay(events, 2, "lru", collisions=lru)
        B.replay(events, 2, "least-stale", collisions=least_stale)
        self.assertEqual(lru, [(1, 1), (1, 2)])
        self.assertEqual(least_stale, lru)  # both layers need every slot at this capacity

    def test_runtime_profile_loads_with_explicit_expert_size(self):
        data = {"tokens": [{"token": 0, "layer_ids": [1], "routes": {"1": [2, 4]}}]}
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "profile.json")
            with open(path, "w") as handle:
                json.dump(data, handle)
            self.assertEqual(B.load_trace(path, 256), ([[0, "decode", 0, 1, [2, 4]]], 256))

    def test_least_stale_leaves_oversized_batch_to_streamed_path(self):
        pool = B.make_policy("least-stale", 2, [])
        pool.access(0, (1, 2))
        self.assertEqual(pool.access(1, (1, 2, 3)), 0)
        self.assertEqual(pool.access(0, (1, 2)), 2)

    def test_current_event_entries_are_not_victims(self):
        pool = B.Lru(2)
        pool.access(0, (1, 2))
        pool.access(0, (3, 4))                      # 4 entries wanted, 2 slots: first two are admitted, rest skipped
        self.assertLessEqual(len(pool.pool), 2)

    def test_prefill_only_queries_and_never_fills(self):
        events = [[0, "prefill", 0, 0, [1, 2]], [0, "decode", 0, 0, [1]], [0, "prefill", 0, 0, [1, 2]]]
        decode_rate, prefill_rate = B.replay(events, 4, "lru")
        self.assertEqual(decode_rate, 0.0)           # first decode access is a miss
        self.assertEqual(prefill_rate, 1 / 4)        # only expert 1 resident for the second prefill

    def test_all_policies_never_exceed_slots(self):
        events = decode(*[(i % 2, tuple((i + j) % 11 for j in range(4))) for i in range(300)])
        for name in ("lru", "arc", "bip8"):
            pool = B.make_policy(name, 5, [])
            for _, _, _, layer, experts in events:
                pool.access(layer, experts)
            size = len(pool.pool) if hasattr(pool, "pool") else len(pool.t1) + len(pool.t2)
            self.assertLessEqual(size, 5, name)

    def test_main_reads_trace_file(self):
        data = {"record_bytes": 1_000_000, "events": decode((0, (1, 2)), (0, (1, 2)))}
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "t.json.gz")
            with gzip.open(path, "wt") as handle:
                json.dump(data, handle)
            events, record = B.load_trace(path)
            self.assertEqual((len(events), record), (2, 1_000_000))
            B.main([path, "--gb", "0.01", "--policies", "lru"])


def synthetic_pinned_trace():
    rng = random.Random(7)
    events = [[0, "decode", t, layer, rng.sample(range(24), 4)] for t in range(40) for layer in range(6)]
    return events + [[0, "prefill", 0, 0, [1, 2, 3]]]


class ExistingPolicyPinTests(unittest.TestCase):
    """Values computed with the file before the cost-aware, byte and prefetch additions."""

    PINNED = {
        "lru": ((0.175, 1 / 3), (0.0, 0.0)),
        "arc": ((0.17604166666666668, 1 / 3), (0.0, 0.0)),
        "bip8": ((0.19166666666666668, 1 / 3), (0.036458333333333336, 0.0)),
        "s3fifo": ((0.18333333333333332, 2 / 3), (0.0, 0.0)),
        "least-stale": ((0.21770833333333334, 1 / 3), (0.0, 0.0)),
        "belady": ((0.47708333333333336, 0.0), (0.19479166666666667, 0.0)),
    }

    def test_synthetic_trace_numbers_are_unchanged(self):
        events = synthetic_pinned_trace()
        for name, (large, small) in self.PINNED.items():
            for slots, expected in ((30, large), (10, small)):
                got = B.replay(events, slots, name)
                self.assertAlmostEqual(got[0], expected[0], 12, (name, slots))
                self.assertAlmostEqual(got[1], expected[1], 12, (name, slots))

    @unittest.skipUnless(os.path.exists(RECORDED_TRACE), "recorded route trace not present")
    def test_recorded_trace_numbers_are_unchanged(self):
        events, _ = B.load_trace(RECORDED_TRACE)
        for name, slots, expected in (("lru", 622, 0.466), ("s3fifo", 622, 0.473), ("least-stale", 622, 0.429),
                                      ("s3fifo", 311, 0.325)):
            self.assertEqual(round(B.replay(events, slots, name)[0], 3), expected, (name, slots))


class CostAwareReplayTests(unittest.TestCase):
    def test_frequency_keeps_the_hot_record_that_lru_evicts(self):
        # 2 slots, one layer: 1 1 1 2 3 1. LRU evicts 1 for 3 (hits 2 of 6). Cost-aware scores 1 above 2 on
        # frequency (p 0.46 vs 0.30; reuse is 0 at this horizon), evicts 2, and hits 1 again (3 of 6).
        events = decode((0, (1,)), (0, (1,)), (0, (1,)), (0, (2,)), (0, (3,)), (0, (1,)))
        self.assertEqual(B.replay(events, 2, "lru")[0], 2 / 6)
        self.assertEqual(B.replay(events, 2, "cost-aware")[0], 3 / 6)

    def test_oversized_row_returns_zero_hits_and_leaves_state_untouched(self):
        pool = B.make_policy("cost-aware", 2, [])
        self.assertEqual(pool.access(0, (1, 2, 3)), 0)
        self.assertEqual((len(pool.slot_of), pool.policy._step), (0, 0))
        self.assertEqual(pool.access(0, (1, 2)), 0)
        self.assertEqual(pool.access(0, (1, 2)), 2)
        self.assertTrue(pool.contains((0, 1)) and not pool.contains((0, 3)))

    def test_never_exceeds_slots_and_free_list_is_consistent(self):
        pool = B.make_policy("cost-aware", 5, [])
        for i in range(300):
            pool.access(i % 3, tuple((i + j) % 11 for j in range(4)))
            self.assertEqual(len(pool.slot_of) + len(pool.free), 5)
            self.assertEqual(sorted([*pool.slot_of.values(), *pool.free]), list(range(5)))

    def test_weights_reach_the_policy(self):
        weights = CostWeights(recent=0.0, alpha=0.5)
        pool = B.make_policy("cost-aware", 3, [], weights)
        self.assertEqual((pool.policy.weights.recent, pool.frequency.alpha), (0.0, 0.5))

    def test_hit_rate_is_hits_over_routed_experts_and_rows_are_decode_only(self):
        events = [[0, "prefill", 0, 0, [1, 2]], *decode((0, (1, 2)), (0, (1, 2)))]
        self.assertEqual(B.replay(events, 4, "cost-aware"), (0.5, 0.0))


def two_token_events():
    return [[0, "decode", 0, 0, [1, 2]], [0, "decode", 0, 1, [1, 2]],
            [0, "decode", 1, 0, [1, 2]], [0, "decode", 1, 1, [1, 2]]]


class ByteMetricTests(unittest.TestCase):
    def test_demand_mb_per_token(self):
        # 4 slots hold all four (layer, expert) pairs of layer 0 and 1: 4 cold misses, 2 tokens, 1 MB records.
        detail = B.replay_detail(two_token_events(), 4, "lru")
        self.assertEqual((detail["misses"], detail["decode_hits"]), (4, 4))
        self.assertEqual(B.byte_metrics(detail, 1_000_000, B.decode_token_count(two_token_events()))["demand_mb_per_token"], 2.0)

    def test_token_is_a_distinct_turn_and_token_pair(self):
        events = [[0, "decode", 0, 0, [1]], [1, "decode", 0, 0, [1]], [1, "decode", 0, 1, [1]], [1, "prefill", 5, 0, [1]]]
        self.assertEqual(B.decode_token_count(events), 2)

    def test_zero_tokens_gives_zero_not_an_error(self):
        self.assertEqual(B.mb_per_token(5, 1_000_000, 0), 0.0)
        self.assertEqual(B.mb_per_token(3, 2_000_000, 4), 1.5)

    def test_prefetch_blocking_and_physical_arithmetic(self):
        detail = {"decode_hit_rate": 0.0, "prefill_resident_rate": 0.0, "decode_hits": 0, "decode_total": 8,
                  "misses": 8, "decode_capacity": 1,
                  "prefetch": {"issued": 6, "used": 5, "wasted": 1, "displaced": 0, "unused_at_end": 1}}
        out = B.byte_metrics(detail, 1_000_000, 4)["prefetch"]
        self.assertEqual((out["blocking_mb_per_token"], out["physical_mb_per_token"]), (0.75, 2.25))
        self.assertEqual((out["precision"], out["covered_share"]), (5 / 6, 5 / 8))


class PredictorTests(unittest.TestCase):
    def test_hand_counted_precision_and_recall(self):
        # alpha 0.5, one layer, rows {1,2} {1,3} {1,2}. Row 1 has an empty table. Row 2 (p1 .5, p2 .5): top-1 {1}
        # correct, top-2 {1,2} one correct. Row 3 (p1 .75, p2 .25, p3 .5): top-1 {1} correct, top-2 {1,3} one correct.
        # Persistence: row 2 predicts {1,2} (one correct), row 3 predicts {1,3} (one correct). Routed 2+2+2.
        events = decode((0, (1, 2)), (0, (1, 3)), (0, (1, 2)))
        out = B.predictor_quality(events, 0.5, ks=(1, 2))
        one, two = out["frequency"]["1"], out["frequency"]["2"]
        self.assertEqual((one["predicted"], one["correct"], one["routed"]), (2, 2, 6))
        self.assertEqual((one["precision"], one["recall"]), (1.0, 2 / 6))
        self.assertEqual((two["predicted"], two["correct"]), (4, 2))
        self.assertEqual((two["precision"], two["recall"]), (0.5, 1 / 3))
        persistence = out["persistence"]
        self.assertEqual((persistence["predicted"], persistence["correct"]), (4, 2))
        self.assertEqual(out["rows"], 3)

    def test_prefill_rows_are_ignored(self):
        events = [[0, "prefill", 0, 0, [1, 2]], *decode((0, (1, 2)))]
        self.assertEqual(B.predictor_quality(events, 0.5, ks=(1,))["routed"], 2)


def periodic(tokens, per_layer_experts):
    """Layers 0 and 1 alternate every token; ``per_layer_experts`` gives each layer's routed experts."""
    return [[0, "decode", t, layer, list(per_layer_experts[layer])] for t in range(tokens) for layer in (0, 1)]


class PrefetchModelTests(unittest.TestCase):
    WEIGHTS = CostWeights(alpha=0.5)

    def run_model(self, events, pool, **config):
        base = dict(slots=2, depth=1, width=1, min_p=0.5, per_layer=1, alpha=0.5) | config
        return B.replay_detail(events, pool, "lru", prefetch=B.PrefetchConfig(**base))

    def test_capacity_comes_out_of_the_pool_and_every_row_after_warmup_is_covered(self):
        # 3 slots, 2 staging: decode capacity 1, so the alternating (0,1) and (1,2) evict each other: 8 misses.
        # Without prefetch 3 slots hold both: 2 misses. Issue after rows 2..7 (6 reads); rows 3..7 demand a staged
        # record (5 used); the record issued after the last row is never demanded (1 wasted).
        events = periodic(4, {0: (1,), 1: (2,)})
        plain = B.replay_detail(events, 3, "lru")
        detail = self.run_model(events, 3)
        self.assertEqual((plain["misses"], plain["decode_capacity"]), (2, 3))
        self.assertEqual((detail["misses"], detail["decode_capacity"]), (8, 1))
        self.assertEqual(detail["prefetch"], {"issued": 6, "used": 5, "wasted": 1, "displaced": 0, "unused_at_end": 1})

    def test_per_row_cap_limits_issues(self):
        # Two experts per layer, 6 slots minus 4 staging = 2 decode slots, so all 16 routed experts miss.
        # R=2 issues both candidates: 12 issued, rows 3..7 demand both (10 used), 2 left at the end.
        # R=1 issues one per row: 6 issued, 5 used, 1 left.
        events = periodic(4, {0: (1, 2), 1: (3, 4)})
        two = self.run_model(events, 6, slots=4, width=2, per_layer=2)
        one = self.run_model(events, 6, slots=4, width=2, per_layer=1)
        self.assertEqual((two["misses"], one["misses"]), (16, 16))
        self.assertEqual({k: two["prefetch"][k] for k in ("issued", "used", "wasted")}, {"issued": 12, "used": 10, "wasted": 2})
        self.assertEqual({k: one["prefetch"][k] for k in ("issued", "used", "wasted")}, {"issued": 6, "used": 5, "wasted": 1})

    def test_width_limits_candidates_per_target_layer(self):
        events = periodic(4, {0: (1, 2), 1: (3, 4)})
        narrow = self.run_model(events, 6, slots=4, width=1, per_layer=2)
        self.assertEqual(narrow["prefetch"]["issued"], 6)

    def test_min_probability_blocks_issues(self):
        events = periodic(4, {0: (1,), 1: (2,)})
        self.assertEqual(self.run_model(events, 3, min_p=0.99)["prefetch"]["issued"], 0)

    def test_slot_is_taken_from_a_passed_entry_only(self):
        class Pool:
            def contains(self, key):
                return False

        frequency = RouteFrequency(4, 16, 0.5)
        for _ in range(2):
            frequency.observe(1, [3, 6])
        frequency.observe(2, [4])
        model = B.StagingModel(Pool(), B.PrefetchConfig(1, 1, 1, 0.5, 1), frequency, True)
        model.next_layer = {0: 1, 1: 2}
        model.after(0, [7])
        self.assertEqual(list(model.staged), [(1, 3)])
        model.after(0, [7])                      # layer 1 not visited yet: the staged entry is not displaceable
        self.assertEqual((model.issued, list(model.staged)), (1, [(1, 3)]))
        model.after(1, [5])                      # layer 1 passed without demand: (1, 3) is displaced for (2, 4)
        self.assertEqual((model.issued, model.displaced, list(model.staged)), (2, 1, [(2, 4)]))
        self.assertEqual(model.summary(), {"issued": 2, "used": 0, "wasted": 2, "displaced": 1, "unused_at_end": 1})

    def test_staging_larger_than_pool_is_refused(self):
        with self.assertRaises(ValueError):
            B.replay_detail(periodic(2, {0: (1,), 1: (2,)}), 2, "lru", prefetch=B.PrefetchConfig(slots=2))

    def test_cost_aware_shares_its_own_frequency_table(self):
        seen = []
        real = B.StagingModel

        def spy(pool, config, frequency, observes):
            seen.append((pool, frequency, observes))
            return real(pool, config, frequency, observes)

        with mock.patch.object(B, "StagingModel", spy):
            detail = B.replay_detail(periodic(6, {0: (1,), 1: (2,)}), 4, "cost-aware", weights=self.WEIGHTS,
                                     prefetch=B.PrefetchConfig(1, 1, 1, 0.5, 1))
            B.replay_detail(periodic(6, {0: (1,), 1: (2,)}), 4, "lru", weights=self.WEIGHTS,
                            prefetch=B.PrefetchConfig(1, 1, 1, 0.5, 1, 0.4))
        (pool, frequency, observes), (_, own, own_observes) = seen
        self.assertTrue(frequency is pool.frequency and not observes)
        self.assertEqual((own.alpha, own_observes), (0.4, True))
        self.assertEqual(detail["decode_capacity"], 3)


class TurnFilterAndOutputTests(unittest.TestCase):
    def events(self):
        return [[turn, "decode", t, 0, [turn + 1, 5]] for turn in range(3) for t in range(2)] + [[1, "prefill", 0, 0, [1]]]

    def test_filter_turns(self):
        events = self.events()
        self.assertEqual(B.filter_turns(events, None), events)
        kept = B.filter_turns(events, [1, 2])
        self.assertEqual({event[0] for event in kept}, {1, 2})
        self.assertEqual(len(kept), 5)

    def write_trace(self, folder):
        path = os.path.join(folder, "t.json.gz")
        with gzip.open(path, "wt") as handle:
            json.dump({"record_bytes": 1_000_000, "events": self.events()}, handle)
        return path

    def test_turns_option_restricts_the_replay(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as out:
            B.main([self.write_trace(folder), "--gb", "0.01", "--policies", "lru", "--turns", "2"])
        self.assertIn("2 decode rows, 2 decode tokens", out.getvalue())
        self.assertIn("turns 2", out.getvalue())

    def test_json_artifact_keys_and_no_file_without_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = self.write_trace(folder)
            target = Path(folder) / "out" / "pool-policy-trace.json"
            target.parent.mkdir()
            weights = Path(folder) / "w.json"
            weights.write_text(json.dumps(CostWeights(recent=0.0).to_mapping()))
            with mock.patch.object(B, "output_path", return_value=target) as patched, \
                    contextlib.redirect_stdout(io.StringIO()):
                B.main([trace, "--gb", "0.01", "--policies", "lru", "cost-aware"])
                self.assertFalse(target.exists())
                patched.assert_not_called()
                B.main([trace, "--gb", "0.01", "--policies", "lru", "cost-aware", "--predict", "--json",
                        "--prefetch-slots", "2", "--weights", str(weights)])
            patched.assert_called_once_with("flashnext", "pool-policy-trace", "pool-policy-trace.json")
            data = json.loads(target.read_text())
        for key in ("trace", "trace_sha256", "record_bytes", "decode_rows", "decode_tokens", "weights", "sizes",
                    "predictor", "prefetch_model", "turns", "policies"):
            self.assertIn(key, data)
        self.assertEqual((data["record_bytes"], data["decode_rows"], data["decode_tokens"]), (1_000_000, 6, 6))
        self.assertEqual(len(data["trace_sha256"]), 64)
        self.assertEqual(data["weights"]["recent"], 0.0)
        entry = data["sizes"][0]["policies"]["cost-aware"]
        for key in ("decode_hit_rate", "prefill_resident_rate", "misses", "demand_mb_per_token", "with_prefetch"):
            self.assertIn(key, entry)
        for key in ("issued", "used", "wasted", "precision", "covered_share", "blocking_mb_per_token",
                    "physical_mb_per_token"):
            self.assertIn(key, entry["with_prefetch"]["prefetch"])
        self.assertIn("upper bound", data["prefetch_model"]["assumption"])
        self.assertEqual(set(data["predictor"]["frequency"]), {"4", "8", "12", "16"})

    def test_bad_weights_file_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()):
            bad = Path(folder) / "w.json"
            bad.write_text('{"recent": -1}')
            with self.assertRaises(SystemExit):
                B.main([self.write_trace(folder), "--policies", "cost-aware", "--weights", str(bad)])


if __name__ == "__main__":
    unittest.main()
