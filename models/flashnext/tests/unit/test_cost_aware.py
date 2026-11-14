"""Cost-aware residency scoring: weights, route frequency and slot victim selection."""

import json
import math
import os
import random
import tempfile
import unittest
import unittest.mock

import numpy as np

from models.shared.cost_aware import CostAwarePolicy, CostWeights, RouteFrequency, WEIGHTS_VERSION


class CostWeightsTest(unittest.TestCase):
    def test_defaults_and_round_trip(self):
        weights = CostWeights()
        self.assertEqual(weights.to_mapping(), {"version": 1, "recent": 0.1, "frequency": 0.2, "reuse": 1.0,
                                                "refill": 0.0, "size": 0.0, "alpha": 0.3})
        self.assertEqual(WEIGHTS_VERSION, 1)
        custom = CostWeights(recent=0.5, frequency=0.0, reuse=2.0, refill=0.25, size=0.1, alpha=1.0)
        self.assertEqual(CostWeights.from_mapping(custom.to_mapping()), custom)

    def test_missing_keys_take_defaults_and_ints_are_floats(self):
        weights = CostWeights.from_mapping({"reuse": 2, "version": 1})
        self.assertEqual(weights, CostWeights(reuse=2.0))
        self.assertIsInstance(weights.reuse, float)

    def test_rejects_bad_mappings(self):
        bad = [
            {"bogus": 1}, {"version": 2}, {"version": True}, {"recent": True}, {"recent": "1"}, {"recent": None},
            {"recent": float("nan")}, {"reuse": float("inf")}, {"frequency": -0.1}, {"refill": -1}, {"size": -1e-9},
            {"alpha": 0.0}, {"alpha": 1.5}, {"alpha": -0.3},
        ]
        for data in bad:
            with self.subTest(data=data), self.assertRaises(ValueError):
                CostWeights.from_mapping(data)
        with self.assertRaises(ValueError):
            CostWeights.from_mapping([1, 2])

    def test_load(self):
        with tempfile.TemporaryDirectory() as folder:
            good = os.path.join(folder, "good.json")
            with open(good, "w") as handle:
                json.dump({"version": 1, "reuse": 3.0, "alpha": 0.5}, handle)
            self.assertEqual(CostWeights.load(good), CostWeights(reuse=3.0, alpha=0.5))
            listing = os.path.join(folder, "list.json")
            garbage = os.path.join(folder, "garbage.json")
            with open(listing, "w") as handle:
                handle.write("[1]")
            with open(garbage, "w") as handle:
                handle.write("{not json")
            for path in (listing, garbage, os.path.join(folder, "missing.json"), folder):
                with self.subTest(path=path), self.assertRaises(ValueError) as caught:
                    CostWeights.load(path)
                self.assertIn(path, str(caught.exception))
            invalid = os.path.join(folder, "invalid.json")
            with open(invalid, "w") as handle:
                json.dump({"alpha": 0}, handle)
            with self.assertRaises(ValueError) as caught:
                CostWeights.load(invalid)
            self.assertIn(invalid, str(caught.exception))

    def test_from_env(self):
        flag = "FLASHNEXT_EXPERT_POOL_COST_WEIGHTS"
        self.assertEqual(CostWeights.from_env({}), CostWeights())
        self.assertEqual(CostWeights.from_env({flag: ""}), CostWeights())
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "weights.json")
            with open(path, "w") as handle:
                json.dump({"frequency": 0.7}, handle)
            self.assertEqual(CostWeights.from_env({flag: path}), CostWeights(frequency=0.7))
            self.assertEqual(CostWeights.from_env({"OTHER": path}, flag="OTHER"), CostWeights(frequency=0.7))
            with unittest.mock.patch.dict(os.environ, {"HOME": folder}):
                self.assertEqual(CostWeights.from_env({flag: "~/weights.json"}), CostWeights(frequency=0.7))
            with self.assertRaises(ValueError):
                CostWeights.from_env({flag: os.path.join(folder, "nope.json")})


class RouteFrequencyTest(unittest.TestCase):
    def test_decay_arithmetic(self):
        freq = RouteFrequency(layers=4, experts=8, alpha=0.3)
        self.assertEqual(freq.table.shape, (4, 8))
        self.assertEqual(freq.table.dtype, np.float64)
        freq.observe(0, [1, 1, 2])
        self.assertAlmostEqual(freq.probability(0, 1), 0.3)
        self.assertAlmostEqual(freq.probability(0, 2), 0.3)
        freq.observe(0, [1])
        self.assertAlmostEqual(freq.probability(0, 1), 0.3 * 0.7 + 0.3)
        self.assertAlmostEqual(freq.probability(0, 2), 0.3 * 0.7)
        self.assertEqual(freq.probability(0, 0), 0.0)
        self.assertTrue(np.all(freq.table[1:] == 0.0))
        for _ in range(200):
            freq.observe(0, [1])
        self.assertTrue(0.0 <= freq.table.min() and freq.table.max() <= 1.0)
        self.assertAlmostEqual(freq.probability(0, 1), 1.0)

    def test_alpha_one_forgets_instantly(self):
        freq = RouteFrequency(layers=2, experts=4, alpha=1.0)
        freq.observe(1, [0, 1])
        freq.observe(1, [2])
        self.assertEqual(list(freq.table[1]), [0.0, 0.0, 1.0, 0.0])

    def test_out_of_range(self):
        freq = RouteFrequency(layers=2, experts=4, alpha=0.5)
        freq.observe(0, [1])
        self.assertAlmostEqual(freq.probability(0, 1), 0.5)
        before = freq.table.copy()
        freq.observe(2, [1])
        freq.observe(-1, [1])
        np.testing.assert_array_equal(freq.table, before)
        freq.observe(0, [-1, 4, 99, 3])
        self.assertAlmostEqual(freq.probability(0, 1), 0.25)
        self.assertAlmostEqual(freq.probability(0, 3), 0.5)
        self.assertEqual(freq.table.sum(), 0.75)
        for layer, expert in ((2, 0), (-1, 0), (0, 4), (0, -1)):
            self.assertEqual(freq.probability(layer, expert), 0.0)
        freq.observe(0, [])
        self.assertAlmostEqual(freq.probability(0, 3), 0.25)

    def test_predict_order_ties_and_exclude(self):
        freq = RouteFrequency(layers=2, experts=16, alpha=0.3)
        freq.observe(0, [1, 2])
        freq.observe(0, [1])
        freq.observe(0, [5, 3])
        # Three visits: expert 1 = (0.3 * 0.7 + 0.3) * 0.7, expert 2 = 0.3 * 0.7 * 0.7, experts 3 and 5 = 0.3.
        got = freq.predict(0, 10)
        self.assertEqual([expert for expert, _ in got], [1, 3, 5, 2])
        self.assertAlmostEqual(got[0][1], (0.3 * 0.7 + 0.3) * 0.7)
        self.assertAlmostEqual(got[1][1], 0.3)
        self.assertAlmostEqual(got[2][1], 0.3)
        self.assertAlmostEqual(got[3][1], 0.3 * 0.7 * 0.7)
        self.assertEqual([expert for expert, _ in freq.predict(0, 3)], [1, 3, 5])
        self.assertEqual([expert for expert, _ in freq.predict(0, 10, exclude={3, 1})], [5, 2])
        self.assertEqual([expert for expert, _ in freq.predict(0, 10, exclude=[3, 5, 1, 2])], [])
        self.assertEqual([expert for expert, _ in freq.predict(0, 2, exclude=(1, 3))], [5, 2])
        self.assertEqual(freq.predict(0, 0), [])
        self.assertEqual(freq.predict(0, -3), [])
        self.assertEqual(freq.predict(1, 4), [])
        self.assertEqual(freq.predict(2, 4), [])
        self.assertEqual(freq.predict(-1, 4), [])
        self.assertTrue(all(isinstance(expert, int) and isinstance(p, float) for expert, p in got))


def policy(slots=4, layers=4, experts=8, **weights):
    return CostAwarePolicy(slots, layers=layers, experts=experts, weights=CostWeights(**weights))


def fill(pol, items):
    """Insert ``(layer, expert)`` keys into slots 0.. at one step each, advancing the pool step between them."""
    for slot, key in enumerate(items):
        pol.advance(key[0])
        pol.insert(key, slot)


class CostAwarePolicyTest(unittest.TestCase):
    def test_frequent_key_outlives_newer_cold_key(self):
        pol = policy(slots=2)
        for _ in range(5):
            pol.frequency.observe(1, [3])
        fill(pol, [(1, 3), (1, 5)])
        self.assertEqual(pol.victims(1), [1])
        self.assertEqual(pol.victims(2), [1, 0])

    def test_recent_term(self):
        pol = policy(slots=3, recent=1.0, frequency=0.0, reuse=0.0)
        pol.advance(0)
        pol.insert((0, 1), 0)
        for _ in range(4):
            pol.advance(0)
        pol.insert((0, 2), 1)
        for _ in range(3):
            pol.advance(0)
        # Slot 1 is 3 steps old (recent), slot 0 is 7 steps old (not recent).
        self.assertEqual(pol.victims(2), [0, 1])
        pol.touch((0, 1), 0)
        # Both recent now: the tie goes to the older last use, slot 1.
        self.assertEqual(pol.victims(2), [1, 0])

    def test_recent_window_edge(self):
        pol = policy(slots=2, layers=4, recent=1.0, frequency=0.5, reuse=0.0)
        pol.frequency.table[0, 1] = 0.5
        pol.advance(0)
        pol.insert((0, 1), 0)
        pol.advance(0)
        pol.insert((0, 2), 1)
        for _ in range(3):
            pol.advance(0)
        # Slot 0 is exactly `layers` steps old: still recent, 1.25 against 1.0.
        self.assertEqual(pol.victims(1), [1])
        pol.advance(0)
        # One step older it is no longer recent: 0.25 against 1.0.
        self.assertEqual(pol.victims(1), [0])

    def test_phase_reuse_prefers_layer_visited_sooner(self):
        # H = 40 / 8 = 5; at layer 0: phase(layer 1) = 1 -> 2 visits, phase(layer 3) = 3 -> 1 visit.
        pol = policy(slots=40, recent=0.0, frequency=0.0, reuse=1.0)
        pol.frequency.table[1, 0] = 0.5
        pol.frequency.table[3, 0] = 0.5
        pol.advance(0)
        pol.insert((3, 0), 0)
        pol.insert((1, 0), 1)
        # reuse: layer 1 = 1 - 0.5**2 = 0.75, layer 3 = 0.5.
        self.assertEqual(pol.victims(1), [0])
        # From layer 2 the order flips: phase(layer 3) = 1, phase(layer 1) = 3.
        pol.advance(2)
        self.assertEqual(pol.victims(1), [1])

    def test_beyond_horizon_has_no_reuse(self):
        # H = 4 / 8 = 0.5 < every phase: reuse is 0 for all keys, so only frequency separates them.
        pol = policy(slots=4, recent=0.0, frequency=1.0, reuse=1.0)
        pol.frequency.table[1, 0] = 0.9
        pol.frequency.table[2, 0] = 0.3
        pol.advance(0)
        pol.insert((1, 0), 0)
        pol.insert((2, 0), 1)
        self.assertEqual(pol.victims(1), [1])

    def test_current_layer_counts_as_one_full_cycle_ahead(self):
        # H = 40 / 8 = 5 and layers = 4: the current layer has phase 4 -> 1 visit; layer 1 has phase 1 -> 2 visits.
        pol = policy(slots=40, recent=0.0, frequency=0.0, reuse=1.0)
        pol.frequency.table[0, 0] = 0.5
        pol.frequency.table[1, 0] = 0.5
        pol.advance(0)
        pol.insert((0, 0), 0)
        pol.insert((1, 0), 1)
        self.assertEqual(pol.victims(1), [0])

    def test_refill_weight_prefers_costlier_layer(self):
        for cheap, costly in ((1, 2), (2, 1)):
            pol = policy(slots=2, recent=0.0, frequency=0.0, reuse=0.0, refill=1.0)
            fill(pol, [(1, 0), (2, 0)])
            pol.observe_refill(cheap, 0.001)
            pol.observe_refill(costly, 0.003)
            # Normalised by the mean (0.002): 0.5 against 1.5, the cheaper layer goes first.
            self.assertEqual(pol.victims(1), [cheap - 1])

    def test_refill_ignored_at_zero_weight_and_unobserved_layer_scores_zero(self):
        pol = policy(slots=2, recent=0.0, frequency=0.0, reuse=0.0, refill=0.0)
        fill(pol, [(1, 0), (2, 0)])
        pol.observe_refill(2, 5.0)
        self.assertEqual(pol.victims(2), [0, 1])
        pol = policy(slots=2, recent=0.0, frequency=0.0, reuse=0.0, refill=1.0)
        fill(pol, [(1, 0), (2, 0)])
        pol.observe_refill(2, 5.0)
        self.assertEqual(pol.victims(1), [0])

    def test_refill_ewma_and_invalid_input(self):
        pol = policy(layers=4)
        pol.observe_refill(1, 1.0)
        pol.observe_refill(1, 2.0)
        self.assertAlmostEqual(pol._refill[1], 1.2)
        for layer, seconds in ((1, float("nan")), (1, float("inf")), (1, -1.0), (4, 1.0), (-1, 1.0)):
            pol.observe_refill(layer, seconds)
        self.assertAlmostEqual(pol._refill[1], 1.2)
        self.assertEqual(int(pol._refill_seen.sum()), 1)

    def test_protected_never_returned(self):
        pol = policy(slots=4)
        fill(pol, [(0, 0), (0, 1), (0, 2), (0, 3)])
        for count in (1, 2):
            self.assertTrue(set(pol.victims(count, protected=[0, 1])).isdisjoint({0, 1}))
        self.assertEqual(sorted(pol.victims(2, protected=iter([0, 1]))), [2, 3])
        with self.assertRaises(RuntimeError):
            pol.victims(3, protected=[0, 1])

    def test_not_enough_victims(self):
        pol = policy(slots=4)
        with self.assertRaises(RuntimeError):
            pol.victims(1)
        fill(pol, [(0, 0), (0, 1)])
        with self.assertRaises(RuntimeError):
            pol.victims(3)
        with self.assertRaises(RuntimeError):
            pol.victims(2, protected=[0])
        self.assertEqual(len(pol.victims(2)), 2)
        self.assertEqual(pol.victims(0), [])
        self.assertEqual(pol.victims(-1), [])
        self.assertEqual(policy(slots=0).victims(0), [])

    def test_tie_order_is_older_use_then_lower_slot(self):
        pol = policy(slots=4, recent=0.0, frequency=0.0, reuse=0.0)
        pol.advance(0)
        pol.insert((0, 0), 2)
        pol.advance(0)
        pol.insert((0, 1), 0)
        pol.insert((0, 2), 1)
        pol.advance(0)
        pol.insert((0, 3), 3)
        self.assertEqual(pol.victims(4), [2, 0, 1, 3])
        pol.advance(0)
        pol.touch((0, 0), 2)
        self.assertEqual(pol.victims(4), [0, 1, 3, 2])

    def test_victims_does_not_modify_state_and_is_deterministic(self):
        def run():
            pol = policy(slots=6, layers=4, experts=8)
            rng = random.Random(7)
            out = []
            for index in range(6):
                pol.frequency.observe(index % 4, [rng.randrange(8) for _ in range(3)])
            fill(pol, [(index % 4, index) for index in range(6)])
            before = pol.stats()
            first = pol.victims(3, protected=[5])
            self.assertEqual(pol.victims(3, protected=[5]), first)
            self.assertEqual(pol.stats(), before)
            out.append(first)
            for slot in first:
                pol.discard(pol.key_of(slot), slot)
            out.append(pol.victims(2))
            return out

        self.assertEqual(run(), run())

    def test_insert_discard_misuse(self):
        pol = policy(slots=2)
        pol.advance(0)
        pol.insert((0, 1), 0)
        with self.assertRaises(ValueError):
            pol.insert((0, 2), 0)
        with self.assertRaises(ValueError):
            pol.discard((0, 2), 0)
        with self.assertRaises(ValueError):
            pol.discard((0, 1), 1)
        self.assertEqual(pol.key_of(0), (0, 1))
        self.assertIsNone(pol.key_of(1))
        pol.discard((0, 1), 0)
        self.assertIsNone(pol.key_of(0))
        with self.assertRaises(ValueError):
            pol.discard((0, 1), 0)
        pol.insert((0, 2), 0)
        self.assertEqual(pol.key_of(0), (0, 2))
        self.assertIsInstance(pol.key_of(0)[0], int)

    def test_out_of_range_keys_do_not_raise(self):
        pol = policy(slots=6, layers=4, experts=8)
        keys = [(99, 0), (-1, 2), (1, 600), (2, -4), (3, 1)]
        pol.advance(1)
        for slot, key in enumerate(keys):
            pol.insert(key, slot)
            pol.touch(key, slot)
        pol.observe_refill(99, 1.0)
        self.assertEqual(sorted(pol.victims(5)), [0, 1, 2, 3, 4])
        self.assertEqual(pol.victims(1, protected=[0, 1, 2, 3]), [4])
        for slot, key in enumerate(keys):
            self.assertEqual(pol.key_of(slot), key)
            pol.discard(key, slot)
        self.assertEqual(pol.stats()["resident"], 0)

    def test_out_of_range_layer_scores_as_cold(self):
        # Layer 99 has p = 0, so it loses to any key with some frequency.
        pol = policy(slots=2, layers=4, recent=0.0, frequency=1.0, reuse=0.0)
        pol.frequency.table[2, 0] = 0.1
        pol.advance(0)
        pol.insert((99, 0), 0)
        pol.insert((2, 0), 1)
        self.assertEqual(pol.victims(1), [0])

    def test_seeded_entries_are_not_recent(self):
        pol = policy(slots=2, recent=1.0, frequency=0.0, reuse=0.0)
        pol.advance(0)
        pol.insert((0, 1), 0)
        pol.advance(0)
        pol.insert((0, 2), 1, seeded=True)
        self.assertEqual(pol.victims(2), [1, 0])
        for _ in range(50):
            pol.advance(0)
        self.assertEqual(pol.victims(1), [1])
        pol.touch((0, 2), 1)
        self.assertEqual(pol.victims(2), [0, 1])

    def test_observe_updates_frequency_and_miss_rate(self):
        pol = policy(layers=4)
        pol.observe(2, [1, 1, 3], misses=0)
        self.assertAlmostEqual(pol.frequency.probability(2, 1), 0.3)
        self.assertAlmostEqual(pol.stats()["miss_rate"], 7.6)
        pol.observe(2, [1], misses=10)
        self.assertAlmostEqual(pol.stats()["miss_rate"], 0.95 * 7.6 + 0.5)
        self.assertAlmostEqual(pol.frequency.probability(2, 1), 0.51)

    def test_stats_and_shared_frequency(self):
        freq = RouteFrequency(layers=4, experts=8, alpha=0.5)
        weights = CostWeights(alpha=0.5)
        pol = CostAwarePolicy(3, layers=4, experts=8, weights=weights, frequency=freq)
        self.assertIs(pol.frequency, freq)
        self.assertIs(pol.weights, weights)
        pol.advance(0)
        pol.insert((0, 1), 0)
        pol.observe(0, [1])
        self.assertEqual(freq.probability(0, 1), 0.5)
        stats = pol.stats()
        self.assertEqual(stats["policy"], "cost-aware")
        self.assertEqual(stats["weights"], weights.to_mapping())
        self.assertEqual(stats["step"], 1)
        self.assertEqual(stats["resident"], 1)
        own = CostAwarePolicy(3, layers=4, experts=8, weights=CostWeights(alpha=0.6))
        self.assertEqual(own.frequency.alpha, 0.6)
        self.assertEqual(CostAwarePolicy(3).frequency.table.shape, (48, 512))

    def test_size_term_is_constant(self):
        a = policy(slots=2, size=0.0)
        b = policy(slots=2, size=3.0)
        for pol in (a, b):
            pol.frequency.observe(1, [3])
            fill(pol, [(1, 3), (1, 5)])
        self.assertEqual(a.victims(2), b.victims(2))

    def test_matches_scalar_reference_score(self):
        rng = random.Random(11)
        layers, experts, slots = 6, 10, 50
        weights = CostWeights(recent=0.3, frequency=0.4, reuse=0.9, refill=0.5, size=0.2, alpha=0.4)
        pol = CostAwarePolicy(slots, layers=layers, experts=experts, weights=weights)
        for _ in range(60):
            pol.frequency.observe(rng.randrange(layers), rng.sample(range(experts), rng.randrange(1, 5)))
        for layer in range(layers):
            if layer != 4:
                pol.observe_refill(layer, rng.uniform(0.001, 0.01))
        keys = [(rng.choice([-2, 0, 1, 2, 3, 4, 5, 9]), rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, -1]))
                for _ in range(slots)]
        for slot, key in enumerate(keys):
            pol.advance(key[0])
            pol.insert(key, slot, seeded=slot % 7 == 0)
            if slot % 5 == 0:
                pol.touch(key, slot)
        for _ in range(3):
            pol.observe(rng.randrange(layers), [1], misses=rng.randrange(0, 12))
        pol.advance(2)
        seen = [pol._refill[layer] for layer in range(layers) if pol._refill_seen[layer]]
        mean = sum(seen) / len(seen)
        horizon = slots / max(pol._miss_rate, 0.5)

        def score(slot):
            layer, expert = keys[slot]
            inside = 0 <= layer < layers
            p = pol.frequency.table[layer, expert] if inside and 0 <= expert < experts else 0.0
            phase = (layer - 2) % layers or layers if inside else layers
            visits = math.floor((horizon - phase) / layers) + 1 if phase <= horizon else 0
            reuse = 1 - (1 - p) ** visits
            recent = 1 if pol._step - pol._last[slot] <= layers else 0
            cost = pol._refill[layer] / mean if inside and pol._refill_seen[layer] else 0.0
            return (weights.recent * recent + weights.frequency * p + weights.reuse * reuse
                    + weights.refill * cost - weights.size, int(pol._last[slot]), slot)

        expected = [slot for _, _, slot in sorted(score(slot) for slot in range(slots))]
        protected = {expected[0], expected[3], 17}
        self.assertEqual(pol.victims(slots), expected)
        self.assertEqual(pol.victims(10, protected=protected), [s for s in expected if s not in protected][:10])

    def test_randomized_pool_consistency(self):
        rng = random.Random(3)
        layers, experts, slots = 6, 20, 40
        pol = CostAwarePolicy(slots, layers=layers, experts=experts)
        where, free, hits_total, total = {}, list(range(slots)), 0, 0
        layer = 0
        for access in range(4000):
            layer = rng.randrange(layers) if rng.random() < 0.1 else (layer + 1) % layers
            chosen = list(dict.fromkeys(rng.choices(range(experts), weights=[1 + (e % 5) ** 2 for e in range(experts)],
                                                    k=rng.randrange(1, 8))))
            pol.advance(layer)
            hit_slots, misses = [], []
            for expert in chosen:
                slot = where.get((layer, expert))
                if slot is None:
                    misses.append(expert)
                else:
                    pol.touch((layer, expert), slot)
                    hit_slots.append(slot)
            need = len(misses) - len(free)
            if need > 0:
                victims = pol.victims(need, protected=hit_slots)
                self.assertEqual(len(set(victims)), need)
                self.assertTrue(set(victims).isdisjoint(hit_slots))
                for slot in victims:
                    key = pol.key_of(slot)
                    self.assertEqual(where[key], slot)
                    pol.discard(key, slot)
                    del where[key]
                    free.append(slot)
            for expert in misses:
                slot = free.pop()
                pol.insert((layer, expert), slot, seeded=rng.random() < 0.05)
                where[(layer, expert)] = slot
            pol.observe(layer, chosen, misses=len(misses))
            hits_total += len(hit_slots)
            total += len(chosen)
            occupied = {slot: key for key, slot in where.items()}
            self.assertEqual(len(occupied), len(where))
            self.assertEqual({slot for slot in range(slots) if pol.key_of(slot) is not None}, set(occupied))
            self.assertEqual({slot: pol.key_of(slot) for slot in occupied}, occupied)
            self.assertEqual(pol.stats()["resident"], len(where))
            self.assertEqual(len(where) + len(free), slots)
        self.assertGreater(hits_total, 0)
        self.assertLess(hits_total, total)


if __name__ == "__main__":
    unittest.main()
