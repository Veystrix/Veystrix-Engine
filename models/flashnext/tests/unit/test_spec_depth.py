#!/usr/bin/env python3
"""Pure tests for the adaptive speculative depth controller."""
from __future__ import annotations

import json
import random
import statistics
import unittest
from collections import Counter

from models.shared.spec_depth import DepthController


class Environment:
    """Deterministic draft/verify/repair world with fixed true parameters."""

    def __init__(self, accept, *, single, row, draft, repair, seed=20240607):
        self.accept = list(accept)
        self.single = single
        self.row = row
        self.draft = draft
        self.repair = repair
        # Seeded, so every run sees the same acceptance outcomes.
        self.random = random.Random(seed)
        # (tokens emitted, seconds spent) of every cycle run so far.
        self.log = []

    def configure(self, accept, *, single, row, draft, repair):
        """Change the true parameters, keeping the random stream and log."""
        self.accept = list(accept)
        self.single, self.row, self.draft, self.repair = single, row, draft, repair

    def cycle(self, controller, depth):
        accepted = 0
        for position in range(depth):
            if self.random.random() < self.accept[position]:
                accepted += 1
            else:
                break
        repair = (accepted + 1) * self.repair if accepted < depth else 0.0
        self.log.append((
            accepted + 1,
            self.draft * depth + self.single + self.row * depth + repair,
        ))
        controller.observe(
            depth, accepted,
            draft_seconds=self.draft * depth,
            verify_seconds=self.single + self.row * depth,
            repair_seconds=repair,
        )

    def rate(self, depth):
        tokens, seconds, reach = 1.0, self.single, 1.0
        seconds += (self.draft + self.row) * depth
        for index in range(depth):
            passed = reach * self.accept[index]
            tokens += passed
            seconds += reach * (1.0 - self.accept[index]) * (index + 1) * self.repair
            reach = passed
        return tokens / seconds

    def best(self, max_depth):
        return max(range(max_depth + 1), key=lambda d: (self.rate(d), -d))


def realised(environment, first, last):
    """Emitted tokens per second over cycles first..last (1-based, inclusive)."""
    window = environment.log[first - 1:last]
    return sum(tokens for tokens, _ in window) / sum(seconds for _, seconds in window)


def case(name, max_depth):
    """The acceptance-test environments, in units of one single-token step."""
    cheap = dict(single=1.0, row=0.15, draft=0.05, repair=1.0)
    accept = {
        "a": [0.95] * max_depth,
        "b": [max(0.9 - 0.1 * index, 0.05) for index in range(max_depth)],
        "c": [0.3] * max_depth,
        "d": [0.8] * max_depth,
        "e": [0.9] * max_depth,
    }[name]
    if name == "d":
        return dict(accept=accept, single=1.0, row=1.0, draft=0.05, repair=1.0)
    if name == "e":
        return dict(accept=accept, **dict(cheap, repair=3.0))
    return dict(accept=accept, **cheap)


def run(controller, environment, cycles):
    """Return the (cycle number, depth) pairs the controller chose."""
    chosen = []
    for number in range(1, cycles + 1):
        depth = controller.choose()
        chosen.append((number, depth))
        environment.cycle(controller, depth)
    return chosen


def steady_choice(controller, chosen, tail=96):
    """Most common depth over the last cycles, ignoring probe and plain samples."""
    plain, probe = controller.plain_every, controller.probe_every
    picks = [
        depth for number, depth in chosen[-tail:]
        if number % plain != 0 and number % probe != 0
    ]
    return Counter(picks).most_common(1)[0][0]


class DepthControllerTest(unittest.TestCase):
    def test_bootstrap_sequence_and_bounds(self):
        controller = DepthController(5)
        environment = Environment([0.8] * 5, single=1.0, row=0.1, draft=0.1, repair=1.0)
        chosen = [depth for _, depth in run(controller, environment, 40)]
        self.assertEqual(chosen[:3], [5, 5, 0])
        self.assertTrue(all(0 <= depth <= 5 for depth in chosen))

    def test_start_depth_is_clamped(self):
        controller = DepthController(4, start_depth=9)
        self.assertEqual(controller.choose(), 4)
        controller = DepthController(4, min_depth=2, start_depth=0)
        self.assertEqual(controller.choose(), 2)

    def test_min_depth_skips_plain_sample_and_stays_in_bounds(self):
        controller = DepthController(5, min_depth=2, start_depth=3, plain_every=4, probe_every=5)
        environment = Environment([0.3] * 5, single=1.0, row=0.5, draft=0.2, repair=1.0)
        chosen = [depth for _, depth in run(controller, environment, 120)]
        self.assertEqual(chosen[:3], [3, 3, chosen[2]])
        self.assertNotIn(0, chosen)
        self.assertNotIn(1, chosen)
        self.assertTrue(all(2 <= depth <= 5 for depth in chosen))
        # A single-row cost is inferred from the verify passes, so every
        # allowed depth has a rate.
        for depth in range(2, 6):
            self.assertIsNotNone(controller.expected_rate(depth))
        self.assertIsNone(controller.stats()["single"])

    def test_default_parameters_reach_near_optimal_rate(self):
        # a: high acceptance, b: acceptance falling with depth, c: low
        # acceptance, d: rows as costly as steps, e: costly repair.
        seeds = range(8)
        for max_depth in (4, 8):
            for name in "abcde":
                with self.subTest(case=name, max_depth=max_depth):
                    params = case(name, max_depth)
                    best = Environment(**params)
                    optimum = best.rate(best.best(max_depth))
                    whole, late = [], []
                    for seed in seeds:
                        controller = DepthController(max_depth)
                        environment = Environment(seed=seed, **params)
                        run(controller, environment, 400)
                        whole.append(realised(environment, 1, 400) / optimum)
                        late.append(realised(environment, 200, 400) / optimum)
                    self.assertGreaterEqual(statistics.median(whole), 0.90, whole)
                    self.assertGreaterEqual(statistics.median(late), 0.95, late)
                    self.assertGreaterEqual(min(late), 0.85, late)

    def test_scenarios_have_the_intended_optima(self):
        best = {name: Environment(**case(name, 8)).best(8) for name in "abcde"}
        self.assertEqual(best, {"a": 4, "b": 1, "c": 0, "d": 0, "e": 1})

    def test_regime_change_recovers_in_both_directions(self):
        for max_depth in (4, 8):
            for first, second in (("a", "c"), ("c", "a")):
                with self.subTest(max_depth=max_depth, change=first + second):
                    target = Environment(**case(second, max_depth))
                    optimum = target.rate(target.best(max_depth))
                    rates = []
                    for seed in range(8):
                        controller = DepthController(max_depth)
                        environment = Environment(seed=seed, **case(first, max_depth))
                        run(controller, environment, 300)
                        environment.configure(**case(second, max_depth))
                        run(controller, environment, 300)
                        rates.append(realised(environment, 451, 600) / optimum)
                    self.assertGreaterEqual(statistics.median(rates), 0.90, rates)
                    self.assertGreaterEqual(min(rates), 0.85, rates)

    def test_one_rejection_moves_a_deep_estimate_by_its_evidence(self):
        controller = DepthController(4, prior=0.8, strength=3.0, decay=1.0)
        for _ in range(12):
            controller.observe(2, 2, draft_seconds=0.1, verify_seconds=1.2)
        before = controller.acceptance()
        # Position 3 was never tested: it sits at the estimate of position 2.
        self.assertAlmostEqual(before[2], before[1])
        controller.observe(3, 2, draft_seconds=0.1, verify_seconds=1.2)
        after = controller.acceptance()
        # One failure against strength 3 moves it by about 1 / (0 + 3), not 0.25.
        self.assertAlmostEqual(after[2], 3.0 * after[1] / (1.0 + 3.0))
        self.assertLess(after[2], before[2])

    @staticmethod
    def frozen(max_depth, accept, **kwargs):
        """A controller whose model is fixed, so only the cadence varies."""
        controller = DepthController(max_depth, **kwargs)
        controller.single, controller.row = 1.0, 0.1
        controller.draft, controller.repair = 0.1, 1.0
        controller.acceptance = lambda: list(accept)
        return controller

    def test_probe_and_plain_cadence(self):
        controller = self.frozen(5, [0.9, 0.9, 0.05, 0.05, 0.05],
                                 probe_every=5, plain_every=12)
        base = controller._best()
        self.assertIn(base, (1, 2))
        interval = controller._probe_interval(base)
        picks = {number: controller.choose() for number in range(1, 61)}
        self.assertEqual([picks[1], picks[2], picks[3]], [5, 5, 0])
        for number in range(4, 61):
            if number % 12 == 0:
                expected = 0  # plain sample, including the number 60 collision
            elif number % interval == 0:
                expected = base + 1  # probe one deeper than the choice
            else:
                expected = base
            self.assertEqual(picks[number], expected, f"cycle {number}")
        self.assertEqual(controller.stats()["cycles"], 60)

    def test_probe_interval_follows_the_modelled_gap(self):
        controller = DepthController(4, probe_every=8)
        # Rate of the next deeper depth relative to the chosen one (depth 1).
        for ratio, interval in ((0.99, 4), (0.9, 8), (0.75, 16), (0.4, 32)):
            controller.last_rates = {1: 1.0, 2: ratio}
            self.assertEqual(controller._probe_interval(1), interval, ratio)
        controller.last_rates = {1: 1.0}
        self.assertEqual(controller._probe_interval(1), 8)
        controller.last_rates = {}
        self.assertEqual(controller._probe_interval(1), 8)

    def test_full_acceptances_climb_between_probes(self):
        controller = self.frozen(4, [0.95, 0.95, 0.2, 0.2], probe_every=10 ** 9,
                                 plain_every=10 ** 9)
        self.assertEqual(controller._best(), 2)
        for _ in range(3):
            controller.choose()
        feed = dict(draft_seconds=0.2, verify_seconds=1.2)
        controller.observe(0, 0, draft_seconds=0.0, verify_seconds=1.0)
        picks = []
        for _ in range(8):
            depth = controller.choose()
            picks.append(depth)
            controller.observe(depth, depth, **feed)
        # A full acceptance at depth 2 probes depth 3 next, but never on two
        # consecutive cycles.
        self.assertEqual(picks, [2, 3, 2, 3, 2, 3, 2, 3])
        # Once a probe of depth 3 is rejected at its last position, the climb
        # stops, however many full acceptances at depth 2 follow.
        self.assertEqual(controller.choose(), 2)
        controller.observe(2, 2, **feed)
        self.assertEqual(controller.choose(), 3)
        controller.observe(3, 2, draft_seconds=0.3, verify_seconds=1.3, repair_seconds=3.0)
        later = []
        for _ in range(4):
            depth = controller.choose()
            later.append(depth)
            controller.observe(depth, depth, **feed)
        self.assertEqual(later, [2, 2, 2, 2])

    def test_probe_is_skipped_at_max_depth(self):
        controller = self.frozen(3, [0.99] * 3, probe_every=2, plain_every=10 ** 9)
        self.assertEqual(controller._best(), 3)
        picks = [controller.choose() for _ in range(30)]
        self.assertEqual(picks[3:], [3] * 27)

    def test_plain_is_skipped_when_chosen_is_zero_or_min_depth_positive(self):
        controller = DepthController(3, probe_every=10 ** 9, plain_every=4)
        controller.single, controller.row, controller.draft, controller.repair = 1.0, 2.0, 1.0, 1.0
        controller.cycles = 3
        self.assertEqual(controller.choose(), 0)
        controller = DepthController(3, min_depth=1, probe_every=10 ** 9, plain_every=1)
        controller.single, controller.row, controller.draft = 1.0, 0.0, 0.0
        controller.cycles = 3
        self.assertGreaterEqual(controller.choose(), 1)

    def test_probe_not_past_max_depth(self):
        controller = DepthController(2, probe_every=1, plain_every=10 ** 9)
        environment = Environment([0.95, 0.95], single=1.0, row=0.01, draft=0.01, repair=1.0)
        chosen = [depth for _, depth in run(controller, environment, 50)]
        self.assertTrue(all(0 <= depth <= 2 for depth in chosen))

    def test_rate_unknown_until_costs_observed(self):
        controller = DepthController(3)
        self.assertIsNone(controller.expected_rate(0))
        controller.observe(3, 3, draft_seconds=0.3, verify_seconds=1.4)
        # No depth-0 cycle yet and depth 0 not observed: only the floor.
        self.assertIsNotNone(controller.expected_rate(3))
        controller = DepthController(3)
        controller.observe(0, 0, draft_seconds=0.0, verify_seconds=1.0)
        self.assertIsNotNone(controller.expected_rate(0))
        self.assertIsNone(controller.expected_rate(2))
        controller.observe(2, 2, draft_seconds=0.2, verify_seconds=1.2)
        self.assertIsNotNone(controller.expected_rate(2))

    def test_model_matches_formula(self):
        controller = DepthController(3, prior=0.5)
        controller.single, controller.row, controller.draft, controller.repair = 1.0, 0.2, 0.1, 0.8
        tokens = 1 + 0.5 + 0.25 + 0.125
        seconds = 0.1 * 3 + 1.0 + 0.2 * 3
        seconds += 0.5 * 1 * 0.8 + 0.5 * 0.5 * 2 * 0.8 + 0.25 * 0.5 * 3 * 0.8
        self.assertAlmostEqual(controller.expected_rate(3), tokens / seconds)
        self.assertAlmostEqual(controller.expected_rate(0), 1.0)

    def test_repair_defaults_to_single_until_observed(self):
        controller = DepthController(1, prior=0.5)
        controller.single, controller.row, controller.draft = 2.0, 0.0, 0.0
        self.assertAlmostEqual(controller.expected_rate(1), 1.5 / (2.0 + 0.5 * 2.0))
        controller.observe(1, 0, draft_seconds=0.0, verify_seconds=2.0, repair_seconds=3.0)
        self.assertEqual(controller.repair, 3.0)

    def test_observe_updates_follow_spec(self):
        controller = DepthController(4, alpha=0.5, prior=0.5, decay=0.5, strength=2.0)
        controller.observe(0, 0, draft_seconds=0.0, verify_seconds=1.0)
        self.assertEqual(controller.single, 1.0)
        self.assertEqual(controller.successes, [0.0] * 4)
        controller.observe(3, 1, draft_seconds=0.6, verify_seconds=1.6, repair_seconds=1.0)
        # Position 1 succeeded, position 2 failed, position 3 was not reached.
        self.assertEqual(controller.successes, [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(controller.failures, [0.0, 1.0, 0.0, 0.0])
        controller.observe(2, 2, draft_seconds=0.2, verify_seconds=1.2)
        # Tested positions decay by one half before the new count is added.
        self.assertEqual(controller.successes, [1.5, 1.0, 0.0, 0.0])
        self.assertEqual(controller.failures, [0.0, 0.5, 0.0, 0.0])
        first = (1.5 + 2.0 * 0.5) / (1.5 + 2.0)
        second = (1.0 + 2.0 * first) / (1.5 + 2.0)
        third = (0.0 + 2.0 * second) / 2.0
        fourth = (0.0 + 2.0 * third) / 2.0
        for got, want in zip(controller.acceptance(), (first, second, third, fourth)):
            self.assertAlmostEqual(got, want)
        controller = DepthController(4, alpha=0.5, prior=0.5)
        controller.observe(0, 0, draft_seconds=0.0, verify_seconds=1.0)
        controller.observe(3, 1, draft_seconds=0.6, verify_seconds=1.6, repair_seconds=1.0)
        self.assertAlmostEqual(controller.draft, 0.2)
        self.assertAlmostEqual(controller.row, 0.2)
        self.assertAlmostEqual(controller.repair, 0.5)
        # A cheaper-than-single verify clamps the row cost at zero.
        controller.observe(1, 1, draft_seconds=0.1, verify_seconds=0.5)
        self.assertGreaterEqual(controller.row, 0.0)
        self.assertEqual(controller.depth_counts, {0: 1, 3: 1, 1: 1})

    def test_argument_validation(self):
        bad_init = [
            dict(max_depth=0), dict(max_depth=-1), dict(max_depth=2.5),
            dict(max_depth=True), dict(max_depth=4, min_depth=-1),
            dict(max_depth=4, min_depth=5), dict(max_depth=4, start_depth=-1),
            dict(max_depth=4, alpha=0.0), dict(max_depth=4, alpha=1.5),
            dict(max_depth=4, alpha=float("nan")), dict(max_depth=4, prior=-0.1),
            dict(max_depth=4, prior=1.1), dict(max_depth=4, prior=float("inf")),
            dict(max_depth=4, probe_every=0), dict(max_depth=4, plain_every=0),
            dict(max_depth=4, decay=0.0), dict(max_depth=4, decay=1.5),
            dict(max_depth=4, decay=float("nan")), dict(max_depth=4, strength=0.0),
            dict(max_depth=4, strength=-1.0), dict(max_depth=4, strength=float("inf")),
        ]
        for kwargs in bad_init:
            with self.subTest(kwargs), self.assertRaises(ValueError):
                DepthController(**kwargs)
        controller = DepthController(3)
        good = dict(draft_seconds=0.1, verify_seconds=1.0)
        bad_observe = [
            ((4, 0), good), ((-1, 0), good), ((2, 3), good), ((2, -1), good),
            ((1.5, 0), good), ((1, 0), dict(good, draft_seconds=-0.1)),
            ((1, 0), dict(good, verify_seconds=float("nan"))),
            ((1, 0), dict(good, verify_seconds=float("inf"))),
            ((1, 0), dict(good, repair_seconds=-1.0)),
            ((1, 0), dict(good, repair_seconds=float("nan"))),
        ]
        for args, kwargs in bad_observe:
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(ValueError):
                controller.observe(*args, **kwargs)
        with self.assertRaises(ValueError):
            controller.expected_rate(4)
        # A rejected observation leaves the model untouched.
        self.assertEqual(controller.stats()["depth_counts"], {})
        self.assertEqual(controller.successes, [0.0] * 3)
        self.assertEqual(controller.failures, [0.0] * 3)

    def test_stats_are_json_serialisable(self):
        controller = DepthController(4)
        environment = Environment([0.7] * 4, single=1.0, row=0.1, draft=0.1, repair=1.0)
        run(controller, environment, 30)
        stats = controller.stats()
        self.assertEqual(stats["cycles"], 30)
        self.assertEqual(sum(stats["depth_counts"].values()), 30)
        self.assertEqual(len(stats["p"]), 4)
        for key in ("single", "row", "draft", "repair", "rates"):
            self.assertIn(key, stats)
        decoded = json.loads(json.dumps(stats))
        self.assertEqual(decoded["cycles"], 30)
        # A fresh controller serialises too.
        json.dumps(DepthController(2).stats())


if __name__ == "__main__":
    unittest.main()
