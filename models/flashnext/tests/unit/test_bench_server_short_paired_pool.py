"""Argument validation and arm checks for the Stage 3 paired options; no server is started."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from models.flashnext.tests.bench import bench_server_short_paired as P
from models.shared.cost_aware import CostWeights

BASE = ["--runtime", "flashnext", "--checkpoint", "/unused", "--performance-v2"]


def parse(*extra, base=BASE):
    return P.build_parser().parse_args([*base, *extra])


def health(policy="lru", mapped=1_000_000_000, prefetch=None):
    return {"cache": {"flashnext_kv": {"expert_pool_policy": policy, "expert_pool_mapped_bytes": mapped,
                                       "expert_pool_stats": {"policy": policy, "prefetch": prefetch}}}}


class ValidationTests(unittest.TestCase):
    def test_valid_combinations(self):
        for extra in (("--pool-policy", "lru", "cost-aware"), ("--pool-policy", "s3fifo", "least-stale"),
                      ("--expert-prefetch",), ("--expert-prefetch", "--fixed-pool-policy", "cost-aware"),
                      ("--pool-policy", "lru", "s3fifo", "--isolated-pool-gb", "1"),
                      ("--expert-prefetch", "--isolated-pool-gb", "0.5")):
            self.assertIsNone(P.pool_option_error(parse(*extra)), extra)

    def test_policy_choices_are_enforced_by_the_parser(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse("--pool-policy", "lru", "arc")

    def test_identical_policies_are_refused(self):
        self.assertIn("different", P.pool_option_error(parse("--pool-policy", "lru", "lru")))

    def test_requires_performance_v2_on_flashnext(self):
        self.assertIn("--performance-v2", P.pool_option_error(
            parse("--pool-policy", "lru", "cost-aware", base=BASE[:-1])))
        self.assertIn("flashnext", P.pool_option_error(
            parse("--expert-prefetch", base=["--runtime", "nemotron35", "--checkpoint", "/x", "--performance-v2"])))

    def test_mutually_exclusive_with_other_isolated_options(self):
        for other in (("--least-stale",), ("--pool-compare", "1", "2"), ("--set-flag", "A=1"),
                      ("--expert-pack", "/x"), ("--large-workload",)):
            for mine in (("--pool-policy", "lru", "cost-aware"), ("--expert-prefetch",)):
                self.assertIn("alone", P.pool_option_error(parse(*mine, *other)), (mine, other))
        self.assertIn("alone", P.pool_option_error(parse("--pool-policy", "lru", "cost-aware", "--expert-prefetch")))

    def test_pool_size_range(self):
        for gb, ok in (("0.5", True), ("1", True), ("2", True), ("0.4", False), ("2.5", False)):
            error = P.pool_option_error(parse("--expert-prefetch", "--isolated-pool-gb", gb))
            self.assertEqual(error is None, ok, gb)

    def test_dependent_options_need_their_parent(self):
        self.assertIn("needs", P.pool_option_error(parse("--isolated-pool-gb", "1")))
        self.assertIn("needs", P.pool_option_error(parse("--fixed-pool-policy", "lru")))
        self.assertIn("--expert-prefetch", P.pool_option_error(parse("--pool-policy", "lru", "cost-aware",
                                                                    "--fixed-pool-policy", "lru")))

    def test_cost_weights_need_a_cost_aware_candidate_and_a_valid_file(self):
        with tempfile.TemporaryDirectory() as folder:
            good, bad = Path(folder) / "good.json", Path(folder) / "bad.json"
            good.write_text(json.dumps(CostWeights(recent=0.2).to_mapping()))
            bad.write_text('{"recent": -1}')
            self.assertIsNone(P.pool_option_error(parse("--pool-policy", "lru", "cost-aware", "--cost-weights", str(good))))
            self.assertIn("cost-aware", P.pool_option_error(parse("--pool-policy", "lru", "s3fifo", "--cost-weights", str(good))))
            self.assertIn("recent", P.pool_option_error(parse("--pool-policy", "lru", "cost-aware", "--cost-weights", str(bad))))
            self.assertIn("cannot read", P.pool_option_error(
                parse("--pool-policy", "lru", "cost-aware", "--cost-weights", str(Path(folder) / "missing.json"))))

    def test_existing_runs_are_not_affected(self):
        self.assertIsNone(P.pool_option_error(parse("--least-stale")))
        self.assertIsNone(P.pool_option_error(parse("--pool-compare", "1", "2")))
        self.assertIsNone(P.pool_option_error(parse()))


class PlanTests(unittest.TestCase):
    def test_policy_ab_sets_the_policy_per_arm_and_weights_on_the_candidate_only(self):
        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder) / "w.json"
            weights.write_text("{}")
            env, expect = P.pool_arm_plan(parse("--pool-policy", "lru", "cost-aware", "--cost-weights", str(weights)))
        self.assertEqual((env["baseline"][P.POOL_POLICY_FLAG], env["candidate"][P.POOL_POLICY_FLAG]), ("lru", "cost-aware"))
        self.assertEqual(env["baseline"][P.COST_WEIGHTS_FLAG], "")
        self.assertTrue(env["candidate"][P.COST_WEIGHTS_FLAG].endswith("w.json"))
        self.assertEqual((env["baseline"][P.PREFETCH_FLAG], env["candidate"][P.PREFETCH_FLAG]), ("0", "0"))
        self.assertEqual(expect["candidate"], {"policy": "cost-aware", "prefetch": False})

    def test_weights_are_not_passed_when_the_candidate_is_another_policy(self):
        env, _ = P.pool_arm_plan(parse("--pool-policy", "cost-aware", "lru"))
        self.assertEqual((env["baseline"][P.COST_WEIGHTS_FLAG], env["candidate"][P.COST_WEIGHTS_FLAG]), ("", ""))

    def test_prefetch_ab_fixes_one_policy_and_flips_the_flag(self):
        env, expect = P.pool_arm_plan(parse("--expert-prefetch"))
        self.assertEqual({e[P.POOL_POLICY_FLAG] for e in env.values()}, {"lru"})
        self.assertEqual((env["baseline"][P.PREFETCH_FLAG], env["candidate"][P.PREFETCH_FLAG]), ("0", "1"))
        env, _ = P.pool_arm_plan(parse("--expert-prefetch", "--fixed-pool-policy", "cost-aware"))
        self.assertEqual({e[P.POOL_POLICY_FLAG] for e in env.values()}, {"cost-aware"})
        self.assertEqual((expect["baseline"]["prefetch"], expect["candidate"]["prefetch"]), (False, True))


class ArmCheckTests(unittest.TestCase):
    ON = {"issued": 3, "staging_slots": 16}

    def test_matching_arms_pass_and_record_their_size(self):
        seen = {}
        self.assertIsNone(P.pool_arm_failure("1-baseline", health(), {"policy": "lru", "prefetch": False}, seen, 2_000_000_000))
        self.assertIsNone(P.pool_arm_failure("1-candidate", health(prefetch=self.ON), {"policy": "lru", "prefetch": True},
                                             seen, 2_000_000_000))
        self.assertEqual(seen, {"1-baseline": 1_000_000_000, "1-candidate": 1_000_000_000})

    def test_wrong_policy_fails(self):
        reason = P.pool_arm_failure("1-candidate", health("lru"), {"policy": "cost-aware", "prefetch": False}, {}, 1)
        self.assertIn("'lru'", reason)
        self.assertIn("'cost-aware'", reason)

    def test_prefetch_must_be_active_in_the_candidate_and_absent_in_the_baseline(self):
        want = {"policy": "lru", "prefetch": True}
        for staged in (None, {"staging_slots": 0}):
            self.assertIn("prefetch", P.pool_arm_failure("c", health(prefetch=staged), want, {}, 1))
        self.assertIn("none was requested", P.pool_arm_failure("b", health(prefetch=self.ON),
                                                               {"policy": "lru", "prefetch": False}, {}, 1))

    def test_different_mapped_bytes_fail_and_name_both_sizes(self):
        seen = {"1-baseline": 1_073_741_824}
        reason = P.pool_arm_failure("1-candidate", health(mapped=536_870_912), {"policy": "lru", "prefetch": False}, seen, 2)
        self.assertIn("536870912", reason)
        self.assertIn("1073741824", reason)
        self.assertNotIn("1-candidate", seen)

    def test_zero_or_missing_mapped_bytes_fail(self):
        want = {"policy": "lru", "prefetch": False}
        seen = {"1-baseline": 1_073_741_824}
        reason = P.pool_arm_failure("1-candidate", health(mapped=0), want, seen, 500_000_000)
        self.assertIn("mapped 0", reason)
        self.assertIn("1073741824", reason)
        self.assertIn("mapped 0", P.pool_arm_failure("1-baseline", health(mapped=0), want, {}, 500_000_000))
        self.assertIn("mapped 0", P.pool_arm_failure("1-baseline", {"cache": {"flashnext_kv": {"expert_pool_policy": "lru"}}},
                                                     want, {}, 1))

    def test_later_round_with_a_resized_pool_fails(self):
        want = {"policy": "lru", "prefetch": False}
        seen = {}
        self.assertIsNone(P.pool_arm_failure("1-baseline", health(mapped=100), want, seen, 1))
        self.assertIn("100", P.pool_arm_failure("2-baseline", health(mapped=200), want, seen, 1))


if __name__ == "__main__":
    unittest.main()
