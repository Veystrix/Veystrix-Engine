"""Context-drafting drafter (VEYSTRIX_LOOKUP_V2) and lossless sampled verification against point-mass drafts."""
import math
import unittest

from models.shared.prompt_lookup import PromptLookup, ThroughputGate, make_gate, make_lookup


class DrafterTests(unittest.TestCase):
    def test_default_drafter_is_unchanged(self):
        lookup = PromptLookup([1, 2, 3, 4, 5, 1, 2])
        self.assertEqual(lookup.propose(4), [3, 4, 5, 1])
        self.assertEqual(lookup.widths, (2, 3, 4))
        self.assertFalse(lookup.wrap)

    def test_v2_uses_the_whole_prompt_and_long_suffixes(self):
        head = [500 + i for i in range(10)]
        tape = head + [9000 + i for i in range(5000)] + head[:8]
        old = make_lookup(tape, v2=False)
        new = make_lookup(tape, v2=True)
        self.assertEqual(old.propose(4), [])                    # the head is more than 2,048 tokens back
        self.assertEqual(new.propose(4), [508, 509, 9000, 9001])
        self.assertEqual(new.last_width, 8)
        self.assertTrue(new.strong)

    def test_a_match_that_overlaps_its_suffix_continues_its_period(self):
        lookup = PromptLookup([], history=64, widths=(2, 3, 4), wrap=True)
        for token in [9, 7, 8, 7, 8, 7, 8]:
            lookup.append(token)
        self.assertEqual(lookup.propose(6), [7, 8, 7, 8, 7, 8])
        plain = PromptLookup([], history=64, widths=(2, 3, 4), wrap=False)
        for token in [9, 7, 8, 7, 8, 7, 8]:
            plain.append(token)
        self.assertLess(len(plain.propose(6)), 6)

    def test_longest_match_wins(self):
        lookup = PromptLookup([1, 2, 3, 9, 9, 0, 1, 2, 3, 4, 4, 4, 1, 2, 3], history=64, widths=(2, 3, 4), wrap=False)
        self.assertEqual(lookup.propose(2)[:1], [4])            # the most recent 3-gram "1 2 3" was followed by 4

    def test_gate_block_doubles_on_a_copy_and_stays_within_the_cap(self):
        gate = make_gate(7, v2=True)
        for _ in range(6):
            gate.observe(False, 1, 0.5)
        sizes = []
        for _ in range(4):
            proposed = gate.depth
            gate.observe(True, proposed + 1, 0.6, proposed=proposed)
            sizes.append(gate.depth)
        self.assertEqual(sizes, [4, 7, 7, 7])
        self.assertEqual(gate.stats()["accepted"], 2 + 4 + 7 + 7)

    def test_default_gate_still_steps_by_one_up_to_four(self):
        gate = make_gate(7, v2=False)
        self.assertEqual((gate.max_depth, gate.double), (4, False))

    def test_a_losing_gate_cools_down_exactly_as_before(self):
        gate = ThroughputGate()
        gate.observe(False, 1, 0.1)
        for _ in range(2):
            gate.observe(True, 1, 1.0, proposed=2)
        self.assertEqual(gate.cooldown, 16)


class SampledVerificationTests(unittest.TestCase):
    """The emitted token must follow the target distribution whatever the draft is."""

    def distribution(self, sampler, row):
        import mlx.core as mx

        kept, masked = sampler.distribution(row)
        probabilities = mx.softmax(masked, axis=-1).tolist()
        ids = list(range(len(probabilities))) if kept is None else kept.tolist()
        return {token: p for token, p in zip(ids, probabilities) if p > 0}

    def check(self, settings, draft, trials=3000, seed=7):
        import mlx.core as mx
        from veystrix.sampling import Sampler

        mx.random.seed(seed)
        rows = mx.array([[2.0, 1.5, 1.0, 0.2, -0.5, -1.0, 0.0, 0.7], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        sampler = Sampler(settings)
        expected = self.distribution(sampler, rows[0])
        counts = {}
        for _ in range(trials):
            accepted, token = sampler.verify_point_mass(rows, [draft])
            first = draft if accepted == 1 else token
            counts[first] = counts.get(first, 0) + 1
        for token in counts:
            self.assertIn(token, expected, "emitted a token the target cannot produce")
        for token, probability in expected.items():
            observed = counts.get(token, 0) / trials
            sigma = math.sqrt(max(probability * (1 - probability), 1e-9) / trials)
            self.assertLess(abs(observed - probability), 5 * sigma + 0.004, (token, observed, probability))

    def test_plain_temperature_and_a_likely_draft(self):
        from veystrix.sampling import Sampling
        self.check(Sampling(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, presence_penalty=0.0), draft=0)

    def test_an_unlikely_draft(self):
        from veystrix.sampling import Sampling
        self.check(Sampling(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, presence_penalty=0.0), draft=5)

    def test_top_k_and_top_p_with_a_draft_outside_the_support(self):
        from veystrix.sampling import Sampling
        self.check(Sampling(temperature=0.8, top_p=0.9, top_k=4, min_p=0.0, presence_penalty=0.0), draft=7)

    def test_top_k_and_top_p_with_a_draft_inside_the_support(self):
        from veystrix.sampling import Sampling
        self.check(Sampling(temperature=0.8, top_p=0.9, top_k=4, min_p=0.0, presence_penalty=0.0), draft=1)

    def test_all_drafts_accepted_gives_a_draw_from_the_last_row(self):
        import mlx.core as mx
        from veystrix.sampling import Sampler, Sampling

        sampler = Sampler(Sampling(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, presence_penalty=0.0))
        rows = mx.array([[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]])
        self.assertEqual(sampler.verify_point_mass(rows, [0, 1]), (2, 2))

    def test_presence_penalty_counts_accepted_drafts_in_later_rows(self):
        import mlx.core as mx
        from veystrix.sampling import Sampler, Sampling

        sampler = Sampler(Sampling(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, presence_penalty=100.0))
        sampler.observe(9)
        rows = mx.array([[50.0, 0.0, 0.0], [50.0, 49.0, 0.0]])
        accepted, token = sampler.verify_point_mass(rows, [0])
        self.assertEqual(accepted, 1)
        self.assertEqual(token, 1)                              # token 0 is now penalised in the last row
        self.assertIn(0, sampler._seen)


if __name__ == "__main__":
    unittest.main()
