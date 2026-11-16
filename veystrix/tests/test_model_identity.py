import unittest

from veystrix import model_identity as mi


class ModelIdentityTests(unittest.TestCase):
    def test_flashnext_identity_matches_the_current_baseline(self):
        q = mi.FLASHNEXT
        self.assertEqual(q.family, "flashnext")
        self.assertEqual(q.display_name, "Veystrix Flash-Next")
        self.assertEqual(q.model_types, frozenset({"qwen4_exp", "qwen4_exp_text"}))
        self.assertEqual(q.served_id, "veystrix-flashnext")
        self.assertEqual(q.context_limit, 262_144)
        self.assertEqual(q.checkpoint_alias, "vontra-mtp")
        self.assertEqual(q.checkpoint_directory, "Qwen3.8-Flash-Next-MLX-4bit-MTP")
        self.assertEqual(q.model_type, "qwen4_exp")
        self.assertEqual(q.text_model_type, "qwen4_exp_text")

    def test_nemotron_identity_matches_the_current_baseline(self):
        n = mi.NEMOTRON
        self.assertEqual(n.family, "nemotron35-lightning")
        self.assertEqual(n.model_types, frozenset({"nemotron_h"}))
        self.assertEqual(n.served_id, "veystrix-nemotron35-lightning")
        self.assertEqual(n.context_limit, 1_048_576)

    def test_lookup_helpers(self):
        self.assertEqual(mi.for_family("flashnext"), mi.FLASHNEXT)
        self.assertEqual(mi.display_name("nemotron35-lightning"), "Veystrix Nemotron 3.5 Lightning")
        self.assertEqual(mi.context_limit("flashnext"), 262_144)
        self.assertEqual(mi.model_types("flashnext"), frozenset({"qwen4_exp", "qwen4_exp_text"}))
        self.assertEqual(mi.served_id("nemotron35-lightning"), "veystrix-nemotron35-lightning")

    def test_unknown_family_is_safe(self):
        self.assertEqual(mi.for_family(None), mi.UNKNOWN)
        self.assertEqual(mi.for_family("not-a-model"), mi.UNKNOWN)
        self.assertEqual(mi.context_limit("not-a-model"), 32_768)
        self.assertEqual(mi.model_types("not-a-model"), frozenset())

    def test_explicit_model_types_are_subsets_of_the_gate(self):
        for identity in mi.IDENTITIES.values():
            for value in (identity.model_type, identity.text_model_type):
                if value:
                    self.assertIn(value, identity.model_types)

    def test_identities_are_immutable(self):
        with self.assertRaises(Exception):
            mi.FLASHNEXT.context_limit = 4096  # type: ignore[misc]
