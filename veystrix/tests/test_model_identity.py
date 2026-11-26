import unittest

from veystrix import model_identity as mi


class ModelIdentityTests(unittest.TestCase):
    def test_inkling2_mini_identity_matches_the_current_baseline(self):
        ink = mi.INKLING2_MINI
        self.assertEqual(ink.family, "inkling2-mini")
        self.assertEqual(ink.display_name, "Inkling 2 Mini")
        self.assertEqual(ink.model_types, frozenset({"inkling2_loop", "inkling2_loop_text"}))
        self.assertEqual(ink.served_id, "veystrix-inkling2-mini")
        self.assertEqual(ink.context_limit, 1_048_576)
        self.assertEqual(ink.checkpoint_alias, "inkling2-mini")
        self.assertEqual(ink.checkpoint_directory, "Inkling-2-Mini-oQ4e-mtp")
        self.assertEqual(ink.model_type, "inkling2_loop")
        self.assertEqual(ink.text_model_type, "inkling2_loop_text")
        self.assertEqual(ink.cpp_runtime, "generic")

    def test_lookup_helpers(self):
        self.assertEqual(mi.for_family("inkling2-mini"), mi.INKLING2_MINI)
        self.assertEqual(mi.display_name("inkling2-mini"), "Inkling 2 Mini")
        self.assertEqual(mi.context_limit("inkling2-mini"), 1_048_576)
        self.assertEqual(mi.model_types("inkling2-mini"), frozenset({"inkling2_loop", "inkling2_loop_text"}))
        self.assertEqual(mi.served_id("inkling2-mini"), "veystrix-inkling2-mini")
        self.assertEqual(mi.for_model_type("inkling2_loop"), mi.INKLING2_MINI)

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
            mi.INKLING2_MINI.context_limit = 4096  # type: ignore[misc]
