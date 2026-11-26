"""Offline test for scripts/cpp_shadow_compare.py (shadow v2)."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "cpp_shadow_compare.py"
    spec = importlib.util.spec_from_file_location("cpp_shadow_compare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ShadowCompareTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.dir = tempfile.TemporaryDirectory()
        self.py_log = Path(self.dir.name) / "py.jsonl"
        self.cpp_log = Path(self.dir.name) / "cpp.jsonl"

    def tearDown(self):
        self.dir.cleanup()

    def _record(self, tokens, digests, **over):
        r = dict(schema=2, prompt_token_ids=[1, 2, 3], seed=7,
                 chunk_boundaries=None, kv="int8", sampler={"greedy": True},
                 emitted_token_ids=tokens, step_digests=digests,
                 digest_source="logit")
        r.update(over)
        return r

    def test_identical_streams_pass(self):
        rec = self._record([10, 11], ["aa", "bb"])
        self.py_log.write_text(json.dumps(rec) + "\n")
        self.cpp_log.write_text(json.dumps(rec) + "\n")
        self.assertEqual(self.mod.main() if False else self._run(), 0)

    def test_token_divergence_fails(self):
        self.py_log.write_text(json.dumps(self._record([10, 11], ["aa", "bb"])) + "\n")
        self.cpp_log.write_text(json.dumps(self._record([10, 99], ["aa", "bb"])) + "\n")
        self.assertEqual(self._run(), 1)

    def test_token_id_source_is_not_a_pass(self):
        self.py_log.write_text(json.dumps(self._record([10], ["aa"])) + "\n")
        self.cpp_log.write_text(json.dumps(self._record([10], ["aa"],
            digest_source="token_id")) + "\n")
        self.assertEqual(self._run(), 1)

    def _run(self):
        import sys
        from unittest.mock import patch
        with patch.object(sys, "argv", ["cmp", str(self.py_log), str(self.cpp_log)]):
            return self.mod.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
