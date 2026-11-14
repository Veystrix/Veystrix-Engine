"""Tokenized fragment cache: exact ids, identity isolation, bounds and reopening."""

import tempfile
import unittest
from pathlib import Path

from veystrix.fragment_cache import TokenizedFragmentCache, identity_for_backend, open_for_backend


class _Tokenizer:
    name_or_path = "tiny-tokenizer"

    def __len__(self):
        return 16


class _Backend:
    model_path = "/models/tiny"
    tokenizer = _Tokenizer()


class FragmentCacheTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_roundtrip_and_identity_isolation(self):
        cache = TokenizedFragmentCache(self.directory, identity_for_backend(_Backend()))
        cache.put("review this diff", [11, 12, 13])
        self.assertEqual(cache.get("review this diff"), [11, 12, 13])
        self.assertIsNone(cache.get("another prompt"))
        stats = cache.stats()
        self.assertEqual(stats["entries"], 1)
        self.assertEqual((stats["hits"], stats["misses"]), (1, 1))

        other = TokenizedFragmentCache(self.directory / "other", {"runtime": "x", "vocab": 999})
        self.assertIsNone(other.get("review this diff"))

    def test_reopening_keeps_entries_and_replace_updates_ids(self):
        cache = TokenizedFragmentCache(self.directory, {"runtime": "x", "vocab": 16})
        cache.put("same text", [1, 2, 3])
        cache.put("same text", [4, 5, 6])
        cache.close()
        reopened = TokenizedFragmentCache(self.directory, {"runtime": "x", "vocab": 16})
        self.assertEqual(reopened.get("same text"), [4, 5, 6])

    def test_entry_and_byte_bounds_evict_oldest(self):
        cache = TokenizedFragmentCache(self.directory, {"runtime": "x"}, max_entries=2, max_bytes=1 << 20)
        for index in range(4):
            cache.put(f"text-{index}", [index] * 4)
        self.assertEqual(cache.stats()["entries"], 2)
        self.assertIsNone(cache.get("text-0"))
        self.assertIsNotNone(cache.get("text-3"))

    def test_oversized_payload_is_not_stored(self):
        cache = TokenizedFragmentCache(self.directory, {"runtime": "x"}, max_bytes=4)
        cache.put("too big", [1, 2, 3])
        self.assertIsNone(cache.get("too big"))

    def test_replaced_tokenizer_does_not_reuse_ids(self):
        checkpoint = self.directory / "model"
        checkpoint.mkdir()
        tokenizer = checkpoint / "tokenizer.json"
        backend = _Backend()
        backend.model_path = str(checkpoint)
        tokenizer.write_text('{"vocab":{"alpha":5}}')
        cache = open_for_backend(backend, self.directory)
        cache.put("alpha", [5])
        cache.close()
        tokenizer.write_text('{"vocab":{"alpha":7}}')
        cache = open_for_backend(backend, self.directory)
        self.addCleanup(cache.close)
        self.assertIsNone(cache.get("alpha"))

    def test_invalid_id_blob_is_a_miss(self):
        cache = TokenizedFragmentCache(self.directory, {"runtime": "x"})
        self.addCleanup(cache.close)
        cache.put("prompt", [5])
        with cache._db:
            cache._db.execute("UPDATE fragments SET ids = ?", (b"x",))
        self.assertIsNone(cache.get("prompt"))


if __name__ == "__main__":
    unittest.main()
