"""Paged KV block table: exact storage, copy-on-write forks, pinning and spill/restore."""

import unittest

import numpy as np

from models.shared.paged_kv import PagedKV


def arrays(length, heads=2, dim=8, seed=0):
    generator = np.random.default_rng(seed)
    return (generator.normal(size=(1, heads, length, dim)).astype(np.float32),
            generator.normal(size=(1, heads, length, dim)).astype(np.float32))


class PagedKVTest(unittest.TestCase):
    def build(self, length=300, page_size=64):
        keys, values = arrays(length)
        cache = PagedKV(page_size)
        for start, end in ((0, 37), (37, 130), (130, 250), (250, length)):
            cache.append(keys[:, :, start:end], values[:, :, start:end])
        return cache, keys, values

    def test_uneven_appends_gather_exactly(self):
        cache, keys, values = self.build()
        gathered_keys, gathered_values = cache.gather()
        np.testing.assert_array_equal(gathered_keys, keys)
        np.testing.assert_array_equal(gathered_values, values)
        self.assertEqual(cache.tokens, 300)
        self.assertEqual(cache.pages, 5)
        self.assertEqual(cache.table(), (0, 1, 2, 3, 4))

    def test_fork_shares_pages_and_copies_only_the_appended_page(self):
        cache, keys, values = self.build()
        child = cache.fork(150)
        self.assertEqual(child.tokens, 150)
        np.testing.assert_array_equal(child.gather()[0], keys[:, :, :150])
        self.assertEqual(child.stats()["shared_pages"], 2)
        # The partial page is copied so the child can extend it without touching the parent.
        self.assertEqual(child.cow_pages, 1)
        child.append(keys[:, :, 150:170], values[:, :, 150:170])
        np.testing.assert_array_equal(child.gather()[0], keys[:, :, :170])
        np.testing.assert_array_equal(cache.gather()[0], keys)

    def test_sink_and_recent_pins_exclude_eviction_candidates(self):
        cache, keys, values = self.build()
        self.assertEqual(cache.pin_sink(1), 1)
        self.assertEqual(cache.pin_recent(1), 1)
        self.assertEqual(cache.victims(), [1, 2, 3])
        with self.assertRaises(ValueError):
            cache.evict(0)
        with self.assertRaises(ValueError):
            cache.evict(4)          # active page
        cache.evict(1)
        self.assertEqual(cache.table(), (0, -1, 2, 3, 4))
        with self.assertRaises(ValueError):
            cache.gather()
        cache.restore(1, keys[:, :, 64:128], values[:, :, 64:128])
        np.testing.assert_array_equal(cache.gather()[0], keys)

    def test_shared_page_survives_release_until_last_fork_drops_it(self):
        cache, keys, values = self.build()
        child = cache.fork(128)
        page = cache.table()[0]
        # Evicting a complete page in the parent must not free bytes the fork still shows.
        cache.evict(0)
        np.testing.assert_array_equal(child.gather()[0], keys[:, :, :128])
        child.evict(0)
        self.assertNotIn(page, child._store.pages)


if __name__ == "__main__":
    unittest.main()
