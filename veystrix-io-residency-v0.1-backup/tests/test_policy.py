import random

import numpy as np
import pytest

from veystrix_residency.policy import (LFU, LRU, OPT, PAGE, POOL, SSD, DoubleResidency,
                                       S3FIFO, top_keys_by_frequency)


def zipf_trace(T=300, L=6, E=64, k=4, a=1.1, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    w = 1.0 / np.arange(1, E + 1) ** a
    perms = [rng.permutation(E) for _ in range(L)]
    steps = []
    for t in range(T):
        for l in range(L):
            p = w / w.sum()
            ids = rng.choice(E, size=k, replace=False, p=p)
            if drift and rng.random() < drift:
                perms[l] = np.roll(perms[l], 1)
            steps.append(tuple(int(l * E + perms[l][i]) for i in ids))
    return steps


def hits(policy, steps):
    n = 0
    for s in steps:
        n += sum(1 for c in policy.step(s) if c == POOL)
    return n


def all_policies(cap, steps):
    k = len(steps[0])
    n_pin = min(cap // 2, cap - k - 1)
    pins = top_keys_by_frequency(steps[: len(steps) // 4], max(n_pin, 0))
    return {
        "lru": LRU(cap), "s3fifo": S3FIFO(cap), "lfu": LFU(cap),
        "pinned": S3FIFO(cap, pinned=pins), "s3fifo_gt0": S3FIFO(cap, promote_gt=0),
    }


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("cap", [8, 20, 60, 150])
def test_opt_is_ceiling(seed, cap):
    steps = zipf_trace(seed=seed, drift=0.05 * (seed % 2))
    h_opt = hits(OPT(cap, steps), steps)
    for name, p in all_policies(cap, steps).items():
        assert hits(p, steps) <= h_opt, name


@pytest.mark.parametrize("seed", range(4))
def test_lru_inclusion_monotone(seed):
    steps = zipf_trace(seed=seed)
    prev = -1
    for cap in [6, 10, 20, 40, 80, 200, 384]:
        h = hits(LRU(cap), steps)
        assert h >= prev
        prev = h


def test_full_capacity_only_compulsory_misses():
    steps = zipf_trace(L=4, E=32)
    uniq = len({x for s in steps for x in s})
    acc = sum(len(s) for s in steps)
    for p in [LRU(uniq), S3FIFO(uniq + 1), LFU(uniq), OPT(uniq, steps)]:
        assert hits(p, steps) == acc - uniq, type(p).__name__


@pytest.mark.parametrize("seed", range(5))
def test_capacity_and_batch_residency_invariants(seed):
    steps = zipf_trace(seed=seed, E=48, k=6)
    cap = 9
    pins = top_keys_by_frequency(steps, 2)
    for p in [LRU(cap), S3FIFO(cap), S3FIFO(cap, pinned=pins), LFU(cap), OPT(cap, steps)]:
        for s in steps:
            p.step(s)
            r = p.resident
            assert len(r) <= cap, type(p).__name__
            assert set(s) <= r, f"{type(p).__name__}: a routed expert was evicted in its own step"


def test_pinned_never_evicted_after_load():
    steps = zipf_trace(seed=3, E=40, k=4)
    pins = top_keys_by_frequency(steps, 5)
    p = S3FIFO(12, pinned=pins)
    loaded = set()
    for s in steps:
        codes = p.step(s)
        for k, c in zip(s, codes):
            if k in pins:
                if k in loaded:
                    assert c == POOL
                loaded.add(k)
        assert loaded <= p.resident


def test_double_residency_with_zero_pagecache_equals_pool():
    steps = zipf_trace(seed=1)
    a = S3FIFO(30)
    b = DoubleResidency(S3FIFO(30), 0)
    for s in steps:
        assert a.step(s) == b.step(s)


def test_double_residency_pagecache_mostly_duplicates():
    """Same total RAM: single tier should not lose to pool+duplicating page cache."""
    wins = 0
    for seed in range(8):
        steps = zipf_trace(seed=seed, T=400)
        pool, pc = 30, 30
        d = DoubleResidency(S3FIFO(pool), pc)
        ssd_double = sum(c == SSD for s in steps for c in d.step(s))
        single = S3FIFO(pool + pc)
        ssd_single = sum(c == SSD for s in steps for c in single.step(s))
        wins += ssd_single <= ssd_double
    assert wins >= 7


def test_rejects_capacity_below_k():
    with pytest.raises(ValueError):
        LRU(3).step((1, 2, 3, 4))


def test_rejects_duplicate_keys():
    with pytest.raises(ValueError):
        LRU(8).step((1, 1))
