import json

import numpy as np
import pytest

from veystrix_residency import probe
from veystrix_residency.policy import POOL, S3FIFO, top_keys_by_frequency
from veystrix_residency.pool import RangeReader, SingleTierExpertPool, from_env
from tests.test_probe_diskstats import write_safetensors


@pytest.fixture
def experts(tmp_path):
    rng = np.random.default_rng(1)
    E, L = 16, 4
    tensors = {}
    for l in range(L):
        for proj in ("gate", "up", "down"):
            for part, m in (("weight", 3000), ("scales", 190), ("biases", 190)):
                tensors[f"layers.{l}.experts.{proj}.{part}"] = ((E, m), rng.bytes(E * m))
    write_safetensors(tmp_path / "m.safetensors", tensors)
    idx = probe.build_index(str(tmp_path), r"layers\.(?P<layer>\d+)\.experts\..*")
    ranges = {(e["layer"], e["expert"]): [tuple(r) for r in e["ranges"]] for e in idx["entries"]}
    raw = open(tmp_path / "m.safetensors", "rb").read()
    truth = {k: b"".join(raw[o:o + n] for _, o, n in v) for k, v in ranges.items()}
    return ranges, truth, E, L


def stream(E, L, T=250, k=4, seed=0):
    rng = np.random.default_rng(seed)
    w = 1 / np.arange(1, E + 1) ** 1.0
    return [tuple((l, int(x)) for x in rng.choice(E, size=k, replace=False, p=w / w.sum()))
            for _ in range(T) for l in range(L)]


@pytest.mark.parametrize("n_slots,seed", [(6, 0), (9, 1), (20, 2), (64, 3)])
def test_bytes_exact_under_eviction(experts, n_slots, seed):
    ranges, truth, E, L = experts
    reader = RangeReader(ranges, nocache=True)
    slot = reader.slot_bytes((0, 0))
    pins = top_keys_by_frequency(stream(E, L, T=30, seed=99), max(0, min(n_slots // 3, n_slots - 5)))
    pool = SingleTierExpertPool(n_slots, slot, reader, pinned=pins)
    for keys in stream(E, L, seed=seed):
        views = pool.get_batch(keys)
        for k, v in zip(keys, views):  # checked AFTER the whole batch was served
            assert bytes(v) == truth[k], k
    reader.close()
    assert 0.0 < pool.hit_rate < 1.0


def test_pool_matches_simulator_sequence(experts):
    ranges, truth, E, L = experts
    reader = RangeReader(ranges)
    s = stream(E, L, seed=5)
    pins = top_keys_by_frequency(s[:200], 4)
    pool = SingleTierExpertPool(14, reader.slot_bytes((0, 0)), reader, pinned=pins)
    sim = S3FIFO(14, pinned=pins)
    for keys in s:
        h0 = pool.hits
        pool.get_batch(keys)
        sim_hits = sum(c == POOL for c in sim.step(keys))
        assert pool.hits - h0 == sim_hits


def test_pinned_read_once(experts):
    ranges, truth, E, L = experts
    counts = {}
    base = RangeReader(ranges)

    def counting(k, out):
        counts[k] = counts.get(k, 0) + 1
        base(k, out)

    s = stream(E, L, seed=7)
    pins = top_keys_by_frequency(s, 5)
    pool = SingleTierExpertPool(10, base.slot_bytes((0, 0)), counting, pinned=pins)
    for keys in s:
        pool.get_batch(keys)
    for p in pins:
        assert counts.get(p, 0) <= 1


def test_wait_slot_free_called_before_every_overwrite(experts):
    ranges, truth, E, L = experts
    reader = RangeReader(ranges)
    seen = []
    pool = SingleTierExpertPool(8, reader.slot_bytes((0, 0)), reader, wait_slot_free=seen.append)
    for keys in stream(E, L, T=50):
        pool.get_batch(keys)
    assert len(seen) == pool.misses


def test_flag_off_is_inert(experts):
    ranges, _, _, _ = experts
    assert from_env(ranges, env={}) is None
    assert from_env(ranges, env={"VEYSTRIX_SINGLE_TIER_POOL": "0"}) is None


def test_flag_on_builds_pool(experts, tmp_path):
    ranges, truth, E, L = experts
    pin = tmp_path / "pins.json"
    pin.write_text(json.dumps([[0, 0], [1, 3]]))
    slot = sum(n for _, _, n in ranges[(0, 0)])
    pool = from_env(ranges, env={"VEYSTRIX_SINGLE_TIER_POOL": "1",
                                 "VEYSTRIX_POOL_BUDGET_MB": str(slot * 12 / 1e6 + 1e-9),
                                 "VEYSTRIX_POOL_PIN_FILE": str(pin)})
    assert pool.policy.cap == 12 and (1, 3) in pool.policy.pinned
    v = pool.get_batch([(1, 3), (2, 2), (0, 5), (3, 1)])
    assert bytes(v[0]) == truth[(1, 3)]


def test_nextuse_pool_bytes_exact_and_matches_sim(experts, tmp_path):
    from veystrix_residency.policy import NextUse, NextUseModel
    ranges, truth, E, L = experts
    calib_stream = stream(E, L, seed=11)
    calib_steps = [tuple(l * E + x for l, x in s) for s in calib_stream]
    model = NextUseModel.fit(calib_steps, L)
    reader = RangeReader(ranges)
    pool = SingleTierExpertPool(10, reader.slot_bytes((0, 0)), reader,
                                policy=NextUse(10, L, E, model), policy_key=lambda k: k[0] * E + k[1])
    sim = NextUse(10, L, E, model)
    for keys in stream(E, L, seed=12):
        h0 = pool.hits
        views = pool.get_batch(keys)
        for k, v in zip(keys, views):
            assert bytes(v) == truth[k]
        assert pool.hits - h0 == sum(c == POOL for c in sim.step([l * E + x for l, x in keys]))


def test_from_env_nextuse(experts, tmp_path):
    ranges, truth, E, L = experts
    tr = tmp_path / "calib.jsonl"
    with open(tr, "w") as f:
        for i, s in enumerate(stream(E, L, T=60, seed=4)):
            f.write(json.dumps({"t": i // L, "l": s[0][0], "e": [x for _, x in s]}) + "\n")
    slot = sum(n for _, _, n in ranges[(0, 0)])
    pool = from_env(ranges, env={"VEYSTRIX_SINGLE_TIER_POOL": "1", "VEYSTRIX_POOL_POLICY": "nextuse",
                                 "VEYSTRIX_POOL_BUDGET_MB": str(slot * 9 / 1e6 + 1e-9),
                                 "VEYSTRIX_POOL_NEXTUSE_CALIB": str(tr)})
    for keys in stream(E, L, T=40, seed=8):
        for k, v in zip(keys, pool.get_batch(keys)):
            assert bytes(v) == truth[k]
