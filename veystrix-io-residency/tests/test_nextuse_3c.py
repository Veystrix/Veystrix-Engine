import json

import numpy as np
import pytest

from veystrix_residency import sim
from veystrix_residency.policy import (LRU, OPT, POOL, S3FIFO, NextUse, NextUseModel,
                                       compulsory_flags)
from veystrix_residency.trace import RouteTraceWriter, load_prefill_counts, load_trace


def sticky_trace(T=400, L=8, E=64, k=6, stay=0.6, a=0.8, seed=0):
    """Per-layer routing with token-to-token persistence plus Zipf background."""
    rng = np.random.default_rng(seed)
    w = 1 / np.arange(1, E + 1) ** a
    perm = [rng.permutation(E) for _ in range(L)]
    prev = [list(rng.choice(E, k, replace=False)) for _ in range(L)]
    steps = []
    for _ in range(T):
        for l in range(L):
            keep = [x for x in prev[l] if rng.random() < stay]
            pool = [x for x in range(E) if x not in keep]
            p = w[pool] / w[pool].sum()
            new = list(rng.choice(pool, k - len(keep), replace=False, p=p))
            cur = keep + new
            prev[l] = cur
            steps.append(tuple(int(l * E + perm[l][x]) for x in cur))
    return steps


def hits(p, steps, start=0):
    n = 0
    for i, s in enumerate(steps):
        c = p.step(s)
        if i >= start:
            n += sum(x == POOL for x in c)
    return n


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("cap", [10, 40, 120])
def test_nextuse_bounded_by_opt_and_invariants(seed, cap):
    L, E = 8, 64
    calib = sticky_trace(seed=100 + seed, L=L, E=E)
    steps = sticky_trace(seed=seed, L=L, E=E)
    m = NextUseModel.fit(calib, L)
    p = NextUse(cap, L, E, m)
    n = 0
    for s in steps:
        c = p.step(s)
        n += sum(x == POOL for x in c)
        assert set(s) <= p.resident and len(p.resident) <= cap
    assert n <= hits(OPT(cap, steps), steps)


def test_nextuse_beats_recency_on_sticky_traces():
    """Mechanism check only (synthetic). Real verdict comes from Veystrix traces."""
    # cap 20 < one token's 48 lookups: LRU hits the cyclic-scan pathology (0 hits).
    # cap 120: comfortable. (Near one token's working set, cap~60, LRU can edge it: a tie zone.)
    L, E = 8, 64
    for cap in (20, 120):
        wins = 0
        for seed in range(5):
            calib = sticky_trace(seed=50 + seed, L=L, E=E)
            steps = sticky_trace(seed=seed, L=L, E=E)
            h_nu = hits(NextUse(cap, L, E, NextUseModel.fit(calib, L)), steps)
            wins += h_nu > max(hits(LRU(cap), steps), hits(S3FIFO(cap), steps))
        assert wins == 5, cap


def test_model_table_is_monotone_sane():
    L = 8
    m = NextUseModel.fit(sticky_trace(seed=9, L=L), L)
    # E[gap | gap >= a] can never be below a
    for b0 in range(m.table.shape[0]):
        for b1 in range(m.table.shape[1]):
            assert (m.table[b0, b1, 1:50] >= np.arange(1, 50) - 1e-9).all()


def test_compulsory_flags_and_steady_metric():
    steps = [(1, 2), (2, 3), (1, 3), (4, 1)]
    assert compulsory_flags(steps) == [[True, True], [False, True], [False, False], [True, False]]
    c = sim.run(OPT(10, steps), steps, 0)
    assert c["comp"] == 4 and c[2] == 4  # infinite-ish cache: only first touches miss


def test_sim_cli_steady_and_prefix_sweep(tmp_path):
    steps = sticky_trace(T=120, L=4, E=32, k=4, seed=3)
    arr = np.array([[ [k % 32 for k in steps[t * 4 + l]] for l in range(4)] for t in range(120)])
    np.save(tmp_path / "tr.npy", arr)
    out = tmp_path / "o.json"
    rc = sim.main(["--trace", str(tmp_path / "tr.npy"), "--expert-bytes", "1000000",
                   "--budgets-mb", "130", "--policies", "opt,s3fifo,nextuse",
                   "--calib-prefix-tokens", "40", "--prefix-sweep", "60,90,120",
                   "--sweep-budget-mb", "130", "--io-mb-per-ms", "3.541", "--non-io-ms", "58.3",
                   "--out", str(out)])
    assert rc == 0
    d = json.loads(out.read_text())
    opt = [r for r in d["rows"] if r["policy"] == "opt"][0]
    assert opt["steady_no_ssd_hit"] == pytest.approx(1.0)  # 130 slots >= 128 keys: zero capacity misses
    comp = [r["compulsory_share"] for r in d["prefix_sweep"]]
    assert comp[0] >= comp[-1]  # first-touch share shrinks as the trace grows


def test_prefill_counts_roundtrip(tmp_path):
    p = tmp_path / "t.jsonl"
    w = RouteTraceWriter(str(p), prefill=True)
    w.record(0, [[1, 2], [2, 3], [2, 5]])
    w.record(1, [[0, 1], [1, 4], [7, 1]])
    w.record(0, [[2, 9]])
    w.record(1, [[1, 4]])
    w.end_token()
    w.close()
    pc = load_prefill_counts(str(p), 16)
    assert pc[0 * 16 + 2] == 3 and pc[1 * 16 + 1] == 3
    steps, T, L, E = load_trace(str(p), 16)
    assert (T, L) == (1, 2) and steps[0] == (2, 9)
