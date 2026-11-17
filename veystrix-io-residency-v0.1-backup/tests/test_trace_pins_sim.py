import json

import numpy as np

from veystrix_residency import pins, sim
from veystrix_residency.trace import RouteTraceWriter, load_trace


def test_writer_roundtrip_and_prefill_skip(tmp_path):
    p = tmp_path / "t.jsonl"
    w = RouteTraceWriter(str(p))
    rng = np.random.default_rng(0)
    arr = np.stack([[rng.choice(32, 4, replace=False) for _ in range(3)] for _ in range(5)])
    w.record(0, [[1, 2, 3, 4], [5, 6, 7, 8]])  # prefill rows -> ignored
    for t in range(5):
        for l in range(3):
            w.record(l, [arr[t, l].tolist()])  # decode shape [1, k]
        w.end_token()
    w.close()
    steps, T, L, E = load_trace(str(p), experts_per_layer=32)
    assert (T, L) == (5, 3)
    assert steps[4] == tuple(1 * 32 + int(x) for x in arr[1, 1])
    np.save(tmp_path / "t.npy", arr)
    assert load_trace(str(tmp_path / "t.npy"), 32)[0] == steps


def test_pins_and_sim_cli(tmp_path):
    rng = np.random.default_rng(2)
    arr = np.stack([[rng.choice(64, 6, replace=False) for _ in range(4)] for _ in range(120)])
    np.save(tmp_path / "tr.npy", arr)
    out = tmp_path / "pins.json"
    assert pins.main(["--trace", str(tmp_path / "tr.npy"), "--expert-bytes", "1000000",
                      "--budget-mb", "40", "--pin-frac", "0.5", "--out", str(out)]) == 0
    pairs = json.loads(out.read_text())
    assert len(pairs) == 20 and all(0 <= l < 4 and 0 <= e < 64 for l, e in pairs)
    rc = sim.main(["--trace", str(tmp_path / "tr.npy"), "--expert-bytes", "1000000",
                   "--budgets-mb", "40,80", "--policies", "s3fifo,opt,double-s3fifo",
                   "--double-pc-mb", "20", "--io-mb-per-ms", "3.541", "--non-io-ms", "58.3",
                   "--calibrate", "s3fifo:40:0.99", "--out", str(tmp_path / "s.json")])
    assert rc == 3  # calibration must FAIL loudly when the sim disagrees with the measurement
    rows = json.loads((tmp_path / "s.json").read_text())["rows"]
    byp = {(r["policy"], r["budget_mb"]): r for r in rows}
    assert byp[("opt", 40.0)]["pool_hit"] >= byp[("s3fifo", 40.0)]["pool_hit"]
    assert byp[("double-s3fifo", 40.0)]["no_ssd_hit"] >= byp[("s3fifo", 40.0)]["pool_hit"]
