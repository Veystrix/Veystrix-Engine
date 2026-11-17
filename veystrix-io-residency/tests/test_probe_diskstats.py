import json
import os
import struct

import numpy as np
import pytest

from veystrix_residency import diskstats, probe


def write_safetensors(path, tensors):
    """tensors: name -> (shape, bytes). Minimal valid safetensors writer."""
    hdr, off, blobs = {}, 0, []
    for name, (shape, data) in tensors.items():
        hdr[name] = {"dtype": "U8", "shape": list(shape), "data_offsets": [off, off + len(data)]}
        off += len(data)
        blobs.append(data)
    h = json.dumps(hdr).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


@pytest.fixture
def model(tmp_path):
    rng = np.random.default_rng(0)
    E, L = 8, 3
    tensors = {}
    for l in range(L):
        for proj, n in (("up", 40000), ("down", 40000)):
            for part, m in (("weight", n), ("scales", n // 16), ("biases", n // 16)):
                tensors[f"layers.{l}.experts.{proj}.{part}"] = ((E, m), rng.bytes(E * m))
        tensors[f"layers.{l}.attn.weight"] = ((128,), rng.bytes(128))
    write_safetensors(tmp_path / "model-00001.safetensors", tensors)
    return tmp_path, tensors, E, L


def test_build_index_ranges_match_bytes(model):
    d, tensors, E, L = model
    idx = probe.build_index(str(d), r"layers\.(?P<layer>\d+)\.experts\..*")
    assert len(idx["entries"]) == E * L
    per_expert = sum(m for (proj, n) in (("up", 40000), ("down", 40000))
                     for m in (n, n // 16, n // 16))
    assert idx["expert_bytes"] == [per_expert]
    path = str(d / "model-00001.safetensors")
    raw = open(path, "rb").read()
    for e in idx["entries"]:
        l, x = e["layer"], e["expert"]
        got = b"".join(raw[o:o + n] for _, o, n in e["ranges"])
        want = b"".join(
            tensors[f"layers.{l}.experts.{p}.{q}"][1][x * m:(x + 1) * m]
            for p, n in (("up", 40000), ("down", 40000))
            for q, m in (("weight", n), ("scales", n // 16), ("biases", n // 16)))
        assert got == want


def test_range_resident_bytes_partial_pages():
    P = probe.PAGE
    res = np.array([True, False, True, True, False])
    size = 5 * P
    assert probe.range_resident_bytes(res, 10, 20, size) == 20
    assert probe.range_resident_bytes(res, P + 5, 10, size) == 0
    # spans pages 0..3: page0 partial (P-100) + page1 0 + page2 full + page3 partial(50)
    assert probe.range_resident_bytes(res, 100, 3 * P - 100 + 50, size) == (P - 100) + P + 50


def test_mincore_sees_cache_state(model):
    d, _, _, _ = model
    path = str(d / "model-00001.safetensors")
    idx = probe.build_index(str(d), r"layers\.(?P<layer>\d+)\.experts\..*")
    fd = os.open(path, os.O_RDONLY)
    os.fsync(fd)
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    snap0 = probe.snapshot(idx)
    if snap0["pagecache_fraction"] > 0.05:
        pytest.skip("filesystem keeps pages resident (tmpfs/overlay); cannot drop cache here")
    target = idx["entries"][5]
    with open(path, "rb") as f:
        for _, o, n in target["ranges"]:
            f.seek(o)
            f.read(n)
    snap1 = probe.snapshot(idx, pool_keys=[[target["layer"], target["expert"]]])
    tb = sum(r[2] for r in target["ranges"])
    assert snap1["dup_bytes"] == tb
    assert snap1["pagecache_expert_bytes"] >= tb


def test_file_residency_after_write_is_resident(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(os.urandom(5 * probe.PAGE + 7))
    r = probe.file_residency(str(p))
    assert r.shape == (6,)
    assert r.all()


SAMPLE_IOREG = '''
+-o IOBlockStorageDriver  <class IOBlockStorageDriver, id 0x100000abc>
    {
      "Statistics" = {"Operations (Write)"=100,"Latency Time (Write)"=0,"Bytes (Read)"=1000000,"Errors (Write)"=0,"Total Time (Read)"=5000000,"Latency Time (Read)"=0,"Retries (Read)"=0,"Errors (Read)"=0,"Total Time (Write)"=9,"Bytes (Write)"=4096,"Operations (Read)"=10,"Retries (Write)"=0}
    }
+-o IOBlockStorageDriver  <class IOBlockStorageDriver, id 0x100000def>
    {
      "Statistics" = {"Operations (Write)"=0,"Bytes (Read)"=50,"Total Time (Read)"=1,"Bytes (Write)"=0,"Operations (Read)"=1}
    }
'''


def test_parse_ioreg_and_delta():
    before = diskstats.parse_ioreg(SAMPLE_IOREG)
    assert len(before) == 2 and before[0]["Bytes (Read)"] == 1000000
    after = [dict(b) for b in before]
    after[0]["Bytes (Read)"] += 436_000_000 * 4
    after[0]["Operations (Read)"] += 4 * 138
    after[0]["Total Time (Read)"] += 2_000_000_000
    r = diskstats.delta(before, after, wall_s=1.0, tokens=4)
    assert r["phys_mb_per_token"] == pytest.approx(436.0)
    assert r["read_ops_per_token"] == pytest.approx(138)
    assert r["mean_read_kb"] == pytest.approx(436e6 / 138 / 1024)
