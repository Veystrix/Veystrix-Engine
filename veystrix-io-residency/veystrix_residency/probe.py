r"""Page-cache residency probe for expert bytes (read-only, zero I/O).

Maps each shard file PROT_READ/MAP_SHARED and calls mincore() — the same
mechanism vmtouch uses on macOS. mmap+mincore never faults a page in, so the
probe does not disturb the cache it measures. Run it from a SEPARATE process
while the engine is mid-decode: the page cache is system-wide.

  # 1) build an index of every routed expert's byte ranges
  python -m veystrix_residency.probe build-index --model-dir ~/models/nemotron \
      --pattern 'backbone\.layers\.(?P<layer>\d+)\.mixer\.experts\..*' --out nemo_index.json

  # 2) during decode, snapshot (optionally with the pool's resident key list)
  python -m veystrix_residency.probe snapshot --index nemo_index.json \
      [--pool-keys pool_keys.json] --out residency_t60.json

The decisive output is `dup_bytes`: expert bytes held BOTH in the pool and in
the page cache. That is RAM a single-tier pool could hand back as new unique
capacity at zero extra footprint.

If your store.py already knows each expert's ranges, skip build-index and dump
its table as {"entries": [{"layer": l, "expert": e, "ranges": [[path, off, len], ...]}]}.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import mmap
import os
import re
import struct
import sys
from typing import Dict, List, Tuple

import numpy as np

PROT_READ = 0x1
MAP_SHARED = 0x1

_libc = ctypes.CDLL(ctypes.util.find_library("c") or None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                       ctypes.c_int, ctypes.c_longlong]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
MAP_FAILED = ctypes.c_void_p(-1).value

PAGE = mmap.PAGESIZE  # 16384 on Apple Silicon


def file_residency(path: str) -> np.ndarray:
    """Boolean array, one entry per page of `path`: resident in page cache?"""
    size = os.path.getsize(path)
    if size == 0:
        return np.zeros(0, dtype=bool)
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = _libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
        if addr is None or addr == MAP_FAILED:
            raise OSError(ctypes.get_errno(), f"mmap failed for {path}")
        try:
            npages = (size + PAGE - 1) // PAGE
            vec = ctypes.create_string_buffer(npages)
            if _libc.mincore(addr, size, vec) != 0:
                raise OSError(ctypes.get_errno(), f"mincore failed for {path}")
            return (np.frombuffer(vec.raw, dtype=np.uint8) & 1).astype(bool)
        finally:
            _libc.munmap(addr, size)
    finally:
        os.close(fd)


def range_resident_bytes(res: np.ndarray, off: int, length: int, file_size: int) -> int:
    """Bytes of [off, off+length) whose page is resident (exact at byte level)."""
    if length <= 0:
        return 0
    end = off + length
    p0, p1 = off // PAGE, (end - 1) // PAGE
    total = 0
    pages = res[p0:p1 + 1]
    if p0 == p1:
        return length if pages[0] else 0
    if pages[0]:
        total += (p0 + 1) * PAGE - off
    if pages[-1]:
        total += end - p1 * PAGE
    total += int(pages[1:-1].sum()) * PAGE
    return total


def read_safetensors_header(path: str) -> Tuple[int, Dict]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    return 8 + n, hdr


def build_index(model_dir: str, pattern: str, experts_dim0: bool = True) -> Dict:
    """Index stacked expert tensors (MLX switch_mlp layout: shape [E, ...]).

    Every tensor whose name matches `pattern` (with a named group `layer`)
    contributes one equal slice per expert along dim 0. Per-expert ranges
    therefore cover all sub-slices (weight/scales/biases x projections).
    """
    rx = re.compile(pattern)
    entries: Dict[Tuple[int, int], List] = {}
    sizes = set()
    for fn in sorted(os.listdir(model_dir)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(model_dir, fn)
        base, hdr = read_safetensors_header(path)
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            m = rx.fullmatch(name)
            if not m:
                continue
            layer = int(m.group("layer"))
            s, e = meta["data_offsets"]
            E = meta["shape"][0]
            nbytes = e - s
            if nbytes % E:
                raise ValueError(f"{name}: {nbytes} bytes not divisible by {E} experts")
            per = nbytes // E
            for x in range(E):
                entries.setdefault((layer, x), []).append([path, base + s + x * per, per])
    out = []
    for (l, x), rngs in sorted(entries.items()):
        sizes.add(sum(r[2] for r in rngs))
        out.append({"layer": l, "expert": x, "ranges": rngs})
    return {"page_size": PAGE, "expert_bytes": sorted(sizes), "entries": out}


def snapshot(index: Dict, pool_keys=None) -> Dict:
    files = sorted({r[0] for e in index["entries"] for r in e["ranges"]})
    res = {p: file_residency(p) for p in files}
    fsize = {p: os.path.getsize(p) for p in files}
    pool = {(int(l), int(x)) for l, x in pool_keys} if pool_keys else set()
    total = resident = dup = 0
    full = partial = 0
    per_layer: Dict[int, int] = {}
    for e in index["entries"]:
        b = sum(r[2] for r in e["ranges"])
        rb = sum(range_resident_bytes(res[p], off, ln, fsize[p]) for p, off, ln in e["ranges"])
        total += b
        resident += rb
        per_layer[e["layer"]] = per_layer.get(e["layer"], 0) + rb
        full += rb == b
        partial += 0 < rb < b
        if (e["layer"], e["expert"]) in pool:
            dup += rb
    return {
        "page_size": PAGE,
        "expert_bytes_total": total,
        "pagecache_expert_bytes": resident,
        "pagecache_fraction": resident / total if total else 0.0,
        "experts_fully_resident": full,
        "experts_partially_resident": partial,
        "pool_keys": len(pool),
        "dup_bytes": dup if pool_keys is not None else None,
        "per_layer_resident_bytes": per_layer,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-index")
    b.add_argument("--model-dir", required=True)
    b.add_argument("--pattern", required=True, help="regex with (?P<layer>\\d+) matching routed-expert tensors")
    b.add_argument("--out", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--index", required=True)
    s.add_argument("--pool-keys", help='JSON list of [layer, expert] currently in the pool')
    s.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "build-index":
        idx = build_index(a.model_dir, a.pattern)
        with open(a.out, "w") as f:
            json.dump(idx, f)
        print(f"{len(idx['entries'])} experts, expert_bytes={idx['expert_bytes']}, page={PAGE}")
        return 0
    with open(a.index) as f:
        idx = json.load(f)
    pk = None
    if a.pool_keys:
        with open(a.pool_keys) as f:
            pk = json.load(f)
    snap = snapshot(idx, pk)
    txt = json.dumps(snap, indent=2)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)
    gb = 1e9
    print(f"expert bytes total      {snap['expert_bytes_total']/gb:8.3f} GB")
    print(f"in page cache           {snap['pagecache_expert_bytes']/gb:8.3f} GB "
          f"({snap['pagecache_fraction']:.1%})")
    if snap["dup_bytes"] is not None:
        print(f"duplicated with pool    {snap['dup_bytes']/gb:8.3f} GB  (pool keys: {snap['pool_keys']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
