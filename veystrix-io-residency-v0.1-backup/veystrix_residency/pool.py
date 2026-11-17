"""Reference single-tier expert pool (opt-in; OFF unless the flag is set).

Hypothesis it tests: on 16 GB, the in-process pool and the OS page cache hold
largely the SAME expert bytes (pread fills both), so every MB given to the pool
buys little unique capacity and adds memory pressure. One tier with F_NOCACHE
reads turns that duplicated RAM into unique capacity at the same footprint.

Flags (read once at engine start; unset = today's behaviour, byte for byte):
  VEYSTRIX_SINGLE_TIER_POOL=1        enable
  VEYSTRIX_POOL_BUDGET_MB=<int>      total pool RAM (set = old pool MB + measured dup MB)
  VEYSTRIX_POOL_PIN_FILE=<json>      optional [[layer, expert], ...] from calibration prompts
  VEYSTRIX_POOL_NOCACHE=1            open expert fds with F_NOCACHE (default 1 when enabled)

Exactness: the pool only decides WHERE bytes come from. Every slot is filled by
reading exactly the expert's on-disk ranges, so the arrays handed to MLX are
identical to today's; routing, weights and kernels are untouched. The tests
check this byte-for-byte under eviction pressure.

Integration notes for store.py / the C++ pager:
  * slot memory should be the existing Metal-backed pool buffer; read_into()
    gets a writable memoryview of one slot.
  * wait_slot_free(slot) MUST block until no in-flight command buffer still
    reads that slot (the evicted expert may belong to the previous layer's
    still-running GPU work). The current pool already has this fence; reuse it.
"""
from __future__ import annotations

import fcntl
import json
import os
import sys
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .policy import POOL, SSD, S3FIFO

F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)  # macOS only


def open_expert_fd(path: str, nocache: bool) -> int:
    fd = os.open(path, os.O_RDONLY)
    if nocache and sys.platform == "darwin":
        fcntl.fcntl(fd, F_NOCACHE, 1)
    return fd


class RangeReader:
    """Fill a slot with one expert's ranges, packed in index order."""

    def __init__(self, ranges: Dict[Hashable, List[Tuple[str, int, int]]], nocache: bool = True):
        self.ranges = ranges
        self.fds: Dict[str, int] = {}
        self.nocache = nocache
        self.bytes_read = 0
        self.preads = 0

    def _fd(self, path: str) -> int:
        fd = self.fds.get(path)
        if fd is None:
            fd = self.fds[path] = open_expert_fd(path, self.nocache)
        return fd

    def slot_bytes(self, key: Hashable) -> int:
        return sum(n for _, _, n in self.ranges[key])

    def __call__(self, key: Hashable, out: memoryview) -> None:
        pos = 0
        for path, off, n in self.ranges[key]:
            got = 0
            while got < n:
                r = os.preadv(self._fd(path), [out[pos + got: pos + n]], off + got)
                if r <= 0:
                    raise IOError(f"short read for {key} at {path}:{off + got}")
                got += r
                self.preads += 1
            pos += n
            self.bytes_read += n

    def close(self) -> None:
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


class SingleTierExpertPool:
    def __init__(self, n_slots: int, slot_bytes: int,
                 read_into: Callable[[Hashable, memoryview], None],
                 pinned: Iterable[Hashable] = (), small_frac: float = 0.1, promote_gt: int = 1,
                 wait_slot_free: Optional[Callable[[int], None]] = None,
                 backing: Optional[np.ndarray] = None):
        self.policy = S3FIFO(n_slots, small_frac=small_frac, pinned=pinned, promote_gt=promote_gt)
        self.slot_bytes = slot_bytes
        self.mem = backing if backing is not None else np.empty((n_slots, slot_bytes), dtype=np.uint8)
        if self.mem.shape != (n_slots, slot_bytes):
            raise ValueError("backing must be [n_slots, slot_bytes] uint8")
        self.read_into = read_into
        self.wait_slot_free = wait_slot_free or (lambda s: None)
        self.slot_of: Dict[Hashable, int] = {}
        self.free: List[int] = list(range(n_slots - 1, -1, -1))
        self.hits = self.misses = 0

    def get_batch(self, keys: Sequence[Hashable]) -> List[memoryview]:
        codes = self.policy.step(keys)
        for v in self.policy.evicted:
            s = self.slot_of.pop(v)
            self.free.append(s)
        views = []
        for k, c in zip(keys, codes):
            if c == POOL:
                self.hits += 1
                s = self.slot_of[k]
            else:
                self.misses += 1
                s = self.free.pop()
                self.wait_slot_free(s)
                self.read_into(k, memoryview(self.mem[s]))
                self.slot_of[k] = s
            views.append(memoryview(self.mem[s]))
        return views

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    def resident_keys(self) -> List[Hashable]:
        """For probe.py --pool-keys."""
        return list(self.slot_of)


def from_env(ranges: Dict[Tuple[int, int], List[Tuple[str, int, int]]],
             env=os.environ) -> Optional[SingleTierExpertPool]:
    if env.get("VEYSTRIX_SINGLE_TIER_POOL") != "1":
        return None
    reader = RangeReader(ranges, nocache=env.get("VEYSTRIX_POOL_NOCACHE", "1") == "1")
    sizes = {reader.slot_bytes(k) for k in ranges}
    if len(sizes) != 1:
        raise ValueError(f"non-uniform expert sizes {sorted(sizes)[:4]}...; pool needs one slot size")
    slot = sizes.pop()
    n_slots = int(float(env["VEYSTRIX_POOL_BUDGET_MB"]) * 1e6 // slot)
    pinned: List[Tuple[int, int]] = []
    if env.get("VEYSTRIX_POOL_PIN_FILE"):
        with open(env["VEYSTRIX_POOL_PIN_FILE"]) as f:
            pinned = [tuple(x) for x in json.load(f)]
    return SingleTierExpertPool(n_slots, slot, reader, pinned=pinned)
