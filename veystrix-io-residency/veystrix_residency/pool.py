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
  VEYSTRIX_POOL_POLICY=s3fifo|nextuse   eviction policy (default s3fifo)
  VEYSTRIX_POOL_NEXTUSE_CALIB=<trace>   calibration trace (different prompts) for nextuse

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

from .policy import POOL, SSD, NextUse, NextUseModel, S3FIFO
from .trace import load_trace

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
                 backing: Optional[np.ndarray] = None, policy=None,
                 policy_key: Optional[Callable[[Hashable], Hashable]] = None):
        self.policy = policy if policy is not None else S3FIFO(
            n_slots, small_frac=small_frac, pinned=pinned, promote_gt=promote_gt)
        if getattr(self.policy, "cap", n_slots) != n_slots:
            raise ValueError("policy capacity must equal n_slots")
        self.pkey = policy_key or (lambda k: k)
        self.from_pkey: Dict[Hashable, Hashable] = {}
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
        pk = [self.pkey(k) for k in keys]
        for k, p in zip(keys, pk):
            self.from_pkey[p] = k
        codes = self.policy.step(pk)
        for pv in self.policy.evicted:
            s = self.slot_of.pop(self.from_pkey[pv])
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
    pol = env.get("VEYSTRIX_POOL_POLICY", "s3fifo")
    if pol == "s3fifo":
        return SingleTierExpertPool(n_slots, slot, reader, pinned=pinned)
    if pol == "nextuse":
        E = max(e for _, e in ranges) + 1
        L = len({l for l, _ in ranges})
        layers = sorted({l for l, _ in ranges})
        lidx = {l: i for i, l in enumerate(layers)}  # MoE-layer index in call order
        calib, _, Lc, Ec = load_trace(env["VEYSTRIX_POOL_NEXTUSE_CALIB"], E)
        if Lc != L:
            raise ValueError(f"calibration trace has {Lc} MoE layers/token, index has {L}")
        model = NextUseModel.fit(calib, L)
        return SingleTierExpertPool(n_slots, slot, reader, policy=NextUse(n_slots, L, E, model),
                                    policy_key=lambda k: lidx[k[0]] * E + k[1])
    raise ValueError(f"unknown VEYSTRIX_POOL_POLICY={pol}")
