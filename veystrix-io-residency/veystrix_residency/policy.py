"""Expert cache policies over a stream of per-layer routing batches.

All policies use slot capacity (uniform expert size — true per model for both
Qwen3.8-Flash-Next and Nemotron). A "step" is one MoE layer call: the k
distinct expert keys routed for that layer. Every key in a step must be
resident at the same time (the matmul uses all of them), so a policy never
evicts a key of the current step.

Each policy's step(keys) returns a list of served-from codes, one per key:
    POOL = 0   served from the in-process pool (no syscall)
    PAGE = 1   pool miss, served from the OS page cache (syscall, no SSD)
    SSD  = 2   pool miss and page-cache miss: physical read

The same classes back the reference pool (pool.py), so the simulator predicts
exactly the hit/miss sequence the implementation produces.
"""
from __future__ import annotations

import heapq
from collections import OrderedDict, deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

POOL, PAGE, SSD = 0, 1, 2


def _check_batch(keys: Sequence[int], cap: int) -> Set[int]:
    batch = set(keys)
    if len(batch) != len(keys):
        raise ValueError("duplicate expert key inside one routing step")
    if cap < len(batch):
        raise ValueError(f"capacity {cap} slots < routed experts per step {len(batch)}")
    return batch


class LRU:
    """Plain LRU. Also used as the page-cache model (codes PAGE instead of POOL)."""

    def __init__(self, cap: int, hit_code: int = POOL):
        self.cap = int(cap)
        self.hit_code = hit_code
        self.od: "OrderedDict[int, None]" = OrderedDict()

    @property
    def resident(self) -> Set[int]:
        return set(self.od)

    def contains(self, k: int) -> bool:
        return k in self.od

    def touch_or_insert(self, k: int, protect: Set[int]) -> bool:
        """Single-key access used by DoubleResidency's page-cache tier."""
        if k in self.od:
            self.od.move_to_end(k)
            return True
        if self.cap <= 0:
            return False
        while len(self.od) >= self.cap:
            self._evict_one(protect)
        self.od[k] = None
        return False

    def _evict_one(self, protect: Set[int]) -> int:
        for victim in self.od:
            if victim not in protect:
                del self.od[victim]
                return victim
        raise RuntimeError("LRU: every resident key is protected")

    def step(self, keys: Sequence[int]) -> List[int]:
        batch = _check_batch(keys, self.cap)
        hits = [k in self.od for k in keys]
        for k, h in zip(keys, hits):
            if h:
                self.od.move_to_end(k)
        for k, h in zip(keys, hits):
            if not h:
                while len(self.od) >= self.cap:
                    self._evict_one(batch)
                self.od[k] = None
        return [self.hit_code if h else SSD for h in hits]


class S3FIFO:
    """S3-FIFO (Yang et al., SOSP 2023) with an optional pinned set.

    Pinned keys get reserved slots, load lazily on first touch, and are never
    evicted. The remaining slots run S3-FIFO: small FIFO S (small_frac of the
    dynamic slots), main FIFO M, ghost FIFO G of keys only (|G| <= |M| slots).
    `promote_gt` is the freq threshold for S->M (paper pseudo-code: freq > 1;
    some implementations use > 0 — calibrate against your engine's s3fifo-pool).
    """

    def __init__(self, cap: int, small_frac: float = 0.1, pinned: Iterable[int] = (),
                 max_freq: int = 3, promote_gt: int = 1):
        self.cap = int(cap)
        self.pinned: Set[int] = set(pinned)
        self.dyn_cap = self.cap - len(self.pinned)
        if self.dyn_cap < 1:
            raise ValueError("pinned set leaves no dynamic slots")
        self.s_cap = max(1, int(round(self.dyn_cap * small_frac)))
        self.g_cap = max(1, self.dyn_cap - self.s_cap)
        self.max_freq = max_freq
        self.promote_gt = promote_gt
        self.S: deque = deque()  # head = left, tail = right
        self.M: deque = deque()
        self.G: "OrderedDict[int, None]" = OrderedDict()
        self.freq: Dict[int, int] = {}
        self.where: Dict[int, str] = {}
        self.pinned_loaded: Set[int] = set()
        self.evicted: List[int] = []  # keys evicted during the last step (pool uses this to free slots)

    @property
    def resident(self) -> Set[int]:
        return set(self.where) | self.pinned_loaded

    def _n_dyn(self) -> int:
        return len(self.S) + len(self.M)

    def _ghost_add(self, k: int) -> None:
        self.G[k] = None
        self.G.move_to_end(k)
        while len(self.G) > self.g_cap:
            self.G.popitem(last=False)

    def _evict_s(self, protect: Set[int]) -> Optional[int]:
        while self.S:
            t = self.S.pop()
            if t in protect or self.freq[t] > self.promote_gt:
                # stays resident: promote to M head (freq carried over, as in the paper)
                self.M.appendleft(t)
                self.where[t] = "M"
                continue
            del self.where[t]
            del self.freq[t]
            self._ghost_add(t)
            self.evicted.append(t)
            return t
        return None

    def _evict_m(self, protect: Set[int]) -> int:
        spins = 0
        while self.M:
            t = self.M.pop()
            if t in protect:
                self.M.appendleft(t)
                spins += 1
                if spins > 4 * (len(self.M) + 1) * (self.max_freq + 1):
                    raise RuntimeError("S3FIFO: every resident key is protected")
                continue
            if self.freq[t] > 0:
                self.freq[t] -= 1
                self.M.appendleft(t)
                continue
            del self.where[t]
            del self.freq[t]
            self.evicted.append(t)
            return t
        raise RuntimeError("S3FIFO: M empty during eviction")

    def _evict_one(self, protect: Set[int]) -> int:
        if len(self.S) >= self.s_cap:
            v = self._evict_s(protect)
            if v is not None:
                return v
        return self._evict_m(protect)

    def step(self, keys: Sequence[int]) -> List[int]:
        batch = _check_batch(keys, self.cap)
        self.evicted = []
        if self.dyn_cap < len(batch - self.pinned) + 1:
            raise ValueError("dynamic slots must exceed routed experts per step")
        codes = []
        misses = []
        for k in keys:
            if k in self.pinned:
                if k in self.pinned_loaded:
                    codes.append(POOL)
                else:
                    self.pinned_loaded.add(k)
                    codes.append(SSD)
                continue
            if k in self.where:
                self.freq[k] = min(self.freq[k] + 1, self.max_freq)
                codes.append(POOL)
            else:
                codes.append(SSD)
                misses.append(k)
        for k in misses:
            while self._n_dyn() >= self.dyn_cap:
                self._evict_one(batch)
            if k in self.G:
                del self.G[k]
                self.M.appendleft(k)
                self.where[k] = "M"
            else:
                self.S.appendleft(k)
                self.where[k] = "S"
            self.freq[k] = 0
        return codes


class LFU:
    """In-cache-history LFU with LRU tie-break (counts persist after eviction)."""

    def __init__(self, cap: int):
        self.cap = int(cap)
        self.count: Dict[int, int] = {}
        self.last: Dict[int, int] = {}
        self.res: Set[int] = set()
        self.heap: list = []
        self.clock = 0

    @property
    def resident(self) -> Set[int]:
        return set(self.res)

    def _evict_one(self, protect: Set[int]) -> None:
        held = []
        while self.heap:
            c, t, k = heapq.heappop(self.heap)
            if k not in self.res or self.count[k] != c or self.last[k] != t:
                continue
            if k in protect:
                held.append((c, t, k))
                continue
            self.res.discard(k)
            for e in held:
                heapq.heappush(self.heap, e)
            return
        raise RuntimeError("LFU: every resident key is protected")

    def step(self, keys: Sequence[int]) -> List[int]:
        batch = _check_batch(keys, self.cap)
        self.clock += 1
        codes = []
        for k in keys:
            hit = k in self.res
            codes.append(POOL if hit else SSD)
            self.count[k] = self.count.get(k, 0) + 1
            self.last[k] = self.clock
        for k, c in zip(keys, codes):
            if c == SSD:
                while len(self.res) >= self.cap:
                    self._evict_one(batch)
                self.res.add(k)
            heapq.heappush(self.heap, (self.count[k], self.last[k], k))
        return codes


INF = float("inf")


class OPT:
    """Belady's MIN: evict the resident key whose next use is farthest away.

    Optimal for uniform object sizes, so it is the hit-rate CEILING for any
    online policy at a given slot capacity. Needs the whole trace up front.
    """

    def __init__(self, cap: int, steps: Sequence[Sequence[int]]):
        self.cap = int(cap)
        self.steps = [tuple(s) for s in steps]
        nxt: List[List[float]] = [[INF] * len(s) for s in self.steps]
        seen: Dict[int, int] = {}
        for i in range(len(self.steps) - 1, -1, -1):
            for j, k in enumerate(self.steps[i]):
                nxt[i][j] = seen.get(k, INF)
            for k in self.steps[i]:
                seen[k] = i
        self.next_occ = nxt
        self.i = 0
        self.res: Set[int] = set()
        self.nu: Dict[int, float] = {}
        self.heap: list = []

    @property
    def resident(self) -> Set[int]:
        return set(self.res)

    def step(self, keys: Sequence[int]) -> List[int]:
        if tuple(keys) != self.steps[self.i]:
            raise ValueError("OPT must be driven with the trace it was built from")
        batch = _check_batch(keys, self.cap)
        codes = [POOL if k in self.res else SSD for k in keys]
        for j, (k, c) in enumerate(zip(keys, codes)):
            self.nu[k] = self.next_occ[self.i][j]
            if c == POOL:
                heapq.heappush(self.heap, (-self.nu[k], k))
        for k, c in zip(keys, codes):
            if c == SSD:
                while len(self.res) >= self.cap:
                    held = []
                    while True:
                        negnu, v = heapq.heappop(self.heap)
                        if v not in self.res or self.nu[v] != -negnu:
                            continue
                        if v in batch:
                            held.append((negnu, v))
                            continue
                        self.res.discard(v)
                        break
                    for e in held:
                        heapq.heappush(self.heap, e)
                self.res.add(k)
                heapq.heappush(self.heap, (-self.nu[k], k))
        self.i += 1
        return codes


class DoubleResidency:
    """Today's likely regime: in-process pool + OS page cache over pool misses.

    Pool misses go through pread, which also populates the page cache (modelled
    as LRU over the pool-miss stream). Those pages are mostly the experts the
    pool just admitted, so the page cache largely DUPLICATES the pool. Compare
    against a single-tier pool given cap = pool_cap + pc_cap.
    """

    def __init__(self, pool, pc_cap: int):
        self.pool = pool
        self.pc = LRU(pc_cap, hit_code=PAGE)

    @property
    def resident(self) -> Set[int]:
        return self.pool.resident | self.pc.resident

    def step(self, keys: Sequence[int]) -> List[int]:
        codes = self.pool.step(keys) if self.pool is not None else [SSD] * len(keys)
        out = []
        for k, c in zip(keys, codes):
            if c == POOL:
                out.append(POOL)
            else:
                out.append(PAGE if self.pc.touch_or_insert(k, set()) else SSD)
        return out


def top_keys_by_frequency(steps: Iterable[Sequence[int]], n: int) -> List[int]:
    cnt: Dict[int, int] = {}
    for s in steps:
        for k in s:
            cnt[k] = cnt.get(k, 0) + 1
    return [k for k, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


# ---------------------------------------------------------------------------
# Calibrated next-use eviction: a table-based Belady approximation.
#
# Belady evicts the key whose next use is farthest away. We cannot see the
# future, but MoE decode has structure: every layer is visited exactly once
# per token, in order. So "time" is measured in tokens, and for each resident
# expert we estimate its next-use token from (a) how long since its last use
# and (b) an exponentially-decayed use count, using gap statistics learned
# from a CALIBRATION trace (different prompts). This is the mechanism FlashMoE
# (arXiv 2601.17063) implements with an ML model; a lookup table is enough to
# test whether the idea pays off on Veystrix traces. Routing is untouched:
# the policy only decides residency, so outputs are bit-identical.
# ---------------------------------------------------------------------------
import numpy as _np


class NextUseModel:
    """E[gap | gap >= age] tables, conditioned on two decayed use counts.

    fast count (decay ~0.9, ~10-token memory) captures short bursts of reuse;
    slow count (decay ~0.99, ~100-token memory) captures standing popularity.
    """

    def __init__(self, table, edges, decays, hmax):
        self.table = table                    # [nb_fast, nb_slow, hmax+1]
        self.edges = [_np.asarray(e, dtype=float) for e in edges]
        self.decays = tuple(decays)
        self.hmax = hmax

    def bins(self, ew_fast, ew_slow):
        return (_np.searchsorted(self.edges[0], ew_fast, side="right"),
                _np.searchsorted(self.edges[1], ew_slow, side="right"))

    @classmethod
    def fit(cls, steps: Sequence[Sequence[int]], steps_per_token: int,
            decays: Sequence[float] = (0.9, 0.99),
            edges: Sequence[Sequence[float]] = ((1.3, 1.7, 2.2, 3.0, 4.0, 5.5),
                                                (1.5, 2.5, 4.0, 6.0, 9.0, 14.0, 22.0)),
            hmax: int = 512) -> "NextUseModel":
        if len(decays) == 1:
            decays = (decays[0], 0.99)
        e0, e1 = (_np.asarray(e, dtype=float) for e in edges)
        nb0, nb1 = len(e0) + 1, len(e1) + 1
        gaps: List[List[int]] = [[] for _ in range(nb0 * nb1)]
        last: Dict[int, int] = {}
        ew: Dict[int, Tuple[float, float]] = {}

        def cell(v):
            return int(_np.searchsorted(e0, v[0], side="right")) * nb1 + int(_np.searchsorted(e1, v[1], side="right"))

        for i, s in enumerate(steps):
            t = i // steps_per_token
            for k in s:
                if k in last:
                    g = t - last[k]
                    gaps[cell(ew[k])].append(min(g, hmax))
                    f, sl = ew[k]
                    ew[k] = (f * decays[0] ** g + 1.0, sl * decays[1] ** g + 1.0)
                else:
                    ew[k] = (1.0, 1.0)
                last[k] = t
        for k in last:  # censored: no reuse seen before the calibration trace ended
            gaps[cell(ew[k])].append(hmax)
        # pool sparse cells toward their fast-count marginal so tiny cells do not dominate
        table = _np.full((nb0, nb1, hmax + 1), float(2 * hmax))
        ages = _np.arange(hmax + 1)

        def cond_mean(g):
            g = _np.sort(_np.asarray(g, dtype=float))
            suffix = _np.concatenate([_np.cumsum(g[::-1])[::-1], [0.0]])
            j = _np.searchsorted(g, ages, side="left")
            cnt = g.size - j
            out = _np.where(cnt > 0, suffix[j] / _np.maximum(cnt, 1), 2.0 * hmax)
            return out

        for b0 in range(nb0):
            marg = [x for b1 in range(nb1) for x in gaps[b0 * nb1 + b1]]
            m_row = cond_mean(marg) if marg else _np.full(hmax + 1, 2.0 * hmax)
            for b1 in range(nb1):
                g = gaps[b0 * nb1 + b1]
                table[b0, b1] = cond_mean(g) if len(g) >= 20 else m_row
        return cls(table, (e0, e1), decays, hmax)


class NextUse:
    """Evict the resident expert with the latest predicted next-use time."""

    def __init__(self, cap: int, steps_per_token: int, experts_per_layer: int, model: NextUseModel):
        self.cap = int(cap)
        self.L = steps_per_token
        self.E = experts_per_layer
        self.m = model
        self.slot_key = _np.full(self.cap, -1, dtype=_np.int64)
        self.slot_of: Dict[int, int] = {}
        self.free = list(range(self.cap - 1, -1, -1))
        self.h_last: Dict[int, int] = {}
        self.h_ew: Dict[int, Tuple[float, float]] = {}
        self.s_last = _np.zeros(self.cap, dtype=_np.int64)
        self.s_f = _np.zeros(self.cap)
        self.s_s = _np.zeros(self.cap)
        self.i = 0
        self.evicted: List[int] = []

    @property
    def resident(self) -> Set[int]:
        return set(self.slot_of)

    def _touch(self, k: int, t: int) -> Tuple[float, float]:
        d0, d1 = self.m.decays
        if k in self.h_last:
            g = t - self.h_last[k]
            f, s = self.h_ew[k]
            v = (f * d0 ** g + 1.0, s * d1 ** g + 1.0) if g > 0 else self.h_ew[k]
        else:
            v = (1.0, 1.0)
        self.h_last[k] = t
        self.h_ew[k] = v
        return v

    def step(self, keys: Sequence[int]) -> List[int]:
        batch = _check_batch(keys, self.cap)
        self.evicted = []
        t = self.i // self.L
        cur_layer = keys[0] // self.E
        codes = [POOL if k in self.slot_of else SSD for k in keys]
        need = sum(1 for c in codes if c == SSD) - len(self.free)
        if need > 0:
            occ = self.slot_key >= 0
            layer = self.slot_key // self.E
            nxt = _np.where(layer > cur_layer, t, t + 1)       # earliest token this expert can be used again
            age = _np.clip(nxt - self.s_last, 1, self.m.hmax)
            b0, b1 = self.m.bins(self.s_f, self.s_s)
            score = self.s_last + self.m.table[b0, b1, age] + layer / self.L
            score = _np.where(occ, score, -_np.inf)
            for k in batch:
                s = self.slot_of.get(k)
                if s is not None:
                    score[s] = -_np.inf
            victims = _np.argpartition(-score, need - 1)[:need]
            for s in victims:
                if not _np.isfinite(score[s]):
                    raise RuntimeError("NextUse: not enough evictable slots")
                v = int(self.slot_key[s])
                del self.slot_of[v]
                self.slot_key[s] = -1
                self.free.append(int(s))
                self.evicted.append(v)
        for k, c in zip(keys, codes):
            f, sl = self._touch(k, t)
            if c == SSD:
                s = self.free.pop()
                self.slot_key[s] = k
                self.slot_of[k] = s
            else:
                s = self.slot_of[k]
            self.s_last[s] = t
            self.s_f[s] = f
            self.s_s[s] = sl
        self.i += 1
        return codes


def compulsory_flags(steps: Sequence[Sequence[int]]) -> List[List[bool]]:
    """True where an access is the first ever touch of that key (a miss for ANY cache)."""
    seen: Set[int] = set()
    out = []
    for s in steps:
        row = []
        for k in s:
            row.append(k not in seen)
            seen.add(k)
        out.append(row)
    return out
