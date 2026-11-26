"""Application-owned pool of expert records (research, off by default).

Every cached expert row the production path reads is copied out of the file
cache, about 880 MB/token and about 37 ms/token in the 2026-09-23 read replay.
This pool keeps whole expert records in one anonymous memory region laid out
like the slab pack (a 4 KiB header, then fixed-stride records), wrapped once
as a Metal buffer. A decode layer addresses every routed expert in place
through the packed-slab kernels (bit 31 plus the slot), so a hit costs no read
and no copy. A miss reads its nine rows into a free or least recently used
slot through ``F_NOCACHE`` descriptors, so the file cache does not hold a
second copy.

A fixed pool cannot follow free memory the way the file cache does. In the
replay a 5 GB pool tied production and a 6 GB pool read 15 to 24 ms/token
faster; 3 and 4 GB lost. ``FLASHNEXT_EXPERT_POOL_GB`` sets the size (0, the
default, disables it). ``FLASHNEXT_EXPERT_POOL_LOCK=1`` mlocks the region so
macOS cannot compress it.

A slot is only overwritten after its last reader finished: layer L fills its
misses after layer L's router sync, which depends on layer L-1's MoE output,
and every slot this layer routes to is moved to the most recent end before a
victim is chosen.
"""
from __future__ import annotations

import fcntl
import logging
import mmap
import os
from pathlib import Path
import threading
from collections import OrderedDict

import numpy as np

from models.shared.cost_aware import CostAwarePolicy, CostWeights, RouteFrequency
from models.shared.least_stale import LeastStale

from .slab_pack import get_slab_layout, libc_mlock

SIZE_FLAG = "FLASHNEXT_EXPERT_POOL_GB"
# Opt-in: decode-phase pool fills read through the file cache (no F_NOCACHE), so experts the pool evicted can still be
# served from free file-cache memory (a second-level victim cache) at no reserved memory. The bytes and their order are
# unchanged, so results are identical; the prefill ring keeps its own uncached descriptors. Unset/0 = the existing behaviour.
BUFFERED_FILL_FLAG = "FLASHNEXT_EXPERT_POOL_BUFFERED_FILL"
LOCK_FLAG = "FLASHNEXT_EXPERT_POOL_LOCK"
POLICY_FLAG = "FLASHNEXT_EXPERT_POOL_POLICY"
NATIVE_PACK_FLAG = "FLASHNEXT_EXPERT_PACK_NATIVE"
FLAT_SLOTS_FLAG = "FLASHNEXT_EXPERT_POOL_FLAT_SLOTS"
POLICIES = ("lru", "s3fifo", "least-stale", "cost-aware")
# Opt-in decode prefetch: predicted records are read into staging slots carved from the pool's own mapping (total
# memory unchanged) and copied into the decode slot if the router asks for them. See decode_prefetch.py.
PREFETCH_FLAG = "FLASHNEXT_EXPERT_PREFETCH"
PREFETCH_SLOTS_FLAG = "FLASHNEXT_EXPERT_PREFETCH_SLOTS"
PREFETCH_DEPTH_FLAG = "FLASHNEXT_EXPERT_PREFETCH_DEPTH"
PREFETCH_WIDTH_FLAG = "FLASHNEXT_EXPERT_PREFETCH_WIDTH"
PREFETCH_MIN_P_FLAG = "FLASHNEXT_EXPERT_PREFETCH_MIN_P"
PREFETCH_WORKERS_FLAG = "FLASHNEXT_EXPERT_PREFETCH_WORKERS"
PREFETCH_GATE_FLAG = "FLASHNEXT_EXPERT_PREFETCH_GATE"
MIN_DECODE_SLOTS = 16
HEADER = 4096
SLOT_BIT = 0x80000000
_PARTS = (
    ("gate_proj", "weight"), ("gate_proj", "scales"), ("gate_proj", "biases"),
    ("up_proj", "weight"), ("up_proj", "scales"), ("up_proj", "biases"),
    ("down_proj", "weight"), ("down_proj", "scales"), ("down_proj", "biases"),
)


# Set only by the opt-in memory manager (memory_manager.py): the size the NEXT pool will be created with. A pool that
# already exists keeps its size; resizing means releasing it first. None means "use FLASHNEXT_EXPERT_POOL_GB".
_BUDGET_GB = [None]


def set_budget_gb(value) -> None:
    _BUDGET_GB[0] = None if value is None else max(0.0, float(value))


def buffered_fill_enabled() -> bool:
    return os.environ.get(BUFFERED_FILL_FLAG, "0") == "1"


def configured_gb() -> float:
    if _BUDGET_GB[0] is not None:
        return _BUDGET_GB[0]
    try:
        return max(0.0, float(os.environ.get(SIZE_FLAG, "0") or 0))
    except ValueError as error:
        raise ValueError(f"{SIZE_FLAG} must be a number of GB") from error


def enabled() -> bool:
    return configured_gb() > 0


def _flag(name, default, kind):
    raw = os.environ.get(name, "").strip() or default
    try:
        if kind == "bool":
            if raw not in ("0", "1"):
                raise ValueError
            return raw == "1"
        if kind == "int":
            value = int(raw)
            if value < 1:
                raise ValueError
            return value
        value = float(raw)
        if not 0.0 <= value <= 1.0:
            raise ValueError
        return value
    except ValueError:
        want = {"bool": "0 or 1", "int": "a positive integer", "prob": "a number from 0 to 1"}[kind]
        raise ValueError(f"{name} must be {want}, got {raw!r}") from None


def prefetch_settings():
    """The decode-prefetch flags (validated), or None when ``FLASHNEXT_EXPERT_PREFETCH`` is off."""
    enabled_ = _flag(PREFETCH_FLAG, "0", "bool")
    settings = {
        "slots": _flag(PREFETCH_SLOTS_FLAG, "16", "int"), "depth": _flag(PREFETCH_DEPTH_FLAG, "2", "int"),
        "width": _flag(PREFETCH_WIDTH_FLAG, "2", "int"), "min_probability": _flag(PREFETCH_MIN_P_FLAG, "0.5", "prob"),
        "workers": _flag(PREFETCH_WORKERS_FLAG, "2", "int"), "gate": _flag(PREFETCH_GATE_FLAG, "1", "bool"),
    }
    return settings if enabled_ else None


class ExpertPool:
    def __init__(self, store, size_bytes: int, group_size: int = 32,
                 lock: bool = False):
        policy = os.environ.get(POLICY_FLAG, "lru").strip().lower() or "lru"
        if policy not in POLICIES:
            raise ValueError(f"{POLICY_FLAG} must be 'lru', 's3fifo', 'least-stale', or 'cost-aware'")
        weights = CostWeights.from_env() if policy == "cost-aware" else None
        prefetch = prefetch_settings()
        self.policy = policy
        layout = get_slab_layout(group_size)
        self.store = store
        self.stride = layout.record_stride
        self.offsets = {key: layout.offset(*key) for key in _PARTS}
        self._row_sizes = {
            key: layout.shape(*key)[0] * layout.shape(*key)[1] * (4 if key[1] == "weight" else 2)
            for key in _PARTS
        }
        total = (int(size_bytes) - HEADER) // self.stride
        if total < MIN_DECODE_SLOTS:
            raise ValueError("expert pool is too small for one decode layer")
        # Staging slots are the physical slots [self.slots, total): never in _free, _slot_of, the flat table or the
        # policy, so everything else sees self.slots as the decode capacity and the mapping length is unchanged.
        staging, self.prefetch_disabled = 0, None
        if prefetch is not None:
            staging = max(0, min(prefetch["slots"], total // 8, total - MIN_DECODE_SLOTS))
            if staging == 0:
                self.prefetch_disabled = (f"no staging slots: {total} slots leave none after the one-eighth cap "
                                          f"and the {MIN_DECODE_SLOTS}-slot decode floor")
        self.slots = total - staging
        self.staging_slots = staging
        length = -(-(HEADER + total * self.stride) // mmap.PAGESIZE) * mmap.PAGESIZE
        self._mm = mmap.mmap(-1, length)
        self.array = np.frombuffer(self._mm, dtype=np.uint8)
        self.locked = bool(lock) and libc_mlock(
            self.array.__array_interface__["data"][0], length
        )
        import mlx.core as mx

        # MLX shape dimensions are 32-bit, so a byte view of more than 2 GiB
        # cannot be wrapped. The kernels address the pool through a char
        # pointer, so a uint32 view of the same memory serves (up to 8 GiB).
        self.buffer_mx = mx.from_dlpack(
            np.frombuffer(self._mm, dtype=np.uint32), copy=False
        )
        self._view = memoryview(self._mm)
        self._slot_of: "OrderedDict[tuple, int]" = OrderedDict()
        self._s3 = _S3FIFO(self.slots) if policy == "s3fifo" else None
        self._least_stale = LeastStale() if policy == "least-stale" else None
        self._cost = (CostAwarePolicy(self.slots, layers=48, experts=512, weights=weights)
                      if policy == "cost-aware" else None)
        self.wants_refill_cost = self._cost is not None and self._cost.weights.refill != 0.0
        self.evictions = 0
        self._free = list(range(self.slots - 1, -1, -1))
        self._flat_slots = None
        if os.environ.get(FLAT_SLOTS_FLAG) == "1":
            self._flat_slots = np.full((48, 512), -1, dtype=np.int32)
        self._fds: dict = {}
        self._fill_fds: dict = {}
        self._fd_lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.pack = None
        self.packed_reads = 0
        self._packed_stats_lock = threading.Lock()
        self.native_pack = os.environ.get(NATIVE_PACK_FLAG) == "1"
        self.pack_mode = os.environ.get("FLASHNEXT_EXPERT_PACK_READ", "record")
        self.pack_extents = 0
        self.pack_extent_bytes = 0
        path = os.environ.get("FLASHNEXT_EXPERT_PACK")
        if path:
            from models.nemotron35.safetensors_index import index_checkpoint
            from models.shared.expert_pack import ExpertPack

            if self.pack_mode not in ("record", "coalesced", "mmap"):
                raise ValueError("FLASHNEXT_EXPERT_PACK_READ must be record, coalesced or mmap")
            if self.native_pack and self.pack_mode != "record":
                raise ValueError("native expert pack reads require record mode")
            try:
                pack = ExpertPack(Path(path).expanduser(), store.dir, index_checkpoint(store.dir), "flashnext",
                                  uncached=self.pack_mode != "mmap" and not buffered_fill_enabled(),
                                  mapped=self.pack_mode == "mmap")
                expected = {f"{projection}.{part}": self._row_sizes[projection, part]
                            for projection, part in _PARTS}
                if pack.parts != expected or pack.payload != self.stride or not any(layer > 0 for layer in pack.layers):
                    pack.close()
                    raise ValueError("packed experts do not match the Metal pool slot layout")
                self.pack = pack
            except (OSError, ValueError) as error:
                logging.getLogger(__name__).warning("Expert pack unavailable, using checkpoint shards: %s", error)
        self.prefetch = self._frequency = None
        self.prefetch_errors = 0
        if staging:
            from .decode_prefetch import DecodePrefetch

            self._frequency = self._cost.frequency if self._cost is not None else RouteFrequency(48, 512, alpha=0.3)
            self.prefetch = DecodePrefetch(
                first_slot=self.slots, slots=staging, read_record=self._read_record, copy_record=self._copy_record,
                frequency=self._frequency, depth=prefetch["depth"], width=prefetch["width"],
                min_probability=prefetch["min_probability"], workers=prefetch["workers"], gate=prefetch["gate"],
                record_bytes=self.stride,
            )

    @property
    def usable_slots(self) -> int:
        return len(self._free) + len(self._slot_of)

    def assign(self, layer: int, experts):
        """Assign a distinct expert set that fits in the pool, or fail before mutation."""
        if isinstance(experts, int):
            experts = [experts]
        distinct = list(dict.fromkeys(experts))
        if len(distinct) > self.usable_slots:
            raise ValueError("requested experts exceed expert pool capacity")
        if self.prefetch is not None:
            self.prefetch.observe(layer)
        if self._cost is not None:
            return self._assign_cost(layer, distinct)
        slot_of = self._slot_of
        table = self._flat_slots
        flat_layer = table[layer] if table is not None and 0 <= layer < table.shape[0] else None
        s3, least_stale = self._s3, self._least_stale
        protected = {(layer, expert) for expert in distinct} if s3 or least_stale else None
        if least_stale is not None:
            least_stale.advance(layer)
        slots, misses, evicted = {}, [], 0
        for expert in distinct:
            key = (layer, expert)
            if flat_layer is not None and 0 <= expert < len(flat_layer):
                entry = int(flat_layer[expert])
                slot = entry if entry >= 0 else None
            else:
                slot = slot_of.get(key)
            if slot is not None:
                if least_stale is not None:
                    least_stale.touch(key)
                elif s3 is None:
                    slot_of.move_to_end(key)
                else:
                    s3.hit(key)
                slots[expert] = slot
        self.hits += len(slots)
        for expert in distinct:
            if expert in slots:
                continue
            if self._free:
                slot = self._free.pop()
            elif least_stale is not None:
                slot = self.discard_slot(least_stale.victim(protected))
                evicted += 1
            elif s3 is None:
                if table is None:
                    _, slot = slot_of.popitem(last=False)
                else:
                    slot = self.discard_slot(next(iter(slot_of)))
                evicted += 1
            else:
                slot = self.discard_slot(s3.victim(protected, self.usable_slots))
                evicted += 1
            key = (layer, expert)
            self._record_slot(key, slot)
            if least_stale is not None:
                least_stale.touch(key)
            elif s3 is not None:
                s3.insert(key)
            slots[expert] = slot
            misses.append((expert, slot))
        self.misses += len(misses)
        self.evictions += evicted
        if self._frequency is not None:
            self._frequency.observe(layer, distinct)
        return slots, misses

    def _assign_cost(self, layer, distinct):
        """Cost-aware assignment, in the order the offline replay uses: advance, touch hits, evict, insert, observe."""
        cost, slot_of, table = self._cost, self._slot_of, self._flat_slots
        flat_layer = table[layer] if table is not None and 0 <= layer < table.shape[0] else None
        cost.advance(layer)
        slots, missing = {}, []
        for expert in distinct:
            if flat_layer is not None and 0 <= expert < len(flat_layer):
                entry = int(flat_layer[expert])
                slot = entry if entry >= 0 else None
            else:
                slot = slot_of.get((layer, expert))
            if slot is None:
                missing.append(expert)
            else:
                cost.touch((layer, expert), slot)
                slots[expert] = slot
        self.hits += len(slots)
        free = self._free
        need = len(missing) - len(free)
        if need > 0:
            for slot in cost.victims(need, protected=slots.values()):
                free.append(self.discard_slot(cost.key_of(slot)))
            self.evictions += need
        misses = []
        for expert in missing:
            slot, key = free.pop(), (layer, expert)
            self._record_slot(key, slot)
            cost.insert(key, slot)
            slots[expert] = slot
            misses.append((expert, slot))
        self.misses += len(misses)
        cost.observe(layer, distinct, misses=len(misses))
        return slots, misses

    def _fd(self, shard: str) -> int:
        fd = self._fds.get(shard)
        if fd is None:
            with self._fd_lock:
                fd = self._fds.get(shard)
                if fd is None:
                    fd = os.open(os.path.join(self.store.dir, shard), os.O_RDONLY)
                    try:
                        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                        try:
                            fcntl.fcntl(fd, getattr(fcntl, "F_RDAHEAD", 45), 0)
                        except OSError:
                            pass
                    except BaseException:
                        os.close(fd)
                        raise
                    self._fds[shard] = fd
        return fd

    def _fill_fd(self, shard: str) -> int:
        """Descriptor for decode fills: the shared uncached one, or (opt-in) a separate buffered one."""
        if not buffered_fill_enabled():
            return self._fd(shard)
        fd = self._fill_fds.get(shard)
        if fd is None:
            with self._fd_lock:
                fd = self._fill_fds.get(shard)
                if fd is None:
                    fd = os.open(os.path.join(self.store.dir, shard), os.O_RDONLY)
                    try:
                        try:
                            fcntl.fcntl(fd, getattr(fcntl, "F_RDAHEAD", 45), 0)  # read-ahead stays off
                        except OSError:
                            pass
                    except BaseException:
                        os.close(fd)
                        raise
                    self._fill_fds[shard] = fd
        return fd

    def _fill_part(self, name, targets) -> None:
        from .store import _profiled_preadv

        ref = self.store.refs[name]
        fd = self._fill_fd(ref.shard)
        view = self._view
        size = ref.row_bytes
        start = 0
        while start < len(targets):
            expert = targets[start][0]
            end = start + 1
            # Sixteen vectors fit the POSIX minimum IOV_MAX.
            while end < min(len(targets), start + 16) and targets[end][0] == expert + end - start:
                end += 1
            buffers = [view[base:base + size] for _, base in targets[start:end]]
            try:
                if _profiled_preadv(fd, buffers, ref.start + expert * size) != size * len(buffers):
                    raise OSError(f"short expert pool read for {name} row {expert}")
            finally:
                for buffer in buffers:
                    buffer.release()
            start = end

    def invalidate(self, layer: int, misses) -> None:
        """Discard failed fills after all their outstanding writes have drained.

        An evicted entry cannot be restored: its slot may already be partly
        overwritten. Return these slots to the free list, never as cache hits.
        """
        for expert, slot in misses:
            key = (layer, expert)
            if self._slot_of.get(key) == slot:
                self._free.append(self.discard_slot(key))

    def discard_slot(self, key):
        """Remove a mapping without making its slot available to decode."""
        if self._s3 is not None:
            self._s3.small.pop(key, None)
            self._s3.main.pop(key, None)
        if self._least_stale is not None:
            self._least_stale.discard(key)
        if self._cost is not None:
            self._cost.discard(key, self._slot_of[key])
        table = self._flat_slots
        if table is not None and 0 <= key[0] < table.shape[0] and 0 <= key[1] < table.shape[1]:
            table[key] = -1
        return self._slot_of.pop(key)

    def _record_slot(self, key, slot):
        self._slot_of[key] = slot
        table = self._flat_slots
        if table is not None and 0 <= key[0] < table.shape[0] and 0 <= key[1] < table.shape[1]:
            table[key] = slot

    def seed_from_ring(self, layer, expert, rows) -> int:
        """Copy one complete, already-loaded expert into a free decode slot."""
        key = (layer, expert)
        if not self._free or key in self._slot_of or rows.keys() != self.offsets.keys():
            return 0
        if any(len(row) != self._row_sizes[part] for part, row in rows.items()):
            return 0
        slot = self._free[-1]
        base = HEADER + slot * self.stride
        for part, row in rows.items():
            offset = base + self.offsets[part]
            self._view[offset:offset + len(row)] = row
        self._free.pop()
        self._record_slot(key, slot)
        if self._s3 is not None:
            self._s3.insert(key)
        if self._least_stale is not None:
            self._least_stale.touch(key, seeded=True)
        if self._cost is not None:
            self._cost.insert(key, slot, seeded=True)
        return sum(map(len, rows.values()))

    def fill(self, prefix: str, misses, submit, chunk: int = 2):
        """Queue the nine row reads of every miss; returns the futures."""
        if chunk < 1:
            raise ValueError("expert pool fill chunk must be positive")
        misses = sorted(misses, key=lambda target: target[0])
        if self.prefetch is not None:
            return self._fill_prefetch(prefix, misses, submit, chunk)
        return self._fill_direct(prefix, misses, submit, chunk)

    def _fill_prefetch(self, prefix, misses, submit, chunk):
        """``fill`` with staged records promoted by copy and the demand gate shut while real reads are outstanding."""
        prefetch, layer = self.prefetch, int(prefix.split(".")[3])
        reads, claimed, gated = [], [], False
        try:
            for expert, slot in misses:
                entry = prefetch.claim((layer, expert))
                if entry is None:
                    reads.append((expert, slot))
                else:
                    claimed.append((entry, expert, slot))
            gated = prefetch.gate and (bool(reads) or any(not entry.ready for entry, _, _ in claimed))
            if gated:
                prefetch.demand_begin()
        except BaseException:
            for entry, _, _ in claimed:
                prefetch.release(entry)
            raise
        futures = []
        try:
            if reads:
                futures = self._fill_direct(prefix, reads, submit, chunk)   # drains its own workers on failure
            while claimed:
                entry, expert, slot = claimed[0]
                futures.append(submit(self._promote, entry, layer, expert, slot))
                claimed.pop(0)
        except BaseException:
            for future in futures:
                try:
                    future.result()
                except BaseException:
                    pass
            for entry, _, _ in claimed:
                prefetch.release(entry)
            if gated:
                prefetch.demand_end()
            raise
        if gated:
            remaining, lock = [len(futures)], threading.Lock()

            def done(_future):
                with lock:
                    remaining[0] -= 1
                    last = remaining[0] == 0
                if last:
                    prefetch.demand_end()

            if not futures:
                prefetch.demand_end()
            for future in futures:
                future.add_done_callback(done)
        return futures

    def _promote(self, entry, layer, expert, slot):
        """Copy a staged record into its decode slot, or read it directly when the prefetch did not deliver."""
        try:
            if not self.prefetch.finish(entry, slot):
                self._read_record(layer, expert, slot)
        finally:
            self.prefetch.release(entry)

    def _read_record(self, layer, expert, slot):
        """Read one whole record into a physical slot (decode or staging)."""
        if self.pack is not None and layer in self.pack.layers:
            self._fill_packed(layer, expert, slot)
        else:
            self._fill_checkpoint_records(layer, [(expert, slot)])

    def _copy_record(self, source, destination):
        size, view = self.stride, self._view
        a, b = HEADER + source * size, HEADER + destination * size
        with view[a:a + size] as src, view[b:b + size] as dst:
            dst[:] = src

    def prefetch_ahead(self, layer) -> None:
        """Queue predictions for the layers after ``layer`` (no-op unless decode prefetch is on)."""
        if self.prefetch is not None:
            try:
                self.prefetch.issue(layer, self._slot_of.__contains__)
            except Exception:             # an optimisation must never fail a decode step
                self.prefetch_errors += 1
                if self.prefetch_errors == 1:
                    logging.getLogger(__name__).exception("decode prefetch issue failed")

    def observe_refill(self, layer, seconds_per_record) -> None:
        if self.wants_refill_cost:
            self._cost.observe_refill(layer, seconds_per_record)

    def stats(self) -> dict:
        total = self.hits + self.misses
        out = {"policy": self.policy, "slots": self.slots, "resident": len(self._slot_of), "hits": self.hits,
               "misses": self.misses, "evictions": self.evictions, "hit_rate": self.hits / total if total else None,
               "staging_slots": self.staging_slots, "prefetch_disabled": self.prefetch_disabled,
               "prefetch_errors": self.prefetch_errors}
        if self._cost is not None:
            out["cost"] = self._cost.stats()
        out["prefetch"] = self.prefetch.stats() if self.prefetch is not None else None
        return out

    def _fill_direct(self, prefix, misses, submit, chunk):
        from .store import expert_read_batch_rows

        futures = []
        try:
            if self.pack is not None:
                layer = int(prefix.split(".")[3])
                if layer in self.pack.layers:
                    if self.native_pack and misses:
                        futures.append(submit(self._fill_native, layer, misses))
                        return futures
                    if self.pack_mode == "record":
                        for expert, slot in misses:
                            futures.append(submit(self._fill_packed, layer, expert, slot))
                    else:
                        for extent in packed_extents(misses):
                            futures.append(submit(self._fill_extent, layer, extent))
                    return futures
            for projection, part in _PARTS:
                name = f"{prefix}.{projection}.{part}"
                offset = HEADER + self.offsets[(projection, part)]
                ref = self.store.refs.get(name)
                batch = expert_read_batch_rows(ref.row_bytes, chunk) if ref is not None else chunk
                for start in range(0, len(misses), batch):
                    targets = [
                        (expert, offset + slot * self.stride)
                        for expert, slot in misses[start:start + batch]
                    ]
                    futures.append(submit(self._fill_part, name, targets))
        except BaseException:
            # The caller cannot see a partial submission's futures. Wait for
            # every worker before allowing it to invalidate and reuse slots.
            for future in futures:
                try:
                    future.result()
                except BaseException:
                    pass
            raise
        return futures

    def _fill_packed(self, layer, expert, slot):
        from .store import _profiled_preadv

        offset = HEADER + slot * self.stride
        with self._view[offset:offset + self.stride] as destination:
            try:
                self.pack.read_into(layer, expert, [destination], read=_profiled_preadv)
            except OSError:
                self._fill_checkpoint_records(layer, [(expert, slot)])
                return
        with self._packed_stats_lock:
            self.packed_reads += 1
            self.pack_extents += 1
            self.pack_extent_bytes += self.stride

    def _fill_extent(self, layer, extent):
        from contextlib import ExitStack
        from .store import _profiled_preadv

        padding = self.pack.stride - self.stride
        with ExitStack() as stack:
            buffers = []
            for _, slot in extent:
                offset = HEADER + slot * self.stride
                buffers.append(stack.enter_context(self._view[offset:offset + self.stride]))
                if padding:
                    buffers.append(memoryview(bytearray(padding)))
            try:
                size = self.pack.read_extent(layer, extent[0][0], buffers, read=_profiled_preadv)
            except OSError:
                self._fill_checkpoint_records(layer, extent)
                return
        with self._packed_stats_lock:
            self.packed_reads += len(extent)
            self.pack_extents += 1
            self.pack_extent_bytes += size

    def _fill_checkpoint_records(self, layer, records):
        for projection, part in _PARTS:
            name = f"language_model.model.layers.{layer}.mlp.switch_mlp.{projection}.{part}"
            offset = HEADER + self.offsets[(projection, part)]
            self._fill_part(name, [(expert, offset + slot * self.stride) for expert, slot in records])

    def _fill_native(self, layer, misses):
        from . import native_read

        base = native_read.address(self.array) + HEADER
        batch = native_read.Batch()
        batch.add_records(self.pack.fd, [self.pack.offset(layer, expert) for expert, _ in misses],
                          self.stride, [base + slot * self.stride for _, slot in misses])
        try:
            batch.run()
        except (OSError, RuntimeError):
            self._fill_checkpoint_records(layer, misses)
            return
        with self._packed_stats_lock:
            self.packed_reads += len(misses)
            self.pack_extents += len(misses)
            self.pack_extent_bytes += len(misses) * self.stride

    def close(self) -> None:
        if self.prefetch is not None:
            self.prefetch.close()         # no prefetch read may write the mapping after this returns
        if self.pack is not None:
            self.pack.close()
        for fd in (*self._fds.values(), *self._fill_fds.values()):
            os.close(fd)
        self._fds.clear()
        self._fill_fds.clear()


def packed_extents(misses, limit=8):
    """Group adjacent file offsets, retaining caller-owned destination slots."""
    start = 0
    while start < len(misses):
        end = start + 1
        while (end < min(len(misses), start + limit) and
               misses[end][0] == misses[end - 1][0] + 1):
            end += 1
        yield misses[start:end]
        start = end


class _S3FIFO:
    def __init__(self, capacity):
        self.capacity = capacity
        self.small = OrderedDict()
        self.main = OrderedDict()
        self.ghost = OrderedDict()

    def hit(self, key):
        lane = self.small if key in self.small else self.main
        lane[key] = min(3, lane[key] + 1)

    def insert(self, key):
        if key in self.ghost:
            del self.ghost[key]
            self.main[key] = 0
        else:
            self.small[key] = 0

    def victim(self, protected, capacity):
        # The batch fits, so at least one resident is outside the requested set.
        # A main-queue entry has at most three second chances before eviction.
        while True:
            prefer_small = len(self.small) >= max(1, capacity // 10)
            lanes = (self.small, self.main) if prefer_small else (self.main, self.small)
            for lane in lanes:
                key = next((candidate for candidate in lane if candidate not in protected), None)
                if key is not None:
                    break
            else:
                raise RuntimeError("expert pool has no unprotected S3-FIFO victim")
            count = lane.pop(key)
            if lane is self.small:
                if count:
                    self.main[key] = 0
                    continue
                self.ghost[key] = None
                if len(self.ghost) > self.capacity:
                    self.ghost.popitem(last=False)
            elif count:
                self.main[key] = count - 1
                continue
            return key


AUTO_FLAG = "FLASHNEXT_EXPERT_POOL_AUTO"
GIB = 2 ** 30
# The guard stops when compressor growth passes 2 GiB while available memory is under 3 GiB, so a pool is only
# allowed if memory still stays above that line AFTER the pool is resident, plus a margin. The pool is filled by
# the first decode tokens (first real chat 2026-11-08: available fell 4.35 to 3.21 GiB within 0.3 s of creation and
# to 2.94 GiB by the first token, with the compressor still 3.4 GiB above its start), so the reserve is its full size.
POOL_FLOOR_BYTES = 3 * GIB
POOL_MARGIN_BYTES = GIB // 2
_HEADROOM_READER = [None]      # tests inject a reader returning (available_bytes, pressure_level)


def auto_enabled() -> bool:
    return os.environ.get(AUTO_FLAG, "0") == "1"


def headroom_decision(available_bytes: int, pool_bytes: int, pressure: int = 1,
                      floor: int = POOL_FLOOR_BYTES, margin: int = POOL_MARGIN_BYTES):
    """Return ``(allowed, reason)``. The pool is optional, so any doubt means no pool."""
    from . import memory_controller

    if memory_controller.forbid_optional():
        return False, f"memory controller is {memory_controller.state().upper()} (no optional allocation)"
    need = int(pool_bytes) + floor + margin
    if pressure >= 2:
        return False, f"memory pressure level {pressure}"
    if available_bytes < need:
        return False, (f"{available_bytes / GIB:.2f} GiB available, a {pool_bytes / GIB:.2f} GiB pool needs "
                       f"{need / GIB:.2f} GiB ({floor / GIB:.0f} GiB guard floor + {margin / GIB:.1f} GiB margin)")
    return True, f"{available_bytes / GIB:.2f} GiB available covers the {need / GIB:.2f} GiB the pool needs"


def _read_headroom():
    if _HEADROOM_READER[0] is not None:
        return _HEADROOM_READER[0]()
    from . import purge_recovery

    reading = purge_recovery.read_system()
    return reading.available, reading.pressure


def _trim_allocator_cache() -> None:
    """Release unused MLX allocator blocks left by the prefill before measuring headroom (idle-safe, no sync)."""
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def reset_decline(store) -> None:
    """Forget an earlier refusal so the next request looks at memory again (deferred, per-request retry)."""
    if store is not None and getattr(store, "_flashnext_pool_declined", None) is not None:
        store._flashnext_pool_declined = None


def _auto_allows(store, size_bytes: int) -> bool:
    _trim_allocator_cache()
    available, pressure = _read_headroom()
    allowed, reason = headroom_decision(available, size_bytes, pressure)
    store._flashnext_pool_decision = {"allowed": allowed, "reason": reason, "available": int(available)}
    if not allowed:
        store._flashnext_pool_declined = reason
        if os.environ.get("FLASHNEXT_CHAT_GUARD") == "1":
            print(f"FlashNext expert pool: not allocated ({reason}); decoding from SSD without a pool "
                  f"(slower, memory-safe)", flush=True)
    return allowed


def for_store(store, group_size: int = 32):
    """The process's pool for this store, created on first use (``None``: run without a pool)."""
    pool = getattr(store, "_flashnext_expert_pool", None)
    planned = getattr(store, "_flashnext_elastic_pool_bytes", None)
    target_bytes = round(configured_gb() * 1e9) if planned is None else planned
    if pool is None and target_bytes > 0:
        if getattr(store, "_flashnext_pool_declined", None) is not None:
            return None                       # refused for this request; reset_decline() lets the next one retry
        target_gb = target_bytes / 1e9
        if planned is None and auto_enabled() and not _auto_allows(store, target_bytes):
            return None
        lock = os.environ.get(LOCK_FLAG, "0") == "1"
        try:
            pool = ExpertPool(
                store, target_bytes, group_size,
                lock=lock,
            )
        except (MemoryError, RuntimeError, OSError):
            # Guarded interactive chat must report a failed allocation rather
            # than silently running with a different pool or masking a Metal error.
            if os.environ.get("FLASHNEXT_CHAT_GUARD") == "1":
                raise
            # Recovery from allocation failure: fall back to 1 GB if larger
            if planned is None and target_gb > 1.0:
                try:
                    pool = ExpertPool(
                        store, int(1.0 * 1e9), group_size,
                        lock=lock,
                    )
                    set_budget_gb(1.0)
                except (MemoryError, RuntimeError, OSError):
                    pool = None
            else:
                pool = None
        if pool is None:
            store._flashnext_pool_declined = "allocation_failed"
        store._flashnext_expert_pool = pool
        if pool is not None and os.environ.get("FLASHNEXT_CHAT_GUARD") == "1":
            print(f"FlashNext expert pool: {len(pool._mm)} bytes "
                  f"({pool.slots} slots; requested {target_gb:g} GB)", flush=True)
    return pool
