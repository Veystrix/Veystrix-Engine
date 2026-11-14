"""Deadline-ordered asynchronous prefetch of expert records into staging slots (opt-in, bit-exact).

Decode visits layers in a fixed order, so the records the router is likely to want a layer or two ahead can be read
while the GPU computes the current layer. A prediction reads one whole record into a staging slot (a slot of the
pool's own mapping that decode never addresses). When the router later demands that record, the pool copies it from
the staging slot into the slot it assigned instead of reading the SSD. Only which bytes are read when changes.

Hazard rules. A staging slot is written only by the one prefetch read that owns it and read only by the copy that
claimed it, after that read finished. A slot returns to the free list only when no read can still write it.
Threads: the decode thread (observe/issue/claim), demand workers (finish/release), prefetch workers (the reads).
No MLX import: the pool supplies ``read_record`` and ``copy_record``.
"""
from __future__ import annotations

import threading

from models.shared.edf_io import DeadlineExecutor

QUEUED, RUNNING, DONE, FAILED, CANCELLED = range(5)


class _Entry:
    __slots__ = ("key", "slot", "deadline", "state", "claimed", "released", "freed")

    def __init__(self, key, slot, deadline):
        self.key, self.slot, self.deadline = key, slot, deadline
        self.state = QUEUED
        self.claimed = self.released = self.freed = False

    @property
    def ready(self) -> bool:
        """True when the record is already staged (the copy needs no SSD read)."""
        return self.state == DONE


class DecodePrefetch:
    def __init__(self, *, first_slot, slots, read_record, copy_record, frequency,
                 depth=2, width=2, min_probability=0.5, workers=2, gate=True, record_bytes=0):
        if slots < 1:
            raise ValueError("decode prefetch needs at least one staging slot")
        self.first_slot, self.slots = int(first_slot), int(slots)
        self.depth, self.width, self.min_probability = int(depth), int(width), float(min_probability)
        self.gate, self.workers, self.record_bytes = bool(gate), int(workers), int(record_bytes)
        self._read_record, self._copy_record, self._frequency = read_record, copy_record, frequency
        self._cond = threading.Condition()
        self._free = list(range(self.first_slot + self.slots - 1, self.first_slot - 1, -1))
        self._entries: dict = {}          # unclaimed entries by (layer, expert)
        self._live: set = set()           # every entry whose slot has not been freed
        self._next: dict = {}             # learned layer order: layer -> following layer
        self._previous = None
        self._step = 0
        self._closed = False
        self._claimed = 0
        self._counts = dict.fromkeys(("issued", "used", "late", "cancelled", "expired", "wasted", "failed"), 0)
        self._gate_lock = threading.Lock()
        self._demand = 0
        self._executor = DeadlineExecutor(self.workers, "decode-prefetch")

    # Decode thread -------------------------------------------------------------------------------------------

    def observe(self, layer) -> None:
        """One pool access: advance the step, learn the layer order, expire passed predictions."""
        with self._cond:
            if self._closed:
                return
            self._step += 1
            step = self._step
            previous, self._previous = self._previous, layer
            if previous is not None and previous != layer:
                self._next[previous] = layer
            for key, entry in list(self._entries.items()):
                if entry.deadline < step and entry.state == QUEUED:
                    entry.state = CANCELLED
                    del self._entries[key]
                    self._counts["expired"] += 1
                    self._free_slot(entry)
        self._executor.cancel_stale(step)

    def issue(self, layer, is_resident) -> int:
        """Queue predictions for the layers after ``layer``. Never blocks; returns how many were queued."""
        with self._cond:
            if self._closed:
                return 0
            step, chain, target = self._step, [], layer
            for distance in range(1, self.depth + 1):
                target = self._next.get(target)
                if target is None:
                    break
                chain.append((distance, target))
        issued = 0
        for distance, target in chain:
            candidates = [(expert, p) for expert, p in self._frequency.predict(target, 64)
                          if p >= self.min_probability and not is_resident((target, expert))]
            queued = 0
            with self._cond:
                if self._closed:
                    return issued
                for expert, _ in candidates:
                    if queued == self.width:
                        break
                    key = (target, expert)
                    if key in self._entries:
                        continue
                    slot = self._take_slot(step)
                    if slot is None:
                        return issued
                    entry = _Entry(key, slot, step + distance)
                    self._entries[key] = entry
                    self._live.add(entry)
                    try:
                        self._executor.submit(self._run, entry, deadline=entry.deadline, demand=False)
                    except BaseException:
                        del self._entries[key]
                        entry.state = CANCELLED
                        self._free_slot(entry)
                        raise
                    self._counts["issued"] += 1
                    queued += 1
                    issued += 1
        return issued

    def claim(self, key):
        """Take ownership of a staged entry (decode thread), or None. A claimed entry cannot be displaced or expired."""
        with self._cond:
            entry = self._entries.pop(key, None) if not self._closed else None
            if entry is not None:
                entry.claimed = True
                self._claimed += 1
            return entry

    # Demand workers ------------------------------------------------------------------------------------------

    def finish(self, entry, destination_slot) -> bool:
        """Copy a claimed record into ``destination_slot``. False means the caller must read it directly."""
        with self._cond:
            if entry.state == QUEUED:
                entry.state = CANCELLED          # never wait on a queued read: the hold may be what queues it
                self._counts["cancelled"] += 1
                return False
            late = entry.state == RUNNING
            while entry.state == RUNNING:
                self._cond.wait()
            if entry.state != DONE:
                return False
        self._copy_record(entry.slot, destination_slot)
        with self._cond:
            self._counts["late" if late else "used"] += 1
        return True

    def release(self, entry) -> None:
        """Give a claimed entry's staging slot back once no read can still write it. Idempotent."""
        with self._cond:
            if entry.released:
                return
            entry.released = True
            self._claimed -= 1
            if entry.state == QUEUED:
                entry.state = CANCELLED
                self._counts["cancelled"] += 1
            if entry.state != RUNNING:
                self._free_slot(entry)           # a running read frees the slot itself when it ends

    def demand_begin(self) -> None:
        with self._gate_lock:
            self._demand += 1
            if self.gate:
                self._executor.hold_speculative(True)

    def demand_end(self) -> None:
        with self._gate_lock:
            self._demand = max(0, self._demand - 1)
            if self.gate and self._demand == 0:
                self._executor.hold_speculative(False)

    # Prefetch workers ----------------------------------------------------------------------------------------

    def _run(self, entry) -> None:
        with self._cond:
            if entry.state != QUEUED or self._closed:
                return
            entry.state = RUNNING
        try:
            self._read_record(entry.key[0], entry.key[1], entry.slot)
        except BaseException:
            outcome = FAILED
        else:
            outcome = DONE
        with self._cond:
            entry.state = outcome
            if outcome == FAILED:
                self._counts["failed"] += 1
                if self._entries.get(entry.key) is entry:
                    del self._entries[entry.key]
                    self._free_slot(entry)
            if entry.released:
                self._free_slot(entry)
            self._cond.notify_all()

    # Internals (hold ``_cond``) ------------------------------------------------------------------------------

    def _free_slot(self, entry) -> None:
        if not entry.freed:
            entry.freed = True
            self._live.discard(entry)
            self._free.append(entry.slot)
            self._cond.notify_all()

    def _take_slot(self, step):
        """A free staging slot, else the oldest completed entry whose deadline has passed, else None."""
        if self._free:
            return self._free.pop()
        stale = [entry for entry in self._entries.values() if entry.state == DONE and entry.deadline < step]
        if not stale:
            return None
        victim = min(stale, key=lambda entry: entry.deadline)
        del self._entries[victim.key]
        self._counts["wasted"] += 1
        self._free_slot(victim)
        return self._free.pop()

    # Lifecycle and stats -------------------------------------------------------------------------------------

    def close(self) -> None:
        """Cancel queued reads, wait for running ones, release the threads. No write happens after this returns."""
        with self._cond:
            self._closed = True
            for entry in self._live:
                if entry.state == QUEUED:
                    entry.state = CANCELLED
            while any(entry.state == RUNNING for entry in self._live):
                self._cond.wait()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def stats(self) -> dict:
        pending = self._executor.pending()
        with self._cond:
            counts = dict(self._counts)
            useful = counts["used"] + counts["late"]
            return {
                **counts, "staged": len(self._entries), "claimed": self._claimed, "free_slots": len(self._free),
                "staging_slots": self.slots, "staging_bytes": self.slots * self.record_bytes, "pending": pending,
                "precision": useful / max(1, useful + counts["wasted"] + counts["expired"]),
                "settings": {"depth": self.depth, "width": self.width, "min_probability": self.min_probability,
                             "workers": self.workers, "gate": self.gate},
            }
