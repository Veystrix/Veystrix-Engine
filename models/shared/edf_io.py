"""Deadline-ordered read executor for SSD-background expert streaming (blueprint EXP-I).

The stock path submits demand misses and speculative prefetches to two separate
fixed thread pools, so a queue of stale prefetches can still delay the demand
read that the active layer is waiting on. This executor keeps one pool and runs
reads in earliest-deadline-first order, with demand ranked before speculative
reads at the same deadline. ``cancel_stale`` drops speculative work whose
deadline has already passed, so a layer that changes its mind does not leave
useless reads occupying an NVMe slot.

This is an opt-in research path. The read function still runs on a worker
thread; only the order in which queued work reaches the workers changes.
"""

from __future__ import annotations

import concurrent.futures as cf
import heapq
import itertools
import threading


class DeadlineExecutor:
    """A fixed worker pool that runs the nearest deadline first."""

    def __init__(self, workers: int, thread_name_prefix: str = "edf-io"):
        if type(workers) is not int or workers < 1:
            raise ValueError("DeadlineExecutor needs at least one worker")
        self.workers = workers
        self._heap: list = []
        self._sequence = itertools.count()
        self._condition = threading.Condition()
        self._shutdown = False
        self._held = False
        self._threads = [threading.Thread(target=self._run, name=f"{thread_name_prefix}-{index}", daemon=True)
                         for index in range(workers)]
        self.submitted = self.completed = self.skipped = self.failed = 0
        for thread in self._threads:
            thread.start()

    def submit(self, fn, /, *args, deadline: int = 0, demand: bool = True, **kwargs) -> cf.Future:
        """Queue ``fn(*args, **kwargs)``. Lower deadlines run first; demand beats prefetch on ties."""
        future: cf.Future = cf.Future()
        with self._condition:
            if self._shutdown:
                raise RuntimeError("DeadlineExecutor is shut down")
            heapq.heappush(self._heap, (int(deadline), 0 if demand else 1, next(self._sequence),
                                        (future, fn, args, kwargs)))
            self.submitted += 1
            self._condition.notify()
        return future

    def cancel_stale(self, before_deadline: int) -> int:
        """Drop queued speculative work for layers older than ``before_deadline``. Returns the count."""
        dropped = 0
        with self._condition:
            keep = []
            for item in self._heap:
                deadline, rank, sequence, (future, _fn, _args, _kwargs) = item
                if rank != 0 and deadline < before_deadline:
                    if future.set_running_or_notify_cancel():
                        future.set_exception(cf.CancelledError("stale prefetch cancelled"))
                    else:
                        future.cancel()
                    self.skipped += 1
                    dropped += 1
                else:
                    keep.append(item)
            if dropped:
                self._heap = keep
                heapq.heapify(self._heap)
        return dropped

    def cancel_layer(self, predicate) -> int:
        """Drop queued speculative work whose ``predicate(deadline)`` is true."""
        dropped = 0
        with self._condition:
            keep = []
            for item in self._heap:
                deadline, rank, _sequence, (future, _fn, _args, _kwargs) = item
                if rank != 0 and predicate(deadline):
                    future.cancel()
                    self.skipped += 1
                    dropped += 1
                else:
                    keep.append(item)
            if dropped:
                self._heap = keep
                heapq.heapify(self._heap)
        return dropped

    def hold_speculative(self, held: bool) -> None:
        """While held, workers run only demand items; speculative items stay queued in order."""
        with self._condition:
            self._held = bool(held)
            if not self._held:
                self._condition.notify_all()

    def pending(self) -> int:
        with self._condition:
            return len(self._heap)

    def _take(self):
        """Pop the next runnable item, or None. While held only demand items qualify."""
        if not self._held:
            return heapq.heappop(self._heap) if self._heap else None
        demand = [item for item in self._heap if item[1] == 0]
        if not demand:
            return None
        item = min(demand, key=lambda entry: entry[:3])
        self._heap.remove(item)
        heapq.heapify(self._heap)
        return item

    def _run(self) -> None:
        while True:
            with self._condition:
                while True:
                    item = self._take()
                    if item is not None or self._shutdown:
                        break
                    self._condition.wait()
                if item is None:
                    return
                _deadline, _rank, _sequence, (future, fn, args, kwargs) = item
            if not future.set_running_or_notify_cancel():
                with self._condition:
                    self.skipped += 1
                continue
            try:
                result = fn(*args, **kwargs)
            except BaseException as error:  # handed to the waiting caller
                future.set_exception(error)
                with self._condition:
                    self.failed += 1
            else:
                future.set_result(result)
                with self._condition:
                    self.completed += 1

    def stats(self) -> dict:
        with self._condition:
            return {"workers": self.workers, "pending": len(self._heap), "submitted": self.submitted,
                    "completed": self.completed, "skipped": self.skipped, "failed": self.failed,
                    "held": self._held}

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        with self._condition:
            self._shutdown = True
            queued, self._heap = self._heap, []
            for _deadline, _rank, _sequence, (future, _fn, _args, _kwargs) in queued:
                if cancel_futures:
                    future.cancel()
                else:
                    if future.set_running_or_notify_cancel():
                        future.set_exception(RuntimeError("DeadlineExecutor shut down"))
                self.skipped += 1
            self._condition.notify_all()
        if wait:
            for thread in self._threads:
                thread.join(timeout=5)
