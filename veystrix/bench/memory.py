"""Sample macOS memory out of band. Missing fields stay null."""
from __future__ import annotations

import re
from threading import Event, Thread
from time import perf_counter_ns

from .provenance import output


def system_sample():
    from models.flashnext.watchdog import swap_used_mib

    vm = output("vm_stat") or ""
    page = re.search(r"page size of (\d+) bytes", vm)
    fields = dict(re.findall(r"(?m)^([^:\n]+):\s+(\d+)\.$", vm))
    queues = ("Pages free", "Pages inactive", "Pages speculative")

    def pages(*names):
        if page and all(name in fields for name in names):
            return sum(int(fields[name]) for name in names) * int(page[1])
        return None

    try:
        swap = round(swap_used_mib() * 2**20)
    except (OSError, ValueError):
        swap = None
    return {
        "available_bytes": pages(*queues),
        "compressor_bytes": pages("Pages occupied by compressor"),
        "swap_used_bytes": swap,
        "swapin_bytes_total": pages("Swapins"),
        "swapout_bytes_total": pages("Swapouts"),
    }


class MemorySampler:
    def __init__(self, origin, interval=1.0, read=system_sample):
        self.origin = origin
        self.interval = interval
        self.read = read
        self.samples = []
        self.stop = Event()
        self.thread = Thread(target=self._run, daemon=True)

    def _sample(self):
        started = perf_counter_ns() - self.origin
        self.samples.append({"ns": started, **self.read()})

    def _run(self):
        while not self.stop.wait(self.interval):
            self._sample()

    def prepare(self):
        self._sample()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self._sample()

    def summary(self):
        result = {"sample_interval_s": self.interval, "samples": self.samples}
        available = [row["available_bytes"] for row in self.samples if row["available_bytes"] is not None]
        result["sampled_available_floor_bytes"] = min(available, default=None)
        for key in ("compressor_bytes", "swap_used_bytes", "swapin_bytes_total", "swapout_bytes_total"):
            first, last = self.samples[0][key], self.samples[-1][key]
            result[key.removesuffix("_total") + "_delta"] = (
                last - first if first is not None and last is not None else None
            )
        return result
