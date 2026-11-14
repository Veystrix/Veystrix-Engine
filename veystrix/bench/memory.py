"""Sample macOS memory out of band. Missing fields stay null."""
from __future__ import annotations

import ctypes
import os
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


class _VMStatistics64(ctypes.Structure):
    """``vm_statistics64`` from <mach/vm_statistics.h>."""

    _fields_ = [
        ("free_count", ctypes.c_uint32), ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32), ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64), ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64), ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64), ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64), ("hits", ctypes.c_uint64), ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32), ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64), ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64), ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32), ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32), ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


_HOST_VM_INFO64 = 4
_HOST = None


def kernel_sample():
    """The fields of ``system_sample`` read from the kernel, without launching ``vm_stat``.

    Forking a process that maps the model costs tens of milliseconds with the
    interpreter lock held, which a once-a-second sampler turns into measurable
    decode overhead. ``vm_stat`` prints free pages net of speculative ones, so
    its free, inactive and speculative lines sum to ``free_count +
    inactive_count`` here. Raises OSError when the call is unavailable.
    """
    global _HOST
    from models.flashnext.watchdog import swap_used_mib

    libc = ctypes.CDLL(None, use_errno=True)
    if _HOST is None:
        libc.mach_host_self.restype = ctypes.c_uint32
        _HOST = libc.mach_host_self()
    stats = _VMStatistics64()
    count = ctypes.c_uint32(ctypes.sizeof(stats) // ctypes.sizeof(ctypes.c_int32))
    call = libc.host_statistics64
    call.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    call.restype = ctypes.c_int
    status = call(_HOST, _HOST_VM_INFO64, ctypes.byref(stats), ctypes.byref(count))
    if status != 0:
        raise OSError(f"host_statistics64 failed with {status}")
    page = os.sysconf("SC_PAGE_SIZE")
    try:
        swap = round(swap_used_mib() * 2**20)
    except (OSError, ValueError):
        swap = None
    return {
        "available_bytes": (stats.free_count + stats.inactive_count) * page,
        "compressor_bytes": stats.compressor_page_count * page,
        "swap_used_bytes": swap,
        "swapin_bytes_total": stats.swapins * page,
        "swapout_bytes_total": stats.swapouts * page,
    }


def sample():
    """One memory reading: the direct kernel call, or ``vm_stat`` where that is unavailable."""
    try:
        return kernel_sample()
    except (OSError, AttributeError):
        return system_sample()


class MemorySampler:
    def __init__(self, origin, interval=1.0, read=sample):
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
