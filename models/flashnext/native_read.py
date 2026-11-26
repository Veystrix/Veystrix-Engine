"""One native call per decode layer for expert reads.

The production read path submits about 70 Python tasks per layer, one per
chunk of rows, and each runs a Python loop around ``os.pread``. An offline
replay of real routes (2026-09-23) priced that path at about 34 ms/token of
fixed overhead beyond the drive, and a fully cached layer cost 1.69 ms against
0.78 ms for a plain eight-thread copy of the same bytes. Here a layer builds
one table of reads with a few numpy operations and ``native_read.c`` performs
them with the GIL released, on a user-interactive concurrent queue. The
destinations and bytes are unchanged.

Packaged builds ship a precompiled ``native_read.bundled.dylib`` so no C source
or compiler is needed on the user's Mac. Source checkouts build once per source
digest into ``~/.cache/flashnext/native``.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

WIDTH = [int(os.environ.get("FLASHNEXT_IO_WORKERS", "8"))]
READ_DTYPE = np.dtype([
    ("fd", "<i4"), ("pad", "<i4"), ("offset", "<i8"),
    ("length", "<i8"), ("dst", "<u8"),
])
_LIB = [None]
_LOCK = threading.Lock()
# Packaged builds ship this precompiled library; no C source or compiler is
# needed on the user's Mac. Source checkouts still build from native_read.c.
BUNDLED_LIBRARY = Path(__file__).with_name("native_read.bundled.dylib")


def _library() -> ctypes.CDLL:
    with _LOCK:
        if _LIB[0] is None:
            _LIB[0] = _load_library()
        return _LIB[0]


def _open_library(library: Path) -> ctypes.CDLL:
    handle = ctypes.CDLL(str(library))
    handle.flashnext_batch_pread.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
    handle.flashnext_batch_pread.restype = ctypes.c_int
    return handle


def _load_library() -> ctypes.CDLL:
    if BUNDLED_LIBRARY.is_file():
        return _open_library(BUNDLED_LIBRARY)
    source = Path(__file__).with_name("native_read.c")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory = Path(os.environ.get("VEYSTRIX_NATIVE_CACHE", "~/.cache/flashnext/native")).expanduser()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    library = directory / f"native_read-{digest}.dylib"
    if not library.exists():
        with tempfile.TemporaryDirectory(prefix=".build-", dir=directory) as scratch:
            partial = Path(scratch) / library.name
            result = subprocess.run(
                ["xcrun", "clang", "-O2", "-dynamiclib", str(source), "-o", str(partial)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(f"native_read build failed: {result.stderr.strip()}")
            partial.replace(library)
    return _open_library(library)


class Batch:
    """Collects one layer's reads as numpy blocks, then runs them in one call."""

    __slots__ = ("blocks",)

    def __init__(self):
        self.blocks = []

    def add(self, fd: int, start: int, row_bytes: int, experts, base: int, stride: int) -> None:
        """Read row ``experts[k]`` to ``base + k * stride`` for every k."""
        experts = np.asarray(experts, dtype=np.int64)
        block = np.empty(len(experts), dtype=READ_DTYPE)
        block["fd"] = fd
        block["pad"] = 0
        block["offset"] = start + experts * row_bytes
        block["length"] = row_bytes
        block["dst"] = base + np.arange(len(experts), dtype=np.uint64) * np.uint64(stride)
        self.blocks.append(block)

    def add_records(self, fd, offsets, row_bytes, destinations):
        """Read complete records into caller-owned, potentially nonadjacent slots."""
        block = np.empty(len(offsets), dtype=READ_DTYPE)
        block["fd"] = fd
        block["pad"] = 0
        block["offset"] = offsets
        block["length"] = row_bytes
        block["dst"] = destinations
        self.blocks.append(block)

    def run(self) -> None:
        if not self.blocks:
            return
        table = self.blocks[0] if len(self.blocks) == 1 else np.concatenate(self.blocks)
        failed = _library().flashnext_batch_pread(
            table.ctypes.data, len(table), WIDTH[0],
        )
        if failed:
            raise OSError(f"{failed} native expert reads failed")


def address(array: np.ndarray) -> int:
    return int(array.__array_interface__["data"][0])
