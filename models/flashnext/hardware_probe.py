"""What this machine can do that some Flash-Next tests assume.

The tests run on a Mac with a real GPU and several GiB of free memory. GitHub's hosted macOS runners are virtual machines with
7 GB of RAM and a virtual GPU: its largest Metal buffer is 3.76 GB, so the "big pool" tests (a 4.6 GB pool) cannot be built;
free memory sits under the 3 GiB floor Veystrix enforces, so prefix-cache snapshots and pools are refused on purpose and cache
restores come back COLD; and zero-copy imports can fail. ``requires_full_mac`` skips those tests, with the reason in the
report, where the machine cannot run them. Set ``VEYSTRIX_CI_RUNNER=1`` (the CI workflow does) to say so explicitly. This is a
test-support module; the engine is unchanged.
"""
import mmap
import os
import unittest

MIN_BUFFER_BYTES = 6 * 2**30     # the largest pool a test builds, with headroom
import mmap
import unittest


def zero_copy_import_works() -> bool:
    """Repeat what ``ExpertPool`` does: wrap an anonymous, page-aligned mmap as an MLX array without a copy."""
    try:
        import mlx.core as mx
        import numpy as np
        with mmap.mmap(-1, mmap.PAGESIZE * 4) as pages:
            view = np.frombuffer(pages, dtype=np.uint32)
            array = mx.from_dlpack(view, copy=False)
            mx.eval(array)
            ok = int(array.size) == len(view)
            del array, view
            return ok
    except Exception:
        return False


def max_buffer_bytes() -> int:
    try:
        import mlx.core as mx
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        return int(info.get("max_buffer_length", 0))
    except Exception:
        return 0


def full_mac() -> bool:
    if os.environ.get("VEYSTRIX_CI_RUNNER") == "1":
        return False
    return zero_copy_import_works() and max_buffer_bytes() >= MIN_BUFFER_BYTES


requires_full_mac = unittest.skipUnless(
    full_mac(), "needs a Mac with a real GPU and free memory (GitHub's hosted macOS runners are virtual, 7 GB)")
requires_zero_copy = requires_full_mac      # the name the tests already use
