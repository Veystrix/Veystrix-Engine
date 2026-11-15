"""Does this machine's MLX accept zero-copy imports of memory-mapped files?

The expert pool and the prefix-cache restore wrap memory-mapped pages as MLX arrays without copying them
(``mx.from_dlpack(..., copy=False)``). That works on a Mac with a real GPU. GitHub's hosted macOS runners report a Metal
GPU, but it is a virtual one: the import fails with "Cannot import a CPU array without a copy" and cache restores come back
empty. Tests that need the import call ``requires_zero_copy`` and are skipped, not failed, where it does not work.
This is a test-support probe; the engine itself is unchanged.
"""
import mmap
import tempfile
import unittest


def zero_copy_import_works() -> bool:
    try:
        import mlx.core as mx
        import numpy as np
        with tempfile.TemporaryFile() as handle:
            handle.truncate(mmap.PAGESIZE * 4)
            with mmap.mmap(handle.fileno(), mmap.PAGESIZE * 4) as pages:
                view = np.frombuffer(pages, dtype=np.uint32)
                array = mx.from_dlpack(view, copy=False)
                mx.eval(array)
                ok = int(array.size) == len(view)
                del array, view
                return ok
    except Exception:
        return False


requires_zero_copy = unittest.skipUnless(
    zero_copy_import_works(), "needs zero-copy MLX imports of memory-mapped files (GitHub's virtual macOS GPUs lack them)")
