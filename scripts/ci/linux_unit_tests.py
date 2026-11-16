"""Run every shared test module that does not need MLX (Linux CI fast lane).

The macOS job still runs the complete suite on real MLX; this job gives fast
feedback on pure-Python changes and needs no Apple hardware.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from veystrix.tests._linux import import_test_module, needs_mlx_at_runtime, test_module_names


def main() -> int:
    names = test_module_names()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    skipped: list[tuple[str, str]] = []
    for name in names:
        if needs_mlx_at_runtime(name):
            skipped.append((name, "uses MLX at runtime"))
            continue
        module, reason = import_test_module(name)
        if module is None:
            skipped.append((name, reason or "needs MLX"))
            continue
        suite.addTests(loader.loadTestsFromModule(module))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(f"\nLinux fast lane: {len(names) - len(skipped)} modules run, "
          f"{len(skipped)} skipped (require MLX)")
    for name, reason in skipped:
        print(f"  skip {name}: {reason}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
