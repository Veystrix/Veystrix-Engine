"""Discovery helpers for the Linux CI fast lane, where MLX is unavailable.

Most shared tests are pure Python and run in seconds on Linux. A minority
import MLX (directly or through a backend) and are skipped there; they still
run in full on the macOS job. The classification is done by attempting the
import, so adding a new pure-Python test needs no list update.
"""
from __future__ import annotations

import importlib
from pathlib import Path

TESTS_DIR = Path(__file__).parent
MLX_MODULES = frozenset({"mlx", "mlx_lm", "mlx_vlm"})

# These import on Linux (MLX is pulled in lazily inside tests/fixtures) but
# exercise real MLX code paths at run time, so they belong to the macOS job.
RUNTIME_MLX_MODULES = frozenset({
    "veystrix.tests.test_concurrency_bench",
    "veystrix.tests.test_interactive_chat_flow",
    "veystrix.tests.test_lookup_v2",
    "veystrix.tests.test_session",
    "veystrix.tests.test_settings",
    "veystrix.tests.test_warm",
    "veystrix.tests.test_server_watchdog_integration",
})


def test_module_names() -> list[str]:
    """Every shared test module, in a stable order."""
    return sorted("veystrix.tests." + path.stem
                  for path in TESTS_DIR.glob("test_*.py"))


def needs_mlx_at_runtime(name: str) -> bool:
    return name in RUNTIME_MLX_MODULES


def import_test_module(name: str):
    """Return ``(module, None)`` or ``(None, reason)`` when the module needs MLX.

    Other import errors are re-raised: they are real breakages, not a reason to
    skip. Only a missing MLX-family dependency is treated as expected on Linux.
    """
    try:
        return importlib.import_module(name), None
    except ModuleNotFoundError as exc:
        top = (exc.name or "").split(".")[0]
        if top in MLX_MODULES:
            return None, f"needs {exc.name}"
        raise
