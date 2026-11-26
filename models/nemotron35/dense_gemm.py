"""Opt-in dense GEMM for Nemotron's large quantized projections: see ``models.shared.dense_gemm``."""
from __future__ import annotations

from models.shared import dense_gemm as _shared
from models.shared.dense_gemm import MIN_ROWS, _bit_equal  # noqa: F401  (re-exported for tests)

FLAG = "VEYSTRIX_NEMOTRON_DENSE_GEMM"


def enabled() -> bool:
    return _shared.flag_enabled(FLAG)


def install() -> None:
    _shared.install(FLAG)
