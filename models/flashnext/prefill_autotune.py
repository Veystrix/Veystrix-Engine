"""Choose a request's chunk shape once, before prefix lookup."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

FLAG = "FLASHNEXT_PREFILL_AUTOTUNE"
SHAPES_FLAG = "FLASHNEXT_PREFILL_SHAPE_PROFILE"
PLAN_VERSION = "headroom-balanced-v1"
CURRENT = ContextVar("flashnext_prefill_choice", default=None)


@dataclass(frozen=True)
class Choice:
    cap: int
    available_bytes: int
    estimated_bytes: int
    measured: bool
    reusable_tokens: int = 0


def choose(total, offset, available, cap, candidates, *, seconds=None):
    from .prefill_budget import chunk_increment_bytes

    if total <= 0 or offset < 0 or cap <= 0:
        raise ValueError("prefill length and cap must be positive, and offset nonnegative")
    sizes = sorted({cap, *(size for size in candidates if 0 < size <= cap)})
    end = offset + total

    def estimate(size):
        width = min(size, total)
        return chunk_increment_bytes(width, end - width)

    # Reserve room for existing allocator/cache behavior in the estimate. The
    # minimum candidate always remains runnable, even with a zero reading.
    fits = [size for size in sizes if estimate(size) <= available // 2]
    eligible = fits or sizes[:1]
    measured = seconds is not None and all(size in seconds for size in eligible)
    if measured:
        chosen = min(eligible, key=lambda size: (math.ceil(total / size) * seconds[size], -size))
    else:
        chosen = eligible[-1]
    return Choice(chosen, available, estimate(chosen), measured)


def load_profile(path):
    data = json.loads(Path(path).expanduser().read_text())
    if data.get("schema_version") != 1:
        raise ValueError("prefill shape profile must use schema version 1")
    seconds = {int(shape): float(value) for shape, value in data["seconds_by_chunk"].items()}
    if any(size <= 0 or not math.isfinite(value) or value <= 0 for size, value in seconds.items()):
        raise ValueError("prefill shape profile needs positive chunk sizes and durations")
    return seconds


def enabled():
    return os.environ.get(FLAG) == "1"


def start(total, offset, *, available_bytes=None, prior_prefix=None):
    if not enabled():
        return None
    from . import memory_manager, prefill

    cap = prefill.configured_chunk_tokens() or prefill.MAX_CHUNK_TOKENS
    # A follow-up with an existing live cache is planned at its absolute
    # context end, even though only its suffix is being computed.
    available = memory_manager.read_system()["available"] if available_bytes is None else available_bytes
    profile = os.environ.get(SHAPES_FLAG)
    seconds = load_profile(profile) if profile else None
    choice = choose(total, offset, available, cap, prefill.CANDIDATE_CHUNKS, seconds=seconds)
    if prior_prefix is not None and offset == 0:
        prior_cap, peek = prior_prefix
        if 0 < prior_cap <= cap and prior_cap != choice.cap:
            from .prefill_budget import chunk_increment_bytes

            marker = CURRENT.set(Choice(prior_cap, available, 0, False))
            try:
                try:
                    reused = peek()
                except (OSError, ValueError):
                    reused = 0
            finally:
                CURRENT.reset(marker)
            if 0 < reused < total:
                width = min(prior_cap, total - reused)
                extra = chunk_increment_bytes(width, total - width)
                if extra <= available // 2:
                    choice = Choice(prior_cap, available, extra, False, reused)
    return CURRENT.set(choice)
