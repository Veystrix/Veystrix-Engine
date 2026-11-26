"""CPU-only unit tests for the offline multi-session simulator (tiny hand-written fixtures, no model, no SSD).

Run from the worktree root:
    .venv/bin/python -m pytest models/shared/tests/unit/test_multisession_sim.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_FLASH_BENCH = _ROOT / "models" / "flashnext" / "tests" / "bench"
for _p in (str(_ROOT), str(_FLASH_BENCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from bench_pool_policy_trace import POLICIES, replay_detail  # noqa: E402
from models.shared.tests.bench.multisession_sim import (  # noqa: E402
    ARRANGEMENT_A, ARRANGEMENT_D,
    build_arrangements, simulate,
)

# Tiny hand-written fixture: 3 tokens, 2 layers, unique experts per layer (so "unique reads" == raw requests).
_TOKENS = [
    {0: (1, 2, 3), 1: (4, 5)},
    {0: (2, 3, 6), 1: (5, 7)},
    {0: (1, 2, 3), 1: (4, 5)},
]
_EVENTS = [[0, "decode", t, layer, list(experts)]
           for t, token in enumerate(_TOKENS) for layer, experts in sorted(token.items())]
_SLOTS = 4  # small enough that replacement matters


def _sim(policy="lru", N=1, arrangement=ARRANGEMENT_A, scheduler="lockstep", seed=0, **kw):
    return simulate("qwen", arrangement, N, scheduler, policy, 1.0, [_TOKENS],
                    session_state_mb=0.0, record_bytes=1000, slots_override=_SLOTS,
                    seed=seed, **kw)


def test_n1_matches_pool_policy_trace():
    """N=1: simulator hit count / unique count / hit rate equal bench_pool_policy_trace on same rows."""
    detail = replay_detail(_EVENTS, _SLOTS, "lru")
    result = _sim("lru")
    assert result["requests"] == detail["decode_total"]
    assert result["unique_reads"] == detail["decode_total"]  # unique experts per row in this fixture
    assert result["hits"] == detail["decode_hits"]
    assert result["misses"] == detail["misses"]
    assert result["hit_rate"] == pytest.approx(detail["decode_hit_rate"])
    # Also check a second policy so the equality is not accidental to LRU.
    for policy in ("arc", "least-stale", "belady"):
        detail = replay_detail(_EVENTS, _SLOTS, policy)
        result = _sim(policy)
        assert result["hits"] == detail["decode_hits"]
        assert result["hit_rate"] == pytest.approx(detail["decode_hit_rate"])


def test_belady_is_upper_bound():
    """Belady's hit rate >= every other policy on the same tiny trace."""
    results = {policy: _sim(policy, N=2) for policy in POLICIES}
    belady = results["belady"]["hit_rate"]
    for policy, result in results.items():
        assert belady >= result["hit_rate"] - 1e-12, (policy, belady, result["hit_rate"])


def test_lockstep_coalescing_is_exactly_half():
    """Two sessions with identical routes: each layer step's raw requests halve -> coalesced fraction 0.5."""
    result = _sim(N=2)
    assert result["requests"] == 2 * sum(len(e) for token in _TOKENS for e in token.values())
    assert result["unique_reads"] == sum(len(e) for token in _TOKENS for e in token.values())
    assert result["coalesced"] == result["unique_reads"]
    assert result["coalesced_fraction"] == pytest.approx(0.5)


def test_deterministic_for_fixed_seed():
    """Same seed -> byte-identical result dicts (random arrangement exercises the RNG path)."""
    a = _sim(policy="lru", N=3, arrangement=ARRANGEMENT_D, seed=123)
    b = _sim(policy="lru", N=3, arrangement=ARRANGEMENT_D, seed=123)
    assert a == b
    # build_arrangements itself is deterministic for a fixed seed too.
    traces = [_TOKENS]
    x = build_arrangements(traces, "qwen", 3, scheduler="interleaved", seed=7)
    y = build_arrangements(traces, "qwen", 3, scheduler="interleaved", seed=7)
    assert x == y
    # A different seed is allowed to change the random arrangement.
    z = build_arrangements(traces, "qwen", 3, scheduler="interleaved", seed=8)
    assert x != z


def test_capacity_shrinks_linearly_with_n():
    """capacity_available_gb == cache_gb - N * session_state_mb / 1024."""
    cache_gb, state_mb = 4.0, 900.0
    for N in (1, 2, 3, 4):
        result = simulate("qwen", ARRANGEMENT_A, N, "lockstep", "lru", cache_gb, [_TOKENS],
                          session_state_mb=state_mb, record_bytes=1000)
        expected = cache_gb - N * state_mb / 1024.0
        assert result["capacity_available_gb"] == pytest.approx(expected)
        assert result["slots"] == int(expected * 1e9) // 1000
