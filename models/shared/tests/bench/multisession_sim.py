"""Offline N-session cache/coalescing simulator for multi-session Phase 0 (no model, no SSD).

This is a *simulator*, not a benchmark run: it reads already-recorded route traces as plain JSON and
replays them through the expert-pool replacement policies in
``models/flashnext/tests/bench/bench_pool_policy_trace.py``. No checkpoint is loaded and no SSD is touched.

It answers: when N decode sessions share one expert/page cache, how many distinct experts must be read
per aggregate token, how much of the traffic coalesces, and what the SSD/GPU/copy-limited throughput is.

    python -m models.shared.tests.bench.multisession_sim            # full sweep -> results/multisession/<ts>-sim/
    python -m models.shared.tests.bench.multisession_sim --help

Number labels used throughout (all numbers in this file are labeled):
  * measured  - read verbatim from a recorded trace file (e.g. Qwen record_bytes=3211264).
  * cited     - taken from an existing results artifact in this repo (e.g. cross-conversation share 0.0627,
                results/flashnext/20261114-132303-route-overlap-manual/route-overlap.json).
  * estimate  - an assumption/derivation here (GPU ms/token, 5.61 MB per Nemotron expert, top-k scaling,
                session-state MB, copy ms/expert, throughput projections, sanity anchors).
  * idea      - arrangement/scheduler semantics, which are modelling constructs, not measurements.

Trace facts (measured):
  Qwen: 3 sessions (turns) x 256 decode tokens x 48 layers, ~8 streamed experts/layer/token. The 8 pinned
        experts/layer are excluded from the trace. Docs describe 10 routed experts/layer; use --topk-scale
        1.25 (10/8) to report that case, labeled estimate.
  Nemotron: 4 distinct 124-128 token traces x 23 MoE layers x 6 experts, 128 experts, after deduplicating
        profile files by a hash of the first 8 tokens' routes. Runs share harness prompts, so token-aligned
        traces are more similar than truly unrelated sessions (idea/estimate caveat).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reuse, do not rewrite: add the repo root and the existing bench directory to sys.path.
_ROOT = Path(__file__).resolve().parents[4]
_FLASH_BENCH = _ROOT / "models" / "flashnext" / "tests" / "bench"
_LOCAL_BENCH = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_FLASH_BENCH), str(_LOCAL_BENCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench_pool_policy_trace import POLICIES, make_policy  # noqa: E402
from bench_route_overlap import decode_routes, load  # noqa: E402
from veystrix.results import RESULTS_ROOT, output_path  # noqa: E402

# Task 2 feeds Task 1: derive the per-session state from the session_state_mb
# calculator (repo-derived configs, bf16 Qwen KV / INT8 Nemotron KV) instead of
# hard-coded guesses. Defaults are at 4K context (the decode traces are short).
from session_state_mb import PRODUCTION_CONFIGS, session_state_mb as _calc_state  # noqa: E402


def _derived_session_state(model: str, ctx_len: int = 4096) -> float:
    return round(_calc_state(model, ctx_len, PRODUCTION_CONFIGS[model]), 1)

# ------------------------------------------------------------------------------------------------------------
# Constants. Every numeric default is labeled in a comment.
# ------------------------------------------------------------------------------------------------------------
QWEN_TRACE = ("/Users/vincent/Documents/Veystrix/results/flashnext/"
              "20260922-221213-route-trace-manual/route-trace.json.gz")
NEMOTRON_GLOB = ("/Users/vincent/Documents/Veystrix/results/nemotron35/"
                 "**/profile/nemotron-decode-*.json")

QWEN_RECORD_BYTES = 3211264                       # measured (trace header)
NEMOTRON_RECORD_BYTES = int(5.61 * 1024 * 1024)   # estimate (cited 5.61 MB per expert)
QWEN_SLOTS = 349                                  # cited production expert pool (349 * 3.21 MB ~= 1.12 GB)
QWEN_PINNED = 8                                   # measured (trace header resident_experts)
QWEN_TOPK_DOCS = 10                               # cited (docs say 10 routed experts/layer)

# derived ESTIMATE from Task 2 session_state_mb.py at 4K context (repo-derived
# configs; no checkpoint read). Override with --qwen-session-state-mb etc.
QWEN_SESSION_STATE_MB = _derived_session_state("qwen")       # ~216 MB
NEMOTRON_SESSION_STATE_MB = _derived_session_state("nemotron")  # ~62 MB
SSD_GBPS = {"qwen": 2.9, "nemotron": 3.3}         # estimate (CLI defaults; "cited" SSD figures)
GPU_MS = {"qwen": 100.0, "nemotron": 38.0}        # estimate (CLI defaults)
HIT_COPY_MS_PER_EXPERT = 0.21                     # estimate (cited 1.28 ms / 6 cached experts)

ARRANGEMENT_A = "distinct-aligned"                # idea
ARRANGEMENT_B = "distinct-shift"                  # idea
ARRANGEMENT_C_PREFIX = "shared-offset-k"          # idea; variants k=7,14,21
ARRANGEMENT_D = "random"                          # idea
DEFAULT_SHARED_OFFSETS = (7, 14, 21)              # idea (task-specified)
DEFAULT_SHIFT = 31                                # idea (task-specified)
DEFAULT_NEMOTRON_MIN_TOKENS = 100                 # filter for full decode traces (measured: full traces >=124)


@dataclass(frozen=True)
class Session:
    """One session's view: a trace plus its token time-shift and (interleaved) layer phase."""
    trace: tuple                   # tuple of tokens; token = dict {layer: tuple(experts)}
    token_offset: int
    layer_offset: int = 0

    def token(self, t, T):
        return self.trace[(self.token_offset + t) % len(self.trace)]


# ------------------------------------------------------------------------------------------------------------
# Trace loading. Both loaders return (traces, meta) where a trace is a list of {layer: tuple(experts)}.
# ------------------------------------------------------------------------------------------------------------
def load_qwen_traces(path=QWEN_TRACE):
    """Qwen route-trace.json.gz -> one trace per turn (measured route data)."""
    data = load(path)
    routes = decode_routes(data["events"])
    traces = []
    for turn in sorted(routes):
        trace = []
        for row in routes[turn]:
            trace.append({int(layer): tuple(int(e) for e in experts) for layer, experts in row.items()})
        traces.append(trace)
    meta = {"record_bytes": int(data["record_bytes"]),
            "pinned_per_layer_measured": int(data.get("resident_experts", QWEN_PINNED)),
            "sessions": len(traces), "tokens": [len(t) for t in traces],
            "trace": str(path), "label": "measured"}
    return traces, meta


def load_nemotron_traces(pattern=NEMOTRON_GLOB, min_tokens=DEFAULT_NEMOTRON_MIN_TOKENS):
    """Dedupe Nemotron decode profiles by a hash of the first 8 tokens' routes (measured route data)."""
    groups = {}
    files = sorted(glob.glob(pattern, recursive=True))
    for path in files:
        try:
            data = json.load(open(path))
        except (OSError, ValueError):
            continue
        tokens = data.get("tokens") or []
        if len(tokens) < min_tokens:
            continue
        signature = json.dumps([tok["routes"] for tok in tokens[:8]], sort_keys=True)
        key = hashlib.sha256(signature.encode()).hexdigest()[:12]
        if key in groups:
            continue
        trace = []
        for tok in tokens:
            trace.append({int(layer): tuple(int(e) for e in experts)
                          for layer, experts in tok["routes"].items()})
        groups[key] = trace
    traces = [groups[key] for key in sorted(groups)]
    meta = {"record_bytes": NEMOTRON_RECORD_BYTES, "record_label": "estimate (5.61 MB per expert)",
            "sessions": len(traces), "tokens": [len(t) for t in traces],
            "files_seen": len(files), "dedup_key": "sha256(routes of first 8 tokens)[:12]",
            "pattern": pattern, "label": "measured (routes), estimate (record bytes)",
            "caveat": ("runs share harness prompts, so token-aligned traces are more similar than truly "
                       "unrelated sessions")}
    return traces, meta


# ------------------------------------------------------------------------------------------------------------
# Deterministic RNG helper: a sim's randomness depends only on (seed, model, arrangement, N, scheduler),
# never on call order.
# ------------------------------------------------------------------------------------------------------------
def _rng_for(seed, *parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    salt = int.from_bytes(digest[:8], "big")
    return np.random.default_rng((int(seed) ^ salt) & 0xFFFFFFFFFFFFFFFF)


def _layers_of(traces):
    return sorted({layer for trace in traces for token in trace for layer in token})


def _empirical_distributions(traces):
    """Per-layer expert histogram and experts-per-layer count histogram (measured from the given traces)."""
    freq, counts = defaultdict(Counter), defaultdict(Counter)
    for trace in traces:
        for token in trace:
            for layer, experts in token.items():
                freq[layer].update(experts)
                counts[layer][len(experts)] += 1
    return freq, counts


def _random_trace(freq, counts, layers, T, rng):
    """Synthetic trace with the same per-layer frequency/count distributions (idea: lower bound)."""
    trace = []
    for _ in range(T):
        token = {}
        for layer in layers:
            opts = counts[layer]
            ks = sorted(opts)
            n = int(rng.choice(ks, p=np.array([opts[k] for k in ks], dtype=float) /
                               sum(opts.values()))) if opts else 0
            ids = sorted(freq[layer])
            probs = np.array([freq[layer][e] for e in ids], dtype=float)
            probs = probs / probs.sum()
            n = min(n, len(ids))
            chosen = rng.choice(ids, size=n, replace=False, p=probs) if n else []
            token[layer] = tuple(int(e) for e in chosen)
        trace.append(token)
    return trace


def build_arrangements(traces, model, N, *, scheduler="lockstep", shift=DEFAULT_SHIFT,
                       shared_offsets=DEFAULT_SHARED_OFFSETS, seed=0, T=None, rng=None):
    """Return {arrangement_name: [Session, ...]} for all N-session trace arrangements (idea).

    A distinct-aligned   : different traces, token-aligned (upper-ish bound; prompts are similar).
    B distinct-shift     : different traces, time-shifted by ``shift`` tokens (wraparound); main unrelated case.
    C shared-offset-k*   : one trace offset by k in {7, 14, 21}; proxy for sub-agents sharing a context.
    D random             : synthetic routes with the same per-layer frequency distribution (lower bound).
    """
    if not traces:
        raise ValueError("no traces supplied")
    if T is None:
        T = max(len(t) for t in traces)
    L = len(_layers_of(traces))
    shared_offsets = tuple(shared_offsets) or DEFAULT_SHARED_OFFSETS

    def phase(s, name):
        # Only interleaved uses the layer phase; lockstep keeps all sessions on the same layer position.
        if scheduler == "lockstep":
            return 0
        local = _rng_for(seed, model, name, N, "phase", s)
        return int(local.integers(0, L))

    out = {}

    def add(name, session_specs):
        sessions = []
        for s, (trace, token_offset) in enumerate(session_specs):
            sessions.append(Session(tuple(trace), token_offset % len(trace), phase(s, name)))
        out[name] = sessions

    # A: different traces, aligned.
    add(ARRANGEMENT_A, [(traces[s % len(traces)], 0) for s in range(N)])
    # B: different traces, shifted by s*shift.
    add(ARRANGEMENT_B, [(traces[s % len(traces)], s * shift) for s in range(N)])
    # C: same trace, offset by k (one arrangement per requested k).
    for k in shared_offsets:
        add(f"{ARRANGEMENT_C_PREFIX}{k}", [(traces[0], s * k) for s in range(N)])
    # D: random routes from the empirical per-layer frequency distribution.
    random_rng = rng if rng is not None else _rng_for(seed, model, ARRANGEMENT_D, N, "routes")
    freq, counts = _empirical_distributions(traces)
    layers = sorted(freq)
    synth = [_random_trace(freq, counts, layers, T, random_rng) for _ in range(N)]
    add(ARRANGEMENT_D, [(trace, 0) for trace in synth])
    return out


def arrangement_names(shared_offsets=DEFAULT_SHARED_OFFSETS):
    return [ARRANGEMENT_A, ARRANGEMENT_B] + [f"{ARRANGEMENT_C_PREFIX}{k}" for k in shared_offsets] + [ARRANGEMENT_D]


# ------------------------------------------------------------------------------------------------------------
# Core simulator.
# ------------------------------------------------------------------------------------------------------------
def _build_steps(sessions, layers, T, scheduler, inflight):
    """Ordered cache accesses as (layer, merged_experts) plus raw request count.

    Lockstep: all sessions visit the same layer at the same time slot, so same-layer requests coalesce.
    Interleaved: each session has a random layer phase; during one in-flight window (``inflight`` time
    slots) requests for the same layer coalesce. With inflight=1 only exactly-coincident requests coalesce.
    A duplicate request removed here is "coalesced", never a cache hit.
    """
    L = len(layers)
    by_block = defaultdict(OrderedDict)  # block -> OrderedDict {layer: merged expert list}
    for t in range(T):
        for p in range(L):
            block = (t * L + p) // max(1, inflight)
            slot = by_block[block]
            for sess in sessions:
                layer = layers[p] if scheduler == "lockstep" else layers[(p + sess.layer_offset) % L]
                slot.setdefault(layer, []).extend(sess.token(t, T).get(layer, ()))
    steps = []
    for block in sorted(by_block):
        for layer, experts in by_block[block].items():
            steps.append((layer, experts))
    return steps


def _capacity_slots(capacity_gb, N, session_state_mb, model, slots_override=None, record_bytes=None):
    """Capacity after reserving N x session_state_mb; converted to slots for the given record size."""
    available_gb = max(0.0, float(capacity_gb) - N * float(session_state_mb) / 1024.0)
    if slots_override is not None:
        slots = int(slots_override)
    else:
        slots = int(available_gb * 1e9) // max(1, int(record_bytes))
    return available_gb, max(1, slots)


# Cache of replay hit counts, keyed by the identity of a steps object shared through ``prebuilt_steps``.
# The reused Belady policy is O(cache_steps x slots), so avoiding a duplicate replay matters for the sweep.
_REPLAY_CACHE = {}


def simulate(model, arrangement, N, scheduler, policy, capacity_gb, traces,
             *, session_state_mb=None, record_bytes=None, ssd_gbps=None, gpu_ms=None,
             hit_copy_ms_per_expert=HIT_COPY_MS_PER_EXPERT, shift=DEFAULT_SHIFT,
             shared_offsets=DEFAULT_SHARED_OFFSETS, seed=0, slots_override=None,
             inflight=1, topk_scale=1.0, prebuilt=None, prebuilt_steps=None, T=None):
    """Replay one (model, arrangement, N, scheduler, policy, capacity) cell -> metrics dict.

    ``traces`` is the list returned by load_qwen_traces/load_nemotron_traces. ``prebuilt`` optionally
    supplies build_arrangements() output to avoid regenerating random traces in the full sweep.
    All numeric defaults are estimates unless the docstring of the loader labels them measured/cited.
    """
    if model not in ("qwen", "nemotron"):
        raise ValueError(f"unknown model {model!r}")
    if scheduler not in ("lockstep", "interleaved"):
        raise ValueError(f"unknown scheduler {scheduler!r}")
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    if T is None:
        T = max(len(t) for t in traces)
    if session_state_mb is None:
        session_state_mb = QWEN_SESSION_STATE_MB if model == "qwen" else NEMOTRON_SESSION_STATE_MB
    if record_bytes is None:
        record_bytes = QWEN_RECORD_BYTES if model == "qwen" else NEMOTRON_RECORD_BYTES
    if ssd_gbps is None:
        ssd_gbps = SSD_GBPS[model]
    if gpu_ms is None:
        gpu_ms = GPU_MS[model]

    if prebuilt is not None:
        sessions = prebuilt[arrangement]
    else:
        sessions = build_arrangements(traces, model, N, scheduler=scheduler, shift=shift,
                                      shared_offsets=shared_offsets, seed=seed, T=T)[arrangement]

    layers = _layers_of(traces)
    available_gb, slots = _capacity_slots(capacity_gb, N, session_state_mb, model, slots_override, record_bytes)
    if prebuilt_steps is not None:
        steps = prebuilt_steps
    else:
        steps = _build_steps(sessions, layers, T, scheduler, inflight)

    unique_reads = sum(len(set(experts)) for _, experts in steps)
    requests = sum(len(experts) for _, experts in steps)
    # Only reuse when the caller pins the steps object (the sweep does); a fresh object's id may be recycled.
    cache_key = (id(steps), policy, slots) if prebuilt_steps is not None else None
    hits = _REPLAY_CACHE.get(cache_key) if cache_key is not None else None
    if hits is None:
        # Belady needs the exact future access order, which ``steps`` already is.
        pool = make_policy(policy, slots, steps)
        hits = sum(pool.access(layer, experts) for layer, experts in steps)
        if cache_key is not None:
            _REPLAY_CACHE[cache_key] = hits

    aggregate_tokens = N * T
    coalesced = requests - unique_reads
    misses = unique_reads - hits
    coalesced_fraction = coalesced / requests if requests else 0.0
    hit_rate = hits / unique_reads if unique_reads else 0.0
    unique_per_step = unique_reads / len(steps) if steps else 0.0
    unique_per_token = unique_reads / aggregate_tokens if aggregate_tokens else 0.0
    mb_per_token = misses * record_bytes / aggregate_tokens / 1e6 if aggregate_tokens else 0.0

    ssd_ms = mb_per_token / ssd_gbps if ssd_gbps else 0.0
    # Nemotron only: a cache hit still costs a page-cache -> GPU copy (estimate 0.21 ms/expert).
    copy_ms = (hits / aggregate_tokens * hit_copy_ms_per_expert) if (model == "nemotron" and aggregate_tokens) else 0.0
    io_perfect = max(ssd_ms, copy_ms)          # SSD transfer and copy overlap perfectly (idea)
    io_none = ssd_ms + copy_ms                 # no overlap (idea)
    ssd_tok_s = ssd_gbps * 1000.0 / mb_per_token if mb_per_token else float("inf")
    gpu_tok_s = 1000.0 / gpu_ms if gpu_ms else float("inf")
    combined_perfect = 1000.0 / max(gpu_ms, io_perfect) if max(gpu_ms, io_perfect) else float("inf")
    combined_none = 1000.0 / (gpu_ms + io_none) if (gpu_ms + io_none) else float("inf")
    limiter = "GPU" if gpu_ms >= max(ssd_ms, copy_ms) else ("SSD" if ssd_ms >= copy_ms else "copy")

    result = {
        "model": model, "arrangement": arrangement, "N": N, "scheduler": scheduler,
        "policy": policy, "capacity_gb": capacity_gb, "session_state_mb": session_state_mb,
        "capacity_available_gb": available_gb, "slots": slots,
        "record_bytes": record_bytes, "ssd_gbps": ssd_gbps, "gpu_ms": gpu_ms,
        "hit_copy_ms_per_expert": hit_copy_ms_per_expert if model == "nemotron" else 0.0,
        "token_positions": T, "aggregate_tokens": aggregate_tokens, "layers": len(layers),
        "cache_steps": len(steps), "requests": requests, "unique_reads": unique_reads,
        "coalesced": coalesced, "coalesced_fraction": coalesced_fraction,
        "hits": hits, "misses": misses, "hit_rate": hit_rate,
        "unique_experts_per_layer_step": unique_per_step,
        "unique_experts_per_aggregate_token": unique_per_token,
        "mb_per_aggregate_token": mb_per_token, "ssd_ms": ssd_ms, "copy_ms": copy_ms,
        "ssd_limited_tok_s": ssd_tok_s, "gpu_limited_tok_s": gpu_tok_s,
        "combined_perfect_overlap_tok_s": combined_perfect,
        "combined_no_overlap_tok_s": combined_none, "limiting_resource": limiter,
        "topk_scale": topk_scale,
    }
    if topk_scale != 1.0:
        # estimate: linear scaling of traffic from ~8 to 10 routed experts/layer (Qwen only).
        result["topk_scale_label"] = "estimate (linear scale; Qwen docs say 10 vs trace's ~8 streamed)"
        result["mb_per_aggregate_token_scaled_estimate"] = mb_per_token * topk_scale
        result["ssd_limited_tok_s_scaled_estimate"] = ssd_tok_s / topk_scale if topk_scale else float("inf")
        result["combined_perfect_overlap_tok_s_scaled_estimate"] = combined_perfect / topk_scale if topk_scale else float("inf")
        result["combined_no_overlap_tok_s_scaled_estimate"] = combined_none / topk_scale if topk_scale else float("inf")
    return result


# ------------------------------------------------------------------------------------------------------------
# Reporting (JSON + markdown) through veystrix.results.output_path.
# ------------------------------------------------------------------------------------------------------------
_TABLE_COLUMNS = [
    "Arrangement", "N", "Scheduler", "Policy", "CapGB", "AvailGB", "slots",
    "uniq/step", "uniq/agg-tok", "coalesced%", "hit%", "MB/agg-tok",
    "SSD tok/s", "GPU tok/s", "perfect tok/s", "none tok/s", "limiter",
]


def _md_row(r):
    return "| " + " | ".join([
        r["arrangement"], str(r["N"]), r["scheduler"], r["policy"],
        f"{r['capacity_gb']:g}", f"{r['capacity_available_gb']:g}", str(r["slots"]),
        f"{r['unique_experts_per_layer_step']:.2f}", f"{r['unique_experts_per_aggregate_token']:.1f}",
        f"{r['coalesced_fraction']*100:.1f}", f"{r['hit_rate']*100:.1f}",
        f"{r['mb_per_aggregate_token']:.1f}",
        f"{r['ssd_limited_tok_s']:.2f}", f"{r['gpu_limited_tok_s']:.2f}",
        f"{r['combined_perfect_overlap_tok_s']:.2f}", f"{r['combined_no_overlap_tok_s']:.2f}",
        r["limiting_resource"],
    ]) + " |"


def _markdown(results, meta, notes, anchors):
    lines = ["# Multi-session cache/coalescing simulation (offline, no model / no SSD)", ""]
    lines += ["All numbers below are simulator outputs (estimates/projects), derived from measured route data.",
              "Row definitions:",
              "- `uniq/step` = distinct (layer,expert) reads per coalesced layer step.",
              "- `uniq/agg-tok` = distinct reads / (N x token positions); the sanity-anchor metric.",
              "- `coalesced%` = duplicate same-layer requests removed by coalescing / raw requests; never a cache hit.",
              "- `hit%` = cache hits / distinct reads; `MB/agg-tok` = misses x record_bytes / (N x tokens) / 1e6.",
              "- `io_ms = MB/agg-tok / ssd_gbps` (+ Nemotron copy ms); perfect = 1000/max(gpu_ms,io_ms),",
              "  none = 1000/(gpu_ms+io_ms). Limiter is the largest of SSD/GPU/copy ms.", ""]
    for model in ("qwen", "nemotron"):
        m = meta.get(model, {})
        lines += [f"## {model}", "",
                  f"Traces: {m.get('sessions')} sessions, token counts {m.get('tokens')}. "
                  f"record_bytes={m.get('record_bytes')} ({m.get('label', '')}).",
                  f"Session state {m.get('session_state_mb')} MB/session (estimate); "
                  f"SSD {m.get('ssd_gbps')} GB/s and GPU {m.get('gpu_ms')} ms/token (estimates).", ""]
        if model == "qwen":
            lines += [f"Measured {QWEN_PINNED} pinned experts/layer are excluded; docs say {QWEN_TOPK_DOCS} routed "
                      "experts/layer (cited). `--topk-scale 1.25` reports the 10-expert case, labeled estimate.", ""]
        else:
            lines += ["Runs share harness prompts, so token-aligned traces are more similar than truly unrelated "
                      "sessions (idea/estimate caveat).", ""]
        lines += ["| " + " | ".join(_TABLE_COLUMNS) + " |",
                  "|" + "|".join(["---"] * len(_TABLE_COLUMNS)) + "|"]
        for r in results:
            if r["model"] == model:
                lines.append(_md_row(r))
        lines.append("")
    lines += ["## Notes and provenance", ""] + [f"- {n}" for n in notes] + ["", "## Sanity anchors", ""]
    lines += [f"- {a}" for a in anchors] + [""]
    return "\n".join(lines)


def _write_outputs(results, meta, notes, anchors, stamp):
    name = f"{stamp}-sim"
    run_dir = RESULTS_ROOT / "multisession" / name
    report = {
        "run": name, "label": "offline simulator; no checkpoint, no SSD",
        "notes": notes, "sanity_anchors": anchors, "trace_meta": meta,
        "results": results,
    }
    jp = output_path("multisession", name, "multisession_sim.json", explicit=run_dir / "multisession_sim.json")
    jp.write_text(json.dumps(report, indent=2) + "\n")
    mp = output_path("multisession", name, "multisession_sim.md", explicit=run_dir / "multisession_sim.md")
    mp.write_text(_markdown(results, meta, notes, anchors))
    return jp, mp


# ------------------------------------------------------------------------------------------------------------
# Parallel sweep support. Workers receive the small traces/steps once via an initializer, then only a cell
# descriptor per task. The reused Belady policy is O(cache_steps x slots), so the sweep is CPU-bound.
# ------------------------------------------------------------------------------------------------------------
_WORKER = {}


def _init_worker(traces, steps_by_cell, settings):
    _WORKER["traces"] = traces
    _WORKER["steps"] = steps_by_cell
    _WORKER["settings"] = settings


def _run_cell(cell):
    model, arrangement, N, scheduler, policy, cap = cell
    s = _WORKER["settings"]
    return simulate(
        model, arrangement, N, scheduler, policy, cap, _WORKER["traces"][model],
        session_state_mb=s["state"][model], record_bytes=s["records"][model],
        ssd_gbps=s["ssd"][model], gpu_ms=s["gpu"][model],
        hit_copy_ms_per_expert=s["hit_copy_ms"], shift=s["shift"],
        shared_offsets=s["shared_offsets"], seed=s["seed"],
        slots_override=s["slots"] if model == "qwen" else None,
        inflight=s["inflight"], topk_scale=s["topk_scale"],
        prebuilt=s["prebuilt"][model, N, scheduler],
        prebuilt_steps=_WORKER["steps"][(model, arrangement, N, scheduler)],
        T=s["T"][model],
    )


# ------------------------------------------------------------------------------------------------------------
# CLI.
# ------------------------------------------------------------------------------------------------------------
def main(argv=None):
    from datetime import datetime

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--qwen-trace", default=QWEN_TRACE)
    p.add_argument("--nemotron-glob", default=NEMOTRON_GLOB)
    p.add_argument("--nemotron-min-tokens", type=int, default=DEFAULT_NEMOTRON_MIN_TOKENS)
    p.add_argument("--models", nargs="+", default=["qwen", "nemotron"], choices=["qwen", "nemotron"])
    p.add_argument("--arrangements", nargs="+", default=arrangement_names())
    p.add_argument("--N", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--schedulers", nargs="+", default=["lockstep", "interleaved"])
    p.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    p.add_argument("--qwen-capacities", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--nemotron-capacities", type=float, nargs="+", default=[2.2, 4.4, 6.6, 8.0])
    p.add_argument("--qwen-session-state-mb", type=float, default=QWEN_SESSION_STATE_MB)
    p.add_argument("--nemotron-session-state-mb", type=float, default=NEMOTRON_SESSION_STATE_MB)
    p.add_argument("--ssd-gbps", type=float, help="override both models' SSD GB/s")
    p.add_argument("--ssd-gbps-qwen", type=float, default=SSD_GBPS["qwen"])
    p.add_argument("--ssd-gbps-nemotron", type=float, default=SSD_GBPS["nemotron"])
    p.add_argument("--gpu-ms", type=float, help="override both models' GPU ms/token")
    p.add_argument("--gpu-ms-qwen", type=float, default=GPU_MS["qwen"])
    p.add_argument("--gpu-ms-nemotron", type=float, default=GPU_MS["nemotron"])
    p.add_argument("--hit-copy-ms-per-expert", type=float, default=HIT_COPY_MS_PER_EXPERT)
    p.add_argument("--shift", type=int, default=DEFAULT_SHIFT)
    p.add_argument("--shared-offsets", type=int, nargs="+", default=list(DEFAULT_SHARED_OFFSETS))
    p.add_argument("--inflight-layers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slots", type=int, default=None,
                   help="pin the Qwen expert pool to N slots (e.g. 349) instead of deriving it from GB")
    p.add_argument("--topk-scale", type=float, default=1.0,
                   help="traffic scale for the 10-vs-8 expert case; 1.25 = 10/8 (estimate)")
    p.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1),
                   help="parallel workers for the sweep (1 = serial)")
    args = p.parse_args(argv)

    ssd = {"qwen": args.ssd_gbps or args.ssd_gbps_qwen, "nemotron": args.ssd_gbps or args.ssd_gbps_nemotron}
    gpu = {"qwen": args.gpu_ms or args.gpu_ms_qwen, "nemotron": args.gpu_ms or args.gpu_ms_nemotron}
    state = {"qwen": args.qwen_session_state_mb, "nemotron": args.nemotron_session_state_mb}
    caps = {"qwen": args.qwen_capacities, "nemotron": args.nemotron_capacities}
    records = {"qwen": QWEN_RECORD_BYTES, "nemotron": NEMOTRON_RECORD_BYTES}

    traces, meta = {}, {}
    for model in args.models:
        if model == "qwen":
            traces[model], meta[model] = load_qwen_traces(args.qwen_trace)
        else:
            traces[model], meta[model] = load_nemotron_traces(args.nemotron_glob, args.nemotron_min_tokens)
        meta[model].update({"session_state_mb": state[model], "ssd_gbps": ssd[model], "gpu_ms": gpu[model]})
        print(f"{model}: {meta[model]['sessions']} traces, token counts {meta[model]['tokens']}, "
              f"record_bytes {meta[model]['record_bytes']} ({meta[model]['label']})")

    prebuilt, steps_by_cell, T_of = {}, {}, {}
    for model in args.models:
        T = max(len(t) for t in traces[model])
        T_of[model] = T
        for N in args.N:
            for scheduler in args.schedulers:
                prebuilt[model, N, scheduler] = build_arrangements(
                    traces[model], model, N, scheduler=scheduler, shift=args.shift,
                    shared_offsets=args.shared_offsets, seed=args.seed, T=T)
                for arrangement in args.arrangements:
                    if arrangement not in prebuilt[model, N, scheduler]:
                        p.error(f"unknown arrangement {arrangement!r}")
                    steps_by_cell[(model, arrangement, N, scheduler)] = _build_steps(
                        prebuilt[model, N, scheduler][arrangement], _layers_of(traces[model]),
                        T, scheduler, args.inflight_layers)

    cells = [(model, arrangement, N, scheduler, policy, cap)
             for model in args.models for arrangement in args.arrangements
             for N in args.N for scheduler in args.schedulers
             for policy in args.policies for cap in caps[model]]
    settings = {"state": state, "records": records, "ssd": ssd, "gpu": gpu,
                "hit_copy_ms": args.hit_copy_ms_per_expert, "shift": args.shift,
                "shared_offsets": args.shared_offsets, "seed": args.seed,
                "slots": args.slots, "inflight": args.inflight_layers,
                "topk_scale": args.topk_scale, "prebuilt": prebuilt, "T": T_of}
    if args.jobs > 1 and len(cells) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker,
                                 initargs=(traces, steps_by_cell, settings)) as pool:
            results = list(pool.map(_run_cell, cells, chunksize=4))
    else:
        _init_worker(traces, steps_by_cell, settings)
        results = [_run_cell(cell) for cell in cells]
    print(f"ran {len(results)} sims")

    def pick(model, arrangement, N, scheduler, policy, cap):
        for r in results:
            if (r["model"] == model and r["arrangement"] == arrangement and r["N"] == N and
                    r["scheduler"] == scheduler and r["policy"] == policy and r["capacity_gb"] == cap):
                return r
        return None

    anchors = []
    for cap in (6.6, 4.4):
        one = pick("nemotron", ARRANGEMENT_B, 1, "lockstep", "lru", cap)
        four = pick("nemotron", ARRANGEMENT_B, 4, "lockstep", "lru", cap)
        if one and four:
            anchors.append(
                f"estimate (anchors): Nemotron B/lru/lockstep {cap:g} GB: uniq/agg-tok "
                f"N=1 {one['unique_experts_per_aggregate_token']:.0f} vs N=4 "
                f"{four['unique_experts_per_aggregate_token']:.0f} (anchor 138 vs ~112); MB/agg-tok "
                f"N=1 {one['mb_per_aggregate_token']:.0f} vs N=4 {four['mb_per_aggregate_token']:.0f} "
                f"(anchor ~190-200 at 6.6 GB, ~280-290 at 4.4 GB). Difference: this model subtracts "
                f"N x session_state_mb from cache GB and coalesces all same-layer requests in lockstep; "
                f"the quick-replay anchor did not state either assumption.")
    # The Qwen anchor holds the production pool fixed at 349 slots for both N=1 and N=3, so the drop
    # isolates sharing (a shrinking capacity would confound it).
    q1 = simulate("qwen", f"{ARRANGEMENT_C_PREFIX}7", 1, "lockstep", "lru", 4.0, traces["qwen"],
                  session_state_mb=state["qwen"], slots_override=QWEN_SLOTS)
    q3 = simulate("qwen", f"{ARRANGEMENT_C_PREFIX}7", 3, "lockstep", "lru", 4.0, traces["qwen"],
                  session_state_mb=state["qwen"], slots_override=QWEN_SLOTS)
    anchors.append(
        f"estimate (anchor): Qwen C(k=7)/lru with the fixed 349-slot pool (cited production size): "
        f"MB/agg-tok N=1 {q1['mb_per_aggregate_token']/1e3:.2f} GB vs N=3 "
        f"{q3['mb_per_aggregate_token']/1e3:.2f} GB (anchor ~1.17 vs ~0.9 GB).")
    anchors.append("cited: cross-conversation same-layer share 0.0627 from "
                   "results/flashnext/20261114-132303-route-overlap-manual/route-overlap.json.")
    anchors.append("idea: A is an upper-ish bound (prompts are similar); D is a lower bound; B is the main "
                   "unrelated-sub-agent case; C is the shared-context proxy.")

    notes = [
        "All throughput figures are estimates/projections, not measurements; no checkpoint or SSD was used.",
        "measured: Qwen record_bytes=3211264, ~8 streamed experts/layer (8 pinned experts excluded from trace).",
        "cited: docs describe 10 routed experts/layer; --topk-scale 1.25 reports that case as an estimate.",
        "estimate: Nemotron record_bytes=5.61*1024*1024 per expert; session state 900/400 MB; "
        "SSD 2.9/3.3 GB/s; GPU 100/38 ms; copy 0.21 ms/expert (from cited 1.28 ms / 6 cached experts).",
        "idea: coalesced duplicates are not cache hits; lockstep coalesces same-layer requests, "
        "interleaved uses random layer phases with an in-flight window of --inflight-layers.",
        "The scheduler and arrangements are modelling constructs; they are not observed runtime schedules.",
        "Caveat: cells where N x session_state_mb exhausts cache_gb clamp to a 1-slot pool; the reused Belady "
        "policy's eviction rule can then underperform ARC (observed on Qwen C7/interleaved/N=4/cap 1-2 GB), "
        "so Belady is an upper bound only for non-degenerate capacities. The tiny-fixture unit test checks the "
        "normal case.",
    ]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jp, mp = _write_outputs(results, meta, notes, anchors, stamp)
    print(f"wrote {jp}")
    print(f"wrote {mp}")
    for a in anchors:
        print("  " + a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
