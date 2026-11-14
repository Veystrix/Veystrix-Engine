"""Host-side decode timing and expert routes for one Nemotron request."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mlx.core as mx

from models.flashnext.decode_profile import summarize

FLAG = "VEYSTRIX_NEMOTRON_DECODE_PROFILE_DIR"
MAX_LAYER_CALLS = 100_000


class DecodeProfile:
    def __init__(self, directory, pool):
        self.directory = Path(directory)
        self.events = []
        self.layers = []
        self.verifier_layer_calls = 0
        self.omitted_layer_calls = 0
        self.sample_end = None
        self.start = time.perf_counter()
        self.initial = pool.stats()
        # Freed buffers MLX keeps for reuse. They are resident but not "active", so nothing else reports them.
        self.cache_begin = self.cache_peak = mx.get_cache_memory()
        self.active_begin = mx.get_active_memory()

    def record(self, layer, routes, started, routed_at, demand_done):
        if len(self.layers) == MAX_LAYER_CALLS:
            self.omitted_layer_calls += 1
            return
        self.events.append((routed_at, layer, list(map(int, routes))))
        self.cache_peak = max(self.cache_peak, mx.get_cache_memory())
        self.sample_end = demand_done
        self.layers.append({
            "layer": layer,
            "route_sync_and_copy_ms": (routed_at - started) * 1000,
            "expert_dispatch_and_wait_ms": (demand_done - routed_at) * 1000,
        })

    def finish(self, pool):
        end = time.perf_counter()
        report = summarize(self.events, self.start, self.sample_end if self.omitted_layer_calls else end)
        report["layer_calls"] = self.layers
        report["layer_calls_omitted"] = self.omitted_layer_calls
        report["verifier_layer_calls_excluded"] = self.verifier_layer_calls
        report["decode_elapsed_s"] = end - self.start
        report["mlx_memory"] = {"cache_bytes_begin": self.cache_begin, "cache_bytes_peak": self.cache_peak,
                                "cache_bytes_end": mx.get_cache_memory(), "active_bytes_begin": self.active_begin,
                                "active_bytes_end": mx.get_active_memory()}
        counters = pool.stats()
        report["pool_delta"] = {name: counters[name] - self.initial[name] for name in (
            "hits", "misses", "bytes_read", "prefetch_issued", "prefetch_used", "prefetch_wasted",
            "read_seconds", "wait_seconds",
        )}
        report["pool_delta"].update({name: counters[name] - self.initial[name]
                                     for name in ("view_hits", "resident_hits", "mapped_experts") if name in counters})
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / f"nemotron-decode-{time.time_ns()}.json").open("x") as stream:
                json.dump(report, stream)
        except OSError:
            pass  # diagnostics must not change the generation result


def begin(pool):
    directory = os.environ.get(FLAG)
    return DecodeProfile(directory, pool) if directory else None
