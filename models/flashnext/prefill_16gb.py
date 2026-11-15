"""Reproducible low-transient Flash-Next prefill profile."""
from __future__ import annotations


PREFILL_16GB = {
    "FLASHNEXT_PREFILL_CHUNK_TOKENS": "128",
    "FLASHNEXT_PREFILL_ORDER": "chunk",
    "FLASHNEXT_PREFILL_RING_EXPERTS": "256",
    "FLASHNEXT_PREFILL_RING_SCRATCH": "1",
    "FLASHNEXT_PREFILL_RING_MIN_TOKENS": "128",
    "FLASHNEXT_UNCACHED_EXPERT_READS": "prefill",
    "FLASHNEXT_EVAL_LINEAR_STATE": "1",
    "FLASHNEXT_PREFILL_CACHE_LIMIT_MB": "512",
}

# Experimental and unmeasured on the real checkpoint: layer-major windows that read only the experts their chunks route
# (``FLASHNEXT_PREFILL_ROUTE_UNION``, models/flashnext/layer_major.py). Unit-tested bit-identical to chunk-major order on a
# tiny model only; the prefix-cache fingerprint carries ``/union-v1`` until a real-checkpoint digest gate has passed.
PREFILL_16GB_UNION = {
    **PREFILL_16GB,
    "FLASHNEXT_PREFILL_ORDER": "layer",
    "FLASHNEXT_PREFILL_ROUTE_UNION": "1",
    "FLASHNEXT_LAYER_MAJOR_MIN_TOKENS": "256",
    "FLASHNEXT_PREFILL_WINDOW_TOKENS": "1024",
    "FLASHNEXT_PREFILL_RING_EXPERTS": "384",
}


def apply(environment):
    for name, value in PREFILL_16GB.items():
        environment.setdefault(name, value)
    return environment
