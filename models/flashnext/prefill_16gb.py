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

def apply(environment):
    for name, value in PREFILL_16GB.items():
        environment.setdefault(name, value)
    return environment
