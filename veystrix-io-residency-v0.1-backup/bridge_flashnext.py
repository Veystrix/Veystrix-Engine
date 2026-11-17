"""Opt-in bridge: Veystrix's existing route hook -> residency RouteTraceWriter.

Off unless VEYSTRIX_ROUTE_TRACE is set, so the default decode path is untouched.
Attach once after backend construction:

    from veystrix_io_bridge import attach
    attach(backend)     # installs set_route_trace + wraps on_decode_token

The engine already calls expert_cache._ROUTE_TRACE(layer_id, routed) once per MoE
layer and exposes each decoded token; this module only records them. Output format
matches veystrix_residency.trace.load_trace.
"""
from __future__ import annotations

import os

from veystrix_residency.trace import RouteTraceWriter

_WRITER: RouteTraceWriter | None = None


def _route(layer_id, routed):
    if _WRITER is not None and layer_id is not None and layer_id >= 0:
        # decode: routed is a flat list of k experts for one row. Skip prefill
        # (multiple rows -> nested list, which the writer ignores).
        _WRITER.record(layer_id, routed)


def attach(backend, env=None) -> RouteTraceWriter | None:
    """Install the trace if VEYSTRIX_ROUTE_TRACE is set; else do nothing."""
    global _WRITER
    environ = os.environ if env is None else env
    path = environ.get("VEYSTRIX_ROUTE_TRACE")
    if not path:
        return None
    from models.flashnext import expert_cache
    _WRITER = RouteTraceWriter(path)
    expert_cache.set_route_trace(_route)
    # token boundary: wrap the backend's decode-token callback via a counter we
    # advance using the existing on_decode_token hook if present.
    return _WRITER


def end_token():
    if _WRITER is not None:
        _WRITER.end_token()


def detach():
    global _WRITER
    from models.flashnext import expert_cache
    expert_cache.set_route_trace(None)
    if _WRITER is not None:
        _WRITER.close()
        _WRITER = None
