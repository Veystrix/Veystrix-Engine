"""Per-prefill expert read statistics (opt-in diagnostic: ``FLASHNEXT_PREFILL_READ_STATS=1``, default off).

Why. Chunk-major prefill re-reads a layer's experts for every chunk. How much of the SSD traffic of a real prompt is such
re-reading, and what the route-union layer-major order (``layer_major.RouteUnion``) saves, is a number to measure, not to
guess. This recorder counts, for every streamed MoE read of one prefill:

* how many experts were read, and how many of them the SAME prefill had already read for that layer (re-reads);
* whether the prefill ring staged them or the read fell back to the ordinary path, and why (the ring's own counters);
* the bytes.

Off, every hook is one list lookup (``ACTIVE[0] is None``); nothing is allocated, nothing is synchronised. On, a hook only
does host-side set arithmetic on lists the read path already built, and one ``PrefillRing.stats()`` snapshot per read.
The recorder never touches the GPU and never raises into the read path.

Hooks: ``StreamingSwitchGLU._one_pass`` (chunk-major reads, multi-row calls only), ``layer_major.ExpertWindow.weights_for``
(layer-major, all experts of a layer) and ``layer_major.RouteUnion.stage`` (route-union groups). ``begin``/``end`` bracket one
prefill (``prefill._prefill_target_checked``); a nested prefill does not start a second recording.
"""
from __future__ import annotations

import os

FLAG = "FLASHNEXT_PREFILL_READ_STATS"
ACTIVE = [None]                 # the live ``Recording`` or None: the only thing a hot-path hook looks at
_LAST = [None]                  # summary of the last completed prefill
_ORDER_RANK = {"chunk": 0, "layer": 1, "layer-union": 2}
_REASON_KEYS = ("fallback_capacity", "fallback_busy", "fallback_error", "fallback_disabled", "fallback_memory")


def enabled() -> bool:
    return os.environ.get(FLAG, "0") == "1"


def expert_bytes(glu) -> int:
    """Bytes of one expert (every projection, weight + scales + biases) of a streamed layer, or 0 when unknown."""
    try:
        store = glu.gate_proj.cache.store
        prefix = glu.gate_proj.cache.prefix.rsplit(".", 1)[0]
        return int(sum(store.refs[f"{prefix}.{p}.{k}"].row_bytes for p in ("gate_proj", "up_proj", "down_proj")
                       for k in ("weight", "scales", "biases")))
    except Exception:
        return 0


def mark(ring):
    """The ring's fallback counters before a read, so ``note`` can tell why it fell back. None without a ring."""
    if ring is None:
        return None
    try:
        return ring.stats()
    except Exception:
        return None


def _reason(ring, before) -> str:
    if ring is None:
        return "no_ring"
    try:
        now = ring.stats()
    except Exception:
        return "unknown"
    if before is not None:
        for key in _REASON_KEYS:
            if now.get(key, 0) > before.get(key, 0):
                return key[len("fallback_"):]
    if now.get("disabled"):
        return "disabled"
    return "unknown"


class Recording:
    def __init__(self, prompt_tokens: int):
        self.prompt_tokens = int(prompt_tokens)
        self.order_rank = -1
        self.seen = {}                  # layer -> experts this prefill has already read for it
        self.calls = []                 # [layer, read, reread, bytes, via_ring, reason]
        self._bytes = {}                # layer -> bytes of one expert

    def note(self, order: str, glu, wanted, via_ring: bool, ring=None, before=None) -> None:
        try:
            layer = int(getattr(glu, "layer_id", -1))
            experts = set(int(e) for e in wanted)
            seen = self.seen.setdefault(layer, set())
            reread = len(experts & seen)
            seen |= experts
            if layer not in self._bytes:
                self._bytes[layer] = expert_bytes(glu)
            reason = None if via_ring else _reason(ring, before)
            self.calls.append([layer, len(experts), reread, len(experts) * self._bytes[layer], bool(via_ring), reason])
            self.order_rank = max(self.order_rank, _ORDER_RANK.get(order, 0))
        except Exception:
            pass                        # a diagnostic must never break a read

    def summary(self) -> dict:
        order = next((name for name, rank in _ORDER_RANK.items() if rank == self.order_rank), "chunk")
        reads = sum(call[1] for call in self.calls)
        rereads = sum(call[2] for call in self.calls)
        reasons = {}
        for call in self.calls:
            if not call[4]:
                reasons[call[5]] = reasons.get(call[5], 0) + 1
        per_layer = {}
        for layer, read, reread, _bytes, _ring, _reason_ in self.calls:
            row = per_layer.setdefault(layer, {"layer": layer, "calls": 0, "read": 0, "reread": 0, "union": 0})
            row["calls"] += 1
            row["read"] += read
            row["reread"] += reread
        for layer, row in per_layer.items():
            row["union"] = len(self.seen.get(layer, ()))
        return {
            "order": order,
            "prompt_tokens": self.prompt_tokens,
            "layers": len(per_layer),
            "calls": len(self.calls),
            "experts_read": reads,
            "experts_reread": rereads,
            "reread_share": round(rereads / reads, 4) if reads else 0.0,
            "bytes_read": sum(call[3] for call in self.calls),
            "ring_staged_calls": sum(1 for call in self.calls if call[4]),
            "fallback_calls": sum(1 for call in self.calls if not call[4]),
            "fallback_reasons": reasons,
            "max_set": max((call[1] for call in self.calls), default=0),
            "per_layer": [per_layer[layer] for layer in sorted(per_layer)],
        }


def begin(prompt_tokens: int):
    """Start recording one prefill. Returns a token for ``end``, or None (disabled, or a prefill is already recording)."""
    if ACTIVE[0] is not None or not enabled():
        return None
    recording = Recording(prompt_tokens)
    ACTIVE[0] = recording
    return recording


def end(token, ok: bool = True) -> None:
    """Stop recording. A completed prefill (``ok``) becomes ``summary()``; a failed one is dropped. Never raises."""
    if token is None:
        return
    try:
        if ACTIVE[0] is token:
            ACTIVE[0] = None
        if ok:
            # A completed prefill that read nothing (a short prompt, a fully cached one) leaves None, never a stale summary.
            _LAST[0] = token.summary() if token.calls else None
    except Exception:
        ACTIVE[0] = None


def summary():
    """JSON-serialisable summary of the last completed prefill, or None when nothing was recorded."""
    last = _LAST[0]
    if last is None:
        return None
    return {**last, "fallback_reasons": dict(last["fallback_reasons"]), "per_layer": [dict(r) for r in last["per_layer"]]}


def reset() -> None:
    """Forget the last summary and any live recording (tests and benchmarks that run several prefills)."""
    _LAST[0] = None
    ACTIVE[0] = None
