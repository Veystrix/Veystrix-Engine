"""Layer-major (windowed) out-of-core prefill. Opt-in: ``FLASHNEXT_PREFILL_ORDER=layer``; the default ``chunk`` is unchanged.

Why. A MoE prompt chunk of 2,048 tokens routes about 400 of a layer's 512 experts, so every chunk re-reads about 60 GB of the
checkpoint (docs/flashnext/architecture-overhaul.md has the arithmetic). The chunk-major loop

    chunk 1 -> layers 0..47, chunk 2 -> layers 0..47, ...

therefore pays the whole SSD pass once per chunk, and its per-layer router sync leaves the GPU idle while the reads run. Here a
*window* of several chunks runs layer-major,

    layer 0 -> chunks 1..n, layer 1 -> chunks 1..n, ...

so a layer's experts are read ONCE per window and reused by every chunk of the window. The window is bounded (default 8,192
tokens): the activation it keeps resident between layers is ``tokens x hc_count x hidden`` bf16 (20 KB per token, 164 MB at
8K), and a window end is a safe cut: every layer's cache is at the same position, so a snapshot or an interruption keeps
everything before it.

Exactness. Layer ``l``'s output for chunk ``c`` depends only on layer ``l-1``'s output for chunk ``c`` and layer ``l``'s own
cache after chunk ``c-1``. Both orders honour exactly those dependencies, with the same chunk shapes, masks and position ids, so
the per-layer arithmetic is the same operations on the same inputs. The one difference is the expert operand: the window
hands ``gather_qmm`` all of the layer's experts and uses the router's indices directly, where the chunk-major path hands it
only that chunk's experts and remapped indices. ``gather_qmm`` computes each (token, expert) product from that expert's own
rows, so the result does not depend on which other experts sit in the operand; tests check this bit for bit on a small model, and
the real checkpoint must pass the token-digest gate before it is promoted. Until it does, the cache fingerprint carries
``identity()`` so layer-major snapshots never mix with chunk-major ones.

Route union (``FLASHNEXT_PREFILL_ROUTE_UNION=1``, opt-in, layer-major only). The all-experts window stages every expert of a
layer (about 1.6 GB of ring at 512 experts) although a few-hundred-token window routes far fewer, and it re-reads nothing
only because it reads everything. With the flag each layer runs in two phases: A) the first half of the layer (attention and
the two hyper-connections, see ``layer_first_half``) for every chunk of the window, then the chunk's router indices are taken
through the real MoE block in a probe mode that reads nothing; B) the union of the chunks' routed experts is staged once (in
greedy consecutive groups when it does not fit the ring half) and every chunk runs the real MoE block on that staged subset
with its indices remapped to staged positions. Every (token, expert) product is the same ``gather_qmm`` on the same expert rows
as in chunk-major order, so the result is bit-identical to it; ``identity()`` still carries ``/union-v1`` until a real-checkpoint
digest gate has passed. A layer the helper cannot split safely runs the all-experts path instead and the refusal is counted.

What it does not do. A layer-major window cannot snapshot at a chunk edge inside it (the later layers have not reached it), so
the cadence of resumable cuts is the window, not the chunk; an interruption loses at most one window of work. Windows still end
at the system-prompt boundary and prefer message anchors. Adaptive chunk re-planning runs between windows only.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from . import prefill_reads

FLAG_ORDER = "FLASHNEXT_PREFILL_ORDER"                  # "chunk" (default) | "layer"
FLAG_WINDOW = "FLASHNEXT_PREFILL_WINDOW_TOKENS"         # tokens per layer-major window
FLAG_HYBRID = "FLASHNEXT_PREFILL_HYBRID_SMALL_WINDOWS"  # opt-in: reference path for small windows
FLAG_PREFETCH = "FLASHNEXT_PREFILL_LAYER_PREFETCH"      # opt-in: read layer l+1's experts while layer l computes
FLAG_INNER_SNAPSHOTS = "FLASHNEXT_PREFILL_INNER_SNAPSHOTS"  # "1" (default): snapshot chunk edges inside a window; "0": cut windows there
FLAG_TAIL_CHUNK = "FLASHNEXT_PREFILL_TAIL_CHUNK"        # "1" (default): a trailing tiny chunk runs chunk-major after the last window
FLAG_MIN_TOKENS = "FLASHNEXT_LAYER_MAJOR_MIN_TOKENS"    # smaller uncached prefills use the chunk-major path
FLAG_ROUTE_UNION = "FLASHNEXT_PREFILL_ROUTE_UNION"      # opt-in: each streamed layer reads only the experts its window routes
UNION_SUFFIX = "/union-v1"
DEFAULT_MIN_TOKENS = 4096
ORDERS = ("chunk", "layer")
PLAN_VERSION = "layer-major-v1"
DEFAULT_WINDOW_TOKENS = 8192
MIN_WINDOW_TOKENS = 1024
MAX_WINDOW_TOKENS = 32768
HYBRID_SMALL_WINDOW_TOKENS = 2048
# A chunk of at most this many tokens takes the decode-style MoE path (``expert_cache`` uses the window only above 8 rows),
# which touches the 2 GB decode pool as a Metal buffer for the first time: measured +1.9 GB of MLX memory and +0.4 s at layer 1,
# held for the rest of the window (results/flashnext/20261110-005029-prefill-profile-manual). It is run after the window instead.
TINY_CHUNK_TOKENS = 8


def order() -> str:
    value = (os.environ.get(FLAG_ORDER, "") or "chunk").strip().lower()
    if value not in ORDERS:
        raise ValueError(f"{FLAG_ORDER} must be one of {', '.join(ORDERS)}")
    return value


def enabled() -> bool:
    return order() == "layer"


def window_tokens() -> int:
    raw = os.environ.get(FLAG_WINDOW, "") or str(DEFAULT_WINDOW_TOKENS)
    try:
        value = int(float(raw))
    except ValueError as error:
        raise ValueError(f"{FLAG_WINDOW} must be a number of tokens") from error
    if not MIN_WINDOW_TOKENS <= value <= MAX_WINDOW_TOKENS:
        raise ValueError(f"{FLAG_WINDOW} must be between {MIN_WINDOW_TOKENS} and {MAX_WINDOW_TOKENS}")
    return value


def identity() -> str:
    """The execution-plan identity for the cache fingerprint: empty for chunk-major (existing keys unchanged)."""
    if not enabled():
        return ""
    hybrid = "/hybrid-small-v1" if hybrid_enabled() else ""
    minimum = min_tokens()
    threshold = f"/min{minimum}" if minimum != DEFAULT_MIN_TOKENS else ""
    union = UNION_SUFFIX if route_union_enabled() else ""
    return f"{PLAN_VERSION}/w{window_tokens()}{hybrid}{threshold}{union}"


def prefetch_enabled() -> bool:
    value = os.environ.get(FLAG_PREFETCH, "0") or "0"
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG_PREFETCH} must be 0 or 1")
    return value == "1"


def route_union_enabled() -> bool:
    """Route-union execution inside layer-major windows (default off). Layer prefetch is ignored while it is on."""
    value = os.environ.get(FLAG_ROUTE_UNION, "0") or "0"
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG_ROUTE_UNION} must be 0 or 1")
    return value == "1"


def tail_chunk_enabled() -> bool:
    value = os.environ.get(FLAG_TAIL_CHUNK, "1") or "1"
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG_TAIL_CHUNK} must be 0 or 1")
    return value == "1"


def inner_snapshots_enabled() -> bool:
    """Snapshot a boundary that falls on a chunk edge inside a window instead of ending the window there.

    Ending a window at the system-prompt boundary makes every expert be read once more (a 75 GB pass at 8K). Each layer's
    cache is instead frozen the moment that layer finishes the chunk ending at the boundary, and the snapshot is assembled
    after the window (``prefix_cache.freeze_cache``). The stored state is the same one the chunk-major order stores there."""
    value = os.environ.get(FLAG_INNER_SNAPSHOTS, "1") or "1"
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG_INNER_SNAPSHOTS} must be 0 or 1")
    return value == "1"


def peel_tail(group):
    """Split a final window into (chunks run layer-major, trailing tiny chunk or None).

    Only a window of more than one chunk whose last chunk has at most ``TINY_CHUNK_TOKENS`` tokens is split. The tail then runs
    through the reference chunk-major path after every layer of the window has finished, so it sees exactly the cache state the
    reference gives it and takes the same executor: results are unchanged."""
    group = list(group)
    if len(group) > 1 and tail_chunk_enabled() and group[-1][1] - group[-1][0] <= TINY_CHUNK_TOKENS:
        return group[:-1], group[-1]
    return group, None


def min_tokens() -> int:
    """Uncached tokens below which a prefill stays chunk-major.

    A layer-major window reads every layer's 512 experts once whatever its size. A 55-token follow-up prefill measured
    about 96 s that way (2026-11-10, DeepSeek Harness effort-turn warm-up) against a few seconds chunk-major. The 8K
    gain (1.70-1.84x) is real, so only prefills large enough to amortize the sweep use it. Below 4096 is unmeasured."""
    value = os.environ.get(FLAG_MIN_TOKENS, "") or str(DEFAULT_MIN_TOKENS)
    try:
        tokens = int(value)
    except ValueError:
        raise ValueError(f"{FLAG_MIN_TOKENS} must be an integer") from None
    if tokens < 0:
        raise ValueError(f"{FLAG_MIN_TOKENS} must not be negative")
    return tokens


def hybrid_enabled() -> bool:
    value = os.environ.get(FLAG_HYBRID, "0") or "0"
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG_HYBRID} must be 0 or 1")
    return value == "1"


# ---------------------------------------------------------------------------------------------------------------- planning

def plan_windows(spans, offset: int, window: int, required=(), anchors=()):
    """Group chunk ``spans`` (relative ``(start, end)`` pairs covering the suffix) into windows of whole chunks.

    A window holds at most ``window`` tokens unless a single chunk is larger. It ends at a ``required`` position (the
    system-prompt boundary) when one falls inside it, otherwise at the last anchor in its second half, otherwise at the last
    chunk edge that fits. Returns a list of span lists; positions are absolute only for ``required`` and ``anchors``.
    """
    spans = [(int(a), int(b)) for a, b in spans]
    if not spans:
        return []
    required = sorted({int(p) for p in required})
    anchors = sorted({int(p) for p in anchors})
    windows, index = [], 0
    while index < len(spans):
        first = spans[index][0]
        limit = first + int(window)
        stop = index
        while stop + 1 < len(spans) and spans[stop + 1][1] <= limit:
            stop += 1                                   # spans[index..stop] fit (the first chunk always does)
        forced = next((k for k in range(index, stop + 1) if offset + spans[k][1] in required), None)
        if forced is not None:
            stop = forced
        elif stop + 1 < len(spans):
            half = first + int(window) // 2
            nice = [k for k in range(index, stop + 1) if offset + spans[k][1] in anchors and spans[k][1] > half]
            if nice:
                stop = nice[-1]
        windows.append(spans[index:stop + 1])
        index = stop + 1
    return windows


# ---------------------------------------------------------------------------------------------------------- cost model

def expert_pass_bytes(experts: int, layers: int, expert_bytes: int) -> int:
    """Bytes of one full pass over every expert of ``layers`` layers."""
    return int(experts) * int(layers) * int(expert_bytes)


def chunk_major_bytes(tokens: int, chunk: int, layers: int, experts_read_per_layer: float, expert_bytes: int) -> int:
    """Physical expert bytes of the chunk-major order: every chunk reads its routed experts in every layer."""
    chunks = -(-int(tokens) // int(chunk))
    return int(chunks * layers * experts_read_per_layer * expert_bytes)


def layer_major_bytes(tokens: int, window: int, layers: int, experts: int, expert_bytes: int) -> int:
    """Physical expert bytes of the layer-major order: every window reads each layer's full expert set once."""
    windows = -(-int(tokens) // int(window))
    return windows * expert_pass_bytes(experts, layers, expert_bytes)


def activation_bytes(tokens: int, hc_count: int, hidden: int, dtype_bytes: int = 2) -> int:
    """Resident hidden states of one window between layers."""
    return int(tokens) * int(hc_count) * int(hidden) * int(dtype_bytes)


def predicted_seconds(tokens: int, compute_tok_s: float, io_bytes: int, io_gb_s: float, overlapped: bool) -> float:
    """Wall time when compute and SSD reads run at the given rates, serialized or perfectly overlapped."""
    compute = tokens / compute_tok_s
    io = io_bytes / (io_gb_s * 1e9)
    return max(compute, io) if overlapped else compute + io


# ----------------------------------------------------------------------------------------------------- expert window

class ExpertWindow:
    """Supplies one layer's complete expert set to every chunk of a window (see ``expert_cache._LAYER_WINDOW``).

    ``ring`` is the prefill ring (``prefill_ring.PrefillRing``) or None. With a ring the layer's experts are read into one
    half with direct I/O and handed out as zero-copy views; without one they are read with the layer's own path (a copy per
    layer). Either way ``end_layer`` must be called only after ``mx.eval`` has returned for every graph that used them: the
    ring's hazard contract (a half is not recycled while the GPU may read it) depends on it.
    """

    def __init__(self, ring=None):
        self.ring = ring
        self._layer = None
        self._weights = None
        self._prefetched = {}                   # layer_id -> (projections, pending reads); at most two layers ahead
        self.stats = {"layers": 0, "bytes": 0, "ring_layers": 0, "fallback_layers": 0, "read_seconds": 0.0,
                      "prefetch_layers": 0, "prefetch_used": 0}
        self._staged = {}                       # layer_id -> prefill_ring.Staging begun on the second ring half
        self.layer_reads = []                   # trace only: [layer_id, read seconds, MLX active MB before, after]
        self.route_union = None                 # a ``RouteUnion`` while the route-union driver owns this layer's MoE calls

    def prefetch(self, glu) -> bool:
        """Queue the next layer's complete expert read on the worker threads without waiting for it (opt-in).

        Fallback path only: each layer's destination is a fresh array, so the reads can never write into memory the layer
        being computed still uses; with a ring (one half) or the pool-reuse path this does nothing. One layer ahead at most:
        about one more layer of operands (1.5 GB at 512 experts) is resident while it overlaps. The bytes read and their order
        are those of the blocking read, so results are identical; only when the wait happens changes. At the start of layer
        ``l`` the window holds layer ``l`` (queued earlier, consumed by its MoE call) and layer ``l+1`` (queued now): never
        more than two, never the same layer twice.
        """
        if route_union_enabled():
            return False                        # a layer's union of routes is unknown a layer ahead: nothing to prefetch
        if glu is not None and self.ring is not None and prefetch_enabled():
            return self._prefetch_ring(glu)
        if (glu is None or self.ring is not None or not prefetch_enabled() or getattr(glu, "layer_id", -1) < 0
                or glu.layer_id in self._prefetched or len(self._prefetched) >= 2):
            return False
        from models.flashnext import expert_cache, expert_io_stats

        pool = getattr(glu.gate_proj.cache.store, "_flashnext_expert_pool", None)
        if expert_io_stats.pool_reuse_enabled() and pool is not None and pool._slot_of:
            return False                        # the pool-reuse read path owns this layer
        projections = (glu.gate_proj, glu.up_proj, glu.down_proj)
        experts = list(range(glu.gate_proj.num_experts))
        self._prefetched[glu.layer_id] = (projections, expert_cache.submit_projection_tasks(projections, experts))
        self.stats["prefetch_layers"] += 1
        return True

    def _prefetch_ring(self, glu) -> bool:
        """Begin the next layer's read on a free ring half (two-half ring only; at most two begun stagings).

        The half is FILLING until ``weights_for`` consumes it, so the layer being computed never shares memory with the
        reads. The bytes, their order and the destination layout are those of the blocking ``stage``: results are identical."""
        layer = getattr(glu, "layer_id", -1)
        ring = self.ring
        # At the start of layer l+1 its own staging is still waiting to be consumed, and layer l+2 may begin on the half layer
        # l just released: at most two begun stagings (never more than the ring has halves; ``begin_fill`` refuses otherwise).
        if layer < 0 or layer in self._staged or len(self._staged) >= 2 or not getattr(ring, "overlap_capable", False):
            return False
        experts = list(range(glu.gate_proj.num_experts))
        staging = ring.stage_begin(glu.gate_proj.cache.store, glu.gate_proj.cache.prefix.rsplit(".", 1)[0], experts)
        if staging is None:
            return False
        self._staged[layer] = staging
        self.stats["prefetch_layers"] += 1
        return True

    def drain(self) -> None:
        """Wait for any queued prefetch to stop writing, then drop it. Safe to call at any time and on every exit path."""
        staged, self._staged = getattr(self, "_staged", None) or {}, {}
        for staging in staged.values():
            self.ring.cancel_stage(staging)     # waits for its workers, then releases the half; never raises
        held, self._prefetched = self._prefetched, {}
        if not held:
            return
        from models.flashnext import expert_cache

        def futures(item):
            if hasattr(item, "futures"):
                yield from item.futures
            elif isinstance(item, (list, tuple)):
                for sub in item:
                    yield from futures(sub)
            else:
                yield item

        for future in futures([pending for _, pending in held.values()]):
            try:
                expert_cache._resolve_future(future, None)
            except Exception as error:
                # A consumed layer's failure is raised by the real wait. This one belongs to a layer that was never consumed
                # (the window ended or aborted first): quiescence is all that matters, but it is counted and logged.
                self.stats["prefetch_drain_errors"] = self.stats.get("prefetch_drain_errors", 0) + 1
                logging.getLogger("flashnext.layer_major").warning("orphaned layer prefetch failed: %r", error)

    def weights_for(self, glu):
        layer = glu.layer_id
        if self._layer == layer and self._weights is not None:
            return self._weights
        began = time.perf_counter()
        try:
            active_before = round(mx.get_active_memory() / 1e6)
        except Exception:
            active_before = None
        experts = list(range(glu.gate_proj.num_experts))
        weights = None
        recording = prefill_reads.ACTIVE[0]                   # FLASHNEXT_PREFILL_READ_STATS; None when off
        stats_mark = prefill_reads.mark(self.ring) if recording is not None else None
        via_ring = False
        if layer in self._prefetched:
            from models.flashnext import expert_cache

            projections, pending = self._prefetched.pop(layer)
            raw = expert_cache._await_projection_tasks(pending)
            weights = [projection.cache.to_mx(chunks) for projection, chunks in zip(projections, raw)]
            self.stats["fallback_layers"] += 1
            self.stats["prefetch_used"] += 1
        elif self.ring is not None:
            staging = self._staged.pop(layer, None)
            if staging is not None:
                weights = self.ring.stage_finish(staging)           # the read began a layer earlier; waits only for its tail
                if weights is not None:
                    self.stats["prefetch_used"] += 1
            if weights is None:
                weights = self.ring.stage(glu.gate_proj.cache.store, glu.gate_proj.cache.prefix.rsplit(".", 1)[0], experts)
            if weights is not None:
                self.stats["ring_layers"] += 1
                via_ring = True
        if weights is None:
            weights = glu._read_weights((glu.gate_proj, glu.up_proj, glu.down_proj), experts)
            self.stats["fallback_layers"] += 1
        self._layer, self._weights = layer, weights
        if recording is not None:
            recording.note("layer", glu, experts, via_ring, self.ring, stats_mark)
        self.stats["layers"] += 1
        self.stats["read_seconds"] += time.perf_counter() - began
        try:
            self.layer_reads.append([layer, round(time.perf_counter() - began, 3), active_before,
                                     round(mx.get_active_memory() / 1e6)])
        except Exception:
            pass
        try:
            store = glu.gate_proj.cache.store
            prefix = glu.gate_proj.cache.prefix.rsplit(".", 1)[0]
            self.stats["bytes"] += len(experts) * sum(
                store.refs[f"{prefix}.{p}.{k}"].row_bytes for p in ("gate_proj", "up_proj", "down_proj")
                for k in ("weight", "scales", "biases"))
        except Exception:
            pass
        return weights

    def end_layer(self) -> None:
        """Drop this layer's operands. Call after the GPU work that used them has completed."""
        self._layer = self._weights = None
        if self.ring is not None:
            self.ring._retire()

    def abort_layer(self) -> None:
        """Fence before releasing a layer whose evaluation did not complete.

        A failed dispatch or evaluation may still have GPU work in flight.
        Keep the operands alive until the ring's normal abort fence succeeds;
        the surrounding ring session retains its pool if fencing fails.
        """
        self.drain()                            # no read may still be writing when the operands are released
        union, self.route_union = self.route_union, None    # stop intercepting MoE calls; ``union`` keeps its operands alive
        if self.ring is not None:
            self.ring.abort()
        self._layer = self._weights = None
        del union

    def begin_union(self, layer, glu):
        """A ``RouteUnion`` for ``layer`` or ``UnionRefused`` (the caller then runs the all-experts path for this layer)."""
        check_layer(layer)
        if getattr(layer.mlp, "switch_mlp", None) is not glu:
            raise UnionRefused("not-streamed", "the layer's switch_mlp is not the streamed block")
        return RouteUnion(self, glu)


# ------------------------------------------------------------------------------------------------------------ route union

class UnionRefused(Exception):
    """Route-union cannot run this layer; the all-experts layer-major path takes it. ``reason`` is a short counter key."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


# What the installed ``Qwen4ExpDecoderLayer.__call__`` uses (checked against its code object, so a changed or patched layer
# is refused instead of being split by a stale copy of its body): attribute names and local variable names, in order.
_LAYER_CALL_NAMES = frozenset({"attn_hyper_connection", "is_linear", "linear_attn", "mlp", "mlp_hyper_connection", "ple",
                               "reshape", "self_attn", "shape"})
_LAYER_CALL_VARNAMES = ("self", "hidden_states", "input_ids", "mask", "cache", "position_ids", "mixed", "hyper_input",
                        "injection_weights", "branch", "injection")


def _layer_call(cls):
    """The function that runs a layer chunk of more than one token: the class's ``__call__``, seen through compile_glue.

    ``compile_glue.apply()`` (every real load) replaces ``__call__`` with a wrapper that handles only one-token decode steps
    itself and delegates every longer input to the original, which is what a prefill chunk of more than 8 rows reaches."""
    call = cls.__call__
    glue = sys.modules.get("models.flashnext.compile_glue")
    if glue is not None and call is getattr(glue, "_decoder_call", None):
        return glue._ORIGINAL.get("decoder")
    return call


def check_layer(layer) -> None:
    """Raise ``UnionRefused`` unless ``layer`` is a stock ``Qwen4ExpDecoderLayer`` that ``layer_first_half`` reproduces."""
    from mlx_vlm.models.qwen4_exp import language as q4

    if not isinstance(layer, q4.Qwen4ExpDecoderLayer):
        raise UnionRefused("layer-type", type(layer).__name__)
    code = getattr(_layer_call(type(layer)), "__code__", None)
    if code is None or frozenset(code.co_names) != _LAYER_CALL_NAMES or tuple(code.co_varnames) != _LAYER_CALL_VARNAMES:
        raise UnionRefused("layer-code", "the decoder layer's __call__ is not the one the split was written against")
    for name in ("attn_hyper_connection", "mlp_hyper_connection", "mlp"):
        if getattr(layer, name, None) is None:
            raise UnionRefused("layer-attributes", f"missing {name}")
    if not isinstance(getattr(layer, "is_linear", None), bool):
        raise UnionRefused("layer-attributes", "missing is_linear")
    if getattr(layer, "linear_attn" if layer.is_linear else "self_attn", None) is None:
        raise UnionRefused("layer-attributes", "missing attention block")


def layer_first_half(layer, hidden_states, input_ids, mask, cache, position_ids):
    """``Qwen4ExpDecoderLayer.__call__`` up to the MoE: returns ``(mixed, hyper_input, injection_weights)`` for ``layer.mlp``.

    The layer's own sub-modules, in the order and with the expressions of the installed ``__call__`` (``check_layer`` pins it).
    Updates ``cache`` exactly as the full layer call does."""
    if "ple" in layer:
        hidden_states = hidden_states + layer.ple(hidden_states, input_ids, cache, mask)
    mixed, hyper_input, injection_weights = layer.attn_hyper_connection(hidden_states)
    if layer.is_linear:
        branch = layer.linear_attn(mixed, mask=mask, cache=cache)
    else:
        branch = layer.self_attn(mixed, mask=mask, cache=cache, position_ids=position_ids)
    injection = branch[..., None, :] * injection_weights[..., None]
    hidden_states = hyper_input + injection.reshape(*hyper_input.shape)
    return layer.mlp_hyper_connection(hidden_states)


def layer_second_half(branch, hyper_input, injection_weights):
    """The rest of ``Qwen4ExpDecoderLayer.__call__`` after ``branch = self.mlp(mixed)``."""
    injection = branch[..., None, :] * injection_weights[..., None]
    return hyper_input + injection.reshape(*hyper_input.shape)


@dataclass
class UnionStep:
    """One step of a layer's second phase: a group of consecutive chunks sharing one staged expert list, or one legacy chunk."""
    chunks: list
    experts: list = field(default_factory=list)     # sorted expert ids staged for the group (empty for a legacy step)
    legacy: bool = False                            # routes unknown (tiny chunk, failed probe): the all-experts path runs it
    oversize: bool = False                          # this chunk alone exceeds the ring half: ordinary read path


def partition_groups(sets, capacity):
    """Greedy consecutive groups of chunks whose expert union fits ``capacity`` (None: unbounded, one group).

    ``sets[c]`` is chunk ``c``'s expert set, or None when its routes are unknown (it becomes a legacy step and ends the group
    before it, so a legacy chunk never runs while a group's staged operands are still in use). A chunk whose own set exceeds
    ``capacity`` is a single-chunk step with ``oversize`` set."""
    steps, group, union = [], [], set()

    def flush():
        nonlocal group, union
        if group:
            steps.append(UnionStep(list(group), sorted(union)))
        group, union = [], set()

    for c, experts in enumerate(sets):
        if experts is None:
            flush()
            steps.append(UnionStep([c], legacy=True))
            continue
        experts = set(experts)
        if capacity is not None and len(experts) > capacity:
            flush()
            steps.append(UnionStep([c], sorted(experts), oversize=True))
            continue
        if group and capacity is not None and len(union | experts) > capacity:
            flush()
        group.append(c)
        union |= experts
    flush()
    return steps


@contextmanager
def _probe_neutral():
    """Run the real MoE block for its router indices only, without double counting anything it records.

    The probe builds the block's graph (router, adaptive top-k, shared expert) once more than the real call does. What that
    would have touched outside the graph is held still: the adaptive top-k patch's route observer, keep statistics and
    ``_OVERLAP`` (which would launch the shared expert early for a result that is thrown away), and the score-sync/top-k
    profile timers. Host-window profiling (``FLASHNEXT_HOST_WINDOW=1``) is not neutralised: it counts real host time."""
    from models.flashnext import expert_cache

    timers = expert_cache._TIMERS
    keys = [k for k in timers if k.startswith("score_sync") or k in ("topk_python", "shared_expert", "router_sync")]
    saved_timers = {k: timers[k] for k in keys}
    adaptive = sys.modules.get("models.flashnext.adaptive_topk")
    saved = None
    if adaptive is not None:
        saved = (adaptive._OVERLAP[0], adaptive._ROUTE_OBSERVER[0], adaptive._KEEP_SUM[0], adaptive._KEEP_COUNT[0],
                 dict(adaptive._LAST_KEEPS))
        adaptive._OVERLAP[0] = False
        adaptive._ROUTE_OBSERVER[0] = None
    try:
        yield
    finally:
        timers.update(saved_timers)
        if adaptive is not None:
            adaptive._OVERLAP[0], adaptive._ROUTE_OBSERVER[0], adaptive._KEEP_SUM[0], adaptive._KEEP_COUNT[0] = saved[:4]
            adaptive._LAST_KEEPS.clear()
            adaptive._LAST_KEEPS.update(saved[4])


class RouteUnion:
    """One streamed layer's route-union state inside an ``ExpertWindow`` (see the module docstring).

    ``StreamingSwitchGLU.__call__`` hands every prompt-sized call to ``call`` while ``window.route_union`` is this object:
    in ``probe`` mode it records the router indices and returns a zero placeholder (no read, no ring use); in ``run`` mode it
    runs the layer's ``_issue`` on the staged operands with the indices remapped to staged positions."""

    def __init__(self, window, glu):
        self.window = window
        self.glu = glu
        self.mode = "probe"
        self.recorded = []                      # indices of the probed calls (probe mode)
        self.checks = []                        # lazy "every routed expert is staged" flags (run mode)
        self.weights = self.table = self.member = None
        self.num_experts = int(glu.gate_proj.num_experts)
        self.groups = []                        # one dict per staged group (trace)

    # -- the MoE call ----------------------------------------------------------------------------------------------------
    def call(self, glu, x, indices, allow_sort):
        from models.flashnext import expert_cache

        # Side effects of ``StreamingSwitchGLU.__call__`` handled here, one by one:
        #  * the progress observer (cancellation point and watchdog tick) runs on every call, probe included, so a cancel is
        #    honoured during phase A as well as phase B;
        #  * a stale ``_routed_host`` hand-off is dropped (only the adaptive one-sync decode path sets it);
        #  * the router sync and ``_ROUTE_TRACE`` belong to the ordinary path, which this never takes: the trace callback is
        #    called once per chunk in run mode from the indices the probe already synchronised.
        observer = expert_cache._PREFILL_PROGRESS
        if observer is not None:
            observer(glu.layer_id)
        if getattr(glu, "_routed_host", None) is not None:
            object.__setattr__(glu, "_routed_host", None)
        if self.mode == "probe":
            self.recorded.append(indices)
            return mx.zeros((*indices.shape, glu.down_proj.output_dims), dtype=x.dtype)
        if self.weights is None:
            raise RuntimeError("route-union: MoE call with no staged experts")
        self.checks.append(mx.all(self.member[indices]))
        local = self.table[indices]
        if expert_cache._ROUTE_TRACE[0] is not None:
            expert_cache._ROUTE_TRACE[0](glu.layer_id, indices.reshape(-1).tolist())
        return glu._issue(mx.expand_dims(x, (-2, -3)), indices, (glu.gate_proj, glu.up_proj, glu.down_proj), local,
                          self.weights, allow_sort, time.perf_counter() if expert_cache._PROFILE else 0.0)

    # -- staging ---------------------------------------------------------------------------------------------------------
    def capacity(self):
        """Experts one staging may hold: the ring half, or None (no ring, or the ring has disabled itself)."""
        ring = self.window.ring
        if ring is None or getattr(ring, "disabled", None):
            return None
        try:
            return int(ring.ring.half_experts)
        except Exception:
            return None

    def stage(self, wanted, routed_hosts):
        """Read ``wanted`` (sorted expert ids) once and make it the operand of the next chunks: ring first, else the layer's
        own read path. ``routed_hosts`` are the group's routed ids (host lists), used only to seed the decode pool."""
        from models.flashnext import expert_cache

        glu, window = self.glu, self.window
        ring = window.ring
        began = time.perf_counter()
        try:
            active_before = round(mx.get_active_memory() / 1e6)
        except Exception:
            active_before = None
        recording = prefill_reads.ACTIVE[0]
        stats_mark = prefill_reads.mark(ring) if recording is not None else None
        expert_cache._prefill_trace("prefill_expert_fetch_begin", layer=glu.layer_id, experts=len(wanted))
        weights = None
        if ring is not None:
            weights = ring.stage(glu.gate_proj.cache.store, glu.gate_proj.cache.prefix.rsplit(".", 1)[0], wanted)
        via_ring = weights is not None
        if via_ring and os.environ.get("FLASHNEXT_PREFILL_SEED_DECODE") == "1" and ring.pending is not None:
            routed = [e for host in routed_hosts for e in host]
            ring.seed_decode(glu.layer_id, {e: i for i, e in enumerate(wanted)}, routed)
        if weights is None:
            weights = glu._read_weights((glu.gate_proj, glu.up_proj, glu.down_proj), wanted)
        expert_cache._prefill_trace("prefill_expert_fetch_end", layer=glu.layer_id,
                                    seconds=round(time.perf_counter() - began, 3))
        lookup = np.zeros(self.num_experts, dtype=np.uint32)
        lookup[wanted] = np.arange(len(wanted), dtype=np.uint32)
        member = np.zeros(self.num_experts, dtype=bool)
        member[wanted] = True
        self.table, self.member, self.weights = mx.array(lookup), mx.array(member), weights
        elapsed = time.perf_counter() - began
        nbytes = len(wanted) * prefill_reads.expert_bytes(glu)
        stats = window.stats
        stats["layers"] += 1
        stats["ring_layers" if via_ring else "fallback_layers"] += 1
        stats["bytes"] += nbytes
        stats["read_seconds"] += elapsed
        try:
            window.layer_reads.append([glu.layer_id, round(elapsed, 3), active_before, round(mx.get_active_memory() / 1e6)])
        except Exception:
            pass
        if recording is not None:
            recording.note("layer-union", glu, wanted, via_ring, ring, stats_mark)
        self.groups.append({"experts": len(wanted), "ring": via_ring, "bytes": nbytes})

    def release(self) -> None:
        """Drop the staged operands and free the ring half. Call only after every graph that used them was evaluated."""
        self.weights = self.table = self.member = None
        if self.window.ring is not None:
            self.window.ring._retire()


def _union_layer(window, union, layer, index, states, inputs, positions, cache, mask_for, chunk_spans, freeze_edges, report,
                 count):
    """One layer in route-union order. Updates ``states`` in place; returns per-chunk (seconds, MLX active MB) lists.

    Phase A (every chunk in order): the layer's first half with the layer's cache, the cache state made concrete (the inner
    snapshot freeze happens here, right after the layer finished that chunk's cache update, as in the chunk loop), then the
    chunk's router indices through the real MoE block in probe mode (a chunk of at most 8 tokens runs the whole layer here). Phase B (group by group): stage the group's experts,
    then run the real MoE block and the second half of every chunk of the group, evaluating each chunk as the chunk loop does."""
    n = len(inputs)
    seconds = [0.0] * n
    memory = [None] * n
    firsts = [None] * n
    sets = [None] * n                           # per chunk: sorted unique routed experts, None when unknown
    hosts = [None] * n                          # per chunk: every routed id (host list), for decode seeding
    layer_cache = cache[index]
    window.route_union = union

    def freeze(c):
        if chunk_spans[c][1] in freeze_edges and c + 1 < n:
            from models.flashnext import prefix_cache

            report.frozen.setdefault(chunk_spans[c][1], [None] * count)[index] = prefix_cache.freeze_cache(layer_cache)

    for c, chunk in enumerate(inputs):
        began = time.perf_counter()
        if chunk.size <= TINY_CHUNK_TOKENS:
            # A decode-sized chunk never takes the window path of the MoE call (and compile_glue may route a one-token layer
            # call differently): run the layer whole, exactly as the chunk loop does, at its place in this phase.
            states[c] = layer(states[c], chunk, mask=mask_for(c), cache=layer_cache, position_ids=positions[c])
            mx.eval(states[c], layer_cache.state)
            freeze(c)
            seconds[c] += time.perf_counter() - began
            memory[c] = _active_mb()
            continue
        mixed, hyper_input, injection_weights = layer_first_half(layer, states[c], chunk, mask_for(c), layer_cache,
                                                                 positions[c])
        union.recorded.clear()
        union.mode = "probe"
        with _probe_neutral():
            layer.mlp(mixed)                    # the placeholder result is discarded unevaluated
        if len(union.recorded) != 1:
            union.recorded.clear()
            report.union_refused["probe-count"] = report.union_refused.get("probe-count", 0) + 1
        mx.eval(mixed, hyper_input, injection_weights, layer_cache.state, union.recorded)
        freeze(c)
        if union.recorded:
            flat = np.array(union.recorded[0]).reshape(-1)
            sets[c], hosts[c] = np.unique(flat).tolist(), flat.tolist()
        union.recorded.clear()
        firsts[c] = (mixed, hyper_input, injection_weights)
        seconds[c] += time.perf_counter() - began

    pending = [c for c in range(n) if firsts[c] is not None]       # chunks whose second half is still to run
    steps = partition_groups([sets[c] for c in pending], union.capacity())
    for step in steps:
        chunks = [pending[i] for i in step.chunks]
        if step.legacy:
            began = time.perf_counter()
            c = chunks[0]
            window.route_union = None           # the existing path: all experts of the layer for >8 rows
            mixed, hyper_input, injection_weights = firsts[c]
            states[c] = layer_second_half(layer.mlp(mixed), hyper_input, injection_weights)
            mx.eval(states[c])
            firsts[c] = None
            window.end_layer()                  # never reuse its operands once a later group may recycle the ring half
            seconds[c] += time.perf_counter() - began
            memory[c] = _active_mb()
            continue
        window.route_union = union
        union.stage(step.experts, [hosts[c] for c in chunks])
        for c in chunks:
            began = time.perf_counter()
            union.mode = "run"
            mixed, hyper_input, injection_weights = firsts[c]
            branch = layer.mlp(mixed)
            if len(union.checks) != 1:
                raise RuntimeError("route-union: the MoE block did not call the streamed experts exactly once")
            routed_ok = union.checks.pop()
            states[c] = layer_second_half(branch, hyper_input, injection_weights)
            mx.eval(states[c], routed_ok)       # the chunk's graph (and its flag) is complete before the next chunk or group
            if not bool(routed_ok.item()):
                raise RuntimeError("route-union: a chunk routed outside its staged experts (probe and execution disagree)")
            firsts[c] = None
            seconds[c] += time.perf_counter() - began
            memory[c] = _active_mb()
        union.release()
    window.route_union = None

    groups = [step for step in steps if not step.legacy]
    ring_groups = sum(1 for g in union.groups if g["ring"])
    report.union_layers += 1
    report.union.append([union.glu.layer_id,
                         len({e for s in sets if s for e in s}), sum(len(s) for s in sets if s), len(groups), ring_groups,
                         len(union.groups) - ring_groups, sum(g["bytes"] for g in union.groups)])
    return seconds, memory


def _active_mb():
    try:
        return round(mx.get_active_memory() / 1e6)
    except Exception:
        return None


# --------------------------------------------------------------------------------------------------------------- driver

def _position_ids(language, inputs, offset: int, cache):
    """The position ids ``LanguageModel.__call__`` builds for a text-only chunk that starts at ``offset``.

    The first chunk (``offset == 0``) goes through ``get_rope_index`` exactly as the model does and records the same state;
    later chunks add the cache offset to the stored rope deltas.
    """
    batch, length = inputs.shape
    if offset == 0 or language._rope_deltas is None:
        position_ids, deltas = language.get_rope_index(inputs, None, None, None)
        language._rope_deltas = mx.zeros((batch, 1), dtype=deltas.dtype)
        language._position_ids = position_ids
        return position_ids
    delta = mx.array(offset + language._rope_deltas)
    if delta.ndim == 0:
        delta = mx.expand_dims(delta, axis=0)
    delta = mx.tile(delta, (batch, 1)) if delta.shape[0] < batch else delta[:batch]
    position_ids = mx.broadcast_to(mx.arange(length).reshape(1, -1), (batch, length))
    position_ids = mx.add(position_ids, delta)
    if language._position_ids is not None and language._position_ids.ndim == 3:
        position_ids = mx.broadcast_to(position_ids[None, ...], (3, batch, length))
    return position_ids


UNION_COLUMNS = ("layer", "union", "chunk_sets", "groups", "ring_groups", "fallback_groups", "expert_bytes")


@dataclass
class WindowReport:
    tokens: int = 0
    chunks: int = 0
    seconds: float = 0.0
    layer_seconds: list = field(default_factory=list)
    layer_reads: list = field(default_factory=list)         # [layer, read s, active MB before, after] per streamed layer
    chunk_seconds: list = field(default_factory=list)       # per layer: seconds of each chunk's evaluation (trace only)
    chunk_memory_mb: list = field(default_factory=list)     # per layer: MLX active MB after each chunk's evaluation (trace only)
    layer_memory_mb: list = field(default_factory=list)     # [active, cache, peak] MLX MB at each layer end (trace only)
    expert_bytes: int = 0
    read_seconds: float = 0.0
    ring_layers: int = 0
    fallback_layers: int = 0
    prefetch_layers: int = 0
    frozen: dict = field(default_factory=dict)              # relative chunk edge -> per-layer frozen caches (inner snapshots)
    # Route-union order only (empty otherwise). ``union`` rows follow ``UNION_COLUMNS``; ``union_refused`` counts why a layer
    # (or a chunk's probe) fell back to the all-experts path. ``ring_layers``/``fallback_layers`` above count staged GROUPS here.
    union_layers: int = 0
    union: list = field(default_factory=list)
    union_refused: dict = field(default_factory=dict)

    def union_summary(self) -> dict:
        rows = self.union
        return {"layers": self.union_layers, "refused": dict(self.union_refused), "columns": list(UNION_COLUMNS),
                "per_layer": [list(row) for row in rows],
                "union_experts": sum(row[1] for row in rows), "chunk_set_experts": sum(row[2] for row in rows),
                "groups": sum(row[3] for row in rows), "ring_groups": sum(row[4] for row in rows),
                "fallback_groups": sum(row[5] for row in rows), "expert_gb": round(sum(row[6] for row in rows) / 1e9, 3)}


def _begin_union(window, layer, report):
    """A ``RouteUnion`` for this layer, or None after counting why the all-experts path runs it instead."""
    glu = _streamed_glu(layer)
    try:
        if glu is None:
            raise UnionRefused("not-streamed", "no StreamingSwitchGLU")
        return window.begin_union(layer, glu)
    except UnionRefused as refusal:
        report.union_refused[refusal.reason] = report.union_refused.get(refusal.reason, 0) + 1
        return None


def _streamed_glu(layer):
    from models.flashnext import expert_cache

    glu = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
    return glu if isinstance(glu, expert_cache.StreamingSwitchGLU) else None


def run_window(language, ids, cache, chunk_spans, offset: int, window=None, checkpoint=None, after_layer=None, freeze_edges=()):
    """Advance every layer's cache over ``ids`` (shape ``(1, T)``) layer-major and return the final hidden states.

    ``chunk_spans`` are relative ``(start, end)`` chunks covering ``ids``; ``offset`` is how many tokens the cache already
    holds. Returns ``(chunk_outputs, report)`` where ``chunk_outputs`` is the list of per-chunk hidden states after the last
    layer and the head's hyper-connection mixer is NOT applied (the caller applies it to the chunk it needs).
    ``window`` is an ``ExpertWindow`` or None (None leaves the layers' own expert path in charge, which is what the
    chunk-major order uses). ``checkpoint(layer)`` runs at each layer end after the layer has been evaluated (a safe point).
    """
    from models.flashnext import expert_cache

    body = language.model
    layers = body.layers
    hc = body.args.hc_count
    count = len(layers)
    report = WindowReport(tokens=int(ids.shape[1]), chunks=len(chunk_spans))
    # ExpertWindow is reused across windows; report only this invocation's
    # traffic and timings rather than its lifetime totals.
    stats_before = dict(window.stats) if window is not None else None
    reads_before = len(window.layer_reads) if window is not None and hasattr(window, "layer_reads") else 0
    freeze_edges = {int(e) for e in freeze_edges}
    began = time.perf_counter()
    inputs = [ids[:, a:b] for a, b in chunk_spans]
    states = []
    positions = []
    for (a, b), chunk in zip(chunk_spans, inputs):
        states.append(mx.tile(body.embed_tokens(chunk), (1, 1, hc)))
        positions.append(_position_ids(language, chunk, offset + a, cache))
    mx.eval(states)
    fa_masks = [None] * len(chunk_spans)
    ssm_masks = [None] * len(chunk_spans)
    from mlx_vlm.models.qwen4_exp import language as q4

    def mask_for(index, layer, c):
        if index == body.fa_idx:
            # The model builds the mask from the first full-attention cache before any layer runs, i.e. at this
            # chunk's start offset; the cache is at exactly that offset here too.
            fa_masks[c] = q4._create_qwen3_5_attention_mask(states[c], cache[body.fa_idx])
        if index == body.ssm_idx:
            ssm_masks[c] = q4._create_qwen3_5_ssm_mask(states[c], cache[body.ssm_idx])
        return ssm_masks[c] if layer.is_linear else fa_masks[c]

    use_union = window is not None and hasattr(window, "begin_union") and route_union_enabled()
    previous = expert_cache._LAYER_WINDOW[0]
    expert_cache._LAYER_WINDOW[0] = window
    try:
        for index, layer in enumerate(layers):
            layer_began = time.perf_counter()
            prefetch = getattr(window, "prefetch", None)
            if prefetch is not None and index + 1 < count:
                prefetch(_streamed_glu(layers[index + 1]))      # overlaps this layer's compute (opt-in, no-op otherwise)
            chunk_times = []
            chunk_memory = []
            union = None
            if use_union:
                union = _begin_union(window, layer, report)
            if union is not None:
                seconds, chunk_memory = _union_layer(window, union, layer, index, states, inputs, positions, cache,
                                                     lambda c, index=index, layer=layer: mask_for(index, layer, c),
                                                     chunk_spans, freeze_edges, report, count)
                chunk_times = [round(value, 3) for value in seconds]
            else:
                for c, chunk in enumerate(inputs):
                    chunk_began = time.perf_counter()
                    mask = mask_for(index, layer, c)
                    states[c] = layer(states[c], chunk, mask=mask, cache=cache[index], position_ids=positions[c])
                    # One evaluation per (layer, chunk): bounds the lazy graph and keeps the cache arrays concrete.
                    mx.eval(states[c], cache[index].state)
                    if chunk_spans[c][1] in freeze_edges and c + 1 < len(chunk_spans):
                        from models.flashnext import prefix_cache

                        report.frozen.setdefault(chunk_spans[c][1], [None] * count)[index] = prefix_cache.freeze_cache(cache[index])
                    chunk_times.append(round(time.perf_counter() - chunk_began, 3))
                    try:
                        chunk_memory.append(round(mx.get_active_memory() / 1e6))
                    except Exception:
                        chunk_memory.append(None)
            report.chunk_seconds.append(chunk_times)
            report.chunk_memory_mb.append(chunk_memory)
            if window is not None:
                window.end_layer()
            if os.environ.get("FLASHNEXT_PREFILL_CLEAR_CACHE", "1") != "0":
                mx.clear_cache()
            report.layer_seconds.append(round(time.perf_counter() - layer_began, 3))
            try:
                report.layer_memory_mb.append([round(mx.get_active_memory() / 1e6), round(mx.get_cache_memory() / 1e6),
                                               round(mx.get_peak_memory() / 1e6)])
            except Exception:
                pass
            if checkpoint is not None:
                checkpoint(index)
            if after_layer is not None:
                after_layer(index, count)
    except BaseException:
        if window is not None:
            window.abort_layer()
        raise
    finally:
        expert_cache._LAYER_WINDOW[0] = previous
        if use_union:
            window.route_union = None
        drain = getattr(window, "drain", None)
        if drain is not None:
            drain()
    report.seconds = time.perf_counter() - began
    if window is not None:
        report.expert_bytes = window.stats["bytes"] - stats_before["bytes"]
        report.read_seconds = window.stats["read_seconds"] - stats_before["read_seconds"]
        report.ring_layers = window.stats["ring_layers"] - stats_before["ring_layers"]
        report.fallback_layers = window.stats["fallback_layers"] - stats_before["fallback_layers"]
        report.prefetch_layers = window.stats.get("prefetch_used", 0) - stats_before.get("prefetch_used", 0)
        report.layer_reads = list(getattr(window, "layer_reads", [])[reads_before:])
    return states, report
