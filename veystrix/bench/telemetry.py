"""Request-local observations. The runtime never synchronizes a device for these."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from time import perf_counter_ns

current = ContextVar("benchmark_trace", default=None)


class Trace:
    def __init__(self, clock=perf_counter_ns, *, capture_shapes=False):
        self.clock = clock
        self.capture_shapes = capture_shapes
        self.origin = clock()
        self.events = []
        self.prefill_tokens = None
        self.decode_read_meter = None
        self.decode_read_bytes = None
        self.counts = Counter()
        self.shapes = Counter()
        self.reads = Counter()
        self.read_lock = Lock()

    @contextmanager
    def activate(self):
        token = current.set(self)
        try:
            yield self
        finally:
            current.reset(token)

    def mark(self, event, **fields):
        self.events.append({"event": event, "ns": self.clock() - self.origin, **fields})

    def end_round(self, kind, started, **fields):
        end = self.clock() - self.origin
        self.events.append({
            "event": kind, "ns": end, "duration_ns": end + self.origin - started, **fields,
        })

    def decode_commit(self):
        self.mark("decode_commit")
        if self.decode_read_meter is not None:
            read = self.decode_read_meter()
            self.decode_read_bytes = read if read >= 0 else None

    def record_reads(self, stats):
        with self.read_lock:
            self.reads["positioned_read_calls"] += stats["pread_calls"]
            self.reads["positioned_read_bytes"] += stats["pread_bytes"]

    def read_snapshot(self):
        with self.read_lock:
            return self.reads.copy()

    def shape(self, operation, arrays, **options):
        shapes = tuple(tuple(x.shape) for x in arrays if x is not None)
        self.shapes[(operation, shapes, tuple(sorted(options.items())))] += 1

    def shape_rows(self):
        return [
            {"operation": name, "operand_shapes": shapes, "options": dict(options), "calls": count}
            for (name, shapes, options), count in self.shapes.items()
        ]


@contextmanager
def mlx_shapes(mx, trace):
    """Observe MLX calls at the Python dispatch boundary in a serial benchmark."""
    operands = ("x", "w", "scales", "biases")
    options = ("transpose", "group_size", "bits", "mode")

    def wrap(name, original):
        def call(*args, **kwargs):
            active = current.get()
            if active is trace:
                if name in ("quantized_matmul", "gather_qmm"):
                    fields = operands
                    if name == "gather_qmm":
                        fields += ("lhs_indices", "rhs_indices")
                    bound = {**dict(zip(fields + options, args)), **kwargs}
                    active.shape("mlx." + name, [bound.get(key) for key in fields], **{
                        key: bound[key] for key in options if key in bound
                    })
                else:
                    active.counts["mlx_" + name + "_calls"] += 1
            return original(*args, **kwargs)
        return call

    names = ("quantized_matmul", "gather_qmm", "eval", "async_eval", "synchronize")
    originals = {name: getattr(mx, name) for name in names}
    try:
        for name, original in originals.items():
            setattr(mx, name, wrap(name, original))
        yield
    finally:
        for name, original in originals.items():
            setattr(mx, name, original)
