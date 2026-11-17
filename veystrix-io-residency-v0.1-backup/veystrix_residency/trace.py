"""Routing-trace capture and loading.

Engine hook (opt-in, off by default): the engine already materialises each
decode layer's top-k ids on the host (`route_tolist`, 48 calls/token on Qwen,
23 on Nemotron). Recording them costs one small write per layer and cannot
change any output.

    from veystrix_residency.trace import RouteTraceWriter
    _TRACE = RouteTraceWriter.from_env()          # VEYSTRIX_ROUTE_TRACE=/path/trace.jsonl
    ...
    ids = route.tolist()                           # existing line
    if _TRACE: _TRACE.record(layer_idx, ids)       # decode only (one row)
    ...
    if _TRACE: _TRACE.end_token()                  # after each decoded token

JSONL line format: {"t": token_index, "l": moe_layer_index, "e": [ids...]}
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np


class RouteTraceWriter:
    def __init__(self, path: str):
        self.f = open(path, "a", buffering=1 << 20)
        self.t = 0

    @classmethod
    def from_env(cls, var: str = "VEYSTRIX_ROUTE_TRACE") -> Optional["RouteTraceWriter"]:
        p = os.environ.get(var)
        return cls(p) if p else None

    def record(self, layer: int, ids: Sequence) -> None:
        if ids and isinstance(ids[0], (list, tuple)):
            if len(ids) != 1:
                return  # prefill rows: not traced (decode-only trace)
            ids = ids[0]
        self.f.write(json.dumps({"t": self.t, "l": int(layer), "e": [int(x) for x in ids]},
                                separators=(",", ":")) + "\n")

    def end_token(self) -> None:
        self.t += 1

    def close(self) -> None:
        self.f.close()


def load_trace(path: str, experts_per_layer: Optional[int] = None
               ) -> Tuple[List[Tuple[int, ...]], int, int, int]:
    """Return (steps, n_tokens, layers_per_token, experts_per_layer).

    steps: one tuple of global keys (layer * E + expert) per MoE layer call,
    in execution order. Accepts .npy [T, L, K] or the JSONL format above.
    """
    if path.endswith(".npy"):
        arr = np.load(path)
        if arr.ndim != 3:
            raise ValueError("npy trace must be [tokens, moe_layers, k]")
        T, L, _ = arr.shape
        E = experts_per_layer or int(arr.max()) + 1
        steps = [tuple(int(l * E + e) for e in arr[t, l]) for t in range(T) for l in range(L)]
        return steps, T, L, E
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                rows.append((int(d["t"]), int(d["l"]), [int(x) for x in d["e"]]))
    if not rows:
        raise ValueError("empty trace")
    E = experts_per_layer or max(max(e) for _, _, e in rows) + 1
    tokens = sorted({t for t, _, _ in rows})
    per_tok = {}
    for t, _, _ in rows:
        per_tok[t] = per_tok.get(t, 0) + 1
    L = max(per_tok.values())
    steps = [tuple(l * E + x for x in e) for _, l, e in rows]
    return steps, len(tokens), L, E
