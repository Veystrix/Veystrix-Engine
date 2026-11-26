"""KV precision policy for the generic C++ runtime: sizes, modes and the ``auto`` plan.

The C++ store (cpp/engine/src/generic/kv_store.hpp) keeps three regions per attention layer:
a protected PREFIX (first P tokens), a MIDDLE history and a RECENT window of W tokens (paged,
256-token pages). Each region has its own precision; K and V history bits are separate.
Every token is quantized once from the original activation (never INT8 -> INT4).

Modes (name -> prefix tokens/bits, recent tokens/bits, history K/V bits):

    bf16          everything bf16 (exact)
    int8          everything INT8 g64 (mlx_lm QuantizedKVCache equivalent)
    k8v4          K INT8, V INT4
    int4          everything INT4
    tiered        recent W bf16 (exact inside the window), history INT8
    tiered-k8v4   prefix 1024 INT8, recent W INT8, history K8/V4
    tiered4       recent W INT8, history INT4
    tiered4+      prefix 1024 INT8, recent W INT8, history INT4  (long-context target)

``auto`` picks, for the context ceiling the server admits, the most precise mode whose KV
fits the plan's KV budget; INT4 history (tiered4+) is eligible only after the real-checkpoint
quality gate recorded a pass (``gate_passed``), otherwise the fallback order is k8v4 history
then INT8 history (coordinator decision, 20 Nov 2026).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3
PAGE = 256
GROUP = 64

# name: (prefix_tokens, prefix_bits, recent_tokens (-1 unbounded, 0 none), recent_bits, hist_k, hist_v)
MODES: dict[str, tuple[int, int, int, int, int, int]] = {
    "bf16": (0, 16, -1, 16, 16, 16),
    "int8": (0, 16, 0, 16, 8, 8),
    "k8v4": (0, 16, 0, 16, 8, 4),
    "int4": (0, 16, 0, 16, 4, 4),
    "k4v2": (0, 16, 0, 16, 4, 2),
    "k3v2": (0, 16, 0, 16, 3, 2),
    "int2": (0, 16, 0, 16, 2, 2),
    "tiered": (0, 16, 8192, 16, 8, 8),
    "tiered-k8v4": (1024, 8, 8192, 8, 8, 4),
    "tiered4": (0, 16, 8192, 8, 4, 4),
    "tiered4+": (1024, 8, 8192, 8, 4, 4),
    # 1M-context lossless-quality mode: exact/INT8 prefix + recent in RAM, per-page min/max key
    # index in RAM (~64 MiB at 1M), cold history pages offloaded to SSD with a bounded hot-page LRU.
    "sparse-paged": (1024, 16, 8192, 16, 4, 4),
    "sparse-paged-k4v2": (1024, 16, 8192, 16, 4, 2),
    "sparse-paged-k2v2": (1024, 16, 8192, 16, 2, 2),
}
CHOICES = ("auto", *MODES)
EXACT = frozenset({"bf16"})
SPARSE_HOT_PAGES_DEFAULT = 256  # 64K hot history tokens cached in RAM across paged offload


def _bpv(bits: int) -> float:
    """Bytes per stored value: packed bits + bf16 scale and bias per group of 64."""
    return 2.0 if bits >= 16 else bits / 8 + 4 / GROUP


@dataclass(frozen=True)
class KVGeometry:
    attention_layers: int
    kv_heads: int
    head_dim: int
    sliding_layers: int = 0
    sliding_window: int = 0

    @property
    def values_per_token(self) -> int:
        """K (or V) values per token across all full-attention visits."""
        return self.attention_layers * self.kv_heads * self.head_dim

    @property
    def sliding_values_per_token(self) -> int:
        """K (or V) values per token across all sliding-window visits (capped at sliding_window)."""
        return self.sliding_layers * self.kv_heads * self.head_dim

    @classmethod
    def from_recipe(cls, recipe: dict) -> "KVGeometry":
        blocks = recipe.get("blocks", {})
        layers = recipe.get("layers", {})
        count = int(layers.get("count", 0))
        pattern = layers.get("pattern") or []
        overrides = layers.get("overrides") or {}
        loop = recipe.get("loop") or {}
        core_start = int(loop.get("core_start", 0)) if loop else 0
        core_end = int(loop.get("core_end", 0)) if loop else 0
        passes = int(loop.get("passes", 1)) if loop else 1
        share_kv = bool(loop.get("share_kv", False)) if loop else False
        attn = 0
        sliding = 0
        sw_size = 0
        heads = dim = 0
        for i in range(count):
            spec = overrides.get(str(i)) or (pattern[i % len(pattern)] if pattern else {})
            block = blocks.get(spec.get("mixer", ""), {})
            if block.get("type", "").startswith("attn."):
                visits = passes if (loop and not share_kv and core_start <= i < core_end) else 1
                win = int(block.get("sliding_window", 0) or 0)
                if win > 0:
                    sliding += visits
                    sw_size = max(sw_size, win)
                else:
                    attn += visits
                heads, dim = int(block["num_kv_heads"]), int(block["head_dim"])
        return cls(attn, heads, dim, sliding_layers=sliding, sliding_window=sw_size)


def kv_bytes(geometry: KVGeometry, mode: str, tokens: int, window: int | None = None,
             prefix: int | None = None) -> int:
    """Peak RAM KV bytes for ``tokens`` of context under ``mode`` (both K and V, all visits)."""
    p, pb, w, rb, hk, hv = MODES[mode]
    if window is not None and w not in (0, -1):
        w = window
    if prefix is not None and p:
        p = prefix
    # Sliding-window visits are strictly capped at sliding_window tokens in RAM.
    sw_tokens = min(tokens, geometry.sliding_window) if geometry.sliding_window > 0 else 0
    sw_bits = 16 if mode in ("bf16", "tiered", "sparse-paged", "sparse-paged-k4v2", "sparse-paged-k2v2") else rb
    sw_total = sw_tokens * geometry.sliding_values_per_token * (_bpv(sw_bits) * 2)

    e = geometry.values_per_token
    if mode.startswith("sparse-paged"):
        # Prefix + recent window in RAM (bf16 exact) + per-page INT8 min/max key index (2 B/dim per page)
        # + bounded hot-page LRU of INT4 cold pages. Full cold history lives on page-aligned SSD.
        pre_n = min(tokens, p)
        rest = max(0, tokens - p)
        rec_n = min(rest, w + PAGE - 1 + 1024)
        n_pages = (tokens + PAGE - 1) // PAGE
        index_bytes = n_pages * e * 2  # INT8 min + INT8 max per K element per 256-tok page
        hot_tokens = min(max(0, rest - w), SPARSE_HOT_PAGES_DEFAULT * PAGE)
        hot_bytes = hot_tokens * e * (_bpv(hk) + _bpv(hv))
        ram_total = (pre_n * e * (_bpv(pb) * 2) + rec_n * e * (_bpv(rb) * 2) +
                     index_bytes + hot_bytes + sw_total)
        return int(ram_total)

    if w < 0 and p == 0:
        return int(tokens * e * (_bpv(rb) * 2) + sw_total)
    total = min(tokens, p) * e * (_bpv(pb) * 2)
    rest = max(0, tokens - p)
    if w >= 0:
        total += rest * e * (_bpv(hk) + _bpv(hv))          # history written once per token
    if w != 0:
        # recent copy: the window, page alignment, and up to 1024 dead rows before compaction
        live = rest if w < 0 else min(rest, w + PAGE - 1 + 1024)
        total += live * e * (_bpv(rb) * 2)
    return int(total + sw_total)


def bytes_per_token(geometry: KVGeometry, mode: str) -> float:
    """Long-run marginal bytes per token (what each extra token of history costs)."""
    p, pb, w, rb, hk, hv = MODES[mode]
    e = geometry.values_per_token
    if w < 0:
        return e * _bpv(rb) * 2
    return e * (_bpv(hk) + _bpv(hv))


GATE_ENV = "VEYSTRIX_KV_INT4_GATE"


def gate_record_path() -> Path:
    return Path(os.environ.get("VEYSTRIX_HOME") or Path.home()).expanduser() / ".veystrix" / "kv-gate.json"


def gate_passed(model_type: str, path: Path | None = None) -> bool:
    """True when the real-checkpoint KV quality gate recorded a pass for INT4 history."""
    if os.environ.get(GATE_ENV) in ("0", "1"):
        return os.environ[GATE_ENV] == "1"
    try:
        data = json.loads((path or gate_record_path()).read_text())
    except (OSError, ValueError):
        return False
    entry = data.get(model_type) if isinstance(data, dict) else None
    return isinstance(entry, dict) and entry.get("tiered4+") == "pass"


def kv_budget(ram_bytes: int, resident_bytes: int, reserve_bytes: int = 4 * GIB,
              fraction: float = 0.35) -> int:
    """KV bytes the plan allows: a fraction of what is left after weights and the OS reserve.

    For a streaming model every KV byte is a byte the page cache cannot hold experts in, so KV
    gets a bounded share, not everything that is free."""
    left = max(0, ram_bytes - resident_bytes - reserve_bytes)
    return max(GIB // 2, int(left * fraction))


@dataclass(frozen=True)
class KVPlan:
    mode: str
    window: int | None
    context: int
    bytes_at_context: int
    budget: int
    reason: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    def profile(self) -> dict:
        out = {"kv": self.mode}
        if self.window is not None:
            out["kv_window"] = self.window
        return out


def plan(geometry: KVGeometry, context: int, budget: int, *, int4_allowed: bool = False,
         requested: str = "auto") -> KVPlan:
    """Choose the KV mode for a context ceiling and a KV byte budget."""
    clamped = ""
    if requested != "auto":
        if requested not in MODES:
            raise ValueError(f"unknown KV mode {requested!r}; use one of {', '.join(CHOICES)}")
        need = kv_bytes(geometry, requested, context)
        if need <= budget:
            return KVPlan(requested, None, context, need, budget, "requested")
        # An explicit mode is a preference, not permission to blow the memory
        # budget. At long contexts e.g. int8 KV can need >17 GiB on a 16 GB
        # Mac, so fall back to the most precise mode that actually fits.
        clamped = f"{requested} needs {need / GIB:.2f} GiB at {context} tokens (> {budget / GIB:.2f} GiB budget); "
        requested = "auto"
    need = kv_bytes(geometry, "bf16", context)
    if need <= budget:
        return KVPlan("bf16", None, context, need, budget,
                      clamped + "bf16 fits at the context ceiling (exact)")
    # bf16 recent window + INT8 history: exact for sessions inside the window; enlarge W while it fits.
    base = kv_bytes(geometry, "tiered", context)
    if base <= budget:
        window = 8192
        while window + PAGE * 32 < context and kv_bytes(geometry, "tiered", context, window=window + PAGE * 32) <= budget:
            window += PAGE * 32
        return KVPlan("tiered", window, context, kv_bytes(geometry, "tiered", context, window=window), budget,
                      clamped + f"bf16 recent window of {window} tokens (exact inside it) + INT8 history")
    order = ["tiered-k8v4"] + (["tiered4+"] if int4_allowed else []) + (["sparse-paged"] if context > 65536 else [])
    for mode in order:
        need = kv_bytes(geometry, mode, context)
        if need <= budget:
            return KVPlan(mode, None, context, need, budget,
                          clamped + f"{mode} is the most precise mode that fits")
    last = order[-1]
    return KVPlan(last, None, context, kv_bytes(geometry, last, context), budget,
                  clamped + f"nothing fits the KV budget at {context} tokens; {last} is the smallest"
                  + ("" if int4_allowed else " allowed (INT4 history waits for the quality gate)")
                  + "; lower the context ceiling")


def max_context(geometry: KVGeometry, mode: str, budget: int, ceiling: int) -> int:
    """Largest page-aligned context whose KV under ``mode`` fits ``budget`` (<= ceiling)."""
    lo, hi = 0, ceiling
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if kv_bytes(geometry, mode, mid) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo if lo == ceiling else (lo // PAGE) * PAGE


def ram_bytes() -> int:
    from veystrix.memory_tiers import physical_bytes
    return physical_bytes()


def plan_for_checkpoint(checkpoint: str | Path, recipe: dict, context: int, requested: str = "auto",
                        ram: int | None = None) -> KVPlan:
    from veystrix.memory_tiers import tier_for
    from veystrix.model_memory import probe_checkpoint
    memory = probe_checkpoint(checkpoint)
    total_ram = ram or ram_bytes()
    tier = tier_for(total_ram)
    resident = memory.resident_dense_bytes if memory else 2 * GIB
    # On 24+ GB Macs that hold Inkling 2 Mini fully resident, reserve the full
    # model weight footprint before sizing KV so KV never crowds out weights.
    if memory and total_ram >= 20 * GIB and memory.total_weight_bytes + 4 * GIB <= total_ram:
        resident = max(resident, memory.total_weight_bytes)
    budget = kv_budget(total_ram, resident, fraction=tier.kv_budget_fraction)
    return plan(KVGeometry.from_recipe(recipe), context, budget,
                int4_allowed=gate_passed(str(recipe.get("model_type", ""))), requested=requested)
