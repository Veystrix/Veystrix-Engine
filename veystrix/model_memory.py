"""What a checkpoint costs in memory, read from its safetensors headers only.

No tensor payload is read, so probing a 100 GB checkpoint takes milliseconds and no RAM.
The numbers feed ``memory_budget_runtime.plan`` so pool/residency sizing follows the
actual model instead of a fixed table.

Classification is by tensor name, which works across families:

* routed experts: ``switch_mlp`` / ``.experts.`` tensors, whose leading dimension is the
  expert count. ``record_bytes`` is the bytes of ONE expert of ONE layer (all its tensors).
* streamed tables: per-layer-embedding / n-gram tables that are read row-wise from SSD on
  demand and are never fully resident.
* resident dense: everything else (attention, SSM, norms, embeddings, lm_head, router).
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

_EXPERT = re.compile(r"(switch_mlp|\.experts\.)")
_NOT_ROUTED = re.compile(r"(shared_expert|shared_experts|router|\.gate\.weight)")
_STREAMED = re.compile(r"(ngram_embedding|ple_embedding|embed_tokens_per_layer)")
_LAYER = re.compile(r"\.layers\.(\d+)\.")
# Tensors the text runtime never loads: vision tower and the MTP draft head (kept off).
_IGNORED = re.compile(r"(vision_tower|(^|\.)visual\.)")
# The generic runtime never loads the MTP head either; the custom runtimes may, so only there.
_IGNORED_GENERIC = re.compile(r"(vision_tower|(^|\.)visual\.|^mtp\.|\.mtp\.)")


def _generic_runtime(root: Path) -> bool:
    try:
        config = json.loads((root / "config.json").read_text())
        from veystrix.model_identity import for_model_type
        text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
        identity = for_model_type(config.get("model_type")) or for_model_type(text.get("model_type"))
        return identity is not None and identity.cpp_runtime == "generic"
    except (OSError, ValueError, AttributeError):
        return False


@dataclass(frozen=True)
class ModelMemory:
    total_bytes: int
    routed_expert_bytes: int
    resident_dense_bytes: int
    streamed_table_bytes: int
    moe_layers: int
    experts_per_layer: int
    record_bytes: int            # one expert of one layer
    ignored_bytes: int = 0       # vision tower / MTP head: on disk, never loaded

    @property
    def expert_keys(self) -> int:
        return self.moe_layers * self.experts_per_layer


_CACHE: dict[tuple, ModelMemory] = {}


def _read_header(path: Path) -> dict:
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


def probe_checkpoint(checkpoint: str | Path) -> ModelMemory | None:
    """Memory profile of a safetensors checkpoint directory, or None if unreadable."""
    root = Path(checkpoint).expanduser()
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        return None
    try:
        key = (str(root.resolve()), tuple((s.name, s.stat().st_size, int(s.stat().st_mtime)) for s in shards))
    except OSError:
        return None
    if key in _CACHE:
        return _CACHE[key]
    total = routed = streamed = ignored = 0
    ignored_re = _IGNORED_GENERIC if _generic_runtime(root) else _IGNORED
    per_layer: dict[str, int] = {}
    experts = 0
    for shard in shards:
        try:
            header = _read_header(shard)
        except (OSError, ValueError, struct.error):
            return None
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            size = int(meta["data_offsets"][1]) - int(meta["data_offsets"][0])
            total += size
            if ignored_re.search(name):
                ignored += size
            elif _EXPERT.search(name) and not _NOT_ROUTED.search(name):
                routed += size
                layer = _LAYER.search(name)
                per_layer[layer.group(1) if layer else "?"] = per_layer.get(layer.group(1) if layer else "?", 0) + size
                shape = meta.get("shape") or []
                if shape:
                    experts = max(experts, int(shape[0]))
            elif _STREAMED.search(name):
                streamed += size
    layers = len(per_layer)
    record = (routed // (layers * experts)) if layers and experts else 0
    result = ModelMemory(total, routed, total - routed - streamed - ignored, streamed, layers, experts, record,
                         ignored)
    _CACHE[key] = result
    return result
