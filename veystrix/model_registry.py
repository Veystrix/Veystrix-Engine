"""Metadata-only inventory of locally installed checkpoints.

IDs identify a canonical installation path, not a model family. Moving a
checkpoint creates a new ID. This registry does not load models or promise
that structural compatibility proves inference correctness.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

from veystrix import model_identity
from veystrix.checkpoints import _json, installed_checkpoints


@dataclass(frozen=True)
class InstalledModel:
    id: str
    family: str
    display_name: str
    checkpoint: str
    architecture: str
    quantization: dict
    native_context: int | None
    verification: str

    def as_dict(self) -> dict:
        return asdict(self)


def installation_id(family: str, path: str | Path) -> str:
    """Identify one canonical checkpoint installation, without inspecting its weights."""
    digest = hashlib.sha256(str(Path(path).expanduser().resolve()).encode("utf-8")).hexdigest()
    return f"veystrix-{family}-{digest}"


UNKNOWN_FAMILY_CONTEXT = model_identity.UNKNOWN.context_limit


def catalog_context(model: InstalledModel) -> int:
    """The context window a harness catalog should advertise for one installation.

    The loaded server's actual (memory-admitted) window always wins over this estimate."""
    identity = model_identity.for_family(model.family)
    limit = identity.context_limit if identity.family != "unknown" else UNKNOWN_FAMILY_CONTEXT
    native = model.native_context
    return min(native, limit) if isinstance(native, int) and native > 0 else limit


def discover(root: Path | None = None) -> list[InstalledModel]:
    """Return structurally discovered installations without reading tensor payloads."""
    records = []
    seen = set()
    for family, path in installed_checkpoints(root):
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        config = _json(path / "config.json")
        text = config.get("text_config") or config.get("llm_config") or config
        if not isinstance(text, dict):
            text = config
        context = text.get("max_position_embeddings")
        if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
            context = None
        quant = config.get("quantization")
        if not isinstance(quant, dict):
            quant = {}
        records.append(InstalledModel(
            id=installation_id(family, path), family=family,
            display_name=path.name, checkpoint=str(path),
            architecture=str(text.get("model_type") or config.get("model_type") or "unknown"),
            quantization=quant, native_context=context,
            verification="recipe_from_headers",
        ))
    return sorted(records, key=lambda record: record.id)


def resolve(model_id: str, root: Path | None = None) -> InstalledModel | None:
    """Resolve only an exact installed ID; never select by an ambiguous family name."""
    return next((model for model in discover(root) if model.id == model_id), None)


def main() -> int:
    """Print metadata for the native application; never load model weights."""
    import json
    print(json.dumps({"object": "list", "data": [dict(m.as_dict(), loaded=False, context_window=catalog_context(m))
                                                 for m in discover()]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
