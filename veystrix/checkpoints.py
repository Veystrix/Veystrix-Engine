"""Find compatible local model checkpoints."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

from veystrix import model_identity


_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
TOKENIZER_IDENTITY_FILES = (*_TOKENIZER_FILES, "special_tokens_map.json", "added_tokens.json",
                            "chat_template.jinja", "vocab.json", "merges.txt", "tokenizer.model", "spiece.model")


def tokenizer_fingerprint(path: Path) -> str | None:
    """Shared checkpoint identity for caches that depend on token IDs."""
    digest = hashlib.sha256(b"veystrix-tokenizer-v1")
    found = False
    for name in TOKENIZER_IDENTITY_FILES:
        file = path / name
        if not file.is_file():
            continue
        digest.update(name.encode())
        digest.update(str(file.stat().st_size).encode())
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        found = True
    return digest.hexdigest() if found else None


def _sane_shard_name(value) -> bool:
    return (
        isinstance(value, str)
        and value.endswith(".safetensors")
        and "/" not in value
        and "\\" not in value
        and value not in (".", "..")
        and not value.startswith(".")
        and not Path(value).is_absolute()
    )


def _missing_shards(path: Path) -> list[str]:
    index_path = path / "model.safetensors.index.json"
    index = _json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return [index_path.name]
    values = list(weight_map.values())
    if not all(_sane_shard_name(value) for value in values):
        return [index_path.name]
    return [name for name in sorted(set(values)) if not (path / name).is_file()]


def _checkpoint_error(label: str, path: Path, missing: list[str]) -> ValueError:
    details = ", ".join(missing) if missing else "model metadata"
    return ValueError(
        f"incomplete or incompatible {label} checkpoint: {path}\n"
        f"missing or invalid: {details}\n"
        f"repair: resume the checkpoint download into {path} and try again"
    )


def model_root() -> Path:
    return Path(os.environ.get("VEYSTRIX_MODEL_ROOT", "~/models")).expanduser()


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _declared_types(config: dict) -> set[str]:
    types = {config.get("model_type")}
    for key in ("text_config", "llm_config"):
        nested = config.get(key)
        if isinstance(nested, dict):
            types.add(nested.get("model_type"))
    return {t for t in types if isinstance(t, str)}


# --------------------------------------------------------------------------- #
# Generic-runtime checkpoint discovery (recipe-driven C++ engine)              #
# Any future model added to model_identity.IDENTITIES works automatically.     #
# --------------------------------------------------------------------------- #

def generic_problems(path: Path, identity: model_identity.ModelIdentity) -> list[str]:
    """Why `path` is not a usable checkpoint for `identity`; empty when valid."""
    path = Path(path)
    config = _json(path / "config.json")
    if not config:
        return ["config.json"]
    if identity.model_types and not (identity.model_types & _declared_types(config)):
        expected = "|".join(sorted(identity.model_types))
        return [f"model_type does not match {expected} ({sorted(_declared_types(config))})"]
    missing = [name for name in _TOKENIZER_FILES if not (path / name).is_file()]
    if (path / "model.safetensors.index.json").is_file():
        missing.extend(_missing_shards(path))
    elif not list(path.glob("*.safetensors")):
        missing.append("*.safetensors")
    if missing:
        return missing
    try:
        from veystrix.recipe import emit_recipe
        emit_recipe(path)
    except Exception as exc:
        return [f"recipe: {exc}"]
    return []


def generic_compatible(path: Path, identity: model_identity.ModelIdentity) -> bool:
    return path.is_dir() and not generic_problems(path, identity)


def installed_for_family(family: str, root: Path | None = None) -> list[Path]:
    identity = model_identity.IDENTITIES.get(family)
    if identity is None:
        return []
    root = root or model_root()
    children = root.iterdir() if root.is_dir() else ()
    found = []
    for path in sorted(children):
        if not path.is_dir():
            continue
        config = _json(path / "config.json")
        if identity.model_types & _declared_types(config) and generic_compatible(path, identity):
            found.append(path)
    return found


def resolve_for_family(family: str, requested: str | os.PathLike[str] | None = None) -> Path:
    """Resolve an explicit or single installed checkpoint for `family`."""
    identity = model_identity.IDENTITIES.get(family)
    if identity is None:
        raise ValueError(f"unknown model family {family!r}")
    root = model_root()
    if requested:
        value = str(requested)
        if identity.checkpoint_alias and value.lower() == identity.checkpoint_alias.lower():
            value = identity.checkpoint_directory or value
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        problems = generic_problems(path, identity)
        if problems:
            raise _checkpoint_error(identity.display_name, path, problems)
        return path.resolve()
    found = installed_for_family(family, root)
    if len(found) == 1:
        return found[0].resolve()
    if not found:
        raise ValueError(
            f"no complete {identity.display_name} checkpoint found; "
            f"run 'veystrix models pull {family}' or pass --checkpoint PATH"
        )
    raise ValueError(
        f"several {identity.display_name} checkpoints found; choose one with --checkpoint: "
        + ", ".join(str(p) for p in found)
    )


def installed_checkpoints(root: Path | None = None) -> list[tuple[str, Path]]:
    """Return every complete checkpoint for all registered model families."""
    choices: list[tuple[str, Path]] = []
    for family in model_identity.IDENTITIES:
        choices.extend((family, path) for path in installed_for_family(family, root))
    return sorted(choices, key=lambda item: (item[0], str(item[1])))


def inkling2_mini_problems(path: Path) -> list[str]:
    return generic_problems(path, model_identity.INKLING2_MINI)


def inkling2_mini_compatible(path: Path) -> bool:
    return generic_compatible(path, model_identity.INKLING2_MINI)


def installed_inkling2_mini(root: Path | None = None) -> list[Path]:
    return installed_for_family("inkling2-mini", root)


def resolve_inkling2_mini(requested: str | os.PathLike[str] | None = None) -> Path:
    return resolve_for_family("inkling2-mini", requested)


def resolve_flashnext(requested: str | os.PathLike[str] | None = None) -> Path:
    return resolve_for_family("flashnext", requested)


def resolve_nemotron35(requested: str | os.PathLike[str] | None = None) -> Path:
    return resolve_for_family("nemotron35-lightning", requested)
