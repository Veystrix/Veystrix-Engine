"""TOML/YAML helpers for harness config writes.

TOML round-trips use ``tomlkit`` (a pinned dependency) when available so user
comments and nested tables survive rewrites; a small stdlib ``tomllib`` + serializer
fallback covers minimal environments. YAML files get a fenced *managed block*:
everything outside the markers is preserved byte-for-byte.
"""
from __future__ import annotations

import json
import re
import tomllib

BEGIN = "# >>> veystrix managed >>>"
END = "# <<< veystrix managed <<<"
_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def toml_loads(text: str):
    """Parse TOML without losing comments or structure.

    Returns a tomlkit document when available (so `veystrix connect` rewrites
    Codex/Kimi configs without dropping nested tables, comments or datetimes),
    otherwise a plain dict via stdlib. Malformed TOML raises instead of being
    silently replaced with `{}` (Sonnet finding 20).
    """
    try:
        import tomlkit
        doc = tomlkit.parse(text)
        return doc
    except ImportError:
        data = tomllib.loads(text)
        if not isinstance(data, dict):
            raise ValueError("TOML root is not a table")
        return data


def _toml_key(key: str) -> str:
    return key if _BARE_KEY.match(key) else json.dumps(str(key))


def _toml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    return json.dumps(str(value))


def toml_dumps(data) -> str:
    # A tomlkit document preserves comments/types and serializes itself; only
    # the minimal fallback serializer handles plain dicts (Sonnet finding 20).
    if hasattr(data, "as_string"):
        return data.as_string()
    lines = []
    for key, value in data.items():
        if not isinstance(value, dict):
            lines.append(f"{_toml_key(key)} = {_toml_scalar(value)}")
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"\n[{'.'.join(_toml_key(k) for k in [key])}]")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    lines.append(f"\n[{_toml_key(key)}.{_toml_key(sub_key)}]")
                    for leaf, leaf_value in sub_value.items():
                        lines.append(f"{_toml_key(leaf)} = {_toml_scalar(leaf_value)}")
                else:
                    lines.append(f"{_toml_key(sub_key)} = {_toml_scalar(sub_value)}")
    return "\n".join(lines).strip() + "\n"


def _yaml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return '""'
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))          # JSON strings are valid YAML scalars


def yaml_dumps(data: dict) -> str:
    def emit(value, indent: int) -> list[str]:
        pad = " " * indent
        out: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (dict, list)) and not item:
                    out.append(f"{pad}{key}: {'{}' if isinstance(item, dict) else '[]'}")
                elif isinstance(item, (dict, list)):
                    out.append(f"{pad}{key}:")
                    out.extend(emit(item, indent + 2))
                else:
                    out.append(f"{pad}{key}: {_yaml_scalar(item)}")
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    sub = emit(item, indent + 4)
                    if sub:
                        out.append(f"{pad}  - {sub[0][indent + 4:]}")
                        out.extend(sub[1:])
                else:
                    out.append(f"{pad}  - {_yaml_scalar(item)}")
        return out
    return "\n".join(emit(data, 0)).strip() + "\n"


def strip_managed(text: str) -> str:
    """``text`` without Veystrix's managed block (markers included)."""
    kept, skipping = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == BEGIN:
            skipping = True
        elif stripped == END:
            skipping = False
        elif not skipping:
            kept.append(line)
    return "\n".join(kept) + ("\n" if kept else "")


def _top_level_key(line: str) -> str | None:
    if not line or line[0] in " \t#-" or ":" not in line:
        return None
    key = line.split(":", 1)[0].strip().strip("'\"")
    return key or None


def yaml_upsert(text: str, block: str) -> str:
    """Replace our managed block, keeping all other lines exactly as they were.

    A top-level key the block defines is removed from the rest of the file (with its nested
    lines): YAML forbids duplicate keys, and harnesses such as Hermes refuse the whole file."""
    owned = {k for k in (_top_level_key(l) for l in block.splitlines()) if k}
    kept: list[str] = []
    skipping = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == BEGIN:
            skipping = True
            continue
        if stripped == END:
            skipping = False
            continue
        if not skipping:
            kept.append(line)
    if owned:
        filtered: list[str] = []
        dropping = False
        for line in kept:
            key = _top_level_key(line)
            if key is not None:
                dropping = key in owned
            elif dropping and line[:1] not in (" ", "\t", "-", "#") and line.strip():
                dropping = False
            if not dropping:
                filtered.append(line)
        kept = filtered
    while kept and not kept[-1].strip():
        kept.pop()
    body = "\n".join(kept)
    managed = f"{BEGIN}\n{block.rstrip()}\n{END}\n"
    return (body + "\n\n" if body else "") + managed
