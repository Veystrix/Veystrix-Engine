"""Public release name from a version tag.

Internally versions are MAJOR.MINOR.PATCH so Sparkle ordering and maintenance
releases work. The public name drops a trailing ``.0``: v1.0.0 -> Veystrix 1.0.
"""
from __future__ import annotations

import sys


def release_title(tag: str) -> str:
    version = tag[1:] if tag.startswith("v") else tag
    parts = version.split(".")
    if len(parts) == 3 and parts[2] == "0":
        version = ".".join(parts[:2])
    return f"Veystrix {version}"


if __name__ == "__main__":
    print(release_title(sys.argv[1]))
