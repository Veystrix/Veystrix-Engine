"""Public release name from a version tag.

The public name is the full three-part version: v1.0.0 -> Veystrix 1.0.0.
Keeping it identical to pyproject.toml and CFBundleShortVersionString avoids
ambiguity between the release page, the app and the update feed.
"""
from __future__ import annotations

import sys


def release_title(tag: str) -> str:
    version = tag[1:] if tag.startswith("v") else tag
    return f"Veystrix {version}"


if __name__ == "__main__":
    print(release_title(sys.argv[1]))
