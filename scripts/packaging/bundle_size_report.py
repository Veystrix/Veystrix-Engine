"""Report what occupies the bundled engine, without modifying it.

Use this before proposing bundle-size changes: it separates the fixed CPython
runtime and MLX Metal library from packages that could actually be pruned.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_app_engine = ROOT / "dist" / "Veystrix.app" / "Contents" / "Resources" / "engine"
DEFAULT_BUNDLE = _app_engine if _app_engine.is_dir() else ROOT / "dist" / "engine"


def directory_sizes(root: Path) -> list[tuple[str, int]]:
    """Immediate child directory sizes of ``root``, largest first."""
    if not root.is_dir():
        return []
    results = [(child.name, _tree_size(child)) for child in sorted(root.iterdir()) if child.is_dir()]
    return sorted(results, key=lambda item: -item[1])


def _tree_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def mib(size: int) -> str:
    return f"{size / 1024 / 1024:8.1f} MiB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", default=str(DEFAULT_BUNDLE))
    args = parser.parse_args()

    root = Path(args.bundle).expanduser()
    python_dir = root / "python"
    site_packages = python_dir / "lib" / "python3.12" / "site-packages"
    print(f"Bundle: {root}")
    for label, target in (("runtime", python_dir), ("site-packages", site_packages),
                          ("engine sources", root / "lib")):
        print(f"  {label:16s} {mib(_tree_size(target))}")
    print("\nLargest runtime directories:")
    for name, size in directory_sizes(python_dir)[:12]:
        print(f"  {mib(size)}  {name}")
    print("\nLargest site-packages:")
    for name, size in directory_sizes(site_packages)[:25]:
        print(f"  {mib(size)}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
