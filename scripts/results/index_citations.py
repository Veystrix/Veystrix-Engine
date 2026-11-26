"""Build a curated index of result runs cited by active documentation.

Nothing is moved, deleted or rewritten. This only reads docs and lists the
evidence they point at, so a large retained history stays discoverable.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC_GLOBS = ("docs/**/*.md", "README.md", "CONTRIBUTING.md")
CITATION = re.compile(r"results/[A-Za-z0-9][A-Za-z0-9._/-]*")
HEADER = """# Results index

This is a generated map of result runs cited by active documentation. It does
**not** move, rewrite or delete any record: the raw evidence stays where it is.
Regenerate with `python scripts/results/index_citations.py --write`.

Counts are the number of active docs that reference the path. Uncited runs are
still retained under `results/<model>/`; see [results/README.md](README.md) for
the storage rules and [docs/testing.md](../docs/testing.md) for the schema.

## Cited runs

| Citations | Run |
|---:|---|
"""
FOOTER = "\n"


def citation_counts(root: Path = ROOT) -> Counter[str]:
    """Count doc citations of existing result paths."""
    counts: Counter[str] = Counter()
    files: set[Path] = set()
    for pattern in DOC_GLOBS:
        files.update(root.glob(pattern))
    for doc in sorted(files):
        if not doc.is_file() or "/archive/" in doc.as_posix():
            continue
        for match in CITATION.findall(doc.read_text(errors="replace")):
            path = match.rstrip("./")
            if len(Path(path).parts) < 3:  # results/<model> is not a run
                continue
            if (root / path).exists():
                counts[path] += 1
    return counts


def render(counts: Counter[str]) -> str:
    lines = [HEADER]
    for path, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {count} | [`{path}`](../{path}) |\n")
    lines.append(FOOTER)
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write results/INDEX.md")
    args = parser.parse_args()
    output = render(citation_counts())
    if args.write:
        (ROOT / "results" / "INDEX.md").write_text(output)
        print(f"wrote results/INDEX.md ({len(citation_counts())} cited runs)")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
