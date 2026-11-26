#!/usr/bin/env python3
"""Measure the whole repository: lines (code/test/comments/blank) by language,
file sizes, and each language's share. Excludes venvs, build/dist and .git.

Usage: .venv/bin/python scripts/analyze_repo.py [--root .] [--out report.md --json report.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Directories never counted (third-party / generated / installed copies).
EXCLUDE_DIRS = {
    ".git", ".venv", "dist", "node_modules", ".build", "build", "__pycache__",
    "site-packages", "V", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".eggs", "egg-info", "checkouts",
}

# extension -> (language, category, comment style)
# category: code | docs | data | config | other
LANG = {
    ".py": ("Python", "code", "hash"),
    ".swift": ("Swift", "code", "slash"),
    ".c": ("C", "code", "slash"),
    ".h": ("C/C++ Header", "code", "slash"),
    ".cpp": ("C++", "code", "slash"),
    ".hpp": ("C++ Header", "code", "slash"),
    ".mm": ("Objective-C++", "code", "slash"),
    ".m": ("Objective-C", "code", "slash"),
    ".metal": ("Metal", "code", "slash"),
    ".rs": ("Rust", "code", "slash"),
    ".go": ("Go", "code", "slash"),
    ".js": ("JavaScript", "code", "slash"),
    ".ts": ("TypeScript", "code", "slash"),
    ".java": ("Java", "code", "slash"),
    ".sh": ("Shell", "code", "hash"),
    ".zsh": ("Shell", "code", "hash"),
    ".rb": ("Ruby", "code", "hash"),
    ".lua": ("Lua", "code", "dash"),
    ".cmake": ("CMake", "code", "hash"),
    ".md": ("Markdown", "docs", "none"),
    ".rst": ("ReStructuredText", "docs", "none"),
    ".txt": ("Plain Text", "docs", "none"),
    ".json": ("JSON", "data", "none"),
    ".jsonl": ("JSON Lines", "data", "none"),
    ".csv": ("CSV", "data", "none"),
    ".yaml": ("YAML", "data", "hash"),
    ".yml": ("YAML", "data", "hash"),
    ".toml": ("TOML", "data", "hash"),
    ".lock": ("Lockfile", "data", "none"),
    ".xml": ("XML", "data", "none"),
    ".plist": ("Plist", "data", "none"),
    ".jinja": ("Jinja", "data", "none"),
    ".gitignore": ("Gitignore", "config", "hash"),
    ".editorconfig": ("Config", "config", "hash"),
    ".cf": ("Config", "config", "hash"),
}

TEST_DIR_PARTS = {"test", "tests", "testing"}
TEST_FILE_RX = re.compile(r"(^test_|_test\.|^test-|^spec_|_spec\.)", re.I)

SLASH_LINE = re.compile(r"^\s*//")
SLASH_BLOCK_START = re.compile(r"/\*")
SLASH_BLOCK_END = re.compile(r"\*/")
DOCSTRING_DELIM = re.compile(r'^\s*(?:"""|\'\'\')')


@dataclass
class FileStat:
    path: str
    language: str
    category: str
    is_test: bool
    size_bytes: int
    total_lines: int
    code: int
    comment: int
    blank: int
    docstring: int = 0


def classify(path: Path) -> tuple[str, str] | None:
    name = path.name
    if name == "CMakeLists.txt":
        return ("CMake", "code")
    if name == "Makefile" or name.startswith("Makefile."):
        return ("Make", "code")
    if name == ".gitignore":
        return ("Gitignore", "config")
    suffix = path.suffix.lower()
    return LANG.get(suffix, (None, None))[:2] if suffix in LANG else None


def is_test_file(path: Path) -> bool:
    parts = set(p.lower() for p in path.parts[:-1])
    if parts & TEST_DIR_PARTS:
        return True
    return bool(TEST_FILE_RX.search(path.name))


def count_lines(path: Path, style: str, category: str) -> tuple[int, int, int, int, int]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return (0, 0, 0, 0, 0)
    lines = text.splitlines()
    blank = code = comment = docstring = 0
    in_block = False
    in_doc = False
    doc_delim = None
    for raw in lines:
        s = raw.strip()
        if not s:
            blank += 1
            continue
        if category in ("data", "docs") and style == "none":
            code += 1  # data/docs lines counted as content
            continue
        if in_block:
            comment += 1
            if SLASH_BLOCK_END.search(s):
                in_block = False
            continue
        if in_doc:
            docstring += 1
            if doc_delim and doc_delim in s:
                in_doc = False
            continue
        # Python docstrings (counted separately as comment-like documentation)
        if style == "hash" and path.suffix.lower() == ".py":
            m = DOCSTRING_DELIM.match(s)
            if m:
                delim = m.group(0).strip()
                docstring += 1
                if s.count(delim) >= 2:
                    pass  # single-line docstring
                else:
                    in_doc = True
                    doc_delim = delim
                continue
        if style == "slash":
            if SLASH_LINE.match(s):
                comment += 1
                continue
            if SLASH_BLOCK_START.search(s):
                comment += 1
                if not SLASH_BLOCK_END.search(s):
                    in_block = True
                continue
            code += 1
            continue
        if style == "hash":
            if s.startswith("#"):
                comment += 1
            else:
                code += 1
            continue
        if style == "dash":
            if s.startswith("--"):
                comment += 1
            else:
                code += 1
            continue
        code += 1
    return (len(lines), code, comment, blank, docstring)


def scan(root: Path) -> list[FileStat]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            cl = classify(p)
            if not cl:
                continue
            lang, cat = cl
            # map comment style
            style = LANG.get(p.suffix.lower(), ("", "", "none"))[2]
            if p.name == "CMakeLists.txt":
                style = "hash"
            if p.name == "Makefile" or p.name.startswith("Makefile."):
                style = "hash"
            total, code, comment, blank, doc = count_lines(p, style, cat)
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            out.append(FileStat(
                path=str(p.relative_to(root)), language=lang, category=cat,
                is_test=is_test_file(p), size_bytes=size, total_lines=total,
                code=code, comment=comment, blank=blank, docstring=doc,
            ))
    return out


def aggregate(stats: list[FileStat]):
    by_lang = defaultdict(lambda: dict(files=0, size=0, total=0, code=0, comment=0, blank=0, doc=0, test_files=0, test_code=0))
    for f in stats:
        a = by_lang[f.language]
        a["files"] += 1
        a["size"] += f.size_bytes
        a["total"] += f.total_lines
        a["code"] += f.code
        a["comment"] += f.comment
        a["blank"] += f.blank
        a["doc"] += f.docstring
        if f.is_test:
            a["test_files"] += 1
            a["test_code"] += f.code
    return by_lang


def fmt_num(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out")
    ap.add_argument("--json")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    stats = scan(root)
    by_lang = aggregate(stats)

    total_code = sum(a["code"] for a in by_lang.values())
    total_prog_code = sum(a["code"] for a in by_lang.values() if a is not None)
    prog_langs = {k: v for k, v in by_lang.items() if k in {
        "Python", "Swift", "C", "C++", "C++ Header", "C/C++ Header", "Objective-C++",
        "Objective-C", "Metal", "Rust", "Go", "JavaScript", "TypeScript", "Java",
        "Shell", "Ruby", "Lua", "CMake", "Make",
    }}
    total_prog = sum(a["code"] for a in prog_langs.values())

    lines = []
    lines.append("# Repository measurement\n")
    lines.append(f"Root: `{root}`  \nFiles counted: **{len(stats)}**  \n"
                 f"Total lines: **{fmt_num(sum(f.total_lines for f in stats))}**  \n"
                 f"Total code/content lines: **{fmt_num(total_code)}**  \n"
                 f"Total size: **{sum(f.size_bytes for f in stats)/1e6:.1f} MB**\n")
    lines.append("\n## By language (code lines = executable or content lines)\n")
    lines.append("| Language | Files | Code lines | Comment | Docstring | Blank | Test code | Size | % of all prog langs |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lang, a in sorted(by_lang.items(), key=lambda kv: -kv[1]["code"]):
        pct = (a["code"] / total_prog * 100) if lang in prog_langs else 0.0
        lines.append(f"| {lang} | {a['files']} | {a['code']:,} | {a['comment']:,} | {a['doc']:,} | "
                     f"{a['blank']:,} | {a['test_code']:,} | {a['size']/1e6:.2f} MB | {pct:.1f}% |")
    lines.append(f"\nTotal programming-language code lines: **{total_prog:,}**\n")
    lines.append("\n## By category\n")
    cat = defaultdict(lambda: [0, 0, 0])
    for f in stats:
        cat[f.category][0] += f.code
        cat[f.category][1] += f.size_bytes
        cat[f.category][2] += 1
    lines.append("| Category | Files | Code/content lines | Size |")
    lines.append("|---|---:|---:|---:|")
    for c, (code, size, n) in sorted(cat.items(), key=lambda kv: -kv[1][0]):
        lines.append(f"| {c} | {n} | {code:,} | {size/1e6:.2f} MB |")
    lines.append("\n## Test vs non-test (code lines)\n")
    tc = sum(a["test_code"] for a in by_lang.values())
    lines.append(f"- Test code lines: **{tc:,}** ({tc/total_code*100:.1f}% of all code/content lines)")
    lines.append(f"- Non-test code lines: **{total_code-tc:,}** ({(total_code-tc)/total_code*100:.1f}%)")
    lines.append("\nNotes: excludes .git, .venv, dist, build, __pycache__, site-packages, .build "
                 "checkouts. Data/docs lines counted as 'code/content'. Python docstrings counted "
                 "separately from `#` comments. Test files detected by path (tests/) or name "
                 "(test_*/ *_test).")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report)
        print(f"\nwrote {args.out}")
    if args.json:
        data = {
            "root": str(root),
            "summary": {
                "files": len(stats),
                "total_lines": sum(f.total_lines for f in stats),
                "total_code": total_code,
                "total_programming_code": total_prog,
                "total_size_bytes": sum(f.size_bytes for f in stats),
                "test_code": tc,
            },
            "by_language": {k: v for k, v in by_lang.items()},
            "files": [asdict(f) for f in stats],
        }
        Path(args.json).write_text(json.dumps(data, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
