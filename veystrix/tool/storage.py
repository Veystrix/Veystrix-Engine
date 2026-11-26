"""SSD prompt-cache storage inspection and reclamation (`veystrix storage`)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys

from veystrix.tool.config import Config

KB = 1024
MB = 1024 ** 2
GB = 1024 ** 3
TB = 1024 ** 4

_SIZE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)\s*$")
_UNIT_MULTIPLIERS = {
    "": GB,
    "b": 1,
    "k": KB,
    "kb": KB,
    "kib": KB,
    "m": MB,
    "mb": MB,
    "mib": MB,
    "g": GB,
    "gb": GB,
    "gib": GB,
    "t": TB,
    "tb": TB,
    "tib": TB,
}


@dataclass(frozen=True)
class CacheEntry:
    family: str
    path: Path
    size_bytes: int
    mtime: float
    tokens: int = 0
    kind: str = "prefix-cache"


def format_bytes(num_bytes: int) -> str:
    value = max(0, int(num_bytes))
    if value >= GB:
        return f"{value / GB:.2f} GB"
    if value >= MB:
        return f"{value / MB:.1f} MB"
    if value >= KB:
        return f"{value / KB:.1f} KB"
    return f"{value} B"


def parse_size(raw: str) -> int | None:
    """Parse a human-readable size (e.g. '500MB', '1.5GB', '2', 'all').

    Returns None for 'all', or a positive byte count. Raises ValueError on invalid input.
    """
    text = (raw or "").strip().lower()
    if not text:
        raise ValueError("empty size")
    if text in ("all", "*"):
        return None
    match = _SIZE_RE.match(text)
    if not match:
        raise ValueError(f"invalid size {raw!r}; use e.g. 500MB, 1.5GB, 2G, or all")
    amount = float(match.group(1))
    unit = match.group(2).lower()
    if unit not in _UNIT_MULTIPLIERS:
        raise ValueError(f"unknown unit {unit!r}; use MB, GB, or all")
    if amount <= 0:
        raise ValueError("amount must be greater than 0")
    return max(1, int(amount * _UNIT_MULTIPLIERS[unit]))


def cache_base_dir(home: Path, config: Config | None = None) -> Path:
    configured = config.get("cache_directory") if config is not None else ""
    if configured:
        return Path(configured).expanduser()
    if os.environ.get("VEYSTRIX_CACHE_ROOT"):
        return Path(os.environ["VEYSTRIX_CACHE_ROOT"]).expanduser()
    if os.environ.get("VEYSTRIX_HOME"):
        return home / ".cache" / "veystrix"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "veystrix"
    return home / ".cache" / "veystrix"


def _dir_size_and_mtime(directory: Path) -> tuple[int, float]:
    total = 0
    latest = 0.0
    try:
        st = directory.stat()
        latest = st.st_mtime
    except OSError:
        return 0, 0.0
    for root, _dirs, files in os.walk(directory):
        for name in files:
            path = Path(root) / name
            try:
                info = path.lstat()
                total += info.st_size
                if info.st_mtime > latest:
                    latest = info.st_mtime
            except OSError:
                continue
    return total, latest


def _read_tokens_count(entry_dir: Path) -> int:
    tok_file = entry_dir / "tokens.json"
    if not tok_file.is_file():
        return 0
    try:
        data = json.loads(tok_file.read_text(encoding="utf-8"))
        toks = data.get("tokens")
        return len(toks) if isinstance(toks, list) else 0
    except (OSError, ValueError, TypeError):
        return 0


def scan_cache(base: Path) -> list[CacheEntry]:
    """Discover all Veystrix SSD prompt-cache entries under `base`."""
    entries: list[CacheEntry] = []
    if not base.is_dir():
        return entries

    # 1. C++ / generic engine persistent prefix caches:
    #    <base>/<family>/sessions/prefix-cache/<digest>/
    try:
        children = sorted(base.iterdir())
    except OSError:
        children = []
    for child in children:
        if not child.is_dir() or child.is_symlink():
            continue
        prefix_dir = child / "sessions" / "prefix-cache"
        if not prefix_dir.is_dir() or prefix_dir.is_symlink():
            continue
        family = child.name
        try:
            snaps = sorted(prefix_dir.iterdir())
        except OSError:
            snaps = []
        for snap in snaps:
            if not snap.is_dir() or snap.is_symlink() or snap.name == ".cas_pages":
                continue
            size, mtime = _dir_size_and_mtime(snap)
            if size <= 0:
                continue
            entries.append(CacheEntry(
                family=family,
                path=snap,
                size_bytes=size,
                mtime=mtime,
                tokens=_read_tokens_count(snap),
                kind="prefix-cache",
            ))

    # 2. Legacy/universal prefix-cache-v1 directories if present:
    #    <base>/prefix-cache-v1/<model>/<fingerprint>/<key>.safetensors + sidecar
    uc_root = base / "prefix-cache-v1"
    if uc_root.is_dir() and not uc_root.is_symlink():
        for family_dir in sorted(uc_root.iterdir()):
            if not family_dir.is_dir() or family_dir.is_symlink() or family_dir.name == "artifacts":
                continue
            for fp_dir in sorted(family_dir.iterdir()):
                if not fp_dir.is_dir() or fp_dir.is_symlink():
                    continue
                for data_file in sorted(fp_dir.glob("*.safetensors")):
                    if not data_file.is_file() or data_file.is_symlink():
                        continue
                    try:
                        info = data_file.lstat()
                    except OSError:
                        continue
                    size = info.st_size
                    mtime = info.st_mtime
                    for ext in (".json", ".tokens"):
                        side = data_file.with_suffix(ext)
                        if side.is_file():
                            try:
                                s_info = side.lstat()
                                size += s_info.st_size
                                mtime = max(mtime, s_info.st_mtime)
                            except OSError:
                                pass
                    entries.append(CacheEntry(
                        family=family_dir.name,
                        path=data_file,
                        size_bytes=size,
                        mtime=mtime,
                        kind="universal-cache",
                    ))

    entries.sort(key=lambda e: e.mtime)  # oldest first (LRU)
    return entries


def storage_report(base: Path) -> dict:
    entries = scan_cache(base)
    by_family: dict[str, dict] = {}
    for entry in entries:
        bucket = by_family.setdefault(entry.family, {"bytes": 0, "snapshots": 0})
        bucket["bytes"] += entry.size_bytes
        bucket["snapshots"] += 1
    total_bytes = sum(e.size_bytes for e in entries)
    probe_dir = base if base.exists() else base.parent
    while not probe_dir.exists() and probe_dir != probe_dir.parent:
        probe_dir = probe_dir.parent
    try:
        free_bytes = shutil.disk_usage(probe_dir).free
    except OSError:
        free_bytes = 0
    return {
        "cache_directory": str(base),
        "total_bytes": total_bytes,
        "total_human": format_bytes(total_bytes),
        "snapshots": len(entries),
        "by_model": {
            fam: {**info, "human": format_bytes(info["bytes"])}
            for fam, info in sorted(by_family.items())
        },
        "disk_free_bytes": free_bytes,
        "disk_free_human": format_bytes(free_bytes),
    }


def _vxkv2_referenced_digests(manifest_path: Path) -> set[str]:
    """Parse a VXKV2 binary manifest and return all referenced 64-hex CAS page digests."""
    digests: set[str] = set()
    try:
        data = manifest_path.read_bytes()
    except OSError:
        return digests
    if len(data) < 20 or data[:8] != b"VXKV2\x00\x00\x00":
        return digests
    pos = 8
    try:
        ver, _pg = struct.unpack_from("<II", data, pos)
        pos += 8
        if ver != 2:
            return digests
        (n_meta,) = struct.unpack_from("<I", data, pos)
        pos += 4
        for _ in range(n_meta):
            (klen,) = struct.unpack_from("<I", data, pos)
            pos += 4 + klen
            (vlen,) = struct.unpack_from("<I", data, pos)
            pos += 4 + vlen
        (n_tensors,) = struct.unpack_from("<I", data, pos)
        pos += 4
        for _ in range(n_tensors):
            (nlen,) = struct.unpack_from("<I", data, pos)
            pos += 4 + nlen
            _dt, ndim = struct.unpack_from("<BB", data, pos)
            pos += 2 + 8 * ndim
            (paged,) = struct.unpack_from("<B", data, pos)
            pos += 1
            if paged:
                (npages,) = struct.unpack_from("<I", data, pos)
                pos += 4
                for _ in range(npages):
                    pos += 8
                    digests.add(data[pos:pos + 64].decode("ascii"))
                    pos += 64
            else:
                (blen,) = struct.unpack_from("<Q", data, pos)
                pos += 8 + blen
    except (struct.error, UnicodeDecodeError, IndexError):
        return digests
    return digests


def gc_cas_pages(prefix_cache_root: Path) -> tuple[int, int]:
    """Delete unreferenced `.vxpg` files under `prefix_cache_root/.cas_pages`."""
    cas_dir = prefix_cache_root / ".cas_pages"
    if not cas_dir.is_dir() or cas_dir.is_symlink():
        return 0, 0
    live: set[str] = set()
    for mf in prefix_cache_root.rglob("snapshot.vxkv2"):
        if mf.is_file():
            live.update(_vxkv2_referenced_digests(mf))
    removed = 0
    reclaimed = 0
    for page_file in cas_dir.rglob("*.vxpg"):
        if not page_file.is_file() or page_file.is_symlink():
            continue
        if page_file.stem not in live:
            try:
                sz = page_file.lstat().st_size
                page_file.unlink()
                removed += 1
                reclaimed += sz
            except OSError:
                continue
    return removed, reclaimed


def _delete_entry(entry: CacheEntry) -> None:
    if entry.kind == "prefix-cache":
        shutil.rmtree(entry.path, ignore_errors=True)
    else:
        entry.path.unlink(missing_ok=True)
        for ext in (".json", ".tokens"):
            entry.path.with_suffix(ext).unlink(missing_ok=True)


def claim_cache(base: Path, target_bytes: int | None, *, dry_run: bool = False) -> dict:
    """Reclaim at least `target_bytes` (or all when None) oldest-first (LRU)."""
    entries = scan_cache(base)
    total_before = sum(e.size_bytes for e in entries)
    requested = total_before if target_bytes is None else min(max(0, target_bytes), total_before)
    reclaimed = 0
    removed = 0
    touched_roots: set[Path] = set()
    for entry in entries:
        if target_bytes is not None and reclaimed >= requested:
            break
        if not dry_run:
            _delete_entry(entry)
            if entry.kind == "prefix-cache":
                touched_roots.add(entry.path.parent)
        reclaimed += entry.size_bytes
        removed += 1
    if not dry_run:
        for root in touched_roots:
            _n_pages, cas_bytes = gc_cas_pages(root)
            reclaimed += cas_bytes
    remaining = max(0, total_before - min(total_before, reclaimed))
    return {
        "cache_directory": str(base),
        "dry_run": dry_run,
        "requested_bytes": total_before if target_bytes is None else target_bytes,
        "reclaimed_bytes": reclaimed,
        "reclaimed_human": format_bytes(reclaimed),
        "removed_snapshots": removed,
        "remaining_bytes": remaining,
        "remaining_human": format_bytes(remaining),
    }


def _print_status(report: dict) -> None:
    print(f"Veystrix SSD cache: {report['total_human']} ({report['snapshots']} snapshots)")
    print(f"Location:           {report['cache_directory']}")
    print(f"Disk free:          {report['disk_free_human']}")
    if report["by_model"]:
        print("\nBy model:")
        for fam, info in report["by_model"].items():
            print(f"  {fam:<18} {info['human']:>10}  ({info['snapshots']} snapshots)")
    else:
        print("\nNo cached snapshots on disk.")
    print("\nReclaim space: veystrix storage claim <amount>   (e.g. 500MB, 1.5GB, all)")


def cmd_storage(argv: list[str], home: Path, config: Config) -> int:
    parser = argparse.ArgumentParser(
        prog="veystrix storage",
        description="Show SSD prompt-cache storage usage and reclaim space.",
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=("status", "claim", "reclaim", "clear"),
        help="status (default), claim/reclaim <amount>, or clear",
    )
    parser.add_argument(
        "amount",
        nargs="?",
        help="amount to reclaim (e.g. 500MB, 1.5GB, 2G, or all); prompts interactively if omitted",
    )
    parser.add_argument("--dry-run", action="store_true", help="preview reclamation without deleting files")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    base = cache_base_dir(home, config)

    if args.action == "status":
        report = storage_report(base)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            _print_status(report)
        return 0

    if args.action == "clear":
        target: int | None = None
    else:
        raw_amount = args.amount
        if not raw_amount:
            report = storage_report(base)
            if report["total_bytes"] == 0:
                if args.json:
                    print(json.dumps(claim_cache(base, 0, dry_run=args.dry_run), indent=2, sort_keys=True))
                else:
                    print("Veystrix SSD cache is already empty (0 B).")
                return 0
            if not sys.stdin.isatty():
                raise SystemExit("veystrix: specify an amount to claim (e.g. veystrix storage claim 1GB or all)")
            _print_status(report)
            try:
                raw_amount = input("\nHow much cache to claim? (e.g. 500MB, 1.5GB, all, or Enter to cancel): ").strip()
            except EOFError:
                return 0
            if not raw_amount:
                print("Cancelled.")
                return 0
        try:
            target = parse_size(raw_amount)
        except ValueError as exc:
            raise SystemExit(f"veystrix: {exc}") from exc

    result = claim_cache(base, target, dry_run=args.dry_run)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        verb = "Would reclaim" if args.dry_run else "Reclaimed"
        print(
            f"{verb} {result['reclaimed_human']} ({result['removed_snapshots']} snapshots); "
            f"remaining cache: {result['remaining_human']}."
        )
    return 0
