"""Physical SSD counters around a command (macOS IOBlockStorageDriver stats).

Wrap the benchmark INSIDE the guard:
  real_run_guard.py ... -- python -m veystrix_residency.diskstats --out disk.json \
      --tokens 96 -- <benchmark command>

Reports physical bytes read, read ops, mean read size, and summed read latency.
This separates the two hypotheses the dead-end registry leaves open:
  * mean read size small (< ~256 KB) and ops/token high  -> IOPS-bound: layout/merging lever
  * mean read size large and MB/s near 3 GB/s            -> bandwidth-bound: only fewer bytes help
and gives the PHYSICAL MB/token to compare with logical pool-miss bytes
(physical < logical => the page cache is already serving part of the misses).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from typing import Dict, List

KEYS = ("Bytes (Read)", "Operations (Read)", "Total Time (Read)", "Latency Time (Read)",
        "Bytes (Write)", "Operations (Write)")
_STAT_RX = re.compile(r'"Statistics"\s*=\s*\{([^}]*)\}')
_KV_RX = re.compile(r'"([^"]+)"\s*=\s*(\d+)')


def parse_ioreg(text: str) -> List[Dict[str, int]]:
    out = []
    for m in _STAT_RX.finditer(text):
        d = {k: int(v) for k, v in _KV_RX.findall(m.group(1))}
        if "Bytes (Read)" in d:
            out.append(d)
    return out


def read_counters() -> List[Dict[str, int]]:
    txt = subprocess.run(["ioreg", "-c", "IOBlockStorageDriver", "-r", "-w", "0", "-k", "Statistics"],
                         capture_output=True, text=True, check=True).stdout
    return parse_ioreg(txt)


def delta(before: List[Dict[str, int]], after: List[Dict[str, int]], wall_s: float, tokens: int) -> Dict:
    if len(before) != len(after):
        raise RuntimeError("block-device set changed during the run")
    diffs = [{k: a.get(k, 0) - b.get(k, 0) for k in KEYS} for b, a in zip(before, after)]
    d = max(diffs, key=lambda x: x["Bytes (Read)"])  # the internal SSD does the work
    ops = d["Operations (Read)"]
    by = d["Bytes (Read)"]
    r = {
        "wall_s": wall_s,
        "phys_read_bytes": by,
        "read_ops": ops,
        "mean_read_kb": by / ops / 1024 if ops else None,
        "read_mb_per_s_wall": by / 1e6 / wall_s if wall_s else None,
        "sum_read_time_ms": d["Total Time (Read)"] / 1e6,
        "mean_read_latency_us": d["Total Time (Read)"] / ops / 1e3 if ops else None,
        "all_devices": diffs,
    }
    if tokens:
        r["phys_mb_per_token"] = by / 1e6 / tokens
        r["read_ops_per_token"] = ops / tokens
    return r


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        print("usage: diskstats [--out F] [--tokens N] -- <command...>", file=sys.stderr)
        return 2
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--tokens", type=int, default=0, help="decoded tokens (for per-token figures)")
    a = ap.parse_args(argv[:i])
    cmd = argv[i + 1:]
    before = read_counters()
    t0 = time.monotonic()
    rc = subprocess.call(cmd)
    wall = time.monotonic() - t0
    after = read_counters()
    r = delta(before, after, wall, a.tokens)
    r["command"] = cmd
    r["returncode"] = rc
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=2)
    print(json.dumps({k: v for k, v in r.items() if k != "all_devices"}, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
