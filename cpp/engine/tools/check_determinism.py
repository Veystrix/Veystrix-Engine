#!/usr/bin/env python3
"""Run the synthetic determinism probe and verify every op's 5 hashes match.

This wraps cpp/engine/tools/determinism_probe.cpp. It never loads the 105 GB
checkpoint: all inputs are synthetic with the exact shapes/dtypes of the real
lm_head / MoE / GDN / SDPA ops, so each mode finishes in seconds.

Usage:
  python3 cpp/engine/tools/check_determinism.py --probe /tmp/det_probe
  python3 cpp/engine/tools/check_determinism.py --probe /tmp/det_probe --mode qmm
"""
import argparse
import re
import subprocess
import sys

MODES = ("qmm", "dqmm", "gdn", "sdpa", "router", "gqmm", "nan", "race",
         "gdnsub", "reduce")

# A "run" line is either [tag] runN sha256=... or [gdn] runN sha256=...
RUN = re.compile(r"^\[([^\]]+)\]\s+run(\d+)\b.*?sha256=([0-9a-f]{64})")
# gdn prints [gdn] runN sha256=... too, so the same regex covers it.
ARGMAX = re.compile(r"^argmax-nan run(\d+) idx=(\d+)")


def check(probe: str, mode: str) -> bool:
    out = subprocess.run([probe, mode], capture_output=True, text=True, check=True).stdout
    groups: dict[tuple[str, str], list[str]] = {}
    for line in out.splitlines():
        m = RUN.match(line)
        if m:
            groups.setdefault((mode, m.group(1)), []).append(m.group(3))
        a = ARGMAX.match(line)
        if a:
            groups.setdefault((mode, "argmax-nan"), []).append("idx=" + a.group(2))

    ok = True
    for (_, tag), hashes in groups.items():
        unique = sorted(set(hashes))
        verdict = "DETERMINISTIC" if len(unique) == 1 else "NONDETERMINISTIC"
        if len(unique) != 1:
            ok = False
        print(f"[{mode}/{tag}] {len(hashes)} runs, {len(unique)} distinct hashes -> {verdict}")
        for h in unique:
            print(f"    {h}")
    if not groups:
        print(f"[{mode}] no run lines parsed", file=sys.stderr)
        return False
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--probe", default="/tmp/det_probe")
    p.add_argument("--mode", choices=MODES + ("all",), default="all")
    a = p.parse_args()
    modes = MODES if a.mode == "all" else (a.mode,)
    all_ok = True
    for m in modes:
        all_ok &= check(a.probe, m)
    print("\nRESULT:", "ALL DETERMINISTIC" if all_ok else "NONDETERMINISM DETECTED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
