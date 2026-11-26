#!/usr/bin/env python3
"""Pre-gate for cross-request GPU/SSD overlap (Sonnet plan step 6).

Reads a VEYSTRIX_CPP_PROFILE profile.jsonl (decode steps only) and computes the
speedup bound for overlapping one session's SSD reads with ANOTHER session's
host build + GPU work on the single driver thread:

  G = GPU wait      = final_eval_ms + moe_eval_ms
  S = SSD read wait = read_wall_ms
  H = host work     = cpu_build_ms + route_sync_ms + sample_ms + digest_ms
  sequential        = G + S + H
  overlapped        = max(S, H + G)          # B's reads || A's host+GPU
  bound             = sequential / overlapped

Sonnet's narrower (G+S+H)/(max(G,S)+H) assumes host work is serial; for N>=2
it can overlap another session's reads, so we also report that conservative
bound. Build the overlap path ONLY when the primary bound clears the gate
(default 1.15x) on real profile data.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def step_parts(row: dict) -> tuple[float, float, float]:
    g = float(row.get("final_eval_ms", 0)) + float(row.get("moe_eval_ms", 0))
    s = float(row.get("read_wall_ms", 0))
    h = (float(row.get("cpu_build_ms", 0)) + float(row.get("route_sync_ms", 0))
         + float(row.get("sample_ms", 0)) + float(row.get("digest_ms", 0)))
    return g, s, h


def bound(g: float, s: float, h: float) -> float:
    seq = g + s + h
    overlap = max(s, h + g)
    return seq / overlap if overlap > 0 else 1.0


def conservative_bound(g: float, s: float, h: float) -> float:
    seq = g + s + h
    serial = max(g, s) + h
    return seq / serial if serial > 0 else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", help="profile.jsonl (or a dir containing it)")
    ap.add_argument("--gate", type=float, default=1.15)
    args = ap.parse_args()
    p = Path(args.profile)
    if p.is_dir():
        p = p / "profile.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    # step 1 is load+prefill; exclude it (cpp_profile.hpp warning).
    decode = [r for r in rows if int(r.get("step", 0)) > 1 and
              float(r.get("read_wall_ms", 0)) < 100000]
    if not decode:
        print("no decode steps in profile")
        return 2
    parts = [step_parts(r) for r in decode]
    b = [bound(*x) for x in parts]
    cb = [conservative_bound(*x) for x in parts]
    med, cmed = statistics.median(b), statistics.median(cb)
    print(f"decode steps={len(parts)}")
    print(f"median G/S/H ms = {statistics.median(x[0] for x in parts):.2f} / "
          f"{statistics.median(x[1] for x in parts):.2f} / "
          f"{statistics.median(x[2] for x in parts):.2f}")
    print(f"overlap bound (cross-session)   median={med:.3f}  gate={args.gate}")
    print(f"conservative bound (Sonnet)     median={cmed:.3f}")
    verdict = "BUILD" if med >= args.gate else "NO-GO"
    print(f"VERDICT: {verdict}")
    return 0 if verdict == "BUILD" else 1


if __name__ == "__main__":
    raise SystemExit(main())
