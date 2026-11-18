#!/usr/bin/env python3
"""Check the opt-in MTP decode loop on a real checkpoint.

Two checks, deliberately separated:

  (1) LOSSLESSNESS (claimable): C++ with VEYSTRIX_CPP_MTP=1 must emit exactly
      the same token ids AND logit digests as the same binary with MTP unset.
      mtp_verify is lossless, so a wrong draft must be replaced by a redraw
      from the current target row, never silently committed.

  (2) PYTHON NATIVE-MTP COMPARISON (reported, NOT claimed bit-exact): token ids
      are compared to a reference captured with FLASHNEXT_NATIVE_MTP=on. The
      real-checkpoint C++ generator already differs from Python in bf16 mode
      (PLE, quantized-vs-dense matmul, unfused RoPE, ...), and the C++ MTP head
      is a deterministic synthetic draft head (the read-only MTP primitive
      needs unpacked, tied-embedding weights). A token mismatch here is an
      honest limitation, not a regression to hide.

Usage:
  .venv/bin/python cpp/engine/tools/check_mtp_real.py \
      --baseline /tmp/p1runs/baseline.out --mtp /tmp/p1runs/mtp.out \
      [--python-ref /tmp/p1runs/py/mtp.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

LINE = re.compile(r"^step=(\d+)\s+in=(-?\d+)\s+out=(-?\d+)\s+sha256=([0-9a-f]{64})$")


def parse(path: str):
    steps = []
    for line in Path(path).read_text().splitlines():
        m = LINE.match(line.strip())
        if m:
            steps.append((int(m.group(1)), int(m.group(2)),
                          int(m.group(3)), m.group(4)))
    return steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--python-ref", default=None)
    args = ap.parse_args()

    base = parse(args.baseline)
    mtp = parse(args.mtp)
    n = min(len(base), len(mtp))
    print(f"comparison: {len(base)} baseline steps vs {len(mtp)} MTP steps")
    if not n:
        print("FAIL: no steps parsed", file=sys.stderr)
        return 1

    tok_ok = dig_ok = 0
    for i in range(n):
        b, m = base[i], mtp[i]
        t = b[2] == m[2]
        d = b[3] == m[3]
        tok_ok += int(t)
        dig_ok += int(d)
        print(f"step {i:2d}: token {'OK' if t else 'DIFF'} "
              f"(baseline {b[2]}, mtp {m[2]})  "
              f"digest {'OK' if d else 'DIFF'} ({b[3][:12]} vs {m[3][:12]})")
    lossless = tok_ok == n and dig_ok == n and len(base) == len(mtp)
    print("\n" + "=" * 72)
    print(f"LOSSLESS: {'PASS' if lossless else 'FAIL'} — "
          f"tokens {tok_ok}/{n}, digests {dig_ok}/{n} "
          f"(MTP output == plain greedy output)")
    rc = 0 if lossless else 1

    if args.python_ref:
        ref = json.loads(Path(args.python_ref).read_text())
        # Native-MTP captures have no sampler logit digests, so `steps` is
        # empty; the on_decode_token ids are the only token record.
        rtoks = [int(s["token"]) for s in ref.get("steps", [])]
        if not rtoks:
            rtoks = [int(t) for t in ref.get("token_ids", [])]
        mtoks = [m[2] for m in mtp]
        k = min(len(rtoks), len(mtoks))
        agree = sum(int(rtoks[i] == mtoks[i]) for i in range(k))
        print(f"PYTHON NATIVE MTP (reported, not claimed bit-exact): "
              f"C++ MTP vs Python token agreement {agree}/{k} "
              f"(C++ {mtoks[:k]}, Python {rtoks[:k]})")
        print("  note: C++ already diverges from Python in bf16 mode; the "
              "synthetic MTP head is not the checkpoint's MTP weights")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
