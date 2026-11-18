#!/usr/bin/env python3
"""Strict parity gate: C++ Nemotron 3.5 Lightning decode tool vs Python ref.

Runs `nemotron_generate` against the real Nemotron-H checkpoint and compares
its greedy token ids AND per-step logit SHA-256 digests to the captured Python
reference (gen_nemotron_reference.py). The gate is strict: any mismatch is a
FAIL (exit 1) reporting the first divergent step; it is never weakened to
"tokens only" or "close enough".

Usage:
  .venv/bin/python cpp/engine/tools/check_nemotron_parity.py \
      --binary cpp/engine/build/nemotron_generate \
      --reference results/nemotron35/<ts>-cpp-ref/reference.json \
      --checkpoint /Users/vincent/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit-mlx-community \
      --tokens 4
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

LINE_RE = re.compile(
    r"^step=(\d+)\s+in=(-?\d+)\s+out=(-?\d+)\s+sha256=([0-9a-f]{64})$"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, help="path to nemotron_generate")
    ap.add_argument("--reference", required=True, help="captured reference.json")
    ap.add_argument("--checkpoint", required=True, help="real checkpoint dir")
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--config", default=None,
                    help="flat config; default: <reference-dir>/nemotron_config.txt")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    ref_path = Path(args.reference).resolve()
    reference = json.loads(ref_path.read_text())
    ref_steps = reference.get("steps", [])
    if not ref_steps:
        print("FAIL: reference.json has no steps", file=sys.stderr)
        return 1

    config = args.config
    if config is None:
        config = str(ref_path.parent / "nemotron_config.txt")
    if not Path(config).exists():
        helper = Path(__file__).parent / "gen_nemotron_config.py"
        subprocess.run(
            [sys.executable, str(helper),
             "--checkpoint", args.checkpoint,
             "--reference", str(ref_path),
             "--out", config],
            check=True,
        )

    cmd = [args.binary, args.checkpoint, str(ref_path), str(args.tokens), config]
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True)
    sys.stderr.write(proc.stderr[-4000:])
    if proc.returncode != 0:
        print(f"FAIL: binary exited {proc.returncode}", file=sys.stderr)
        return 1

    got: list[dict] = []
    for line in proc.stdout.splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            if line.strip():
                print(f"WARN: unparsed output line: {line!r}", file=sys.stderr)
            continue
        got.append({"step": int(m.group(1)), "in": int(m.group(2)),
                    "out": int(m.group(3)), "digest": m.group(4)})

    n = min(len(got), len(ref_steps))
    print(f"comparison: {len(got)} C++ steps vs {len(ref_steps)} reference "
          f"steps (requested {args.tokens})")

    token_ok = 0
    digest_ok = 0
    first_token_div = None
    first_digest_div = None
    for i in range(n):
        exp = ref_steps[i]
        g = got[i]
        t_match = g["out"] == int(exp["token"])
        d_match = g["digest"] == exp["logit_sha256"]
        token_ok += int(t_match)
        digest_ok += int(d_match)
        if not t_match and first_token_div is None:
            first_token_div = i
        if not d_match and first_digest_div is None:
            first_digest_div = i
        print(f"step {i:2d}: token {'OK' if t_match else 'DIFF'} "
              f"(got {g['out']:6d}, ref {int(exp['token']):6d})  "
              f"digest {'OK' if d_match else 'DIFF'} "
              f"(got {g['digest'][:12]}, ref {exp['logit_sha256'][:12]})")

    if len(got) < len(ref_steps):
        print(f"FAIL: C++ produced {len(got)} steps, reference has "
              f"{len(ref_steps)}", file=sys.stderr)

    all_ok = (token_ok == len(ref_steps) and digest_ok == len(ref_steps)
              and len(got) == len(ref_steps))
    print("\n" + "=" * 72)
    if all_ok:
        print(f"PASS: {token_ok}/{len(ref_steps)} tokens and "
              f"{digest_ok}/{len(ref_steps)} logit digests match the reference")
        return 0

    print(f"FAIL: token ids {token_ok}/{len(ref_steps)}, "
          f"logit digests {digest_ok}/{len(ref_steps)}")
    if first_token_div is not None:
        s = ref_steps[first_token_div]
        g = got[first_token_div]
        print(f"  first TOKEN divergence at step {first_token_div}: "
              f"C++={g['out']} Python={s['token']}")
    if first_digest_div is not None:
        s = ref_steps[first_digest_div]
        g = got[first_digest_div]
        print(f"  first DIGEST divergence at step {first_digest_div}: "
              f"C++={g['digest']} Python={s['logit_sha256']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
