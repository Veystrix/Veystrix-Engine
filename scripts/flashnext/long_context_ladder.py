#!/usr/bin/env python3
"""Drive a single logical session to 65k / 262144 / 1048576 tokens.

Uses the segmented snapshot protocol: one guarded process per bounded chunk,
each restoring the previous snapshot and appending the next chunk. The guard's
5 GiB swap ceiling is therefore measured per segment (fresh process), while the
snapshot SHA chain preserves the exact computation across the whole prompt.

This is the ONLY way to reach >32k on a 16 GB M4 under the hard swap ceiling;
a single S=65536 forward is not possible.

Usage:
  long_context_ladder.py --tokens 65536 --chunk 2048 --out results/cpp/ladder-65k
"""
from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "scripts/flashnext/real_run_guard.py"
DRIVER = ROOT / "cpp/engine/capi/test_long_context.py"
PY = ROOT / ".venv/bin/python"
LIB = os.environ.get("VEYSTRIX_CPP_LIB") or next((str(p) for p in (ROOT / "cpp/engine/build/libveystrix_c.dylib", ROOT / "cpp/engine/build-live/libveystrix_c.dylib") if p.is_file()), str(ROOT / "cpp/engine/build/libveystrix_c.dylib"))
CKPT = Path("/Users/vincent/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit-mlx-community")
CONFIG = ROOT / "results/cpp/20261118-p0-baseline-nemotron/nemotron_config.txt"


def run(mode: str, seg: int, out: Path, tokens: int, chunk: int, context: int) -> dict:
    log = out / f"guard_{mode}{seg}.jsonl"
    env = {
        "LC_MODE": mode, "LC_SEG": str(seg), "LC_CHUNK": str(chunk),
        "VEYSTRIX_CPP_NEMOTRON_CHUNKED": "1",
        "VEYSTRIX_CPP_MLX_CACHE_BYTES": str(512 * 2**20),
        "LC_CONTEXT": str(context),
        "LC_COLD": "65536", "LC_INC": "65536",
    }
    import os
    cmd = [sys.executable, str(GUARD), "--log", str(log),
           "--floor-gib", "1.5", "--swap-growth-mib", "5120",
           "--swap-ceiling-mib", "5120", "--compressor-growth-mib", "8192",
           "--thermal-level", "10", "--max-seconds", "1800", "--",
           str(PY), str(DRIVER), str(LIB), str(CKPT), str(CONFIG),
           str(out / "prompt.json"), str(out / "snap")]
    p = subprocess.run(cmd, cwd=str(ROOT), env={**os.environ, **env},
                       capture_output=True, text=True)
    summary = {"mode": mode, "seg": seg, "rc": p.returncode}
    for line in log.read_text().splitlines():
        d = json.loads(line)
        if d.get("event") == "end":
            summary.update(stopped=d.get("stopped"),
                           swapout=d.get("swapout_mib"), seconds=d.get("seconds"))
    last = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if last:
        summary["result"] = json.loads(last[-1])
    if p.returncode != 0 or summary.get("stopped"):
        summary["stderr"] = p.stderr[-2000:]
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=65536)
    ap.add_argument("--chunk", type=int, default=65536,
                    help="tokens per segment; C++ splits into internal 2048-chunks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cooldown", type=int, default=20)
    ap.add_argument("--context", type=int, default=1048576)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Fixed deterministic prompt, same as other gates. Generate with the venv
    # python (this script may be launched with system python which lacks numpy).
    import subprocess as _sp
    gen = ("import json,numpy as np;"
           f"ids=np.random.default_rng(20261117).integers(0,200000,size={args.tokens}).tolist();"
           "open(__import__('sys').argv[1],'w').write(json.dumps({'prompt_ids':ids,'prompt_tokens':len(ids)}))")
    _sp.run([str(PY), "-c", gen, str(out / "prompt.json")], check=True, cwd=str(ROOT))

    nseg = (args.tokens + args.chunk - 1) // args.chunk
    results = []
    results.append(run("init", 1, out, args.tokens, args.chunk, args.context))
    print("init:", results[-1], flush=True)
    for seg in range(2, nseg):
        time.sleep(args.cooldown)
        results.append(run("step", seg, out, args.tokens, args.chunk, args.context))
        print(f"step {seg}:", results[-1], flush=True)
        if results[-1].get("stopped") or results[-1]["rc"] != 0:
            print("segment failed, stopping ladder", flush=True)
            break
    else:
        time.sleep(args.cooldown)
        results.append(run("final", nseg, out, args.tokens, args.chunk, args.context))
        print("final:", results[-1], flush=True)
    (out / "ladder.json").write_text(json.dumps(results, indent=2) + "\n")
    ok = all(r.get("stopped") is None and r["rc"] == 0 for r in results)
    print("LADDER:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
