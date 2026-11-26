#!/usr/bin/env python3
"""Run the Qwen overlap bit-exact gate for N=2/3/4 under the real-run guard."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DYLIB = os.environ.get("VEYSTRIX_CPP_LIB") or next((str(p) for p in (ROOT / "cpp/engine/build/libveystrix_c.dylib", ROOT / "cpp/engine/build-live/libveystrix_c.dylib") if p.is_file()), str(ROOT / "cpp/engine/build/libveystrix_c.dylib"))
CKPT = "/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
CONFIG = str(ROOT / "results/cpp/20261119-g-append-qwen/real_config.txt")
TEST = str(ROOT / "cpp/engine/capi/test_qwen_overlap_parity.py")
OUT = ROOT / "results/cpp/qwen-overlap-gate"
OUT.mkdir(parents=True, exist_ok=True)

GUARD = [
    sys.executable, str(ROOT / "scripts/flashnext/real_run_guard.py"),
    "--floor-gib", "1.5", "--swap-growth-mib", "5120",
    "--swap-ceiling-mib", "5120", "--compressor-growth-mib", "8192",
    "--thermal-level", "10", "--max-seconds", "1800",
]

def run(N, overlap, steps):
    tag = f"n{N}-{overlap}"
    log = OUT / f"guard_{tag}.jsonl"
    cmd = GUARD + ["--log", str(log), "--"] + [
        sys.executable, TEST, DYLIB, CKPT, CONFIG, str(N), str(steps)]
    env = os.environ.copy()
    env["VEYSTRIX_CPP_POOL"] = "s3fifo"
    env["VEYSTRIX_CPP_QWEN_OVERLAP"] = overlap
    env["VEYSTRIX_CPP_MLX_CACHE_BYTES"] = "536870912"
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    end = [json.loads(l) for l in log.read_text().splitlines()
           if json.loads(l).get("event") == "end"]
    guard = end[-1] if end else {}
    # The guard appends its own "guard: {...}" summary line to stdout; the
    # child's JSON is the line that parses as a list of per-session streams.
    data = None
    for line in reversed(r.stdout.strip().splitlines()):
        try:
            candidate = json.loads(line)
        except Exception:
            continue
        if isinstance(candidate, list):
            data = candidate
            break
    return {
        "rc": r.returncode,
        "guard_stopped": guard.get("stopped"),
        "guard_swapout": guard.get("swapout_mib"),
        "data": data,
        "stderr_tail": r.stderr.strip().splitlines()[-5:],
    }

verdict = {}
for N in (2, 3, 4):
    steps = 4
    serial = run(N, "0", steps)
    batch = run(N, "1", steps)
    ok = False
    detail = ""
    if serial["data"] and batch["data"]:
        ok = serial["data"] == batch["data"]
        detail = "tokens+digests match" if ok else "MISMATCH"
    else:
        detail = f"no data (serial rc={serial['rc']} batch rc={batch['rc']})"
    clean = (serial["guard_stopped"] is None and batch["guard_stopped"] is None)
    verdict[f"N={N}"] = {
        "ok": bool(ok and clean), "detail": detail, "guard_clean": clean,
        "serial": serial, "batch": batch,
    }
    print(f"N={N}: {'PASS' if ok and clean else 'FAIL'} — {detail}", flush=True)

(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
print("VERDICT:", "PASS" if all(v["ok"] for v in verdict.values()) else "FAIL")
