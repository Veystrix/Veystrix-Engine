#!/usr/bin/env python3
"""Guarded Qwen overlap 3-pair A/B (ABBAAB): serial vs batched throughput.

Each arm runs under real_run_guard with the owner's flags. N=2 sessions,
configurable decode steps. Emitted token+digest streams must be byte-identical
across every arm. Median A (serial) vs B (overlap) and per-pair speedups are
reported; the design gate is >=+10% on every pair with byte streams within 2%.
"""
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DYLIB = os.environ.get("VEYSTRIX_CPP_LIB") or next((str(p) for p in (ROOT / "cpp/engine/build/libveystrix_c.dylib", ROOT / "cpp/engine/build-live/libveystrix_c.dylib") if p.is_file()), str(ROOT / "cpp/engine/build/libveystrix_c.dylib"))
CKPT = "/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
CONFIG = str(ROOT / "results/cpp/20261119-g-append-qwen/real_config.txt")
TEST = str(ROOT / "cpp/engine/capi/test_qwen_overlap_parity.py")
OUT = ROOT / "results/cpp/qwen-overlap-ab"
OUT.mkdir(parents=True, exist_ok=True)
N = 2
STEPS = 24
COOLDOWN = 15

GUARD = [
    sys.executable, str(ROOT / "scripts/flashnext/real_run_guard.py"),
    "--floor-gib", "1.5", "--swap-growth-mib", "5120",
    "--swap-ceiling-mib", "5120", "--compressor-growth-mib", "8192",
    "--thermal-level", "10", "--max-seconds", "900",
]


def arm(label, overlap):
    log = OUT / f"guard_{label}.jsonl"
    cmd = GUARD + ["--log", str(log), "--"] + [
        sys.executable, TEST, DYLIB, CKPT, CONFIG, str(N), str(STEPS)]
    env = os.environ.copy()
    env.update(VEYSTRIX_CPP_POOL="s3fifo",
               VEYSTRIX_CPP_QWEN_OVERLAP=overlap,
               VEYSTRIX_CPP_MLX_CACHE_BYTES="536870912")
    t0 = time.monotonic()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.monotonic() - t0
    end = [json.loads(l) for l in log.read_text().splitlines()
           if json.loads(l).get("event") == "end"]
    guard = end[-1] if end else {}
    stream = None
    for line in reversed(r.stdout.strip().splitlines()):
        try:
            cand = json.loads(line)
        except Exception:
            continue
        if isinstance(cand, list):
            stream = cand
            break
    time.sleep(COOLDOWN)
    return {"label": label, "mode": overlap, "rc": r.returncode,
            "guard_stopped": guard.get("stopped"), "guard_secs": guard.get("seconds"),
            "wall": round(wall, 2), "stream": stream}


# ABBAAB
order = [("1", "0"), ("2", "1"), ("3", "1"), ("4", "0"), ("5", "0"), ("6", "1")]
arms = [arm(lbl, mode) for lbl, mode in order]
streams = [json.dumps(a["stream"], sort_keys=True) for a in arms]
byte_identical = all(s == streams[0] for s in streams)

A = [a for a in arms if a["mode"] == "0"]
B = [a for a in arms if a["mode"] == "1"]
medA = statistics.median(a["guard_secs"] for a in A)
medB = statistics.median(b["guard_secs"] for b in B)
speedup = medA / medB
# pairs: (arm1 vs arm2), (arm4 vs arm3... careful with ABBAAB ordering)
pair_specs = [(0, 1), (3, 2), (4, 5)]
pairs = []
for ai, bi in pair_specs:
    a, b = arms[ai], arms[bi]
    pairs.append(round(a["guard_secs"] / b["guard_secs"], 4))

verdict = {
    "byte_identical": byte_identical,
    "median_serial_s": medA, "median_overlap_s": medB,
    "median_speedup": round(speedup, 4),
    "pair_speedups": pairs,
    "all_pairs_ge_10pct": all(x >= 1.10 for x in pairs),
    "guard_clean": all(a["guard_stopped"] is None for a in arms),
    "arms": arms,
}
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
print(json.dumps({k: v for k, v in verdict.items() if k != "arms"}, indent=2))
for a in arms:
    print(a["label"], "mode", a["mode"], "sec", a["guard_secs"],
          "stopped", a["guard_stopped"])
