#!/usr/bin/env python3
"""Decode-only Qwen overlap A/B (ABBA), guarded. Reads the decode_s line."""
import json, os, statistics, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DYLIB = os.environ.get("VEYSTRIX_CPP_LIB") or next((str(p) for p in (ROOT / "cpp/engine/build/libveystrix_c.dylib", ROOT / "cpp/engine/build-live/libveystrix_c.dylib") if p.is_file()), str(ROOT / "cpp/engine/build/libveystrix_c.dylib"))
CKPT = "/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
CONFIG = str(ROOT / "results/cpp/20261119-g-append-qwen/real_config.txt")
TEST = str(ROOT / "cpp/engine/capi/test_qwen_overlap_parity.py")
OUT = ROOT / "results/cpp/qwen-overlap-decode-ab"; OUT.mkdir(parents=True, exist_ok=True)
GUARD = [sys.executable, str(ROOT / "scripts/flashnext/real_run_guard.py"),
         "--floor-gib", "1.5", "--swap-growth-mib", "5120", "--swap-ceiling-mib", "5120",
         "--compressor-growth-mib", "8192", "--thermal-level", "10", "--max-seconds", "900"]

def arm(label, overlap, steps=96):
    log = OUT / f"guard_{label}.jsonl"
    cmd = GUARD + ["--log", str(log), "--"] + [sys.executable, TEST, DYLIB, CKPT, CONFIG, "2", str(steps)]
    env = os.environ.copy(); env.update(VEYSTRIX_CPP_POOL="s3fifo", VEYSTRIX_CPP_QWEN_OVERLAP=overlap,
                                        VEYSTRIX_CPP_MLX_CACHE_BYTES="536870912")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    timing = None
    for line in reversed(r.stdout.strip().splitlines()):
        try: d = json.loads(line)
        except Exception: continue
        if isinstance(d, dict) and "decode_s" in d: timing = d; break
    end = [json.loads(l) for l in log.read_text().splitlines() if json.loads(l).get("event") == "end"]
    stopped = end[-1].get("stopped") if end else "?"
    time.sleep(12)
    return {"label": label, "mode": overlap, "decode_s": timing["decode_s"] if timing else None, "stopped": stopped}

arms = [arm("1","0"), arm("2","1"), arm("3","1"), arm("4","0")]
ds = {a["mode"]: a["decode_s"] for a in arms}
pairs = [round(arms[0]["decode_s"]/arms[1]["decode_s"],4), round(arms[3]["decode_s"]/arms[2]["decode_s"],4)]
verdict = {"decode_s": {a["label"]: a["decode_s"] for a in arms}, "pair_speedups": pairs,
           "guard_clean": all(a["stopped"] is None for a in arms), "arms": arms}
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
print(json.dumps({k:v for k,v in verdict.items() if k != "arms"}, indent=2))
for a in arms: print(a["label"], "mode", a["mode"], "decode_s", a["decode_s"], "stopped", a["stopped"])
