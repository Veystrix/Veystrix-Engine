#!/usr/bin/env python3
"""Guarded bit-exact parity runner for ``nemotron_generate``.

Enforces the owner's real-run rules verbatim:

  * before EACH run: wait until no ``real_run_guard`` is active;
  * ``sysctl vm.swapusage`` before each run (logged);
  * 20 s cool-down between runs;
  * every real run wrapped in ``/Users/vincent/Documents/Veystrix/
    scripts/flashnext/real_run_guard.py`` with floor-gib 1.5,
    swap-growth-mib 5120, compressor-growth-mib 8192, thermal-level 2;
  * STOP the whole plan if any guard reports ``stopped`` non-null.

Compares C++ greedy token ids AND per-step float32 logit SHA-256 digests to a
captured Python reference, exactly (never "tokens only").

Usage:
  .venv/bin/python cpp/engine/tools/run_nemo_long_gate.py --plan plan.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

VEYR_DIR = Path("/Users/vincent/Documents/Veystrix")
GUARD = VEYR_DIR / "scripts/flashnext/real_run_guard.py"
PYTHON = VEYR_DIR / ".venv/bin/python"
FLOOR_GIB = 1.5
SWAP_GROWTH_MIB = 5120
COMPRESSOR_GROWTH_MIB = 8192
THERMAL_LEVEL = 2
COOLDOWN_S = 20.0
STEP_RE = re.compile(r"^step=(\d+)\s+in=(-?\d+)\s+out=(-?\d+)\s+sha256=([0-9a-f]{64})$")
GUARD_RE = re.compile(r"^guard: (\{.*\})$")


def wait_for_guards() -> None:
    while True:
        out = subprocess.run(
            "ps aux | grep real_run_guard | grep -v grep",
            shell=True, capture_output=True, text=True,
        ).stdout.strip()
        if not out:
            print("[serialize] no other guarded run active", flush=True)
            return
        print(f"[serialize] another guarded run is active — waiting:\n{out}", flush=True)
        time.sleep(10)


def sysctl_before(label: str, result_dir: Path) -> str:
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True,
                         text=True, check=True).stdout.strip()
    line = f"{datetime.now(timezone.utc).isoformat()} [{label}] {out}"
    with (result_dir / "sysctl_before.log").open("a") as fh:
        fh.write(line + "\n")
    print(f"[swap-before] {label}: {out}", flush=True)
    return out


def run_one(spec: dict, result_dir: Path, guard_log: Path, raw_log: Path) -> dict:
    label = spec["label"]
    cmd = [spec["binary"], spec["checkpoint"], spec["reference"],
           str(spec["tokens"]), spec["config"]]
    env = os.environ.copy()
    env.pop("VEYSTRIX_NEMOTRON_KV", None)
    if spec.get("kv") == "bf16":
        env["VEYSTRIX_NEMOTRON_KV"] = "bf16"
    env.pop("VEYSTRIX_CPP_PROFILE", None)  # profile off: digests stay ON
    env.pop("VEYSTRIX_CPP_NO_DIGEST", None)
    for k, v in spec.get("env", {}).items():
        env[k] = str(v)

    guard_cmd = [
        str(PYTHON), str(GUARD),
        "--floor-gib", str(FLOOR_GIB),
        "--swap-growth-mib", str(SWAP_GROWTH_MIB),
        "--compressor-growth-mib", str(COMPRESSOR_GROWTH_MIB),
        "--thermal-level", str(THERMAL_LEVEL),
        "--max-seconds", str(spec.get("budget", 900)),
        "--log", str(guard_log), "--",
    ] + cmd
    print(f"[guard] {label}: $ {shlex.join(guard_cmd)}", flush=True)

    t0 = time.monotonic()
    with raw_log.open("w") as raw:
        proc = subprocess.Popen(guard_cmd, cwd=str(VEYR_DIR), env=env,
                                stdout=raw, stderr=subprocess.STDOUT, text=True)
        rc = proc.wait()
    ended = time.monotonic()

    summary = None
    got = []
    for line in raw_log.read_text().splitlines():
        m = GUARD_RE.match(line.strip())
        if m:
            summary = json.loads(m.group(1))
        m = STEP_RE.match(line.strip())
        if m:
            got.append({"step": int(m.group(1)), "out": int(m.group(3)),
                        "digest": m.group(4)})
    if summary is None:
        rows = [json.loads(x) for x in guard_log.read_text().splitlines() if x.strip()]
        summary = next((r for r in reversed(rows) if r.get("event") == "end"), None)
    if summary is None:
        summary = {"stopped": "guard summary missing", "returncode": rc}

    expected = json.loads(Path(spec["reference"]).read_text())["steps"]
    n = min(len(got), len(expected))
    tok_ok = dig_ok = 0
    first = None
    for i in range(n):
        t_ok = got[i]["out"] == int(expected[i]["token"])
        d_ok = got[i]["digest"] == expected[i]["logit_sha256"]
        tok_ok += int(t_ok)
        dig_ok += int(d_ok)
        if first is None and not (t_ok and d_ok):
            first = {"step": i, "cpp_token": got[i]["out"],
                     "python_token": int(expected[i]["token"]),
                     "cpp_digest": got[i]["digest"],
                     "python_digest": expected[i]["logit_sha256"]}
    total = len(expected)
    passed = (tok_ok == total and dig_ok == total and len(got) == total)
    out = {
        "label": label, "kv": spec.get("kv", "int8"), "passed": passed,
        "tokens_matched": f"{tok_ok}/{total}", "digests_matched": f"{dig_ok}/{total}",
        "cpp_steps": len(got), "reference_steps": total, "first_divergence": first,
        "guard": {"stopped": summary.get("stopped"), "returncode": rc,
                  "seconds": round(ended - t0, 1),
                  "swap_growth_mib": summary.get("swap_peak_mib", 0) - summary.get("swap_baseline_mib", 0)},
        "guard_log": str(guard_log), "raw_log": str(raw_log),
        "cpp_tokens": [g["out"] for g in got],
        "cpp_digests": [g["digest"] for g in got],
    }
    print(f"[result] {label}: kv={out['kv']} passed={passed} "
          f"tokens={out['tokens_matched']} digests={out['digests_matched']} "
          f"stopped={summary.get('stopped')!r}", flush=True)
    if first is not None:
        print(f"[result] {label}: first divergence {json.dumps(first)}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="JSON list of run specs")
    ap.add_argument("--out", required=True, help="result JSON path")
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    result_dir = Path(args.out).resolve().parent
    result_dir.mkdir(parents=True, exist_ok=True)
    results = []
    last_end = None
    for i, spec in enumerate(plan):
        wait_for_guards()
        if last_end is not None:
            wait = COOLDOWN_S - (time.monotonic() - last_end)
            if wait > 0:
                print(f"[cooldown] {wait:.1f}s", flush=True)
                time.sleep(wait)
        sysctl_before(spec["label"], result_dir)
        label = spec["label"]
        rec = run_one(spec, result_dir,
                      result_dir / f"guard_{label}.jsonl",
                      result_dir / f"raw_{label}.log")
        last_end = time.monotonic()
        results.append(rec)
        if rec["guard"]["stopped"] is not None:
            print("[guard] STOP NON-NULL — ABORTING PLAN", flush=True)
            (result_dir / "guard_stop.json").write_text(
                json.dumps({"label": label, "guard": rec["guard"]}, indent=2) + "\n")
            Path(args.out).write_text(json.dumps({"results": results}, indent=2) + "\n")
            return 90
    Path(args.out).write_text(json.dumps({"results": results}, indent=2) + "\n")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
