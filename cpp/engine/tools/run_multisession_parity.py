#!/usr/bin/env python3
"""OWNER-RUN end-to-end real-checkpoint multi-session bit-identity gate.

For N sessions with DISTINCT prompts this script:
  1. writes N flat configs from the real checkpoint config (one base config is
     generated once with gen_real_config.py, then only prompt_ids /
     prompt_tokens / prompt_sha256 are replaced per session),
  2. runs EACH session SOLO via real_checkpoint_generate -- under
     scripts/flashnext/real_run_guard.py -- and records its per-step
     (token, logit SHA-256) stream,
  3. runs the concurrent multisession_real tool -- also under the guard -- with
     all N sessions interleaved on one thread,
  4. asserts that EVERY session's tokens AND digests match its SOLO stream
     exactly. Any mismatch is a hard FAIL reported with the first divergence.

Every real-checkpoint run goes through the guard from /Users/vincent/Documents/Veystrix.
`sysctl vm.swapusage` is checked before each run (the ~3.4 GiB WebKit baseline is
expected and OK). If the guard STOPS a run (non-null `stopped` / exit 90), the
whole harness stops immediately -- no further real runs are attempted. Cool-down
gaps are inserted between runs so the machine can shed heat.

No git commands. No files are written outside the worktree / an owned results
directory. This script never weakens the gate to "tokens only".

Usage (from anywhere; runs are executed with cwd=/Users/vincent/Documents/Veystrix):
  .venv/bin/python cpp/engine/tools/run_multisession_parity.py \
      --sessions 2 --tokens 4 --verify-dir /Users/vincent/Documents/Veystrix/results/flashnext/ms-parity
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

VEYSTRIX = Path("/Users/vincent/Documents/Veystrix")
DEFAULT_CKPT = Path("/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP")
DEFAULT_REF = VEYSTRIX / "results/flashnext/20261117-222244-real-reference/reference.json"
GUARD = VEYSTRIX / "scripts/flashnext/real_run_guard.py"
GEN_CONFIG = Path(__file__).resolve().parent / "gen_real_config.py"

SOLO_LINE = re.compile(
    r"^step=(\d+)\s+in=(-?\d+)\s+out=(-?\d+)\s+sha256=([0-9a-f]{64})$")
MULTI_LINE = re.compile(
    r"^session=(\d+)\s+step=(\d+)\s+token=(-?\d+)\s+sha256=([0-9a-f]{64})$")


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--reference", default=str(DEFAULT_REF))
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--prompt-len", type=int, default=16)
    ap.add_argument("--prompt-seed-base", type=int, default=20261118)
    ap.add_argument("--solo-binary", default="/tmp/real_gen_ms")
    ap.add_argument("--multi-binary", default="/tmp/multisession_real")
    ap.add_argument("--verify-dir", default=None,
                    help="output directory (default: results/flashnext/ms-real-parity-<ts>)")
    ap.add_argument("--budget", type=int, default=900,
                    help="per-run --max-seconds passed to the guard")
    ap.add_argument("--cooldown", type=int, default=25,
                    help="seconds of cool-down between real runs")
    ap.add_argument("--floor-gib", type=float, default=1.5)
    a = ap.parse_args()
    if a.verify_dir is None:
        a.verify_dir = str(VEYSTRIX / "results/flashnext" /
                           ("ms-real-parity-" + time.strftime("%Y%m%d-%H%M%S")))
    return a


def check_swap() -> None:
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True,
                         text=True, check=True).stdout.strip()
    log("[preflight] " + out + "  (baseline ~3.4 GiB WebKit swap is OK)")


def guard_summary(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith("guard: {"):
            try:
                return json.loads(line[len("guard: "):])
            except json.JSONDecodeError:
                return None
    return None


def run_guarded(name: str, cmd: list[str], log_path: Path,
                budget: int, floor_gib: float, cooldown: int) -> subprocess.CompletedProcess:
    check_swap()
    guard_cmd = [
        sys.executable, str(GUARD),
        "--log", str(log_path),
        "--floor-gib", str(floor_gib),
        "--swap-growth-mib", "5120",
        "--compressor-growth-mib", "8192",
        "--thermal-level", "2",
        "--max-seconds", str(budget),
        "--",
    ] + [str(x) for x in cmd]
    log(f"\n[$ {name}] " + " ".join(guard_cmd))
    began = time.time()
    proc = subprocess.run(guard_cmd, cwd=str(VEYSTRIX), text=True,
                          capture_output=True)
    (log_path.with_suffix(".stdout")).write_text(proc.stdout)
    (log_path.with_suffix(".stderr")).write_text(proc.stderr)
    log(f"[{name}] returncode={proc.returncode} elapsed={time.time()-began:.1f}s")
    summary = guard_summary(proc.stdout)
    if summary is not None:
        log(f"[{name}] guard summary: {json.dumps(summary)}")
    if proc.returncode == 90 or (summary and summary.get("stopped")):
        raise RuntimeError(f"GUARD STOPPED {name}: {summary}")
    if proc.returncode == 91:
        raise RuntimeError(f"GUARD REFUSED {name}: {proc.stderr.strip()[-500:]}")
    if cooldown:
        log(f"[cooldown] {cooldown}s")
        time.sleep(cooldown)
    return proc


def make_configs(args: argparse.Namespace, out: Path) -> list[Path]:
    import numpy as np

    base = out / "base_config.txt"
    if not base.exists():
        gen = [
            sys.executable, str(GEN_CONFIG),
            "--checkpoint", args.checkpoint,
            "--reference", args.reference,
            "--out", str(base),
        ]
        subprocess.run(gen, cwd=str(VEYSTRIX), check=True)
    text = base.read_text()
    cfg = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            cfg[k] = v
    vocab_bound = min(200000, int(cfg.get("vocab", 200000)))
    paths: list[Path] = []
    for i in range(args.sessions):
        rng = np.random.default_rng(args.prompt_seed_base + i)
        prompt = rng.integers(0, vocab_bound, size=args.prompt_len).tolist()
        sha = hashlib.sha256(
            b"".join(int(t).to_bytes(4, "little", signed=True) for t in prompt)
        ).hexdigest()
        c = dict(cfg)
        c["prompt_ids"] = ",".join(str(t) for t in prompt)
        c["prompt_tokens"] = str(len(prompt))
        c["prompt_sha256"] = sha
        p = out / f"session_{i:03d}.config.txt"
        p.write_text("".join(f"{k}={v}\n" for k, v in c.items()))
        paths.append(p)
        log(f"[config] session {i}: {len(prompt)} tokens, sha={sha[:12]} prompt={prompt}")
    return paths


def main() -> int:
    args = parse_args()
    out = Path(args.verify_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log(f"verify dir: {out}")
    for binary in (args.solo_binary, args.multi_binary):
        if not Path(binary).exists():
            log(f"FAIL: binary missing: {binary}")
            return 1

    configs = make_configs(args, out)
    solo_streams: list[list[dict]] = []
    try:
        # ---- Step 1: each session SOLO via the real_checkpoint_generate binary.
        for i, cfg_path in enumerate(configs):
            cmd = [args.solo_binary, args.checkpoint, args.reference,
                   str(args.tokens), str(cfg_path)]
            proc = run_guarded(f"solo-session-{i}", cmd,
                               out / f"solo_{i:03d}_guard.jsonl",
                               args.budget, args.floor_gib, args.cooldown)
            if proc.returncode != 0:
                log(f"FAIL: solo session {i} exited {proc.returncode}")
                return 1
            records = []
            for line in proc.stdout.splitlines():
                m = SOLO_LINE.match(line.strip())
                if m:
                    records.append({"step": int(m.group(1)),
                                    "token": int(m.group(3)),
                                    "sha256": m.group(4)})
            if len(records) != args.tokens:
                log(f"FAIL: solo session {i} produced {len(records)} steps, "
                    f"expected {args.tokens}")
                return 1
            solo_streams.append(records)
            log(f"[solo] session {i}: {len(records)} steps, "
                f"first digest={records[0]['sha256']}")

        # ---- Step 2: all sessions CONCURRENT via multisession_real.
        cmd = [args.multi_binary, args.checkpoint, str(args.sessions),
               str(args.tokens)] + [str(c) for c in configs]
        proc = run_guarded("multisession", cmd,
                           out / "multisession_guard.jsonl",
                           args.budget, args.floor_gib, 0)
        if proc.returncode != 0:
            log(f"FAIL: multisession tool exited {proc.returncode}")
            return 1
        multi: list[dict] = []
        for line in proc.stderr.splitlines():
            m = MULTI_LINE.match(line.strip())
            if m:
                multi.append({"session": int(m.group(1)),
                              "step": int(m.group(2)),
                              "token": int(m.group(3)),
                              "sha256": m.group(4)})
        log(f"[multi] parsed {len(multi)} step lines")

        # ---- Step 3: strict per-session comparison.
        ok = True
        for i in range(args.sessions):
            got = [r for r in multi if r["session"] == i]
            got.sort(key=lambda r: r["step"])
            exp = solo_streams[i]
            if len(got) != len(exp):
                log(f"FAIL session {i}: produced {len(got)} steps, "
                    f"solo has {len(exp)}")
                ok = False
                continue
            for s, (g, e) in enumerate(zip(got, exp)):
                if g["step"] != s or g["token"] != e["token"] or \
                        g["sha256"] != e["sha256"]:
                    ok = False
                    log(f"FAIL session {i} step {s}: "
                        f"multi(token={g['token']}, sha256={g['sha256']}) "
                        f"!= solo(token={e['token']}, sha256={e['sha256']})")
                    break
            else:
                log(f"PASS session {i}: {len(exp)}/{len(exp)} tokens and digests "
                    f"bit-identical to solo (step0 sha256={exp[0]['sha256']})")

        summary = {
            "sessions": args.sessions, "tokens": args.tokens,
            "prompt_seed_base": args.prompt_seed_base,
            "prompt_len": args.prompt_len, "pass": ok,
            "solo_digests": [[r["sha256"] for r in s] for s in solo_streams],
            "multi_digests": [[r["sha256"] for r in sorted(
                (x for x in multi if x["session"] == i), key=lambda x: x["step"])]
                for i in range(args.sessions)],
        }
        (out / "verdict.json").write_text(json.dumps(summary, indent=2) + "\n")
        log("\n" + "=" * 72)
        if ok:
            log(f"PASS: all {args.sessions} sessions bit-identical to SOLO "
                f"({args.tokens} tokens each).")
            return 0
        log("FAIL: at least one session diverged from SOLO; see first divergence above.")
        return 1
    except RuntimeError as e:
        log("STOP: " + str(e))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
