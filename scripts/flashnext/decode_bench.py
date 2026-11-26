#!/usr/bin/env python3
"""Decode tok/s bench for the C++ engine through the real server (model_router).

Starts ``veystrix.model_router`` with VEYSTRIX_ENGINE=cpp, optional speed-knob env overrides,
sends N identical greedy requests and reports the engine's own ``decode_tokens_per_second``
from the flight recorder (not SSE chunk counts).

  .venv/bin/python scripts/flashnext/decode_bench.py --model ornith --runs 3 \
      --env VEYSTRIX_GENERIC_PIPE=1 --env VEYSTRIX_CPP_KEEPWARM=1
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = {
    "ornith": "~/models/Ornith-1.5-35B-A3B-MLX-4bit",
    "flashnext": "~/models/Qwen3.8-Flash-Next-MLX-4bit-MTP",
}
PROMPT = ("Write a detailed, step-by-step explanation of how a hash map handles collisions, "
          "with a short Python example.")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(port: int, proc: subprocess.Popen, timeout: float) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            raise SystemExit(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if json.loads(r.read() or b"{}").get("status") in ("ok", "ready"):
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise SystemExit("server did not become ready")


def request(port: int, model: str, tokens: int) -> None:
    body = json.dumps({"model": f"veystrix-{model}", "max_tokens": tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": PROMPT}], "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(CHECKPOINTS), default="ornith")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE override for the engine")
    ap.add_argument("--kv", default="")
    ap.add_argument("--timeout", type=float, default=900)
    args = ap.parse_args()

    port = free_port()
    env = os.environ.copy()
    env["VEYSTRIX_ENGINE"] = "cpp"
    env.setdefault("MallocLargeCache", "0")
    if args.kv:
        env["VEYSTRIX_KV_MODE"] = args.kv
    for item in args.env:
        k, _, v = item.partition("=")
        env[k] = v
    python = str(ROOT / ".venv" / "bin" / "python")
    cmd = [python, "-u", "-m", "veystrix.model_router", "--host", "127.0.0.1", "--port", str(port),
           "--model", args.model, "--checkpoint", str(Path(CHECKPOINTS[args.model]).expanduser())]
    log = Path(os.environ.get("TMPDIR", "/tmp")) / f"decode-bench-{port}.log"
    with log.open("w") as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    try:
        wait_ready(port, proc, args.timeout)
        request(port, args.model, 8)  # warm-up
        for _ in range(args.runs):
            request(port, args.model, args.tokens)
    finally:
        proc.send_signal(2)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    recorder = Path("~/.flashnext/logs").expanduser() / f"server-{proc.pid}.jsonl"
    rates = []
    for line in recorder.read_text().splitlines():
        d = json.loads(line)
        if d.get("event") == "generate_end" and int(d.get("completion_tokens", 0)) >= args.tokens // 2:
            rates.append(float(d["decode_tokens_per_second"]))
    speed = next((l.strip() for l in log.read_text().splitlines() if "speed=[" in l), "")
    print(json.dumps({"model": args.model, "env": args.env, "rates": rates,
                      "median": statistics.median(rates) if rates else None, "engine": speed[-220:]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
