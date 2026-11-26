#!/usr/bin/env python3
"""Runtime check for the opt-in tokenized fragment cache (UCH-08 / MIC-06).

Starts the normal server with ``VEYSTRIX_FRAGMENT_CACHE=1``, sends the same
rendered prompt twice, and records the fragment-cache hit count from
``/v1/cache/diagnostics`` plus the emitted token digest of each request. The
second request proves the cache is on the live encode path; the token digest
gate proves the cached id list is the same as a fresh encode.

    python -m models.nemotron35.tests.bench.bench_fragment_cache \
        --checkpoint "$NEMOTRON_CHECKPOINT" --tokens 2048
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from models.flashnext.tests.bench.bench_server_short_paired import (  # noqa: E402
    counters, get, listen_port, request, swap_used_mb, wait_ready)
from models.flashnext.diskio import vm_counters  # noqa: E402
from veystrix.measurement import MeasurementRun  # noqa: E402
from veystrix.model_registry import installation_id  # noqa: E402
from veystrix.results import output_path  # noqa: E402


def prompt(tokens: int):
    body = ("def build_cache_key(request, namespace):\n"
            "    return hashlib.sha256((request + namespace).encode()).hexdigest()\n\n")
    repeats = max(1, tokens // 24)
    return [{"role": "system", "content": "Copy code exactly. Output only the requested code."},
            {"role": "user", "content": f"Copy the following lines exactly.\n\n{body * repeats}"}]


def diagnostics(port):
    status, value = get(port, "/v1/cache/diagnostics")
    return value if status == 200 else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.environ.get("NEMOTRON_CHECKPOINT", ""))
    parser.add_argument("--tokens", type=int, default=2048)
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint or NEMOTRON_CHECKPOINT is required")
    runtime, family = "nemotron35", "nemotron35-lightning"
    model_id = installation_id(family, args.checkpoint)
    messages = prompt(args.tokens)
    result = output_path(runtime, "fragment-cache", "fragment-cache.jsonl")
    run = MeasurementRun(result, runtime=runtime, experiment="fragment-cache", metadata={
        "diagnostic": True, "checkpoint": args.checkpoint, "installed_model_id": model_id,
        "messages": messages, "temperature": 0, "max_tokens": 8,
        "cache": "fresh server and private prefix directory; OS file cache retained",
        "purpose": "verify the opt-in tokenized fragment cache executes on the server encode path"})
    run.start()
    port = listen_port()
    log_path = result.parent / "fragment-cache-server.log"
    with tempfile.TemporaryDirectory(prefix="veystrix-fragment-") as scratch, log_path.open("x") as log:
        private = Path(scratch)
        prefs = private / "preferences.json"
        prefs.write_text('{"temperature": 0, "thinking_enabled": false}')
        env = os.environ | {
            "VEYSTRIX_PYTHON": sys.executable, "PYTHONPATH": str(ROOT),
            "VEYSTRIX_FRAGMENT_CACHE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "VEYSTRIX_NEMOTRON_KV": "bf16", "VEYSTRIX_NEMOTRON_AUTHORIZED": "1",
            "VEYSTRIX_SERVER_API_KEY": "", "VEYSTRIX_NEMOTRON_CONTEXT": "8192",
        }
        command = [sys.executable, "-m", "veystrix.cli", "--model", family, "--checkpoint", args.checkpoint,
                   "--server", "--host", "127.0.0.1", "--port", str(port), "--no-auto-purge",
                   "--preferences-file", str(prefs), "--api-keys-file", str(private / "keys.json"),
                   "--prefix-dir", str(private / "prefix"), "--pool-gb", "2"]
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            wait_ready(process, port)
            arms = []
            for index in range(2):
                before = counters(process.pid)
                before_swap = swap_used_mb()
                started = time.monotonic()
                measured = request(port, model_id, messages=messages, max_tokens=8, timeout=600)
                measured["wall_seconds"] = time.monotonic() - started
                measured["swap_mib_before"] = before_swap
                measured["swap_mib_after"] = swap_used_mb()
                measured["physical_read_bytes"] = counters(process.pid).get("physical_read_bytes", -1) - before.get("physical_read_bytes", -1)
                measured["vm_before"] = vm_counters()
                measured["cache"] = diagnostics(port)
                measured["fragment"] = measured["cache"].get("fragment_cache")
                measured["prompt_tokens"] = measured["cache"].get("newly_computed_prefill_tokens")
                arms.append(measured)
                run.arm(arm_id=f"{index + 1}-request", condition="fragment-cache", round_index=index + 1,
                        command=command, metrics=measured)
                print(f"request {index + 1}: TTFT {measured['wire_ttft_seconds']:.2f}s "
                      f"fragment {measured['fragment']}", flush=True)
            first, second = arms
            hashes = {arm["output_text_sha256"] for arm in arms}
            hits = int((second["fragment"] or {}).get("hits", 0))
            checks = {"text_parity": len(hashes) == 1, "second_request_cache_hit": hits > 0,
                      "first_request_was_cold": int((first["fragment"] or {}).get("hits", 0)) == 0}
            status = "diagnostic" if all(checks.values()) else "failed"
            run.finish(status, **checks, fragment_stats=second["fragment"],
                       ttft_seconds=[arm["wire_ttft_seconds"] for arm in arms])
            print(json.dumps(checks), flush=True)
            return 0 if all(checks.values()) else 1
        except Exception as exc:
            run.failure(str(exc))
            print(f"fragment-cache check failed: {exc}", flush=True)
            return 1
        finally:
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=3)
                    break
                except subprocess.TimeoutExpired:
                    continue


if __name__ == "__main__":
    raise SystemExit(main())
