"""Record guarded Nemotron host latency and routes on a coding request."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.flashnext.tests.bench.bench_server_short_paired import execute, large_coding_workload
from models.nemotron35.decode_profile import FLAG
from veystrix.measurement import MeasurementRun, source_hash
from veystrix.model_registry import installation_id
from veystrix.results import ROOT, output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--pool-gb", type=float, default=2)
    args = parser.parse_args()
    if not 1 <= args.tokens <= 2048 or not 0 <= args.pool_gb <= 6:
        parser.error("use 1–2048 output tokens and a pool from 0–6 GiB")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    result = output_path("nemotron35", "decode-profile", "decode-profile.jsonl")
    profiles = result.parent / "profiles"
    settings = {FLAG: str(profiles), "VEYSTRIX_NEMOTRON_KV": "int8", "VEYSTRIX_NEMOTRON_CONTEXT": "16384",
                "VEYSTRIX_NEMOTRON_NATIVE_MTP": "0", "VEYSTRIX_NEMOTRON_PROMPT_LOOKUP": "0",
                "VEYSTRIX_BENCH_TOKEN_DIGEST": "1", "VEYSTRIX_BENCH_LOGIT_DIGEST": "1"}
    messages = large_coding_workload()
    model_id = installation_id("nemotron35-lightning", str(checkpoint))
    run = MeasurementRun(result, runtime="nemotron35", experiment="decode-profile", metadata={
        "diagnostic": True, "checkpoint": str(checkpoint), "model_id": model_id, "settings": settings,
        "messages": messages, "sampling": "greedy", "reasoning_effort": "none", "pool_gib": args.pool_gb,
        "output_tokens": args.tokens, "source_sha256": source_hash(Path(__file__)),
    })
    run.start()
    try:
        command, metrics = execute(ROOT, "nemotron35", str(checkpoint), model_id, "profile", result,
                                   messages=messages, max_tokens=args.tokens, settings=settings, pool_gb=args.pool_gb)
        run.arm(arm_id="profile", condition="host-and-routes", round_index=1, command=command, metrics=metrics)
        profile, = profiles.glob("nemotron-decode-*.json")
        rows = json.loads(profile.read_text())
        if not rows["tokens"] or not metrics["token_sha256"] or not metrics["prefill_logit_sha256"]:
            raise ValueError("missing route, token or logit evidence")
        run.finish("diagnostic", profile=str(profile.relative_to(result.parent)), routed_forwards=len(rows["tokens"]),
                   pool_delta=rows["pool_delta"], prompt_tokens=metrics["runtime_usage"]["prompt"])
        print(result)
    except (Exception, KeyboardInterrupt) as error:
        run.failure(str(error), arm_id="profile")
        run.finish("failed")
        print(f"Profile failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
