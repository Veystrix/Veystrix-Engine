"""Paired A/B for Qwen K=2 (exact speculative depth 2) under the guard.

One arm per process (MTP on/off via constructor), same prompt/seed, greedy.
Collects output digest, decode tok/s, physical SSD bytes/token. Run via guard.

  .venv/bin/python experiments/ane/ab_k2_qwen.py --arm baseline|candidate --tokens 128

Owner authorized this specific guarded run on 2026-11-17.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from veystrix.results import output_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    ap.add_argument("--tokens", type=int, default=128)
    args = ap.parse_args()

    from models.flashnext.settings.launch import apply_chat_environment
    apply_chat_environment(os.environ)
    os.environ.update({
        "FLASHNEXT_EXPERT_POOL_GB": "0", "FLASHNEXT_SLAB_GLOBAL": "0",
        "FLASHNEXT_SLAB_PACK": "0", "FLASHNEXT_PROMPT_LOOKUP": "0",
        "FLASHNEXT_PREFIX_CACHE": "0", "FLASHNEXT_GPU_KEEPWARM": "0",
        "FLASHNEXT_PREFILL_CACHE_LIMIT_MB": "256",
    })
    from veystrix.backends.flashnext import FlashNextBackend
    kw = {"native_mtp": "on" if args.arm == "candidate" else "off"}
    if args.arm == "candidate":
        kw["mtp_depth"] = 2
    from models.flashnext.checkpoint_policy import resolve_resident_experts
    backend = FlashNextBackend(model_path=None,
                               resident_experts=resolve_resident_experts(None, None) or 8, **kw)

    from models.flashnext.diskio import ReadMeter
    from veystrix.bench.qualification import prompt_ids

    ids = prompt_ids(backend, 64)
    meter = ReadMeter()
    backend.reset()
    backend.append_tokens(ids)
    times = []
    bmarks = []

    def on_prefilled():
        meter.reset()

    def on_token(_v, _p):
        times.append(time.monotonic())
        bmarks.append(meter.bytes_since())
        meter.reset()

    t0 = time.monotonic()
    out, stats = backend.generate(args.tokens, on_prefilled=on_prefilled, on_decode_token=on_token)
    wall = time.monotonic() - t0
    digest = hashlib.sha256(out.encode()).hexdigest()
    decode_s = (times[-1] - times[0]) if len(times) > 1 else wall
    tps = (len(times) - 1) / decode_s if decode_s else 0
    report = {
        "arm": args.arm, "mtp": kw, "tokens": stats.tokens,
        "output_sha256": digest, "decode_tok_s": round(tps, 3),
        "wall_s": round(wall, 2),
        "physical_mb_per_token": round(sum(max(0, b) for b in bmarks) / 1e6 / max(1, len(bmarks)), 1),
    }
    dest = output_path("flashnext", "k2-ab", f"{args.arm}.json")
    dest.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
