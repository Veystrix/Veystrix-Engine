"""Live route-union measurement on the CURRENT Flash-Next checkpoint.

Authorized guarded run (2026-11-17). Measures the consecutive-token expert-set
union ratio with the model as it actually runs today (10 routed experts/token),
which the saved trace could not (it was an 8-expert config). No production code
is changed; this only observes the existing route-trace hook.

Run under: scripts/flashnext/real_run_guard.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from veystrix.results import output_path  # noqa: E402

PROMPT = "Explain mixture-of-experts routing and top-k expert selection briefly."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=128)
    args = ap.parse_args()

    from models.flashnext.settings.launch import apply_chat_environment
    apply_chat_environment(os.environ)
    # Override with the proven low-memory settings from the qualification harness
    # (the chat defaults request a 6 GiB pool + slab pack, which trips the 3 GiB guard).
    os.environ.update({
        "FLASHNEXT_EXPERT_POOL_GB": "0",
        "FLASHNEXT_SLAB_GLOBAL": "0",
        "FLASHNEXT_SLAB_PACK": "0",
        "FLASHNEXT_PROMPT_LOOKUP": "0",
        "FLASHNEXT_PREFIX_CACHE": "0",
        "FLASHNEXT_GPU_KEEPWARM": "0",
        "FLASHNEXT_PREFILL_CACHE_LIMIT_MB": "256",
    })
    source = Path(os.path.expanduser("~/.cache/flashnext"))
    private = Path(tempfile.mkdtemp(prefix="flashnext-live-union-"))
    for item in list(source.glob("pins.json")) + list(source.glob("slab-frozen-*.json")):
        shutil.copy2(item, private / item.name)
    os.environ["FLASHNEXT_PIN_CACHE"] = str(private / "pins.json")

    from veystrix.checkpoints import resolve_flashnext
    from models.flashnext.checkpoint_policy import resolve_resident_experts
    checkpoint = str(resolve_flashnext(os.environ.get("VEYSTRIX_FLASHNEXT_MODEL")))
    resident = resolve_resident_experts(None, checkpoint) or 32

    from veystrix.backends.flashnext import FlashNextBackend
    from models.flashnext import expert_cache
    from models.flashnext.diskio import ReadMeter, free_memory_mb

    backend = FlashNextBackend(model_path=checkpoint, resident_experts=resident)
    state = {"phase": "prefill", "token": 0}
    per_token: dict[tuple, set] = {}
    pinned = None
    slab = None

    def trace(layer_id, routed):
        if state["phase"] == "decode":
            per_token[(int(layer_id), int(state["token"]))] = set(int(e) for e in routed)

    expert_cache.set_route_trace(trace)
    meter = ReadMeter()
    per_token_bytes = []
    try:
        backend.reset()
        text = (f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n"
                "<think>\n\n</think>\n\n")
        backend.append_text(text)

        def on_prefilled():
            state["phase"] = "decode"
            meter.reset()

        def on_token(_value, _piece):
            per_token_bytes.append(meter.bytes_since())
            meter.reset()
            state["token"] += 1

        free_before = free_memory_mb()
        out, stats = backend.generate(args.tokens, on_prefilled=on_prefilled,
                                      on_decode_token=on_token)
        pinned = {str(k): sorted(v) for k, v in backend.routing.pinned.items()}
    finally:
        expert_cache.set_route_trace(None)

    # Compute the K=2 union ratio with the exact spec formula.
    tokens = sorted({t for _, t in per_token})
    layers = sorted({l for l, _ in per_token})
    ratios = []
    for t1, t2 in zip(tokens, tokens[1:]):
        su = sa = 0
        for layer in layers:
            a = per_token.get((layer, t1), set())
            b = per_token.get((layer, t2), set())
            su += len(a | b)
            sa += len(a) + len(b)
        if sa:
            ratios.append(su / sa)

    experts_per_call = [len(v) for v in per_token.values()]
    msgs = {
        "checkpoint": checkpoint, "date": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "tokens": stats.tokens, "layers": len(layers),
        "experts_per_call_median": statistics.median(experts_per_call),
        "union_ratio_k2_mean": statistics.mean(ratios) if ratios else None,
        "union_ratio_k2_median": statistics.median(ratios) if ratios else None,
        "union_ratio_k2_p10": sorted(ratios)[len(ratios)//10] if ratios else None,
        "union_ratio_k2_p90": sorted(ratios)[9*len(ratios)//10] if ratios else None,
        "physical_mb_per_token": sum(max(0, b) for b in per_token_bytes) / 1e6 / max(1, len(per_token_bytes)),
        "free_mb_before": free_before,
        "output_sha256_prefix": __import__("hashlib").sha256(out.encode()).hexdigest()[:16],
    }
    out_path = output_path("flashnext", "live-route-union", "union.json")
    out_path.write_text(json.dumps(msgs, indent=1))
    print(json.dumps(msgs, indent=1), flush=True)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
