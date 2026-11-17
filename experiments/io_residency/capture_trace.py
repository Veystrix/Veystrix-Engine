"""Guarded long-decode route trace capture (diagnostic only; no output change).

Loads the backend directly, installs the residency RouteTraceWriter on the
engine's existing set_route_trace hook, clears EOS stops to force a fixed-length
decode, and writes the trace JSONL. Token boundaries are tracked via the
on_decode_token callback. No model behavior changes; only recording.

  python experiments/io_residency/capture_trace.py --model flashnext \
      --checkpoint <path> --tokens 1200 --source-tokens 128 --out <trace.jsonl> \
      [--calib-out <trace2.jsonl> --calib-source-tokens 600]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESIDENCY = ROOT / "veystrix-io-residency"
sys.path.insert(0, str(RESIDENCY))


def build_prompt(backend, source_tokens, variant=0):
    from veystrix.bench.qualification import prompt_ids
    return prompt_ids(backend, source_tokens, variant=variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("flashnext", "nemotron35"), required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokens", type=int, default=1200)
    ap.add_argument("--source-tokens", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-out", default=None)
    ap.add_argument("--calib-source-tokens", type=int, default=600)
    args = ap.parse_args()

    from veystrix_residency.trace import RouteTraceWriter

    # Install the writer on the engine's existing route hook.
    writer = RouteTraceWriter(args.out)
    calib_writer = RouteTraceWriter(args.calib_out) if args.calib_out else None
    active = [writer]
    counter = {"t": 0}

    if args.model == "flashnext":
        from models.flashnext import expert_cache
        def record(layer_id, routed):
            if layer_id is not None and layer_id >= 0 and routed and not isinstance(routed[0], (list, tuple)):
                active[0].record(layer_id, routed)
        expert_cache.set_route_trace(record)
        from veystrix.backends.flashnext import FlashNextBackend
        from models.flashnext.checkpoint_policy import resolve_resident_experts
        backend = FlashNextBackend(model_path=None, resident_experts=resolve_resident_experts(None, None) or 8)
    else:
        # Nemotron: it already has a decode-profile route writer; use it directly.
        from models.nemotron35 import streamed_experts
        def record(layer_id, routed):
            if layer_id is not None and layer_id >= 0 and routed and not isinstance(routed[0], (list, tuple)):
                active[0].record(layer_id, routed)
        # Nemotron's hook is via the pool's decode_profile; set_route_trace exists on flashnext only.
        # Fall back: use the residency writer only for flashnext; Nemotron uses bench_decode_profile.
        raise SystemExit("Nemotron: use bench_decode_profile --tokens 1200 (already records routes)")

    # Suppress EOS to force a fixed-length decode.
    backend.stops = set()
    ids = build_prompt(backend, args.source_tokens)
    backend.append_tokens(ids)

    def on_token(_v, _p):
        active[0].end_token()
        counter["t"] += 1

    t0 = time.monotonic()
    out, stats = backend.generate(args.tokens, on_decode_token=on_token)
    print(f"decoded {counter['t']} tokens in {time.monotonic()-t0:.1f}s; finish={getattr(stats,'finish','?')}")

    expert_cache.set_route_trace(None)
    writer.close()
    if calib_writer:
        calib_writer.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
