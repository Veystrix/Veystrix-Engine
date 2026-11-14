"""Measure, from a recorded route trace, how much expert traffic three reuse ideas could remove (offline).

Reads a route-trace file written by ``bench_route_trace`` (``events`` rows ``[turn, phase, token, layer, [experts]]``)
and answers, with the recorded routing instead of an assumption:

* prefill reuse: for prompts cut into chunks of ``--chunk`` tokens, how many experts a layer reads when every chunk
  reads its own set (chunk-major order) against the union over a window of ``--window`` tokens (route-union
  layer-major order). Decode tokens stand in for prompt tokens: the trace records only the distinct experts of a
  whole prefill call, not per-token routes.
* cross-conversation reuse: the share of one conversation's experts at a layer that another conversation routes at
  the same layer in the same step (what co-stepping two requests layer by layer could share).
* consecutive-token reuse and static popularity: the share of a token's experts its predecessor used, and the share
  of all routings the K most frequent experts of each layer cover.

Experts pinned in the recorded run never reach the trace, so every figure describes the streamed remainder.

    python -m models.flashnext.tests.bench.bench_route_overlap TRACE.json.gz [--chunk 128] [--window 512] [--json]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from veystrix.results import output_path


def load(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def decode_routes(events):
    """turn -> ordered list of {layer: experts} per decode token."""
    turns = defaultdict(lambda: defaultdict(dict))
    for turn, phase, token, layer, experts in events:
        if phase == "decode":
            turns[turn][token][layer] = experts
    return {turn: [tokens[key] for key in sorted(tokens)] for turn, tokens in sorted(turns.items())}


def layer_ids(routes):
    return sorted({layer for tokens in routes.values() for row in tokens for layer in row})


def union(rows, layer):
    found = set()
    for row in rows:
        found.update(row.get(layer, ()))
    return found


def window_unions(routes, width):
    """Union size per layer of every non-overlapping run of ``width`` consecutive tokens within a turn."""
    sizes = []
    for tokens in routes.values():
        for start in range(0, len(tokens) - width + 1, width):
            sizes.extend(len(union(tokens[start:start + width], layer)) for layer in layer_ids(routes))
    return sizes


def describe(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"mean": round(statistics.fmean(ordered), 2), "p50": ordered[len(ordered) // 2],
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], "max": ordered[-1], "samples": len(ordered)}


def prefill_reuse(routes, chunk, window):
    """Chunk-major reads against the window union, with windows assembled from whole chunks of any turns."""
    chunks = [tokens[start:start + chunk] for tokens in routes.values()
              for start in range(0, len(tokens) - chunk + 1, chunk)]
    per_window = max(1, window // chunk)
    summed, united = [], []
    for start in range(0, len(chunks) - per_window + 1, per_window):
        group = chunks[start:start + per_window]
        for layer in layer_ids(routes):
            sets = [union(rows, layer) for rows in group]
            summed.append(sum(len(found) for found in sets))
            united.append(len(set().union(*sets)))
    if not united:
        return None
    return {"chunk_tokens": chunk, "window_tokens": per_window * chunk, "windows": len(united) // len(layer_ids(routes)),
            "chunk_major_experts_per_layer": describe(summed), "window_union_experts_per_layer": describe(united),
            "reads_ratio_chunk_major_over_union": round(sum(summed) / sum(united), 3)}


def cross_conversation(routes):
    """Share of conversation A's experts at (step, layer) that conversation B routes at the same (step, layer)."""
    shares = []
    for left, right in combinations(sorted(routes), 2):
        for a, b in zip(routes[left], routes[right]):
            for layer, experts in a.items():
                other = b.get(layer)
                if experts and other:
                    shares.append(len(set(experts) & set(other)) / len(set(experts)))
    return round(statistics.fmean(shares), 4) if shares else None


def consecutive(routes):
    shares = []
    for tokens in routes.values():
        for before, after in zip(tokens, tokens[1:]):
            for layer, experts in after.items():
                if experts and layer in before:
                    shares.append(len(set(experts) & set(before[layer])) / len(set(experts)))
    return round(statistics.fmean(shares), 4) if shares else None


def popularity(routes, counts=(4, 8, 16, 32, 64, 128)):
    by_layer, total = defaultdict(Counter), 0
    for tokens in routes.values():
        for row in tokens:
            for layer, experts in row.items():
                by_layer[layer].update(experts)
                total += len(experts)
    return {str(count): round(sum(sum(hits for _, hits in by_layer[layer].most_common(count)) for layer in by_layer) / total, 4)
            for count in counts} if total else None


def analyse(trace, chunk, window):
    routes = decode_routes(trace["events"])
    return {
        "decode_tokens": sum(len(tokens) for tokens in routes.values()), "turns": len(routes),
        "layers": len(layer_ids(routes)), "record_bytes": trace.get("record_bytes"),
        "pinned_per_layer_in_recording": trace.get("resident_experts"),
        "experts_per_token_per_layer": describe(window_unions(routes, 1)),
        "window_union_experts_per_layer": {str(width): describe(window_unions(routes, width))
                                           for width in (16, 64, 128, 256) if any(len(t) >= width for t in routes.values())},
        "prefill_reuse": prefill_reuse(routes, chunk, window),
        "cross_conversation_same_layer_share": cross_conversation(routes),
        "consecutive_token_share": consecutive(routes),
        "top_k_per_layer_routing_share": popularity(routes),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("trace", type=Path)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--json", action="store_true", help="write results/flashnext/<run>/route-overlap.json")
    args = parser.parse_args(argv)
    if args.chunk < 1 or args.window < args.chunk:
        parser.error("the window must hold at least one chunk")
    report = analyse(load(args.trace), args.chunk, args.window)
    report["trace"] = str(args.trace)
    report["trace_sha256"] = hashlib.sha256(args.trace.read_bytes()).hexdigest()
    print(json.dumps(report, indent=2))
    if args.json:
        out = output_path("flashnext", "route-overlap", "route-overlap.json")
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
