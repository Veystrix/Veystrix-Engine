#!/usr/bin/env python3
"""E3 (offline) - predictor recall from a recorded route trace.

Opus's zero-training pre-gate: can a CHEAP, non-routing-changing predictor
anticipate enough of a layer's routed experts to make prefetch worthwhile?
No model, no SSD, no permission - just the real 1,200-token traces we already
captured in results/io_residency/20261117/.

Three zero-training predictors, none of which change routing:
  persistence : this token, layer L is predicted by the PREVIOUS token's layer L
  cross-layer : this token, layer L is predicted by this token's layer L-1
  frequency   : the K most frequently routed experts in layer L so far

For each K we report:
  precision@K = |predicted ∩ routed| / K          (of what we prefetch, how much is used)
  recall@K    = |predicted ∩ routed| / |routed|    (of what is used, how much we covered)
  hit@K       = fraction of rows with >=1 correct expert

Opus's rule: "below 0.6 recall@14 means train the head." Note that the long
traces route ~170 experts/layer for Qwen (adaptive top-k at full width), so
recall@14 is bounded by 14/170 ~ 0.08 for a 14-wide prefetch; the meaningful
read for prefetch value is PRECISION@K and the coverage curve across larger K.
Both are reported so the pre-gate is not misread.

Usage:
  .venv/bin/python -m models.flashnext.tests.bench.bench_predictor_recall \
      results/io_residency/20261117/qwen_long_trace_dedup.jsonl --label qwen
  .venv/bin/python -m models.flashnext.tests.bench.bench_predictor_recall \
      results/io_residency/20261117/nemo_long_trace.jsonl --label nemotron
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

from veystrix.results import output_path

KS = (10, 14, 20, 50, 100, 170)


def load(path: Path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "t" in obj and "l" in obj and "e" in obj:
                rows.append((int(obj["t"]), int(obj["l"]), set(int(e) for e in obj["e"])))
    return rows


def top_k(items, k):
    """K most common experts from a Counter, deterministic tie-break by id."""
    return [e for e, _ in sorted(items.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def first_k(pool, k):
    """K expert ids from a set, deterministic by numeric id."""
    return sorted(pool)[:k]


def score(predicted, routed):
    p = set(predicted)
    inter = p & routed
    return {
        "precision": len(inter) / len(p) if p else 0.0,
        "recall": len(inter) / len(routed) if routed else 0.0,
        "hit": 1.0 if inter else 0.0,
    }


def analyse(rows, ks=KS):
    # structure: tokens[t][layer] = set(experts)
    tokens = collections.defaultdict(dict)
    for t, layer, experts in rows:
        tokens[t][layer] = experts
    timesteps = sorted(tokens)
    layers = sorted({layer for _, layer, _ in rows})

    freq = collections.defaultdict(lambda: collections.Counter())
    results = {name: {str(k): [0.0, 0.0, 0.0, 0] for k in ks}
               for name in ("persistence", "cross_layer", "frequency")}
    route_sizes = []

    for t in timesteps:
        for layer in layers:
            routed = tokens[t].get(layer)
            if not routed:
                continue
            route_sizes.append(len(routed))
            predictions = {
                "persistence": tokens.get(t - 1, {}).get(layer, set()),
                "cross_layer": tokens[t].get(layer - 1, set()),
                "frequency": set(top_k(freq[layer], max(ks))),
            }
            for name, pool in predictions.items():
                for k in ks:
                    pred = first_k(pool, k) if isinstance(pool, set) else list(pool)[:k]
                    s = score(pred, routed)
                    agg = results[name][str(k)]
                    agg[0] += s["precision"]
                    agg[1] += s["recall"]
                    agg[2] += s["hit"]
                    agg[3] += 1
            freq[layer].update(routed)

    out = {}
    n_rows = sum(results["persistence"][str(ks[0])][3] for _ in [0])
    for name, table in results.items():
        out[name] = {}
        for k in ks:
            prec, rec, hit, count = table[str(k)]
            out[name][str(k)] = {
                "precision": round(prec / count, 4) if count else 0.0,
                "recall": round(rec / count, 4) if count else 0.0,
                "hit": round(hit / count, 4) if count else 0.0,
                "rows": count,
            }
    out["_meta"] = {
        "rows": len(rows), "timesteps": len(timesteps), "layers": len(layers),
        "mean_routed_per_layer": round(sum(route_sizes) / len(route_sizes), 2)
        if route_sizes else 0.0,
        "ks": list(ks),
    }
    return out


def text(out):
    lines = []
    meta = out["_meta"]
    lines.append(f"rows={meta['rows']} timesteps={meta['timesteps']} layers={meta['layers']} "
                 f"mean routed/layer={meta['mean_routed_per_layer']}")
    lines.append(f"{'predictor':<12} {'K':>4} {'precision':>10} {'recall':>8} {'hit':>8}")
    for name in ("persistence", "cross_layer", "frequency"):
        for k, vals in out[name].items():
            lines.append(f"{name:<12} {k:>4} {vals['precision']:>10.3f} "
                         f"{vals['recall']:>8.3f} {vals['hit']:>8.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--label", default="trace")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = load(args.trace)
    out = analyse(rows)
    out["_meta"]["trace_sha256"] = hashlib.sha256(args.trace.read_bytes()).hexdigest()
    out["_meta"]["label"] = args.label
    report = text(out)
    print(report, flush=True)

    target = output_path("flashnext", "predictor-recall", f"{args.label}.json", args.json)
    target.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
