"""Offline route-union test for the ANE-drafter hypothesis.

Uses the per-token routed-expert sets already recorded in
results/flashnext/*route-trace*/route-trace.json.gz. No model run, no permission
needed. Computes, for K consecutive decode tokens, how many distinct experts
are touched versus the sum of the per-token sets. A low ratio means one batched
verification could share SSD reads across K tokens — the condition under which
ANE drafting + exact GPU verification could beat plain decode.
"""
from __future__ import annotations

import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT / "results/flashnext/20260922-221213-route-trace-manual/route-trace.json.gz"
KS = (2, 3, 4, 6)


def load():
    with gzip.open(TRACE, "rt") as f:
        data = json.load(f)
    # events: [turn, phase, token, layer, [expert ids]]
    by_key = defaultdict(set)
    for turn, phase, token, layer, experts in data["events"]:
        if phase == "decode":
            by_key[(turn, int(layer), int(token))] = set(experts)
    return by_key, data


def analyze(by_key, k):
    # index tokens per (turn, layer)
    seq = defaultdict(list)
    for (turn, layer, token), experts in by_key.items():
        seq[(turn, layer)].append((token, experts))
    ratios = []
    absolute = []
    for (turn, layer), rows in seq.items():
        rows.sort()
        for i in range(len(rows) - k + 1):
            window = rows[i:i + k]
            union = set().union(*(e for _, e in window))
            total = sum(len(e) for _, e in window)
            if total:
                ratios.append(len(union) / total)
                absolute.append(len(union))
    return ratios, absolute


def main():
    by_key, data = load()
    print(f"trace: {TRACE.name}")
    print(f"decode events: {len(by_key)} sets, {len(set(k[2] for k in by_key))} tokens, "
          f"{len(set(k[1] for k in by_key))} layers")
    sizes = [len(v) for v in by_key.values()]
    print(f"experts per (token,layer): median {statistics.median(sizes)}, "
          f"mean {statistics.mean(sizes):.2f}, min {min(sizes)}, max {max(sizes)}")
    print()
    print(f"{'K':>2} {'union/sum ratio':>18} {'union experts':>14} {'implied bytes ratio':>20}")
    for k in KS:
        ratios, absolute = analyze(by_key, k)
        r = statistics.median(ratios)
        print(f"{k:>2} {r:>18.3f} {statistics.median(absolute):>14.0f} {r:>20.3f}")
    print()
    print("Interpretation (Fable's gate): ratio < 0.6 => a K-token batch shares enough")
    print("expert reads to be worth an ANE drafter + exact GPU verify; near 1.0 =>")
    print("native MTP's failure repeats (each drafted row costs a full token's reads).")


if __name__ == "__main__":
    main()
