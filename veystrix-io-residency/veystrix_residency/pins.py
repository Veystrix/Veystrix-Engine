"""Write a static pin file from a CALIBRATION trace (different prompts from the A/B set).

  python -m veystrix_residency.pins --trace calib.jsonl --expert-bytes 5612544 \
      --budget-mb 4096 --pin-frac 0.5 --out pins.json

Pins the most frequently routed (layer, expert) pairs. Routing itself is
unchanged, so pinning cannot affect outputs; it only protects hot experts
from being flushed by the cold tail.
"""
from __future__ import annotations

import argparse
import json
import sys

from .policy import top_keys_by_frequency
from .trace import load_trace


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--expert-bytes", type=int, required=True)
    ap.add_argument("--budget-mb", type=float, required=True)
    ap.add_argument("--pin-frac", type=float, default=0.5)
    ap.add_argument("--experts-per-layer", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    steps, T, L, E = load_trace(a.trace, a.experts_per_layer)
    n = int(int(a.budget_mb * 1e6 // a.expert_bytes) * a.pin_frac)
    keys = top_keys_by_frequency(steps, n)
    pairs = [[k // E, k % E] for k in keys]
    with open(a.out, "w") as f:
        json.dump(pairs, f)
    print(f"pinned {len(pairs)} experts ({len(pairs) * a.expert_bytes / 1e9:.2f} GB) from {T} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
