"""Calibrate ``CostWeights`` offline on a recorded route trace, with a holdout (no model, no SSD).

Coordinate descent over ``recent``, ``frequency``, ``reuse`` and ``alpha`` from the defaults, a small grid per
coordinate, two passes. The objective is the mean decode hit rate over the pool sizes on the TRAIN turns.
``refill`` and ``size`` are not searched: the trace has no timing (nothing to scale ``refill`` by) and all
records have the same size (``size`` is a constant offset that cannot change an ordering).

The HOLDOUT turns report, per size, the default weights, the calibrated weights and the ``lru``, ``s3fifo``,
``least-stale`` and ``belady`` references from ``bench_pool_policy_trace.replay``. Each replay starts with an
empty pool, so the holdout numbers describe a cold pool at the start of that turn; in a real session the pool is
warm from earlier turns. The calibrated weights are written to ``--out`` only when their mean holdout hit rate
beats the defaults by at least ``MIN_GAIN`` (0.5 percentage points); otherwise the defaults stand. A JSON report
with the search trajectory, train and holdout numbers and the decision always goes to
``results/flashnext/<run>/calibration.json``.

    python -m models.flashnext.tests.bench.calibrate_cost_weights TRACE.json.gz [--gb 0.5 1 2] --out weights.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from models.flashnext.tests.bench.bench_pool_policy_trace import file_sha256, filter_turns, load_trace, replay
from models.shared.cost_aware import CostWeights
from veystrix.results import output_path

GRID = {
    "recent": (0.0, 0.05, 0.1, 0.2, 0.4),
    "frequency": (0.0, 0.1, 0.2, 0.5, 1.0),
    "reuse": (0.0, 0.5, 1.0, 2.0),
    "alpha": (0.1, 0.2, 0.3, 0.4, 0.5),
}
PASSES = 2
MIN_GAIN = 0.005
REFERENCES = ("lru", "s3fifo", "least-stale", "belady")


def slots_for(gbs, record):
    return [int(gb * 1e9) // record for gb in gbs]


def evaluate(events, slots, weights):
    """Decode hit rate of ``cost-aware`` per pool size."""
    return [replay(events, size, "cost-aware", weights=weights)[0] for size in slots]


def calibrate(events, slots, *, grid=None, passes=None, start=None):
    """Coordinate descent on the mean hit rate; returns ``(weights, trajectory)``. Ties keep the current value."""
    grid, passes = grid or GRID, PASSES if passes is None else passes
    weights = start or CostWeights()
    cache, trajectory = {}, []

    def score(candidate):
        key = tuple(sorted(candidate.to_mapping().items()))
        if key not in cache:
            cache[key] = evaluate(events, slots, candidate)
        rates = cache[key]
        return sum(rates) / len(rates), rates

    best, best_rates = score(weights)
    trajectory.append({"pass": 0, "coordinate": None, "value": None, "mean": best, "per_size": best_rates,
                       "weights": weights.to_mapping()})
    for index in range(1, passes + 1):
        for name, values in grid.items():
            current = getattr(weights, name)
            for value in values:
                if value == current:
                    continue
                candidate = replace(weights, **{name: value})
                mean, rates = score(candidate)
                trajectory.append({"pass": index, "coordinate": name, "value": value, "mean": mean,
                                   "per_size": rates, "weights": candidate.to_mapping()})
                if mean > best + 1e-12:
                    best, best_rates, weights = mean, rates, candidate
    return weights, trajectory


def holdout_report(events, slots, weights, references=REFERENCES):
    """Per size decode hit rate on ``events`` for the default weights, the calibrated weights and the references."""
    rows = {"default": evaluate(events, slots, CostWeights()), "calibrated": evaluate(events, slots, weights)}
    for name in references:
        rows[name] = [replay(events, size, name)[0] for size in slots]
    return {name: {"per_size": rates, "mean": sum(rates) / len(rates)} for name, rates in rows.items()}


def decide(holdout, minimum=None):
    minimum = MIN_GAIN if minimum is None else minimum
    gain = holdout["calibrated"]["mean"] - holdout["default"]["mean"]
    return {"gain": gain, "minimum_gain": minimum, "write_weights": gain >= minimum}


def run(events, record, gbs, train_turns, holdout_turns, *, grid=None, passes=None):
    slots = slots_for(gbs, record)
    train = filter_turns(events, train_turns)
    test = filter_turns(events, holdout_turns)
    weights, trajectory = calibrate(train, slots, grid=grid, passes=passes)
    holdout = holdout_report(test, slots, weights)
    return {"gb": list(gbs), "slots": slots, "train_turns": list(train_turns), "holdout_turns": list(holdout_turns),
            "weights": weights.to_mapping(), "trajectory": trajectory,
            "train": {"default": trajectory[0]["per_size"], "calibrated": evaluate(train, slots, weights)},
            "holdout": holdout, "decision": decide(holdout)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("trace")
    parser.add_argument("--record-bytes", type=int)
    parser.add_argument("--gb", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--train-turns", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--holdout-turns", type=int, nargs="+", default=[2])
    parser.add_argument("--out", help="write the calibrated weights JSON here when the holdout gain is enough")
    args = parser.parse_args(argv)
    if set(args.train_turns) & set(args.holdout_turns):
        parser.error("train and holdout turns must not overlap")
    try:
        events, record = load_trace(args.trace, args.record_bytes)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    report = run(events, record, args.gb, args.train_turns, args.holdout_turns)
    report |= {"trace": str(Path(args.trace).resolve()), "trace_sha256": file_sha256(args.trace),
               "record_bytes": record, "grid": {name: list(values) for name, values in GRID.items()},
               "passes": PASSES, "out": None}
    train = report["train"]
    print(f"train turns {args.train_turns}: default mean {sum(train['default']) / len(args.gb):.1%}"
          f" -> calibrated {sum(train['calibrated']) / len(args.gb):.1%}; weights {report['weights']}")
    for name, row in report["holdout"].items():
        cells = ", ".join(f"{gb:g} GB {rate:.1%}" for gb, rate in zip(args.gb, row["per_size"]))
        print(f"holdout turns {args.holdout_turns} {name}: {cells}; mean {row['mean']:.1%}")
    decision = report["decision"]
    if decision["write_weights"] and args.out:
        Path(args.out).write_text(json.dumps(report["weights"], indent=2) + "\n")
        report["out"] = args.out
        print(f"holdout gain {decision['gain']:+.2%} >= {MIN_GAIN:.1%}: wrote {args.out}")
    elif decision["write_weights"]:
        print(f"holdout gain {decision['gain']:+.2%} >= {MIN_GAIN:.1%}: pass --out to write the weights")
    else:
        print(f"holdout gain {decision['gain']:+.2%} < {MIN_GAIN:.1%}: the defaults stand, nothing written")
    target = output_path("flashnext", "cost-weights-calibration", "calibration.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {target}")
    return report


if __name__ == "__main__":
    main()
