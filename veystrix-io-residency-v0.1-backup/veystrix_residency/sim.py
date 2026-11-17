"""Trace-driven expert-cache simulator.

Answers, before any hardware A/B: for a given RAM budget, what hit rate and
physical MB/token does each policy reach on YOUR routing trace, and what is the
ceiling (Belady OPT)? Converts that to predicted io ms and tok/s using your
measured effective bandwidth and non-I/O time.

Example (Nemotron, numbers from the guarded 96-token profile):
  python -m veystrix_residency.sim --trace nemo_trace.jsonl \
      --expert-bytes 5612544 --budgets-mb 1024,2048,3072,4096,5120,6144 \
      --policies s3fifo,lru,lfu,pinned-s3fifo,opt,double-s3fifo \
      --double-pc-mb 2048 --io-mb-per-ms 3.541 --non-io-ms 58.3 \
      --calibrate s3fifo:2048:0.437 --out sim_nemotron.json

Calibration gate: --calibrate POLICY:BUDGET_MB:MEASURED_HIT reruns your
production config and FAILS (exit 3) if the simulated hit rate is more than
--calibrate-tol away from the measured one. Do not trust predictions from an
uncalibrated simulator.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Sequence

from .policy import (LFU, LRU, OPT, PAGE, POOL, SSD, DoubleResidency, S3FIFO,
                     top_keys_by_frequency)
from .trace import load_trace


def make_policy(name: str, cap: int, steps, pin_source_steps, args):
    if name == "lru":
        return LRU(cap)
    if name == "s3fifo":
        return S3FIFO(cap, small_frac=args.small_frac, promote_gt=args.promote_gt)
    if name == "lfu":
        return LFU(cap)
    if name == "opt":
        return OPT(cap, steps)
    if name == "pinned-s3fifo":
        n_pin = int(cap * args.pin_frac)
        pins = top_keys_by_frequency(pin_source_steps, n_pin)
        return S3FIFO(cap, small_frac=args.small_frac, pinned=pins, promote_gt=args.promote_gt)
    if name == "pagecache-only":
        return LRU(cap, hit_code=PAGE)
    if name.startswith("double-"):
        inner = name[len("double-"):]
        pc_slots = int(args.double_pc_mb * 1e6 // args.expert_bytes)
        pool = make_policy(inner, cap, steps, pin_source_steps, args)
        return DoubleResidency(pool, pc_slots)
    raise ValueError(f"unknown policy {name}")


def run(policy, steps: Sequence[Sequence[int]], eval_from_step: int) -> Dict[str, int]:
    c = {POOL: 0, PAGE: 0, SSD: 0}
    for i, s in enumerate(steps):
        codes = policy.step(s)
        if i >= eval_from_step:
            for x in codes:
                c[x] += 1
    return c


def summarize(name, budget_mb, counts, n_eval_tokens, total_expert_bytes, args):
    acc = counts[POOL] + counts[PAGE] + counts[SSD]
    ssd_mb_tok = counts[SSD] * args.expert_bytes / 1e6 / max(n_eval_tokens, 1)
    logical_mb_tok = acc * args.expert_bytes / 1e6 / max(n_eval_tokens, 1)
    row = {
        "policy": name,
        "budget_mb": budget_mb,
        "coverage": budget_mb * 1e6 / total_expert_bytes if total_expert_bytes else None,
        "pool_hit": counts[POOL] / acc if acc else 0.0,
        "no_ssd_hit": (counts[POOL] + counts[PAGE]) / acc if acc else 0.0,
        "logical_mb_per_token": logical_mb_tok,
        "ssd_mb_per_token": ssd_mb_tok,
    }
    if args.io_mb_per_ms:
        io_ms = ssd_mb_tok / args.io_mb_per_ms
        row["pred_io_ms"] = io_ms
        if args.non_io_ms is not None:
            row["pred_tok_s"] = 1000.0 / (args.non_io_ms + io_ms)
    return row


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--pin-trace", help="calibration trace from DIFFERENT prompts for pinned-* policies")
    ap.add_argument("--experts-per-layer", type=int)
    ap.add_argument("--expert-bytes", type=int, required=True, help="bytes per routed expert (all sub-slices)")
    ap.add_argument("--budgets-mb", required=True)
    ap.add_argument("--policies", default="lru,s3fifo,lfu,pinned-s3fifo,opt")
    ap.add_argument("--warmup-tokens", type=int, default=8)
    ap.add_argument("--small-frac", type=float, default=0.1)
    ap.add_argument("--promote-gt", type=int, default=1)
    ap.add_argument("--pin-frac", type=float, default=0.5)
    ap.add_argument("--double-pc-mb", type=float, default=0.0,
                    help="page-cache MB holding expert pages alongside the pool (measure with probe.py)")
    ap.add_argument("--io-mb-per-ms", type=float, help="measured effective MB per ms of io wait (Nemotron: 3.541)")
    ap.add_argument("--non-io-ms", type=float, help="measured non-I/O ms/token (Nemotron: 58.3, Qwen: 143.2)")
    ap.add_argument("--total-experts", type=int, help="override total routed experts (layers*E)")
    ap.add_argument("--calibrate", help="POLICY:BUDGET_MB:MEASURED_HIT, e.g. s3fifo:2048:0.437")
    ap.add_argument("--calibrate-tol", type=float, default=0.03)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    steps, T, L, E = load_trace(args.trace, args.experts_per_layer)
    pin_steps = load_trace(args.pin_trace, E)[0] if args.pin_trace else steps[: args.warmup_tokens * L]
    total_keys = args.total_experts or L * E
    total_bytes = total_keys * args.expert_bytes
    eval_from = args.warmup_tokens * L
    n_eval_tokens = max(T - args.warmup_tokens, 1)
    k = len(steps[0])

    out = {"trace": args.trace, "tokens": T, "moe_layers_per_token": L, "experts_per_layer": E,
           "k": k, "expert_bytes": args.expert_bytes, "total_expert_gb": total_bytes / 1e9,
           "unique_keys_in_trace": len({x for s in steps for x in s}),
           "warmup_tokens": args.warmup_tokens, "rows": []}
    if not args.pin_trace and any(p.startswith("pinned") or "pinned" in p for p in args.policies.split(",")):
        print("WARNING: pinned policies use frequencies from the warmup tokens of the SAME trace; "
              "pass --pin-trace from different prompts for an honest estimate", file=sys.stderr)

    status = 0
    if args.calibrate:
        pol, b, measured = args.calibrate.split(":")
        cap = int(float(b) * 1e6 // args.expert_bytes)
        cnt = run(make_policy(pol, cap, steps, pin_steps, args), steps, eval_from)
        row = summarize(pol, float(b), cnt, n_eval_tokens, total_bytes, args)
        diff = row["pool_hit"] - float(measured)
        ok = abs(diff) <= args.calibrate_tol
        out["calibration"] = {"policy": pol, "budget_mb": float(b), "measured_hit": float(measured),
                              "sim_hit": row["pool_hit"], "diff": diff, "pass": ok}
        print(f"CALIBRATION {pol}@{b}MB: sim {row['pool_hit']:.3f} vs measured {float(measured):.3f} "
              f"-> {'PASS' if ok else 'FAIL'}")
        if not ok:
            status = 3

    hdr = f"{'policy':<18}{'MB':>7}{'cover':>7}{'poolhit':>9}{'noSSD':>8}{'SSD MB/t':>10}{'io ms':>8}{'tok/s':>7}"
    print(hdr)
    for b in [float(x) for x in args.budgets_mb.split(",")]:
        cap = int(b * 1e6 // args.expert_bytes)
        for p in args.policies.split(","):
            try:
                pol = make_policy(p, cap, steps, pin_steps, args)
                row = summarize(p, b, run(pol, steps, eval_from), n_eval_tokens, total_bytes, args)
            except ValueError as e:
                row = {"policy": p, "budget_mb": b, "error": str(e)}
                print(f"{p:<18}{b:>7.0f}  skipped: {e}")
                out["rows"].append(row)
                continue
            out["rows"].append(row)
            print(f"{p:<18}{b:>7.0f}{row['coverage']:>7.1%}{row['pool_hit']:>9.3f}{row['no_ssd_hit']:>8.3f}"
                  f"{row['ssd_mb_per_token']:>10.1f}{row.get('pred_io_ms', float('nan')):>8.1f}"
                  f"{row.get('pred_tok_s', float('nan')):>7.2f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    return status


if __name__ == "__main__":
    sys.exit(main())
