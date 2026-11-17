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

from .policy import (LFU, LRU, OPT, PAGE, POOL, SSD, DoubleResidency, NextUse, NextUseModel,
                     S3FIFO, compulsory_flags, top_keys_by_frequency)
from .trace import load_prefill_counts, load_trace


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
    if name == "nextuse":
        if getattr(args, "_nextuse_model", None) is None:
            args._nextuse_model = NextUseModel.fit(pin_source_steps, args._L,
                                                   decays=(args.nextuse_decay, args.nextuse_slow_decay))
        return NextUse(cap, args._L, args._E, args._nextuse_model)
    if name == "prefill-pinned-s3fifo":
        if not args._prefill_counts:
            raise ValueError("trace has no prefill_counts line (capture with VEYSTRIX_ROUTE_TRACE_PREFILL=1)")
        n_pin = int(cap * args.pin_frac)
        ranked = sorted(args._prefill_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n_pin]
        return S3FIFO(cap, small_frac=args.small_frac, pinned=[k for k, _ in ranked], promote_gt=args.promote_gt)
    if name == "pagecache-only":
        return LRU(cap, hit_code=PAGE)
    if name.startswith("double-"):
        inner = name[len("double-"):]
        pc_slots = int(args.double_pc_mb * 1e6 // args.expert_bytes)
        pool = make_policy(inner, cap, steps, pin_source_steps, args)
        return DoubleResidency(pool, pc_slots)
    raise ValueError(f"unknown policy {name}")


def run(policy, steps: Sequence[Sequence[int]], eval_from_step: int, comp=None) -> Dict:
    c = {POOL: 0, PAGE: 0, SSD: 0, "comp": 0}
    comp = comp if comp is not None else compulsory_flags(steps)
    for i, s in enumerate(steps):
        codes = policy.step(s)
        if i >= eval_from_step:
            for x, first in zip(codes, comp[i]):
                c[x] += 1
                c["comp"] += first
    return c


def summarize(name, budget_mb, counts, n_eval_tokens, total_expert_bytes, args):
    acc = counts[POOL] + counts[PAGE] + counts[SSD]
    comp = counts.get("comp", 0)
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
        # 3C split: first-touch misses happen for ANY cache and fade with run length.
        "compulsory_share": comp / acc if acc else 0.0,
        "capacity_miss_share": (counts[SSD] - comp) / acc if acc else 0.0,
        # hit rate over non-first-touch accesses = steady-state estimate for long decodes
        "steady_no_ssd_hit": (counts[POOL] + counts[PAGE]) / (acc - comp) if acc > comp else 1.0,
    }
    steady_ssd_mb = (1 - row["steady_no_ssd_hit"]) * logical_mb_tok
    row["steady_ssd_mb_per_token"] = steady_ssd_mb
    if args.io_mb_per_ms:
        io_ms = ssd_mb_tok / args.io_mb_per_ms
        row["pred_io_ms"] = io_ms
        if args.non_io_ms is not None:
            row["pred_tok_s"] = 1000.0 / (args.non_io_ms + io_ms)
            row["steady_pred_tok_s"] = 1000.0 / (args.non_io_ms + steady_ssd_mb / args.io_mb_per_ms)
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
    ap.add_argument("--nextuse-decay", type=float, default=0.9)
    ap.add_argument("--nextuse-slow-decay", type=float, default=0.99)
    ap.add_argument("--calib-prefix-tokens", type=int, default=0,
                    help="no --pin-trace: fit pins/nextuse on the first N tokens and evaluate only after them")
    ap.add_argument("--prefix-sweep", help="e.g. 24,48,72,96: re-run --sweep-policy on growing trace prefixes")
    ap.add_argument("--sweep-policy", default="opt")
    ap.add_argument("--sweep-budget-mb", type=float, default=6144)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    steps, T, L, E = load_trace(args.trace, args.experts_per_layer)
    args._L, args._E, args._nextuse_model = L, E, None
    args._prefill_counts = load_prefill_counts(args.trace, E)
    start_tok = args.warmup_tokens
    if args.pin_trace:
        pin_steps = load_trace(args.pin_trace, E)[0]
    elif args.calib_prefix_tokens:
        pin_steps = steps[: args.calib_prefix_tokens * L]
        start_tok = max(start_tok, args.calib_prefix_tokens)
    else:
        pin_steps = steps[: args.warmup_tokens * L]
    total_keys = args.total_experts or L * E
    total_bytes = total_keys * args.expert_bytes
    eval_from = start_tok * L
    n_eval_tokens = max(T - start_tok, 1)
    comp = compulsory_flags(steps)
    k = len(steps[0])

    out = {"trace": args.trace, "tokens": T, "moe_layers_per_token": L, "experts_per_layer": E,
           "k": k, "expert_bytes": args.expert_bytes, "total_expert_gb": total_bytes / 1e9,
           "unique_keys_in_trace": len({x for s in steps for x in s}),
           "warmup_tokens": args.warmup_tokens, "rows": []}
    if not args.pin_trace and not args.calib_prefix_tokens and any(("pinned" in p and "prefill" not in p) or p == "nextuse"
                                  for p in args.policies.split(",")):
        print("WARNING: pinned/nextuse are fitted on the SAME trace's leading tokens; pass --pin-trace "
              "from different prompts (or --calib-prefix-tokens) for an honest estimate", file=sys.stderr)

    status = 0
    if args.calibrate:
        pol, b, measured = args.calibrate.split(":")
        cap = int(float(b) * 1e6 // args.expert_bytes)
        cnt = run(make_policy(pol, cap, steps, pin_steps, args), steps, eval_from, comp)
        row = summarize(pol, float(b), cnt, n_eval_tokens, total_bytes, args)
        diff = row["pool_hit"] - float(measured)
        ok = abs(diff) <= args.calibrate_tol
        out["calibration"] = {"policy": pol, "budget_mb": float(b), "measured_hit": float(measured),
                              "sim_hit": row["pool_hit"], "diff": diff, "pass": ok}
        print(f"CALIBRATION {pol}@{b}MB: sim {row['pool_hit']:.3f} vs measured {float(measured):.3f} "
              f"-> {'PASS' if ok else 'FAIL'}")
        if not ok:
            status = 3

    if args.prefix_sweep:
        cap = int(args.sweep_budget_mb * 1e6 // args.expert_bytes)
        out["prefix_sweep"] = []
        print(f"PREFIX SWEEP {args.sweep_policy}@{args.sweep_budget_mb:.0f}MB "
              f"(a 'ceiling' that rises with N is a trace-length artefact)")
        for n in [int(x) for x in args.prefix_sweep.split(",")]:
            if n <= start_tok or n > T:
                print(f"  N={n:>5}  skipped (must be in ({start_tok}, {T}])")
                continue
            sub = steps[: n * L]
            subargs = argparse.Namespace(**vars(args))
            subargs._nextuse_model = None
            c = run(make_policy(args.sweep_policy, cap, sub, pin_steps, subargs), sub, eval_from, comp[: n * L])
            r = summarize(args.sweep_policy, args.sweep_budget_mb, c, max(n - start_tok, 1), total_bytes, args)
            r["tokens"] = n
            out["prefix_sweep"].append(r)
            print(f"  N={n:>5}  hit {r['no_ssd_hit']:.3f}  compulsory {r['compulsory_share']:.3f}  "
                  f"capacity-miss {r['capacity_miss_share']:.3f}  steady {r['steady_no_ssd_hit']:.3f}")

    hdr = (f"{'policy':<22}{'MB':>7}{'cover':>7}{'poolhit':>9}{'noSSD':>8}{'cold':>7}{'steady':>8}"
           f"{'SSD MB/t':>10}{'tok/s':>7}{'st tok/s':>9}")
    print(hdr)
    for b in [float(x) for x in args.budgets_mb.split(",")]:
        cap = int(b * 1e6 // args.expert_bytes)
        for p in args.policies.split(","):
            try:
                pol = make_policy(p, cap, steps, pin_steps, args)
                row = summarize(p, b, run(pol, steps, eval_from, comp), n_eval_tokens, total_bytes, args)
            except ValueError as e:
                row = {"policy": p, "budget_mb": b, "error": str(e)}
                print(f"{p:<22}{b:>7.0f}  skipped: {e}")
                out["rows"].append(row)
                continue
            out["rows"].append(row)
            print(f"{p:<22}{b:>7.0f}{row['coverage']:>7.1%}{row['pool_hit']:>9.3f}{row['no_ssd_hit']:>8.3f}"
                  f"{row['compulsory_share']:>7.3f}{row['steady_no_ssd_hit']:>8.3f}"
                  f"{row['ssd_mb_per_token']:>10.1f}{row.get('pred_tok_s', float('nan')):>7.2f}"
                  f"{row.get('steady_pred_tok_s', float('nan')):>9.2f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    return status


if __name__ == "__main__":
    sys.exit(main())
