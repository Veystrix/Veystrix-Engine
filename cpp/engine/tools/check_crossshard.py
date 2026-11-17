#!/usr/bin/env python3
"""Wrapper for the cross-shard MoE ctest: generate a fresh synthetic checkpoint,
run the binary twice with VEYSTRIX_CPP_VERIFY_SLOTS=1 and require identical step
SHAs, then run once with the verifier off and require success.
"""
import argparse, os, subprocess, sys, tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--binary", required=True)
ap.add_argument("--generator", required=True)
a = ap.parse_args()

with tempfile.TemporaryDirectory(prefix="crossshard_") as d:
    r = subprocess.run([sys.executable, a.generator, d],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("generator failed:", r.stdout, r.stderr); sys.exit(1)
    print("[gen]", r.stdout.strip())

    def run(verify):
        env = dict(os.environ)
        env.pop("VEYSTRIX_CPP_VERIFY_SLOTS", None)
        if verify:
            env["VEYSTRIX_CPP_VERIFY_SLOTS"] = "1"
        p = subprocess.run([a.binary, d], capture_output=True, text=True, env=env)
        return p

    runs = []
    for label, verify in (("A(on)", True), ("B(on)", True), ("C(off)", False)):
        p = run(verify)
        print(f"--- run {label}: rc={p.returncode} ---")
        if p.stdout.strip(): print(p.stdout.strip())
        if p.returncode != 0:
            print(p.stderr); sys.exit(1)
        runs.append(p.stdout)

    shas = lambda s: [ln for ln in s.splitlines() if ln.startswith("step=")]
    if shas(runs[0]) != shas(runs[1]):
        print("RUN-TO-RUN SHA MISMATCH"); sys.exit(1)
    if len(shas(runs[0])) != 40:
        print("expected 40 step SHAs"); sys.exit(1)

print("check_crossshard OK (2x verifier-on identical + verifier-off pass)")
